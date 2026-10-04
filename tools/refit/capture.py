#!/usr/bin/env python3
"""Stage 6 capture driver: send the refit's prompts to a running server one at a time and file each prompt's capture
in a directory of its own (`run`); add the projector captures to running ridge sums and delete them once a saved
checkpoint holds them (`acc`, CPU, alongside or after `run`).

  capture.py run --what activations|state --prompts P.jsonl [...] --engine-dir DIR --store DIR [--port 8100]
  capture.py acc --store DIR --set train --sums DIR [--follow] [--every 40] [--threads 8]

run: the server must have been started with RADIANCE_KVA_CAPTURE (activations, mode off) or RADIANCE_KVA_CAPTURE_STATE
  (state, any mode) pointing at a mount of --engine-dir. First the tokenizer probe (docs.py's manifest) goes through
  /tokenize and must give the same ids. Each prompt goes as token ids to /v1/completions (max_tokens 1, temperature 0;
  usage.prompt_tokens must equal the ids sent, no cached tokens). Its records are the new capture.jsonl / state.jsonl
  lines: there must be exactly one per chunk (per rank for states) of the chunks the prompt makes -- [k*chunk,
  min(N, (k+1)*chunk)) -- each keyed by its first position and the FNV-1a hash of its ids, recomputed here. A
  short count means the engine replayed a recorded pass, which never calls the plugin's step() (radiance
  core/runtime/ctx.cpp:1258-1278): the run stops, naming it (serve with recording off). The prompt's files are moved
  into <store>/<key>/ with their jsonl lines and a meta.json, written last, through a .tmp dir and a rename, so a
  directory without meta.json is never read. Resumable: a prompt whose directory exists, or that the sums hold, is
  skipped. At the end <store>/run-done.json is written (acc --follow exits on it).
acc: qfn.fit.Sums of the plain config at split S without the `final` target (not captured), raw and chat apart, in
  qfn.bigcap's checkpoint format (acc-{raw,chat}-s<S>.pt: state + documents), restored with qfn.bigcap.restore, so
  qfn.fit.stats_parts reads them. Each document is one Sums.add (its chunks concatenated); its directory is deleted
  only after a checkpoint that holds it is on disk. A document is added at most once and its row count must equal
  the rows its capture.jsonl lines state.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert  # noqa: E402
import kva_research  # noqa: E402

JSONL = {"activations": "capture.jsonl", "state": "state.jsonl"}
HTTP_TIMEOUT = 900
DONE = "run-done.json"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fnv1a64(ids):
    """The plugin's chunk hash (arch/kva_dump.h chunk_key): FNV-1a 64 over the ids' little-endian int32 bytes."""
    h = 1469598103934665603
    for byte in np.asarray(ids, dtype="<i4").tobytes():
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def expected_chunks(ids, chunk):
    """{chunk key 'p<P>.h<H16>': (P, n_tok)} of the chunks a single-sequence prefill of `ids` runs."""
    out = {}
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        out[f"p{start}.h{fnv1a64(part)}"] = (start, len(part))
    return out


def entry_key(what, entry):
    """'p<P>.h<H16>' of a jsonl entry ('chunk.p..h..' prefix, or 'state.p..h...r<rank>.npy' file)."""
    name = entry["prefix"] if what == "activations" else entry["file"]
    return ".".join(name.split(".")[1:3])


def entry_files(what, engine_dir, entry):
    if what == "state":
        return [engine_dir / entry["file"]]
    return sorted(engine_dir.glob(entry["prefix"] + ".*"))


def http(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"POST {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")


def tokenizer_probe(base, manifest):
    for probe in manifest["probe"]:
        got = http(base, "/tokenize", {"prompt": probe["text"], "add_special_tokens": False}).get("tokens")
        if got != probe["ids"]:
            raise SystemExit(f"tokenizer probe: /tokenize gives {len(got or [])} ids, docs.py gave {len(probe['ids'])} "
                             "(first difference decides: the served vocab is not the tokenizer the prompts used)")
    log(f"tokenizer probe: {len(manifest['probe'])} texts tokenise identically")


def read_lines(path, offset):
    """(entries after line `offset`, new line count)."""
    if not path.exists():
        return [], 0
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[offset:], len(lines)


def check_entries(what, entries, expected, prompt):
    """The new jsonl lines must be exactly one per expected chunk (per rank for states), nothing else."""
    world = max([e.get("world", 1) for e in entries] or [1])
    seen = {}
    for e in entries:
        key = entry_key(what, e)
        if key not in expected or (e["chunk_start"], e["n_tok"]) != expected[key]:
            raise SystemExit(f"{prompt['key']}: record {key} ({e['chunk_start']}, {e['n_tok']}) is not one of its chunks")
        seen.setdefault(key, set()).add(e.get("rank", 0))
    short = [k for k in expected if seen.get(k) != set(range(world))]
    if short:
        raise SystemExit(f"{prompt['key']}: {len(short)} of {len(expected)} chunks not captured (e.g. {short[:2]}): the "
                         "engine replayed recorded passes, which skip the plugin's step(); serve with recording off")


def file_prompt(what, engine_dir, store, prompt, entries, response):
    """Move the prompt's files and jsonl lines into <store>/<key>/ (meta.json last, then the rename)."""
    dest = store / prompt["key"]
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for e in entries:
        files = entry_files(what, engine_dir, e)
        if not files:
            raise SystemExit(f"{prompt['key']}: no files for {e}")
        for f in files:
            shutil.move(str(f), tmp / f.name)
    (tmp / JSONL[what]).write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    meta = {k: prompt[k] for k in prompt if k != "ids"}
    meta.update(ids_sha256=hashlib.sha256(np.asarray(prompt["ids"], "<i4").tobytes()).hexdigest(),
                records=len(entries), rows=sum(e.get("rows", 0) for e in entries),
                usage=response.get("usage"), timings=response.get("timings"))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def summed_keys(sums_dir):
    """Document keys the acc checkpoints hold (only their docs lists are read: mmap)."""
    import torch
    keys = set()
    for path in Path(sums_dir).glob("acc-*-s*.pt") if sums_dir else []:
        keys |= set(torch.load(path, map_location="cpu", mmap=True, weights_only=True)["docs"])
    return keys


def cmd_run(a):
    base = f"http://127.0.0.1:{a.port}"
    engine_dir, store = Path(a.engine_dir), Path(a.store)
    store.mkdir(parents=True, exist_ok=True)
    (store / DONE).unlink(missing_ok=True)
    tokenizer_probe(base, json.loads((Path(a.prompts[0]).parent / "manifest.json").read_text()))
    prompts = [json.loads(x) for p in a.prompts for x in open(p, encoding="utf-8") if x.strip()]
    done = summed_keys(a.sums) | {p["key"] for p in prompts if (store / p["key"] / "meta.json").exists()}
    todo = [p for p in prompts if p["key"] not in done]
    log(f"{len(prompts)} prompts, {len(prompts) - len(todo)} already captured, {sum(p['n_tokens'] for p in todo)} "
        f"tokens to go")
    jsonl, t0, tokens = engine_dir / JSONL[a.what], time.time(), 0
    for i, prompt in enumerate(todo, 1):
        _, offset = read_lines(jsonl, 0)
        r = http(base, "/v1/completions", {"model": "m", "prompt": prompt["ids"], "max_tokens": 1, "temperature": 0})
        usage = r.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if usage.get("prompt_tokens") != prompt["n_tokens"] or cached:
            raise SystemExit(f"{prompt['key']}: prompt_tokens {usage.get('prompt_tokens')} (sent {prompt['n_tokens']}), "
                             f"cached {cached}")
        entries, _ = read_lines(jsonl, offset)
        check_entries(a.what, entries, expected_chunks(prompt["ids"], a.chunk), prompt)
        file_prompt(a.what, engine_dir, store, prompt, entries, r)
        tokens += prompt["n_tokens"]
        rate = tokens / (time.time() - t0)
        left = sum(p["n_tokens"] for p in todo[i:])
        log(f"[{i}/{len(todo)}] {prompt['key']}: {prompt['n_tokens']} tokens, {len(entries)} records, "
            f"prompt_ms {(r.get('timings') or {}).get('prompt_ms')}; {rate:.0f} tok/s, ~{left / rate / 60:.0f} min left")
    (store / DONE).write_text(json.dumps(dict(prompts=len(prompts), captured=len(todo),
                                              finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))) + "\n")
    log(f"run done: {len(todo)} prompts captured in {(time.time() - t0) / 60:.1f} min")


# ---- acc -------------------------------------------------------------------------------------------
def layout_for(entry):
    """qfn.fit.Layout('plain') at the capture's split, without the `final` target; the model's numbers come from
    the capture's own capture.jsonl line (no model number typed here)."""
    from qfn import fit
    from qfn.weights import Checkpoint
    config = dict(num_hidden_layers=entry["layers"][-1] + 1, hidden_size=entry["hidden"], hc_count=entry["hc"])
    lay = fit.Layout("plain", entry["split"], Checkpoint(tensors={}, config=config))
    lay.final = False                    # final_multi_hidden is not captured (it feeds only MTP)
    return lay


def ready(store, prompt_set):
    """Filed prompts of one set (<store>/<set>/<format>/<name>/meta.json), complete by construction."""
    return sorted(p.parent for p in store.glob(f"{prompt_set}/*/*/meta.json"))


def cmd_acc(a):
    kva_research.root()
    import torch
    from qfn import bigcap, fit
    torch.set_num_threads(a.threads)
    store, sums_dir = Path(a.store), Path(a.sums)
    sums_dir.mkdir(parents=True, exist_ok=True)
    state = {}                           # fmt -> [Sums, docs, path, dirty]
    pending, added, t0 = [], 0, time.time()

    def checkpoint():
        for fmt, (sums, docs, path, dirty) in state.items():
            if dirty:
                bigcap.save_ckpt(path, sums, docs, sums.lay.split, fmt, [str(store)])
                state[fmt][3] = False
        for d in pending:
            shutil.rmtree(d)
        log(f"checkpoint: " + ", ".join(f"{f} {len(s[1])} docs {s[0].n} rows" for f, s in state.items())
            + f"; deleted {len(pending)} capture dirs")
        pending.clear()

    while True:
        new = [d for d in ready(store, a.set) if d not in pending]
        for d in new:
            meta = json.loads((d / "meta.json").read_text())
            entries = convert.chunks(d)
            if not state.get(meta["format"]):
                path = sums_dir / f"acc-{meta['format']}-s{entries[0]['split']}.pt"
                sums, old = fit.Sums(layout_for(entries[0])), bigcap.load_ckpt(path)
                if old is not None:
                    bigcap.restore(sums, old["state"], "cpu")
                state[meta["format"]] = [sums, list(old["docs"]) if old else [], path, False]
            sums, docs = state[meta["format"]][:2]
            if meta["key"] in docs:      # already summed (a crash between checkpoint and delete)
                pending.append(d)
                continue
            n0 = sums.n
            sums.add(convert.read_document(d), {}, "cpu")
            if sums.n - n0 != meta["rows"]:
                raise SystemExit(f"{meta['key']}: added {sums.n - n0} rows, its capture states {meta['rows']}")
            docs.append(meta["key"])
            state[meta["format"]][3] = True
            pending.append(d)
            added += 1
            log(f"acc {meta['key']}: +{meta['rows']} rows ({added} docs, {time.time() - t0:.0f}s)")
            if added % a.every == 0:
                checkpoint()
        if not new:
            if not a.follow or (store / DONE).exists() and not [d for d in ready(store, a.set) if d not in pending]:
                break
            time.sleep(10)
    checkpoint()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--what", choices=sorted(JSONL), required=True)
    r.add_argument("--prompts", nargs="+", required=True, help="docs.py's jsonl files (manifest.json beside them)")
    r.add_argument("--engine-dir", required=True, help="host side of the engine's capture directory mount")
    r.add_argument("--store", required=True, help="where each prompt's capture is filed")
    r.add_argument("--sums", default="", help="acc's sums dir: prompts its checkpoints hold are skipped")
    r.add_argument("--port", type=int, default=8100)
    r.add_argument("--chunk", type=int, default=2048, help="the served prefill chunk (--max-num-batched-tokens)")
    c = sub.add_parser("acc")
    c.add_argument("--store", required=True, help="run's store of projector captures")
    c.add_argument("--set", default="train", help="the prompt set to sum (held-out documents stay as captured)")
    c.add_argument("--sums", required=True)
    c.add_argument("--follow", action="store_true", help="keep polling until run-done.json and nothing left")
    c.add_argument("--every", type=int, default=40, help="checkpoint (then delete) every N documents")
    c.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    {"run": cmd_run, "acc": cmd_acc}[a.command](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
