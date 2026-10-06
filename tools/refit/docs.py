#!/usr/bin/env python3
"""The Stage 6 refit's prompts, tokenised once: the projector's training and held-out documents (each raw AND as a
chat user turn) and the +st fit prompts. Writes <out>/{train,held,sterm}.jsonl (one prompt a line, token ids
included) and <out>/manifest.json (counts, tokens, the sha256 of every text and of every file written).

  docs.py build   --plans DIR --tokenizer tokenizer.json --overlap overlap.json --out DIR
  docs.py overlap --prompts DIR --suite DIR [--locked DIR]    (word 50-gram check of the held + sterm texts)

Which documents, and why (KVA-FACTS §4, §7; notes/refit.md §1):
  train  the shipped fit's own documents: FN-BIGCAP batch b0 (FN-CAP2's 71 overlap-clean calibration documents) and
         batch b1 (80 impact-corpus documents), so the refit differs from the shipped fit in the engine whose
         activations it reads and not in its data. Each raw and as a chat user turn (the shipped fit chat-copied 22
         of b0's 71: FN-CAP2's time box); the chat share is set by row weighting at the solve either way.
  held   the shipped fit's held-out set: FN-CAP2's 5 held documents, raw and chat (the lambda choice).
  sterm  plan-sterm's prompts minus the protocol-1.1 exclusions (sterm4 shares 50-grams with the raw LongBench pool),
         plus clean impact documents of batch b2 (in neither the projector's training nor its held set) cut to
         --extra-tokens, until there are >= MIN_PROMPTS prompts and >= MIN_CHUNK_ENDS approximate chunk ends (R43),
         plus one prompt of margin. Radiance approximates whole chunks only (a chunk with >= tail tokens after it),
         so an N-token prompt gives floor((N - tail) / chunk) chunk ends, fewer than tcc's clipped chunks gave.
Chat text = qfn.steps.chat_text(text, i) with i = the document's index in FN-CAP2's capture order (its 99 calibration
documents in plan order, then its held documents), as FN-CAP2 and FN-BIGCAP built their chat copies (checked against
FN-BIGCAP's stored chat texts); batch b1's chat texts are taken from its plan file as captured.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ridgefill_research  # noqa: E402

MIN_PROMPTS = 13        # R43: the shipped +st fit's prompt count
MIN_CHUNK_ENDS = 50     # R43: chunk ends per late GDN layer
PROBE_CHARS = 4000      # the tokenizer probe: the start of the first raw and first chat prompt, re-tokenised by the
                        # served engine's /tokenize before any capture (capture.py refuses on a difference)


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_ends(n_tokens, tail, chunk):
    """Approximate chunk ends of an n-token prompt: chunks [k*chunk, (k+1)*chunk) with >= tail tokens after them."""
    return max(0, (n_tokens - tail) // chunk)


def fncap2_order(plans_dir):
    """{name: (index, text)} in FN-CAP2's capture order: plan-train's calibration documents, then the held ones."""
    from ridgefill import data as kd
    train = [x for x in json.loads((kd.INPUTS / "plan-train.json").read_text()) if x["kind"] == "calib"]
    held = kd.calib_docs("held")
    if len(train) != 99 or len(held) != 5:
        raise SystemExit(f"FN-CAP2 order: {len(train)} calibration + {len(held)} held documents, expected 99 + 5")
    return {d["name"]: (i, d["text"]) for i, d in enumerate(train + held)}


def train_items(plans_dir, excluded):
    """[(source, name, format, text)] for batch b0 (raw + chat) and batch b1 (raw + chat as captured)."""
    from qfn.steps import chat_text
    order = fncap2_order(plans_dir)
    b0 = [d["name"] for d in json.loads((plans_dir / "PLAN-b0.json").read_text()) if d["kind"] == "train"]
    stored = {d["name"]: d["text"] for d in json.loads((plans_dir / "PLAN-b4.json").read_text())
              if d["kind"] == "chatold"}
    items = []
    for name in b0:
        if f"calib-train-{name}" in excluded:
            raise SystemExit(f"b0 document {name} is on the overlap exclusion list")
        index, text = order[name]
        chat = chat_text(text, index)
        if name in stored and stored[name] != chat:
            raise SystemExit(f"{name}: chat text differs from FN-BIGCAP's stored copy (index {index})")
        items += [("fncap2", name, "raw", text), ("fncap2", name, "chat", chat)]
    for d in json.loads((plans_dir / "PLAN-b1.json").read_text()):
        if f"impact-{d['name']}" in excluded:
            raise SystemExit(f"b1 document {d['name']} is on the overlap exclusion list")
        if d["kind"] == "big" and sha256_text(d["text"]) != d["sha256"]:
            raise SystemExit(f"b1 {d['name']}: text differs from its manifest sha256")
        items.append(("impact", d["name"], "raw" if d["kind"] == "big" else "chat", d["text"]))
    return items


def held_items(plans_dir):
    from ridgefill import data as kd
    from qfn.steps import chat_text
    order = fncap2_order(plans_dir)
    items = []
    for d in kd.calib_docs("held"):
        index, text = order[d["name"]]
        items += [("fncap2", d["name"], "raw", text), ("fncap2", d["name"], "chat", chat_text(text, index))]
    return items


def sterm_items(plans_dir, encode, tail, chunk, extra_tokens):
    """[(source, name, format, text, cut)]: plan-sterm minus exclusions, then b2 impact documents cut to extra_tokens."""
    from ridgefill import data as kd
    items = [("sterm", d["name"], "raw", d["text"], None) for d in kd.calib_docs("sterm")]
    ends = sum(chunk_ends(len(encode(t)), tail, chunk) for *_, t, _ in items)
    extra = (d for d in json.loads((plans_dir / "PLAN-b2.json").read_text())
             if d["kind"] == "big" and d["tokens"] >= extra_tokens)
    margin = 1                           # one prompt more than the minimum, in case a chunk end goes unmatched
    while margin >= 0:
        if len(items) >= MIN_PROMPTS and ends >= MIN_CHUNK_ENDS:
            margin -= 1
            if margin < 0:
                break
        d = next(extra)
        items.append(("impact", d["name"], "raw", d["text"], extra_tokens))
        ends += chunk_ends(extra_tokens, tail, chunk)
    return items, ends


def write_set(out, name, items, encode, max_prompt, tail, chunk):
    rows = []
    for source, doc, fmt, text, *cut in items:
        ids = encode(text)
        limit = min([max_prompt] + [c for c in cut if c])
        rows.append(dict(set=name, key=f"{name}/{fmt}/{source}-{doc}", source=source, name=doc, format=fmt,
                         text_sha256=sha256_text(text), text_tokens=len(ids), cut=len(ids) > limit,
                         n_tokens=min(len(ids), limit), chunk_ends=chunk_ends(min(len(ids), limit), tail, chunk),
                         ids=ids[:limit]))
    path = out / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path, rows


def cmd_build(a):
    ridgefill_research.root()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids  # noqa: E731
    plans, out = Path(a.plans), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    excluded = set(json.loads(Path(a.overlap).read_text())["excluded"])
    sterm, ends = sterm_items(plans, encode, a.tail, a.chunk, a.extra_tokens)
    sets = {"train": train_items(plans, excluded), "held": held_items(plans), "sterm": sterm}
    manifest = dict(tokenizer_sha256=hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest(),
                    max_prompt=a.max_prompt, tail=a.tail, chunk=a.chunk, extra_tokens=a.extra_tokens, sets={})
    for name, items in sets.items():
        path, rows = write_set(out, name, items, encode, a.max_prompt, a.tail, a.chunk)
        manifest["sets"][name] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      prompts=len(rows), tokens=sum(r["n_tokens"] for r in rows),
                                      raw=sum(r["format"] == "raw" for r in rows),
                                      chat=sum(r["format"] == "chat" for r in rows), cut=sum(r["cut"] for r in rows),
                                      chunk_ends=sum(r["chunk_ends"] for r in rows),
                                      docs=[dict(key=r["key"], text_sha256=r["text_sha256"], n_tokens=r["n_tokens"])
                                            for r in rows])
        print(f"{name}: {len(rows)} prompts, {manifest['sets'][name]['tokens']} tokens "
              f"(raw {manifest['sets'][name]['raw']}, chat {manifest['sets'][name]['chat']}, "
              f"cut {manifest['sets'][name]['cut']}) -> {path}")
    print(f"sterm: {ends} approximate chunk ends at tail {a.tail}, chunk {a.chunk}")
    (*_, raw), (*_, chat) = sets["train"][:2]           # one raw prompt; one chat prompt's head + tail (both frames)
    probe = [raw[:PROBE_CHARS], chat[:PROBE_CHARS // 2] + chat[-PROBE_CHARS // 2:]]
    manifest["probe"] = [dict(text=t, ids=encode(t)) for t in probe]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def cmd_overlap(a):
    """Every held and sterm text against the suite (and the locked set) with FN-BIGCAP's own 50-gram check."""
    ridgefill_research.root()
    from ridgefill import data as kd
    from qfn import bigcap_overlap as ov
    prompts = Path(a.prompts)
    keys = {r["key"] for s in ("held", "sterm") for r in map(json.loads, open(prompts / f"{s}.jsonl"))}
    texts = {f"sterm-{d['name']}": d["text"] for d in json.loads((kd.INPUTS / "plan-sterm.json").read_text())}
    texts.update({f"held-{d['name']}": d["text"] for d in kd.calib_docs("held")})
    plans = Path(a.plans)
    texts.update({f"impact-{d['name']}": d["text"] for d in json.loads((plans / "PLAN-b2.json").read_text())
                  if d["kind"] == "big" and f"sterm/raw/impact-{d['name']}" in keys})
    files = ov.eval_files(Path(a.suite), Path(a.locked)) if a.locked else \
        [p for p in sorted(Path(a.suite).rglob("*")) if p.is_file() and p.suffix in ov.EXTS and p.name != "MANIFEST.json"]
    hits, scanned = ov.check(texts, files)
    report = dict(method="qfn.bigcap_overlap word 50-grams", texts=sorted(texts), eval_files=len(files),
                  eval_grams=sum(scanned.values()), hits=hits, overlapping=sorted(hits))
    (prompts / "overlap.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{len(texts)} texts vs {len(files)} files: overlapping {sorted(hits)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--plans", required=True, help="FN-BIGCAP plan dir (PLAN-b0/b1/b2/b4.json)")
    b.add_argument("--tokenizer", required=True, help="the checkpoint's tokenizer.json (the container's vocab source)")
    b.add_argument("--overlap", required=True, help="FN-BIGCAP's overlap.json (its 'excluded' list is refused)")
    b.add_argument("--out", required=True)
    b.add_argument("--max-prompt", type=int, default=49151,
                   help="longest prompt: the served --max-model-len minus the one generated token")
    b.add_argument("--tail", type=int, default=512, help="+st fit tail (tcc's FIT_TAIL, b0/drive.py)")
    b.add_argument("--chunk", type=int, default=2048, help="the served prefill chunk (--max-num-batched-tokens)")
    b.add_argument("--extra-tokens", type=int, default=10240, help="length of each added sterm prompt")
    o = sub.add_parser("overlap")
    o.add_argument("--prompts", required=True, help="the build's output dir")
    o.add_argument("--plans", required=True, help="FN-BIGCAP plan dir (for the added sterm texts)")
    o.add_argument("--suite", required=True, help="the Flash-Next suite's data dir")
    o.add_argument("--locked", default="", help="the locked test set dir (optional)")
    a = ap.parse_args(argv)
    {"build": cmd_build, "overlap": cmd_overlap}[a.command](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
