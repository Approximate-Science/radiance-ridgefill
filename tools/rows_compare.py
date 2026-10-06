#!/usr/bin/env python3
"""Quality mode's exact rows, checked against the paper's rule (research repo fnlev/rules.py, the oracle).

  rows_compare.py fixture --ppl ppl.jsonl --out FIXTURE.json  COMMON
      R33: for each doc, its first --window token ids and the rows fnlev.rules keeps when that window is the
      whole prompt; the kernel test feeds the ids to ridgefill_rowsel and must get these rows. The fixture also
      carries the scores of the kept ids it uses (every other id is -inf), so the test needs no sidecar.
  rows_compare.py compare --corpus corpus.jsonl (--dump rows.jsonl | --simulate) [--tail T] [--out R.json]  COMMON
      R39: per doc, the rows chosen chunk by chunk (from the engine's dump, or simulated with the same rule on
      each --window chunk) vs fnlev.rules over the whole prompt at P = N - T, as a Jaccard index.
  COMMON = --tokenizer DIR --sidecar ridgefill-sidecar.safetensors --freq F --fnlev-root RESEARCH_REPO [--window 2048]

The share and classes come from the sidecar's metadata; the sidecar's freq hash must equal --freq's.

Engine dump format (RADIANCE_RIDGEFILL_DUMP=<dir> writes <dir>/rows.jsonl), one JSON object per approximate chunk:
  {"chunk_start": absolute position of the chunk's first row, "n_tok": rows in the chunk,
   "rows_idx": chunk-relative selected rows, ascending (-1 padding may be kept; it is dropped here),
   "token_ids": the chunk's n_tok token ids}
Each chunk is matched to its corpus doc by its token ids at chunk_start, so docs may arrive in any order.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ridgefill_rules as R  # noqa: E402

FIXTURE_FORMAT = "ridgefill-rowsel-fixture-1"


def load_sidecar(path, freq):
    """(score [vocab] f32, share, classes, sha256 of the score bytes) from the sidecar."""
    with safe_open(str(path), "np") as f:
        meta = f.metadata() or {}
        score = f.get_tensor("ridgefill.rowsel.score")
    if meta.get("ridgefill.src.freq.sha256") != hashlib.sha256(Path(freq).read_bytes()).hexdigest():
        raise SystemExit(f"{path} was built from another unigram table than --freq {freq}")
    score = np.ascontiguousarray(score, dtype="<f4")
    return (score, float(meta["ridgefill.rowsel.share"]), meta["ridgefill.rowsel.classes"].split(","),
            hashlib.sha256(score.tobytes()).hexdigest())


def load_oracle(fnlev_root, tokenizer_dir, freq, tok):
    """fnlev.rules.Rules from the research repo: the code that made the paper's row lists."""
    if not (Path(fnlev_root) / "fnlev" / "rules.py").is_file():
        raise SystemExit(f"--fnlev-root {fnlev_root}: no fnlev/rules.py (expected the RidgeFill research repo)")
    sys.path.insert(0, str(fnlev_root))
    from fnlev import rules
    return rules.Rules(model_dir=tokenizer_dir, freq=freq, tok=tok)


def encode(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def chunk_rows(ids, P, window, score, share):
    """Absolute rows kept when every whole --window chunk before P selects on its own (D11)."""
    return [c + r for c in range(0, P - window + 1, window) for r in R.select_rows(ids[c:c + window], score, share)]


def fixture(args, tok, oracle, score, share, classes, score_sha):
    spec = {"rule": "class56", "share": share, "classes": classes}
    ppl = Path(args.ppl)
    docs = []
    for line in ppl.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        ids = encode(tok, doc["prompt"])[:args.window]
        rows, info = oracle.rows(spec, ids, len(ids), doc["id"])
        ours = R.select_rows(ids, score, share)
        if rows != ours:
            raise SystemExit(f"{doc['id']}: the sidecar's table selects {len(ours)} rows, fnlev.rules {len(rows)}; "
                             f"first difference {sorted(set(rows) ^ set(ours))[:5]}")
        docs.append(dict(doc=doc["id"], token_ids=ids, matches=info["matches"], k=len(rows), rows=rows,
                         random_count=len(rows)))
        print(f"{doc['id']:12s} window {len(ids)}  matches {info['matches']:4d}  k {len(rows):3d}  "
              f"({len(rows) / len(ids):.1%} of rows)  transcription == fnlev.rules")
    # The kept ids of these windows and their scores, so a test can rebuild the table these rows need
    # (every other id is -inf) without the 1.3 GB sidecar; score_sha256 ties them to the full table.
    kept = sorted({t for d in docs for t in d["token_ids"] if np.isfinite(score[t])})
    out = dict(format=FIXTURE_FORMAT, window=args.window, share=share, classes=classes,
               score_tensor="ridgefill.rowsel.score", score_sha256=score_sha, vocab=len(score),
               kept_ids=kept, kept_scores=[float(score[t]) for t in kept],
               ppl=ppl.name, ppl_sha256=hashlib.sha256(ppl.read_bytes()).hexdigest(),
               rule="fnlev/rules.py class56, rank rarity (table), ties by position; k = round(share * matches), "
                    "Python round = half to even", docs=docs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


def read_dump(path, docs):
    """{doc index: set of absolute rows} and {doc index: set of chunk starts} from the engine's rows.jsonl."""
    rows, starts = {}, {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        rec = json.loads(line)
        start, ids = rec["chunk_start"], rec["token_ids"]
        owners = [i for i, d in enumerate(docs) if d["ids"][start:start + len(ids)] == ids]
        if len(owners) != 1:
            raise SystemExit(f"{path}:{number}: chunk at {start} matches {len(owners)} corpus docs (need exactly 1)")
        if rec["n_tok"] != len(ids) or any(r >= len(ids) for r in rec["rows_idx"]):
            raise SystemExit(f"{path}:{number}: n_tok {rec['n_tok']} / rows_idx out of range for {len(ids)} ids")
        rows.setdefault(owners[0], set()).update(start + r for r in rec["rows_idx"] if r >= 0)
        starts.setdefault(owners[0], set()).add(start)
    return rows, starts


def compare(args, tok, oracle, score, share, classes):
    spec = {"rule": "class56", "share": share, "classes": classes}
    docs = []
    for line in Path(args.corpus).read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        ids = encode(tok, doc["prompt"])
        docs.append(dict(source=doc["source"], ids=ids, P=len(ids) - args.tail, score_from=doc["score_from"]))
    dumped, starts = read_dump(args.dump, docs) if args.dump else ({}, {})
    report = []
    for i, d in enumerate(docs):
        whole = set(oracle.rows(spec, d["ids"], d["P"], d["source"])[0])
        simulated = set(chunk_rows(d["ids"], d["P"], args.window, score, share))
        chunked = dumped.get(i, set()) if args.dump else simulated
        union = whole | chunked
        rec = dict(source=d["source"], N=len(d["ids"]), P=d["P"], score_from_matches=d["score_from"] == d["P"],
                   whole=len(whole), chunked=len(chunked), both=len(whole & chunked),
                   jaccard=len(whole & chunked) / len(union) if union else 1.0, chunked_share=len(chunked) / d["P"])
        if args.dump:
            want = set(range(0, d["P"] - args.window + 1, args.window))
            rec.update(dump_equals_simulated=chunked == simulated, chunks_missing=sorted(want - starts.get(i, set())))
        report.append(rec)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    mean = sum(r["jaccard"] for r in report) / len(report)
    share_rows = sum(r["chunked"] for r in report) / sum(r["P"] for r in report)
    print(f"mean Jaccard {mean:.4f} over {len(report)} docs ({'engine dump' if args.dump else 'simulated chunks'}); "
          f"in-chunk rows {share_rows:.2%} of bulk rows")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(mean_jaccard=mean, rows_share=share_rows, window=args.window,
                                                  tail=args.tail, source="dump" if args.dump else "simulated",
                                                  docs=report), indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("fixture", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--tokenizer", required=True, help="checkpoint dir holding the model's tokenizer")
        p.add_argument("--sidecar", required=True, help="ridgefill-sidecar.safetensors (score table, share, classes)")
        p.add_argument("--freq", required=True, help="the unigram table the sidecar was built from")
        p.add_argument("--fnlev-root", required=True, help="RidgeFill research repo root (holds fnlev/rules.py)")
        p.add_argument("--window", type=int, default=2048, help="prefill chunk = --max-num-batched-tokens")
        if name == "fixture":
            p.add_argument("--ppl", required=True, help="ppl.jsonl: one {id, prompt} per line")
            p.add_argument("--out", required=True, help="fixture JSON to write")
        else:
            p.add_argument("--corpus", required=True, help="KL corpus from tools/kld_corpus.py")
            how = p.add_mutually_exclusive_group(required=True)
            how.add_argument("--dump", help="the engine's rows.jsonl (format in this file's header)")
            how.add_argument("--simulate", action="store_true", help="select per chunk here, with the same rule")
            p.add_argument("--tail", type=int, default=2048, help="exact tail T; P = N - T")
            p.add_argument("--out", help="report JSON")
    args = ap.parse_args(argv)
    for flag in ("sidecar", "freq", "ppl", "corpus", "dump"):
        value = getattr(args, flag, None)
        if value and not Path(value).is_file():
            raise SystemExit(f"--{flag} {value} not found")
    score, share, classes, score_sha = load_sidecar(args.sidecar, args.freq)
    tok = R.load_tokenizer(args.tokenizer)
    oracle = load_oracle(args.fnlev_root, args.tokenizer, args.freq, tok)
    if args.command == "fixture":
        fixture(args, tok, oracle, score, share, classes, score_sha)
    else:
        compare(args, tok, oracle, score, share, classes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
