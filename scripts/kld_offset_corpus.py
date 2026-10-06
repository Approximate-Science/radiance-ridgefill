#!/usr/bin/env python3
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""A KL-mode corpus whose docs end half a chunk past a chunk boundary (R49, R50, R51).

  kld_offset_corpus.py --corpus corpus/quick9.jsonl --tokenizer DIR --out corpus/quick9-off1024.jsonl
                       [--chunk 2048] [--offset 1024] [--tail 2048]

Each doc of an existing corpus (N tokens, a multiple of --chunk) is cut to N - chunk + offset tokens, so with
T = tail the chunk before the last one straddles the bulk end: its first `chunk - (T - offset)` rows are bulk
and the rest exact (PLAN-FIX §4). score_from = N' - tail, as the 2048-cut corpus scores, so a doc's per-row
ΔNLL is comparable doc for doc with that run's. The cut reuses tools/kld_corpus.py's decode/encode
round-trip search with `offset` as its quantum (N' is a multiple of offset). Writes <out> and
<out>.manifest.json with every doc's N', the expected approximate-step count and the cut's move.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kld_corpus as K  # noqa: E402
import ridgefill_rules as R  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, help="a kld_corpus.py corpus (every doc a multiple of --chunk)")
    ap.add_argument("--tokenizer", required=True, help="checkpoint dir holding the model's tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--offset", type=int, default=1024)
    ap.add_argument("--tail", type=int, default=2048)
    args = ap.parse_args(argv)
    if not 0 < args.offset < args.chunk or args.chunk % args.offset:
        raise SystemExit(f"--offset {args.offset} must divide --chunk {args.chunk} and be smaller")
    tok = R.load_tokenizer(args.tokenizer)
    docs = [json.loads(x) for x in Path(args.corpus).read_text(encoding="utf-8").splitlines() if x.strip()]
    lines, records = [], []
    for d in docs:
        n = len(K.encode(tok, d["prompt"]))
        if n % args.chunk:
            raise SystemExit(f"{d['source']}: {n} tokens is not a multiple of --chunk {args.chunk}")
        prompt, rec = K.cut_doc(tok, d["prompt"], args.offset, n - args.chunk + args.offset)
        n2 = rec["n"]
        steps = -(-(n2 - args.tail) // args.chunk)   # ceil: the bulk chunks, the straddling one included
        lines.append(json.dumps({"prompt": prompt, "score_from": n2 - args.tail, "source": d["source"]}))
        records.append(dict(source=d["source"], n_from=n, n=n2, score_from=n2 - args.tail, approx_steps=steps,
                            move=rec["move"], ids_equal=rec["ids_equal"]))
        print(f"{d['source']:12s} N {n:6d} -> {n2:6d}  score_from {n2 - args.tail:6d}  approx steps {steps}  "
              f"move {rec['move']:+d}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = dict(corpus=str(args.corpus), corpus_sha256=hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
                    chunk=args.chunk, offset=args.offset, tail=args.tail,
                    out_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    approx_steps_total=sum(r["approx_steps"] for r in records), docs=records)
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(lines)} docs, {manifest['approx_steps_total']} approximate steps expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
