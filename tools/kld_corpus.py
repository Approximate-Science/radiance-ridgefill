#!/usr/bin/env python3
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""ppl.jsonl -> a radiance KL-mode corpus whose every doc is a whole number of prefill chunks.

  kld_corpus.py --ppl ppl.jsonl --tokenizer DIR --out corpus/quick9.jsonl [--chunk 2048] [--tail 2048]
                [--max-tokens 32768]

Each doc is cut to the largest multiple of --chunk tokens it holds (at most --max-tokens), so its last chunk
is exactly the exact tail and `score_from = N - tail` is that chunk's first row (HANDOVER Stage 0.6). A KL line
is {"prompt", "score_from", "source"} (radiance docs/TOOLS.md, "The KL mode"); the engine re-tokenises
`prompt`, so the cut ids are decoded and re-encoded and the re-encoded length must be exactly N: when the
decode/encode round trip is not stable at the cut, the cut moves by a few tokens until it is, and the move is
recorded. Per-doc N, the move and whether the re-encoded ids equal the original prefix go to
<out>.manifest.json and stdout.

Defaults are the measurement protocol's: 2048-token chunks (--max-num-batched-tokens 2048), T = 2048, and the
32,768-token cap of the longest quick-tier bucket.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ridgefill_rules as R  # noqa: E402

SEARCH = 64     # how far the cut may move to find a round-trip-stable boundary before the doc is refused


def encode(tok, text):
    """Ids as the engine's KL mode sees a raw prompt: no BOS added, special-token strings recognised."""
    return tok(text, add_special_tokens=False).input_ids


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def cut_doc(tok, text, chunk, max_tokens):
    """(prompt text, record): the longest prefix that re-encodes to exactly a multiple of `chunk` tokens."""
    ids = encode(tok, text)
    target = min(len(ids), max_tokens) // chunk * chunk
    for move in [0] + [m for d in range(1, SEARCH + 1) for m in (-d, d)]:
        cut = target + move
        if cut <= 0 or cut > len(ids):
            continue
        prompt = decode(tok, ids[:cut])
        again = encode(tok, prompt)
        if len(again) == target:
            return prompt, dict(n_original=len(ids), n=target, cut_at=cut, move=move, ids_equal=again == ids[:target])
    raise SystemExit(f"no cut within {SEARCH} tokens of {target} re-encodes to exactly {target} tokens")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ppl", required=True, help="ppl.jsonl: one {id, prompt, ...} per line")
    ap.add_argument("--tokenizer", required=True, help="checkpoint dir holding the model's tokenizer")
    ap.add_argument("--out", required=True, help="corpus JSONL to write (its directory is created)")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk = --max-num-batched-tokens")
    ap.add_argument("--tail", type=int, default=2048, help="exact tail T; score_from = N - T")
    ap.add_argument("--max-tokens", type=int, default=32768, help="cap on a doc's length")
    args = ap.parse_args(argv)
    ppl, out = Path(args.ppl), Path(args.out)
    if not ppl.is_file():
        raise SystemExit(f"--ppl {ppl} not found")
    if args.tail % args.chunk or args.tail > args.max_tokens - args.chunk:
        raise SystemExit(f"--tail {args.tail} must be a multiple of --chunk {args.chunk} with a bulk chunk before it")
    tok = R.load_tokenizer(args.tokenizer)
    docs = [json.loads(line) for line in ppl.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines, records = [], []
    for doc in docs:
        prompt, rec = cut_doc(tok, doc["prompt"], args.chunk, args.max_tokens)
        if rec["n"] - args.tail < args.chunk:
            raise SystemExit(f"{doc['id']}: {rec['n']} tokens leave no bulk chunk before a {args.tail}-token tail")
        lines.append(json.dumps({"prompt": prompt, "score_from": rec["n"] - args.tail, "source": doc["id"]}))
        records.append(dict(source=doc["id"], score_from=rec["n"] - args.tail,
                            bulk_chunks=(rec["n"] - args.tail) // args.chunk, **rec))
        print(f"{doc['id']:12s} N_orig {rec['n_original']:6d}  N {rec['n']:6d}  score_from {rec['n'] - args.tail:6d}  "
              f"move {rec['move']:+d}  ids_equal {rec['ids_equal']}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = dict(ppl=ppl.name, ppl_sha256=hashlib.sha256(ppl.read_bytes()).hexdigest(), chunk=args.chunk,
                    tail=args.tail, max_tokens=args.max_tokens, corpus_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    bulk_chunks_total=sum(r["bulk_chunks"] for r in records), docs=records)
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} docs; {manifest['bulk_chunks_total']} bulk chunks = the approximate-step "
          f"count R18 expects) and {out}.manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
