#!/usr/bin/env python3
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""R45: the engine's per-chunk mask dump against the rule, transcribed here apart from the plugin.

  mask_rule.py --dump DIR --sidecar F [--lengths 9216,10000,...] [--chunk 2048] [--tail 2048] [--tile 64]

DIR holds the RADIANCE_RIDGEFILL_DUMP files of a quality-mode run of one-sequence prompts (prefix cache off, no
decoders): mask.jsonl and rows.jsonl, one line per masked approximate chunk, in step order. For each prompt
(a chunk_start of 0 begins one) the rule is recomputed from N alone:

  chunks [p, p + min(chunk, N - p)), n_ahead = min(N - end, chunk);
  b = n_tok if n_ahead >= T else n_tok - ceil_tile(T - n_ahead); approximated iff n_ahead > 0 and b > 0;
  mask = 0 on [b, n_tok); on [0, b) the class rule (tools/ridgefill_rules.select_rows over ids[0:b] with the
  sidecar's score table and share); bounds = {0, b, b, n_tok}.

N is each prompt's entry of --lengths, in send order (a capped n_ahead makes N unrecoverable from the dump:
16,384 and 16,385 dump alike); the dump's own largest end + n_ahead must be consistent with it. Every expected approximate chunk must be present (a missing one is a replayed pass: rerun with
--profile-ops) and every dumped field must equal the rule. Exits 1 on any difference, naming it.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ridgefill_rules as R  # noqa: E402
from safetensors import safe_open  # noqa: E402


def expected_chunks(n, chunk, tail, tile):
    """[(start, n_tok, n_ahead, b)] for every approximated chunk of an N-token prompt."""
    out = []
    for p in range(0, n, chunk):
        n_tok = min(chunk, n - p)
        ahead = min(n - p - n_tok, chunk)
        b = n_tok if ahead >= tail else n_tok - -(-(tail - ahead) // tile) * tile
        if ahead > 0 and b > 0:
            out.append((p, n_tok, ahead, b))
    return out


def prompts(mask_lines, rows_lines):
    """Group the dump's chunks by prompt: a chunk_start of 0 begins one."""
    groups = []
    for m, r in zip(mask_lines, rows_lines):
        if m["chunk_start"] == 0:
            groups.append([])
        groups[-1].append((m, r))
    return groups


def check_prompt(group, n, score, share, args):
    errors = []
    seen = max(m["chunk_start"] + m["n_tok"] + m["n_ahead"] for m, _ in group)
    if seen > n or (seen < n and seen - group[-1][0]["chunk_start"] - group[-1][0]["n_tok"] < args.chunk):
        errors.append(f"the dump reaches {seen} tokens, inconsistent with N {n}")
    want = expected_chunks(n, args.chunk, args.tail, args.tile)
    got = {m["chunk_start"]: (m, r) for m, r in group}
    if sorted(got) != [w[0] for w in want]:
        errors.append(f"chunks {sorted(got)} != expected {[w[0] for w in want]}")
    for p, n_tok, ahead, b in want:
        if p not in got:
            continue
        m, r = got[p]
        ids = r["token_ids"]
        rule = R.select_rows(ids[:b], score, share)
        mask = "".join("0" if i >= b or i in set(rule) else "1" for i in range(n_tok))
        for key, have, need in (("n_tok", m["n_tok"], n_tok), ("n_ahead", m["n_ahead"], ahead), ("b", m["b"], b),
                                ("bounds", m["bounds"], [0, b, b, n_tok]), ("mask", m["mask"], mask),
                                ("rows_idx", r["rows_idx"], rule)):
            if have != need:
                errors.append(f"chunk {p}: {key} differs" + ("" if key in ("mask", "rows_idx") else f": {have} != {need}"))
    print(f"N {n:6d}: {len(want)} approximate chunks, b per chunk {[w[3] for w in want]}, "
          f"exact class rows {sum(len(got[w[0]][1]['rows_idx']) for w in want if w[0] in got)}: "
          f"{'EQUAL to the rule' if not errors else 'DIFFERS'}")
    for e in errors:
        print("   ", e)
    return not errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--sidecar", required=True, help="the sidecar safetensors holding ridgefill.rowsel.score")
    ap.add_argument("--lengths", required=True, help="comma-separated prompt lengths, in the order the run sent them")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--tail", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=64)
    args = ap.parse_args(argv)
    with safe_open(args.sidecar, "np") as f:
        score = f.get_tensor("ridgefill.rowsel.score")
        share = float((f.metadata() or {})["ridgefill.rowsel.share"])
    read = lambda name: [json.loads(x) for x in (Path(args.dump) / name).read_text().splitlines() if x.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x]
    groups = prompts(read("mask.jsonl"), read("rows.jsonl"))
    if len(groups) != len(lengths):
        print(f"{len(groups)} prompts in the dump, {len(lengths)} sent")
        return 1
    ok = [check_prompt(g, n, score, share, args) for g, n in zip(groups, lengths)]
    print(f"{sum(ok)} of {len(ok)} prompts equal the rule")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
