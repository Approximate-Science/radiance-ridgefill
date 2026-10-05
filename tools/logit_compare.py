#!/usr/bin/env python3
"""logit_compare.py -- gate 1: a decoder's next-token distributions in two captures, token by token.

Each side is <dump dir>:<manifest.json>:<segment>:<sequence>, segment "batched" or "soloN" (the lines
logit_capture.py recorded), sequence the decoder's index in that request's FIRST step (its place in the
batch; 0 for a solo). A decoder is then FOLLOWED, not looked up by index -- once the long prompt starts
decoding the engine reorders the decode entries, so an index can point at another sequence: its row
in a later step is the one at position + 1 whose input token is the greedy pick of its previous row
(temperature 0), preferring the same index on a tie. The ranks' rows are joined (identical rows: each
rank holds the whole vocabulary; else concatenated in rank order).
Compared position by position while the two sides' input tokens agree (after the first input
difference the contexts differ and nothing is comparable): KL(A || B) of the softmax, and top-1
agreement. Prints n, mean / max KL, top-1 agreement, and where the inputs first differ.

usage: logit_compare.py A_SIDE B_SIDE
"""
import json
import os
import sys

import numpy as np


def side(spec):
    dump, manifest, seg, seq = spec.rsplit(":", 3)
    m = json.load(open(manifest))
    lo, hi = m["batched"] if seg == "batched" else m["solo"][int(seg[4:])]
    lines = [json.loads(x) for x in open(os.path.join(dump, "logits.jsonl")).read().splitlines()[lo:hi]]
    steps = {}   # step file key -> {rank: (rows array, row list)}, in order
    for j in lines:
        steps.setdefault(j["file"].rsplit(".r", 1)[0], {})[j["rank"]] = (np.load(os.path.join(dump, j["file"])), j["rows"])
    out, want_pos, want_tok, want_seq = {}, None, None, int(seq)
    for ranks in steps.values():
        rows0 = ranks[min(ranks)][1]
        cands = [r for r in rows0 if (want_pos is None and r[2] == want_seq) or
                 (want_pos is not None and r[3] == want_pos and r[4] == want_tok)]
        if not cands:
            continue
        r = next((c for c in cands if c[2] == want_seq), cands[0])
        parts = [ranks[k][0][r[0]] for k in sorted(ranks)]
        same = all(np.array_equal(parts[0], x) for x in parts[1:])
        logits = parts[0] if same else np.concatenate(parts)
        out[r[3]] = (r[4], logits)
        want_pos, want_tok, want_seq = r[3] + 1, int(logits.argmax()), r[2]
    return out


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main():
    a, b = side(sys.argv[1]), side(sys.argv[2])
    kls, agree, first_diff = [], 0, None
    for pos in sorted(set(a) & set(b)):
        (ta, la), (tb, lb) = a[pos], b[pos]
        if ta != tb:
            first_diff = pos
            break
        pa, pb = log_softmax(la), log_softmax(lb)
        kls.append(float(np.sum(np.exp(pa) * (pa - pb))))
        agree += int(la.argmax() == lb.argmax())
    n = len(kls)
    print(json.dumps({"n": n, "kl_mean": float(np.mean(kls)) if n else None, "kl_max": max(kls) if n else None,
                      "top1": agree / n if n else None, "first_input_difference": first_diff,
                      "bytes_identical": n > 0 and all(k == 0.0 for k in kls)}))


if __name__ == "__main__":
    main()
