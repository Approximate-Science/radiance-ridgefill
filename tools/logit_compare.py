#!/usr/bin/env python3
"""logit_compare.py -- gate 1: a decoder's next-token distributions in two captures, token by token.

Each side is <dump dir>:<manifest.json>:<segment>:<sequence>, segment "batched" or "soloN" (the lines
logit_capture.py recorded), sequence the decoder's index in that request (its position in the batch;
0 for a solo). Per side, every logits row of that sequence is keyed by its position; the ranks' rows
are joined (identical rows: each rank holds the whole vocabulary; else concatenated in rank order).
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
    lines = open(os.path.join(dump, "logits.jsonl")).read().splitlines()[lo:hi]
    by_pos = {}
    for line in lines:
        j = json.loads(line)
        rows = np.load(os.path.join(dump, j["file"]))
        for out_j, row, s, pos, tok in j["rows"]:
            if s == int(seq):
                by_pos.setdefault(pos, {"token": tok, "ranks": {}})["ranks"][j["rank"]] = rows[out_j]
    out = {}
    for pos, v in by_pos.items():
        rs = [v["ranks"][r] for r in sorted(v["ranks"])]
        same = all(np.array_equal(rs[0], x) for x in rs[1:])
        out[pos] = (v["token"], rs[0] if same else np.concatenate(rs))
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
