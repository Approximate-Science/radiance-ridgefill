#!/usr/bin/env python3
"""Lever check (not the gap): does a correction fitted only on LATE chunk ends serve late tail starts better?
The tail of a long prompt starts at P = 14,336 or 30,720, while the fit's chunk ends sit at 2,048..8,192, and the
first chunk end (2,048: the fill started from a zero state) has a different mean error (state_err.py
position_dependence). Scores, at chunk ends >= 6,144 only, the error energy removed by: shipped C, the LOO radiance
mean over all ends, and the LOO radiance mean over ends >= 4,096. Same captures as state_err.py. JSON to stdout."""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from state_err import records, load_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--shipped", nargs=2, required=True)
    a = ap.parse_args()
    ex, pr = records(a.exact), records(a.pred)
    ship = [load_c(p) for p in a.shipped]
    E = collections.defaultdict(list)   # (prompt, rank) -> [(end, E [18,24,128,128])]
    layers = None
    for k, r in sorted(pr.items()):
        if not r["approximate"]:
            continue
        layers = r["layers"]
        E[(k[0], k[3])].append((k[2] + 1, np.load(ex[k]["path"]).astype(np.float64) - np.load(r["path"]).astype(np.float64)))
    tot = collections.defaultdict(float)
    for rank in (0, 1):
        keys = [k for k in E if k[1] == rank]
        Cs = np.stack([ship[rank][L] for L in layers])
        for k in keys:
            rest_all = [e for kk in keys if kk != k for (end, e) in E[kk]]
            rest_late = [e for kk in keys if kk != k for (end, e) in E[kk] if end >= 4096]
            C_all, C_late = np.mean(rest_all, axis=0), np.mean(rest_late, axis=0)
            for end, e in E[k]:
                if end < 6144:
                    continue
                tot["ee"] += float((e ** 2).sum())
                tot["ship"] += float(((e - Cs) ** 2).sum())
                tot["all"] += float(((e - C_all) ** 2).sum())
                tot["late"] += float(((e - C_late) ** 2).sum())
                tot["n"] += 1
    print(json.dumps({"late_ends_scored": int(tot["n"]),
                      "removed_at_late_ends": {"shipped": round(1 - tot["ship"] / tot["ee"], 4),
                                               "loo_all_ends": round(1 - tot["all"] / tot["ee"], 4),
                                               "loo_ends_ge_4096": round(1 - tot["late"] / tot["ee"], 4)}}, indent=1))


if __name__ == "__main__":
    main()
