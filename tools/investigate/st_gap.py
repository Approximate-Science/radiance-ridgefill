#!/usr/bin/env python3
"""The +st effect, radiance vs tcc, on the 6 quick-tier docs whose last 512 tokens are identical.

Prints per-doc st effects (st arm minus fill arm, nats, last 512 scored tokens), for radiance
(deterministic engine, same boot) and for every tcc pairing that exists on disk, and the paired
difference radiance - tcc with a bootstrap 95% CI over docs. Also tcc's same-arm repeat (big-st on two
boots) as its noise floor. Read-only.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rescore import EVID, ref_layout, read_rows, tcc_arm, doc_rows

rng = np.random.default_rng(20261004)


def boot(x, B=20000):
    x = np.asarray(x)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return [round(float(v), 4) for v in np.percentile(m, [2.5, 97.5])]


def main():
    docs, n = ref_layout()
    six = [d for d in docs if not d["source"].startswith("ppl/8k")]
    R = {a: read_rows(os.path.join(EVID, p), n) for a, p in
         {"fill": "stage3/kld-fill-f60f893.json", "st": "stage4/kld-st.json",
          "quality": "stage5/kld-quality.json"}.items()}
    ours = {a: np.array([doc_rows(R[a], d, last=512)[:, 1].astype(np.float64).mean()
                         - doc_rows(R[a], d, last=512)[:, 2].astype(np.float64).mean() for d in six])
            for a in R}
    T = {a: tcc_arm(p) for a, p in {"exact": "iter-01/exact", "nost02": "iter-02/big-nost",
                                     "st02": "iter-02/big-st", "st04": "iter-04/big-st",
                                     "q04": "iter-04/x24big-class56-s25-rarity-st"}.items()}
    tcc = {a: np.array([T[a][d["source"]]["nll_all"] - T["exact"][d["source"]]["nll_all"] for d in six])
           for a in T}
    out = {"docs": [d["source"] for d in six]}
    eff_r = ours["st"] - ours["fill"]
    out["radiance_st_effect"] = {"per_doc": eff_r.round(4).tolist(), "mean": round(eff_r.mean(), 4)}
    for name, (a, b) in {"tcc_same_boot_iter02": ("nost02", "st02"),
                         "tcc_cross_boot_02_04": ("nost02", "st04")}.items():
        e = tcc[b] - tcc[a]
        out[name] = {"per_doc": e.round(4).tolist(), "mean": round(e.mean(), 4), "ci": boot(e),
                     "radiance_minus_tcc": round((eff_r - e).mean(), 4), "ci_paired": boot(eff_r - e)}
    rep = tcc["st04"] - tcc["st02"]
    out["tcc_same_arm_repeat_st04_minus_st02"] = {"per_doc": rep.round(4).tolist(), "mean": round(rep.mean(), 4),
                                                   "sd_per_doc": round(rep.std(ddof=1), 4)}
    qr, qt = ours["quality"] - ours["st"], tcc["q04"] - tcc["st04"]
    out["quality_minus_st"] = {"radiance": round(qr.mean(), 4), "tcc_iter04_same_boot": round(qt.mean(), 4),
                               "radiance_minus_tcc": round((qr - qt).mean(), 4), "ci_paired": boot(qr - qt)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
