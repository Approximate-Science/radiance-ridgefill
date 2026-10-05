#!/usr/bin/env python3
"""INV-final13: radiance fill / fill+st on tcc's 13 final-tier 16k+32k docs vs tcc's same-server final tier.

Radiance: KL-mode .rows (col 1 candidate NLL, col 2 reference NLL) against a stock-engine reference recorded on the
same boot; tcc: tests/results/final/{exact,ours-nost,ours}/ppl.jsonl, one server session, last 512 tokens.
Checks the last-512 token ids are the same, then prints per-doc deltas and paired bootstrap CIs. Read-only."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rescore import read_rows, tcc_arm

rng = np.random.default_rng(20261004)


def boot(x, B=20000):
    x = np.asarray(x)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return [round(float(v), 4) for v in np.percentile(m, [2.5, 97.5])]


def layout(ref):
    k = json.load(open(os.path.join(ref, "kld.json")))
    toks = np.fromfile(os.path.join(ref, "tokens.i32"), dtype=np.int32)
    off, docs = 0, []
    for d in k["docs"]:
        d = dict(d, ids=toks[off:off + d["tokens"]])
        off += d["tokens"]
        docs.append(d)
    return docs, k["rows"]


def main():
    ref, prefix = sys.argv[1], sys.argv[2]
    docs, n = layout(ref)
    R = {a: read_rows(f"{prefix}-{a}.json", n) for a in ("fill", "st")}
    T = {a: tcc_arm("final/" + a) for a in ("exact", "ours-nost", "ours")}
    rows = []
    for d in docs:
        nsc = d["tokens"] - 1 - d["score_from"]
        t = T["exact"][d["source"]]
        assert (np.array(t["ids"]) == d["ids"][-512:]).all(), d["source"]
        r = {"doc": d["source"]}
        for a in R:
            seg = R[a][d["row0"]:d["row0"] + nsc].astype(np.float64)
            r["rad_" + a] = seg[-512:, 1].mean() - seg[-512:, 2].mean()
            r["rad_" + a + "_2047"] = seg[:, 1].mean() - seg[:, 2].mean()
        r["tcc_fill"] = T["ours-nost"][d["source"]]["nll_all"] - t["nll_all"]
        r["tcc_st"] = T["ours"][d["source"]]["nll_all"] - t["nll_all"]
        rows.append(r)
    g = lambda k: np.array([r[k] for r in rows])
    er, et = g("rad_st") - g("rad_fill"), g("tcc_st") - g("tcc_fill")
    out = {"docs": [r["doc"] for r in rows],
           "radiance": {"fill": round(g("rad_fill").mean(), 4), "st": round(g("rad_st").mean(), 4),
                        "st_effect": round(er.mean(), 4), "st_effect_ci": boot(er),
                        "fill_2047": round(g("rad_fill_2047").mean(), 4), "st_2047": round(g("rad_st_2047").mean(), 4)},
           "tcc_same_server": {"fill": round(g("tcc_fill").mean(), 4), "st": round(g("tcc_st").mean(), 4),
                               "st_effect": round(et.mean(), 4), "st_effect_ci": boot(et)},
           "radiance_minus_tcc": {"fill": round((g("rad_fill") - g("tcc_fill")).mean(), 4),
                                  "fill_ci": boot(g("rad_fill") - g("tcc_fill")),
                                  "st": round((g("rad_st") - g("tcc_st")).mean(), 4),
                                  "st_ci": boot(g("rad_st") - g("tcc_st")),
                                  "st_effect": round((er - et).mean(), 4), "st_effect_ci": boot(er - et)},
           "per_doc": {k: g(k).round(4).tolist() for k in ("rad_fill", "rad_st", "tcc_fill", "tcc_st")}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
