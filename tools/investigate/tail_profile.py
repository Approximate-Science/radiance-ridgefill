#!/usr/bin/env python3
"""Where in the exact tail the KVA loss sits on radiance: mean NLL delta vs exact by tail position bucket
(512-token quarters of the 2,047 scored rows), per arm, over the 9 quick docs or the 6 docs tcc matches.
Row i of a doc predicts token score_from + i + 1, so quarter q covers tokens P+1+512q .. P+512(q+1). Read-only."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rescore import EVID, ref_layout, read_rows, doc_rows

ARMS = {"fill": "stage3/kld-fill-f60f893.json", "st": "stage4/kld-st.json", "swap": "stage4/kld-swap.json",
        "quality": "stage5/kld-quality.json", "random": "stage5/kld-random.json"}


def main():
    docs, n = ref_layout()
    out = {}
    for subset, ds in {"all9": docs, "six": [d for d in docs if not d["source"].startswith("ppl/8k")]}.items():
        res = {}
        for a, p in ARMS.items():
            R = read_rows(os.path.join(EVID, p), n)
            q = np.zeros((len(ds), 4))
            for j, d in enumerate(ds):
                seg = doc_rows(R, d).astype(np.float64)
                dl = seg[:, 1] - seg[:, 2]
                # quarters by distance from P: rows [0,511) [511,1023) [1023,1535) [1535,2047) (the last = tcc's window)
                edges = [0, 511, 1023, 1535, 2047]
                q[j] = [dl[edges[k]:edges[k + 1]].mean() for k in range(4)]
            res[a] = q.mean(axis=0).round(4).tolist()
        res["st_minus_fill"] = (np.array(res["st"]) - np.array(res["fill"])).round(4).tolist()
        res["quality_minus_st"] = (np.array(res["quality"]) - np.array(res["st"])).round(4).tolist()
        out[subset] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
