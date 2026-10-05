#!/usr/bin/env python3
"""On-engine check of the +st apply, from RADIANCE_KVA_CAPTURE_STATE captures (labbook INV-apply, INV-apply-neg).

Arms are speed runs of the same prompts that differ only in the correction (alpha / copy); `--base` is the alpha=0
arm, `--exact` the exact run. Approximate records are taken after the scan and BEFORE the apply; exact-chunk (tail)
records after the step (notes/arch.md "Capture").
  chunk ends   every arm vs base: max |diff| / rms(base) per layer -- equal unless something leaked across a chunk
  tail end     per layer, sum |S_arm(N) - S_exact(N)|^2 over prompts and both ranks, and its share of the base's;
               plus the linear-response check D(arm) = S_arm(N) - S_base(N) vs alpha * D(alpha=1)
Layer 24 is the clean one: its tail inputs depend only on layers 0..23, which every arm runs exactly, so its tail
recurrence is the same linear map of the starting state in every arm. JSON to stdout.
"""
import argparse, collections, json, os
import numpy as np


def records(root):
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "state.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            r["path"] = os.path.join(root, name, r["file"])
            out[(name, r["chunk_start"], r["rank"])] = r
    return out


def tail_keys(recs):
    """(prompt, rank) -> key of the last exact record and of the last approximate one."""
    last, lastp = {}, {}
    for k, r in recs.items():
        pr = (k[0], k[2])
        if r["approximate"]:
            if pr not in lastp or r["last_position"] > recs[lastp[pr]]["last_position"]:
                lastp[pr] = k
        elif pr not in last or r["last_position"] > recs[last[pr]]["last_position"]:
            last[pr] = k
    return last, lastp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", required=True)
    ap.add_argument("--base", required=True, help="name=dir of the alpha=0 arm")
    ap.add_argument("--arm", action="append", default=[], help="name=alpha=dir (alpha used for the linearity check)")
    ap.add_argument("--same", action="append", default=[], help="name=dir: another no-correction run (determinism)")
    args = ap.parse_args()
    ex = records(args.exact)
    bname, bdir = args.base.split("=", 1)
    base = records(bdir)
    arms = {}
    for a in args.arm:
        nm, al, d = a.split("=", 2)
        arms[nm] = (float(al), records(d))
    same = {s.split("=", 1)[0]: records(s.split("=", 1)[1]) for s in args.same}
    layers = next(iter(base.values()))["layers"]
    out = {"layers": layers, "chunk_ends": {}, "tail": {}, "linear": {}}
    # chunk ends: approximate records
    for nm, recs in list({k: v[1] for k, v in arms.items()}.items()) + list(same.items()):
        worst = np.zeros(len(layers)); first_equal = True; n = 0
        for k, r in base.items():
            if not r["approximate"]:
                continue
            o = recs[k]
            Sb, So = np.load(r["path"]), np.load(o["path"])
            if k[1] == 0:
                first_equal &= bool(np.array_equal(Sb, So))
            rms = np.sqrt((Sb.astype(np.float64) ** 2).mean(axis=(1, 2, 3)))
            worst = np.maximum(worst, np.abs(Sb - So).max(axis=(1, 2, 3)) / rms)
            n += 1
        out["chunk_ends"][nm] = {"records": n, "first_chunk_bitwise_equal": first_equal,
                                 "max_rel_diff_by_layer": [float(f"{v:.3g}") for v in worst]}
    # tail end
    last_b, lastp_b = tail_keys(base)
    err = collections.defaultdict(lambda: np.zeros(len(layers)))
    lin_num = collections.defaultdict(lambda: np.zeros(len(layers)))
    lin_den = collections.defaultdict(lambda: np.zeros(len(layers)))
    d1 = {}
    for pr, k in sorted(last_b.items()):
        Sx = np.load(ex[k]["path"]).astype(np.float64)
        Sb = np.load(base[k]["path"]).astype(np.float64)
        err[bname] += ((Sb - Sx) ** 2).sum(axis=(1, 2, 3))
        for nm, (al, recs) in arms.items():
            Sa = np.load(recs[k]["path"]).astype(np.float64)
            err[nm] += ((Sa - Sx) ** 2).sum(axis=(1, 2, 3))
            if al == 1.0 and nm not in d1:
                d1[(nm, pr)] = Sa - Sb
        # linear response of every non-unit-alpha arm against the alpha=1 shipped arm
        ref = next((nm for nm, (al, _) in arms.items() if al == 1.0), None)
        for nm, (al, recs) in arms.items():
            if al in (0.0, 1.0) or ref is None:
                continue
            Da = np.load(recs[k]["path"]).astype(np.float64) - Sb
            D1 = d1[(ref, pr)]
            lin_num[nm] += ((Da - al * D1) ** 2).sum(axis=(1, 2, 3))
            lin_den[nm] += ((al * D1) ** 2).sum(axis=(1, 2, 3))
    for nm, e in err.items():
        out["tail"][nm] = {"err_energy_share_of_base": (e / err[bname]).round(4).tolist()}
    for nm in lin_num:
        out["linear"][nm] = {"rel_dev_by_layer": [float(f"{v:.3g}") for v in np.sqrt(lin_num[nm] / lin_den[nm])]}
    out["tail_prompts_x_ranks"] = len(last_b)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
