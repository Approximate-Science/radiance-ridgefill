#!/usr/bin/env python3
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""How much of radiance's GDN fill error at a chunk end does a constant correction C remove?

Inputs: two capture dirs written by RADIANCE_RIDGEFILL_CAPTURE_STATE (notes/arch.md "Capture"): an exact run
(mode off) and an approximate run (speed, nothing applied before the copy), same prompts, same chunking.
Per (prompt, chunk end, rank) the files are f32 [n_late_gdn, H_local, 128 (V), 128 (K)].
E = S_exact - S_pred at every approximate chunk end. Reports, per layer (both ranks pooled):
  rel      sqrt(sum |E|^2 / sum |S_exact|^2)
  removed  1 - sum |E - C|^2 / sum |E|^2   for C = shipped (tcc fit), the in-sample mean, the
           leave-one-prompt-out mean (the honest radiance fit), and alpha* x shipped (best scalar)
  alpha*   sum <E, C_ship> / (n |C_ship|^2)
  orient   cos(mean E, C_ship) and the same against controls: C_ship transposed per head (V<->K),
           heads shifted by one inside a rank, heads in the tiled order (value head j <-> key head j % 16)
Also the tail: |S_pred(N) - S_exact(N)| at the end of the exact tail relative to |E| at its start.
Read-only; JSON to stdout.
"""
import argparse, collections, json, os
import numpy as np
import torch


def records(root):
    out = {}
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        p = os.path.join(d, "state.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            r["path"] = os.path.join(d, r["file"])
            out[(name, r["chunk_start"], r["last_position"], r["rank"])] = r
    return out


def load_c(path):
    d = torch.load(path, map_location="cpu")
    return {int(L): (d["sum"][L].double() / d["count"][L]).numpy() for L in d["sum"]}


def tiled_perm(n_v=48, n_k=16):
    """index j in the tiled order (key head j % n_k) -> HF grouped index (key head h // (n_v/n_k))."""
    rep = n_v // n_k
    return np.array([(j % n_k) * rep + j // n_k for j in range(n_v)])


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--shipped", nargs=2, required=True, help="rank0.pt rank1.pt")
    ap.add_argument("--refit", nargs=2, help="rank0.pt rank1.pt (optional, for the cosine)")
    args = ap.parse_args()
    ex, pr = records(args.exact), records(args.pred)
    ship = [load_c(p) for p in args.shipped]
    refit = [load_c(p) for p in args.refit] if args.refit else None
    layers = None
    # per (prompt, rank): sums over that prompt's chunk ends, so the LOO mean is exact
    S_p = collections.defaultdict(lambda: 0.0)
    n_p = collections.Counter()
    EE = collections.defaultdict(float)   # (L, rank) -> sum |E|^2
    SS = collections.defaultdict(float)   # (L, rank) -> sum |S_exact|^2
    tail = collections.defaultdict(lambda: [0.0, 0.0])  # (L) -> [sum |E_N|^2, sum |E_P|^2 at the same prompts]
    by_end = collections.defaultdict(lambda: 0.0)
    n_end = collections.Counter()
    last_EP = {}
    for key, r in sorted(pr.items()):
        name, start, last, rank = key
        e = ex.get(key)
        if e is None:
            raise SystemExit(f"no exact record for {key}")
        layers = r["layers"]
        Sx = np.load(e["path"]).astype(np.float64)
        Sp = np.load(r["path"]).astype(np.float64)
        E = Sx - Sp
        if r["approximate"]:
            S_p[(name, rank)] = S_p[(name, rank)] + E
            n_p[(name, rank)] += 1
            for i, L in enumerate(layers):
                EE[(L, rank)] += float((E[i] ** 2).sum())
                SS[(L, rank)] += float((Sx[i] ** 2).sum())
            bucket = "2048" if last + 1 == 2048 else ("4096" if last + 1 == 4096 else ">=6144")
            by_end[(bucket, rank)] = by_end[(bucket, rank)] + E
            n_end[(bucket, rank)] += 1
            prev = last_EP.get((name, rank))
            if prev is None or prev[0] < last:
                last_EP[(name, rank)] = (last, E)
        else:
            lp = last_EP.get((name, rank))
            for i, L in enumerate(layers):
                tail[(L, rank)][0] += float((E[i] ** 2).sum())
                tail[(L, rank)][1] += float((lp[1][i] ** 2).sum()) if lp else float("nan")
    out = {"prompts": len({k[0] for k in n_p}), "chunk_ends_per_rank": sum(v for k, v in n_p.items() if k[1] == 0),
           "layers": {}}
    perm = tiled_perm()
    tot = collections.defaultdict(float)
    for i, L in enumerate(layers):
        row = collections.defaultdict(float)
        for rank in (0, 1):
            keys = [k for k in n_p if k[1] == rank]
            S = sum(S_p[k][i] for k in keys)
            n = sum(n_p[k] for k in keys)
            C_in = S / n
            Cs = ship[rank][L]
            ee = EE[(L, rank)]
            def resid(C, Ssum=S, nn=n):
                return ee - 2 * float((Ssum * C).sum()) + nn * float((C * C).sum())
            loo = 0.0
            for k in keys:
                C_loo = (S - S_p[k][i]) / (n - n_p[k])
                Sk = S_p[k][i]
                loo += - 2 * float((Sk * C_loo).sum()) + n_p[k] * float((C_loo * C_loo).sum())
            loo += ee
            a = float((S * Cs).sum()) / (n * float((Cs * Cs).sum()))
            row["ee"] += ee
            row["ss"] += SS[(L, rank)]
            row["r_ship"] += resid(Cs)
            row["r_in"] += resid(C_in)
            row["r_loo"] += loo
            row["r_alpha"] += resid(a * Cs)
            row["sum_ES"] += float((S * Cs).sum())
            row["n_CC"] += n * float((Cs * Cs).sum())
            full_ship = np.concatenate([ship[0][L], ship[1][L]])          # [48, V, K]
            ctrl = {"same": Cs,
                    "transposed": np.swapaxes(Cs, -1, -2),
                    "shift1": np.roll(Cs, 1, axis=0),
                    "tiled": full_ship[perm][rank * 24:(rank + 1) * 24]}
            for nm, C in ctrl.items():
                row["cos_meanE_" + nm] += cos(C_in, C) / 2
            if refit:
                Cr = refit[rank][L]
                for nm, C in ctrl.items():
                    row["cos_refit_" + nm] += cos(Cr, C) / 2
            t = tail[(L, rank)]
            row["tail_EN"] += t[0]
            row["tail_EP"] += t[1]
        rec = {"rel": round((row["ee"] / row["ss"]) ** 0.5, 4),
               "removed_shipped": round(1 - row["r_ship"] / row["ee"], 4),
               "removed_insample": round(1 - row["r_in"] / row["ee"], 4),
               "removed_loo": round(1 - row["r_loo"] / row["ee"], 4),
               "removed_alpha_shipped": round(1 - row["r_alpha"] / row["ee"], 4),
               "alpha_star": round(row["sum_ES"] / row["n_CC"], 3),
               "tail_EN_over_EP": round((row["tail_EN"] / row["tail_EP"]) ** 0.5, 4)}
        rec.update({k: round(v, 4) for k, v in row.items() if k.startswith("cos_")})
        out["layers"][str(L)] = rec
        for k in ("ee", "r_ship", "r_in", "r_loo", "r_alpha", "sum_ES", "n_CC"):
            tot[k] += row[k]
        for k, v in row.items():
            if k.startswith("cos_"):
                tot[k] += v / len(layers)
    out["all_layers"] = {"removed_shipped": round(1 - tot["r_ship"] / tot["ee"], 4),
                         "removed_insample": round(1 - tot["r_in"] / tot["ee"], 4),
                         "removed_loo": round(1 - tot["r_loo"] / tot["ee"], 4),
                         "removed_alpha_shipped": round(1 - tot["r_alpha"] / tot["ee"], 4),
                         "alpha_star": round(tot["sum_ES"] / tot["n_CC"], 3),
                         "shipped_over_loo": round((tot["ee"] - tot["r_ship"]) / (tot["ee"] - tot["r_loo"]), 3)}
    out["all_layers"].update({k: round(v, 4) for k, v in tot.items() if k.startswith("cos_")})
    # does the mean error depend on where the chunk ends? cos between bucket means, per layer, rank 0
    pos = {}
    for i, L in enumerate(layers):
        m = {b: by_end[(b, 0)][i] / n_end[(b, 0)] for b in ("2048", "4096", ">=6144") if n_end[(b, 0)]}
        pos[str(L)] = {"cos_2048_vs_late": round(cos(m["2048"], m[">=6144"]), 3),
                       "norm_late_over_2048": round(float(np.linalg.norm(m[">=6144"]) / np.linalg.norm(m["2048"])), 3)}
    out["position_dependence_rank0"] = pos
    out["n_end_rank0"] = {b: n_end[(b, 0)] for b in ("2048", "4096", ">=6144")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
