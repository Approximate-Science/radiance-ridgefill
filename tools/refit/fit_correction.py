#!/usr/bin/env python3
"""Stage 6 correction refit (R43): C_L = mean over approximate chunk ends of (S_exact - S_pred) per late delta-net
layer and TP rank, from capture.py's filed state captures of an exact run and a speed run of the same prompts, written
in b0/st_hook.py's fit format so tools/kva_sidecar.py --names refit --st reads it unchanged.

  fit_correction.py --exact DIR --pred DIR --out DIR [--min-count 50] [--min-prompts 13] [--shipped R0.pt R1.pt]

Matching: a speed-run record with "approximate": true is paired with the exact-run record of the same chunk (first
position + hash of its ids, i.e. the same prompt and chunk), the same rank and the same last position; the exact
record must be an `off`-mode, non-approximate one. Both are the state slot after that chunk (the speed one before any
correction apply: capture_state copies between the scan and the apply, notes/arch.md). Exact-run records with no
approximate partner (the tail chunks) are not used.
Output (per rank r): <out>/kva-radiance-s<S>-st.rank<r>.pt = {"sum": {L: f32 [H_local, V, K]}, "count": {L: n}} with
H_local the rank's contiguous value heads [r*H_local, (r+1)*H_local) (checked against each record's "heads"), so
kva_sidecar's cat(rank0, rank1) is the model's head order. Sums are accumulated in f64 and stored f32.
report-radiance-st.json: prompts, chunk ends per layer, and per layer: the relative state error |S_exact - S_pred| /
|S_exact|, the in-sample share of the error's energy the constant removes (n |C|^2 / sum |d|^2), and, with --shipped,
the cosine between the refit and the shipped C (each rank's heads).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def records(store, approximate):
    """{(chunk key, rank): (path, entry, prompt dir)} of the filed state records with this approximate flag."""
    out = {}
    for meta in sorted(Path(store).glob("*/*/*/meta.json")):
        for line in (meta.parent / "state.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["approximate"] != approximate:
                continue
            key = (".".join(e["file"].split(".")[1:3]), e["rank"])
            out[key] = (meta.parent / e["file"], e, meta.parent.name)
    return out


def load_state(path, entry):
    s = np.load(path)
    heads = entry["heads"][1] - entry["heads"][0]
    if s.dtype != np.float32 or s.ndim != 4 or s.shape[:2] != (len(entry["layers"]), heads):
        raise SystemExit(f"{path}: {s.dtype} {s.shape}, expected f32 [{len(entry['layers'])}, {heads}, V, K]")
    return torch.from_numpy(s).double()


def fit(exact_store, pred_store):
    """Per rank: layers, f64 sums of d = S_exact - S_pred, counts, sum |d|^2, sum |S_exact|^2, prompts used."""
    exact, pred = records(exact_store, False), records(pred_store, True)
    acc = {}
    for key, (ppath, pe, prompt) in sorted(pred.items()):
        if key not in exact:
            raise SystemExit(f"{ppath}: no exact-run record of chunk {key[0]} rank {key[1]}")
        epath, ee, _ = exact[key]
        if (ee["last_position"], ee["layers"], ee["heads"], ee["mode"]) != \
                (pe["last_position"], pe["layers"], pe["heads"], "off"):
            raise SystemExit(f"{epath} vs {ppath}: last position / layers / heads differ, or the exact run was not off")
        rank, heads = pe["rank"], pe["heads"]
        if heads[0] != rank * (heads[1] - heads[0]):
            raise SystemExit(f"{ppath}: rank {rank} holds heads {heads}, not the contiguous split")
        s_exact = load_state(epath, ee)
        d = s_exact - load_state(ppath, pe)
        a = acc.setdefault(rank, dict(layers=pe["layers"], sum=torch.zeros_like(d), count=0, dd=torch.zeros(d.shape[0]),
                                      ss=torch.zeros(d.shape[0]), prompts=set()))
        if a["layers"] != pe["layers"]:
            raise SystemExit(f"{ppath}: layers {pe['layers']} differ from this rank's first record's")
        a["sum"] += d
        a["count"] += 1
        a["dd"] += d.square().flatten(1).sum(1)
        a["ss"] += s_exact.square().flatten(1).sum(1)
        a["prompts"].add(prompt)
    if not acc:
        raise SystemExit(f"{pred_store}: no approximate state records")
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exact", required=True, help="capture.py run --what state store of the exact (off) run")
    ap.add_argument("--pred", required=True, help="the same for the speed run (refit projector, no correction)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=50, help="R43: chunk ends per layer")
    ap.add_argument("--min-prompts", type=int, default=13, help="R43: prompts")
    ap.add_argument("--shipped", nargs=2, metavar=("RANK0", "RANK1"), help="the shipped correction, for the cosine")
    a = ap.parse_args(argv)
    acc = fit(a.exact, a.pred)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = min(acc[0]["layers"])
    report = dict(exact=a.exact, pred=a.pred, ranks={})
    for rank, r in sorted(acc.items()):
        n, prompts = r["count"], len(r["prompts"])
        if n < a.min_count or prompts < a.min_prompts:
            raise SystemExit(f"rank {rank}: {n} chunk ends from {prompts} prompts; R43 needs >= {a.min_count} from "
                             f">= {a.min_prompts}")
        c = r["sum"] / n
        energy = (n * c.square().flatten(1).sum(1) / r["dd"]).tolist()
        rel = (r["dd"] / r["ss"]).sqrt().tolist()
        path = out / f"kva-radiance-s{split}-st.rank{rank}.pt"
        torch.save({"sum": {L: r["sum"][i].float().contiguous() for i, L in enumerate(r["layers"])},
                    "count": {L: n for L in r["layers"]}}, path)
        info = dict(file=path.name, chunk_ends=n, prompts=sorted(r["prompts"]),
                    layers={str(L): dict(rel_error=round(rel[i], 5), constant_share=round(energy[i], 5),
                                         c_norm=round(c[i].norm().item(), 4)) for i, L in enumerate(r["layers"])})
        if a.shipped:
            old = torch.load(a.shipped[rank], map_location="cpu", weights_only=True)
            for i, L in enumerate(r["layers"]):
                ref = (old["sum"][L] / old["count"][L]).double().flatten()
                info["layers"][str(L)]["cos_shipped"] = round(torch.nn.functional.cosine_similarity(
                    c[i].flatten(), ref, dim=0).item(), 4)
        report["ranks"][str(rank)] = info
        log(f"rank {rank}: {n} chunk ends, {prompts} prompts; rel error {min(rel):.3f}..{max(rel):.3f}, constant "
            f"removes {min(energy):.3f}..{max(energy):.3f} of the error energy -> {path}")
    (out / "report-radiance-st.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
