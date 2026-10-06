#!/usr/bin/env python3
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""boundary_cos.py -- R17: the plugin's dumped boundary stream vs a tcc capture's boundary_24.

  boundary_cos.py --dump DIR --capture CAPTURE.pt [--capture CAPTURE.pt ...] [--ids IDS.json]

DIR is a RADIANCE_RIDGEFILL_DUMP directory (boundary.jsonl + boundary.p<P>.npy, f32 [n_tok, hc*n_embd]: the
residual stream entering layer S of the approximate chunk starting at absolute position P). Each capture
file holds rows at absolute `positions` (stride 8) with `boundary_24` [rows, hc*n_embd] bf16 and
`input_ids`. For every capture whose positions fall in a dumped chunk: the dumped token ids must equal
the capture's input_ids at those positions (else the rows are not the same tokens and the row is
refused), then per-row cosine of the dumped row vs boundary_24. --ids (the full prompt's ids, as
scripts/send_doc.py --out wrote them) is checked against every capture's input_ids too.
Prints one JSON object: per capture n, min, mean, median, p01 of the cosines, and the pooled values.
Needs numpy and torch (the research venv).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--capture", action="append", required=True)
    ap.add_argument("--ids")
    a = ap.parse_args()
    dump = Path(a.dump)
    chunks = [json.loads(x) for x in (dump / "boundary.jsonl").read_text().splitlines() if x.strip()]
    full = json.loads(Path(a.ids).read_text()) if a.ids else None
    out, pooled = {}, []
    for cap_path in a.capture:
        cap = torch.load(cap_path, map_location="cpu", weights_only=True)
        pos = cap["positions"].numpy()
        cids = cap["input_ids"].numpy()
        if full is not None and not all(full[p] == i for p, i in zip(pos.tolist(), cids.tolist())):
            raise SystemExit(f"{cap_path}: the prompt's ids differ from the capture's input_ids")
        hit = [c for c in chunks if c["chunk_start"] <= pos[0] and pos[-1] < c["chunk_start"] + c["n_tok"]]
        if not hit:
            out[cap_path] = "no dumped chunk covers positions %d..%d" % (pos[0], pos[-1])
            continue
        c = hit[-1]
        rel = pos - c["chunk_start"]
        dids = np.asarray(c["token_ids"])[rel]
        if not np.array_equal(dids, cids):
            raise SystemExit(f"{cap_path}: dumped token ids differ from the capture's at {int((dids != cids).sum())} rows")
        mine = np.load(dump / c["file"], mmap_mode="r")[rel].astype(np.float64)
        ref = cap["boundary_24"].float().numpy().astype(np.float64)
        cos = (mine * ref).sum(1) / (np.linalg.norm(mine, axis=1) * np.linalg.norm(ref, axis=1))
        pooled.extend(cos.tolist())
        out[cap_path] = dict(chunk_start=c["chunk_start"], n=int(cos.size), min=float(cos.min()),
                             mean=float(cos.mean()), median=float(np.median(cos)),
                             p01=float(np.percentile(cos, 1)), argmin_position=int(pos[int(cos.argmin())]),
                             norm_ratio_median=float(np.median(np.linalg.norm(mine, axis=1) /
                                                               np.linalg.norm(ref, axis=1))))
    p = np.asarray(pooled)
    if p.size:
        out["pooled"] = dict(n=int(p.size), min=float(p.min()), mean=float(p.mean()),
                             median=float(np.median(p)), frac_below_0_98=float((p < 0.98).mean()))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
