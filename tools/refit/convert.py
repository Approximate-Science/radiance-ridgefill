#!/usr/bin/env python3
"""Radiance's projector captures (RADIANCE_RIDGEFILL_CAPTURE, notes/arch.md "Capture") as tcc Capture records, so the
research pipeline's own code reads them unchanged (R42): qfn.fit.Sums.add takes the record dict directly (no file
round trip for the training rows), and fit.Held / fit.records read the held-out documents from the capture_NNNNN.pt
files this tool writes.

  convert.py pt    CAPTURE_DIR OUT_DIR                    one capture_NNNNN.pt per chunk, in position order
  convert.py check CAPTURE_DIR --ckpt MODEL_DIR [--json OUT]   the HC identity check (instrument check)

A record (b0/capture_files.py, the fields tcc's Capture.flush writes; final_multi_hidden is not captured, the
`final` map feeds only MTP, which radiance's RidgeFill path does not use):
  boundary_S       bf16 [R, hc*hidden]  b_h entering layer S at the captured rows
  block_input_L    bf16 [R, hidden]     layer L's bf16 block input x right after its connection read, L >= S
  positions        int64 [R]            absolute positions of the captured rows (multiples of the stride)
  input_ids        int32 [R]            their token ids
  stride           int
HC identity check (KVA-FACTS §7): qfn.hc.mix(boundary_S, layer S's read weights) must reproduce block_input_S. The
weights come from a checkpoint dir (qfn.weights.Checkpoint, e.g. tcc's bf16 HC file), so a cosine below 1 measures
how far radiance's own connection-read numerics (its stored HC weights) sit from those weights; on tcc's captures
the same check gave 0.99999780 (report-big-s24.json).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ridgefill_research  # noqa: E402


def chunks(capture_dir):
    """capture.jsonl entries of the chunks whose files are present, in position order; the last entry per prefix
    wins (a rerun of the same chunk rewrites its files)."""
    capture_dir = Path(capture_dir)
    entries = {}
    with open(capture_dir / "capture.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                entries[e["prefix"]] = e
    present = [e for e in entries.values() if (capture_dir / f"{e['prefix']}.rows.npy").is_file()]
    return sorted(present, key=lambda e: e["chunk_start"])


def bf16(path, rows, width):
    """A '<u2' .npy of bf16 bits as a bf16 tensor, shape-checked."""
    bits = np.load(path)
    if bits.dtype != np.dtype("<u2") or bits.shape != (rows, width):
        raise SystemExit(f"{path}: {bits.dtype} {bits.shape}, expected <u2 ({rows}, {width})")
    return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)


def read_chunk(capture_dir, entry):
    """One chunk's tcc Capture record."""
    base = Path(capture_dir) / entry["prefix"]
    rows = np.load(f"{base}.rows.npy").astype(np.int64)
    ids = np.load(f"{base}.ids.npy")
    pos = np.load(f"{base}.pos.npy")
    r, split, hidden, hc = entry["rows"], entry["split"], entry["hidden"], entry["hc"]
    if len(rows) != r or len(ids) != entry["n_tok"] or len(pos) != entry["n_tok"]:
        raise SystemExit(f"{base}: rows {len(rows)} ids {len(ids)} pos {len(pos)} vs capture.jsonl {entry}")
    if entry["layers"][0] != split or int(pos[0]) != entry["chunk_start"]:
        raise SystemExit(f"{base}: layers start {entry['layers'][0]} / first position {pos[0]} vs {entry}")
    rec = {f"boundary_{split}": bf16(f"{base}.boundary.npy", r, hc * hidden)}
    for layer in entry["layers"]:
        rec[f"block_input_{layer}"] = bf16(f"{base}.bi.{layer}.npy", r, hidden)
    rec["positions"] = torch.from_numpy(pos[rows].astype(np.int64))
    rec["input_ids"] = torch.from_numpy(ids[rows].astype(np.int32))
    rec["stride"] = int(entry["stride"])
    if (rec["positions"] % rec["stride"]).any():
        raise SystemExit(f"{base}: a captured row's position is not a multiple of the stride")
    return rec


def read_document(capture_dir):
    """Every chunk of one document's capture dir as one record (rows concatenated in position order)."""
    recs = [read_chunk(capture_dir, e) for e in chunks(capture_dir)]
    if not recs:
        raise SystemExit(f"{capture_dir}: no captured chunks")
    out = {k: torch.cat([r[k] for r in recs]) for k in recs[0] if k != "stride"}
    out["stride"] = recs[0]["stride"]
    return out


def write_pt(capture_dir, out_dir):
    """One capture_NNNNN.pt per chunk (NNNNN = position order), the files fit.Held / fit.records glob."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = chunks(capture_dir)
    for i, e in enumerate(entries):
        torch.save(read_chunk(capture_dir, e), out_dir / f"capture_{i:05d}.pt")
    return len(entries)


def hc_identity(rec, ck):
    """(mean row cosine of qfn.hc.mix(boundary_S, ck's layer-S read weights) vs block_input_S, S); ck is a
    qfn.weights.Checkpoint."""
    ridgefill_research.root()
    import torch.nn.functional as F
    from qfn import hc as H
    split = min(int(k.split("_")[1]) for k in rec if k.startswith("boundary_"))
    base = H.mix(rec[f"boundary_{split}"], ck.hc_read(split))
    return F.cosine_similarity(base, rec[f"block_input_{split}"].float(), dim=-1).mean().item(), split


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pt")
    p.add_argument("capture_dir")
    p.add_argument("out_dir")
    c = sub.add_parser("check")
    c.add_argument("capture_dir")
    c.add_argument("--ckpt", required=True, help="checkpoint dir with layer S's connection-read weights")
    c.add_argument("--json", default="", help="also write the result here")
    a = ap.parse_args(argv)
    if a.command == "pt":
        print(f"wrote {write_pt(a.capture_dir, a.out_dir)} capture file(s) to {a.out_dir}")
        return 0
    rec = read_document(a.capture_dir)
    ridgefill_research.root()
    from qfn.weights import Checkpoint
    cos, split = hc_identity(rec, Checkpoint(a.ckpt))
    result = dict(capture_dir=str(a.capture_dir), rows=int(rec["positions"].numel()), split=split, hc_identity_cos=cos)
    print(json.dumps(result))
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
