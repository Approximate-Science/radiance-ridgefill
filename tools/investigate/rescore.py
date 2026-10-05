#!/usr/bin/env python3
"""Rescore radiance KL-mode .rows files on tcc's window and compare with tcc's quick-tier ppl.jsonl.

Why: radiance's KL mode scores every exact-tail position (rows [score_from, n-1)); tcc's quick tier
scored only the last 512 tokens of a 9,216 / 16,384 / 32,768-token prompt. This script checks the
alignment (doc order from the reference's kld.json, token ids vs tcc's stored ids) and prints per-doc
deltas for any window. Read-only; prints JSON lines.
"""
import argparse, json, os, sys
import numpy as np

EVID = os.path.expanduser("~/AI-Work/radiance-kva-plugin-20261004/evidence")
TCC = os.path.expanduser("~/AI-Work/kva-flashnext-iterate/tests/results")
REF = os.path.join(os.path.dirname(__file__), "../../data/kld/ref-stage0")

ARMS = {  # radiance arms: label -> report path (relative to evidence)
    "exact": "stage0/kld-exact-vs-ref.json",
    "fill": "stage4/speed-fill-f60f893.json",  # placeholder, replaced below if missing
}


def ref_layout():
    k = json.load(open(os.path.join(REF, "kld.json")))
    toks = np.fromfile(os.path.join(REF, "tokens.i32"), dtype=np.int32)
    docs, off = [], 0
    for d in k["docs"]:
        d = dict(d)
        d["ids"] = toks[off:off + d["tokens"]]
        off += d["tokens"]
        docs.append(d)
    assert off == len(toks), (off, len(toks))
    return docs, k["rows"]


def read_rows(path, n_rows):
    a = np.fromfile(path + ".rows", dtype=np.float32)
    assert a.size == n_rows * 4, (path, a.size)
    return a.reshape(n_rows, 4)


def tcc_arm(name):
    out = {}
    for l in open(os.path.join(TCC, name, "ppl.jsonl")):
        r = json.loads(l)
        out[r["id"]] = r
    return out


def doc_rows(rows, d, last=None, first=None):
    """rows of doc d; row i predicts token score_from + i + 1. last=k: the last k scored tokens."""
    n = d["tokens"] - 1 - d["score_from"]
    seg = rows[d["row0"]:d["row0"] + n]
    if last is not None:
        seg = seg[-last:]
    if first is not None:
        seg = seg[:first]
    return seg
