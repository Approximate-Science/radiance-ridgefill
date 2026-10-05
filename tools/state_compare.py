#!/usr/bin/env python3
"""state_compare.py -- R61: the decoders' late delta-net states after each mixed step, KVA vs off.

Two RADIANCE_KVA_CAPTURE_STATE directories of the SAME arrangement (scripts/conc.sh text: one batched
request, so the same decoder token rides beside the same prefill chunk in both runs) hold mixed.jsonl
and one mixed.<key>.r<R>.npy per mixed step and rank (arch/kva_dump.h). Steps are matched by key
(the step's first position and the FNV-1a of all its token ids) and rank; per matched step, each
decoder slot (sequence index < n_seq_decode) must be byte-identical, and the prefill slot (the last
sequence) of an approximate step is reported as the positive control (it must differ in speed and
quality). Prints one line per matched step and a verdict.

usage: state_compare.py <kva_dir> <off_dir>
"""
import json
import os
import sys

import numpy as np


def index(d):
    out = {}
    for line in open(os.path.join(d, "mixed.jsonl")):
        j = json.loads(line)
        out[(j["key"], j["rank"])] = j
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: state_compare.py <kva_dir> <off_dir>")
    kva_dir, off_dir = sys.argv[1:]
    kva, off = index(kva_dir), index(off_dir)
    shared = sorted(set(kva) & set(off), key=lambda k: (kva[k]["starts"], k[1]))
    print(f"mixed steps: kva {len(kva)}, off {len(off)}, matched {len(shared)}")
    bad = approx = moved = 0
    for key in shared:
        a, b = kva[key], off[key]
        x = np.load(os.path.join(kva_dir, a["file"]))
        y = np.load(os.path.join(off_dir, b["file"]))
        d = a["n_seq_decode"]
        same = [bool(np.array_equal(x[i].view(np.uint32), y[i].view(np.uint32))) for i in range(d)]
        last = x[-1].astype(np.float64) - y[-1].astype(np.float64)
        prefill_moved = bool(np.any(last != 0))
        bad += not all(same)
        approx += a["approximate"]
        moved += a["approximate"] and prefill_moved
        print(f"  {key[0]} r{key[1]} starts {a['starts']} approximate {a['approximate']}: decoder slots "
              f"{'identical' if all(same) else 'DIFFER ' + str(same)}; prefill slot max |diff| "
              f"{np.abs(last).max():.3e}")
    print(f"state_compare: {len(shared) - bad}/{len(shared)} steps with every decoder slot byte-identical; "
          f"prefill slot moved on {moved}/{approx} approximate steps (positive control)")
    sys.exit(1 if bad or not shared else 0)


if __name__ == "__main__":
    main()
