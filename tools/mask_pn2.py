#!/usr/bin/env python3
"""mask_pn2.py -- R58': on every dumped masked step with two (or more) prefill entries, the earlier
sequences' rows are all exact.

Reads <dump>/mask.jsonl (RADIANCE_RIDGEFILL_DUMP, arch/ridgefill_dump.h): per masked step the device mask (one
character a row, '1' = approximated) and bounds {s, b', b', e} of the LAST sequence. A step whose
prefill count n_seq - n_seq_decode is >= 2 must have no '1' before s (the decoders' rows and every
earlier prefill's rows), and no '1' at or past b'.

usage: mask_pn2.py <dump_dir>
"""
import json
import os
import sys

if len(sys.argv) != 2:
    sys.exit("usage: mask_pn2.py <dump_dir>")
steps = [json.loads(l) for l in open(os.path.join(sys.argv[1], "mask.jsonl"))]
two = [j for j in steps if j["n_seq"] - j["n_seq_decode"] >= 2]
bad = [j for j in two if "1" in j["mask"][:j["bounds"][0]] or "1" in j["mask"][j["bounds"][1]:]]
for j in two:
    s, e = j["bounds"][0], j["bounds"][1]
    print(f"  n_tok {j['n_tok']} n_seq {j['n_seq']} (decoders {j['n_seq_decode']}) s {s} b' {e}: rows before s "
          f"approximated {j['mask'][:s].count('1')}, window approximated {j['mask'][s:e].count('1')}/{e - s}")
print(f"mask_pn2: {len(steps)} masked steps, {len(two)} with two prefills, {len(bad)} with an earlier row approximated")
sys.exit(1 if bad or not two else 0)
