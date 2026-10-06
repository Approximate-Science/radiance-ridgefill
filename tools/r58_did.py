#!/usr/bin/env python3
"""r58_did.py -- R58': does sharing a step with another prompt move a RidgeFill mode's tails more than it
moves stock's?

Two prefills share a step only at a doc hand-over, and the top-up shifts every later doc's chunk
boundaries, which moves even the stock engine's numbers (the GEMM-shape / chunking floor). So the
test is a difference of differences, per doc, over the reports' last L scored positions
(scripts/kl_tail.py's dNLL vs the stock reference):

    (mode at --max-num-seqs 2 - mode at 1) - (exact at 2 - exact at 1)

with a bootstrap 95% interval over docs (10,000 resamples, seed 0). The exact-at-1 report may be
omitted when the reference itself is exact at 1 (its dNLL is 0 by construction).

usage: r58_did.py --corpus C --last L --mode-1 A.json --mode-2 B.json --exact-2 E2.json [--exact-1 E1.json]
"""
import argparse
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

KL_TAIL = Path(__file__).resolve().parent.parent / "scripts" / "kl_tail.py"


def per_doc(corpus, last, report):
    out = subprocess.run([sys.executable, str(KL_TAIL), "--corpus", corpus, "--last", str(last), report],
                         capture_output=True, text=True, check=True).stdout
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s+(\S+/\S+)\s+dNLL ([+-][\d.]+)", out, re.M)}


def main():
    ap = argparse.ArgumentParser()
    for name in ("--corpus", "--mode-1", "--mode-2", "--exact-2"):
        ap.add_argument(name, required=True)
    ap.add_argument("--exact-1")
    ap.add_argument("--last", type=int, default=512)
    a = ap.parse_args()
    m1, m2 = per_doc(a.corpus, a.last, a.mode_1), per_doc(a.corpus, a.last, a.mode_2)
    e2 = per_doc(a.corpus, a.last, a.exact_2)
    e1 = per_doc(a.corpus, a.last, a.exact_1) if a.exact_1 else {k: 0.0 for k in e2}
    docs = sorted(m1)
    d = [(m2[k] - m1[k]) - (e2[k] - e1[k]) for k in docs]
    rng = random.Random(0)
    boot = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    print(f"last {a.last}: (mode s2 - s1) - (exact s2 - s1) = {statistics.mean(d):+.5f} "
          f"[{boot[250]:+.5f}, {boot[9750]:+.5f}] over {len(d)} docs; "
          f"exact's own s2 - s1 {statistics.mean(e2[k] - e1[k] for k in docs):+.5f}")


if __name__ == "__main__":
    main()
