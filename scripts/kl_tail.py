#!/usr/bin/env python3
"""kl_tail.py -- KL-mode reports scored on the LAST L positions of each doc only (R100's headline).

  kl_tail.py --corpus quick9.jsonl --last 512 A.json [B.json]

WHY. The headline quality protocol (Dylan, 2026-10-04): with the exact tail extended to T = 2,560 at
chunk 2,048, scoring only each doc's last 512 positions means every scored token has >= 2,048 exact
tokens before it, as a generated token does. A report's whole scored window is [score_from, N-1); this
reads the report's .rows (radiance core/kld.cpp:586-590: [positions, 4] f32 in reference row order -- KL,
the candidate's NLL, the reference's NLL, top-1 agreement) and keeps the last L rows of every doc.

PRINTS, per report: per-doc dNLL = mean(candidate NLL - reference NLL) over the doc's last L rows (paired
with the exact reference by construction), mean KL and top-1 over them; the mean over docs with a
bootstrap 95% CI (10,000 resamples over docs, seed 0, percentile method -- tools/paired.py's); and the
ppl ratio exp(mean dNLL). With B: the paired per-doc difference B - A of those dNLLs and its CI.
Doc order and per-doc position counts come from the corpus (reference row order) and the report's
by_source section; a mismatch with the .rows length is an error, named. numpy only.
"""
import argparse
import json
import math
import sys

import numpy as np

N_BOOTSTRAP, SEED = 10000, 0


def fail(msg):
    raise SystemExit(f"kl_tail: {msg}")


def bootstrap(x):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    means = x[rng.integers(0, len(x), size=(N_BOOTSTRAP, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_doc(report, sources, last):
    """{source: (dNLL, KL, top1)} over each doc's last `last` rows."""
    rep = json.load(open(report))
    rows = np.fromfile(report + ".rows", dtype=np.float32)
    if rows.size % 4:
        fail(f"{report}.rows is not a [positions, 4] f32 array")
    rows = rows.reshape(-1, 4)
    counts = [rep["by_source"][s]["positions"] for s in sources]
    if sum(counts) != len(rows):
        fail(f"{report}.rows holds {len(rows)} positions; by_source says {sum(counts)}")
    out, at = {}, 0
    for s, n in zip(sources, counts):
        if n < last:
            fail(f"{report}: {s} scores {n} positions, fewer than --last {last}")
        r = rows[at + n - last:at + n]
        out[s] = (float((r[:, 1] - r[:, 2]).mean()), float(r[:, 0].mean()), float(r[:, 3].mean()))
        at += n
    return out


def summarize(name, docs):
    d = [v[0] for v in docs.values()]
    m, lo, hi = bootstrap(d)
    print(f"{name}: dNLL vs exact {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  ppl ratio {math.exp(m):.4f}  "
          f"KL {np.mean([v[1] for v in docs.values()]):.4f}  top-1 {np.mean([v[2] for v in docs.values()]):.4f}")
    for s, v in docs.items():
        print(f"    {s:12s} dNLL {v[0]:+.5f}  KL {v[1]:.4f}  top-1 {v[2]:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, help="the KL corpus the reference was recorded from")
    ap.add_argument("--last", type=int, default=512)
    ap.add_argument("reports", nargs="+")
    args = ap.parse_args(argv)
    if len(args.reports) > 2:
        fail("one report, or two to compare (B - A)")
    sources = [json.loads(x)["source"] for x in open(args.corpus, encoding="utf-8") if x.strip()]
    docs = [per_doc(r, sources, args.last) for r in args.reports]
    print(f"last {args.last} positions of each of {len(sources)} docs")
    for r, d in zip(args.reports, docs):
        summarize(r, d)
    if len(docs) == 2:
        m, lo, hi = bootstrap([docs[1][s][0] - docs[0][s][0] for s in sources])
        print(f"paired B - A: {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  ({'B worse' if lo > 0 else 'B better' if hi < 0 else 'CI includes 0'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
