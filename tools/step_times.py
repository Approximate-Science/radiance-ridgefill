#!/usr/bin/env python3
"""step_times.py -- unprofiled per-step wall time by step kind, from conc.py's timestamped log windows.

The engine prints one line per step as it plans it (RADIANCE_LOG_STEPS=1); docker's --timestamps
stamps each line, so the time between two consecutive step lines is one step's wall time (host
planning + device). Grouped per rep by kind: the long prompt's big chunks (n_tok > 1,000), its short
chunks (the 64-row checkpoint remainders, 20 < n_tok <= 1,000) and decode-only steps; printed as the
median over the settled reps of each kind's mean ms. --profile-ops is not used: it restores
synchronisations and its times do not compare across arms.

usage: step_times.py <conc-label.json> [--skip 2]
"""
import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime

STEP = re.compile(r"^(\S+) .*step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")


def stamp(s):
    """docker's RFC 3339 stamp with nanoseconds -> datetime (microseconds kept)."""
    head, _, frac = s.rstrip("Z").partition(".")
    return datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}")


def kinds(path):
    steps = [(stamp(m.group(1)), int(m.group(5))) for m in map(STEP.match, open(path, errors="replace")) if m]
    out = defaultdict(list)
    for (t0, tok), (t1, _) in zip(steps, steps[1:]):
        kind = "big" if tok > 1000 else "short" if tok > 20 else "decode"
        out[kind].append(1000 * (t1 - t0).total_seconds())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--skip", type=int, default=2)
    a = ap.parse_args()
    res = json.load(open(a.result))
    logdir = a.result[:-len(".json")] + ".logs"
    cells = defaultdict(lambda: defaultdict(list))
    for r in res["reps"]:
        if r["rep"] < a.skip or not r.get("log"):
            continue
        for kind, ms in kinds(os.path.join(logdir, r["log"]["file"])).items():
            cells[(r["length"], r["decoders"])][kind].append((len(ms), statistics.mean(ms)))
    for (length, c), ks in sorted(cells.items()):
        parts = [f"{k} n {statistics.median(x[0] for x in v):.0f} x {statistics.median(x[1] for x in v):6.1f} ms"
                 for k, v in sorted(ks.items())]
        print(f"{res['label']} {length} C={c}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
