#!/usr/bin/env python3
"""conc_steps.py -- R57: every step of a long prompt that the rule says is approximate logged one.

Reads a conc.py ttft result and its per-rep log windows (the engine's RADIANCE_LOG_STEPS lines and
the plugin's "kva: approximate step" lines) and, step by step, recomputes PLAN-FIX §2's rule from the
engine's own numbers: the long prompt's chunk q = n_tok - (n_seq - 1)(1 + spec) (each decoder verifies 1 + spec rows;
the long prompt is the step's last entry), its context = the sum of its earlier chunks, n_ahead =
min(N - ctx - q, max_tok), b = n_tok if n_ahead >= T else n_tok - roundup_G(T - n_ahead), s_lb =
n_tok - q; approximate iff n_ahead > 0 and b > s_lb. Each expected step must be followed by a plugin
line with the same n_tok and n_ahead (any path); a shortfall names the step. A replayed pass logs
nothing, so a shortfall is first checked against replay before it is called a miss.

usage: conc_steps.py <conc-label.json> [--tail 2048] [--tile 64] [--max-tok 2048] [--spec 0]
"""
import argparse
import json
import os
import re

STEP = re.compile(r"step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")
KVA = re.compile(r"approximate step \((\w+), (\d+) tokens, (\d+) ahead, b (\d+), s_lb (\d+), D (\d+), Pn (\d+)")


def expected(steps, length, a):
    """[(step, n_tok, n_ahead, approximate)] for the long prompt's chunks, from the engine's step lines."""
    out, ctx = [], 0
    for number, phase, n_seq, n_tok in steps:
        if phase == "decode" or ctx >= length:
            continue
        q = n_tok - (n_seq - 1) * (1 + a.spec)
        ahead = min(length - ctx - q, a.max_tok)
        ctx += q
        if ahead <= 0:
            out.append((number, n_tok, 0, False))
            continue
        need = a.tail - ahead
        b = n_tok if ahead >= a.tail else n_tok - (need + a.tile - 1) // a.tile * a.tile
        out.append((number, n_tok, ahead, b > n_tok - q))
    return out


def check(rep, logdir, a):
    text = open(os.path.join(logdir, rep["log"]["file"])).read().splitlines()
    steps, kva, last = [], {}, None
    for line in text:
        m = STEP.search(line)
        if m:
            last = int(m.group(1))
            steps.append((last, m.group(2), int(m.group(3)), int(m.group(4))))
            continue
        k = KVA.search(line)
        if k and last is not None:
            kva[last] = (int(k.group(2)), int(k.group(3)), k.group(1))
    want = expected(steps, rep["length"], a)
    misses = [(s, n, h) for s, n, h, ok in want if ok and (s not in kva or kva[s][:2] != (n, h))]
    extra = [s for s in kva if s not in {w[0] for w in want if w[3]}]
    return sum(w[3] for w in want), len(kva), misses, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--tail", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--max-tok", type=int, default=2048)
    ap.add_argument("--spec", type=int, default=0, help="draft depth: each decoder verifies 1 + spec rows")
    a = ap.parse_args()
    res = json.load(open(a.result))
    logdir = a.result[:-len(".json")] + ".logs"
    bad = 0
    for rep in res["reps"]:
        if not rep.get("log"):
            continue
        want, got, misses, extra = check(rep, logdir, a)
        bad += bool(misses or extra)
        print(f"{rep['length']:6d} C={rep['decoders']} rep {rep['rep']}: expected {want} approximate, "
              f"logged {got}" + (f"; MISSING at steps {misses}" if misses else "") +
              (f"; UNEXPECTED at steps {extra}" if extra else ""))
    print("conc_steps: every expected step logged" if not bad else f"conc_steps: {bad} rep(s) differ")


if __name__ == "__main__":
    main()
