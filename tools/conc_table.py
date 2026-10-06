#!/usr/bin/env python3
"""conc_table.py -- R55 / R56 from conc.py ttft results: settled medians and the ratios the rows read.

Per (length, C): the long prompt's prompt_ms median over the settled reps (reps[skip:], A.1's warmed
protocol reads reps 3-7), and the decoders' median token gap during the prefill and alone (the median
of the per-rep medians). Then, for each RidgeFill result against the exact one: the speedup exact/RidgeFill at each
C, R55's ratio speedup(C) / speedup(0), and R56's ratio RidgeFill gap / exact gap during the prefill, each
with a bootstrap 95% interval over the settled reps (unpaired, 2,000 resamples, seed 0).

usage: conc_table.py <exact.json> <ridgefill.json> [<ridgefill.json> ...] [--skip 2]
"""
import argparse
import json
import random
import statistics


def settled(res, skip):
    """(length, C) -> {"prompt": [...], "gap": [...], "alone": [...]} over the settled reps."""
    out = {}
    for r in res["reps"]:
        if r["rep"] < skip:
            continue
        cell = out.setdefault((r["length"], r["decoders"]), {"prompt": [], "gap": [], "alone": []})
        cell["prompt"].append(r["prompt_ms"])
        for key, field in (("gap", "decoder_gap_prefill_ms"), ("alone", "decoder_gap_alone_ms")):
            if r.get(field):
                cell[key].append(r[field]["median"])
    return out


def boot_ratio(num, den, rng, n=2000):
    """Bootstrap 95% interval of median(num) / median(den), unpaired."""
    xs = sorted(statistics.median(rng.choices(num, k=len(num))) / statistics.median(rng.choices(den, k=len(den)))
                for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n)]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exact")
    ap.add_argument("ridgefill", nargs="+")
    ap.add_argument("--skip", type=int, default=2)
    a = ap.parse_args()
    rng = random.Random(0)
    ex = settled(json.load(open(a.exact)), a.skip)
    print("| arm | length | C | prompt_ms | speedup | R55 speedup(C)/speedup(0) | decoder gap prefill ms | "
          "R56 RidgeFill/exact gap | decoder gap alone ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (length, c), e in sorted(ex.items()):
        print(f"| exact | {length} | {c} | {med(e['prompt']):.0f} | 1 | | {med(e['gap']):.1f} | | {med(e['alone']):.1f} |")
    for path in a.ridgefill:
        res = json.load(open(path))
        kv = settled(res, a.skip)
        for (length, c), k in sorted(kv.items()):
            e = ex.get((length, c))
            if not e:
                continue
            sp = med(e["prompt"]) / med(k["prompt"])
            lo, hi = boot_ratio(e["prompt"], k["prompt"], rng)
            r55 = ""
            k0, e0 = kv.get((length, 0)), ex.get((length, 0))
            if c and k0 and e0:
                r55 = f"{sp / (med(e0['prompt']) / med(k0['prompt'])):.3f}"
            r56 = ""
            if k["gap"] and e["gap"]:
                glo, ghi = boot_ratio(k["gap"], e["gap"], rng)
                r56 = f"{med(k['gap']) / med(e['gap']):.3f} [{glo:.3f}, {ghi:.3f}]"
            print(f"| {res['label']} | {length} | {c} | {med(k['prompt']):.0f} | {sp:.3f} [{lo:.3f}, {hi:.3f}] | "
                  f"{r55} | {med(k['gap']):.1f} | {r56} | {med(k['alone']):.1f} |")


if __name__ == "__main__":
    main()
