#!/usr/bin/env python3
"""profile_steps.py -- per-step op lists out of a --profile-ops log (R16, R26, R36).

  profile_steps.py LOG [--step K] [--from-layer 24] [--json OUT]

The engine prints a CUMULATIVE per-op table every RADIANCE_PROFILE_EVERY steps per rank (radiance
core/runtime/ctx.cpp dump_profile: calls, total ms, us/call, site, join, op, geometry, #index, first
weight). Run with RADIANCE_PROFILE_EVERY=1 and the difference of two consecutive tables of one rank is
the ops that step issued. Each op's layer is read off its first weight (blk.L.*, kva.proj.L.*, kva.st.L);
an op without a weight takes the layer of the nearest declared op before it that has one.

Prints, per rank and per step: the number of ops issued, and for layers >= --from-layer the op names
with their call deltas and mean us/call; --step K prints that table's step in full. Standard library only.
"""
import argparse
import json
import re
from collections import defaultdict

ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s\s(\S+)\s\s(.*?)\s\s#(\d+)\s(\S+)\s*$")
HEAD = re.compile(r"^per-op timing, rank (\d+)")
LAYER = re.compile(r"^(?:blk|kva\.proj|kva\.projr|kva\.st|kva\.stswap|kva\.str)\.(\d+)\b")


def parse(path):
    tables, cur = [], None          # [(rank, {index: (calls, ms, op, geom, wname)})]
    for line in open(path, errors="replace"):
        m = HEAD.match(line.strip())
        if m:
            cur = (int(m.group(1)), {})
            tables.append(cur)
            continue
        if cur is None:
            continue
        r = ROW.match(line.rstrip("\n"))
        if r:
            cur[1][int(r.group(8))] = (int(r.group(1)), float(r.group(2)), r.group(6), r.group(7), r.group(9))
        elif line.strip().startswith("total"):
            cur = None
    return tables


def layer_map(tables):
    seen = {}
    for _, t in tables:
        for i, v in t.items():
            seen[i] = v[4]
    out, last = {}, None
    for i in sorted(seen):
        m = LAYER.match(seen[i])
        if m:
            last = int(m.group(1))
        out[i] = last
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--step", type=int)
    ap.add_argument("--from-layer", type=int, default=24)
    ap.add_argument("--json")
    a = ap.parse_args()
    tables = parse(a.log)
    lay = layer_map(tables)
    prev, steps = defaultdict(dict), []
    for rank, t in tables:
        d = {}
        for i, (calls, ms, op, geom, w) in t.items():
            pc, pm = prev[rank].get(i, (0, 0.0))[:2]
            if calls > pc:
                d[i] = dict(op=op, geom=geom, weight=w, layer=lay.get(i), calls=calls - pc,
                            us=1000.0 * (ms - pm) / (calls - pc))
        prev[rank] = t
        steps.append(dict(rank=rank, ops=d))
    k = defaultdict(int)
    for s in steps:
        k[s["rank"]] += 1
        s["n"] = k[s["rank"]]
        late = defaultdict(lambda: [0, 0.0])
        for x in s["ops"].values():
            if x["layer"] is not None and x["layer"] >= a.from_layer:
                late[x["op"]][0] += x["calls"]
                late[x["op"]][1] += x["us"] * x["calls"]
        lst = ", ".join(f"{op} x{c} ({t / c:.0f} us)" for op, (c, t) in sorted(late.items()))
        print(f"rank {s['rank']} table {s['n']}: {len(s['ops'])} ops issued; layers >= {a.from_layer}: {lst or '(none)'}")
        if a.step is not None and s["n"] == a.step:
            for i, x in sorted(s["ops"].items()):
                print(f"    #{i:<5d} L{x['layer']!s:>4}  {x['op']:<28s} x{x['calls']:<3d} {x['us']:10.1f} us  {x['weight']}  [{x['geom']}]")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(steps, f)


if __name__ == "__main__":
    main()
