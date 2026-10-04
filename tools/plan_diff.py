#!/usr/bin/env python3
"""The gate before `rad-convert --reuse C --in-place`: compare the run's plan with what C holds.

  plan_diff.py --plan PLAN.log --container RAD_INFO_V.txt [--expect-new SHARD.safetensors ...] [--out R.json]

PLAN.log is the stderr of `rad-convert ... --plan-only -v`; RAD_INFO_V.txt is the stdout of `rad-info -v C`.
--expect-new names the shard(s) whose tensors are the only weights allowed to be new (repeatable); without it
no weight may be new, which is the check AFTER an append: the extended container must hold exactly the plan.

Why it is needed: --plan-only prints every planned weight but not which are new, reused or missing, and the
in-place writer refuses a weight whose encoding changed but NOT one that is missing from the plan -- the new
header names only what this run planned, so a weight left out silently leaves the container
(notes/sidecar.md §7). PASS = new names are exactly the expected set, none dropped, none changed (encoding,
or quantiser + options).
"""
import argparse
import json
import re
import sys
from pathlib import Path

UNITS = {"B", "KiB", "MiB", "GiB", "TiB"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plan_rows(text):
    """{name: (encoding, provenance)} from `rad-convert --plan-only -v` debug lines
    (`<tag> rad_convert.cpp:N    name  encoding  size unit  as is | quantiser options`)."""
    rows = {}
    for line in ANSI.sub("", text).splitlines():
        site, sep, msg = line.partition("  ")
        if "rad_convert.cpp:" not in site or not sep:
            continue
        t = msg.split()
        if len(t) >= 5 and t[3] in UNITS:
            rows[t[0]] = (t[1], "checkpoint" if t[4:] == ["as", "is"] else " ".join(t[4:]))
    return rows


def container_rows(text):
    """{name: (encoding, provenance)} from `rad-info -v` tensor rows
    (`name  encoding  shape  size unit  @offset  checkpoint | quantiser options`)."""
    rows = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) >= 7 and t[4] in UNITS and t[5].startswith("@"):
            rows[t[0]] = (t[1], " ".join(t[6:]))
    return rows


def shard_names(path):
    with open(path, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    return {k for k in header if k != "__metadata__"}


def diff(plan, held, expected):
    new = sorted(set(plan) - set(held))
    return dict(planned=len(plan), held=len(held), new=new, dropped=sorted(set(held) - set(plan)),
                changed=sorted(n for n in set(plan) & set(held) if plan[n] != held[n]),
                unexpected_new=sorted(set(new) - expected), expected_missing=sorted(expected - set(new)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="stderr of rad-convert --plan-only -v")
    ap.add_argument("--container", required=True, help="stdout of rad-info -v on the container")
    ap.add_argument("--expect-new", action="append", default=[], help="shard whose tensors may be new")
    ap.add_argument("--out", help="report JSON")
    args = ap.parse_args(argv)
    for path in [args.plan, args.container, *args.expect_new]:
        if not Path(path).is_file():
            raise SystemExit(f"{path} not found")
    plan = plan_rows(Path(args.plan).read_text(encoding="utf-8", errors="replace"))
    held = container_rows(Path(args.container).read_text(encoding="utf-8", errors="replace"))
    if not plan or not held:
        raise SystemExit(f"parsed {len(plan)} plan rows and {len(held)} container rows; need both (was -v given?)")
    expected = set().union(set(), *(shard_names(p) for p in args.expect_new))
    report = diff(plan, held, expected)
    ok = not (report["dropped"] or report["changed"] or report["unexpected_new"] or report["expected_missing"])
    for key in ("new", "dropped", "changed", "unexpected_new", "expected_missing"):
        items = report[key]
        print(f"{key:17s} {len(items):5d}  {', '.join(items[:6])}{' ...' if len(items) > 6 else ''}")
    for name in report["changed"][:10]:
        print(f"  changed {name}: container {held[name]} -> plan {plan[name]}")
    print(f"planned {report['planned']}, held {report['held']}: {'PASS' if ok else 'FAIL'}")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(report, passed=ok), indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
