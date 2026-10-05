#!/usr/bin/env python3
"""hazard_rate.py -- Stage C (#2, prefix cache on): which requests' exact tail overlapped
cached positions, at what rate, and does the log-side count agree with the plugin's device
counter (R65, R68).

  hazard_rate.py --records responses.jsonl [--records more.jsonl ...]
                   [--plugin-log server.log] [--tail 2048] [--out report.json] [--require-match]

`--records` lines are JSON objects carrying the request's prompt length and the prefix-cache
hit count, in any of these shapes (the first that resolves wins):

  {"id": "branch/000/B", "prompt_len": 8749, "cache_n": 4096}          compact
  {"id": "...", "prompt_tokens": 8749, "cache_n": 4096}                compact, response spelling
  {"id": "...", "usage": {"prompt_tokens": ...}, "timings": {"cache_n": ..., "prompt_n": ...}}
                                                                       a whole chat/completion response

prompt_len falls back to timings.cache_n + timings.prompt_n (GUIDE.md §5.9: prompt_n + cache_n
is the prompt).  A record missing both spellings is refused by line number.

A request is FLAGGED when its exact tail [prompt_len - T, prompt_len) reached positions the
prefix cache supplied -- the superset condition of PLAN-FIX §5.4:

  flagged:  cache_n > prompt_len - T          (max(0, ...) guards a prompt shorter than T)

The count of overlapped positions is  max(0, cache_n - max(0, prompt_len - T)).  This is a
SUPERSET of the true hazard (positions the producer approximated): an append-only turn can
overlap the cache in its tail yet be exact, which is why C of the branch corpus is built with
a computed prompt >= T and must NOT be flagged, while a branch (N2 - T < P) always is.

`--plugin-log` parses the log line the KVA plugin prints once per request that resumed from a
cached checkpoint, reporting its device hazard counter (PLAN-FIX §5.4):

  HAZARD_LOG_RE below is the contract.  It accepts
      kva: hazard <positions> positions
      kva: hazard request <id> <positions> positions
  (extra text around the marker is ignored; a `request <id>` names the request so the two
  counts can be cross-checked per request, not just in total).  On the branch corpus the
  plugin counter must equal the records-side overlap (R65: the counter equals
  min(N1-N2, T-(N2-P)), which branch_corpus.py constructs to equal max(0, P-(N2-T))); on
  append-only traffic the plugin must report 0 and hazard_rate must flag nothing (R68).
  --require-match exits 1 when the two instruments disagree.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The plugin's hazard log line -- the contract with the engine-side instrument
# (PLAN-FIX §5.4).  One line per request that resumed from a cached checkpoint:
#
#     kva: hazard 1395 positions
#     kva: hazard request branch/000/B 1395 positions
#
# `positions` is the device hazard counter for that request (named group "positions"; the
# optional `request <id>` names it, named group "request").  A line that matches nothing is
# counted as unparsed and reported, never silently dropped.
# ---------------------------------------------------------------------------
HAZARD_LOG_RE = re.compile(
    r"\bkva:\s+hazard\b(?:\s+request\s+(?P<request>\S+))?\s+(?P<positions>\d+)\s+positions\b")


def die(msg):
    print(f"hazard_rate: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Records.
# ---------------------------------------------------------------------------

def prompt_len_of(rec, lineno):
    for key in ("prompt_len", "prompt_tokens"):
        if isinstance(rec.get(key), int):
            return rec[key]
    usage = rec.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
        return usage["prompt_tokens"]
    timings = rec.get("timings") or {}
    if isinstance(timings, dict) and isinstance(timings.get("cache_n"), int) \
            and isinstance(timings.get("prompt_n"), int):
        return timings["cache_n"] + timings["prompt_n"]
    return None


def cache_n_of(rec, lineno):
    if isinstance(rec.get("cache_n"), int):
        return rec["cache_n"]
    timings = rec.get("timings")
    if isinstance(timings, dict) and isinstance(timings.get("cache_n"), int):
        return timings["cache_n"]
    return None


def load_records(paths):
    rows = []
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            die(f"--records {path} unreadable: {e}")
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                die(f"{path}:{lineno}: not JSON: {e}")
            prompt_len = prompt_len_of(rec, lineno)
            cache_n = cache_n_of(rec, lineno)
            if prompt_len is None or prompt_len < 0:
                die(f"{path}:{lineno}: no prompt length (prompt_len / prompt_tokens / "
                    f"usage.prompt_tokens / timings.cache_n + timings.prompt_n)")
            if cache_n is None or cache_n < 0:
                die(f"{path}:{lineno}: no timings.cache_n")
            timings = rec.get("timings") or {}
            rows.append({
                "id": rec.get("id") or rec.get("request") or f"{Path(path).name}:{lineno}",
                "prompt_len": prompt_len,
                "cache_n": cache_n,
                "prompt_n": timings.get("prompt_n") if isinstance(timings, dict) else None,
            })
    if not rows:
        die(f"no records in: {', '.join(str(p) for p in paths)}")
    return rows


# ---------------------------------------------------------------------------
# The plugin log.
# ---------------------------------------------------------------------------

def parse_plugin_log(path):
    """{"lines": n, "positions_total": s, "by_request": {id: positions}, "unnamed": [..],
    "unparsed": [lines the contract regex did not match]}."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        die(f"--plugin-log unreadable: {e}")
    out = {"lines": 0, "positions_total": 0, "by_request": {}, "unnamed": [], "unparsed": []}
    for lineno, line in enumerate(text.splitlines(), 1):
        if "kva" not in line:          # fast reject; the regex is the authority
            continue
        m = HAZARD_LOG_RE.search(line)
        if not m:
            out["unparsed"].append(f"{lineno}: {line.strip()[:120]}")
            continue
        positions = int(m.group("positions"))
        out["lines"] += 1
        out["positions_total"] += positions
        request = m.group("request")
        if request is None:
            out["unnamed"].append(positions)
        else:
            out["by_request"][request] = out["by_request"].get(request, 0) + positions
    return out


# ---------------------------------------------------------------------------
# The measurement.
# ---------------------------------------------------------------------------

def flag_and_overlap(prompt_len, cache_n, tail):
    """(flagged, overlap): did the exact tail [N-T, N) reach cached positions, and how many.

    flagged  iff cache_n > max(0, N - T)  -- PLAN-FIX §5.4's superset condition
              cache_n > prompt_len - T, with max(0, ..) so a prompt shorter than T is judged
              by whether anything was cached at all, not by a negative tail start.
    overlap  = max(0, cache_n - max(0, N - T)), the number of tail positions the cache served.
    """
    tail_start = max(0, prompt_len - tail)
    return cache_n > tail_start, max(0, cache_n - tail_start)


def summarize(rows, plugin, tail):
    per_request, flagged, overlap_total = [], 0, 0
    for row in rows:
        flagged_i, overlap = flag_and_overlap(row["prompt_len"], row["cache_n"], tail)
        flagged += flagged_i
        overlap_total += overlap
        per_request.append(dict(row, tail_start=row["prompt_len"] - tail,
                                flagged=flagged_i, overlap=overlap,
                                plugin_positions=(plugin["by_request"].get(row["id"])
                                                  if plugin is not None else None)))
    # UNNAMED LINES ARE PAIRED IN ORDER. The plugin cannot know request ids; it logs one line per
    # request whose counter moved, after that request ran. When every line is unnamed and there is one
    # per flagged request, the i-th line belongs to the i-th flagged request (both chronological).
    if plugin is not None and plugin["unnamed"] and not plugin["by_request"]:
        flagged_rows = [r for r in per_request if r["flagged"]]
        if len(flagged_rows) == len(plugin["unnamed"]):
            for r, positions in zip(flagged_rows, plugin["unnamed"]):
                r["plugin_positions"] = positions
    summary = {
        "tail": tail,
        "n_requests": len(rows),
        "n_flagged": flagged,
        "rate": flagged / len(rows),
        "overlap_total": overlap_total,
        "per_request": per_request,
        "plugin": plugin,
    }
    if plugin is not None:
        differing = [{"id": r["id"], "overlap": r["overlap"],
                      "plugin_positions": r["plugin_positions"]}
                     for r in per_request
                     if r["plugin_positions"] is not None
                     and r["plugin_positions"] != r["overlap"]]
        records_only = [r["id"] for r in per_request
                        if r["flagged"] and r["plugin_positions"] is None]
        plugin_only = sorted(set(plugin["by_request"]) - {r["id"] for r in per_request})
        agree = (plugin["positions_total"] == overlap_total and not differing
                 and not records_only)
        summary["cross_check"] = {
            "agree": agree,
            "plugin_lines": plugin["lines"],
            "plugin_positions_total": plugin["positions_total"],
            "records_overlap_total": overlap_total,
            "per_id_differing": differing,
            "flagged_without_plugin_line": records_only,
            "plugin_ids_without_record": plugin_only,
            "unnamed_plugin_lines": list(plugin["unnamed"]),
        }
    return summary


def print_report(summary, sources):
    tail = summary["tail"]
    print(f"hazard_rate: tail T = {tail}; {summary['n_requests']} requests from "
          f"{', '.join(str(s) for s in sources)}")
    print(f"{'id':24s} {'prompt_len':>10s} {'cache_n':>8s} {'tail_start':>10s} "
          f"{'flagged':>7s} {'overlap':>7s} {'plugin':>7s}")
    for r in summary["per_request"]:
        print(f"{r['id']:24s} {r['prompt_len']:10d} {r['cache_n']:8d} {r['tail_start']:10d} "
              f"{('yes' if r['flagged'] else '-'):>7s} {r['overlap']:7d} "
              f"{(str(r['plugin_positions']) if r['plugin_positions'] is not None else '-'):>7s}")
    print(f"flagged {summary['n_flagged']} of {summary['n_requests']} "
          f"(rate {summary['rate']:.3f}); overlapped positions from records: "
          f"{summary['overlap_total']}")
    if (cc := summary.get("cross_check")) is not None:
        print(f"plugin log: {cc['plugin_lines']} hazard lines, {cc['plugin_positions_total']} "
              f"positions total; records say {cc['records_overlap_total']}")
        if cc["unnamed_plugin_lines"]:
            print(f"  (unnamed plugin lines, counted in the total only: "
                  f"{cc['unnamed_plugin_lines']})")
        for d in cc["per_id_differing"]:
            print(f"  DIFFER {d['id']}: records overlap {d['overlap']} vs plugin "
                  f"{d['plugin_positions']} positions")
        for i in cc["flagged_without_plugin_line"]:
            print(f"  no plugin line for flagged request {i}")
        for i in cc["plugin_ids_without_record"]:
            print(f"  plugin line for a request not in the records: {i}")
        print(f"cross-check: {'AGREE' if cc['agree'] else 'DISAGREE'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hazard_rate.py",
                                description=__doc__.splitlines()[0])
    ap.add_argument("--records", nargs="+", required=True,
                    help="JSONL of response records (see the module docstring for the shapes)")
    ap.add_argument("--plugin-log", default=None,
                    help="server log to parse the plugin's `kva: hazard ... positions` lines "
                         f"(contract: {HAZARD_LOG_RE.pattern})")
    ap.add_argument("--tail", type=int, default=2048,
                    help="T, the exact tail (kva.tail; default 2048)")
    ap.add_argument("--out", default=None, help="write the machine-readable summary here")
    ap.add_argument("--require-match", action="store_true",
                    help="exit 1 when the records and the plugin log disagree (needs "
                         "--plugin-log)")
    args = ap.parse_args(argv)
    if args.tail < 1:
        die("--tail must be positive")
    if args.require_match and not args.plugin_log:
        die("--require-match needs --plugin-log")

    rows = load_records(args.records)
    plugin = parse_plugin_log(args.plugin_log) if args.plugin_log else None
    summary = summarize(rows, plugin, args.tail)
    print_report(summary, args.records)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    if args.require_match and not summary["cross_check"]["agree"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())