#!/usr/bin/env python3
"""settle.py -- how long a freshly started server takes to reach its settled prefill speed, and what the expert mover
does meanwhile. Run right after scripts/serve.sh reports /health; it starts nothing.

  settle.py --out OUT.json [--cycles 15] [--lengths "1024 2048 4096 8192"] [--poll 1.0]

WHAT IT MEASURES: `cycles` rounds of one prefill-only request at each length (max_tokens 1, temperature 0, a
different leading nonce each, --no-prefix-cache asserted as tools/speed.py does), back to back, timings.prompt_ms
and wall time of each; and, every `poll` seconds on a second thread, the engine's /stats counters that say what the
placement engine is doing (experts: promotions, demotions, readmits, routed vs routed-resident, dispatches, staged
units; link: mover h2d bytes, streamed-expert bytes, staged bytes, all-reduce bytes; cards: util, power, temp).

SETTLED: per length, the reference is the median of the last 4 cycles; the settle cycle is the first cycle from which
every later value stays within 2% of it; time-to-settled is that request's start, seconds after the script began
(which is right after /health). Printed per length with the first-cycle and settled values, and the mover counters at
the settle point and at the end.

Prompts are built from RK_DOCS exactly as tools/speed.py builds them (its own functions). Standard library only.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speed as S  # noqa: E402

EXPERT_KEYS = ("promotions", "demotions", "readmits", "distinct_promoted", "routed", "routed_res", "dispatches",
               "refused", "starved_slab", "staged_units")
LINK_KEYS = ("h2d_bytes", "stream_bytes", "staged_bytes", "ar_bytes")


def poll_stats(stop, every, out, t0):
    while not stop.is_set():
        try:
            with urllib.request.urlopen(S.BASE + "/stats", timeout=5) as r:
                s = json.loads(r.read())
            e, link = s.get("experts") or {}, s.get("link") or {}
            row = {"t": round(time.time() - t0, 2)}
            row.update({k: e.get(k) for k in EXPERT_KEYS})
            row.update({k: link.get(k) for k in LINK_KEYS})
            row["cards"] = [{k: c.get(k) for k in ("index", "util", "power_w", "temp_c")} for c in s.get("cards", [])]
            out.append(row)
        except Exception as ex:  # a missed poll is a gap in the series, not a failed measurement
            out.append({"t": round(time.time() - t0, 2), "error": str(ex)[:120]})
        stop.wait(every)


def settle_point(values, tol=0.02, tail=4):
    """(reference, first index from which every value is within tol of the reference)."""
    ref = statistics.median(values[-tail:])
    k = len(values) - 1
    while k > 0 and abs(values[k - 1] / ref - 1) <= tol:
        k -= 1
    return ref, k


def stats_at(stats, t):
    rows = [r for r in stats if "error" not in r and r["t"] <= t]
    return rows[-1] if rows else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=15)
    ap.add_argument("--lengths", default="1024 2048 4096 8192")
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--port", default="8100")
    ap.add_argument("--docs", required=True)
    a = ap.parse_args()
    S.BASE = f"http://127.0.0.1:{a.port}"
    lengths = [int(x) for x in a.lengths.split()]
    doc_ids = [S.tokenize(d["prompt"]) for d in S.load_docs(a.docs)]
    t0 = time.time()
    stats, stop = [], threading.Event()
    th = threading.Thread(target=poll_stats, args=(stop, a.poll, stats, t0), daemon=True)
    th.start()
    reqs, n = [], 0
    for cycle in range(a.cycles):
        for L in lengths:
            n += 1
            nonce = f"Settle nonce {n}.\n"
            ts = time.time() - t0
            rec = S.measure_once(L, nonce, S.tokenize(nonce), doc_ids)
            reqs.append({"cycle": cycle, "len": L, "t_start": round(ts, 2), "t_end": round(time.time() - t0, 2),
                         "prompt_ms": rec["prompt_ms"]})
    stop.set()
    th.join()
    report = {"lengths": lengths, "cycles": a.cycles, "requests": reqs, "stats": stats, "per_length": {}}
    end = stats_at(stats, 1e9)
    for L in lengths:
        series = [r for r in reqs if r["len"] == L]
        vals = [r["prompt_ms"] for r in series]
        ref, k = settle_point(vals)
        at = stats_at(stats, series[k]["t_start"])
        report["per_length"][L] = {"first": vals[0], "settled": ref, "settle_cycle": k,
                                   "t_settled": series[k]["t_start"], "series": vals}
        mv = lambda r: f"prom {r['promotions']} dem {r['demotions']} readm {r['readmits']} res {r['routed_res']}/{r['routed']}" if r else "-"
        print(f"{L:6d}: first {vals[0]:8.1f}  settled {ref:8.1f}  at cycle {k:2d} (t {series[k]['t_start']:6.1f} s)  "
              f"mover then [{mv(at)}]  end [{mv(end)}]")
    Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
