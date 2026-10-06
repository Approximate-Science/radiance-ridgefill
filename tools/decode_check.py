#!/usr/bin/env python3
"""decode_check.py -- is a decode gap after a long prompt the expert mover's backlog, or the step itself?

  decode_check.py --docs JSONL --out OUT.json [--reps 3] [--long 16384] [--short 200] [--answer 256] [--idle 60]

WHY: the release session's round a measured decode right after a 16K prompt in the same request -- stock 11.2-11.9
ms/token, quality 8.1-9.7 -- while a request moved ~131 GB host->device on stock and ~36-39 GB on quality (the
quality pass skips the late layers' experts). If stock's prefill churns the expert cache and its decode pays the mover's
backlog, the gap closes once the mover has settled: S4/ediag2 had settled decode equal (10.70 vs 10.71 ms/step).

Two arms, `reps` each, one request at a time:
  after-long   a `long`-token prompt + `answer` greedy tokens in ONE request (what round a measured)
  settled      a `long`-token prompt (max_tokens 1), then the server idle >= `idle` s, then a `short`-token prompt +
               `answer` greedy tokens
Per answered request: decode ms/token (timings.predicted_per_token_ms), tokens a step (draft counts, depth 3), and the
rank-0 /stats mover and link counters before and after it (deltas). Prompts are tools/speed.py's (a leading nonce
each, so no prefix-cache hit). Standard library only.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speed as S  # noqa: E402

KEYS = (("experts", "promotions"), ("experts", "demotions"), ("experts", "readmits"), ("experts", "routed"),
        ("experts", "routed_res"), ("link", "h2d_bytes"), ("link", "stream_bytes"), ("link", "staged_bytes"))


def stats():
    with urllib.request.urlopen(S.BASE + "/stats", timeout=10) as r:
        s = json.loads(r.read())
    return {f"{a}.{b}": (s.get(a) or {}).get(b) or 0 for a, b in KEYS}


def idle_for(seconds):
    """Wait until no request is running, then `seconds` more of idleness."""
    deadline = time.time() + 600
    while time.time() < deadline:
        with urllib.request.urlopen(S.BASE + "/metrics", timeout=30) as r:
            running = sum(float(l.rpartition(" ")[2]) for l in r.read().decode().splitlines()
                          if l.startswith("vllm:num_requests_running"))
        if running == 0:
            break
        time.sleep(1)
    time.sleep(seconds)


def ask(ids, n, depth=3):
    before = stats()
    r = S.http_json("POST", S.BASE + "/v1/completions",
                    {"model": "m", "prompt": ids, "max_tokens": n, "temperature": 0, "ignore_eos": True})
    after = stats()
    t = r.get("timings") or {}
    dn, da = t.get("draft_n") or 0, t.get("draft_n_accepted") or 0
    steps = dn / depth if dn else 0
    return {"prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), "prompt_ms": t.get("prompt_ms"),
            "decode_ms_per_token": t.get("predicted_per_token_ms"),
            "tokens_per_step": (da + steps) / steps if steps else None,
            "mover": {k: after[k] - before[k] for k in before}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--long", type=int, default=16384)
    ap.add_argument("--short", type=int, default=200)
    ap.add_argument("--answer", type=int, default=256)
    ap.add_argument("--idle", type=float, default=60)
    ap.add_argument("--port", default="8100")
    a = ap.parse_args()
    S.BASE = f"http://127.0.0.1:{a.port}"
    doc_ids = [S.tokenize(d["prompt"]) for d in S.load_docs(a.docs)]
    out = {"after-long": [], "settled": []}
    for i in range(a.reps):
        idle_for(5)
        out["after-long"].append(ask(S.build_prompt(a.long, doc_ids, S.tokenize(f"Decode check {i}.\n")), a.answer))
        idle_for(5)
        ask(S.build_prompt(a.long, doc_ids, S.tokenize(f"Decode warm {i}.\n")), 1)
        idle_for(a.idle)
        rec = ask(S.build_prompt(a.short, doc_ids, S.tokenize(f"Short {i}.\n")), a.answer)
        rec["idle_s"] = a.idle
        out["settled"].append(rec)
        for k in out:
            x = out[k][-1]
            print(f"rep {i} {k:10s}: decode {x['decode_ms_per_token']:.2f} ms/tok, tokens/step "
                  f"{x['tokens_per_step'] and round(x['tokens_per_step'], 3)}, h2d {x['mover']['link.h2d_bytes'] / 2**30:.1f} GiB, "
                  f"promotions {x['mover']['experts.promotions']}", flush=True)
    for k in out:
        print(f"median {k}: decode {statistics.median(x['decode_ms_per_token'] for x in out[k]):.2f} ms/tok")
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
