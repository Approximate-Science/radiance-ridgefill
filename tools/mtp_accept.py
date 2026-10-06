#!/usr/bin/env python3
"""mtp_accept.py -- the drafting head's acceptance after a long prefill: R70's instrument (and R97's link bytes).

  mtp_accept.py --docs JSONL --out OUT.json [--length 16384] [--answer 256] [--reps 5] [--depth 3] [--port 8100]

WHAT IT DOES: `reps` requests, each a `length`-token prompt built from the corpus exactly as tools/speed.py builds
its prompts (its own functions; a different leading nonce each, so no two prompts share a prefix) and an
`answer`-token greedy continuation on /v1/completions. Per request: timings.draft_n and draft_n_accepted (the engine
reports them when the server drafts), predicted_n, decode ms a token, the text's sha256, and the delta of the rank-0
/stats link counters (h2d_bytes, stream_bytes, staged_bytes) across the request -- what the expert mover and the
stager moved for it.

TOKENS A STEP: a verify step drafts `depth` tokens and emits the accepted ones plus one, so steps = draft_n / depth
and tokens a step = (draft_n_accepted + steps) / steps. Printed per request and as the median over the reps; the
server under test is compared with a stock server of the same flags on the same prompts (R70: RidgeFill with the final map
0.95-1.0x stock, RADIANCE_RIDGEFILL_FINAL=off 0.85-0.92x).

Standard library only.
"""
import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speed as S  # noqa: E402

LINK = ("h2d_bytes", "stream_bytes", "staged_bytes")


def link():
    with urllib.request.urlopen(S.BASE + "/stats", timeout=10) as r:
        s = json.loads(r.read())
    return {k: (s.get("link") or {}).get(k) or 0 for k in LINK}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--answer", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--port", default="8100")
    a = ap.parse_args()
    S.BASE = f"http://127.0.0.1:{a.port}"
    doc_ids = [S.tokenize(d["prompt"]) for d in S.load_docs(a.docs)]
    rows = []
    for i in range(a.reps):
        nonce = f"MTP nonce {i}.\n"
        ids = S.build_prompt(a.length, doc_ids, S.tokenize(nonce))
        before = link()
        r = S.http_json("POST", S.BASE + "/v1/completions",
                        {"model": "m", "prompt": ids, "max_tokens": a.answer, "temperature": 0, "seed": 12345,
                         "ignore_eos": True})
        after = link()
        t = r.get("timings") or {}
        dn, da = t.get("draft_n") or 0, t.get("draft_n_accepted") or 0
        steps = dn / a.depth if dn else 0
        text = r["choices"][0].get("text") or ""
        row = {"nonce": nonce, "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
               "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"),
               "predicted_per_token_ms": t.get("predicted_per_token_ms"), "draft_n": dn, "draft_n_accepted": da,
               "accept": da / dn if dn else None, "tokens_per_step": (da + steps) / steps if steps else None,
               "sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
               "link": {k: after[k] - before[k] for k in LINK}}
        rows.append(row)
        print(f"rep {i}: prompt {row['prompt_tokens']} ({row['prompt_ms']:.0f} ms)  tokens/step "
              f"{row['tokens_per_step'] if row['tokens_per_step'] is None else round(row['tokens_per_step'], 3)}  "
              f"accept {da}/{dn}  decode {row['predicted_per_token_ms']} ms/tok  h2d {row['link']['h2d_bytes'] / 2**20:.0f} MiB  "
              f"text {row['sha256'][:12]}", flush=True)
    tps = [r["tokens_per_step"] for r in rows if r["tokens_per_step"] is not None]
    summary = {"length": a.length, "answer": a.answer, "depth": a.depth, "reps": rows,
               "tokens_per_step_median": statistics.median(tps) if tps else None}
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(f"median tokens/step {summary['tokens_per_step_median']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
