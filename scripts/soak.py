#!/usr/bin/env python3
"""soak.py -- a mixed request load against the server on RK_PORT, for the tape audit (R99).

  soak.py --docs ppl.jsonl --requests 400 --clients 8 [--seed 0] [--out soak.json]

WHY. Every host decision of an approximate pass must be a function of keyed batch fields (R15/R99); a
decision that read anything else would make a replayed recording differ from a fresh issue, and the engine's
tape audit fails such a step with RAD_E_STATE. A mixed load -- short chats decoding while medium and long
prompts prefill beside them -- produces the shapes the plugin approximates beside decoders (D > 0, Pn >= 1)
and the small mixed steps the engine records and replays. This drives it; the caller greps the server log.

LOAD (seeded): 70% short prompts (~100-600 tokens, 48 generated), 25% medium (~3-6K tokens, 16 generated),
5% long (~12-20K tokens, 8 generated), temperature 0.7, from --clients concurrent threads. Prompt text is a
window of a document of --docs (one {"prompt"} per line), cut by characters (~4 a token), with a per-request
nonce first so no two prompts share a prefix. PRINTS a progress line every 50 requests and a summary:
requests ok / failed (HTTP status or exception named), prompt and generated token totals, wall time.
Exits 1 if any request failed. Standard library only.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request

KINDS = [("short", 0.70, (400, 2400), 48), ("medium", 0.25, (12000, 24000), 16), ("long", 0.05, (48000, 80000), 8)]


def plan(docs, n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        r, acc = rng.random(), 0.0
        for kind, share, (lo, hi), gen in KINDS:
            acc += share
            if r <= acc:
                break
        text = docs[rng.randrange(len(docs))]
        chars = min(rng.randrange(lo, hi), len(text) - 1)
        start = rng.randrange(0, len(text) - chars) if len(text) > chars else 0
        out.append((kind, f"Soak request {i}.\n" + text[start:start + chars], gen))
    return out


def send(base, prompt, gen, timeout):
    body = {"model": "m", "prompt": prompt, "max_tokens": gen, "temperature": 0.7, "seed": 7}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", required=True)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    docs = [json.loads(x)["prompt"] for x in open(args.docs, encoding="utf-8") if x.strip()]
    work = plan(docs, args.requests, args.seed)
    base = f"http://127.0.0.1:{os.environ.get('RK_PORT', '8100')}"
    lock, results, nxt = threading.Lock(), [], [0]
    t0 = time.time()

    def client():
        while True:
            with lock:
                if nxt[0] >= len(work):
                    return
                i = nxt[0]
                nxt[0] += 1
            kind, prompt, gen = work[i]
            try:
                u = send(base, prompt, gen, args.timeout).get("usage", {})
                rec = dict(i=i, kind=kind, ok=True, prompt=u.get("prompt_tokens", 0), gen=u.get("completion_tokens", 0))
            except Exception as e:   # noqa: BLE001 -- every failure is reported by name below
                rec = dict(i=i, kind=kind, ok=False, error=f"{type(e).__name__}: {e}")
            with lock:
                results.append(rec)
                if len(results) % 50 == 0:
                    print(f"  {len(results)}/{len(work)} done, {time.time() - t0:.0f} s", flush=True)

    threads = [threading.Thread(target=client) for _ in range(args.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [r for r in results if not r["ok"]]
    summary = dict(requests=len(results), failed=len(bad), wall_s=round(time.time() - t0, 1),
                   prompt_tokens=sum(r.get("prompt", 0) for r in results), gen_tokens=sum(r.get("gen", 0) for r in results),
                   by_kind={k: sum(1 for r in results if r["kind"] == k) for k, *_ in KINDS}, errors=[r["error"] for r in bad][:10])
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(summary=summary, results=sorted(results, key=lambda r: r["i"])), f)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
