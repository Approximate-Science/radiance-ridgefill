#!/usr/bin/env python3
"""turns.py -- two-turn conversations against the running server (Stage C: R62, R64, R66).

Per conversation k (RK_TURN_LENGTHS, default 8192 12288 16384 20480 24576 28672 32768 9216 18432 30720
tokens): turn 1 = a nonce + RK_DOCS text cut to the length, 64 tokens at temperature 0; turn 2 = turn 1's
prompt ids + its answer re-tokenized + a question + RK_TURN_MORE (default 2560 >= T) further doc tokens,
128 tokens at temperature 0 -- append-only, so a prefix cache resumes turn 2 from its last checkpoint
inside turn 1. Recorded per turn: timings.cache_n / prompt_n / prompt_ms and the answer (+ sha256).

usage: RK_DOCS=<ppl.jsonl> turns.py <out.json>     (RK_PORT)
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import conc  # noqa: E402


def ask(ids, tokens):
    r = conc.post("/v1/completions", {"model": "m", "prompt": ids, "max_tokens": tokens, "temperature": 0})
    t, u = r.get("timings") or {}, r.get("usage") or {}
    text = r["choices"][0]["text"]
    return {"prompt_tokens": u.get("prompt_tokens"), "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
            "prompt_ms": t.get("prompt_ms"), "text": text, "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}


def main():
    conc.BASE = f"http://127.0.0.1:{os.environ.get('RK_PORT', '8100')}"
    docs = [json.loads(l)["prompt"] for l in open(os.environ["RK_DOCS"]) if l.strip()]
    doc_ids = [conc.tokenize(d) for d in docs]
    pool = [t for d in doc_ids for t in d]
    lengths = [int(x) for x in os.environ.get("RK_TURN_LENGTHS",
                                              "8192 12288 16384 20480 24576 28672 32768 9216 18432 30720").split()]
    more = int(os.environ.get("RK_TURN_MORE", "2560"))
    question = conc.tokenize("\n\nQuestion: summarise the passage above in two sentences.\nAnswer:")
    out = []
    for k, n in enumerate(lengths):
        conc.wait_idle()
        ids1 = conc.long_prompt(n, doc_ids, 1000 + k)
        t1 = ask(ids1, 64)
        start = (k * 7919) % (len(pool) - more)
        ids2 = ids1 + conc.tokenize(t1["text"]) + question + pool[start:start + more]
        t2 = ask(ids2, 128)
        out.append({"conversation": k, "length": n, "turn1": t1, "turn2": t2})
        print(f"  conv {k} ({n}): turn1 cache_n {t1['cache_n']}  turn2 cache_n {t2['cache_n']} prompt_n "
              f"{t2['prompt_n']} prompt_ms {t2['prompt_ms']:.0f}  answer {t2['sha']}", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
