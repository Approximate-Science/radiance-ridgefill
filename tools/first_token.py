#!/usr/bin/env python3
"""first_token.py -- is a decoder's solo-vs-concurrent divergence at its first token a near-tie?

For each of conc.py's text questions: the first generated token and its logprob solo, and inside the
batched arrangement [questions..., the 32K prompt] (conc.py text's shape), against the running server.
The server returns the chosen token's logprob only (no alternatives, core/server/oai.cpp:1831-1845),
so a near-tie shows as two DIFFERENT first tokens each near log(1/2) = -0.693; a token chosen at
p ~ 1 in one arm and replaced in the other is not the ident.sh noise class.

usage: RK_DOCS=<ppl.jsonl> first_token.py <out.json>   (RK_PORT, RK_TEXT_LENGTH as conc.py)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import conc  # noqa: E402

conc.BASE = f"http://127.0.0.1:{os.environ.get('RK_PORT', '8100')}"
qs = ["Implement an LRU cache in Go with a fixed capacity and a small test.",
      "Explain why merge sort is O(n log n), step by step.",
      "Write a short story about a lighthouse keeper who finds a map.",
      "Describe how a hash table resolves collisions, with examples."]
prompts = [conc.tokenize(f"Question: {q}\nAnswer:") for q in qs]
docs = [json.loads(l)["prompt"] for l in open(os.environ["RK_DOCS"]) if l.strip()]
long_ids = conc.long_prompt(int(os.environ.get("RK_TEXT_LENGTH", "32768")), [conc.tokenize(d) for d in docs], 1)


def top(choice):
    lp = choice.get("logprobs") or {}
    return [(lp.get("tokens") or ["?"])[0], (lp.get("token_logprobs") or [float("nan")])[0]]


out = {"solo": [], "batched": []}
for p in prompts:
    r = conc.post("/v1/completions", {"model": "m", "prompt": p, "max_tokens": 1, "temperature": 0, "logprobs": 5})
    out["solo"].append(top(r["choices"][0]))
r = conc.post("/v1/completions", {"model": "m", "prompt": prompts + [long_ids], "max_tokens": 1,
                                  "temperature": 0, "logprobs": 5})
for c in sorted(r["choices"], key=lambda c: c["index"])[:4]:
    out["batched"].append(top(c))
for i in range(4):
    (a, la), (b, lb) = out["solo"][i], out["batched"][i]
    print(f"q{i}: solo {a!r} {la:.4f}  batched {b!r} {lb:.4f}  {'same token' if a == b else 'DIFFERENT token'}")
json.dump(out, open(sys.argv[1], "w"), indent=1)
