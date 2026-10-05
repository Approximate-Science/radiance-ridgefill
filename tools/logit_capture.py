#!/usr/bin/env python3
"""logit_capture.py -- requests for gate 1 against a server started with RADIANCE_KVA_DUMP_LOGITS=<dir>
(mounted from the host at <host dir>) and --profile-ops (no replayed passes, so every step dumps).

Sends conc.py text's arrangement once -- ONE request whose prompt is [decoder_1 .. decoder_D, the long
prompt], RK_TEXT_TOKENS tokens at temperature 0 -- then each decoder alone, and records which lines of
<host dir>/logits.jsonl each request wrote: <out.json> = {"batched": [lo, hi], "solo": [[lo, hi], ...],
"texts": ...}. tools/logit_compare.py reads it.

usage: RK_DOCS=<ppl.jsonl> logit_capture.py <host dump dir> <out.json>   (RK_PORT, RK_TEXT_D, RK_TEXT_TOKENS)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import conc  # noqa: E402


def lines(path):
    return sum(1 for _ in open(path)) if os.path.exists(path) else 0


def main():
    dump, out = sys.argv[1], sys.argv[2]
    conc.BASE = f"http://127.0.0.1:{os.environ.get('RK_PORT', '8100')}"
    jl = os.path.join(dump, "logits.jsonl")
    d = int(os.environ.get("RK_TEXT_D", "4"))
    tokens = int(os.environ.get("RK_TEXT_TOKENS", "128"))
    qs = ["Implement an LRU cache in Go with a fixed capacity and a small test.",
          "Explain why merge sort is O(n log n), step by step.",
          "Write a short story about a lighthouse keeper who finds a map.",
          "Describe how a hash table resolves collisions, with examples."]
    prompts = [conc.tokenize(f"Question: {q}\nAnswer:") for q in qs[:d]]
    docs = [json.loads(l)["prompt"] for l in open(os.environ["RK_DOCS"]) if l.strip()]
    long_ids = conc.long_prompt(32768, [conc.tokenize(t) for t in docs], 1)
    res = {"solo": [], "texts": {}}
    conc.wait_idle()
    lo = lines(jl)
    r = conc.post("/v1/completions", {"model": "m", "prompt": prompts + [long_ids], "max_tokens": tokens, "temperature": 0})
    res["batched"] = [lo, lines(jl)]
    res["texts"]["batched"] = [c["text"] for c in sorted(r["choices"], key=lambda c: c["index"])]
    for i, p in enumerate(prompts):
        conc.wait_idle()
        lo = lines(jl)
        r = conc.post("/v1/completions", {"model": "m", "prompt": p, "max_tokens": tokens, "temperature": 0})
        res["solo"].append([lo, lines(jl)])
        res["texts"][f"solo{i}"] = r["choices"][0]["text"]
    json.dump(res, open(out, "w"), indent=1)
    print(f"logit_capture: batched lines {res['batched']}, solo {res['solo']}")


if __name__ == "__main__":
    main()
