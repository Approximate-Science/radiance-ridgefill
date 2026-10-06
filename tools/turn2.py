#!/usr/bin/env python3
"""turn2.py -- R64 (redefined): the second turn of a conversation whose first turn's KV comes from the prefix cache,
against the same turn recomputed (--no-prefix-cache) and against exact, through the HTTP logits capture.

  turn2.py build --docs JSONL --out CONV.json [--n 5] [--context 12000] [--answer 64]
  turn2.py run   --conv CONV.json --dump HOST_DUMP_DIR --out MANIFEST.json [--answer 64]

WHY: with RidgeFill on, a cached prefix holds the KV the first turn computed -- its bulk approximated, its last T rows
exact -- and the second turn reuses it; without the cache the second turn approximates its whole bulk afresh. The
bar is that the cached second turn is no further from exact than the recomputed one, within noise.

build (against the REFERENCE server, the exact engine): n conversations from the corpus documents (one each, starting
at document 3), each turn 1 = a document cut to `context` tokens + a question, answered greedily (`answer` tokens);
turn 2 = turn 1 + that answer + a follow-up. Everything is kept as token ids, so every server is sent the same ids.

run (against the server under test, started with RADIANCE_RIDGEFILL_DUMP_LOGITS=<dir> -- mounted from HOST_DUMP_DIR -- and
--profile-ops so no pass is replayed): per conversation, turn 1 + its answer as one prefill (max_tokens 1, so a
cache server holds exactly turn 2's prefix), then turn 2, greedy, `answer` tokens. The manifest records which lines
of <HOST_DUMP_DIR>/logits.jsonl turn 2 wrote ({"solo": [[lo, hi], ...]}, tools/logit_compare.py's form: side
"<dir>:<manifest>:solo<i>:0"), the cached-token count each turn 2 reported, and its text.

Standard library only (tools/conc.py's HTTP helpers).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conc  # noqa: E402

Q1 = "\n\nUser: Summarise the document above in three sentences.\nAssistant:"
Q2 = "\nUser: Now list the three most specific facts the document states, each with where it appears.\nAssistant:"


def lines(path):
    return sum(1 for _ in open(path)) if os.path.exists(path) else 0


def complete(ids, n):
    r = conc.post("/v1/completions", {"model": "m", "prompt": ids, "max_tokens": n, "temperature": 0})
    return r["choices"][0]["text"], r.get("usage") or {}


def build(a):
    docs = [json.loads(x)["prompt"] for x in open(a.docs) if x.strip()]
    convs = []
    for i in range(a.n):
        doc = conc.tokenize(docs[3 + i])
        if len(doc) < a.context:
            conc.die(f"document {3 + i} has {len(doc)} tokens, fewer than --context {a.context}")
        turn1 = conc.tokenize("User: Read this document.\n\n") + doc[:a.context] + conc.tokenize(Q1)
        conc.wait_idle()
        text, _ = complete(turn1, a.answer)
        answer = conc.tokenize(text) if text else []
        convs.append({"doc": 3 + i, "turn1": turn1, "answer_text": text, "answer": answer,
                      "turn2": turn1 + answer + conc.tokenize(Q2)})
        print(f"conversation {i}: turn 1 {len(turn1)} tokens, answer {len(answer)}, turn 2 {len(convs[-1]['turn2'])}",
              flush=True)
    json.dump(convs, open(a.out, "w"))


def run(a):
    convs = json.load(open(a.conv))
    jl = os.path.join(a.dump, "logits.jsonl")
    res = {"solo": [], "cached": [], "texts": []}
    for i, c in enumerate(convs):
        conc.wait_idle()
        complete(c["turn1"] + c["answer"], 1)
        conc.wait_idle()
        lo = lines(jl)
        text, usage = complete(c["turn2"], a.answer)
        conc.wait_idle()
        res["solo"].append([lo, lines(jl)])
        res["cached"].append((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        res["texts"].append(text)
        print(f"conversation {i}: turn 2 {usage.get('prompt_tokens')} tokens, cached {res['cached'][-1]}, "
              f"dump lines {res['solo'][-1]}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--docs", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=5)
    b.add_argument("--context", type=int, default=12000)
    b.add_argument("--answer", type=int, default=64)
    r = sub.add_parser("run")
    r.add_argument("--conv", required=True)
    r.add_argument("--dump", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--answer", type=int, default=64)
    for p in (b, r):
        p.add_argument("--port", default=os.environ.get("RK_PORT", "8100"))
    a = ap.parse_args()
    conc.BASE = f"http://127.0.0.1:{a.port}"
    build(a) if a.cmd == "build" else run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
