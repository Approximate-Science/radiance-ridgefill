#!/usr/bin/env python3
"""branch_send.py -- send tools/branch_corpus.py's triples to the running prefix-cache server (R65, R68).

Per triple, in order: A (the producer; its prefill writes the checkpoints), then B (the branch, which
resumes inside A's document and ends sooner), then C (the append-only continuation); each as one
/v1/chat/completions request, max_tokens 1, temperature 0. After each, a one-token flush request: the
plugin logs its hazard counter on a LATER step, so the flush makes that line land before the next
triple member. Each response is written to <records.jsonl> as {"id": "<triple id>/<A|B|C>", "usage",
"timings"}, the shape tools/hazard_rate.py --records reads.

usage: branch_send.py <branches.jsonl> <records.jsonl>     (RK_PORT)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import conc  # noqa: E402


def chat(messages):
    return conc.post("/v1/chat/completions", {"model": "m", "messages": messages, "max_tokens": 1,
                                              "temperature": 0})


def main():
    conc.BASE = f"http://127.0.0.1:{os.environ.get('RK_PORT', '8100')}"
    triples = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    with open(sys.argv[2], "w") as out:
        for t in triples:
            for part in ("A", "B", "C"):
                conc.wait_idle()
                r = chat(t[part]["messages"])
                rec = {"id": f"{t['id']}/{part}", "usage": r.get("usage"), "timings": r.get("timings")}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                chat([{"role": "user", "content": "ok"}])
                tm = rec["timings"] or {}
                print(f"  {rec['id']}: prompt {(rec['usage'] or {}).get('prompt_tokens')} cache_n {tm.get('cache_n')} "
                      f"prompt_n {tm.get('prompt_n')}", flush=True)


if __name__ == "__main__":
    main()
