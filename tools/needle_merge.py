#!/usr/bin/env python3
"""needle_merge.py -- the release session's needle glue (scripts/release_session.sh part `needle`).

  needle_merge.py DIR             DIR/corpus-s{1,2,3}.jsonl -> DIR/corpus.jsonl (ids prefixed "S<seed>/", so three
                                  seeds = three sets of distinct keys and numbers) and DIR/smoke.jsonl (seed 1, 16K,
                                  single needle: the five bulk depths and the 0.98 tail control)
  needle_merge.py --gate RESULTS  the smoke gate on stock: PASS when >= 5 of the 6 are right and the tail control is
                                  right (orchestrator, 2026-10-05: a full run is worth nothing if stock cannot answer);
                                  prints the replies either way
"""
import json
import sys


def merge(d):
    with open(f"{d}/corpus.jsonl", "w") as out, open(f"{d}/smoke.jsonl", "w") as smoke:
        for s in (1, 2, 3):
            for line in open(f"{d}/corpus-s{s}.jsonl"):
                item = json.loads(line)
                item["id"] = f"S{s}/{item['id']}"
                out.write(json.dumps(item) + "\n")
                if s == 1 and item["length"] == 16384 and item["variant"] == "single":
                    smoke.write(json.dumps(item) + "\n")


def gate(path):
    r = [json.loads(line) for line in open(path) if line.strip()]
    tail = [x for x in r if x["region"] == "tail"]
    right = sum(x["correct"] for x in r)
    ok = len(r) == 6 and right >= 5 and tail and all(x["correct"] for x in tail)
    print(("PASS" if ok else "FAIL") + f" {right}/{len(r)} (tail {sum(x['correct'] for x in tail)}/{len(tail)}); replies: "
          + " | ".join(repr(x["reply"][:60]) for x in r))


if __name__ == "__main__":
    gate(sys.argv[2]) if sys.argv[1] == "--gate" else merge(sys.argv[1])
