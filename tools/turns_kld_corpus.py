#!/usr/bin/env python3
"""turns_kld_corpus.py -- R64's corpus (redefined 2026-10-05): append-only two-turn pairs for the KL mode.

For each document of two KL corpora cut from the same sources -- turn 1 from the shorter (quick9-off1024:
k*2048 + 1024 tokens), turn 2 from the longer (quick9: (k+1)*2048), whose text must START WITH turn 1's --
writes turn 1 then turn 2, sources prefixed "turn1/" and "turn2/". In the KL mode at --max-num-seqs 1 the
docs run in order, so on a prefix-cache server turn 2 resumes from turn 1's last checkpoint; turn 2 keeps
quick9's score_from (its last T positions), which is where cached and recomputed turn 2 are compared,
each against the exact reference of this same corpus.

usage: turns_kld_corpus.py <turn1 corpus.jsonl> <turn2 corpus.jsonl> <out.jsonl>
"""
import json
import sys


def main():
    t1 = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    t2 = {j["source"]: j for j in (json.loads(l) for l in open(sys.argv[2]) if l.strip())}
    with open(sys.argv[3], "w") as out:
        for a in t1:
            b = t2.get(a["source"])
            if not b or not b["prompt"].startswith(a["prompt"]) or len(b["prompt"]) <= len(a["prompt"]):
                sys.exit(f"turns_kld_corpus: {a['source']}: turn 2 is not an append of turn 1")
            out.write(json.dumps({"prompt": a["prompt"], "score_from": a["score_from"], "source": "turn1/" + a["source"]}) + "\n")
            out.write(json.dumps({"prompt": b["prompt"], "score_from": b["score_from"], "source": "turn2/" + b["source"]}) + "\n")
    print(f"turns_kld_corpus: {len(t1)} pairs -> {sys.argv[3]}")


if __name__ == "__main__":
    main()
