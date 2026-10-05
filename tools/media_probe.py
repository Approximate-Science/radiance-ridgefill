#!/usr/bin/env python3
"""media_probe.py -- R75's instrument: the next-token distribution after an image + long text, read through the
engine's buffer probe, against the same text without the image.

  media_probe.py run --corpus DIR/media.jsonl --container NAME --out OUT.json [--cuts "44000 48000 52000 56000"]
  media_probe.py compare REF.json CAND.json

WHY THE PROBE: the server returns no logprobs and the KL mode reads text corpora only, so the tail NLL R75 names cannot
be had for a prompt with a picture in it. The engine's own answer for a serving comparison (core/engine_probe.cpp,
"WHY THIS SHAPE AND NOT A PERPLEXITY") is N distinct prompts at max_tokens 1 with the sampler's candidate planes
dumped: `sampler.cand_idx` / `sampler.cand_val` hold the step's top candidates with global ids and their logits.
The server must run with -vv and RK_DOCKER_EXTRA="-e RADIANCE_DUMP_BUF=sampler.cand_idx,sampler.cand_val
-e RADIANCE_DUMP_HEAD=20"; it synchronises every step, so its timings mean nothing.

run: for every picture of the corpus (tools/media_corpus.py) and every cut, two chat prompts that differ only by the
picture -- [image, text[:cut] + Q] and [text[:cut] + Q], Q asking for the next word of the text, which the cut leaves
mid-sentence -- each at max_tokens 1, top_k 20, temperature 1. After each response, the last cand_idx/cand_val dump
lines in `docker logs NAME` since the request was sent are the prompt's last row. Prompts run one at a time.

compare: per prompt, top-1 agreement and the top-K KL(P_ref || Q_cand) over the reference's top 20, both renormalised
over that set (a reference id missing from the candidate's 20 takes the candidate's 20th logit, so the KL is a lower
bound there); means split by picture / text-only. R75's question is whether the picture rows' gap to the reference
is within the text-only rows' gap (the band) -- the same 20 texts with and without a picture before them.

Standard library only.
"""
import argparse
import base64
import datetime
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

Q = "\n\nReply with only the next word of the text above, which stops mid-sentence."
DUMP = re.compile(r"step (\d+)\S*\s+sampler\.cand_(idx|val)\s+head((?:\s+\S+)+)\s*$")


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def last_dump(container, since):
    out = subprocess.run(["docker", "logs", "--since", since, container], capture_output=True, text=True)
    got = {}
    for line in (out.stdout + out.stderr).splitlines():
        m = DUMP.search(line)
        if m:
            got[m.group(2)] = (int(m.group(1)), m.group(3).split())
    if set(got) != {"idx", "val"} or got["idx"][0] != got["val"][0]:
        return None
    return {"step": got["idx"][0], "idx": [int(x) for x in got["idx"][1]], "val": [float(x) for x in got["val"][1]]}


def run(a):
    corpus = Path(a.corpus)
    rows = []
    for doc in (json.loads(x) for x in corpus.read_text().splitlines() if x.strip()):
        text = json.loads(Path(doc["docs"]).read_text().splitlines()[doc["doc_index"]])["prompt"]
        url = "data:image/png;base64," + base64.b64encode((corpus.parent / doc["image"]).read_bytes()).decode()
        for cut in (int(c) for c in a.cuts.split()):
            for picture in (True, False):
                parts = ([{"type": "image_url", "image_url": {"url": url}}] if picture else []) + \
                        [{"type": "text", "text": text[:cut] + Q}]
                since = datetime.datetime.now(datetime.timezone.utc).isoformat()
                r = post(a.port, {"model": "m", "messages": [{"role": "user", "content": parts}], "max_tokens": 1,
                                  "temperature": 1.0, "top_k": 20, "seed": 75,
                                  "chat_template_kwargs": {"enable_thinking": False}})
                time.sleep(0.5)
                d = last_dump(a.container, since)
                if d is None:
                    sys.exit(f"media_probe: no cand_idx/cand_val dump for {doc['id']} cut {cut}: is the server at -vv "
                             f"with RADIANCE_DUMP_BUF set?")
                rec = {"id": doc["id"], "cut": cut, "picture": picture,
                       "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), **d}
                rows.append(rec)
                print(f"{doc['id']} cut {cut} {'picture' if picture else 'text   '} prompt {rec['prompt_tokens']} "
                      f"top1 {d['idx'][0]}", flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


def kl_top(ref, cand):
    """KL(P_ref || Q_cand) over the reference's candidate ids, both renormalised over them."""
    floor = min(cand["val"])
    q_of = dict(zip(cand["idx"], cand["val"]))
    lp, lq = ref["val"], [q_of.get(i, floor) for i in ref["idx"]]
    zp, zq = max(lp), max(lq)
    sp = math.log(sum(math.exp(x - zp) for x in lp)) + zp
    sq = math.log(sum(math.exp(x - zq) for x in lq)) + zq
    return sum(math.exp(p - sp) * ((p - sp) - (q - sq)) for p, q in zip(lp, lq))


def compare(a):
    ref = {(r["id"], r["cut"], r["picture"]): r for r in json.loads(Path(a.ref).read_text())}
    cand = {(r["id"], r["cut"], r["picture"]): r for r in json.loads(Path(a.cand).read_text())}
    for picture in (True, False):
        keys = sorted(k for k in ref if k[2] == picture and k in cand)
        top1 = [ref[k]["idx"][0] == cand[k]["idx"][0] for k in keys]
        kls = [kl_top(ref[k], cand[k]) for k in keys]
        print(f"{'picture  ' if picture else 'text-only'}: {len(keys)} prompts  top-1 agreement {sum(top1)}/{len(keys)}  "
              f"top-20 KL mean {sum(kls) / len(kls):.5f}  max {max(kls):.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--corpus", required=True)
    r.add_argument("--container", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cuts", default="44000 48000 52000 56000")
    r.add_argument("--port", default="8100")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("cand")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
