#!/usr/bin/env python3
"""conc.py -- a long prompt beside decoding users, against the server already running (scripts/conc.sh).

Two measurements, both on one server in one mode (Stage B, PLAN-FIX §3, REQUIREMENTS-FIX R54-R60):

  ttft   the fnconc pattern (radiance scripts/fnconc.sh), BOTH halves on one clock: C streaming
         decoders (ignore_eos, temperature 0) reach steady state, then one long prompt (max_tokens
         1) is sent; recorded per rep: the long prompt's timings.prompt_ms (R55), every decoder
         token's arrival time -- the gaps between them alone and during the prefill are the
         decoders' ms/step (R56) -- the engine's step counters over the prefill, and the engine log
         lines of the prefill window: the plugin's "kva: approximate step" lines and, with
         RADIANCE_LOG_STEPS=1, the engine's step lines, so the approximate-step count can be held
         to the rule step by step (R57). C = 0 is the solo prefill.
  text   the tiercross pattern made deterministic: ONE completions request whose prompt is a batch
         [decoder_1 .. decoder_D, long prompt] -- the server admits them together, so every run of
         every mode puts the same decoder token beside the same prefill chunk -- max_tokens
         RK_TEXT_TOKENS at temperature 0; recorded: each decoder's text and its sha256, the long
         prompt's, the draft counters (R60), and each decoder alone. R54 compares a mode's decoder
         texts with `off`'s, byte for byte.

Inputs (env): RK_PORT (8100), RK_DOCS (JSONL of {"prompt"}, the long prompt's source text, cut to
exactly the length with a leading nonce), RK_LENGTHS ("16384 32768"), RK_CONC ("0 1 4 8"), RK_REPS
(7), RK_CONTAINER (the server's container, for its log), RK_EVIDENCE/RK_STAGE (output dir), RK_TEXT_D
("1 4"), RK_TEXT_TOKENS (256), RK_TEXT_LENGTH (32768). Output: <dir>/conc-<label>.json and, per rep,
the log window <dir>/conc-<label>.logs/<length>-c<C>-r<rep>.log.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

BASE = ""
TIMEOUT = 3600.0


def die(msg):
    print(f"conc: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def metrics():
    """The engine's counters, name -> value, summed over a name's label sets."""
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        out = {}
        for line in r.read().decode().splitlines():
            if line and not line.startswith("#"):
                name, _, value = line.rpartition(" ")
                name = name.split("{")[0].strip()
                try:
                    out[name] = out.get(name, 0.0) + float(value)
                except ValueError:
                    pass
        return out


def wait_idle(limit=300):
    for _ in range(limit):
        if metrics().get("vllm:num_requests_running", 0) == 0:
            return
        time.sleep(1)
    die("the server never went idle")


def tokenize(text):
    ids = post("/tokenize", {"prompt": text, "add_special_tokens": False}).get("tokens")
    if not ids:
        die(f"/tokenize returned nothing for {text[:60]!r}")
    return ids


def long_prompt(length, doc_ids, n):
    """A nonce then the docs' ids, cut to exactly `length` (speed.py's rule: a fresh nonce a request)."""
    ids = tokenize(f"Measurement nonce {n}.\n")
    while len(ids) < length:
        for d in doc_ids:
            ids.extend(d)
    return ids[:length]


class Decoder(threading.Thread):
    """One streaming decoder: every token's arrival time, until told to stop (then it disconnects,
    which cancels the request)."""

    def __init__(self, prompt):
        super().__init__(daemon=True)
        self.prompt, self.times, self.stop, self.error = prompt, [], threading.Event(), None

    def run(self):
        body = {"model": "m", "prompt": self.prompt, "max_tokens": 8192, "ignore_eos": True,
                "temperature": 0, "stream": True}
        req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                for line in r:
                    if line.startswith(b"data: {"):
                        self.times.append(time.monotonic())
                    if self.stop.is_set():
                        return
        except Exception as e:   # a disconnect we asked for is not an error
            if not self.stop.is_set():
                self.error = repr(e)


def gaps(decoders, lo, hi):
    """Milliseconds between consecutive tokens of each decoder whose later token arrived in [lo, hi]."""
    out = []
    for d in decoders:
        out += [1000 * (b - a) for a, b in zip(d.times, d.times[1:]) if lo <= b <= hi]
    return out


def summary(xs):
    if not xs:
        return None
    s = sorted(xs)
    return {"n": len(s), "median": statistics.median(s), "p90": s[int(0.9 * (len(s) - 1))], "max": s[-1]}


def log_window(since, until, path):
    """The container's log between two wall-clock instants, kept beside the result."""
    container = os.environ.get("RK_CONTAINER")
    if not container:
        return None
    p = subprocess.run(["docker", "logs", "--since", since, "--until", until, container],
                       capture_output=True, text=True)
    text = p.stdout + p.stderr
    with open(path, "w") as f:
        f.write(text)
    lines = text.splitlines()
    approx = [l for l in lines if "kva: approximate step" in l]
    return {"approximate_steps": len(approx), "step_lines": sum(" step " in l and "n_seq=" in l for l in lines),
            "pn2_steps": sum("Pn 2," in l for l in approx), "file": os.path.basename(path)}


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def one_rep(length, c, n, doc_ids, logdir, tag):
    wait_idle()
    decoders = [Decoder(f"d{tag}{i} count upward and do not stop:") for i in range(c)]
    for d in decoders:
        d.start()
    deadline = time.monotonic() + 120
    while c and min(len(d.times) for d in decoders) < 16:
        if time.monotonic() > deadline:
            die("decoders did not reach 16 tokens in 120 s")
        time.sleep(0.05)
    a0 = time.monotonic()
    time.sleep(2.0 if c else 0)
    ids = long_prompt(length, doc_ids, n)
    m0, w0, t0 = metrics(), utc(), time.monotonic()
    r = post("/v1/completions", {"model": "m", "prompt": ids, "max_tokens": 1, "temperature": 0})
    t1, w1, m1 = time.monotonic(), utc(), metrics()
    for d in decoders:
        d.stop.set()
    for d in decoders:
        d.join(timeout=60)
    usage, timings = r.get("usage") or {}, r.get("timings") or {}
    if usage.get("prompt_tokens") != length:
        die(f"prompt_tokens {usage.get('prompt_tokens')} != {length}")
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        die("a cache hit: --no-prefix-cache is not in effect")
    delta = {k: m1.get(k, 0) - m0.get(k, 0) for k in
             ("radiance:engine_steps_total", "radiance:engine_steps_mixed_total",
              "radiance:engine_steps_prefill_total", "radiance:decode_tokens_total")}
    steps = delta["radiance:engine_steps_total"]
    return {"length": length, "decoders": c, "prompt_ms": timings.get("prompt_ms"),
            "wall_ms": 1000 * (t1 - t0), "since": w0, "until": w1,
            "decoder_gap_alone_ms": summary(gaps(decoders, a0, t0)),
            "decoder_gap_prefill_ms": summary(gaps(decoders, t0, t1)),
            "counters": delta, "ms_per_step_wall": 1000 * (t1 - t0) / steps if steps else None,
            "decoder_errors": [d.error for d in decoders if d.error],
            "log": log_window(w0, w1, os.path.join(logdir, f"{length}-c{c}-{tag}.log"))}


def run_ttft(label, out_dir):
    docs = [json.loads(l)["prompt"] for l in open(os.environ["RK_DOCS"]) if l.strip()]
    doc_ids = [tokenize(d) for d in docs]
    lengths = [int(x) for x in os.environ.get("RK_LENGTHS", "16384 32768").split()]
    concs = [int(x) for x in os.environ.get("RK_CONC", "0 1 4 8").split()]
    reps = int(os.environ.get("RK_REPS", "7"))
    logdir = os.path.join(out_dir, f"conc-{label}.logs")
    os.makedirs(logdir, exist_ok=True)
    result, n = {"label": label, "kind": "ttft", "reps": []}, 0
    for length in lengths:
        for c in concs:
            for rep in range(reps):
                n += 1
                rec = one_rep(length, c, n, doc_ids, logdir, f"r{rep}")
                rec["rep"] = rep
                result["reps"].append(rec)
                g = rec["decoder_gap_prefill_ms"]
                print(f"  {length:6d} C={c} rep {rep}: prompt {rec['prompt_ms']:9.1f} ms"
                      f"  decoder gap median {g['median'] if g else float('nan'):7.1f} ms"
                      f"  approx lines {rec['log']['approximate_steps'] if rec['log'] else '-'}", flush=True)
    return result


def run_text(label, out_dir):
    docs = [json.loads(l)["prompt"] for l in open(os.environ["RK_DOCS"]) if l.strip()]
    doc_ids = [tokenize(d) for d in docs]
    length = int(os.environ.get("RK_TEXT_LENGTH", "32768"))
    tokens = int(os.environ.get("RK_TEXT_TOKENS", "256"))
    reps = int(os.environ.get("RK_REPS", "3"))
    questions = ["Implement an LRU cache in Go with a fixed capacity and a small test.",
                 "Explain why merge sort is O(n log n), step by step.",
                 "Write a short story about a lighthouse keeper who finds a map.",
                 "Describe how a hash table resolves collisions, with examples."]
    result = {"label": label, "kind": "text", "runs": []}
    for d in [int(x) for x in os.environ.get("RK_TEXT_D", "1 4").split()]:
        prompts = [tokenize(f"Question: {q}\nAnswer:") for q in questions[:d]]
        for rep in range(reps):
            wait_idle()
            m0, w0 = metrics(), utc()
            body = {"model": "m", "prompt": prompts + [long_prompt(length, doc_ids, 1)],
                    "max_tokens": tokens, "temperature": 0}
            r = post("/v1/completions", body)
            m1, w1 = metrics(), utc()
            texts = [c["text"] for c in sorted(r["choices"], key=lambda c: c["index"])]
            run = {"decoders": d, "rep": rep, "texts": texts[:d],
                   "sha": [hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts],
                   "draft": {k: m1.get(k, 0) - m0.get(k, 0) for k in
                             ("radiance:draft_tokens_total", "radiance:draft_accepted_total")},
                   "log": log_window(w0, w1, os.path.join(out_dir, f"text-{label}-d{d}-r{rep}.log"))}
            result["runs"].append(run)
            print(f"  text D={d} rep {rep}: {' '.join(run['sha'])}  draft {run['draft']}", flush=True)
        for i, p in enumerate(prompts):   # each decoder alone: R54's "equals A solo" half
            wait_idle()
            r = post("/v1/completions", {"model": "m", "prompt": p, "max_tokens": tokens, "temperature": 0})
            t = r["choices"][0]["text"]
            result["runs"].append({"decoders": d, "solo": i, "texts": [t],
                                   "sha": [hashlib.sha256(t.encode()).hexdigest()[:16]]})
            print(f"  text D={d} solo {i}: {result['runs'][-1]['sha'][0]}", flush=True)
    return result


def run_pair(label, out_dir):
    """R58': two long prompts in ONE batched request (admitted together) beside RK_CONC decoders."""
    docs = [json.loads(l)["prompt"] for l in open(os.environ["RK_DOCS"]) if l.strip()]
    doc_ids = [tokenize(d) for d in docs]
    length = int(os.environ.get("RK_LENGTHS", "32768").split()[0])
    c = int(os.environ.get("RK_CONC", "8").split()[-1])
    result = {"label": label, "kind": "pair", "reps": []}
    for rep in range(int(os.environ.get("RK_REPS", "1"))):
        wait_idle()
        decoders = [Decoder(f"p{rep}{i} count upward and do not stop:") for i in range(c)]
        for d in decoders:
            d.start()
        while c and min(len(d.times) for d in decoders) < 16:
            time.sleep(0.05)
        m0, w0, t0 = metrics(), utc(), time.monotonic()
        prompts = [long_prompt(length, doc_ids, 2 * rep + 1), long_prompt(length, doc_ids, 2 * rep + 2)]
        r = post("/v1/completions", {"model": "m", "prompt": prompts, "max_tokens": 1, "temperature": 0})
        t1, w1, m1 = time.monotonic(), utc(), metrics()
        for d in decoders:
            d.stop.set()
        for d in decoders:
            d.join(timeout=60)
        rec = {"rep": rep, "length": length, "decoders": c, "wall_ms": 1000 * (t1 - t0),
               "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
               "decoder_gap_prefill_ms": summary(gaps(decoders, t0, t1)),
               "steps": m1.get("radiance:engine_steps_total", 0) - m0.get("radiance:engine_steps_total", 0),
               "log": log_window(w0, w1, os.path.join(out_dir, f"pair-{label}-r{rep}.log"))}
        result["reps"].append(rec)
        print(f"  pair rep {rep}: wall {rec['wall_ms']:.0f} ms, approximate lines {rec['log']['approximate_steps'] if rec['log'] else '-'}"
              f", with two prefills {rec['log']['pn2_steps'] if rec['log'] else '-'}", flush=True)
    return result


def main():
    global BASE
    kinds = {"ttft": run_ttft, "text": run_text, "pair": run_pair}
    if len(sys.argv) != 3 or sys.argv[1] not in kinds:
        die("usage: conc.py ttft|text|pair <label>")
    kind, label = sys.argv[1], sys.argv[2]
    BASE = f"http://127.0.0.1:{os.environ.get('RK_PORT', '8100')}"
    out_dir = os.path.join(os.environ["RK_EVIDENCE"], os.environ.get("RK_STAGE", "scratch"))
    os.makedirs(out_dir, exist_ok=True)
    res = kinds[kind](label, out_dir)
    res["finished"] = utc()
    path = os.path.join(out_dir, f"conc-{label}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"conc: wrote {path}")


if __name__ == "__main__":
    main()
