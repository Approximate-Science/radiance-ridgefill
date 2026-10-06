#!/usr/bin/env python3
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""speed.py -- prefill latency at exact prompt lengths against a running server.

Called by scripts/speed.sh (which checks the inputs); see that header for the protocol.
This file is the measurement. Python standard library only.

WHAT IT DOES
  1. reads the docs from RK_DOCS (JSONL, one {"prompt", ...} per line) and tokenises each
     once through the served model's own tokenizer: POST /tokenize
     {"prompt": ..., "add_special_tokens": false} -> {"count", "max_model_len", "tokens"}
     (radiance core/server/admin.cpp:119-196; add_special_tokens=false so the length is
     exact and no BOS is silently prepended);
  2. for every length in RK_LENGTHS: one warm-up then RK_REPS requests, each with a
     DIFFERENT short leading nonce, the docs' token ids concatenated and cut to the
     target length minus the nonce;
  3. sends the token ids directly: /v1/completions accepts "prompt": [ids]
     (radiance core/server/oai.cpp:1549-1568, "token arrays must contain integers only"),
     so the length is exact by construction; usage.prompt_tokens is verified against the
     target on every response and a mismatch fails loudly with both numbers;
  4. records timings.prompt_ms (GUIDE.md §5.9), usage.prompt_tokens and the cached-token
     count (usage.prompt_tokens_details.cached_tokens and timings.cache_n; both must be 0:
     --no-prefix-cache is on, and a cache hit would answer a rep for free and read as a
     speedup);
  5. runs scripts/preflight.sh before every sample (the warm-up included) and aborts on
     failure;
  6. writes $RK_EVIDENCE/$RK_STAGE/speed-<label>.json with per-rep values, median and
     min/max per length, the preflight lines, the exact-tail note per length and the
     server's /server_info (or /v1/models) response for provenance, then prints the table.

Env vars (defaults in scripts/speed.sh and scripts/common.sh):
  RK_DOCS               required: JSONL with a "prompt" field per line
  RK_LENGTHS            default "9216 16384 32768": prompt token counts, EXACT
  RK_REPS               default 5
  RK_PORT               default 8100
  RK_STAGE              default scratch
  RK_EVIDENCE           default <repo>/evidence
  RK_SPEED_HTTP_TIMEOUT default 3600 (seconds per request)
  RK_NONCE_TEXT         default "Measurement nonce {n}.\\n": the request with sequence
                        number n gets nonce n (n counts across the whole run)
"""

import json
import os
import re
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE = None            # "http://127.0.0.1:<port>", set in main
HTTP_TIMEOUT = 3600.0
PREFLIGHT_PATH = None


def die(msg):
    print(f"speed: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def http_json(method, url, body=None, required=True, parse=True):
    """One HTTP request; dies with the endpoint and status named on failure.
    parse=False: only care about the status (for /health), return True on 200.
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            if not parse:
                return True
            payload = resp.read().decode()
    except urllib.error.HTTPError as e:
        if not required:
            return None
        die(f"{method} {url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
    except urllib.error.URLError as e:
        if not required:
            return None
        die(f"{method} {url} failed: {e.reason}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        die(f"{method} {url} answered something that is not JSON: {e}")


def tokenize(text):
    """Token ids for text, via the served model's /tokenize (add_special_tokens off)."""
    r = http_json("POST", BASE + "/tokenize",
                  {"prompt": text, "add_special_tokens": False})
    ids = r.get("tokens") if isinstance(r, dict) else None
    if not isinstance(ids, list) or not ids:
        die(f"/tokenize returned no tokens for: {text[:80]!r}")
    if r.get("count") != len(ids):
        die(f"/tokenize count {r.get('count')} != {len(ids)} tokens for {text[:80]!r}")
    return ids


def load_docs(path):
    """The docs of RK_DOCS; only the "prompt" field is used."""
    docs = []
    try:
        f = open(path, encoding="utf-8")
    except OSError as e:
        die(f"RK_DOCS unreadable: {e}")
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError as e:
                die(f"{path}:{lineno}: not JSON: {e}")
            if "prompt" not in j:
                die(f'{path}:{lineno}: no "prompt" field')
            docs.append(j)
    if not docs:
        die(f"RK_DOCS holds no documents: {path}")
    return docs


def build_prompt(target_len, doc_ids, nonce_ids):
    """nonce_ids + the docs' ids concatenated and cut to target_len - len(nonce_ids).

    Pure: takes the pieces, returns the id list, exactly target_len long.
    """
    need = target_len - len(nonce_ids)
    if need <= 0:
        die(f"the nonce is {len(nonce_ids)} tokens, longer than the target {target_len}")
    total = sum(len(d) for d in doc_ids)
    if total < need:
        die(f"RK_DOCS holds {total} tokens in total, less than the {need} needed "
            f"for a {target_len}-token prompt")
    ids = []
    while len(ids) < need:
        for d in doc_ids:
            ids.extend(d)
    return nonce_ids + ids[:need]


def tail_note(length):
    """The exact-tail size the plugin is expected to leave at this length.

    The chunk is 2048 (RK_FLAGS --max-num-batched-tokens) and RidgeFill's T is 2048: a chunk
    with fewer than T tokens after it runs exact. At a multiple of 2048 the tail is
    exactly T; at 9216 the chunk at 6144 sees n_ahead = 1024 < T and runs exact, so the
    tail is 3072, not 2048 (HANDOVER §5 Stage 0 step 5).
    """
    chunk = 2048
    if length < 2 * chunk:
        return length, (f"{length} is below one {chunk} chunk: everything runs exact")
    if length % chunk == 0:
        return chunk, (f"{length} is a multiple of the {chunk} chunk: "
                       f"the exact tail is exactly T = {chunk}")
    start = (length // chunk - 1) * chunk
    ahead = length - (start + chunk)
    tail = chunk + length % chunk
    return tail, (f"{length} is {length / chunk:.1f} chunks: the chunk at {start} sees "
                  f"n_ahead = {ahead} < T = {chunk} and runs exact, so the exact tail is "
                  f"{tail}, not {chunk}")


def preflight():
    """Run scripts/preflight.sh; abort the measurement on failure."""
    try:
        p = subprocess.run([PREFLIGHT_PATH], capture_output=True, text=True)
    except OSError as e:
        die(f"cannot run {PREFLIGHT_PATH}: {e}")
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        die(f"preflight failed (exit {p.returncode}); the sample is not taken")
    return p.stdout.splitlines()


def measure_once(target_len, nonce_text, nonce_ids, doc_ids):
    """One prefill request; returns the per-rep record or dies naming the mismatch."""
    ids = build_prompt(target_len, doc_ids, nonce_ids)
    body = {"model": "m", "prompt": ids, "max_tokens": 1, "temperature": 0}
    r = http_json("POST", BASE + "/v1/completions", body)
    timings = r.get("timings") or {}
    usage = r.get("usage") or {}
    prompt_ms = timings.get("prompt_ms")
    if prompt_ms is None:
        die(f"response has no timings.prompt_ms: {json.dumps(r)[:300]}")
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens != target_len:
        die(f"usage.prompt_tokens {prompt_tokens} != the target {target_len} "
            f"(the prompt was cut to exactly {target_len} ids)")
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
              timings.get("cache_n", 0))
    if any(c != 0 for c in cached):
        die(f"cached tokens {cached} on a {target_len}-token prompt: "
            f"--no-prefix-cache is not in effect on this server")
    return {
        "nonce": nonce_text,
        "prompt_ms": float(prompt_ms),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": max(cached),
        "prompt_n": timings.get("prompt_n"),
        "preflight": [],
    }


def main():
    global BASE, HTTP_TIMEOUT, PREFLIGHT_PATH
    if len(sys.argv) != 2:
        die("usage: speed.py <label>")
    label = sys.argv[1]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        die(f"label {label!r} may only use [A-Za-z0-9._-] (it names the output file)")

    env = os.environ
    docs_path = env.get("RK_DOCS", "")
    if not docs_path:
        die('RK_DOCS is not set: the JSONL whose "prompt" fields build the prompts')
    try:
        lengths = [int(x) for x in env.get("RK_LENGTHS", "9216 16384 32768").split()]
    except ValueError:
        die("RK_LENGTHS must be integers")
    if not lengths or any(l <= 0 for l in lengths):
        die("RK_LENGTHS must be positive integers")
    reps = int(env.get("RK_REPS", "5"))
    if reps < 1:
        die("RK_REPS must be >= 1")
    port = env.get("RK_PORT", "8100")
    BASE = f"http://127.0.0.1:{port}"
    HTTP_TIMEOUT = float(env.get("RK_SPEED_HTTP_TIMEOUT", "3600"))
    stage = env.get("RK_STAGE", "scratch")
    nonce_fmt = env.get("RK_NONCE_TEXT", "Measurement nonce {n}.\n")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evidence = env.get("RK_EVIDENCE", os.path.join(repo, "evidence"))
    PREFLIGHT_PATH = os.path.join(repo, "scripts", "preflight.sh")
    if not os.access(PREFLIGHT_PATH, os.X_OK):
        die(f"preflight script missing or not executable: {PREFLIGHT_PATH}")

    # the server must already be up (scripts/serve.sh put it there)
    if not http_json("GET", BASE + "/health", parse=False):
        die(f"no server on {BASE} (scripts/serve.sh starts one)")

    docs = load_docs(docs_path)
    doc_ids = [tokenize(d["prompt"]) for d in docs]
    print(f"speed: server {BASE}, {len(docs)} docs from {docs_path}, "
          f"lengths {lengths}, {reps} reps each after one warm-up")

    # provenance: the whole configuration of the running engine
    server = http_json("GET", BASE + "/server_info", required=False)
    if server is not None:
        provenance = {"endpoint": "/server_info", "response": server}
    else:
        provenance = {"endpoint": "/v1/models", "response": http_json("GET", BASE + "/v1/models")}

    nonce_seq = 0
    lengths_detail = {}
    for length in lengths:
        tail, note = tail_note(length)
        entry = {"expected_exact_tail_tokens": tail, "tail_note": note}

        # one warm-up (a different nonce, counted like any other request), then the reps
        warm = preflight()
        nonce_text = nonce_fmt.format(n=nonce_seq)
        nonce_seq += 1
        entry["warmup"] = measure_once(length, nonce_text, tokenize(nonce_text), doc_ids)
        entry["warmup"]["preflight"] = warm

        rep_records = []
        for rep in range(1, reps + 1):
            pre_lines = preflight()
            nonce_text = nonce_fmt.format(n=nonce_seq)
            nonce_seq += 1
            record = measure_once(length, nonce_text, tokenize(nonce_text), doc_ids)
            record["rep"] = rep
            record["preflight"] = pre_lines
            rep_records.append(record)
            print(f"  {length:>6} tok  rep {rep}: {record['prompt_ms']:9.1f} ms  "
                  f"prompt_tokens {record['prompt_tokens']}  cached {record['cached_tokens']}")
        entry["reps"] = rep_records

        prompt_ms_values = [r["prompt_ms"] for r in rep_records]
        entry["median_prompt_ms"] = statistics.median(prompt_ms_values)
        entry["min_prompt_ms"] = min(prompt_ms_values)
        entry["max_prompt_ms"] = max(prompt_ms_values)
        lengths_detail[str(length)] = entry
        print(f"  {length:>6} tok  median {entry['median_prompt_ms']:9.1f} ms  "
              f"min {entry['min_prompt_ms']:9.1f}  max {entry['max_prompt_ms']:9.1f}  "
              f"(exact tail {tail})")

    out_dir = os.path.join(evidence, stage)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"speed-{label}.json")
    if os.path.exists(out_path):
        die(f"refusing to overwrite {out_path}")
    out = {
        "label": label,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "docs": os.path.abspath(docs_path),
        "lengths": lengths,
        "reps": reps,
        "port": port,
        "nonce_format": nonce_fmt,
        "timings_field": "timings.prompt_ms (GUIDE.md §5.9)",
        "server": provenance,
        "lengths_detail": lengths_detail,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"speed: wrote {out_path}")


if __name__ == "__main__":
    main()