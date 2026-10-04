#!/usr/bin/env python3
"""send_doc.py -- send ONE prefill-only request built from a document to the server on RK_PORT.

For the debug gates that need exactly one known prompt (R16/R36 --profile-ops on a 16,384-token
prompt, R17 the boundary dump on a held capture document), not for timing (tools/speed.py is the
timing instrument).

  scripts/send_doc.py --jsonl FILE (--name NAME | --index I) [--field prompt] [--length L] [--out F]

The document's text (the line whose "name"/"id" is NAME, or line I) is tokenised by the SERVED model's
tokenizer (POST /tokenize, add_special_tokens false, so no BOS is prepended), cut to L tokens when
--length is given (refused when the document is shorter), and sent as token ids to /v1/completions
with max_tokens 1, temperature 0. usage.prompt_tokens must equal the id count. Prints one JSON line:
the id count, the sha256 of the ids, usage and timings. --out also writes the ids (JSON list), so the
caller can check them against a capture's input_ids. Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request


def post(base, path, body, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def pick(lines, name, index, field):
    docs = [json.loads(x) for x in lines if x.strip()]
    if name is None:
        return docs[index][field]
    for d in docs:
        if d.get("name") == name or d.get("id") == name:
            return d[field]
    sys.exit(f"send_doc: no line with name/id {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--name")
    who.add_argument("--index", type=int)
    ap.add_argument("--field", default="prompt")
    ap.add_argument("--length", type=int)
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=float, default=3600.0)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{os.environ.get('RK_PORT', '8100')}"
    with open(a.jsonl, encoding="utf-8") as f:
        text = pick(f.read().splitlines(), a.name, a.index, a.field)
    ids = post(base, "/tokenize", {"prompt": text, "add_special_tokens": False}, a.timeout)["tokens"]
    if a.length is not None:
        if len(ids) < a.length:
            sys.exit(f"send_doc: the document has {len(ids)} tokens, fewer than --length {a.length}")
        ids = ids[:a.length]
    if a.out:
        with open(a.out, "w") as f:
            json.dump(ids, f)
    r = post(base, "/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0}, a.timeout)
    got = r.get("usage", {}).get("prompt_tokens")
    if got != len(ids):
        sys.exit(f"send_doc: usage.prompt_tokens {got} != {len(ids)} ids sent")
    print(json.dumps({"n_ids": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      "usage": r.get("usage"), "timings": r.get("timings")}))


if __name__ == "__main__":
    main()
