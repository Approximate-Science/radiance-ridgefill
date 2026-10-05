#!/usr/bin/env python3
"""media_ident.py -- one image-then-text chat request against the server on --port, greedy, hashed: R76's instrument.

  media_ident.py --image FILE --docs JSONL [--index I] [--chars N] [--max-tokens 64] --out OUT.json

WHAT IT SENDS: /v1/chat/completions with ONE user message whose content is the image (a base64 data: URL -- the
engine fetches nothing) followed by the text of document I of the corpus cut to N characters and a closing
instruction, temperature 0, a fixed seed, thinking off, max_tokens tokens. The image comes first so every chunk after
the one holding its rows is text whose rotary position runs behind its index (abi/rad_runtime.h, multimodal): the
chunks the plugin approximates after an image.

WHAT IT RECORDS: usage (prompt_tokens must reach --min-prompt, default 10,000, or the run fails), timings, the
completion text and its sha256, and per-token logprobs with the top 5 alternatives when this build's sampler returns
them (the request is retried without them when the server refuses logprobs; `logprobs` is then null). Two servers of
the same flags are compared by the sha256 and, when present, the logprobs, byte for byte.

Standard library only.
"""
import argparse
import base64
import hashlib
import json
import mimetypes
import sys
import urllib.error
import urllib.request


def post(base, body, timeout):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--field", default="prompt")
    ap.add_argument("--chars", type=int, default=48000)
    ap.add_argument("--min-prompt", type=int, default=10000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--port", default="8100")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    docs = [json.loads(x) for x in open(a.docs, encoding="utf-8") if x.strip()]
    text = docs[a.index][a.field]
    if len(text) < a.chars:
        sys.exit(f"media_ident: document {a.index} has {len(text)} characters, fewer than --chars {a.chars}")
    raw = open(a.image, "rb").read()
    mime = mimetypes.guess_type(a.image)[0] or "application/octet-stream"
    url = f"data:{mime};base64," + base64.b64encode(raw).decode()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text[:a.chars] + "\n\nDescribe the image, then summarise the text above in two sentences."},
        ]}],
        "max_tokens": a.max_tokens, "temperature": 0, "seed": 12345,
        "chat_template_kwargs": {"enable_thinking": False},
        "logprobs": True, "top_logprobs": 5,
    }
    base = f"http://127.0.0.1:{a.port}"
    try:
        r = post(base, body, a.timeout)
    except urllib.error.HTTPError as e:
        if e.code not in (400, 501) or b"logprobs" not in e.read():
            raise
        body.pop("logprobs")
        body.pop("top_logprobs")
        r = post(base, body, a.timeout)
    ch = r["choices"][0]
    content = ch["message"]["content"] or ""
    rec = {
        "image": a.image, "image_sha256": hashlib.sha256(raw).hexdigest(), "docs": a.docs, "index": a.index,
        "chars": a.chars, "usage": r.get("usage"), "timings": r.get("timings"),
        "finish_reason": ch.get("finish_reason"), "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "logprobs": ch.get("logprobs"),
    }
    open(a.out, "w").write(json.dumps(rec, indent=1))
    pt = (r.get("usage") or {}).get("prompt_tokens", 0)
    print(json.dumps({"prompt_tokens": pt, "sha256": rec["sha256"][:16], "logprobs": rec["logprobs"] is not None,
                      "prompt_ms": (r.get("timings") or {}).get("prompt_ms")}))
    if pt < a.min_prompt:
        sys.exit(f"media_ident: prompt_tokens {pt} < --min-prompt {a.min_prompt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
