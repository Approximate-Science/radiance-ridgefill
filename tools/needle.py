#!/usr/bin/env python3
"""needle.py -- needle-in-a-haystack retrieval test against a radiance server.

Perplexity on long docs cannot catch a retrieval failure: the score is a mean over
positions, the exact tail dominates it, and one needle lost in the approximated bulk
moves it by less than the run-to-run noise.  This harness makes the failure visible
directly: a fact ("The special magic number for <key> is <7 digits>.") is buried at a
known token depth in filler cut to an EXACT token count, the model is asked to return
the number, and the reply is either right or wrong.

Why the bulk/tail labelling: the RidgeFill prefill-acceleration plugin approximates most of a
long prompt but keeps the last T = 2048 tokens exact (README.md `RADIANCE_RIDGEFILL_TAIL`;
tools/speed.py tail_note).  A needle inside that exact tail can be retrieved from the
approximated run too; a needle in the bulk before it is where approximation loses
facts.  Every corpus item records which side its needle is on, and `compare` reports
accuracy separately for the two.

Subcommands (details in notes/needle.md):
  build    fill from the docs' "prompt" fields, insert needles at relative depths, cut
           to EXACT token counts with the server's /tokenize, write corpus.jsonl;
  run      send each item as token ids to /v1/completions (temperature 0), parse the
           first 7-digit number in the reply, write results.jsonl (resumable);
  compare  two results files: accuracy overall and by length / depth / bulk-vs-tail /
           variant, the paired disagreement table with an EXACT McNemar p-value, and
           the items where the two disagree.

Endpoint shapes (radiance core/server/admin.cpp:119-196, core/server/oai.cpp:1549-1568;
GUIDE.md §5.7/§5.9): POST /tokenize {"prompt", "add_special_tokens": false} ->
{"count", "tokens"}; POST /v1/completions {"prompt": [int, ...]} -> choices[0].text,
usage.prompt_tokens, timings.prompt_ms.  Python standard library only.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import urllib.error
import urllib.request

BASE = None            # "http://host:port", set by each subcommand
HTTP_TIMEOUT = 3600.0

TAIL_TOKENS = 2048     # the plugin's exact tail T (README.md RADIANCE_RIDGEFILL_TAIL)
N_DIGITS = 7           # every magic number is exactly 7 digits
WRAP = ([], [])        # --chat: the template's ids before and after a user message (chat_wrap)


# A fixed pool of common words for the keys: no external word list is available to a
# stdlib-only tool, and the keys only need to be ordinary, unlikely-in-filler words.
COMMON_WORDS = [
    "harbor", "lantern", "meadow", "compass", "ember", "river", "canyon", "orchard",
    "glacier", "willow", "beacon", "falcon", "cedar", "prairie", "coral", "summit",
    "harvest", "cinder", "atlas", "quartz", "juniper", "fjord", "tundra", "saffron",
    "marble", "delta", "lagoon", "monsoon", "copper", "granite", "basalt", "amber",
    "cobalt", "clover", "driftwood", "pumice", "obsidian", "thistle", "sundial",
    "vellum", "caravan", "pelican", "osprey", "badger", "lynx", "heron", "otter",
    "walnut", "almond", "barley", "pencil", "tunnel", "anchor", "basket", "boulder",
    "ribbon", "shovel", "ticket", "whistle", "blanket", "candle", "ladder", "pocket",
]


def die(msg):
    print(f"needle: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- HTTP + tokeniser

def http_json(method, url, body=None, required=True, parse=True):
    """One HTTP request; dies with the endpoint and status named on failure."""
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
        die(f"{method} {url} failed: {e.reason}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        die(f"{method} {url} answered something that is not JSON: {e}")


def tokenize(text):
    """Token ids for text, via the served model's /tokenize (add_special_tokens off,
    so no BOS is silently prepended and counts are exact)."""
    r = http_json("POST", BASE + "/tokenize",
                  {"prompt": text, "add_special_tokens": False})
    ids = r.get("tokens") if isinstance(r, dict) else None
    if not isinstance(ids, list) or not ids:
        die(f"/tokenize returned no tokens for: {text[:80]!r}")
    if r.get("count") != len(ids):
        die(f"/tokenize count {r.get('count')} != {len(ids)} tokens for {text[:80]!r}")
    return ids


def load_docs(path):
    """The filler docs; only the "prompt" field is used (same contract as speed.py)."""
    try:
        f = open(path, encoding="utf-8")
    except OSError as e:
        die(f"--docs unreadable: {e}")
    docs = []
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
        die(f"--docs holds no documents: {path}")
    return docs


def read_jsonl(path, what):
    """A JSONL file as a list of dicts; every line must be a JSON object."""
    try:
        f = open(path, encoding="utf-8")
    except OSError as e:
        die(f"cannot read the {what} {path}: {e}")
    rows = []
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError as e:
                die(f"{path}:{lineno}: not JSON: {e}")
            if not isinstance(j, dict):
                die(f"{path}:{lineno}: not a JSON object")
            rows.append(j)
    return rows


# ---------------------------------------------------------------- pure pieces

def parse_int_list(text, what):
    try:
        values = [int(x) for x in re.split(r"[,\s]+", text.strip()) if x]
    except ValueError:
        die(f"--{what} must be integers, comma or space separated: {text!r}")
    if not values or any(v <= 0 for v in values):
        die(f"--{what} must be positive integers")
    if len(set(values)) != len(values):
        die(f"--{what} holds duplicates")
    return values


def parse_float_list(text, what):
    try:
        values = [float(x) for x in re.split(r"[,\s]+", text.strip()) if x]
    except ValueError:
        die(f"--{what} must be numbers between 0 and 1: {text!r}")
    if not values:
        die(f"--{what} is empty")
    if any(not (0.0 <= v <= 1.0) for v in values):
        die(f"--{what} must be relative depths in [0, 1]")
    if len(set(values)) != len(values):
        die(f"--{what} holds duplicates")
    return values


def needle_sentence(key, number):
    return f"The special magic number for {key} is {number}."


def question_sentence(key):
    return f"What is the special magic number for {key}? Answer with the number only."


def filler_prefix(doc_ids, count):
    """The docs' token ids concatenated and repeated, cut to exactly `count` ids."""
    if count <= 0:
        return []
    ids = []
    while len(ids) < count:
        for d in doc_ids:
            ids.extend(d)
    return ids[:count]


def needle_position(length, depth, filler_total):
    """The token index the needle block starts at.

    The depth is relative to the WHOLE prompt (0 = first filler token, ~1 = against the
    question), then clamped into the filler region, which ends where the needles start
    pushing the question off the end.
    """
    pos = int(round(depth * (length - 1)))
    return max(0, min(pos, max(filler_total, 0)))


def region_at(position, length, tail=TAIL_TOKENS):
    """Which side of the exact/approximated boundary a token index is on."""
    return "tail" if position >= length - tail else "bulk"


def first_seven_digits(text):
    """The first run of exactly 7 digits in the reply, or None.

    Runs of 6 or 8+ digits do not match: an 8-digit number would contain a misleading
    7-digit substring, and the needles' numbers are always exactly 7 digits.
    """
    m = re.search(r"(?<!\d)(\d{%d})(?!\d)" % N_DIGITS, text)
    return m.group(1) if m else None


def random_words(rng, k):
    """k distinct keys from the pool, from the seeded RNG (deterministic per seed)."""
    if k > len(COMMON_WORDS):
        die(f"--keys {k} exceeds the {len(COMMON_WORDS)}-word key pool")
    keys, seen = [], set()
    while len(keys) < k:
        w = COMMON_WORDS[rng.randrange(len(COMMON_WORDS))]
        if w not in seen:
            keys.append(w)
            seen.add(w)
    return keys


def random_numbers(rng, k):
    """k distinct 7-digit numbers (1,000,000 .. 9,999,999)."""
    nums, seen = [], set()
    while len(nums) < k:
        n = rng.randrange(10 ** (N_DIGITS - 1), 10 ** N_DIGITS)
        if n not in seen:
            nums.append(n)
            seen.add(n)
    return nums


def mcnemar_exact_p(b, c):
    """Two-sided EXACT McNemar p-value from the disagreement counts.

    b = pairs A right/B wrong, c = pairs A wrong/B right; the null is that each
    disagreeing pair is equally likely to fall on either side, so
    p = min(1, 2 * P(X <= min(b, c))) with X ~ Binomial(b + c, 1/2).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail_p = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail_p)


# ---------------------------------------------------------------- build

def chat_wrap():
    """The model's chat template around ONE user message, generation prompt on, thinking off: the ids the server's
    own template renders before and after the message's content. A raw prompt makes an instruct model continue the
    document or open a reasoning block, and a 16-token reply then holds no number (2026-10-05: stock 0/28)."""
    probe = "needle probe content"
    r = http_json("POST", BASE + "/tokenize",
                  {"messages": [{"role": "user", "content": probe}], "add_generation_prompt": True,
                   "chat_template_kwargs": {"enable_thinking": False}})
    rendered, content = r.get("tokens") or [], tokenize(probe)
    for i in range(len(rendered) - len(content) + 1):
        if rendered[i:i + len(content)] == content:
            return rendered[:i], rendered[i + len(content):]
    die("the chat template's render does not contain the message's own ids: cannot wrap the prompts")


def build_item(rng, doc_ids, length, depth, variant, n_keys, seed):
    """One corpus item: filler + needle block at `depth` + the question, EXACTLY
    `length` tokens.  Pure given the pre-tokenised pieces' ids (tokenize() is the
    only server call in here)."""
    keys = random_words(rng, 1 if variant == "single" else n_keys)
    numbers = random_numbers(rng, len(keys))
    asked = 0 if variant == "single" else rng.randrange(len(keys))

    needle_ids = [tokenize("\n" + needle_sentence(k, n))
                  for k, n in zip(keys, numbers)]
    question_ids = tokenize("\n\n" + question_sentence(keys[asked]))
    block_len = sum(len(x) for x in needle_ids)
    head, tail = WRAP
    filler_total = length - block_len - len(question_ids) - len(head) - len(tail)
    if filler_total <= 0:
        die(f"length {length} has no room for filler: the needle block ({block_len} "
            f"tokens) and the question ({len(question_ids)}) alone exceed it")

    pos = needle_position(length, depth, filler_total)
    filler = filler_prefix(doc_ids, filler_total)

    starts, cursor = [], len(head) + pos
    for ids in needle_ids:
        starts.append(cursor)
        cursor += len(ids)

    prompt_ids = head + filler[:pos]
    for ids in needle_ids:
        prompt_ids.extend(ids)
    prompt_ids.extend(filler[pos:])
    prompt_ids.extend(question_ids)
    prompt_ids.extend(tail)
    if len(prompt_ids) != length:
        die(f"assembled {len(prompt_ids)} tokens, expected exactly {length} "
            f"(filler {filler_total} + needles {block_len} + "
            f"question {len(question_ids)})")

    needles = []
    for ids, (k, n), s in zip(needle_ids, zip(keys, numbers), starts):
        region = region_at(s, length)
        needles.append({"key": k, "number": str(n), "position": s,
                        "region": region, "in_tail": region == "tail"})

    asked_needle = needles[asked]
    return {
        "id": f"L{length}/D{depth:.2f}/{variant}",
        "variant": variant,
        "length": length,
        "depth": depth,
        "depths": [depth] * len(keys),
        "seed": seed,
        "exact_tail_tokens": TAIL_TOKENS,
        "keys": keys,
        "asked_index": asked,
        "key": keys[asked],
        "expected": str(numbers[asked]),
        "question": question_sentence(keys[asked]),
        "needles": needles,
        "needle_pos": starts[asked],
        "region": asked_needle["region"],
        "in_tail": asked_needle["in_tail"],
        "prompt_tokens": len(prompt_ids),
        "prompt_ids": prompt_ids,
    }


def cmd_build(a):
    global BASE, HTTP_TIMEOUT
    BASE, HTTP_TIMEOUT = a.server, float(a.timeout)
    lengths = parse_int_list(a.lengths, "lengths")
    depths = parse_float_list(a.depths, "depths")
    if a.keys < 2:
        die(f"--keys must be >= 2 (the multi-key variant needs more than one needle)")
    if os.path.exists(a.out):
        die(f"refusing to overwrite {a.out}")

    docs = load_docs(a.docs)
    print(f"needle build: server {BASE}, {len(docs)} docs from {a.docs}, "
          f"lengths {lengths}, depths {depths}, seed {a.seed}, "
          f"multi-key variant with {a.keys} needles")
    doc_ids = [tokenize(d["prompt"]) for d in docs]
    global WRAP
    WRAP = ([], [])
    if getattr(a, "chat", False):
        WRAP = chat_wrap()
        print(f"needle build: --chat: {len(WRAP[0])} template ids before the message, {len(WRAP[1])} after")
    total_doc_tokens = sum(len(x) for x in doc_ids)
    longest = max(lengths)
    if total_doc_tokens < longest:
        print(f"needle build: note: the docs hold {total_doc_tokens} tokens in total; "
              f"they are cycled to fill the {longest}-token prompts")

    rng = random.Random(a.seed)
    n_items = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for length in lengths:
            for depth in depths:
                for variant in ("single", "multi"):
                    item = build_item(rng, doc_ids, length, depth, variant,
                                      a.keys, a.seed)
                    out.write(json.dumps(item) + "\n")
                    n_items += 1
                    print(f"  {item['id']}: needle at token {item['needle_pos']} "
                          f"({item['region']}), expected {item['expected']}")
    print(f"needle build: wrote {n_items} items to {a.out}")


# ---------------------------------------------------------------- run

def cmd_run(a):
    global BASE, HTTP_TIMEOUT
    BASE, HTTP_TIMEOUT = a.server, float(a.timeout)
    items = read_jsonl(a.corpus, "corpus")
    if not items:
        die(f"the corpus {a.corpus} holds no items")

    seen_ids = set()
    for item in items:
        if "id" not in item:
            die(f'the corpus {a.corpus} holds a line without an "id"')
        if "prompt_ids" not in item or "expected" not in item:
            die(f'{item.get("id", "?")} has no "prompt_ids"/"expected": '
                f"not a needle corpus line?")
        if item["id"] in seen_ids:
            die(f'{item["id"]} appears twice in {a.corpus}')
        seen_ids.add(item["id"])

    done = set()
    if os.path.exists(a.out):
        for rec in read_jsonl(a.out, "results"):
            if "id" not in rec:
                die(f'{a.out} holds a line without an "id"')
            done.add(rec["id"])
    pending = [it for it in items if it["id"] not in done]
    print(f"needle run: server {BASE}, {len(items)} corpus items, "
          f"{len(done)} already in {a.out}, {len(pending)} to run, "
          f"max_tokens {a.max_tokens}, temperature 0")

    n_correct = 0
    with open(a.out, "a", encoding="utf-8") as out:
        for i, item in enumerate(pending, 1):
            body = {"model": "m", "prompt": item["prompt_ids"],
                    "max_tokens": a.max_tokens, "temperature": 0}
            r = http_json("POST", BASE + "/v1/completions", body)
            choices = r.get("choices")
            text = choices[0].get("text") if choices else None
            if not isinstance(text, str):
                die(f'{item["id"]}: response has no choices[0].text: '
                    f"{json.dumps(r)[:300]}")
            timings = r.get("timings") or {}
            usage = r.get("usage") or {}
            prompt_ms = timings.get("prompt_ms")
            if prompt_ms is None:
                die(f'{item["id"]}: response has no timings.prompt_ms: '
                    f"{json.dumps(r)[:300]}")
            prompt_tokens = usage.get("prompt_tokens")
            if prompt_tokens != item["prompt_tokens"]:
                die(f'{item["id"]}: usage.prompt_tokens {prompt_tokens} != the '
                    f'{item["prompt_tokens"]} ids sent (the prompt was cut to exactly '
                    f'that many tokens at build time)')

            parsed = first_seven_digits(text)
            correct = parsed == item["expected"]
            n_correct += correct
            rec = {
                "id": item["id"],
                "correct": correct,
                "parsed": parsed,
                "expected": item["expected"],
                "reply": text,
                "prompt_ms": float(prompt_ms),
                "prompt_tokens": prompt_tokens,
                # copied from the corpus so `compare` needs the results files only
                "length": item["length"],
                "depth": item["depth"],
                "region": item["region"],
                "variant": item["variant"],
                "key": item["key"],
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            status = "correct" if correct else "WRONG"
            print(f'  {i}/{len(pending)} {item["id"]}: {status}  parsed {parsed}  '
                  f'prompt_ms {rec["prompt_ms"]:.1f}')
    if pending:
        print(f"needle run: {n_correct}/{len(pending)} correct "
              f"({100.0 * n_correct / len(pending):.1f}%), appended to {a.out}")
    else:
        print(f"needle run: nothing to do (all {len(done)} ids are already in {a.out})")


# ---------------------------------------------------------------- compare

def _accuracy(rows):
    n = len(rows)
    k = sum(1 for r in rows if r["correct"])
    return f"{k}/{n} ({100.0 * k / n:.1f}%)" if n else "n/a"


def cmd_compare(a):
    a_rows = read_jsonl(a.a_file, "results A")
    b_rows = read_jsonl(a.b_file, "results B")
    if not a_rows or not b_rows:
        die("both results files must hold at least one line")

    A, B = {}, {}
    for rows, path, table in ((a_rows, a.a_file, A), (b_rows, a.b_file, B)):
        for rec in rows:
            for key in ("id", "correct"):
                if key not in rec:
                    die(f'{path} holds a line without "{key}": not a results file?')
            if rec["id"] in table:
                die(f'{rec["id"]} appears twice in {path}')
            table[rec["id"]] = rec

    only_a = set(A) - set(B)
    only_b = set(B) - set(A)
    if only_a or only_b:
        die(f"the results files do not cover the same items: "
            f"{len(only_a)} only in A (e.g. {sorted(only_a)[:3]}), "
            f"{len(only_b)} only in B (e.g. {sorted(only_b)[:3]})")

    order = [rec["id"] for rec in a_rows]        # keep A's file order
    n = len(order)

    def line(label, keyfn):
        groups = {}
        for rid in order:
            groups.setdefault(keyfn(A[rid]), []).append(rid)
        for key in sorted(groups, key=str):
            rids = groups[key]
            print(f"  {label} {key}: A {_accuracy([A[r] for r in rids])}   "
                  f"B {_accuracy([B[r] for r in rids])}   (n={len(rids)})")

    print(f"needle compare: A {a.a_file}  B {a.b_file}  ({n} paired items)")
    print(f"overall: A {_accuracy([A[r] for r in order])}   "
          f"B {_accuracy([B[r] for r in order])}")
    print("by length:")
    line("length", lambda r: r.get("length"))
    print("by depth bucket:")
    line("depth", lambda r: r.get("depth"))
    print("by region (bulk = inside the approximated prefix, tail = inside the "
          f"last {TAIL_TOKENS} exact tokens):")
    line("region", lambda r: r.get("region"))
    print("by variant:")
    line("variant", lambda r: r.get("variant"))

    both_right = sum(1 for r in order if A[r]["correct"] and B[r]["correct"])
    both_wrong = sum(1 for r in order if not A[r]["correct"] and not B[r]["correct"])
    b = sum(1 for r in order if A[r]["correct"] and not B[r]["correct"])
    c = sum(1 for r in order if not A[r]["correct"] and B[r]["correct"])
    p = mcnemar_exact_p(b, c)
    print("paired disagreement:")
    print(f"  both right: {both_right}   both wrong: {both_wrong}   "
          f"A right / B wrong: {b}   A wrong / B right: {c}")
    print(f"  exact McNemar p = {p:.6g}   (b + c = {b + c})")

    differing = [r for r in order if A[r]["correct"] != B[r]["correct"]]
    print(f"items where A and B disagree ({len(differing)}):")
    for rid in differing:
        winner, loser = (A[rid], B[rid]) if A[rid]["correct"] else (B[rid], A[rid])
        side = "A" if A[rid]["correct"] else "B"
        reply = loser.get("reply", "")
        snippet = reply if len(reply) <= 48 else reply[:45] + "..."
        print(f'  {rid}  {side} right ({winner.get("expected")}); other side '
              f'parsed {loser.get("parsed")!r}, reply {snippet!r}')


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="needle.py",
        description="Needle-in-a-haystack retrieval test against a radiance "
                    "OpenAI-compatible server (see notes/needle.md).")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a corpus of exact-length prompts with "
                                     "needles at known depths")
    b.add_argument("--docs", required=True,
                   help="JSONL with a 'prompt' field per line (the filler)")
    b.add_argument("--lengths", default="8192,16384,32768",
                   help="exact prompt token counts, comma/space separated")
    b.add_argument("--depths", default="0.05,0.25,0.5,0.75,0.9,0.98",
                   help="relative needle depths in [0,1], comma/space separated")
    b.add_argument("--keys", type=int, default=4,
                   help="needles in the multi-key variant (the question asks one)")
    b.add_argument("--chat", action="store_true",
                   help="wrap each prompt in the server's chat template (one user turn, thinking off)")
    b.add_argument("--seed", type=int, default=0,
                   help="seed for keys, numbers and the asked needle")
    b.add_argument("--server", default="http://127.0.0.1:8100")
    b.add_argument("--out", required=True, help="corpus JSONL to write")
    b.add_argument("--timeout", type=float, default=HTTP_TIMEOUT)
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run", help="ask every corpus item, resumable")
    r.add_argument("--corpus", required=True)
    r.add_argument("--server", default="http://127.0.0.1:8100")
    r.add_argument("--out", required=True, help="results JSONL (appended, resumable)")
    r.add_argument("--max-tokens", type=int, default=16)
    r.add_argument("--timeout", type=float, default=HTTP_TIMEOUT)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="two results files: accuracy, disagreement "
                                       "table, exact McNemar")
    c.add_argument("a_file")
    c.add_argument("b_file")
    c.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()