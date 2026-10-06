#!/usr/bin/env python3
"""branch_corpus.py -- build the Stage C (#2, prefix cache on) branch-hazard corpus.

  branch_corpus.py build --docs ppl.jsonl --n 20 --seed 0 --interval 2048 --tail 2048 \\
                         --server http://127.0.0.1:8100 --out branches.jsonl [--min-prompt 16384]

Each output line is one TRIPLE of chat conversations (messages lists) over one long document,
the shape R65 asks for: (doc, long question, short question).

  A  the producer: system + user(document + long question) + assistant(placeholder answer)
     + user(follow-up).  Its rendered prompt is N1 >= --min-prompt tokens (default 16K); the
     prefill writes prefix-cache checkpoints at every multiple of --interval.
  B  the branch: the SAME document cut at a character position inside it + a SHORT question,
     so B shares A's token prefix up to inside the document and ends sooner (N2 < N1).  The
     cut is searched so that  N2 - T < P <= shared prefix,  where P is the largest checkpoint
     multiple <= the shared prefix (hits resume only at checkpoint multiples,
     core/mem/prefix.cpp:290-302) and T = --tail: the HAZARD case of PLAN-FIX §5.2 -- B's
     exact tail [N2-T, N2) reaches cached positions the producer approximated.  The predicted
     number of B's tail positions that come from cache is  max(0, P - (N2 - T)).  The
     construction also keeps P <= N1 - T, so that number equals the DD-A device-counter oracle
     min(N1-N2, T-(N2-P)) every overlapped position really was approximated by the producer.
  C  the append-only continuation (the safe case): A + assistant(long placeholder answer) +
     user(new question), N3 > N1.  The placeholder answer is long enough that C's own computed
     prompt covers its whole exact tail, so no tail position comes from cache:
     max(0, P3 - (N3 - T)) == 0, and hazard_rate.py must not flag it (R64/R65/R68).

All lengths are exact, measured with the served model's own tokenizer: POST /tokenize takes
`messages`, renders the chat template and answers the count chat serves (GUIDE.md §5.6,
core/server/admin.cpp:119-196); the `tokens` array it returns is what the shared prefixes are
measured in.  --interval is the number on the engine's startup line
`chunk geometry: ... checkpoint interval 2048 ...` (core/sched/geometry.cpp:156).
Deterministic per --seed.  Python standard library only.  Writes <out> and <out>.manifest.json.
"""

import argparse
import hashlib
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

HTTP_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# The conversation pieces.  Every text here is a placeholder: the measurement reads only the
# token lengths and the cache counters, never the model's answers.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ("You are a careful assistant. Answer strictly and only from the document "
                 "the user provides.")

LONG_QUESTIONS = [
    "Read the document above carefully and answer in your own words: what are the main "
    "subjects it describes, what does it report about each one's habits, range and diet, and "
    "which claims should a reader treat with caution? Structure the answer under three "
    "headings and quote the passages you rely on.",
    "Summarise the document above for a reader who has never seen it: what kind of text is "
    "it, what does it establish, and what does it leave undecided? Give one paragraph per "
    "point and cite the supporting passage for each.",
    "Using only the document above, list every distinct topic it touches, say where in the "
    "text each topic appears, and explain how the topics relate to one another. Finish with "
    "the single sentence you would keep if only one could survive.",
    "What question does the document above answer, what evidence does it give, and where is "
    "that evidence weakest? Answer in three short sections and point at the exact passages.",
    "Extract from the document above every factual claim it makes, rank the ten most "
    "load-bearing ones, and justify the ranking with quotations from the text.",
    "Rewrite the document above as a briefing note: the situation, what is known, what is "
    "unknown, and the three questions a careful reader should ask next. Quote the text where "
    "it matters.",
]

FOLLOWUPS = [
    "Now answer the same question again in exactly three bullet points, and name the passage "
    "that supports each bullet.",
    "Give the passage a title of at most six words and one sentence on why that title fits.",
    "Which single sentence of the document carries the most information? Quote it and say why.",
    "Answer the question again, but this time for each claim say whether the document itself "
    "vouches for it or reports it from elsewhere.",
]

ANSWER_PLACEHOLDERS = [
    "Placeholder answer recorded by the Stage C branch corpus; the measurement reads only "
    "prompt lengths and cache counters, so this text stands in for the model's first answer.",
    "A stand-in first answer for the Stage C branch corpus. Nothing here is read for content; "
    "only the rendered length of this turn matters to the measurement.",
    "This is the corpus's placeholder first answer. Its wording is irrelevant; what matters is "
    "that the conversation continues past the document with a second question.",
    "Stage C corpus placeholder. The first answer would be generated here; the corpus replaces "
    "it with fixed text so the prompts are reproducible.",
]

SHORT_QUESTIONS = [
    "In one sentence, summarise the passage above.",
    "List the three most important facts in the passage above.",
    "Give the passage a title of at most six words.",
    "Name the main subject of the passage above.",
    "Quote the first sentence of the passage above.",
    "What kind of text is the passage above?",
]

NEW_QUESTIONS = [
    "Finally, name the single most surprising claim in the document and justify your choice in "
    "two sentences.",
    "One last question: which part of the document would you re-read before trusting any "
    "summary of it, and why?",
    "To close: state the one thing the document proved to you and the one thing it left open.",
    "Last question: if you could add one paragraph to the document, what would it cover and "
    "where would it go?",
]

# C's placeholder answer is this paragraph repeated enough times that the turn-2 request's own
# computed prompt is longer than the exact tail (see the docstring).
ANSWER_PARAGRAPH = ("Placeholder continuation answer for the Stage C branch corpus. It exists to "
                    "make this append-only turn long enough that the request's own computed "
                    "prompt covers its whole exact tail, so no tail position is served from the "
                    "prefix cache. ")

# The checkpoint index (P = k * interval) each triple aims its branch at, in order; clamped to
# what the document and the producer can support (see build_triple).
K_SEQUENCE = (2, 5, 3, 7, 1, 4, 6, 8)

HEADER_TOKENS = 32      # allowance for the system turn + user-turn header before the document
MAX_ADJUST_STEPS = 128  # fine-adjust moves after the binary search, each one /tokenize call


def die(msg):
    print(f"branch_corpus: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# The server: /tokenize only (GUIDE.md §5.6; core/server/admin.cpp:119-196).
# ---------------------------------------------------------------------------

def http_json(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            payload = resp.read().decode()
    except urllib.error.HTTPError as e:
        die(f"POST {url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as e:
        die(f"POST {url} failed: {e.reason}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        die(f"POST {url} answered something that is not JSON: {e}")


def _ids_of(r, what):
    ids = r.get("tokens") if isinstance(r, dict) else None
    if not isinstance(ids, list) or not ids:
        die(f"/tokenize returned no tokens for: {what[:80]!r}")
    if r.get("count") != len(ids):
        die(f"/tokenize count {r.get('count')} != {len(ids)} tokens for: {what[:80]!r}")
    return ids


def tokenize_chat(server, messages):
    """Exact rendered length (and ids) of the chat prompt the server would serve for `messages`.

    /tokenize takes `messages`, renders the chat template with the generation prompt (its
    add_generation_prompt default is true -- exactly what a chat request gets) and tokenises
    what it produced, so `count` is the prompt chat serves.
    """
    return _ids_of(http_json(server + "/tokenize",
                             {"messages": messages, "add_generation_prompt": True}),
                   json.dumps(messages)[:120])


def tokenize_text(server, text):
    """Raw-text token ids, no BOS added (the same call speed.py makes)."""
    return _ids_of(http_json(server + "/tokenize",
                             {"prompt": text, "add_special_tokens": False}), text)


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def largest_checkpoint(n, interval):
    """Hits resume only at checkpoint multiples (prefix.cpp:290-302): the resume point of a
    request whose longest matching prefix is n tokens is the largest multiple <= n."""
    return (n // interval) * interval


# ---------------------------------------------------------------------------
# The conversations.
# ---------------------------------------------------------------------------

def build_producer(doc, question, answer, followup):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": doc + "\n\nQuestion: " + question},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": followup},
    ]


def build_branch(doc, cut, question):
    """The branch: the document cut at `cut` characters, then a SHORT question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": doc[:cut] + "\n\nQuestion: " + question},
    ]


def build_continuation(producer, long_answer, question):
    """A + assistant(answer) + user(question): the append-only continuation."""
    return producer + [
        {"role": "assistant", "content": long_answer},
        {"role": "user", "content": question},
    ]


def long_answer_text(target_chars):
    """ANSWER_PARAGRAPH repeated to >= target_chars (the caller retries with more if needed)."""
    reps = max(1, -(-target_chars // len(ANSWER_PARAGRAPH)))
    return ANSWER_PARAGRAPH * reps


def load_docs(path):
    docs = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        die(f"--docs unreadable: {e}")
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError as e:
            die(f"{path}:{lineno}: not JSON: {e}")
        if "prompt" not in j:
            die(f'{path}:{lineno}: no "prompt" field')
        source = j.get("id") or j.get("source") or f"doc/{len(docs)}"
        docs.append((source, j["prompt"]))
    if not docs:
        die(f"--docs holds no documents: {path}")
    return docs


# ---------------------------------------------------------------------------
# One triple.
# ---------------------------------------------------------------------------

def build_triple(server, doc, source, index, interval, tail, min_prompt, rng, doc_tokens):
    """Build and measure one (A, B, C) triple; returns the record, or (None, reason).

    The rng draws are taken first, in a fixed order, whether or not the triple is skipped, so
    the output stays a pure function of (docs, n, seed, interval, tail, server tokenizer).
    """
    question = rng.choice(LONG_QUESTIONS)
    answer = rng.choice(ANSWER_PLACEHOLDERS)
    followup = rng.choice(FOLLOWUPS)
    short_question = rng.choice(SHORT_QUESTIONS)
    new_question = rng.choice(NEW_QUESTIONS)

    # --- A: the producer -------------------------------------------------
    A = build_producer(doc, question, answer, followup)
    ids_A = tokenize_chat(server, A)
    n1 = len(ids_A)
    if n1 < min_prompt:
        return None, (f"producer prompt {n1} < --min-prompt {min_prompt}")

    # How far past checkpoint k the shared prefix is aimed (the "slack").  Two feasibility
    # caps on the checkpoint index k, and a jitter so the predicted overlap T - slack - suffix
    # varies across the corpus instead of repeating:
    #   the cut must stay inside the document (shared prefix <= header + doc tokens), and
    #   P = k*interval must be <= N1 - tail, so every overlapped cached position lies in the
    #   range the producer approximated (p < N1 - T) and the predicted superset count equals
    #   the DD-A device-counter oracle min(N1-N2, T-(N2-P)).
    slack = max(2, interval // 8 + rng.randint(0, interval // 4))
    chars_per_token = (len(doc) / doc_tokens) if doc_tokens else 1.0
    max_k = min((doc_tokens + HEADER_TOKENS - 2 * slack) // interval,
                (n1 - tail) // interval)
    if max_k < 1:
        return None, (f"no checkpoint fits: doc {doc_tokens} tokens, producer {n1}, "
                       f"interval {interval}, tail {tail}")
    k = min(K_SEQUENCE[index % len(K_SEQUENCE)], max_k)
    target_shared = k * interval + slack   # land the shared prefix just past checkpoint k

    # --- B: the branch ---------------------------------------------------
    # Binary-search the character cut for "shared prefix >= target", then fine-adjust so that
    # P lands on checkpoint k and B stays short enough (N2 - T < P).  shared(cut) is not
    # perfectly monotone at BPE boundaries; the adjust loop absorbs that.
    def measure(cut):
        B = build_branch(doc, cut, short_question)
        ids_B = tokenize_chat(server, B)
        return B, ids_B, len(ids_B), common_prefix_len(ids_A, ids_B)

    lo, hi = 0, len(doc)
    while lo < hi:
        mid = (lo + hi) // 2
        if measure(mid)[3] >= target_shared:
            hi = mid
        else:
            lo = mid + 1
    cut = lo

    step = max(1, int(chars_per_token))
    B = ids_B = None
    for _ in range(MAX_ADJUST_STEPS):
        B, ids_B, n2, shared = measure(cut)
        p = largest_checkpoint(shared, interval)
        if shared < k * interval:
            cut += step
        elif shared >= (k + 1) * interval or n2 - tail >= p:
            cut -= step
        else:
            break
        if not 0 < cut < len(doc):
            die(f"branch search left the document (cut {cut}, doc {len(doc)} chars) for "
                f"{source}: shorten the question pools or lower the checkpoint index")
    B, ids_B, n2, shared = measure(cut)
    p = largest_checkpoint(shared, interval)

    predicted = max(0, p - (n2 - tail))                 # tail positions served from cache
    bound = min(n1 - n2, tail - (n2 - p))               # DD-A: what the device counter should say
    checks = {
        "n1_ge_min_prompt": n1 >= min_prompt,
        "n2_lt_n1": n2 < n1,
        "branch_inside_document": 0 < cut < len(doc),
        "n2_minus_tail_lt_p": n2 - tail < p,
        "p_le_shared_prefix": p <= shared,
        "p_pos_multiple_of_interval": p > 0 and p % interval == 0,
        "p_le_n1_minus_tail": p <= n1 - tail,
        "predicted_positive": predicted > 0,
        "bound_equals_prediction": bound == predicted,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        die(f"the B construction failed for {source}: {', '.join(bad)} "
            f"(n1={n1} n2={n2} shared={shared} p={p} cut={cut} k={k})")

    # --- C: the append-only continuation (the safe case) ------------------
    # The placeholder answer is sized so C's own computed prompt (N3 - P3) covers the whole
    # exact tail: predicted max(0, P3 - (N3 - T)) == 0.  A tail-and-a-quarter of placeholder
    # text leaves margin for any P3; the retry doubles it if a strange template disagrees.
    answer_chars = int(tail * 5 / 4 * chars_per_token) + len(ANSWER_PARAGRAPH)
    C = ids_C = None
    for _ in range(4):
        C = build_continuation(A, long_answer_text(answer_chars), new_question)
        ids_C = tokenize_chat(server, C)
        n3 = len(ids_C)
        shared_c = common_prefix_len(ids_A, ids_C)
        if shared_c > n1:
            die(f"C shares more than all of A for {source}: shared {shared_c} > n1 {n1}")
        p3 = largest_checkpoint(shared_c, interval)
        c_pred = max(0, p3 - (n3 - tail))
        if c_pred == 0:
            break
        answer_chars *= 2
    else:
        die(f"the C construction failed for {source}: predicted {c_pred} != 0 "
            f"(n3={n3} p3={p3} tail={tail})")

    record = {
        "id": f"branch/{index:03d}",
        "source": source,
        "interval": interval,
        "tail": tail,
        "n1": n1, "n2": n2, "n3": n3,
        "A": {"messages": A, "n": n1},
        "B": {"messages": B, "n": n2, "shared_prefix": shared, "p": p, "k": k,
              "cut_chars": cut, "tail_start": n2 - tail,
              "predicted_tail_from_cache": predicted, "hazard_bound": bound},
        "C": {"messages": C, "n": n3, "shared_prefix": shared_c, "p": p3,
              "predicted_tail_from_cache": c_pred},
        "checks": dict(checks, n3_gt_n1=n3 > n1, c_predicted_zero=c_pred == 0),
    }
    if not all(record["checks"].values()):
        die(f"internal check failed for {source}: {json.dumps(record['checks'])}")
    return record, None


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="branch_corpus.py",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the branch-hazard corpus")
    b.add_argument("--docs", required=True,
                   help="ppl.jsonl: one {id, prompt, ...} per line (only prompt is used)")
    b.add_argument("--n", type=int, default=20, help="number of triples (docs are cycled)")
    b.add_argument("--seed", type=int, default=0, help="conversation-text seed; the output "
                   "is a pure function of it")
    b.add_argument("--interval", type=int, default=2048,
                   help="the checkpoint interval: read it off the engine's startup line "
                        '"chunk geometry: ... checkpoint interval N ..."')
    b.add_argument("--tail", type=int, default=2048, help="T, the exact tail (ridgefill.tail)")
    b.add_argument("--min-prompt", type=int, default=16384,
                   help="the producer A must render to at least this many tokens")
    b.add_argument("--server", required=True, help="server base URL, e.g. http://127.0.0.1:8100")
    b.add_argument("--out", required=True, help="output JSONL (a .manifest.json is written "
                   "beside it)")
    args = ap.parse_args(argv)

    if args.n < 1:
        die("--n must be >= 1")
    if args.interval < 1 or args.tail < 1:
        die("--interval and --tail must be positive")
    if args.min_prompt < 1:
        die("--min-prompt must be positive")
    server = args.server.rstrip("/")
    if not server.startswith("http"):
        die(f"--server must be an http URL, got {args.server!r}")

    docs = load_docs(args.docs)
    doc_tokens = [len(tokenize_text(server, doc)) for _, doc in docs]
    if all(t == 0 for t in doc_tokens):
        die("every document tokenises to zero tokens")

    rng = random.Random(args.seed)
    records, skipped = [], []
    for i in range(args.n):
        source, doc = docs[i % len(docs)]
        record, why = build_triple(server, doc, source, i, args.interval, args.tail,
                                   args.min_prompt, rng, doc_tokens[i % len(docs)])
        if record is None:
            skipped.append((f"{i:03d}", source, why))
            print(f"branch/{i:03d}  {source:12s}  skipped: {why}", flush=True)
            continue
        records.append(record)
        B, C = record["B"], record["C"]
        print(f"{record['id']}  {source:12s}  N1 {record['n1']:6d}  N2 {record['n2']:6d}  "
              f"shared {B['shared_prefix']:6d}  P {B['p']:6d}  tail-from-cache "
              f"{B['predicted_tail_from_cache']:5d}  | C N3 {record['n3']:6d}  P3 {C['p']:6d}  "
              f"from-cache {C['predicted_tail_from_cache']}", flush=True)

    if not records:
        die("no triple was built: every document was skipped")
    if skipped:
        print(f"skipped {len(skipped)} of {args.n} triples (doc too short); "
              f"{len(records)} written")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    manifest = {
        "docs": str(Path(args.docs).resolve()),
        "docs_sha256": hashlib.sha256(Path(args.docs).read_bytes()).hexdigest(),
        "n": args.n, "seed": args.seed,
        "interval": args.interval, "tail": args.tail, "min_prompt": args.min_prompt,
        "server": server,
        "skipped": [{"triple": t, "source": s, "why": w} for t, s, w in skipped],
        "triples": [{"id": r["id"], "source": r["source"],
                     "n1": r["n1"], "n2": r["n2"], "n3": r["n3"],
                     "shared_prefix": r["B"]["shared_prefix"], "p": r["B"]["p"],
                     "predicted_tail_from_cache": r["B"]["predicted_tail_from_cache"],
                     "hazard_bound": r["B"]["hazard_bound"],
                     "c_shared_prefix": r["C"]["shared_prefix"], "c_p": r["C"]["p"],
                     "c_predicted_tail_from_cache": r["C"]["predicted_tail_from_cache"]}
                    for r in records],
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                 encoding="utf-8")
    print(f"wrote {out} ({len(records)} triples) and {out}.manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())