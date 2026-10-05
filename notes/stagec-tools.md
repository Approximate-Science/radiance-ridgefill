# Notes — Stage C tools (`tools/branch_corpus.py`, `tools/hazard_rate.py`)

Written by the Stage C tools worker, 2026-10-05. Engine facts below were read from the
radiance tree at tag v1.0.8, commit `140987f` (read-only); nothing here needs the plugin or a
GPU to run — `branch_corpus.py` talks only to `/tokenize`, `hazard_rate.py` reads files. Both
are Python standard library only. Tests: `tests/test_branch_corpus.py`,
`tests/test_hazard_rate.py` (pytest, a fake `/tokenize` server in a thread).

## What Stage C measures and why these two tools

PLAN-FIX §5 (DD-A) fixes the #2 cache invariant: a request that resumes from a cached
checkpoint P keeps an exact tail `[N2−T, N2)`, but when it is a **branch that ends sooner than
the producer** (`N2 − T < P ≤ shared prefix`), the oldest positions of that tail were
projected late-K/V by the producer — the hazard, bounded by `min(N1−N2, T−(N2−P))`. R65
measures it with 20 (doc, long question, short question) triples and the device hazard
counter (PLAN-FIX §5.4); R68 cross-checks the counter against a log-side instrument over
response `timings`. These tools are those two instruments; the **engine-side** counter and its
`kva_hazard_read` plumbing are another worker's, and this file does not design them.

Engine facts these tools rely on (all verified in the tree):

- Hits resume only at checkpoint multiples: `core/mem/prefix.cpp:290-302`
  (`for (p = (n_attn_tokens / interval_) * interval_; ...)`) — so a request whose longest
  matching prefix is `shared` resumes at `P = (shared / interval) * interval` and reports
  `timings.cache_n = P` (GUIDE.md §5.9: `prompt_n + cache_n` is the prompt).
- The checkpoint interval is on the startup line `chunk geometry: quantum … , checkpoint
  interval N, max_tok …` (`core/sched/geometry.cpp:156`) — read N off it and pass it as
  `--interval`.
- `POST /tokenize` takes `messages`, renders the chat template (generation prompt included,
  its default) and returns `{"count", "tokens"}` — "the count it gives is the prompt chat
  serves" (GUIDE.md §5.6, `core/server/admin.cpp:119-196`). All lengths below are therefore
  exact, not estimated.

## `tools/branch_corpus.py`

```
python3 tools/branch_corpus.py build \
    --docs /var/home/dylan/AI-Work/kva-flashnext-tests-data/samples/quick/ppl.jsonl \
    --n 20 --seed 0 --interval 2048 --tail 2048 \
    --server http://127.0.0.1:8100 --out evidence/stagec/branches.jsonl
```

Builds `--n` triples (documents are cycled; docs whose producer renders below
`--min-prompt` 16384 tokens are skipped with a note — at 2048 geometry the 16k/32k quick
buckets qualify, the 8k bucket does not). One JSONL line per triple:

| field | meaning |
|---|---|
| `A` | the producer: system + user(document + long question) + assistant(placeholder) + user(follow-up); `A.n = N1 ≥ 16384` |
| `B` | the branch: the same document cut inside it + a SHORT question, `N2 < N1`; `B.shared_prefix` = the exact common token prefix with A, `B.p = P` the largest checkpoint multiple ≤ it, `B.predicted_tail_from_cache = max(0, P − (N2 − T))` |
| `B.hazard_bound` | `min(N1−N2, T−(N2−P))` — the DD-A oracle the device counter is checked against (R65); the construction keeps `P ≤ N1 − T`, so it equals the predicted count and every overlapped position really was approximated by the producer |
| `C` | the append-only continuation: A + assistant(long placeholder answer) + user(new question), `N3 > N1`, with `C.predicted_tail_from_cache = max(0, P3 − (N3 − T)) = 0` |
| `n1 n2 n3 interval tail checks` | flat mirrors and the per-triple assertion results (all must be true or the tool dies naming the triple) |

The cut is found by binary search over character positions plus a fine-adjust loop
(`/tokenize` each step), because the shared token prefix is not perfectly monotone at BPE
boundaries; the loop lands `shared ∈ [k·I, (k+1)·I)` with `N2 − T < P`, checkpoint index k
varying by triple (`K_SEQUENCE`) and slack jittered per seed so the predicted overlap varies
(T − slack − question overhead ≈ 1200–1750 at T = 2048). C's placeholder answer is sized at
a tail and a quarter so C's **own computed prompt covers its whole exact tail**
(`N3 − P3 ≥ T`): no tail position comes from cache and `hazard_rate.py` must not flag it.
Output is a pure function of (docs, n, seed, interval, tail, server tokenizer); a
`.manifest.json` with the lengths and sha256s lands beside the corpus.

**Serving recipe** (the driver is not part of this change): per triple, chat requests
`temperature 0, max_tokens 1`, in order A → B → C, recording per request the id
(`<triple>/A|B|C`), `usage.prompt_tokens` and `timings.cache_n`/`prompt_n` into a JSONL for
`hazard_rate.py`. Serve on the cache profile (`fnserve.sh` cache flags, `--no-prefix-cache`
absent, `--checkpoint-slots` large enough that retention never runs — retention only evicts
on slot exhaustion, `prefix.cpp:683-694`, and an evicted checkpoint would move P). Triples
over the same document share that document's prefix, so either serve one document's triples
on one boot or call `/reset_prefix_cache` (`core/server/admin.cpp`) between triples: the
hazard arithmetic holds either way, but the clean reading is one producer chain per triple.

## `tools/hazard_rate.py`

```
python3 tools/hazard_rate.py --records responses.jsonl --plugin-log server.log \
    --tail 2048 [--out report.json] [--require-match]
```

Reads response records (JSONL; `{"id", "prompt_len", "cache_n"}`, `prompt_tokens`, or whole
chat/completion responses with `usage`/`timings` — `prompt_len` falls back to
`cache_n + prompt_n`), flags each request whose exact tail reached cached positions —

```
flagged:  cache_n > prompt_len − T          (PLAN-FIX §5.4's superset condition;
overlap:  max(0, cache_n − max(0, prompt_len − T))   max(0,·) guards a prompt shorter than T)
```

— reports the per-request table, the overlapped-position count and the overall rate. This is
the **superset** of the true hazard (an append-only turn can overlap the cache in its tail
yet be exact), which is exactly why C is built unflaggable.

`--plugin-log` parses the line the plugin prints once per request that resumed from a cached
checkpoint, carrying the device hazard counter. **The contract is the named constant
`HAZARD_LOG_RE` in the tool** — the engine-side worker implements the emitter to match it:

```python
HAZARD_LOG_RE = re.compile(
    r"\bkva:\s+hazard\b(?:\s+request\s+(?P<request>\S+))?\s+(?P<positions>\d+)\s+positions\b")
```

i.e. `kva: hazard <positions> positions` or `kva: hazard request <id> <positions> positions`
(extra line text is ignored; a `kva` line the regex does not match is reported as unparsed,
never dropped; unnamed lines count toward the total only). `--require-match` exits 1 when
the two instruments disagree. On the branch corpus they must agree per request (the counter
equals `min(N1−N2, T−(N2−P))`, which the construction makes equal to the records-side
overlap); on append-only traffic the plugin must report 0 and the flag must stay down.

## What each tool proves

- `branch_corpus.py`: R65's precondition is **constructed, not sampled** — every line
  guarantees `N2 − T < P ≤ shared prefix` with exact `/tokenize` lengths, so a device counter
  that disagrees with `hazard_bound` is wrong by arithmetic, not by corpus accident; the C
  legs are the R64-shaped append-only control (predicted 0) that R65 also requires
  ("append-only and regenerate corpora: counter 0").
- `hazard_rate.py`: R68's log-side instrument — it must flag exactly the B legs of the corpus
  and none of the A/C legs, and its totals must equal the plugin counter's on the same run
  (`--require-match` is the gate form). The unit tests cover the arithmetic on hand-made
  records (including the `cache_n == prompt_len − T` boundary), the log contract regex, and
  an end-to-end pass where records hand-made from a corpus build are flagged exactly as
  predicted.

## Caveats

- `--tail` is the plugin's `kva.tail` (T). The defaults match the protocol (2048/2048); the
  two knobs are kept separate because `--checkpoint-interval` below `max_tok` is the
  documented operator route (R84).
- The predicted counts are the **cache-side superset**; whether an overlapped position was
  actually approximated is the producer's `N1 − T` boundary, which the B construction pins
  (`P ≤ N1 − T`) — outside this corpus the log-side flag alone is not evidence of a hazard,
  only of a cache overlap in the tail.
- `/tokenize` renders with the server's chat template; the lengths are exact for that server
  and template. Re-run the builder if the template or `--interval` changes.