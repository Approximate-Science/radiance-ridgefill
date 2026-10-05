# Notes — the needle-in-a-haystack retrieval harness (`tools/needle.py`)

Written with the harness, 2026-10-05. The engine cannot run in this sandbox, so the
harness is built and tested against a fake HTTP server shaped like the real endpoints
(`tests/test_needle.py`); the endpoint shapes come from the radiance tree at v1.0.8,
commit 140987f (read-only), as documented in `notes/scripts.md` and used by
`tools/speed.py`. Python standard library only, like `speed.py`.

## What it measures and why perplexity is not enough

The KVA plugin accelerates prefill by approximating most of a long prompt and keeping
only the last T = 2048 tokens exact (README.md `RADIANCE_KVA_TAIL`; `tools/speed.py`
`tail_note` is the arithmetic at the protocol lengths). Perplexity on a long doc is a
mean over positions and the exact tail dominates it, so a fact lost in the approximated
bulk moves the score by less than run-to-run noise: a KL/perplexity gate can pass while
the engine has quietly stopped retrieving anything from the first N − T tokens. This
harness makes that failure binary: it buries one fact at a known token depth, asks the
model to return it, and the reply is either right or wrong. Both the position and the
bulk/tail side of the boundary are recorded per item, so `compare` can say not just
"KVA is worse" but "KVA loses exactly the needles in the approximated bulk".

## Usage

```sh
# 1. Build the corpus (filler from the docs' "prompt" fields, needles at the depths,
#    every prompt cut to an EXACT token count through the served model's tokenizer):
python3 tools/needle.py build \
  --docs /var/home/dylan/AI-Work/kva-flashnext-tests-data/samples/quick/ppl.jsonl \
  --lengths 8192,16384,32768 --depths 0.05,0.25,0.5,0.75,0.9,0.98 \
  --keys 4 --seed 0 --server http://127.0.0.1:8100 --out corpus.jsonl

# 2. Run it against each engine (stock, then the KVA build), resumable:
python3 tools/needle.py run --corpus corpus.jsonl \
  --server http://127.0.0.1:8100 --out results-stock.jsonl --max-tokens 16
python3 tools/needle.py run --corpus corpus.jsonl \
  --server http://127.0.0.1:8101 --out results-kva.jsonl

# 3. Compare:
python3 tools/needle.py compare results-stock.jsonl results-kva.jsonl
```

`build` refuses to overwrite its output; `run` appends one JSON line per item to `--out`
and skips any id already present, so an interrupted run is resumed by re-invoking the
same command; `compare` exits non-zero naming the ids if the two results files do not
cover exactly the same items.

## The corpus

Per (length, depth) two variants are written:

- **single** — one needle, the question asks for it;
- **multi** — `--keys` needles (default 4, distinct keys and distinct 7-digit numbers)
  in one block at the same depth, and the question asks for one of them chosen by the
  seeded RNG: the model must pick the asked key, not recite any number it half-remembers.

The needle is `The special magic number for <key> is <7 digits>.` and the question,
always the last thing in the prompt, is `What is the special magic number for <key>?
Answer with the number only.` Keys are drawn from a fixed in-file pool of common words
(a stdlib-only tool has no word list on tap); keys, numbers and the asked index all
come from `--seed`, so a seed reproduces a corpus byte for byte (tested).

Exact lengths work like `speed.py`: every filler doc is tokenised once through
`POST /tokenize` with `add_special_tokens: false` (no BOS is silently prepended,
core/server/admin.cpp:119-196), the docs' ids are concatenated, cycled and cut so that
filler + needle block + question is EXACTLY the target token count, and the ids are
sent to `POST /v1/completions` as `"prompt": [ids]` (core/server/oai.cpp:1549-1568).
`run` verifies `usage.prompt_tokens` against the sent count on every response.

Every corpus item records: `id` (`L<length>/D<depth>/<variant>`), `length`, `depth` and
`depths`, `variant`, `keys`, `key`/`asked_index` (the asked needle), `expected` (its
7-digit number), the `question`, one entry per needle with its token `position`,
`region` and `in_tail`, `needle_pos`/`region`/`in_tail` for the asked needle,
`exact_tail_tokens` (2048), `seed`, `prompt_tokens` and the full `prompt_ids`.

## Depths, and why they cluster near the tail boundary

`region` labels a position `tail` if it lies in the last 2048 tokens of the prompt
(`position >= length - 2048`), `bulk` if it lies in the approximated region before
that. The depth is relative to the whole prompt (`round(depth * (length - 1))`,
clamped into the filler), so the same depth list samples different sides of the
boundary as the length grows — the default depths are chosen so that 0.9 and 0.98
straddle it at 16K/32K:

| length | boundary (length − 2048) | depth → needle token (side) |
|---|---|---|
| 8192 | 6144 | 0.05 → 410 (bulk), 0.25 → 2048 (bulk), 0.5 → 4096 (bulk), **0.75 → 6143 (bulk, by ONE token)**, 0.9 → 7372 (tail), 0.98 → 8027 (tail) |
| 16384 | 14336 | 0.05 → 819 (bulk), 0.25 → 4096 (bulk), 0.5 → 8192 (bulk), 0.75 → 12287 (bulk), **0.9 → 14745 (tail)**, 0.98 → 16055 (tail) |
| 32768 | 30720 | 0.05 → 1638 (bulk), 0.25 → 8192 (bulk), 0.5 → 16384 (bulk), 0.75 → 24575 (bulk), **0.9 → 29490 (bulk)**, **0.98 → 32112 (tail)** |

Reading the straddle: at 16K a 0.9-depth needle sits 1,639 tokens from the end —
inside the 2,048-token exact tail, where KVA is expected to retrieve it; at 32K the
same 0.9 depth sits 3,278 tokens from the end — inside the approximated bulk, where a
correct answer would prove nothing about approximation and a wrong one is the expected
signature. Only 0.98 survives in the tail at 32K, and at 8K even 0.75 lands one token
before the boundary, which is why the positions are computed and labelled exactly
rather than assumed from the depth. The question itself is the last ~16 tokens and so
always inside the tail. A KVA-vs-stock `compare` that shows stock near 100% everywhere
but KVA collapsing on `bulk` (with `tail` intact) is the direct evidence that the
plugin's approximation is losing retrievable facts; KVA matching stock on `tail`
at every length is the direct evidence that the exact tail really is exact.

## `run` and `compare`

`run` sends each item at temperature 0 with `--max-tokens` (default 16) and parses the
first run of exactly 7 digits in `choices[0].text` — runs of 6 or 8+ digits do not
match, since the needles' numbers are always exactly 7 digits and an 8-digit run
would contain a misleading 7-digit substring. A reply with no 7-digit number counts as
incorrect (`parsed: null`), never as a crash. Each result line carries the reply, the
parsed value, `timings.prompt_ms` and the corpus metadata (length, depth, region,
variant), so `compare` needs the two results files only.

`compare` prints accuracy for A and B overall and by length, depth bucket, bulk-vs-tail
region and variant, then the paired disagreement table (both right / both wrong /
A right-B wrong `b` / A wrong-B right `c`) with an EXACT two-sided McNemar p-value:
`p = min(1, 2 · P(X ≤ min(b, c)))`, `X ~ Binomial(b + c, 1/2)` — every disagreeing pair
is a Bernoulli trial under the null that the two engines are equally likely to be the
right one, and the exact binomial sum avoids any large-sample chi-square approximation
(the interesting tables here are small: 2 items per length×depth×variant cell). It
ends with the list of items where A and B disagree, each with the loser's reply.

## Tests

`tests/test_needle.py` runs the harness against a fake radiance server in a thread: a
word-level `/tokenize` (exact counts, `add_special_tokens=false` asserted) and a
`/v1/completions` that decodes the prompt, finds the asked needle and answers its
number — always (mode `stock`, the control) or only when the needle sits in the last
2048 prompt tokens (mode `kva`, the plugin's failure mode). Covered: exact token
lengths and needle token positions (including that the block really is the needle and
the question really is last), determinism per seed, distinct keys/numbers, the
bulk/tail flip across the boundary, 7-digit parsing edge cases, resume (done ids are
never re-sent), wrong-answer recording, McNemar on known tables, and the full
build → run → run → compare path with the expected accuracies, disagreement counts
and p-value.