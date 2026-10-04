#!/bin/sh
# speed.sh -- prefill latency at EXACT prompt lengths against the server already running on
# RK_PORT (scripts/serve.sh put it there; this script starts nothing).
#
# WHAT IT MEASURES: timings.prompt_ms (radiance docs/GUIDE.md §5.9: from the first step the
# scheduler gave the request to its first token) of prefill-only requests (max_tokens 1,
# temperature 0) at EXACTLY RK_LENGTHS tokens per prompt -- default 9216 16384 32768.
# 9216 is 4.5 chunks of 2048: the chunk at 6144 sees n_ahead = 1024 < T = 2048 and runs
# exact, so 9216's exact tail is 3072, not 2048, and its speedup sits below the row-exact
# number (HANDOVER §5 Stage 0 step 5; the evidence file carries this note per length).
# One warm-up then RK_REPS reps per length; the median of the reps is the number the
# report carries, with min and max for the spread. scripts/preflight.sh runs before
# every sample (the warm-up included) and a failure aborts the whole measurement.
#
# WHY THE PROMPTS ARE PROSE: the docs of RK_DOCS are tokenised with the served model's own
# tokenizer (POST /tokenize), concatenated and cut to the target length, with a DIFFERENT
# short leading nonce per request -- the prefix cache matches from the START of a prompt,
# so a leading nonce invalidates the whole match (scripts/fnpf.sh's rule). --no-prefix-cache
# is on in every serve anyway; the cached-token count is recorded and must be 0.
#
# USAGE: RK_DOCS=<quick ppl.jsonl> scripts/speed.sh <label>
#   label: goes into the output filename speed-<label>.json
#   RK_DOCS: REQUIRED, no default: JSONL, one {"id", "bucket", "prompt"} per line (the
#            quick quality set, .../samples/quick/ppl.jsonl); only "prompt" is used.
#
# Env vars (defaults here or in scripts/common.sh):
#   RK_DOCS              required (above)
#   RK_LENGTHS           default "9216 16384 32768": prompt token counts, EXACT
#   RK_REPS              default 5: reps per length after one warm-up
#   RK_PORT              default 8100
#   RK_STAGE             default scratch: output dir $RK_EVIDENCE/$RK_STAGE
#   RK_EVIDENCE          default <repo>/evidence
#   RK_SPEED_HTTP_TIMEOUT default 3600 s per request (a 32K prefill is minutes)
# The measurement itself is tools/speed.py (Python standard library only); this wrapper
# checks the inputs and hands over.

set -eu

. "$(dirname "$0")/common.sh"

[ "$#" -eq 1 ] || rk_die "usage: RK_DOCS=<quick ppl.jsonl> scripts/speed.sh <label>"
label=$1
[ -n "${RK_DOCS:-}" ] || rk_die 'RK_DOCS is not set: the JSONL whose "prompt" fields build the prompts (the quick quality set, .../samples/quick/ppl.jsonl)'
[ -r "$RK_DOCS" ] || rk_die "RK_DOCS is not a readable file: $RK_DOCS"
command -v python3 >/dev/null 2>&1 || rk_die "python3 is not on PATH"
exec python3 "$RK_TOOLS/speed.py" "$label"