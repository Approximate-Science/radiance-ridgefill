#!/bin/sh
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# conc.sh -- a long prompt beside decoding users, against the server scripts/serve.sh left running
# (Stage B: REQUIREMENTS-FIX R54-R57, R60). It starts nothing.
#
#   scripts/conc.sh ttft <label>   the fnconc pattern, BOTH halves on one clock: C streaming decoders
#                                  reach steady state, then one long prompt; per rep the prompt's
#                                  timings.prompt_ms (R55), every decoder token's arrival time (their
#                                  gaps alone and during the prefill = the decoders' ms/step, R56),
#                                  the engine's step counters, and the server log of the prefill
#                                  window -- the plugin's approximate-step lines counted, and with
#                                  RADIANCE_LOG_STEPS=1 on the server the engine's step lines, which
#                                  tools/conc_steps.py holds to the rule step by step (R57). C = 0 is
#                                  the solo prefill, the reference each C's speedup is read against.
#   scripts/conc.sh text <label>   the tiercross pattern made deterministic: ONE request whose prompt
#                                  is the batch [decoder_1 .. decoder_D, a 32K prompt], so every run
#                                  of every mode puts the same decoder token beside the same prefill
#                                  chunk; each decoder's text (temperature 0), its sha256, the draft
#                                  counters (R60), and each decoder alone. R54: a mode's decoder texts
#                                  equal off's byte for byte; RADIANCE_RIDGEFILL_MASK=all must change them.
#   scripts/conc.sh accept <label> R60's acceptance half: the long prompt, then RK_TEXT_D decoders 0.5 s
#                                  later as SEPARATE requests, each reporting its own draft_n /
#                                  draft_n_accepted (a batched request's counts sum all its choices,
#                                  the approximated prompt's included).
#
# Env (defaults in tools/conc.py / scripts/common.sh): RK_DOCS (required: JSONL of {"prompt"}, the long
# prompt's text), RK_PORT, RK_EVIDENCE, RK_STAGE (output dir under RK_EVIDENCE), RK_CONTAINER (default
# radiance-ridgefill-<mode of the running container>), RK_LENGTHS "16384 32768", RK_CONC "0 1 4 8", RK_REPS
# (7 ttft / 3 text), RK_TEXT_D "1 4", RK_TEXT_TOKENS 256, RK_TEXT_LENGTH 32768.
# Output: $RK_EVIDENCE/$RK_STAGE/conc-<label>.json (+ conc-<label>.logs/, text-<label>-*.log).
set -eu

. "$(dirname "$0")/common.sh"

[ "$#" -eq 2 ] || rk_die "usage: RK_DOCS=<ppl.jsonl> scripts/conc.sh ttft|text|accept <label>"
[ -n "${RK_DOCS:-}" ] && [ -r "$RK_DOCS" ] || rk_die "RK_DOCS is not a readable file: ${RK_DOCS:-unset}"
if [ -z "${RK_CONTAINER:-}" ]; then
    RK_CONTAINER=$(docker ps --filter name=radiance-ridgefill- --format '{{.Names}}' | head -1)
    [ -n "$RK_CONTAINER" ] || rk_die "no radiance-ridgefill container is running (scripts/serve.sh first)"
fi
export RK_CONTAINER RK_DOCS
exec python3 "$RK_TOOLS/conc.py" "$1" "$2"
