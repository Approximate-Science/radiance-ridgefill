#!/bin/sh
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# grade.sh -- the KL-mode quality measurement (radiance docs/TOOLS.md lines 470-506).
#
# WHAT IT MEASURES: how far a RidgeFill mode is from the exact engine on the quick quality set,
# scored on the exact tail only: per-position KL divergence, both engines' perplexity and
# top-1 agreement, per source. The reports (and their .rows files) are compared doc by
# doc with tools/paired.py.
#
# USAGE:
#   scripts/grade.sh record <ref_dir> <corpus.jsonl>    record the reference, ONCE (stock engine)
#   scripts/grade.sh <mode> <ref_dir> <out.json>        score one candidate against it
#     mode: exact (the stock engine: the exact-vs-exact noise floor) | off | plumb | speed | quality
#     <corpus.jsonl>: the KL-mode corpus (tools/kld_corpus.py output), one
#       {"prompt", "score_from", "source"} per line, docs CUT to 2048-token multiples.
#
# Both forms run the engine in the FOREGROUND (docker run --rm) with RK_FLAGS plus
# --max-num-seqs 1: KL mode admits every doc at once and the scheduler tops up a step's
# leftover budget with the next doc's first chunk -- a step with n_seq == 2 (approximated since
# Stage A's device mask); --max-num-seqs 1 forbids the top-up (HANDOVER §5 Stage 0 step 6).
# The reference and every candidate run with the SAME flags (same-boot protocol, HANDOVER
# §7; the noise floor is exact-vs-exact under this exact configuration -- TOOLS.md's bf16
# reference advice is for measuring a quantisation, not this).
#
# scripts/preflight.sh runs before each container. Refuses to overwrite an existing
# reference dir or out file (and their .rows / .log sidecars). The full container log is
# kept beside the output: <out>.log (candidates), <ref_dir>.log (record).
#
# After a candidate run the count of "ridgefill: approximate step" log lines is printed (the
# plugin logs one per approximate step); RK_EXPECT_APPROX, when set, is the expected
# count and a mismatch exits non-zero with both numbers (the bulk-chunk gate, HANDOVER R18).
# The plugin logs from step(), and a RECORDED pass replays without calling step() (radiance
# core/runtime/ctx.cpp run_pass): from radiance 1.1.x the 2048-token approximate pass is recorded
# and replayed, so the log shows 38 of quick9's 67 bulk chunks while the rows are byte-identical to
# 1.0.13's (gate A, 2026-10-06). So when RK_EXPECT_APPROX is set the candidate runs with
# RADIANCE_DEBUG_ARGSHA=1, which turns pass recording off (ctx.cpp prepare; its digest touches
# only MTP draft pass 1, off here) and every pass is issued live: the count is the whole count again.
#
# Env vars (defaults in scripts/common.sh unless noted):
#   RK_MODEL           required: host path to the .rad container
#   RK_FLAGS           the shared engine flags (common.sh)
#   RK_EXPECT_APPROX   unset by default: the expected approximate-step count (candidates)
#   RK_KLD_SEQS        default 1: --max-num-seqs of a CANDIDATE run. 2 lets the scheduler top a
#                      step up with the next doc's first chunk -- two prefills in one step, the
#                      shape R58' scores (scripts/two_prompts.sh); the reference stays at 1
#   plus the mode's RADIANCE_RIDGEFILL*/RADIANCE_LOG_STEPS pass-through into the container.

set -eu

. "$(dirname "$0")/common.sh"

usage() {
    rk_die "usage: scripts/grade.sh record <ref_dir> <corpus.jsonl> | scripts/grade.sh <mode> <ref_dir> <out.json>"
}

rk_args=$(mktemp) || rk_die "mktemp failed"
trap 'rm -f "$rk_args"' EXIT INT TERM

# run the command line collected in $rk_args, keep the container's whole log in $1
run_logged() {
    log=$1
    set --
    while IFS= read -r rk_arg; do
        [ -n "$rk_arg" ] || continue
        set -- "$@" "$rk_arg"
    done < "$rk_args"
    if ! docker run "$@" > "$log" 2>&1; then
        printf 'grade: the container failed; last lines of %s:\n' "$log" >&2
        tail -20 "$log" >&2 || true
        exit 1
    fi
}

model_arg="/models/$(basename "$RK_MODEL")"

[ "$#" -ge 1 ] || usage
if [ "$1" = record ]; then
    [ "$#" -eq 3 ] || usage
    rk_require_model
    ref_dir=$(realpath -m -- "$2")   # docker -v needs absolute host paths
    corpus=$(realpath -m -- "$3")
    [ -r "$corpus" ] || rk_die "the corpus is not a readable file: $corpus"
    if [ -d "$ref_dir" ] && [ -n "$(ls -A "$ref_dir" 2>/dev/null)" ]; then
        rk_die "refusing to overwrite an existing reference dir: $ref_dir"
    fi
    log=$ref_dir.log
    [ ! -e "$log" ] || rk_die "refusing to overwrite the log: $log"
    mkdir -p "$ref_dir"
    "$RK_SCRIPTS/preflight.sh" || rk_die "preflight failed before the record run; nothing was started"
    {
        printf '%s\n' --rm
        rk_docker_prefix exact     # the reference is the STOCK engine by definition
        printf '%s\n' -v "$ref_dir":/data/ref -v "$corpus":/data/corpus.jsonl:ro
        printf '%s\n' "$RK_IMAGE" --model "$model_arg"
        # shellcheck disable=SC2086  # RK_FLAGS is one flag or value per word by construction
        printf '%s\n' $RK_FLAGS --max-num-seqs 1
        printf '%s\n' --kld-record /data/ref --kld-corpus /data/corpus.jsonl
    } > "$rk_args"
    run_logged "$log"
    [ -f "$ref_dir/kld.json" ] || rk_die "the run reported success but $ref_dir/kld.json is missing; log: $log"
    printf 'grade: reference recorded in %s (log: %s)\n' "$ref_dir" "$log"
    exit 0
fi

[ "$#" -eq 3 ] || usage
mode=$1
ref_dir=$(realpath -m -- "$2")   # docker -v needs absolute host paths
out=$(realpath -m -- "$3")
rk_mode_validate "$mode"
rk_require_model
[ -f "$ref_dir/kld.json" ] || rk_die "not a finished reference (no kld.json): $ref_dir"
for f in "$out" "$out.rows" "$out.log"; do
    [ ! -e "$f" ] || rk_die "refusing to overwrite: $f"
done
out_dir=$(dirname "$out")
mkdir -p "$out_dir"
log=$out.log
"$RK_SCRIPTS/preflight.sh" || rk_die "preflight failed before the candidate run; nothing was started"
{
    printf '%s\n' --rm
    rk_docker_prefix "$mode"
    # every pass live when the approximate steps are counted (header): a replayed pass logs nothing
    [ -z "${RK_EXPECT_APPROX:-}" ] || printf '%s\n' -e RADIANCE_DEBUG_ARGSHA=1
    printf '%s\n' -v "$ref_dir":/data/ref:ro -v "$out_dir":/data/out
    printf '%s\n' "$RK_IMAGE" --model "$model_arg"
    # shellcheck disable=SC2086  # RK_FLAGS is one flag or value per word by construction
    printf '%s\n' $RK_FLAGS --max-num-seqs "${RK_KLD_SEQS:-1}"
    printf '%s\n' --kld-ref /data/ref --kld-out "/data/out/$(basename "$out")"
} > "$rk_args"
run_logged "$log"

# the plugin logs one line per approximate step; the count proves the bulk chunks ran
# approximate rather than falling back to exact (HANDOVER R18)
approx=$(grep -c 'ridgefill: approximate step' "$log" || true)
printf 'grade: %s approximate steps: %s (log: %s)\n' "$mode" "$approx" "$log"
if [ -n "${RK_EXPECT_APPROX:-}" ]; then
    if [ "$approx" != "$RK_EXPECT_APPROX" ]; then
        rk_die "approximate-step count $approx != expected $RK_EXPECT_APPROX (RK_EXPECT_APPROX)"
    fi
fi