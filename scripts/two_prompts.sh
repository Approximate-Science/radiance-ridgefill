#!/bin/sh
# two_prompts.sh -- R58': two non-final prefill chunks in one step are both served correctly, and the
# last one is approximated (PLAN-FIX §3: the device mask ignores every row before the last sequence).
#
# When do two prefills share a step? A non-final chunk ends on the scheduler's quantum, so a step's
# budget is left over only when a prompt's FINAL chunk is short; the scheduler tops the step up with
# the next prompt's first chunk (scripts/grade.sh's note). So the shape is "A's final chunk, exact,
# then B's chunk, approximated" -- at --max-num-batched-tokens 2048 and 8192 alike.
#
#   scripts/two_prompts.sh kl <mode> <ref_dir> <out.json>
#       the KL half: scripts/grade.sh with --max-num-seqs 2, so the corpus' docs run in pairs (quick9's
#       two 32K docs together) and every pair's hand-over is a two-prefill step. Both tails are scored
#       against the stock reference; compare with the same mode's --max-num-seqs 1 run per doc
#       (scripts/kl_tail.py): the difference must sit at the GEMM-shape floor that exact-at-2 vs
#       exact-at-1 shows. RK_FLAGS selects 2048 or 8192.
#   scripts/two_prompts.sh live <label>
#       against the server scripts/serve.sh left running (start it with RADIANCE_LOG_STEPS=1 and
#       RADIANCE_KVA_DUMP=<dir> mounted writable): two RK_LENGTHS prompts in ONE batched request beside
#       RK_CONC decoders (default 8); the log window's approximate lines with Pn 2 are counted, and
#       tools/mask_pn2.py <dump dir> checks the dumped mask of every two-prefill step: no row before
#       the last sequence approximated.
set -eu

. "$(dirname "$0")/common.sh"

[ "$#" -ge 1 ] || rk_die "usage: scripts/two_prompts.sh kl <mode> <ref_dir> <out.json> | live <label>"
case $1 in
kl)
    shift
    [ "$#" -eq 3 ] || rk_die "usage: scripts/two_prompts.sh kl <mode> <ref_dir> <out.json>"
    RK_KLD_SEQS=2 exec "$RK_SCRIPTS/grade.sh" "$@" ;;
live)
    [ "$#" -eq 2 ] || rk_die "usage: scripts/two_prompts.sh live <label>"
    [ -n "${RK_DOCS:-}" ] && [ -r "$RK_DOCS" ] || rk_die "RK_DOCS is not a readable file: ${RK_DOCS:-unset}"
    RK_CONTAINER=${RK_CONTAINER:-$(docker ps --filter name=radiance-kva- --format '{{.Names}}' | head -1)}
    [ -n "$RK_CONTAINER" ] || rk_die "no radiance-kva container is running (scripts/serve.sh first)"
    : "${RK_LENGTHS:=32768}" "${RK_CONC:=8}"
    export RK_CONTAINER RK_DOCS RK_LENGTHS RK_CONC
    exec python3 "$RK_TOOLS/conc.py" pair "$2" ;;
*)
    rk_die "unknown form '$1': kl or live" ;;
esac
