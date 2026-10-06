#!/bin/sh
# once.sh -- run the engine ONCE in the foreground with the shared flags plus extra args, for the
# engine modes that print and exit (--debug-graph, --debug-placement, --debug-selection) or any
# one-off run whose log is the evidence. Output goes to stdout+stderr; the caller redirects it.
#
#   scripts/once.sh <mode> [extra engine args...]
#
# Same env vars as serve.sh (common.sh): RK_MODEL, RK_IMAGE, RK_PLUGIN_HOME, RK_FLAGS, and every
# RADIANCE_RIDGEFILL* / RADIANCE_LOG_STEPS passed through.
set -eu
. "$(dirname "$0")/common.sh"
[ "$#" -ge 1 ] || rk_die "usage: scripts/once.sh <mode> [extra engine args...]"
mode=$1
shift
rk_tmp=$(mktemp) || rk_die "mktemp failed"
trap 'rm -f "$rk_tmp"' EXIT INT TERM
{
    printf '%s\n' --rm
    rk_docker_prefix "$mode"
    printf '%s\n' "$RK_IMAGE" --model "/models/$(basename "$RK_MODEL")"
    # shellcheck disable=SC2086  # RK_FLAGS is one flag or value per word by construction
    printf '%s\n' $RK_FLAGS
    printf '%s\n' "$@"
} > "$rk_tmp"
set --
while IFS= read -r rk_arg; do
    [ -n "$rk_arg" ] || continue
    set -- "$@" "$rk_arg"
done < "$rk_tmp"
docker run "$@"
