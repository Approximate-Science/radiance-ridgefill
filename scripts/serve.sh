#!/bin/sh
# serve.sh -- start the engine in one KVA mode as a detached container and leave it up.
#
# WHAT IT IS FOR: the deployment every speed/quality number is taken against; it measures
# nothing itself. scripts/speed.sh attaches to the server this leaves running.
#
# USAGE: scripts/serve.sh <mode> [extra engine args...]
#   mode: exact (the stock engine, no plugin -- the baseline) | off | plumb | speed |
#   quality (see scripts/common.sh for what each mode means).
#
# WHAT IT DOES: runs scripts/preflight.sh first (fail closed before touching the cards),
# refuses if any radiance-kva-* container is already running (Flash-Next needs both
# cards), starts radiance-kva-<mode> detached with the standard prefix (common.sh) and
# RK_FLAGS plus --max-num-seqs 8 (fnserve.sh's serving default), waits for GET /health
# (RK_SERVE_TIMEOUT, progress every 30 s), and records the container id, the image
# digest, the model file size and the full command in
# $RK_EVIDENCE/serve-<mode>-<UTC timestamp>.cmd for provenance.
#
# Env vars (defaults in scripts/common.sh unless noted):
#   RK_MODEL          required: host path to the .rad container
#   RK_PORT           8100
#   RK_IMAGE          stilldeadcode/radiance:1.0.8
#   RK_PLUGIN_HOME    <repo>/home (mounted at /plugins for every mode except exact)
#   RK_EVIDENCE       <repo>/evidence
#   RK_FLAGS          the shared engine flags (common.sh)
#   RK_SERVE_SEQS     8: --max-num-seqs for the serving deployment
#   RK_SERVE_TIMEOUT  1800 s: how long to wait for /health (a 114 GiB model load is slow)
#   plus any RADIANCE_KVA*/RADIANCE_LOG_STEPS in the caller's environment, passed through
#   to the container.

set -eu

. "$(dirname "$0")/common.sh"

[ "$#" -ge 1 ] || rk_die "usage: scripts/serve.sh <mode> [extra engine args...]"
mode=$1
shift
rk_mode_validate "$mode"
rk_require_model
: "${RK_SERVE_TIMEOUT:=1800}"
: "${RK_SERVE_SEQS:=8}"

container=radiance-kva-$mode
running=$(docker ps --filter name=radiance-kva- --format '{{.Names}}' || true)
[ -z "$running" ] || rk_die "refusing to start: a radiance-kva container is already running: $running"

# fail closed before touching the cards: squatters, leftover VRAM, amdgpu kernel errors
"$RK_SCRIPTS/preflight.sh" || rk_die "preflight failed before serve; nothing was started"

# the caller's extra engine args, one per line, appended at the end of the command
rk_extra=$(mktemp) || rk_die "mktemp failed"
printf '%s\n' "$@" > "$rk_extra"

# the whole docker command line, one argument per line (POSIX sh has no arrays)
rk_args=$(mktemp) || rk_die "mktemp failed"
trap 'rm -f "$rk_args" "$rk_extra"' EXIT INT TERM
{
    printf '%s\n' -d --name "$container"
    rk_docker_prefix "$mode"
    printf '%s\n' "$RK_IMAGE" --model "/models/$(basename "$RK_MODEL")"
    # shellcheck disable=SC2086  # RK_FLAGS is one flag or value per word by construction
    printf '%s\n' $RK_FLAGS
    printf '%s\n' --host 0.0.0.0 --port "$RK_PORT" --max-num-seqs "$RK_SERVE_SEQS"
    cat "$rk_extra"
} > "$rk_args"

digest=$(docker image inspect --format '{{.Id}}' "$RK_IMAGE" 2>/dev/null || true)
[ -n "$digest" ] || rk_die "docker image inspect failed for $RK_IMAGE (is the image pulled?)"

mkdir -p "$RK_EVIDENCE"
cmd_file=$RK_EVIDENCE/serve-$mode-$(date -u +%Y%m%dT%H%M%SZ).cmd
{
    printf '# serve: container %s (mode %s) started %s UTC\n' \
        "$container" "$mode" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '# image: %s\n' "$RK_IMAGE"
    printf '# image digest: %s\n' "$digest"
    printf '# model: %s (%s bytes)\n' "$RK_MODEL" "$(wc -c < "$RK_MODEL" | tr -d ' ')"
    printf '# port: %s\n' "$RK_PORT"
    printf '# docker run command, one argument per line (paths with spaces stay unambiguous):\n'
    printf 'docker run\n'
    while IFS= read -r rk_arg; do
        [ -n "$rk_arg" ] || continue
        printf '  %s\n' "$rk_arg"
    done < "$rk_args"
} > "$cmd_file"
printf 'command recorded in %s\n' "$cmd_file"

# materialise the command line and start
set --
while IFS= read -r rk_arg; do
    [ -n "$rk_arg" ] || continue
    set -- "$@" "$rk_arg"
done < "$rk_args"
if ! cid=$(docker run "$@" 2>&1); then
    rk_die "docker run failed: $cid"
fi

# wait for /health, progress every 30 s, refuse to outlive the container
printf 'container %s starting (id %s); waiting for /health on port %s (timeout %ss)\n' \
    "$container" "$cid" "$RK_PORT" "$RK_SERVE_TIMEOUT"
waited=0
while :; do
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' \
        "http://127.0.0.1:$RK_PORT/health" 2>/dev/null || true)
    [ "$code" = 200 ] && break
    state=$(docker container inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)
    [ "$state" = true ] || rk_die "container $container exited before /health answered; log: docker logs $container"
    if [ "$waited" -ge "$RK_SERVE_TIMEOUT" ]; then
        rk_die "/health did not answer within ${RK_SERVE_TIMEOUT}s on port $RK_PORT"
    fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 30)) -eq 0 ]; then
        printf '  ...still waiting after %ss\n' "$waited"
    fi
done

# the health answer must be OUR container's, not something else on the port: preflight
# (a) already refused any other engine on the box, this catches the rest
state=$(docker container inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)
[ "$state" = true ] || rk_die "container $container is not running; log: docker logs $container"

printf 'UP: container %s (id %s) on port %s\n' "$container" "$cid" "$RK_PORT"
printf '  completions  http://127.0.0.1:%s/v1/completions\n' "$RK_PORT"