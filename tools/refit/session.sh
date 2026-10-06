#!/bin/sh
# session.sh -- the Stage 6 refit's GPU sessions, each the WHOLE serve -> capture -> stop sequence under the GPU
# lock (WORKER-RULES: the lock file is the caller's RK_GPU_LOCK; flock blocks until the holder releases it).
#
# USAGE: tools/refit/session.sh exact|speed
#   exact  (1) `off` + RADIANCE_RIDGEFILL_CAPTURE: the projector's held-out then training documents (activations);
#          (2) `off` + RADIANCE_RIDGEFILL_CAPTURE_STATE: the +st prompts' exact late delta-net states.
#   speed  `speed` with the refit projector, no correction (RADIANCE_RIDGEFILL_PROJ=refit, RADIANCE_RIDGEFILL_ST=refit while
#          the container holds no ridgefill.str.*), tail 512 (tcc's FIT_TAIL) + RADIANCE_RIDGEFILL_CAPTURE_STATE: the +st
#          prompts' predicted states. Needs ridgefill.projr.* appended to the container first.
#
# Recording is turned off with RADIANCE_DEBUG_ARGSHA=1 (radiance core/runtime/ctx.cpp:120 disables pass recording
# under it; its digest code runs only on draft pass 1, which these runs never have), because a replayed pass never
# calls the plugin's step() and would silently skip its capture. --profile-ops does the same but restores a
# synchronisation around every launch (core/engine.cpp:818-821: several times slower). capture.py still checks
# that every chunk was captured and stops if not.
#
# Env: RK_MODEL (required), RK_PLUGIN_HOME (required: the frozen home built from the capture commit),
#      RK_GPU_LOCK (required), RK_REFIT_DATA (default <repo>/data/refit), RK_PORT (8100), plus scripts/common.sh's.
# These set RADIANCE_RIDGEFILL_PROJ/_ST/_DECLARE, which the current plugin refuses: run them only against the frozen
# home they were written for (see tools/dev/README.md).
set -eu

[ "$#" -eq 1 ] || { echo "usage: tools/refit/session.sh exact|speed" >&2; exit 1; }
phase=$1
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo=$(CDPATH= cd "$here/../.." && pwd)
: "${RK_MODEL:?the .rad container}" "${RK_PLUGIN_HOME:?the frozen plugin home}" "${RK_GPU_LOCK:?the GPU lock file}"
: "${RK_REFIT_DATA:=$repo/data/refit}"
: "${RK_PORT:=8100}"
: "${RK_PYTHON:=python3}"
export RK_MODEL RK_PLUGIN_HOME RK_PORT
prompts=$RK_REFIT_DATA/prompts
log=$RK_REFIT_DATA/session-$phase-$(date -u +%Y%m%dT%H%M%SZ).log

# kernel_clean WHEN -- no amdgpu MES/SMU/timeout/reset line since the lock was taken (10 min before it for "before")
kernel_clean() {
    hits=$(journalctl -k --since "$since" 2>/dev/null | grep -iE 'amdgpu.*(MES|SMU|timeout|reset)' || true)
    [ -z "$hits" ] || { printf 'kernel log has amdgpu MES/SMU/timeout/reset lines:\n%s\n' "$hits" >&2; return 1; }
    echo "kernel log clean ($1)"
}

# serve_capture MODE ENGINE_DIR WHAT STORE PROMPT_FILES... -- with the capture env already exported. Every step is
# checked by hand: `set -e` does not apply inside a function called from an `||` list.
serve_capture() {
    mode=$1 engine=$2 what=$3 store=$4
    shift 4
    mkdir -p "$engine" "$store" || return 1
    if ! RK_DOCKER_EXTRA="-v $engine:/cap -e RADIANCE_DEBUG_ARGSHA=1" "$repo/scripts/serve.sh" "$mode"; then
        docker logs radiance-ridgefill-"$mode" > "$store/engine-$mode.log" 2>&1 || true
        "$repo/scripts/stop.sh"
        return 1
    fi
    status=0
    "$RK_PYTHON" "$here/capture.py" run --what "$what" --prompts "$@" --engine-dir "$engine" --store "$store" \
        --sums "$RK_REFIT_DATA/sums" --port "$RK_PORT" || status=$?
    docker logs radiance-ridgefill-"$mode" > "$store/engine-$mode.log" 2>&1 || true
    "$repo/scripts/stop.sh" || status=1
    return $status
}

session() {
    kernel_clean before || return 1
    case $phase in
    exact)
        RADIANCE_RIDGEFILL_CAPTURE=/cap; export RADIANCE_RIDGEFILL_CAPTURE
        serve_capture off "$RK_REFIT_DATA/engine-act" activations "$RK_REFIT_DATA/store" \
            "$prompts/held.jsonl" "$prompts/train.jsonl" || return 1
        unset RADIANCE_RIDGEFILL_CAPTURE
        kernel_clean between || return 1
        RADIANCE_RIDGEFILL_CAPTURE_STATE=/cap; export RADIANCE_RIDGEFILL_CAPTURE_STATE
        serve_capture off "$RK_REFIT_DATA/engine-state-exact" state "$RK_REFIT_DATA/state-exact" \
            "$prompts/sterm.jsonl" || return 1
        ;;
    speed)
        RADIANCE_RIDGEFILL_CAPTURE_STATE=/cap RADIANCE_RIDGEFILL_PROJ=refit RADIANCE_RIDGEFILL_ST=refit RADIANCE_RIDGEFILL_TAIL=512
        export RADIANCE_RIDGEFILL_CAPTURE_STATE RADIANCE_RIDGEFILL_PROJ RADIANCE_RIDGEFILL_ST RADIANCE_RIDGEFILL_TAIL
        serve_capture speed "$RK_REFIT_DATA/engine-state-speed" state "$RK_REFIT_DATA/state-speed" \
            "$prompts/sterm.jsonl" || return 1
        ;;
    *) echo "phase '$phase' is not exact|speed" >&2; return 1 ;;
    esac
    kernel_clean after
}

echo "session $phase: waiting for $RK_GPU_LOCK (log $log)"
exec 9>>"$RK_GPU_LOCK"
flock 9
echo "session $phase: lock held $(date -u +%H:%M:%SZ)"
since=$(date -d '-10 min' '+%Y-%m-%d %H:%M:%S')
if session > "$log" 2>&1; then
    echo "session $phase: done $(date -u +%H:%M:%SZ) (log $log)"
else
    echo "session $phase: FAILED $(date -u +%H:%M:%SZ) (log $log)" >&2
    tail -n 20 "$log" >&2
    exit 1
fi
