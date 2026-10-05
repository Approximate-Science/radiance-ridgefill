#!/bin/sh
# session.sh -- the investigation's GPU session: GDN state captures of the 14 +st prompts in speed mode with the
# correction at several strengths, each serve -> capture -> stop, ALL under one hold of the GPU lock.
#
# WHY: instrument check of the +st apply on the engine (notes/investigate.md, labbook INV-apply). alpha=0 is the
# no-correction control with the SAME declared weights (same placement); alpha=1 is the shipped correction. The
# captures are the plugin's RADIANCE_KVA_CAPTURE_STATE files (notes/arch.md "Capture"): pre-apply at approximate
# chunk ends, after the step on exact chunks (the tail).
#
# USAGE: tools/investigate/session.sh ARM [ARM...]   ARM = name:alpha[:st]  e.g. st1:1 st0:0 swap1:1:swap
# Env: RK_MODEL, RK_PLUGIN_HOME (frozen capture home), RK_GPU_LOCK (required); RK_INV_DATA (<repo>/data/investigate);
#      RK_TAIL (512), RK_PROJ (refit), RK_ST (shipped), RK_PROMPTS (<repo>/data/refit/prompts/sterm.jsonl).
set -eu
[ "$#" -ge 1 ] || { echo "usage: tools/investigate/session.sh name:alpha[:st] [...]" >&2; exit 1; }
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo=$(CDPATH= cd "$here/../.." && pwd)
: "${RK_MODEL:?the .rad container}" "${RK_PLUGIN_HOME:?the frozen plugin home}" "${RK_GPU_LOCK:?the GPU lock file}"
: "${RK_INV_DATA:=$repo/data/investigate}" "${RK_TAIL:=512}" "${RK_PROJ:=refit}" "${RK_ST:=shipped}"
: "${RK_PROMPTS:=$repo/data/refit/prompts/sterm.jsonl}" "${RK_PORT:=8100}" "${RK_PYTHON:=python3}"
export RK_MODEL RK_PLUGIN_HOME RK_PORT
log=$RK_INV_DATA/session-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$RK_INV_DATA"

kernel_clean() {
    hits=$(journalctl -k --since "$since" 2>/dev/null | grep -iE 'amdgpu.*(MES|SMU|timeout|reset)' || true)
    [ -z "$hits" ] || { printf 'kernel log has amdgpu MES/SMU/timeout/reset lines:\n%s\n' "$hits" >&2; return 1; }
    echo "kernel log clean ($1) $(date -u +%H:%M:%SZ)"
}

one_arm() {
    name=${1%%:*} rest=${1#*:}
    alpha=${rest%%:*} st=$RK_ST
    case $rest in *:*) st=${rest#*:} ;; esac
    engine=$RK_INV_DATA/engine-$name store=$RK_INV_DATA/state-$name
    mkdir -p "$engine" "$store" || return 1
    RADIANCE_KVA_CAPTURE_STATE=/cap RADIANCE_KVA_PROJ=$RK_PROJ RADIANCE_KVA_ST=$st RADIANCE_KVA_TAIL=$RK_TAIL \
        RADIANCE_KVA_ALPHA=$alpha
    export RADIANCE_KVA_CAPTURE_STATE RADIANCE_KVA_PROJ RADIANCE_KVA_ST RADIANCE_KVA_TAIL RADIANCE_KVA_ALPHA
    echo "arm $name: alpha $alpha proj $RK_PROJ st $st tail $RK_TAIL"
    if ! RK_DOCKER_EXTRA="-v $engine:/cap -e RADIANCE_DEBUG_ARGSHA=1" "$repo/scripts/serve.sh" speed; then
        docker logs radiance-kva-speed > "$store/engine-speed.log" 2>&1 || true
        "$repo/scripts/stop.sh"
        return 1
    fi
    status=0
    "$RK_PYTHON" "$repo/tools/refit/capture.py" run --what state --prompts "$RK_PROMPTS" --engine-dir "$engine" \
        --store "$store" --port "$RK_PORT" || status=$?
    docker logs radiance-kva-speed > "$store/engine-speed.log" 2>&1 || true
    "$repo/scripts/stop.sh" || status=1
    return $status
}

session() {
    kernel_clean before || return 1
    for arm in "$@"; do
        one_arm "$arm" || return 1
        kernel_clean "after $arm" || return 1
    done
}

echo "session: waiting for $RK_GPU_LOCK (log $log)"
exec 9>>"$RK_GPU_LOCK"
flock 9
echo "session: lock held $(date -u +%H:%M:%SZ)"
since=$(date -d '-10 min' '+%Y-%m-%d %H:%M:%S')
if session "$@" > "$log" 2>&1; then
    echo "session: done $(date -u +%H:%M:%SZ) (log $log)"
else
    echo "session: FAILED $(date -u +%H:%M:%SZ) (log $log)" >&2
    tail -n 30 "$log" >&2
    exit 1
fi
