#!/bin/sh
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# kl_session.sh -- KL-mode runs of the investigation, ALL under one hold of the GPU lock: record a reference on a
# corpus (stock engine), then speed-mode candidates at the given correction strengths.
#
# WHY: INV-final13 -- radiance on tcc's final-tier 16k/32k docs, so the +st effect is compared with tcc's best
# same-server pair on the same tokens (notes/investigate.md).
#
# USAGE: tools/investigate/kl_session.sh CORPUS REF_DIR OUT_PREFIX ARM [ARM...]   ARM = name:alpha
#   writes OUT_PREFIX-<name>.json (+ .rows, .log). The reference is recorded only if REF_DIR has no kld.json.
# Env: RK_MODEL, RK_PLUGIN_HOME (frozen home), RK_GPU_LOCK (required); RK_EXPECT_APPROX (bulk chunks of CORPUS).
set -eu
[ "$#" -ge 4 ] || { echo "usage: kl_session.sh CORPUS REF_DIR OUT_PREFIX name:alpha [...]" >&2; exit 1; }
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo=$(CDPATH= cd "$here/../.." && pwd)
: "${RK_MODEL:?the .rad container}" "${RK_PLUGIN_HOME:?the frozen plugin home}" "${RK_GPU_LOCK:?the GPU lock file}"
export RK_MODEL RK_PLUGIN_HOME
corpus=$1 ref=$2 prefix=$3
shift 3
log=$(dirname "$prefix")/kl-session-$(date -u +%Y%m%dT%H%M%SZ).log

kernel_clean() {
    hits=$(journalctl -k --since "$since" 2>/dev/null | grep -iE 'amdgpu.*(MES|SMU|timeout|reset)' || true)
    [ -z "$hits" ] || { printf 'kernel log has amdgpu MES/SMU/timeout/reset lines:\n%s\n' "$hits" >&2; return 1; }
    echo "kernel log clean ($1) $(date -u +%H:%M:%SZ)"
}

session() {
    kernel_clean before || return 1
    if [ ! -f "$ref/kld.json" ]; then
        "$repo/scripts/grade.sh" record "$ref" "$corpus" || return 1
        kernel_clean "after record" || return 1
    fi
    for arm in "$@"; do
        name=${arm%%:*} alpha=${arm#*:}
        RADIANCE_RIDGEFILL_ALPHA=$alpha RADIANCE_RIDGEFILL_PROJ=shipped RADIANCE_RIDGEFILL_ST=shipped
        export RADIANCE_RIDGEFILL_ALPHA RADIANCE_RIDGEFILL_PROJ RADIANCE_RIDGEFILL_ST
        echo "arm $name: speed, alpha $alpha"
        "$repo/scripts/grade.sh" speed "$ref" "$prefix-$name.json" || return 1
        kernel_clean "after $name" || return 1
    done
}

echo "kl session: waiting for $RK_GPU_LOCK (log $log)"
exec 9>>"$RK_GPU_LOCK"
flock 9
echo "kl session: lock held $(date -u +%H:%M:%SZ)"
since=$(date -d '-10 min' '+%Y-%m-%d %H:%M:%S')
if session "$@" > "$log" 2>&1; then
    echo "kl session: done $(date -u +%H:%M:%SZ) (log $log)"
else
    echo "kl session: FAILED $(date -u +%H:%M:%SZ) (log $log)" >&2
    tail -n 30 "$log" >&2
    exit 1
fi
