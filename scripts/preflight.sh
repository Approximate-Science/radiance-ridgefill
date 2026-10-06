#!/bin/sh
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# preflight.sh -- fail closed before any GPU measurement: a number taken on a busy or
# unhealthy machine is not a measurement. scripts/speed.sh runs it before every sample,
# scripts/serve.sh and scripts/grade.sh before each container.
#
# Exits non-zero with a named reason if:
#   (a) any process other than this repo's own tooling and the engines inside the
#       radiance-ridgefill-* containers matches radiance|llama|vllm|r9v AND holds /dev/kfd open (pgrep -af; the check's
#       own process tree is excluded -- the script, its launcher, every ancestor: a
#       squatter is by definition not in it -- and the report is filtered with awk,
#       not grep, so no grep process ever matches);
#   (b) no radiance-ridgefill container runs and any discrete GPU (mem_info_vram_total > 8 GiB)
#       holds more than RK_PREFLIGHT_VRAM_MIB MiB of VRAM;
#   (c) journalctl -k --since "$RK_PREFLIGHT_SINCE" has lines matching
#       amdgpu.*(MES|SMU|ring.*timeout|GPU reset) -- a MES or SMU error invalidates every
#       result after it (HANDOVER §3).
#
# Prints one provenance line first (UTC time, then per card: PCI id, VRAM used,
# temperature, current sclk), and "preflight: OK" on success; the failure reason goes to
# stderr. The caller keeps the stdout lines in the evidence file.
#
# Env vars:
#   RK_PREFLIGHT_SINCE     default -30min: the journalctl kernel-log window
#   RK_PREFLIGHT_VRAM_MIB  default 1024: the per-card VRAM ceiling for check (b)

set -u

. "$(dirname "$0")/common.sh"
: "${RK_PREFLIGHT_SINCE:=-30min}"
: "${RK_PREFLIGHT_VRAM_MIB:=1024}"

fail() {
    printf 'preflight: FAIL: %s\n' "$*" >&2
    exit 1
}

tmp_klog=$(mktemp) || fail "mktemp failed"
tmp_squat=$(mktemp) || fail "mktemp failed"
trap 'rm -f "$tmp_klog" "$tmp_squat"' EXIT INT TERM

# ---- the provenance line: UTC time, per card PCI id + VRAM used + temperature + sclk.
# The card's clock is part of the measurement (scripts/fnpf.sh's rule): prefill drifts
# down as the part heats, so a thermostat reads as a regression unless the temperature
# and sclk are recorded beside the number.
provenance=$(date -u +%Y-%m-%dT%H:%M:%SZ)
n_cards=0
for dev in /sys/class/drm/card*/device; do
    [ -r "$dev/mem_info_vram_total" ] || continue
    total=$(cat "$dev/mem_info_vram_total") || fail "cannot read $dev/mem_info_vram_total"
    [ "$total" -gt 8589934592 ] || continue             # discrete only: > 8 GiB
    n_cards=$((n_cards + 1))
    pci=$(basename "$(readlink -f "$dev")")
    used_mib=$(awk -v b="$(cat "$dev/mem_info_vram_used")" 'BEGIN { printf "%.0f", b / 1048576 }')
    temp=''
    for h in "$dev"/hwmon/hwmon*/temp1_input; do
        [ -r "$h" ] || continue
        temp=$(awk -v m="$(cat "$h")" 'BEGIN { printf "%.1f", m / 1000 }')
        break
    done
    sclk=''
    if [ -r "$dev/pp_dpm_sclk" ]; then
        # the line with '*' is the current clock; print its value (e.g. 2900M)
        sclk=$(awk '/\*/ { for (i = 1; i <= NF; i++) if ($i ~ /M/) { print $i; exit } }' \
            "$dev/pp_dpm_sclk")
    fi
    provenance="$provenance  $pci vram=${used_mib}MiB temp=${temp:-?}C sclk=${sclk:-?}"
done
printf '%s\n' "$provenance"
[ "$n_cards" -gt 0 ] || fail "no discrete GPU (mem_info_vram_total > 8 GiB) under /sys/class/drm"

# ---- (a) squatters: any radiance|llama|vllm|r9v process that is not ours.
# Ours = this check's own process tree (the script, its launcher and every ancestor:
# the operator's shell, an agent or CI wrapper that deliberately ran this check -- a
# squatter is by definition NOT in it), this repo's own tooling (command lines
# containing the repo path) and the processes inside the radiance-ridgefill-* containers.
# A process whose command line merely QUOTES the word (a wrapper embedding the pattern)
# is not an engine either; no real radiance/llama/vllm/r9v carries a "|" in its name.
ours=' '
pid=$$
while [ "$pid" != 1 ] && [ -r "/proc/$pid/status" ]; do
    ppid=$(awk '/^PPid:/ { print $2 }' /proc/$pid/status 2>/dev/null)
    [ -n "$ppid" ] || break
    ours="$ours$ppid "
    pid=$ppid
done
containers=$(docker ps --filter name=radiance-ridgefill- --format '{{.Names}}' 2>/dev/null) \
    || fail "docker ps failed: cannot tell which radiance-ridgefill containers are ours"
for c in $containers; do
    ours="$ours$(docker top "$c" -eo pid 2>/dev/null | awk 'NR > 1 { printf "%s ", $1 }')"
done
if pgrep -af 'radiance|llama|vllm|r9v' > "$tmp_squat" 2>/dev/null; then
    bad=$(awk -v me="$$" -v repo="$RK_REPO" -v ours="$ours" \
        -v pattern='radiance|llama|vllm|r9v' '
        {
            pid = $1
            rest = $0; sub(/^[0-9]+[ \t]*/, "", rest)
            if (pid == me) next
            if (repo != "" && index(rest, repo) != 0) next
            if (index(ours, " " pid " ") != 0) next
            # a process that merely QUOTES the check pattern (a wrapper like
            # `bash -c "scripts/preflight.sh"` or an agent harness) is not an engine;
            # no real radiance/llama/vllm/r9v process carries a "|" in its name
            if (index(rest, pattern) != 0) next
            print
        }' "$tmp_squat")
    # A name match is a squatter only if it holds the GPU (/dev/kfd open). A port forwarder
    # or a log tailer named after an engine is not; a LIVE process whose fds we cannot read is
    # counted (fail closed). One that exited between pgrep and this check is not: a gone
    # process holds no GPU, and counting it aborted samples on every short-lived ls or grep
    # whose argv merely named a path containing the pattern (Stage E S3, 2026-10-05).
    bad=$(printf '%s\n' "$bad" | while read -r spid srest; do
        [ -n "$spid" ] || continue
        if ls -l "/proc/$spid/fd" 2>/dev/null | grep -q '/dev/kfd'; then
            printf '%s %s\n' "$spid" "$srest"
        elif ! ls "/proc/$spid/fd" >/dev/null 2>&1 && [ -d "/proc/$spid" ]; then
            printf '%s %s\n' "$spid" "$srest"
        fi
    done)
    if [ -n "$bad" ]; then
        printf '%s\n' "$bad" >&2
        fail "a process outside this repo matches radiance|llama|vllm|r9v (lines above)"
    fi
fi

# ---- (b) leftover VRAM when no radiance-ridgefill container runs.
if [ -z "$containers" ]; then
    limit_bytes=$((RK_PREFLIGHT_VRAM_MIB * 1048576))
    for dev in /sys/class/drm/card*/device; do
        [ -r "$dev/mem_info_vram_total" ] || continue
        total=$(cat "$dev/mem_info_vram_total")
        [ "$total" -gt 8589934592 ] || continue
        used=$(cat "$dev/mem_info_vram_used")
        if [ "$used" -gt "$limit_bytes" ]; then
            pci=$(basename "$(readlink -f "$dev")")
            used_mib=$(awk -v b="$used" 'BEGIN { printf "%.0f", b / 1048576 }')
            fail "$pci holds ${used_mib} MiB of VRAM with no radiance-ridgefill container running (ceiling ${RK_PREFLIGHT_VRAM_MIB} MiB)"
        fi
    done
fi

# ---- (c) the kernel log: a MES/SMU error invalidates everything measured after it.
if ! journalctl -k --since "$RK_PREFLIGHT_SINCE" > "$tmp_klog" 2>/dev/null; then
    fail "journalctl -k --since $RK_PREFLIGHT_SINCE is unreadable; the amdgpu health window cannot be checked"
fi
if grep -iE 'amdgpu.*(MES|SMU|ring.*timeout|GPU reset)' "$tmp_klog" >/dev/null 2>&1; then
    grep -iE 'amdgpu.*(MES|SMU|ring.*timeout|GPU reset)' "$tmp_klog" | head -5 >&2
    fail "the kernel log since $RK_PREFLIGHT_SINCE holds amdgpu MES/SMU/timeout/reset lines (above); a MES or SMU error invalidates every result after it"
fi

# ---- (d) GPU page faults: RECORDED, not refused. radiance's release profile page-faults the card as a server
# stops after long prompts -- stock too: on 2026-10-06 all 14 fault bursts fell within 3 s of a container stop
# (e2e cases 0/3/4, ramp-stock, betterbench probes; gate A2 notes). Refusing would stop every session after the
# first such server; the count and the last fault's time go into the evidence instead. A fault whose time falls
# INSIDE a measurement invalidates that measurement.
faults=$(grep -ciE 'amdgpu.*(page fault|PROTECTION_FAULT)' "$tmp_klog")
last=$(grep -iE 'amdgpu.*(page fault|PROTECTION_FAULT)' "$tmp_klog" | tail -1 | awk '{print $3}')
printf 'preflight: %s amdgpu page-fault line(s) since %s%s\n' "$faults" "$RK_PREFLIGHT_SINCE" "${last:+, the last at $last}"

printf 'preflight: OK\n'