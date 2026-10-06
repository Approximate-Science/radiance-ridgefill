#!/bin/sh
# stop.sh -- stop every radiance-ridgefill-* container and wait until the cards let go of their
# VRAM, so the next measurement starts from a clean machine (preflight.sh (b) would
# otherwise fail it, and a number taken next to a half-released card is not a number).
#
# USAGE: scripts/stop.sh     (no arguments; stops and removes EVERY radiance-ridgefill-* container)
#
# Env vars:
#   RK_STOP_TIMEOUT   default 300 s: how long to wait for VRAM to drop
#   RK_STOP_VRAM_MIB  default 1024: the per-card ceiling, the same threshold preflight.sh
#                     applies when no radiance-ridgefill container runs

set -eu

. "$(dirname "$0")/common.sh"
: "${RK_STOP_TIMEOUT:=300}"
: "${RK_STOP_VRAM_MIB:=1024}"

if ! stopped=$(docker ps -aq --filter name=radiance-ridgefill- --format '{{.Names}}' 2>/dev/null); then
    rk_die "docker ps failed: is the docker daemon running?"
fi
if [ -z "$stopped" ]; then
    printf 'stop: no radiance-ridgefill-* container to stop\n'
else
    for c in $stopped; do
        # SIGTERM first with a 60 s grace period (docs/DOCKER.md: the engine finishes its
        # step and releases the cards on SIGTERM; compose/common.yaml uses the same 60 s)
        printf 'stop: stopping %s (SIGTERM, up to 60 s to finish its step)\n' "$c"
        docker stop -t 60 "$c" >/dev/null
        docker rm "$c" >/dev/null
    done
fi

# VRAM must fall below the ceiling on every discrete card (mem_info_vram_total > 8 GiB);
# the kernel driver releases the allocation while the container's teardown runs.
limit_bytes=$((RK_STOP_VRAM_MIB * 1048576))

vram_under_limit() {
    for dev in /sys/class/drm/card*/device; do
        [ -r "$dev/mem_info_vram_total" ] || continue
        total=$(cat "$dev/mem_info_vram_total")
        [ "$total" -gt 8589934592 ] || continue        # discrete only: > 8 GiB
        used=$(cat "$dev/mem_info_vram_used")
        if [ "$used" -gt "$limit_bytes" ]; then return 1; fi
    done
    return 0
}

waited=0
until vram_under_limit; do
    if [ "$waited" -ge "$RK_STOP_TIMEOUT" ]; then
        for dev in /sys/class/drm/card*/device; do
            [ -r "$dev/mem_info_vram_total" ] || continue
            total=$(cat "$dev/mem_info_vram_total")
            [ "$total" -gt 8589934592 ] || continue
            pci=$(basename "$(readlink -f "$dev")")
            used_mib=$(awk -v b="$(cat "$dev/mem_info_vram_used")" \
                'BEGIN { printf "%.0f", b / 1048576 }')
            printf 'stop: %s still holds %s MiB\n' "$pci" "$used_mib"
        done
        rk_die "VRAM did not drop below ${RK_STOP_VRAM_MIB} MiB within ${RK_STOP_TIMEOUT}s"
    fi
    sleep 2
    waited=$((waited + 2))
    if [ $((waited % 30)) -eq 0 ]; then
        printf 'stop: still waiting after %ss\n' "$waited"
    fi
done
printf 'stop: VRAM below %s MiB on every card after %ss\n' "$RK_STOP_VRAM_MIB" "$waited"