#!/bin/sh
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# tp1_check.sh -- the published model at TENSOR-PARALLEL 1 on ONE card: does it fit, is `off` stock, and how
# close is int8 quality to exact there. One session; run it under the GPU queue:
#   RK_PLUGIN_HOME=data/home-<short> gpuq.sh tp1 scripts/tp1_check.sh [<pci address, default 0000:13:00.0>]
#
#   0. The card pinned: ROCR_VISIBLE_DEVICES = scripts/card_index.sh <pci>, passed into every container
#      (RK_DOCKER_EXTRA), and the engine's own device line must say it uses ONE card.
#   1. FIT + TP1 R3: the stock engine (`exact`) at --tp 1 with scripts/common.sh's RK_FLAGS (expert_tiered, fp8
#      KV, 2048-token steps, 49152 context) and three memory flags changed for one card (Dylan, 10-06):
#        --host-pool-mib $RK_TP1_HOST_POOL_MIB (default 40960): the routed experts the card cannot hold live in
#          pinned host RAM. At TP1 they are ~58 GiB; the 10-05 attempt's card held 15.7 GiB of them, so ~43.6 GiB
#          is off-card, and 1.1.1's card keeps what the pool cannot (no --expert-vs-cache-ratio any more).
#          Refused here, by name, above the driver's pinned cap (ttm pages_limit) or above MemAvailable less 8 GiB.
#        --ngram-placement disk: the 47.68 GiB n-gram table read from the model file, so RAM goes to experts.
#        --gpu-headroom-mib $RK_TP1_HEADROOM_MIB: 96 (the profile's own) on a card with no connected display,
#          3072 (common.sh's display-card value) on one that drives a display -- read from the card's DRM
#          connectors, not assumed.
#      Radiance computes the card's budget at startup and refuses a shortfall by name; if it does, that reason
#      is printed with the budget it computed, and the session STOPS (exit 3). Otherwise scripts/ident.sh
#      records the TP1 R3 (stock's own six hashes at TP1).
#   2. `off` with the plugin home at --tp 1: ident must equal that TP1 R3.
#   3. A TP1 exact KL reference, recorded in this session (grade.sh record): the TP2 reference is not a TP1
#      one -- the all-reduce sums partials in another order, so TP1's exact logits differ from TP2's in the
#      low bits, and a candidate is scored against the exact engine it runs beside.
#   4. int8 quality at T2560 against it (67 approximate steps), scored last-512 paired vs exact
#      (scripts/kl_tail.py, R100's protocol), beside TP2's (G13: +0.00103 [-0.01348, +0.01529]).
# Logs: evidence/tpx/tp1-<UTC>/session.log. Exit 0 all green, 1 a check failed, 3 does not fit, 4 refused to
# start (a serving container is running).
set -u

. "$(dirname "$0")/common.sh"

pci=${1:-0000:13:00.0}
[ -n "${RK_PLUGIN_HOME:-}" ] && [ -f "$RK_PLUGIN_HOME/architectures/qwen4exp_fp8.so" ] ||
    rk_die "RK_PLUGIN_HOME must be a frozen home (scripts/frozen_home.sh <commit>): '${RK_PLUGIN_HOME:-}'"
export RK_PLUGIN_HOME RK_MODEL=${RK_MODEL:-/var/home/dylan/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad}
E=$RK_REPO/evidence/tpx/tp1-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$E"
D=$(readlink -f "$RK_REPO/data")
C=$RK_REPO/corpus/quick9.jsonl
REF=$D/kld/ref-tp1-$(date -u +%Y%m%d)
PY=/var/home/dylan/projects/research/kva/.venv/bin/python; [ -x "$PY" ] || PY=python3
T0=$(date '+%Y-%m-%d %H:%M:%S')
log() { echo "$*" | tee -a "$E/session.log"; }
klog() { if journalctl -k --since "$T0" | grep -qiE 'amdgpu.*(MES|SMU|timeout|reset)'; then log "STOP: kernel log"; exit 3; fi; }
fin() { docker logs "radiance-ridgefill-$2" > "$E/$1.serve.log" 2>&1; "$RK_SCRIPTS/stop.sh" > /dev/null 2>&1; klog; }
srv=$(docker ps --format '{{.Names}} {{.Image}}' | awk '$2 !~ /build/ {print $1}' | tr '\n' ' ')
[ -z "$srv" ] || { log "STOP: a serving container is running: $srv"; exit 4; }

idx=$("$RK_SCRIPTS/card_index.sh" "$pci") || { log "STOP: no card at $pci"; exit 1; }
RK_FLAGS=$(printf '%s' "$RK_FLAGS" | sed 's/--tp [0-9]*/--tp 1/')
case " $RK_FLAGS " in *" --tp 1 "*) ;; *) log "STOP: RK_FLAGS has no --tp to set: $RK_FLAGS"; exit 1 ;; esac

# memory flags for one card (header, step 1)
displays=$(cat /sys/bus/pci/devices/"$pci"/drm/card*/card*-*/status 2>/dev/null | grep -c '^connected$')
[ "$displays" -gt 0 ] && hr_default=3072 || hr_default=96
: "${RK_TP1_HOST_POOL_MIB:=40960}" "${RK_TP1_HEADROOM_MIB:=$hr_default}"
avail_mib=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
[ "$RK_TP1_HOST_POOL_MIB" -le $((avail_mib - 8192)) ] ||
    { log "STOP: --host-pool-mib $RK_TP1_HOST_POOL_MIB exceeds MemAvailable $avail_mib MiB less 8 GiB"; exit 1; }
if [ -r /sys/module/ttm/parameters/pages_limit ]; then
    ttm_mib=$(( $(cat /sys/module/ttm/parameters/pages_limit) * $(getconf PAGESIZE) / 1048576 ))
    [ "$RK_TP1_HOST_POOL_MIB" -le "$ttm_mib" ] ||
        { log "STOP: --host-pool-mib $RK_TP1_HOST_POOL_MIB exceeds the driver's pinned cap (ttm pages_limit) $ttm_mib MiB"; exit 1; }
fi
RK_FLAGS=$(printf '%s' "$RK_FLAGS" | sed -e "s/--host-pool-mib [0-9]*/--host-pool-mib $RK_TP1_HOST_POOL_MIB/" \
    -e "s/--gpu-headroom-mib [0-9]*/--gpu-headroom-mib $RK_TP1_HEADROOM_MIB/" -e 's/ *--ngram-placement [a-z]*//')
RK_FLAGS="$RK_FLAGS --ngram-placement disk"
for want in "--host-pool-mib $RK_TP1_HOST_POOL_MIB" "--gpu-headroom-mib $RK_TP1_HEADROOM_MIB"; do
    case " $RK_FLAGS " in *" $want "*) ;; *) log "STOP: RK_FLAGS has no '$want': $RK_FLAGS"; exit 1 ;; esac
done
export RK_FLAGS RK_DOCKER_EXTRA="-e ROCR_VISIBLE_DEVICES=$idx"
I8="$RK_DOCKER_EXTRA -v $D/projector-ridgefill-qwen38fn-int8:/projector:ro"
log "tp1 start $(date -u +%FT%TZ) boot $(cat /proc/sys/kernel/random/boot_id) card $pci = ROCR_VISIBLE_DEVICES $idx"
log "  image $RK_IMAGE; home $RK_PLUGIN_HOME $(sha256sum "$RK_PLUGIN_HOME"/*/*.so | cut -c1-16 | tr '\n' ' ')"
log "  flags: $RK_FLAGS"
rc=0

# 1. fit + the TP1 R3
if ! "$RK_SCRIPTS/serve.sh" exact > "$E/exact.serve.out" 2>&1; then
    fin exact exact
    log "DOES NOT FIT (or did not start) at --tp 1 on one card: $(tail -3 "$E/exact.serve.out" | tr '\n' '|')"
    sed -n '/VRAM budget, per card/,/unclaimed/p' "$E/exact.serve.log" | sed 's/^/    /' | tee -a "$E/session.log"
    grep -iE 'refus|cannot|does not fit|short|exceed|too large' "$E/exact.serve.log" | head -8 | sed 's/^/    /' | tee -a "$E/session.log"
    exit 3
fi
"$RK_SCRIPTS/ident.sh" > "$E/tp1-R3.ident" 2>&1 || rc=1
fin exact exact
log "  device: $(grep -m1 -E '^. device: ' "$E/exact.serve.log" | cut -c1-200)"
grep -qE 'device: AQL backend, 1 device' "$E/exact.serve.log" ||
    { log "STOP: the engine did not run on exactly one device (see $E/exact.serve.log)"; exit 1; }
log "FITS at --tp 1 on one card -- the budget the engine computed:"
sed -n '/VRAM budget, per card/,/unclaimed/p' "$E/exact.serve.log" | sed 's/^/    /' | tee -a "$E/session.log"
grep -E 'mover: .*slab slots|placement: (token embedding|n-gram table)' "$E/exact.serve.log" | head -4 | cut -c1-200 | sed 's/^/    /' | tee -a "$E/session.log"
log "  TP1 R3 recorded: $E/tp1-R3.ident"; sed 's/^/    /' "$E/tp1-R3.ident" | tee -a "$E/session.log"

# 2. off at TP1
"$RK_SCRIPTS/serve.sh" off > "$E/off.serve.out" 2>&1 && "$RK_SCRIPTS/ident.sh" > "$E/off.ident" 2>&1; fin off off
if [ -s "$E/tp1-R3.ident" ] && diff -q "$E/off.ident" "$E/tp1-R3.ident" > /dev/null; then log "off at TP1: ident EQUALS the TP1 R3"
else log "off at TP1: ident DIFFERS from the TP1 R3 (or a serve failed)"; diff "$E/off.ident" "$E/tp1-R3.ident" | head -8 | tee -a "$E/session.log"; rc=1; fi

# 3. the TP1 exact reference, this session
if [ -d "$REF" ]; then log "STOP: $REF exists (one reference a day; move it to rerun)"; exit 1; fi
if ! "$RK_SCRIPTS/grade.sh" record "$REF" "$C" > "$E/ref.out" 2>&1; then
    log "TP1 reference FAILED: $(tail -2 "$E/ref.out" | tr '\n' '|')"; exit 1; fi
log "TP1 exact reference: $REF ($(tail -1 "$E/ref.out"))"
klog

# 4. int8 quality T2560 against it
env RK_DOCKER_EXTRA="$I8" RADIANCE_RIDGEFILL_TAIL=2560 RADIANCE_RIDGEFILL_PROJECTOR=/projector RADIANCE_RIDGEFILL_FINAL=off \
    RADIANCE_RIDGEFILL_SCORE_BULK=1 RK_EXPECT_APPROX=67 "$RK_SCRIPTS/grade.sh" quality "$REF" "$E/i8-quality-t2560.json" \
    > "$E/quality.out" 2>&1 || rc=1
log "int8 quality T2560 at TP1: $(tail -1 "$E/quality.out")"
log "    $(grep -E 'holds the projector|REFUSED|forwarding' "$E/i8-quality-t2560.json.log" | sort -u | cut -c1-160 | tr '\n' '|')"
"$PY" "$RK_REPO/scripts/kl_tail.py" --corpus "$C" --last 512 "$E/i8-quality-t2560.json" 2>&1 | grep -v '^    ppl/' | tee -a "$E/session.log"
log "    TP2 (G13, the same protocol): dNLL +0.00103 [-0.01348, +0.01529], KL 0.0368, top-1 0.9128"
klog
log "tp1 end $(date -u +%FT%TZ) rc $rc"
exit $rc
