#!/bin/sh
# release_session.sh -- THE FINAL RELEASE SESSION: one GPU session, from one commit of main.
#
#   gpuq.sh release env RK_RELEASE_VERSION=0.2.0 sh scripts/release_session.sh   (from a checkout of main's HEAD)
#
# Every server runs radiance's DEFAULT flags, as a user would: RK_FLAGS="--tp 2", no --max-num-seqs (the scripts'
# `default`), so MTP is auto, the prefix cache is on, steps are 8,192 tokens, the context is the model's. Only the
# R64 servers add --num-speculative-tokens 0 --profile-ops (the logits capture follows one greedy decoder and must
# see every pass).
#   0  frozen home of HEAD (scripts/frozen_home.sh: the build runs the host tests)
#   1  packages (tools/package.py: plugin + the int8 projector, no template) into $RK_RELEASE_DIST; SHA256SUMS of the
#      tarballs and of each extracted directory verified
#   2  the fresh-engine gate (scripts/e2e_fresh.sh): stock image, published model (sha256 checked), clean
#      RADIANCE_HOME = the extracted plugin then the image's own home, the extracted projector at <model dir>/projector
#   3  headline numbers (the S4 protocol): stock / quality / speed, two rounds interleaved, each server warmed with
#      settle.py (3 cycles at 2K/16K/32K) then speed.sh RK_REPS=7 at 16K and 32K; decode at C = 1 with MTP on
#      (tools/mtp_accept.py: 16K prompt, 256 greedy tokens, 3 reps) on stock and quality
#   4  needle (tools/needle.py): one corpus, built once, run on round a's three servers; compare with McNemar
#   5  R64 redefined (tools/turn2.py): turn 2 with the prefix cache vs --no-prefix-cache vs exact (mode off),
#      logits captured over HTTP (RADIANCE_KVA_DUMP_LOGITS), compared by tools/logit_compare.py
# The plugin and projector every server uses are the EXTRACTED PACKAGES (what a user installs), the projector found by
# discovery beside the model (no RADIANCE_KVA_PROJECTOR).
set -u
W=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$W" || exit 1
D=$(readlink -f "$W/data")
COMMIT=$(git rev-parse HEAD) SHORT=$(git rev-parse --short HEAD)
E=$W/evidence/release; mkdir -p "$E"
: "${RK_RELEASE_VERSION:=0.2.0}"
: "${RK_RELEASE_DIST:=/var/home/dylan/AI-Work/radiance-kva-plugin-20261004/dist}"
export RK_MODEL=/var/home/dylan/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad
export RK_DOCS=/var/home/dylan/AI-Work/kva-flashnext-tests-data/samples/quick/ppl.jsonl
export RK_FLAGS="--tp 2" RK_SERVE_SEQS=default RK_E2E_SEQS=default RK_STAGE=release
PY=/var/home/dylan/projects/research/kva/.venv/bin/python
T0=$(date '+%Y-%m-%d %H:%M:%S')
log() { echo "$*" | tee -a "$E/session.log"; }
klog() { if journalctl -k --since "$T0" | grep -qiE 'amdgpu.*(MES|SMU|timeout|reset)'; then log "STOP: kernel log"; exit 3; fi; }
quiet() {
  n=0; while :; do l=$(cut -d' ' -f1 /proc/loadavg)
    busy=$(pgrep -xc 'cc1|cc1plus|g\+\+|c\+\+|ld|ninja|hipcc|clang|clang\+\+|clang-[0-9]+|kernel_test|arch_static_tes' || true)
    if awk -v l="$l" 'BEGIN{exit !(l < 2.5)}' && [ "$busy" = 0 ]; then break; fi
    [ $n -ge 60 ] && break; sleep 10; n=$((n + 1)); done
  echo "load $(cut -d' ' -f1 /proc/loadavg) busy $busy waited $((n * 10))s"
}
log "RELEASE start $(date -u +%FT%TZ) boot $(cat /proc/sys/kernel/random/boot_id) commit $COMMIT version $RK_RELEASE_VERSION"

# 0 + 1 ------------------------------------------------------------------------------------------------
RK_RADIANCE_SRC=/var/home/dylan/projects/inference/radiance scripts/frozen_home.sh "$SHORT" > "$E/frozen_home.out" 2>&1 ||
  { log "frozen home FAILED: $(grep -E 'FAIL|error' "$E/frozen_home.out" | head -5 | tr '\n' '|')"; exit 2; }
log "home $SHORT: $(grep 'tests passed' "$E/frozen_home.out")"
python3 tools/package.py --home "$D/home-$SHORT" --projector "$D/projector-qwen38fn-int8" --out "$RK_RELEASE_DIST" \
  --version "$RK_RELEASE_VERSION" --commit "$COMMIT" --radiance-version 1.0.8 > "$E/package.out" 2>&1 ||
  { log "package FAILED: $(tail -3 "$E/package.out" | tr '\n' '|')"; exit 2; }
(cd "$RK_RELEASE_DIST" && sha256sum -c SHA256SUMS) > "$E/dist-sums.txt" 2>&1 || { log "dist SHA256SUMS FAILED"; exit 2; }
X=$E/extract; rm -rf "$X"; mkdir -p "$X"
for t in "$RK_RELEASE_DIST"/*.tar.gz; do tar -xzf "$t" -C "$X"; done
for d in "$X"/*/; do (cd "$d" && sha256sum -c SHA256SUMS) >> "$E/dist-sums.txt" 2>&1 || { log "SHA256SUMS FAILED in $d"; exit 2; }; done
log "dist: $(cd "$RK_RELEASE_DIST" && ls -1 | tr '\n' ' '); every SHA256SUMS verified ($(grep -c ': OK$' "$E/dist-sums.txt") files)"
cat "$RK_RELEASE_DIST/SHA256SUMS" | tee -a "$E/session.log"
PLUGIN=$(ls -d "$X"/radiance-kva-*/ | head -1); PROJ=$(ls -d "$X"/projector-qwen3.8-flash-next-*/ | head -1)
export RK_PLUGIN_HOME=${PLUGIN%/}
MOUNT="-v ${PROJ%/}:/models/projector:ro"   # the extracted projector beside the model, as a user lays it out

# 2 ----------------------------------------------------------------------------------------------------
RK_DIST=$RK_RELEASE_DIST RK_E2E_WORK=$E/e2e scripts/e2e_fresh.sh > "$E/e2e.out" 2>&1; rc=$?
log "e2e exit $rc: $(grep -E '^e2e: case' "$E/e2e.out" | sed 's/^e2e: //' | tr '\n' '|')"
klog

# 3 + 4 ------------------------------------------------------------------------------------------------
N=$E/needle; mkdir -p "$N"
needle_build() {   # three seeds = three sets of distinct keys and numbers; ids prefixed by the seed
  for s in 1 2 3; do
    python3 tools/needle.py build --docs "$RK_DOCS" --lengths 16384,32768 --depths 0.10,0.30,0.50,0.70,0.85,0.98 \
      --keys 4 --seed $s --out "$N/corpus-s$s.jsonl" > "$N/build-s$s.out" 2>&1 || { log "needle build FAILED (seed $s)"; return 1; }
  done
  python3 - "$N" <<'EOF'
import json, sys
n = sys.argv[1]
with open(f"{n}/corpus.jsonl", "w") as out:
    for s in (1, 2, 3):
        for line in open(f"{n}/corpus-s{s}.jsonl"):
            item = json.loads(line); item["id"] = f"S{s}/{item['id']}"; out.write(json.dumps(item) + "\n")
EOF
  log "needle corpus: $(wc -l < "$N/corpus.jsonl") items (16K/32K, depths 0.10-0.85 in the approximated bulk + 0.98 tail, single + 4-key, seeds 1-3)"
}
arm() {   # label mode: serve, warm, time, decode (stock/quality), needle (round a)
  label=$1 mode=$2; r=${label##*-}
  q=$(quiet)
  if env RK_DOCKER_EXTRA="$MOUNT" scripts/serve.sh "$mode" > "$E/serve-$label.out" 2>&1; then
    [ "$label" = exact-a ] && needle_build
    python3 tools/settle.py --out "$E/warm-$label.json" --docs "$RK_DOCS" --cycles 3 --lengths "2048 16384 32768" > "$E/warm-$label.txt" 2>&1
    RK_LENGTHS="16384 32768" RK_REPS=7 scripts/speed.sh "$label" > "$E/speed-$label.out" 2>&1 || log "$label: speed.sh FAILED"
    [ "$mode" = speed ] || python3 tools/mtp_accept.py --docs "$RK_DOCS" --reps 3 --out "$E/decode-$label.json" > "$E/decode-$label.txt" 2>&1
    [ "$r" = a ] && python3 tools/needle.py run --corpus "$N/corpus.jsonl" --out "$N/results-$mode.jsonl" > "$N/run-$mode.out" 2>&1
    docker logs "radiance-kva-$mode" > "$E/$label.serve.log" 2>&1
    log "== $label ($mode; $q) $(grep -m1 -oE 'I\[0\] mover: [0-9]+ slab slots' "$E/$label.serve.log")"
    sed 's/^/    warm /' "$E/warm-$label.txt" | tee -a "$E/session.log"
    log "    $(grep -E 'median' "$E/speed-$label.out" | tr -s ' ' | tr '\n' '|')"
    [ -f "$E/decode-$label.txt" ] && log "    decode: $(tail -1 "$E/decode-$label.txt"); $(grep -o 'decode [0-9.]* ms/tok' "$E/decode-$label.txt" | tr '\n' ' ')"
  else log "== $label: serve FAILED -- $(tail -2 "$E/serve-$label.out" | tr '\n' '|')"
       docker logs "radiance-kva-$mode" > "$E/$label.serve.log" 2>&1; fi
  scripts/stop.sh > /dev/null 2>&1; klog
}
for r in a b; do arm exact-$r exact; arm quality-$r quality; arm speed-$r speed; done
for m in quality speed; do
  log "== needle stock vs $m"; python3 tools/needle.py compare "$N/results-exact.jsonl" "$N/results-$m.jsonl" 2>&1 | tee "$N/compare-$m.txt" | head -40 | tee -a "$E/session.log"
done

# 5 ----------------------------------------------------------------------------------------------------
R=$E/r64; mkdir -p "$R"
r64() {   # label mode extra-flags
  label=$1 mode=$2 extra=$3
  mkdir -p "$R/dump-$label"
  if env RK_FLAGS="--tp 2 --num-speculative-tokens 0 $extra" RK_DOCKER_EXTRA="$MOUNT -v $R/dump-$label:/dump" \
       RADIANCE_KVA_DUMP_LOGITS=/dump scripts/serve.sh "$mode" --profile-ops > "$R/serve-$label.out" 2>&1; then
    [ -f "$R/conv.json" ] || python3 tools/turn2.py build --docs "$RK_DOCS" --out "$R/conv.json" > "$R/build.out" 2>&1
    python3 tools/turn2.py run --conv "$R/conv.json" --dump "$R/dump-$label" --out "$R/manifest-$label.json" > "$R/run-$label.out" 2>&1
    log "R64 $label: $(tr '\n' '|' < "$R/run-$label.out")"
  else log "R64 $label: serve FAILED"; fi
  docker logs "radiance-kva-$mode" > "$R/$label.serve.log" 2>&1; scripts/stop.sh > /dev/null 2>&1; klog
}
r64 exact off ""
r64 cache quality ""
r64 nocache quality "--no-prefix-cache"
for i in 0 1 2 3 4; do
  for c in cache nocache; do
    printf 'conv %s %s vs exact: ' $i $c | tee -a "$E/session.log"
    $PY tools/logit_compare.py "$R/dump-exact:$R/manifest-exact.json:solo$i:0" "$R/dump-$c:$R/manifest-$c.json:solo$i:0" 2>&1 | tee -a "$E/session.log"
  done
done
log "klog after: $(journalctl -k --since "$T0" | grep -ciE 'amdgpu.*(MES|SMU|timeout|reset)')"
log "RELEASE end $(date -u +%FT%TZ)"
