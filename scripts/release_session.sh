#!/bin/sh
# release_session.sh -- THE FINAL RELEASE SESSION: one GPU session, from one commit of main.
#
#   gpuq.sh release env RK_RELEASE_VERSION=0.1.0 sh scripts/release_session.sh   (from a checkout of main's HEAD)
#
# A VERSION CHANGE RE-RUNS ONLY THE VERSION-DEPENDENT PART (the measurements never read the version string):
#   gpuq.sh release-pkg env RK_RELEASE_PARTS="package e2e" RK_RELEASE_VERSION=<x.y.z> RK_RELEASE_DIST=<fresh dir> \
#       sh scripts/release_session.sh
# RK_RELEASE_PARTS (default "package e2e headline needle r64") picks the parts (needle rides on the headline's
# round-RK_RELEASE_NEEDLE_ROUND servers); the frozen home of RK_RELEASE_COMMIT (default HEAD)
# is reused when it exists; extraction and the e2e work dir are per version (evidence/release/{extract,e2e}-<version>).
#
# Every server runs radiance 1.0.13's shipped flashnext profile (RK_RELEASE_FLAGS, below: MTP 3, prefix cache on with
# host and disk tiers, 2,048-token steps, wht6 wire, 8 sequences), headroom 3,072 MiB instead of 96. Only the
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
#      logits captured over HTTP (RADIANCE_RIDGEFILL_DUMP_LOGITS), compared by tools/logit_compare.py
# The plugin and projector every server uses are the EXTRACTED PACKAGES (what a user installs), the projector found by
# discovery beside the model (no RADIANCE_RIDGEFILL_PROJECTOR).
set -u
W=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$W" || exit 1
D=$(readlink -f "$W/data")
COMMIT=$(git rev-parse "${RK_RELEASE_COMMIT:-HEAD}") SHORT=$(git rev-parse --short "${RK_RELEASE_COMMIT:-HEAD}")
: "${RK_RELEASE_PARTS:=package e2e headline needle r64}"
has() { case " $RK_RELEASE_PARTS " in *" $1 "*) return 0 ;; esac; return 1; }
E=$W/evidence/release; mkdir -p "$E"
: "${RK_RELEASE_VERSION:=0.1.0}"
# the radiance release the plugin is built against and served on (1.0.13 since the rebase, Dylan 2026-10-05):
: "${RK_RADIANCE_VERSION:=1.0.13}"
: "${RK_RADIANCE_SRC:=$D/radiance-src-1.0.13}"
: "${RK_BUILD_IMAGE:=radiance-build:1.0.13}"
: "${RK_IMAGE:=stilldeadcode/radiance:$RK_RADIANCE_VERSION}"
export RK_RADIANCE_SRC RK_BUILD_IMAGE RK_IMAGE
: "${RK_RELEASE_DIST:=/var/home/dylan/AI-Work/radiance-kva-plugin-20261004/dist}"
export RK_MODEL=/var/home/dylan/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad
export RK_DOCS=/var/home/dylan/AI-Work/kva-flashnext-tests-data/samples/quick/ppl.jsonl
# THE USER'S CONFIG (orchestrator, 2026-10-05): radiance 1.0.13's own shipped profile for this model,
# deploy/compose/flashnext.yaml, with ONE deviation -- --gpu-headroom-mib 3072 instead of 96, because the display runs
# on card 0000:03:00.0 (Dylan's rule). Its prefix cache (VRAM + 4 GiB host + 128 GiB disk under --prefix-cache-dir) is
# ON: every server gets a FRESH cache dir (RK_CACHE_DIR, mounted at /kvcache by scripts/common.sh), so no server reuses
# another's KV, and every timed prompt carries a leading nonce, so no TTFT is answered from the cache. Radiance's pure
# defaults do not load this model on two 32 GB cards (--host-pool-mib 0: "DID NOT FIT", evidence/r1013/def13-a.serve.log).
# common.yaml's docker settings are mirrored by scripts/common.sh (devices, seccomp=unconfined, --init, memlock -1,
# models read-only); NOT mirrored here: network_mode host (it hides the GPUs under rootless docker: loopback -p
# instead), user 1000:1000 (rootless root already maps to the host user), group_add (the scratch image has no group
# entries; the device nodes are world-rw here), --api-key / restart / healthcheck (test runs).
: "${RK_RELEASE_FLAGS:=--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 3072 --expert-vs-cache-ratio 0.82 --kv-cache-dtype fp8 --prefix-cache-host-mib 4096 --prefix-cache-dir /kvcache --prefix-cache-disk-mib 131072 --num-speculative-tokens 3 --max-num-batched-tokens 2048}"
# the prefix cache's disk tier can fill its volume (2026-10-05: 19 GB a server filled /var/mnt/qwen-storage): the
# cache root is a parameter, one server's dir at a time, removed before the next
K=${RK_RELEASE_CACHE_ROOT:-$E/kvcache}; mkdir -p "$K"
drop_cache() {    # every server's prefix-cache dir goes as soon as the server stops (its size logged first)
  for c in "$K"/*/; do [ -d "$c" ] && { echo "  kvcache $(basename "$c"): $(du -sh "$c" | cut -f1)" >> "$E/kvcache.txt"; rm -rf "$c"; }; done
}
fresh_cache() {   # label: a fresh prefix-cache dir for the next server, the volume's free space logged first
  drop_cache
  echo "  $1: $(df -h --output=avail "$K" | tail -1 | tr -d ' ') free on the cache volume ($K)" >> "$E/kvcache.txt"
  export RK_CACHE_DIR=$K/$1; mkdir -p "$RK_CACHE_DIR"
}
export RK_FLAGS="$RK_RELEASE_FLAGS" RK_SERVE_SEQS=default RK_E2E_SEQS=default RK_STAGE=release
# scripts/common.sh's Stage C cache profile appends its own prefix-cache flags whenever RK_CACHE_DIR is set (and drops
# --no-prefix-cache); the engine takes the last value, so they must carry the profile's sizes (2026-10-05: its default
# 32,768 MiB disk tier silently replaced the profile's 131,072 -- never reached, 19 GB at most)
export RK_CACHE_HOST_MIB=4096 RK_CACHE_DISK_MIB=131072
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
log "RELEASE start $(date -u +%FT%TZ) boot $(cat /proc/sys/kernel/random/boot_id) parts '$RK_RELEASE_PARTS' commit $COMMIT version $RK_RELEASE_VERSION; radiance $RK_RADIANCE_VERSION ($RK_IMAGE, build $RK_BUILD_IMAGE, source $RK_RADIANCE_SRC)"

# 0 + 1 ------------------------------------------------------------------------------------------------
X=$E/extract-$RK_RELEASE_VERSION
if has package; then
if [ -f "$D/home-$SHORT/kernels/ridgefill.so" ] && [ -f "$E/frozen_home-$SHORT.out" ]; then
  log "home $SHORT: reused ($(grep 'tests passed' "$E/frozen_home-$SHORT.out"))"
else
  scripts/frozen_home.sh "$SHORT" > "$E/frozen_home-$SHORT.out" 2>&1 ||
    { log "frozen home FAILED: $(grep -E 'FAIL|error' "$E/frozen_home-$SHORT.out" | head -5 | tr '\n' '|')"; exit 2; }
  log "home $SHORT: $(grep 'tests passed' "$E/frozen_home-$SHORT.out")"
fi
python3 tools/package.py --home "$D/home-$SHORT" --projector "$D/projector-ridgefill-qwen38fn-int8" --out "$RK_RELEASE_DIST" \
  --version "$RK_RELEASE_VERSION" --commit "$COMMIT" --radiance-version "$RK_RADIANCE_VERSION" > "$E/package-$RK_RELEASE_VERSION.out" 2>&1 ||
  { log "package FAILED: $(tail -3 "$E/package-$RK_RELEASE_VERSION.out" | tr '\n' '|')"; exit 2; }
(cd "$RK_RELEASE_DIST" && sha256sum -c SHA256SUMS) > "$E/dist-sums.txt" 2>&1 || { log "dist SHA256SUMS FAILED"; exit 2; }
rm -rf "$X"; mkdir -p "$X"
for t in "$RK_RELEASE_DIST"/*.tar.gz; do tar -xzf "$t" -C "$X"; done
for d in "$X"/*/; do (cd "$d" && sha256sum -c SHA256SUMS) >> "$E/dist-sums.txt" 2>&1 || { log "SHA256SUMS FAILED in $d"; exit 2; }; done
log "dist: $(cd "$RK_RELEASE_DIST" && ls -1 | tr '\n' ' '); every SHA256SUMS verified ($(grep -c ': OK$' "$E/dist-sums.txt") files)"
cat "$RK_RELEASE_DIST/SHA256SUMS" | tee -a "$E/session.log"
fi
[ -d "$X" ] || { log "no extracted packages for version $RK_RELEASE_VERSION ($X): run the package part"; exit 2; }
PLUGIN=$(ls -d "$X"/radiance-ridgefill-*/ | head -1); PROJ=$(ls -d "$X"/ridgefill-projector-qwen3.8-flash-next-*/ | head -1)
export RK_PLUGIN_HOME=${PLUGIN%/}
MOUNT="-v ${PROJ%/}:/models/projector:ro"   # the extracted projector beside the model, as a user lays it out

# 2 ----------------------------------------------------------------------------------------------------
if has e2e; then
RK_DIST=$RK_RELEASE_DIST RK_E2E_WORK=$E/e2e-$RK_RELEASE_VERSION RK_E2E_CACHE_ROOT=$K/e2e scripts/e2e_fresh.sh > "$E/e2e-$RK_RELEASE_VERSION.out" 2>&1; rc=$?
log "e2e exit $rc: $(grep -E '^e2e: case' "$E/e2e-$RK_RELEASE_VERSION.out" | sed 's/^e2e: //' | tr '\n' '|')"
klog
fi

# 3 + 4 headline, and the needle on round RK_RELEASE_NEEDLE_ROUND's servers (default b) ------------------
# The needle prompts are chat prompts (the template's user turn, thinking off), max_tokens 32; stock builds the corpus
# and must pass the 6-item smoke gate at 16K (>= 5/6, the tail control right) before any full run.
N=$E/needle; mkdir -p "$N"
NEEDLE_OK=0
needle_on() {   # mode (its server is up)
  if [ "$1" = exact ]; then
    for s in 1 2 3; do
      python3 tools/needle.py build --chat --docs "$RK_DOCS" --lengths 16384,32768 --depths 0.10,0.30,0.50,0.70,0.85,0.98 \
        --keys 4 --seed $s --out "$N/corpus-s$s.jsonl" > "$N/build-s$s.out" 2>&1 || log "needle build FAILED (seed $s)"
    done
    python3 tools/needle_merge.py "$N"
    log "needle corpus: $(wc -l < "$N/corpus.jsonl") items (seeds 1-3 x 16K/32K x depths 0.10-0.85 bulk + 0.98 tail x single/4-key), $(grep -o -- '--chat: .*' "$N/build-s1.out")"
    python3 tools/needle.py run --corpus "$N/smoke.jsonl" --out "$N/smoke-exact.jsonl" --max-tokens 32 > "$N/smoke-exact.out" 2>&1
    gate=$(python3 tools/needle_merge.py --gate "$N/smoke-exact.jsonl")
    log "needle smoke (stock, 6 items at 16K): $gate"
    case "$gate" in PASS*) NEEDLE_OK=1 ;; *) log "needle: STOPPED at the smoke gate"; return 0 ;; esac
  fi
  [ "$NEEDLE_OK" = 1 ] || return 0
  python3 tools/needle.py run --corpus "$N/corpus.jsonl" --out "$N/results-$1.jsonl" --max-tokens 32 > "$N/run-$1.out" 2>&1
  log "    needle $1: $(tail -1 "$N/run-$1.out")"
}
arm() {   # label mode: serve, warm, time, decode (stock/quality), the needle on its round
  label=$1 mode=$2
  q=$(quiet); fresh_cache "$label"
  if env RK_DOCKER_EXTRA="$MOUNT" scripts/serve.sh "$mode" > "$E/serve-$label.out" 2>&1; then
    python3 tools/settle.py --out "$E/warm-$label.json" --docs "$RK_DOCS" --cycles 3 --lengths "2048 16384 32768" > "$E/warm-$label.txt" 2>&1
    RK_LENGTHS="16384 32768" RK_REPS=7 scripts/speed.sh "$label" > "$E/speed-$label.out" 2>&1 || log "$label: speed.sh FAILED"
    [ "$mode" = speed ] || python3 tools/mtp_accept.py --docs "$RK_DOCS" --reps 3 --out "$E/decode-$label.json" > "$E/decode-$label.txt" 2>&1
    log "== $label ($mode; $q) $(docker logs "radiance-ridgefill-$mode" 2>&1 | grep -m1 -oE 'I\[0\] mover: [0-9]+ slab slots')"
    sed 's/^/    warm /' "$E/warm-$label.txt" | tee -a "$E/session.log"
    log "    $(grep -E 'median' "$E/speed-$label.out" | tr -s ' ' | tr '\n' '|')"
    [ -f "$E/decode-$label.txt" ] && log "    decode: $(tail -1 "$E/decode-$label.txt"); $(grep -o 'decode [0-9.]* ms/tok' "$E/decode-$label.txt" | tr '\n' ' ')"
    has needle && [ "${label##*-}" = "${RK_RELEASE_NEEDLE_ROUND:-b}" ] && needle_on "$mode"
  else log "== $label: serve FAILED -- $(tail -2 "$E/serve-$label.out" | tr '\n' '|')"; fi
  docker logs "radiance-ridgefill-$mode" > "$E/$label.serve.log" 2>&1
  scripts/stop.sh > /dev/null 2>&1; drop_cache; klog
}
if has headline; then
  for a in ${RK_RELEASE_ARMS:-exact-a quality-a speed-a exact-b quality-b speed-b}; do arm "$a" "${a%-*}"; done
fi
if [ "$NEEDLE_OK" = 1 ]; then
  for m in quality speed; do
    log "== needle stock vs $m"; python3 tools/needle.py compare "$N/results-exact.jsonl" "$N/results-$m.jsonl" 2>&1 | tee "$N/compare-$m.txt" | head -40 | tee -a "$E/session.log"
  done
fi

# 5 ----------------------------------------------------------------------------------------------------
if has r64; then
R=$E/r64; mkdir -p "$R"
r64() {   # label mode extra-flags
  label=$1 mode=$2 extra=$3
  mkdir -p "$R/dump-$label"; fresh_cache "r64-$label"
  # the profile with MTP off (the capture follows one greedy decoder a row a step) and --profile-ops (no replayed pass)
  f=$(echo "$RK_RELEASE_FLAGS" | sed 's/--num-speculative-tokens 3/--num-speculative-tokens 0/')
  # the no-cache arm: no prefix-cache flag at all and no RK_CACHE_DIR, or common.sh turns the cache back on
  case "$extra" in *--no-prefix-cache*)
    unset RK_CACHE_DIR
    f=$(echo "$f" | sed -e 's/ --prefix-cache-host-mib [0-9]*//' -e 's| --prefix-cache-dir [^ ]*||' -e 's/ --prefix-cache-disk-mib [0-9]*//') ;;
  esac
  if env RK_FLAGS="$f $extra" RK_DOCKER_EXTRA="$MOUNT -v $R/dump-$label:/dump" \
       RADIANCE_RIDGEFILL_DUMP_LOGITS=/dump scripts/serve.sh "$mode" --profile-ops > "$R/serve-$label.out" 2>&1; then
    [ -f "$R/conv.json" ] || python3 tools/turn2.py build --docs "$RK_DOCS" --out "$R/conv.json" > "$R/build.out" 2>&1
    python3 tools/turn2.py run --conv "$R/conv.json" --dump "$R/dump-$label" --out "$R/manifest-$label.json" > "$R/run-$label.out" 2>&1
    log "R64 $label: $(tr '\n' '|' < "$R/run-$label.out")"
  else log "R64 $label: serve FAILED"; fi
  docker logs "radiance-ridgefill-$mode" > "$R/$label.serve.log" 2>&1; scripts/stop.sh > /dev/null 2>&1; drop_cache; klog
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
fi
fresh_cache done; rm -rf "$K/done"; log "prefix-cache dirs used: $(tr '\n' ' ' < "$E/kvcache.txt" 2>/dev/null)"
log "klog after: $(journalctl -k --since "$T0" | grep -ciE 'amdgpu.*(MES|SMU|timeout|reset)')"
log "RELEASE end $(date -u +%FT%TZ)"
