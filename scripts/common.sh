#!/bin/sh
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# common.sh -- shared configuration, mode validation and the docker-run prefix for every
# measurement script in this repo (serve.sh, stop.sh, preflight.sh, speed.sh, grade.sh).
# SOURCED, never executed.
#
# Env vars and their defaults (every path/port is overridable; nothing machine-specific
# is hard-coded -- the repo location comes from this file's own place):
#   RK_MODEL        required, no default: host path to the .rad container. Its directory
#                   is mounted read-only at /models in every container and the engine is
#                   pointed at /models/<basename>.
#   RK_PORT         default 8100: the port the served engine listens on.
#   RK_IMAGE        default stilldeadcode/radiance:1.0.8 (the runtime image, docs/DOCKER.md).
#   RK_PLUGIN_HOME  default <repo>/home: the plugin home (architectures/, kernels/), mounted
#                   at /plugins for every mode except exact.
#   RK_EVIDENCE     default <repo>/evidence: where measurement outputs go.
#   RK_FLAGS        default below: the engine flags EVERY run shares. Overridable, but a
#                   run with different flags is not comparable with earlier evidence.
#
# The mode argument serve.sh/grade.sh take:
#   exact   the stock engine, the image's own plugin home, no RADIANCE_RIDGEFILL -- the baseline
#   off     the plugin is mounted but RidgeFill is disabled
#   plumb   the fill path is wired but the late layers still run exactly (plumbing control)
#   speed   bulk chunks of layers 24-47 approximated (the speed mode)
#   quality speed plus the rarest rows of the kept classes run exactly (the quality mode)

# --- where this repo is (from the sourcing script's own $0; every default path derives
# from it, so the repo can live anywhere)
RK_SCRIPTS=$(CDPATH= cd "$(dirname "$0")" && pwd) || exit 1
RK_REPO=$(CDPATH= cd "$RK_SCRIPTS/.." && pwd) || exit 1
RK_TOOLS=$RK_REPO/tools

: "${RK_PORT:=8100}"
: "${RK_IMAGE:=stilldeadcode/radiance:1.0.13}"
: "${RK_PLUGIN_HOME:=$RK_REPO/home}"
: "${RK_EVIDENCE:=$RK_REPO/evidence}"
# Exported so a script that hands over to a child process (speed.sh -> tools/speed.py)
# sees the same values the shell does, including a caller's UNEXPORTED override.
export RK_PORT RK_IMAGE RK_PLUGIN_HOME RK_EVIDENCE

# The engine flags every run shares (HANDOVER §7 measurement protocol). Every flag name
# verified against the flag table in radiance core/config.cpp lines 31-190:
#   --tp 2                        tensor parallel over the two cards (config.cpp:37)
#   --kv-cache-dtype fp8          the production KV cache dtype (config.cpp:56)
#   --tp-wire exact               exact cross-rank wire: wht6 is lossy and would put wire
#                                 noise into every quality number (config.cpp:38)
#   --max-num-batched-tokens 2048 the prefill chunk. RidgeFill's bulk chunk is T = 2048, so the
#                                 chunk must stay 2048 (fnserve.sh: "THE CHUNK IS 2048";
#                                 config.cpp:40)
#   --no-prefix-cache             a cache hit would answer a speed rep for free and read
#                                 as a speedup; speed.sh also checks cached_tokens == 0
#                                 on every response (config.cpp:85)
#   --num-speculative-tokens 0    MTP off: the protocol measures plain prefill
#                                 (config.cpp:89)
#   --max-model-len 49152         the measurement context (config.cpp:42)
#   --gpu-headroom-mib 3072       VRAM left unclaimed on every card. One card may drive a
#                                 desktop display, and the engine takes one headroom for all
#                                 cards; 3 GiB keeps the display responsive. fnserve.sh's 96
#                                 is the production shape, this is the measurement shape
#                                 (config.cpp:44). Fewer resident experts than production, the
#                                 same for every arm.
# Mirrored from radiance scripts/fnserve.sh because Flash-Next does not start without
# them (the container's routed experts do not fit both cards' VRAM):
#   --placement expert_tiered     the routed experts that do not fit stream from a pinned
#                                 host pool (fnserve.sh; config.cpp:71)
#   --host-pool-mib 12288         that pinned host pool; it also holds the KL-mode logits
#                                 copy in pinned host memory (fnserve.sh; config.cpp:57)
#   --expert-vs-cache-ratio 0.82  the production weights-vs-KV split (fnserve.sh); at
#                                 --max-model-len 49152 it only widens the expert side
#                                 (config.cpp:51)
: "${RK_FLAGS:=--tp 2 --kv-cache-dtype fp8 --tp-wire exact --max-num-batched-tokens 2048 --no-prefix-cache --num-speculative-tokens 0 --max-model-len 49152 --gpu-headroom-mib 3072 --placement expert_tiered --host-pool-mib 12288 --expert-vs-cache-ratio 0.82}"

# THE CACHE PROFILE (Stage C, HANDOVER-FIX §4): RK_CACHE_DIR=<host dir> serves with fnserve.sh's
# prefix-cache flags instead of --no-prefix-cache -- finished turns copied to a 4 GiB host tier and on
# to disk under the mounted dir (radiance scripts/fnserve.sh:102-103; RK_CACHE_HOST_MIB /
# RK_CACHE_DISK_MIB override the sizes). Everything else in RK_FLAGS stays, so a cache run differs
# from the measured baseline by the cache alone.
if [ -n "${RK_CACHE_DIR:-}" ]; then
    RK_FLAGS="$(printf '%s' "$RK_FLAGS" | sed 's/ *--no-prefix-cache//') --prefix-cache-host-mib ${RK_CACHE_HOST_MIB:-4096} --prefix-cache-dir /kvcache --prefix-cache-disk-mib ${RK_CACHE_DISK_MIB:-32768}"
fi

# rk_die -- print to stderr, naming the missing thing, and exit non-zero.
rk_die() {
    printf 'radiance-ridgefill: %s\n' "$*" >&2
    exit 1
}

# rk_mode_validate MODE -- one of exact|off|plumb|speed|quality, else die naming it.
rk_mode_validate() {
    case $1 in
    exact|off|plumb|speed|quality) ;;
    *) rk_die "mode '$1' is not one of exact|off|plumb|speed|quality" ;;
    esac
}

# rk_require_model -- RK_MODEL must be set and name the .rad container; die naming it.
rk_require_model() {
    [ -n "${RK_MODEL:-}" ] || rk_die "RK_MODEL is not set: the host path to the .rad container"
    [ -f "$RK_MODEL" ] || rk_die "RK_MODEL is not a file: $RK_MODEL"
}

# rk_docker_prefix MODE -- print the docker-run prefix for MODE, ONE ARGUMENT PER LINE.
# POSIX sh has no arrays, so callers collect the lines into "$@" through a temp file:
#     rk_tmp=$(mktemp)
#     rk_docker_prefix "$MODE" > "$rk_tmp"
#     set --
#     while IFS= read -r rk_arg; do set -- "$@" "$rk_arg"; done < "$rk_tmp"
#     rm -f "$rk_tmp"
# (the read loop must not be in a pipeline: a pipeline puts it in a subshell and the
# set -- would be lost).
#
# What the prefix carries and why (docs/DOCKER.md "Run it"):
#   --device /dev/kfd --device /dev/dri   the GPUs. No --group-add: the runtime image is FROM
#                                        scratch with no video/render entries (docker refuses
#                                        the names), and under rootless Docker the host's kfd/
#                                        render nodes must be world-rw anyway
#   --security-opt label=disable        SELinux hosts: the container label cannot open the
#                                        device nodes otherwise (found in the kernel tests)
#   --security-opt seccomp=unconfined   io_uring (the n-gram table) and the NUMA binding
#                                        the ROCm runtime makes for pinned host memory
#   --init                              the engine finishes its step and releases the
#                                        cards on SIGTERM instead of dying as PID 1
#   -p 127.0.0.1:PORT:PORT              publish the server port on loopback. NOT --network host:
#                                        under rootless Docker it hides every GPU from the HIP
#                                        runtime ("built with HIP but no device is visible";
#                                        measured 2026-10-04, each flag tried alone)
#   --ulimit memlock=-1                 the pinned host pool needs unlimited memlock
#                                        (deploy/compose/common.yaml does the same)
#   -v <model dir>:/models:ro          the model, read-only
# and the mode's home:
#   exact:  -e RADIANCE_HOME=/opt/radiance/share/radiance      (stock, no plugin mounted)
#   other:  -v $RK_PLUGIN_HOME:/plugins:ro
#           -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance   (plugin first)
#           -e RADIANCE_RIDGEFILL=<MODE>
# Every RADIANCE_RIDGEFILL* (except RADIANCE_RIDGEFILL itself, which the mode argument sets),
# RADIANCE_LOG_STEPS, RADIANCE_PROFILE_EVERY and RADIANCE_DEBUG_ROUTING in the caller's
# environment is passed through (-e NAME, the caller's value), then RK_DOCKER_EXTRA's words.
rk_docker_prefix() {
    rk_mode_validate "$1"
    rk_require_model
    printf '%s\n' \
        --device /dev/kfd \
        --device /dev/dri \
        --security-opt seccomp=unconfined \
        --security-opt label=disable \
        --init \
        -p "127.0.0.1:$RK_PORT:$RK_PORT" \
        --ulimit memlock=-1 \
        -v "$(dirname "$(readlink -f "$RK_MODEL")")":/models:ro
    [ -z "${RK_CACHE_DIR:-}" ] || printf '%s\n' -v "$RK_CACHE_DIR":/kvcache
    if [ "$1" = exact ]; then
        printf '%s\n' -e RADIANCE_HOME=/opt/radiance/share/radiance
    else
        [ -d "$RK_PLUGIN_HOME" ] || rk_die "RK_PLUGIN_HOME is not a directory: $RK_PLUGIN_HOME"
        printf '%s\n' -v "$RK_PLUGIN_HOME":/plugins:ro
        printf '%s\n' -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance
        printf '%s\n' -e RADIANCE_RIDGEFILL="$1"
    fi
    for rk_name in $(env | cut -d= -f1 | LC_ALL=C sort -u |
                     grep -E '^RADIANCE_RIDGEFILL|^RADIANCE_LOG_STEPS$|^RADIANCE_PROFILE_EVERY$|^RADIANCE_DEBUG_ROUTING$' |
                     grep -v '^RADIANCE_RIDGEFILL$'); do
        printf '%s\n' -e "$rk_name"
    done
    # RK_DOCKER_EXTRA: extra docker-run arguments, one per WORD (no spaces inside one), e.g.
    # RK_DOCKER_EXTRA="-v /host/dump:/dump" for RADIANCE_RIDGEFILL_DUMP=/dump, which needs a writable
    # mount. Debug runs only; empty by default, so measured runs are unchanged.
    # shellcheck disable=SC2086  # one argument per word by construction
    [ -z "${RK_DOCKER_EXTRA:-}" ] || printf '%s\n' $RK_DOCKER_EXTRA
}