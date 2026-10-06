#!/bin/sh
# kernels_gpu_check.sh -- kernel_test's "gpu" group (every device row against its host row) on ONE card.
#
# USAGE: scripts/kernels_gpu_check.sh <commit> [<pci address, default 0000:13:00.0>]
#
#   1. (no GPU) scripts/frozen_home.sh <commit>: git archive into data/src-<short>, the build in
#      $RK_BUILD_IMAGE (HIP on: the image's radiance has it), ctest -LE gpu there.
#   2. (GPU, under gpuq) the same build tree's `ctest -L gpu` in the same image, the card pinned with
#      ROCR_VISIBLE_DEVICES (scripts/card_index.sh maps the address). The device leg itself runs in seconds
#      (notes/gates.md Gate 1: 1.99 s); the lock is held for one container start and that.
#   Log: evidence/tpx/kernels-gpu-<short>.log; exit 0 when every gpu case passed on the named card.
set -eu

. "$(dirname "$0")/common.sh"

[ "$#" -ge 1 ] || rk_die "usage: scripts/kernels_gpu_check.sh <commit> [<pci address>]"
pci=${2:-0000:13:00.0}
short=$(git -C "$RK_REPO" rev-parse --short "$1") || rk_die "not a commit: $1"
export RK_BUILD_IMAGE RK_RADIANCE_SRC
E=$RK_REPO/evidence/tpx; mkdir -p "$E"
log=$E/kernels-gpu-$short.log

"$RK_SCRIPTS/frozen_home.sh" "$short" > "$E/kernels-build-$short.out" 2>&1 ||
    rk_die "the build or its host tests failed: $E/kernels-build-$short.out"
idx=$("$RK_SCRIPTS/card_index.sh" "$pci")
src=$RK_REPO/data/src-$short
/var/home/dylan/AI-Work/radiance-kva-plugin-20261004/gpuq.sh "kernels-gpu-$short" \
    docker run --rm --security-opt label=disable --device /dev/kfd --device /dev/dri \
        --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES="$idx" \
        -v "$(readlink -f "$RK_RADIANCE_SRC")":/rsrc:ro -v "$src":/src -w /src "$RK_BUILD_IMAGE" \
        ctest --test-dir /src/build -L gpu --output-on-failure -V > "$log" 2>&1 || true
grep -E "visible|PCI|device vs host|heads|ok |FAIL|check\(s\)|tests passed" "$log" | head -60
grep -qi "device 0 of 1 visible, PCI $pci" "$log" ||
    rk_die "the run did not see exactly card $pci (ROCR_VISIBLE_DEVICES=$idx): $(grep -i 'visible' "$log" | head -1)"
grep -q "100% tests passed" "$log" || rk_die "a gpu case failed: $log"
echo "kernels_gpu_check: PASS on $pci (commit $short); $log"
