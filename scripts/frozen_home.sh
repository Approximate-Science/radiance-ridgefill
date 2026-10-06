#!/bin/sh
# frozen_home.sh -- build the plugin home a measurement runs against from ONE commit, so every
# number names the exact source it came from and no working-tree state can leak into it.
#
# USAGE: scripts/frozen_home.sh <commit>
#
# WHAT IT DOES: `git archive <commit>` into data/src-<short>/ (data/ is git-ignored), builds it in
# the radiance-build image against the image's own radiance install and a read-only mount of the
# radiance source checkout, runs the host test suite there (`ctest -LE gpu`), and copies the two
# plugins into data/home-<short>/{architectures,kernels}. Prints the home and both sha256s.
# Idempotent: an existing home for the commit is rebuilt from scratch.
#
# Env vars:
#   RK_RADIANCE_SRC  the radiance source checkout at the installed release (required; the arch
#                    plugin #includes the in-tree qwen4exp_fp8.cpp from it)
#   RK_BUILD_IMAGE   default radiance-build
#   RK_HOME_TAG      appended to both directory names (data/src-<short><tag>, data/home-<short><tag>), so
#                    one commit built against two radiance releases keeps two homes
#                    (scripts/update_radiance.sh passes -r<release>); default empty
set -eu

. "$(dirname "$0")/common.sh"

[ "$#" -eq 1 ] || rk_die "usage: scripts/frozen_home.sh <commit>"
[ -n "${RK_RADIANCE_SRC:-}" ] || rk_die "RK_RADIANCE_SRC is not set: the radiance source checkout"
[ -f "$RK_RADIANCE_SRC/arch/qwen4exp_fp8/qwen4exp_fp8.cpp" ] || rk_die "RK_RADIANCE_SRC is not a radiance checkout: $RK_RADIANCE_SRC"
: "${RK_BUILD_IMAGE:=radiance-build:1.0.13}"

short=$(git -C "$RK_REPO" rev-parse --short "$1") || rk_die "not a commit: $1"
src=$RK_REPO/data/src-$short${RK_HOME_TAG:-}
home=$RK_REPO/data/home-$short${RK_HOME_TAG:-}
rm -rf "$src" "$home"
mkdir -p "$src" "$home/architectures" "$home/kernels"
git -C "$RK_REPO" archive "$short" | tar -x -C "$src"

docker run --rm --security-opt label=disable \
    -v "$(readlink -f "$RK_RADIANCE_SRC")":/rsrc:ro -v "$src":/src -w /src "$RK_BUILD_IMAGE" \
    bash -c 'cmake -S /src -B /src/build -G Ninja -DCMAKE_PREFIX_PATH=/stage/opt/radiance -DRADIANCE_SRC=/rsrc >/dev/null &&
             cmake --build /src/build >/dev/null && ctest --test-dir /src/build -LE gpu --output-on-failure' ||
    rk_die "build or host tests failed for $short"

cp "$src/build/radiance_home/architectures/qwen4exp_fp8.so" "$home/architectures/"
cp "$src/build/radiance_home/kernels/ridgefill.so" "$home/kernels/"
printf 'home %s (commit %s)\n' "$home" "$(git -C "$RK_REPO" rev-parse "$short")"
sha256sum "$home/architectures/qwen4exp_fp8.so" "$home/kernels/ridgefill.so"
