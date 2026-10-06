#!/bin/sh
# append_prep.sh -- everything before a Stage 6 in-place append EXCEPT the append: the refit sidecar shard, its R13
# self-check, the header-only stub naming the shipped shard + the refit shard, the container's current rad-info,
# `rad-convert --plan-only -v` and the plan diff gate (notes/sidecar.md §7). Writes nothing to the container (its
# directory is mounted read-only here) and needs no GPU. The real append is the orchestrator's (notes/refit.md §7).
#
# USAGE: tools/refit/append_prep.sh projr|full
#   projr  append #1: the refit projector alone (ridgefill.projr.*; no ridgefill.str.*, so RADIANCE_RIDGEFILL_ST=refit = no correction)
#   full   append #2: the full refit set (ridgefill.projr.* byte-identical, reused by name; ridgefill.str.* new)
# Env: RK_MODEL (the .rad), RK_PLUGIN_HOME (home whose arch declares every copy under RADIANCE_RIDGEFILL_DECLARE=all),
#      RK_RECIPE (default: the recipe beside the container), RK_PYTHON, RK_REFIT_DATA (default <repo>/data/refit),
#      RK_STUB_REVISION (the source checkpoint commit the shipped stub pinned, data/stub/stub-manifest.json).
# These set RADIANCE_RIDGEFILL_PROJ/_ST/_DECLARE, which the current plugin refuses: run them only against the frozen
# home they were written for (see tools/dev/README.md).
set -eu

[ "$#" -eq 1 ] || { echo "usage: tools/refit/append_prep.sh projr|full" >&2; exit 1; }
which=$1
# Physical paths throughout: the stub's shard symlinks are relative, and a relative path computed between a
# symlinked spelling (/home/...) and a resolved one (/var/home/...) points nowhere inside the /ridgefill mount.
here=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
repo=$(CDPATH= cd "$here/../.." && pwd -P)
: "${RK_MODEL:?the .rad container}" "${RK_PLUGIN_HOME:?the plugin home}" "${RK_PYTHON:?python with torch + safetensors}"
: "${RK_REFIT_DATA:=$repo/data/refit}"
RK_REFIT_DATA=$(readlink -f "$RK_REFIT_DATA")
models=$(dirname "$(readlink -f "$RK_MODEL")")
rad=$(basename "$RK_MODEL")
: "${RK_RECIPE:=$models/qwen4exp-w4nl64-i8-hc8m.recipe}"
: "${RK_STUB_REVISION:=$("$RK_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['commit'])" "$repo/data/stub/stub-manifest.json")}"
proj=$RK_REFIT_DATA/proj/ridgefill-radiance-s24.safetensors
case $which in
projr) st_args="" ;;
full)  st_args="--st $RK_REFIT_DATA/st/ridgefill-radiance-s24-st.rank0.pt $RK_REFIT_DATA/st/ridgefill-radiance-s24-st.rank1.pt" ;;
*) echo "'$which' is not projr|full" >&2; exit 1 ;;
esac
side=$RK_REFIT_DATA/sidecar-$which
stub=$RK_REFIT_DATA/stub-$which
ev=$RK_REFIT_DATA/append-$which
mkdir -p "$ev"
rel() { r=$(readlink -f "$1"); printf '%s' "${r#"$repo"/}"; }   # repo-relative path: the repo is mounted at /ridgefill

# shellcheck disable=SC2086  # st_args is empty or two paths without spaces
"$RK_PYTHON" "$repo/tools/dev/ridgefill_sidecar.py" build --names refit --proj "$proj" $st_args --out "$side"
# shellcheck disable=SC2086
"$RK_PYTHON" "$repo/tools/dev/ridgefill_sidecar.py" verify "$side/ridgefill-sidecar-refit.safetensors" --names refit --proj "$proj" \
    $st_args > "$ev/verify-shard.txt" || { cat "$ev/verify-shard.txt"; echo "shard verify FAILED" >&2; exit 1; }
cat "$ev/verify-shard.txt"
rm -rf "$stub"
"$RK_PYTHON" "$repo/tools/dev/stub_checkpoint.py" --repo Qwen/Qwen3.8-Flash-Next --revision "$RK_STUB_REVISION" \
    --out "$stub" --extra "$repo/data/sidecar/ridgefill-sidecar.safetensors" --extra "$side/ridgefill-sidecar-refit.safetensors"
sort -u "$repo/data/sidecar/rad-convert-set.txt" "$side/rad-convert-set.txt" > "$ev/set.txt"
dup=$(cut -d= -f1 "$ev/set.txt" | sort | uniq -d)
[ -z "$dup" ] || { echo "conflicting --set keys: $dup" >&2; exit 1; }

run() {   # the build image with the repo read-only at /ridgefill and the model dir read-only at /models
    docker run --rm --security-opt label=disable -e RADIANCE_RIDGEFILL_DECLARE=all -e CALIB=calib/w4nl-calib \
        -v "$models":/models:ro -v "$repo":/ridgefill:ro radiance-build "$@"
}
run /stage/opt/radiance/bin/rad-info -v "/models/$rad" > "$ev/rad-info-v-before.txt"
run /stage/opt/radiance/bin/rad-info --meta "/models/$rad" > "$ev/rad-info-meta-before.txt"
sets=$(sed 's/^/--set /' "$ev/set.txt" | tr '\n' ' ')
# shellcheck disable=SC2086  # one --set key=value pair per word by construction (asserted in ridgefill_sidecar)
run /stage/opt/radiance/bin/rad-convert "/ridgefill/$(rel "$stub")" --reuse "/models/$rad" --in-place \
    --recipe "/models/$(basename "$RK_RECIPE")" --home "/ridgefill/$(rel "$RK_PLUGIN_HOME"):/stage/opt/radiance/share/radiance" \
    $sets --set ridgefill.mode=off --set ridgefill.tail=2048 --plan-only -v > "$ev/plan-only.log" 2>&1
# The tensors allowed to be new: the whole shard for projr; for full only ridgefill.str.* (ridgefill.projr.* is already held
# after append #1), written as a shard of just those names because plan_diff reads names from a shard header.
expect=$side/ridgefill-sidecar-refit.safetensors
if [ "$which" = full ]; then
    expect=$ev/expect-new.safetensors
    "$RK_PYTHON" -c "import sys; from safetensors.torch import load_file, save_file
t = load_file(sys.argv[1]); save_file({k: v for k, v in t.items() if k.startswith('ridgefill.str.')}, sys.argv[2])" \
        "$side/ridgefill-sidecar-refit.safetensors" "$expect"
fi
"$RK_PYTHON" "$repo/tools/dev/plan_diff.py" --plan "$ev/plan-only.log" --container "$ev/rad-info-v-before.txt" \
    --expect-new "$expect" --out "$ev/plan-diff.json" > "$ev/plan-diff.txt" ||
    { cat "$ev/plan-diff.txt"; echo "plan diff FAILED" >&2; exit 1; }
cat "$ev/plan-diff.txt"
