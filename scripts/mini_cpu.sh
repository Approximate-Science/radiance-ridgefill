#!/bin/sh
# mini_cpu.sh -- the no-GPU correctness run for this plugin, one command (notes/mini-model.md).
#
# WHAT IT DOES. Builds radiance host-only and the plugins host-only (README's host-only section)
# if their build trees are missing, regenerates the mini Qwen4-Exp checkpoint if missing
# (tools/mini_model.py: same architecture, random weights, the real vocab and tokenizer), then runs
# every check the CPU can carry, printing PASS/FAIL with the evidence line that decided it.
#
# WHAT IT CANNOT DO (radiance 1.0.8, found and argued in notes/mini-model.md §4): the stock
# qwen4exp architecture declares ops only libr4d (the DEVICE library) has rows for, so the host
# backend cannot get past declare -- not rad-convert, not serving, stock home or plugin home
# alike. The script stops AT that wall and prints the exact op list; the container, the projector
# and the RADIANCE_RIDGEFILL speed/quality runs behind it are unreachable until those ops grow host
# rows. Everything up to the wall is green and stays green.
#
# Usage: scripts/mini_cpu.sh [job-dir]     default job-dir: <repo>/../mini-run
# Needs: cmake >= 3.21, g++ (C++20), and the research venv's python (RIDGEFILL_PYTHON, below) with
#        numpy/safetensors for mini_model.py and pytest for the Python tier.

set -u

repo=$(cd "$(dirname "$0")/.." && pwd)
job=${1:-$repo/../mini-run}
RADIANCE_SRC=${RADIANCE_SRC:-/var/home/dylan/projects/inference/radiance}
RIDGEFILL_PYTHON=${RIDGEFILL_PYTHON:-/var/home/dylan/projects/research/kva/.venv/bin/python}
mkdir -p "$job"
pass=0; fail=0

say() { printf '%s\n' "$*"; }
ok()  { say "PASS: $*"; pass=$((pass + 1)); }
no()  { say "FAIL: $*"; fail=$((fail + 1)); }

# ---------------------------------------------------------------- step 1: the host-only builds

if [ ! -x "$job/build-radiance-host/install/bin/radiance" ]; then
    say "== building radiance host-only (RAD_WITH_HIP=OFF) into $job/build-radiance-host"
    cmake -S "$RADIANCE_SRC" -B "$job/build-radiance-host" -DRAD_WITH_HIP=OFF -DRAD_WITH_FFMPEG=OFF \
          -DRAD_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$job/build-radiance-host/install" >/dev/null ||
        { no "radiance host-only configure"; exit 1; }
    cmake --build "$job/build-radiance-host" -j || { no "radiance host-only build"; exit 1; }
    cmake --install "$job/build-radiance-host" >/dev/null || { no "radiance host-only install"; exit 1; }
fi
RAD="$job/build-radiance-host/install"
ok "radiance host-only built: $RAD/bin/radiance (host backend, no HIP)"

if [ ! -d "$job/build-host/radiance_home" ]; then
    say "== building the plugins host-only into $job/build-host"
    cmake -S "$repo" -B "$job/build-host" -DCMAKE_PREFIX_PATH="$RAD" -DRADIANCE_SRC="$RADIANCE_SRC" \
          -DRAD_WITH_HIP=OFF >/dev/null || { no "plugin host-only configure"; exit 1; }
    cmake --build "$job/build-host" -j || { no "plugin host-only build"; exit 1; }
fi
plug="$job/build-host/radiance_home"
[ -f "$plug/architectures/qwen4exp_fp8.so" ] && [ -f "$plug/kernels/ridgefill.so" ] &&
    ok "plugin home: $plug (architectures/qwen4exp_fp8.so shadows the install's, kernels/ridgefill.so)" ||
    no "plugin home incomplete under $plug"

# ------------------------------------------------------- step 1b: the no-GPU test gates

if (cd "$job/build-host" && ctest -LE gpu --output-on-failure >/tmp/mini-ctest.log 2>&1); then
    ok "plugin ctest -LE gpu: $(grep -o '[0-9]*% tests passed' /tmp/mini-ctest.log | tail -1)"
else
    no "plugin ctest -LE gpu"; sed -n '1,40p' /tmp/mini-ctest.log
fi
if (cd "$repo" && "$RIDGEFILL_PYTHON" -m pytest tests -q >/tmp/mini-pytest.log 2>&1); then
    ok "python tier: $(tail -1 /tmp/mini-pytest.log)"
else
    no "python tier"; tail -5 /tmp/mini-pytest.log
fi

# ------------------------------------------------------------- step 2: the mini checkpoint

if [ ! -f "$job/mini/ckpt/model.safetensors" ]; then
    say "== generating the mini checkpoint (tools/mini_model.py)"
    "$RIDGEFILL_PYTHON" "$repo/tools/mini_model.py" --out "$job/mini/ckpt" || { no "mini_model.py"; exit 1; }
fi
sz=$(stat -c %s "$job/mini/ckpt/model.safetensors")
ok "mini checkpoint: $job/mini/ckpt ($(echo "$sz/1073741824" | bc -l | cut -c1-5) GiB, $(ls "$job/mini/ckpt" | wc -l) files)"

# The namespace gate: rad-convert's declare walks every name map and shape before any kernel
# question, so a plan whose ONLY complaints are the device-only op list proves the checkpoint is
# the architecture's own -- every tensor found, every shape accepted.
HOME2="$plug:$RAD/share/radiance"
plan=$(RADIANCE_HOME="$HOME2" "$RAD/bin/rad-convert" "$job/mini/ckpt" -o /dev/null \
        --recipe "$repo/scripts/mini-qwen4exp.recipe" --plan-only 2>&1)
missing_ops=$(printf '%s\n' "$plan" | grep -o "no loaded kernel plugin declares op '[a-z0-9_]*'" |
              sort -u | sed "s/.*op '//;s/'//" | tr '\n' ' ')
other_errors=$(printf '%s\n' "$plan" | grep '^E ' |
               grep -v "no loaded kernel plugin declares op" |
               grep -v "rad_arch_declare returned" | head -3)
if [ -n "$other_errors" ]; then
    no "checkpoint namespace: rad-convert --plan-only has non-op errors: $other_errors"
elif [ -n "$missing_ops" ]; then
    say "NOTE: checkpoint namespace is complete (no tensor/shape/name errors); declare stops on ops with no host row: $missing_ops"
else
    ok "checkpoint namespace: rad-convert --plan-only declares clean"
fi

# --------------------------------------------- step 3: serve it on the CPU backend
# This is where the wall is (notes/mini-model.md §4): the stock arch declares device-only ops,
# so the host backend cannot get past declare. Both homes are tried: the plugin home with
# RADIANCE_RIDGEFILL=off must fail on exactly the same ops as the stock home -- mode off runs the
# in-tree declare unchanged, and identical failure is the only comparison the CPU can still make.

stock_ops=""
for home_kind in stock plugin; do
    [ "$home_kind" = stock ] && HOME3="$RAD/share/radiance" || HOME3="$plug:$RAD/share/radiance"
    log=/tmp/mini-serve-$home_kind.log
    RADIANCE_HOME="$HOME3" RADIANCE_RIDGEFILL=off timeout 300 "$RAD/bin/radiance" \
        --model "$job/mini/ckpt" --debug-accept-reference-kernels --tp 1 \
        --max-num-batched-tokens 512 --max-model-len 8192 --port 18123 >"$log" 2>&1
    rc=$?
    ops=$(grep -o "no loaded kernel plugin declares op '[a-z0-9_]*'" "$log" | sort -u |
          sed "s/.*op '//;s/'//" | tr '\n' ' ')
    if [ $rc -eq 0 ]; then
        no "$home_kind home: server exited 0 without staying up (unexpected)"
    elif [ -n "$ops" ]; then
        say "STOP ($home_kind home): CPU serving fails at declare, exit $rc -- ops with no host row:"
        for op in $ops; do say "    $op"; done
        if [ "$home_kind" = stock ]; then
            stock_ops=$ops
            no "stock home: serve --debug-accept-reference-kernels --tp 1 (the stock qwen4exp declare needs device-only ops)"
        elif [ "$ops" = "$stock_ops" ]; then
            say "NOTE: plugin home, RADIANCE_RIDGEFILL=off: fails on exactly the stock home's op list -- mode off runs the in-tree declare unchanged"
            no "plugin home RADIANCE_RIDGEFILL=off: byte-identical output to stock cannot be shown (neither home gets past declare)"
        else
            no "plugin home: failed on a different op list than stock: $ops"
        fi
    else
        no "$home_kind home: server failed for another reason (exit $rc)"; tail -5 "$log"
    fi
done

say ""
say "== $pass PASS, $fail FAIL. The container/projector steps behind the wall (mini container via"
say "   rad-convert, tools/dev/mini_projector.py, RADIANCE_RIDGEFILL=speed/quality runs) are unreachable"
say "   until the ops above grow host rows in radiance; see notes/mini-model.md §4-§5."
[ "$fail" -eq 0 ] || exit 1
exit 0