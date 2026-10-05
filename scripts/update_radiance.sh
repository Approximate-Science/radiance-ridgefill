#!/bin/sh
# update_radiance.sh -- try the plugin against another radiance release in one command.
#
# USAGE: scripts/update_radiance.sh [--gpu-smoke] <radiance tag or commit>      e.g. v1.0.14
#
# WHAT IT DOES, stopping at the first red step:
#   a. SOURCE   `git archive` of the release from the radiance checkout ($RK_RADIANCE_REPO) into
#               data/radiance-src-<name>/, made read-only. <name> is the release (1.0.14) when the commit is
#               that release's tag, else <release>-g<short>. An existing directory is verified against the
#               commit (tar --compare) and reused.
#   b. BUILD    a host-only install of that radiance (build-radiance-<name>-host/, reused when its engine
#               carries the release string) and the plugin against it (build-host-<name>/). CMake's own
#               checks refuse a source/install mismatch. A compile error is the first answer: it names
#               the file and line that no longer fits the release.
#   c. TESTS    ctest -LE gpu -- the static oracle (`off` and every approximate path held issue for
#               issue against THAT release's in-tree plugin), adapter_core, the purity gate, the kernel
#               host rows -- then pytest. A failing oracle case is printed with its first difference.
#   d. PACKAGE  only when a-c are green and the release's build image exists (radiance-build:<release>):
#               the plugin's committed HEAD built in that image (scripts/frozen_home.sh, which runs the
#               host suite again in there) into data/home-<short>-r<name>, then tools/package.py into
#               dist/radiance-kva-r<name>-<short>/. No image: SKIPPED, with the commands that make one.
#   e. VERDICT  PASS or FAIL, and -- either way -- which in-tree files the adapters COPY changed between
#               the pinned release (RADIANCE_RELEASE) and this one, and which adapter file copies each
#               (arch/*.copies): the files to port when the oracle fails, and to review when it does not.
#
# --gpu-smoke (needs d's home and stilldeadcode/radiance:<release>): queues ONE gpuq session -- stock and
# off ident on the release's runtime image (must be equal), an exact KL reference for it
# (data/kld/ref-r<name>, recorded unless present), then int8 quality T2560 and int8 speed T2048 against
# it, scored last-512 paired vs exact (scripts/kl_tail.py, R100's protocol). ~15-20 min of GPU; the
# session's log is printed as the command to watch.
#
# Exit 0 PASS, 1 FAIL, 2 usage or a missing input. Nothing outside this repo's data/, build-*/ and dist/
# is written; the radiance checkout is only read (git rev-parse / show / archive / diff).
#
# Env vars:
#   RK_RADIANCE_REPO  the radiance git checkout (default /var/home/dylan/projects/inference/radiance)
#   RK_CMAKE          cmake >= 3.21 (default: `cmake` on PATH, else ~/.local/bin/cmake)
#   RK_PYTHON         python for pytest (default python3)
#   RK_PROJECTOR      the folder to package and smoke-test (default data/projector-qwen38fn-int8)
#   RK_JOBS           build parallelism (default 4, at nice 19)
set -u

. "$(dirname "$0")/common.sh"

smoke=0
case "${1:-}" in --gpu-smoke) smoke=1; shift ;; esac
[ "$#" -eq 1 ] || { echo "usage: scripts/update_radiance.sh [--gpu-smoke] <radiance tag or commit>" >&2; exit 2; }
ref=$1
RREPO=${RK_RADIANCE_REPO:-/var/home/dylan/projects/inference/radiance}
PY=${RK_PYTHON:-python3}
JOBS=${RK_JOBS:-4}
PROJ=${RK_PROJECTOR:-$RK_REPO/data/projector-qwen38fn-int8}
cmake_ok() {  # cmake >= 3.21, the minimum of both trees' CMakeLists.txt
    v=$("$1" --version 2>/dev/null | sed -n '1s/^cmake version \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
    [ -n "$v" ] || return 1
    major=${v% *} minor=${v#* }
    [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 21 ]; }
}
CM=""
for c in ${RK_CMAKE:-} cmake "$HOME/.local/bin/cmake"; do cmake_ok "$c" && { CM=$c; break; }; done
[ -n "$CM" ] || { echo "update_radiance: no cmake >= 3.21 (set RK_CMAKE)" >&2; exit 2; }

# ---------------------------------------------------------------- a. the release
commit=$(git -C "$RREPO" rev-parse --verify --quiet "$ref^{commit}") ||
    { echo "update_radiance: '$ref' is not a commit of $RREPO" >&2; exit 2; }
short=$(printf '%s' "$commit" | cut -c1-8)
ver=$(git -C "$RREPO" show "$commit:CMakeLists.txt" | sed -n 's/^project(radiance VERSION \([0-9][0-9.]*\).*/\1/p')
[ -n "$ver" ] || { echo "update_radiance: no project(radiance VERSION ...) at $commit" >&2; exit 2; }
if [ "$(git -C "$RREPO" describe --exact-match --tags "$commit" 2>/dev/null)" = "v$ver" ]; then name=$ver
else name=$ver-g$short; fi
SRC=$RK_REPO/data/radiance-src-$name
LOG=$RK_REPO/data/update-$name
mkdir -p "$LOG"
: > "$LOG/summary.txt"
say() { printf '%s\n' "$*" | tee -a "$LOG/summary.txt"; }
step() { printf '  %-14s %s\n' "$1" "$2" | tee -a "$LOG/summary.txt"; }

pin=$(cut -d' ' -f2 "$RK_REPO/RADIANCE_RELEASE" 2>/dev/null || true)
pinver=$(cut -d' ' -f1 "$RK_REPO/RADIANCE_RELEASE" 2>/dev/null || true)
plugin_sha=$(git -C "$RK_REPO" rev-parse HEAD)
say "update_radiance: radiance $ver ($commit, '$ref') for plugin $(git -C "$RK_REPO" rev-parse --short HEAD) on $(git -C "$RK_REPO" branch --show-current); pinned ${pinver:-none}"
say "  logs: $LOG"

# e's report, which every exit prints: what the adapters copy that this release changed.
copies_report() {
    [ -n "$pin" ] || { say "  (no RADIANCE_RELEASE pin: cannot say what changed)"; return; }
    [ "$pin" = "$commit" ] && { say "  in-tree changes since the pin: none (this IS the pinned release)"; return; }
    if git -C "$RREPO" diff --quiet "$pin" "$commit" -- abi; then say "  abi/ since $pinver: unchanged"
    else say "  abi/ since $pinver: CHANGED -- $(git -C "$RREPO" diff --shortstat "$pin" "$commit" -- abi)"; fi
    say "  in-tree files the adapters copy (arch/*.copies), $pinver -> $ver:"
    grep -hv '^#' "$RK_REPO"/arch/*.copies | awk 'NF == 2' | while read -r mine theirs; do
        if git -C "$RREPO" diff --quiet "$pin" "$commit" -- "$theirs"; then st="unchanged"
        else st="CHANGED ($(git -C "$RREPO" diff --numstat "$pin" "$commit" -- "$theirs" | awk '{print "+"$1" -"$2}'))"; fi
        printf '    %-26s copies %-42s %s\n' "$mine" "$theirs" "$st" | tee -a "$LOG/summary.txt"
    done
    say "  everything else changed in arch/ (called, not copied):"
    git -C "$RREPO" diff --stat "$pin" "$commit" -- arch | sed '$d' | sed 's/^/    /' | tee -a "$LOG/summary.txt"
}
finish() {  # verdict exit-code
    copies_report
    say "RESULT: $1"
    exit "$2"
}

if [ -d "$SRC" ]; then
    diffs=$(git -C "$RREPO" archive "$commit" | tar -d -C "$SRC" 2>&1 | grep -vE 'Mod time differs|Mode differs|Uid differs|Gid differs' || true)
    if [ -n "$diffs" ]; then step source "FAIL: $SRC exists and is not $commit: $(printf '%s' "$diffs" | head -3 | tr '\n' '|')"; finish FAIL 1; fi
    step source "OK $SRC (existing, verified against $short)"
else
    mkdir -p "$SRC" && git -C "$RREPO" archive "$commit" | tar -x -C "$SRC" && chmod -R a-w "$SRC" ||
        { step source "FAIL: git archive into $SRC"; finish FAIL 1; }
    step source "OK $SRC (archived, read-only)"
fi

# ---------------------------------------------------------------- b. builds
HB=$RK_REPO/build-radiance-$name-host
if [ -x "$HB/install/bin/radiance" ] && strings "$HB/install/bin/radiance" | grep -qxF "$ver"; then
    step "host install" "OK $HB/install (existing, carries $ver)"
else
    if nice -n 19 "$CM" -S "$SRC" -B "$HB" -DRAD_WITH_HIP=OFF -DRAD_WITH_FFMPEG=OFF -DRAD_BUILD_TESTS=OFF \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HB/install" > "$LOG/radiance-host.log" 2>&1 &&
       nice -n 19 "$CM" --build "$HB" -j "$JOBS" >> "$LOG/radiance-host.log" 2>&1 &&
       "$CM" --install "$HB" >> "$LOG/radiance-host.log" 2>&1; then
        step "host install" "OK $HB/install (built)"
    else
        step "host install" "FAIL: radiance $ver does not build host-only ($LOG/radiance-host.log): $(grep -m3 -E 'error' "$LOG/radiance-host.log" | tr '\n' '|')"
        finish FAIL 1
    fi
fi
PB=$RK_REPO/build-host-$name
if ! "$CM" -S "$RK_REPO" -B "$PB" -DCMAKE_PREFIX_PATH="$HB/install" -DRADIANCE_SRC="$SRC" -DRAD_WITH_HIP=OFF \
        > "$LOG/plugin-configure.log" 2>&1; then
    step "plugin build" "FAIL at configure ($LOG/plugin-configure.log):"
    grep -A4 'CMake Error' "$LOG/plugin-configure.log" | head -12 | sed 's/^/      /' | tee -a "$LOG/summary.txt"
    finish FAIL 1
fi
if ! nice -n 19 "$CM" --build "$PB" -j "$JOBS" > "$LOG/plugin-build.log" 2>&1; then
    step "plugin build" "FAIL: the plugin does not compile against radiance $ver ($LOG/plugin-build.log). First errors:"
    grep -E 'error:' "$LOG/plugin-build.log" | sed "s|$RK_REPO/||" | sort -u | head -12 | sed 's/^/      /' | tee -a "$LOG/summary.txt"
    finish FAIL 1
fi
step "plugin build" "OK $PB"

# ---------------------------------------------------------------- c. tests
(cd "$PB" && ctest -LE gpu --output-on-failure) > "$LOG/ctest.log" 2>&1
ct=$?
step "ctest -LE gpu" "$( [ $ct = 0 ] && echo OK || echo FAIL): $(grep -E 'tests passed' "$LOG/ctest.log" | tail -1)"
if [ $ct != 0 ]; then
    grep -E '^ *[0-9]+ - .*\(Failed\)' "$LOG/ctest.log" | sed 's/^/      /' | tee -a "$LOG/summary.txt"
    for t in arch_static_test adapter_core_test; do
        [ -x "$PB/tests/$t" ] || continue
        "$PB/tests/$t" > "$LOG/$t.log" 2>&1
        grep -E '^  FAIL|first difference' "$LOG/$t.log" | sort | uniq -c | head -20 | sed "s/^/      [$t] /" | tee -a "$LOG/summary.txt"
    done
    finish FAIL 1
fi
for t in arch_static_test adapter_core_test; do   # their check counts, which ctest does not print
    "$PB/tests/$t" > "$LOG/$t.log" 2>&1
    printf '      %-18s %s\n' "$t" "$(grep -E 'check\(s\) over' "$LOG/$t.log")" | tee -a "$LOG/summary.txt"
done
(cd "$RK_REPO" && "$PY" -m pytest -q tests) > "$LOG/pytest.log" 2>&1
pt=$?
step pytest "$( [ $pt = 0 ] && echo OK || echo FAIL): $(tail -1 "$LOG/pytest.log")"
[ $pt = 0 ] || finish FAIL 1

# ---------------------------------------------------------------- d. packages
BIMG=radiance-build:$ver
home=""
if ! docker image inspect "$BIMG" > /dev/null 2>&1; then
    step package "SKIPPED: no build image $BIMG. Make it: (cd $RREPO && docker/build.sh -r $commit --target build && docker tag radiance-build $BIMG)"
elif [ -n "$(git -C "$RK_REPO" status --porcelain -- arch kernels tests CMakeLists.txt)" ]; then
    step package "SKIPPED: arch/ kernels/ tests/ or CMakeLists.txt has uncommitted changes; a package is built from a commit"
else
    hs=$(git -C "$RK_REPO" rev-parse --short HEAD)
    if RK_HOME_TAG=-r$name RK_BUILD_IMAGE=$BIMG RK_RADIANCE_SRC=$SRC "$RK_SCRIPTS/frozen_home.sh" HEAD > "$LOG/frozen_home.log" 2>&1; then
        home=$RK_REPO/data/home-$hs-r$name
        pver=$(sed -n 's/^RAD_ARCH_PLUGIN([^,]*, *"[^"]*", *"[^"]*", *"\([0-9.]*\)".*/\1/p' "$RK_REPO/arch/qwen4exp_kva.cpp")
        abi=$(sed -n 's/^set(PACKAGE_VERSION "\([0-9.]*\)")/\1/p' "$HB"/install/lib*/cmake/radiance/*ersion*.cmake 2>/dev/null | head -1)
        out=$RK_REPO/dist/radiance-kva-r$name-$hs
        if [ -e "$out" ]; then
            step package "SKIPPED: $out exists (remove it to rebuild); device home $home"
        elif "$PY" "$RK_REPO/tools/package.py" --home "$home" --projector "$PROJ" --out "$out" --version "$pver" \
                --commit "$plugin_sha" --radiance-version "$ver" ${abi:+--abi-version "$abi"} > "$LOG/package.log" 2>&1; then
            step package "OK $out (plugin $pver, radiance $ver, ABI ${abi:-?}; device home $home)"
        else
            step package "FAIL: tools/package.py ($LOG/package.log): $(tail -2 "$LOG/package.log" | tr '\n' '|')"
            finish FAIL 1
        fi
    else
        step package "FAIL: the device build in $BIMG ($LOG/frozen_home.log): $(grep -m3 -E 'error|FAIL' "$LOG/frozen_home.log" | tr '\n' '|')"
        finish FAIL 1
    fi
fi

# ---------------------------------------------------------------- the optional GPU smoke
if [ $smoke = 1 ]; then
    RIMG=stilldeadcode/radiance:$ver
    if [ -z "$home" ]; then step "gpu smoke" "NOT QUEUED: no device home (package step above)"
    elif ! docker image inspect "$RIMG" > /dev/null 2>&1; then step "gpu smoke" "NOT QUEUED: no runtime image $RIMG"
    else
        S=$LOG/smoke.sh
        cat > "$S" <<EOF
#!/bin/sh
# generated by scripts/update_radiance.sh for radiance $ver ($commit), plugin home $home
set -u
cd $RK_REPO || exit 1
E=$LOG/smoke; mkdir -p \$E
export RK_IMAGE=$RIMG RK_PLUGIN_HOME=$home RK_STAGE=update-$name
export RK_MODEL=\${RK_MODEL:-${RK_MODEL:-/var/home/dylan/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad}}
I8="RK_DOCKER_EXTRA=-v $PROJ:/projector:ro"
REF=$RK_REPO/data/kld/ref-r$name
C=$RK_REPO/corpus/quick9.jsonl
T0=\$(date '+%Y-%m-%d %H:%M:%S')
log() { echo "\$*" | tee -a \$E/session.log; }
klog() { if journalctl -k --since "\$T0" | grep -qiE 'amdgpu.*(MES|SMU|timeout|reset)'; then log "STOP: kernel log"; exit 3; fi; }
fin() { docker logs radiance-kva-\$2 > \$E/\$1.serve.log 2>&1; scripts/stop.sh > /dev/null 2>&1; klog; }
srv=\$(docker ps --format '{{.Names}} {{.Image}}' | awk '\$2 !~ /build/ {print \$1}' | tr '\n' ' ')
[ -z "\$srv" ] || { log "STOP: a serving container is running: \$srv"; exit 4; }
log "smoke start \$(date -u +%FT%TZ) boot \$(cat /proc/sys/kernel/random/boot_id) radiance $ver home $home"
scripts/serve.sh exact > \$E/exact.serve.out 2>&1 && scripts/ident.sh > \$E/exact.ident 2>&1; fin exact exact
scripts/serve.sh off > \$E/off.serve.out 2>&1 && scripts/ident.sh > \$E/off.ident 2>&1; fin off off
if [ -s \$E/exact.ident ] && diff -q \$E/exact.ident \$E/off.ident > /dev/null; then log "ident: off EQUALS stock -- PASS"
else log "ident: off DIFFERS from stock (or a serve failed) -- FAIL"; diff \$E/exact.ident \$E/off.ident | head -8 | tee -a \$E/session.log; fi
if [ ! -d \$REF ]; then scripts/grade.sh record \$REF \$C > \$E/ref.out 2>&1 || { log "reference FAILED: \$(tail -2 \$E/ref.out | tr '\n' '|')"; exit 5; }; log "reference recorded: \$REF"; fi
env RADIANCE_KVA_TAIL=2560 RADIANCE_KVA_PROJECTOR=/projector RADIANCE_KVA_SCORE_BULK=1 RK_EXPECT_APPROX=67 "\$I8" scripts/grade.sh quality \$REF \$E/quality-t2560.json > \$E/quality.out 2>&1; log "quality T2560: \$(tail -1 \$E/quality.out)"; klog
env RADIANCE_KVA_PROJECTOR=/projector RADIANCE_KVA_SCORE_BULK=1 RK_EXPECT_APPROX=67 "\$I8" scripts/grade.sh speed \$REF \$E/speed-t2048.json > \$E/speed.out 2>&1; log "speed T2048: \$(tail -1 \$E/speed.out)"; klog
$PY scripts/kl_tail.py --corpus \$C --last 512 \$E/quality-t2560.json \$E/speed-t2048.json 2>&1 | grep -v '^    ppl/' | tee -a \$E/session.log
log "smoke end \$(date -u +%FT%TZ)"
EOF
        nohup setsid /var/home/dylan/AI-Work/radiance-kva-plugin-20261004/gpuq.sh "update-$name" sh "$S" \
            > "$LOG/smoke.queue" 2>&1 < /dev/null &
        sleep 2
        step "gpu smoke" "QUEUED: $(head -1 "$LOG/smoke.queue"); watch $LOG/smoke/session.log"
    fi
fi
finish PASS 0
