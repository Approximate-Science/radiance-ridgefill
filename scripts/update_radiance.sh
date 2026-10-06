#!/bin/sh
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# update_radiance.sh -- is the plugin compatible with another radiance release? One command, one verdict.
#
# USAGE: scripts/update_radiance.sh <radiance tag or commit> [--host-only] [--ci] [--gpu-smoke]
#        e.g.  scripts/update_radiance.sh v1.0.14
#
# EXIT: 0 COMPATIBLE, 1 INCOMPATIBLE, 2 INFRASTRUCTURE (the answer could not be computed: a bad tag, the
# radiance checkout, the toolchain, a missing Python package). 2 is never reported as compatible.
#
# WHAT IT DOES, stopping at the first red step:
#   a. SOURCE   `git archive` of the release from the radiance checkout ($RK_RADIANCE_REPO) into
#               data/radiance-src-<name>/, made read-only. <name> is the release (1.0.14) when the commit is
#               that release's tag, else <release>-g<short>. An existing directory is verified against the
#               commit (tar --compare) and reused.
#   b. ABI      RAD_ABI_VERSION (abi/rad_abi.h) against the pinned release's: a change is INCOMPATIBLE
#               whatever the tests say (a plugin is loaded by ABI number; a new number is a new contract).
#   c. BUILD    a host-only install of that radiance (build-radiance-<name>-host/, reused when its engine
#               carries the release string) and the plugin against it (build-host-<name>/). The plugin's
#               configure checks and a compile error are INCOMPATIBLE answers: the error names the file and
#               line that no longer fits the release.
#   d. TESTS    ctest -LE gpu -- the static oracle (`off` and every approximate path held issue for issue
#               against THAT release's in-tree plugin), adapter_core, the purity gate, the kernel host rows
#               -- then pytest. Every failing oracle case is printed with its first difference and the
#               adapter files it points at: those whose COPIED in-tree source changed (arch/*.copies).
#   e. PACKAGE  (not with --host-only) when a-d are green and radiance-build:<release> exists: the
#               committed HEAD built in that image (scripts/frozen_home.sh) and tools/package.py into
#               dist/radiance-ridgefill-r<name>-<short>/; no image: SKIPPED, naming the commands that make one.
#   f. REPORT   always: the WARNING section -- every file of radiance's abi/, arch/common/ and
#               arch/qwen4exp_fp8/ that changed between the pinned release (RADIANCE_VERSION) and this one
#               -- and which adapter file copies each changed file. Loud, but not failing: a compatible
#               release with changed blocks still deserves a look.
#
# --ci         also prints GitHub/Forgejo workflow commands (::error:: for the verdict and each failure,
#              ::warning:: for each changed in-tree file) and writes the report to data/update-<name>/
#              report.md for the workflow to upload.
# --host-only  stops after d (no docker): what CI runs.
# --gpu-smoke  (needs e's home and stilldeadcode/radiance:<release>) queues ONE gpuq session: stock and off
#              ident on the release's runtime image (must be equal), an exact KL reference for it
#              (data/kld/ref-r<name>, recorded unless present), int8 quality T2560 and int8 speed T2048
#              scored last-512 paired vs exact (scripts/kl_tail.py, R100's protocol). ~15-20 min of GPU.
#
# Nothing outside this repo's data/, build-*/ and dist/ is written; the radiance checkout is only read.
#
# Env vars:
#   RK_RADIANCE_REPO  a radiance git checkout to read (only read: it must already hold the tag). Unset: this
#                     script keeps its own clone in data/radiance (cloned from RK_RADIANCE_URL on first use, its
#                     tags fetched when the release asked for is not in it yet)
#   RK_RADIANCE_URL   where that clone comes from (default https://codeberg.org/StillDeadcode/radiance.git)
#   RK_MODEL          --gpu-smoke only: the stock published .rad file (required; nothing is guessed)
#   RK_GPUQ           --gpu-smoke only: a command that queues a GPU session (`$RK_GPUQ <label> <cmd...>`), for a
#                     machine shared with other GPU work; unset, the smoke session starts directly
#   RK_CMAKE          cmake >= 3.21 (default: `cmake` on PATH, else ~/.local/bin/cmake)
#   RK_PYTHON         python for pytest (default python3)
#   RK_PROJECTOR      the folder to package and smoke-test (default data/projector-ridgefill-qwen38fn-int8)
#   RK_JOBS           build parallelism (default 4, at nice 19)
set -u

. "$(dirname "$0")/common.sh"

smoke=0 hostonly=0 ci=0 ref=""
for a in "$@"; do
    case $a in
        --gpu-smoke) smoke=1 ;;
        --host-only) hostonly=1 ;;
        --ci)        ci=1 ;;
        -*)          echo "update_radiance: unknown option $a" >&2; exit 2 ;;
        *)           [ -z "$ref" ] || { echo "update_radiance: two releases given ($ref, $a)" >&2; exit 2; }; ref=$a ;;
    esac
done
[ -n "$ref" ] || { echo "usage: scripts/update_radiance.sh <radiance tag or commit> [--host-only] [--ci] [--gpu-smoke]" >&2; exit 2; }
[ $hostonly = 1 ] && [ $smoke = 1 ] && { echo "update_radiance: --gpu-smoke needs the device build --host-only skips" >&2; exit 2; }
RREPO=${RK_RADIANCE_REPO:-$RK_REPO/data/radiance}
RURL=${RK_RADIANCE_URL:-https://codeberg.org/StillDeadcode/radiance.git}
PY=${RK_PYTHON:-python3}
JOBS=${RK_JOBS:-4}
PROJ=${RK_PROJECTOR:-$RK_REPO/data/projector-ridgefill-qwen38fn-int8}

infra() { echo "update_radiance: INFRASTRUCTURE: $*" >&2; [ $ci = 1 ] && echo "::error title=radiance watch: infrastructure::$*"; exit 2; }

cmake_ok() {  # cmake >= 3.21, the minimum of both trees' CMakeLists.txt
    v=$("$1" --version 2>/dev/null | sed -n '1s/^cmake version \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
    [ -n "$v" ] || return 1
    major=${v% *} minor=${v#* }
    [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 21 ]; }
}
CM=""
for c in ${RK_CMAKE:-} cmake "$HOME/.local/bin/cmake"; do cmake_ok "$c" && { CM=$c; break; }; done
[ -n "$CM" ] || infra "no cmake >= 3.21 (set RK_CMAKE)"
if [ -z "${RK_RADIANCE_REPO:-}" ]; then   # our own clone: make it, and fetch when the release is new to it
    [ -d "$RREPO/.git" ] || git clone -q --filter=blob:none "$RURL" "$RREPO" || infra "git clone $RURL into $RREPO"
    git -C "$RREPO" rev-parse --verify --quiet "$ref^{commit}" > /dev/null ||
        git -C "$RREPO" fetch -q --tags origin || infra "git fetch --tags in $RREPO"
fi
git -C "$RREPO" rev-parse --git-dir > /dev/null 2>&1 || infra "$RREPO is not a radiance git checkout (RK_RADIANCE_REPO)"
[ $smoke = 0 ] || [ -f "${RK_MODEL:-}" ] || infra "--gpu-smoke needs RK_MODEL, the stock published .rad file"
[ $smoke = 0 ] || [ -f "$RK_REPO/corpus/quick9.jsonl" ] ||
    infra "--gpu-smoke needs the KL corpus corpus/quick9.jsonl (maintainer data, not in git: docs/REBASING.md)"

# ---------------------------------------------------------------- a. the release
commit=$(git -C "$RREPO" rev-parse --verify --quiet "$ref^{commit}") || infra "'$ref' is not a commit of $RREPO"
short=$(printf '%s' "$commit" | cut -c1-8)
ver=$(git -C "$RREPO" show "$commit:CMakeLists.txt" | sed -n 's/^project(radiance VERSION \([0-9][0-9.]*\).*/\1/p')
[ -n "$ver" ] || infra "no project(radiance VERSION ...) at $commit"
if [ "$(git -C "$RREPO" describe --exact-match --tags "$commit" 2>/dev/null)" = "v$ver" ]; then name=$ver
else name=$ver-g$short; fi
SRC=$RK_REPO/data/radiance-src-$name
LOG=$RK_REPO/data/update-$name
mkdir -p "$LOG" || infra "cannot create $LOG"
: > "$LOG/summary.txt"
say()  { printf '%s\n' "$*" | tee -a "$LOG/summary.txt"; }
step() { printf '  %-14s %s\n' "$1" "$2" | tee -a "$LOG/summary.txt"; }
ann()  { [ $ci = 1 ] && printf '::%s title=%s::%s\n' "$1" "$2" "$3"; return 0; }

pin=$(cut -d' ' -f2 "$RK_REPO/RADIANCE_VERSION" 2>/dev/null || true)
pinver=$(cut -d' ' -f1 "$RK_REPO/RADIANCE_VERSION" 2>/dev/null || true)
[ -n "$pin" ] || infra "RADIANCE_VERSION (the pinned release: '<version> <commit>') is missing"
git -C "$RREPO" cat-file -e "$pin^{commit}" 2>/dev/null || infra "the pinned commit $pin is not in $RREPO (fetch its tags)"
plugin_sha=$(git -C "$RK_REPO" rev-parse HEAD)
say "update_radiance: radiance $ver ($commit, '$ref') for plugin $(git -C "$RK_REPO" rev-parse --short HEAD) ($(git -C "$RK_REPO" branch --show-current 2>/dev/null || echo detached)); pinned $pinver"
say "  logs: $LOG"

# THE ADAPTER FILES A CHANGE POINTS AT: those whose copied in-tree source (arch/*.copies) changed since the pin.
copies() { grep -hv '^#' "$RK_REPO"/arch/*.copies | awk 'NF == 2'; }
suspects=$(copies | while read -r mine theirs; do
    git -C "$RREPO" diff --quiet "$pin" "$commit" -- "$theirs" || echo "$mine"; done | sort -u | tr '\n' ' ')

# f. the report every exit prints
report() {
    say ""
    say "WARNINGS -- radiance files changed $pinver -> $name (abi/, arch/common/, arch/qwen4exp_fp8/):"
    changed=$(git -C "$RREPO" diff --numstat "$pin" "$commit" -- abi arch/common arch/qwen4exp_fp8)
    if [ -z "$changed" ]; then say "  none"
    else
        printf '%s\n' "$changed" | while read -r add del path; do
            who=$(copies | awk -v p="$path" '$2 == p {print $1}' | sort -u | tr '\n' ' ')
            line="$path (+$add -$del)${who:+ -- copied by ${who% }}"
            say "  $line"
            ann warning "radiance $pinver -> $name changed $path" "$line"
        done
    fi
    say "  adapter copies (arch/*.copies):"
    copies | while read -r mine theirs; do
        if git -C "$RREPO" diff --quiet "$pin" "$commit" -- "$theirs"; then st=unchanged; else st=CHANGED; fi
        printf '    %-26s copies %-42s %s\n' "$mine" "$theirs" "$st" | tee -a "$LOG/summary.txt"
    done
}
verdict() {  # word exit-code reason
    report
    say ""
    say "RESULT: $1${3:+ -- $3}"
    [ $2 = 1 ] && ann error "radiance $ver: $1" "${3:-}"
    if [ $ci = 1 ]; then
        { echo "# radiance watch: $ver ($commit) -- $1"; echo; echo '```'; cat "$LOG/summary.txt"; echo '```'; } > "$LOG/report.md"
    fi
    exit "$2"
}
incompatible() { verdict INCOMPATIBLE 1 "$*"; }

if [ -d "$SRC" ]; then
    diffs=$(git -C "$RREPO" archive "$commit" | tar -d -C "$SRC" 2>&1 | grep -vE 'Mod time differs|Mode differs|Uid differs|Gid differs' || true)
    [ -z "$diffs" ] || infra "$SRC exists and is not $commit: $(printf '%s' "$diffs" | head -3 | tr '\n' '|')"
    step source "OK $SRC (existing, verified against $short)"
else
    { mkdir -p "$SRC" && git -C "$RREPO" archive "$commit" | tar -x -C "$SRC" && chmod -R a-w "$SRC"; } ||
        infra "git archive of $commit into $SRC"
    step source "OK $SRC (archived, read-only)"
fi

# ---------------------------------------------------------------- b. the ABI number
abi_of() { git -C "$RREPO" show "$1:abi/rad_abi.h" 2>/dev/null | sed -n 's/^#define[ \t]*RAD_ABI_VERSION[ \t]*\([0-9]*\).*/\1/p'; }
abi_pin=$(abi_of "$pin") abi_new=$(abi_of "$commit")
abi_bad=""
if [ "$abi_pin" = "$abi_new" ]; then step ABI "OK RAD_ABI_VERSION $abi_new (as $pinver)"
else step ABI "CHANGED: RAD_ABI_VERSION $abi_pin ($pinver) -> $abi_new ($ver)"; abi_bad="RAD_ABI_VERSION $abi_pin -> $abi_new"; fi

# ---------------------------------------------------------------- c. builds
HB=$RK_REPO/build-radiance-$name-host
if [ -x "$HB/install/bin/radiance" ] && strings "$HB/install/bin/radiance" | grep -qxF "$ver"; then
    step "host install" "OK $HB/install (existing, carries $ver)"
elif nice -n 19 "$CM" -S "$SRC" -B "$HB" -DRAD_WITH_HIP=OFF -DRAD_WITH_FFMPEG=OFF -DRAD_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HB/install" > "$LOG/radiance-host.log" 2>&1 &&
     nice -n 19 "$CM" --build "$HB" -j "$JOBS" >> "$LOG/radiance-host.log" 2>&1 &&
     "$CM" --install "$HB" >> "$LOG/radiance-host.log" 2>&1; then
    step "host install" "OK $HB/install (built)"
else
    step "host install" "FAIL ($LOG/radiance-host.log): $(grep -m3 -E 'error|Error' "$LOG/radiance-host.log" | tr '\n' '|')"
    report; infra "radiance $ver itself does not build host-only here (toolchain?): $LOG/radiance-host.log"
fi
PB=$RK_REPO/build-host-$name
if ! "$CM" -S "$RK_REPO" -B "$PB" -DCMAKE_PREFIX_PATH="$HB/install" -DRADIANCE_SRC="$SRC" -DRAD_WITH_HIP=OFF \
        > "$LOG/plugin-configure.log" 2>&1; then
    step "plugin build" "FAIL at configure ($LOG/plugin-configure.log):"
    grep -A4 'CMake Error' "$LOG/plugin-configure.log" | head -12 | sed 's/^/      /' | tee -a "$LOG/summary.txt"
    incompatible "the plugin's configure refuses radiance $ver"
fi
if ! nice -n 19 "$CM" --build "$PB" -j "$JOBS" > "$LOG/plugin-build.log" 2>&1; then
    step "plugin build" "FAIL: the plugin does not compile against radiance $ver ($LOG/plugin-build.log). First errors:"
    errs=$(grep -E 'error:' "$LOG/plugin-build.log" | sed "s|$RK_REPO/||" | sort -u | sort -t: -k1,1 -k2,2n | head -12)
    printf '%s\n' "$errs" | sed 's/^/      /' | tee -a "$LOG/summary.txt"
    printf '%s\n' "$errs" | while IFS= read -r e; do ann error "compile" "$e"; done
    files=$(printf '%s\n' "$errs" | cut -d: -f1 | sort -u | tr '\n' ' ')
    incompatible "compile errors in ${files% }${suspects:+; copied sources changed for ${suspects% }}"
fi
step "plugin build" "OK $PB"

# ---------------------------------------------------------------- d. tests
(cd "$PB" && ctest -LE gpu --output-on-failure) > "$LOG/ctest.log" 2>&1
ct=$?
step "ctest -LE gpu" "$( [ $ct = 0 ] && echo OK || echo FAIL): $(grep -E 'tests passed' "$LOG/ctest.log" | tail -1)"
failed_cases=""
for t in arch_static_test adapter_core_test; do
    "$PB/tests/$t" > "$LOG/$t.log" 2>&1
    printf '      %-18s %s\n' "$t" "$(grep -E 'check\(s\) over|failure\(s\)' "$LOG/$t.log" | tail -1)" | tee -a "$LOG/summary.txt"
    for c in $(grep -E '^  FAIL' "$LOG/$t.log" | awk '{print $2}' | sort -u); do
        failed_cases="$failed_cases $c"
        diff1=$(awk -v c="$c" '$1 == "FAIL" && $2 == c {f = 1} f && /first difference/ {print; exit}' "$LOG/$t.log" | sed 's/^ *//')
        line="[$t] $c -> ${suspects:-no copied source changed: a called block (WARNINGS below)}${diff1:+ ($diff1)}"
        say "      FAIL $line"
        ann error "static oracle" "$line"
    done
done
if [ $ct != 0 ]; then
    grep -E '^ *[0-9]+ - .*\(Failed\)' "$LOG/ctest.log" | sed 's/^/      /' | tee -a "$LOG/summary.txt"
    incompatible "static oracle / host tests fail (${failed_cases# }) -- port ${suspects:-the called blocks listed under WARNINGS}"
fi
(cd "$RK_REPO" && "$PY" -m pytest -q tests) > "$LOG/pytest.log" 2>&1
pt=$?
step pytest "$( [ $pt = 0 ] && echo OK || echo FAIL): $(tail -1 "$LOG/pytest.log")"
if [ $pt != 0 ]; then
    grep -qE 'ModuleNotFoundError|ImportError|No module named' "$LOG/pytest.log" &&
        { report; infra "pytest could not import its requirements (pip install -r tools/requirements.txt): $LOG/pytest.log"; }
    incompatible "pytest fails ($LOG/pytest.log)"
fi
[ -z "$abi_bad" ] || incompatible "$abi_bad: a plugin is loaded by ABI number"

# ---------------------------------------------------------------- e. packages
home=""
if [ $hostonly = 1 ]; then
    step package "not built (--host-only)"
else
    BIMG=radiance-build:$ver
    if ! docker image inspect "$BIMG" > /dev/null 2>&1; then
        step package "SKIPPED: no build image $BIMG. Make it: (cd $RREPO && docker/build.sh -r $commit --target build && docker tag radiance-build $BIMG)"
    elif [ -n "$(git -C "$RK_REPO" status --porcelain -- arch kernels tests CMakeLists.txt)" ]; then
        step package "SKIPPED: arch/ kernels/ tests/ or CMakeLists.txt has uncommitted changes; a package is built from a commit"
    else
        hs=$(git -C "$RK_REPO" rev-parse --short HEAD)
        if RK_HOME_TAG=-r$name RK_BUILD_IMAGE=$BIMG RK_RADIANCE_SRC=$SRC "$RK_SCRIPTS/frozen_home.sh" HEAD > "$LOG/frozen_home.log" 2>&1; then
            home=$RK_REPO/data/home-$hs-r$name
            pver=$(sed -n 's/^RAD_ARCH_PLUGIN([^,]*, *"[^"]*", *"[^"]*", *"\([0-9.]*\)".*/\1/p' "$RK_REPO/arch/qwen4exp_ridgefill.cpp")
            abi=$(sed -n 's/^set(PACKAGE_VERSION "\([0-9.]*\)")/\1/p' "$HB"/install/lib*/cmake/radiance/*ersion*.cmake 2>/dev/null | head -1)
            out=$RK_REPO/dist/radiance-ridgefill-r$name-$hs
            if [ -e "$out" ]; then
                step package "SKIPPED: $out exists (remove it to rebuild); device home $home"
            elif "$PY" "$RK_REPO/tools/package.py" --home "$home" --projector "$PROJ" --out "$out" --version "$pver" \
                    --commit "$plugin_sha" --radiance-version "$ver" ${abi:+--abi-version "$abi"} > "$LOG/package.log" 2>&1; then
                step package "OK $out (plugin $pver, radiance $ver, ABI ${abi:-?}; device home $home)"
            else
                step package "FAIL: tools/package.py ($LOG/package.log): $(tail -2 "$LOG/package.log" | tr '\n' '|')"
                report; infra "packaging failed after a compatible verdict: $LOG/package.log"
            fi
        else
            step package "FAIL: the device build in $BIMG ($LOG/frozen_home.log): $(grep -m3 -E 'error|FAIL' "$LOG/frozen_home.log" | tr '\n' '|')"
            incompatible "the device build in $BIMG fails (the host build passed): $LOG/frozen_home.log"
        fi
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
export RK_MODEL=$RK_MODEL
I8="RK_DOCKER_EXTRA=-v $PROJ:/projector:ro"
REF=$RK_REPO/data/kld/ref-r$name
C=$RK_REPO/corpus/quick9.jsonl
T0=\$(date '+%Y-%m-%d %H:%M:%S')
log() { echo "\$*" | tee -a \$E/session.log; }
klog() { if journalctl -k --since "\$T0" | grep -qiE 'amdgpu.*(MES|SMU|timeout|reset)'; then log "STOP: kernel log"; exit 3; fi; }
fin() { docker logs radiance-ridgefill-\$2 > \$E/\$1.serve.log 2>&1; scripts/stop.sh > /dev/null 2>&1; klog; }
srv=\$(docker ps --format '{{.Names}} {{.Image}}' | awk '\$2 !~ /build/ {print \$1}' | tr '\n' ' ')
[ -z "\$srv" ] || { log "STOP: a serving container is running: \$srv"; exit 4; }
log "smoke start \$(date -u +%FT%TZ) boot \$(cat /proc/sys/kernel/random/boot_id) radiance $ver home $home"
scripts/serve.sh exact > \$E/exact.serve.out 2>&1 && scripts/ident.sh > \$E/exact.ident 2>&1; fin exact exact
scripts/serve.sh off > \$E/off.serve.out 2>&1 && scripts/ident.sh > \$E/off.ident 2>&1; fin off off
if [ -s \$E/exact.ident ] && diff -q \$E/exact.ident \$E/off.ident > /dev/null; then log "ident: off EQUALS stock -- PASS"
else log "ident: off DIFFERS from stock (or a serve failed) -- FAIL"; diff \$E/exact.ident \$E/off.ident | head -8 | tee -a \$E/session.log; fi
if [ ! -d \$REF ]; then scripts/grade.sh record \$REF \$C > \$E/ref.out 2>&1 || { log "reference FAILED: \$(tail -2 \$E/ref.out | tr '\n' '|')"; exit 5; }; log "reference recorded: \$REF"; fi
env RADIANCE_RIDGEFILL_TAIL=2560 RADIANCE_RIDGEFILL_PROJECTOR=/projector RADIANCE_RIDGEFILL_SCORE_BULK=1 RK_EXPECT_APPROX=67 "\$I8" scripts/grade.sh quality \$REF \$E/quality-t2560.json > \$E/quality.out 2>&1; log "quality T2560: \$(tail -1 \$E/quality.out)"; klog
env RADIANCE_RIDGEFILL_PROJECTOR=/projector RADIANCE_RIDGEFILL_SCORE_BULK=1 RK_EXPECT_APPROX=67 "\$I8" scripts/grade.sh speed \$REF \$E/speed-t2048.json > \$E/speed.out 2>&1; log "speed T2048: \$(tail -1 \$E/speed.out)"; klog
$PY scripts/kl_tail.py --corpus \$C --last 512 \$E/quality-t2560.json \$E/speed-t2048.json 2>&1 | grep -v '^    ppl/' | tee -a \$E/session.log
log "smoke end \$(date -u +%FT%TZ)"
EOF
        # shellcheck disable=SC2086  # RK_GPUQ is a command and its own words
        nohup setsid ${RK_GPUQ:+$RK_GPUQ "update-$name"} sh "$S" > "$LOG/smoke.queue" 2>&1 < /dev/null &
        sleep 2
        step "gpu smoke" "STARTED${RK_GPUQ:+ (queued with $RK_GPUQ)}; watch $LOG/smoke/session.log"
    fi
fi
verdict COMPATIBLE 0
