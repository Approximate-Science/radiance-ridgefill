#!/bin/sh
# e2e_fresh.sh -- the FRESH-ENGINE end-to-end check of the packaged artifacts. THE RELEASE GATE.
#
# WHAT IT PROVES (PLAN-FIX "DECISIONS RECORDED", FINAL DELIVERABLE): the tarballs tools/package.py
# built install and behave on a stock radiance: a stock stilldeadcode/radiance runtime image, the
# stock published .rad container, a clean RADIANCE_HOME holding ONLY the packaged plugin, the
# projector folder unpacked from the package, the template produced by the merge tool.
#
# THE MODEL FILE IS NEVER COPIED (it is 114 GiB): <work>/model/<basename> is a SYMLINK to RK_MODEL,
# for this script's own host-side reference and evidence. Docker bind mounts do NOT resolve
# symlink targets across the mount (the target path would not exist inside the container), so
# every container mounts the model's REAL directory at /models (scripts/common.sh's rule), and
# the projector folder is put beside the model INSIDE the container through a second bind mount
# at /models/projector (the loader's discovery order: $RADIANCE_KVA_PROJECTOR -> beside --model
# -> beside the resolved file; the mount makes /models/projector the folder discovery finds).
#
# THE CASES (every run carries the same engine flags as scripts/common.sh RK_FLAGS):
#   0  the stock-engine baseline: image's own plugin home, no projector. ident.sh's six hashes
#      and one 16,384-token TTFT -- the numbers every other case is compared against.
#   1  no projector mounted (an EMPTY dir shadows any real projector/ the model dir may hold):
#      the plugin logs the no-projector line and serves stock -> ident EQUALS the baseline.
#   2  projector mounted, RADIANCE_KVA=off: off never looks -> ident EQUALS the baseline.
#   3  RADIANCE_KVA=quality: the startup log has the projector line with 0 warning(s), and a
#      16,384-token prompt (tools/speed.py's prompt builder) is FASTER than the baseline TTFT.
#   4  RADIANCE_KVA=speed: the same checks.
#   5  a corrupted projector copy (one byte flipped in one proj file, in a scratch copy): the
#      startup REFUSES that folder BY NAME and serves stock -> ident EQUALS the baseline.
#   6  --override-chat-template with the packaged pre-merged template and NO kva kwarg:
#      requests render byte-identically to the unmodified template -> ident EQUALS the baseline.
#   7  the same with "chat_template_kwargs": {"kva": "on"}: PENDING until Stage F lands (the
#      marker erasure) -- SKIPPED with that reason unless RK_E2E_STAGE_F=1, which expects the
#      request to succeed and the log to show the marker detected.
#
# OUTPUT: <work>/e2e-report.json (every case with pass/fail/skipped and its evidence); exit
# status 0 only when no case failed. Every container this script starts is stopped and removed
# (EXIT/INT/TERM trap), so a failed gate leaves no server behind.
#
# USAGE
#   RK_DIST=<dist dir from tools/package.py> RK_MODEL=<stock published .rad> \
#   RK_E2E_WORK=<scratch dir to create> scripts/e2e_fresh.sh
#
# Env vars (defaults in scripts/common.sh unless noted):
#   RK_DIST               required, no default: the dist/ tools/package.py wrote
#   RK_MODEL              required, no default: the stock published .rad container
#   RK_E2E_WORK           required, no default: the scratch dir this script creates (evidence
#                         stays there; refuse if it exists non-empty)
#   RK_IMAGE              stilldeadcode/radiance:1.0.8
#   RK_PORT               8100
#   RK_FLAGS              the shared engine flags (common.sh); every case runs with them
#   RK_E2E_SERVE_TIMEOUT  default 1800 s: how long to wait for /health a case (114 GiB load)
#   RK_E2E_TTFT_LEN       default 16384: the TTFT prompt length
#   RK_E2E_TTFT_REPS      default 3: timed reps after one warm-up (median is compared)
#   RK_E2E_PREFLIGHT     default 1: run scripts/preflight.sh before each container; 0 skips it
#                         (debug only -- a gate number next to a squatter is not a number)
#   RK_E2E_MODEL_SHA256   default the published sha256 of the stock container
#                         (0af5e962...aa4d20, notes/aprime.md R144): refuse a model file that is
#                         not it. Set to an empty string to skip the (slow) hash of 114 GiB.
#   RK_E2E_STAGE_F        default unset: case 7 prints SKIPPED; set to 1 to run it (needs the
#                         Stage F marker erasure in the plugin)

set -eu

. "$(dirname "$0")/common.sh"

# ---------------------------------------------------------------- inputs and layout

[ -n "${RK_DIST:-}" ] || rk_die "RK_DIST is not set: the dist directory tools/package.py wrote"
[ -d "$RK_DIST" ] || rk_die "RK_DIST is not a directory: $RK_DIST"
rk_require_model
[ -n "${RK_E2E_WORK:-}" ] || rk_die "RK_E2E_WORK is not set: the scratch dir to create"
for e_cmd in docker curl sha256sum tar python3 jq; do
    command -v "$e_cmd" >/dev/null 2>&1 || rk_die "$e_cmd is not on PATH"
done
if [ -e "$RK_E2E_WORK" ] && [ -n "$(ls -A "$RK_E2E_WORK" 2>/dev/null)" ]; then
    rk_die "RK_E2E_WORK exists and is not empty: $RK_E2E_WORK (the gate creates it fresh)"
fi

: "${RK_E2E_SERVE_TIMEOUT:=1800}"
: "${RK_E2E_TTFT_LEN:=16384}"
: "${RK_E2E_TTFT_REPS:=3}"
: "${RK_E2E_PREFLIGHT:=1}"
: "${RK_E2E_STAGE_F:=0}"
: "${RK_E2E_MODEL_SHA256:=0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20}"
E2E_TTFT_LEN=$RK_E2E_TTFT_LEN
E2E_TTFT_REPS=$RK_E2E_TTFT_REPS
export RK_FLAGS RK_MODEL RK_DIST RK_E2E_TTFT_LEN RK_E2E_TTFT_REPS   # the report assembler reads them

# a leftover server would fight for the cards and the port
e_running=$(docker ps --filter name=radiance-kva- --format '{{.Names}}' 2>/dev/null || true)
[ -z "$e_running" ] || rk_die "refusing to start: a radiance-kva container is already running: $e_running"

if ! mkdir -p "$RK_E2E_WORK" 2>/dev/null; then
    rk_die "cannot create RK_E2E_WORK: $RK_E2E_WORK"
fi
E2E_WORK=$(CDPATH= cd "$RK_E2E_WORK" && pwd) || rk_die "cannot enter RK_E2E_WORK: $RK_E2E_WORK"
mkdir "$E2E_WORK/extract" "$E2E_WORK/model" "$E2E_WORK/logs" "$E2E_WORK/ident" "$E2E_WORK/cases"
E2E_EXTRACT=$E2E_WORK/extract
E2E_LOGS=$E2E_WORK/logs
E2E_IDENT=$E2E_WORK/ident
E2E_CASES=$E2E_WORK/cases
E2E_EMPTY_PROJECTOR=$E2E_WORK/empty-projector
mkdir "$E2E_EMPTY_PROJECTOR"      # shadows any real projector/ beside the model (case 1)

printf 'e2e: work %s; dist %s; image %s; model %s\n' "$E2E_WORK" "$RK_DIST" "$RK_IMAGE" "$RK_MODEL"

# ---------------------------------------------------------------- (a) extract + verify the packages

# the tarballs' own hashes first, then each directory's SHA256SUMS after extraction:
# nothing runs against an artifact that does not verify
[ -f "$RK_DIST/SHA256SUMS" ] || rk_die "the dist dir has no SHA256SUMS: $RK_DIST"
(cd "$RK_DIST" && sha256sum -c SHA256SUMS) || rk_die "the dist tarballs do not match $RK_DIST/SHA256SUMS"

set -- "$RK_DIST"/radiance-kva-*.tar.gz
[ "$#" -eq 1 ] || rk_die "expected exactly one radiance-kva-<version>.tar.gz in $RK_DIST, found: $*"
E2E_PLUGIN_TARBALL=$1
for e_tb in "$E2E_PLUGIN_TARBALL" "$RK_DIST/projector-qwen3.8-flash-next.tar.gz" "$RK_DIST/kva-chat-template.tar.gz"; do
    [ -f "$e_tb" ] || rk_die "the dist dir lacks the packaged tarball: $e_tb"
    e_root=$(tar -tzf "$e_tb" | head -1)
    e_want=$(basename "$e_tb" .tar.gz)/
    [ "$e_root" = "$e_want" ] || rk_die "$e_tb does not contain a single root dir $e_want (first entry: $e_root)"
    tar -xzf "$e_tb" -C "$E2E_EXTRACT"
done

E2E_PLUGIN_VERSION=$(basename "$E2E_PLUGIN_TARBALL" .tar.gz)
E2E_PLUGIN_VERSION=${E2E_PLUGIN_VERSION#radiance-kva-}
E2E_PLUGIN_HOME=$E2E_EXTRACT/radiance-kva-$E2E_PLUGIN_VERSION
E2E_PROJECTOR=$E2E_EXTRACT/projector-qwen3.8-flash-next
E2E_TEMPLATE_DIR=$E2E_EXTRACT/kva-chat-template
export RK_PLUGIN_HOME="$E2E_PLUGIN_HOME"     # rk_docker_prefix mounts it at /plugins:ro
[ -f "$E2E_PLUGIN_HOME/architectures/qwen4exp_fp8.so" ] || rk_die "the packaged plugin home has no architectures/qwen4exp_fp8.so: $E2E_PLUGIN_HOME"
[ -f "$E2E_PLUGIN_HOME/kernels/kva.so" ] || rk_die "the packaged plugin home has no kernels/kva.so: $E2E_PLUGIN_HOME"
[ -f "$E2E_TEMPLATE_DIR/chat_template.jinja" ] || rk_die "the template package has no chat_template.jinja: $E2E_TEMPLATE_DIR"
for e_dir in "$E2E_PLUGIN_HOME" "$E2E_PROJECTOR" "$E2E_TEMPLATE_DIR"; do
    (cd "$e_dir" && sha256sum -c SHA256SUMS) ||
        rk_die "the extracted package does not match its SHA256SUMS: $e_dir"
done
printf 'e2e: extracted and verified radiance-kva-%s, projector-qwen3.8-flash-next, kva-chat-template\n' "$E2E_PLUGIN_VERSION"

# ---------------------------------------------------------------- (b) the model: a symlink, never a copy

# <work>/model/<basename> is a SYMLINK to the real file (host-side reference only -- it would
# dangle inside a bind mount, which is why the containers mount the REAL directory at /models).
mkdir -p "$E2E_WORK/model"
ln -s "$(readlink -f "$RK_MODEL")" "$E2E_WORK/model/$(basename "$RK_MODEL")"

# the model must be the stock published container (unless RK_E2E_MODEL_SHA256 is emptied)
if [ -n "$RK_E2E_MODEL_SHA256" ]; then
    printf 'e2e: verifying the model sha256 (%s is 114 GiB; this takes minutes)\n' "$(basename "$RK_MODEL")"
    e_got=$(sha256sum "$RK_MODEL" | cut -d' ' -f1) || rk_die "sha256sum failed on $RK_MODEL"
    if [ "$e_got" != "$RK_E2E_MODEL_SHA256" ]; then
        rk_die "the model is not the stock published container: sha256 $e_got, expected $RK_E2E_MODEL_SHA256 ($RK_MODEL); set RK_E2E_MODEL_SHA256 only if you know better"
    fi
    printf '%s\n' "$e_got" > "$E2E_WORK/model-sha256"
    printf 'e2e: model sha256 ok (%s)\n' "$e_got"
else
    printf 'not verified (RK_E2E_MODEL_SHA256 empty)\n' > "$E2E_WORK/model-sha256"
fi

# ---------------------------------------------------------------- the helpers

# e2e_die REASON -- record the abort, stop every container, write the report, exit 1.
e2e_die() {
    printf 'radiance-kva e2e: FAIL: %s\n' "$*" >&2
    printf '%s\n' "$*" > "$E2E_CASES/abort.reason"
    e2e_cleanup
    e2e_report
    exit 1
}

# stop and remove every container this script started (the trap and e2e_die both call this)
E2E_STARTED=''
e2e_cleanup() {
    for e_c in $E2E_STARTED; do
        printf 'e2e: stopping %s\n' "$e_c"
        docker stop -t 60 "$e_c" >/dev/null 2>&1 || true
        docker rm "$e_c" >/dev/null 2>&1 || true
    done
    E2E_STARTED=''
}
trap e2e_cleanup EXIT INT TERM

# e2e_ev ID LINE -- append one evidence line to the case's record.
e2e_ev() {
    printf '%s\n' "$2" >> "$E2E_CASES/$1.evidence"
}

# e2e_status ID TITLE RC -- record the case pass/fail (RC 0 = pass).
e2e_status() {
    e_st=pass
    [ "$3" -eq 0 ] || e_st=fail
    printf '%s\n%s\n' "$e_st" "$2" > "$E2E_CASES/$1.meta"
    printf 'e2e: case %s: %s -> %s\n' "$1" "$2" "$e_st"
}

# e2e_skipped ID TITLE REASON -- record a case that did not run.
e2e_skipped() {
    printf 'skipped\n%s\n' "$2" > "$E2E_CASES/$1.meta"
    e2e_ev "$1" "SKIPPED: $3"
    printf 'e2e: case %s: %s -> SKIPPED (%s)\n' "$1" "$2" "$3"
}

# e2e_serve NAME MODE PROJECTOR TEMPLATE [extra engine args...]
#   start container radiance-kva-e2e-<NAME> with the standard prefix (common.sh), the same
#   RK_FLAGS as every other measurement, the packaged plugin home as the only plugin home
#   (exact: the image's own home, no plugin), plus the projector / template bind mounts;
#   wait for /health. The whole command line is kept in logs/<NAME>.cmd.
e2e_serve() {
    e_name=$1; e_mode=$2; e_proj=$3; e_tmpl=$4; shift 4
    rk_mode_validate "$e_mode"
    if [ "$RK_E2E_PREFLIGHT" -eq 1 ]; then
        "$RK_SCRIPTS/preflight.sh" > "$E2E_LOGS/$e_name.preflight.txt" 2>&1 ||
            e2e_die "preflight failed before case $e_name; nothing was started"
        e2e_ev "$e_name" "preflight: OK ($(cat "$E2E_LOGS/$e_name.preflight.txt" | tail -1))"
    fi
    e_container=radiance-kva-e2e-$e_name
    e_args=$E2E_LOGS/$e_name.cmd
    {
        printf '%s\n' -d --name "$e_container"
        rk_docker_prefix "$e_mode"
        if [ -n "$e_proj" ]; then printf '%s\n' -v "$e_proj":/models/projector:ro; fi
        if [ -n "$e_tmpl" ]; then printf '%s\n' -v "$e_tmpl":/templates:ro; fi
        printf '%s\n' "$RK_IMAGE" --model "/models/$(basename "$RK_MODEL")"
        # shellcheck disable=SC2086  # RK_FLAGS is one flag or value per word by construction
        printf '%s\n' $RK_FLAGS
        printf '%s\n' --host 0.0.0.0 --port "$RK_PORT" --max-num-seqs 8
        if [ -n "$e_tmpl" ]; then printf '%s\n' --override-chat-template /templates/chat_template.jinja; fi
        if [ "$#" -gt 0 ]; then printf '%s\n' "$@"; fi
    } > "$e_args"
    e2e_ev "$e_name" "docker command line (one argument per line): logs/$e_name.cmd"
    set --
    while IFS= read -r e_arg; do
        [ -n "$e_arg" ] || continue
        set -- "$@" "$e_arg"
    done < "$e_args"
    if ! e_cid=$(docker run "$@" 2>&1); then
        e2e_die "docker run failed for case $e_name: $e_cid (command line: $e_args)"
    fi
    E2E_STARTED="$E2E_STARTED $e_container"
    printf 'e2e: case %s: container %s (id %s) starting; waiting for /health on port %s\n' \
        "$e_name" "$e_container" "$e_cid" "$RK_PORT"
    e_waited=0
    while :; do
        e_code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' \
            "http://127.0.0.1:$RK_PORT/health" 2>/dev/null || true)
        [ "$e_code" = 200 ] && break
        e_state=$(docker container inspect -f '{{.State.Running}}' "$e_container" 2>/dev/null || true)
        [ "$e_state" = true ] ||
            e2e_die "container $e_container exited before /health answered; log: docker logs $e_container"
        if [ "$e_waited" -ge "$RK_E2E_SERVE_TIMEOUT" ]; then
            e2e_die "/health did not answer within ${RK_E2E_SERVE_TIMEOUT}s for case $e_name"
        fi
        sleep 5
        e_waited=$((e_waited + 5))
        if [ $((e_waited % 30)) -eq 0 ]; then printf '  ...still waiting after %ss\n' "$e_waited"; fi
    done
    e2e_ev "$e_name" "server up: container $e_container, port $RK_PORT, mode $e_mode"
}

# e2e_stop NAME -- capture the container's whole log (the evidence the log checks grep),
# then stop and remove it.
e2e_stop() {
    e_container=radiance-kva-e2e-$1
    docker logs "$e_container" > "$E2E_LOGS/$1.log" 2>&1 ||
        e2e_die "docker logs failed for $e_container"
    e2e_ev "$1" "container log: logs/$1.log"
    docker stop -t 60 "$e_container" >/dev/null ||
        e2e_die "docker stop failed for $e_container"
    docker rm "$e_container" >/dev/null ||
        e2e_die "docker rm failed for $e_container"
    e_new=''
    for e_c in $E2E_STARTED; do [ "$e_c" = "$e_container" ] || e_new="$e_new $e_c"; done
    E2E_STARTED=${e_new# }
}

# e2e_ident ID LABEL -- run scripts/ident.sh, keep ident/<LABEL>.ident, evidence the output.
e2e_ident() {
    if ! "$RK_SCRIPTS/ident.sh" > "$E2E_IDENT/$2.ident" 2> "$E2E_LOGS/$2.ident.err"; then
        e2e_ev "$1" "FAIL: scripts/ident.sh exited non-zero (stderr in logs/$2.ident.err)"
        return 1
    fi
    e2e_ev "$1" "ident output (ident/$2.ident):"
    while IFS= read -r e_line; do
        e2e_ev "$1" "  $e_line"
    done < "$E2E_IDENT/$2.ident"
    return 0
}

# e2e_ident_equals_stock ID LABEL -- the six hashes must equal the stock baseline's.
e2e_ident_equals_stock() {
    if cmp -s "$E2E_IDENT/stock.ident" "$E2E_IDENT/$2.ident"; then
        e2e_ev "$1" "PASS: the six ident hashes equal the stock baseline (ident/stock.ident)"
        return 0
    fi
    e2e_ev "$1" "FAIL: the ident hashes differ from the stock baseline:"
    diff "$E2E_IDENT/stock.ident" "$E2E_IDENT/$2.ident" | while IFS= read -r e_line; do
        e2e_ev "$1" "  $e_line"
    done
    return 1
}

# e2e_ttft -- one warm-up + RK_E2E_TTFT_REPS timed prefills of exactly RK_E2E_TTFT_LEN tokens
# (tools/speed.py's own prompt builder: nonce + docs tokenised by the served model's /tokenize,
# cut to the exact length, cached-token count checked), median printed on stdout.
e2e_ttft() {
    python3 - "$RK_TOOLS" "$RK_PORT" "$E2E_TTFT_LEN" "$E2E_TTFT_REPS" <<'PY'
import os, statistics, sys

tools, port, length, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, tools)
import speed  # tools/speed.py: standard-library-only, no server contact at import

speed.BASE = f"http://127.0.0.1:{port}"
speed.HTTP_TIMEOUT = float(os.environ.get("RK_E2E_TTFT_HTTP_TIMEOUT", "3600"))

# the filler: prose, grown until one /tokenize covers the whole target length
filler = ("The KVA end-to-end gate fills its prefill prompt with plain prose like this, "
          "tokenised by the served model's own tokenizer. ") * 64
doc_ids = [speed.tokenize(filler)]
while sum(len(d) for d in doc_ids) < length:
    filler += filler
    doc_ids = [speed.tokenize(filler)]

times = []
for n in range(reps + 1):                      # rep 0 is the warm-up, like scripts/speed.sh
    nonce = f"e2e nonce {n}.\n"
    rec = speed.measure_once(length, nonce, speed.tokenize(nonce), doc_ids)
    tag = 'warmup' if n == 0 else f'rep {n}'
    print(f"  ttft {tag}: {rec['prompt_ms']:.1f} ms "
          f"({rec['prompt_tokens']} tokens, cached {rec['cached_tokens']})", file=sys.stderr)
    if n:
        times.append(rec["prompt_ms"])
print(f"{statistics.median(times):.1f}")
PY
}

# e2e_report -- assemble <work>/e2e-report.json from the per-case records; a case the gate
# never reached is recorded as failed. Prints the summary; writes failed-count.
e2e_report() {
    python3 - "$E2E_WORK" <<'PY'
import json, os, sys
from pathlib import Path

work = Path(sys.argv[1])
ids = ["0", "1", "2", "3", "4", "5", "6", "7"]
titles = {
    "0": "stock baseline (image's own plugin home, no projector)",
    "1": "no projector mounted: plugin serves stock (ident == stock)",
    "2": "projector + RADIANCE_KVA=off: ident == stock",
    "3": "RADIANCE_KVA=quality: projector 0-warning line + TTFT faster than stock",
    "4": "RADIANCE_KVA=speed: projector 0-warning line + TTFT faster than stock",
    "5": "corrupted projector: refused by name, ident == stock",
    "6": "--override-chat-template, no kva kwarg: ident == stock",
    "7": "--override-chat-template + kva:on request (Stage F)",
}

def read(name):
    path = work / name
    return path.read_text().strip() if path.is_file() else None

def lines(name):
    path = work / name
    return path.read_text().splitlines() if path.is_file() else []

cases, failed = [], []
for cid in ids:
    meta = work / "cases" / f"{cid}.meta"
    if meta.is_file():
        status, name = meta.read_text().splitlines()[:2]
    else:
        status, name = "fail", titles[cid] + " -- not run (the gate aborted earlier)"
    ev_path = work / "cases" / f"{cid}.evidence"
    cases.append({"id": cid, "name": name, "status": status,
                  "evidence": ev_path.read_text().splitlines() if ev_path.is_file() else []})
    if status == "fail":
        failed.append(cid)

stock_ttft = read("stock-ttft-ms")
report = {
    "tool": "scripts/e2e_fresh.sh",
    "gate": "fresh-engine end-to-end check of the packaged release (PLAN-FIX FINAL DELIVERABLE)",
    "image": os.environ.get("RK_IMAGE"),
    "model": os.environ.get("RK_MODEL"),
    "model_sha256": read("model-sha256") or "not verified",
    "dist": os.environ.get("RK_DIST"),
    "work": str(work),
    "rk_flags": os.environ.get("RK_FLAGS"),
    "ttft_length": os.environ.get("RK_E2E_TTFT_LEN"),
    "ttft_reps": os.environ.get("RK_E2E_TTFT_REPS"),
    "stock_ident_file": "ident/stock.ident",
    "stock_ident": lines("ident/stock.ident"),
    "stock_ttft_ms": float(stock_ttft) if stock_ttft else None,
    "stage_f": os.environ.get("RK_E2E_STAGE_F", "0") == "1",
    "cases": cases,
    "failed": failed,
    "all_passed": not failed,
}
(work / "e2e-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
(work / "failed-count").write_text(f"{len(failed)}\n")

print("e2e: report written to", work / "e2e-report.json")
for case in cases:
    print(f"  case {case['id']}: {case['status']:<7} {case['name']}")
print(f"e2e: {len(failed)} failed case(s)")
PY
}

# ---------------------------------------------------------------- step 0: the stock baseline

printf 'e2e: step 0: the stock engine (no plugin, no projector)\n'
e2e_serve 0 exact '' ''
if ! e2e_ident 0 stock; then
    e2e_die "scripts/ident.sh failed against the stock engine"
fi
if ! E2E_STOCK_TTFT=$(e2e_ttft); then
    e2e_die "the stock TTFT measurement failed"
fi
printf '%s\n' "$E2E_STOCK_TTFT" > "$E2E_WORK/stock-ttft-ms"
e2e_ev 0 "stock TTFT at ${E2E_TTFT_LEN} tokens (median of ${E2E_TTFT_REPS} reps): ${E2E_STOCK_TTFT} ms"
e2e_stop 0
e2e_status 0 "stock baseline taken (ident + TTFT)" 0

# ---------------------------------------------------------------- case 1: no projector mounted

e2e_serve 1 quality "$E2E_EMPTY_PROJECTOR" ''
e2e_ident 1 1 || e2e_die "scripts/ident.sh failed in case 1"
e2e_stop 1
e_rc=0
if grep -q 'KVA: no projector folder' "$E2E_LOGS/1.log"; then
    e2e_ev 1 "PASS: the log has the no-projector line:"
    grep 'KVA: no projector folder' "$E2E_LOGS/1.log" | while IFS= read -r e_line; do
        e2e_ev 1 "  $e_line"
    done
else
    e2e_ev 1 "FAIL: no 'KVA: no projector folder' line in logs/1.log (the plugin did not report serving stock)"
    e_rc=1
fi
e2e_ident_equals_stock 1 1 || e_rc=1
e2e_status 1 "no projector mounted: plugin serves stock (ident == stock)" "$e_rc"

# ---------------------------------------------------------------- case 2: projector + RADIANCE_KVA=off

e2e_serve 2 off "$E2E_PROJECTOR" ''
e2e_ident 2 2 || e2e_die "scripts/ident.sh failed in case 2"
e2e_stop 2
e_rc=0
e2e_ident_equals_stock 2 2 || e_rc=1
e2e_status 2 "projector + RADIANCE_KVA=off: ident == stock" "$e_rc"

# ---------------------------------------------------------------- case 3: RADIANCE_KVA=quality

e2e_serve 3 quality "$E2E_PROJECTOR" ''
if ! E2E_TTFT=$(e2e_ttft); then
    e2e_ev 3 "FAIL: the TTFT measurement failed against the quality server"
    E2E_TTFT=''
fi
e2e_stop 3
e_rc=0
if grep -Eq 'KVA: projector .* matches .* 0 warning\(s\)' "$E2E_LOGS/3.log"; then
    e2e_ev 3 "PASS: the startup log has the projector line with 0 warning(s):"
    grep 'KVA: projector' "$E2E_LOGS/3.log" | while IFS= read -r e_line; do
        e2e_ev 3 "  $e_line"
    done
else
    e2e_ev 3 "FAIL: no 'KVA: projector ... matches ... 0 warning(s)' line in logs/3.log"
    e_rc=1
fi
if [ -n "$E2E_TTFT" ]; then
    e2e_ev 3 "quality TTFT at ${E2E_TTFT_LEN} tokens (median of ${E2E_TTFT_REPS} reps): ${E2E_TTFT} ms (stock ${E2E_STOCK_TTFT} ms)"
    if awk -v a="$E2E_TTFT" -v b="$E2E_STOCK_TTFT" 'BEGIN{exit !(a < b)}'; then
        e2e_ev 3 "PASS: quality is faster than stock at ${E2E_TTFT_LEN} tokens"
    else
        e2e_ev 3 "FAIL: quality is not faster than stock (${E2E_TTFT} ms >= ${E2E_STOCK_TTFT} ms)"
        e_rc=1
    fi
else
    e_rc=1
fi
e2e_status 3 "RADIANCE_KVA=quality: projector 0-warning line + TTFT faster than stock" "$e_rc"

# ---------------------------------------------------------------- case 4: RADIANCE_KVA=speed

e2e_serve 4 speed "$E2E_PROJECTOR" ''
if ! E2E_TTFT=$(e2e_ttft); then
    e2e_ev 4 "FAIL: the TTFT measurement failed against the speed server"
    E2E_TTFT=''
fi
e2e_stop 4
e_rc=0
if grep -Eq 'KVA: projector .* matches .* 0 warning\(s\)' "$E2E_LOGS/4.log"; then
    e2e_ev 4 "PASS: the startup log has the projector line with 0 warning(s):"
    grep 'KVA: projector' "$E2E_LOGS/4.log" | while IFS= read -r e_line; do
        e2e_ev 4 "  $e_line"
    done
else
    e2e_ev 4 "FAIL: no 'KVA: projector ... matches ... 0 warning(s)' line in logs/4.log"
    e_rc=1
fi
if [ -n "$E2E_TTFT" ]; then
    e2e_ev 4 "speed TTFT at ${E2E_TTFT_LEN} tokens (median of ${E2E_TTFT_REPS} reps): ${E2E_TTFT} ms (stock ${E2E_STOCK_TTFT} ms)"
    if awk -v a="$E2E_TTFT" -v b="$E2E_STOCK_TTFT" 'BEGIN{exit !(a < b)}'; then
        e2e_ev 4 "PASS: speed is faster than stock at ${E2E_TTFT_LEN} tokens"
    else
        e2e_ev 4 "FAIL: speed is not faster than stock (${E2E_TTFT} ms >= ${E2E_STOCK_TTFT} ms)"
        e_rc=1
    fi
else
    e_rc=1
fi
e2e_status 4 "RADIANCE_KVA=speed: projector 0-warning line + TTFT faster than stock" "$e_rc"

# ---------------------------------------------------------------- case 5: a corrupted projector copy

# a scratch copy with ONE BYTE flipped in one proj file: the folder must be refused BY NAME
# (its manifest no longer matches) and the engine must serve stock
E2E_CORRUPT=$E2E_WORK/corrupt-projector
rm -rf "$E2E_CORRUPT"
cp -R "$E2E_PROJECTOR" "$E2E_CORRUPT"
set -- "$E2E_CORRUPT"/proj.*.safetensors
if [ ! -f "$1" ]; then
    e2e_die "no proj.*.safetensors in $E2E_PROJECTOR (cannot build the corrupted copy)"
fi
E2E_CORRUPT_FILE=$(basename "$1")
python3 - "$1" <<'PY'
import sys
path = sys.argv[1]
with open(path, "r+b") as f:
    f.seek(-1, 2)
    byte = f.read(1)
    f.seek(-1, 2)
    f.write(bytes([byte[0] ^ 0xFF]))
PY
printf 'e2e: corrupted %s in %s (last byte flipped)\n' "$E2E_CORRUPT_FILE" "$E2E_CORRUPT"
e2e_ev 5 "corrupted copy: $E2E_CORRUPT ($E2E_CORRUPT_FILE, last byte flipped)"
e2e_serve 5 quality "$E2E_CORRUPT" ''
e2e_ident 5 5 || e2e_die "scripts/ident.sh failed in case 5"
e2e_stop 5
e_rc=0
if grep -q 'REFUSED' "$E2E_LOGS/5.log" && grep -q "$E2E_CORRUPT_FILE" "$E2E_LOGS/5.log" \
    && grep -q 'serving stock' "$E2E_LOGS/5.log"; then
    e2e_ev 5 "PASS: the log refuses the folder naming $E2E_CORRUPT_FILE and serves stock:"
    grep 'KVA: projector' "$E2E_LOGS/5.log" | while IFS= read -r e_line; do
        e2e_ev 5 "  $e_line"
    done
else
    e2e_ev 5 "FAIL: the log does not refuse $E2E_CORRUPT_FILE by name (no REFUSED + file + 'serving stock' lines in logs/5.log)"
    e_rc=1
fi
e2e_ident_equals_stock 5 5 || e_rc=1
e2e_status 5 "corrupted projector: refused by name, ident == stock" "$e_rc"

# ---------------------------------------------------------------- case 6: the template, no kva kwarg

# --override-chat-template with the packaged pre-merged template; requests WITHOUT the kva
# kwarg must render byte-identically to the unmodified template -> the stock hashes
e2e_serve 6 off "$E2E_PROJECTOR" "$E2E_TEMPLATE_DIR"
e2e_ident 6 6 || e2e_die "scripts/ident.sh failed in case 6"
e2e_stop 6
e_rc=0
if grep -q 'chat_template.jinja' "$E2E_LOGS/6.log"; then
    e2e_ev 6 "PASS: the log names the override template:"
    grep 'chat_template.jinja' "$E2E_LOGS/6.log" | while IFS= read -r e_line; do
        e2e_ev 6 "  $e_line"
    done
else
    e2e_ev 6 "NOTE: logs/6.log does not name chat_template.jinja (not gated; the engine may not log the override path)"
fi
e2e_ident_equals_stock 6 6 || e_rc=1
e2e_status 6 "--override-chat-template, no kva kwarg: ident == stock" "$e_rc"

# ---------------------------------------------------------------- case 7: the kva:on request (Stage F)

if [ "$RK_E2E_STAGE_F" != 1 ]; then
    e2e_skipped 7 "--override-chat-template + kva:on request (Stage F)" \
        "PENDING: Stage F (the per-request marker erasure) has not landed; set RK_E2E_STAGE_F=1 to run it once it has"
else
    e2e_serve 7 off "$E2E_PROJECTOR" "$E2E_TEMPLATE_DIR"
    printf '%s' '{"model":"m","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"kva":"on"}}' \
        > "$E2E_WORK/case7-request.json"
    e_code=$(curl -s -m 900 -o "$E2E_WORK/case7-response.json" -w '%{http_code}' \
        -H 'Content-Type: application/json' -d @"$E2E_WORK/case7-request.json" \
        "http://127.0.0.1:$RK_PORT/v1/chat/completions" || true)
    e2e_stop 7
    e_rc=0
    if [ "$e_code" = 200 ]; then
        e2e_ev 7 "PASS: the kva:on request succeeded (HTTP 200, response in case7-response.json)"
    else
        e2e_ev 7 "FAIL: the kva:on request did not succeed (HTTP ${e_code:-none})"
        e_rc=1
    fi
    if grep -qi 'marker' "$E2E_LOGS/7.log"; then
        e2e_ev 7 "PASS: the log shows the marker detected:"
        grep -i 'marker' "$E2E_LOGS/7.log" | while IFS= read -r e_line; do
            e2e_ev 7 "  $e_line"
        done
    else
        e2e_ev 7 "FAIL: no marker line in logs/7.log"
        e_rc=1
    fi
    e2e_status 7 "--override-chat-template + kva:on request (Stage F)" "$e_rc"
fi

# ---------------------------------------------------------------- the report and the verdict

e2e_report
e_failures=$(cat "$E2E_WORK/failed-count")
if [ "$e_failures" -eq 0 ]; then
    printf 'e2e: ALL CASES PASSED (case 7 skipped does not fail the gate); the release is verified end to end\n'
    exit 0
fi
printf 'radiance-kva e2e: FAIL: %s case(s) failed; see %s/e2e-report.json\n' "$e_failures" "$E2E_WORK" >&2
exit 1