#!/usr/bin/env python3
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""ridgefill_report.py -- one paste-able report for a RidgeFill problem.

Run it on the machine that serves, point it at the server's log, and paste the output into the issue:

    python3 ridgefill_report.py --log server.log                      # a log file
    python3 ridgefill_report.py --container my-radiance               # docker logs of a container
    python3 ridgefill_report.py --log server.log --url http://127.0.0.1:8000 --projector <model dir>/projector

It reads, never changes: the log, this plugin folder (VERSION.json, .so hashes), the projector folder (manifest,
file hashes), /proc and /sys (RAM, pinned-memory cap, GPUs, PCIe links), RADIANCE_* variables, the kernel log's
amdgpu lines when readable, and the server's /version when --url is given. The first section is a DIAGNOSIS: every
plugin line in the log is matched against the catalog below, which names the cause and the fix (codes RF-xxx,
docs/TROUBLESHOOTING.md). Home directory, user name and host name are replaced by <home>, <user>, <host>.
Python 3.8+, standard library only. Exit 0 when the report was written, 2 on bad arguments.
"""
import argparse, getpass, glob, hashlib, json, os, platform, re, socket, subprocess, sys, urllib.request

# ------------------------------------------------------------------------------------------- the catalog
# (code, level, regex on one log line, what it means, what to do). First match wins, so specific before general.
# tests/test_ridgefill_report.py checks that every message the plugin's C++ can print matches an entry here.
CATALOG = [
    ("RF-001", "ok", r"RidgeFill: projector .* matches ",
     "RidgeFill loaded and the projector matches this model.", ""),
    ("RF-002", "ok", r"RidgeFill: rank \d+ holds the projector",
     "The projector is in host memory on this rank.", ""),
    ("RF-003", "ok", r"ridgefill: approximate step",
     "A prompt chunk was approximated (one line a live step; a replayed recorded pass logs none).", ""),
    ("RF-004", "info", r"ridgefill: hazard -?\d+ positions",
     "A branched conversation resumed approximated positions from the prefix cache.",
     "Informational. See README 'Prefix cache'."),
    ("RF-005", "info", r"RidgeFill: rank \d+ holds nothing \(plumb",
     "Mode plumb: the plugin path runs without fitted tensors (a test mode).", "Use speed or quality to serve."),
    ("RF-006", "ok", r"plugin \S*qwen4exp_fp8\.so is shadowed by \S*",
     "The engine found RidgeFill's architecture first on the plugin search path.", ""),
    ("RF-101", "warn", r"RidgeFill: no projector folder \(looked at",
     "No projector found; the server runs stock.",
     "Put the projector folder at <model dir>/projector or set RADIANCE_RIDGEFILL_PROJECTOR (inside the "
     "container if you use Docker: mount it)."),
    ("RF-102", "error", r"RidgeFill: projector .* REFUSED, it cannot run on this model",
     "The projector was fitted for a different model (tokenizer, geometry or tensor names differ); serving stock.",
     "Use the projector made for this model file (README 'Requirements' names the model and its SHA256)."),
    ("RF-103", "error", r"RidgeFill: projector .* REFUSED: ",
     "The projector folder is damaged or incomplete (a file is missing, corrupt or not the one the manifest "
     "names); serving stock.", "Download the projector again and check it against its SHA256SUMS."),
    ("RF-104", "warn", r"RidgeFill: WARNING: projector .* fitted on another variant",
     "The projector runs, but was fitted on a variant of this model (quantisation or weight anchors differ).",
     "Expect quality to differ from the published numbers; use the projector for this exact file if you can."),
    ("RF-105", "error", r"selects exact rows from the table .* holds none; serving stock",
     "Mode quality needs the projector's row table and this projector has none; serving stock.",
     "Use the shipped projector, or RADIANCE_RIDGEFILL=speed."),
    ("RF-106", "error", r"RidgeFill: rank \d+ could not allocate .* for the projector",
     "Not enough host memory (or VRAM for the staging slot) for the projector on this rank.",
     "Free RAM (the projector maps ~640 MiB a rank) or lower other host-memory users (pinned pools, n-gram table)."),
    ("RF-107", "error", r"RidgeFill: rank \d+: the projector upload failed",
     "Copying the projector to the GPU failed.", "Check the kernel log for amdgpu errors (in this report) and GPU health."),
    ("RF-108", "error", r"RidgeFill: rank \d+ cannot load the int8 projector",
     "The int8 projector files could not be read on this rank.", "Download the projector again; check SHA256SUMS."),
    ("RF-201", "warn", r"built against radiance .* forwarding to the engine",
     "This plugin was built for another radiance release; it hands every request to the engine's own "
     "architecture, so RidgeFill is OFF.",
     "Use the RidgeFill build for your radiance release, or build one (docs/REBASING.md)."),
    ("RF-202", "error", r"built against radiance .* no in-tree architectures/",
     "Engine release mismatch, and the engine's own architecture is not on RADIANCE_HOME behind the plugin, "
     "so startup fails.", "Set RADIANCE_HOME=<plugin dir>:<engine share dir> (plugin first). A home given only "
     "as --radiance-home is invisible to plugins."),
    ("RF-203", "error", r"radiance: \S+: cannot load \S+: ",
     "The engine's own architecture file could not be loaded by the plugin's release guard.",
     "Check the engine install and RADIANCE_HOME."),
    ("RF-204", "error", r"exports no rad_arch_declare/step",
     "A file named like the engine's architecture is not an architecture plugin.", "Check RADIANCE_HOME."),
    ("RF-205", "error", r"plugin '\S*ridgefill\S*' declines this machine",
     "ridgefill.so does not serve this GPU, so RidgeFill's kernels are missing.",
     "RidgeFill ships for the GPU targets in VERSION.json (gfx1201). Other GPUs need a build for them."),
    ("RF-206", "error", r"kernel plugin '\S*ridgefill\S*' (has no device code|declares it was built for)",
     "ridgefill.so has no code for this GPU.", "As RF-205."),
    ("RF-301", "error", r": \S+ is '.*'; it takes ",
     "A RADIANCE_RIDGEFILL* variable has a value it does not take; startup refused.",
     "Fix the value named in the line (README 'Configuration')."),
    ("RF-302", "error", r"is retired with the container append",
     "A retired switch (RADIANCE_RIDGEFILL_PROJ, _ST or _DECLARE) is set; startup refused.", "Unset it."),
    ("RF-303", "error", r"is retired: the projector is always streamed",
     "A retired switch (RADIANCE_RIDGEFILL_PROJ_PLACE or _PROJ_RING) is set; startup refused.", "Unset it."),
    ("RF-304", "error", r"ridgefill\.tail is -?\d+ tokens and the largest step is",
     "The exact tail is too long for the step size; startup refused.",
     "Lower RADIANCE_RIDGEFILL_TAIL or raise --max-num-batched-tokens."),
    ("RF-305", "error", r"ridgefill\.tail is -?\d+ tokens; the shortest exact tail",
     "The exact tail is shorter than the method was measured at; startup refused.", "Raise RADIANCE_RIDGEFILL_TAIL."),
    ("RF-306", "warn", r"the container's ridgefill\.mode=\S+ is ignored",
     "The mode comes from RADIANCE_RIDGEFILL only.", "Set RADIANCE_RIDGEFILL."),
    ("RF-307", "error", r"RADIANCE_RIDGEFILL_MASK=all approximates rows and mode plumb",
     "A debug switch that cannot combine with mode plumb.", "Unset RADIANCE_RIDGEFILL_MASK."),
    ("RF-308", "error", r"the row share is \S+; it takes \(0, 1\]",
     "RADIANCE_RIDGEFILL_SHARE is out of range; startup refused.", "Use a value in (0, 1]."),
    ("RF-309", "error", r"the projector starts at layer .* n-gram embedding enters",
     "The projector's split layer cannot serve this model.", "Use the projector made for this model."),
    ("RF-310", "error", r"runs the MoE calibration tap",
     "An engine calibration run with RidgeFill on.", "Calibrate with RADIANCE_RIDGEFILL=off."),
    ("RF-311", "error", r"block owns its input norm",
     "The engine's model build is one this plugin release cannot serve.", "Open an issue with this report."),
    ("RF-401", "error", r"no kernel serves the (int8 projector's|staging ring's copy|projector \(ridgefill_gemm)",
     "A kernel the projector needs is missing: the engine's kernel libraries (libr4d) did not load.",
     "Keep the engine's own share directory on RADIANCE_HOME after the plugin directory."),
    ("RF-402", "error", r"no kernel library serves it -- ridgefill\.so is missing",
     "ridgefill.so is not on RADIANCE_HOME, or declines this GPU.",
     "Install kernels/ridgefill.so beside architectures/qwen4exp_fp8.so in the plugin directory."),
    ("RF-403", "info", r"so speed mode takes the masked path",
     "Speed mode uses its slower (masked) path for chunks that straddle the tail.", "Informational."),
    ("RF-501", "error", r"does not end on the delta net's chunk tile",
     "An internal invariant broke (the scheduler cut a chunk RidgeFill cannot approximate); the step was refused.",
     "Open an issue with this report and the full log."),
    ("RF-901", "warn", r"DEBUG RADIANCE_RIDGEFILL_",
     "A debug switch is set; output is not the served configuration.", "Unset RADIANCE_RIDGEFILL_FORCE_*/SHIFT_B/MASK."),
    ("RF-902", "error", r"RADIANCE_RIDGEFILL_(CAPTURE|DUMP)\w*: (device read failed|cannot (append to|write))",
     "A capture/dump tool (development only) could not read or write.", "Check the dump directory; unset the switch to serve."),
    ("RF-903", "error", r"(a capture needs the split layer|RADIANCE_RIDGEFILL_CAPTURE records exact runs|"
                        r"RADIANCE_RIDGEFILL_CAPTURE_STATE copies the delta-net state)",
     "A capture tool (development only) is misconfigured.", "Unset RADIANCE_RIDGEFILL_CAPTURE*."),
]
_COMPILED = [(c, lvl, re.compile(rx), what, fix) for c, lvl, rx, what, fix in CATALOG]
PLUGIN_LINE = re.compile(r"ridgefill|qwen4exp_fp8", re.I)
ENGINE_ERROR = re.compile(r"^E(\[\d+\])? ")
RANK = {"error": 0, "warn": 1, "info": 2, "ok": 3}


def classify(line):
    """The catalog entry a log line matches, or None."""
    for entry in _COMPILED:
        if entry[2].search(line):
            return entry
    return None


def diagnose(lines):
    """(findings, unclassified, engine_errors): findings = [(entry, count, first line)], worst first."""
    found, unclassified, engine_errors = {}, [], []
    for line in lines:
        entry = classify(line)
        if entry:
            n, first = found.get(entry[0], (0, line))
            found[entry[0]] = (n + 1, first)
        elif ENGINE_ERROR.match(line):
            engine_errors.append(line)
        elif PLUGIN_LINE.search(line) and line.startswith("radiance: "):
            unclassified.append(line)
    by_code = {e[0]: e for e in _COMPILED}
    findings = sorted(((by_code[c], n, first) for c, (n, first) in found.items()), key=lambda f: (RANK[f[0][1]], f[0][0]))
    return findings, unclassified, engine_errors


def headline(findings, engine_errors, ridgefill_env):
    """One sentence: is RidgeFill running, and if not, the first reason."""
    codes = {f[0][0] for f in findings}
    problems = [f for f in findings if f[0][1] in ("error", "warn") and f[0][0] != "RF-901"]
    if problems:
        return f"{problems[0][0][0]}: {problems[0][0][3]}"
    if "RF-001" in codes:
        return "RidgeFill is running: the projector matches this model" + (
            "" if "RF-003" in codes else " (no approximate step in this log: prompts shorter than the tail, or mode off)")
    if engine_errors:
        return "The engine logged errors (below); RidgeFill gave no reason of its own"
    if "RF-006" not in codes:
        return ("RidgeFill did not load: the engine never saw its architecture first on the plugin search path. "
                "Put the plugin directory first on RADIANCE_HOME")
    if (ridgefill_env or "off") == "off":
        return "RidgeFill loaded but is off (RADIANCE_RIDGEFILL unset or off)"
    return "RidgeFill loaded but logged no projector line; attach the full log"


# ------------------------------------------------------------------------------------------- collectors
def run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def read(path, default=None):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return default


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def host_facts():
    mem = dict(line.split(":", 1) for line in (read("/proc/meminfo", "") or "").splitlines() if ":" in line)
    gib = lambda k: f"{int(mem[k].split()[0]) / 2**20:.1f} GiB" if k in mem else "?"
    cpu = next((l.split(":", 1)[1].strip() for l in (read("/proc/cpuinfo", "") or "").splitlines()
                if l.startswith("model name")), platform.processor() or "?")
    pages = read("/sys/module/ttm/parameters/pages_limit")
    pinned = f"{int(pages) * os.sysconf('SC_PAGE_SIZE') / 2**30:.1f} GiB" if pages and pages.isdigit() else "not readable"
    return [("OS", platform.platform()), ("CPU", f"{cpu}, {os.cpu_count()} threads"),
            ("RAM", f"{gib('MemTotal')} total, {gib('MemAvailable')} available"),
            ("Pinned-memory cap (ttm pages_limit)", pinned),
            ("Docker", (run(["docker", "--version"]) or "not found").strip()),
            ("Python", platform.python_version())]


def gpu_facts():
    rows = []
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if read(f"{dev}/vendor") != "0x1002":
            continue
        slot = next((l.split("=", 1)[1] for l in (read(f"{dev}/uevent", "") or "").splitlines()
                     if l.startswith("PCI_SLOT_NAME=")), "?")
        vram = read(f"{dev}/mem_info_vram_total")
        used = read(f"{dev}/mem_info_vram_used")
        link = f"{read(f'{dev}/current_link_speed', '?')} x{read(f'{dev}/current_link_width', '?')} " \
               f"(max {read(f'{dev}/max_link_speed', '?')} x{read(f'{dev}/max_link_width', '?')})"
        card = os.path.basename(os.path.dirname(dev))
        displays = sum(read(p) == "connected" for p in glob.glob(f"/sys/class/drm/{card}-*/status"))
        size = f"{int(vram) / 2**30:.1f} GiB VRAM, {int(used) / 2**30:.1f} used" if vram and used else "VRAM ?"
        rows.append((slot, f"device {read(f'{dev}/device', '?')}, {size}, PCIe {link}, {displays} display(s)"))
    for node in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        props = dict(l.split(None, 1) for l in (read(node, "") or "").splitlines() if " " in l)
        if props.get("gfx_target_version", "0") != "0":
            v = int(props["gfx_target_version"])
            rows.append((f"kfd node {node.split('/')[-2]}", f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}"))
    return rows or [("GPUs", "no AMD GPU in /sys/class/drm")]


GPU_FAULT = re.compile(r"amdgpu.*(page fault|PROTECTION_FAULT|failed to respond|ring \S+ timeout|job timed out|"
                       r"GPU reset|reset begin|hang)", re.I)


def kernel_log():
    """amdgpu failures this boot (not its routine firmware lines), with times, so they can be set against the run."""
    text = run(["journalctl", "-k", "-b", "--no-pager", "-q", "-o", "short-iso"], timeout=20)
    if text is None:
        return "journalctl -k not readable (run as a user in the systemd-journal group, or attach `dmesg -T`)."
    bad = [l for l in text.splitlines() if GPU_FAULT.search(l)]
    if not bad:
        return "No amdgpu fault, timeout or reset this boot."
    return (f"{len(bad)} amdgpu fault/timeout/reset line(s) this boot, first {bad[0][:25]}, last {bad[-1][:25]}. "
            "A fault during the failing run makes everything after it untrustworthy; one from before it may not matter."
            + "".join("\n    " + l[:200] for l in bad[-3:]))


def plugin_facts(plugin_dir):
    rows = []
    version = read(os.path.join(plugin_dir, "VERSION.json"))
    if version:
        v = json.loads(version)
        rows.append(("Plugin", f"{v.get('plugin')} {v.get('plugin_version')} (commit {v.get('plugin_commit')}), "
                               f"built for radiance {v.get('radiance_version')}, GPU {', '.join(v.get('gpu_targets', []))}"))
    sums = {}
    for line in (read(os.path.join(plugin_dir, "SHA256SUMS"), "") or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    for so in sorted(glob.glob(os.path.join(plugin_dir, "*", "*.so"))):
        rel = os.path.relpath(so, plugin_dir)
        digest = sha256(so)
        check = "" if rel not in sums else (" (matches SHA256SUMS)" if sums[rel] == digest else " (DIFFERS from SHA256SUMS)")
        rows.append((rel, digest[:16] + check))
    return rows or [("Plugin", f"no VERSION.json or .so files in {plugin_dir}")]


def projector_facts(folder):
    manifest = read(os.path.join(folder, "ridgefill.json"))
    if manifest is None:
        return [("Projector", f"no ridgefill.json in {folder}")]
    m = json.loads(manifest)
    files = m.get("files", {})
    bad = [n for n, want in files.items()
           if not os.path.isfile(os.path.join(folder, n)) or sha256(os.path.join(folder, n)) != want]
    return [("Projector", f"{m.get('name')} (adapter {m.get('adapter')}, split {m.get('split')})"),
            ("Projector files", f"{len(files) - len(bad)}/{len(files)} match the manifest" +
             (f"; missing or corrupt: {', '.join(bad[:5])}" if bad else ""))]


def server_facts(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=5) as r:
            v = json.loads(r.read())
        return [("Engine (/version)", f"{v.get('engine')} {v.get('version')}, ABI {v.get('abi_version')}")]
    except (OSError, ValueError) as e:
        return [("Engine (/version)", f"no answer from {url}: {e}")]


def log_lines(args):
    if args.log:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    if args.container:
        text = run(["docker", "logs", args.container], timeout=60)
        if text is None:
            sys.exit(f"ridgefill_report: `docker logs {args.container}` failed")
        return text.splitlines()
    return []


# ------------------------------------------------------------------------------------------- report
def redact(text):
    for value, name in ((os.path.expanduser("~"), "<home>"), (getpass.getuser(), "<user>"), (socket.gethostname(), "<host>")):
        if value and len(value) > 2:
            text = text.replace(value, name)
    return text


def table(rows):
    return "\n".join(f"- **{k}:** {v}" for k, v in rows)


def render(args, lines):
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("RADIANCE")}
    findings, unclassified, engine_errors = diagnose(lines)
    out = ["# RidgeFill report", ""]
    out += ["## Diagnosis", "", "**" + (headline(findings, engine_errors, env.get("RADIANCE_RIDGEFILL")) if lines
                                        else "No log given: run again with --log FILE or --container NAME") + "**", ""]
    for (code, level, _, what, fix), n, first in findings:
        out.append(f"- `{code}` {level} x{n}: {what}" + (f" **Fix:** {fix}" if fix else ""))
        if level in ("error", "warn"):
            out.append(f"    `{first[:300]}`")
    if unclassified:
        out += ["", "Plugin lines the catalog does not know (please keep them in the issue):", ""]
        out += [f"    {l[:300]}" for l in dict.fromkeys(unclassified)]
    if engine_errors:
        out += ["", "Engine error lines:", ""] + [f"    {l[:300]}" for l in dict.fromkeys(engine_errors)]
    out += ["", "## Plugin", "", table(plugin_facts(args.plugin_dir))]
    if args.projector:
        out += ["", table(projector_facts(args.projector))]
    if args.url:
        out += ["", table(server_facts(args.url))]
    out += ["", "## Machine", "", table(host_facts()), "", "## GPUs", "", table(gpu_facts()),
            "", "## Kernel log", "", kernel_log(),
            "", "## Environment", "", table(env.items()) if env else "No RADIANCE* variables in this shell "
            "(if the server runs in a container, they are set there: list them in the issue)."]
    if lines:
        startup = [l for l in lines if re.match(r"^[IWE](\[\d+\])? (device|model|placement|plugin|kv|declare):", l)]
        out += ["", "## Log excerpt", "", "<details><summary>startup lines and the last 40 lines</summary>", "", "```"]
        out += startup[:40] + ["..."] + lines[-40:] + ["```", "</details>"]
    return redact("\n".join(out)) + "\n"


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--log", help="the server's log file (stderr)")
    src.add_argument("--container", help="a docker container to read `docker logs` from")
    p.add_argument("--plugin-dir", default=here, help="the RidgeFill plugin folder (default: this script's folder)")
    p.add_argument("--projector", default=os.environ.get("RADIANCE_RIDGEFILL_PROJECTOR"),
                   help="the projector folder to verify (default: $RADIANCE_RIDGEFILL_PROJECTOR)")
    p.add_argument("--url", help="the server's base URL, for /version")
    p.add_argument("--out", help="write the report here as well as to stdout")
    args = p.parse_args(argv)
    if args.log and not os.path.isfile(args.log):
        p.error(f"--log {args.log}: no such file")
    report = render(args, log_lines(args))
    sys.stdout.write(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
