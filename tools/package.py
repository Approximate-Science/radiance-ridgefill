#!/usr/bin/env python3
"""package.py -- build the radiance-kva release distribution (PACKAGING.md §0-§5).

  package.py --home <frozen plugin home> --projector <projector folder> \
               --template-spec <kva-marker-spec.json> --base-template <model chat template> \
               --out <dist dir> --version <x.y.z> --commit <git sha> \
               [--radiance-version <x.y.z>] [--gpu-targets gfx....] [--abi-version <x.y.z>]

Produces, under --out:
  radiance-kva-<version>/        architectures/qwen4exp_fp8.so, kernels/kva.so, README.md
                                 (this repo's), LICENSE (the repo's), VERSION.json, SHA256SUMS
  projector-qwen3.8-flash-next/  an exact copy of the projector folder (its kva.json hashes
                                 verified BEFORE copying; a mismatch refuses by name) + SHA256SUMS
  kva-chat-template/             kva_template.py, the marker spec, a pre-merged template
                                 (kva_template.py merge of --base-template), a README, SHA256SUMS
  <each>.tar.gz                  a tarball of each directory, byte-deterministic (sorted names,
                                 fixed mtime/uid/gid, gzip mtime 0): two runs give identical bytes
  SHA256SUMS                     the three tarballs

Every file is written mode 0644. Refuses by name on any missing input; writes nothing
outside --out (and refuses an --out that overlaps an input). Standard library only.

The frozen home is what scripts/frozen_home.sh built; the projector folder is what
tools/kva_projector.py built; the merge runs the very tools/kva_template.py that ships
inside the kva-chat-template package.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

PROG = "package"

PLUGIN_NAME = "radiance-kva"
PROJECTOR_DIRNAME = "projector-qwen3.8-flash-next"
TEMPLATE_DIRNAME = "kva-chat-template"
ARCH_SO = "architectures/qwen4exp_fp8.so"
KERNEL_SO = "kernels/kva.so"

VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")
GFX_RE = re.compile(rb"gfx[0-9a-z]+")

TEMPLATE_README = """# KVA chat template -- per-request KVA for radiance

For a radiance server serving Qwen3.8-Flash-Next with the radiance-kva plugin installed
and a matching projector folder loaded. Everything the marker does (tokens, ids, dials,
the switch kwarg) is read from {spec_name}; kva_template.py holds no marker constants.

## Serve the model's own template (what most users want)

`chat_template.jinja` here is the model's chat template with the kva marker block merged
in front (its base bytes are appended completely unchanged). Point radiance at it:

    radiance ... --override-chat-template /path/to/chat_template.jinja

A request opts in per request:

    "chat_template_kwargs": {{"kva": "on"}}        dials: kva_share / kva_alpha / kva_tail

Without the kwarg every request renders byte-identically to the unmodified template and
runs stock.

## Serve your OWN chat template

Merge the marker into it; your template's bytes stay unchanged:

    python3 kva_template.py merge --spec {spec_name} --base your-template.jinja --out your-template.kva.jinja
    radiance ... --override-chat-template /path/to/your-template.kva.jinja

Verify a merged file byte for byte, or remove the block again:

    python3 kva_template.py check --spec {spec_name} --base your-template.jinja --merged your-template.kva.jinja
    python3 kva_template.py strip --merged your-template.kva.jinja --out your-template.jinja

## Docker

The template path must name a file visible INSIDE the container; mount it read-only and
pass the container-side path:

    docker run ... -v $PWD:/templates:ro <image> ... \\
        --override-chat-template /templates/chat_template.jinja

## Caveats

- Only point --override-chat-template at a merged template when the radiance-kva plugin
  is installed AND a usable projector folder is loaded: without the plugin the marker
  renders into the prompt unerased and those requests degrade (the plugin's startup
  guard refuses this combination by name).
- Without the template every request runs stock; server-wide KVA needs
  RADIANCE_KVA=quality or RADIANCE_KVA=speed instead.
"""


def die(message: str) -> None:
    print(f"{PROG}: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        die(f"{what} is not a directory: {path}")
    return path


def require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        die(f"{what} is missing: {path}")
    return path


# ------------------------------------------------------------------ the .so readers

def version_strings_in(data: bytes) -> list[str]:
    """Every NUL-delimited string in *data* that is exactly d.d.d.

    KVA_RADIANCE_VERSION (arch/CMakeLists.txt) compiles the radiance release into the arch
    plugin as one such string -- the same strings CMake's release check reads in the engine
    binary (repo CMakeLists.txt). Version tags like GLIBCXX_3.4.32 do not match: the whole
    NUL-delimited string must be the bare release number.
    """
    found = set()
    for raw in data.split(b"\x00"):
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            continue
        if VERSION_RE.fullmatch(text):
            found.add(text)
    return sorted(found)


def gpu_targets_in(data: bytes) -> list[str]:
    """Every gfx... string in *data* (the AMDGPU targets the binary was compiled for)."""
    return sorted({m.group().decode("ascii") for m in GFX_RE.finditer(data)})


def read_radiance_version(arch_so: Path, override: str | None) -> tuple[str, str]:
    """The radiance release the plugin was built against, and how it was obtained."""
    candidates = version_strings_in(arch_so.read_bytes())
    if override is not None:
        if candidates and override not in candidates:
            die(f"--radiance-version {override} is not among the release strings embedded in "
                f"{arch_so}: {candidates}")
        if candidates:
            return override, (f"--radiance-version argument; also embedded in {ARCH_SO} "
                               f"(KVA_RADIANCE_VERSION, arch/CMakeLists.txt)")
        return override, "--radiance-version argument (no NUL-delimited d.d.d string found in the .so)"
    if len(candidates) == 1:
        return candidates[0], (f"the NUL-delimited release string embedded in {ARCH_SO} "
                               "(KVA_RADIANCE_VERSION, set by arch/CMakeLists.txt from RADIANCE_SRC)")
    if not candidates:
        die(f"no NUL-delimited d.d.d release string found in {arch_so} (KVA_RADIANCE_VERSION); "
            "pass --radiance-version to record it")
    die(f"{arch_so} carries several release strings {candidates}; pass --radiance-version "
        "to pick the radiance release this plugin was built against")


def read_gpu_targets(archives: list[Path], override: str | None) -> tuple[list[str], str]:
    """The GPU targets the plugins were compiled for, and how they were obtained."""
    if override is not None:
        targets = [t.strip() for t in override.split(",") if t.strip()]
        if not targets:
            die("--gpu-targets is empty after splitting on commas")
        return targets, "--gpu-targets argument"
    found: set[str] = set()
    for path in archives:
        found.update(gpu_targets_in(path.read_bytes()))
    targets = sorted(found)
    if targets:
        return targets, "gfx* strings extracted from the packaged .so files"
    return [], ("no gfx target string found in the packaged .so files (a host-only build); "
                "pass --gpu-targets to record them")


# ------------------------------------------------------------------ the projector folder

def verify_projector(projector: Path) -> dict:
    """Check the projector folder against its kva.json BEFORE anything is copied.

    Refuses by name on: a missing kva.json, a manifest without a files map, a listed file
    that is missing or hashes differently, a file on disk the manifest does not list, and
    anything in the folder that is not a regular file.
    """
    manifest_path = require_file(projector / "kva.json", f"the projector manifest of {projector}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"cannot read the projector manifest {manifest_path}: {exc}")
    if not isinstance(manifest, dict):
        die(f"{manifest_path}: the manifest must be a JSON object")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        die(f"{manifest_path}: the manifest has no 'files' map (every file's sha256)")
    on_disk = {p.name: p for p in projector.iterdir()}
    for name in sorted(p for p in on_disk if not on_disk[p].is_file()):
        die(f"{projector / name}: not a regular file (the projector folder is files only)")
    extra = sorted(set(on_disk) - set(files) - {"kva.json"})
    if extra:
        die(f"the projector manifest {manifest_path} does not list: {', '.join(extra)}")
    for name in sorted(files):
        path = projector / name
        if not path.is_file():
            die(f"the projector manifest lists {name}, which is missing in {projector}")
        want = files[name]
        if not isinstance(want, str):
            die(f"{manifest_path}: files['{name}'] is not a sha256 string")
        have = sha256_file(path)
        if have != want:
            die(f"the projector file {path} hashes {have} but its manifest {manifest_path} "
                f"lists {want}")
    return manifest


def copy_file(src: Path, dst: Path) -> None:
    """Copy one file, mode 0644 (every packaged file is 0644)."""
    shutil.copyfile(src, dst)
    dst.chmod(0o644)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o644)


def write_sums(directory: Path) -> None:
    """SHA256SUMS of every file under *directory* (relative, sorted), mode 0644."""
    names = sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())
    lines = [f"{sha256_file(directory / name)}  {name}" for name in names]
    write_text(directory / "SHA256SUMS", "\n".join(lines) + "\n")


# ------------------------------------------------------------------ the deterministic tarball

def deterministic_tar(src_dir: Path, dest: Path, arc_root: str) -> None:
    """A byte-deterministic <arc_root>.tar.gz of *src_dir*'s contents.

    Sorted member names, mtime 0, uid/gid 0, no uname/gname, files 0644, directories 0755,
    gzip header with mtime 0 and no file name: two runs over the same tree give identical
    bytes (the ustar format carries no pax headers whose bytes could vary).
    """
    entries: list[tuple[str, Path, bool]] = [(arc_root + "/", src_dir, True)]
    for path in sorted(p for p in src_dir.rglob("*")):
        rel = arc_root + "/" + str(path.relative_to(src_dir))
        entries.append((rel + "/" if path.is_dir() else rel, path, path.is_dir()))
    with open(dest, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for name, path, is_dir in sorted(entries):
                    info = tarfile.TarInfo(name)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    if is_dir:
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        tar.addfile(info)
                    else:
                        info.size = path.stat().st_size
                        info.mode = 0o644
                        with open(path, "rb") as f:
                            tar.addfile(info, f)


# ------------------------------------------------------------------ VERSION.json

def build_version_json(version: str, commit: str, radiance_version: str,
                       radiance_version_source: str, gpu_targets: list[str],
                       gpu_targets_source: str, abi_version: str | None,
                       abi_source: str) -> str:
    """VERSION.json: the plugin's version, commit and build target, each with its source.

    No timestamps and no absolute paths, so two runs give identical bytes.
    """
    doc = {
        "plugin": PLUGIN_NAME,
        "plugin_version": version,
        "plugin_commit": commit,
        "radiance_version": radiance_version,
        "radiance_abi": abi_version,
        "gpu_targets": gpu_targets,
        "fields": {
            "plugin_version": "--version argument",
            "plugin_commit": ("--commit argument (the commit scripts/frozen_home.sh built "
                              "the plugin home from)"),
            "radiance_version": radiance_version_source,
            "radiance_abi": abi_source,
            "gpu_targets": gpu_targets_source,
        },
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


# ------------------------------------------------------------------ packaging

def overlap(out: Path, other: Path, name: str) -> None:
    """Refuse an --out that is inside an input or contains one."""
    if out == other or out in other.parents or other in out.parents:
        die(f"--out {out} overlaps the {name} {other}; nothing may be written into an input")


def merge_template(tool: Path, spec: Path, base: Path, out: Path) -> None:
    """chat_template.jinja = the kva block + base, via the very tool that ships in the box."""
    merged = out / "chat_template.jinja"
    run = [sys.executable, str(tool), "merge", "--spec", str(spec), "--base", str(base),
           "--out", str(merged)]
    result = subprocess.run(run, capture_output=True, text=True)
    if result.returncode != 0:
        die(f"kva_template.py merge failed ({result.stderr.strip()})")
    check = subprocess.run([sys.executable, str(tool), "check", "--spec", str(spec),
                            "--base", str(base), "--merged", str(merged)],
                           capture_output=True, text=True)
    if check.returncode != 0:
        die(f"kva_template.py check refused the merged template: {check.stderr.strip()}")
    merged.chmod(0o644)


def package(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parent.parent

    # ---- every input, checked and named before anything is written
    home = require_dir(Path(args.home), "--home (the frozen plugin home)")
    arch_so = require_file(home / ARCH_SO, f"the architecture plugin of {home}")
    kernel_so = require_file(home / KERNEL_SO, f"the kernel library of {home}")
    projector = require_dir(Path(args.projector), "--projector (the projector folder)")
    verify_projector(projector)          # hashes verified BEFORE any copying
    spec = require_file(Path(args.template_spec), "--template-spec (kva-marker-spec.json)")
    base = require_file(Path(args.base_template), "--base-template (the model's chat template)")
    readme = require_file(repo / "README.md", "the repo README")
    license_ = require_file(repo / "LICENSE", "the repo LICENSE")
    template_tool = require_file(repo / "tools" / "kva_template.py", "the template tool")

    if not VERSION_RE.fullmatch(args.version):
        die(f"--version must be x.y.z, got {args.version!r}")
    if not COMMIT_RE.fullmatch(args.commit):
        die(f"--commit must be a git sha (7-40 hex chars), got {args.commit!r}")
    if args.abi_version is not None and not VERSION_RE.fullmatch(args.abi_version):
        die(f"--abi-version must be x.y.z, got {args.abi_version!r}")

    out = Path(args.out)
    if out.exists():
        if not out.is_dir():
            die(f"--out exists and is not a directory: {out}")
        if any(out.iterdir()):
            die(f"--out is not empty: {out} (package into a fresh directory)")
    overlap(out.resolve(), home.resolve(), "frozen plugin home")
    overlap(out.resolve(), projector.resolve(), "projector folder")
    overlap(out.resolve(), spec.resolve(), "marker spec")
    overlap(out.resolve(), base.resolve(), "base template")

    radiance_version, radiance_source = read_radiance_version(arch_so, args.radiance_version)
    gpu_targets, gpu_source = read_gpu_targets([arch_so, kernel_so], args.gpu_targets)
    if args.abi_version is not None:
        abi_source = "--abi-version argument"
    else:
        abi_source = ("not recorded in the packaged inputs; pass --abi-version to record the "
                      "radiance ABI the plugin was linked against")

    out.mkdir(parents=True)
    plugin_dir = out / f"{PLUGIN_NAME}-{args.version}"
    projector_dir = out / PROJECTOR_DIRNAME
    template_dir = out / TEMPLATE_DIRNAME

    # ---- radiance-kva-<version>/ : the plugin home plus its provenance
    (plugin_dir / "architectures").mkdir(parents=True)
    (plugin_dir / "kernels").mkdir()
    copy_file(arch_so, plugin_dir / ARCH_SO)
    copy_file(kernel_so, plugin_dir / KERNEL_SO)
    copy_file(readme, plugin_dir / "README.md")
    copy_file(license_, plugin_dir / "LICENSE")
    write_text(plugin_dir / "VERSION.json",
               build_version_json(args.version, args.commit, radiance_version, radiance_source,
                                  gpu_targets, gpu_source, args.abi_version, abi_source))
    write_sums(plugin_dir)

    # ---- projector-qwen3.8-flash-next/ : the exact folder, hashes already verified
    projector_dir.mkdir()
    for path in sorted(projector.iterdir()):
        copy_file(path, projector_dir / path.name)
    write_sums(projector_dir)

    # ---- kva-chat-template/ : the tool, the spec, a pre-merged template, a README
    template_dir.mkdir()
    copy_file(template_tool, template_dir / "kva_template.py")
    copy_file(spec, template_dir / spec.name)
    merge_template(template_tool, spec, base, template_dir)
    write_text(template_dir / "README.md", TEMPLATE_README.format(spec_name=spec.name))
    write_sums(template_dir)

    # ---- the tarballs (deterministic) and dist/SHA256SUMS over them
    tarballs = []
    for src_dir, arc_root in ((plugin_dir, plugin_dir.name),
                              (projector_dir, PROJECTOR_DIRNAME),
                              (template_dir, TEMPLATE_DIRNAME)):
        dest = out / f"{arc_root}.tar.gz"
        deterministic_tar(src_dir, dest, arc_root)
        tarballs.append(dest)
    lines = [f"{sha256_file(t)}  {t.name}" for t in tarballs]
    write_text(out / "SHA256SUMS", "\n".join(lines) + "\n")

    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"packaged {PLUGIN_NAME}-{args.version} (commit {args.commit}, radiance "
          f"{radiance_version}, gpu targets {gpu_targets or 'none recorded'}) into {out}")
    print(f"  {plugin_dir.name}/ ({ARCH_SO}, {KERNEL_SO}, README.md, LICENSE, VERSION.json)")
    print(f"  {PROJECTOR_DIRNAME}/ ({len(list(projector_dir.iterdir()))} files, manifest verified)")
    print(f"  {TEMPLATE_DIRNAME}/ (kva_template.py, {spec.name}, chat_template.jinja, README.md)")
    print(f"  3 deterministic tarballs + SHA256SUMS; {n_files} files total")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Build the radiance-kva release distribution (plugins, projector folder, "
                    "chat-template package, deterministic tarballs).")
    parser.add_argument("--home", required=True,
                        help="the frozen plugin home (scripts/frozen_home.sh output: "
                             f"{ARCH_SO} and {KERNEL_SO})")
    parser.add_argument("--projector", required=True,
                        help="the projector folder (tools/kva_projector.py build output, "
                             "with its kva.json manifest)")
    parser.add_argument("--template-spec", required=True,
                        help="kva-marker-spec.json (tools/kva_template.py's spec)")
    parser.add_argument("--base-template", required=True,
                        help="the model's own chat template (the pre-merged template's base)")
    parser.add_argument("--out", required=True, help="the dist directory to create (must be "
                        "empty or absent)")
    parser.add_argument("--version", required=True, help="the plugin version, x.y.z")
    parser.add_argument("--commit", required=True, help="the git commit the home was built from")
    parser.add_argument("--radiance-version", help="the radiance release the plugin was built "
                        "against; by default read from the arch .so (KVA_RADIANCE_VERSION)")
    parser.add_argument("--gpu-targets", help="comma-separated GPU targets (e.g. gfx1201); by "
                        "default extracted from the packaged .so files")
    parser.add_argument("--abi-version", help="the radiance ABI version the plugin was linked "
                        "against (not recorded in the .so; pass it to include it in VERSION.json)")
    args = parser.parse_args(argv)
    return package(args)


if __name__ == "__main__":
    sys.exit(main())