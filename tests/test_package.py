"""test_package.py -- tests for tools/package.py (the release packager).

package.py derives the repo root from its own location, so the tests build a SCRATCH
REPO (tools/package.py + tools/kva_template.py copied byte-identical, a README, a
LICENSE) and run the copy; the real repo is never written to. The fake inputs mirror
the real shapes: a frozen home (architectures/qwen4exp_fp8.so + kernels/kva.so), the
projector folder (files + a kva.json manifest with real sha256s, built for BOTH dtypes
the release may ship -- bf16 and the int8 folder -- so the package naming is exercised),
a marker spec and a base chat template. Standard library + pytest only; everything runs
offline.

The release shape: the chat-template package is NOT built by default (per-request KVA is
parked, notes/future/per-request.md) -- the template tests pass --with-template.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "tools" / "package.py"
TEMPLATE_TOOL = REPO / "tools" / "kva_template.py"


# ------------------------------------------------------------------ the fixtures' builders

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read())
    return digest.hexdigest()


def make_spec(path: Path) -> Path:
    """A valid 64-token marker spec (the same structure as the real kva-marker-spec.json,
    which the template tool refuses to hard-code anything from)."""
    spec = {
        "length": 64,
        "pattern": {"offsets": [0, 59],
                    "tokens": ["<|quad_start|>", "<|quad_end|>"],
                    "rule": "alternate, starting with the first token"},
        "dials": [
            {"offset": 60, "kwarg": "kva_share", "default": "0.25",
             "table": {"0.10": "<|box_start|>", "0.25": "<|box_end|>",
                       "0.50": "<|object_ref_start|>"}},
            {"offset": 61, "kwarg": "kva_alpha", "default": "1.0",
             "table": {"0": "<|box_start|>", "0.5": "<|box_end|>",
                       "1.0": "<|object_ref_start|>"}},
            {"offset": 62, "kwarg": "kva_tail", "default": "2048",
             "table": {"1024": "<|box_start|>", "2048": "<|box_end|>",
                       "2560": "<|object_ref_start|>", "3072": "<|object_ref_end|>"}},
        ],
        "end": {"offset": 63, "token": "<|im_end|>"},
        "switch": {"kwarg": "kva", "on_value": "on"},
        "token_ids": {"<|quad_start|>": 248051, "<|quad_end|>": 248052,
                      "<|box_start|>": 248049, "<|box_end|>": 248050,
                      "<|object_ref_start|>": 248047, "<|object_ref_end|>": 248048,
                      "<|im_end|>": 248044},
    }
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


def make_home(home: Path) -> Path:
    """A frozen plugin home: the two .so files, with the radiance release string
    (KVA_RADIANCE_VERSION, arch/CMakeLists.txt) and a gfx target embedded as the real
    HIP build would carry them."""
    (home / "architectures").mkdir(parents=True)
    (home / "kernels").mkdir()
    arch = (home / "architectures" / "qwen4exp_fp8.so").write_bytes(
        b"ELF fake\n\x001.0.8\x00GLIBCXX_3.4.32\x00amdgfx target gfx1201\x00\x00")
    (home / "kernels" / "kva.so").write_bytes(
        b"ELF fake kernel\x00gfx1201\x00kva_gemm_nt_bias\x00")
    return home


def make_projector(projector: Path, dtype: str = "bf16") -> Path:
    """The projector folder's shape (notes/aprime.md §2, small): a few tensor files,
    chat_template.jinja, README.md and the kva.json manifest listing every file's sha256.
    The manifest carries its own projector.dtype (bf16, or int8 -- the folder
    `tools/kva_projector.py int8` builds), which is what names the package."""
    projector.mkdir(parents=True, exist_ok=True)
    names = []
    for layer in (24, 25, 47):
        name = f"proj.L{layer}.safetensors"
        (projector / name).write_bytes(f"fake projector tensors layer {layer} {dtype}\n".encode())
        names.append(name)
    (projector / "correction.safetensors").write_bytes(b"fake correction tensors\n")
    (projector / "rowsel.safetensors").write_bytes(b"fake rowsel tables\n")
    (projector / "chat_template.jinja").write_text("merged template placeholder\n",
                                                   encoding="utf-8")
    (projector / "README.md").write_text("# projector\n", encoding="utf-8")
    files = {name: sha256_file(projector / name) for name in names}
    files["correction.safetensors"] = sha256_file(projector / "correction.safetensors")
    files["rowsel.safetensors"] = sha256_file(projector / "rowsel.safetensors")
    files["chat_template.jinja"] = sha256_file(projector / "chat_template.jinja")
    files["README.md"] = sha256_file(projector / "README.md")
    manifest = {"format": 1, "plugin_min_version": "0.3.0", "adapter": "qwen4exp",
                "adapter_abi": 1, "split": 24, "files": files,
                "projector": {"dtype": dtype, "layout": "plain_nk",
                               "files": {str(L): f"proj.L{L}.safetensors" for L in (24, 25, 47)}}}
    (projector / "kva.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                        encoding="utf-8")
    return projector


def make_repo(dst: Path, with_license: bool = True) -> Path:
    """A scratch repo: package.py and kva_template.py copied byte-identical from the real
    one, a README, a LICENSE (or none)."""
    tools = dst / "tools"
    tools.mkdir(parents=True)
    shutil.copyfile(PACKAGE, tools / "package.py")
    shutil.copyfile(TEMPLATE_TOOL, tools / "kva_template.py")
    (dst / "README.md").write_text("# radiance-kva (test repo README)\n", encoding="utf-8")
    if with_license:
        (dst / "LICENSE").write_text("MIT test license\n", encoding="utf-8")
    return dst


@pytest.fixture
def inputs(tmp_path):
    """All valid inputs plus a scratch repo; returns a dict of paths."""
    repo = make_repo(tmp_path / "repo")
    base = tmp_path / "base-template.jinja"
    base.write_text(
        "{%- for m in messages %}{{ m['content'] }}{% endfor %}\n", encoding="utf-8")
    return {
        "repo": repo,
        "packager": repo / "tools" / "package.py",
        "home": make_home(tmp_path / "home"),
        "projector": make_projector(tmp_path / "projector"),
        "projector_int8": make_projector(tmp_path / "projector-int8", dtype="int8"),
        "spec": make_spec(tmp_path / "kva-marker-spec.json"),
        "base": base,
    }


def run_packager(inputs, out, version="0.3.0", commit="0123456789abcdef",
                 extra=(), template=False, expect_rc=0):
    """Run the packager. template=True adds --with-template + --template-spec +
    --base-template (the release shape leaves them out: the chat-template package is
    parked behind the flag)."""
    cmd = [sys.executable, str(inputs["packager"]),
           "--home", str(inputs["home"]),
           "--projector", str(inputs["projector"]),
           "--out", str(out),
           "--version", version,
           "--commit", commit]
    if template:
        cmd += ["--with-template", "--template-spec", str(inputs["spec"]),
                "--base-template", str(inputs["base"])]
    cmd += list(extra)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if expect_rc is not None:
        assert result.returncode == expect_rc, result.stderr
    return result


def check_sums(dir_path: Path):
    """sha256sum -c must pass in the directory, and every file must be mode 0644."""
    result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=dir_path,
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for path in dir_path.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o644, path


# ------------------------------------------------------------------ the layout

def test_layout_and_sums_no_template_by_default(inputs, tmp_path):
    """The release shape: the chat-template package is NOT built (per-request KVA is
    parked, notes/future/per-request.md); two packages, two tarballs."""
    out = tmp_path / "dist"
    result = run_packager(inputs, out)
    assert "radiance-kva-0.3.0" in result.stdout

    plugin = out / "radiance-kva-0.3.0"
    assert (plugin / "architectures" / "qwen4exp_fp8.so").is_file()
    assert (plugin / "kernels" / "kva.so").is_file()
    assert (plugin / "README.md").read_text(encoding="utf-8") == \
        (inputs["repo"] / "README.md").read_text(encoding="utf-8")
    assert (plugin / "LICENSE").is_file()
    assert (plugin / "VERSION.json").is_file()
    check_sums(plugin)

    projector = out / "projector-qwen3.8-flash-next-bf16"     # named by the manifest's dtype
    assert projector.is_dir()
    # an exact copy: every input file, byte-identical, nothing missing or extra
    # (LICENSE and SHA256SUMS are the two additions packaging makes on top)
    for name in sorted(p.name for p in inputs["projector"].iterdir()):
        assert (projector / name).is_file(), name
        assert sha256_file(projector / name) == sha256_file(inputs["projector"] / name), name
    assert sorted(p.name for p in projector.iterdir()
                  if p.name not in ("SHA256SUMS", "LICENSE")) == \
        sorted(p.name for p in inputs["projector"].iterdir())
    assert (projector / "LICENSE").read_bytes() == \
        (inputs["repo"] / "LICENSE").read_bytes()
    sums_lines = (projector / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert any(line.endswith("  LICENSE") for line in sums_lines)   # LICENSE is summed
    check_sums(projector)

    # no chat-template package anywhere: not built by default
    assert not (out / "kva-chat-template").exists()
    for tarball in ("radiance-kva-0.3.0.tar.gz", "projector-qwen3.8-flash-next-bf16.tar.gz"):
        assert (out / tarball).is_file()
    assert not (out / "kva-chat-template.tar.gz").exists()
    dist_sums = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(dist_sums) == 2
    for line in dist_sums:
        digest, name = line.split("  ", 1)
        assert digest == sha256_file(out / name)
    assert "dtype bf16" in result.stdout


def test_int8_projector_folder_names_the_package_int8(inputs, tmp_path):
    """The shipped projector may be the int8 folder (built by `kva_projector.py int8`):
    the package is named after the dtype the manifest ITSELF carries, never hard-coded."""
    inputs["projector"] = inputs["projector_int8"]
    out = tmp_path / "dist"
    result = run_packager(inputs, out)
    projector = out / "projector-qwen3.8-flash-next-int8"
    assert projector.is_dir()
    assert not (out / "projector-qwen3.8-flash-next-bf16").exists()
    # an exact copy of the int8 folder, byte-identical, nothing missing or extra
    for name in sorted(p.name for p in inputs["projector_int8"].iterdir()):
        assert (projector / name).is_file(), name
        assert sha256_file(projector / name) == sha256_file(inputs["projector_int8"] / name), name
    check_sums(projector)
    assert (out / "projector-qwen3.8-flash-next-int8.tar.gz").is_file()
    dist_sums = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in dist_sums] == \
        ["radiance-kva-0.3.0.tar.gz", "projector-qwen3.8-flash-next-int8.tar.gz"]
    assert "dtype int8" in result.stdout


def test_template_package_built_only_with_the_flag(inputs, tmp_path):
    """--with-template opts into the chat-template package (parked by default); the
    layout inside it is the old one, and dist/SHA256SUMS then covers three tarballs."""
    out = tmp_path / "dist"
    run_packager(inputs, out, template=True)

    template = out / "kva-chat-template"
    assert (template / "kva_template.py").read_bytes() == TEMPLATE_TOOL.read_bytes()
    assert (template / "kva-marker-spec.json").is_file()
    assert (template / "chat_template.jinja").is_file()
    assert (template / "README.md").is_file()
    assert (template / "LICENSE").read_bytes() == \
        (inputs["repo"] / "LICENSE").read_bytes()
    template_readme = (template / "README.md").read_text(encoding="utf-8")
    assert "--override-chat-template" in template_readme
    assert "PARKED" in template_readme          # it says the feature is parked
    sums_lines = (template / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert any(line.endswith("  LICENSE") for line in sums_lines)   # LICENSE is summed
    check_sums(template)

    for tarball in ("radiance-kva-0.3.0.tar.gz", "projector-qwen3.8-flash-next-bf16.tar.gz",
                    "kva-chat-template.tar.gz"):
        assert (out / tarball).is_file()
    dist_sums = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(dist_sums) == 3
    for line in dist_sums:
        digest, name = line.split("  ", 1)
        assert digest == sha256_file(out / name)


def test_version_json_records_every_field_and_its_source(inputs, tmp_path):
    out = tmp_path / "dist"
    run_packager(inputs, out, version="1.2.3", commit="0123456789abcdef",
                 extra=["--abi-version", "15.0.0"])
    doc = json.loads((out / "radiance-kva-1.2.3" / "VERSION.json").read_text(encoding="utf-8"))
    assert doc["plugin_version"] == "1.2.3"
    assert doc["plugin_commit"] == "0123456789abcdef"
    # read from the .so (KVA_RADIANCE_VERSION): the fake arch .so embeds 1.0.8
    assert doc["radiance_version"] == "1.0.8"
    assert doc["radiance_abi"] == "15.0.0"
    assert doc["gpu_targets"] == ["gfx1201"]
    sources = doc["fields"]
    assert set(sources) == {"plugin_version", "plugin_commit", "radiance_version",
                            "radiance_abi", "gpu_targets"}
    assert "1.0.8" in sources["radiance_version"] or "KVA_RADIANCE_VERSION" in \
        sources["radiance_version"]
    assert "--abi-version" in sources["radiance_abi"]
    assert sources["gpu_targets"] == "gfx* strings extracted from the packaged .so files"
    # no timestamps: the same inputs give the same VERSION.json bytes on any run
    assert "created" not in doc and "utc" not in doc and "time" not in doc


def test_merged_template_is_snippet_plus_base(inputs, tmp_path):
    out = tmp_path / "dist"
    run_packager(inputs, out, template=True)
    merged = (out / "kva-chat-template" / "chat_template.jinja")
    base = inputs["base"].read_bytes()
    # verify with the tool itself, and check the base bytes ride along unchanged
    result = subprocess.run(
        [sys.executable, str(TEMPLATE_TOOL), "check", "--spec", str(inputs["spec"]),
         "--base", str(inputs["base"]), "--merged", str(merged)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    block = merged.read_bytes()[: -len(base)]
    assert block.endswith(b"{#- kva-marker-end v1 #}")
    assert merged.read_bytes().endswith(base)


def test_sha_sums_refuse_on_mismatch(inputs, tmp_path):
    """The packaged plugin's SHA256SUMS must actually catch a changed file,
    and so must the projector's and the template's (their LICENSE included)."""
    out = tmp_path / "dist"
    run_packager(inputs, out, template=True)
    plugin = out / "radiance-kva-0.3.0"
    (plugin / "README.md").write_text("tampered\n", encoding="utf-8")
    result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=plugin,
                            capture_output=True, text=True)
    assert result.returncode != 0

    projector = out / "projector-qwen3.8-flash-next-bf16"
    (projector / "LICENSE").write_text("tampered license\n", encoding="utf-8")
    result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=projector,
                            capture_output=True, text=True)
    assert result.returncode != 0

    template = out / "kva-chat-template"
    (template / "LICENSE").write_text("tampered license\n", encoding="utf-8")
    result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=template,
                            capture_output=True, text=True)
    assert result.returncode != 0


# ------------------------------------------------------------------ determinism

def test_deterministic_tarballs(inputs, tmp_path):
    """Two runs over the same inputs give identical tarball bytes (and the same
    dist/SHA256SUMS), so a release can be re-verified bit for bit."""
    out1, out2 = tmp_path / "dist1", tmp_path / "dist2"
    run_packager(inputs, out1, template=True)
    run_packager(inputs, out2, template=True)
    for name in ("radiance-kva-0.3.0.tar.gz", "projector-qwen3.8-flash-next-bf16.tar.gz",
                 "kva-chat-template.tar.gz"):
        assert sha256_file(out1 / name) == sha256_file(out2 / name), name
    assert (out1 / "SHA256SUMS").read_bytes() == (out2 / "SHA256SUMS").read_bytes()
    assert (out1 / "radiance-kva-0.3.0" / "VERSION.json").read_bytes() == \
        (out2 / "radiance-kva-0.3.0" / "VERSION.json").read_bytes()


def test_tarball_metadata_is_pinned(inputs, tmp_path):
    """Sorted member names, fixed mtime/uid/gid, files 0644: nothing of the builder's
    environment can leak into the tarball."""
    out = tmp_path / "dist"
    run_packager(inputs, out)
    with tarfile.open(out / "radiance-kva-0.3.0.tar.gz") as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == sorted(m.name for m in members)
    for member in members:
        assert member.mtime == 0
        assert member.uid == 0 and member.gid == 0
        assert member.uname == "" and member.gname == ""
        assert member.mode == (0o755 if member.isdir() else 0o644)
    assert members[0].isdir() and members[0].name == "radiance-kva-0.3.0"
    assert "radiance-kva-0.3.0/VERSION.json" in [m.name for m in members]


# ------------------------------------------------------------------ refusals

def test_missing_license_refuses_by_name(inputs, tmp_path):
    inputs["repo"] = make_repo(tmp_path / "repo-nolic", with_license=False)
    inputs["packager"] = inputs["repo"] / "tools" / "package.py"
    result = run_packager(inputs, tmp_path / "dist", expect_rc=2)
    assert "LICENSE" in result.stderr
    assert str(inputs["repo"] / "LICENSE") in result.stderr
    assert not (tmp_path / "dist").exists()


def test_projector_hash_mismatch_refuses_by_name(inputs, tmp_path):
    corrupt = inputs["projector"] / "proj.L25.safetensors"
    data = bytearray(corrupt.read_bytes())
    data[-1] ^= 0xFF
    corrupt.write_bytes(bytes(data))
    result = run_packager(inputs, tmp_path / "dist", expect_rc=2)
    assert "proj.L25.safetensors" in result.stderr
    assert "manifest" in result.stderr


def test_projector_missing_file_refuses_by_name(inputs, tmp_path):
    (inputs["projector"] / "rowsel.safetensors").unlink()
    result = run_packager(inputs, tmp_path / "dist", expect_rc=2)
    assert "rowsel.safetensors" in result.stderr


def test_projector_extra_file_refuses_by_name(inputs, tmp_path):
    (inputs["projector"] / "unlisted.bin").write_bytes(b"not in the manifest\n")
    result = run_packager(inputs, tmp_path / "dist", expect_rc=2)
    assert "unlisted.bin" in result.stderr


def test_projector_dtype_refuses_by_name(inputs, tmp_path):
    """The package is named by the manifest's own projector.dtype: a manifest without
    the field, or a dtype that cannot name a directory, refuses naming kva.json."""
    manifest_path = inputs["projector"] / "kva.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["projector"]
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    result = run_packager(inputs, tmp_path / "dist", expect_rc=2)
    assert "kva.json" in result.stderr and "projector" in result.stderr
    assert not (tmp_path / "dist").exists()

    manifest["projector"] = {"dtype": "bf16/../evil", "layout": "plain_nk"}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    result = run_packager(inputs, tmp_path / "dist2", expect_rc=2)
    assert "kva.json" in result.stderr and "dtype" in result.stderr
    assert not (tmp_path / "dist2").exists()


def test_missing_home_inputs_refuse_by_name(inputs, tmp_path):
    (inputs["home"] / "kernels" / "kva.so").unlink()
    result = run_packager(inputs, tmp_path / "dist", expect_rc=2)
    assert "kernels/kva.so" in result.stderr
    assert not (tmp_path / "dist").exists()


def test_with_template_gates_spec_and_base_by_name(inputs, tmp_path):
    """--with-template needs its two inputs (refused by name), and the inputs without
    the flag refuse too: nothing silently builds or half-builds the parked package."""
    # the flag without the spec
    result = run_packager(inputs, tmp_path / "dist", template=True,
                          extra=["--template-spec", ""], expect_rc=2)
    assert "--with-template" in result.stderr and "--template-spec" in result.stderr
    # the flag without the base
    result = run_packager(inputs, tmp_path / "dist", template=True,
                          extra=["--base-template", ""], expect_rc=2)
    assert "--with-template" in result.stderr and "--base-template" in result.stderr
    # the spec or base WITHOUT the flag: not an input at all this release
    result = run_packager(inputs, tmp_path / "dist",
                          extra=["--template-spec", str(inputs["spec"])], expect_rc=2)
    assert "--with-template" in result.stderr
    result = run_packager(inputs, tmp_path / "dist",
                          extra=["--base-template", str(inputs["base"])], expect_rc=2)
    assert "--with-template" in result.stderr
    assert not (tmp_path / "dist").exists()


def test_missing_spec_and_base_refuse_by_name(inputs, tmp_path):
    """With --with-template a missing spec or base file refuses by name -- but without
    the flag they are not inputs at all and a run missing both succeeds (the package
    that needs them is parked)."""
    inputs["spec"].unlink()
    inputs["base"].unlink()
    result = run_packager(inputs, tmp_path / "dist-default", expect_rc=0)   # not inputs
    assert (tmp_path / "dist-default" / "radiance-kva-0.3.0").is_dir()

    missing = tmp_path / "nope.json"
    result = run_packager(inputs, tmp_path / "dist", template=True,
                          extra=["--template-spec", str(missing)], expect_rc=2)
    assert "--template-spec" in result.stderr and str(missing) in result.stderr

    make_spec(inputs["spec"])
    missing = tmp_path / "nope.jinja"
    result = run_packager(inputs, tmp_path / "dist2", template=True,
                          extra=["--base-template", str(missing)], expect_rc=2)
    assert "--base-template" in result.stderr and str(missing) in result.stderr


def test_refuses_non_empty_out_and_overlapping_out(inputs, tmp_path):
    out = tmp_path / "dist"
    run_packager(inputs, out)
    result = run_packager(inputs, out, expect_rc=2)          # not empty
    assert "not empty" in result.stderr

    inside = inputs["projector"] / "dist-inside"            # --out inside an input
    result = run_packager(inputs, inside, expect_rc=2)
    assert "overlaps" in result.stderr
    assert not inside.exists()


def test_bad_version_and_commit_refuse(inputs, tmp_path):
    result = run_packager(inputs, tmp_path / "dist", version="0.3", expect_rc=2)
    assert "--version" in result.stderr
    result = run_packager(inputs, tmp_path / "dist", commit="HEAD", expect_rc=2)
    assert "--commit" in result.stderr


# ------------------------------------------------------------------ the pure helpers

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pkg():
    return load_module(PACKAGE, "package_under_test")


def test_version_strings_ignores_prefixed_tags(pkg):
    data = (b"\x001.0.8\x00" b"GLIBCXX_3.4.32\x00" b"GCC: (GNU) 13.2.0\x00"
            b"\x002.39.2\x00")
    # GLIBCXX_3.4.32 and 13.2.0 are parts of larger strings; only bare d.d.d counts
    assert pkg.version_strings_in(data) == ["1.0.8", "2.39.2"]


def test_gpu_targets_extraction(pkg):
    data = b"gfx1201 gfx1201 gfx94 gfx942:local-no-dup\x00"
    assert pkg.gpu_targets_in(data) == ["gfx1201", "gfx94", "gfx942"]


def test_radiance_version_requires_disambiguation(pkg, tmp_path):
    so = tmp_path / "qwen4exp_fp8.so"
    so.write_bytes(b"\x001.0.8\x00\x002.0.0\x00")
    with pytest.raises(SystemExit):
        pkg.read_radiance_version(so, None)          # ambiguous without a hint
    version, source = pkg.read_radiance_version(so, "2.0.0")
    assert version == "2.0.0" and "embedded" in source
    with pytest.raises(SystemExit):
        pkg.read_radiance_version(so, "3.0.0")       # not among the embedded strings
    so.write_bytes(b"\x001.0.8\x00")
    version, source = pkg.read_radiance_version(so, None)
    assert version == "1.0.8" and "KVA_RADIANCE_VERSION" in source
    so.write_bytes(b"no bare release string here")
    version, source = pkg.read_radiance_version(so, "1.0.8")  # accepted, flagged as not found
    assert version == "1.0.8" and "no NUL-delimited d.d.d string found" in source
    with pytest.raises(SystemExit):
        pkg.read_radiance_version(so, None)          # nothing embedded, no override


def test_write_sums_covers_every_file_sorted(pkg, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("b\n")
    (tmp_path / "sub" / "a.txt").write_text("a\n")
    pkg.write_sums(tmp_path)
    lines = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == ["b.txt", "sub/a.txt"]
    for line in lines:
        digest, name = line.split("  ", 1)
        assert digest == sha256_file(tmp_path / name)


def test_deterministic_tar_function_is_pure(pkg, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.txt").write_text("x\n")
    (src / "d").mkdir()
    (src / "d" / "y.txt").write_text("y\n")
    os.utime(src / "x.txt", (1000, 1000))           # a different mtime must not leak
    pkg.deterministic_tar(src, tmp_path / "a.tar.gz", "src")
    os.utime(src / "x.txt", (2000, 2000))
    pkg.deterministic_tar(src, tmp_path / "b.tar.gz", "src")
    assert sha256_file(tmp_path / "a.tar.gz") == sha256_file(tmp_path / "b.tar.gz")
    with tarfile.open(tmp_path / "a.tar.gz") as tar:
        names = [m.name for m in tar.getmembers()]
    assert names == ["src", "src/d", "src/d/y.txt", "src/x.txt"]