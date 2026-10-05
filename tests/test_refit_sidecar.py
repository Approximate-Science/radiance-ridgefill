"""tools/kva_sidecar.py --names refit for Stage 6's two appends: the projector alone first (no kva.str.*, so
RADIANCE_KVA_ST=refit serves with no correction while the correction is being fitted), then the full refit set,
whose projector tensors must be byte-identical to the first build's (the second append reuses them by name)."""
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "dev"))   # the append route (A')
import kva_sidecar as K  # noqa: E402


@pytest.fixture
def inputs(tmp_path):
    save_file({f"layer.{L}": torch.randn(4, 9).bfloat16() for L in (2, 3)}, str(tmp_path / "proj.safetensors"))
    for rank in (0, 1):
        torch.save({"sum": {2: torch.full((2, 3, 3), 6.0 + rank)}, "count": {2: 3}}, tmp_path / f"st{rank}.pt")
    return tmp_path


def build(root, out, *extra):
    K.main(["build", "--names", "refit", "--proj", str(root / "proj.safetensors"), *extra, "--out", str(root / out)])
    shard = root / out / K.NAMES["refit"]["shard"]
    with safe_open(str(shard), "pt") as f:
        meta = f.metadata()
    return load_file(str(shard)), meta, (root / out / K.SET_FILE).read_text()


def test_projector_alone_has_no_correction(inputs):
    tensors, meta, set_file = build(inputs, "projr")
    assert sorted(tensors) == ["kva.projr.2.bias", "kva.projr.2.weight", "kva.projr.3.bias", "kva.projr.3.weight"]
    assert sorted(k for k in meta if k.startswith("kva.src.")) == ["kva.src.projr.sha256"]
    assert "kva.src.projr.sha256=" in set_file and "str0" not in set_file
    proj = load_file(str(inputs / "proj.safetensors"))
    assert torch.equal(tensors["kva.projr.3.weight"], proj["layer.3"][:, :-1])
    assert torch.equal(tensors["kva.projr.3.bias"], proj["layer.3"][:, -1])
    shard = inputs / "projr" / K.NAMES["refit"]["shard"]
    assert K.main(["verify", str(shard), "--names", "refit", "--proj", str(inputs / "proj.safetensors")]) == 0


def test_full_refit_set_reuses_the_same_projector_bytes(inputs):
    first, _, _ = build(inputs, "projr")
    full, meta, _ = build(inputs, "full", "--st", str(inputs / "st0.pt"), str(inputs / "st1.pt"))
    assert sorted(k for k in full if k.startswith("kva.str.")) == ["kva.str.2"]
    assert all(torch.equal(first[k], full[k]) for k in first)
    assert torch.equal(full["kva.str.2"][:2], torch.full((2, 3, 3), 2.0))      # rank 0's heads first: 6 / 3
    assert {"kva.src.str0.sha256", "kva.src.str1.sha256"} <= set(meta)


def test_shipped_still_requires_the_correction(inputs):
    with pytest.raises(SystemExit, match="needs --st"):
        K.main(["build", "--proj", str(inputs / "proj.safetensors"), "--out", str(inputs / "x")])
