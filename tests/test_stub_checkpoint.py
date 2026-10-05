"""tools/stub_checkpoint.py with HTTP replaced by a local file (no network). Run: python -m pytest tests/"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "dev"))   # the append route (A')
import stub_checkpoint as S  # noqa: E402


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A real shard served by byte range, as the HF resolve endpoint does."""
    shard = tmp_path / "remote.safetensors"
    save_file({"a.weight": torch.ones(4, 8, dtype=torch.bfloat16), "b.bias": torch.ones(8)}, str(shard))
    data = shard.read_bytes()
    monkeypatch.setattr(S, "get", lambda url, byte_range=None: data[byte_range[0]:byte_range[1] + 1])
    return data


def test_stub_has_the_real_header_and_size_and_zero_data(tmp_path, remote):
    out = tmp_path / "stub.safetensors"
    S.stub_shard("url", len(remote), out)
    header_end = 8 + int.from_bytes(remote[:8], "little")
    stub = out.read_bytes()
    assert len(stub) == len(remote) and stub[:header_end] == remote[:header_end]
    assert stub[header_end:] == bytes(len(remote) - header_end)


def test_stub_opens_as_safetensors_with_the_real_names_and_shapes(tmp_path, remote):
    out = tmp_path / "stub.safetensors"
    S.stub_shard("url", len(remote), out)
    with safe_open(str(out), "pt") as f:
        assert sorted(f.keys()) == ["a.weight", "b.bias"]
        assert list(f.get_slice("a.weight").get_shape()) == [4, 8]


def test_stub_refuses_a_size_that_disagrees_with_the_header(tmp_path, remote):
    with pytest.raises(SystemExit, match="but the file is"):
        S.stub_shard("url", len(remote) + 1, tmp_path / "stub.safetensors")


def test_extra_shard_is_linked_relatively_and_indexed(tmp_path):
    extra = tmp_path / "side" / "kva.safetensors"
    extra.parent.mkdir()
    save_file({"kva.st.24": torch.zeros(2)}, str(extra))
    out = tmp_path / "stub"
    out.mkdir()
    index = {"weight_map": {"a.weight": "model-1.safetensors"}}
    assert S.add_extra(out, index, extra) == 1
    assert index["weight_map"]["kva.st.24"] == "kva.safetensors"
    assert os.readlink(out / "kva.safetensors") == os.path.join("..", "side", "kva.safetensors")


def test_extra_shard_name_clash_is_refused(tmp_path):
    extra = tmp_path / "kva.safetensors"
    save_file({"a.weight": torch.zeros(2)}, str(extra))
    with pytest.raises(SystemExit, match="already in the checkpoint"):
        S.add_extra(tmp_path, {"weight_map": {"a.weight": "model-1.safetensors"}}, extra)
