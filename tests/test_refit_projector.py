"""tools/refit/fit_projector.py end to end on synthetic captures whose block inputs ARE a linear map of the boundary:
filed -> summed (capture.py acc) -> solved -> written. The refit must recover the map (held-out cosine ~1, far above a
random "shipped" projector scored on the same rows), write tcc's per-layer layout without `final`, and be readable by
tools/kva_sidecar.py. Needs KVA_RESEARCH_ROOT; SKIPPED without it."""
import argparse
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "refit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
pytestmark = pytest.mark.skipif(not os.environ.get("KVA_RESEARCH_ROOT"), reason="KVA_RESEARCH_ROOT not set")

from test_refit_capture import CHUNK, HC, HIDDEN, LAYERS, SPLIT, write_prompt  # noqa: E402

N_LAYERS = LAYERS[-1] + 1
PREFIX = "model.language_model.layers."


def write_checkpoint(root):
    """config.json + index + one shard holding the late layers' connection-read weights (all delta-net layers)."""
    g = torch.Generator().manual_seed(3)
    tensors = {}
    for L in LAYERS:
        p = f"{PREFIX}{L}.attn_hyper_connection."
        tensors[p + "hc_norm.weight"] = torch.randn(HC * HIDDEN, generator=g) * 0.1
        tensors[p + "input_mix_weight_down.weight"] = torch.randn(3, HC * HIDDEN, generator=g)
        tensors[p + "input_mix_weight_up.weight"] = torch.randn(HC * HIDDEN, 3, generator=g)
    root.mkdir()
    save_file(tensors, str(root / "model.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors"
                                                                                  for k in tensors}}))
    (root / "config.json").write_text(json.dumps({"text_config": dict(
        num_hidden_layers=N_LAYERS, hidden_size=HIDDEN, hc_count=HC, rms_norm_eps=1e-6,
        layer_types=["linear_attention"] * N_LAYERS)}))


def test_refit_recovers_a_linear_map_and_beats_a_random_one(tmp_path):
    import capture
    import fit_projector
    import kva_sidecar
    g = torch.Generator().manual_seed(7)
    maps = {L: (torch.randn(HIDDEN, HC * HIDDEN, generator=g) * 0.3, torch.randn(HIDDEN, generator=g)) for L in LAYERS}
    bi_of = lambda L, b: b @ maps[L][0].T + maps[L][1]  # noqa: E731
    engine, store = tmp_path / "engine", tmp_path / "store"
    docs = [("train", f, i) for i in range(12) for f in ("raw", "chat")] + [("held", f, 100) for f in ("raw", "chat")]
    for prompt_set, fmt, i in docs:
        ids = list(range(1000 * i + (fmt == "chat"), 1000 * i + 8 * CHUNK))
        jsonl = engine / "capture.jsonl"
        offset = len(jsonl.read_text().splitlines()) if jsonl.exists() else 0
        write_prompt(engine, ids, seed=i * 2 + (fmt == "chat"), bi_of=bi_of)
        entries, _ = capture.read_lines(jsonl, offset)
        prompt = dict(key=f"{prompt_set}/{fmt}/syn-doc{i}", format=fmt, ids=ids, n_tokens=len(ids))
        capture.file_prompt("activations", engine, store, prompt, entries, {"usage": {}})
    capture.cmd_acc(argparse.Namespace(store=str(store), set="train", sums=str(tmp_path / "sums"), follow=False,
                                       every=100, threads=1))
    write_checkpoint(tmp_path / "ckpt")
    shipped = {f"layer.{L}": torch.randn(HIDDEN, HC * HIDDEN + 1, generator=g).bfloat16() for L in LAYERS}
    save_file(shipped, str(tmp_path / "shipped.safetensors"))
    out = tmp_path / "proj"
    fit_projector.main(["--sums", str(tmp_path / "sums"), "--held", str(store / "held"), "--ckpt", str(tmp_path / "ckpt"),
                        "--shipped", str(tmp_path / "shipped.safetensors"), "--out", str(out), "--threads", "1"])
    report = json.loads((out / f"report-radiance-s{SPLIT}.json").read_text())
    assert report["chosen_lambda"] in report["lambdas"]
    assert report["rows"]["raw"] == report["rows"]["chat"] == 12 * (8 * CHUNK // 8)      # 16 rows a document
    assert report["heldout"]["refit"]["bi"] > 0.99 > report["heldout"]["shipped"]["bi"]
    written = load_file(str(out / f"kva-radiance-s{SPLIT}.safetensors"))
    assert sorted(written) == [f"layer.{L}" for L in LAYERS]
    assert all(t.dtype == torch.bfloat16 and t.shape == (HIDDEN, HC * HIDDEN + 1) for t in written.values())
    split, tensors = kva_sidecar.projector_tensors(out / f"kva-radiance-s{SPLIT}.safetensors", "kva.projr")
    assert split == SPLIT and torch.allclose(tensors[f"kva.projr.{SPLIT}.weight"].float(), maps[SPLIT][0], atol=0.05)
