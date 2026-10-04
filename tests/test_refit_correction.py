"""tools/refit/fit_correction.py on synthetic state captures in the plugin's layout (notes/arch.md "Capture"):
C = mean(S_exact - S_pred) over matched approximate chunk ends, per rank, in st_hook's format with rank 0's heads
first after tools/kva_sidecar.py; tail chunks unused; refusals for a missing exact record and for too few chunk ends."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "refit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fit_correction as FC  # noqa: E402
import kva_sidecar as K  # noqa: E402

LAYERS, HEADS, D = [2, 3], 2, 3          # per rank: 2 heads of a 4-head model, 3 x 3 states


def write_store(root, mode, prompts, value):
    """prompts: {name: [(P, n_tok, approximate)]}; state value(prompt, P, rank) fills each record."""
    for name, chunks in prompts.items():
        d = root / "sterm" / "raw" / name
        d.mkdir(parents=True)
        lines = []
        for p, n, approx in chunks:
            for rank in (0, 1):
                f = f"state.p{p}.h{name:0>16}.r{rank}.npy"
                np.save(d / f, np.full((len(LAYERS), HEADS, D, D), value(name, p, rank), dtype=np.float32))
                lines.append(dict(file=f, chunk_start=p, last_position=p + n - 1, n_tok=n, rank=rank, world=2,
                                  heads=[rank * HEADS, (rank + 1) * HEADS], layers=LAYERS, approximate=approx,
                                  mode=mode, applied_before_copy=False))
        (d / "state.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (d / "meta.json").write_text("{}")


PROMPTS = {"a": [(0, 16, True), (16, 16, True), (32, 8, False)], "b": [(0, 16, True), (16, 4, False)]}


def stores(tmp_path, pred_prompts=PROMPTS):
    exact_prompts = {k: [(p, n, False) for p, n, _ in v] for k, v in PROMPTS.items()}
    write_store(tmp_path / "exact", "off", exact_prompts, lambda name, p, rank: 10.0 + rank)
    # error d = exact - pred: rank 0 -> 1, 2, 3 at the three approximate ends; rank 1 -> 10x that
    errors = {("a", 0): 1.0, ("a", 16): 2.0, ("b", 0): 3.0}
    write_store(tmp_path / "pred", "speed", pred_prompts,
                lambda name, p, rank: 10.0 + rank - errors.get((name, p), 99.0) * (1 if rank == 0 else 10))
    return str(tmp_path / "exact"), str(tmp_path / "pred")


def test_constant_is_the_mean_error_in_st_hook_format(tmp_path):
    exact, pred = stores(tmp_path)
    out = tmp_path / "st"
    FC.main(["--exact", exact, "--pred", pred, "--out", str(out), "--min-count", "3", "--min-prompts", "2"])
    r0 = torch.load(out / "kva-radiance-s2-st.rank0.pt", weights_only=True)
    assert r0["count"] == {2: 3, 3: 3} and torch.allclose(r0["sum"][3], torch.full((HEADS, D, D), 6.0))
    st = K.correction_tensors(out / "kva-radiance-s2-st.rank0.pt", out / "kva-radiance-s2-st.rank1.pt", 2, "kva.str", None)
    assert torch.allclose(st["kva.str.2"][:HEADS], torch.full((HEADS, D, D), 2.0))       # rank 0's heads first
    assert torch.allclose(st["kva.str.2"][HEADS:], torch.full((HEADS, D, D), 20.0))
    report = json.loads((out / "report-radiance-st.json").read_text())
    assert report["ranks"]["0"]["chunk_ends"] == 3 and report["ranks"]["0"]["prompts"] == ["a", "b"]
    assert abs(report["ranks"]["0"]["layers"]["2"]["constant_share"] - 12 / 14) < 1e-4    # 3*2^2 / (1+4+9)


def test_too_few_chunk_ends_refuse(tmp_path):
    exact, pred = stores(tmp_path)
    with pytest.raises(SystemExit, match="R43 needs"):
        FC.main(["--exact", exact, "--pred", pred, "--out", str(tmp_path / "st")])


def test_an_unmatched_approximate_record_refuses(tmp_path):
    extra = dict(PROMPTS, c=[(0, 16, True)])
    exact, pred = stores(tmp_path, extra)
    with pytest.raises(SystemExit, match="no exact-run record"):
        FC.main(["--exact", exact, "--pred", pred, "--out", str(tmp_path / "st"), "--min-count", "1",
                 "--min-prompts", "1"])
