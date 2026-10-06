"""tools/refit/convert.py + capture.py on synthetic captures written in the plugin's layout (notes/arch.md
"Capture"): the record the research code reads, the chunk keys, the replay refusal, filing and the running sums.
Needs RIDGEFILL_RESEARCH_ROOT (the research code is imported read-only); without it every test here is SKIPPED."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "refit"))
pytestmark = pytest.mark.skipif(not os.environ.get("RIDGEFILL_RESEARCH_ROOT"), reason="RIDGEFILL_RESEARCH_ROOT not set")

SPLIT, LAYERS, HIDDEN, HC, CHUNK, STRIDE = 2, [2, 3], 4, 4, 16, 8


def bf16_bits(t):
    return t.to(torch.bfloat16).view(torch.int16).numpy().view("<u2")


def write_chunk(d, ids, start, rng, bi_of=None):
    """One chunk's files + capture.jsonl line, as arch/ridgefill_dump.h writes them; bi_of(layer, boundary) gives the
    block inputs (random when absent)."""
    import capture
    n = len(ids)
    pos = np.arange(start, start + n, dtype="<i4")
    rows = np.array([i for i in range(n) if (start + i) % STRIDE == 0], dtype="<i4")
    prefix = f"chunk.p{start}.h{capture.fnv1a64(ids)}"
    b = torch.from_numpy(rng.standard_normal((len(rows), HC * HIDDEN))).float()
    np.save(d / f"{prefix}.boundary.npy", bf16_bits(b))
    for layer in LAYERS:
        bi = bi_of(layer, b.bfloat16().float()) if bi_of else torch.from_numpy(rng.standard_normal((len(rows), HIDDEN)))
        np.save(d / f"{prefix}.bi.{layer}.npy", bf16_bits(bi))
    np.save(d / f"{prefix}.rows.npy", rows)
    np.save(d / f"{prefix}.ids.npy", np.asarray(ids, dtype="<i4"))
    np.save(d / f"{prefix}.pos.npy", pos)
    entry = dict(prefix=prefix, chunk_start=start, n_tok=n, stride=STRIDE, rows=len(rows), split=SPLIT,
                 layers=LAYERS, hidden=HIDDEN, hc=HC, dtype="bf16")
    with open(d / "capture.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def write_prompt(d, ids, seed=0, bi_of=None):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    return [write_chunk(d, ids[s:s + CHUNK], s, rng, bi_of) for s in range(0, len(ids), CHUNK)]


@pytest.fixture
def mods():
    import capture
    import convert
    import ridgefill_research
    ridgefill_research.root()
    return capture, convert


def test_read_chunk_is_the_tcc_record(tmp_path, mods):
    capture, convert = mods
    ids = list(range(100, 140))                     # 40 tokens: chunks 16 + 16 + 8
    write_prompt(tmp_path, ids)
    entries = convert.chunks(tmp_path)
    assert [e["chunk_start"] for e in entries] == [0, 16, 32]
    rec = convert.read_chunk(tmp_path, entries[1])
    raw = np.load(tmp_path / f"{entries[1]['prefix']}.boundary.npy")
    assert rec[f"boundary_{SPLIT}"].dtype == torch.bfloat16
    assert np.array_equal(rec[f"boundary_{SPLIT}"].view(torch.int16).numpy().view("<u2"), raw)
    assert rec["positions"].tolist() == [16, 24] and rec["input_ids"].tolist() == [116, 124]
    assert rec["positions"].dtype == torch.int64 and rec["input_ids"].dtype == torch.int32
    doc = convert.read_document(tmp_path)
    assert doc["positions"].tolist() == [0, 8, 16, 24, 32] and doc[f"block_input_{LAYERS[-1]}"].shape == (5, HIDDEN)


def test_pt_files_read_by_the_research_loaders(tmp_path, mods):
    _, convert = mods
    from b0 import capture_files as cf
    from qfn import fit
    write_prompt(tmp_path / "cap", list(range(40)))
    assert convert.write_pt(tmp_path / "cap", tmp_path / "pt") == 3
    files = cf.files(tmp_path / "pt")
    assert [f.name for f in files] == ["capture_00000.pt", "capture_00001.pt", "capture_00002.pt"]
    keys = [f"boundary_{SPLIT}"] + [f"block_input_{L}" for L in LAYERS]
    got = list(fit.records([tmp_path / "pt"], keys))
    assert sum(r[keys[0]].shape[0] for _, r in got) == 5
    assert cf.positions(cf.load(files[2])).tolist() == [32]


def test_sums_add_takes_the_record_and_matches_the_scatter(tmp_path, mods):
    capture, convert = mods
    write_prompt(tmp_path, list(range(64)))
    entries = convert.chunks(tmp_path)
    lay = capture.layout_for(entries[0])
    assert lay.targets == [f"bi.{L}" for L in LAYERS] and lay.dim == HC * HIDDEN
    from qfn import fit
    sums = fit.Sums(lay)
    rec = convert.read_document(tmp_path)
    sums.add(rec, {}, "cpu")
    x = rec[f"boundary_{SPLIT}"].double()
    xc = x - x.mean(0)
    assert sums.n == 8
    assert torch.allclose(sums.sxx, xc.T @ xc, atol=1e-4)


def test_hc_identity_reproduces_the_read(tmp_path, mods):
    capture, convert = mods
    from qfn import hc as H
    from qfn.weights import Checkpoint, PREFIX
    g = torch.Generator().manual_seed(1)
    names = {f"{PREFIX}{SPLIT}.attn_hyper_connection.hc_norm.weight": torch.randn(HC * HIDDEN, generator=g) * 0.1,
             f"{PREFIX}{SPLIT}.attn_hyper_connection.input_mix_weight_down.weight": torch.randn(3, HC * HIDDEN, generator=g),
             f"{PREFIX}{SPLIT}.attn_hyper_connection.input_mix_weight_up.weight": torch.randn(HC * HIDDEN, 3, generator=g)}
    ck = Checkpoint(tensors=names, config=dict(num_hidden_layers=4, hidden_size=HIDDEN, hc_count=HC))
    b = torch.randn(6, HC * HIDDEN, generator=g).to(torch.bfloat16)
    rec = {f"boundary_{SPLIT}": b, f"block_input_{SPLIT}": H.mix(b, ck.hc_read(SPLIT)).to(torch.bfloat16)}
    cos, split = convert.hc_identity(rec, ck)
    assert split == SPLIT and cos > 0.999
    rec[f"block_input_{SPLIT}"] = torch.randn(6, HIDDEN, generator=g).to(torch.bfloat16)   # negative control
    assert convert.hc_identity(rec, ck)[0] < 0.9


def test_chunk_keys_and_the_replay_refusal(tmp_path, mods):
    capture, _ = mods
    ids = list(range(40))
    expected = capture.expected_chunks(ids, CHUNK)
    entries = write_prompt(tmp_path, ids)
    assert sorted(expected) == sorted(capture.entry_key("activations", e) for e in entries)
    prompt = dict(key="train/raw/x-doc")
    capture.check_entries("activations", entries, expected, prompt)
    with pytest.raises(SystemExit, match="replayed"):
        capture.check_entries("activations", entries[:2], expected, prompt)
    wrong = dict(entries[0], prefix=entries[0]["prefix"].replace("p0.", "p8."))
    with pytest.raises(SystemExit, match="not one of its chunks"):
        capture.check_entries("activations", [wrong] + entries[1:], expected, prompt)


def test_state_records_need_every_rank(tmp_path, mods):
    capture, _ = mods
    ids = list(range(20))
    expected = capture.expected_chunks(ids, CHUNK)
    entries = [dict(file=f"state.{k}.r{r}.npy", chunk_start=s, n_tok=n, rank=r, world=2)
               for k, (s, n) in expected.items() for r in (0, 1)]
    capture.check_entries("state", entries, expected, dict(key="sterm/raw/x"))
    with pytest.raises(SystemExit, match="replayed"):
        capture.check_entries("state", entries[:-1], expected, dict(key="sterm/raw/x"))


def test_file_then_acc_deletes_only_after_the_checkpoint(tmp_path, mods):
    capture, convert = mods
    from qfn import fit
    engine, store, sums = tmp_path / "engine", tmp_path / "store", tmp_path / "sums"
    for i, fmt in enumerate(("raw", "chat", "raw")):
        ids = list(range(1000 * i, 1000 * i + 24))
        offset = len((engine / "capture.jsonl").read_text().splitlines()) if (engine / "capture.jsonl").exists() else 0
        write_prompt(engine, ids, seed=i)
        entries, _ = capture.read_lines(engine / "capture.jsonl", offset)
        prompt = dict(key=f"train/{fmt}/src-doc{i}", format=fmt, ids=ids, n_tokens=len(ids))
        capture.file_prompt("activations", engine, store, prompt, entries, {"usage": {}})
    assert not list(engine.glob("chunk.*")) and len(capture.ready(store, "train")) == 3
    args = argparse.Namespace(store=str(store), set="train", sums=str(sums), follow=False, every=100, threads=1)
    capture.cmd_acc(args)
    assert capture.ready(store, "train") == []
    raw = fit.stats_parts(sums / f"acc-raw-s{SPLIT}.pt")[0]
    assert raw[0] == "raw" and raw[1]["n"] == 6
    assert capture.summed_keys(sums) == {"train/raw/src-doc0", "train/chat/src-doc1", "train/raw/src-doc2"}
    capture.cmd_acc(args)                                     # rerun: nothing new, nothing added twice
    assert fit.stats_parts(sums / f"acc-raw-s{SPLIT}.pt")[0][1]["n"] == 6
