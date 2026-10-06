# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""tools/ridgefill_sidecar.py on small synthetic inputs (no model files, no GPU), plus a check of a real build when
RIDGEFILL_SIDECAR (+ RIDGEFILL_TOKENIZER) point at one; that check is SKIPPED otherwise, never passed.
Run: python -m pytest tests/
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "dev"))   # the append route (A')
import ridgefill_rules as R  # noqa: E402
import ridgefill_sidecar as K  # noqa: E402

INF = float("-inf")
# Byte-level BPE pieces: "Ġ" is the space byte, "Ċ" the newline byte.
PIECES = ["ĠThe", "ing", "Ġthe", "3.14", "Ġ", "Ċ", "x2", ",", "<|im_end|>"]
VOCAB = 12                      # model rows past the tokenizer's 9 ids are padding


class FakeTokenizer:
    added_tokens_decoder = {8: "<|im_end|>"}

    def __len__(self):
        return len(PIECES)

    def convert_ids_to_tokens(self, ids):
        return [PIECES[i] for i in ids]


def write_inputs(root):
    proj = {"layer.2": torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7).bfloat16(),
            "layer.3": -torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7).bfloat16(),
            "final": torch.zeros(8, 7, dtype=torch.bfloat16)}
    save_file(proj, str(root / "proj.safetensors"))
    for rank, base in ((0, 1.0), (1, 100.0)):
        sums = {2: torch.full((2, 3, 3), base * 8), 3: torch.full((2, 3, 3), base * 4)}
        torch.save({"sum": sums, "count": {2: 4, 3: 2}}, root / f"st.rank{rank}.pt")
    save_file({"logfreq": -torch.arange(1, VOCAB + 1, dtype=torch.float32)}, str(root / "freq.safetensors"))
    tok_dir = root / "tok"
    tok_dir.mkdir()
    (tok_dir / "config.json").write_text(json.dumps({"text_config": {"vocab_size": VOCAB}}))
    (tok_dir / "tokenizer.json").write_text("{}")
    (tok_dir / "tokenizer_config.json").write_text("{}")
    return ["--proj", str(root / "proj.safetensors"), "--st", str(root / "st.rank0.pt"), str(root / "st.rank1.pt"),
            "--freq", str(root / "freq.safetensors"), "--tokenizer", str(tok_dir)]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("sidecar")
    source_args = write_inputs(root)
    original = R.load_tokenizer
    R.load_tokenizer = lambda path: FakeTokenizer()
    try:
        K.main(["build", *source_args, "--out", str(root / "out")])
    finally:
        R.load_tokenizer = original
    shard = root / "out" / K.SHARD
    with safe_open(str(shard), "pt") as f:
        meta = f.metadata()
    return dict(root=root, args=source_args, shard=shard, tensors=load_file(str(shard)), meta=meta)


def test_projector_weight_is_every_column_but_the_last(built):
    want = torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7)[:, :6].bfloat16()
    assert torch.equal(built["tensors"]["ridgefill.proj.2.weight"], want)


def test_projector_bias_is_the_last_column(built):
    want = torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7)[:, 6].bfloat16()
    assert torch.equal(built["tensors"]["ridgefill.proj.2.bias"], want)


def test_projector_dtypes_stay_bf16(built):
    assert built["tensors"]["ridgefill.proj.3.weight"].dtype == torch.bfloat16
    assert built["tensors"]["ridgefill.proj.3.bias"].dtype == torch.bfloat16


def test_final_is_not_converted(built):
    assert not any("final" in name for name in built["tensors"])


def test_correction_is_sum_over_count_rank0_heads_first(built):
    # layer 2: rank 0 = 8 / 4 = 2, rank 1 = 800 / 4 = 200; heads [0, 2) are rank 0's.
    st = built["tensors"]["ridgefill.st.2"]
    assert st.shape == (4, 3, 3) and st.dtype == torch.float32
    assert torch.equal(st[:2], torch.full((2, 3, 3), 2.0))
    assert torch.equal(st[2:], torch.full((2, 3, 3), 200.0))


def test_swapped_control_puts_rank1_heads_first(built):
    swap = built["tensors"]["ridgefill.stswap.3"]
    assert torch.equal(swap[:2], torch.full((2, 3, 3), 200.0))
    assert torch.equal(swap[2:], torch.full((2, 3, 3), 2.0))


def test_score_is_rarity_for_kept_classes(built):
    score = built["tensors"]["ridgefill.rowsel.score"]
    # " The" (id 0, cap): -logfreq = 1; "ing" (id 1, piece): 2; "x2" (id 6, mixed): 7
    assert [score[0].item(), score[1].item(), score[6].item()] == [1.0, 2.0, 7.0]


def test_score_is_minus_inf_for_dropped_classes_specials_and_padding(built):
    score = built["tensors"]["ridgefill.rowsel.score"]
    dropped = [2, 3, 4, 5, 7, 8, 9, 10, 11]   # " the", "3.14", " ", "\n", ",", <|im_end|>, 3 padding rows
    assert [score[i].item() for i in dropped] == [INF] * len(dropped)


def test_score_tables_cover_the_model_vocab(built):
    for name in ("ridgefill.rowsel.score", "ridgefill.rowsel.score_none", "ridgefill.rowsel.score_all"):
        assert built["tensors"][name].shape == (VOCAB,) and built["tensors"][name].dtype == torch.float32


def test_score_none_selects_nothing(built):
    assert torch.equal(built["tensors"]["ridgefill.rowsel.score_none"], torch.full((VOCAB,), INF))


def test_score_all_keeps_every_id_at_one_score(built):
    assert torch.equal(built["tensors"]["ridgefill.rowsel.score_all"], torch.zeros(VOCAB))


def test_metadata_names_split_share_and_classes(built):
    meta = built["meta"]
    assert (meta["ridgefill.split"], meta["ridgefill.rowsel.share"], meta["ridgefill.rowsel.classes"]) == ("2", "0.25", "cap,mixed,piece")


def test_metadata_hashes_are_the_source_files(built):
    root = built["root"]
    assert built["meta"]["ridgefill.src.proj.sha256"] == K.sha256_file(root / "proj.safetensors")
    assert built["meta"]["ridgefill.src.st1.sha256"] == K.sha256_file(root / "st.rank1.pt")
    assert built["meta"]["ridgefill.src.tokenizer_json.sha256"] == K.sha256_file(root / "tok" / "tokenizer.json")


def test_index_names_only_the_shard(built):
    index = json.loads((built["shard"].parent / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {K.SHARD}
    assert sorted(index["weight_map"]) == sorted(built["tensors"])


def test_set_file_carries_the_header_metadata(built):
    lines = (built["shard"].parent / K.SET_FILE).read_text().splitlines()
    assert dict(line.split("=", 1) for line in lines) == built["meta"]


def test_verify_passes_on_its_own_sidecar(built):
    assert K.main(["verify", str(built["shard"]), *built["args"]]) == 0


def test_verify_fails_when_the_rank_files_are_swapped(built):
    args = list(built["args"])
    i = args.index("--st")
    args[i + 1], args[i + 2] = args[i + 2], args[i + 1]
    assert K.main(["verify", str(built["shard"]), *args]) == 1


def test_verify_reads_saved_rad_info_meta_output(built, tmp_path):
    rows = "".join(f"  {k}  {v}\n" for k, v in built["meta"].items())
    saved = tmp_path / "meta.txt"
    saved.write_text("model.rad\n\nmetadata (99)\n  n_layers  48\n" + rows)
    assert K.main(["verify", str(saved), *built["args"]]) == 0


def test_two_builds_of_the_same_inputs_are_byte_identical(built, tmp_path):
    # Regression: safetensors writes __metadata__ in hash order, so the shard's sha256 changed run to run.
    original = R.load_tokenizer
    R.load_tokenizer = lambda path: FakeTokenizer()
    try:
        for _ in range(3):
            K.main(["build", *built["args"], "--out", str(tmp_path / "again")])
            assert (tmp_path / "again" / K.SHARD).read_bytes() == built["shard"].read_bytes()
    finally:
        R.load_tokenizer = original


def test_build_refuses_a_freq_table_of_the_wrong_length(built, tmp_path):
    args = list(built["args"])
    save_file({"logfreq": torch.zeros(VOCAB - 1)}, str(tmp_path / "short.safetensors"))
    args[args.index("--freq") + 1] = str(tmp_path / "short.safetensors")
    with pytest.raises(SystemExit, match="vocab rows"):
        K.main(["build", *args, "--out", str(tmp_path / "out")])


def test_build_refuses_a_missing_input_by_name(built, tmp_path):
    args = list(built["args"])
    args[args.index("--proj") + 1] = str(tmp_path / "absent.safetensors")
    with pytest.raises(SystemExit, match="proj: "):
        K.main(["build", *args, "--out", str(tmp_path / "out")])


@pytest.mark.skipif(not (os.environ.get("RIDGEFILL_SIDECAR") and os.environ.get("RIDGEFILL_TOKENIZER")),
                    reason="needs RIDGEFILL_SIDECAR (a built ridgefill-sidecar.safetensors) and RIDGEFILL_TOKENIZER")
def test_real_sidecar_layout_and_known_tokens():
    tensors = load_file(os.environ["RIDGEFILL_SIDECAR"])
    vocab = R.model_vocab(os.environ["RIDGEFILL_TOKENIZER"])
    tok = R.load_tokenizer(os.environ["RIDGEFILL_TOKENIZER"])
    score = tensors["ridgefill.rowsel.score"]
    proj = sorted(int(k.split(".")[2]) for k in tensors if k.startswith("ridgefill.proj.") and k.endswith(".weight"))
    st = sorted(int(k.split(".")[2]) for k in tensors if k.startswith("ridgefill.st."))
    assert proj == list(range(proj[0], proj[-1] + 1)) and len(tensors) == 2 * len(proj) + 2 * len(st) + 3
    w, b = tensors[f"ridgefill.proj.{proj[0]}.weight"], tensors[f"ridgefill.proj.{proj[0]}.bias"]
    assert w.dtype == b.dtype == torch.bfloat16 and b.shape == (w.shape[0],)
    assert tensors[f"ridgefill.st.{st[0]}"].dtype == torch.float32 and score.shape == (vocab,)
    ids = {text: tok.encode(text, add_special_tokens=False) for text in (" The", "ing", " the", "3.14")}
    assert len(ids[" The"]) == len(ids["ing"]) == len(ids[" the"]) == 1
    assert math.isfinite(score[ids[" The"][0]]) and math.isfinite(score[ids["ing"][0]])
    assert score[ids[" the"][0]] == INF and all(score[i] == INF for i in ids["3.14"])


@pytest.fixture(scope="module")
def refit(built):
    out = built["root"] / "refit"
    args = [a for a in built["args"]]
    cut = args.index("--freq")
    source_args = args[:cut]                      # --proj and --st only
    K.main(["build", "--names", "refit", *source_args, "--out", str(out)])
    shard = out / K.NAMES["refit"]["shard"]
    with safe_open(str(shard), "pt") as f:
        meta = f.metadata()
    return dict(args=source_args, shard=shard, tensors=load_file(str(shard)), meta=meta)


def test_refit_names_are_projr_and_str_only(refit):
    assert sorted(refit["tensors"]) == ["ridgefill.projr.2.bias", "ridgefill.projr.2.weight", "ridgefill.projr.3.bias",
                                        "ridgefill.projr.3.weight", "ridgefill.str.2", "ridgefill.str.3"]


def test_refit_tensors_equal_the_shipped_layout(built, refit):
    assert torch.equal(refit["tensors"]["ridgefill.projr.2.weight"], built["tensors"]["ridgefill.proj.2.weight"])
    assert torch.equal(refit["tensors"]["ridgefill.str.3"], built["tensors"]["ridgefill.st.3"])


def test_refit_hash_keys_do_not_collide_with_the_shipped_ones(built, refit):
    assert sorted(k for k in refit["meta"] if k.startswith("ridgefill.src.")) == [
        "ridgefill.src.projr.sha256", "ridgefill.src.str0.sha256", "ridgefill.src.str1.sha256"]
    assert refit["meta"]["ridgefill.src.projr.sha256"] == built["meta"]["ridgefill.src.proj.sha256"]


def test_refit_verify_passes(refit):
    assert K.main(["verify", str(refit["shard"]), "--names", "refit", *refit["args"]]) == 0


def test_refit_refuses_row_selection_inputs(built, tmp_path):
    with pytest.raises(SystemExit, match="drop --freq"):
        K.main(["build", "--names", "refit", *built["args"], "--out", str(tmp_path / "x")])


def test_shipped_requires_row_selection_inputs(refit, tmp_path):
    with pytest.raises(SystemExit, match="--freq and --tokenizer are required"):
        K.main(["build", *refit["args"], "--out", str(tmp_path / "x")])
