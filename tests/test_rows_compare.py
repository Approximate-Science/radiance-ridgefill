# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""tools/rows_compare.py helpers, and the R33 fixture against a built sidecar (RIDGEFILL_SIDECAR; SKIPPED without it).
Run: python -m pytest tests/
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ridgefill_rules as R  # noqa: E402
import rows_compare as C  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rowsel_quick9.json"


def test_chunk_rows_select_within_each_whole_chunk_and_offset_them():
    # ids 1 score 5 (kept), ids 0 never kept; share 1 keeps every match. Chunk 2 = [4, 6) is past P = 5's last
    # whole chunk, so it selects nothing.
    score = np.array([-np.inf, 5.0], dtype=np.float32)
    assert C.chunk_rows([1, 0, 0, 1, 1, 1], P=5, window=2, score=score, share=1.0) == [0, 3]


def test_dump_rows_go_to_the_doc_whose_ids_they_carry(tmp_path):
    docs = [{"ids": [1, 2, 3, 4]}, {"ids": [5, 6, 7, 8]}]
    dump = tmp_path / "rows.jsonl"
    dump.write_text(json.dumps({"chunk_start": 2, "n_tok": 2, "rows_idx": [1, -1], "token_ids": [7, 8]}) + "\n")
    rows, starts = C.read_dump(dump, docs)
    assert rows == {1: {3}} and starts == {1: {2}}


def test_dump_chunk_matching_no_doc_is_refused(tmp_path):
    dump = tmp_path / "rows.jsonl"
    dump.write_text(json.dumps({"chunk_start": 0, "n_tok": 1, "rows_idx": [0], "token_ids": [9]}) + "\n")
    with pytest.raises(SystemExit, match="matches 0 corpus docs"):
        C.read_dump(dump, [{"ids": [1, 2]}])


def test_dump_row_past_the_chunk_is_refused(tmp_path):
    dump = tmp_path / "rows.jsonl"
    dump.write_text(json.dumps({"chunk_start": 0, "n_tok": 2, "rows_idx": [2], "token_ids": [1, 2]}) + "\n")
    with pytest.raises(SystemExit, match="out of range"):
        C.read_dump(dump, [{"ids": [1, 2]}])


def test_fixture_rows_follow_from_its_own_kept_scores():
    # What the C++ kernel test does: a vocab table of -inf with the fixture's kept ids set.
    fixture = json.loads(FIXTURE.read_text())
    score = np.full(fixture["vocab"], -np.inf, dtype=np.float32)
    score[fixture["kept_ids"]] = fixture["kept_scores"]
    got = [R.select_rows(doc["token_ids"], score, fixture["share"]) for doc in fixture["docs"]]
    assert got == [doc["rows"] for doc in fixture["docs"]]


def test_fixture_has_a_half_to_even_rounding_case():
    # 578 matches * 0.25 = 144.5 -> 144 rows: a kernel rounding half away from zero keeps 145 and fails.
    fixture = json.loads(FIXTURE.read_text())
    assert any(doc["matches"] % 4 == 2 and doc["k"] == doc["matches"] // 4 for doc in fixture["docs"])


@pytest.mark.skipif(not os.environ.get("RIDGEFILL_SIDECAR"), reason="needs RIDGEFILL_SIDECAR (a built ridgefill-sidecar.safetensors)")
def test_fixture_matches_the_sidecar_score_table():
    fixture = json.loads(FIXTURE.read_text())
    with safe_open(os.environ["RIDGEFILL_SIDECAR"], "np") as f:
        score = np.ascontiguousarray(f.get_tensor(fixture["score_tensor"]), dtype="<f4")
    assert hashlib.sha256(score.tobytes()).hexdigest() == fixture["score_sha256"]
    assert score[fixture["kept_ids"]].tolist() == fixture["kept_scores"]
    got = [R.select_rows(doc["token_ids"], score, fixture["share"]) for doc in fixture["docs"]]
    assert got == [doc["rows"] for doc in fixture["docs"]]
