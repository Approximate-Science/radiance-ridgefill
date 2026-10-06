"""tools/kld_corpus.py with a character-level fake tokenizer (no model files). Run: python -m pytest tests/"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kld_corpus as C  # noqa: E402
import ridgefill_rules as R  # noqa: E402


class CharTokenizer:
    """One token per character, except that a text ENDING in "!" gets one extra empty token: a stand-in for a
    BPE whose encoding of a cut prefix is not the prefix of the whole text's encoding."""

    def __call__(self, text, add_special_tokens=False):
        ids = list(text) + ([""] if text.endswith("!") else [])
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return "".join(ids)


def test_cut_is_the_largest_chunk_multiple():
    prompt, rec = C.cut_doc(CharTokenizer(), "abcdefghij", chunk=4, max_tokens=32)
    assert (prompt, rec["n"], rec["move"], rec["ids_equal"]) == ("abcdefgh", 8, 0, True)


def test_cut_respects_the_token_cap():
    prompt, rec = C.cut_doc(CharTokenizer(), "abcdefghijklmnop", chunk=4, max_tokens=12)
    assert (prompt, rec["n"]) == ("abcdefghijkl", 12)


def test_unstable_boundary_moves_the_cut_until_the_reencoded_length_is_exact():
    # Cutting at 8 ends the text in "!", which re-encodes to 9 tokens; one token earlier re-encodes to 8.
    prompt, rec = C.cut_doc(CharTokenizer(), "abcdef!!gh", chunk=4, max_tokens=32)
    assert (prompt, rec["n"], rec["cut_at"], rec["move"], rec["ids_equal"]) == ("abcdef!", 8, 7, -1, False)


def test_corpus_lines_score_the_last_tail(tmp_path, monkeypatch):
    ppl = tmp_path / "ppl.jsonl"
    ppl.write_text(json.dumps({"id": "doc/0", "prompt": "x" * 13}) + "\n")
    monkeypatch.setattr(R, "load_tokenizer", lambda path: CharTokenizer())
    C.main(["--ppl", str(ppl), "--tokenizer", "unused", "--out", str(tmp_path / "c.jsonl"),
            "--chunk", "4", "--tail", "4", "--max-tokens", "32"])
    line = json.loads((tmp_path / "c.jsonl").read_text())
    manifest = json.loads((tmp_path / "c.jsonl.manifest.json").read_text())
    assert line == {"prompt": "x" * 12, "score_from": 8, "source": "doc/0"}
    assert manifest["bulk_chunks_total"] == 2


def test_refuses_a_doc_with_no_bulk_chunk(tmp_path, monkeypatch):
    ppl = tmp_path / "ppl.jsonl"
    ppl.write_text(json.dumps({"id": "short", "prompt": "x" * 7}) + "\n")
    monkeypatch.setattr(R, "load_tokenizer", lambda path: CharTokenizer())
    with pytest.raises(SystemExit, match="no bulk chunk"):
        C.main(["--ppl", str(ppl), "--tokenizer", "unused", "--out", str(tmp_path / "c.jsonl"),
                "--chunk", "4", "--tail", "4", "--max-tokens", "32"])
