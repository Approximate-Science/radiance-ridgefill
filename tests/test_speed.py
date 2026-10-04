"""Unit tests for the pure parts of tools/speed.py (no server needed): the docs
loader, the prompt builder and the exact-tail note."""

import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEED = os.path.join(REPO, "tools", "speed.py")


def load_speed():
    spec = importlib.util.spec_from_file_location("speed", SPEED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_docs(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return str(path)


def test_load_docs_reads_the_prompt_field(tmp_path):
    speed = load_speed()
    path = write_docs(tmp_path / "ppl.jsonl", [
        {"id": "ppl/8k/0", "bucket": "8k", "prompt": "first doc"},
        {"id": "ppl/16k/1", "bucket": "16k", "prompt": "second doc"},
    ])
    docs = speed.load_docs(path)
    assert [d["prompt"] for d in docs] == ["first doc", "second doc"]


def test_load_docs_refuses_a_line_without_prompt(tmp_path, capsys):
    speed = load_speed()
    path = write_docs(tmp_path / "ppl.jsonl", [{"id": "x", "bucket": "8k"}])
    with pytest.raises(SystemExit):
        speed.load_docs(path)
    assert '"prompt"' in capsys.readouterr().err


def test_load_docs_refuses_an_empty_file(tmp_path, capsys):
    speed = load_speed()
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(SystemExit):
        speed.load_docs(str(path))
    assert "no documents" in capsys.readouterr().err


def test_build_prompt_is_exactly_the_target_length():
    speed = load_speed()
    doc_ids = [list(range(1, 6000)), list(range(20000, 25000))]
    nonce_ids = [99, 98, 97]
    ids = speed.build_prompt(9216, doc_ids, nonce_ids)
    assert len(ids) == 9216
    assert ids[:3] == [99, 98, 97]                 # the leading nonce, uncut
    assert ids[3] == 1                            # then the first doc's ids
    assert set(ids[3:]) <= set(range(1, 6000)) | set(range(20000, 25000))


def test_build_prompt_concatenates_then_cuts():
    speed = load_speed()
    doc_ids = [[1, 2, 3], [4, 5]]
    ids = speed.build_prompt(6, doc_ids, [9])
    assert ids == [9, 1, 2, 3, 4, 5]              # nonce, then doc0 + doc1, no room to cut
    ids = speed.build_prompt(5, doc_ids, [9])
    assert ids == [9, 1, 2, 3, 4]                 # cut mid-doc


def test_build_prompt_refuses_too_few_tokens(capsys):
    speed = load_speed()
    with pytest.raises(SystemExit):
        speed.build_prompt(100, [[1, 2, 3]], [9])
    err = capsys.readouterr().err
    assert "3 tokens" in err and "99 needed" in err


def test_tail_note_at_the_three_protocol_lengths():
    speed = load_speed()
    tail, note = speed.tail_note(9216)
    assert tail == 3072
    assert "3072" in note and "n_ahead = 1024" in note and "chunk at 6144" in note
    tail, note = speed.tail_note(16384)
    assert tail == 2048
    assert "exactly T = 2048" in note
    tail, note = speed.tail_note(32768)
    assert tail == 2048