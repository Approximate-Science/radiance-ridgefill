"""Unit tests for tools/paired.py on synthetic reports, .rows files and corpora.

Synthetic inputs only: the reports follow the engine's --kld-out shape (radiance
core/kld.cpp:562-579), the .rows files are [positions, 4] float32 in reference row order
(KL, candidate NLL, reference NLL, top-1 agreement; core/kld.cpp:586-590).
"""

import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRED = os.path.join(REPO, "tools", "paired.py")


def load_paired():
    spec = importlib.util.spec_from_file_location("paired", PAIRED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def make_report(by_source_ppl, kld_mean=0.001, kld_p99=0.004, top1=0.99,
               with_by_source=True, all_ppl=(3.0, 2.99)):
    rep = {
        "model": "candidate.rad",
        "reference": "ref.rad",
        "docs": len(by_source_ppl),
        "all": {
            "positions": 9,
            "kld": {"mean": kld_mean, "median": kld_mean / 2, "p99": kld_p99},
            "top1_agreement": top1,
            "ppl": {"candidate": all_ppl[0], "reference": all_ppl[1], "ratio": 1.0},
        },
        "by_source": {},
    }
    if with_by_source:
        for source, ppl in by_source_ppl.items():
            rep["by_source"][source] = {
                "positions": 3,
                "docs": 1,
                "kld": {"mean": kld_mean},
                "top1_agreement": top1,
                "ppl": {"candidate": ppl, "reference": 2.99},
            }
    return rep


# ---------------------------------------------------------------- the report path

def test_per_doc_nll_from_report_is_log_of_ppl(tmp_path):
    paired = load_paired()
    rep = make_report({"doc0": 3.0, "doc1": 3.09})
    path = write_json(tmp_path / "a.json", rep)
    nll = paired.per_doc_nll_from_report(rep, path)
    assert nll == {"doc0": math.log(3.0), "doc1": math.log(3.09)}


def test_report_path_main_prints_per_doc_and_ci(tmp_path, capsys):
    paired = load_paired()
    a = write_json(tmp_path / "a.json", make_report({"doc0": 3.0, "doc1": 3.0, "doc2": 3.0}))
    b = write_json(tmp_path / "b.json", make_report({"doc0": 3.03, "doc1": 3.03, "doc2": 3.03},
                                                     kld_mean=0.002, kld_p99=0.006, top1=0.98))
    paired.main([str(a), str(b)])
    out = capsys.readouterr().out
    expected_diff = math.log(3.03) - math.log(3.0)
    assert f"{expected_diff:.6f}" in out
    assert "mean difference B - A:" in out
    assert "95% CI [" in out
    assert "doc0" in out
    # the summary lines of each report
    assert "mean KL 0.001000" in out and "mean KL 0.002000" in out
    assert "99th KL 0.004000" in out and "99th KL 0.006000" in out
    assert "top-1 99.000%" in out and "top-1 98.000%" in out


def test_source_mismatch_is_an_error(tmp_path, capsys):
    paired = load_paired()
    a = write_json(tmp_path / "a.json", make_report({"doc0": 3.0, "doc1": 3.0}))
    b = write_json(tmp_path / "b.json", make_report({"doc0": 3.0, "doc2": 3.0}))
    with pytest.raises(SystemExit):
        paired.main([str(a), str(b)])
    err = capsys.readouterr().err
    assert "different sources" in err
    assert "doc1" in err and "doc2" in err


# ---------------------------------------------------------------- the .rows path

def write_rows_and_corpus(tmp_path, report_name, doc_nlls, corpus_rows=None):
    """Write <report_name> (no by_source), <report_name>.rows and a corpus with
    matching doc order: doc i has corpus_rows[i] scored positions (default 3)."""
    if corpus_rows is None:
        corpus_rows = [3] * len(doc_nlls)
    rows = np.zeros((sum(corpus_rows), 4), dtype=np.float32)
    offset = 0
    corpus_lines = []
    for i, (nll, n_rows) in enumerate(zip(doc_nlls, corpus_rows)):
        rows[offset:offset + n_rows, 0] = 0.001     # KL
        rows[offset:offset + n_rows, 1] = nll       # candidate NLL
        rows[offset:offset + n_rows, 2] = nll - 0.01  # reference NLL
        rows[offset:offset + n_rows, 3] = 1         # argmax agreed
        offset += n_rows
        tokens = n_rows + 1 + 2048                  # score_from = N - T, T = 2048
        corpus_lines.append({"prompt": f"doc {i}", "tokens": tokens,
                             "score_from": tokens - n_rows - 1, "source": f"doc{i}"})
    report = write_json(tmp_path / report_name, make_report({}, with_by_source=False))
    rows.tofile(tmp_path / (report_name + ".rows"))
    corpus = tmp_path / "quick9.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(json.dumps(line) + "\n")
    return str(report), str(corpus)


def test_rows_fallback_matches_report_path(tmp_path, capsys):
    paired = load_paired()
    # the same per-doc NLLs through both paths: by_source ppl = exp(mean nll)
    doc_nlls = [1.5, 2.5, 3.0]
    a_rows, corpus = write_rows_and_corpus(tmp_path, "a.json", doc_nlls, corpus_rows=[3, 3, 3])
    nll_from_rows = paired.per_doc_nll_from_rows(a_rows, corpus)
    a_rep = make_report({f"doc{i}": math.exp(n) for i, n in enumerate(doc_nlls)})
    nll_from_report = paired.per_doc_nll_from_report(a_rep, "a.json")
    assert nll_from_rows == pytest.approx(nll_from_report, rel=1e-6)


def test_rows_fallback_used_when_report_has_no_by_source(tmp_path, capsys):
    paired = load_paired()
    a, corpus = write_rows_and_corpus(tmp_path, "a.json", [1.0, 2.0], corpus_rows=[4, 2])
    b, _ = write_rows_and_corpus(tmp_path, "b.json", [1.1, 2.2], corpus_rows=[4, 2])
    paired.main([a, b, "--corpus", corpus])
    out = capsys.readouterr().out
    # per-doc means: doc0 = 1.1 - 1.0 = 0.1, doc1 = 2.2 - 2.0 = 0.2, mean = 0.15
    assert f"{1.0:.6f}" in out and f"{1.1:.6f}" in out and f"{2.2:.6f}" in out
    assert "mean difference B - A: 0.150000" in out
    assert "95% CI [" in out


def test_rows_size_mismatch_is_an_error(tmp_path, capsys):
    paired = load_paired()
    a, corpus = write_rows_and_corpus(tmp_path, "a.json", [1.0, 2.0], corpus_rows=[4, 2])
    # drop two POSITIONS (8 floats) from the rows file
    rows = np.fromfile(a + ".rows", dtype=np.float32)
    rows[:-8].tofile(a + ".rows")
    with pytest.raises(SystemExit):
        paired.per_doc_nll_from_rows(a, corpus)
    err = capsys.readouterr().err
    assert "positions" in err and "6" in err and "4" in err


def test_corpus_without_tokens_is_an_error(tmp_path, capsys):
    paired = load_paired()
    corpus = tmp_path / "c.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "x", "score_from": 5}) + "\n")
    with pytest.raises(SystemExit):
        paired.docs_from_corpus(str(corpus))
    err = capsys.readouterr().err
    assert '"tokens"' in err and "c.jsonl:1" in err


def test_missing_rows_without_corpus_is_an_error(tmp_path, capsys):
    paired = load_paired()
    path = write_json(tmp_path / "a.json", make_report({}, with_by_source=False))
    with pytest.raises(SystemExit):
        paired.per_doc_nll(load_paired().load_report(path), str(path), None)
    err = capsys.readouterr().err
    assert "--corpus" in err


# ---------------------------------------------------------------- the bootstrap

def test_bootstrap_is_reproducible():
    paired = load_paired()
    diffs = [0.1, -0.2, 0.3, 0.05, -0.1]
    assert paired.bootstrap_ci(diffs) == paired.bootstrap_ci(diffs)


def test_bootstrap_ci_brackets_the_mean():
    paired = load_paired()
    rng = np.random.default_rng(1)
    diffs = list(rng.normal(0.01, 0.05, 50))
    lo, hi = paired.bootstrap_ci(diffs)
    mean = float(np.mean(diffs))
    assert lo <= mean <= hi


# ---------------------------------------------------------------- the CLI

def test_cli_end_to_end(tmp_path):
    a = write_json(tmp_path / "a.json", make_report({"doc0": 3.0, "doc1": 4.0}))
    b = write_json(tmp_path / "b.json", make_report({"doc0": 3.03, "doc1": 4.08}))
    result = subprocess.run([sys.executable, PAIRED, str(a), str(b)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "mean difference B - A:" in result.stdout
    assert "doc0" in result.stdout and "doc1" in result.stdout
    assert "top-1" in result.stdout


def test_cli_refuses_nonexistent_report(tmp_path):
    result = subprocess.run([sys.executable, PAIRED,
                             str(tmp_path / "missing.json"), str(tmp_path / "b.json")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "missing.json" in result.stderr