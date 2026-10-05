"""tools/branch_corpus.py end-to-end against a fake /tokenize server running in a thread.

The fake tokenizer is character-level (one token per character) and the fake chat template is
a fixed Qwen-shaped render, so every length in the records can be re-derived independently in
the test and compared with what the tool recorded.  Run: python -m pytest tests/
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import branch_corpus as B  # noqa: E402


# ---------------------------------------------------------------------------
# The fake server: POST /tokenize with a char-level tokenizer and a fixed template.
# ---------------------------------------------------------------------------

def render(messages):
    """The fake chat template + generation prompt: what /tokenize and chat both serve."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + "<|im_start|>assistant\n"


def fake_ids(text):
    return [ord(c) for c in text]


class FakeTokenizeServer:
    """A thread hosting POST /tokenize (prompt form and messages form) on an ephemeral port."""

    def __init__(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/tokenize" or ("prompt" in body and "messages" in body):
                    self.send_error(400)
                    return
                if "messages" in body:
                    text = render(body["messages"])
                elif isinstance(body.get("prompt"), str):
                    text = body["prompt"]
                else:
                    self.send_error(400)
                    return
                toks = fake_ids(text)
                payload = {"count": len(toks), "max_model_len": 1000000, "tokens": toks,
                           "token_strs": None}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()
        self.thread.join()


def make_ppl(path, sizes, base="the quick brown fox jumps over the lazy dog. "):
    """A ppl.jsonl of `sizes` character counts; the text is filler, only lengths matter."""
    with open(path, "w", encoding="utf-8") as f:
        for i, size in enumerate(sizes):
            text = (base * (-(-size // len(base))))[:size]
            f.write(json.dumps({"id": f"doc/{i}", "prompt": text}) + "\n")
    return str(path)


@pytest.fixture()
def server():
    srv = FakeTokenizeServer()
    yield srv
    srv.close()


def build(srv, tmp_path, ppl, n, seed, interval=256, tail=256, min_prompt=1024, name="branches"):
    out = str(tmp_path / f"{name}.jsonl")
    B.main(["build", "--docs", ppl, "--n", str(n), "--seed", str(seed),
            "--interval", str(interval), "--tail", str(tail), "--min-prompt", str(min_prompt),
            "--server", srv.url, "--out", out])
    lines = [json.loads(line) for line in Path(out).read_text().splitlines()]
    return out, lines


# ---------------------------------------------------------------------------

def test_lengths_are_exact_and_the_hazard_inequality_holds_for_every_B(server, tmp_path):
    ppl = make_ppl(tmp_path / "ppl.jsonl", [3000, 5000])
    out, lines = build(server, tmp_path, ppl, n=3, seed=0)
    assert len(lines) == 3
    for t in lines:
        A, Bleg, C = t["A"], t["B"], t["C"]
        # exact lengths: the recorded n equals the fake server's own count for these messages
        assert t["n1"] == len(fake_ids(render(A["messages"]))) == A["n"]
        assert t["n2"] == len(fake_ids(render(Bleg["messages"]))) == Bleg["n"]
        assert t["n3"] == len(fake_ids(render(C["messages"]))) == C["n"]
        # the shared prefix is measured in id space and P is its largest interval multiple
        ids_A = fake_ids(render(A["messages"]))
        ids_B = fake_ids(render(Bleg["messages"]))
        ids_C = fake_ids(render(C["messages"]))
        shared = B.common_prefix_len(ids_A, ids_B)
        assert shared == Bleg["shared_prefix"]
        assert Bleg["p"] == (shared // t["interval"]) * t["interval"]
        assert C["shared_prefix"] == B.common_prefix_len(ids_A, ids_C)
        assert C["p"] == (C["shared_prefix"] // t["interval"]) * t["interval"]
        # the R65 precondition: N2 - T < P <= shared prefix, P a positive checkpoint multiple
        assert t["n2"] - t["tail"] < Bleg["p"] <= Bleg["shared_prefix"]
        assert Bleg["p"] > 0 and Bleg["p"] % t["interval"] == 0
        # B branches inside the document and is shorter than the producer
        assert t["n2"] < t["n1"]
        doc = A["messages"][1]["content"].split("\n\nQuestion: ")[0]
        assert 0 < Bleg["cut_chars"] < len(doc)
        assert Bleg["messages"][1]["content"].startswith(doc[:Bleg["cut_chars"]])
        assert not Bleg["messages"][1]["content"].startswith(doc[:Bleg["cut_chars"] + 1])
        # the prediction: max(0, P - (N2 - T)), and (P <= N1 - T kept by the construction)
        # it equals the DD-A device-counter oracle min(N1-N2, T-(N2-P))
        assert Bleg["predicted_tail_from_cache"] == \
            max(0, Bleg["p"] - (t["n2"] - t["tail"]))
        assert Bleg["hazard_bound"] == min(t["n1"] - t["n2"], t["tail"] - (t["n2"] - Bleg["p"]))
        assert Bleg["predicted_tail_from_cache"] == Bleg["hazard_bound"] > 0
        assert Bleg["p"] <= t["n1"] - t["tail"]
        # C: append-only, longer than A, and no tail position comes from cache
        assert t["n3"] > t["n1"]
        assert C["messages"][:len(A["messages"])] == A["messages"]
        assert C["predicted_tail_from_cache"] == \
            max(0, C["p"] - (t["n3"] - t["tail"])) == 0
        # and C's own computed prompt covers the whole tail (what keeps it unflagged)
        assert t["n3"] - C["p"] >= t["tail"]
        assert all(t["checks"].values())


def test_deterministic_per_seed_and_seed_changes_the_text(server, tmp_path):
    ppl = make_ppl(tmp_path / "ppl.jsonl", [4000, 4200])
    out0, lines0 = build(server, tmp_path, ppl, n=2, seed=0, name="b0")
    out0b, lines0b = build(server, tmp_path, ppl, n=2, seed=0, name="b0b")
    out1, lines1 = build(server, tmp_path, ppl, n=2, seed=1, name="b1")
    # same seed -> byte-identical corpus and manifest
    assert Path(out0).read_bytes() == Path(out0b).read_bytes()
    assert Path(out0 + ".manifest.json").read_bytes() == Path(out0b + ".manifest.json").read_bytes()
    # another seed -> the same geometry can hold, but the placeholder texts must move
    texts0 = {json.dumps(t["A"]["messages"]) + json.dumps(t["B"]["messages"]) for t in lines0}
    texts1 = {json.dumps(t["A"]["messages"]) + json.dumps(t["B"]["messages"]) for t in lines1}
    assert texts0 != texts1
    # the manifest carries the lengths for provenance
    manifest = json.loads(Path(out0 + ".manifest.json").read_text())
    assert manifest["seed"] == 0 and manifest["interval"] == 256 and manifest["tail"] == 256
    assert [m["id"] for m in manifest["triples"]] == [t["id"] for t in lines0]


def test_short_docs_are_skipped_and_none_left_is_a_failure(server, tmp_path, capsys):
    ppl = make_ppl(tmp_path / "ppl.jsonl", [3000, 100])
    out, lines = build(server, tmp_path, ppl, n=2, seed=0)     # doc/1 never reaches 1024 tokens
    assert len(lines) == 1
    assert lines[0]["id"] == "branch/000"
    assert "skipped 1 of 2" in capsys.readouterr().out

    ppl_tiny = make_ppl(tmp_path / "tiny.jsonl", [100, 150])
    with pytest.raises(SystemExit):
        build(server, tmp_path, ppl_tiny, n=2, seed=0, name="tiny")
    assert "no triple was built" in capsys.readouterr().err


def test_records_are_refused_when_fields_are_missing(server, tmp_path):
    (tmp_path / "bad.jsonl").write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        build(server, tmp_path, str(tmp_path / "bad.jsonl"), n=1, seed=0, name="bad")
    with pytest.raises(SystemExit):
        B.main(["build", "--docs", str(tmp_path / "missing.jsonl"), "--n", "1", "--seed", "0",
                "--interval", "256", "--tail", "256", "--server", server.url,
                "--out", str(tmp_path / "o.jsonl")])