"""tools/hazard_rate.py: the flag/overlap arithmetic on hand-made records, the plugin-log
contract regex, the cross-check, and one end-to-end pass where the records are hand-made from
a branch_corpus build against the fake /tokenize server.  Run: python -m pytest tests/
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import branch_corpus as B      # noqa: E402
import hazard_rate as H        # noqa: E402


# ---------------------------------------------------------------------------
# The flag / overlap arithmetic on hand-made records.
# ---------------------------------------------------------------------------

ROWS = [
    # a branch leg: tail start 6100-2048=4052 < cache 4096 -> flagged, 44 overlapped positions
    {"id": "branch/000/B", "prompt_len": 6100, "cache_n": 4096},
    # an append-only turn: tail start 19210-2048=17162 >= cache 16384 -> not flagged
    {"id": "branch/000/C", "prompt_len": 19210, "cache_n": 16384},
    # the boundary: cache_n == prompt_len - T is NOT a flag (the tail starts exactly at P)
    {"id": "edge/equal", "prompt_len": 4096, "cache_n": 2048},
    # one position past the boundary
    {"id": "edge/one", "prompt_len": 4095, "cache_n": 2048},
    # a fresh request: nothing cached, nothing flagged (also guards a prompt shorter than T,
    # where the literal N - T is negative)
    {"id": "fresh", "prompt_len": 900, "cache_n": 0},
    # a short request fully served from cache: its whole prompt is its own exact tail
    {"id": "tiny-hit", "prompt_len": 900, "cache_n": 100},
]


def test_flag_and_overlap_arithmetic():
    flagged = {r["id"]: H.flag_and_overlap(r["prompt_len"], r["cache_n"], 2048) for r in ROWS}
    assert flagged["branch/000/B"] == (True, 44)
    assert flagged["branch/000/C"] == (False, 0)
    assert flagged["edge/equal"] == (False, 0)
    assert flagged["edge/one"] == (True, 1)
    assert flagged["fresh"] == (False, 0)
    assert flagged["tiny-hit"] == (True, 100)


def test_summarize_reports_rate_and_overlaps():
    s = H.summarize([dict(r, prompt_n=None) for r in ROWS], None, 2048)
    assert s["n_requests"] == 6 and s["n_flagged"] == 3
    assert s["rate"] == pytest.approx(0.5)
    assert s["overlap_total"] == 44 + 1 + 100
    assert "cross_check" not in s


def test_records_loader_accepts_full_response_objects(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps({"id": "full/response", "usage": {"prompt_tokens": 6100},
                    "timings": {"cache_n": 4096, "prompt_n": 2004, "prompt_ms": 900.0}}) + "\n"
        + json.dumps({"prompt_tokens": 19210, "timings": {"cache_n": 16384, "prompt_n": 2826}}) + "\n",
        encoding="utf-8")
    rows = H.load_records([str(path)])
    assert rows[0]["id"] == "full/response" and rows[0]["prompt_len"] == 6100 \
        and rows[0]["cache_n"] == 4096
    # the second line has no id: it is named by file and line, and prompt_len falls back to
    # cache_n + prompt_n
    assert rows[1]["prompt_len"] == 19210 and rows[1]["cache_n"] == 16384
    assert "responses.jsonl:2" in rows[1]["id"]


def test_missing_fields_are_refused(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "cache_n": 10}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        H.load_records([str(path)])
    assert "no prompt length" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The plugin-log contract.
# ---------------------------------------------------------------------------

def test_the_hazard_log_regex_matches_the_two_spellings_and_rejects_the_rest():
    m = H.HAZARD_LOG_RE.search("radiance: kva: hazard 1395 positions (mode=quality)")
    assert m and m.group("positions") == "1395" and m.group("request") is None
    m = H.HAZARD_LOG_RE.search("2026-10-05T12:00:00 kva: hazard request branch/007/B "
                               "512 positions n_checkpoints=3")
    assert m and m.group("request") == "branch/007/B" and m.group("positions") == "512"
    assert H.HAZARD_LOG_RE.search("kva: hazard request branch/007/B 0 positions")
    for line in ("kva: hazard positions",          # no count
                 "kva: hazards 44 positions",       # not the marker word
                 "kvaa: hazard 44 positions",       # not the marker prefix
                 "kva: hazard 44 rows"):            # not the marker unit
        assert H.HAZARD_LOG_RE.search(line) is None, line


def test_parse_plugin_log_counts_and_reports_unparsed(tmp_path):
    path = tmp_path / "server.log"
    path.write_text(
        "radiance: serving on 8100\n"
        "radiance: kva: hazard request branch/000/B 44 positions\n"
        "radiance: kva: hazard 1 positions\n"
        "some unrelated kva noise\n"
        "radiance: kva: hazard request branch/000/C 0 positions\n", encoding="utf-8")
    p = H.parse_plugin_log(str(path))
    assert p["lines"] == 3 and p["positions_total"] == 45
    assert p["by_request"] == {"branch/000/B": 44, "branch/000/C": 0}
    assert p["unnamed"] == [1]
    assert len(p["unparsed"]) == 1 and "unrelated kva noise" in p["unparsed"][0]


def test_cross_check_agrees_and_require_match_passes(tmp_path, capsys):
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    plugin = tmp_path / "plugin.log"
    plugin.write_text(
        "kva: hazard request branch/000/B 44 positions\n"
        "kva: hazard request edge/one 1 positions\n"
        "kva: hazard request tiny-hit 100 positions\n"
        "kva: hazard request branch/000/C 0 positions\n", encoding="utf-8")
    assert H.main(["--records", str(records), "--plugin-log", str(plugin), "--tail", "2048",
                   "--require-match"]) == 0
    out = capsys.readouterr().out
    assert "flagged 3 of 6" in out and "AGREE" in out


def test_cross_check_disagreement_fails_require_match(tmp_path, capsys):
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    plugin = tmp_path / "plugin.log"
    # the device counter says 50 for a request the records side counts as 44
    plugin.write_text("kva: hazard request branch/000/B 50 positions\n", encoding="utf-8")
    assert H.main(["--records", str(records), "--plugin-log", str(plugin), "--tail", "2048",
                   "--require-match"]) == 1
    out = capsys.readouterr().out
    assert "DISAGREE" in out and "DIFFER branch/000/B" in out


def test_main_writes_the_machine_readable_summary(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    out = tmp_path / "report.json"
    assert H.main(["--records", str(records), "--tail", "2048", "--out", str(out)]) == 0
    s = json.loads(out.read_text())
    assert s["n_flagged"] == 3 and s["overlap_total"] == 145
    assert [r["id"] for r in s["per_request"] if r["flagged"]] == \
        ["branch/000/B", "edge/one", "tiny-hit"]


# ---------------------------------------------------------------------------
# End to end with the fake /tokenize server: what branch_corpus predicts is exactly what
# hazard_rate flags, and the plugin log built from the predictions cross-checks clean.
# ---------------------------------------------------------------------------

def render(messages):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) \
        + "<|im_start|>assistant\n"


class FakeTokenizeServer:
    def __init__(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/tokenize":
                    self.send_error(404)
                    return
                text = render(body["messages"]) if "messages" in body else body["prompt"]
                toks = [ord(c) for c in text]
                data = json.dumps({"count": len(toks), "max_model_len": 1000000,
                                   "tokens": toks, "token_strs": None}).encode()
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


def test_the_corpus_predictions_are_exactly_what_hazard_rate_flags(tmp_path, capsys):
    srv = FakeTokenizeServer()
    try:
        ppl = tmp_path / "ppl.jsonl"
        base = "the quick brown fox jumps over the lazy dog. "
        text = (base * 120)[:3000]
        ppl.write_text(json.dumps({"id": "doc/0", "prompt": text}) + "\n", encoding="utf-8")
        out = tmp_path / "branches.jsonl"
        B.main(["build", "--docs", str(ppl), "--n", "2", "--seed", "0",
                "--interval", "256", "--tail", "256", "--min-prompt", "1024",
                "--server", srv.url, "--out", str(out)])
        triples = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(triples) == 2
    finally:
        srv.close()

    # hand-made response records: what a driver would note down serving A -> B -> C per triple
    # (A cold, B resuming at P, C resuming at P3), plus the plugin lines the device would print
    records, plugin_lines = [], []
    for t in triples:
        records += [
            {"id": t["id"] + "/A", "prompt_len": t["n1"], "cache_n": 0},
            {"id": t["id"] + "/B", "prompt_len": t["n2"], "cache_n": t["B"]["p"]},
            {"id": t["id"] + "/C", "prompt_len": t["n3"], "cache_n": t["C"]["p"]},
        ]
        plugin_lines.append(f"radiance: kva: hazard request {t['id']}/B "
                            f"{t['B']['predicted_tail_from_cache']} positions")
        plugin_lines.append(f"radiance: kva: hazard request {t['id']}/C 0 positions")
    records_path = tmp_path / "responses.jsonl"
    records_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    plugin_path = tmp_path / "server.log"
    plugin_path.write_text("\n".join(plugin_lines) + "\n", encoding="utf-8")

    assert H.main(["--records", str(records_path), "--plugin-log", str(plugin_path),
                   "--tail", "256", "--require-match"]) == 0
    out = capsys.readouterr().out
    s = H.summarize(H.load_records([str(records_path)]),
                    H.parse_plugin_log(str(plugin_path)), 256)
    # the B legs are flagged, with exactly the predicted overlap; A and C are not flagged
    flagged = {r["id"]: r for r in s["per_request"] if r["flagged"]}
    assert set(flagged) == {t["id"] + "/B" for t in triples}
    for t in triples:
        assert flagged[t["id"] + "/B"]["overlap"] == t["B"]["predicted_tail_from_cache"]
    assert s["rate"] == pytest.approx(2 / 6)
    assert "AGREE" in out