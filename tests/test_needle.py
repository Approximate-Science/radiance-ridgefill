# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""Unit + end-to-end tests for tools/needle.py against a fake radiance server.

The engine cannot run in this sandbox, so the HTTP shapes come from a fake server
in a thread (tests/test_speed.py and notes/scripts.md document the real shapes):
POST /tokenize {"prompt", "add_special_tokens": false} -> {"count", "tokens"}, and
POST /v1/completions {"prompt": [ids]} -> choices[0].text, usage.prompt_tokens,
timings.prompt_ms (radiance core/server/admin.cpp:119-196, core/server/oai.cpp:1549-1568).

The fake tokenizer is word-level (whitespace split, one id per word), so token counts
are exact and predictable.  The fake completions endpoint simulates the two engines
under test: mode "stock" always finds the needle anywhere in the prompt; mode "ridgefill"
finds it only when its number token sits in the last 2048 prompt tokens (the plugin's
exact tail, README.md RADIANCE_RIDGEFILL_TAIL), which is exactly the failure the harness
must catch.
"""

import importlib.util
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDLE = os.path.join(REPO, "tools", "needle.py")


def load_needle():
    spec = importlib.util.spec_from_file_location("needle", NEEDLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- the fake server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        srv = self.server
        if self.path == "/tokenize" and "messages" in body:
            # the chat form: a word-level stand-in for the model's template, one user turn + the generation prompt
            assert body.get("chat_template_kwargs") == {"enable_thinking": False}, "the needle asks with thinking off"
            words = (["<|im_start|>user"] + body["messages"][0]["content"].split()
                     + ["<|im_end|>", "<|im_start|>assistant", "<think></think>"])
            ids = [srv.word_id(w) for w in words]
            self._json({"count": len(ids), "tokens": ids, "max_model_len": 49152})
        elif self.path == "/tokenize":
            srv.n_tokenize += 1
            words = body["prompt"].split()
            ids = [srv.word_id(w) for w in words]
            assert not body.get("add_special_tokens"), \
                "needle.py must pass add_special_tokens=false (exact counts, no BOS)"
            self._json({"count": len(ids), "tokens": ids, "max_model_len": 49152})
        elif self.path == "/v1/completions":
            srv.n_completions += 1
            ids = body["prompt"]
            assert (isinstance(ids, list) and ids
                    and all(isinstance(i, int) for i in ids)), \
                "the prompt must be a non-empty list of token ids"
            assert body.get("temperature") == 0, "retrieval must run at temperature 0"
            words = [srv.rev.get(i) for i in ids]
            # the asked key: the question's "for <key>?" in the last few tokens
            key = None
            for i in range(len(words) - 1, max(len(words) - 40, 0), -1):
                if words[i - 1] == "for" and words[i].endswith("?"):
                    key = words[i][:-1]
                    break
            # the needle "... for <key> is <number>." somewhere in the prompt
            number, pos = None, None
            if key:
                for j in range(len(words) - 2):
                    if (words[j] == key and words[j + 1] == "is"
                            and words[j + 2].endswith(".")):
                        number, pos = words[j + 2][:-1], j + 2
                        break
            # stock sees the whole prompt; ridgefill only its last 2048 tokens
            visible = number is not None and (
                srv.mode == "stock" or pos >= len(ids) - srv.tail_tokens)
            text = " " + number if visible else " I'm sorry, I don't know."
            self._json({"choices": [{"text": text}],
                        "usage": {"prompt_tokens": len(ids)},
                        "timings": {"prompt_ms": 12.5, "prompt_n": len(ids)}})
        else:
            self.send_error(404)


class FakeRadiance(ThreadingHTTPServer):
    """Word-level tokenizer + a completions endpoint with a configurable tail."""

    def __init__(self, mode="stock", tail_tokens=2048):
        super().__init__(("127.0.0.1", 0), Handler)
        self.mode = mode
        self.tail_tokens = tail_tokens
        self.lock = threading.Lock()
        self.vocab = {}          # word -> id, assigned in first-seen order
        self.rev = {}            # id -> word
        self.n_tokenize = 0
        self.n_completions = 0

    def word_id(self, word):
        with self.lock:
            if word not in self.vocab:
                self.vocab[word] = 1000 + len(self.vocab)
                self.rev[self.vocab[word]] = word
            return self.vocab[word]

    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture
def stock_server():
    srv = FakeRadiance(mode="stock")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def ridgefill_server():
    srv = FakeRadiance(mode="ridgefill")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


# ---------------------------------------------------------------- helpers

FILLER_DOC = " ".join(["filler", "text", "about", "raccoons", "and", "oysters"] * 80)


def write_docs(path):
    path.write_text(json.dumps({"id": "d0", "prompt": FILLER_DOC}) + "\n"
                    + json.dumps({"id": "d1", "prompt": FILLER_DOC}) + "\n")
    return str(path)


def build(needle, server, docs, out, lengths="512,1024", depths="0.05,0.5,0.98",
          keys=4, seed=0, chat=False):
    needle.cmd_build(SimpleNamespace(
        docs=docs, lengths=lengths, depths=depths, keys=keys, seed=seed,
        server=server.url(), out=str(out), timeout=30.0, chat=chat))
    return [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]


def run(needle, server, corpus, out, max_tokens=16):
    needle.cmd_run(SimpleNamespace(
        corpus=str(corpus), server=server.url(), out=str(out),
        max_tokens=max_tokens, timeout=30.0))
    return [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]


def word_len(text):
    """What the fake tokenizer counts for a piece of text."""
    return len(text.split())


# ---------------------------------------------------------------- pure pieces

def test_first_seven_digits():
    needle = load_needle()
    assert needle.first_seven_digits("The answer is 6254109.") == "6254109"
    assert needle.first_seven_digits("  1234567\n") == "1234567"
    assert needle.first_seven_digits("6254109 then 7348291") == "6254109"   # first one
    assert needle.first_seven_digits("short 123456") is None               # 6 digits
    assert needle.first_seven_digits("long 12345678") is None               # 8 digits
    assert needle.first_seven_digits("no digits at all") is None
    assert needle.first_seven_digits("code 0042401?") == "0042401"         # leading zeros


def test_mcnemar_exact_p_on_known_tables():
    needle = load_needle()
    # b=1, c=5: p = 2 * (C(6,0) + C(6,1)) / 2^6 = 2 * 7/64 = 0.21875
    assert needle.mcnemar_exact_p(1, 5) == pytest.approx(0.21875)
    # symmetric
    assert needle.mcnemar_exact_p(5, 1) == pytest.approx(0.21875)
    # b=8, c=2: p = 2 * (1 + 10 + 45)/1024 = 0.109375
    assert needle.mcnemar_exact_p(8, 2) == pytest.approx(0.109375)
    # no disagreements at all
    assert needle.mcnemar_exact_p(0, 0) == 1.0
    # heavy overlap in the middle: 2 * 42/64 > 1, capped at 1
    assert needle.mcnemar_exact_p(3, 3) == 1.0
    # the smallest possible signal: b + c = 1 can never reach significance
    assert needle.mcnemar_exact_p(1, 0) == pytest.approx(2.0 * 0.5)
    # cross-check one table against the direct binomial sum
    b, c = 4, 9
    n = b + c
    direct = 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    assert needle.mcnemar_exact_p(b, c) == pytest.approx(min(1.0, direct))


def test_needle_position_and_region_boundary():
    needle = load_needle()
    # depth is relative to the whole prompt: 0.9 of 16384 tokens -> token 14745
    assert needle.needle_position(16384, 0.9, 10 ** 9) == 14745
    assert needle.needle_position(32768, 0.98, 10 ** 9) == 32112
    # clamped into the filler region (the filler ends before the needle + question)
    assert needle.needle_position(512, 0.98, 491) == 491
    assert needle.needle_position(512, 0.05, 491) == 26
    # the exact-tail boundary at 16384 is token 14336 (the last 2048 are exact)
    assert needle.region_at(16384 - 2048, 16384) == "tail"
    assert needle.region_at(16384 - 2049, 16384) == "bulk"
    # 0.9 at 16384 lands inside the tail, 0.9 at 32768 lands in the bulk:
    # the depths straddle the boundary as the lengths grow
    assert needle.region_at(needle.needle_position(16384, 0.9, 10 ** 9), 16384) == "tail"
    assert needle.region_at(needle.needle_position(32768, 0.9, 10 ** 9), 32768) == "bulk"
    # a prompt shorter than the tail is exact everywhere
    assert needle.region_at(0, 512) == "tail"


def test_parse_lists_reject_bad_input(capsys):
    needle = load_needle()
    for bad in ("8192,abc", "", "0"):
        with pytest.raises(SystemExit):
            needle.parse_int_list(bad, "lengths")
    assert needle.parse_int_list("8192, 16384 32768", "lengths") == [8192, 16384, 32768]
    with pytest.raises(SystemExit):
        needle.parse_float_list("1.5", "depths")
    assert needle.parse_float_list("0.05,0.9", "depths") == [0.05, 0.9]


# ---------------------------------------------------------------- build

def test_build_exact_lengths_and_positions(stock_server, tmp_path):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    items = build(needle, stock_server, docs, tmp_path / "corpus.jsonl",
                  lengths="512,1024", depths="0.05,0.5,0.98")
    assert len(items) == 2 * 3 * 2            # lengths x depths x {single, multi}
    assert len({it["id"] for it in items}) == len(items)   # ids unique

    for item in items:
        # EXACT token count, by construction
        assert item["prompt_tokens"] == item["length"]
        assert len(item["prompt_ids"]) == item["length"]

        n_keys = 1 if item["variant"] == "single" else 4
        assert len(item["needles"]) == n_keys
        assert len(set(item["keys"])) == n_keys              # distinct keys
        assert all(len(nd["number"]) == 7 for nd in item["needles"])
        assert item["expected"] == item["needles"][item["asked_index"]]["number"]
        assert len({nd["number"] for nd in item["needles"]}) == n_keys   # distinct

        # with the word-level fake: 8 words per needle, 13 per question
        needle_words = 8 * n_keys
        question_words = 13
        filler_total = item["length"] - needle_words - question_words
        depth = item["depth"]
        block_start = needle.needle_position(item["length"], depth, filler_total)
        assert item["needles"][0]["position"] == block_start
        assert item["needle_pos"] == item["needles"][item["asked_index"]]["position"]

        # the ids really are the needle block at that position, then the question
        ids = item["prompt_ids"]
        assert ids[block_start] == stock_server.word_id("The")
        assert ids[-question_words] == stock_server.word_id("What")
        assert ids[block_start + needle_words - 1] in {
            stock_server.word_id(nd["number"] + ".") for nd in item["needles"]}

        # positions of the needles inside the block are consecutive
        starts = [nd["position"] for nd in item["needles"]]
        assert starts == [block_start + 8 * i for i in range(n_keys)]

        # the region label agrees with the position, not with the nominal depth
        for nd in item["needles"]:
            assert nd["region"] == needle.region_at(nd["position"], item["length"])
            assert nd["in_tail"] == (nd["region"] == "tail")
        assert item["region"] == item["needles"][item["asked_index"]]["region"]

        # the question asks for exactly the asked key
        assert item["key"] in item["question"]
        assert item["question"].endswith("Answer with the number only.")


def test_build_short_prompt_is_all_tail(stock_server, tmp_path):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    items = build(needle, stock_server, docs, tmp_path / "c.jsonl",
                  lengths="512", depths="0.05,0.98")
    # a 512-token prompt is entirely inside the 2048-token exact tail
    assert all(it["region"] == "tail" and it["in_tail"] for it in items)


def test_build_region_flips_across_the_tail_boundary(stock_server, tmp_path):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    items = {it["id"]: it for it in build(
        needle, stock_server, docs, tmp_path / "c.jsonl",
        lengths="4096", depths="0.4,0.9")}
    assert items["L4096/D0.40/single"]["region"] == "bulk"
    assert items["L4096/D0.90/single"]["region"] == "tail"


def test_build_is_deterministic_per_seed(stock_server, tmp_path):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    build(needle, stock_server, docs, tmp_path / "a.jsonl", seed=0)
    build(needle, stock_server, docs, tmp_path / "b.jsonl", seed=0)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_build_seed_changes_keys_and_numbers(stock_server, tmp_path):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    zero = build(needle, stock_server, docs, tmp_path / "s0.jsonl",
                 lengths="512", depths="0.5", seed=0)
    one = build(needle, stock_server, docs, tmp_path / "s1.jsonl",
                lengths="512", depths="0.5", seed=1)
    assert all(a["length"] == b["length"] for a, b in zip(zero, one))
    assert any(a["expected"] != b["expected"] for a, b in zip(zero, one))


def test_build_refuses_overwrite_and_bad_args(stock_server, tmp_path, capsys):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    out = tmp_path / "c.jsonl"
    build(needle, stock_server, docs, out, lengths="512", depths="0.5")
    with pytest.raises(SystemExit):
        build(needle, stock_server, docs, out, lengths="512", depths="0.5")
    assert "refusing to overwrite" in capsys.readouterr().err
    with pytest.raises(SystemExit):          # room for filler only
        build(needle, stock_server, docs, tmp_path / "tiny.jsonl",
              lengths="16", depths="0.5")


# ---------------------------------------------------------------- run

def test_run_against_stock_and_parsing(stock_server, tmp_path):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    items = build(needle, stock_server, docs, tmp_path / "corpus.jsonl",
                  lengths="512,1024", depths="0.5", keys=3)
    recs = run(needle, stock_server, tmp_path / "corpus.jsonl",
               tmp_path / "res.jsonl", max_tokens=16)
    assert len(recs) == len(items) == 4
    assert stock_server.n_completions == 4
    for rec, item in zip(recs, items):
        assert rec["id"] == item["id"]
        assert rec["correct"] is True
        assert rec["parsed"] == item["expected"]
        assert rec["prompt_ms"] == 12.5
        assert rec["prompt_tokens"] == item["length"]
        # the metadata compare needs travels with every record
        for key in ("length", "depth", "region", "variant", "key"):
            assert rec[key] == item[key]


def test_run_resume_skips_done_ids(stock_server, tmp_path, capsys):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    items = build(needle, stock_server, docs, tmp_path / "corpus.jsonl",
                  lengths="512", depths="0.25,0.75")
    out = tmp_path / "res.jsonl"

    # a partial results file: the first item is already there
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": items[0]["id"], "correct": True}) + "\n")
    run(needle, stock_server, tmp_path / "corpus.jsonl", out)
    lines = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    assert len(lines) == len(items)
    assert stock_server.n_completions == len(items) - 1    # the done id was skipped
    assert sum(1 for r in lines if r["correct"]) == len(items)

    # a second full run changes nothing and sends nothing
    before = stock_server.n_completions
    run(needle, stock_server, tmp_path / "corpus.jsonl", out)
    assert stock_server.n_completions == before
    after = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    assert len(after) == len(items)                       # no duplicate lines
    assert "nothing to do" in capsys.readouterr().out


def test_run_records_wrong_answers(ridgefill_server, tmp_path):
    """The ridgefill-mode fake only retrieves needles inside the last 2048 tokens."""
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    items = build(needle, ridgefill_server, docs, tmp_path / "corpus.jsonl",
                  lengths="3072", depths="0.1,0.9")
    recs = run(needle, ridgefill_server, tmp_path / "corpus.jsonl", tmp_path / "res.jsonl")
    by_id = {r["id"]: r for r in recs}
    for item in items:
        rec = by_id[item["id"]]
        assert rec["correct"] == (item["region"] == "tail")
        if item["region"] == "bulk":
            assert rec["parsed"] is None
            assert "don't know" in rec["reply"]
        else:
            assert rec["parsed"] == item["expected"]


# ---------------------------------------------------------------- compare

def test_compare_end_to_end(stock_server, ridgefill_server, tmp_path, capsys):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    corpus = tmp_path / "corpus.jsonl"
    build(needle, stock_server, docs, corpus, lengths="3072", depths="0.1,0.9")
    # the same corpus built against the ridgefill server: identical ids (same tokenize
    # order), so each server's word-level vocabulary covers them
    items = build(needle, ridgefill_server, docs, tmp_path / "corpus_ridgefill.jsonl",
                  lengths="3072", depths="0.1,0.9")
    with open(corpus, encoding="utf-8") as f:
        assert [it["id"] for it in items] == \
               [json.loads(l)["id"] for l in f if l.strip()]

    a = tmp_path / "stock.jsonl"
    b = tmp_path / "ridgefill.jsonl"
    run(needle, stock_server, corpus, a)               # the stock engine: all right
    run(needle, ridgefill_server, tmp_path / "corpus_ridgefill.jsonl", b)

    needle.cmd_compare(SimpleNamespace(a_file=str(a), b_file=str(b)))
    out = capsys.readouterr().out
    assert "A 4/4 (100.0%)" in out and "B 2/4 (50.0%)" in out
    assert "by region" in out
    assert "bulk: A 2/2 (100.0%)   B 0/2 (0.0%)" in out
    assert "tail: A 2/2 (100.0%)   B 2/2 (100.0%)" in out
    assert "A right / B wrong: 2   A wrong / B right: 0" in out
    # b=2, c=0: p = 2 * C(2,0)/2^2 = 0.5
    assert "exact McNemar p = 0.5" in out
    assert "L3072/D0.10/single" in out and "L3072/D0.10/multi" in out
    assert "L3072/D0.90/single" not in out.split("disagree")[1]  # tail items agree


def test_compare_refuses_mismatched_ids(stock_server, tmp_path, capsys):
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    corpus = tmp_path / "corpus.jsonl"
    items = build(needle, stock_server, docs, corpus, lengths="512", depths="0.5")
    a = tmp_path / "a.jsonl"
    run(needle, stock_server, corpus, a)
    b = tmp_path / "b.jsonl"
    with open(b, "w", encoding="utf-8") as f:          # B covers only one item
        f.write(json.dumps({"id": items[0]["id"], "correct": False}) + "\n")
    with pytest.raises(SystemExit):
        needle.cmd_compare(SimpleNamespace(a_file=str(a), b_file=str(b)))
    assert "only in A" in capsys.readouterr().err

def test_build_chat_wraps_each_prompt_in_the_template_at_the_exact_length(stock_server, tmp_path):
    """--chat: every prompt is the template's user turn (thinking off) around filler + needles + question, still
    EXACTLY `length` ids, the needle positions shifted by the template's head; stock retrieves every needle."""
    needle = load_needle()
    docs = write_docs(tmp_path / "docs.jsonl")
    items = build(needle, stock_server, docs, tmp_path / "c.jsonl", chat=True)
    head = [stock_server.vocab["<|im_start|>user"]]
    tail = [stock_server.vocab[w] for w in ("<|im_end|>", "<|im_start|>assistant", "<think></think>")]
    for it in items:
        ids = it["prompt_ids"]
        assert len(ids) == it["length"] == it["prompt_tokens"]
        assert ids[:1] == head and ids[-3:] == tail
        for n in it["needles"]:
            assert stock_server.rev[ids[n["position"]]] == "The"
    needle.WRAP = ([], [])   # module state: the next test builds raw prompts
    res = run(needle, stock_server, tmp_path / "c.jsonl", tmp_path / "r.jsonl")
    assert all(r["correct"] for r in res)
