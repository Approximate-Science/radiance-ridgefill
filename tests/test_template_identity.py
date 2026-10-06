"""Tests for tools/template_identity.py (the engine-side identity check).

Three layers, matching what the tool itself can prove offline:

  * the suite (what `suite` writes: shapes, ridgefill variants, spellings);
  * the pure diff rules on hand-made id lists (no server, no tokenizer);
  * an end-to-end run against a FAKE radiance server whose /tokenize renders
    a chat template through jinja2 and tokenises with the real served-model
    tokenizer named by $RIDGEFILL_TEST_TOKENIZER (the served model's tokenizer directory)
    when it is present (the whole layer is skipped when it is not, naming
    that path; RIDGEFILL_TEST_TOKENIZER overrides it). The fake is a stand-in: the
    engine renders with minja, so the real R119 check runs `render` against
    two real servers; here the fake only exercises the plumbing end to end.

The reply-format check (R120) is tested against logs with the real startup
line's bytes (the entry is one physical log line; the extraction also copes
with an entry that spans lines, up to the next log record).
"""

import contextlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.ridgefill_template import build_block, load_spec  # noqa: E402
import tools.template_identity as ti  # noqa: E402

TOOL = REPO_ROOT / "tools" / "template_identity.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SPEC_PATH = FIXTURES / "ridgefill-marker-spec.json"
BASE_TEMPLATE_PATH = FIXTURES / "container-chat-template.jinja"

# The tokenizer the engine is stated to serve (the shipped model's own); the
# whole fake-server layer skips without it, naming the path.
TOKENIZER_ENV = "RIDGEFILL_TEST_TOKENIZER"
TOKENIZER_DIR = Path(os.environ.get(TOKENIZER_ENV, ""))
needs_tokenizer = pytest.mark.skipif(
    not os.environ.get(TOKENIZER_ENV) or not TOKENIZER_DIR.is_dir(),
    reason=f"set {TOKENIZER_ENV} to the served model's tokenizer directory",
)

SHAPE_NAMES = [
    "system_user",
    "user_only",
    "user_assistant_user",
    "tools_call_result",
    "enable_thinking_true",
    "enable_thinking_false",
    "thinking_budget",
    "reasoning_effort_low",
    "reasoning_effort_xhigh",
    "no_generation_prompt",
    "long_special_token_text",
]
VARIANT_SUFFIXES = ["", "__on", "__on_share", "__on_alpha", "__on_tail", "__on_invalid"]


def run_cli(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def spec():
    return load_spec(str(SPEC_PATH))


@pytest.fixture(scope="module")
def tokenizer():
    # Only reached from tests marked needs_tokenizer, so the directory exists.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def suite_from_cli(tmp_path):
    out = tmp_path / "suite.json"
    result = run_cli("suite", "--out", out)
    assert result.returncode == 0, result.stderr
    return out, json.loads(out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------

def test_suite_covers_every_shape_in_every_ridgefill_variant(tmp_path):
    path, suite = suite_from_cli(tmp_path)
    assert suite["endpoint"] == "/tokenize"
    assert set(suite["usage"]) == {"render_stock", "render_merged", "diff", "replyfmt"}

    names = [case["name"] for case in suite["cases"]]
    expected = [shape + suffix for shape in SHAPE_NAMES for suffix in VARIANT_SUFFIXES]
    assert names == expected                      # 66 cases, no duplicates
    assert len(set(names)) == len(names)

    by_name = {case["name"]: case for case in suite["cases"]}
    for shape, suffix in [(s, x) for s in SHAPE_NAMES for x in VARIANT_SUFFIXES]:
        case = by_name[shape + suffix]
        request = case["request"]
        assert isinstance(request["messages"], list) and request["messages"]
        # every case tokenises the rendered prompt the way chat does
        assert request["add_special_tokens"] is False
        if shape == "no_generation_prompt":
            assert request["add_generation_prompt"] is False
        else:
            assert request["add_generation_prompt"] is True

        # the ridgefill variant folded into the shape's own kwargs
        kwargs = request.get("chat_template_kwargs") or {}
        if suffix == "":
            assert "ridgefill" not in kwargs                 # no ridgefill kwarg at all
            assert case["expect"] == "identical"
        elif suffix == "__on_invalid":
            # the shape's own kwargs (if any) ride along under the ridgefill ones
            assert kwargs["ridgefill"] == "on" and kwargs["ridgefill_share"] == "0.75"
            assert case["expect"] == "identical"      # unknown dial keeps it off
        else:
            assert kwargs["ridgefill"] == "on"
            assert case["expect"] == "marker"
        if suffix == "__on_share":
            assert kwargs["ridgefill_share"] == "0.50"
        if suffix == "__on_alpha":
            assert kwargs["ridgefill_alpha"] == "0.5"
        if suffix == "__on_tail":
            assert kwargs["ridgefill_tail"] == "2560"

    # the shape's own kwargs ride along under the ridgefill variants
    kwargs = by_name["enable_thinking_false__on_share"]["request"]["chat_template_kwargs"]
    assert kwargs == {"enable_thinking": False, "ridgefill": "on", "ridgefill_share": "0.50"}
    assert "ridgefill" not in (by_name["enable_thinking_true"]["request"].get("chat_template_kwargs") or {})

    # the shape fields the task names
    assert by_name["thinking_budget"]["request"]["thinking_budget"] == 2048
    assert by_name["reasoning_effort_low"]["request"]["chat_template_kwargs"] == \
        {"reasoning_effort": "low"}
    assert by_name["reasoning_effort_xhigh"]["request"]["chat_template_kwargs"] == \
        {"reasoning_effort": "xhigh"}                   # the template's own high level
    tools = by_name["tools_call_result__on"]["request"]["tools"]
    assert tools and tools[0]["function"]["name"] == "read_file"
    messages = by_name["tools_call_result__on"]["request"]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    long_content = by_name["long_special_token_text__on"]["request"]["messages"][0]["content"]
    assert "<|quad_start|>" in long_content and len(long_content) > 1000
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_suite_is_deterministic(tmp_path):
    _, first = suite_from_cli(tmp_path)
    out2 = tmp_path / "suite2.json"
    assert run_cli("suite", "--out", out2).returncode == 0
    assert json.loads(out2.read_text(encoding="utf-8")) == first


# ---------------------------------------------------------------------------
# Marker resolution (the spec's rules, re-derived independently of the tool)
# ---------------------------------------------------------------------------

def expected_ids(spec, dial_values=None):
    """A third derivation of the marker ids: walk the spec's own tables."""
    dial_values = dial_values or {}
    lo, hi = spec["pattern"]["offsets"]
    dial_by_offset = {d["offset"]: d for d in spec["dials"]}
    ids = []
    for offset in range(spec["length"]):
        if lo <= offset <= hi:
            token = spec["pattern"]["tokens"][(offset - lo) % 2]
        elif offset in dial_by_offset:
            dial = dial_by_offset[offset]
            token = dial["table"][dial_values.get(dial["kwarg"], dial["default"])]
        else:
            token = spec["end"]["token"]
        ids.append(spec["token_ids"][token])
    return ids


def test_resolve_marker_matches_the_spec_tables(spec):
    ids, reason = ti.resolve_marker(spec, {"ridgefill": "on"})
    assert ids == expected_ids(spec)                     # dials at their defaults
    assert "ridgefill_share=0.25" in reason and "ridgefill_alpha=1.0" in reason \
        and "ridgefill_tail=2048" in reason
    assert ids[0] == 248051 and ids[1] == 248052        # the pattern, pinned
    assert ids[63] == 248044                            # the end token
    assert ids[60:63] == [248050, 248047, 248050]       # share .25, alpha 1.0, tail 2048
    assert len(ids) == spec["length"] == 64


@pytest.mark.parametrize("kwargs,dial_values", [
    ({"ridgefill": "on", "ridgefill_share": "0.50"}, {"ridgefill_share": "0.50"}),
    ({"ridgefill": "on", "ridgefill_alpha": "0.5"}, {"ridgefill_alpha": "0.5"}),
    ({"ridgefill": "on", "ridgefill_tail": "2560"}, {"ridgefill_tail": "2560"}),
    ({"ridgefill": "on", "ridgefill_share": "0.10", "ridgefill_alpha": "0", "ridgefill_tail": "3072"},
     {"ridgefill_share": "0.10", "ridgefill_alpha": "0", "ridgefill_tail": "3072"}),
])
def test_resolve_marker_resolves_every_dial(spec, kwargs, dial_values):
    ids, reason = ti.resolve_marker(spec, kwargs)
    assert ids == expected_ids(spec, dial_values)
    for kwarg, value in dial_values.items():
        assert f"{kwarg}={value}" in reason


@pytest.mark.parametrize("kwargs,reason_part", [
    ({}, "no `ridgefill` kwarg"),
    ({"ridgefill": "off"}, "not the on value"),
    ({"ridgefill": True}, "not the on value"),                # a boolean is not the string "on"
    ({"enable_thinking": False}, "no `ridgefill` kwarg"),
    ({"ridgefill": "on", "ridgefill_share": "0.75"}, "not one of its table's values"),
    ({"ridgefill": "on", "ridgefill_share": 0.5}, "not one of its table's values"),
    ({"ridgefill": "on", "ridgefill_alpha": "nine"}, "not one of its table's values"),
])
def test_resolve_marker_off_cases_name_the_reason(spec, kwargs, reason_part):
    ids, reason = ti.resolve_marker(spec, kwargs)
    assert ids is None
    assert reason_part in reason


def test_dial_key_stringifies_json_values():
    assert ti._dial_key("0.50") == "0.50"
    assert ti._dial_key(2048) == "2048"
    assert ti._dial_key(0.5) == "0.5"                   # so 0.5 never matches "0.50"
    assert ti._dial_key(True) == "true"
    assert ti._dial_key(None) == "null"
    assert ti._dial_key({"a": 1}) is None               # no rendering: matches nothing


# ---------------------------------------------------------------------------
# diff rules on hand-made id lists
# ---------------------------------------------------------------------------

SUITE_SHA = "0" * 64


def ids_case(name, expect, kwargs, tokens):
    return {"name": name, "expect": expect, "chat_template_kwargs": kwargs,
            "count": len(tokens), "tokens": tokens}


def write_ids(path, cases, suite_sha=SUITE_SHA):
    doc = {"server": "fake://", "endpoint": "/tokenize", "suite_sha256": suite_sha,
           "n_cases": len(cases), "cases": cases}
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")


def run_diff(tmp_path, stock_cases, merged_cases):
    stock, merged = tmp_path / "stock.json", tmp_path / "merged.json"
    write_ids(stock, stock_cases)
    write_ids(merged, merged_cases)
    return run_cli("diff", "--stock", stock, "--merged", merged, "--spec", SPEC_PATH)


def test_diff_passes_when_the_rules_hold(tmp_path, spec):
    marker = expected_ids(spec)
    stock = [ids_case("plain", "identical", {}, [10, 11, 12]),
             ids_case("on", "marker", {"ridgefill": "on"}, [10, 11, 12]),
             ids_case("tail", "marker", {"ridgefill": "on", "ridgefill_tail": "2560"}, [20, 21]),
             ids_case("invalid", "identical", {"ridgefill": "on", "ridgefill_share": "0.75"}, [30, 31])]
    merged = [ids_case("plain", "identical", {}, [10, 11, 12]),
              ids_case("on", "marker", {"ridgefill": "on"}, marker + [10, 11, 12]),
              ids_case("tail", "marker", {"ridgefill": "on", "ridgefill_tail": "2560"},
                       expected_ids(spec, {"ridgefill_tail": "2560"}) + [20, 21]),
              ids_case("invalid", "identical", {"ridgefill": "on", "ridgefill_share": "0.75"}, [30, 31])]
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 cases: 4 pass, 0 fail" in result.stdout
    assert "OK:" in result.stdout


def test_diff_fails_when_the_marker_is_missing(tmp_path, spec):
    marker = expected_ids(spec)
    stock = [ids_case("on", "marker", {"ridgefill": "on"}, [10, 11])]
    merged = [ids_case("on", "marker", {"ridgefill": "on"}, [10, 11])]   # no marker rendered
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 1
    assert "the 64 marker ids are missing" in result.stdout
    assert "1 fail" in result.stdout


def test_diff_fails_when_a_request_without_ridgefill_gains_the_marker(tmp_path, spec):
    marker = expected_ids(spec)
    stock = [ids_case("plain", "identical", {}, [10, 11])]
    merged = [ids_case("plain", "identical", {}, marker + [10, 11])]
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 1
    assert "no marker was expected" in result.stdout


def test_diff_fails_when_a_dial_token_is_wrong(tmp_path, spec):
    stock = [ids_case("tail", "marker", {"ridgefill": "on", "ridgefill_tail": "2560"}, [10])]
    # the default marker: offset 62 carries the 2048 token, not 2560's
    merged = [ids_case("tail", "marker", {"ridgefill": "on", "ridgefill_tail": "2560"},
                       expected_ids(spec) + [10])]
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 1
    assert "marker part differs at offset 62" in result.stdout


def test_diff_fails_when_the_base_part_differs(tmp_path, spec):
    stock = [ids_case("on", "marker", {"ridgefill": "on"}, [10, 11, 12])]
    merged = [ids_case("on", "marker", {"ridgefill": "on"},
                       expected_ids(spec) + [10, 99, 12])]
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 1
    assert "part after the marker differs from stock from index 1" in result.stdout


def test_diff_fails_when_identical_ids_differ(tmp_path):
    stock = [ids_case("plain", "identical", {}, [10, 11, 12])]
    merged = [ids_case("plain", "identical", {}, [10, 99, 12])]
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 1
    assert "the ids differ from index 1" in result.stdout
    stock = [ids_case("plain", "identical", {}, [10, 11])]
    merged = [ids_case("plain", "identical", {}, [10, 11, 12])]
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 1
    assert "differ in length" in result.stdout


def test_diff_resolves_numeric_dial_values(tmp_path, spec):
    # JSON numbers resolve through |string: 2560 -> "2560" matches; 0.5 -> "0.5"
    # does not match ridgefill_share's "0.50", so the request stays off
    stock = [ids_case("tail", "marker", {"ridgefill": "on", "ridgefill_tail": 2560}, [10]),
             ids_case("share", "identical", {"ridgefill": "on", "ridgefill_share": 0.5}, [20])]
    merged = [ids_case("tail", "marker", {"ridgefill": "on", "ridgefill_tail": 2560},
                       expected_ids(spec, {"ridgefill_tail": "2560"}) + [10]),
              ids_case("share", "identical", {"ridgefill": "on", "ridgefill_share": 0.5}, [20])]
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 0, result.stdout + result.stderr


def test_diff_fails_when_the_suites_expectation_is_wrong(tmp_path, spec):
    # the suite declares a marker, but the dial value keeps the request off
    stock = [ids_case("bad", "marker", {"ridgefill": "on", "ridgefill_share": "0.75"}, [10])]
    merged = [ids_case("bad", "marker", {"ridgefill": "on", "ridgefill_share": "0.75"}, [10])]
    result = run_diff(tmp_path, stock, merged)
    assert result.returncode == 1
    assert "declares 'marker'" in result.stdout
    assert "resolves the request as 'identical'" in result.stdout


def test_diff_refuses_mismatched_id_files(tmp_path, spec):
    marker = expected_ids(spec)
    one = [ids_case("a", "identical", {}, [1, 2])]
    result = run_diff(tmp_path, one, one + [ids_case("b", "identical", {}, [3])])
    assert result.returncode == 1
    assert "do not carry the same cases" in result.stderr

    stock, merged = tmp_path / "s.json", tmp_path / "m.json"
    write_ids(stock, one)
    write_ids(merged, one, suite_sha="1" * 64)
    result = run_cli("diff", "--stock", stock, "--merged", merged, "--spec", SPEC_PATH)
    assert result.returncode == 1
    assert "different suites" in result.stderr


# ---------------------------------------------------------------------------
# replyfmt (R120)
# ---------------------------------------------------------------------------

# The real startup entry, byte for byte the one the engine prints for this
# model's template (the reasoning and tool-call markers it derives, with
# their literal backslash-n escapes, all on one physical log line). The
# fixture is written programmatically from a real startup log, so a
# display mangling of the tags cannot corrupt it.
REPLY_ENTRY = ('chat: reply format template: reasoning "<think>\\n".."\\n</think>\\n\\n"; calls "<tool_call>\\n""<function="NAME">\\n" "<parameter="ARG">\\n"""VALUE"</parameter>\\n" "</function>\\n""</tool_call>"; grammar on "<tool_call>\\n"')


def write_log(path, entry, prefix="I ", before="I chat: template loaded (8952 bytes)\n",
             after="I sampling defaults for requests that send none\n"):
    path.write_text(before + prefix + entry + "\n" + after, encoding="utf-8")


def test_replyfmt_passes_on_identical_entries(tmp_path):
    stock, merged = tmp_path / "stock.log", tmp_path / "merged.log"
    write_log(stock, REPLY_ENTRY)
    write_log(merged, REPLY_ENTRY, prefix="2026-10-05T06:27:41 I ",   # prefix is stripped
              before="I step batch: 7.19 MiB pinned staging ring\n")
    result = run_cli("replyfmt", "--stock-log", stock, "--merged-log", merged)
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout and "R120" in result.stdout


def test_replyfmt_fails_on_a_changed_reply_format(tmp_path):
    stock, merged = tmp_path / "stock.log", tmp_path / "merged.log"
    write_log(stock, REPLY_ENTRY)
    write_log(merged, REPLY_ENTRY.replace('"<parameter="ARG">\\n"', '"<parameter="ARG">\\n\\n"'))
    result = run_cli("replyfmt", "--stock-log", stock, "--merged-log", merged)
    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "differs from stock" in result.stderr


def test_replyfmt_refuses_logs_without_the_line(tmp_path):
    stock, merged = tmp_path / "stock.log", tmp_path / "merged.log"
    write_log(stock, REPLY_ENTRY)
    merged.write_text("I chat: template loaded (8952 bytes)\nI serving on http://0.0.0.0:8100\n",
                     encoding="utf-8")
    result = run_cli("replyfmt", "--stock-log", stock, "--merged-log", merged)
    assert result.returncode == 1
    assert str(merged) in result.stderr
    assert "no `chat: reply format template:`" in result.stderr

    twice = tmp_path / "twice.log"
    twice.write_text("I " + REPLY_ENTRY + "\nI " + REPLY_ENTRY + "\n", encoding="utf-8")
    result = run_cli("replyfmt", "--stock-log", twice, "--merged-log", stock)
    assert result.returncode == 1
    assert "2 `chat: reply format template:` lines" in result.stderr


# ---------------------------------------------------------------------------
# The fake radiance server (jinja2 + the served model's tokenizer)
# ---------------------------------------------------------------------------

class _Refused(Exception):
    """The request the fake refused (becomes the 400's message)."""


def _normalize_message(message):
    """What the engine hands the template: tool-call arguments as an object
    (llama.cpp parses the OAI wire string), null content as ''."""
    message = dict(message)
    if message.get("content") is None:
        message["content"] = ""
    if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
        calls = []
        for call in message["tool_calls"]:
            call = dict(call)
            function = dict(call.get("function") or {})
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except ValueError:
                    pass
            call["function"] = function
            calls.append(call)
        message["tool_calls"] = calls
    return message


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/tokenize":
            self._respond(404, {"error": {"message": f"no {self.path} endpoint"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            tokens = self.server.radiance.render_tokenize(body)
        except _Refused as exc:
            self._respond(400, {"error": {"message": str(exc)}})
        else:
            self._respond(200, {"count": len(tokens), "max_model_len": 200000,
                                "tokens": tokens, "token_strs": None})

    def _respond(self, code, doc):
        payload = json.dumps(doc).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class FakeRadiance:
    """A /tokenize stand-in shaped like the engine's (admin.cpp): renders the
    messages with the template, tokenises with parse_special and no BOS."""

    def __init__(self, template_source, tokenizer):
        import jinja2

        def raise_exception(message):
            raise _Refused(message)

        env = jinja2.Environment()
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: "2026-01-01"
        self.template = env.from_string(template_source)
        self.tokenizer = tokenizer

    def render_tokenize(self, body):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise _Refused("messages must be a non-empty array")
        context = {"messages": [_normalize_message(m) for m in messages],
                   "bos_token": "", "eos_token": ""}
        tools = body.get("tools")
        if isinstance(tools, list) and tools:
            context["tools"] = tools
        if body.get("add_generation_prompt", True):
            context["add_generation_prompt"] = True
        context.update(body.get("chat_template_kwargs") or {})
        # the engine overwrites the kwargs' enable_thinking with its own bool
        # (chat-auto-parser-helpers.cpp); the fake's deployment default is on
        context["enable_thinking"] = True
        prompt = self.template.render(**context)
        return self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    @contextlib.contextmanager
    def serve(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.radiance = self
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


def render_suite(tmp_path, url, suite_path, name):
    out = tmp_path / name
    result = run_cli("render", "--server", url, "--suite", suite_path, "--out", out)
    assert result.returncode == 0, result.stderr
    return out, json.loads(out.read_text(encoding="utf-8"))


@needs_tokenizer
def test_end_to_end_stock_vs_merged_over_the_fake_servers(tmp_path, spec, tokenizer):
    base = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    suite_path, _ = suite_from_cli(tmp_path)
    with FakeRadiance(base, tokenizer).serve() as stock_url, \
            FakeRadiance(build_block(spec) + base, tokenizer).serve() as merged_url:
        stock_file, stock_doc = render_suite(tmp_path, stock_url, suite_path, "stock-ids.json")
        merged_file, merged_doc = render_suite(tmp_path, merged_url, suite_path, "merged-ids.json")
        assert stock_doc["n_cases"] == merged_doc["n_cases"] == 66
        result = run_cli("diff", "--stock", stock_file, "--merged", merged_file,
                         "--spec", SPEC_PATH)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "66 cases: 66 pass, 0 fail (22 identical, 44 marker+64)" in result.stdout

    # spot-check the two rules on the ids themselves
    stock_by_name = {c["name"]: c for c in stock_doc["cases"]}
    merged_by_name = {c["name"]: c for c in merged_doc["cases"]}
    assert merged_by_name["system_user"]["tokens"] == stock_by_name["system_user"]["tokens"]
    marker = expected_ids(spec)
    on = merged_by_name["system_user__on"]["tokens"]
    assert on == marker + stock_by_name["system_user__on"]["tokens"]
    assert on[:64] == marker
    assert merged_by_name["system_user__on_share"]["tokens"][:64] == \
        expected_ids(spec, {"ridgefill_share": "0.50"})
    assert merged_by_name["long_special_token_text__on_invalid"]["tokens"] == \
        stock_by_name["long_special_token_text__on_invalid"]["tokens"]


@needs_tokenizer
def test_end_to_end_a_broken_gate_is_caught(tmp_path, spec, tokenizer):
    """A gate that fires on every request is exactly the regression `diff`
    exists to catch: the cases without the kwarg gain the marker."""
    base = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    broken_gate = build_block(spec).replace("ridgefill is defined and ridgefill == 'on'", "true")
    suite_path, _ = suite_from_cli(tmp_path)
    with FakeRadiance(base, tokenizer).serve() as stock_url, \
            FakeRadiance(broken_gate + base, tokenizer).serve() as merged_url:
        stock_file, _ = render_suite(tmp_path, stock_url, suite_path, "stock-ids.json")
        merged_file, _ = render_suite(tmp_path, merged_url, suite_path, "merged-ids.json")
        result = run_cli("diff", "--stock", stock_file, "--merged", merged_file,
                         "--spec", SPEC_PATH)
    assert result.returncode == 1
    # every no-ridgefill case gained the marker; the invalid-dial cases stay off even
    # with a broken switch, because the dial checks (_ridgefill_ok_*) are in the gate
    assert "55 pass, 11 fail (11 identical, 44 marker+64)" in result.stdout
    assert "no marker was expected" in result.stdout