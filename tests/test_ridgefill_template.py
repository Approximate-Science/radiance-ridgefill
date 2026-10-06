# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/ridgefill_template.py (RidgeFill chat-template marker snippet).

Offline verification only — the inference engine is never run:
  * jinja2 stands in for the template engine (with minimal stubs for the
    non-Jinja callables both base templates use: `raise_exception`;
    `strftime_now` is provided too, though neither base needs it);
  * a shipped tokenizer verifies the marker is exactly the spec's 64 ids.

The spec and the container base template are portable fixtures beside this
file; the operator template and the tokenizer are OPTIONAL machine-local
inputs, pointed at by `RIDGEFILL_TEST_OPERATOR_TEMPLATE` and `RIDGEFILL_TEST_TOKENIZER`.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jinja2 import Environment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.ridgefill_template import build_block, load_spec  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPEC_PATH = FIXTURES_DIR / "ridgefill-marker-spec.json"

OPERATOR_TEMPLATE_ENV = "RIDGEFILL_TEST_OPERATOR_TEMPLATE"
TOKENIZER_ENV = "RIDGEFILL_TEST_TOKENIZER"
_operator_template = os.environ.get(OPERATOR_TEMPLATE_ENV)
OPERATOR_TEMPLATE = Path(_operator_template) if _operator_template else None
_tokenizer_dir = os.environ.get(TOKENIZER_ENV)
TOKENIZER_DIR = Path(_tokenizer_dir) if _tokenizer_dir else None

needs_operator_template = pytest.mark.skipif(
    OPERATOR_TEMPLATE is None or not OPERATOR_TEMPLATE.is_file(),
    reason=f"optional machine-local input: set {OPERATOR_TEMPLATE_ENV} to the operator base "
           "template (it is unset, or its file is missing)",
)
needs_tokenizer = pytest.mark.skipif(
    TOKENIZER_DIR is None or not TOKENIZER_DIR.is_dir(),
    reason=f"optional machine-local input: set {TOKENIZER_ENV} to the tokenizer directory "
           "(it is unset, or its directory is missing)",
)

BASE_PATHS = {
    "container": FIXTURES_DIR / "container-chat-template.jinja",
    "operator": OPERATOR_TEMPLATE,
}


def _base_params(names):
    """A param per base template; the operator one is skipped when its
    machine-local env var is unset or missing (the skip reason names it)."""
    params = []
    for name in names:
        marks = [needs_operator_template] if name == "operator" else []
        params.append(pytest.param(name, marks=marks))
    return params


TOOL = REPO_ROOT / "tools" / "ridgefill_template.py"

SCENARIOS = [
    (
        "system_user_gen",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        True,
    ),
    (
        "system_user_no_gen",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about GPUs."},
        ],
        False,
    ),
    (
        "user_only",
        [{"role": "user", "content": "Explain KV caches in one sentence."}],
        True,
    ),
    (
        "user_assistant_user",
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there. How can I help?"},
            {"role": "user", "content": "Name three colors."},
        ],
        True,
    ),
]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spec():
    return load_spec(str(SPEC_PATH))


@pytest.fixture(scope="session")
def block(spec):
    return build_block(spec)


@pytest.fixture(scope="session")
def tokenizer():
    # Only reached from tests marked with `needs_tokenizer`, so TOKENIZER_DIR is set.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


@pytest.fixture()
def env():
    """jinja2 environment with the minimal stubs the base templates need.

    Both base templates call `raise_exception(...)`; neither calls
    `strftime_now`, but it is stubbed anyway so other operator templates
    can be dropped into BASE_PATHS without surprise.
    """

    def raise_exception(message):
        raise ValueError(message)

    environment = Environment()
    environment.globals["raise_exception"] = raise_exception
    environment.globals["strftime_now"] = lambda fmt: "2026-01-01"
    return environment


def render(env, source, **context):
    return env.from_string(source).render(**context)


def base_source(name):
    return BASE_PATHS[name].read_text(encoding="utf-8")


def merged_source(block, name):
    return block + base_source(name)


def run_cli(*args, binary_stdout=False):
    result = subprocess.run(
        [sys.executable, str(TOOL), *map(str, args)],
        capture_output=True,
        text=not binary_stdout,
    )
    return result


def expected_marker(spec, dial_values=None):
    """Independent re-implementation of the marker string from the spec JSON."""
    dial_values = dial_values or {}
    lo, hi = spec["pattern"]["offsets"]
    pattern_tokens = spec["pattern"]["tokens"]
    dial_by_offset = {dial["offset"]: dial for dial in spec["dials"]}
    parts = []
    for offset in range(spec["length"]):
        if lo <= offset <= hi:
            parts.append(pattern_tokens[(offset - lo) % 2])
        elif offset in dial_by_offset:
            dial = dial_by_offset[offset]
            value = dial_values.get(dial["kwarg"], dial["default"])
            parts.append(dial["table"][value])
        else:
            parts.append(spec["end"]["token"])
    return "".join(parts)


def expected_ids(spec, dial_values=None):
    """The spec's token id at each of the 64 marker offsets."""
    dial_values = dial_values or {}
    lo, hi = spec["pattern"]["offsets"]
    pattern_tokens = spec["pattern"]["tokens"]
    dial_by_offset = {dial["offset"]: dial for dial in spec["dials"]}
    ids = []
    for offset in range(spec["length"]):
        if lo <= offset <= hi:
            token = pattern_tokens[(offset - lo) % 2]
        elif offset in dial_by_offset:
            dial = dial_by_offset[offset]
            token = dial["table"][dial_values.get(dial["kwarg"], dial["default"])]
        else:
            token = spec["end"]["token"]
        ids.append(spec["token_ids"][token])
    return ids


def write_spec_variant(tmp_path, mutate, name="spec.json"):
    """Copy the real spec with one mutation applied, for malformed-spec tests."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    mutate(spec)
    path = tmp_path / name
    path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI: snippet / merge / check / strip
# ---------------------------------------------------------------------------

def test_snippet_prints_the_block_exactly(tmp_path, block):
    result = run_cli("snippet", "--spec", SPEC_PATH, binary_stdout=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == block.encode("utf-8")  # byte-exact, no trailing newline


def test_check_accepts_a_fresh_merge(tmp_path, block):
    merged = tmp_path / "merged.jinja"
    assert run_cli("merge", "--spec", SPEC_PATH, "--base", BASE_PATHS["container"],
                   "--out", merged).returncode == 0
    result = run_cli("check", "--spec", SPEC_PATH, "--base", BASE_PATHS["container"],
                     "--merged", merged)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK:")


def test_check_names_the_reason_on_corruption(tmp_path):
    merged = tmp_path / "merged.jinja"
    run_cli("merge", "--spec", SPEC_PATH, "--base", BASE_PATHS["container"], "--out", merged)
    data = bytearray(merged.read_bytes())
    data[len(data) - 10] ^= 0x01  # corrupt inside the base part
    merged.write_bytes(bytes(data))

    result = run_cli("check", "--spec", SPEC_PATH, "--base", BASE_PATHS["container"],
                     "--merged", merged)
    assert result.returncode != 0
    assert "base part differs" in result.stderr
    assert str(BASE_PATHS["container"]) in result.stderr

    data = bytearray(merged.read_bytes())
    data[100] ^= 0x01  # corrupt inside the marker block
    merged.write_bytes(bytes(data))
    result = run_cli("check", "--spec", SPEC_PATH, "--base", BASE_PATHS["container"],
                     "--merged", merged)
    assert result.returncode != 0
    assert "marker block does not match spec" in result.stderr


@pytest.mark.parametrize("base_name", _base_params(BASE_PATHS))
def test_merged_file_is_block_plus_base_bytes(tmp_path, block, base_name):
    out = tmp_path / f"merged-{base_name}.jinja"
    result = run_cli("merge", "--spec", SPEC_PATH, "--base", BASE_PATHS[base_name], "--out", out)
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == block.encode("utf-8") + BASE_PATHS[base_name].read_bytes()


@pytest.mark.parametrize("base_name", _base_params(BASE_PATHS))
def test_strip_round_trips_a_merge(tmp_path, base_name):
    merged = tmp_path / "merged.jinja"
    stripped = tmp_path / "base.jinja"
    assert run_cli("merge", "--spec", SPEC_PATH, "--base", BASE_PATHS[base_name],
                   "--out", merged).returncode == 0
    result = run_cli("strip", "--merged", merged, "--out", stripped)
    assert result.returncode == 0, result.stderr
    assert stripped.read_bytes() == BASE_PATHS[base_name].read_bytes()


def test_merge_refuses_a_second_merge(tmp_path):
    merged = tmp_path / "merged.jinja"
    twice = tmp_path / "merged-twice.jinja"
    assert run_cli("merge", "--spec", SPEC_PATH, "--base", BASE_PATHS["container"],
                   "--out", merged).returncode == 0
    result = run_cli("merge", "--spec", SPEC_PATH, "--base", merged, "--out", twice)
    assert result.returncode != 0
    assert "already contains a ridgefill marker block" in result.stderr
    assert str(merged) in result.stderr  # the reason names the file
    assert not twice.exists()


def test_strip_refuses_a_file_without_marker(tmp_path):
    out = tmp_path / "base.jinja"
    result = run_cli("strip", "--merged", BASE_PATHS["container"], "--out", out)
    assert result.returncode != 0
    assert "no ridgefill marker block to strip" in result.stderr
    assert str(BASE_PATHS["container"]) in result.stderr
    assert not out.exists()


# ---------------------------------------------------------------------------
# Rendering: requests WITHOUT the kwarg are byte-identical to the base
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base_name", _base_params(BASE_PATHS))
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_off_case_renders_byte_identically(env, block, base_name, scenario):
    _, messages, add_generation_prompt = scenario
    context = dict(messages=messages, add_generation_prompt=add_generation_prompt)
    base = render(env, base_source(base_name), **context)
    off = render(env, merged_source(block, base_name), **context)
    assert off == base


# ---------------------------------------------------------------------------
# Rendering: requests WITH ridgefill="on" get exactly the 64-token marker first
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base_name", _base_params(BASE_PATHS))
@pytest.mark.parametrize("scenario", [SCENARIOS[0], SCENARIOS[3]])
def test_on_case_renders_marker_plus_base(env, block, spec, base_name, scenario):
    _, messages, add_generation_prompt = scenario
    context = dict(messages=messages, add_generation_prompt=add_generation_prompt)
    marker = expected_marker(spec)
    base = render(env, base_source(base_name), **context)
    on = render(env, merged_source(block, base_name), ridgefill="on", **context)
    assert on == marker + base  # marker is the very first thing in the prompt


def test_block_alone_emits_nothing_when_off(env, block):
    assert render(env, block) == ""
    assert render(env, block, ridgefill="off") == ""
    assert render(env, block, messages=[{"role": "user", "content": "hi"}]) == ""


@pytest.mark.parametrize(
    "ridgefill_value", ["off", "ON", "on ", "", 1, True, None], ids=lambda v: repr(v)
)
def test_non_on_switch_values_disable_the_marker(env, block, ridgefill_value):
    context = dict(messages=SCENARIOS[0][1], add_generation_prompt=True, ridgefill=ridgefill_value)
    base = render(env, base_source("container"), **context)
    assert render(env, merged_source(block, "container"), **context) == base


@needs_tokenizer
def test_marker_tokenizes_to_exactly_the_spec_ids(spec, tokenizer):
    marker = expected_marker(spec)
    ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    assert len(ids) == spec["length"] == 64
    assert ids == expected_ids(spec)


# ---------------------------------------------------------------------------
# Dials: values, defaults, unknown values
# ---------------------------------------------------------------------------

def dial_cases(spec):
    for dial in spec["dials"]:
        for value in dial["table"]:
            yield pytest.param(dial["kwarg"], value, id=f"{dial['kwarg']}={value}")


@pytest.mark.parametrize("kwarg,value", list(dial_cases(json.loads(SPEC_PATH.read_text()))))
@needs_tokenizer
def test_each_dial_value_selects_its_token(env, spec, tokenizer, kwarg, value):
    context = dict(ridgefill="on", **{kwarg: value})
    marker = render(env, build_block(spec), **context)
    expected = expected_marker(spec, {kwarg: value})
    assert marker == expected
    ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    assert ids == expected_ids(spec, {kwarg: value})


def test_defaults_used_when_dials_absent(env, spec):
    with_defaults = render(env, build_block(spec), ridgefill="on")
    explicit = render(
        env,
        build_block(spec),
        ridgefill="on",
        **{dial["kwarg"]: dial["default"] for dial in spec["dials"]},
    )
    assert with_defaults == explicit == expected_marker(spec)


@pytest.mark.parametrize(
    "kwarg", [d["kwarg"] for d in json.loads(SPEC_PATH.read_text())["dials"]]
)
def test_unknown_dial_value_suppresses_the_marker(env, block, kwarg):
    context = dict(messages=SCENARIOS[0][1], add_generation_prompt=True,
                   ridgefill="on", **{kwarg: "not-a-table-value"})
    base = render(env, base_source("container"), **context)
    assert render(env, merged_source(block, "container"), **context) == base


@needs_tokenizer
def test_numeric_dial_values_match_the_string_tables(env, spec, tokenizer):
    """chat_template_kwargs arrive as JSON: numbers like 0.5 / 2048 must resolve
    via |string to the table keys '0.5' / '2048'."""
    marker = render(env, build_block(spec), ridgefill="on", ridgefill_alpha=0.5, ridgefill_tail=2048)
    assert marker == expected_marker(spec, {"ridgefill_alpha": "0.5", "ridgefill_tail": "2048"})
    ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    assert ids == expected_ids(spec, {"ridgefill_alpha": "0.5", "ridgefill_tail": "2048"})


# ---------------------------------------------------------------------------
# Malformed specs are refused, each with a named reason (spec path included)
# ---------------------------------------------------------------------------

def run_snippet(spec_path):
    return run_cli("snippet", "--spec", spec_path)


def assert_refused(tmp_path, mutate, reason_substring):
    spec_path = write_spec_variant(tmp_path, mutate)
    result = run_snippet(spec_path)
    assert result.returncode != 0
    assert reason_substring in result.stderr
    assert str(spec_path) in result.stderr  # the reason names the spec file
    assert result.stdout == ""  # nothing printed on refusal


def test_spec_with_wrong_length_is_refused(tmp_path):
    assert_refused(tmp_path, lambda s: s.update(length=63), "length must be 64, got 63")


def test_spec_with_overlapping_offsets_is_refused(tmp_path):
    assert_refused(
        tmp_path,
        lambda s: s["dials"][0].update(offset=59),  # collides with the pattern
        "overlapping offsets",
    )


def test_spec_with_missing_offsets_is_refused(tmp_path):
    def drop_share_dial(spec):
        spec["dials"] = [d for d in spec["dials"] if d["kwarg"] != "ridgefill_share"]

    assert_refused(tmp_path, drop_share_dial, "missing offset 60")


def test_spec_referencing_unknown_token_is_refused(tmp_path):
    assert_refused(
        tmp_path,
        lambda s: s["token_ids"].pop("<|box_end|>"),
        "unknown token referenced: '<|box_end|>'",
    )


def test_spec_with_dial_default_not_in_table_is_refused(tmp_path):
    assert_refused(
        tmp_path,
        lambda s: s["dials"][0].update(default="0.75"),
        "default '0.75' not in its table",
    )


def test_spec_with_invalid_json_is_refused(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    result = run_snippet(path)
    assert result.returncode != 0
    assert "invalid JSON" in result.stderr
    assert str(path) in result.stderr