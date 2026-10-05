#!/usr/bin/env python3
"""KVA chat-template marker snippet tool.

Builds a Jinja snippet that, when merged in front of any chat template, makes
the template render a fixed 64-token marker as the very first thing in the
prompt — only when the request opts in via chat_template_kwargs
(`"kva": "on"`, plus optional dials). The radiance KVA plugin detects and
erases the marker on the GPU. Requests without the kwarg must render
byte-identically to the unmodified template.

Everything about the marker (token strings, ids, offsets, dial tables,
defaults, the switch kwarg and its on-value) is read from the marker spec
JSON; nothing is hard-coded here beyond the plugin's fixed marker length.

Subcommands:
  snippet --spec SPEC                          print the snippet alone
  merge   --spec SPEC --base B.jinja --out O   O = snippet + B, B unchanged
  check   --spec SPEC --base B --merged M      verify M == snippet + B exactly
  strip   --merged M --out B                   B = M minus a merged snippet

The snippet uses only minja-safe constructs (llama.cpp's Jinja subset):
`{#- -#}` comments, `{%- if %}`/`{%- set %}`, `is defined`, `==`, `and`,
the `|string` filter, string `~` concatenation, and whitespace-control
dashes. It therefore renders to nothing at all — not even a newline — when
the request did not opt in, and to exactly the marker tokens when it did.

Standard library only; no third-party imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Marker comments the tool emits. `merge` refuses any base that already
# contains one of these (idempotence); `strip` locates the merged block by
# the pair. Keep the two strings non-overlapping prefixes of each other.
MARKER_START = "{#- kva-marker v1"
MARKER_END = "{#- kva-marker-end v1 #}"

# The GPU side of the plugin detects exactly this many marker tokens; a spec
# that does not describe this many tokens cannot be used and is refused.
REQUIRED_LENGTH = 64

# The only fixed-token pattern rule this tool knows how to generate.
PATTERN_RULE = "alternate, starting with the first token"

PROG = "kva_template"


class SpecError(Exception):
    """The marker spec is malformed (message names the spec file)."""


# ---------------------------------------------------------------------------
# Spec loading and validation
# ---------------------------------------------------------------------------

def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_str(value) -> bool:
    return isinstance(value, str) and value != ""


def load_spec(spec_path: str) -> dict:
    """Load and validate the marker spec. Raises SpecError with a named reason."""
    path = Path(spec_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read spec {spec_path}: {exc.strerror or exc}") from exc
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpecError(f"{spec_path}: invalid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise SpecError(f"{spec_path}: spec must be a JSON object")
    _validate(spec, str(spec_path))
    return spec


def _validate(spec: dict, where: str) -> None:
    def err(reason: str) -> None:
        raise SpecError(f"{where}: {reason}")

    # -- length ---------------------------------------------------------
    length = spec.get("length")
    if not _is_int(length):
        err("length must be an integer")
    if length != REQUIRED_LENGTH:
        err(f"length must be {REQUIRED_LENGTH}, got {length}")

    # -- pattern segment ------------------------------------------------
    pattern = spec.get("pattern")
    if not isinstance(pattern, dict):
        err("pattern must be an object")
    p_offs = pattern.get("offsets")
    p_toks = pattern.get("tokens")
    rule = pattern.get("rule")
    if not (isinstance(p_offs, list) and len(p_offs) == 2 and all(_is_int(o) for o in p_offs)):
        err("pattern.offsets must be a pair of integers [first, last]")
    if p_offs[0] < 0 or p_offs[0] > p_offs[1]:
        err(f"pattern.offsets must be a non-negative increasing pair, got {p_offs}")
    if not (isinstance(p_toks, list) and len(p_toks) == 2 and all(_is_nonempty_str(t) for t in p_toks)):
        err("pattern.tokens must be two non-empty strings")
    if rule != PATTERN_RULE:
        err(f"unsupported pattern.rule {rule!r}; this tool only implements {PATTERN_RULE!r}")
    segments = [(p_offs[0], p_offs[1], "pattern")]

    # -- dial segments --------------------------------------------------
    dials = spec.get("dials")
    if not isinstance(dials, list):
        err("dials must be a list")
    dial_kwargs: set[str] = set()
    for i, dial in enumerate(dials):
        if not isinstance(dial, dict):
            err(f"dials[{i}] must be an object")
        offset = dial.get("offset")
        if not _is_int(offset):
            err(f"dials[{i}].offset must be an integer")
        kwarg = dial.get("kwarg")
        if not _is_nonempty_str(kwarg):
            err(f"dials[{i}].kwarg must be a non-empty string")
        if kwarg in dial_kwargs:
            err(f"duplicate dial kwarg {kwarg!r}")
        dial_kwargs.add(kwarg)
        default = dial.get("default")
        table = dial.get("table")
        if not _is_nonempty_str(default):
            err(f"dial {kwarg}: default must be a non-empty string")
        if not (
            isinstance(table, dict)
            and table
            and all(_is_nonempty_str(k) and _is_nonempty_str(v) for k, v in table.items())
        ):
            err(f"dial {kwarg}: table must be a non-empty string-to-string map")
        if default not in table:
            err(f"dial {kwarg}: default {default!r} not in its table")
        segments.append((offset, offset, f"dial {kwarg}"))

    # -- end segment ----------------------------------------------------
    end = spec.get("end")
    if not isinstance(end, dict):
        err("end must be an object")
    if not _is_int(end.get("offset")):
        err("end.offset must be an integer")
    if not _is_nonempty_str(end.get("token")):
        err("end.token must be a non-empty string")
    segments.append((end["offset"], end["offset"], "end"))

    # -- switch ----------------------------------------------------------
    switch = spec.get("switch")
    if not isinstance(switch, dict):
        err("switch must be an object")
    if not _is_nonempty_str(switch.get("kwarg")):
        err("switch.kwarg must be a non-empty string")
    if not _is_nonempty_str(switch.get("on_value")):
        err("switch.on_value must be a non-empty string")
    if switch["kwarg"] in dial_kwargs:
        err(f"switch kwarg {switch['kwarg']!r} collides with a dial kwarg")

    # -- token ids --------------------------------------------------------
    token_ids = spec.get("token_ids")
    if not (
        isinstance(token_ids, dict)
        and token_ids
        and all(_is_nonempty_str(k) and _is_int(v) for k, v in token_ids.items())
    ):
        err("token_ids must be a non-empty string-to-integer map")
    for offset in range(p_offs[0], p_offs[1] + 1):
        token = p_toks[(offset - p_offs[0]) % 2]
        if token not in token_ids:
            err(f"unknown token referenced: {token!r} (pattern, offset {offset})")
    for dial in dials:
        for value, token in dial["table"].items():
            if token not in token_ids:
                err(f"unknown token referenced: {token!r} (dial {dial['kwarg']}, value {value!r})")
    if end["token"] not in token_ids:
        err(f"unknown token referenced: {end['token']!r} (end)")

    # -- tiling: offsets must cover 0..length-1 with no gap or overlap ----
    segments.sort(key=lambda seg: seg[0])
    for start, stop, label in segments:
        if stop >= length:
            err(f"{label} offset {stop} is beyond length-1 = {length - 1}")
    pos = 0
    for start, stop, label in segments:
        if start > pos:
            err(f"missing offset {pos}" if start == pos + 1 else f"missing offsets {pos}-{start - 1}")
        if start < pos:
            err(f"overlapping offsets: {label} starts at {start} but offsets up to {pos - 1} are already covered")
        pos = stop + 1
    if pos != length:
        err(f"missing offsets {pos}-{length - 1}" if pos < length else "internal tiling error")


# ---------------------------------------------------------------------------
# Snippet generation
# ---------------------------------------------------------------------------

def _jinja_str(text: str) -> str:
    """Quote *text* as a single-quoted Jinja string literal."""
    escaped = (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f"'{escaped}'"


def _dial_lines(dial: dict) -> list[str]:
    """Lines that resolve one dial kwarg into `_kva_tok_<offset>` and `_kva_ok_<offset>`.

    Absent kwarg  -> default token, ok.
    Known value   -> table token, ok.
    Unknown value -> empty token, not ok (whole marker is suppressed).
    """
    offset = dial["offset"]
    kwarg = dial["kwarg"]
    tok_var = f"_kva_tok_{offset}"
    ok_var = f"_kva_ok_{offset}"
    default_token = dial["table"][dial["default"]]
    lines = [
        f"{{%- set {tok_var} = {_jinja_str(default_token)} -%}}",
        f"{{%- set {ok_var} = true -%}}",
        f"{{%- if {kwarg} is defined -%}}",
        f"{{%- set {tok_var} = '' -%}}",
        f"{{%- set {ok_var} = false -%}}",
    ]
    for value, token in dial["table"].items():
        lines += [
            f"{{%- if {kwarg}|string == {_jinja_str(value)} -%}}",
            f"{{%- set {tok_var} = {_jinja_str(token)} -%}}",
            f"{{%- set {ok_var} = true -%}}",
            "{%- endif -%}",
        ]
    lines.append("{%- endif -%}")
    return lines


def build_block(spec: dict) -> str:
    """Build the snippet: a marker comment, dial resolution, and one gated output.

    The returned string has no trailing newline; `merge` appends the base
    template's bytes directly after it.
    """
    length = spec["length"]
    p_lo, p_hi = spec["pattern"]["offsets"]
    p_toks = spec["pattern"]["tokens"]
    end_offset = spec["end"]["offset"]
    end_token = spec["end"]["token"]
    dial_by_offset = {dial["offset"]: dial for dial in spec["dials"]}
    switch_kwarg = spec["switch"]["kwarg"]
    switch_on = spec["switch"]["on_value"]

    # Walk every offset once: consecutive fixed tokens are fused into one
    # string literal, dial offsets become `_kva_tok_<offset>` variables.
    parts: list[str] = []  # expression chunks joined with ~
    literal: list[str] = []
    for offset in range(length):
        if offset in dial_by_offset:
            if literal:
                parts.append(_jinja_str("".join(literal)))
                literal = []
            parts.append(f"_kva_tok_{offset}")
        elif p_lo <= offset <= p_hi:
            literal.append(p_toks[(offset - p_lo) % 2])
        elif offset == end_offset:
            literal.append(end_token)
        else:  # unreachable after validation
            raise SpecError(f"spec covers offset {offset} with no token")
    if literal:
        parts.append(_jinja_str("".join(literal)))

    # The comment names length, the switch and the dials so a merged file is
    # self-describing; `merge`/`strip` find the block by these comments.
    meta = [f"length={length}", f"switch={switch_kwarg}", f"switch_value={switch_on}"]
    if spec["dials"]:
        meta.append("dials=" + ",".join(dial["kwarg"] for dial in spec["dials"]))
    comment = "{#- kva-marker v1 " + " ".join(meta) + " -#}"

    conditions = [f"{switch_kwarg} is defined", f"{switch_kwarg} == {_jinja_str(switch_on)}"]
    conditions += [f"_kva_ok_{dial['offset']}" for dial in spec["dials"]]

    gate = (
        "{%- if " + " and ".join(conditions) + " -%}"
        + "{{- " + " ~ ".join(parts) + " -}}"
        + "{%- endif %}" + MARKER_END
    )

    lines = [comment]
    for dial in spec["dials"]:
        lines += _dial_lines(dial)
    lines.append(gate)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()  # OSError propagates to main()


def _write_bytes(path: str, data: bytes) -> None:
    Path(path).write_bytes(data)  # OSError propagates to main()


def _first_diff(a: bytes, b: bytes) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _error(message: str) -> int:
    print(f"{PROG}: error: {message}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_snippet(args: argparse.Namespace) -> int:
    block = build_block(load_spec(args.spec))
    # Write exactly the snippet, no trailing newline, so that shell
    # redirection reproduces the merged file byte for byte.
    sys.stdout.write(block)
    sys.stdout.flush()
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    block = build_block(load_spec(args.spec)).encode("utf-8")
    base = _read_bytes(args.base)
    if MARKER_START.encode("utf-8") in base or MARKER_END.encode("utf-8") in base:
        return _error(f"{args.base}: already contains a kva marker block; refusing to merge again")
    if Path(args.out).resolve() == Path(args.base).resolve():
        return _error(f"{args.out}: output would overwrite the base template {args.base}")
    _write_bytes(args.out, block + base)
    print(f"merged: {args.out} = {len(block)}-byte kva block + {args.base} ({len(base)} bytes, unchanged)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    block = build_block(load_spec(args.spec)).encode("utf-8")
    base = _read_bytes(args.base)
    merged = _read_bytes(args.merged)
    if merged == block + base:
        print(f"OK: {args.merged} is the {len(block)}-byte kva block for spec {args.spec} "
              f"followed by {args.base} ({len(base)} bytes)")
        return 0
    if len(merged) >= len(block) and merged[: len(block)] == block:
        rest = merged[len(block):]
        n = _first_diff(rest, base)
        reason = f"base part differs from {args.base} at byte {len(block) + n}"
        if len(rest) != len(base):
            reason += f" (lengths {len(rest)} vs {len(base)})"
    elif len(merged) < len(block) and merged == block[: len(merged)]:
        reason = f"file ends after {len(merged)} bytes; the kva block for spec {args.spec} is {len(block)} bytes"
    else:
        n = _first_diff(merged[: len(block)], block)
        reason = f"kva marker block does not match spec {args.spec} at byte {n}"
    return _error(f"{args.merged}: {reason}")


def cmd_strip(args: argparse.Namespace) -> int:
    merged = _read_bytes(args.merged)
    end_marker = MARKER_END.encode("utf-8")
    idx = merged.find(end_marker)
    if idx < 0:
        return _error(f"{args.merged}: no kva marker block to strip ({MARKER_END} not found)")
    block = merged[: idx + len(end_marker)]
    if not block.startswith(MARKER_START.encode("utf-8")):
        return _error(f"{args.merged}: kva marker end found but the leading marker comment is missing or corrupt")
    if Path(args.out).resolve() == Path(args.merged).resolve():
        return _error(f"{args.out}: output would overwrite {args.merged}")
    _write_bytes(args.out, merged[idx + len(end_marker):])
    print(f"stripped: {args.out} = {args.merged} minus {len(block)}-byte kva block")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Build, merge, verify and strip the KVA chat-template marker snippet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("snippet", help="print the snippet alone (no trailing newline)")
    p.add_argument("--spec", required=True, help="marker spec JSON (read from, never written)")
    p.set_defaults(func=cmd_snippet)

    p = sub.add_parser("merge", help="write OUT = snippet + BASE (BASE bytes unchanged)")
    p.add_argument("--spec", required=True, help="marker spec JSON")
    p.add_argument("--base", required=True, help="the operator's chat template")
    p.add_argument("--out", required=True, help="merged template to write")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("check", help="verify MERGED == snippet + BASE byte for byte")
    p.add_argument("--spec", required=True, help="marker spec JSON")
    p.add_argument("--base", required=True, help="the operator's chat template")
    p.add_argument("--merged", required=True, help="merged template to verify")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("strip", help="remove a previously merged snippet (round trip)")
    p.add_argument("--merged", required=True, help="merged template to strip")
    p.add_argument("--out", required=True, help="base template to write")
    p.set_defaults(func=cmd_strip)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SpecError as exc:
        return _error(str(exc))
    except OSError as exc:
        return _error(f"{exc.filename or exc}: {exc.strerror or exc}")


if __name__ == "__main__":
    sys.exit(main())