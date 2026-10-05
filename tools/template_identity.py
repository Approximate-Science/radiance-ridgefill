#!/usr/bin/env python3
"""template_identity.py -- the engine-side identity check for the merged chat template.

The engine renders chat templates with minja (llama.cpp's Jinja subset), not
jinja2, so offline renders prove nothing: only the engine's own rendering
counts. This tool drives a running radiance server's `/tokenize` endpoint,
which renders `messages` with the same chat template the chat endpoints use
and tokenises what it produced (parse_special=true, add_special_tokens=false
-- core/server/admin.cpp, `/tokenize`), and checks R119/R120 of
fix-246/REQUIREMENTS-FIX.md against it:

  R119  every request WITHOUT the `kva` kwarg renders byte-identically to the
        stock template (0 differing ids over the suite); every request WITH
        `chat_template_kwargs {"kva": "on"}` renders the stock prompt with
        exactly the 64 marker ids in front; a dial value outside its table
        keeps the request off (identical to stock, no marker).
  R120  the engine's startup reply-format line ("chat: reply format template:
        ...", derived from the template: reasoning markers, tool-call parser)
        is identical with the stock and the merged template.

Subcommands:
  suite    --out suite.json                          write the request suite
  render   --server URL --suite S.json --out I.json  POST every case to
                                                    /tokenize, keep the ids
  diff     --stock A.json --merged B.json --spec SPEC.json
                                                   compare the two id files,
                                                   per-case table, exit 0/1
  replyfmt --stock-log A.log --merged-log B.log     the startup reply-format
                                                   line must be identical

The suite's cases are /tokenize bodies, so the ids they render are the ids the
chat endpoints serve. `render` stores the ids per case; `diff` re-derives each
case's expectation from the marker spec (switch, dial tables, defaults, token
ids -- resolved the way the merged snippet resolves them: `|string` after the
JSON value arrives) and cross-checks it against the expectation the suite
declares, so a case that claims a marker but does not opt in fails loudly.

Suite field spellings (read off the engine, see each case's comment):
  * `add_generation_prompt` and `tools` are read by /tokenize directly.
  * `reasoning_effort` is sent as a chat_template_kwargs key: /tokenize merges
    chat_template_kwargs into the template context but never reads the
    top-level reasoning_effort field (parse_thinking runs on the chat
    endpoints only), so the kwargs spelling is the one that exercises the
    template's effort branch through this endpoint. "high" is not one of this
    template's efforts (xhigh|medium|low -- it refuses 'high' with a 400
    naming the legal set, which leaves no ids to compare), so the suite's
    high case is the template's own high level, "xhigh".
  * `thinking_budget` is a sampler knob that never reaches the template; the
    case proves the field is accepted and the render unaffected.
  * `chat_template_kwargs {"enable_thinking": true/false}` is what the
    request carries; through /tokenize the engine renders with its
    deployment-default enable_thinking either way (the kwargs' bool is
    overwritten with the server's own -- chat-auto-parser-helpers.cpp), so on
    a default server both cases render with thinking on. Identity between
    stock and merged is what is checked, and it holds either way.

Standard library only. tools/kva_template.py (also stdlib-only) supplies the
spec loader; the fake radiance used by the tests lives in the test file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __package__:  # imported as tools.template_identity
    from .kva_template import SpecError, load_spec
else:            # run as a script: tools/ is on sys.path
    from kva_template import SpecError, load_spec

PROG = "template_identity"
ENDPOINT = "/tokenize"
REPLY_MARKER = "chat: reply format template:"
# A new radiance log record starts with its level letter and a space; the
# reply-format entry's own continuation lines (the template's embedded
# strings) never do, so this delimits where the entry ends.
REPLY_RECORD_START = re.compile(r"^[TIWEC] ", re.MULTILINE)

# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute path"}
                },
                "required": ["path"],
            },
        },
    }
]

TOOL_CALL_MESSAGES = [
    {"role": "user", "content": "What is in /etc/hosts? Use the read_file tool."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\": \"/etc/hosts\"}",
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n",
    },
]

LONG_USER_CONTENT = (
    "A customer pasted raw control-token text into the support chat and the "
    "pipeline stored it verbatim. Below is the transcript, exactly as typed; "
    "the marker-looking strings inside it are data, not control tokens.\n\n"
    + "".join(
        f"Turn {i}: the customer wrote \"<|quad_start|> please check batch "
        f"{i}, it reports <|quad_end|> after the merge step\" and the agent "
        f"answered about batch {i} without interpreting the markers; the "
        f"stored audit line reads <|im_start|>batch {i}<|im_end|> in the "
        f"nightly export, and the diff for batch {i} was empty.\n\n"
        for i in range(1, 9)
    )
    + "Summarise in one sentence why none of these pasted markers can change "
      "how the request is served, and confirm that the literal text "
      "<|quad_start|> a user types is just data inside their message."
)

SYSTEM_USER = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is the capital of France?"},
]

# (name, base request fields). `add_generation_prompt` defaults to true; only
# the shape that exercises its absence names it. `chat_template_kwargs` here
# are the shape's own kwargs -- the kva variants merge on top of them.
SHAPES = [
    ("system_user", {"messages": SYSTEM_USER}),
    ("user_only", {"messages": [
        {"role": "user",
         "content": "Explain what a KV cache is in one sentence."}]}),
    ("user_assistant_user", {"messages": [
        {"role": "user",
         "content": "Hello! Can you help me with a shell one-liner?"},
        {"role": "assistant",
         "content": "Of course. What should it do?"},
        {"role": "user",
         "content": "List the five largest files under /var/log, "
                    "human-readable sizes."}]}),
    ("tools_call_result", {"messages": TOOL_CALL_MESSAGES, "tools": TOOLS}),
    ("enable_thinking_true",
     {"messages": SYSTEM_USER, "chat_template_kwargs": {"enable_thinking": True}}),
    ("enable_thinking_false",
     {"messages": SYSTEM_USER, "chat_template_kwargs": {"enable_thinking": False}}),
    ("thinking_budget", {"messages": SYSTEM_USER, "thinking_budget": 2048}),
    ("reasoning_effort_low",
     {"messages": SYSTEM_USER, "chat_template_kwargs": {"reasoning_effort": "low"}}),
    # the template's own high level is "xhigh"; "high" it refuses with a 400
    ("reasoning_effort_xhigh",
     {"messages": SYSTEM_USER, "chat_template_kwargs": {"reasoning_effort": "xhigh"}}),
    ("no_generation_prompt",
     {"messages": SYSTEM_USER, "add_generation_prompt": False}),
    ("long_special_token_text", {"messages": [
        {"role": "user", "content": LONG_USER_CONTENT}]}),
]

# (name suffix, kva kwargs merged into the shape's kwargs, declared expect).
# None = do not add any kva kwarg (the shape's own kwargs stand).
VARIANTS = [
    ("", None, "identical"),
    ("__on", {"kva": "on"}, "marker"),
    ("__on_share", {"kva": "on", "kva_share": "0.50"}, "marker"),
    ("__on_alpha", {"kva": "on", "kva_alpha": "0.5"}, "marker"),
    ("__on_tail", {"kva": "on", "kva_tail": "2560"}, "marker"),
    # 0.75 is not in kva_share's table: the request stays off (identical)
    ("__on_invalid", {"kva": "on", "kva_share": "0.75"}, "identical"),
]

SUITE_COMMENT = (
    "R119 request suite: /tokenize bodies whose ids the stock and the merged "
    "(kva-marker) template must render identically for every case without "
    "`kva: on`; with `kva: on` the merged template must render the same ids "
    "with exactly the 64 marker ids of kva-marker-spec.json in front (dials "
    "resolved from the spec); an invalid dial value keeps the request off. "
    "Spellings: reasoning_effort rides chat_template_kwargs because /tokenize "
    "does not read the top-level field (parse_thinking runs on the chat "
    "endpoints only); the template's high effort is `xhigh` (it refuses "
    "`high` with a 400 naming the legal set, so `high` has no ids to "
    "compare); thinking_budget is a sampler knob that never reaches the "
    "template; through /tokenize the engine renders with its "
    "deployment-default enable_thinking whatever the kwargs say (the kwargs' "
    "bool is overwritten with the server's own), so the identity check is "
    "what these cases prove."
)


def build_suite() -> dict:
    cases = []
    for shape_name, fields in SHAPES:
        for suffix, kva_kwargs, expect in VARIANTS:
            request = dict(fields)
            kwargs = dict(fields.get("chat_template_kwargs") or {})
            if kva_kwargs is not None:
                kwargs.update(kva_kwargs)
            if kwargs:
                request["chat_template_kwargs"] = kwargs
            request.setdefault("add_generation_prompt", True)
            # the engine's chat form tokenises with add_special_tokens=false
            # (the rendered template already opens the prompt) and
            # parse_special=true (its control tokens recognised as themselves)
            request["add_special_tokens"] = False
            cases.append({
                "name": shape_name + suffix,
                "expect": expect,
                "request": request,
            })
    return {
        "comment": SUITE_COMMENT,
        "tool": "tools/template_identity.py",
        "endpoint": ENDPOINT,
        "usage": {
            "render_stock": f"{PROG} render --server URL --suite suite.json --out stock-ids.json",
            "render_merged": f"{PROG} render --server URL --suite suite.json --out merged-ids.json",
            "diff": f"{PROG} diff --stock stock-ids.json --merged merged-ids.json "
                    "--spec kva-marker-spec.json",
            "replyfmt": f"{PROG} replyfmt --stock-log stock.log --merged-log merged.log",
        },
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Marker resolution -- the spec's rules, re-derived here independently of the
# generated snippet (diff must not trust the very template it is checking)
# ---------------------------------------------------------------------------

def _dial_key(value):
    """How the snippet's `|string` filter stringifies a chat_template_kwargs
    JSON value: strings stay, 2048 -> '2048', 0.5 -> '0.5' (so the number 0.5
    does NOT match the table key '0.50'), booleans/null render as JSON spells
    them, and types with no defined rendering (objects, arrays) match nothing.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    return None


def resolve_marker(spec: dict, kwargs):
    """Resolve one case's chat_template_kwargs against the marker spec.

    Returns (ids, reason): ids is the list of 64 marker token ids when the
    request opts in (the switch is the on value and every dial that is set
    resolves into its table, absent dials taking their defaults), and None
    when it does not -- reason says which, in every case.
    """
    switch = spec["switch"]["kwarg"]
    on_value = spec["switch"]["on_value"]
    if switch not in kwargs:
        return None, f"no `{switch}` kwarg"
    value = kwargs[switch]
    if not (isinstance(value, str) and value == on_value):
        return None, f"`{switch}` is {json.dumps(value)}, not the on value {on_value!r}"

    lo, hi = spec["pattern"]["offsets"]
    pattern_tokens = spec["pattern"]["tokens"]
    dial_by_offset = {dial["offset"]: dial for dial in spec["dials"]}
    dial_values = {}
    ids = []
    for offset in range(spec["length"]):
        if lo <= offset <= hi:
            token = pattern_tokens[(offset - lo) % 2]
        elif offset in dial_by_offset:
            dial = dial_by_offset[offset]
            if dial["kwarg"] in kwargs:
                key = _dial_key(kwargs[dial["kwarg"]])
                if key is None or key not in dial["table"]:
                    got = json.dumps(kwargs[dial["kwarg"]])
                    return None, (f"dial `{dial['kwarg']}` is {got}, which is not one "
                                  f"of its table's values; the request stays off")
                token = dial["table"][key]
                dial_values[dial["kwarg"]] = key
            else:
                token = dial["table"][dial["default"]]
                dial_values[dial["kwarg"]] = dial["default"]
        elif offset == spec["end"]["offset"]:
            token = spec["end"]["token"]
        else:  # unreachable: load_spec refuses a spec with gaps or overlaps
            raise SpecError(f"spec covers offset {offset} with no token")
        ids.append(spec["token_ids"][token])

    dials = ", ".join(f"{k}={v}" for k, v in sorted(dial_values.items())) or "dials at defaults"
    return ids, f"`{switch}` is on ({dials})"


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------

def _error(message: str) -> int:
    print(f"{PROG}: error: {message}", file=sys.stderr)
    return 1


def _write_json(path: str, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def cmd_suite(args: argparse.Namespace) -> int:
    _write_json(args.out, build_suite())
    n = len(SHAPES) * len(VARIANTS)
    print(f"wrote {args.out}: {n} cases ({len(SHAPES)} shapes x {len(VARIANTS)} "
          "kva variants); render it against the stock and the merged server, "
          "then diff")
    return 0


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

class RenderError(Exception):
    """A case could not be rendered (the message names the case and server)."""


def _tokenize(server: str, request: dict, timeout: float) -> list[int]:
    url = server.rstrip("/") + ENDPOINT
    body = json.dumps(request).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RenderError(f"{url} answered HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RenderError(f"cannot reach {url}: {exc.reason}") from exc
    except (ValueError, OSError) as exc:
        raise RenderError(f"{url} did not answer JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RenderError(f"{url} did not answer a JSON object")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not all(
            isinstance(t, int) and not isinstance(t, bool) for t in tokens):
        raise RenderError(f"{url} answered no integer token array")
    if payload.get("count") != len(tokens):
        raise RenderError(f"{url} answered count {payload.get('count')} "
                           f"for {len(tokens)} tokens")
    return tokens


def _load_ids_doc(path: str, what: str) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RenderError(f"cannot read {what} {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise RenderError(f"{what} {path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("cases"), list):
        raise RenderError(f"{what} {path} must be an object with a cases array "
                           "(as `render` writes)")
    return doc


def cmd_render(args: argparse.Namespace) -> int:
    try:
        suite_path = Path(args.suite)
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _error(f"cannot read the suite {args.suite}: {exc}")
    cases = suite.get("cases") if isinstance(suite, dict) else None
    if not isinstance(cases, list) or not cases:
        return _error(f"the suite {args.suite} has no cases array")

    rendered = []
    try:
        for case in cases:
            name = case.get("name", "?")
            request = case.get("request")
            if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
                return _error(f"case {name!r} has no /tokenize request with messages")
            tokens = _tokenize(args.server, request, args.timeout)
            print(f"{len(tokens):6d} ids  {name}")
            rendered.append({
                "name": name,
                "expect": case.get("expect"),
                "chat_template_kwargs": request.get("chat_template_kwargs"),
                "count": len(tokens),
                "tokens": tokens,
            })
    except RenderError as exc:
        return _error(f"case {name!r}: {exc}")

    suite_sha = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    _write_json(args.out, {
        "server": args.server.rstrip("/"),
        "endpoint": ENDPOINT,
        "suite": str(args.suite),
        "suite_sha256": suite_sha,
        "n_cases": len(rendered),
        "cases": rendered,
    })
    print(f"rendered {len(rendered)} cases from {args.suite} against "
          f"{args.server.rstrip('/')}{ENDPOINT} into {args.out}")
    return 0


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def _first_diff(a: list[int], b: list[int]):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _identical_detail(stock: list[int], merged: list[int]) -> str:
    if len(stock) != len(merged):
        return f"the ids differ in length: stock {len(stock)}, merged {len(merged)}"
    i = _first_diff(stock, merged)
    if i is not None:
        return f"the ids differ from index {i} (stock id {stock[i]}, merged id {merged[i]})"
    return "the ids are identical"


def _marker_detail(spec_ids: list[int], stock: list[int], merged: list[int]) -> str:
    if merged == stock:
        return "merged equals stock: the 64 marker ids are missing"
    if len(merged) == len(stock):
        return f"merged has stock's length {len(stock)}: the 64 marker ids are missing"
    i = _first_diff(merged, spec_ids + stock)
    if i is None:
        return "the ids are exactly the marker ids followed by stock's"
    if i < len(spec_ids):
        return (f"the marker part differs at offset {i} "
                f"(merged id {merged[i]}, the spec wants {spec_ids[i]})")
    j = i - len(spec_ids)
    return (f"the part after the marker differs from stock from index {j} "
            f"(merged id {merged[i]}, stock id {stock[j]})")


def cmd_diff(args: argparse.Namespace) -> int:
    try:
        spec = load_spec(args.spec)
        stock_doc = _load_ids_doc(args.stock, "--stock")
        merged_doc = _load_ids_doc(args.merged, "--merged")
    except (SpecError, RenderError) as exc:
        return _error(str(exc))

    sha_stock, sha_merged = stock_doc.get("suite_sha256"), merged_doc.get("suite_sha256")
    if not sha_stock or not sha_merged:
        return _error("both id files must come from `render` (no suite sha256 recorded)")
    if sha_stock != sha_merged:
        return _error(f"the two id files come from different suites "
                      f"(--stock {sha_stock[:12]}..., --merged {sha_merged[:12]}...)")
    stock_cases = stock_doc["cases"]
    merged_cases = merged_doc["cases"]
    if [c.get("name") for c in stock_cases] != [c.get("name") for c in merged_cases]:
        return _error("the two id files do not carry the same cases in the same order")

    rows = []
    n_identical = n_marker = n_fail = 0
    for stock_case, merged_case in zip(stock_cases, merged_cases):
        name = stock_case.get("name", "?")
        kwargs = stock_case.get("chat_template_kwargs") or {}
        spec_ids, reason = resolve_marker(spec, kwargs)
        derived = "marker" if spec_ids is not None else "identical"
        declared = stock_case.get("expect")
        stock_ids = stock_case.get("tokens")
        merged_ids = merged_case.get("tokens")
        detail = ""
        ok = isinstance(stock_ids, list) and isinstance(merged_ids, list)
        if not ok:
            detail = "the case carries no token array"
        elif declared is not None and declared != derived:
            ok = False
            detail = (f"the suite declares {declared!r} but the spec resolves the "
                      f"request as {derived!r} ({reason})")
        elif spec_ids is None:
            if stock_ids != merged_ids:
                ok = False
                detail = f"no marker was expected, but {_identical_detail(stock_ids, merged_ids)}"
        else:
            if merged_ids != spec_ids + stock_ids:
                ok = False
                detail = _marker_detail(spec_ids, stock_ids, merged_ids)
        if ok:
            n_identical += derived == "identical"
            n_marker += derived == "marker"
        else:
            n_fail += 1
        rows.append((ok, derived, len(stock_ids) if isinstance(stock_ids, list) else -1,
                     len(merged_ids) if isinstance(merged_ids, list) else -1, name, detail))

    print(f"{'verdict':7}  {'expect':11} {'stock':>7} {'merged':>7}  case")
    for ok, derived, n_stock, n_merged, name, detail in rows:
        print(f"{'ok' if ok else 'FAIL':7}  {derived:11} {n_stock:7d} {n_merged:7d}  {name}")
        if not ok:
            print(f"{'':7}  {detail}")
    print(f"{len(rows)} cases: {len(rows) - n_fail} pass, {n_fail} fail "
          f"({n_identical} identical, {n_marker} marker+{spec['length']})")
    if n_fail:
        print("FAIL: the merged template is not identical to stock where it must be")
        return 1
    print("OK: every request without the kva kwarg is identical to stock, and "
          f"every kva:on request is stock with exactly the {spec['length']} "
          "marker ids in front (R119)")
    return 0


# ---------------------------------------------------------------------------
# replyfmt (R120)
# ---------------------------------------------------------------------------

class ReplyFmtError(Exception):
    """A log has no single reply-format entry (the message names the log)."""


def reply_format_entry(log_path: str) -> str:
    """The reply-format startup entry of a radiance log, prefix stripped.

    The entry can span several physical lines (the template's markers embed
    real newlines), so it runs from the marker to the next line that starts a
    new log record (level letter + space).
    """
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ReplyFmtError(f"cannot read {log_path}: {exc.strerror or exc}") from exc
    starts = [m.start() for m in re.finditer(re.escape(REPLY_MARKER), text)]
    if not starts:
        raise ReplyFmtError(f"{log_path}: no `{REPLY_MARKER}` startup line")
    if len(starts) > 1:
        raise ReplyFmtError(f"{log_path}: {len(starts)} `{REPLY_MARKER}` lines; "
                            "expected the one startup line")
    rest = text[starts[0]:]
    nxt = REPLY_RECORD_START.search(rest)
    return (rest if nxt is None else rest[:nxt.start()]).rstrip("\n")


def cmd_replyfmt(args: argparse.Namespace) -> int:
    try:
        stock = reply_format_entry(args.stock_log)
        merged = reply_format_entry(args.merged_log)
    except ReplyFmtError as exc:
        return _error(str(exc))
    print(f"stock:  {stock}")
    print(f"merged: {merged}")
    if stock == merged:
        print("OK: the reply format the engine derives is identical with the stock "
              "and the merged template (R120)")
        return 0
    print("FAIL: the reply format the engine derives from the merged template "
          "differs from stock's (R120)", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Engine-side identity check for the merged chat template: "
                    "suite -> render against two servers' /tokenize -> diff; "
                    "plus the reply-format line check (R119/R120).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("suite", help="write the request suite")
    p.add_argument("--out", required=True, help="suite JSON to write")
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser("render", help="POST every suite case to a server's /tokenize")
    p.add_argument("--server", required=True,
                   help="the radiance server's base URL (e.g. http://localhost:8000)")
    p.add_argument("--suite", required=True, help="the suite (from `suite`)")
    p.add_argument("--out", required=True, help="id file to write (input to `diff`)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-case HTTP timeout in seconds (default 60)")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("diff", help="compare stock and merged id files case by case")
    p.add_argument("--stock", required=True, help="ids rendered against the stock template")
    p.add_argument("--merged", required=True, help="ids rendered against the merged template")
    p.add_argument("--spec", required=True, help="kva-marker-spec.json")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("replyfmt", help="the reply-format startup line must be identical")
    p.add_argument("--stock-log", required=True, help="startup log of the stock server")
    p.add_argument("--merged-log", required=True, help="startup log of the merged server")
    p.set_defaults(func=cmd_replyfmt)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OSError as exc:
        return _error(f"{exc.filename or exc}: {exc.strerror or exc}")


if __name__ == "__main__":
    sys.exit(main())