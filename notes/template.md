# KVA chat-template marker tool — `tools/kva_template.py`

Offline, standard-library-only tool that turns the marker spec into a Jinja
snippet, merges it in front of any operator chat template, verifies merges,
and strips them again. The radiance KVA plugin erases the marker on the GPU;
this repo only makes sure the *template* side emits it correctly.

Everything about the marker (token strings, ids, offsets, dial tables and
defaults, the switch kwarg and its on-value) is read from the spec at
`/var/home/dylan/AI-Work/radiance-kva-plugin-20261004/worker-context/kva-marker-spec.json`
— the tool has no hard-coded tokens or offsets, only the plugin's fixed marker
length of 64 and the one pattern rule it knows.

## Usage

```sh
V=/var/home/dylan/projects/research/kva/.venv/bin/python   # stdlib-only tool; any python works

# Print the snippet alone (exactly the snippet, no trailing newline,
# so `> file.jinja` reproduces merge output byte for byte):
$V tools/kva_template.py snippet --spec kva-marker-spec.json

# Merge: OUT = snippet + BASE, BASE's bytes appended completely unchanged:
$V tools/kva_template.py merge --spec kva-marker-spec.json \
      --base sharp-v22-dylan.jinja --out sharp-v22-dylan.kva.jinja

# Verify a merged file byte for byte (exit 0 = OK, 1 = reason on stderr):
$V tools/kva_template.py check --spec kva-marker-spec.json \
      --base sharp-v22-dylan.jinja --merged sharp-v22-dylan.kva.jinja

# Remove a previously merged block (exact round trip of merge):
$V tools/kva_template.py strip --merged sharp-v22-dylan.kva.jinja --out sharp-v22-dylan.jinja
```

Refusals (non-zero exit, reason on stderr, always naming the file):
`merge` refuses a base that already contains a kva marker block
(`{#- kva-marker v1 … -#}` or the end comment) — idempotence — and an output
path equal to the base; `strip` refuses files without a marker block;
malformed specs are refused by every subcommand that reads one.

## Snippet behavior

| request (`chat_template_kwargs`)                        | rendered prefix                          |
|---------------------------------------------------------|------------------------------------------|
| no `kva` kwarg                                          | nothing at all (byte-identical to base)  |
| `kva` ≠ `"on"` (e.g. `"off"`, `1`, `true`, `"ON"`)      | nothing at all                           |
| `kva="on"`, dials absent or all in their tables          | the 64-token marker, first thing in prompt |
| `kva="on"`, any dial set to a value not in its table    | nothing at all (request stays off)       |

Absent dials use their spec defaults; dial values are compared after `|string`,
so JSON numbers work too (`kva_tail: 2048` matches the key `"2048"`;
`kva_share: 0.5` does *not* match `"0.50"` and keeps the request off).

## Generated snippet for the current spec

Produced by `kva_template.py snippet --spec kva-marker-spec.json`:

```jinja
{#- kva-marker v1 length=64 switch=kva switch_value=on dials=kva_share,kva_alpha,kva_tail -#}
{%- set _kva_tok_60 = '<|box_end|>' -%}
{%- set _kva_ok_60 = true -%}
{%- if kva_share is defined -%}
{%- set _kva_tok_60 = '' -%}
{%- set _kva_ok_60 = false -%}
{%- if kva_share|string == '0.10' -%}
{%- set _kva_tok_60 = '<|box_start|>' -%}
{%- set _kva_ok_60 = true -%}
{%- endif -%}
{%- if kva_share|string == '0.25' -%}
{%- set _kva_tok_60 = '<|box_end|>' -%}
{%- set _kva_ok_60 = true -%}
{%- endif -%}
{%- if kva_share|string == '0.50' -%}
{%- set _kva_tok_60 = '<|object_ref_start|>' -%}
{%- set _kva_ok_60 = true -%}
{%- endif -%}
{%- endif -%}
{%- set _kva_tok_61 = '<|object_ref_start|>' -%}
{%- set _kva_ok_61 = true -%}
{%- if kva_alpha is defined -%}
{%- set _kva_tok_61 = '' -%}
{%- set _kva_ok_61 = false -%}
{%- if kva_alpha|string == '0' -%}
{%- set _kva_tok_61 = '<|box_start|>' -%}
{%- set _kva_ok_61 = true -%}
{%- endif -%}
{%- if kva_alpha|string == '0.5' -%}
{%- set _kva_tok_61 = '<|box_end|>' -%}
{%- set _kva_ok_61 = true -%}
{%- endif -%}
{%- if kva_alpha|string == '1.0' -%}
{%- set _kva_tok_61 = '<|object_ref_start|>' -%}
{%- set _kva_ok_61 = true -%}
{%- endif -%}
{%- endif -%}
{%- set _kva_tok_62 = '<|box_end|>' -%}
{%- set _kva_ok_62 = true -%}
{%- if kva_tail is defined -%}
{%- set _kva_tok_62 = '' -%}
{%- set _kva_ok_62 = false -%}
{%- if kva_tail|string == '1024' -%}
{%- set _kva_tok_62 = '<|box_start|>' -%}
{%- set _kva_ok_62 = true -%}
{%- endif -%}
{%- if kva_tail|string == '2048' -%}
{%- set _kva_tok_62 = '<|box_end|>' -%}
{%- set _kva_ok_62 = true -%}
{%- endif -%}
{%- if kva_tail|string == '2560' -%}
{%- set _kva_tok_62 = '<|object_ref_start|>' -%}
{%- set _kva_ok_62 = true -%}
{%- endif -%}
{%- if kva_tail|string == '3072' -%}
{%- set _kva_tok_62 = '<|object_ref_end|>' -%}
{%- set _kva_ok_62 = true -%}
{%- endif -%}
{%- endif -%}
{%- if kva is defined and kva == 'on' and _kva_ok_60 and _kva_ok_61 and _kva_ok_62 -%}{{- '<|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|><|quad_start|><|quad_end|>' ~ _kva_tok_60 ~ _kva_tok_61 ~ _kva_tok_62 ~ '<|endoftext|>' -}}{%- endif %}{#- kva-marker-end v1 #}
```

## Why requests without the kwarg render byte-identically

- Every tag in the snippet uses whitespace-control dashes (`{%- … -%}`), so
  the block emits no output, not even a newline, when it does not fire.
- The block's last tag is the end comment `{#- kva-marker-end v1 #}` with **no**
  trailing strip dash, and `merge` appends the base bytes directly after it:
  nothing before, between, or after the block is consumed, so a base with
  leading whitespace or plain text is preserved too.
- The snippet only ever assigns `_kva_*`-prefixed variables, which no known
  template uses, so it cannot change the base template's logic.

## minja compatibility choices

The engine renders chat templates with minja (llama.cpp's Jinja subset). The
snippet deliberately uses only:

- `{#- -#}` comments, `{%- if %}`/`{%- endif %}`, `{%- set %}`
- `is defined`, `==`, `and`
- the `|string` filter
- string `~` concatenation, string literals only (no dict literals, no
  subscripting, no loops, no macros, no other filters)

Explicitly avoided: `in` on dict literals (a static `==` comparison per dial
value instead), inline `x if c else y` expressions, dict lookups, `default()`.
Every construct except the comments already appears in the container's own
chat template, which the engine renders today; comments are the one construct
verified only against the minja documentation (see *Limits*).

## Tests — `tests/test_kva_template.py`

Run with the venv python (has `jinja2`, `transformers`, `pytest`):

```sh
/var/home/dylan/projects/research/kva/.venv/bin/python -m pytest tests/test_kva_template.py -q
```

Covered:
- merge output == snippet + base, byte-exact, for **both** base templates
  (container `container-chat-template.jinja`, operator `sharp-v22-dylan.jinja`);
  `strip(merge(x)) == x` for both.
- rendering the merged template with jinja2 — system+user, user-only,
  user+assistant+user, and `add_generation_prompt=True`, with **no** kva
  kwarg — is byte-identical to rendering the base for both templates.
- with `kva="on"`: output = marker string + base output, and the marker
  tokenized with the served model's tokenizer (special tokens parsed,
  `add_special_tokens=False`, no BOS) is exactly 64 ids equal to the spec's
  id at each offset (0–59 pattern, 60–62 selected dial tokens, 63 end).
- every dial value maps to its token; defaults when absent; unknown dial
  value → no marker; `kva="off"` and other values → no marker; numeric dial
  values resolve through `|string`.
- malformed specs refused with a named reason, one test per class: length ≠ 64,
  overlapping offsets, missing offsets, unknown token referenced, dial default
  not in its table, invalid JSON; merging twice refused.

Stubs for jinja2 rendering: both base templates call `raise_exception(...)`,
so tests provide `raise_exception` (raises `ValueError`); `strftime_now` is
also stubbed for future templates, though neither base uses it. Nothing else
is needed — both bases render under jinja2 3.1.6 defaults.

Latest result:

```
51 passed in 3.01s
```

## Engine-side identity check — `tools/template_identity.py`

The offline renders above are jinja2; the engine renders with minja, so the
R119/R120 check that counts runs through a live radiance server's `/tokenize`
(which renders `messages` with the same template the chat endpoints use and
tokenises what it produced: parse_special=true, add_special_tokens=false —
`core/server/admin.cpp`). Standard library only:

```sh
$V tools/template_identity.py suite    --out suite.json            # 66 cases
$V tools/template_identity.py render  --server http://STOCK:8000 --suite suite.json --out stock-ids.json
$V tools/template_identity.py render  --server http://MERGED:8000 --suite suite.json --out merged-ids.json
$V tools/template_identity.py diff    --stock stock-ids.json --merged merged-ids.json \
                                     --spec kva-marker-spec.json     # exit 0/1, per-case table
$V tools/template_identity.py replyfmt --stock-log stock.log --merged-log merged.log
```

The suite is 11 shapes x 6 kva variants. Shapes: system+user; user only;
user/assistant/user; tools + an assistant tool_call + a tool result;
`chat_template_kwargs` enable_thinking true/false; `thinking_budget` 2048;
reasoning_effort low/xhigh; `add_generation_prompt` false; a long user message
typing `<|quad_start|>`/`<|quad_end|>`/`<|im_start|>` as data. Variants: plain
(no kva kwarg), `kva:on`, `kva:on` + each dial (`kva_share "0.50"`,
`kva_alpha "0.5"`, `kva_tail "2560"`), and `kva:on` with the invalid dial
`kva_share "0.75"` (which keeps the request off).

`diff` re-derives every case's expectation from the spec (switch, dial
tables, defaults, token ids, resolved as the snippet resolves them: `|string`
after the JSON value arrives, so the number 0.5 does not match the key
"0.50"), cross-checks it against the expectation the suite declares, and
requires: no-kva and invalid-dial cases byte-identical to stock; `kva:on`
cases exactly the 64 marker ids in front of stock's ids. `replyfmt` compares
the engines' startup `chat: reply format template: ...` entries (R120).

Field spellings were read off the engine, not the guide's chat sections
(see the tool docstring for the details): `/tokenize` merges
`chat_template_kwargs` into the template context but never reads the
request's top-level `reasoning_effort` (parse_thinking runs on the chat
endpoints only), so the effort cases use the kwargs spelling; the template's
own high effort is `xhigh` (it refuses `high` with a 400 naming the legal
set, which leaves no ids to compare); `thinking_budget` is a sampler knob
that never reaches the template; and the kwargs' `enable_thinking` bool is
overwritten with the server's own (`chat-auto-parser-helpers.cpp`), so on a
default server both enable_thinking cases render with thinking on — identity
is what those cases check.

## Tests — `tests/test_template_identity.py`

Run with the venv python (needs `jinja2`, `transformers`, `pytest`):

```sh
/var/home/dylan/projects/research/kva/.venv/bin/python -m pytest tests/test_template_identity.py -q
```

Covered: the suite's 66 cases (shapes, variant kwargs merging, spellings,
determinism); the pure diff rules on hand-made id lists (pass cases, missing
marker, marker gained where none was expected, wrong dial token, numeric dial
resolution, declared-vs-derived mismatch, mismatched id files); `replyfmt`
against logs with the real startup line's bytes; and an end-to-end run
(suite → render → diff) against a FAKE radiance server whose /tokenize renders
via jinja2 and tokenises with the shipped model's tokenizer at
`/var/home/dylan/models-boot/tcc-qwen38-flash-next-mxfp4-fp8-gptq/` (skipped
when absent, naming the path), including a broken-marker-gate server that
`diff` must catch.

Latest result:

```
29 passed in 4.59s
```

## Limits / not verified offline

- **minja itself was never run** on this machine (no engine or llama.cpp checkout
  served here; the fake radiance in the tests renders with jinja2). The live R119/R120
  check — the one that counts — is now driven by `tools/template_identity.py`
  (see above): run `suite`, `render` against the stock and the merged server, and
  `diff` + `replyfmt` on a machine that can reach them.
- Token identity was verified with the tokenizer at
  `/var/home/dylan/models-boot/tcc-qwen38-flash-next-mxfp4-fp8-gptq/` only;
  the engine's own tokenizer is stated to be the same.
- jinja2 default environment settings are used in tests (`keep_trailing_newline`
  false, no trim/lstrip blocks). Byte-identity compares base and merged under
  identical settings; if the engine's settings differ, both sides shift the
  same way, but that combination was not exercised.