# KVA Chat Template Integration

This package provides the chat-template integration tooling for the Radiance KVA (RidgeFill) plugin (`qwen4exp_kva`). It enables per-request opt-in and dynamic parameter tuning using a standard Jinja2 chat-template snippet.

## Why a Snippet?

In production, operators often maintain customized chat templates featuring custom system prompts, tool-calling schemas, reasoning tags, or tailored whitespace conventions. Replacing an operator's template with a monolithic file would overwrite these customizations.

Instead, the KVA template integration is delivered as a modular Jinja snippet that is prepended to *any* existing base template:
- **Zero Inactive Footprint:** When KVA is not requested, every tag in the snippet evaluates to empty text. The rendered output is byte-identical to rendering your base template alone, introducing no extra spaces or newlines.
- **Variable Isolation:** All internal variables use the `_kva_*` prefix, preventing collisions with base template logic.
- **Engine-Safe Syntax:** The snippet is restricted to minja-compatible Jinja constructs (simple `if`/`endif`, equality comparisons, string filters, and concatenation) supported natively by Radiance.

## Merging with Your Chat Template

The standalone tool `tools/kva_template.py` handles merging, verification, and stripping using Python's standard library alone.

### 1. Merge the Snippet into Your Base Template

Prepend the KVA marker block to your existing template:

```sh
python3 tools/kva_template.py merge \
  --spec <model dir>/projector/kva.json \
  --base /path/to/my-custom-template.jinja \
  --out /path/to/my-custom-template.kva.jinja
```

*Note:* `merge` is idempotent; it refuses to re-merge if a KVA marker comment (`{#- kva-marker v1 ... -#}`) is already present.

### 2. Verify or Revert When Needed

- **Verify a merged template byte-for-byte:**
  ```sh
  python3 tools/kva_template.py check \
    --spec <model dir>/projector/kva.json \
    --base /path/to/my-custom-template.jinja \
    --merged /path/to/my-custom-template.kva.jinja
  ```
- **Strip the marker block to restore the pristine base template:**
  ```sh
  python3 tools/kva_template.py strip \
    --merged /path/to/my-custom-template.kva.jinja \
    --out /path/to/restored-template.jinja
  ```
- **Inspect the raw snippet alone:**
  ```sh
  python3 tools/kva_template.py snippet --spec <model dir>/projector/kva.json
  ```

## Passing the Template to Radiance

To serve your merged template, pass the `--override-chat-template` flag at server startup:

```sh
radiance \
  --model /path/to/models/qwen3.8-next-flash-fp8-iq4r-moe.rad \
  --override-chat-template /path/to/my-custom-template.kva.jinja \
  ...
```

## How a Request Opts In

> **Note:** Per-request opt-in via chat template kwargs is **coming in the next release** (Stage F). When enabled, clients opt in per request as described below.

Clients opt in per request by including `chat_template_kwargs` in the `/v1/chat/completions` API payload:

```json
{
  "model": "qwen3.8-next-flash",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Analyze the attached long technical document..."}
  ],
  "chat_template_kwargs": {
    "kva": "on"
  }
}
```

When `kva: "on"` is present, the template emits a 64-token special marker at position 0 of the prompt. The KVA plugin detects this marker, decodes any requested dials, erases the marker tokens on the GPU, and accelerates the prefill pass.

## Template Dials

Clients can fine-tune approximation trade-offs on a per-request basis by supplying optional dials inside `chat_template_kwargs`:

| Dial Argument | Allowed Values | Default Value | Token Mapped (Offset) | Function & Description |
|---|---|---|---|---|
| `kva_share` | `"0.10"`, `"0.25"`, `"0.50"` | `"0.25"` | Offset 60:<br>`"0.10"` → `<|box_start|>`<br>`"0.25"` → `<|box_end|>`<br>`"0.50"` → `<|object_ref_start|>` | **Class Retention Share:** Fraction of high-frequency class tokens kept exact in the approximation window. Higher values retain more exact compute. |
| `kva_alpha` | `"0"`, `"0.5"`, `"1.0"` | `"1.0"` | Offset 61:<br>`"0"` → `<|box_start|>`<br>`"0.5"` → `<|box_end|>`<br>`"1.0"` → `<|object_ref_start|>` | **Correction Strength:** Scaling factor for the recurrent GDN terminal state correction. |
| `kva_tail` | `"1024"`, `"2048"`, `"2560"`, `"3072"` | `"2048"` | Offset 62:<br>`"1024"` → `<|box_start|>`<br>`"2048"` → `<|box_end|>`<br>`"2560"` → `<|object_ref_start|>`<br>`"3072"` → `<|object_ref_end|>` | **Exact Tail Length ($T$):** Trailing prompt tokens computed with stock exact inference. Must satisfy $T < 2 \times \text{step} - 64$. |

### Dial Formatting and Fallbacks

- Values are converted using `|string`, allowing either string literals (`"2048"`) or JSON numbers (`2048`).
- If any specified dial contains an unrecognized value (e.g. `kva_share: "0.40"`), the template suppresses the entire marker block. The request automatically executes as stock exact inference.

## Default Stock Behavior (Without `kva: "on"`)

When a request omits `kva`, sets `kva: "off"`, or provides invalid dial options:
- The snippet renders nothing.
- The prompt matches the base template byte for byte.
- The server processes the request with standard, stock exact inference.

## Important Caveat: Deployment Guard

> **WARNING: Do not serve a merged chat template on a Radiance instance that lacks the KVA plugin.**

If an operator deploys a merged template to a standard Radiance engine *without* loading the KVA plugin:
1. Requests with `"kva": "on"` will render the 64-token control marker into the prompt.
2. Because the standard engine lacks the KVA plugin, the 64 marker tokens will **not** be erased on the GPU.
3. The language model will ingest the raw special control tokens (`<|quad_start|>`, `<|quad_end|>`, etc.) directly as part of the context, leading to corrupted context and severe output quality degradation.

Ensure that the KVA plugin is properly installed on `$RADIANCE_HOME` and confirmed active in the startup logs before serving merged templates with per-request opt-in enabled.
