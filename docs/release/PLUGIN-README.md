# Radiance KVA Plugin

The Radiance KVA (RidgeFill) plugin accelerates long-prompt prefill for Qwen3.8-Flash-Next by approximating late layers (layers 24 through 47) for all but the final ~2,048 prompt tokens — using a closed-form ridge fit that fills the late-layer caches from layer 24's residual stream. The plugin is disabled by default, and its quality mode shows no measurable difference from stock on the measured set (9 long documents, last 512 tokens, paired; ΔNLL +0.0021, 95% CI [−0.0126, +0.0157]).

## Requirements

Before installing the plugin, ensure your deployment matches the verified environment:
- **Radiance Engine:** Radiance release `1.0.8`. The plugin binary includes an engine release guard that verifies the engine binary at startup.
- **Base Model:** Stock published model `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe` (LFS SHA256 `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20`, 121,969,901,568 bytes). The container remains unmodified—no weights or metadata are appended.
- **GPU Architecture:** AMD `gfx1201` GPUs.
- **GPU Topology:** Two GPUs with tensor parallelism (`--tp 2`), matching the recorded measurement configuration.

## Installation

Follow these steps to set up the plugin alongside your existing Radiance server:

1. **Keep your stock model file intact:** Retain your unmodified `qwen3.8-next-flash-fp8-iq4r-moe.rad` container as published.
2. **Download the plugin binaries:** Obtain `<plugin tarball>` and extract it to `<plugin dir>`. The directory structure provides `architectures/qwen4exp_fp8.so` and `kernels/kva.so`.
3. **Download the projector directory:** Download the projector folder directly into a `projector/` subdirectory beside your model file (e.g., `hf download <org>/<projector repo> --local-dir <model dir>/projector`).
4. **Set `RADIANCE_HOME`:** Prepend the plugin path to `RADIANCE_HOME` using the runtime image's path so the engine shadows the in-tree architecture:
   ```sh
   export RADIANCE_HOME=<plugin dir>:/opt/radiance/share/radiance
   ```
5. **(Optional) Configure chat template:** If you plan to use per-request opt-in once available, supply `--override-chat-template <model dir>/projector/chat_template.jinja` (or your merged template from `tools/kva_template.py`).
6. **Mount directory in Docker:** If running inside Docker, mount the model's containing *directory* (e.g. `-v /data/models:/models`), rather than mounting only the single `.rad` file. The plugin discovers `projector/` relative to the model directory.
7. **Start Radiance with your selected mode:** Launch the engine with your desired operating mode (e.g., `RADIANCE_KVA=quality`). Verify in the startup log that the projector was matched with zero warnings:
   ```
   KVA: projector /models/projector matches qwen4exp: arch ok, metadata 11/11, tokenizer ok, encodings 489/489, anchors 3/3, 0 warning(s)
   ```

## Operating Modes

### Server-Wide Modes

The operating mode is configured at engine startup via the `RADIANCE_KVA` environment variable:

- `off` (default): Stock Radiance execution, byte for byte. Even if a projector folder is present, the plugin declares the in-tree graph and allocates no auxiliary VRAM.
- `quality`: Recommended production acceleration. Approximates late layers for bulk prompt tokens, retains exact computation for selected class tokens (default share 0.25), and applies recurrent GDN terminal state corrections. Prefill quality shows no measurable difference from stock on the measured set (9 long documents, last 512 tokens, paired; ΔNLL +0.0021, 95% CI [−0.0126, +0.0157]).
- `speed`: Maximum prefill throughput. Executes lean fill and tail-only straddling across bulk prompt tokens, bypassing late-layer attention, connections, and MoE compute until the exact tail.
- `plumb`: Verification and diagnostics mode. Routes stock computation through the masked pipeline without approximation to verify numerical identity with stock kernels.

### Per-Request On/Off

> **Note:** Per-request opt-in via chat template kwargs is **coming in the next release** (Stage F). In the current release, modes are selected server-wide.

When enabled in the upcoming release:
- Requests opt in individually by passing `"chat_template_kwargs": {"kva": "on"}` in the OpenAI-compatible `/v1/chat/completions` API payload.
- Requests without `"kva": "on"` run stock exact inference.
- Operators can tune behavior per request using template dials:
  - `kva_share`: Share of class tokens kept exact (`"0.10"`, `"0.25"`, `"0.50"`; default `"0.25"`).
  - `kva_alpha`: GDN state correction strength (`"0"`, `"0.5"`, `"1.0"`; default `"1.0"`).
  - `kva_tail`: Exact tail length in tokens (`"1024"`, `"2048"`, `"2560"`, `"3072"`; default `"2048"`).

## What It Costs

The table below lists the trade-offs and resource costs across operating modes and placement configurations. Note that projector placement default is `TBD (author decision)` between `vram` and `host`:

| Dimension | Option / Setting | Measured Effect / Cost | Reference & Context |
|---|---|---|---|
| **Quality vs Speed** | `RADIANCE_KVA=quality` (T=2048, default guard) | **TTFT Speedup:** 1.19x at 9,216 tokens (-907 ms), 1.43x at 16,384 tokens (-2,940 ms), 1.83x at 32,768 tokens (-8,661 ms).<br>**Quality:** ΔNLL +0.00212 [-0.01255, +0.01565] at T=2560; ppl ratio 1.0021; top-1 91.56% (last 512 tokens). | Warmed server medians; 95% CI [−0.0126, +0.0157] includes 0, showing no measurable difference from stock. |
| **Quality vs Speed** | `RADIANCE_KVA=speed` (T=2048) | **TTFT Speedup:** 1.40x at 9,216 tokens (-1,626 ms), 1.86x at 16,384 tokens (-4,501 ms), 2.13x at 32,768 tokens (-10,090 ms).<br>**Quality:** ΔNLL +0.02382 on last 512 tokens (+0.04432 whole tail). | Fastest prefill; tail-only straddle saves 788 ms over exact chunk. |
| **Projector Placement** | `RADIANCE_KVA_PROJ_PLACE=vram`<br>*(default: TBD (author decision))* | **VRAM:** ~1.22 GiB per GPU card (~1,228 MiB held per rank). Reduces resident expert slab slots by ~970 slots per rank (-6.5% to -6.7%, 16,883 down to 15,745 slots).<br>**Prefill TTFT:** Fastest ON prefill (9,657 ms at 16K; 13,536 ms at 32K). | Best for prefill-heavy workloads where expert cache eviction is acceptable. Default choice is TBD (author decision). |
| **Projector Placement** | `RADIANCE_KVA_PROJ_PLACE=host`<br>*(default: TBD (author decision))* | **VRAM:** 128 MiB per rank (recurrent correction, row table, plus 2 × 50 MiB VRAM staging slots). Recovers ~970 expert slots per rank (+6.2% resident experts, up to 16,713 slots).<br>**Prefill TTFT:** +6% vs VRAM at 16K (10,230 ms; level with stock exact), +10% vs VRAM at 32K (14,942 ms; 25% faster than stock exact). | Uses the DD-L staging ring (lane 1 staging copy). Avoids card memory pressure. Default choice is TBD (author decision). |
| **VRAM Placement Side Effect** | Projector in VRAM (`RADIANCE_KVA_PROJ_PLACE=vram`) | On a KVA server with the projector in VRAM, chunks that run exact are slower than stock (16K all-exact 11.6 s vs 9.6 s) and short prompts run at 0.85–0.94x of stock due to fewer resident expert slots (staging buffer growth). | TBD (Stage E measurement, int8 halves it). |
| **Concurrency (decoders present)** | Mixed traffic (prefill beside active decoders) | Speed mode keeps 67–89% of its solo speedup, still faster than stock in every measured case (1.34–2.30x faster than stock). Decoder text is byte-identical to KVA-off. | Measured in Stage B (`notes/stageb.md`, R54/R55). |
| **Host Staging Ring** | `RADIANCE_KVA_PROJ_RING=0` (zero-copy gate) | **Prefill TTFT:** 2.7x slower than stock exact (27,096 ms at 16K; 54,104 ms at 32K) due to per-M-tile PCIe re-reads. | Do not disable the staging ring (`PROJ_RING=1` default). |
| **Exact Tail vs Chunk Size** | `RADIANCE_KVA_TAIL=2048` (default) | Exact tail must satisfy $T \le 2 \times \text{step} - 64$. At chunk size 2048, $T < 4,032$. Startup refuses if exceeded. Prompts shorter than $T + \text{chunk}$ run stock exact. | To use small chunks with large tail, set `--checkpoint-interval` below `--max-num-batched-tokens`. |
| **Multimodal / Vision Steps** | Steps containing media | Steps with image/media inputs run stock exact compute (0 approximate steps). | Derive logic detects media rows and disables approximation for that step. |
| **Prefix Cache & Branching** | Prefix cache interactions | Prefix cache storage capacity is unaffected (no per-token KV groups added). In per-request mode, the 64-token marker prepended to ON requests means ON and OFF requests do not share prefix cache trees; two ON requests branch normally. A conversation that is edited/branched to a shorter continuation can get part of its exact tail from approximated cache; the plugin counts these events; plain follow-up turns and regenerations are unaffected. | Cache snapshots record sequence state cleanly. |

## Troubleshooting

The table below explains common startup messages, refusals, and warnings:

| Log Message Pattern | Cause | Engine Behavior & Recommended Action |
|---|---|---|
| `KVA: no projector folder (looked at ...); serving stock` | No projector directory found at `$RADIANCE_KVA_PROJECTOR` or `<model dir>/projector`. | Engine serves stock inference cleanly. Verify projector path or Docker volume mount. |
| `REFUSED, it cannot run on this model: metadata '<key>' is '<x>' in this model and '<y>' in the projector's ...; serving stock` | Geometric mismatch (dimensions, layer count, `hc_count`, layer roles) between projector manifest and model. | Refused by name. Engine falls back to stock inference. Use a projector fitted for this model. |
| `REFUSED ... the tokenizer differs (vocab sha256 <x> here, <y> in the projector's model); serving stock` | Vocabulary hash mismatch. Row selection tables require exact token ID mapping. | Refused by name. Engine falls back to stock inference. Check that model tokenizer matches projector tokenizer. |
| `REFUSED: the manifest lists <file>, which is missing or unreadable in <dir>; serving stock` | Corrupted download, missing safetensors file, or failed SHA256 checksum. | Refused by name. Engine falls back to stock inference. Re-download projector folder. |
| `WARNING: tensor <name> hashes <x> here and <y> in the model the projector was fitted on ... -- it runs, but was fitted on another variant` | Base tensor anchor mismatch, differing quantization format, or fine-tuned model variant. | Logged as a warning (naming both sides); KVA **runs**. Safe to proceed if intended. |
| `WARNING: built against radiance 1.0.8, and the engine ... carries release string(s) ...; forwarding to the engine's own architecture ...` | Engine binary release does not match `1.0.8`. | Release guard forwards execution to the engine's in-tree `qwen4exp_fp8.so` (stock exact, KVA off). If in-tree plugin is absent, startup fails by name. |
| `radiance: qwen4exp_kva: kva.tail is ... and the largest step is ...; none is once the tail exceeds ...` | Configured tail $T$ exceeds maximum allowable tokens for current chunk size ($2 \times \text{step} - 64$). | Engine refuses startup. Lower `RADIANCE_KVA_TAIL` or increase batch token step. |
| `radiance: qwen4exp_kva: <switch> is retired with the container append...` | Deprecated switch set (`RADIANCE_KVA_PROJ`, `RADIANCE_KVA_ST`, `RADIANCE_KVA_DECLARE`). | Engine refuses startup. Unset deprecated environment variables. |

## Compatibility Guarantees

The plugin maintains strict fallbacks to ensure existing setups remain operational:
1. **Missing Projector:** If no projector folder is present, the engine automatically serves stock Radiance without modification.
2. **Incompatible Projector:** Any structural, dimension, or tokenizer mismatch causes the projector to be refused by name, immediately falling back to stock inference.
3. **Model Variant Mismatches:** Differences in quantization encoding or weight anchors emit clear warnings naming both variants, but continue running KVA.
4. **Engine Version Guard:** If the Radiance binary differs from release `1.0.8`, the plugin automatically forwards execution to the stock in-tree architecture on `$RADIANCE_HOME`, keeping the server functional in stock mode.
