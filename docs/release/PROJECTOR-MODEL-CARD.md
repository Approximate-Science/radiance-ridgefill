---
license: apache-2.0
base_model: StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe
tags:
  - radiance
  - kva
  - ridgefill
  - prefill-acceleration
  - qwen3.8-flash-next
---

# KVA Projector for Qwen3.8-Flash-Next

This repository provides the auxiliary projection maps, recurrent terminal state corrections, and token frequency tables required by the Radiance KVA (RidgeFill) plugin (`qwen4exp_kva`) to accelerate long-prompt prefill on `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`.

## What This Folder Is (and Is Not)

- **Not a Standalone Model:** This repository does not contain language model weights, token embeddings, or an execution graph. It cannot generate text on its own, nor can it be loaded by standard inference frameworks like Hugging Face `transformers` or `vLLM`.
- **An Auxiliary Plugin Package:** This folder contains specialized linear projection tensors and recurrent state correction matrices fitted to the late-layer representations of Qwen3.8-Flash-Next. It operates exclusively when loaded by the Radiance KVA architecture plugin (`qwen4exp_kva`) alongside an untouched stock model container.

## Files and Directory Structure

The repository contains 29 files totaling approximately 1.32 GB (1,318,066,885 bytes), defined by `kva.json`:

| File(s) | Format & Dtype | Size | Description |
|---|---|---|---|
| `kva.json` | JSON (Format 1) | ~30 KB | Repository manifest containing model architecture fingerprints (`qwen4exp`, dimensions, layer roles), dial token tables, fit metadata, and SHA256 hashes of all files. |
| `proj.L24.safetensors` … `proj.L47.safetensors` (24 files) | SafeTensors (BF16) | ~52.4 MB each | Late-layer projection maps for layers 24 through 47. Each file contains `proj.L.weight` (`[2560, 10240]`) and `proj.L.bias` (`[2560]`). |
| `correction.safetensors` | SafeTensors (FP32) | ~56.6 MB | Recurrent GDN terminal state correction tensors `st.L` (`[48, 128, 128]`) across 18 recurrent late layers (layers 24–26, 28–30, 32–34, 36–38, 40–42, and 44–46). |
| `rowsel.safetensors` | SafeTensors (FP32) | ~3.0 MB | Vocabulary frequency scoring tables (`score`, `score_none`, `score_all` of shape `[248320]`) used by quality mode to identify high-information class tokens. |
| `chat_template.jinja` | Jinja2 Template | ~11.8 KB | Chat template containing the 64-token KVA marker generator and dial parsers merged in front of the model's base chat template. |
| `README.md` | Markdown | ~1 KB | Directory reference and quick-start instructions. |

## How the Projector Was Fitted

The shipped projector (the one in the folder) is `kva-big-s24`: centred ridge regression with an unpenalised bias, with regularization $\lambda = 0.03$ chosen by held-out mean block-input cosine. It was fitted on 425,789 rows (248,923 raw + 176,866 chat-formatted; chat weighted to a 0.5 share) from 151 raw documents + 102 chat copies at row stride 8, captured on a different engine (tcclaviger's vLLM build serving an mxfp4/fp8 quantisation of the same model); held-out block-input cosine 0.7525.

The shipped GDN correction was fitted as the mean terminal-state error over 13 prompts, with 62 chunk ends per layer per rank.

A refit on radiance's own activations (151 docs, 498,204 weighted rows; 14 prompts for the correction) raised held-out cosine by +0.0035 but did not change quality (paired NLL difference within noise), so the shipped fit is kept.

## Measured Quality and Speed

### Measurement Protocol

All metrics were gathered under the following controlled evaluation protocol:
- **Environment:** Radiance engine release `1.0.8`, execution image `stilldeadcode/radiance:1.0.8`, boot ID `75e3e39b-cc5b-49de-8087-a4791f372a92`.
- **Hardware:** AMD `gfx1201` GPUs (2 cards, tensor parallelism `--tp 2`, `--gpu-headroom-mib 3072`).
- **Scoring Window:** Paired per-document evaluation on the last 512 tokens across the 9 `quick9` perplexity benchmark documents (lengths 8,192, 16,384, and 32,768 tokens) with bootstrap 95% confidence intervals.
- **Timing:** Time-to-first-token (TTFT) medians collected on warmed servers across repeated runs (`RK_REPS=7`, discarding initial warm-up).

### Quality Results

| Mode / Configuration | Scoring Window | ΔNLL vs Exact [95% CI] | Perplexity Ratio | Top-1 Match | Mean KL Divergence |
|---|---|---|---|---|---|
| **Stock Exact** (Baseline) | Last 512 tokens | 0.00000 | 1.0000 | 100.0% | 1.15e-7 |
| **Quality Mode** ($T=2560$, default) | Last 512 tokens | **+0.00212** [-0.01255, +0.01565] | **1.0021** | **91.56%** | **0.0368** |
| **Quality Mode** ($T=2048$) | Last 512 tokens | +0.00576 [-0.01307, +0.02245] | 1.0058 | 89.50% | 0.0592 |
| **Speed Mode** ($T=2048$, lean) | Last 512 tokens | +0.02382 [+0.00292, +0.04388] | 1.0241 | 88.74% | 0.0672 |

In quality mode ($T=2560$), the 95% confidence interval for ΔNLL includes zero, showing no measurable difference from stock on the measured set (9 long documents, last 512 tokens, paired; ΔNLL +0.0021, 95% CI [−0.0126, +0.0157]).

### Prefill Speedup (Warmed Server TTFT)

| Prompt Length | Stock Exact (ms) | Quality Mode ($T=2048$) | Speedup (Quality) | Speed Mode ($T=2048$) | Speedup (Speed) |
|---|---|---|---|---|---|
| **9,216 tokens** | 5,662 ms | 4,738–4,773 ms | **1.19x** (-907 ms) | 4,034–4,038 ms | **1.40x** (-1,626 ms) |
| **16,384 tokens** | 9,742–9,751 ms | 6,797–6,816 ms | **1.43x** (-2,940 ms) | 5,244–5,245 ms | **1.86x** (-4,501 ms) |
| **32,768 tokens** | 19,045–19,058 ms | 10,385–10,404 ms | **1.83x** (-8,661 ms) | 8,958–8,966 ms | **2.13x** (-10,090 ms) |

### Concurrency (Decoders Present)

Under mixed traffic (long-prompt prefill concurrent with active decoders), speed mode retains 67–89% of its solo speedup and remains faster than stock in every measured case (1.34–2.30x speedup). Active decoder text generation is byte-identical to running with KVA off (`notes/stageb.md`, R54/R55).

## Limitations

- **Model Compatibility:** This projector is fitted specifically for `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`. It cannot be used with other model architectures or differing layer layouts.
- **Short Prompts:** Prompts shorter than $T + \text{chunk size}$ (~2,560 tokens) cannot be approximated and run at stock exact speed.
- **Bulk Row Logits:** Logits produced on approximated bulk rows do not reflect the exact model distribution. For perplexity scoring, score only the exact tail (the last $T$ tokens) or run with `RADIANCE_KVA_SCORE_BULK=1` for tail-only evaluation.
- **Resource Footprint & Placement Default:** Projector placement default between VRAM and host is `TBD (author decision)`. Loading projector maps into GPU VRAM requires ~1.22 GiB per card (~970 fewer resident expert slots), which causes chunks that run exact to be slower than stock (16K all-exact 11.6 s vs 9.6 s) and short prompts to run at 0.85–0.94x of stock (`TBD (Stage E measurement, int8 halves it)`). Alternatively, host-mapped placement eliminates this VRAM footprint at the cost of +6% to +10% prefill execution time.
- **Prefix Cache Branching:** A conversation that is edited or branched to a shorter continuation can get part of its exact tail from approximated cache; the plugin counts these events, while plain follow-up turns and regenerations are unaffected.
- **Multimodal Inputs:** Steps carrying image or vision embeddings run stock exact compute.

## Attribution

TBD by the author
