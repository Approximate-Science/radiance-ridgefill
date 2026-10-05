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

# KVA Projector for Qwen3.8-Flash-Next (INT8)

Auxiliary projection maps, recurrent terminal-state corrections, and token
frequency tables for the Radiance KVA plugin (`qwen4exp_kva`) on
`StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`. The plugin is Apache-2.0; this
projector package is Apache-2.0.

## What this folder is (and is not)

- **INT8 only.** The maps are int8 codes plus bf16 scales. The bf16 projector
  is only the int8 builder's input (`tools/kva_projector.py int8 --from
  <bf16 folder> --out <dir>`); it is not shipped and no release test covers
  bf16 serving.
- **Not a standalone model.** No language model weights, embeddings, or
  execution graph. It cannot generate text alone and cannot be loaded by
  `transformers` or `vLLM`. It runs only when loaded by the plugin alongside
  the untouched stock model container.

## Files

28 files, 698,753,818 bytes, listed with SHA256 hashes in `kva.json`
(`projector.dtype: "i8"`, layout `i8_row128`, encoding `i8*bf16[1x128]`).
The package directory is named `projector-qwen3.8-flash-next-i8` by
`tools/package.py` after that dtype field. `kva.json` SHA256:
`5b699e27e88d2e27cb546c174c6cb6f257e20433555e96b37fe17781aa3a85ae`.

| File(s) | Contents | Size |
|---|---|---|
| `kva.json` | Manifest (format 1): architecture fingerprints, fit metadata, and SHA256 of every file | 30,468 bytes |
| `proj8.L24.safetensors` … `proj8.L47.safetensors` (24 files) | Late-layer maps for layers 24–47. Each: `proj.L.codes` int8 `[2560, 10240]`, `proj.L.scale` bf16 `[2560, 80]`, `proj.L.bias` bf16 `[2560]` | 26,629,416 bytes each |
| `correction.safetensors` | Recurrent GDN terminal-state corrections `st.L` f32 `[48, 128, 128]` for 18 layers (24–26, 28–30, 32–34, 36–38, 40–42, 44–46) | 56,624,584 bytes |
| `rowsel.safetensors` | Vocabulary frequency tables `score`, `score_none`, `score_all`, f32 `[248320]` | 2,980,120 bytes |
| `chat_template.jinja` | Parked per-request marker template (not used by this release) | 11,805 bytes |
| `README.md` | Directory reference | 857 bytes |

## How the projector was fitted

The shipped maps are `kva-big-s24`: centered ridge regression with an
unpenalized bias, regularization λ = 0.03 chosen by held-out mean block-input
cosine. Fitted on 425,789 rows (248,923 raw + 176,866 chat-formatted; chat
weighted to a 0.5 share) from 151 raw documents + 102 chat copies at row
stride 8, captured on a different engine (tcclaviger's vLLM build serving an
mxfp4/fp8 quantization of the same model); held-out block-input cosine 0.7525.

The shipped GDN correction was fitted as the mean terminal-state error over 13
prompts, with 62 chunk ends per layer per rank.

A refit on radiance's own activations (151 docs, 498,204 weighted rows; 14
prompts for the correction) raised held-out cosine by +0.0035 but did not
change quality (paired NLL difference within noise), so the shipped fit is
kept.

The int8 conversion (`tools/kva_projector.py int8`): per-128 absmax int8 with
bf16 scale — each row is stored as int8 codes at absmax/127 per 128 columns
with a bf16 scale, codes rounded half-to-even against the rounded scale.
Worst |w − dequant| is 0.39% of a map's max |w| across the 24 maps.

## Quality

Measured on radiance 1.0.8, two gfx1201 cards, `--tp 2` (TP1 expected to work,
TP4 untested), paired per document vs stock exact.

Paired int8 − bf16 (same boot):

| Configuration | Paired ΔNLL int8 − bf16 [95% CI] |
|---|---|
| Quality, T 2560, last 512 | −0.00109 [−0.00400, +0.00215] |
| Quality, T 2048, whole tail | +0.00101 [−0.00111, +0.00336] |
| Quality, T 2048, last 512 | +0.00169 [−0.00146, +0.00493] |
| Speed, T 2048, last 512 | +0.00041 [−0.00450, +0.00527] |

Every paired CI includes 0: int8 holds the bf16 projector's level. Headline
quality vs stock exact (quality mode, last 512, paired): int8 +0.00103
[−0.01348, +0.01529]; bf16 +0.00212 [−0.01255, +0.01565].

Headline quality on the release build is re-measured by the final release
session: «FINAL: quality-mode ΔNLL vs stock exact with 95% CI, protocol».

## Speed

Int8 halves the streamed bytes (0.639 GB vs 1.259 GB per pass per rank) and
the ring slot (25.4 MiB vs 50 MiB), and is 2.5–4.7% faster on TTFT than bf16
through the ring. Headline TTFT on the release build is re-measured by the
final release session: «FINAL: TTFT speedup vs stock at 16K/32K tokens,
warmed-server medians, protocol».

## Limitations

- Only for `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`. Geometry,
  tokenizer, or tensor mismatches are refused by name (server runs stock);
  quantization or anchor differences warn and run.
- Prompts shorter than T + chunk run stock exact.
- Logits on approximated bulk rows are not the model's; score only the exact
  tail, or run with `RADIANCE_KVA_SCORE_BULK=1` for tail-only evaluation.
- Steps carrying image rows run stock exact; approximation resumes after.
- The folder's `final` MTP map variant, if present, is used only when
  `RADIANCE_KVA_FINAL` is not `off` and MTP drafting is on; default is off.
