# Radiance RidgeFill Plugin

The Radiance RidgeFill plugin (`qwen4exp_ridgefill`) speeds up long-prompt prefill for
Qwen3.8-Flash-Next. It approximates layers 24 through 47 for bulk prompt tokens
using a fitted projector, and computes the last `T` prompt tokens exactly
(`T` = `RADIANCE_RIDGEFILL_TAIL`, default 2048). The mode is server-wide, set once at
startup. The shipped projector is INT8 only.

By Dylan Johnston and tcclaviger, Apache-2.0 (see `NOTICE`). Cite with DOI
[10.5281/zenodo.23179168](https://doi.org/10.5281/zenodo.23179168); BibTeX and the prior work
RidgeFill builds on are at the end.

## Requirements

- **Radiance engine:** release `1.0.13`. The plugin checks the engine binary at
  startup and forwards to the engine's own architecture (RidgeFill off) or refuses to
  start on a mismatch; see Troubleshooting.
- **Base model:** the stock published container
  `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe` (LFS SHA256
  `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20`,
  121,969,901,568 bytes). The container is never modified.
- **GPUs:** two AMD `gfx1201` cards, `--tp 2` (the tested configuration).
  TP1 is expected to work (the engine declares whole weights with no
  collectives; the static test oracle covers TP1). TP4 is untested.
- **License:** Apache-2.0 for the plugin and the projector.

## Installation

1. **Keep the stock model file intact.** Serve the published `.rad` file as is.
2. **Download the plugin** into a plugin directory (it needs only these two files):
   ```sh
   hf download Dyluhn/radiance-ridgefill --include "architectures/*" "kernels/*" --local-dir <plugin dir>
   ```
   The same repo's `release/` folder holds the release tarballs and their `SHA256SUMS`.
3. **Download the projector** into a `projector/` folder beside the model file:
   ```sh
   hf download Dyluhn/ridgefill-projector-qwen3.8-flash-next-i8 --local-dir <model dir>/projector
   ```
   To keep it elsewhere, set `RADIANCE_RIDGEFILL_PROJECTOR=<dir>`, which is checked first.
4. **Put the plugin first on `RADIANCE_HOME`:**
   ```sh
   export RADIANCE_HOME=<plugin dir>:/opt/radiance/share/radiance
   ```
5. **Docker: mount the model's directory** (e.g. `-v /data/models:/models`),
   not just the single `.rad` file, so the plugin can find `projector/`
   beside the model.
6. **Start with a mode** (e.g. `RADIANCE_RIDGEFILL=quality`) and check the startup
   log for the `RidgeFill: projector … matches …` line. If it is missing, RidgeFill is not
   active; see below.

## Modes

The mode is server-wide only, read once from the environment at startup
(`arch/ridgefill_config.h`):

- `off` (default): stock execution. Nothing is declared and no projector VRAM
  is held, even if a projector folder is present.
- `quality`: approximates bulk tokens, keeps selected class tokens exact
  (default share 0.25), and applies the fitted terminal-state correction.
- `speed`: lean fill with tail-only straddling on bulk tokens; skips late-layer
  attention and MoE compute until the exact tail.
- `plumb`: diagnostics. Runs stock computation through the plugin pipeline
  without approximation. Reads no fitted tensor.

Per-request ON/OFF (a `ridgefill` chat-template kwarg) is parked for a future update
(`notes/future/per-request.md`); this release has no per-request switch.

## Switches

Every switch is read once at startup (`arch/ridgefill_config.h`). Values and
defaults:

| Switch | Values | Default |
|---|---|---|
| `RADIANCE_RIDGEFILL` | `off\|plumb\|speed\|quality` | `off` |
| `RADIANCE_RIDGEFILL_TAIL` | token count, at least 512 | `2048` |
| `RADIANCE_RIDGEFILL_ALPHA` | correction strength in [0, 1] | `1` |
| `RADIANCE_RIDGEFILL_ROWSEL` | `class\|random\|all` | `class` |
| `RADIANCE_RIDGEFILL_SHARE` | share of class rows kept exact, in (0, 1] | `0.25` |
| `RADIANCE_RIDGEFILL_STAGE` | `auto\|stock` (expert stager lever) | `auto` |
| `RADIANCE_RIDGEFILL_STAGE_ROWS` | row count | unlimited |
| `RADIANCE_RIDGEFILL_MIN_BULK_ROWS` | a pass approximates only past this many bulk rows | `1024` |
| `RADIANCE_RIDGEFILL_CKPT_FLOOR` | exact rows kept at a prefix-cache checkpoint | `0` |
| `RADIANCE_RIDGEFILL_FINAL` | `on\|off` (MTP final map, when held and MTP is on) | `off` |
| `RADIANCE_RIDGEFILL_SCORE_BULK` | `1` (approximate in KL scoring mode; then score only the exact tail) | unset |
| `RADIANCE_RIDGEFILL_PROJECTOR` | projector directory | `<model dir>/projector` |
| `RADIANCE_RIDGEFILL_ROWSEL_TABLE` | `class\|none\|all` | `class` |

Debug-only switches (each changes what an approximate pass computes):
`RADIANCE_RIDGEFILL_STRADDLE` (`split\|end`, default `split`),
`RADIANCE_RIDGEFILL_FORCE_SPLIT`, `RADIANCE_RIDGEFILL_SHIFT_B`,
`RADIANCE_RIDGEFILL_FORCE_STREAM`, `RADIANCE_RIDGEFILL_TAIL_ONLY`,
`RADIANCE_RIDGEFILL_MASK` (`all`, refused in `plumb`).

Startup refuses a tail above `2 × step − tile` (`step` =
`--max-num-batched-tokens`), because no chunk could ever be approximated past
it, and refuses a tail below 512.

## What it costs

Measured on radiance 1.0.13 with its own `deploy/compose/flashnext.yaml` profile (MTP 3, prefix cache on, 8 sequences, 2,048-token steps; `--gpu-headroom-mib 3072` because the display ran on one card), int8 projector, two AMD R9700 (gfx1201), `--tp 2`. Speed: time to first token, warmed and settled servers, median of 7, two rounds within 0.3% of each other. Quality: last 512 tokens of 9 long documents, paired against exact, bootstrap 95% CI.

| Dimension | Effect / Cost |
|---|---|
| Quality mode speedup | **1.49x** at 16K tokens (12.26 s → 8.22 s), **2.10x** at 32K (24.06 s → 11.46 s) |
| Quality mode quality | ΔNLL +0.0010 per token (95% CI −0.0135 to +0.0153): no measurable difference from exact; perplexity ratio 1.0010, top-1 agreement 91.3% |
| Speed mode speedup | **1.97x** at 16K (12.26 s → 6.24 s), **2.55x** at 32K (24.06 s → 9.45 s) |
| Speed mode quality | ΔNLL +0.0242 per token (95% CI +0.0040 to +0.0446): a small measurable cost; perplexity ratio 1.0245, top-1 agreement 89.4% |
| Retrieval (needle in a haystack) | 72/72 for stock, quality and speed: one fact (or the right one of four) hidden at 10–85% depth of a 16K or 32K chat prompt, plus exact-tail controls; no disagreements with stock |
| Decode | Equal to stock, with MTP on: 8.87 vs 9.37 ms/token right after a 16K prompt, 8.18 vs 7.99 settled (stock vs quality) |
| Prompts that run the stock path on a RidgeFill server | ≈ +0.9% (2K) / +1.2% (8K) settled prefill time vs stock, because the plugin's VRAM displaces resident experts. Short prompts and the exact tail are unaffected in output, only in time. Decode equals stock. |
| Projector memory | The projector always lives in host RAM (≈637 MiB host-mapped per rank for int8) and is streamed through ONE VRAM staging slot of 25.4 MiB per rank. Total plugin VRAM ≈ 61 MiB per card (slot + arena buffers). There is no VRAM placement. |
| MTP drafting (`--num-speculative-tokens`, default auto) | Works with the plugin; decode is unaffected. The MTP `final` map is off by default and optional: +1.8% drafted tokens per step (paired, measured on radiance 1.0.8) for about +105 MiB VRAM per rank (computed from the plugin's declarations). See `docs/MTP-FINAL-MAP.md` in the plugin repo. |
| Concurrency | Quality mode keeps co-batched decoders byte-identical to off. Speed mode's decoders-beside-prefill path does not promise byte identity; its decoders stay within stock's own solo-vs-batched variation. |
| Prefix cache | Works like stock. A branched or edited conversation can resume from a checkpoint whose cached positions were approximated; the plugin counts these and logs `ridgefill: hazard N positions (total M)`. `RADIANCE_RIDGEFILL_CKPT_FLOOR` (default 0) trades cache reuse for exactness. |
| Images | Steps carrying image rows run the stock path; approximation resumes on later text steps. |

## Is RidgeFill active

The ONLY signal is the startup log. Either the projector matched:

```
radiance: qwen4exp_ridgefill: RidgeFill: projector <dir> (found <how>) matches <model>: arch ok, <checks>, <N> warning(s); split <S>, <correction held|no correction>, <MiB> MiB in <files> files; RidgeFill projector <name> by <authors> (<license>, doi:<doi>)
```

(`arch/ridgefill_projector.h`; `<how>` is `$RADIANCE_RIDGEFILL_PROJECTOR=<dir>`,
`beside --model <path> …`, or `beside the resolved model file <path>`. The
shipped folder reports split 24 with correction held, and its credit from
`ridgefill.json`: `RidgeFill projector ridgefill-projector-qwen3.8-flash-next-i8 by
Dylan Johnston and tcclaviger (Apache-2.0, doi:10.5281/zenodo.23179168)`.) Each rank then logs
what it holds, e.g.:

```
radiance: qwen4exp_ridgefill: RidgeFill: rank 0 holds the projector: <H> MiB host-mapped (int8 maps, correction, row table), 25.4 MiB VRAM (the staging ring's one slot)
```

(`plumb` instead logs `RidgeFill: rank <n> holds nothing (plumb reads no fitted
tensor)`.) Or the projector was refused and the server runs stock:

```
radiance: qwen4exp_ridgefill: RidgeFill: no projector folder (looked at <places>); serving stock
radiance: qwen4exp_ridgefill: RidgeFill: projector <dir> (found <how>) REFUSED, it cannot run on this model: <reason> [<checks>]; serving stock
```

If neither a `matches` line nor a `REFUSED … serving stock` line is present,
RidgeFill did not load; check `RADIANCE_HOME` order.

## Troubleshooting

| Log line | Meaning |
|---|---|
| `RidgeFill: no projector folder (looked at …); serving stock` | No `ridgefill.json` at `$RADIANCE_RIDGEFILL_PROJECTOR` or `<model dir>/projector`. Stock serving. Check the path or Docker mount. |
| `RidgeFill: projector … REFUSED, it cannot run on this model: …; serving stock` | Dimension, layer, tokenizer, or tensor mismatch with this model. Stock serving. Use the projector fitted for the model. |
| `RidgeFill: WARNING: projector …: … -- it runs, but was fitted on another variant` | Quantization or weight-anchor difference only. RidgeFill still runs. |
| `WARNING: built against radiance 1.0.13, and the engine … carries release string(s) …; forwarding to the engine's own architecture …, RidgeFill off` | Engine is not release 1.0.13. Stock serving through the engine's in-tree architecture. |
| `built against radiance 1.0.13, and the engine …; no in-tree architectures/… on $RADIANCE_HOME to forward to, so this plugin declines …` | Engine mismatch and no in-tree architecture behind the plugin on `RADIANCE_HOME`. Startup fails. Fix `RADIANCE_HOME` order (a home given only as `--radiance-home` is invisible to plugins). |
| `ridgefill.tail is … and the largest step is …` | Tail above `2 × step − tile`. Startup refused. Lower `RADIANCE_RIDGEFILL_TAIL` or raise `--max-num-batched-tokens` (or set `--checkpoint-interval` below it). |
| `ridgefill.tail is …; the shortest exact tail this method was measured at is …` | Tail below 512. Startup refused. |
| `RADIANCE_RIDGEFILL_PROJ_PLACE=… is retired: the projector is always streamed from host memory through the staging ring …` | Retired placement switch set. Startup refused. Unset `RADIANCE_RIDGEFILL_PROJ_PLACE` and `RADIANCE_RIDGEFILL_PROJ_RING`. |
| `RADIANCE_RIDGEFILL_PROJ=… is retired with the container append: …` (likewise `_ST`, `_DECLARE`) | Retired container-append switch set. Startup refused. Unset it; tensors come from the projector folder. |
| `ridgefill: hazard <n> positions (total <m>)` | Informational: a branched conversation resumed `<n>` approximated positions from prefix cache. See above. |

## Compatibility

- Missing or incompatible projector: the server runs stock (see the `serving
  stock` lines above). A tokenizer or geometry mismatch refuses by name;
  quantization or anchor differences warn and run.
- Engine other than 1.0.13: forwarded to the in-tree architecture (RidgeFill off) or
  refused at startup, never silently approximated. Forwarding needs the engine's own
  in-tree `qwen4exp_fp8.so` on `$RADIANCE_HOME` behind this plugin's home; a home given
  only as `--radiance-home` is invisible to plugins, so the start then fails by name.
  Each plugin build targets one radiance release; a build for a newer release is made
  with `scripts/update_radiance.sh <tag>` (see the repository README).
- Radiance's own MTP drafting works with the plugin; the plugin's optional
  `final` map only applies when the folder holds it and MTP is on.
- Prefix cache behaves like stock; see the hazard counter above.

## For model authors

Porting RidgeFill to another model means writing one adapter against the model-free
core; the projector folder needs no schema code. The contract, with qwen4exp
as the worked example, is `docs/ADDING-A-MODEL.md`.

## Credit and prior work

RidgeFill is by **Dylan Johnston** (author) and **tcclaviger** (co-author), released under
Apache-2.0. Cite it with DOI [10.5281/zenodo.23179168](https://doi.org/10.5281/zenodo.23179168)
(BibTeX below).

The idea that a long prompt's late-layer caches need not come from running every late layer in
full on every prompt token is not new:

- **DeepSeek-V4.1-Flash's Causal Encoder-Decoder** (DeepSeek technical report,
  [arXiv:2609.19969](https://arxiv.org/abs/2609.19969), §2.2) builds the decoder's global KV cache
  from the encoder's outputs, so most prompt tokens skip the full decoder computation in prefill.
- **YOCO**, "You Only Cache Once" (Sun et al., 2024, [arXiv:2405.05254](https://arxiv.org/abs/2405.05254)):
  the self-decoder's KV cache serves the cross-decoder, so prefill can exit early.
- **kishida's [Q3-8B-KVA-Projector](https://huggingface.co/kishida/Q3-8B-KVA-Projector)**
  ("Late Layer KV Approximation Projector for Qwen3-8B") is the first retrofit of the idea onto an
  existing model. "KVA" is that work's name; this project used it only as a working name.

What RidgeFill adds:

- a **closed-form ridge fit**, with no gradient training;
- maps to the late layers' **inputs**, not to their K/V, so each late layer computes its own K/V
  and its own recurrent state;
- an **exact tail**: the last `T` prompt tokens always run every layer exactly;
- **exact-row selection**: quality mode keeps a selected share of bulk rows exact;
- the **recurrent-state correction** for the gated delta net's state;
- the **radiance plugin**: an architecture plugin and a kernel library that serve the stock
  published model, unmodified.

## Citation

```bibtex
@software{johnston_ridgefill_2026,
  author  = {Johnston, Dylan and {tcclaviger}},
  title   = {RidgeFill},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  doi     = {10.5281/zenodo.23179168},
  url     = {https://doi.org/10.5281/zenodo.23179168}
}
```
