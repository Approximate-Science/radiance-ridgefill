# Radiance KVA Plugin

The Radiance KVA plugin (`qwen4exp_kva`) speeds up long-prompt prefill for
Qwen3.8-Flash-Next. It approximates layers 24 through 47 for bulk prompt tokens
using a fitted projector, and computes the last `T` prompt tokens exactly
(`T` = `RADIANCE_KVA_TAIL`, default 2048). The mode is server-wide, set once at
startup. The shipped projector is INT8 only.

## Requirements

- **Radiance engine:** release `1.0.13`. The plugin checks the engine binary at
  startup and forwards to the engine's own architecture (KVA off) or refuses to
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
2. **Unpack the plugin.** Put `architectures/qwen4exp_fp8.so` and
   `kernels/kva.so` in a plugin directory.
3. **Unpack the projector.** Copy the projector package
   `projector-qwen3.8-flash-next-i8` (named by `tools/package.py` after the
   dtype in the folder manifest's `projector.dtype`) into a `projector/`
   subdirectory beside the model file (e.g. `<model dir>/projector/`), so it
   holds `kva.json` plus the map files. To point elsewhere, set
   `RADIANCE_KVA_PROJECTOR=<dir>`, which is checked first.
4. **Put the plugin first on `RADIANCE_HOME`:**
   ```sh
   export RADIANCE_HOME=<plugin dir>:/opt/radiance/share/radiance
   ```
5. **Docker: mount the model's directory** (e.g. `-v /data/models:/models`),
   not just the single `.rad` file, so the plugin can find `projector/`
   beside the model.
6. **Start with a mode** (e.g. `RADIANCE_KVA=quality`) and check the startup
   log for the `KVA: projector … matches …` line. If it is missing, KVA is not
   active; see below.

## Modes

The mode is server-wide only, read once from the environment at startup
(`arch/kva_config.h`):

- `off` (default): stock execution. Nothing is declared and no projector VRAM
  is held, even if a projector folder is present.
- `quality`: approximates bulk tokens, keeps selected class tokens exact
  (default share 0.25), and applies the fitted terminal-state correction.
- `speed`: lean fill with tail-only straddling on bulk tokens; skips late-layer
  attention and MoE compute until the exact tail.
- `plumb`: diagnostics. Runs stock computation through the plugin pipeline
  without approximation. Reads no fitted tensor.

Per-request ON/OFF (a `kva` chat-template kwarg) is parked for a future update
(`notes/future/per-request.md`); this release has no per-request switch.

## Switches

Every switch is read once at startup (`arch/kva_config.h`). Values and
defaults:

| Switch | Values | Default |
|---|---|---|
| `RADIANCE_KVA` | `off\|plumb\|speed\|quality` | `off` |
| `RADIANCE_KVA_TAIL` | token count, at least 512 | `2048` |
| `RADIANCE_KVA_ALPHA` | correction strength in [0, 1] | `1` |
| `RADIANCE_KVA_ROWSEL` | `class\|random\|all` | `class` |
| `RADIANCE_KVA_SHARE` | share of class rows kept exact, in (0, 1] | `0.25` |
| `RADIANCE_KVA_STAGE` | `auto\|stock` (expert stager lever) | `auto` |
| `RADIANCE_KVA_STAGE_ROWS` | row count | unlimited |
| `RADIANCE_KVA_MIN_BULK_ROWS` | a pass approximates only past this many bulk rows | `1024` |
| `RADIANCE_KVA_CKPT_FLOOR` | exact rows kept at a prefix-cache checkpoint | `0` |
| `RADIANCE_KVA_FINAL` | `on\|off` (MTP final map, when held and MTP is on) | `off` |
| `RADIANCE_KVA_SCORE_BULK` | `1` (approximate in KL scoring mode; then score only the exact tail) | unset |
| `RADIANCE_KVA_PROJECTOR` | projector directory | `<model dir>/projector` |
| `RADIANCE_KVA_ROWSEL_TABLE` | `class\|none\|all` | `class` |

Debug-only switches (each changes what an approximate pass computes):
`RADIANCE_KVA_STRADDLE` (`split\|end`, default `split`),
`RADIANCE_KVA_FORCE_SPLIT`, `RADIANCE_KVA_SHIFT_B`,
`RADIANCE_KVA_FORCE_STREAM`, `RADIANCE_KVA_TAIL_ONLY`,
`RADIANCE_KVA_MASK` (`all`, refused in `plumb`).

Startup refuses a tail above `2 × step − tile` (`step` =
`--max-num-batched-tokens`), because no chunk could ever be approximated past
it, and refuses a tail below 512.

## What it costs

Headline numbers are re-measured on the release build; placeholders below are
filled in by the final release session.

| Dimension | Effect / Cost |
|---|---|
| Quality mode speedup (default flags) | «FINAL: TTFT speedup vs stock at 9K/16K/32K tokens, warmed-server medians» |
| Quality mode quality (default flags) | «FINAL: ΔNLL vs stock, last-512 paired, with 95% CI» |
| Speed mode speedup (default flags) | «FINAL: TTFT speedup vs stock at 9K/16K/32K tokens, warmed-server medians» |
| Speed mode quality (default flags) | «FINAL: ΔNLL vs stock, last-512 paired, with 95% CI» |
| Prompts that run the stock path on a KVA server | ≈ +0.9% (2K) / +1.2% (8K) settled prefill time vs stock, because the plugin's VRAM displaces resident experts. Short prompts and the exact tail are unaffected in output, only in time. Decode equals stock. |
| Projector memory | The projector always lives in host RAM (≈637 MiB host-mapped per rank for int8) and is streamed through ONE VRAM staging slot of 25.4 MiB per rank. Total plugin VRAM ≈ 61 MiB per card (slot + arena buffers). There is no VRAM placement. |
| MTP drafting (`--num-speculative-tokens`, default auto) | Works with the plugin; decode is unaffected. The MTP `final` map is off by default and optional: +1.8% drafted tokens per step (paired) for +25 MiB VRAM per rank. |
| Concurrency | Quality mode keeps co-batched decoders byte-identical to off. Speed mode's decoders-beside-prefill path does not promise byte identity; its decoders stay within stock's own solo-vs-batched variation. |
| Prefix cache | Works like stock. A branched or edited conversation can resume from a checkpoint whose cached positions were approximated; the plugin counts these and logs `kva: hazard N positions (total M)`. `RADIANCE_KVA_CKPT_FLOOR` (default 0) trades cache reuse for exactness. |
| Images | Steps carrying image rows run the stock path; approximation resumes on later text steps. |

## Is KVA active

The ONLY signal is the startup log. Either the projector matched:

```
radiance: qwen4exp_kva: KVA: projector <dir> (found <how>) matches <model>: arch ok, <checks>, <N> warning(s); split <S>, <correction held|no correction>, <MiB> MiB in <files> files
```

(`arch/kva_projector.h`; `<how>` is `$RADIANCE_KVA_PROJECTOR=<dir>`,
`beside --model <path> …`, or `beside the resolved model file <path>`. The
shipped folder reports split 24 with correction held.) Each rank then logs
what it holds, e.g.:

```
radiance: qwen4exp_kva: KVA: rank 0 holds the projector: <H> MiB host-mapped (int8 maps, correction, row table), 25.4 MiB VRAM (the staging ring's one slot)
```

(`plumb` instead logs `KVA: rank <n> holds nothing (plumb reads no fitted
tensor)`.) Or the projector was refused and the server runs stock:

```
radiance: qwen4exp_kva: KVA: no projector folder (looked at <places>); serving stock
radiance: qwen4exp_kva: KVA: projector <dir> (found <how>) REFUSED, it cannot run on this model: <reason> [<checks>]; serving stock
```

If neither a `matches` line nor a `REFUSED … serving stock` line is present,
KVA did not load; check `RADIANCE_HOME` order.

## Troubleshooting

| Log line | Meaning |
|---|---|
| `KVA: no projector folder (looked at …); serving stock` | No `kva.json` at `$RADIANCE_KVA_PROJECTOR` or `<model dir>/projector`. Stock serving. Check the path or Docker mount. |
| `KVA: projector … REFUSED, it cannot run on this model: …; serving stock` | Dimension, layer, tokenizer, or tensor mismatch with this model. Stock serving. Use the projector fitted for the model. |
| `KVA: WARNING: projector …: … -- it runs, but was fitted on another variant` | Quantization or weight-anchor difference only. KVA still runs. |
| `WARNING: built against radiance 1.0.13, and the engine … carries release string(s) …; forwarding to the engine's own architecture …, KVA off` | Engine is not release 1.0.13. Stock serving through the engine's in-tree architecture. |
| `built against radiance 1.0.13, and the engine …; no in-tree architectures/… on $RADIANCE_HOME to forward to, so this plugin declines …` | Engine mismatch and no in-tree architecture behind the plugin on `RADIANCE_HOME`. Startup fails. Fix `RADIANCE_HOME` order (a home given only as `--radiance-home` is invisible to plugins). |
| `kva.tail is … and the largest step is …` | Tail above `2 × step − tile`. Startup refused. Lower `RADIANCE_KVA_TAIL` or raise `--max-num-batched-tokens` (or set `--checkpoint-interval` below it). |
| `kva.tail is …; the shortest exact tail this method was measured at is …` | Tail below 512. Startup refused. |
| `RADIANCE_KVA_PROJ_PLACE=… is retired: the projector is always streamed from host memory through the staging ring …` | Retired placement switch set. Startup refused. Unset `RADIANCE_KVA_PROJ_PLACE` and `RADIANCE_KVA_PROJ_RING`. |
| `RADIANCE_KVA_PROJ=… is retired with the container append: …` (likewise `_ST`, `_DECLARE`) | Retired container-append switch set. Startup refused. Unset it; tensors come from the projector folder. |
| `kva: hazard <n> positions (total <m>)` | Informational: a branched conversation resumed `<n>` approximated positions from prefix cache. See above. |

## Compatibility

- Missing or incompatible projector: the server runs stock (see the `serving
  stock` lines above). A tokenizer or geometry mismatch refuses by name;
  quantization or anchor differences warn and run.
- Engine other than 1.0.13: forwarded to the in-tree architecture (KVA off) or
  refused at startup, never silently approximated. Forwarding needs the engine's own
  in-tree `qwen4exp_fp8.so` on `$RADIANCE_HOME` behind this plugin's home; a home given
  only as `--radiance-home` is invisible to plugins, so the start then fails by name.
  Each plugin build targets one radiance release; a build for a newer release is made
  with `scripts/update_radiance.sh <tag>` (see the repository README).
- Radiance's own MTP drafting works with the plugin; the plugin's optional
  `final` map only applies when the folder holds it and MTP is on.
- Prefix cache behaves like stock; see the hazard counter above.

## For model authors

Porting KVA to another model means writing one adapter against the model-free
core; the projector folder needs no schema code. The contract, with qwen4exp
as the worked example, is `docs/ADDING-A-MODEL.md`.
