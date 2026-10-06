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

- **Radiance engine:** release `1.1.1` (commit 7001841). The plugin checks the engine binary at
  startup and forwards to the engine's own architecture (RidgeFill off) or refuses to
  start on a mismatch; see Troubleshooting.
- **Base model:** the stock published container
  `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe` (LFS SHA256
  `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20`,
  121,969,901,568 bytes). The container is never modified.
- **GPUs:** AMD `gfx1201` (Radeon AI PRO R9700). By tensor-parallel size:

  | `--tp` | status |
  |---|---|
  | 2 | **tested**: every number in this README |
  | 1 | **tested for correctness** on one 32 GB card (radiance 1.1.1): `off` is byte-identical to stock and int8 quality matches TP2's (last-512 ΔNLL +0.0028 vs +0.0010). The routed experts that do not fit on the card (~44 GiB) live in pinned host RAM: `--host-pool-mib 40960 --ngram-placement disk` (the n-gram table read from the model file so RAM goes to experts). That pool must fit both in free RAM and under the driver's pinned-memory cap (`ttm.pages_limit`, half of RAM by default). Measured there (release profile otherwise): 32K prefill 67.8 s stock, 57.8 s quality (1.17x), 41.0 s speed (1.66x); requests that do not approximate +0.3% vs stock; decode 13.5 tok/s, quality 0.8% below stock. |
  | 3 | **built and tested without the hardware**: radiance 1.1.1 serves three ranks (uneven delta-net heads, one rank without attention) and the plugin follows it -- the static test oracle and the GPU kernels at TP3's 18- and 15-head ranks pass -- but no three-card run has been made |
  | 4 | untested |
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

## How much faster

Measured on two AMD R9700 (gfx1201), `--tp 2`, radiance 1.0.13 with its own `deploy/compose/flashnext.yaml` profile (expert_tiered offload, MTP 3, prefix cache on, 8 sequences, 2,048-token steps; `--gpu-headroom-mib 3072` because the display ran on one card), int8 projector. Stock is the same server with `RADIANCE_RIDGEFILL=off`. Prefill = the engine's prompt time.

- **Typical use: a 32K-token prompt prefills 1.24x faster in quality mode and 1.70x faster in speed mode.** This is a fresh server, or long prompts mixed with chat (measured at 1 long prompt to 6 short chats; 95% CI 1.21–1.27x and 1.68–1.74x). A fresh server's first long prompts get 1.07–1.13x and 1.47–1.56x.
- **Sustained long-prompt traffic: 2.05x (quality) and 2.81x (speed) at 32K, 1.85x and 2.33x at 16K**, once the server has served about 15–18 long prompts back to back with nothing in between.
- **Short prompts and decode: no loss beyond ~1%.** Decode on a fresh server is within 0.5% of stock; after long-prompt traffic it is 3–4% faster than stock. Prompts too short to approximate (under about 2K tokens) take the stock path and prefill 0.4–1.0% slower than stock (see "What it costs").

| 32K-token prompt, prefill vs stock | quality | speed |
|---|---|---|
| fresh server, or long prompts mixed with chat | 1.24x | 1.70x |
| after ~15–18 long prompts back to back | 2.05x | 2.81x |

Why it depends on the traffic: with expert_tiered offload, radiance moves VRAM expert slots toward the layers a server's requests use. RidgeFill's long prompts use few experts in layers 24–47, so the cache shifts toward layers 0–23, at most 8 moves per layer and slot class per request; that takes ~15–18 long prompts. Short chats and decode use every layer and pull the cache back within about 10 requests. If all experts fit in VRAM there is no mover and this ramp should not apply, but that has not been measured.

On BetterBench's prefill sweep (fresh servers, 1.6K / 5.9K / 11.8K / 23.6K / 47K real tokens): quality 0.97 / 1.19 / 1.28 / 1.38 / 1.45x, speed 0.97 / 1.08 / 1.27 / 1.48 / 1.50x. At 1.6K nothing is approximated; those packages cost 3% there, two thirds of it an extra prefill stage that 0.1.0-r3 removed (now +0.9%).

Earlier copies of this README gave 2.10x / 2.55x at 32K. Those were measured partway through the ramp above and hold only for sustained long-prompt traffic.

## What it costs

Same hardware and profile as above. Quality: last 512 tokens of 9 long documents, paired against exact, bootstrap 95% CI.

| Dimension | Effect / Cost |
|---|---|
| Quality mode quality | ΔNLL +0.0010 per token (95% CI −0.0135 to +0.0153): no measurable difference from exact; perplexity ratio 1.0010, top-1 agreement 91.3% |
| Speed mode quality | ΔNLL +0.0242 per token (95% CI +0.0040 to +0.0446): a small measurable cost; perplexity ratio 1.0245, top-1 agreement 89.4% |
| Retrieval (needle in a haystack) | 72/72 for stock, quality and speed: one fact (or the right one of four) hidden at 10–85% depth of a 16K or 32K chat prompt, plus exact-tail controls; no disagreements with stock |
| Decode | No loss, with MTP on. Short chats paired with stock: −0.3% (quality) / −0.5% (speed) on a fresh server; +3.3–3.9% after long-prompt traffic and under mixed traffic; 8 concurrent chats within noise |
| Prompts that run the stock path on a RidgeFill server | +0.4–1.0% prefill time vs stock at 512–2,000 tokens, and +0.8% on prefix-cache hits (paired, fresh servers, 2026-10-06), because the plugin's VRAM displaces about 47 resident expert slots a card. A stock step of more than 1,024 tokens that directly follows an approximated one can still stage one extra expert layer, because radiance's prefill stager reuses the previous pass's reach. Output is unaffected, only time. |
| Projector memory | The projector always lives in host RAM (≈637 MiB host-mapped per rank for int8) and is streamed through ONE VRAM staging slot of 25.4 MiB per rank. Total plugin VRAM ≈ 70 MiB per card, measured as the claimable drop on card 0 (the 25.4 MiB slot, ~3 MiB of kernel code and 43 MiB of activation-arena buffers). There is no VRAM placement. |
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

Run the report on the serving machine and read its first line; it names the problem and the fix:

```sh
python3 ridgefill_report.py --container <your radiance container>   # or --log <server stderr file>
```

It ships in this folder, needs only Python 3.8+, changes nothing, and checks this plugin's and the projector's
files against their hashes (`--projector <dir>`). Every message RidgeFill can print has a code and a fix in
`TROUBLESHOOTING.md` (also in this folder). To report a problem, paste the whole report into the issue.

## Compatibility

- Missing or incompatible projector: the server runs stock (see the `serving
  stock` lines above). A tokenizer or geometry mismatch refuses by name;
  quantization or anchor differences warn and run.
- Engine other than 1.1.1: forwarded to the in-tree architecture (RidgeFill off) or
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
