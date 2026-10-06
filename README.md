# radiance-ridgefill

RidgeFill prefill for radiance's Qwen3.8-Flash-Next (`qwen4exp`), as plugins: an architecture
plugin that shadows the in-tree `qwen4exp_fp8.so` (plugin name `qwen4exp_ridgefill`), the `ridgefill.so` kernel
library, and `tools/ridgefill_projector.py`, which builds the projector folder the plugin loads from
`<model dir>/projector/`. The model file stays exactly as published (the old container append is retired:
`tools/dev/README.md`).

The plugin is a model-independent core (`arch/ridgefill_*.h`) and one adapter per model (`arch/qwen4exp_*`);
adding a model is docs/ADDING-A-MODEL.md.

RidgeFill is by Dylan Johnston and tcclaviger (Apache-2.0, `NOTICE`); cite it with DOI
[10.5281/zenodo.23179168](https://doi.org/10.5281/zenodo.23179168) (`CITATION.cff`; BibTeX and the prior work
it builds on at the end of this file).

## What it gains

Measured on two AMD R9700 (gfx1201), `--tp 2`, radiance 1.0.13's flashnext profile (expert_tiered offload,
MTP 3, `--gpu-headroom-mib 3072`), int8 projector, against the same server with `RADIANCE_RIDGEFILL=off`
(notes/ramp.md, notes/mixed-traffic.md):

- **Typical use: a 32K-token prompt prefills 1.24x faster (quality) / 1.70x (speed)** on a fresh server or with
  long prompts mixed with chat (1 long : 6 short). A fresh server's first long prompts: 1.07–1.13x / 1.47–1.56x.
- **Sustained long-prompt traffic: 2.05x / 2.81x at 32K** (1.85x / 2.33x at 16K), after ~15–18 long prompts
  back to back, once radiance's expert cache has shifted toward layers 0–23. Short chats and decode shift it back.
- **Short prompts and decode: no loss** (decode −0.3% / −0.5% on a fresh server, +3.3–3.9% after long-prompt
  traffic).

docs/HOW-IT-WORKS.md "Headline results" has the mechanism and the BetterBench sweep. The 2.10x / 2.55x of the
release session were measured partway through the cache shift and hold only for sustained long-prompt traffic.

## Build

Two inputs: an **installed radiance** and a **source checkout of the same release**. The arch plugin
compiles radiance's `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`, which radiance does not install; configure
refuses a checkout whose release or headers differ from the install, naming both. Building either tree
needs CMake ≥ 3.21 (the `cmake_minimum_required` of both top-level `CMakeLists.txt` files) and, on the
Docker path, Ninja — which, like the compiler, is inside the build image, not on the host.

### In Docker (the path the served plugins come from)

Prerequisites: Docker with BuildKit (`docker buildx`) on the host — nothing else; the build stage
carries its own cmake, ninja and g++-14, so no compiler, Ninja or ROCm install is needed outside it.
The `radiance-build` image is made by radiance's `docker/build.sh`: it streams `git archive` of the
chosen commit as the build context to `docker buildx` and stops at the Dockerfile's `build` stage
(`rocm/dev-ubuntu-24.04` plus cmake, ninja-build and g++-14), which configures with Ninja and installs
radiance under `DESTDIR=/stage` — that staged install at `/stage/opt/radiance` is what the snippet
below builds the plugins against.

1. Build radiance's compiler image once, from the radiance checkout at the release you serve:
   ```sh
   (cd <radiance> && docker/build.sh -r <commit> --target build)     # image: radiance-build
   ```
2. Build, run the no-GPU tests, install into `./home`:
   ```sh
   docker run --rm -v <radiance>:/rsrc:ro -v "$PWD":/ridgefill -w /ridgefill radiance-build bash -c '
     cmake -S /ridgefill -B /ridgefill/build-hip -G Ninja -DCMAKE_PREFIX_PATH=/stage/opt/radiance -DRADIANCE_SRC=/rsrc &&
     cmake --build /ridgefill/build-hip &&
     ctest --test-dir /ridgefill/build-hip -LE gpu --output-on-failure &&
     cmake --install /ridgefill/build-hip'
   ```
   `home/architectures/qwen4exp_fp8.so` and `home/kernels/ridgefill.so` are then a `$RADIANCE_HOME` to put in
   front of the installation's. Serve only plugins built in this image: a host compiler's libstdc++ is
   newer than the runtime image's.
3. Check what the engine will load:
   ```sh
   docker run --rm -v "$PWD/home":/plugins:ro radiance-build \
     /stage/opt/radiance/bin/rad-info --plugins --home /plugins:/stage/opt/radiance/share/radiance
   ```
   Expect the installed `qwen4exp_fp8.so` logged as shadowed, `qwen4exp_ridgefill ... architecture qwen4exp`
   from `/plugins/architectures/qwen4exp_fp8.so`, and `ridgefill` among the kernel libraries.

GPU targets come from the radiance install (`gfx1201` in the default image) or `-DRAD_GPU_TARGETS=...`.

### Host-only (no ROCm, no GPU)

Build radiance host-only outside radiance's source tree, then the plugins against it; the tests need no card:
```sh
cmake -S <radiance> -B build-radiance-host -DRAD_WITH_HIP=OFF -DRAD_WITH_FFMPEG=OFF -DRAD_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/build-radiance-host/install"
cmake --build build-radiance-host -j && cmake --install build-radiance-host
cmake -S . -B build-host -DCMAKE_PREFIX_PATH="$PWD/build-radiance-host/install" -DRADIANCE_SRC=<radiance> \
      -DRAD_WITH_HIP=OFF
cmake --build build-host -j && (cd build-host && ctest -LE gpu --output-on-failure)
```

The Python tier (the `tests/test_*.py` suites and the tools they exercise) needs no build at all:
```sh
pip install -r tools/requirements.txt
python -m pytest tests -q
```
Any test that needs machine-local data is SKIPPED unless you opt in; each skip reason names the env
var to set. The full set (grep `tests/`): `RIDGEFILL_RESEARCH_ROOT` (a checkout of the research repo the
refit imports read-only), `RIDGEFILL_TOKENIZER` (a checkpoint directory), `RIDGEFILL_SIDECAR` (a sidecar
built by `tools/dev/ridgefill_sidecar.py`), `RIDGEFILL_TEST_TOKENIZER` (a tokenizer directory) and
`RIDGEFILL_TEST_OPERATOR_TEMPLATE` (an operator base chat template). Without them the run is green on the
suites that fake their inputs.

`-DRAD_WITH_HIP=OFF` leaves out the device rows even against a HIP install. Every `build*/` and `home/`
directory is git-ignored.

## Updating to a new radiance release

The plugin is built against ONE radiance release (`RADIANCE_VERSION` names it, with its commit): the arch
plugin compiles that release's in-tree Qwen4-Exp source, and the release guard forwards to the engine's own
architecture on any other. To try a new release:
```sh
scripts/update_radiance.sh v1.0.14              # + device build and packages when radiance-build:1.0.14 exists
scripts/update_radiance.sh v1.0.14 --host-only  # no docker: ~15 min the first time, ~1 min after
scripts/update_radiance.sh v1.0.14 --gpu-smoke  # + one queued GPU session, ~15-20 min
```
Exit 0 COMPATIBLE, 1 INCOMPATIBLE, 2 INFRASTRUCTURE (the answer could not be computed -- never reported
as compatible).
It archives the release from the radiance checkout into `data/radiance-src-<release>/` (read-only), builds a
host-only install of it and the plugin against it, runs `ctest -LE gpu` -- the static oracle holds `off` and
every approximate path issue for issue against THAT release's in-tree plugin -- and pytest, and on green
builds the device plugins in `radiance-build:<release>` and the release packages (`tools/package.py`) into
`dist/radiance-ridgefill-r<release>-<commit>/`. It ends `RESULT: PASS` or `RESULT: FAIL` (exit 0 / 1); logs and a
summary in `data/update-<release>/`. A change of RAD_ABI_VERSION is INCOMPATIBLE on its own. `--gpu-smoke` adds one `gpuq` session on `stilldeadcode/radiance:<release>`:
stock and `off` ident (must match), an exact KL reference for the release, int8 quality T2560 and int8
speed T2048 scored last-512 paired vs exact.

Either way it lists which in-tree files the adapter COPIES changed since the pinned release, and which
adapter file copies each (`arch/*.copies`). When the oracle fails, those are the files to port: a compile
error names the line that no longer fits (e.g. on v1.0.8 this branch fails in `arch/qwen4exp_moe.h` because
1.0.10's four-class MoE fields do not exist there), a failing static case prints its first difference.
Port the adapter file, rerun until PASS, then update `RADIANCE_VERSION` and the release README's version.
The 1.0.8 -> 1.0.13 port is the worked example (notes/rebase-1.0.13.md): one file changed.

### CI: radiance release watch

`.github/workflows/radiance-watch.yml` (GitHub Actions; Forgejo / Codeberg Actions runs the same file) runs
daily, on every push and pull request, and by hand with an optional `radiance_tag`. It resolves radiance's
newest release tag (`git ls-remote --tags` on codeberg.org/StillDeadcode/radiance, highest semver), clones
radiance, and runs `scripts/update_radiance.sh <tag> --host-only --ci` on an ordinary ubuntu-24.04 runner:
no GPU, no ROCm, no docker. It installs cmake (>= 3.21), g++-14, binutils and Python 3.12 with
`tools/requirements.txt` (torch from the CPU index). The job fails on INCOMPATIBLE (a build failure, an ABI
number change, any static-oracle mismatch) and on INFRASTRUCTURE; each failing oracle case is an `::error::`
naming the adapter files it points at and the first in-tree op that moved. Every file of radiance's `abi/`,
`arch/common/` and `arch/qwen4exp_fp8/` changed since `RADIANCE_VERSION` is a `::warning::` either way. The
report (`data/update-<release>/`, report.md + every log) is uploaded as an artifact.

## Switches (read once, at startup)

The MODE comes from the environment only, default `off`: a container's `ridgefill.mode` is ignored (and said so
once in the log), and so are any `ridgefill.*` weights an earlier append left in it. The fitted tensors come from
the projector folder: `$RADIANCE_RIDGEFILL_PROJECTOR`, else `projector/` beside the `--model` path as typed, else
beside the file it resolves to (mount the model's DIRECTORY in Docker). A folder that cannot run on the model
(other dimensions, layer layout or tokenizer, a missing or corrupt file) is refused by name and the engine
serves stock; one fitted on another variant of the model (quantisation, recipe, base weights) is a WARNING
naming both, and runs. `arch/ridgefill_config.h` is the full list of startup switches; the capture switches
live outside it -- `RADIANCE_RIDGEFILL_CAPTURE`/`_STATE` are read in `arch/qwen4exp_ridgefill.cpp`,
`RADIANCE_RIDGEFILL_CAPTURE_SPLIT` in `arch/ridgefill_declare_masked.h`, which with `arch/ridgefill_declare.h` declares
what a capture writes.

| switch | values | meaning |
|---|---|---|
| `RADIANCE_RIDGEFILL` | `off` (default), `plumb`, `speed`, `quality` | `off` is stock radiance, byte for byte, even when the container holds ridgefill.* tensors |
| `RADIANCE_RIDGEFILL_TAIL` | tokens, default 2048, ≥ 512, ≤ 2 × `--max-num-batched-tokens` minus the delta net's chunk tile (64 on this model) | the exact tail T. Above one step a full chunk proves only one step ahead, so its last T − step rows (rounded to 64) also run exact (lower coverage); at 2 × step − 64 and beyond no row could ever be approximated, so startup refuses |
| `RADIANCE_RIDGEFILL_ALPHA` | 0..1, default 1 | correction strength |
| `RADIANCE_RIDGEFILL_ROWSEL` | `class`, `random`, `all` | quality mode's exact-row rule |
| `RADIANCE_RIDGEFILL_SHARE` | (0, 1], default 0.25 | share of a window's class matches kept exact |
| `RADIANCE_RIDGEFILL_PROJECTOR` | a directory | the projector folder, ahead of `projector/` beside the model. A folder built by `tools/ridgefill_projector.py int8 --from <bf16 folder> --out <dir>` holds the maps in int8 (the container trunk's own `i8*bf16[1x128]` encoding, 0.65 GiB instead of 1.23 GiB, half the bytes each approximated chunk streams); point this switch at it to use it |
| `RADIANCE_RIDGEFILL_FINAL` | `off` (default), `on` | optional, not in the release: with MTP on (`--num-speculative-tokens` > 0) and a folder holding the `final` map (`tools/ridgefill_projector.py final`), each approximated chunk predicts its bulk rows' final stream, which the drafting head reads. Measured +1.8% drafted tokens a step (R70, notes/staged.md) for a 50 MiB slot instead of 25.4 MiB with int8 maps, +200 MiB host memory and +210 MB a chunk over the link. `off` declares, holds and streams nothing of the map, whatever the folder holds |
| `RADIANCE_RIDGEFILL_ROWSEL_TABLE` | `class`/`none`/`all` | which of the folder's row tables quality mode uses (`none`/`all` are controls) |
| `RADIANCE_RIDGEFILL_STAGE` | `auto` (default), `stock` | the expert-stager lever: `auto` lets the late layers stream only their routed experts on an approximate pass (notes/impl.md §2); `stock` leaves staging as it is |
| `RADIANCE_RIDGEFILL_STAGE_ROWS` | rows | `auto` streams only when the pass has at most this many exact rows, else the pass runs the stock step. Default: no limit -- measured on warmed servers, streaming beats the stock step by 220-240 ms a straddling chunk at 512-1,984 exact rows, and a 64-row limit cost 331 ms at 9,216 and ran every T 2560 chunk exact (notes/impl.md, Stage A.1 R96) |
| `RADIANCE_RIDGEFILL_SCORE_BULK` | `1` | KL mode (`--kld-ref`) serves stock unless set: logits on approximated rows are not the model's, so set it only when scoring the exact tail |
| `RADIANCE_RIDGEFILL_MIN_BULK_ROWS` | rows | a pass approximates only with at least this many bulk rows, else it runs the stock step. Default 1,024 (the projector is always in host memory): every approximate pass streams the whole projector, and the 64-row checkpoint remainders a prompt alternates with beside decoders cost 208-222 ms approximated vs 61-92 ms stock -- decoders riding them got 2.2-3.5x slower (notes/stageb.md session 2c) |
| `RADIANCE_RIDGEFILL_STRADDLE`, `RADIANCE_RIDGEFILL_FORCE_SPLIT`, `RADIANCE_RIDGEFILL_SHIFT_B`, `RADIANCE_RIDGEFILL_FORCE_STREAM`, `RADIANCE_RIDGEFILL_MASK` | `split`/`end`, rows, ±rows, `1`, `all` | gate-only debug switches (R50, R47, R51, R94; `MASK=all` approximates every row before the bulk end, decoders included -- R54's negative control, speed/quality only); each is said loudly at startup |
| `RADIANCE_RIDGEFILL_TAIL_ONLY` | `1` (default), `0` | gate-only debug switch: `0` makes speed-mode straddling chunks -- and speed chunks beside decoders -- take the masked path instead of the tail-only / decoders path (their oracles, A.1 and Stage B) |
| `RADIANCE_RIDGEFILL_DUMP_LOGITS` | a directory | debug only, synchronises: every live pass's logits rows on every rank, each named by sequence/position/token (Stage B gate 1, `tools/logit_compare.py`); run with `--profile-ops` so no pass is replayed |
| `RADIANCE_RIDGEFILL_DUMP` | a directory | debug only, synchronises mid-step: the layer-S stream of every approximate chunk (`boundary.p<P>.npy` + `boundary.jsonl`) and the device mask of every masked one (`mask.jsonl`, `rows.jsonl`) |
| `RADIANCE_RIDGEFILL_CAPTURE` | a directory | refit/debug only: in `off` mode rank 0 records the stock step of every single-sequence prefill -- host copies of the stream entering layer S and of every late layer's block input. Refused at startup unless `RADIANCE_RIDGEFILL=off`: the code says it "records exact runs" and refuses with `RAD_E_INVAL` (`arch/qwen4exp_ridgefill.cpp`) |
| `RADIANCE_RIDGEFILL_CAPTURE_STATE` | a directory | refit/debug only: on every single-sequence prefill chunk, copies whatever late delta-net state the step did not already record, one file per (chunk, rank). Unlike `RADIANCE_RIDGEFILL_CAPTURE`, the code does NOT refuse it with the mode on -- it is accepted in any mode |
| `RADIANCE_RIDGEFILL_CAPTURE_SPLIT` | a layer index | refit/debug only: the split layer S a capture uses when no usable projector folder exists (a capture fits a projector, so there may be no folder yet); declares nothing; without a folder, startup refuses unless it names a layer between the PLE and the last (`arch/ridgefill_declare_masked.h`) |
| `RADIANCE_RIDGEFILL_PROJ`, `RADIANCE_RIDGEFILL_ST`, `RADIANCE_RIDGEFILL_DECLARE` | -- | retired with the container append; refused by name |
| `RADIANCE_RIDGEFILL_PROJ_PLACE`, `RADIANCE_RIDGEFILL_PROJ_RING` | -- | retired: the projector is always streamed from host memory (below); refused by name |

**Where the projector lives:** in host memory, always. Each rank holds the maps, its correction heads and the row
table in one host-mapped block (1.2 GiB bf16, 0.6 GiB int8) and ONE VRAM slot the size of one late layer's map
(50 MiB bf16, 25.4 MiB int8); on every approximated chunk, as soon as layer L's GEMM has read the slot, the second
lane copies layer L+1's map into it while the rest of layer L computes. Every request pays that slot and the
plugin's arena buffers in resident experts (the layer-S stream, kept with bf16 maps or MTP only, and the projected
inputs), so they are kept as small as the pass allows (notes/stagee.md §14).

**What a request that does not use RidgeFill pays (the documented residual, accepted):** with the int8 folder, about 67 MiB
of VRAM a rank -- the ring's 25.4 MiB slot and 41 MiB of plugin activation buffers -- that the engine would otherwise
give to resident experts (47 slab slots a card on the release config). Prompts RidgeFill does not approximate,
paired with stock on fresh servers (radiance 1.0.13 flashnext profile, 0.1.0-r3): **+0.4% to +1.0% prefill time at
512-2,000 tokens, +0.8% on prefix-cache hits** (notes/stockpath-fix.md; before r3, 1,025-2,048 tokens paid +2.5%
for an extra prefill stage, notes/stock-path-cost.md). A stock step of more than 1,024 tokens that directly
follows an approximated one can still stage one extra expert layer, because radiance's stager reuses the previous
pass's reach. Decode is unchanged. What RidgeFill gains on long prompts is under "What it gains"
above.
Keeping the maps in VRAM instead (an earlier option, removed) cost ~1,100 expert slots a card and made a
configuration stock radiance serves refuse to start (`--max-num-batched-tokens 8192 --max-num-seqs 10`: the pinned
pool overflowed), and the plugin cannot see the engine's budget when it declares (notes/stagee.md §8).

A mode with no usable projector serves stock and says why. A configuration that cannot run refuses at
startup by name (tail of two steps less a tile or more, ridgefill.so missing, or libr4d not loaded -- ridgefill.so's
`ridgefill_gemm_nt_bias` forwards to libr4d's `gemm_nt_bias` row). Each approximate step logs one `ridgefill: approximate step` line
(rank 0) with the numbers it was decided from.

**Images and video:** a step that runs the vision encoder, holds media rows or has rows whose rotary components
differ runs the stock step; the text chunks after it are approximated again, at their rotary positions (which run
behind the token index after an image), exactly where the stock step reads them. Draft-head passes always run stock.

**A shorter step for a long tail, with no plugin change:** `--checkpoint-interval` below
`--max-num-batched-tokens` gives each request smaller chunks while `n_ahead` stays capped at the step, at the
cost of more snapshot writes.

**Engine-release guard:** at load the plugin requires the engine binary to carry exactly the radiance release
it was built against. On a mismatch it forwards to the engine's own in-tree Qwen4-Exp architecture (stock, RidgeFill
off, logged with both releases and the engine's sha256) when that file is on `$RADIANCE_HOME` after this
plugin's home, and declines otherwise (startup then fails by name; a home given only as `--radiance-home` is
not visible to a plugin).

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
