# radiance-kva

KVA / RidgeFill prefill for radiance's Qwen3.8-Flash-Next (`qwen4exp`), as plugins: an architecture
plugin that shadows the in-tree `qwen4exp_fp8.so` (plugin name `qwen4exp_kva`), the `kva.so` kernel
library, and `tools/kva_projector.py`, which builds the projector folder the plugin loads from
`<model dir>/projector/`. The model file stays exactly as published (the old container append is retired:
`tools/dev/README.md`).

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
   docker run --rm -v <radiance>:/rsrc:ro -v "$PWD":/kva -w /kva radiance-build bash -c '
     cmake -S /kva -B /kva/build-hip -G Ninja -DCMAKE_PREFIX_PATH=/stage/opt/radiance -DRADIANCE_SRC=/rsrc &&
     cmake --build /kva/build-hip &&
     ctest --test-dir /kva/build-hip -LE gpu --output-on-failure &&
     cmake --install /kva/build-hip'
   ```
   `home/architectures/qwen4exp_fp8.so` and `home/kernels/kva.so` are then a `$RADIANCE_HOME` to put in
   front of the installation's. Serve only plugins built in this image: a host compiler's libstdc++ is
   newer than the runtime image's.
3. Check what the engine will load:
   ```sh
   docker run --rm -v "$PWD/home":/plugins:ro radiance-build \
     /stage/opt/radiance/bin/rad-info --plugins --home /plugins:/stage/opt/radiance/share/radiance
   ```
   Expect the installed `qwen4exp_fp8.so` logged as shadowed, `qwen4exp_kva ... architecture qwen4exp`
   from `/plugins/architectures/qwen4exp_fp8.so`, and `kva` among the kernel libraries.

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
var to set. The full set (grep `tests/`): `KVA_RESEARCH_ROOT` (a checkout of the research repo the
refit imports read-only), `KVA_TOKENIZER` (a checkpoint directory), `KVA_SIDECAR` (a built
`kva-sidecar.safetensors`), `KVA_TEST_TOKENIZER` (a tokenizer directory) and
`KVA_TEST_OPERATOR_TEMPLATE` (an operator base chat template). Without them the run is green on the
suites that fake their inputs.

`-DRAD_WITH_HIP=OFF` leaves out the device rows even against a HIP install. Every `build*/` and `home/`
directory is git-ignored.

## Switches (read once, at startup)

The MODE comes from the environment only, default `off`: a container's `kva.mode` is ignored (and said so
once in the log), and so are any `kva.*` weights an earlier append left in it. The fitted tensors come from
the projector folder: `$RADIANCE_KVA_PROJECTOR`, else `projector/` beside the `--model` path as typed, else
beside the file it resolves to (mount the model's DIRECTORY in Docker). A folder that cannot run on the model
(other dimensions, layer layout or tokenizer, a missing or corrupt file) is refused by name and the engine
serves stock; one fitted on another variant of the model (quantisation, recipe, base weights) is a WARNING
naming both, and runs. `arch/kva_config.h` is the full list of startup switches; the capture switches
live outside it -- `RADIANCE_KVA_CAPTURE`/`_STATE` are read in `arch/qwen4exp_kva.cpp`,
`RADIANCE_KVA_CAPTURE_SPLIT` in `arch/kva_declare_masked.h`, which with `arch/kva_declare.h` declares
what a capture writes.

| switch | values | meaning |
|---|---|---|
| `RADIANCE_KVA` | `off` (default), `plumb`, `speed`, `quality` | `off` is stock radiance, byte for byte, even when the container holds kva.* tensors |
| `RADIANCE_KVA_TAIL` | tokens, default 2048, ≥ 512, ≤ 2 × `--max-num-batched-tokens` minus the delta net's chunk tile (64 on this model) | the exact tail T. Above one step a full chunk proves only one step ahead, so its last T − step rows (rounded to 64) also run exact (lower coverage); at 2 × step − 64 and beyond no row could ever be approximated, so startup refuses |
| `RADIANCE_KVA_ALPHA` | 0..1, default 1 | correction strength |
| `RADIANCE_KVA_ROWSEL` | `class`, `random`, `all` | quality mode's exact-row rule |
| `RADIANCE_KVA_SHARE` | (0, 1], default 0.25 | share of a window's class matches kept exact |
| `RADIANCE_KVA_PROJECTOR` | a directory | the projector folder, ahead of `projector/` beside the model. A folder built by `tools/kva_projector.py int8 --from <bf16 folder> --out <dir>` holds the maps in int8 (the container trunk's own `i8*bf16[1x128]` encoding, 0.65 GiB instead of 1.23 GiB, half the bytes each approximated chunk streams); point this switch at it to use it |
| `RADIANCE_KVA_FINAL` | `on` (default), `off` | with MTP on (`--num-speculative-tokens` > 0) and a folder holding the `final` map (`tools/kva_projector.py final`), each approximated chunk predicts its bulk rows' final stream, which the drafting head reads (+210 MB a chunk over the link, +200 MiB host memory, a 50 MiB slot even with int8 maps); `off` is the control |
| `RADIANCE_KVA_ROWSEL_TABLE` | `class`/`none`/`all` | which of the folder's row tables quality mode uses (`none`/`all` are controls) |
| `RADIANCE_KVA_STAGE` | `auto` (default), `stock` | the expert-stager lever: `auto` lets the late layers stream only their routed experts on an approximate pass (notes/impl.md §2); `stock` leaves staging as it is |
| `RADIANCE_KVA_STAGE_ROWS` | rows | `auto` streams only when the pass has at most this many exact rows, else the pass runs the stock step. Default: no limit -- measured on warmed servers, streaming beats the stock step by 220-240 ms a straddling chunk at 512-1,984 exact rows, and a 64-row limit cost 331 ms at 9,216 and ran every T 2560 chunk exact (notes/impl.md, Stage A.1 R96) |
| `RADIANCE_KVA_SCORE_BULK` | `1` | KL mode (`--kld-ref`) serves stock unless set: logits on approximated rows are not the model's, so set it only when scoring the exact tail |
| `RADIANCE_KVA_MIN_BULK_ROWS` | rows | a pass approximates only with at least this many bulk rows, else it runs the stock step. Default 1,024 (the projector is always in host memory): every approximate pass streams the whole projector, and the 64-row checkpoint remainders a prompt alternates with beside decoders cost 208-222 ms approximated vs 61-92 ms stock -- decoders riding them got 2.2-3.5x slower (notes/stageb.md session 2c) |
| `RADIANCE_KVA_STRADDLE`, `RADIANCE_KVA_FORCE_SPLIT`, `RADIANCE_KVA_SHIFT_B`, `RADIANCE_KVA_FORCE_STREAM`, `RADIANCE_KVA_MASK` | `split`/`end`, rows, ±rows, `1`, `all` | gate-only debug switches (R50, R47, R51, R94; `MASK=all` approximates every row before the bulk end, decoders included -- R54's negative control, speed/quality only); each is said loudly at startup |
| `RADIANCE_KVA_TAIL_ONLY` | `1` (default), `0` | gate-only debug switch: `0` makes speed-mode straddling chunks -- and speed chunks beside decoders -- take the masked path instead of the tail-only / decoders path (their oracles, A.1 and Stage B) |
| `RADIANCE_KVA_DUMP_LOGITS` | a directory | debug only, synchronises: every live pass's logits rows on every rank, each named by sequence/position/token (Stage B gate 1, `tools/logit_compare.py`); run with `--profile-ops` so no pass is replayed |
| `RADIANCE_KVA_DUMP` | a directory | debug only, synchronises mid-step: the layer-S stream of every approximate chunk (`boundary.p<P>.npy` + `boundary.jsonl`) and the device mask of every masked one (`mask.jsonl`, `rows.jsonl`) |
| `RADIANCE_KVA_CAPTURE` | a directory | refit/debug only: in `off` mode rank 0 records the stock step of every single-sequence prefill -- host copies of the stream entering layer S and of every late layer's block input. Refused at startup unless `RADIANCE_KVA=off`: the code says it "records exact runs" and refuses with `RAD_E_INVAL` (`arch/qwen4exp_kva.cpp`) |
| `RADIANCE_KVA_CAPTURE_STATE` | a directory | refit/debug only: on every single-sequence prefill chunk, copies whatever late delta-net state the step did not already record, one file per (chunk, rank). Unlike `RADIANCE_KVA_CAPTURE`, the code does NOT refuse it with the mode on -- it is accepted in any mode |
| `RADIANCE_KVA_CAPTURE_SPLIT` | a layer index | refit/debug only: the split layer S a capture uses when no usable projector folder exists (a capture fits a projector, so there may be no folder yet); declares nothing; without a folder, startup refuses unless it names a layer between the PLE and the last (`arch/kva_declare_masked.h`) |
| `RADIANCE_KVA_PROJ`, `RADIANCE_KVA_ST`, `RADIANCE_KVA_DECLARE` | -- | retired with the container append; refused by name |
| `RADIANCE_KVA_PROJ_PLACE`, `RADIANCE_KVA_PROJ_RING` | -- | retired: the projector is always streamed from host memory (below); refused by name |

**Where the projector lives:** in host memory, always. Each rank holds the maps, its correction heads and the row
table in one host-mapped block (1.2 GiB bf16, 0.6 GiB int8) and ONE VRAM slot the size of one late layer's map
(50 MiB bf16, 25.4 MiB int8); on every approximated chunk, as soon as layer L's GEMM has read the slot, the second
lane copies layer L+1's map into it while the rest of layer L computes. Every request pays that slot and the
plugin's arena buffers in resident experts (the layer-S stream, kept with bf16 maps or MTP only, and the projected
inputs), so they are kept as small as the pass allows (notes/stagee.md §14).
Keeping the maps in VRAM instead (an earlier option, removed) cost ~1,100 expert slots a card and made a
configuration stock radiance serves refuse to start (`--max-num-batched-tokens 8192 --max-num-seqs 10`: the pinned
pool overflowed), and the plugin cannot see the engine's budget when it declares (notes/stagee.md §8).

A mode with no usable projector serves stock and says why. A configuration that cannot run refuses at
startup by name (tail of two steps less a tile or more, kva.so missing, or libr4d not loaded -- kva.so's
`kva_gemm_nt_bias` forwards to libr4d's `gemm_nt_bias` row). Each approximate step logs one `kva: approximate step` line
(rank 0) with the numbers it was decided from.

**Images and video:** a step that runs the vision encoder, holds media rows or has rows whose rotary components
differ runs the stock step; the text chunks after it are approximated again, at their rotary positions (which run
behind the token index after an image), exactly where the stock step reads them. Draft-head passes always run stock.

**A shorter step for a long tail, with no plugin change:** `--checkpoint-interval` below
`--max-num-batched-tokens` gives each request smaller chunks while `n_ahead` stays capped at the step, at the
cost of more snapshot writes.

**Engine-release guard:** at load the plugin requires the engine binary to carry exactly the radiance release
it was built against. On a mismatch it forwards to the engine's own in-tree Qwen4-Exp architecture (stock, KVA
off, logged with both releases and the engine's sha256) when that file is on `$RADIANCE_HOME` after this
plugin's home, and declines otherwise (startup then fails by name; a home given only as `--radiance-home` is
not visible to a plugin).
