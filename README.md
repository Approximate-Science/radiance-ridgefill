# radiance-kva

KVA / RidgeFill prefill for radiance's Qwen3.8-Flash-Next (`qwen4exp`), as plugins: an architecture
plugin that shadows the in-tree `qwen4exp_fp8.so` (plugin name `qwen4exp_kva`), the `kva.so` kernel
library, and `tools/kva_sidecar.py`, which puts the fitted tensors into the served container.

## Build

Two inputs: an **installed radiance** and a **source checkout of the same release**. The arch plugin
compiles radiance's `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`, which radiance does not install; configure
refuses a checkout whose release or headers differ from the install, naming both.

### In Docker (the path the served plugins come from)

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

Build radiance host-only outside its tree, then the plugins against it; the tests need no card:
```sh
cmake -S <radiance> -B build-radiance-host -DRAD_WITH_HIP=OFF -DRAD_WITH_FFMPEG=OFF -DRAD_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/build-radiance-host/install"
cmake --build build-radiance-host -j && cmake --install build-radiance-host
cmake -S . -B build-host -DCMAKE_PREFIX_PATH="$PWD/build-radiance-host/install" -DRADIANCE_SRC=<radiance> \
      -DRAD_WITH_HIP=OFF
cmake --build build-host -j && (cd build-host && ctest -LE gpu --output-on-failure)
```
`-DRAD_WITH_HIP=OFF` leaves out the device rows even against a HIP install. Every `build*/` and `home/`
directory is git-ignored.

## Switches (read once, at startup)

The MODE comes from the environment only, default `off`: a container's `kva.mode` is ignored (and said so
once in the log). Other settings: container metadata (`rad-convert --set`) first, environment overrides.
`arch/kva_config.h` is the full list.

| switch | values | meaning |
|---|---|---|
| `RADIANCE_KVA` | `off` (default), `plumb`, `speed`, `quality` | `off` is stock radiance, byte for byte, even when the container holds kva.* tensors |
| `RADIANCE_KVA_TAIL` / `kva.tail` | tokens, default 2048, ≥ 512, ≤ 2 × `--max-num-batched-tokens` − 64 | the exact tail T. Above one step a full chunk proves only one step ahead, so its last T − step rows (rounded to 64) also run exact (lower coverage); at 2 × step − 64 and beyond no row could ever be approximated, so startup refuses |
| `RADIANCE_KVA_ALPHA` | 0..1, default 1 | correction strength |
| `RADIANCE_KVA_ROWSEL` | `class`, `random`, `all` | quality mode's exact-row rule |
| `RADIANCE_KVA_SHARE` / `kva.rowsel.share` | (0, 1], default 0.25 | share of a window's class matches kept exact |
| `RADIANCE_KVA_PROJ`, `RADIANCE_KVA_ST`, `RADIANCE_KVA_ROWSEL_TABLE` | `shipped`/`refit`, `shipped`/`swap`/`refit`, `class`/`none`/`all` | which copy of each fitted tensor is served (controls and refits share the container) |
| `RADIANCE_KVA_STAGE` | `auto` (default), `stock` | the expert-stager lever: `auto` lets the late layers stream only their routed experts on an approximate pass (notes/impl.md §2); `stock` leaves staging as it is |
| `RADIANCE_KVA_STAGE_ROWS` | rows | `auto` streams only when the pass has at most this many exact rows (default: always; R96 measures the crossover) |
| `RADIANCE_KVA_SCORE_BULK` | `1` | KL mode (`--kld-ref`) serves stock unless set: logits on approximated rows are not the model's, so set it only when scoring the exact tail |
| `RADIANCE_KVA_STRADDLE`, `RADIANCE_KVA_FORCE_SPLIT`, `RADIANCE_KVA_SHIFT_B`, `RADIANCE_KVA_FORCE_STREAM` | `split`/`end`, rows, ±rows, `1` | gate-only debug switches (R50, R47, R51, R94); each is said loudly at startup |
| `RADIANCE_KVA_DUMP` | a directory | debug only, synchronises mid-step: the layer-S stream of every approximate chunk (`boundary.p<P>.npy` + `boundary.jsonl`) and the device mask of every masked one (`mask.jsonl`, `rows.jsonl`) |
| `RADIANCE_KVA_DECLARE` | `all` | **set it for every `rad-convert` run with this plugin**: rad-convert writes only declared tensors, and an `--in-place` append drops any it was not shown |

A mode that cannot run refuses at startup by name (tail of two steps less a tile or more, no projector,
quality without its row table, kva.so missing). Each approximate step logs one `kva: approximate step` line
(rank 0) with the numbers it was decided from.

**A shorter step for a long tail, with no plugin change:** `--checkpoint-interval` below
`--max-num-batched-tokens` gives each request smaller chunks while `n_ahead` stays capped at the step, at the
cost of more snapshot writes.

**Engine-release guard:** at load the plugin requires the engine binary to carry exactly the radiance release
it was built against. On a mismatch it forwards to the engine's own in-tree Qwen4-Exp architecture (stock, KVA
off, logged with both releases and the engine's sha256) when that file is on `$RADIANCE_HOME` after this
plugin's home, and declines otherwise (startup then fails by name; a home given only as `--radiance-home` is
not visible to a plugin).
