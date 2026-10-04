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

Container metadata (`rad-convert --set`) first, environment overrides; `arch/kva_config.h` is the full list.

| switch | values | meaning |
|---|---|---|
| `RADIANCE_KVA` / `kva.mode` | `off` (default), `plumb`, `speed`, `quality` | `off` is stock radiance, byte for byte, even when the container holds kva.* tensors |
| `RADIANCE_KVA_TAIL` / `kva.tail` | tokens, default 2048, ≥ 512, ≤ `--max-num-batched-tokens` | the exact tail T |
| `RADIANCE_KVA_ALPHA` | 0..1, default 1 | correction strength |
| `RADIANCE_KVA_ROWSEL` | `class`, `random`, `all` | quality mode's exact-row rule |
| `RADIANCE_KVA_SHARE` / `kva.rowsel.share` | (0, 1], default 0.25 | share of a chunk's class matches kept exact; the row cap derives from it |
| `RADIANCE_KVA_PROJ`, `RADIANCE_KVA_ST`, `RADIANCE_KVA_ROWSEL_TABLE` | `shipped`/`refit`, `shipped`/`swap`/`refit`, `class`/`none`/`all` | which copy of each fitted tensor is served (controls and refits share the container) |
| `RADIANCE_KVA_DECLARE` | `all` | **set it for every `rad-convert` run with this plugin**: rad-convert writes only declared tensors, and an `--in-place` append drops any it was not shown |

A mode that cannot run refuses at startup by name (tail longer than a step, no projector, quality without its
row table, kva.so missing). In this build every mode but `off` still refuses as not implemented.
