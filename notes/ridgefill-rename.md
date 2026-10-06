# The rename to RidgeFill, and the credit (2026-10-06)

KVA was the working name; it names kishida's Q3-8B-KVA-Projector, so the method ships as **RidgeFill**
(Dylan's decision), by Dylan Johnston and tcclaviger, Apache-2.0, doi:10.5281/zenodo.23179168.

## What changed

- Everything that ships or builds: plugin id `qwen4exp_ridgefill` (the file stays `qwen4exp_fp8.so`),
  `ridgefill.so` and its `ridgefill_*` ops, `namespace ridgefill`, `arch/ridgefill_*.h`, `RADIANCE_RIDGEFILL*`
  (no aliases), the `RidgeFill:` / `ridgefill:` log lines and every parser, the `ridgefill.json` manifest, the
  packages `radiance-ridgefill-<ver>` and `ridgefill-projector-qwen3.8-flash-next-<dtype>`, the prose.
- Credit: `NOTICE` (repo root and both packages, summed), `CITATION.cff`, SPDX headers, and inside the
  projector: `ridgefill.json` leads with name/authors/license/doi/homepage, every `.safetensors` header carries
  name/authors/license/doi, and the startup `matches` line prints the manifest's credit.
- `data/projector-ridgefill-qwen38fn-int8` = `tools/ridgefill_projector.py credit --from
  data/projector-qwen38fn-int8`: 93 tensors byte-identical (the tool's per-tensor check, and again through the
  safetensors reader), the parked marker template rebuilt over the same base bytes; `ridgefill.json` 30,833 bytes,
  sha256 `795363fa153f0e7e16454af6acabd644709e703d84de0dd9535565226bb51faa`, 27 files 698,726,607 bytes.

## Checks

- Host build against radiance 1.0.13: `ctest -LE gpu` 5/5; arch_static **68 cases / 1,144,182 checks**
  (c36f4d7, before the rename, built the same way: 68 / 1,144,180 -- the +2 are the credit line's checks);
  adapter_core 5 / 115; purity gate green. pytest 215 passed / 33 skipped (212 before + 3 credit tests).
- Frozen home 540c096 in `radiance-build:1.0.13` (its host suite 5/5); neither `.so` contains "kva".
- One gpuq session, 06:57-07:19Z (evidence/ridgefill/session.log, scripts/session.sh beside it):
  - package 0.1.0 into `~/AI-Work/radiance-kva-plugin-20261004/dist-ridgefill-0.1.0/`:
    `radiance-ridgefill-0.1.0.tar.gz` 7f83cfc1d8284b014757f8428cfcb54693f3312c547c84887b58770ba75c3d0d,
    `ridgefill-projector-qwen3.8-flash-next-i8.tar.gz` df7eef048d2c5014aab99a6fd07d34325af02eafa2efe4148dce19de0ce2e85e;
    every SHA256SUMS verified (39 files).
  - Fresh-engine e2e on the extracted packages, release config (1.0.13 flashnext profile, headroom 3072):
    cases 0-6 PASS (7 parked). Stock 16K TTFT 12,774 ms, quality 10,385, speed 7,791 (3 reps, unsettled
    servers: a gate, not the headline). Cases 3/4 see the credit on the `matches` line.
  - G13's KL protocol on the extracted plugin and projector: int8 quality T2560 and speed T2048, 67
    approximate steps each; both `.rows` files **byte-identical** to G13's pre-rename rows; last-512 scoring
    +0.00103 [-0.01348, +0.01529] and +0.02423 [+0.00404, +0.04460], as before to the last digit.
  - Prefix caches under `~/.cache/radiance-kva-release/ridgefill/` (the root held another session's dir), all
    removed; klog 0.
