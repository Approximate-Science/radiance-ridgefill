# NOTES — radiance-kva lab notebook

Provenance on every table: plugin commit, radiance `140987f` (v1.0.8), container sha256, flags, date.
Container: `qwen3.8-next-flash-fp8-iq4r-moe.rad` from HF `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`
(LFS sha256 `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20`, 121,969,901,568 B).

## Run-wide deviations from HANDOVER (decided 2026-10-04)
- `--gpu-headroom-mib 3072` on every run (production default 96): Dylan's display is on the R9700 at
  0000:03:00.0 and he asked for headroom on that card. The flag is per-engine, so both cards keep 3 GiB.
  Applied to reference and candidate alike; absolute TTFT is not comparable with the author's 96-MiB numbers.

## Stage 0
