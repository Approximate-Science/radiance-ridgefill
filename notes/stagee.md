# notes/stagee.md -- Stage E: the projector's VRAM cost, the int8 projector, untested combinations

Worker: account-B Opus 5.5, worktree `radiance-kva-wt-e`, branch `stage-e` off `0e25c16` (main after A′'s R144
re-run). radiance `140987f` (v1.0.8), read only. Plan: `fix-246/HANDOVER-FIX.md` §6, `PLAN-FIX.md` §6.4, §14-§17 and
"DECISIONS RECORDED", `PACKAGING.md`, `REFUTATION-3-selfload.md` §2.2, `REQUIREMENTS-FIX.md` R77-R80 (+ v1 R85-R91).
Model: the PUBLISHED file `~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad` (sha256 `0af5e962…4d20`, never written);
projector `~/models/rad/projector/` by discovery unless a row says `RADIANCE_KVA_PROJECTOR`. Boot
`75e3e39b-cc5b-49de-8087-a4791f372a92`. Image `stilldeadcode/radiance:1.0.8`, `RK_FLAGS` of `scripts/common.sh`
(`--gpu-headroom-mib 3072`). Every engine arm under `flock gpu.lock`; scripts in `evidence/stagee/scripts/`.

## 0. Predictions registered before the runs (labbook ids)

| id | row | prediction | kill |
|---|---|---|---|
| HE-R80-slots | R80 | resident slab slots a rank: off = exact ±0.3%; vram bf16 −6…−7.5% (≈ −1,100); host ring −0.5…−1% (128 MiB) | vram within 3% of exact, or host ring > 2% down |
| HE-held-2k | R80 | the pure VRAM cost (FORCE_SPLIT=2048: projector held, every pass stock), 2,048-token prompt: vram +8…+20% TTFT vs exact (A.1: +223 ms at 2,112); host ring 0…+3% | vram within ±3% of exact |
| HE-held-1k | R78 | 1,024-token prompt (one unstaged chunk): vram +0…+6%, host ring ±2% | vram > +10% |
| HE-R77-dec | R77 | decode ms/step at C = 1: off = exact ±1%; vram bf16 +1…+4%; host ring ±1%. C = 32: vram +2…+6% | vram C=1 > +8% or < 0 beyond noise |
| HE-R79-kl | R79 | int8 projector vs bf16, paired per doc, last 512 at T 2560 and whole tail at T 2048: |ΔNLL| ≤ 0.002, CI includes 0 | CI excludes 0 or |Δ| > 0.005 |
| HE-R79-slots | R79 | int8 recovers ≈ half the vram slots lost (≈ +550 a rank vs bf16) | < +400 |
