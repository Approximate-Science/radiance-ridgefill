# notes/rebase-1.5.0.md -- the plugin on radiance 1.5.0 (b93b531), and the sustained-regime regression

Branch rebase-1.5.0 off rebase-1.4.0 (1705335; notes/rebase-1.4.0.md has the 1.4.0 port and the hold).
Machine: two AMD Radeon AI PRO R9700 (gfx1201, 32 GB), one card on a PCIe Gen5 x16 path and one on Gen4 x4.
Every number here comes from shipped defaults plus the flags named: nothing is tuned to these links.

## Host verdict (`scripts/update_radiance.sh v1.5.0 --host-only`, RK_RADIANCE_REPO = the local radiance checkout)
- **COMPATIBLE at the 1.4.0 pin, no port:** ABI 15; ctest 5/5 (arch_static 1,160,347 checks over 72 cases);
  pytest 228 passed / 33 skipped.
- **What changed that the adapter copies:** only `rad_block_gdn_fp8.h`, which changes a declaration (the
  rotation split of a rotated `ssm_inz`). `rad_block_mlp_fp8.h` skips its gated fold for ParoQuant 5-bit.
  `rad_fp8.h` (+305) is ParoQuant paths; `rad_builder.h` / `rad_encoding.h` add encodings. heat.h is unchanged:
  the credit cap is still 512.
- **Pin:** 5df38e0.
- **Harness** (88a9bff, "scripts: radiance 1.5.0's prefix-cache flags"):
  - 1.5.0 renamed `--prefix-cache-host-mib` to `--prefix-cache-mib`, and refuses `--prefix-cache-dir` /
    `--prefix-cache-disk-mib` by name.
  - The release flags follow 1.5.0's `deploy/compose/flashnext.yaml` (`--prefix-cache-mib 4096`).
  - `common.sh`'s cache profile appends `--prefix-cache-mib`.

## The sustained-regime regression (Dylan, 1.4.0: "The quality mode is much slower than it was before")
### Same protocol on every engine
- **Server:** one fresh server an arm; the baseline container; the serving pattern (tp2 wht6, `--max-num-seqs 8`,
  len 200000, expert_tiered, pool 12288, headroom 3072, MTP 3 by the engine's default, no prefix-cache flags).
- **Plugin:** the int8 folder at /models/projector; only the RidgeFill home is mounted.
- **Requests:** 32,768 x1, 16,384 x1, 32,768 x22, 16,384 x3, back to back (ramp.py: token ids, max_tokens 1,
  distinct nonce, 0 cached on every request). Then dec.py: 10 short chats, 256 tokens.
- **Evidence (git-ignored):** 1.3.0 / 1.4.0 in rebase-1.4.0's data/ab140, 1.5.0 in data/ab150.

### 32K prompt time (s)
"last5" = median of the last five 32K requests. Rank 0 resident units are the /stats `layer_resident` sums,
start -> end.

| engine, arm | fresh | #20 | last5 | resident L0-23 | resident L24-47 |
|---|---|---|---|---|---|
| 1.3.0 quality | 23.77 | 17.05 | **16.90** | 8,456 -> 10,263 | 8,438 -> 5,079 |
| 1.3.0 speed | 15.95 | 12.39 | **12.19** | 8,371 -> 9,929 | 8,342 -> 5,367 |
| 1.4.0 quality (pre-fix) | 21.79 | 22.98 | 22.75 | 8,392 -> 7,814 | 8,337 -> 7,202 |
| 1.4.0 speed (probe change) | 15.04 | 17.63 | 17.63 | 8,396 -> 7,989 | 8,376 -> 7,035 |
| 1.5.0 quality, pre-fix (d6bef4c's plugin) | 19.64 | 21.85 | 21.85 | 8,482 -> 7,838 | 8,473 -> 7,229 |
| 1.5.0 quality, heat-credit fix | 20.31 | 23.13 | 23.16 | 8,482 -> 8,268 | 8,473 -> 6,797 |
| 1.5.0 speed, pre-fix | 14.74 | 17.53 | 17.52 | 8,482 -> 8,019 | 8,473 -> 7,048 |
| 1.5.0 speed, heat-credit fix | 14.74 | 17.01 | 17.00 | 8,482 -> 7,994 | 8,473 -> 7,072 |
| 1.5.0 quality, SHIPPED (88a9bff: probe change) | 20.04 | 21.68 | 21.64 | 8,486 -> 7,836 | 8,435 -> 7,224 |
| 1.5.0 speed, SHIPPED | 14.76 | 16.92 | 16.92 | 8,481 -> 8,016 | 8,440 -> 7,045 |
| 1.5.0 off (stock) | 22.86 | 28.68 | 28.67 | 8,517 -> 7,822 | 8,489 -> 7,444 |

- **Fresh servers:** 1.5.0 is faster than 1.3.0 (quality 19.64 vs 23.77, speed 14.74 vs 15.95).
- **After ~20 back-to-back long prompts:** 1.3.0's cache moves ~1,800 slots to layers 0-23 and it gets faster.
  1.4.0 and 1.5.0 do not move it, and get slower.
- **Against the same engine's stock (1.5.0, last5):** quality 1.31x (pre-fix) / 1.24x (fix), speed 1.64x / 1.69x.
- **Stock itself gets slower** over the same ramp on 1.5.0 (22.86 -> 28.67 s).

### The cause: two rules new in radiance 1.4.0, both still in 1.5.0
1. **The heat credit cap.** `HeatParams::credit_cap` = 512 selections a layer a step (core/place/heat.cpp
   `observe`). A quality bulk chunk routes every row in layers 0-23 and only its exact rows in 24-47. Both are
   over 512, so both are credited the same and the cache has no reason to favour 0-23. Through 1.3.0 the credit
   was the count.
   - A layer is also decayed only when it is credited. Speed's lean pass reports no late layer, so their heat
     is frozen.
2. **No swaps behind a stager-armed pass** (core/engine.cpp, "NO SWAPS BEHIND A PASS THE STAGER ARMED"; absent
   from 1.3.0). Every chunk of a long prompt is stager-armed, so back-to-back long prompts give the mover about
   one unarmed step a request.
   - First ten requests: 895-918 promotions in every arm, 64 a step. 1.3.0 made 272 a request early.
   - Last ten: exactly 1,044 a request in every RidgeFill arm (10,170 over the ten for stock), 96-97% of them
     readmits of units it had just demoted. Staged bytes a 32K prompt RISE over the ramp in every arm
     (stock 279 -> 351 GiB, quality 148 -> 176).
   - Which passes the stager arms is the scheduler's decision; an architecture plugin has no say.

### The plugin-side fix, measured: it does not restore the ramp
- **What it does** (side branch `heat-credit-1.5.0` = 89fd650, local only):
  - On a pass with the drop arm and no decoder, and on every lean pass, every late layer reports a zero
    histogram. Their heat decays and is not credited.
  - The zeros live in the projector's host block, zero by construction.
  - Static test: `an_approximate_pass_credits_the_late_layers_nothing_unless_a_decoder_rides_it`.
- **Effect on 1.5.0, last5:** quality 23.16 vs 21.85 s pre-fix (+6%), speed 17.00 vs 17.52 s (-3%). Fresh:
  quality 20.31 vs 19.64, speed equal. One server an arm, so these are inside what one server can resolve.
  Neither is near the bar (1.3.0: 16.90 / 12.19).
- **Decode after the ramp** (median of the 10 chats, predicted tok/s): quality 86.25 fix / 66.98 pre-fix, speed
  80.30 / 80.41, stock 86.50. Paired by chat: quality fix / pre 1.247 (0.994-1.558), speed 1.000, quality fix /
  stock 0.993. One server each; the quality pre-fix arm's decode is the outlier.
- **Decision:** the update ships WITHOUT it (reverted on rebase-1.5.0: 66beb97, 1757555). It buys no prefill, its
  decode effect is one noisy pair, and the limiter is rule 2, which the plugin cannot reach.
- **Upstream ask:** drafted, NOT sent (notes/future/upstream-credit-cap.md).

### The probe change (b471150), kept
- **Why:** 1.4.0's prefill stager is a ring paced three layers back. One probe of the last trunk layer behind
  layer S-4's gate-up moves it past the end, so layers 24-47 stream; probing every late layer behind S-3 (what
  1.3.0's two-buffer stager needed) let the ring stage layer 24.
- **Measured on 1.4.0, quality ramp:** stage overflows ~2,800 -> ~300 a 32K request, stage_layers 408 -> 378.
  Prefill within noise (the ramp's fresh 32K 20.90 vs 21.79 s). It is part of the gated binary below.

## Gates on the update (commit 88a9bff: pin + probe change + harness; home data/home-88a9bff, qwen4exp_fp8.so
228160e5, ridgefill.so 464ec6a0). The home is built in radiance-build:1.3.0 (ROCm 7.2.4) against the 1.5.0
image's own /opt/radiance; configure: "radiance 1.5.0: RADIANCE_SRC and /opt/radiance/bin/radiance agree";
ctest 5/5 in the image. No radiance-build:1.5.0 image exists; `--gpu-smoke`'s steps were run by hand.

**G1, correctness (measurement flags, data/gate150/g1): PASS, every number equal to 1.3.0's and 1.4.0's.**
| check | result |
|---|---|
| ident stock vs off | the six hashes EQUAL; stock's equal 1.4.0's |
| exact KL reference ref-r1.5.0 | rows.f32, tokens.i32, logp.f16 BYTE-IDENTICAL to ref-r1.4.0 |
| int8 quality T2560 | 67 approximate steps; rows BYTE-IDENTICAL to G13's; dNLL +0.00103 [-0.01348, +0.01529] |
| int8 speed T2048 | 67 approximate steps; rows BYTE-IDENTICAL to G13's; dNLL +0.02423 [+0.00404, +0.04460] |

**The shipped binary against stock on 1.5.0, same protocol:**
- **32K, fresh:** quality 1.14x, speed 1.55x. **16K, the second prompt:** 1.10x / 1.52x.
- **32K, last5:** 1.33x / 1.69x. **16K after the ramp:** 1.28x / 1.56x.
- **Decode after the ramp,** paired by chat with off: 0.993 [0.656, 1.001] (quality; one chat 0.656) and
  0.993 [0.983, 1.093] (speed). Quality's median is 78.78 tok/s, against off's 86.50.

**G2, the release config (1.5.0's flashnext flags, `--prefix-cache-mib 4096`, headroom 3072; data/gate150/g2): PASS.**
- **Package:** tools/package.py through scripts/release_session.sh RK_RELEASE_PARTS="package e2e", the frozen home
  reused. Plugin tarball 4fa32804; projector tarball 203b5a20, byte-identical to the 1.3.0 gate's; 41 checksums OK.
- **Fresh-engine e2e:** cases 0-6 pass, 7 SKIPPED (parked per-request feature) -- as on 1.3.0.
- **Stock-path ABCA** (quality, speed, stock, quality2; 2 warm-ups + 12 rounds of 512/1024/1600/2000 tokens,
  paired by request with stock):

| | 512 | 1024 | 1600 | 2000 | pooled |
|---|---|---|---|---|---|
| quality / stock | -0.50% | -1.93% | +1.14% | +1.11% | **+0.87%** [-0.96, +1.41] |
| speed / stock | +0.95% | +0.66% | +1.38% | +0.87% | **+0.94%** [+0.68, +1.21] |
| quality2 / stock | -0.08% | -1.70% | +1.25% | +1.22% | **+0.90%** [-0.73, +1.54] |

  The drift (quality2 / quality) is +0.03%. That is the ~1% residual 1.3.0 accepted (+0.94-0.96%).
- **Decode vs stock** (10 chats each, fresh servers): quality 0.9968, speed 0.9962, quality2 0.9958. 30/30
  texts identical (1.3.0: 0.9941 / 0.9946 / 0.9911).
- **Docs:** README, the release README and HOW-IT-WORKS now carry the 1.5.0 numbers above. The packages gated
  here were made from 88a9bff, before that text.

**Not done:** a radiance-build:1.5.0 image. The homes come from radiance-build:1.3.0 (ROCm 7.2.4) against the 1.5.0
image's /opt/radiance (configure checks they agree), and `--gpu-smoke`'s steps ran by hand as G1. Mixed traffic
(long prompts between chats, where decode steps are unarmed and the mover swaps) was not measured on 1.5.0.
