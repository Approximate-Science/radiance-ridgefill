# notes/rebase-1.4.0.md -- the plugin on radiance 1.4.0 (dc339b4), installed beside radiance-exl3

1.4.0 (tag 2026-10-12): the next-layer expert prefetch at decode (`rad_expert_prefetch` / `_step` / `_mark`,
abi/rad_runtime.h; `MoeFP8::xp_next` linked in `qwen4exp_fp8.cpp` declare), collectives that end their link hold
early (`rad_kernel_collective`), the MTP head's history pass storing only K/V and indexer keys
(`AttnGatedFP8::store_kv`, `QsaIndexer::store` / `store_keys`, `MoeFP8::step_rows`), `--busy-prefill-chunk`.
ABI 15, unchanged.

## Host verdict (`RK_RADIANCE_REPO=<radiance checkout> scripts/update_radiance.sh v1.4.0 --host-only`)
- At the 1.3.0 pin: **INCOMPATIBLE**. `arch_static_test` did not link: the in-tree MoE now calls the three
  prefetch functions, which the test fake did not define. Four copied sources changed: `rad_block_moe_fp8.h`,
  `rad_block_attn_gated_fp8.h`, `rad_qsa.h`, `qwen4exp_fp8.cpp`.
- Port (50ba02f): `arch/qwen4exp_moe.h` issues `rad_expert_prefetch` at the router input,
  `rad_expert_prefetch_step` ahead of the routed GEMMs and `rad_expert_prefetch_mark` after down, where
  `MoeFP8::pass` does. `tests/rad_fake.h` records the three calls as pseudo-ops, so the oracle holds their order
  (mutation check: dropping the ported prediction fails `plumb_is_the_stock_step_through_the_masked_path`,
  "first difference ... (expert prefetch)"). No other port was needed: `rad_qsa.h` only splits `step()` into
  `store_keys()` and the query half, with the same issues (`qwen4exp_fill.h`'s `qsa_keys` is `store_keys` behind
  `step()`'s `atb` guard); `rad_block_attn_gated_fp8.h` only adds `store_kv()`; the `qwen4exp_fp8.cpp` change
  is the layer link, which the plugin compiles in.
- After the port: **COMPATIBLE**: ctest 5/5 (arch_static 1,160,335 checks over 71 cases, adapter_core 115
  over 5), pytest 228 passed / 33 skipped. Pin d6bef4c; rerun against the new pin: COMPATIBLE, no changed files.

## Device home (data/home-d6bef4c-r1.4.0)
Built in `radiance-build:1.3.0` (ROCm 7.2.4) against the 1.4.0 runtime image's own `/opt/radiance` (copied out of
`stilldeadcode/radiance:1.4.0`; its headers are byte-identical to the 1.4.0 dev-install's), `RADIANCE_SRC` =
`data/radiance-src-1.4.0`; configure: "radiance 1.4.0: RADIANCE_SRC and /opt/radiance/bin/radiance agree";
ctest 5/5 in the image. `qwen4exp_fp8.so` ba735f46 (carries "1.4.0"), `ridgefill.so` 9f7ed3f6.
EXL3 home: the release dry-run tarball (5263e0a = main e0f1d69's binaries), tarball sha256 OK, its four .so OK
against its SHA256SUMS.

## Both homes together (`RADIANCE_HOME=<ridgefill>:<exl3>:/opt/radiance/share/radiance`)
`rad-info --plugins`: one load message, `qwen4exp_fp8.so is shadowed by /ridgefill/architectures/qwen4exp_fp8.so`
(expected); `qwen4exp_ridgefill` claims `qwen4exp @ (unquantised)`, `qwen4exp_exl3` claims `qwen4exp @ exl3`;
ridgefill, exl3r4d, exl3ref load as kernel libraries beside libr4d / libavx / libref. No duplicate key, no other
load-order message, in rad-info or in any of the 9 server logs below.

## GPU (two R9700, gpu.lock; data/gpu140/session.log; kernel log clean throughout)
The display card held ~1.7 GiB of desktop VRAM (librewolf, plasmashell, Xwayland, ...), above the scripts'
1,024 MiB idle ceiling, so B1 ran with `RK_PREFLIGHT_VRAM_MIB=2560 RK_STOP_VRAM_MIB=2560` (the first B1
attempt stopped at preflight with nothing started). `--gpu-headroom-mib 3072` covers the desktop, as before.

**A. EXL3 container, serving pattern (tp2 wht6, seqs 8, len 200000, expert_tiered, pool 11793), both homes,
`RADIANCE_RIDGEFILL=quality` + a projector named (the worst case): PASS.** Up in 290 s. `model: qwen4exp
arch=qwen4exp quant=exl3`; `architecture plugin qwen4exp_exl3@0.1.0`; `I qwen4exp_exl3: prefill staging OFF
(automatic): a pooled expert's zero-copy reads a step of 2048 tokens 1.00 ... against 1.50 to stage; the serving
cards' narrowest host path 7.9 GB/s`. Chat: "Paris". radiance ident.sh: temp 0 `005dcf9faa178d1e` /
`cd44e6cae09e5b92`, 0.7 `13917adf2a177a6a` / `1df93140902a2c95`, 1.0 `4da13637e32d5ff3` / `f5dbddb484280fa8`.
RidgeFill did not engage: no RidgeFill line at all; ridgefill.so is loaded but takes no op band (kernel selection
exl3r4d 344, exl3ref 344, libavx 2626, libr4d 3171).

**B1. Baseline container, RidgeFill's correctness gate with the measurement flags (scripts/common.sh RK_FLAGS),
both homes on every plugin run: PASS, every number equal to 1.3.0's gate.**
| check | result |
|---|---|
| ident: stock (image home) vs off (both homes) | the six hashes EQUAL; and equal to 1.3.0's stock |
| exact KL reference ref-r1.4.0 | rows.f32, tokens.i32, logp.f16 BYTE-IDENTICAL to ref-r1.3.0 |
| int8 quality T2560 | 67 approximate steps; rows BYTE-IDENTICAL to G13's; dNLL +0.00103 [-0.01348, +0.01529], KL 0.0368, top-1 0.9128 |
| int8 speed T2048 | 67 approximate steps; rows BYTE-IDENTICAL to G13's; dNLL +0.02423 [+0.00404, +0.04460], KL 0.0669, top-1 0.8937 |
Folder line: `matches qwen4exp: arch ok, metadata 11/11, tokenizer ok, encodings 489/489, anchors 3/3, 0
warning(s)`; each rank `holds the projector: 637.8 MiB host-mapped ..., 25.4 MiB VRAM`.

**B2. Baseline container, serving pattern (pool 12288; MTP depth 3 by the engine's default), both homes, the int8
folder at /models/projector (found beside the model), one fresh server per mode, the same two prompts
(ramp.py: token ids, max_tokens 1, distinct nonce, cached 0). Prefill = timings.prompt_ms.**
| prompt | off | quality | speed |
|---|---|---|---|
| 32,000 tokens | 23,199 ms = 1,379 tok/s | 20,935 ms = 1,529 tok/s (1.108x), 15 approximate-step lines | 15,481 ms = 2,067 tok/s (1.499x), 15 |
| 16,384 tokens (second prompt) | 11,745 ms = 1,395 tok/s | 10,695 ms = 1,532 tok/s (1.098x), 7 | 8,154 ms = 2,009 tok/s (1.440x), 7 |
Single requests on fresh servers, no repeats: inside the README's fresh-server band (first long prompts
1.07-1.13x / 1.47-1.56x). Not a ramp or stock-path measurement; the stock-path cost was not re-measured on 1.4.0.

## Not done here
The packaged release (tools/package.py), the fresh-engine e2e cases, the stock-path ABCA and decode pairing of the
1.3.0 gate; `--gpu-smoke` itself (it needs `radiance-build:1.4.0`, which does not exist; the steps it runs were run
by hand above). No publishing, no push.

## STATUS 2026-10-13 21:05Z -- ON HOLD (Dylan: "just hold"). NOT MERGED, NOT PUSHED. Quality and speed REGRESSED.

Dylan: "The quality mode is much slower than it was before." Confirmed: the regression is in the SUSTAINED regime
(the expert-cache shift of notes/ramp.md). Fresh-server numbers are unchanged.

### The A/B (data/ab140/, git-ignored; ab.sh, ramp.py, residency.py, analyze.py, session.log)
- **Engines:** radiance 1.3.0 + main@6de05fb's home (data/home-4426e85 in the main checkout) vs 1.4.0 + this
  branch's home. Only the RidgeFill home is mounted on both.
- **Setup:** baseline container, the serving pattern (tp2 wht6, seqs 8, len 200000, expert_tiered, pool 12288,
  headroom 3072, MTP 3 by default, no prefix-cache flags), int8 folder at /models/projector.
- **Requests:** one fresh server per arm, then 32,768 x1, 16,384 x1, 32,768 x22, 16,384 x3 (ramp.py: token ids,
  max_tokens 1, distinct nonce, 0 cached on every request).
- **Prompt times** (s, timings.prompt_ms). "last5" = median of the last five 32K requests; "end 16K" = median of
  the last two 16K requests. Rank 0 resident units are the /stats `layer_resident` sums for layers 0-23 / 24-47.

| arm | 32K fresh | 16K fresh | 32K #20 | 32K last5 | end 16K | resident L0-23 / L24-47, start -> end |
|---|---|---|---|---|---|---|
| 1.3.0 quality | 23.77 | 12.21 | 17.05 | 16.90 | 8.16 | 8,456 / 8,438 -> 10,263 / 5,079 |
| 1.4.0 quality (d6bef4c) | 21.79 | 10.87 | 22.98 | 22.75 | 11.80 | 8,392 / 8,337 -> 7,814 / 7,202 |
| 1.4.0 quality, probe fix (b471150) | 20.90 | 10.68 | 24.93 | 23.44 | 11.68 | 8,389 / 8,376 -> 7,821 / 7,208 |
| 1.3.0 speed | 15.95 | 8.39 | 12.39 | 12.19 | 6.65 | 8,371 / 8,342 -> 9,929 / 5,367 |
| 1.4.0 speed (b471150) | 15.04 | 7.84 | 17.63 | 17.63 | 9.20 | 8,396 / 8,376 -> 7,989 / 7,035 |

- **Fresh:** 1.4.0 is 4-11% faster than 1.3.0.
- **After ~20 long prompts:** 1.3.0 quality falls to 16.9 s and speed to 12.2 s. On 1.4.0 they do not fall:
  quality 22.8 s (1.35x slower than 1.3.0) and speed 17.6 s (1.45x).
- **Why the time does not fall:** on 1.3.0 rank 0's cache moves ~1,800 slots to layers 0-23, and staged bytes fall
  152 -> 88 GiB a 32K request. On 1.4.0 both halves lose residents and staged bytes RISE 151 -> 182 GiB.
- **Not measured:** the `off` pair (stopped by the hold); decode before/after.

### Cause (found, the fix not proven)
- **The credit cap.** radiance 1.4.0's heat engine credits each layer's routing histogram at most
  `HeatParams::credit_cap` = 512 selections a step (core/place/heat.cpp `observe`; new in 1.4.0; no flag or env,
  a struct default).
  - A quality bulk chunk routes every row in layers 0-23 (~20,480 selections) and only its exact rows in 24-47.
    Both are over 512, so they get the same credit and the cache has no reason to favour 0-23.
  - Through 1.3.0 the credit was the count, and that imbalance drove the shift.
- **Decay only on report.** In 1.4.0 a layer's heat decays only when the layer is reported (decay 0.95 per
  credit, was 0.98). Speed's lean path reports nothing for the late layers, so their heat neither grows nor
  decays, and speed does not shift either. The two together are the likely mechanism for both modes.
- **Other 1.4.0 changes** in the same place, not isolated: 64 moves a step (was 8 a dispatch; compile-time
  RAD_PLACE_MOVES_PER_STEP), the mover rewrite, the prefill stager's ring.
- **Ruled out: the stager probes** (my first suspicion). Same staged layers a pass in both versions
  (408-410 a 32K request).
  - The probe change (commit b471150: one probe of the last layer behind S-4, for 1.4.0's ring) lowers stage
    overflows ~2,800 -> ~300 and stage_layers 408 -> 378 a request.
  - It does not restore the ramp (row 3). It is kept as a correct adaptation, measured only by that ramp.

### The untried plugin-side fix (commit 00546fe, home data/home-00546fe-r1.4.0 built, 5/5 host tests)
- **What it does:** on a pass with the drop arm and no decoder (quality's bulk chunks), the late layers report
  the zero offsets as their histogram. Their heat then decays and is not credited by the prompt's few exact rows.
  Early layers keep their capped 512, and decode steps and passes with decoders are untouched.
- **Static test:** a_dropping_pass_credits_the_late_layers_nothing_unless_a_decoder_rides_it.
- **NOT MEASURED ON GPU:** the queued ramp + decode (credit fix / pre-fix / 1.3.0) was stopped by the hold.
- **Speed is not covered.** Its lean path reports no late-layer histogram at all. The analogous fix (report zeros
  for the late layers on lean and straddle passes, so they decay) is not written.
- **Upstream ask:** if the plugin side cannot restore the 1.3.0 ramp, the ask is drafted, not sent, in
  notes/future/upstream-credit-cap.md: a credit cap knob, or a credit weight in RadRouting.

### Gates
- **Done (on d6bef4c, before both fixes):**
  - host verdict COMPATIBLE;
  - ident off == stock;
  - ref-r1.4.0 byte-identical to 1.3.0's;
  - KL rows (quality T2560 / speed T2048, 67/67, rows == G13);
  - EXL3 coexistence.
- **Not done:**
  - every GPU gate on the fixed homes (b471150, 00546fe): ident, KL rows;
  - the quality/speed/off ramp with the credit fix;
  - decode before/after;
  - stock-path cost;
  - packaging (tools/package.py) and the fresh-engine e2e cases;
  - radiance-build:1.4.0 (not built: the homes come from radiance-build:1.3.0 against the 1.4.0 image's
    /opt/radiance);
  - HOW-IT-WORKS / impl.md text on the probe (still describes S-3 and every late layer).
