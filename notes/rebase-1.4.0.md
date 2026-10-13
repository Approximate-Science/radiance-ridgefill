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
