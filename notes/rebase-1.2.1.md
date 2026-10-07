# notes/rebase-1.2.1.md -- the plugin on radiance 1.2.1 (d23c3e0, seen upstream 2026-10-07 09:03Z)

One release commit, found by the lead's periodic upstream check (Dylan: keep the plugin current).

## What 1.2.1 changed, and what it means here

| change | here |
|---|---|
| per-expert checkpoints (AutoRound / GPTQ exports that ship experts one by one): `qwen4exp_fp8.cpp` declare asks `moe_expert_sources` (new, `rad_block_moe_fp8.h` +25) instead of naming the stacked tensors | declare-side only. The adapter copies `step()` (qwen4exp_adapter.h) and `MoeFP8::pass` (qwen4exp_moe.h), neither changed; a `.rad` container is read by declared name and answers "stacked" as before. `update_radiance.sh` flags both files CHANGED at file level; the oracle holds every path |
| checkpoint restore keeps a handed-out snapshot until its step ran (#33); headroom warning names the value that clears it (#32); `reasoning_effort` levels probed and published (#31); `rad-convert` reads Quark MXFP4 / AutoRound / GPTQ releases (#29); `tool_choice` grammars | scheduler / server / converter; no plugin path reads them |

`scripts/update_radiance.sh v1.2.1 --host-only` (data/update-1.2.1): **COMPATIBLE**, ABI 15, 71 oracle cases /
1,160,080 checks, adapter_core 5/115, pytest 228/33. Pin fae456a: `RADIANCE_VERSION` = `1.2.1 d23c3e0`, docs, and
the install step's tag `v0.1.0-radiance-1.2.1`. Images: `stilldeadcode/radiance:1.2.1` pulled; `radiance-build:1.2.1`
= `docker/build.sh -r d23c3e0 --target build` from data/radiance (also moved `:latest`). Plugin home fae456a:
qwen4exp_fp8.so 51019e96 (carries "1.2.1"), ridgefill.so a63c29bc (unchanged).

## Gate (data/lead-20261006/gate121, the 1.2.0 gate's protocol; reference compared with 1.2.0's)

09:10-09:50Z, boot 75e3e39b, stilldeadcode/radiance:1.2.1 (sha256:5741b286). **PASS** -- labbook H121-* all confirmed.

| check | result |
|---|---|
| `off` ident == stock; 1.2.1 stock R3 == 1.0.13's | **PASS** |
| exact KL reference ref-r1.2.1 vs ref-r1.2.0 | **byte-identical** |
| int8 quality T2560 / speed T2048 | 67 / 67 steps; `.rows` **byte-identical to G13**; dNLL +0.00103 / +0.02423 |
| package + fresh-engine e2e 0-6 | **all pass**; plugin b0f7bb38, projector 203b5a20 (gate121/dist, local; the projector differs from 0ac218bf only in its card) |
| stock-path (ABCA, paired) | quality +0.35% [+0.26, +0.41], speed +0.40% [+0.36, +0.45], quality2 +0.42%; drift +0.02% |
| decode vs stock (10 chats) | 1.0002 / 1.0003 / 0.9996; 30/30 answers identical |
