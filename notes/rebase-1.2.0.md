# notes/rebase-1.2.0.md -- the plugin on radiance 1.2.0 (e95d397, released 2026-10-06 19:36Z)

1.2.0 appeared while the 1.1.1 work was being wrapped up (Dylan: "before you wrap up, rebase again to make sure we
are current"). One release commit.

## What 1.2.0 changed, and what it means here

| change | here |
|---|---|
| MXFP4 weights for block-fp8 dense Qwen3.5/3.6/3.8 (`rad_fp8.h` +86, `rad_block_mlp_fp8.h` fold guard, new libr4d GEMMs) | none: Flash-Next is not MXFP4, and no adapter copy covers either file |
| collectives without peer-to-peer (`--p2p auto|on|off`, a host-memory twin; libr4d `r4d_ar_entry`, the 2-rank exact one-shot) | none in code; this machine has peer-to-peer, so `auto` keeps the peer kernels -- H120-ref tests that the exact logits did not move |
| stop strings apply to the answer, not the reasoning block; a prefix hit ends one drafter window early | engine-side; no plugin path reads them |

`scripts/update_radiance.sh v1.2.0 --host-only` (data/update-1.2.0): **COMPATIBLE**, ABI 15 = pin, ctest -LE gpu
5/5, pytest green, every adapter copy unchanged. Pin c40a2ca: `RADIANCE_VERSION` = `1.2.0 e95d397`, docs.
Images: `stilldeadcode/radiance:1.2.0` pulled; `radiance-build:1.2.0` = `docker/build.sh -r e95d397 --target
build` from data/radiance (this script's own clone; the tag also moved `radiance-build:latest`).
Plugin home c40a2ca: qwen4exp_fp8.so a5a2189b (carries "1.2.0"), ridgefill.so a63c29bc (unchanged).

## Gate (data/lead-20261006/gate120, gate_a.sh's protocol)

20:09-20:49Z, boot 75e3e39b, stilldeadcode/radiance:1.2.0 (sha256:665775ed). **PASS** -- labbook H120-* all confirmed.

| check | result |
|---|---|
| `off` ident == stock (tp2 exact wire); 1.2.0 stock R3 == 1.0.13's | **PASS** |
| exact KL reference ref-r1.2.0 vs ref-r1.1.1 | **byte-identical** (the all-reduce changes did not move the peer path) |
| int8 quality T2560 / speed T2048 | 67 / 67 steps; `.rows` **byte-identical to G13**; dNLL +0.00103 / +0.02423 |
| package + fresh-engine e2e 0-6 | **all pass**; plugin 873b0d12, projector 0ac218bf (gate120/dist, local) |
| stock-path (ABCA, paired) | quality +0.74% [+0.59, +0.79], speed +0.74% [+0.62, +0.80], quality2 +0.69%; drift -0.01% |
| decode vs stock (10 chats) | 0.9966 / 0.9965 / 0.9963; 30/30 answers identical |

TP1 and TP3 were not rerun on 1.2.0 (TP1 measured on 1.1.1; the plugin code is unchanged, and 1.2.0 computes this
model's exact logits as 1.1.1 did).
