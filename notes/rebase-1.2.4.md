# notes/rebase-1.2.4.md -- the plugin on radiance 1.2.4 (7d84086, seen upstream 2026-10-07 21:43Z)

1.2.4: EmbeddingGemma 2 (text / image / video / audio embeddings), a fix for multi-image requests on servers that
lend the activation arena to experts (Flash-Next with expert_tiered, #43), the Qwen vision towers' attention on the
card again, a startup warning when amdgpu cwsr_enable is 0 (#42). `abi/rad_builder.h` and `abi/rad_runtime.h` grow
without a new ABI number (15). No file the adapter copies changed. Host: COMPATIBLE (ctest 5/5, pytest 228/33).
Pin f189847; images stilldeadcode/radiance:1.2.4 (pulled), radiance-build:1.2.4 (built; moved :latest); home
f189847: qwen4exp_fp8.so ecf0b030 (carries "1.2.4"), ridgefill.so 8c5b4ada (unchanged).

## Gate (data/lead-20261006/gate124, 21:50-22:49Z): PASS -- labbook H124-* confirmed
| check | result |
|---|---|
| off == stock; ref-r1.2.4 == ref-r1.2.3; rows == G13, 67/67; e2e 0-6 | **PASS** (plugin ffa6ff5e, projector 203b5a20) |
| stock-path, arms that started quiet | speed +0.54% [+0.31, +0.76], quality2 +0.81% [+0.73, +0.85]; held stock 343 / plugin 372-376 |
| decode vs stock | 0.9925 / 0.9936 / 0.9956, 30/30 identical |

The first quality arm is not used: another session's CPU-only KVA test run kept the load at 6.84 through quiet()'s
600 s wait, and the arm started anyway (+0.74% [+0.29, +0.78]). The GPUs were not shared (that job hides them).
