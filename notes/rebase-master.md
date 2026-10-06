# notes/rebase-master.md -- the plugin on radiance 1.1.1 (7001841, radiance `main`; 2026-10-06)

radiance has no `master` branch; its default branch `main` is at **7001841 = tag v1.1.1** (1.1.0 = 82b5835, both
released 10-06), and that is the "new master" this rebases onto (Dylan's request; HANDOVER-ridgefill-master-tp.md
D1). Commits: 4b39032 (the port), 6f487ac (the pin), 6179188 (the approximate-step count under pass replay).

## What 1.1.0 / 1.1.1 changed that the plugin sees

| in-tree change | what it means here | done |
|---|---|---|
| fused router `router_gemm_topk_scatter`: every MoE pass of <= 64 rows issues router + top-k + sort + shared gate as one op | the adapter's hand-issued MoE copy (qwen4exp_moe.h) must issue the same op on undropped <= 64-row passes; a dropping pass keeps router + topk + drop + scatter and issues the shared gate itself. Before the port: link errors in the test fake, then 12 oracle cases "gemm_nt vs router_gemm_topk_scatter" | ported (4b39032) |
| new ABI exports `rad_weight_shard_span_parts`, `rad_kv_group_zero` (ABI number unchanged, 15) | the oracle's fake builder records them | 4b39032 |
| `--expert-vs-cache-ratio` removed (a server given it does not start); the experts keep what the host pool cannot hold, the KV cache the rest | dropped from RK_FLAGS and the release flags | 6f487ac |
| any world size (`--tp 3`: uneven delta-net heads, an attention-zero rank) | not needed at TP2; branch `tp3` (notes/tp3.md) | separate |
| kernel changes (hc_read prefill form, hc_up rows, moe_gemm_q decode); linear-state snapshots in pinned host RAM | none for the plugin's code; upstream says "identical text" -- **measured identical logits** (below) | -- |
| pass recording: the 2048-token approximate pass is now recorded and replayed (radiance core/runtime/ctx.cpp `run_pass`, unchanged file; what now lets the pass record lies elsewhere) | a replayed pass does not call the plugin's `step()`, so the plugin's per-step log line counts only live passes: grade.sh's 67-step gate read 38 | 6179188: grade.sh counts with recording off (`RADIANCE_DEBUG_ARGSHA=1`) |

`RADIANCE_VERSION` = `1.1.1 7001841`; `scripts/common.sh` derives RK_IMAGE, RK_BUILD_IMAGE and RK_RADIANCE_SRC from
it, so the next bump is one line (6f487ac).

## Host verdict (`scripts/update_radiance.sh 7001841`, data/update-1.1.1/)

Before 4b39032 INCOMPATIBLE (link errors, then the 12 oracle cases); after it **COMPATIBLE, exit 0**: ctest -LE gpu
5/5, arch_static 69 cases / 1,157,049 checks, adapter_core 5/115, purity green, pytest 215 passed / 33 skipped.
Mutants: shared gate always skipped -> 43 failures; dropping pass fused -> 1 failure.

## Gate A -- the engine gate on 1.1.1 (data/lead-20261006/gateA, gateA2, countcheck; boot 75e3e39b)

| check | result |
|---|---|
| `off` ident == stock ident (six hashes, tp2 exact wire) | **PASS** |
| 1.1.1 stock R3 == 1.0.13's | equal |
| exact KL reference ref-r1.1.1 vs ref-1.0.13 (rows, tokens, logp) | **byte-identical**: the engine computes this model as 1.0.13 did |
| int8 quality T2560 / speed T2048 `.rows` vs G13 | **byte-identical**; last-512 dNLL +0.00103 [-0.01348, +0.01529] and +0.02423 [+0.00404, +0.04460] (= G13) |
| approximate steps (grade.sh's gate) | 38 / 38 with recording on (one "recorded pass" line); **67 / 67** with 6179188 (countcheck), rows still identical |
| package + fresh-engine e2e 0-6 (gateA2) | **all pass** (plugin a07e7e70, projector 0ac218bf = r3's) |
| stock-path cost (ABCA, 12 rotated rounds, paired) | quality +0.76% [+0.73, +0.82], speed +0.74% [+0.63, +0.77], quality2 +0.73%; drift -0.12%; at 1,600 tokens +0.94% / +0.86% with CI upper +1.12 / +1.14 (lower +0.81 / +0.77): r3's level (+0.78..+0.93) |
| decode vs stock (10 chats, paired) | 0.9968 / 0.9970 / 0.9965 (>= 0.99); 30/30 answers byte-identical to stock; MTP acceptance 0.743 on every arm |

Labbook: H111-ident, -e2e, -stockpath, -decode confirmed; H111-rows **refuted** on its mechanism -- it predicted
1.1.1's kernel changes would move the logits; they did not. **Gate A: PASS** -- merge-ready.

Gate A's first part 2 failed at docker mount time, not in the plugin: `~/models/rad/projector`, the mount point the
scripts bind the projector over (inside the read-only model mount), had been deleted in a storage cleanup. It was
recreated empty and part 2 rerun as gateA2.

GPU page faults: `amdgpu [gfxhub] page fault` bursts at server STOP after long prompts under the release profile,
stock included; all 14 of 2026-10-06 fell within 3 s of a container stop. The gates' kernel-log checks looked
only for MES/SMU/timeout/reset; preflight.sh now records page faults (count and last time) in every run's evidence.
