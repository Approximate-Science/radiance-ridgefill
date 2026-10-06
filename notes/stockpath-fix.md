# notes/stockpath-fix.md -- the extra prefill stage on prompts RidgeFill does not approximate, found and removed (2026-10-06)

notes/stock-path-cost.md measured prompts RidgeFill does not approximate prefilling +0.9% (512, 1,024 tokens) and
+2.5% (1,600, 2,000 tokens) slower than stock. Rank 0's /stats gave two causes: VRAM displacement at every length,
and, above 1,024 tokens only, 50 prefill layer stages a request against stock's 49. Dylan's rule: such requests
lose no speed against stock, with a ~1% residual accepted. The extra stage broke that rule.

## Cause
- **The hazard instrument's op.** `ridgefill_step.h` `core_step` calls `hazard_issue` (`ridgefill_hazard.h`) after
  `a.stock_step` on every speed/quality pass. It issues `ridgefill_hazard` whenever the last sequence still has
  tail ahead (span = T - n_ahead > 0), which includes every final prefill chunk.
- **Where the op sat.** It was declared in `decl_hazard` (`ridgefill_declare_masked.h:195` at 9763b74), after the
  whole in-tree graph, MTP head included. Op handles follow declaration order.
- **What the engine does with that.** radiance 1.0.13's `PrefillStager` (`core/place/stager.cpp`) arms for steps
  of more than 1,024 tokens. At a routed layer's first op it stages the next layer's pooled experts, but only if
  `will_issue` says this kind of pass reaches that layer. `will_issue` compares the layer's first op with
  `pass_last_[draft_pass]`, the highest op handle the previous pass of that kind issued (`end_pass`).
- **The result.** With the hazard op's handle as the trunk pass's highest, every trunk pass of more than 1,024
  tokens staged the MTP head's routed layer (the 49th) at layer 47's first op, and released it unread at the
  pass's end. Stock stages that layer once, in the head's own pass. That is +1 stage, ~150 units (~one layer's
  pooled experts), on every such request.
- Prompts of 512 and 1,024 tokens never arm the stager, so they pay only the displacement.

## Diagnosis (one gpuq session, 3.2 GPU minutes)
- 10:07:34-10:10:45Z, evidence `~/AI-Work/radiance-kva-plugin-20261004/evidence/ridgefix-diag-20261006/`
  (diag.sh, stages.py, stages.txt, serve logs). Release config as the stock-path run, `RADIANCE_LOG_STEPS=1`.
- Three fresh servers: stock, quality on the r2 packages, quality on the frozen home of 7722c7a (the only change:
  no hazard op on a stock pass that can count nothing). 2 warm-ups, then 3 rounds of 1,024 / 1,600 / 2,000 tokens.
- Labbook HRIDGEFIX-diag (seq 551) **confirmed**:

| server | stage_layers 1,024 / 1,600 / 2,000 | staged units 1,600 / 2,000 |
|---|---|---|
| stock | 0 / 49 / 49 | 7,024 / 7,040 |
| quality r2 | 0 / 50 / 50 | 7,230 / 7,246 |
| quality 7722c7a | 0 / 49 / 49 | 7,084 / 7,100 |

- Every request on every server was one prefill step of `n_tok` = its length at `ctx=0` (`RADIANCE_LOG_STEPS`), so
  the passes are the same. Removing only the stock pass's hazard op removed the 50th stage. The remaining +60
  units are the displacement.

## The fix (branch fix/stockpath-prefetch)
- **7722c7a**: `may_count` (`ridgefill_hazard.h`). A stock pass whose rule can count nothing issues no hazard op:
  no context (`max_ctx_len` 0 -- exact, `bound_bucket(0)` is 0), or one prefill sequence whose rows cover the whole
  span. A fresh prompt's stock path is then the in-tree step op for op.
- **22c14b1**: `core_declare_first` (`ridgefill_step.h`). The hazard op is declared BEFORE the in-tree graph
  (speed and quality only), so it is never a pass's highest op. This also covers the stock passes that can count:
  a prefix hit with a 1,025-2,047-row step, or several prefill sequences in one step. When the projector proves
  unusable, the op is declared and never issued, which the builder allows (`abi/rad_builder.h`
  `rad_op_resolved`). The in-tree ops' handles shift by one; nothing reads them as numbers.
- **Approximated requests:** their passes issue later-declared RidgeFill ops, so their own stager bound and staging
  are unchanged. The bound is learned from the previous pass of the same kind, so an approximated chunk that
  follows a stock pass of more than 1,024 tokens now skips one wasted head-layer stage. That is a timing effect
  only; the KL `.rows` gate checks the outputs.
- **What remains:** a stock pass right after an approximated pass of more than 1,024 tokens still stages the head
  once, from that pass's high bound. That is engine behaviour, and removing it would mean changing approximated
  passes. Not done.
- **Regression check:** `no_stock_pass_issues_an_op_above_the_in_tree_steps_highest` (tests/arch_static_test.cpp).
  With MTP declared, no stock pass issues an op above the in-tree step's highest on the same batch. Shapes: fresh
  1,600 / 2,000, span-covering 2,048, beside decoders, two prefix hits, two prefill sequences. Passes that can
  count still end with the op.
  - It fails on main (8 checks).
  - It fails on a mutant that declares the op after the graph (the 4 prefix-hit and two-sequence checks).
  - The two serve-stock tests now expect the one leading, never-issued op.
- Host, against radiance 1.0.13: `ctest -LE gpu` 5/5, arch_static 69 cases / 1,144,315 checks, adapter_core 5,
  purity gate green, pytest 215 passed / 33 skipped. The frozen homes of 7722c7a and 22c14b1 pass the same suite
  inside `radiance-build:1.0.13`.
- Binaries: only `qwen4exp_fp8.so` changes (22c14b1: c42eb60e...). `ridgefill.so` is byte-identical to r2's
  (a63c29bc...).

## Verification (one gpuq session, 37.5 GPU minutes)
- 10:25:49-11:03:20Z, evidence `~/AI-Work/radiance-kva-plugin-20261004/evidence/ridgefix-verify-20261006/`:
  verify.sh, prefix.py, verify_analyze.py, analyze.out (stockpath's analyze.py, unchanged), verify-tables.md,
  session.log, serve logs.
- Kernel log: 0 amdgpu MES/SMU/timeout/reset lines before and after every server. Every prefix-cache dir under
  `~/.cache/radiance-kva-release/` was removed after its server.
- Labbook: predictions seq 559-561 were registered before the session.
  - HRIDGEFIX-stockpath **confirmed**.
  - HRIDGEFIX-kl **confirmed**.
  - HRIDGEFIX-32k **inconclusive** (below).
- **Packages `dist-ridgefill-0.1.0-r3`** (version 0.1.0, from the frozen home of 22c14b1 + data/projector-ridgefill-qwen38fn-int8):
  - `radiance-ridgefill-0.1.0.tar.gz` cb9ed31f3e3b993bf5391bc78779b066ab233b2b4ddaff9d9fa99b6709da1554;
  - `ridgefill-projector-qwen3.8-flash-next-i8.tar.gz` 88a46a39bf62445f85b38c12f50e8b2d59c043f33b93d66e4f6752d036a1778b
    (= r2's);
  - 39 SHA256SUMS files verified. Their READMEs are still r2's text.
- **e2e cases 0-6 PASS** on the extracted r3 packages, release config (7 parked).

**(i) The stock-path test** (stockpath.sh's protocol: throwaway, quality, speed, stock, quality2, 12 rotated rounds,
paired by request, median [95% CI]):

| tokens | quality / stock | speed / stock | quality2 / stock | drift q2 / q | before (stock-path-cost.md, quality) |
|---|---|---|---|---|---|
| 512 | +0.38% [+0.15, +0.50] | +0.65% [+0.48, +0.76] | +0.51% [+0.32, +0.70] | +0.16% | +0.92% |
| 1,024 | +0.54% [+0.45, +0.77] | +0.69% [+0.35, +0.87] | +0.83% [+0.75, +1.09] | +0.31% | +0.89% |
| 1,600 | **+0.85%** [+0.80, +1.03] | **+0.93%** [+0.81, +1.05] | **+0.93%** [+0.83, +1.05] | +0.01% | **+2.59%** |
| 2,000 | **+0.78%** [+0.75, +0.80] | **+0.80%** [+0.76, +0.82] | **+0.80%** [+0.77, +0.82] | +0.02% | **+2.46%** |

- Prefill layer stages per request at 1,600 / 2,000: **49 on every server**, stock included. Before the fix it
  was 50 against 49.
- Staged units: 7,094 / 7,097 against stock's 7,034 / 7,035. The +60 is the displacement.
- 0 approximate steps on all 200 requests.
- The 512 / 1,024 figures are lower than the previous session's (+0.9%). Those lengths never arm the stager, so
  the difference is session-to-session spread of the displacement cost, not the fix. Across both sessions that
  cost is +0.4% to +1.0%.

**The prefix-hit block** (every stock-path server, after its rounds):
- A 2,048-token primer, then 9 requests of the same prefix + a 1,200 / 1,600 / 2,000-token suffix (prefix.py).
- `cached` was 2,048 on all 9 and there were 0 approximate steps. Each request was one stock step at context
  2,048: the shape whose hazard rule may count, so the op is still issued.
- Layer stages: **49 on every request on every server**, the same as stock.
- prompt_ms paired with stock: quality +0.84% [+0.79, +0.87], speed +0.88% [+0.82, +0.88], quality2 +0.82%.

**(ii) KL rows:** G13's protocol on the extracted r3 packages, int8 quality T2560 and speed T2048. Both `.rows` are
**byte-identical** to evidence/rebase13/g13's, with 67 approximate steps each.

**(iii) e2e:** cases 0-6 PASS (above).

**(iv) 5 x 32K, fresh quality server, against the ramp run's first 5** (evidence/ramp-20261006, r1 packages,
07:5xZ). **Not met as written:**

| request | this run (ms) | ramp (ms) | diff |
|---|---|---|---|
| 1 | 21,062 | 20,198 | **+4.28%** |
| 2 | 20,282 | 20,221 | +0.30% |
| 3 | 19,765 | 19,704 | +0.31% |
| 4 | 18,903 | 18,842 | +0.33% |
| 5 | 18,051 | 17,993 | +0.32% |
| sum | 98,063 | 96,957 | +1.14% |

- **The plugin's work is the same on every one of these requests.** Rank 0's counters match the ramp's exactly:
  streamed bytes (128.85 / 131.76 / 120.17 / 108.75 / 97.66 GiB), staged units, promotions and resident routed
  experts, 410 layer stages, and 15 approximate steps. The arena plan and slab slots are identical too (15,664 /
  15,712).
- What differs is the session. Card 0's sclk sampled after each request was 3,134-3,183 MHz against 3,207-3,291
  MHz in the ramp run. This server followed the KL servers (grade.sh's own flags), where the ramp's quality
  server followed a stock throwaway.
- The uniform +0.3% on requests 2-5 and the one-off +4.3% on the first long request are consistent with that
  state. But a cross-session comparison with no same-session control cannot show "not slower".
- **Decision then (the conservative option): fix/stockpath-prefetch was held unmerged and r3 was not repackaged.**
  The ABA session below settled it.

**(iv) settled: r3 vs r2 in one session (ABA, 11.9 GPU minutes).**
- 11:15:04-11:27:01Z, evidence `~/AI-Work/radiance-kva-plugin-20261004/evidence/ridgefix-aba-20261006/` (aba.sh,
  aba_analyze.py, aba-tables.md). Fresh quality servers r3a, r2, r3b; 5 x 32K each, with the same seeded
  prompts; release config. Kernel log 0; cache dirs removed.

| request | r3a ms | r2 ms | r3b ms | mean(r3) / r2 | r3b / r3a |
|---|---|---|---|---|---|
| 1 | 20,174 | 20,189 | 20,209 | +0.01% | +0.17% |
| 2 | 20,214 | 20,226 | 20,228 | -0.03% | +0.07% |
| 3 | 19,702 | 19,691 | 19,707 | +0.07% | +0.02% |
| 4 | 18,835 | 18,852 | 18,850 | -0.05% | +0.08% |
| 5 | 17,984 | 18,007 | 17,994 | -0.10% | +0.05% |

- **Gate PASSES.** Over requests 2-5, d = median(mean(r3a, r3b) / r2 - 1) = -0.04%, and the r3-vs-r3 spread
  s = median |r3b / r3a - 1| = 0.06%. Request 1 is +0.01%.
- Both r3 servers match the ramp run to the request, here as in the verify session: 20.17-20.21 s against 20.20 s
  on request 1. So the verify session's +0.3% / +4.3% was that session's state.
- **Counters.** r3a and r3b are identical to each other and to the ramp run's r1 counters. r2 differs by up to 60
  staged units and 0.04 GiB streamed a request, although r2's binaries equal r1's. That is the mover's
  run-to-run variation, not an r3 effect.
- Labbook HRIDGEFIX-aba is recorded **refuted**, on the registered kill's "any counter difference" clause only. Its
  timing claim holds.
- Merged into main and repackaged as r3 with the corrected READMEs (notes/release-session.md).

## What the plugin's VRAM is made of (read only, nothing changed)
Startup budget, card 0, quality vs stock (evidence/stockpath-20261006 serve logs): claimable 27.30 vs 27.37 GiB,
**-70 MiB a card**. The engine splits that by `--expert-vs-cache-ratio`: experts -60 MiB (47 slab slots), KV
cache -10 MiB. The often-quoted ~61 MiB is Stage E's figure for the same pieces, before the 1.0.13 arena plan.

| piece | MiB a card | where | avoidable? |
|---|---|---|---|
| staging-ring slot | 25.4 | `rad_dev_alloc`, "already held" (+28.5 with code) | no while the projector streams from host; a half-layer slot (~-12.7 MiB, ~-0.2%) was rejected (notes/stagee.md §14) |
| ridgefill.so code objects | ~3 | "already held" | no while the library is loaded |
| layer-S stream's int8 codes + scales | 20.0 + 0.6 | arena, whole program | only with arena aliasing, which is the engine's (below) |
| projected block input + its int8 codes/scales | 10.0 + ~5.2 | arena, whole program | same |
| in-tree buffers made whole-program (stream, x, x codes, route ids, gdn a,b) | ~7 (the rest of the +43) | arena packing | same |
| mask, bounds, zero offsets, probe rows | < 0.1 | arena | negligible |
| per-sequence KV groups (applied, rho, meta) | small | KV share | negligible |

- **Arena: +43 MiB** at the full plan (940.12 vs 897.31 MiB). It scales with rows: +10 MiB at 256 tokens and +23.6
  MiB at 1,024, which is ~23 KiB a row.
- The per-buffer split is computed from the declarations (2,048 rows, width 10,240, n_embd 2,560, int8 group 128).
  The engine's buffer plan was not dumped (`--debug-graph` not run). The ~7 MiB packing remainder is the
  difference, not a measurement.
- **Why the arena part is whole-program:** every RidgeFill op is declared after the in-tree graph, and the planner
  packs transients by declaration order. A buffer that a later-declared op reads in the middle of the pass must
  therefore live for the whole program (`rad_buf_concurrent`). The plugin cannot interleave its declarations with
  the in-tree declare.
- Stage E found that sharing those bytes with in-tree transients (arena aliasing, "(d)") is not plugin-side.
- **Nothing here is avoidable plugin-side without a behaviour change.** The 25.4 MiB slot could halve at a cost
  Dylan declined. The arena pieces need engine support.
