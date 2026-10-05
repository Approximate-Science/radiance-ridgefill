# notes/stageb.md -- Stage B: concurrency (approximation beside decoding users and other prompts)

Worker: account-B Opus 5.5, worktree `radiance-kva-wt-b`, branch `stage-b` (rebased on main `02b674e`:
the guard streams every masked pass by default, 0d75987). radiance `140987f` (v1.0.8), read only.
Model: the PUBLISHED container (`0af5e962…4d20`, no kva.* inside); projector folder
`~/models/rad/projector/` found by discovery. Boot `75e3e39b-cc5b-49de-8087-a4791f372a92` (KL
reference `data/kld/ref-stage0` valid). Plan: fix-246 HANDOVER-FIX §3, PLAN-FIX §3/§8, REQUIREMENTS-FIX
R53′–R61, REFUTATION §3. Labbook predictions HB-R54-text … HB-R60-spec (seq 368–374) registered
before any engine run.

## 1. What Stage A already had, and what Stage B added

Stage A built the mixed-step mechanism the plan asks for (notes/impl.md §1, §3): the decode half
issued verbatim over `[0, DT)` with `num_accepted` (`arch/kva_layer.h` `gdn_decode_half`), the
other prefill sequences' scan as stock (`gdn_prefill_scans`, `P > 1`), the correction and decay
sums on index row `n_seq − 1` with M = 1 (`last_slot`), the device mask that only ever marks rows
of the last sequence's window `[s, b)`. Stage B adds no serving mechanism. It adds:

| commit | what | why |
|---|---|---|
| `a0ce022` | kernels: `kva_mask` mode `step` (mask 1 on `[0, b′)` when the window is not empty; bounds keep `s`) | R54's negative control: a gate that says "decoders unchanged" must be shown able to fail |
| `1252304` | arch: `RADIANCE_KVA_MASK=all` (declares the mask op in `step` mode, projector from row 0, refused in plumb, said loudly); R53′ static set; `RADIANCE_KVA_CAPTURE_STATE` on MIXED steps (every sequence's late delta-net states after the step, `mixed.jsonl` + one `.npy` a step and rank) | R53′, R54, R61 |
| `58ac8d4` | `scripts/conc.sh` (ttft = fnconc both halves + log windows; text = deterministic tiercross), `scripts/two_prompts.sh` (R58′: KL at `--max-num-seqs 2`, live pair + 8 decoders), `tools/conc_steps.py` (R57), `tools/state_compare.py` (R61), `tools/mask_pn2.py` (R58′); `mask.jsonl` gains `n_seq`/`n_seq_decode`; `grade.sh` `RK_KLD_SEQS` | the engine gates |
| `02058bb` | conc.py reads labelled metrics | `wait_idle` would not have waited |

**Why the text test is a batched request.** R54 compares a decoder's text with `off` "in the SAME
arrangement". A decoder sent first and a long prompt sent a moment later land on different steps
from run to run (wall clock), and a decoder token computed beside a 2,048-row chunk vs alone runs
its dense GEMMs at a different M -- the ident.sh noise class (`narrow_ksplit` from M). One
`/v1/completions` request whose prompt is the batch `[decoder_1 .. decoder_D, 32K prompt]` is
admitted together, so every run puts the same decoder token beside the same prefill chunk; the
step lines (`RADIANCE_LOG_STEPS=1`) are kept to show it.

**Why R58′'s tails are scored in KL mode.** The API has no prompt logprobs (`echo` only prepends
text, `core/server/server.cpp:610`), so a concurrently served prompt's tail cannot be scored over
HTTP. The KL mode admits every doc at once (`core/engine_kld.cpp`, "EVERY DOCUMENT AT ONCE"), and
at `--max-num-seqs 2` the scheduler tops a step up with the next doc's first chunk -- exactly the
two-prefill step (A's final chunk exact, B's chunk approximated). Non-final chunks end on the
quantum, so this hand-over is the ONLY way two prefills share a step at these flags
(`scheduler.cpp:656-690`); decoders do not create more of them (2,048 − D rounds down to 1,984 and
the 64 − D rows left hold no quantum).

## 2. Static (host, build-host; 51 cases, 680,841 checks after the rebase)

New cases: `r53_mixed_steps_are_the_in_tree_step_with_only_the_last_sequence_masked` (D ∈ {1, 4},
decoders verifying 1 or 1 + n_spec 3 rows, one or two prefill entries, whole-bulk and straddling,
speed and quality -- 16 shapes, oracle = `qwen4exp_fp8::step` on the same batch with only the
masked path's substitutions); `the_mask_all_control_approximates_every_row_before_the_bulk_end`;
`a_mixed_step_capture_copies_every_sequences_late_states_after_the_step`. Kernel host:
`mask_window_clamping` checks `step` on every window shape.

Mutants (scratch copy `/tmp/stageb-mut`, `evidence/stageB/sessions/mutate_b.py`,
`evidence/stageB/mutants-1.txt`) -- every one CAUGHT:
```
B1 correction over every sequence            B2 correction on index row 0
B3 decode half skipped                       B4 decode half without num_accepted
B5 decode half depth from max_q_len          B6 kva_drop_rows omitted
B7 stager probes omitted (the plan's "stock down handle" mutant; the alternate handle is not
   declarable, notes/impl.md §2)             B8 projector from the decode rows (s_lb = DT)
B9 other prefill's scan dropped              B10 conv/kkt over the decode rows' cu
B11 gated norm over every row                B12 MASK=all ignored by the declare
B13 MASK=all keeps the projector at s_lb     B14 MASK=all accepted in plumb
B15 mixed capture reads the last slot only   B16 mixed capture skipped on approximate steps
K1 step mode from s (host row)               K2 step mode from 0 on an empty window
```

## 3. Engine gates

(results below as they land; frozen home `data/home-58ac8d4`: arch 3ce44304…, kva.so 9d80ee76…)

### Session 1 -- correctness (2026-10-05 10:08-10:23Z; `evidence/stageB/session1.log`, script `sessions/s1.sh`)
Boot 75e3e39b…, model 121,969,901,568 B (published), image stilldeadcode/radiance:1.0.8, RK_FLAGS of
scripts/common.sh, `RADIANCE_LOG_STEPS=1` on every server. Kernel log clean before and after.

- **kva_mask device leg** (card 0000:13:00.0, `ROCR_VISIBLE_DEVICES=1`, "device 0 of 1 visible, PCI
  0000:13:00.0"): 240 configurations (192 before + 48 `step`) device mask and bounds == host; 1,076
  checks; ctest gpu 1/1.
- **R6/R7 green**: off @ 58ac8d4, ident.sh = R3's six hashes (dc7115567e4d85e4 baba87f4bfdebdca /
  74624bdf06eb6837 bc36faa6cefaa6fe / aa87d500fb3831d1 fc9a051f9310d638).
- **The arrangement is deterministic**: off, 3 reps each, D = 1 and D = 4: every text hash identical
  rep to rep. Step shape (step log): step 0 = the decoders' prompts (final chunks) + the 32K prompt's
  first 1,984 rows (a two-prefill step, approximated, Pn 2); then the 32K prompt's chunks ALTERNATE
  1,984 and 64 rows beside the decoder rows -- the scheduler stops each chunk at the 2,048 checkpoint
  boundary (`scheduler.cpp:656-690`), so a 32K prompt beside a decoder takes ~33 steps, not 16, in
  every mode (stock behaviour). Every step with bulk rows is approximated (30 lines a request).
- **R54 green** (decoders' text beside an approximated 32K prefill, same arrangement as off):

| mode | decoder texts byte-identical to off | the 32K prompt's own continuation |
|---|---|---|
| speed (180 approximate steps, all masked/stream) | **20 / 20** (D=1 ×3, D=4 ×3 = 15 concurrent + 5 solo) | differs (it is approximated) |
| quality (180, masked/stream) | **20 / 20** | differs |
| quality + `RADIANCE_KVA_MASK=all` (negative control, 60) | **0 / 5 concurrent** (diverge after 0-2 characters); 5/5 solos identical (no prefill beside them) | differs |

  The control proves the gate can fail, and fails hard. "Equals A solo except where ident.sh's header
  allows": **stock itself** gives a decoder a different text beside a 32K prefill than alone -- off
  concurrent vs off solo share 0 / 0 / 462 / 14 characters (D=4 decoders 0-3), the first divergence
  at the first generated token after `Answer:` (`' '` vs `'\n'`, both continuations sensible). KVA
  changes none of it (its texts equal off's concurrent AND off's solo texts byte for byte).
  **Session 1b (10:24-10:26Z) confirms it is the stock engine's own**: `exact` (the image's plugin
  home, nothing of ours mounted), same arrangement: D=1 `a1fb40bb…`, D=4 `bd6ac2fc… 35b2d8e6… cd60a13d…
  3807e191…`, solos `2a82de60… 6ea5a478… 130b1312… 570771a1…` -- the same hashes as off, speed and
  quality. (Reading the first token's margin was not possible: this build's sampler refuses
  `logprobs`, HTTP 400.) Whether a first-token flip between M = 21 and M = 2,005 is a "defect" in the
  ident header's sense is a stock-engine question; KVA's bar -- identical to off in the same
  arrangement -- holds byte for byte.
- **R61 green** (`RADIANCE_KVA_CAPTURE_STATE` on mixed steps, D = 1, 1 rep a mode; captures
  `data/stageB/r61-{off,quality,speed}`, 10 GB): off logged 31 MIXED steps; all 31 × 2 ranks matched
  by key in quality and in speed:

| comparison | matched mixed steps, every decoder slot byte-identical | prefill slot moved (positive control) |
|---|---|---|
| quality vs off | **62 / 62** | 58 / 58 approximate steps (max abs diff 0.76-2.1) |
| speed vs off | **62 / 62** | 58 / 58 |

  (The 4 non-approximate matched steps are the prompt's last chunks; their prefill slot differs too,
  because it continues from approximated state.)

### Session 2 -- R55 / R56 / R57 (2026-10-05 10:37-11:46Z; `session2.log`, `sessions/s2.sh`)
fnconc pattern (`scripts/conc.sh ttft`): C streaming decoders (ignore_eos, temp 0) reach 16 tokens,
2 s alone, then one long prompt (max_tokens 1); warmed servers (one discarded pass), RK_REPS 7 read
reps 3-7, quiet host before each label, `--max-num-seqs 9` (see the trap below), RADIANCE_LOG_STEPS=1
on every arm. Tables by `tools/conc_table.py` (bootstrap 95% CI, 2,000 resamples, unpaired).

**Trap found and fixed first (named):** at the serving default `--max-num-seqs 8`, eight decoders fill
every slot, the long prompt QUEUES until they finish (8,144 decode-only steps) and then prefills alone
-- its prompt_ms is a solo number filed as C = 8. That run is kept in `evidence/stageB/invalid-seqs8/`;
conc.py now refuses a rep with no mixed step or > 2 s queued (642c43b).

| arm | length | C | prompt_ms | speedup vs exact | **R55**: speedup(C)/speedup(0) | decoder gap median, prefill (ms) | **R56** KVA/exact median gap |
|---|---|---|---|---|---|---|---|
| exact | 16K | 0 / 1 / 4 / 8 | 9,633 / 10,041 / 10,222 / 10,301 | 1 | | - / 64.6 / 77.8 / 92.3 | |
| exact | 32K | 0 / 1 / 4 / 8 | 19,382 / 20,239 / 20,515 / 20,660 | 1 | | - / 72.0 / 86.6 / 99.7 | |
| quality | 16K | 0 | 6,134 | 1.570 [1.516, 1.615] | | | |
| quality | 16K | 1 | 6,686 | 1.502 [1.485, 1.510] | **0.956** | 67.2 | 1.040 [0.848, 1.105] |
| quality | 16K | 4 | 7,315 | 1.397 [1.378, 1.440] | **0.890** | 79.2 | 1.018 [0.761, 1.223] |
| quality | 16K | 8 | 7,712 | 1.336 [1.266, 1.396] | **0.851** | 91.7 | 0.994 [0.758, 1.539] |
| quality | 32K | 0 | 10,709 | 1.810 [1.666, 1.865] | | | |
| quality | 32K | 1 | 11,702 | 1.730 [1.720, 1.735] | **0.956** | 70.9 | 0.985 [0.802, 1.069] |
| quality | 32K | 4 | 12,878 | 1.593 [1.568, 1.629] | **0.880** | 91.7 | 1.059 [0.694, 1.255] |
| quality | 32K | 8 | 13,313 | 1.552 [1.491, 1.615] | **0.857** | 80.2 | 0.805 [0.459, 1.228] |
| speed | 16K | 0 | 3,905 | 2.467 [2.376, 2.553] | | | |
| speed | 16K | 1 | 5,428 | 1.850 [1.821, 1.860] | **0.750** | 58.2 | 0.900 [0.775, 1.082] |
| speed | 16K | 4 | 5,905 | 1.731 [1.706, 1.811] | **0.702** | 74.2 | 0.953 [0.726, 1.088] |
| speed | 16K | 8 | 6,247 | 1.649 [1.580, 1.729] | **0.668** | 90.8 | 0.984 [0.654, 1.328] |
| speed | 32K | 0 | 6,849 | 2.830 [2.685, 2.847] | | | |
| speed | 32K | 1 | 8,790 | 2.303 [2.260, 2.317] | **0.814** | 73.6 | 1.022 [0.693, 1.079] |
| speed | 32K | 4 | 9,542 | 2.150 [2.126, 2.243] | **0.760** | 64.7 | 0.747 [0.576, 1.203] |
| speed | 32K | 8 | 9,843 | 2.099 [1.920, 2.185] | **0.742** | 87.9 | 0.882 [0.406, 1.066] |

Decoders' longest and p90 gaps during the prefill (median over the settled reps):

| arm | 16K C 1 / 4 / 8: p90 ms (KVA / exact) | 16K max ms (KVA / exact) | 32K p90 | 32K max |
|---|---|---|---|---|
| quality | 706/1,199 · 787/1,211 · 843/1,210 | 1,371/1,211 · 1,389/1,217 · 1,394/1,219 | 657/1,211 · 735/1,219 · 759/1,221 | 1,347/1,218 · 1,368/1,224 · 1,373/1,229 |
| speed | 524/1,199 · 575/1,211 · 608/1,210 | 1,365/1,211 · 1,385/1,217 · 1,389/1,219 | 445/1,211 · 487/1,219 · 501/1,221 | 1,342/1,218 · 1,365/1,224 · 1,365/1,229 |

Reading:
- **R55 RED.** Speed keeps 0.67-0.81 of its solo speedup with ANY decoder beside it (prediction
  HB-R55-speed: ~0.75-0.85, confirmed in direction; slightly worse). Mechanism (by construction,
  `kva_plan.h`): the lean fill needs `D == 0`; one decoder moves every bulk chunk onto the masked path,
  which runs every late block over all rows. Quality keeps 0.956 at C = 1 but 0.85-0.89 at C = 4 and 8
  (prediction 0.95-1.0: refuted). The long prompt is still 1.34-2.30x faster than stock in every cell.
- **R56: medians green, the longest stall red.** Decoders' median and p90 gaps during the prefill are
  at or below stock's (KVA steps are shorter); but their single longest gap is +10-14% (1,342-1,394 vs
  1,211-1,229 ms) in every cell. Leading suspect (to test, session 2c): the prompt's last chunks run the
  stock step on an ON server, which holds ~1.2 GiB less expert VRAM a card (A.1's residual) -- see 2b.
- **R57 GREEN** (`tools/conc_steps.py`, every rep of quality and speed): every chunk the rule
  approximates is logged or replayed, none unexpected. With decoders, the scheduler stops each chunk at
  the 2,048 checkpoint boundary, so chunks alternate 1,984 / 64 rows; the 64-row passes are below the
  stager's 1,025-row arming, are recorded on their second sighting and replayed silently after (e.g.
  32K at C = 8: 30 expected = 15 logged + 15 replayed). A replay issues the recorded approximate pass
  (tape audit), so they are counted, not missed (d04861f).
- Decode-only gaps before the prefill (2 s window, noisy): exact 11.9 / 17.8 / 22.7 ms (16K run, C 1/4/8)
  vs quality 12.1 / 19.2 / 33.7 and speed 12.0 / 19.7 / 32.3; at 32K 28.0 vs 26.7 / 28.6 -- inconsistent;
  measured properly (cstep-style steady state) in session 4.

### Session 2b -- the mechanism of quality's C-penalty, first attempt (11:49-11:57Z; `session2b.log`)
- HB-R55-decexperts (registered before; "the decoders' routed experts, streamed zero-copy on masked
  passes, are the C-penalty") -- **refuted**. `--profile-ops`, 16K, C = 0 vs 8, per step on rank 1 (the x4
  card): quality's big-chunk late `moe_gemm_q` 232.1 -> 231.2 ms a step (no growth with decoders);
  exact's 39.0 -> 52.4 ms. Short (64-row) steps: exact 185 ms, quality 211 ms a step (all ops, profiled).
  Profiled times do not compare across arms (profiling restores synchronisations; quality's big step
  reads 671 ms profiled while it is the FASTER arm unprofiled), so session 2c times steps unprofiled.
- The stager-lever-off arm (`RADIANCE_KVA_STAGE=stock`) cannot test it: with the lever off the
  never-slower guard sends every masked pass to the stock step (0 approximate steps). What it did
  measure: **an ON (quality) server running every chunk exact takes 11.6 s at 16K vs the stock server's
  9.6 s (+20%)** -- A.1's projector-VRAM residual (fewer resident experts, bigger staging buffers),
  on every exact chunk an ON server runs.

### Session 3 -- R58′ (12:42-13:06Z; `session3.log`, `sessions/s3.sh`; 3b for the 8,192 live arm)
**First attempt on quick9 produced no two-prefill step** (kept in `evidence/stageB/r58-quick9-nopn2/`):
quick9's docs are cut to 2,048 multiples, so every doc's final chunk fills its step and the scheduler
never tops a step up with the next doc. The corpus that exercises R58′ is `quick9-off1024` (k·2,048 +
1,024 tokens, reference `data/kld/ref-off1024`, same boot): each doc's 1,024-row final chunk leaves the
step's other 1,024 rows to the next doc's first chunk (12 two-prefill steps a run at 2,048).
**Second attempt (12:41Z) aborted on a lane breach** -- see "Incident" below.

KL half: `scripts/two_prompts.sh kl` (= grade.sh at `--max-num-seqs 2`), every candidate
`RADIANCE_KVA_SCORE_BULK=1`. "s1/s2" = `--max-num-seqs` 1/2. Scoring `scripts/kl_tail.py`; R58′'s test
`tools/r58_did.py`: per doc, (mode s2 − s1) − (exact s2 − s1), bootstrap 95% over the 9 docs.

| chunk | run | approx. steps (Pn 2) | dNLL vs stock ref, last 512 | KL | top-1 |
|---|---|---|---|---|---|
| 2,048 | exact s2 (the floor) | - (12 two-prefill steps) | +0.00389 [−0.00365, +0.00984] | 0.0149 | 0.9475 |
| 2,048 | quality s1 / s2 | 67 / **71 (12)** | +0.01040 / +0.01762 | 0.0486 / 0.0517 | 0.906 / 0.899 |
| 2,048 | speed s1 / s2 | 67 / **71 (12)** | +0.02546 / +0.03037 | 0.0549 / 0.0580 | 0.895 / 0.897 |
| 8,192 | exact s1 / s2 | - (36 of 40 steps two-prefill) | 0 (rows = the 2,048 reference) / +0.00342 | 0 / 0.0151 | 1 / 0.946 |
| 8,192 | quality s1 / s2 | 54 / **32 (29)** | +0.01040 (rows = 2,048's) / +0.00831 | 0.0486 / 0.0353 | 0.906 / 0.919 |

| R58′ difference of differences | last 512 | whole tail (2,047) |
|---|---|---|
| quality, 2,048 | +0.00333 [−0.00581, +0.01276] | −0.00057 [−0.00525, +0.00365] |
| speed, 2,048 | +0.00102 [−0.00574, +0.00755] | −0.00218 [−0.00771, +0.00276] |
| quality, 8,192 | −0.00552 [−0.01265, +0.00210] | **−0.01447 [−0.02049, −0.00766]** (better) |

Reading:
- **Stock moves too.** A hand-over top-up shifts every later doc's chunk boundaries by 1,024 rows, and
  stock's tails move by KL 0.015 / top-1 0.947 against its own solo reference -- the floor R58′ is read
  against. At 2,048 neither mode moves beyond it (every CI includes 0): **green**.
- At 8,192 with one prompt in flight the chunks are still 2,048 rows (the scheduler also cuts at the
  2,048 checkpoint interval, `scheduler.cpp:656-668`): s1 rows are byte-identical to the 2,048 run.
  With two prompts it packs two 2,048-row chunks into one 4,096-row step, and only the step's LAST
  sequence is approximated -- 22 of quality's bulk chunks ran exact (54 -> 32 approximate steps). The
  whole tail is then BETTER than solo (−0.0145, CI excludes 0) -- the design's "every other row
  exact", paid in speed, not a defect.
- Live half, 2,048 (`two_prompts.sh live`, two 32K prompts in one request beside 8 decoders, `--max-num-seqs
  10`, `RADIANCE_KVA_DUMP`): 45 approximate steps, **16 with Pn 2**; `tools/mask_pn2.py`: **0 rows before
  the last sequence approximated** on all 16 (e.g. n_tok 1,992, s = 72, window 1,838/1,920 approximated --
  the 8 decoder rows and the 64 rows of the other prompt all exact).
- Live half, 8,192: the quality server (projector in VRAM) **refused to start** at `--max-num-batched-tokens
  8192 --max-num-seqs 10`: "DID NOT FIT 461 x blk.*.ffn_gate_up_exps (554.94 MiB): the host pool is full
  and no --weights-disk-tier was given" (`serve-pair-t8192-repro.log`). Session 3b checks whether stock
  starts at those flags and runs the arm with the projector in host memory.

**Incident (12:41Z, reported to the coordinator).** My session took the GPU lock while Stage E's
`radiance-kva-quality` server was still up on port 8100; my KL runs failed on the port (no GPU work), and
the session's cleanup (`scripts/stop.sh`, which stops every `radiance-kva-*` container) stopped THEIR
server mid-measurement. Every Stage B session now refuses to start anything, and exits, if any radiance
container is up when it holds the lock (`guard()` in `sessions/s{2c,3,3b,4}.sh`).

### Session 3b -- the 8,192 live arm (13:12-13:16Z; `session3b.log`)
- At `--max-num-batched-tokens 8192 --max-num-seqs 10` (RK_FLAGS otherwise): **stock (exact) and off
  start** (health OK); **quality with the projector in VRAM refuses to start** ("DID NOT FIT 461 x
  blk.*.ffn_gate_up_exps (554.94 MiB): the host pool is full and no --weights-disk-tier was given").
  The 1.2 GiB a card the VRAM projector holds turns a configuration stock serves into one that does not
  start -- by name, and `RADIANCE_KVA_PROJ_PLACE=host` (or `--weights-disk-tier`, or a larger
  `--host-pool-mib`) serves it. Finding for the placement trade (Stage E / DD-I), not a Stage B
  mechanism; reported.
- Live pair at 8,192 with `RADIANCE_KVA_PROJ_PLACE=host`: 15 approximate steps, **all 15 with Pn 2**
  (n_tok 4,104 = 8 decoders + the first prompt's 2,048-row chunk + the second's); `mask_pn2.py`: **0 rows
  before the last sequence approximated** (window 1,922-1,955 of 2,048 approximated: quality's class rows).

**R58′ GREEN**: both tails at stock's own two-prefill floor at 2,048 (quality and speed), better than
solo at 8,192 (more rows exact by design); the last sequence approximated (12 / 29 Pn-2 KL steps, 16 /
15 live); earlier sequences' rows all exact (31 live two-prefill masks checked).

### Session 2c -- R55 with the interleaved instrument, and where the time goes (13:20-13:38Z; `session2c.log`)
16K, C = 0 / 1 / 8, **rep-major** (`RK_INTERLEAVE=1`: every C once a rep, so the heat engine's drift falls
on every C alike), 6 reps reading 3-6, warmed, quiet host (every rep: load <= 1.16, 0 compilers -- now
recorded per rep). Per-step wall time from the step lines' docker timestamps (`tools/step_times.py`;
unprofiled), "big" = the prompt's 2,048-row chunks, "short" = the 64-row checkpoint remainders decoders
cause.

| arm | C | prompt_ms | speedup | **R55 ratio** | big chunk ms (n 8) | short ms (n 8) | decoder gap median / p90 / max ms |
|---|---|---|---|---|---|---|---|
| exact | 0 / 1 / 8 | 9,689 / 10,090 / 10,330 | 1 | | 1,211 / 1,200 / 1,200 | - / 61 / 92 | - · 64 / 1,207 / 1,212 · 110 / 1,208 / 1,212 |
| quality (vram) | 0 | 6,527 | 1.484 [1.475, 1.494] | | 816 | | |
| quality (vram) | 1 | 6,858 | 1.471 [1.463, 1.483] | **0.991** | 798 | 60 | 69 / 736 / 1,379 |
| quality (vram) | 8 | 7,356 | 1.404 [1.349, 1.458] | **0.946** | 835 | 84 | 90 / 791 / 1,386 |
| speed (vram) | 0 | 4,154 | 2.333 [2.272, 2.350] | | **519 (lean)** | | |
| speed (vram) | 1 | 5,549 | 1.818 [1.791, 1.836] | **0.780** | **637 (masked)** | 57 | 67 / 540 / 1,368 |
| speed (vram) | 8 | 6,137 | 1.683 [1.525, 1.810] | **0.722** | **677 (masked)** | 90 | 102 / 599 / 1,373 |
| quality (host) | 0 | 7,041 | 1.376 [1.369, 1.387] | | 880 | | |
| quality (host) | 1 | 8,539 | 1.182 [1.177, 1.189] | **0.859** | 860 | **208** | **226** / 827 / 1,204 |
| quality (host) | 8 | 8,958 | 1.153 [1.089, 1.193] | **0.838** | 898 | **222** | **245** / 888 / 1,209 |

Reading:
- **Session 2's quality R55 (0.85-0.89) was the instrument**: its block order measured all C = 0 reps
  first. Interleaved, quality keeps 0.991 / 0.946 at 16K -- in band. Its big chunks barely move with
  decoders (816 -> 835 ms) and its 64-row remainders are cheaper than stock's; what decoders add is
  mostly those remainder steps, a near-constant absolute cost in both arms, which shrinks a large ratio.
  32K and C = 4 re-measured the same way in session 5.
- **Speed: R55 RED, mechanism measured**: one decoder moves every bulk chunk from the lean fill (519 ms) to
  the masked path (637-677 ms). Kept 0.78 / 0.72 of its speedup (still 1.68-1.82x stock).
- **R56's longest stall = the projector-VRAM residual, confirmed**: with the projector in host memory the
  decoders' longest gap is stock's (1,204-1,209 vs 1,212 ms); in VRAM it is +13% (1,368-1,386 ms) --
  the prompt's exact chunk on a server holding ~1.2 GiB less expert VRAM a card.
- **NEW (host placement + decoders): decoders' median gap during an approximated prefill is 2.2-3.5x
  stock's** (226-245 vs 64-110 ms): each 64-row remainder pass costs ~210 ms masked vs 61-92 ms stock,
  because a masked pass streams every late layer's projector map whatever its row count. A cost to OTHER
  users, not on the allowed list ("host = slower ON requests"). Plugin-only fix proposed to the
  coordinator (not built: Stage E owns placement): approximate a pass only when its bulk rows (b − s_lb,
  keyed) exceed the placement's break-even -- ~650-700 rows from these step times for host (masked ≈ 187 +
  0.33 ms a row vs stock ≈ 24 + 0.57); none needed for vram (64-row masked 57-84 ms vs stock 61-92).

### Session 4 -- R60 speculating decoders, drift, steady-state decode (from 13:42Z; `session4.log`)
Servers with `--num-speculative-tokens 3` (decoders verify 4 rows a step; draft passes run stock),
`conc.sh text` (the batched arrangement), 3 reps, D = 1 and 4:

| mode | approximate steps | decoder texts vs off (same arrangement) | solos |
|---|---|---|---|
| off | 0 | (reference) D=1 `49309ffb…`; D=4 `bd6ac2fc… 0ff98137… 7e8fd42c… 0d4f722f…` | `2a82de60…` etc. = the non-speculative solos |
| quality | 162 | **identical, 15 / 15 concurrent texts** | identical |
| speed | 162 | **identical, 15 / 15** | identical |

The server-wide draft counters differ (D=1: off 200/342, quality 179/402, speed 183/390) but a batched
request's draft counts sum ALL its choices (`core/server/server.cpp` `timings_of`), the approximated
prompt's own 256 generated tokens included -- whose prefix is approximated by design (its MTP head reads
the layer-S stream for bulk rows: Stage D's `final` map, DD-D). The decoders' own acceptance is a
function of their own tokens and exact hidden states, both identical here; it is measured directly
(decoders as separate requests, each with its own `draft_n`) in session 5 (`conc.sh accept`, 14e1cf7).

Drift check (exact again at the end, session 2's shape): 32K solo 19,348-19,612 ms (session 2: 19,382),
C = 8 20,606-21,275 (one rep 23,085) vs 20,660 -- stock was stable through the day.

Steady-state decode, no prefill (radiance `scripts/cstep2.sh`, read only; `--max-num-seqs 9`; two
windows each, after a warm pass):

| server | C = 1 ms/step | C = 8 ms/step |
|---|---|---|
| exact (stock) | 13.94 / 13.94 | 14.44 / 14.19 |
| quality, projector in VRAM | 13.92 / 13.94 | 14.47 / 14.19 |
| quality, projector in host memory | 13.82 / 13.82 | **26.54 / 15.42** |

An ON server with the projector in VRAM decodes exactly like stock; session 2's 2-second "alone" gaps
were noise. With the projector in host memory, decode at C = 8 is +87% in the first window and +8.7%
after -- a decode-only step runs no KVA op, so this is the placement's effect on the expert tiers (host
pool / SSD), Stage E's to explain; reported.

### Session 5 -- R55 quality, interleaved, both lengths (14:21-15:1xZ; `session5.log`)
Same instrument as 2c (rep-major, 6 reps reading 3-6, warmed, `--max-num-seqs 9`), C = 0 / 1 / 4 / 8,
16K and 32K, exact vs quality (VRAM placement).

| length | exact prompt_ms C 0/1/4/8 | quality prompt_ms C 0/1/4/8 | speedup C 0 | **R55 ratio C 1 / 4 / 8** | R56 KVA/exact median gap C 1 / 4 / 8 |
|---|---|---|---|---|---|
| 16K | 9,703 / 10,129 / 10,222 / 10,278 | 7,190 / 7,762 / 7,939 / 8,136 | 1.350 [1.327, 1.359] | **0.967 / 0.954 / 0.936** | 1.51 [1.29, 1.57] / 1.41 [1.08, 1.65] / 1.27 [1.01, 1.59] |
| 32K | 19,415 / 20,317 / 20,578 / 20,998 | 12,572 / 13,619 / 14,143 / 14,799 | 1.544 [1.520, 1.551] | **0.966 / 0.942 / 0.919** | 1.45 [1.23, 1.50] / 1.26 [0.98, 1.44] / 1.09 [0.86, 1.57] |

Per-step (step_times.py): quality big chunks 16K 899 / 882 / 895 / 911 ms, 32K 786 / 765 / 789 / 817 ms
(C 0/1/4/8) -- decoders barely move them; exact 1,202-1,213 ms throughout.

Reading:
- **R55 for quality: GREEN at every cell** (0.919-0.967 here; 0.991 / 0.946 at 16K in 2c). Session 2's
  0.85-0.89 was the block-order instrument.
- **Hygiene note**: another lane's build ran during the EXACT arm's first 16K reps (14:25-14:28Z, up to
  111 compiler processes, load 29.9 -- recorded per rep). Exact's prompt times did not move (C 0:
  9,689-9,724 ms vs 2c's 9,689), so the baseline stands; the quality arm ran on a quiet host (load <=
  1.25, 0 compilers).
- **This quality server was slower than 2c's** (16K solo speedup 1.35x vs 1.48x; 32K 1.54x vs session
  2's 1.81x), and its decoders were slower even BEFORE the prefill (decode-only gap 15.2 / 32.3 / 37.0
  ms vs exact 11.6 / 17.6 / 23.3). That decode-only slowness is a server-level effect -- the same
  symptom Stage E reports for plain decode on quality servers (their lane) -- not the mixed-step
  mechanism; it is also why R56's ratio is > 1 in this session (1.09-1.51) while 2c and session 2 read
  0.8-1.07. Session 4's steady-state decode on a VRAM-placed quality server read equal to stock
  (13.92 vs 13.94 ms at C = 1). So decode speed on an ON server varies from boot to boot; R56 cannot be
  called green until E's diagnosis lands.

**R60 acceptance half (session 5, 14:5xZ)**: `conc.sh accept`, `--num-speculative-tokens 3`, the 32K
prompt then 4 decoders as SEPARATE requests 0.5 s later, 5 rounds:

| server | decoder 0 | decoder 1 | decoder 2 | decoder 3 | all 20 answers |
|---|---|---|---|---|---|
| off | 181/222 ×5 | 163-165/270-276 | 159-160/285-288 | 164/279 ×5 | **3,344 / 5,298 = 0.6312** |
| quality | 181/222 ×4, 180/225 | 163-164/273-276 | 160/285 ×5 | 164/279 ×5 | **3,342 / 5,304 = 0.6301** |

Every decoder's text identical across rounds and across the two servers (4 distinct texts per arm,
the same 4). **R60 GREEN**: texts identical to off (batched arrangement, sessions 4; separate requests,
here), acceptance within the round-to-round spread of each arm.

## 4. Decisions of 15:10Z (Dylan via the orchestrator) and what was built

- **R55 speed: option (A), "decoders full, bulk lean"** -- 173dfc4. `PATH_DECODERS` in `kva_plan.h`: speed,
  one prefill entry whose chunk is all bulk, decoders beside, every late attention layer on the per-row
  sparse form (`straddle_ok`), the stager lever on, not `MASK=all`. Per late layer (`decoders_layer`,
  `kva_layer.h`): the bulk rows [DT, n) get the lean pieces (projector from the layer-S stream + codes,
  K/V or the delta net's projections + corrected scan); the decoder rows [0, DT) get the in-tree layer
  over that range with the in-tree helpers' r0/rows -- connection read/write, attention per row (the
  query path over every row, as A.1's straddle: an M-RoPE plane cannot be column-sliced), the delta
  net's decode half verbatim and its out-projection over [0, DT), MoE over [0, DT) (`moe_layer` gained an
  end row) with the stager probes so their experts stream. The decoders' dense GEMMs run at M = DT (the
  narrow kernel a decode-only step uses): byte-identity to off is NOT required on this path (Dylan); the
  bar is stock's own solo-vs-batched variation. Quality, plumb, two prefills, straddles, dense-attention
  shapes, `MASK=all` and `TAIL_ONLY=0` keep the masked path. Static: the in-tree-step oracle
  (`decoders_expected`) at TP1 and TP2 rank 0, D 1/4, n_spec 0/3 (54 cases, 680,899 checks); mutants
  D1-D12 all caught (`evidence/stageB/mutants-decoders.txt`; D6 "decoders path in quality" needed a
  planner-level case, as A.1's A10 did).
- **Gate 1's instrument** -- fd9e174. The server refuses `logprobs` (HTTP 400), so
  `RADIANCE_KVA_DUMP_LOGITS=<dir>` (debug, read at declare, any mode incl. off, issues nothing) dumps
  each live pass's logits rows on every rank with each row's sequence/position/token;
  `tools/logit_capture.py` sends the batched arrangement and each decoder alone and records which dump
  lines belong to which request; `tools/logit_compare.py` gives a decoder's per-token KL(A||B) and top-1
  agreement while the two sides' inputs agree. Floor = off batched vs off solo (the ksplit-from-M
  class, same session); candidate = speed batched vs off batched; quality batched vs off batched must be
  byte-identical. Captures run with `--profile-ops` (no replayed pass skips step()).
- **R56 host guard: GO** -- 185e31b (planner rule, keyed `b - s_lb`, vram no threshold).
- **VRAM placement refusing to start at 8192 / 10 seqs** -> Stage E (no placement code touched here).
- Prediction HB-R55-speedA registered before any run (labbook seq 418).

### Session 7a / 8a -- R6/R7 and gate 1 for the decoders path (15:39-16:05Z; home `data/home-fd9e174`:
### arch fceea9dd…, kva.so 9d80ee76…; host placement from 15:40Z, Dylan: the VRAM placement is removed)
- **R6/R7 green at fd9e174**: off ident = R3's six hashes (`ident-off-fd9e174.txt`, session 7a; 7a was then
  stopped and its captures rerun with host placement as session 8a).
- **Gate 1 (session 8a)**: `--profile-ops` servers with `RADIANCE_KVA_DUMP_LOGITS`, `RADIANCE_KVA_PROJ_PLACE=host`;
  one request [4 decoders, 32K prompt] (128 tokens) + each decoder alone. Speed took the decoders path on 14
  of its 15 approximate steps (1 masked: the step where the decoders' prompts are prefilled beside the
  first chunk -- no decoder is decoding yet, Pn 5). `tools/logit_compare.py` follows each decoder as a
  greedy chain (9697d44: a batch index stops naming the same sequence once the long prompt decodes).

| decoder | floor: off batched vs off solo -- mean KL / max KL / top-1 (positions) | **(A) speed batched vs off batched** -- mean KL / max / top-1 (positions) | quality batched vs off batched |
|---|---|---|---|
| 0 | 0.737 / 0.737 / 0.00 (1) | **0.0013 / 0.029 / 0.963 (27)** | byte-identical (54) |
| 1 | 0.097 / 0.097 / 0.00 (1) | **0.0029 / 0.029 / 0.987 (76)** | byte-identical (129) |
| 2 | 0.316 / 16.3 / 0.981 (52) | **0.0016 / 0.035 / 0.990 (105)** | byte-identical (129) |
| 3 | 0.023 / 0.044 / 0.75 (4) | **0.0005 / 0.002 / 0.929 (14)** | byte-identical (129) |

  **Gate 1 GREEN**: every decoder's mean KL vs off in the same arrangement is below stock's own
  solo-vs-batched difference for that decoder (33-567x), its max KL below the floor's, its top-1 disagreement
  (one flip, at the position where its text then diverges: after 26 / 75 / 104 / 10 generated tokens)
  at or below the floor's. Quality (unchanged path) stays byte-identical to off. Reading the floor: the
  stock engine's logits for the SAME prompt differ by 1-2 logit units between "prefilled beside a 32K
  chunk" and "prefilled alone" (decoder 0: top token 1946 at 18.82 vs 198 at 19.03) -- a large class,
  stock's own, measured here, not assumed.

### Session 8b/8c/8e -- R55 / R56 / R57 on the deployed configuration (projector in host memory, guard
### 1,024 bulk rows), home `data/home-fd9e174`, interleaved, 6 reps reading 3-6 (16:00-17:10Z)

| arm | length | C 0 / 1 / 4 / 8 prompt_ms | solo speedup | **R55 ratio C 1 / 4 / 8** | decoder gap median, KVA / exact (ms) C 1/4/8 | longest gap KVA / exact (ms) |
|---|---|---|---|---|---|---|
| exact | 16K | 9,696 / 10,127 / 10,222 / 10,336 | 1 | | 66 / 81 / 101 | 1,213 |
| exact | 32K | 19,418 / 20,304 / 20,585 / 21,184 | 1 | | 76 / 90 / 130 | 1,219-1,226 |
| **speed (A)** | 16K | 5,089 / 5,573 / 5,749 / 5,922 | 1.905 [1.870, 1.923] | **0.954 / 0.933 / 0.916** | 85 / 94 / 109 | 1,204-1,213 |
| **speed (A)** | 32K | 8,534 / 9,674 / 10,046 / 10,274 | 2.275 [2.240, 2.292] | **0.922 / 0.901 / 0.906** | 100 / 110 / 113 | 1,197-1,204 |
| quality | 16K (8c; 32K stalled, below) | 7,895 / 8,473 / 8,601 / 8,996 | 1.228 | 0.973 / 0.968 / 0.936 | 102 / 109 / 127 | -- |

Per step (step_times.py), speed beside decoders: big chunks 620-644 ms (16K) and 524-547 ms (32K) vs
533-636 ms lean solo -- the decoders path costs the long prompt almost nothing; stock's big chunk is
1,202-1,214 ms. The 64-row remainders now run the stock step (guard): 76-96 ms on this ON server vs
64-117 ms on the stock server.

- **R55 GREEN for speed (A) and quality** on the deployed configuration (every cell >= 0.90; speed 32K
  C = 4 is 0.901). Prediction HB-R55-speedA: ratio confirmed (>= 0.9); big-chunk cost confirmed (predicted
  530-570 ms at 32K: 524-547).
- **R56: longest gap and p90 GREEN, median RED.** Decoders' longest stall <= stock's in every cell (speed:
  1,197-1,213 vs 1,213-1,226 ms) and their p90 is less than half of stock's (484-570 vs ~1,210 ms); but
  their MEDIAN gap is 8-31% above stock's in 5 of 6 speed cells (32K C=8: 0.87) and 26-54% for quality.
  The median sits on the 64-row remainder steps, which run the STOCK step on an ON server with the
  projector in host memory, and those cost more than on the stock server (also visible before the
  prefill: session 4's decode-only C = 8 15.4 vs 14.2 ms, +87% while warming). That is Stage E's
  decode-cost finding on host placement, not the mixed-step path; R56's median goes green only with E's
  fix. With the guard, the 2.2-3.5x of session 2c is gone (median now 1.05-1.54x).
- **R57 GREEN** with the guard (`conc_steps.py --min-bulk-rows 1024`: every expected step logged or
  replayed, none unexpected; the guard's stock remainders are expected stock).
- **Open: a stall under load, cause unknown.** Twice at 32K with C = 8 (speed 8b at rep 1; quality 8c at
  rep 0), the eight new streaming decoders did not reach 16 tokens within 120 s right after a 32K C = 4
  rep; the speed rerun (8e) completed every rep. Kernel log clean both times. The server logs were
  removed with the containers; conc.py now prints the decoders' errors and the running/waiting counts on
  this failure (000c6c2) and sessions follow the server log (`docker logs -f`), so the next occurrence
  leaves evidence. Seen in two modes, so not tied to the decoders path.
- **Not run (pace change, 17:20Z):** R60 on the decoders path (speculating decoders now take it in speed
  mode; static R53'/n_spec 3 shapes green; the session `sessions/s8d.sh` is written); quality 32K
  interleaved on host placement.

## 5. Final row table (stage-b @ merge 2899f91; plugin code at fd9e174 = frozen home `data/home-fd9e174`)

| row | state | evidence |
|---|---|---|
| R53′ | **green** static: 16 mixed shapes + decoders path (TP1/TP2, D 1/4, n_spec 0/3), 54 cases; mutants B1-B16, K1-K2, G1-G5, D1-D12 all caught | §2, §4 |
| R54 | **green**: masked path (quality; speed before A) byte-identical to off, 20/20 + 20/20; MASK=all control diverges; speed (A): gate 1 below stock's own floor (mean KL 0.0005-0.0029 vs 0.023-0.74), quality byte-identical | sessions 1, 8a |
| R61 | **green** (masked path): 62/62 mixed steps decoder slots byte-identical, both modes; the decoders path is gate-1-checked instead (bytes may move by design) | session 1 |
| R57 | **green** (C 0/1/4/8, both modes, guard-aware) | sessions 2, 8e |
| R58′ | **green**: both tails at stock's two-prefill floor (2,048), better at 8,192; 31 Pn-2 masks, 0 earlier rows approximated | sessions 3, 3b |
| R55 | **green** on host placement: speed (A) 0.901-0.954, quality 0.936-0.973 (16K) / 0.919-0.967 (vram, both lengths, session 5) | sessions 5, 8 |
| R56 | **partial**: longest gap and p90 <= stock; median +5-54% above stock on host placement = E's ON-server stock-step cost | sessions 2c, 8 |
| R60 | **green** on the masked path (texts identical, acceptance 0.6312 vs 0.6301); **not run** on the decoders path | sessions 4, 5 |
| R6/R7 | **green** at 58ac8d4 and fd9e174 (off ident = R3) | sessions 1, 7a |
| R56 host guard | built (185e31b), median gap 2.2-3.5x -> 1.05-1.54x stock | session 8 |
