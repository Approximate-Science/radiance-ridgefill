# notes/impl.md — IMPL lane: PLAN-FIX v2 Stage A (device mask, row-exact tail, stager lever, guard)

Owner: IMPL lane (2026-10-04, late evening). Plan: `~/AI-Work/radiance-kva-plugin-20261004/fix-246/`
(PLAN-FIX v2, REQUIREMENTS-FIX R45–R99, HANDOVER-FIX v2). radiance `140987f` (v1.0.8), read only.
Decisions in force (orchestrator, plan defaults): DD-F device mask replaces compaction (deleted, not
flagged); DD-B stager lever behind `RADIANCE_KVA_STAGE`; DD-C both straddle variants
(`RADIANCE_KVA_STRADDLE=split|end`); DD-E forward on a version mismatch. DD-A (hazard counter,
`kva.ckpt_floor`) is Stage C and DD-D (`final`) Stage D: neither is built here.

## 1. kva.so schemas for Stage A (the arch side issues exactly these)

Operands are POSITIONAL. `?` = optional (`RAD_NONE`). Every param is required. No param carries
`RAD_PROLE_SEQ_CHUNK` (R98) and, with `cap` gone, none carries `RAD_PROLE_CAPACITY` either.

**A kernel learns a per-step host number only through an operand's EXTENT.** The engine hands a
kernel the ranged parameter at its BAND's upper bound, not the issued value (radiance
`abi/rad_abi.h:54` "ranges collapsed", `core/runtime/issue.cpp:811-814`); an operand's row count is
per issue and is what a recorded pass replays. So `kva_mask` reads the bulk end `b` as the extent of
its `token_ids` operand, and `M` is issued at `b` only to select the band.

| op | params | operands (in order) |
|---|---|---|
| `kva_mask` (replaces `kva_rowsel`) | `M` (range, issued at b) · `share` f64 · `seed` int · `mode` str `none`/`class`/`random`/`all` | 0 `cu_last` in i32 [2] = `praw(cu_seqlens + n_seq−1, I32, 2)` → {s, e} · 1 `token_ids` in i32 [b] (rows [0, b) of the step; **its extent is b**) · 2 `positions` in i32 [b] or component-major [c, b] (row 0 at its strides; random mode only reads it) · 3 `score` **weight** f32 [vocab], optional (class/random need it) · 4 `mask` out i32 [n] (n = its extent = n_tok; every row written) · 5 `bounds` out i32 [4] |
| `kva_select` | `M` (range, rows) | 0 `mask` in i32 [n] (n = its extent) · 1 `x_src` in [n, w0] · 2 `q_src`? in [n, w1] · 3 `s_src`? in [n, w2] · 4 `x` inout [n, w0] · 5 `q`? inout [n, w1] · 6 `s`? inout [n, w2] |
| `kva_drop_rows` | `M` (range, rows) · `top_k` int | 0 `mask` in i32 [n] · 1 `ids` inout i32 [n, ≥top_k] |
| `kva_rho_update` | unchanged: `M` · `n_head` | unchanged 0–5 (`a` [n, n_head] at its strides, `mask`, `A_log`w, `dt_bias`w, `ND` inout, `state_idx` [1, pitch]) · **6 `bounds`? in i32 [≥1]** |
| `kva_state_correct` | unchanged: `M` · `mode` · `alpha` · `n_head` · `sd0` · `sd1` | unchanged 0–6 (`state`, `state_idx`, `applied`, `applied_idx`, `C`w, `ND`?, `nd_idx`?) · **7 `bounds`? in i32 [≥2]** |
| `kva_state_read` | unchanged | unchanged |

Semantics:

- `kva_mask`: s = cu_last[0], e = cu_last[1]; b = extent of `token_ids`; b′ = min(max(b, s), e).
  The window is W = [s, b′). `mask[i] = 0` for every row outside W. Inside W: `none` → 1 (every row
  approximated: speed mode); `all` → 0 (every row exact: plumb, and quality's all-rows oracle);
  `class` → a row MATCHES when `score[token_ids[i]]` is finite (an id outside [0, vocab) does not);
  k = rint(share × matches in W), half to even (Python's `round`, fnlev/rules.py); a row is kept
  (mask 0) iff it matches and fewer than k matching rows of W rank before it by (score descending,
  row ascending); every other row of W is 1. `random` → the same k, over ALL rows of W, ranked by
  `kva_row_hash(seed, positions[i])` ascending, ties by row; kept rows 0, others 1. This is
  `kva_rowsel`'s rule run over W instead of the whole chunk, with no cap and no `rows_idx`.
  `bounds = {s, b′, b′, e}` (the GDN split's two `cu` pairs, PLAN-FIX §4). Refused: s < 0, s > e,
  e > n.
- `kva_select`: for every row i < n with `mask[i] == 1`, row i of each present destination is
  overwritten with row i of its source, byte for byte; nothing else is written. Each (src, dst) pair
  is present or absent together, has one dtype and one row width (shape[1]); row pitches come off
  each tensor (last stride 1); both have ≥ n rows. Dtype-agnostic (bf16 x, int8 or E4M3 codes, f32
  scales).
- `kva_drop_rows`: for every row i < n with `mask[i] == 1`, `ids[i, 0..top_k) = −1`; nothing else is
  written (columns ≥ top_k and unmasked rows untouched). ids rows ≥ n, row pitch off the tensor.
- `kva_rho_update` + `bounds`: the recurrence runs over rows [lo, n) with lo = clamp(bounds[0], 0,
  n) and n = `a`'s extent; absent = [0, n), i.e. today. (The arch side issues `a` at b rows for the
  split variant and at n_tok rows for the end-of-chunk variant, with bounds = kva_mask's.)
- `kva_state_correct` + `bounds`: when present and `bounds[1] <= bounds[0]` the op writes nothing in
  either mode (the step's last sequence had no bulk row, so its correction stays as the previous
  chunk end left it); otherwise exactly today's op. **Deviation from PLAN-FIX §7**, which gave
  bounds to the rho update only: without it a sequence that never had a bulk row (possible only with
  two prefill chunks in one step, where the host cannot see s) would get alpha·C added by `apply`
  (rho is 1 while D is 0).

## 2. The stager lever as built: zero-row probes (plan DD-B's mechanism is not declarable)

**Evidence that the plan's alternate handle cannot exist** (`evidence/stageA/try1/r94-plumb-alt-1.json.log`,
plugin 5dad465, 2026-10-05 02:27Z): every serving mode refused at declare with

    E[0] declare: op 'moe_gemm_q': 0 weight operands declared; the schema has 0 fixed weight operands and 4 weight
    TABLE operands, so 0 must divide evenly among the tables and it does not.

The core needs >= 1 weight per table operand (`core/build/rad_builder.cpp:691-697`, `rest <= 0` refused); a
table's run is fixed by its declaration at issue (`core/runtime/issue.cpp:254-259`, "A table's run is fixed by
the declaration"); and declaring an op with a layer's real expert run after the graph widens that layer's
stager span to the end of the program (`rad_builder.cpp:1153-1170`, `stager.cpp:21-46`), which turns staging
off for EVERY pass (exact ones included) -- a stock-speed regression, so not an option. REFUTATION M1 read
`issue.cpp:136-138` ("a plugin may legally issue a table the declaration did not describe") but missed both the
declare-time count rule and the issue-time run check.

**What replaces it (stock handles only, no new declared expert use).** The stager reacts only to op HANDLES
(`before_op` on a layer's first expert op: wait, release a held buffer whose `read_` is set, `start(next)`;
`after_op` on its last: `read_ = true`; `start()` streams when the target buffer is held, `stager.cpp:87-168`).
So on a masked pass, right behind layer S-3's gate-up GEMM -- when the stager holds S-3 (waited) and S-2
(copying) and has released neither -- the plugin issues the gate-up handle of every layer from S-1 to the last
as a PROBE: one token, every expert offset zero (`kva_mask` writes them into `kva_zero_offsets`), output into
the plugin's `kva_probe_rows`. With zero offsets the kernel treats every slot as dropped: it reads no weight and
writes zeros into its own output rows only (`libr4d/r4d_moe.hip:909-916`). Trace (S = 24):

| issue | stager | effect |
|---|---|---|
| op_gu(21) (real) | wait 21; release 20; start(22) -> staged | holds {21, 22} |
| probes op_gu(23), op_gu(24) .. op_gu(47) | start(24) .. start(48): target buffer held -> streamed, `started_` set | no copy |
| op_dn(21) | read_[21] | |
| op_gu(22) | wait 22; release 21; start(23) -> staged (S-1 runs every row) | holds {23, 22} |
| op_gu(23) | wait 23; release 22; start(24): already started | nothing |
| op_gu(24..47) | release; start(next): already started | every late layer zero-copies only its routed experts |

Correctness never depends on it: a probe moves no number (R94 checks bytes); placement can only make it fall
back toward today's staging (a layer with no non-resident unit holds no buffer). The MTP head's routed layer is
also started (streamed) by the probe of layer 47, so it is not staged for nothing (REFUTATION §7.3). Cost: 25
tiny launches and residency resyncs per masked pass. `RADIANCE_KVA_STAGE=stock` = no probes (the tested
fallback); `RADIANCE_KVA_FORCE_STREAM=1` forces them (R94). Needs S >= 3. The lean fill is unchanged (no probes;
its one wasted copy of layer S is today's behaviour, measured in Stage 3).

**Instrument note.** The mover's end-of-run `h2d` counts the mover's own copies (staging, promotions), not the
kernels' zero-copy reads of streamed experts. R95's "h2d falls" therefore shows staging removed, not total PCIe
traffic; TTFT is the governing number.

## 3. What was built (commits 42ac7fd, 05c9c3a, 5dad465, 4d33aef, 0fd3a65, 68d5d92) and decisions with cost

- **DD-F**: compaction path deleted (`cap`, `h_R/x_R/y_R`, `kva_rows_idx`, gather/scatter, `kva_rowsel`). One
  mechanism: in-place device mask. `kva_mask` runs once a masked pass AHEAD of layer 0 (it reads only the batch;
  the plan put it after S-1 -- moved so the probes' zero offsets exist by layer S-3).
- **derive()** (`qwen4exp_kva.cpp`): stock | lean | masked from keyed fields only; lean = speed && D == 0 &&
  Pn == 1 && b == n_tok. `s_lb` uses `max_q_len_prefill` in a mixed phase and `max_q_len` in a pure prefill
  (the batch leaves the former 0 there).
- **Straddle (DD-C)**: `split` = scan [s,b) -> rho over [s,b) -> apply -> scan [b,e); `end` = one scan, rho
  over the whole chunk (mask 0 on the tail), apply after. In SPEED mode `end` applies an undecayed alpha (speed
  keeps no decay sums) -- so R50 is measured in quality mode.
- **kva_state_correct + bounds** (deviation from PLAN-FIX §7, which bounded rho only): a step whose last
  sequence has no bulk row (possible only with two prefills, where s is device data) must not get alpha*C added
  by `apply` (rho = 1 while D = 0). Undo and apply both skip when bounds[1] <= bounds[0].
- **Alignment**: a one-sequence approximated chunk off the 64-row tile fails the step by name (`rad_step_fail`);
  the scheduler never cuts one (non-final chunks start and end on the quantum, `scheduler.cpp:666-688`).
  `RADIANCE_KVA_FORCE_SPLIT` bypasses it on purpose (R47's negative control).
- **Debug switches** apply to every approximate pass: FORCE_SPLIT (b = n_tok - N), SHIFT_B (b += N, clamped),
  FORCE_STREAM, STRADDLE; all said loudly at declare.
- **Mode from RADIANCE_KVA only** (R81); the container's `kva.mode` is said ignored once (rank 0, real declare).
  KL mode (`max_out_rows > 0`) serves stock unless `RADIANCE_KVA_SCORE_BULK=1` (R73).
- **Tails past the step (R100)**: refused only when T > 2C - G (no chunk could hold a provably-bulk row); a
  capped n_ahead gives b = n_tok - ceil_G(T - C).
- **Release guard (DD-E)**: `rad_plugin_open` -> forward probe/declare/step to the in-tree file found on
  $RADIANCE_HOME after this plugin's own, or decline. No `chat_format` forward: neither file exports one at
  1.0.8 (a future release that adds one would serve its template's default while forwarded -- named residual).
- **Deferred by the orchestrator's decisions**: hazard counter / `kv_kva_meta` / `kva.ckpt_floor` (DD-A, Stage
  C); `final` map (DD-D, Stage D); `RADIANCE_KVA_MASK=all` (R54's negative control, Stage B).

### Static results (host, `build-host`, 2026-10-05)
`arch_static_test`: 40 cases, 464,789 checks, all ok. The oracle for every masked pass is the in-tree
`qwen4exp_fp8::step` issue list on the same batch with exactly the named substitutions (mask ahead of layer 0;
probes behind layer S-3's gate-up, each derived from that layer's own stock gate-up issue; the stream copy;
projection/select after each late connection read; undo/rho/apply around the last sequence's scan, split when
asked; drop between top-k and scatter). Cases: R52' truth table (22 rows), R73, stage switch + threshold,
plumb (stock / stream / forced split / 64-row fused top-k), quality TP1 + TP2 rank 0 + 64 rows, straddle
split/end/speed, mixed steps (2 decoders; two prefills; decoder + two prefills), tile refusal, lean fill,
R93 (first expert op = gate-up, last = down), R81, tail lift, guard scan/sha256/forward table.
Mutants (scratch copy, not committed), 21, each caught: M1 media guard dropped; M2 s_lb = DT; M3 probes after
the down GEMM; M4 no drop; M5 correction over every sequence; M6 never split; M7 quality takes the lean fill;
M8 KL mode approximates; M9 probes from layer S; M10 release scan ignores NULs; M11 undo without bounds; M12
projector over the whole step; M13 misaligned chunk served; M14 container kva.mode a switch; M15 probes read
the real offsets; M16 probes in layer S-2; M17 mask writes no zeros; M18 stage stock still streams; M19 a capped
n_ahead taken as enough; M20 the old one-step refusal; M21 refusal at 2C. (Names of the catching cases:
`mutate.py` output pasted in §4.)
`kernel_test` host: 737 checks; gpu (card 0000:13:00.0, 02:44Z): 945 checks incl. kva_mask zeros, all ok.

## 4. Engine gates (radiance 140987f, image stilldeadcode/radiance:1.0.8, boot 75e3e39b…, RK_FLAGS of
## scripts/common.sh, quick9 KL reference data/kld/ref-stage0, every KL candidate with SCORE_BULK=1 unless said)

### Session 1 -- correctness oracles (plugin 0fd3a65, home data/home-0fd3a65: arch 9d0fdf47…, kva.so 2cfe6699…;
### 2026-10-05 02:44-03:05Z; log evidence/stageA/session1.log)
Kernel device group first (card 0000:13:00.0, ROCR_VISIBLE_DEVICES=1): 945 checks, ctest gpu 1/1 passed --
kva_mask 192 configurations incl. the zeros output, kva_select 7.58 MB, kva_drop_rows 12 configurations, all ==
host bytewise.

| gate | run | approx steps | rows vs stage-0 exact | KL mean / max | rank-0 mover h2d |
|---|---|---|---|---|---|
| R94 boot 1 | plumb, FORCE_STREAM=1 | 67 | **identical** | 1.15e-7 / 1.21e-6 | 287.15 GiB |
| R94 boot 2 | same | 67 | **identical** | same | 287.02 GiB |
| R94 boot 3 | same | 67 | **identical** | same | 287.02 GiB |
| R94 control | plumb, STAGE=stock (no probes) | 67 | identical | same | **518.59 GiB** |
| R47 | plumb, FORCE_SPLIT=1024 | 67 | **identical** | same | 288.64 GiB |
| R47 negative | plumb, FORCE_SPLIT=1000 (off the tile) | 67 | **differ** | 10.0 / 19.7, top-1 0.0001, ppl x17,316 | 273.83 GiB |
| R35' | quality, ROWSEL_TABLE=all, SHARE=1 | 67 | **identical** | 1.15e-7 / 1.21e-6 | 294.32 GiB |
| R73 | quality, SCORE_BULK unset | **0** | identical | same | 529.19 GiB |

Reading: the probes move no number (R94 green, 3/3); the split scan with no correction is exact at a tile
boundary and the check can fail (R47 green, its negative control catastrophic, as an unaligned scan must be);
the whole masked path with every row exact reproduces stock (R35' green). The lever's mechanism is real: with
it the mover's copies fall from 518.6 to 287 GiB per run on the same arm, and come back with STAGE=stock
(R95's mechanism half; the mover counts its copies, not zero-copy reads -- §2). Stage-5 compaction for
comparison: quality all-rows 529.19 GiB.

### Mutation run (scratch copy of arch/ + tests/, 2026-10-05; `mutate.py`, not committed) -- catching cases
```
M1 derive ignores media rows: the_approximate_decision_truth_table
M2 s_lb is the decode rows only: a_mixed_step_corrects_only_the_last_sequence, the_approximate_decision_truth_table
M3 probes after the down GEMM: a_mixed_step_..., a_straddling_chunk_..., plumb_is_the_stock_step_..., quality_masks_rows_...
M4 bulk rows not dropped: a_mixed_step_..., a_straddling_chunk_..., quality_masks_rows_...
M5 correction over every sequence: a_mixed_step_corrects_only_the_last_sequence
M6 straddle never splits: a_straddling_chunk_..., plumb_is_the_stock_step_...
M7 quality takes the lean fill: a_chunk_off_the_tile_..., kl_mode_..., plumb_..., quality_masks_rows_..., truth_table
M8 KL mode approximates: kl_mode_serves_stock_unless_score_bulk
M9 probes start at layer S: a_mixed_step_..., a_straddling_chunk_..., plumb_..., quality_masks_rows_...
M10 release scan ignores the NULs: the_release_scan_counts_nul_delimited_copies
M11 undo without bounds on the masked path: a_mixed_step_..., a_straddling_chunk_..., quality_masks_rows_...
M12 projector over the whole step: a_mixed_step_..., a_straddling_chunk_...
M13 misaligned chunk served: a_chunk_off_the_tile_fails_the_step_by_name
M14 container kva.mode is a switch: the_container_mode_is_ignored_and_said_so
M15 probes read the real expert offsets: a_mixed_step_..., a_straddling_chunk_..., plumb_..., quality_masks_rows_...
M16 probes ride in layer S-2: a_mixed_step_..., a_straddling_chunk_..., plumb_..., quality_masks_rows_...
M17 mask writes no zeros: a_mixed_step_..., a_straddling_chunk_..., plumb_..., quality_masks_rows_...
M18 stage stock still streams: plumb_is_the_stock_step_through_the_masked_path, the_stage_switch_and_threshold_gate_the_lever
M19 a capped n_ahead taken as enough: a_tail_past_the_step_approximates_the_provable_rows
M20 the old one-step refusal: a_tail_of_two_steps_less_a_tile_or_more_is_refused, a_tail_past_the_step_...
M21 refusal at 2C, not 2C - G: a_tail_of_two_steps_less_a_tile_or_more_is_refused
```
Earlier arm, now obsolete with the alternates gone: "MoE down on op_dn always" and "alternate declared with
weights" were each caught by the alternate-handle cases before the lever changed.

### Session 2 -- policy gates, R45, the quality headline, R49-R51 (plugin 68d5d92, home data/home-68d5d92:
### arch d50fa23f…, kva.so 2cfe6699…; 2026-10-05 03:07-03:33Z; log evidence/stageA/session2.log)
- **R81 green**: `RADIANCE_KVA` set empty (the plugin reads empty as unset), container kva.mode=off: ident.sh
  = R3's six hashes; the log says once "the container's kva.mode=off is ignored; the mode comes from
  RADIANCE_KVA only". (Static: a container saying kva.mode=quality with the switch unset declares the in-tree
  graph, `the_container_mode_is_ignored_and_said_so`.)
- **R83 green**: engine binary patched 1.0.8 -> 1.0.9 (scratch copy, sha256 63332b00…, mounted over
  /opt/radiance/bin/radiance), mode quality asked: forwarded, ident = R3, log
  `WARNING: built against radiance 1.0.8, and the engine /proc/self/exe (sha256 63332b00…) carries release
  string(s) '0.46.1, 1.0.9' (1.0.8 0 times); forwarding to the engine's own architecture
  /opt/radiance/share/radiance/architectures/qwen4exp_fp8.so, KVA off`. With the installation's
  architectures/ hidden: `plugin 'qwen4exp_kva' declines this machine` then `E no architecture plugin claims
  'qwen4exp'` -- startup fails by name. Real binary: both plugins load (session 1, every run).
- **R84 green**: `--max-num-batched-tokens 1024`, tail 2048: refused, "kva.tail is 2048 tokens and the largest
  step is 1024 ... none is once the tail exceeds 1984 (2 x 1024 - 64)"; `RADIANCE_KVA_TAIL=1024` at step 1024
  serves (03:34Z re-run alone, evidence/stageA/r84-tail1024.serve.log; the in-session attempt hit a harness
  race right after the refused container exited). README names `--checkpoint-interval`.
- **R45 green** (engine half): quality + RADIANCE_KVA_DUMP + --profile-ops, prompts of 9216 / 10000 / 16384 /
  16385 / 32767 tokens: `scripts/mask_rule.py` -> every chunk's n_tok, n_ahead, b, bounds, full mask and exact
  rows equal the rule recomputed from N alone; b per straddling chunk 1024 / 1792 / -- / -- / 1984; 5 of 5.
- **Speed regression**: speed KL rows byte-identical to Stage 4's speed+st (home f60f893): the lean path
  is unchanged.

Quality, paired per doc vs exact, bootstrap 95% CI over the 9 docs (`scripts/kl_tail.py`; labbook records
stageA-*):

| run | scoring | dNLL vs exact | KL mean | top-1 |
|---|---|---|---|---|
| Stage 5 compaction quality (f60f893) | whole tail (2,047) | +0.02356 [+0.01152, +0.03630] | 0.0677 | 0.8832 |
| quality T 2048, masked path | whole tail | **+0.02282 [+0.01155, +0.03464]** | 0.0680 | 0.8838 |
| -- paired masked - compaction | | -0.00074 [-0.00246, +0.00046] | | |
| quality T 2048 | last 512 | +0.00576 [-0.01307, +0.02245] | 0.0592 | 0.8950 |
| **quality T 2560 (R100 HEADLINE)** | **last 512** | **+0.00212 [-0.01255, +0.01565]** | **0.0368** | **0.9156** |
| -- paired T 2560 - T 2048 | last 512 | -0.00364 [-0.01291, +0.00665] | | |
| speed T 2048 (lean) | last 512 | +0.02382 [+0.00292, +0.04388] | 0.0672 | 0.8874 |

Offset corpus (quick9 cut to k*2048 + 1024, its own exact reference data/kld/ref-off1024 recorded 03:2xZ, same
boot; exact vs it: 0 by construction):

| gate | comparison | paired difference |
|---|---|---|
| **R49 green** | dNLL(off1024, split) - dNLL(2048-cut), whole tail | +0.00399 [-0.00158, +0.00966] (CI includes 0, within +-0.01) |
| **R50** | split - end, whole tail | **-0.00876 [-0.01123, -0.00647]**: split better, in the -0.01..0 band -> keep split |
| R50 | split - end, last 512 | +0.00026 [-0.00231, +0.00292] |
| **R51 green** | SHIFT_B=+2048 - split, last 512 | **+0.00704 [+0.00123, +0.01277]**: worse, CI excludes 0 |
| R51 | SHIFT_B=-64 - split, last 512 | -0.00258 [-0.00563, +0.00051]: not worse |

R51's whole-tail number for +2048 (+1.01 nats) is not a quality reading: that scored window then contains
approximated rows, whose logits are not the model's (the SCORE_BULK caveat) -- which is why the last-512
protocol is the one to read.
