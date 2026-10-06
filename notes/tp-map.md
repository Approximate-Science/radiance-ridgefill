# notes/tp-map.md -- TP 1 / 2 / 4 map for the RidgeFill plugin on radiance 1.0.8 (report only)
> Report by an open-model worker (GLM-5.3 via opencode-go), written against main at 9af94f7 and spot-checked by the
> orchestrator (the engine facts in summary items 1, 4, 5 verified in the radiance source). Summary item 9 is STALE:
> the int8 projector is built on branch stage-e (Stage E), which this map did not see. TP1/TP4 memory figures are
> estimates, as marked.


Model facts used throughout (Qwen3.8-Flash-Next): 48 layers = 12 x (3 GatedDeltaNet + 1 QSA attention)
(rad:arch/qwen4exp_fp8/qwen4exp_fp8.cpp:22-23); hidden 2560, hc_count 4 (stream 10240), 512 experts of
640 at top-10 + shared 640, 24 query heads + 2 KV heads at head_dim 256, vocab 248320
(rad:qwen4exp_fp8.cpp:24-26); GDN 16 key heads / 48 value heads at 128/128 (rad:docs/ARCHITECTURES.md:73-74).
Measured config is always `--tp 2` (scripts/common.sh RK_FLAGS, notes/gates.md:15). `rad:` =
/var/home/dylan/projects/inference/radiance at 140987f; other paths are this repo.

## 0. Summary (10 lines)

1. **TP1: works today** -- engine declares whole weights, no collectives (rad:core/format/share.cpp:23); plugin is
   world-aware and its static oracle already covers TP1 (tests/arch_static_test.cpp:484,1314). Only cost is memory: one
   32 GB card must hold the whole non-expert trunk (~7 GiB est., §2.3) and the full expert set (~2x TP2's per-rank slice).
2. **TP2: works today, measured** (NOTES.md:34, notes/gates.md, notes/aprime.md throughout).
3. **TP4: works with fixes X** -- the engine declares it (every divisibility check passes on the real head counts,
   §1.3) and no plugin code is hard-coded to two ranks (§3); fix X = extend the static test's world loops {1,2} -> {1,2,4}
   (tests/arch_static_test.cpp:484,744,1314,1447) and accept the §6 perf degradations. Not runnable on this 2-card box
   (rad:core/engine_bringup.cpp:2462).
4. **TP4 degrade (engine behaviour, not plugin-fixable)**: the fused all-reduce write exists only at world==2
   (rad:arch/common/rad_block_hc.h:297), so at TP4 every block issues its own `all_reduce`; MoE expert shares are
   unequal by parity at 4 ranks (rad:qwen4exp_fp8.cpp:495-498).
5. The projector maps are **replicated on every rank at every TP** (1,228 MiB/rank bf16); the only TP-scaled plugin
   weight is the correction head slice (54/27/13.5 MiB at TP1/2/4, arch/ridgefill_projector.h:217,224).
6. Mask / row selection is **identical on all ranks by construction** (pure function of the replicated batch + the
   full-vocab score table; no rank input, no broadcast) -- correct at 1/2/4 (notes/impl.md:241-246; kernels/ridgefill.h).
7. Host placement moves exactly 1.26 GB/pass/rank over PCIe at **every** TP (notes/aprime.md:202); on rank 1's Gen4 x4
   (~7.2 GB/s, machine fact) that is the measured +6%/+10% TTFT (NOTES.md (d)).
8. A **per-rank-sharded projector is possible plugin-side** (sliced `ridgefill_gemm_nt_bias` + the in-tree `all_reduce` row):
   saves 614 MiB VRAM and half the GEMM at TP2 but adds ~480 MiB/pass of all-reduce wire -- net slower on this box,
   ~time-neutral only on symmetric fast links (§4.3).
9. The **int8 projector (~0.61 GiB/rank, Stage E, unbuilt)** is blocked plugin-side on the forwarded row's dtype
   constraint -- the engine matches `dtype in {bf16}` only (notes/aprime.md:221; kernels/forward.cpp:41-42).
10. Nothing in arch/ or kernels/ is hard-coded to two ranks (grep §3.1); the two-rank assumptions live in the engine
    (rad:rad_block_hc.h:297) and in the test loops (tests/arch_static_test.cpp:484,744,1314,1447).

## 1. Engine support: does radiance 1.0.8 run qwen4exp_fp8 at this TP?

### 1.1 Flags and bounds
- `--tp N`, N >= 1 (rad:core/config.cpp:37,341,656); `--tp-wire exact|wht6` (rad:core/config.cpp:38,342; measured runs
  use exact, notes/gates.md:15). `sc.world_size = cfg_.tp` (rad:core/engine.cpp:142).
- TP N needs N usable devices of one architecture (rad:core/engine_bringup.cpp:2462).
- World sizes past 8 are not served: libr4d supplies 2/4/8-rank all-reduce (rad:core/device/host.cpp:56). So 1, 2, 4
  are all inside the served set.

### 1.2 How each thing splits (code)
- **Attention heads**: query heads divide by world or declare refuses (rad:arch/common/rad_arch.h:768). KV heads: when
  world > n_head_kv_all (2), world must be a multiple of it, then each rank declares n_head_kv = 1 and reads its own
  head's row-span via `kv_off` -- i.e. **KV heads replicate at TP4** (rad:rad_arch.h:787-799; the world-4 example is
  spelled out at rad:arch/common/rad_block_attn_gated_fp8.h:184-195: ranks {0,1} need KV head 0, ranks {2,3} head 1).
- **GDN heads**: key and value head counts each divide by world or declare refuses
  (rad:arch/qwen4exp_fp8/qwen4exp_fp8.cpp:393-394).
- **MoE experts**: the router stays replicated; each rank computes a column slice of every routed expert and the block's
  all-reduce sums the partials (rad:qwen4exp_fp8.cpp:66-72,490-497). An expert is 640 = 5 x 128-column blocks, split
  contiguously per parity over the ranks; declare refuses only when the ranks cannot each take a whole block of both
  parities (rad:qwen4exp_fp8.cpp:542-574). Whole-expert parallelism is used only for calibration / bf16 experts
  (rad:qwen4exp_fp8.cpp:516-529). The shared expert (640 columns) splits by whole 128-blocks with the all-reduce summing
  partials (rad:qwen4exp_fp8.cpp:411-433).
- **Hyper-connection stream (10240)**: replicated on every rank, every hc weight RAD_SHARD_NONE; the all-reduce rides
  in the hc write **only at world==2** (`ar_hc_write`, rad:arch/common/rad_block_hc.h:44-46,241-247,297); at other
  worlds each block declares and issues its own `all_reduce` (rad:arch/common/rad_block_gdn_fp8.h:379-381,511-512;
  rad:rad_block_attn_gated_fp8.h:422-424,560-561; rad:arch/common/rad_block_moe_fp8.h:1303-1305,1490-1491).
- **LM head / embeddings**: both row-sharded over the ragged vocab split (rad:qwen4exp_fp8.cpp:865-869;
  shard_extent/shard_begin rad:rad_arch.h:676-687,814-815); core restates and checks the split
  (rad:core/engine.cpp:121-142). The MTP draft head needs a 16-aligned vocab share (rad:rad_arch.h:825-833 area).
- **PLE (51B table)**: replicated, Tier::Mapped from the container file, host gather, no collective
  (rad:arch/common/rad_block_ple.h:27-32,155-171,218).

### 1.3 Divisibility vs the real counts -- does TP4 divide?
| item | count | TP1 | TP2 | TP4 | cite |
|---|---|---|---|---|---|
| query heads | 24 | whole | 12 | 6 -- divides | rad:rad_arch.h:768 |
| KV heads | 2 | 2 | 1 (split) | replicate: 4 % 2 == 0, n_head_kv 1 + kv_off | rad:rad_arch.h:787-799 |
| GDN key heads | 16 | whole | 8 | 4 -- divides | rad:qwen4exp_fp8.cpp:393 |
| GDN value heads | 48 | whole | 24 | 12 -- divides | rad:qwen4exp_fp8.cpp:394 |
| expert intermediate | 640 = 5x128 | whole | 2-3 blocks/expert (uneven) | 1-2 blocks/expert (uneven, documented) | rad:qwen4exp_fp8.cpp:495-498,542-574 |
| shared expert 640 | 5 blocks | whole | 3/2 blocks | 1-2 blocks + all-reduce partials | rad:qwen4exp_fp8.cpp:411-433 |
| expert count 512 | (split only for calib/bf16) | -- | 256 | 128 -- divides | rad:qwen4exp_fp8.cpp:520-529,465 |
| vocab 248320 | 62080 x 4 exact | whole | 124160 | 62080; 62080 % 16 == 0 (MTP ok) | rad:rad_arch.h:676-687,814; rad:qwen4exp_fp8.cpp:865-869 |
| hc stream 10240 | replicated | -- | -- | -- | rad:rad_block_hc.h:44-46 |

**Verdict**: TP1 and TP2 run (TP2 measured, NOTES.md:34 onward); TP4 is declarable -- no check refuses it on the real
counts -- but has never been run in this repo (unknown at runtime; 4 devices required,
rad:core/engine_bringup.cpp:2462).

### 1.4 Collectives per layer
- TP1: none (`op_ar` declared only at world>1, rad:rad_block_gdn_fp8.h:379).
- TP2: the hc writes carry the all-reduce (`ar_hc_write` x2 per layer) and the MoE output gather is fused
  (`ar_gather_hc_write`, declared only when the fused write exists, rad:rad_block_hc.h:297-310,316-356); measured op
  counts per late layer at notes/gates.md:157.
- TP4: no fused write (world==2 only) -> per block a separate `all_reduce` (rad:rad_block_gdn_fp8.h:511-512,
  rad:rad_block_attn_gated_fp8.h:560-561, rad:rad_block_moe_fp8.h:1490-1491) and the MoE gather is the plain
  `op_gather` (rad:rad_block_hc.h:316 returns early without `op_ar_write`). LM head: no per-layer collective; the
  sampler all-gathers top-k candidate pairs per sampled position (rad:core/sample/sampler.cpp:144,194,258-261).

## 2. Memory per card

### 2.1 What replicates vs splits (code)
- Splits by TP: attn qkv/out (RAD_SHARD_ROW stacked / COL, rad:rad_block_attn_gated_fp8.h:238-243,420), GDN
  in/out projections (rad:notes/arch.md:129-131 cites rad:rad_block_gdn_fp8.h:376-377), token_embd + lm_head
  (ROW, rad:qwen4exp_fp8.cpp:865-869), GDN a_log/dt_bias (ROW, notes/arch.md:147-148), shared expert (blocks,
  rad:qwen4exp_fp8.cpp:411-433), expert slices (rad:qwen4exp_fp8.cpp:490-497).
- Replicates at every TP: every hc weight (rad:rad_block_hc.h:241-247), router, conv1d, norms, PLE table and
  projections (rad:rad_block_ple.h:155-171,218-257), the 10240 stream itself (rad:rad_block_hc.h:44-46).

### 2.2 Measured TP2 anchors (per rank / per card)
- static non-expert weights in VRAM: 4.03 GiB (ON-server static 5.25 = 4.03 + 1.22 projector, notes/impl.md:181).
- experts resident via slab cache: 19.93 GiB, 16,847 slots (NOTES.md:34); ~16,880 slots stock vs 15,745 ON
  (notes/impl.md:181-182); ~18,371 folder-vram / 19,296 host-ring (notes/aprime.md:313).
- plugin held (folder, vram placement): 1.50-1.53 GiB = 1,228.1 MiB maps + 27 MiB correction + ~0.25 GiB buffers
  (notes/aprime.md:167-168,204,312); plumb "already held" 306.25/331.59 MiB (notes/aprime.md:312).
- plugin held (host placement): 434/430-459.62 MiB VRAM + 1,200.5 MiB host-mapped, ring 128.0 MiB VRAM
  (notes/aprime.md:204,249,312-313).
- host expert pool: `--host-pool-mib 12288` per rank (notes/gates.md:15).

### 2.3 Per-TP scaling (ESTIMATES marked; only the TP2 column is measured)
- static trunk: split portion scales 1/TP, replicated portion constant (§2.1). The replicated portion is not measured;
  from the per-weight sizes (hc ~0.62 GiB + router ~0.12 GiB + PLE projections ~0.06 GiB + misc, sizes from
  rad:rad_block_hc.h:123-124,227 and rad:qwen4exp_fp8.cpp:24-26) it is ~0.9-1.1 GiB, so **TP1 ~6.8-7.2 GiB ESTIMATE**
  and **TP4 ~2.4-2.6 GiB ESTIMATE** per card. Exact split: unknown from the notes.
- experts: a rank's expert bytes = full/TP (every rank slices every expert, rad:qwen4exp_fp8.cpp:490-497). TP2
  measured 19.93 GiB resident (NOTES.md:34); TP1 working set ~2x that on one 32 GB card (~half resident, more staging;
  ESTIMATE); TP4 ~10 GiB per rank -> plausibly all resident (ESTIMATE).
- attention KV (fp8): 12 layers x per-rank KV heads x 256 x 2 B/token = 3,072 B/token at TP2 and TP4 (1 head each;
  TP4's is replicated, rad:rad_arch.h:794-795), 6,144 at TP1; ~2.4 GiB / 8 seqs x 49,152 ctx (ESTIMATE from shapes,
  rad:qwen4exp_fp8.cpp:24-26, notes/gates.md:15 `--max-model-len 49152`).
- GDN state: 36 layers x per-rank value heads x 128 x 128 x 4 B = 108/54/27 MiB per seq at TP1/2/4 (shape
  arch/ridgefill_declare.h:88; f32, arch/ridgefill_declare_masked.h:201-206).

### 2.4 Plugin per-card cost (per rank; maps are REPLICATED at every TP, arch/ridgefill_projector.h:249-266)
Pieces: maps+biases 1,228.1 MiB bf16 (notes/aprime.md:168); correction slice 54/27/13.5 MiB at TP1/2/4
(arch/ridgefill_projector.h:217,224; 18 layers x 48/world heads x 128x128x4 B); row tables (score*) 0.95 MiB, quality only,
full vocab replicated (arch/ridgefill_projector.h:83-86,241); boundary/plugin buffers ~0.25 GiB (b_hs
[max_tok, 10240] bf16 ~40 MiB at max_tok 2048, arch/ridgefill_declare_masked.h:33; x_P, mask, bounds, probe rows --
notes/aprime.md:167-168 difference); ring slots 2 x 50.0 MiB VRAM, host placement only (arch/ridgefill_projector.h:199).

| placement | TP1 | TP2 (measured) | TP4 |
|---|---|---|---|
| bf16, vram | ~1.53 GiB VRAM | 1.50-1.53 GiB (notes/aprime.md:204,312) | ~1.49 GiB VRAM (only the 13.5 MiB correction differs) |
| bf16, host | 1,200.5 MiB host-mapped + ~460 MiB VRAM | 434/430-459.6 MiB VRAM + 1,200.5 host (notes/aprime.md:204,249) | 1,200.5 host + ~420 MiB VRAM |
| int8, vram (Stage E, unbuilt) | ~0.92 GiB ESTIMATE | ~0.89 GiB (maps ~614 MiB + st + buffers) | ~0.88 GiB |
| int8, host (unbuilt) | ~600 MiB host + ~0.41 GiB VRAM | ~600 host + ~0.39 GiB (ring slots 2 x 25.6 MiB) | ~600 host + ~0.38 GiB |

The int8 column is blocked today: the engine selects the forwarded row only for `dtype in {bf16}`
(notes/aprime.md:221); kernels/forward.cpp:41-52 copies libr4d/libref `gemm_nt_bias` rows, so an int8 map needs a
forwarded int8 row plus an int8 folder format -- plugin-side work, listed as Stage E / notes/arch.md:377.

## 3. Plugin correctness per rank

### 3.1 Every place the plugin reads rank/TP (grep of arch/, kernels/, tests/)
- arch/qwen4exp_ridgefill.cpp:50 (rank-0 mode note), :62-66 (g_model[ctx->rank]), :212-213 (step), :220-224 (rank-0 dump),
  :235 (rank-0 log line), :261 (capture), :293-299 (state capture: heads [rank*H, (rank+1)*H), `world` in the jsonl),
  :314 (g_ridgefill[rad_rank(c)]), :323/:329 (rank-0 capture/log).
- arch/ridgefill_declare.h:77 (g_ridgefill[MAX_RANKS]), :34-36 (st per rank), :88 (d.n_head_kv = m.gcfg.n_head_v, per rank),
  :117-118/:129 (op params take the per-rank head count).
- arch/ridgefill_declare_masked.h:141-142 (upload_rank(ctx->rank), g_upload[ctx->rank]), :201-206 (state shape per-rank heads).
- arch/ridgefill_projector.h:55 (g_upload[MAX_RANKS]), :82 (st shape check x world), :153-171 (rank-named error paths),
  :217/:224 (per-rank correction bytes, slice offset `rank * heads`), :244 (st operand rows = per-rank heads),
  :249-266 (upload_rank, per-rank log), :273-277 (free).
- arch/ridgefill_layer.h:38 (k.st = this rank's heads), :51/:62/:85/:126 (per-rank n_head_v from m.gcfg),
  :181/:270/:312 (the blocks' own all_reduce guards), :210-218
  (project_masked: full n and wide per rank).
- arch/ridgefill_dump.h:225,244-254 (state file per rank, heads [rank*heads, (rank+1)*heads), world field).
- arch/ridgefill_fill.h:40,106 (H / n_head_kv from the per-rank block configs).
- **Hard-coded to two ranks: none found.** The only `world == 2` branch is in the engine (rad:rad_block_hc.h:297); the
  plugin's only "/ 2"-shaped constant is unrelated (arch/ridgefill_config.h:203, an env-parse bound).

### 3.2 Per mechanism
- **GDN correction head slicing**: correct at TP1 (rank 0 takes all 48 heads, offset 0) and TP4 (rank r takes heads
  [12r, 12r+12) -- `st->data + rank * heads` with heads = per-rank bytes, arch/ridgefill_projector.h:217,224). This matches
  the engine's contiguous GDN head split (notes/arch.md:296: heads [rank*H_local, (rank+1)*H_local), "the contiguous
  split of the delta net's own weights"). The folder check multiplies by world (arch/ridgefill_projector.h:82), so ONE
  correction.safetensors [48,128,128] serves every TP. 4 uploads: upload_rank runs once per rank per process
  (arch/ridgefill_projector.h:44-47,249) -- at TP4 that is 4 x 1,228 MiB of maps (correct, wasteful; §4.3).
- **Row selection / mask**: `ridgefill_mask` is a pure function of cu_last, token_ids, positions, the score table
  ([n_vocab_all], replicated) and the seed (schema notes/impl.md:239-246; kernels/ridgefill.h:96+). No rank input -> computed
  redundantly, byte-identically, on every rank; no broadcast exists or is needed. Correct at 1/2/4.
- **MoE drop-row masking under expert sharding**: at every TP all ranks route the same experts (router replicated,
  rad:qwen4exp_fp8.cpp:490-492) and compute a column slice of each, so the drop must be identical per rank -- and is,
  because it reads the same mask (arch/ridgefill_moe.h:72-74). Dropped slots cost nothing per rank (arch/ridgefill_moe.h:9-11,
  citing rad:libr4d/r4d_moe.hip:477-479,909-916,1580-1586). The probe issue is rank-consistent
  (arch/ridgefill_moe.h:25-51; probed_expert per rank rad:qwen4exp_fp8.cpp:465).
- **Tail-only straddle (speed)**: attn_straddle / gdn_straddle issue in-tree handles with per-rank head counts
  (arch/ridgefill_layer.h:279-341); at TP4 attention runs 6 query heads + 1 replicated KV head (kv_off, rad:rad_arch.h:794-795)
  through the block's own declared handles (arch/ridgefill_fill.h:101-114). Correct; the `op_ar` guards mirror the in-tree
  step (`if (d.op_ar && !ar_taken(...))`, arch/ridgefill_layer.h:181,270,312; arch/ridgefill_moe.h:139-141), which is what makes TP4's
  separate-collective world consistent.
- **Projector GEMM**: full [M, 2560] output per rank over the full [2560, 10240] map, no collective
  (arch/ridgefill_layer.h:210-218,252-257,343-347). Correct at every TP (the 10240 stream is replicated,
  rad:rad_block_hc.h:44-46); at TP4 it is 4x redundant compute (§4.1).
- **TP1 specifics**: no collectives anywhere (op_ar only at world>1, rad:rad_block_gdn_fp8.h:379); the plugin's TP1
  behaviour is already oracle-tested (tests/arch_static_test.cpp:484,1314; notes/impl.md:347 "quality TP1 + TP2").

## 4. Performance per TP derivable from code

### 4.1 Projector GEMM per rank (identical at every TP -- replicated)
- FLOPs per layer per pass: 2 * M * 2560 * 10240 (M = projected rows); at M = 2048 that is 107 GFLOP/layer, 2.58 TFLOP
  for the 24 late layers per rank per pass. Measured 1,464-1,737 us/call (notes/gates.md:74) -> 35-42 ms/pass/rank.
- Bytes per layer: map 52.4 MB (VRAM or ring slot) + h_S read M x 10240 x 2 B + output M x 2560 x 2 B
  (shapes arch/ridgefill_layer.h:212; arch/ridgefill_projector.h:70,199).
- These do NOT shrink with TP: each rank needs the full projected stream because the 10240 stream and hc weights are
  replicated (rad:rad_block_hc.h:44-46,241-247).

### 4.2 Host placement PCIe per approximated chunk per rank
- The ring moves exactly 24 x 2561 x 10240 x 2 B = **1.26 GB per approximate pass per rank**, one read of each map
  (notes/aprime.md:202) -- the same per rank at TP1/2/4 (maps replicate). On rank 1's Gen4 x4 (~7.2 GB/s, machine fact)
  that is a >= 175 ms/pass floor; measured TTFT +573 ms (+6%) at 16K, +1,405 ms (+10%) at 32K vs vram
  (NOTES.md (d); notes/aprime.md:196-199).

### 4.3 Per-rank-SHARDED projector (each rank holds 1/TP of the 2560 output rows of proj.L)
- Possible plugin-side: slice the proj_w operand in take_operands (arch/ridgefill_projector.h:236-244) to rows
  [rank*n/TP, (rank+1)*n/TP), keep each rank's bias columns, and issue the in-tree `all_reduce` row (world 2/4/8,
  rad:core/device/host.cpp:56) over the [M, 2560] block input x after the GEMM -- the same collective the blocks
  already issue (rad:rad_block_gdn_fp8.h:511-512). Each rank's GEMM output covers disjoint columns, so the
  all-reduce (over a zero-filled scratch, or per-slice writes into a zeroed x) reconstructs full x on every rank.
- Cost/saving at TP2 (bf16, vram, M = 2048): map VRAM 1,228 -> 614 MiB/rank (~+660 slab slots at ~0.93 MiB/slot,
  NOTES.md (d): ~970 slots per 0.88 GiB); GEMM time halves (~20 ms/pass saved, notes/gates.md:74); new all-reduce
  ~2 x 10 MiB x 24 layers = **~480 MiB/pass/rank of wire** -> ~66 ms at 7.2 GB/s. **Net slower on this box**
  (-46 ms/pass); roughly time-neutral only on symmetric fast links; the memory win is real. At TP4: map 308 MiB/rank,
  GEMM /4, all-reduce wire unchanged (full x width) -- wire grows relative to the saving. At TP1 there is nothing to
  shard into. Notes/arch.md:377 already lists "int8 or row-shard + all_gather" as the Stage-8 arms.

## 5. Testability on this box (2 x 32 GB AMD)
- **TP1 and TP2 run** (TP2 is the measured configuration, notes/gates.md:15; TP1 needs one device,
  rad:core/engine_bringup.cpp:2462).
- **TP4 cannot run**: the engine refuses --tp 4 with 2 usable devices (rad:core/engine_bringup.cpp:2462).
- **What can validate TP4 without cards**: tests/arch_static_test.cpp is parameterised by rank count -- served_ctx
  (rank, world) at :263-266, declare_pair (rank, world) at :821, and the world loops at :484, :744, :1314
  ({1,128},{2,128},{1,64}) and :1447. The harness records op names/params for any world and the oracle is the REAL
  in-tree step on the same ctx (masked_expected -> run_step(qwen4exp_fp8::step...), :1242-1247), so extending the
  loops to {1,2,4} exercises the real TP4 declare paths (KV replication, MoE parity split, per-block all_reduce) and
  the real plugin paths against the in-tree oracle. Limits: the mock builder accepts every op (no kernel registry),
  so it proves issue-list equivalence, not collective execution; libr4d's 4-rank all-reduce itself cannot be validated
  without 4 cards (unknown). Kernel tests are world-agnostic (ops take n_head as a parameter, notes/adapter-map.md
  kernels/ table; tests/kernel_test.cpp).

## 6. Ranked gap list (plugin-side fixes only; radiance source may not change)

**TP1** (correctness: no gap found -- §3.2; static oracle covers TP1 at tests/arch_static_test.cpp:484,1314):
1. **Memory pressure (rank 1)**: one card holds static ~6.8-7.2 GiB (ESTIMATE §2.3) + plugin 1.53 GiB (vram) + slab
   cache ~20 GiB + KV; the expert working set doubles vs TP2 (NOTES.md:34 x2). Plugin-side fix: default to host
   placement (built, `RADIANCE_RIDGEFILL_PROJ_PLACE=host`, notes/aprime.md:196-199) and build the int8 projector (Stage E,
   notes/arch.md:377) -- needs a forwarded int8 row in kernels/forward.cpp:41-76 (today the engine matches
   `dtype in {bf16}` only, notes/aprime.md:221).
2. **Unmeasured (rank 2)**: no TP1 run exists in the notes; TTFT/quality at TP1 unknown. Fix: run the §5 harness arms
   at --tp 1 on one card.

**TP4**:
1. **Static-test coverage (rank 1)**: the world loops stop at 2 (tests/arch_static_test.cpp:484,744,1314,1447), so
   the TP4 declare paths (KV replication rad:rad_arch.h:787-799; parity MoE split rad:qwen4exp_fp8.cpp:542-574;
   separate per-block all_reduce) are never exercised anywhere. Fix: extend the loops to {1,2,4} and add the
   correction-slice assertion at world 4 (the :739-780 case already computes `rank * heads * 4` from the per-rank
   model, so it generalises).
2. **Projector replication (rank 2, memory)**: 4 x 1,228 MiB = 4.9 GiB of VRAM across the cards for identical maps
   (arch/ridgefill_projector.h:249-266). Fix (optional, §4.3): per-rank sharded projector via sliced operands + the in-tree
   `all_reduce` row -- memory -921 MiB/card at TP4, but net slower on this box's asymmetric links; only worth it for
   the slab-slot headroom.
3. **Separate collectives (rank 3, perf, NOT plugin-fixable)**: no fused `ar_hc_write` outside world==2
   (rad:rad_block_hc.h:297) -> one extra `all_reduce` op per block per layer at TP4 (rad:rad_block_gdn_fp8.h:511-512).
   The plugin already mirrors this exactly (arch/ridgefill_layer.h:181,270,312; arch/ridgefill_moe.h:139-141), so it stays
   consistent; nothing plugin-side can recover the fusion.
4. **Unequal MoE shares (rank 4, perf, NOT plugin-fixable)**: at 4 ranks the parity split gives ranks 0/1 three blocks
   of alternate expert pairs and ranks 2/3 two (rad:qwen4exp_fp8.cpp:495-498) -> the all-reduce waits on ranks 0/1.
   Affects stock and approximate passes alike.
5. **Replicated attention KV head (rank 5, memory, NOT plugin-fixable)**: each of the 4 ranks reads a full KV head
   (rad:rad_arch.h:794-795) -- byte-identical KV weights on 2 ranks; per-rank attention KV cache equals TP2's.
6. **Unvalidated 4-rank collective path (rank 6, residual unknown)**: even with the static test extended, libr4d's
   4-rank all-reduce is not exercised without 4 cards (§5); no plugin-side fix can close this, only a 4-card run.
