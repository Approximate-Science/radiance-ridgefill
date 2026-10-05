# notes/split.md -- the core/adapter split: progress (account-B worker, 2026-10-05; updated ~20:37Z)

Executing notes/adapter-split-spec.md on branch `split`, checked out in worktree
`radiance-kva-wt-b` (the first worktree, `radiance-kva-wt-split`, was outside the directories this
session's file tools may touch; I removed it instead of working around the denial). Host build:
`build-host` (`cmake -S . -B build-host -DCMAKE_PREFIX_PATH=<radiance-kva>/build-radiance-host/install
-DRADIANCE_SRC=<radiance>`).

## Base

`split` = main 7bbf161 merged with stage-e 5a3115b -> **1238dde**. Conflicts resolved: kva_config.h
(stage-e makes min_bulk_rows always the host default 1024; stage-c's ckpt_floor kept beside it),
kva_declare_masked.h (stage-c's decl_hazard, then stage-e's decl_final). After the merge: ctest -LE gpu
3/3, arch_static 63 cases / 1,125,462 checks green.

## Steps done (each one commit; every check = build + ctest -LE gpu 3/3 + arch_static all green)

| step | commit | what | deviation from the spec |
|---|---|---|---|
| 1 | 7e1f901 | guard takes the shadowed stem (`kShadowSo` in qwen4exp_kva.cpp) | none |
| 2 | 6f20ba5 | arch/kva_adapter.h (facts) + arch/qwen4exp_adapter.h `adapter_of(m)`; bridge case `the_adapter_facts_match_the_model` (TP1 + both TP2 ranks) | facts only: each hook joins in the step that moves its caller (a hook's signature names Kva/Pass/StateDump, which are not in namespace kva yet). `buf_x_q/s` = the producer's pair `a_x.cq()/cs()`, not `q8/s8`. Per-layer facts are owned `std::vector<uint8_t>`, not raw pointers. `straddle_layers`/`qsa_exact_to` not facts yet (they depend on the split; 5b) |
| 3 | 5a960fe | read_config(meta, c, min_tail = 512, default_tail = 2048); `Config::min_tail`; check_mode refuses below it | none |
| 4 | 1d3ac33 | projector (load_folder, check_tensors, row_block, plan_*, take_operands, upload_rank) reads `Kva::ad` | `check_tensors` stays CORE, not a hook: every shape it checks is a fact and the map dtype (bf16/i8) is the manifest's -- Dylan: "treat the map dtype as data from the folder manifest". New fact: `world` |
| 5a (facts) | 2d7a531 | check_mode (tile), check_fill (split_lo / calibrated / ext_in), decl_fill / decl_fill_i8 / decl_ring (wide, n_embd, dtype, buffers) | new fact `dtype` (op-param spelling). Still model-named in decl_fill: the quantiser declare (`k.quant.declare(b, g, m.a_x, n)`) -> `declare_model` hook |
| 5a (namespace) | 41e860f | kva_json/guard/folder/match/plan/config/int8/projector/dump + Kva/g_kva -> `namespace kva`; each header ends `namespace qwen4exp_kva { using namespace kva; }` so the static test is unchanged | `Upload` became ambiguous in take_upload (a second `Upload` is visible) -> qualified `kva::Upload` |
| 5a (hooks) | 777e48d | decl_late_group + decl_kernel_ops -> qwen4exp_adapter.h as `decl_state_ops`; the fill's quantiser -> `declare_model`; decl_fill takes no model | hooks read `g_model[ctx->rank]` (the model the declare passed before) |
| 5b | 51a86b3 | decl_probes / decl_projected / decl_masked / take_* / decl_hazard / capture_split / decl_state_read read facts; `declare_codes` hook (Kva::xp's code pair, same point in the declare order); straddle_missing -> adapter | `straddle_layers`/`qsa_exact_to` are per-layer FACTS (`straddle_lack[]`, `qsa_exact_to[]`) the core reduces from the split up, not adapter-computed scalars; new fact `buf_route_ids`. Still model-named: `decl_selected` (check_mode's max_tok, stage-e's decl_final) |
| 6a | 8a1658e | block pieces -> arch/qwen4exp_blocks.h; kva_fill.h -> qwen4exp_fill.h; kva_moe.h -> qwen4exp_moe.h | gdn_decoders (stage-b) moved with them |

## Remaining (in order)

1. kva_final.h (stage-e's MTP final map) and decl_selected: still take the model.
2. 6b (drivers over conn/late_block/ffn hooks), 7 (kva_step.h; delete the bridge case), 8 (rad_fake.h,
   folder_fixture.h), 9 (tests/adapter_core_test.cpp, toy 4-layer dense adapter; core_purity.cmake),
   10 (docs/ADDING-A-MODEL.md from spec §6, README pointer).
4. Comment pass per file as it is touched (radiance arch density: why + cost, no noise). Done so far
   only in the new/edited blocks of steps 1-5a.
5. GPU gate (trimmed, ONE gpuq session, ~25-30 min): off ident once + KL `.rows` byte-identity for
   quality and speed on the **int8** folder (Dylan: int8 is the only shipped projector) vs main's
   recorded rows. Not queued yet -- the split is not finished.

## Also done this session

- notes/stagec.md: R66 recorded as CUT, "covered by R62 + E's held cost" (orchestrator).
- c3b waiter: not in the gpuq queue at 20:21Z (already gone; nothing to cancel).
