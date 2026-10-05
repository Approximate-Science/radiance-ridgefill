# notes/split.md -- the core/adapter split: progress (account-B worker, 2026-10-05 ~20:30Z)

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

## Remaining (in order)

1. **Namespace move** (the rest of 5a; do it FIRST, as one mechanical commit): kva_config.h,
   kva_plan.h, kva_json.h, kva_folder.h, kva_match.h, kva_guard.h, kva_dump.h, then Kva/g_kva -> `namespace
   kva` with `using namespace rad::arch;` inside, and `namespace qwen4exp_kva { using namespace kva; }`
   so arch_static_test's `qwen4exp_kva::X` names keep resolving. Only then can the hook signatures
   (`declare_model`, `decl_state_ops`, `late_block`, `ffn`, `conn`, ...) name Kva in kva_adapter.h.
2. 5a hooks: decl_late_group + decl_kernel_ops -> qwen4exp_adapter.h behind `decl_state_ops`;
   quantiser -> `declare_model`.
3. 5b, 6a, 6b, 7 (delete the bridge case), 8, 9 (tests/adapter_core_test.cpp, toy 4-layer dense
   adapter; core_purity.cmake), 10 (docs/ADDING-A-MODEL.md from spec §6, README pointer).
4. Comment pass per file as it is touched (radiance arch density: why + cost, no noise). Done so far
   only in the new/edited blocks of steps 1-5a.
5. GPU gate (trimmed, ONE gpuq session, ~25-30 min): off ident once + KL `.rows` byte-identity for
   quality and speed on the **int8** folder (Dylan: int8 is the only shipped projector) vs main's
   recorded rows. Not queued yet -- the split is not finished.

## Also done this session

- notes/stagec.md: R66 recorded as CUT, "covered by R62 + E's held cost" (orchestrator).
- c3b waiter: not in the gpuq queue at 20:21Z (already gone; nothing to cancel).
