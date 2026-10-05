# notes/split.md -- the core/adapter split: progress (account-B worker, 2026-10-05; session 2 from 20:54Z)

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
| 6b (projection) | 74c5a85 | project_masked / project_bulk / project_beside / project / quantise read buf_stream / buf_x / buf_x_q / buf_x_s | none |
| 6b (drivers) | bc4431c | masked / straddle / decoders / fill_layer = core skeletons over new hooks `conn` / `late_block` / `ffn`; kva_layer.h -> namespace kva | `ffn` takes `to` (decoders' [0, DT)); dbg_resid lives in `conn` after a write (spec: inside conn/late_block) |
| (cleanup) | 1015dae | the per-header `namespace qwen4exp_kva { using namespace kva; }` trailers removed (kva_declare.h's directive covers every later reopening); guards KVA_*_H | -- |
| 9 (part) | 251ce01 | tests/core_purity.cmake = ctest `core_headers_name_no_arch` (host label) | SELF-TIGHTENING while the split runs: `KVA_PENDING` lists unconverted headers; fails on a model name outside the list AND on a listed header that became pure. Negative control (append `qwen4exp` to kva_json.h) fails naming the line |
| (purity) | d59b232 | kva_layer/plan/folder pure (citations reworded) | pending 8 of 15: config, declare, declare_masked, dump, final, guard, hazard, projector |
| (purity) | 61fe0c2 | the 48 core `radiance: qwen4exp_kva:` prefixes -> `radiance: %s:` of `kva::g_log_name` (new arch/kva_log.h; set by rad_plugin_open before the guard speaks, and by declare) | a global mirror of the `log_name` fact, because the guard/read_config/upload speak with no Kva in scope. Same served text: arch_static's stderr has 283 `radiance: qwen4exp_kva:` lines, 0 `radiance: kva:` |
| (purity) | 2491088 | kva_guard.h WHY block reworded | pending 4 of 16: kva_declare.h, kva_declare_masked.h, kva_final.h, kva_hazard.h |
| (purity) | 2df39a2 | kva_hazard / kva_final / kva_declare / kva_declare_masked -> namespace kva; decl_selected takes no model (check_mode reads ctx->max_tok = the model's max_tok at the real declare, geom_from); final map reads facts (hc = wide / n_embd); slot_row/last_slot -> kva_layer.h | purity: 16 headers, NONE pending; the gate drops its pending list |
| 7 | 7d5209f | arch/kva_step.h: core_declare, derive (`can_stream`: probe_depth routed layers below S), mask_rows, copy_stream, approximate_step, log_pass, single_prefill, misaligned, core_step; hooks prologue / stock_layer(probes) / epilogue / stock_step / capture_step / finish_state / capture_mixed; fact n_vocab. Bridge case deleted | hook BODIES live in qwen4exp_adapter.h (adapter_of must see them), not the .cpp; qwen4exp_kva.cpp = scaffolding + declare/step/probe thins + exports. arch_static back to 63 cases / 1,125,462 checks (the pre-split count) |
| 8 | 97ea361 | tests/rad_fake.h (recording builder/ctx, fake device, Env, stderr_of), tests/folder_fixture.h (tiny container, in-memory folder) | case list hash identical before/after |
| 9 | fb36645 | tests/adapter_core_test.cpp (toy 4-layer dense adapter, spec §4 cases 1-5; 115 checks) + arch/kva_core.h (umbrella, adds <arch/rad_fp8.h>) | target has NO ${RADIANCE_SRC}/arch on its include path and #errors on the qwen4exp plugin. `kva_drop_rows` declared only with top_k > 0 (qwen4exp's graph unchanged). Negative controls, each failing its case: drop always declared, min_tail 512, routed + probe_depth 1, match_name qwen4exp |
| merge | 705bd0d | stage-e a005952 (D1's frozen home 406e746 green, ctest 3/3) | one conflict (Stage C's hazard line on both sides; kept split's). No code change from the merge; README held-cost paragraph + E/D notes |
| 10 (part) | 86ae8ba | docs/ADDING-A-MODEL.md + README pointer | written to the interface AS BUILT (check_tensors core, declare_codes hook, int8 ships), with a status line for the step-7 hooks |

## Remaining

1. GPU gate G1 (evidence/split/scripts/g1.sh; queued 21:14Z on gpuq as `split-g1`, behind staged-d2):
   frozen home of `split` HEAD at session start (its build runs the host suite), off ident x1 vs R3, int8
   speed + quality KL `.rows` cmp'd to Stage E S4's i8 rows (home 5a3115b, same boot 75e3e39b),
   RADIANCE_KVA_FINAL=off. Predictions labbook seq 446 (HS-split-off-ident), 447 (HS-split-i8-rows).
2. Stage E's final-map default flip (Dylan: MTP final map out of the release, default off): merge
   stage-e again when it lands.

Host suite at 705bd0d: ctest -LE gpu 5/5 (kernel_test, arch_static 63 / 1,125,462, adapter_core
5 / 115, core_headers_name_no_arch, python), pytest tests 209 passed / 33 skipped.

## Also done this session

- notes/stagec.md: R66 recorded as CUT, "covered by R62 + E's held cost" (orchestrator).
- c3b waiter: not in the gpuq queue at 20:21Z (already gone; nothing to cancel).
