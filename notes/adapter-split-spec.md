# notes/adapter-split-spec.md -- the CORE / ADAPTER split, specified for execution (report only)

> Goal (Dylan): a future model -- dense, MoE or hybrid -- is a new ADAPTER plus its own static oracle,
> with the core untouched: mostly drop-in. No second adapter is built here; the proof is a host-only
> TOY adapter (§4). Inputs: notes/adapter-map.md (§1 per-function map, §2 the 15-point interface, §4
> the ranked facts), notes/tp-map.md, arch/, kernels/, tests/arch_static_test.cpp, the CMake files,
> radiance docs/PLUGIN.md §11. adapter-map.md's line numbers are approximate; every reference below
> was re-located in this clone by symbol. Nothing here is speculative: each item cites its symbol.

## 0. Ground rules

* The split runs on main AFTER stage-c and stage-e merge (their pieces' homes: §1.4). Before step 1,
  re-run `ctest -LE gpu` and re-locate the symbols the two branches moved.
* Every step of §2 is one commit that leaves the tree building and `ctest -LE gpu` green. The
  per-step check is the STATIC ORACLE: tests/arch_static_test.cpp holds declare graphs and step issue
  lists element-for-element against the in-tree qwen4exp_fp8 plugin (`check_same_graph`, `differ`),
  so a moved fact that changed a number, an operand or a refusal line fails loudly. The engine-level
  byte-identity gate runs ONCE, after the last step (§5).
* Namespaces: each core file, as it is converted, moves to `namespace kva`; arch/qwen4exp_adapter.h
  opens `namespace qwen4exp_kva` with using-declarations for everything the adapter and the static
  test name (`using kva::Kva; using kva::g_kva; ...`), so tests/arch_static_test.cpp keeps compiling
  unchanged until §2 step 8 touches it. The plugin's exports and claim stay `qwen4exp_kva`.
* No behaviour change anywhere in the split: same issues, same refusals, same stderr text (several
  static cases match refusal strings). Only `radiance: qwen4exp_kva:` prefixes become `%s:` of the
  adapter's `log_name` fact, with the same value.
* Radiance's contract (PLUGIN.md §11) is untouched: each adapter is one .so claiming `(arch_id,
  quant)`, shadowing the in-tree stem on $RADIANCE_HOME. The core is header-only and compiled INTO
  each adapter .so; kernels/kva.so stays shared (its ops take geometry as parameters).

## 1. Target layout

### 1.1 CORE -- model-independent; includes only installed radiance headers (`<arch/rad_arch.h>`
### and siblings), never any in-tree architecture. No core file may name a model type.

| file | contents (unchanged unless noted) |
|---|---|
| arch/kva_json.h | `Json`, `JsonReader`, `json_parse` -- pure reader |
| arch/kva_folder.h | `FolderPlace`/`find_folder`, `Folder`/`read_folder`, `read_safetensors`, `map_file`, `sha256_bytes` |
| arch/kva_match.h | `Container`/`open_container`/`entry_sha256`/`vocab_sha256`, `Match`/`match_*` -- adapter name is a parameter |
| arch/kva_plan.h | `PlanIn`/`PlanConfig`/`Pass`/`bulk_end`/`plan_pass` -- pure numbers |
| arch/kva_config.h | `Config`, env parsing; `kMinTail` and the `tail` default become adapter facts (§3.10) |
| arch/kva_dump.h | `dump_*`, `Capture`/`capture_*`, `StateDump`/`state_end`/`mixed_state_end` -- widths as arguments |
| arch/kva_guard.h | `count_version`/`releases_in`/`sha256_*`/`real_path`/`engine_object`/`Forward`/`take_forward`/`open_guard`; `find_shadowed` gains a `so` argument (§3.11) |
| arch/kva_adapter.h | **NEW -- the ONE interface header** (§1.3) |
| arch/kva_projector.h | `Loaded`/`load_folder` (match name from the adapter), `Upload`/`upload_rank`/`plan_rank`/`plan_maps`/`take_operands`/`fill_block`/`free_upload*` -- geometry from facts, schema via the `check_tensors` hook |
| arch/kva_declare.h, arch/kva_declare_masked.h | `Kva`/`g_kva` (block-set fields become optional), `check_mode`/`check_fill`/`decl_fill`/`decl_masked`/`decl_projected`/`decl_probes`/`decl_state_read`/`note_*`/`take_folder`/`take_upload`/`decl_selected`/`capture_split` |
| arch/kva_layer.h | `ring_copy`/`ring_next`, `project`/`project_masked`/`project_bulk`/`project_rows`, `quantise`, and the four late-layer drivers (`masked_layer`/`straddle_layer`/`decoders_layer`/`fill_layer`) as skeletons calling adapter hooks |
| arch/kva_step.h | **NEW**: `derive`/`log_pass`/`single_prefill`/`misaligned`/`mask_rows`/`copy_stream`/`approximate_step`/`step` (from qwen4exp_kva.cpp), `note_meta_mode` |
| kernels/ | unchanged: the ops are generic `[n_head, sd0, sd1]` linear-state rows; `kva_rho_update`'s GDN decay semantics are only ever *issued* by a GDN adapter (adapter-map §1 kernels table) |

### 1.2 ADAPTER (qwen4exp) -- everything that names `qwen4exp_fp8::`

| file | contents |
|---|---|
| arch/qwen4exp_kva.cpp | the include/shadow scaffolding (`RAD_ARCH_NO_EXPORTS`, `#include <qwen4exp_fp8/qwen4exp_fp8.cpp>`), `declare`/`step`/`probe` thins, the exports (`RAD_ARCH_PLUGIN(qwen4exp_kva, "qwen4exp", "", ...)`), `capture_step`/`finish_state`/`capture_mixed`, `probe_arm` |
| arch/qwen4exp_adapter.h | **NEW**: `adapter_of(const qwen4exp_fp8::Model&, rank)` filling every fact; the hooks: `check_tensors` (from kva_projector.h), `declare_model` (decl_fill's quantiser + decl_projected's codes), `decl_state_ops` (decl_late_group + decl_kernel_ops), `prologue`/`stock_layer`/`epilogue`/`conn` (qwen4exp_kva.cpp prologue/layer/epilogue), `late_block`/`ffn` (the block issues), `stock_step` (= qwen4exp_fp8::step), the capture hooks; the `using` block; `kMinTail`'s value (512) |
| arch/qwen4exp_blocks.h | from kva_layer.h: `slot_row`/`last_slot`, `correct`/`decay_sums`/`copy_state`/`read_state`, `gdn_*` (decode half, scans, masked, tail rows, straddle), `attn_rows` |
| arch/qwen4exp_fill.h | kva_fill.h renamed: `gdn_project`/`gdn_scan`/`qsa_keys`/`attn_kv` (verbatim in-tree issue copies) |
| arch/qwen4exp_moe.h | kva_moe.h renamed: `MoeArm`/`moe_probe`/`moe_route`/`moe_experts`/`moe_pass`/`moe_layer` (MoeFP8 verbatim; any MoeFP8 model may include it) |
| tests/arch_static_test.cpp | the qwen4exp oracle, unchanged after step 8 |
| arch/CMakeLists.txt | per-plugin `rad_add_plugin` entries (one per adapter; see §6) |

### 1.3 The adapter interface: arch/kva_adapter.h (namespace `kva`)

A plain struct of facts + function pointers. Every field carries a one-line contract; the right
column of §3 says where qwen4exp supplies it today. Nothing speculative: exactly what the core's
declare and step paths read (adapter-map §2's fifteen points).

```cpp
struct StateShape { int64_t n_head, sd0, sd1; };          /* this rank's; {0,0,0} = no state */
struct KvaAdapter {
    /* identity (§2.1) */
    const char* log_name;    /* stderr prefix of every refusal/note ("qwen4exp_kva") */
    const char* match_name;  /* the projector manifest's "adapter" string ("qwen4exp") */
    const char* shadow_so;   /* the in-tree .so the guard forwards to ("qwen4exp_fp8.so") */
    /* geometry facts (§2.5, §2.8, §2.4, §2.14) */
    int64_t n_layer, n_embd, n_vocab_all, wide; /* wide = the stream feeding K: hccfg.hc*n_embd */
    int64_t tile;            /* bulk-end tile G; 1 when no recurrent block makes the checks vacuous */
    int64_t split_lo;        /* lowest valid split layer; = ple_layer + 1; 0 when no constraint */
    int64_t probe_depth;     /* routed layers needed below S for the stager probes (3) */
    int64_t top_k, n_expert, n_ff_exp;  /* MoE facts; top_k == 0 = dense: no drop arm, no probes */
    int64_t min_tail, default_tail;     /* the measured tail facts (512 / 2048) */
    int64_t qsa_exact_to;     bool straddle_layers;  /* §2.12; the adapter computes both at declare */
    const char* dtype;       /* the stream dtype for the cast/gemm op params ("bf16") */
    const char* act_dtype;   /* the activation dtype of the stream/block-input buffers ("bf16") */
    StateShape state;        /* §2.7: the recurrent state's per-rank shape; {0,0,0} = none */
    /* per-layer fact arrays, length n_layer (§2.3, §2.9, §2.11) */
    const bool *full,        /* attention layer? */
               *ext_in,      /* the block takes a pre-normed input (check_fill) */
               *calibrated,  /* the block runs a calibration tap (check_fill refuses it) */
               *routed;      /* routed FFN: probes, stream_ok, the drop arm */
    /* buffers the core issues against (§2.2, §2.6); 0 = absent */
    rad_buf buf_stream, buf_x, buf_x_q, buf_x_s, buf_logits;  /* b_h, a_x.x, a_x.q8, a_x.s8, b_logits */
    /* hooks (§2.1, §2.6, §2.7, §2.10, §2.13); nullptr = capability absent, the core skips it */
    std::string (*check_tensors)(const Folder&, const KvaAdapter&, Loaded*);   /* the folder schema */
    int         (*declare_model)(RadBuilder*, const RadBuildCtx*, Kva&);       /* quantiser + codes */
    const char* (*decl_state_ops)(RadBuilder*, const RadBuildCtx*, Kva&);     /* correction ops */
    void (*prologue)(RadCtx*, const RadBatch*);                                /* embedding/media/rope */
    void (*stock_layer)(RadCtx*, int64_t li, const RadBatch*, const MoeArm*);  /* one exact layer */
    void (*epilogue)(RadCtx*, const RadBatch*);                                /* last conn + logits */
    void (*conn)(RadCtx*, int64_t li, bool ffn, bool write, int64_t T, int64_t r0, int64_t rows);
    void (*late_block)(RadCtx*, const Kva&, int64_t li, const RadBatch*, const Pass&,
                       StateDump*, Path path, int64_t r0, int64_t rows);      /* the 4 block variants */
    void (*ffn)(RadCtx*, const Kva&, int64_t li, const RadBatch*, int64_t r0, int64_t rows,
                rad_op drop, rad_buf mask);                                    /* MoE / dense MLP */
    void (*stock_step)(RadCtx*, const RadBatch*);                              /* the in-tree step */
    void (*capture_step)(RadCtx*, const Kva&, const RadBatch*);               /* RADIANCE_KVA_CAPTURE */
    void (*finish_state)(RadCtx*, const Kva&, const RadBatch*, StateDump&, bool);
    void (*capture_mixed)(RadCtx*, const Kva&, const RadBatch*, bool);
};
```

`MoeArm` moves with the MoE issues (qwen4exp_moe.h); the core passes the drop op and mask buffer to
`ffn`, which builds its own arm. `late_block` receives the path (LEAN/MASKED/STRADDLE/DECODERS) and
the row window, because the correction/scan interleaving is the block's own (kva_layer.h
gdn_prefill_scans). The core owns everything around it: conn, projection, select, ring, mask.

### 1.4 Where the stage-c / stage-e pieces land (decided now so the branches merge into core)

* The hazard op (`kva_hazard_read` and its step-side log line): kernels/ + arch/kva_step.h -- CORE.
* The streaming-only projector ring (maps always host RAM, one VRAM slot): kva_projector.h
  `plan_maps`/`take_operands` + kva_layer.h `ring_copy`/`ring_next` -- CORE; every width in them is
  already the `wide` fact.
* Int8 maps: kva_folder.h dtype handling, kva_projector.h `take_operands`, the decl_fill gemm dtype
  param, kernels/forward.cpp's forwarded row -- CORE/kernels; the folder's int8 schema names go in
  the adapter's `check_tensors`.
* The MTP "final" map: kva_match.h/kva_projector.h manifest and upload flow -- CORE; its shape check
  in the adapter's `check_tensors`.
* Media handling: derive's eligibility gates (`batch->enc`, `n_mm_rows`) -- CORE (kva_step.h); the
  media rows' issues -- the `prologue` hook (already adapter).

## 2. The move list -- in an order that keeps the build and the static test green after each step.
## One commit per step, its check named (every check: build + `arch_static`, plus the cases named;
## `ctest -LE gpu`). §0's namespace rule applies inside every step.

| # | move (symbol -> where) | check (static cases that pin it) |
|---|---|---|
| 1 | kva_guard.h: `find_shadowed`/`open_guard` take the shadow stem as an argument; qwen4exp_kva.cpp passes `"qwen4exp_fp8.so"` from a local constant. Nothing else moves. | `the_release_scan_counts_nul_delimited_copies`, `the_forward_table_starts_empty` |
| 2 | NEW arch/kva_adapter.h (the struct, §1.3) + NEW arch/qwen4exp_adapter.h: `adapter_of(m, rank)` filling every fact from the model (`m.g`, `m.hccfg.hc`, `m.gcfg`, `m.moecfg`, `m.ple_layer`, `m.layers[].full/mlp/gdn/attn`, `m.a_x`, `m.b_h`, `m.b_logits`); the using-declaration block. No caller changes yet. Add a temporary case `the_adapter_facts_match_the_model` asserting each fact against the direct reads (wide == `m.hccfg.hc * m.g.n_embd`, tile == `m.gcfg.chunk`, split_lo == `m.ple_layer + 1`, per-layer full/ext_in/routed, `state` == `{m.gcfg.n_head_v, head_v, head_k}`). | the new case + full suite |
| 3 | kva_config.h: `kMinTail` and `Config::tail`'s default come from the adapter -- `read_config` gains trailing defaulted parameters (`min_tail = 512`, `default_tail = 2048`), so the two static cases that call `read_config` directly stay unchanged; the declare passes the adapter's values. | `a_tail_below_the_measured_minimum_is_refused`, `a_tail_of_two_steps_less_a_tile_or_more_is_refused`, `a_tail_past_the_step_approximates_the_provable_rows` |
| 4 | kva_projector.h: drop the `Model&` parameter -- `check_tensors`'s body moves to qwen4exp_adapter.h (the hook); `load_folder` takes the adapter (`match_model(..., a.match_name)`); `plan_maps`/`plan_rank`/`take_operands`/`upload_rank` read the `n_embd`/`wide`/`n_layer`/`n_vocab_all` facts and `state` (per-rank correction bytes = `n_head*sd0*sd1*4`; folder check `{n_head*world, sd0, sd1}`). | `a_mode_without_a_usable_projector_serves_the_in_tree_graph`, `speed_takes_the_folder_and_declares_its_kernel_ops`, `the_raw_operands_point_at_their_tensors_copies`, `host_placement_puts_the_maps_in_host_mapped_memory`, `the_staging_ring_copies_each_map_a_layer_ahead_on_lane_1`, `a_projector_fitted_on_another_variant_warns_and_runs` |
| 5a | kva_declare.h: `Kva`/`g_kva` to namespace kva (fields unchanged; they become optional capabilities); `decl_late_group`+`decl_kernel_ops` move to qwen4exp_adapter.h behind `decl_state_ops`; `check_mode` reads `tile`/`min_tail` facts; `check_fill` reads `split_lo`/`ext_in[]`/`calibrated[]`; `decl_fill` reads `n_embd`/`wide`/`dtype`/`buf_stream`/`buf_x` and calls `declare_model` for `k.quant`. | `speed_adds_its_ops_after_the_in_tree_graph`, `a_mode_whose_kva_op_no_kernel_serves_is_refused_by_name`, `each_routed_layers_expert_span_is_gate_up_to_down` |
| 5b | kva_declare_masked.h: `decl_probes` reads `top_k`/`n_expert`/`n_ff_exp` (skipped whole when no layer is `routed`); `decl_projected` SPLITS -- the fact-parameterised parts (`b_hs`, `xp.x`, `op_cast`, `op_select`, `op_drop`) stay core, only the model-owning parts move into `declare_model` (`xp.declare_qs`'s code pair mirroring the model's `a_x`, `q8_fed`, the `b_eids` concurrency, and `k.quant.declare`); `decl_masked`/`note_*`/`take_folder`/`take_upload`/`decl_selected`/`capture_split` read facts; `decl_state_read` reads `state`; the QSA loop filling `straddle_layers`/`qsa_exact_to` moves to qwen4exp_adapter.h. | `the_masked_paths_buffers_take_the_whole_program`, `a_sizing_declare_matches_the_real_one`, `the_correction_is_declared_and_issued_only_when_held`, `a_capture_without_a_folder_takes_its_split_from_the_env_not_the_container` |
| 6a | Pure moves, no interface change: kva_layer.h's `slot_row`/`last_slot`/`correct`/`decay_sums`/`copy_state`/`read_state`/`gdn_decode_half`/`gdn_scan_over`/`gdn_prefill_front`/`gdn_prefill_scans`/`gdn_masked`/`gdn_out_rows`/`gdn_tail_rows`/`gdn_straddle`/`attn_rows` -> arch/qwen4exp_blocks.h; kva_fill.h -> arch/qwen4exp_fill.h; kva_moe.h -> arch/qwen4exp_moe.h; qwen4exp_kva.cpp's include order updated. | full suite (the oracles compare issue lists, so a pure move that changed anything fails) |
| 6b | kva_layer.h: the four drivers become core skeletons over hooks -- `masked_layer`/`straddle_layer`/`decoders_layer` call `conn`/`late_block`/`ffn` and keep `project_masked`/`project_bulk`/`project_rows`/`ring_next` (facts `buf_stream`/`buf_x`/`buf_x_q`/`buf_x_s`); `fill_layer` keeps `project`/`quantise` and calls `late_block(PATH_LEAN)`; the `dbg_resid` calls move inside the adapter's `conn`/`late_block` hooks (they name `m.b_h`/`m.a_x`). | `plumb_is_the_stock_step_through_the_masked_path`, `quality_masks_rows_in_place_through_the_in_tree_layer`, `a_straddling_chunk_splits_the_last_scan_at_the_bulk_end`, `speed_beside_decoders_runs_full_late_blocks_over_the_decoder_rows_only`, `a_speed_straddle_runs_its_late_blocks_over_the_tail_rows_only`, `r53_mixed_steps_are_the_in_tree_step_with_only_the_last_sequence_masked`, `speed_fills_late_layers_with_their_cache_writing_ops_only`, `at_tp2_each_rank_corrects_its_own_heads` |
| 7 | qwen4exp_kva.cpp -> arch/kva_step.h (CORE): `note_meta_mode`, `derive` (`stream_ok` becomes `count(routed[i], i < split) >= probe_depth`; `straddle_ok` from the facts), `mask_rows`, `copy_stream`, `log_pass`, `single_prefill`, `misaligned` (`tile` fact), `approximate_step` (skeleton: `prologue`/`stock_layer` hooks, `ring_copy`, `late_block` per path, `epilogue`), `step` (dispatch + the `stock_step`/`capture_step`/`finish_state`/`capture_mixed` hooks). Stays in the adapter: the scaffolding, the `declare`/`step`/`probe` thins, `prologue`/`layer`/`epilogue`/`probe_arm`/`capture_step`/`finish_state`/`capture_mixed`, the exports. The temporary case from step 2 is deleted. | every oracle case: `off_declares_exactly_the_in_tree_graph`, `off_step_issues_the_in_tree_sequence`, all masked/straddle/decoders/lean cases, `a_chunk_off_the_tile_fails_the_step_by_name`, `capture_records_an_exact_chunk_and_issues_the_stock_sequence`, `state_capture_copies_each_late_state_before_the_apply`, `a_mixed_step_capture_copies_every_sequences_late_states_after_the_step`, `the_logits_dump_names_each_rows_sequence_and_changes_no_issue` |
| 8 | Test plumbing, no case changes: the recording fakes (`RadBuilder`/`RadCtx` + the `rad_*` C row) move to tests/rad_fake.h; `tiny_container*`/`safetensors`/`write_folder`/`sha_of` move to tests/folder_fixture.h; arch_static_test.cpp includes both. | arch_static green, unchanged case list |
| 9 | NEW tests/adapter_core_test.cpp (§4) + tests/core_purity.cmake + tests/CMakeLists.txt entries. | the new suite + the purity gate |
| 10 | docs/ADDING-A-MODEL.md (§6) + a README pointer; then the §5 engine gate (its baseline arms are recorded on current main after the stage-c/e merge). | docs only; §5 |

Commit size guide: 5a/5b and 6a/6b are split so each stays reviewable; if a step's diff exceeds
~400 lines, split it by symbol, not by concept.

## 3. The hard-coded qwen4exp facts (adapter-map §4) -- each an adapter field or a metadata read

| # | fact (symbol today) | becomes | qwen4exp supplies |
|---|---|---|---|
| 1 | The block-issue code: kva_fill.h `gdn_project`/`gdn_scan`/`qsa_keys`/`attn_kv`, kva_layer.h `gdn_*`/`attn_rows`, kva_moe.h `moe_*` | ADAPTER hooks `late_block`/`ffn`/`stock_layer` (+ qwen4exp_blocks/fill/moe.h); never metadata | the verbatim in-tree issues, pinned to radiance 140987f |
| 2 | `stream_ok = k.split >= 3` (qwen4exp_kva.cpp `derive`) | a core rule over the `routed[]` fact + the `probe_depth` field (3) | `routed[]` = layers with a routed MoeFP8 FFN; `probe_depth` 3 (the probes ride behind layer S-3's gate-up) |
| 3 | The ext_in contract (`check_fill`) | the `ext_in[]` fact; `check_fill` stays core and refuses by name | `lay.full ? lay.attn.ext_in : lay.gdn.ext_in` |
| 4 | `wide = m.hccfg.hc * m.g.n_embd` (copy_stream, decl_fill, decl_projected, plan_maps, project_*) | the `wide` fact | `m.hccfg.hc * m.g.n_embd` (10240) |
| 5 | `k.tile = m.gcfg.chunk` (`declare`, `check_mode`, `misaligned`) | the `tile` fact; kva_plan.h's `tile = 64` default stays a pre-adapter fallback only | `m.gcfg.chunk` (64) |
| 6 | Recurrent-state shapes (`decl_late_group`, `decl_kernel_ops`, `decl_state_read`, `read_state`) | the `state` fact {n_head, sd0, sd1} + the `decl_state_ops` hook; the rho op stays adapter-issued (its GDN decay semantics live with the adapter's issues) | `m.gcfg.{n_head_v, head_v, head_k}` per rank |
| 7 | `m.ple_layer` (check_fill, capture_split, check_tensors' split bound) | the `split_lo` fact (= ple_layer + 1); a model with no PLE supplies 0 | `m.ple_layer + 1` |
| 8 | `moecfg.top_k`, probe sizes (`decl_probes`, the drop op in decl_projected) | the `top_k`/`n_expert`/`n_ff_exp` facts; 0 = dense and the declarations drop out | `m.moecfg.top_k`, `l.mlp.c.{n_expert, n_ff_exp}` |
| 9 | `straddle_layers`' QSA form (`decl_masked`'s loop, `derive`) | the `straddle_layers` + `qsa_exact_to` fields, computed by the adapter's declare | `a.qsa_sel && a.qsa_sequ && a.op_attn_gq`, `a.qsa_exact_to` |
| 10 | `kMinTail = 512`, tail default 2048 (kva_config.h) | the `min_tail`/`default_tail` fields (fit facts per model family; a manifest metadata read is possible later, nothing today reads one) | 512 / 2048 (KVA-FACTS §5 measurements) |
| 11 | The shadow/guard strings (`find_shadowed`, the plugin claim, arch/CMakeLists.txt's stem) | the `shadow_so` field; the claim stays in the adapter .cpp's `RAD_ARCH_PLUGIN`; the stem stays in the adapter's CMake entry | `"qwen4exp_fp8.so"`, `("qwen4exp", "")`, target `qwen4exp_fp8` |

## 4. The proof without a second model -- tests/adapter_core_test.cpp (host-only, no GPU, no
## radiance engine run, no in-tree architecture)

The toy adapter, defined in the test file itself: 4 layers, all dense attention (`full[]` all
true), `n_embd` 64, `wide` 64 (a plain residual), `tile` 1, `split_lo` 0, `state` {0,0,0} (no
recurrent block), no MoE (`top_k` 0, no `routed` layer), `ext_in[]` all true, `calibrated[]`
none, `straddle_layers` false, `min_tail` 256, `default_tail` 1024, `dtype` "bf16"; every hook a
no-op except `check_tensors` (the toy schema: `proj.L.weight [64, 64]` bf16 + `.bias`, `score`
f32 [vocab], no `st`). The file includes ONLY core headers (kva_adapter.h and the kva_* core
set) plus the shared fakes -- never `qwen4exp_kva.cpp`, never anything under `${RADIANCE_SRC}/arch`,
and it does not define `RAD_ARCH_NO_EXPORTS`. Cases:

1. `the_planner_serves_a_dense_toy` -- `derive` through the toy's facts + `plan_pass` rows: a
   whole-bulk chunk in speed takes PATH_LEAN; a straddling chunk falls to PATH_STOCK (no straddle,
   no stream); quality falls to PATH_STOCK (no stream); plumb still takes PATH_MASKED; decoders
   beside a bulk chunk fall to PATH_STOCK. Pins facts #2/#5/#9 with no qwen4exp anywhere.
2. `the_toy_projector_folder_is_matched_and_refused` -- `match_model` + the toy's `check_tensors`:
   the toy folder (tests/folder_fixture.h) is accepted (`have_proj`, split 2); a folder naming
   another adapter is refused "for adapter"; a wrong metadata key, a wrong `vocab_sha256`, a
   misshapen `proj.2.weight` and a hole in the maps are each refused by name.
3. `the_toy_config_takes_its_tail_from_the_adapter` -- `read_config` with the toy's facts: the
   tail defaults to 1024; a tail of 128 is refused naming the toy's 256 minimum (wrong under
   qwen4exp's 512 -- pinning fact #10).
4. `the_toy_declare_adds_only_generic_ops` -- a full core declare with the toy on the shared fake
   builder: after the toy's own (empty) graph the ops added are exactly `kva_mask`, `cast`,
   `kva_select` and one `kva_gemm_nt_bias` per late layer (N 64, K 64); NO `kva_state_correct`, NO
   `kva_drop_rows`, no probe buffers (`b_zeros == 0`), no quantiser op; a lean toy step issues
   exactly the projector GEMMs (one per late layer, `device_calls == 0`); a plumb declare copies
   nothing. Pins that the GDN/MoE capabilities dropped OUT rather than mis-declared.
5. `no_core_header_names_the_in_tree_architecture` -- the compile itself: this TU builds the
   whole core against the toy with the in-tree source nowhere on its include path. Backed by the
   build-time gate `core_headers_name_no_arch` (tests/core_purity.cmake, step 9): every
   arch/kva_*.h must be free of `qwen4exp`, `GdnFP8`, `MoeFP8`, `AttnGatedFP8`, `QsaIndexer`,
   `HyperConn`, `gcfg`, `hccfg`, `moecfg`, `ple_layer` (the model-type and config names; the
   buffer-name comments in kva_adapter.h are deliberately excluded from the grep).

Together with the unchanged arch_static suite (qwen4exp through the same core), this proves the
core serves two adapters that share no block type -- the second deliberately trivial.

## 5. The byte-identity gate for the refactor

Protocol (notes/gates.md provenance throughout): the baseline and the candidate arms on the SAME
HOST boot (the boot id is checked and recorded per run, gates.md provenance table), the frozen
flags of scripts/common.sh, the quick9 KL corpus, `RK_EXPECT_APPROX=67`. Baseline arms run on
current main (post stage-c/e merge) into fresh files (`grade.sh` refuses to overwrite); candidate
arms after step 10; both under the GPU lock. `grade.sh` is the row producer (`<out>.rows`);
`tools/paired.py` is not needed -- the gate is equality, not a CI.

| gate | what must hold | run | GPU minutes |
|---|---|---|---|
| S0 static | the whole host suite, every commit and once at the end | `ctest -LE gpu` | 0 (no GPU) |
| S1 kernels | the device rows still match the host rows (kernels/ untouched) | `kva_kernels_gpu` | ~5 incl. build (the device test itself ran in 2 s, gates.md Gate 1) |
| S2 off ≡ stock, text | the six `ident.sh` hashes equal between a stock (`exact`) server and an `off` server | `serve.sh exact` + `scripts/ident.sh`; `serve.sh off` + ident | ~35 EST: two 114 GiB model loads, bounded by `RK_SERVE_TIMEOUT` 1800 s each (~10 min each ESTIMATE) + 6 short generations (< 1 min) |
| S3 KL byte-identity (the gate) | (a) `grade.sh speed` and (b) `grade.sh quality`: `.rows` byte-identical (`cmp`) to the baseline arms on the same host boot, equal ppl ratio / top-1 / KL mean, 67 approximate steps; (c) `grade.sh quality` with `RADIANCE_KVA_ROWSEL=all RADIANCE_KVA_SHARE=1 RADIANCE_KVA_ROWSEL_TABLE=all`: `.rows` byte-identical to the exact reference `data/kld/ref-stage0` (R35's boot-invariant property, gates.md Gate 7b) -- the strongest end-to-end masked-path check | 3 candidate + 2 baseline `grade.sh` runs | ~80 EST: five runs, each a foreground container = one model load (~10 min ESTIMATE) + one quick9 KL pass (~6 min by derivation: 67 prefill chunks at 0.9-1.2 s, stageb.md session-3 per-step lines, + 9 x 2,047 decode positions at ~14 ms, stageb.md:319, ~= 83 s + 258 s) |
| S4 TTFT smoke (not byte-gated) | speedup vs exact within run noise (the stage-4 numbers, gates.md results table: 1.22 / 1.57 / 1.96x) | `serve.sh speed` + `speed.sh` (3 lengths x 6 prompts at 4.9-19.8 s each) | ~15 EST: one load + ~4 min of requests |

Total ≈ 2.5-3 GPU-hours, load-dominated. ESTIMATE marks the load times: no measured load time
exists in the notes, only the 1800 s bound and the 114 GiB container size (scripts/serve.sh). If
the budget is tight, S4 is the droppable arm; S2/S3 are not. Success: every `cmp` clean, every
number equal to the baseline's; any difference is a regression to bisect before merge -- the
split changes no bytes by construction (§0).

## 6. docs/ADDING-A-MODEL.md -- the checklist this spec ships in step 10 (as written, ≤ 25 lines)

```
ADDING A MODEL (dense, MoE or hybrid): one adapter, no core change.
0. Read radiance docs/PLUGIN.md §11: your adapter is ONE .so claiming (arch_id, quant),
   shadowing the in-tree stem on $RADIANCE_HOME. Copy arch/qwen4exp_kva.cpp's scaffolding
   (RAD_ARCH_NO_EXPORTS + the #include of your in-tree arch .cpp) and arch/CMakeLists.txt's
   rad_add_plugin entry (stem = the shadowed .so's name).
1. Write <model>_adapter.h filling every KvaAdapter fact from your in-tree Model
   (arch/kva_adapter.h is the contract; each field's comment is its semantics).
2. Implement the hooks: conn (your connection read/write), prologue / stock_layer / epilogue
   (verbatim copies of your in-tree step), late_block (your blocks in the four approximate
   variants -- qwen4exp_blocks.h is the pattern), ffn (your MoE with the drop arm, or your dense
   MLP), check_tensors (your projector folder schema), declare_model (your quantiser + projected
   codes), decl_state_ops (your correction ops; nullptr when no recurrent state), stock_step,
   and capture_step / finish_state / capture_mixed if you keep the debug captures.
3. Reuse arch/qwen4exp_moe.h as-is if your FFN is MoeFP8; otherwise write your own lean pieces
   (the lean fill needs exactly your blocks' cache-writing ops).
4. Static test: copy tests/arch_static_test.cpp's oracle pattern for your model -- off must
   declare and issue exactly what your in-tree plugin does; add your derive truth-table rows,
   your projector refusals, and your own facts-vs-model case (§4's pattern).
5. Fit the projector: serve with RADIANCE_KVA=off and your capture hooks, then
   tools/kva_projector.py fit (notes/refit.md); the manifest's "adapter" is your match_name,
   its model.meta keys name the facts your check_tensors pins, and your measured tail goes
   into the min_tail / default_tail facts.
6. Gate: ctest -LE gpu, then the §5 engine gate for your model (off ≡ stock, the KL row
   byte-identity, quality/all-rows == exact).
```

## 7. What this spec deliberately does NOT do

* No second adapter is built; the toy is a test, not a plugin. qwen35moe_fp8's known blockers --
  the own-norm contract and the capture-side normed block input (adapter-map §3 candidate 1) --
  are exactly what the `ext_in[]` fact and the `capture_step` hook make expressible; that work
  stays with whoever builds that adapter.
* No radiance source change, no new ABI, no engine release bump: the guard's KVA_RADIANCE_VERSION
  mechanics are untouched and PLUGIN.md §11's contract is served as-is.
* No speculative interface fields: §1.3 is exactly what the core's declare and step paths read
  today (adapter-map §2's fifteen points, closed).