# notes/adapter-map.md -- CORE vs ADAPTER map, the adapter interface, and second-adapter candidates
> Prep map by an open-model worker (GLM-5.3), spot-checked by the orchestrator: the claims checked hold, but
> some file:line references are off by a few lines (e.g. `stream_ok` is qwen4exp_kva.cpp:118, the plan tile
> default is kva_plan.h:51). Re-locate by symbol before editing; Stage B/E merges will shift lines further.


Report only. Every claim carries file:line in this clone (arch/, kernels/) or in the radiance
1.0.8 checkout at /var/home/dylan/projects/inference/radiance (cited as `rad:`).

## 1. Per function/struct: CORE (model-independent), ADAPTER (qwen4exp-specific), MIXED

### arch/kva_json.h -- CORE (whole file)
`Json` (kva_json.h:19), `JsonReader` (:41), `json_parse` (:161): a pure reader; no model type appears.

### arch/kva_folder.h -- CORE (whole file)
`FolderPlace`/`find_folder` (:43/:103), `mapped_container`/`typed_model` (:28/:53), `FolderTensor`/`Folder` (:134/:141), `read_safetensors`/`read_folder` (:196/:235), `sha256_bytes`/`map_file` (:153/:172): reads the process and the folder's own files, not the model (header: kva_folder.h:7); tensor names are data.

### arch/kva_match.h -- CORE
`Container`/`open_container`/`entry_sha256`/`vocab_sha256` (kva_match.h:34/40/64/77): container-format checks. `Match`/`match_meta`/`match_encodings`/`match_anchors`/`match_model` (:99/106/119/133/146): header claims model-agnostic (kva_match.h:8-10); the one model fact -- the adapter name -- is a parameter (`adapter`, :146) and the arch id comes from `meta->arch_id` (:152).

### arch/kva_guard.h -- MIXED
- CORE: `count_version`/`releases_in`/`sha256_block`/`sha256_file`/`real_path`/`engine_object` (kva_guard.h:35,63,84,125,146,152), `Forward`/`g_forward`/`take_forward`/`open_guard` (:174-177,178,182,206) -- release-integrity mechanics.
- MIXED: `find_shadowed` (:159-172) -- only line 165 is model-specific: the hard-coded string `architectures/qwen4exp_fp8.so`; the mechanism is CORE.

### arch/kva_plan.h -- CORE (whole file)
`Path`/`PlanIn`/`PlanConfig`/`Pass`/`bulk_end`/`plan_pass` (kva_plan.h:24,36,48,57,65,70): pure numbers in/out; header: "model-agnostic" (kva_plan.h:3); the adapter fills `PlanIn` (:8-9 names that contract).

### arch/kva_config.h -- CORE
`Mode`..`Place` enums + names (kva_config.h:76-86), `Config` (:91), env parsing (:116-236): mode vocabulary, no model type. `kMinTail` = 512 and `tail` = 2048 (:89,:95) are CORE code whose VALUES are measurements on qwen4exp (KVA-FACTS §5) -- see §4.

### arch/kva_dump.h -- CORE
`dump_read`/`dump_chunk`/`dump_line`/`dump_npy`/`dump_npy_f32` (kva_dump.h:39,52,58,69,88), `dump_boundary`/`dump_mask` (:99,:123 -- widths arrive as arguments; callers pass `hccfg.hc*n`, qwen4exp_kva.cpp:221), `Capture`/`capture_begin`/`capture_rows`/`capture_end`/`StateDump`/`state_end` (:162,170,185,202,219,224 -- split/hidden/hc/heads/V/K all parametersised, :202,:224).

### arch/kva_projector.h -- MIXED
- CORE: `Loaded`/`g_loaded`/`Upload`/`g_upload`/`tensor`/`shaped` (kva_projector.h:32,39,46,62,64,66); `Piece`/`place_piece`/`fill_block`/`dev_at`/`free_upload` (:138-190); `free_uploads` (:273).
- MIXED: `load_folder` (:92) -- model-independent flow, adapter string at :110.
- MIXED: `Layout`/`plan_maps`/`plan_rank`/`take_operands`/`upload_rank` (:193,197,216,233,249) -- flow is CORE; model-specific lines: `wide = hccfg.hc*n` (:199,:234), ring block `(n+1)*wide*2` (:199), correction head slice `g.n_head_v*head_v*head_k*4` with the `full`-layer test (:219-223,:239).
- ADAPTER: `check_tensors` (:68-86) -- the whole body is qwen4exp geometry: split vs `m.ple_layer` (:71), proj shape `{n, wide}` (:70-77), st shape `{gcfg.n_head_v*g.world, head_v, head_k}` (:81-82), which layers are delta-net (`!full`, :79).

### arch/kva_declare.h -- MIXED
- MIXED: `struct Kva` (kva_declare.h:21-60) -- mostly CORE state, but fields encode THIS model's block set: `tile` = the delta net's chunk (:32-33), `op_undo/op_apply/op_rho`/`kv_applied`/`kv_rho` (:40,45-48, GDN correction), `straddle_layers`/`qsa_exact_to` (:55-58, QSA per-row sparse form), `n_zeros/b_probe` (:51-53, MoE stager). `g_kva` (:77) CORE.
- ADAPTER: `decl_late_group` (:83-97) -- binds a KV group only to non-full layers (:93) with `d.n_head_kv = m.gcfg.n_head_v` (:88), delta-net state facts. `decl_kernel_ops` (:100-136) -- corrects only delta-net layers (`lay.full: continue`, :113), reads `m.gcfg` head shapes (:117-118,:129) and the in-tree GDN buffers `lay.gdn.w_a_log/w_dt_bias/w.ab` (:131-132). `check_fill` (:172-201) -- three qwen4exp facts: the PLE layer (:173-177), the MoE calibration tap `lay.mlp.op_gram_gu` (:185-191), the ext_in norm contract (:190).
- MIXED: `check_mode` (:138-170) -- tail-bound math is CORE; model-specific lines: :149 (`G = m.gcfg.chunk`, the tile) and the refusal text naming "the delta net" (:160-166).
- MIXED: `decl_fill` (:204-236) -- projector GEMM declare is CORE-shaped; model-specific: :210/:222 (`wide = m.hccfg.hc * g.n_embd` feeding K), :223-225 (in-tree buffer names `m.b_h`, `m.a_x`), :215 (the model's own quantiser config).

### arch/kva_declare_masked.h -- MIXED
- ADAPTER: `decl_probes` (:18-33) -- sizes from `l.mlp.c.n_expert/top_k/n_ff_exp` (:20-24), routed-MoE facts. `decl_state_read` (:204-226) -- the state shape `m.gcfg.{n_head_v,head_v,head_k}` (:207,:211-212) and delta-net naming (:216).
- MIXED: `decl_projected` (:38-60) -- buffers/ops are CORE-shaped; model-specific: :33 (`b_hs` width `m.hccfg.hc*n`), :39 (`xp` quant pair mirrors `m.a_x`), :52 (`m.b_eids`, an in-tree MoE buffer).
- MIXED: `decl_masked` (:63-90) -- mask/bounds and the mode's row rule are CORE (:67-77); model-specific: :48 (`RAD_INT("top_k", m.moecfg.top_k)` for `kva_drop_rows`), :78-86 (`straddle_layers` gated on the attention's QSA per-row sparse form `qsa_sel/qsa_sequ/op_attn_gq/qsa_exact_to`).
- CORE: `note_config`/`note_debug`/`take_folder`/`take_upload`/`decl_selected` (:93-160) -- folder/config plumbing.
- MIXED: `capture_split` (:183-201) -- split-from-folder is CORE; :187/:199 test against `m.ple_layer` (the PLE constraint).

### arch/qwen4exp_kva.cpp -- MIXED (the adapter's spine around CORE-shaped flow)
| item | line | class | why |
|---|---|---|---|
| include/shadow scaffolding | qwen4exp_kva.cpp:28-38,342-350 | ADAPTER | `#include <qwen4exp_fp8/qwen4exp_fp8.cpp>` (30), the plugin claim `("qwen4exp", "")` and name (346-350) |
| `note_meta_mode`, `declare` | qwen4exp_kva.cpp:49,55 | ADAPTER | reads `qwen4exp_fp8::g_model` (62-63), `m.gcfg.chunk` (86), the layer count (82), and routes to adapter declares |
| `derive` | qwen4exp_kva.cpp:105-132 | MIXED | the PlanIn bridge the plan header names (kva_plan.h:8-9); model-specific lines: 123-124 (`stream_ok = k.split >= 3`, needs three ROUTED layers below S for the probes), 125-126 (`straddle_ok` from the QSA exactness bound), 128 (`pc.tile = k.tile`, the delta net's) |
| `prologue`, `layer`, `epilogue` | qwen4exp_kva.cpp:135,148,167 | ADAPTER | verbatim copies of the in-tree step (comments cite qwen4exp_fp8.cpp:1391-1404, 1407-1425, 1428-1437): PLE (140,152), `hc_mix/hc_ffn` reads/writes (143,146,174-177), the 97th connection `m.mixer` (176) |
| `mask_rows`, `copy_stream` | qwen4exp_kva.cpp:179,188 | MIXED | kva_mask/cast issues are CORE ops; `copy_stream` reads `m.hccfg.hc * m.g.n_embd` (190) |
| `probe_arm` | qwen4exp_kva.cpp:201-208 | ADAPTER | MoE stager probes over `m.layers[x].mlp` (207) |
| `approximate_step` | qwen4exp_kva.cpp:210-236 | MIXED | the pass skeleton (prologue, exact layers 0..S-1, then per-path late layers, epilogue) is the CORE flow; adapter lines: 218 (`li == k.split - 3` probe placement), the calls into adapter issues |
| `log_pass`, `single_prefill` | qwen4exp_kva.cpp:238,250 | CORE | counting + keyed batch shape only |
| `capture_step` | qwen4exp_kva.cpp:260-290 | ADAPTER | issues the in-tree layers by hand; captures `m.b_h` at `hccfg.hc*n` (269), the block input `m.a_x.x` after the hc read (272-274) |
| `finish_state` | qwen4exp_kva.cpp:292-303 | ADAPTER | only delta-net layers (`!full`, 295), `m.gcfg` head dims (298) |
| `misaligned` | qwen4exp_kva.cpp:305-309 | MIXED | the invariant is CORE (a chunk must split on the tile); `k.tile` is the delta net's (309) |
| `step` | qwen4exp_kva.cpp:312-333 | MIXED | dispatch flow is CORE; falls through to `qwen4exp_fp8::step` for stock passes (328) |
| `probe` | qwen4exp_kva.cpp:335-338 | ADAPTER | forwards to `qwen4exp_fp8::probe` |

### arch/kva_fill.h -- ADAPTER (whole file)
Verbatim in-tree issue copies pinned to radiance 140987f block types: `gdn_project`/`gdn_scan`
(kva_fill.h:27,34, GdnFP8), `qsa_keys` (kva_fill.h:65, QsaIndexer -- a qwen4exp-only block,
rad:arch/qwen4exp_fp8/qwen4exp_fp8.cpp:17 names QSA as NEW), `attn_kv` (kva_fill.h:101,
AttnGatedFP8). A model without these exact block types needs its own fill file.

### arch/kva_layer.h -- MIXED
| item | line | class | why |
|---|---|---|---|
| `last_slot` | kva_layer.h:20 | CORE | KV-group slot addressing |
| `correct` | kva_layer.h:31-44 | ADAPTER | the GDN state correction over `m.kv_state`/`k.kv_applied` groups |
| `decay_sums`, `read_state` | kva_layer.h:46-76 | ADAPTER | GDN a|b/A_log/dt_bias decay (49-56), `gcfg` state shape (61-63) |
| `gdn_decode_half`, `gdn_scan_over`, `gdn_prefill_front`, `gdn_prefill_scans`, `gdn_masked` | kva_layer.h:78-190 | ADAPTER | the GdnFP8 block, split around the correction |
| `ring_copy`, `ring_next` | kva_layer.h:192-208 | CORE | lane/copy mechanics, no model type |
| `project_masked`, `project_bulk`, `project` | kva_layer.h:210,252,343 | MIXED | kva_gemm_nt_bias + quantiser issues are CORE ops; model-specific line: 212/254 (`wide = m.hccfg.hc * n` feeding K), 217/256/345 (the in-tree buffers `m.b_hs`, `m.a_x`) |
| `quantise` | kva_layer.h:351-358 | MIXED | the quant op is CORE; the buffer names `m.a_x` are the model's |
| `masked_layer`, `straddle_layer`, `fill_layer` | kva_layer.h:231,324,361 | MIXED | these skeletons ARE the core's three paths (read connection -> project/select -> block whole or fill pieces -> write connection); the model-specific lines inside each: 237-239/330-331/367-374 (the `full ? attention : gdn` role split), 246 (MoE drop), 294-322 `attn_straddle` (QSA per-row sparse attention) |
| `gdn_tail_rows`, `gdn_straddle` | kva_layer.h:263,279 | ADAPTER | GdnFP8 tail rows |

### arch/kva_moe.h -- ADAPTER (block-type, reusable by any MoeFP8 model)
`MoeArm`/`moe_probe`/`moe_route`/`moe_experts`/`moe_pass`/`moe_layer` (kva_moe.h:25-160) are
MoeFP8 verbatim (comment: rad_block_moe_fp8.h:1341-1496). Not qwen4exp-specific -- any radiance
model whose FFN is MoeFP8 (qwen35moe_fp8) reuses them as-is; a dense FFN needs a trivial sibling
(issue the MLP, no drop arm).

### kernels/ -- CORE op semantics, one GDN-shaped exception
| item | line | class | why |
|---|---|---|---|
| `kva_mask`, `kva_select`, `kva_drop_rows`, `kva_state_read`, `kva_gemm_nt_bias` | kernels/kva.h:96-235, rows.cpp:74-133, forward.cpp | CORE | geometry comes in as params (`share`, `top_k`, `n_head/sd0/sd1`); kva_gemm_nt_bias is the engine's own row forwarded (forward.cpp) |
| `kva_rho_update`, `kva_state_correct` | kernels/kva.h:118-160, 170-235; mask.hip, rho.hip, state_correct.hip | ADAPTER-flavoured CORE | the ops are generic linear-state [n_head, sd0, sd1], but `kva_rho_update`'s semantics are the GDN decay (softplus of A_log/dt_bias, kva.h:62-66,79-96) -- a different recurrent block needs its own rho or none |

## 2. The adapter interface the core needs

Each fact/hook, with where qwen4exp supplies it today:

1. **Identity and shadowing.** arch id + quant for the plugin claim (qwen4exp_kva.cpp:346-350), the
   .so stem to shadow (arch/CMakeLists.txt:9), the shadowed in-tree file name for the guard's
   forward (kva_guard.h:165), the adapter string for the projector match (kva_projector.h:110), and
   how the in-tree .cpp is included with exports suppressed (qwen4exp_kva.cpp:28-30).
2. **The model handle.** declare/step read one in-tree `Model` struct: `g_model[rank]` with
   `g` (Geom), `layers`, `gcfg` (GdnFP8::Config), `hccfg` (HyperConn::Config), `moecfg`, `ple_layer`,
   `b_h`, `a_x`, `mixer`, `enter`, `mrows`, `ple` (qwen4exp_kva.cpp:62-63,135-177). The core needs
   this behind an adapter-facing view.
3. **Layer roles.** which layers are "full" (attention) vs recurrent, per layer (qwen4exp_kva.cpp:154,
   kva_declare.h:93,113; kva_projector.h:79; kva_layer.h:237).
4. **Split layer S and its constraints.** S from the projector manifest (kva_projector.h:70), with
   the adapter's constraint "above the PLE layer, below the last" (kva_projector.h:71-73,
   kva_declare.h:173-177, kva_declare_masked.h:187,199) and, for a capture without a folder,
   RADIANCE_KVA_CAPTURE_SPLIT under the same constraint (kva_declare_masked.h:189).
5. **Residual width.** `wide = hccfg.hc * n_embd` -- the projector's K, the boundary buffer and the
   copy widths (qwen4exp_kva.cpp:190; kva_declare.h:210,222; kva_declare_masked.h:33,40;
   kva_projector.h:70,199,234; kva_layer.h:212,254). For a plain-residual model the adapter
   supplies `n_embd` (hc = 1, or a different connection entirely -- see 6).
6. **Connection read/write.** the op pair that produces the block input from the stream and writes
   the block output back, issued per layer and sliced by rows: `l.hc_mix/hc_ffn.read/write`
   (qwen4exp_kva.cpp:143-146,174-177; kva_layer.h:237,246,330-333; capture_step qwen4exp_kva.cpp:272,278)
   plus the 97th connection in the epilogue (`m.mixer.read`, qwen4exp_kva.cpp:176). A plain-norm
   model supplies `x = norm(x); x += h` (or the in-tree fused NormQuant) -- and the ext_in contract
   flips (see 9).
7. **Recurrent blocks and their state.** which layers carry state, the KV group, the per-layer state
   shape (heads, V, K), the scan/conv ops, and the undo/apply correction: `m.gcfg.{n_head_v,head_v,
   head_k,chunk}`, `m.kv_state` (kva_declare.h:88-95,117-118; kva_declare_masked.h:207,211-212;
   kva_layer.h:46-76,141-190; kva_fill.h:27-63; qwen4exp_kva.cpp:292-303).
8. **Bulk-end tile G.** the tile the scheduler's quantum and the recurrent scan force bulk ends onto:
   `k.tile = m.gcfg.chunk` (qwen4exp_kva.cpp:86), used by the plan (`pc.tile`, qwen4exp_kva.cpp:128;
   kva_plan.h:49,66), the tail bound (`G = m.gcfg.chunk`, kva_declare.h:149) and the misaligned
   refusal (qwen4exp_kva.cpp:305-309). A model with no recurrent block supplies 1 and the checks
   go vacuous.
9. **Input-norm contract.** whether the block owns its input norm (the fill assumes it does not:
   `ext_in` must hold, kva_declare.h:190,196-201). The adapter must state this and the issue code
   must match (qwen4exp passes `Src::norm = nullptr`, rad:qwen4exp_fp8.cpp:44-46).
10. **Cache-writing pieces per block type** (the lean fill): for attention, the K/V path
    (kva_fill.h:101-114 + qsa_keys 65-99); for the delta net, projections + conv + scan
    (kva_fill.h:27-63); the adapter enumerates them per block type.
11. **MoE dispatch masking.** the drop arm and top_k (kva_declare_masked.h:48; kva_moe.h:53-71),
    the routing-report suppression for bulk rows (kva_moe.h:142-160), the stager probe issue and
    its requirement of >= 3 routed layers below S (qwen4exp_kva.cpp:123-124,201-208; kva_moe.h:35-51),
    and the calibration-tap refusal (kva_declare.h:185-191).
12. **Per-row sparse attention (the straddle path).** whether every late attention layer has the
    per-row gated form and its exactness bound: `qsa_sel/qsa_sequ/op_attn_gq/qsa_exact_to`
    (kva_declare_masked.h:78-86; qwen4exp_kva.cpp:125-126; kva_layer.h:294-322). Absent ->
    `straddle_ok` false and the plan falls back (kva_plan.h:92).
13. **Projector folder schema.** tensor names and shapes: `proj.L.weight [n, wide]` bf16 + bias,
    `st.L [n_head_v*world, head_v, head_k]` f32 all-or-none, `score* [n_vocab_all]` f32
    (kva_projector.h:70-86), plus the manifest's model.meta/encodings/anchors keys (kva_match.h:106-145).
14. **Mode-level measurements.** tail default and minimum (kva_config.h:89,95) -- arguably fit
    facts per model family, not adapter code; today they are CORE constants.
15. **The static oracle.** tests/arch_static_test.cpp holds every issue list to the in-tree plugin;
    each adapter ships its own (qwen4exp_kva.cpp:23-24 says so).

## 3. In-tree radiance 1.0.8 architectures (rad:arch/) and second-adapter candidates

| arch id | file (rad:arch/) | shape |
|---|---|---|
| `llama_fp8` | llama_fp8/llama_fp8.cpp | dense, full attention only (its header: "no gated delta net, no layer schedule") |
| `llama_dense_fp8` | llama_dense_fp8/llama_dense_fp8.cpp | the same graph over an unquantised checkpoint (3 macros + include of llama_fp8.cpp) |
| `qwen35_bf16` | qwen35_bf16/qwen35_bf16.cpp | dense hybrid: 48 GDN + 16 full-attention layers (`full_attention_interval 4`), bf16 |
| `qwen35_fp8` | qwen35_fp8/qwen35_fp8.cpp | the same, block-scaled fp8 |
| `qwen35moe_fp8` | qwen35moe_fp8/qwen35moe_fp8.cpp | the same graph with the FFN routed (3 macros + include of qwen35_fp8.cpp) |
| `qwen4exp_fp8` | qwen4exp_fp8/qwen4exp_fp8.cpp | hybrid MoE + gated residual (hc) + PLE + QSA (this plugin's base) |

### Candidate 1: `qwen35moe_fp8` -- the best second adapter
Same block set as qwen4exp minus the new pieces (rad:qwen4exp_fp8.cpp:10-19: GdnFP8, gated
attention, routed FFN reused; hc/PLE/QSA are NEW). What an adapter would need:
- the identity/shadow facts of §2.1 for `qwen35moe` (claim, .so stem, guard string, include);
- a **connection hook that is not HyperConn**: qwen35's blocks own GemmaRMSNorm inputs
  (rad:qwen35_fp8.cpp:1041,1053,1072 set `Src::norm`; ext_in=false per rad:rad_block_gdn_fp8.h:213),
  so the ext_in contract inverts and the core's `check_fill` refusal fires as written
  (kva_declare.h:190,196-201) -- the fill/masked path must project the block input differently and
  the capture side must find the normed block input inside the block, which the in-tree step does
  not expose (qwen4exp_kva.cpp:272-274 reads `m.a_x.x` right after the hc read);
- residual width = `n_embd` (5120 at 27B), no `hccfg` (every `wide` line of §2.5);
- no PLE (ple_layer absent -> the split constraint of §2.4 becomes "0 <= S < n_layer") and no QSA:
  `straddle_layers` stays false (kva_declare_masked.h:78-86), speed mode loses the tail-only
  straddle and serves masked (the plan already handles it, kva_plan.h:92) -- a capability loss, not
  a correctness one;
- MoE facts: stager lever and kva_drop_rows transfer as-is (kva_moe.h has no qwen4exp type);
- state correction: GdnFP8's shape is read from qwen35's own gcfg -- `kva_state_correct`,
  `kva_rho_update` apply unchanged (kernels/kva.h:170-235).

Projector fit inputs for it: boundary = the residual stream entering layer S, [n_tok, 5120]
bf16 (kva_dump.h:99 writes what it is handed); per-late-layer block inputs = the NORMED input --
**the blocker**: qwen35's norm lives inside the block (fused NormQuant, rad:qwen35_fp8.cpp:985-995),
so `RADIANCE_KVA_CAPTURE`'s current hook (qwen4exp_kva.cpp:272) has no buffer to read; either the
fit targets the pre-norm `x` (the projector then must not reproduce the norm's output, changing
what kva_select overwrites) or the adapter adds a capture variant. Stated plainly: without
resolving the input-norm contract, no projector can be fitted or served.

### Candidate 2: `llama_fp8` / `llama_dense_fp8` -- the model-independence proof
Dense, full attention, no recurrent state at all. Adapter needs: everything above with tile = 1
(no misaligned check, no correction groups, no rho -- kva_declare.h:83-136 drops out), a new
attention K/V fill for AttnFP8 (kva_fill.h:101 is AttnGatedFP8; llama's attention is a different
block, rad:arch/common/rad_block_attn_fp8.h), and no MoE arm (kva_moe.h unused; the masked path
just runs the dense MLP, the drop arm is a no-op). Blockers: (a) the same own-norm contract
(llama blocks norm internally); (b) the lean fill's value must come from K/V writes alone -- fine;
(c) whether AttnFP8 has a per-row sparse form decides if a straddle path exists at all -- none is
visible in rad:rad_block_attn_fp8.h, so `straddle_ok` is false and only lean/masked remain.
It is the cleanest proof that CORE is model-independent but rewrites the most issue code;
qwen35moe_fp8 reuses the most and is therefore the better second adapter.

## 4. qwen4exp facts hard-coded instead of read from model metadata, ranked by breakage

1. **The block-issue code itself** -- kva_fill.h:27-114, kva_layer.h:78-190/263-322, kva_moe.h:35-160
   are verbatim copies of the in-tree GdnFP8/QsaIndexer/AttnGatedFP8/MoeFP8 issues, pinned to
   radiance 140987f line numbers. Not read from metadata at all; a second model with different
   block types does not compile against them, and an engine release that changes a block breaks
   them silently (only the release guard kva_guard.h:206 and the static test stand in the way).
   Breaks a second model the hardest.
2. **`stream_ok = k.split >= 3`** (qwen4exp_kva.cpp:124) -- assumes >= 3 ROUTED (MoE) layers below S
   carry the stager probes (kva_moe.h:35-51; qwen4exp_kva.cpp:201-208). A dense or
   sparsely-routed model loses streaming and with it the masked path (kva_plan.h:95-96), i.e.
   every mode but lean falls back to stock.
3. **The ext_in / own-norm contract** (kva_declare.h:190,196-201) -- hard-coded to "blocks take a
   pre-normed input"; qwen35 and llama violate it and are refused, correctly, but no metadata key
   says which side a model is on.
4. **`wide = m.hccfg.hc * n_embd`** (qwen4exp_kva.cpp:190; kva_declare.h:210,222;
   kva_declare_masked.h:33,40; kva_projector.h:70,199,234; kva_layer.h:212,254) -- the hyper-
   connection width is structural, not in metadata; a plain-residual model has no `hccfg` and the
   projector's K, all buffer widths and the ring block size are wrong (refused by shape check
   kva_projector.h:74-77, so loud, not silent).
5. **The delta-net tile `G = m.gcfg.chunk`** (qwen4exp_kva.cpp:86; kva_declare.h:149) and the
   misaligned refusal (qwen4exp_kva.cpp:305-309) -- a model with no 64-row recurrent tile gets a
   wrong bulk-end rounding (kva_plan.h:66) if the adapter passes the wrong tile; today the plan
   defaults `tile = 64` when unset (kva_plan.h:49).
6. **Recurrent-state shapes** `n_head_v/head_v/head_k` (kva_declare.h:88,117-118,129;
   kva_declare_masked.h:207,211-212; kva_projector.h:81-82; qwen4exp_kva.cpp:298) -- read from the
   in-tree GdnFP8 config, not the container metadata; a different recurrent block (dflash2, dspark --
   rad:arch/common/rad_block_dflash2.h, rad_block_dspark.h) needs different correction ops; the
   kernels' `kva_rho_update` GDN decay (kernels/kva.h:79-96) is semantic, not shape, and does not
   transfer.
7. **`ple_layer`** (kva_declare.h:173-177; kva_declare_masked.h:187,199; kva_projector.h:71-73;
   qwen4exp_kva.cpp:140,152,270) -- harmless elsewhere (absent -> the constraint is vacuous), but
   it is an architectural fact with no metadata key.
8. **`moecfg.top_k` in the drop op and the probe sizes** (kva_declare_masked.h:20-24,48) -- MoE
   facts; a dense model has no `moecfg` and these declarations must drop out.
9. **`straddle_layers`' QSA per-row sparse form** (kva_declare_masked.h:78-86;
   qwen4exp_kva.cpp:125-126) -- qwen4exp-only block feature; degrades to no-straddle, which is
   handled, but nothing in metadata announces it.
10. **`kMinTail = 512`, tail default 2048** (kva_config.h:89,95) -- measured on qwen4exp
    (KVA-FACTS §5), applied to every model; a model whose projector fits at another tail is
    refused below 512 or silently served at 2048. Breaks nothing structurally; wrong operating
    point only.
11. **The shadow/guard strings** (kva_guard.h:165; qwen4exp_kva.cpp:346-350; arch/CMakeLists.txt:9) --
    per-adapter constants, trivial but required for each new plugin.