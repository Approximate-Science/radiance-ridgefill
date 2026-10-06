# Stage F implementation plan (F.2→F.10 minus void) — REPORT ONLY

Scope: per-request ON/OFF via 64-token marker (`worker-context/ridgefill-marker-spec.json`:
0–59 quad_start(248051)/quad_end(248052) alternate, 60 share / 61 alpha / 62 tail dials,
63 `<|endoftext|>` 248044; on iff `chat_template_kwargs.ridgefill=="on"`).
F.1 done: `notes/stagef-verify.md` 14/14 CONFIRMED (2 with line drift, §2/§4).
Binding deltas (override design): projector ALWAYS host-RAM + 2-slot VRAM ring
(`arch/ridgefill_config.h:84-91`, `arch/ridgefill_projector.h:plan_maps`, `notes/stageb.md` §4
"host placement from 15:40Z, VRAM placement removed"); speed-beside-decoders = path A
(`arch/ridgefill_plan.h:42` PATH_DECODERS, `arch/ridgefill_layer.h:396` `decoders_layer`,
`notes/stageb.md` §4 173dfc4); B merged; C/D in progress on other branches.
Line refs below are this repo at 86cf8bc unless `engine:` (= radiance 140987f, read-only).

## 0. Void / must-change design lines (do not build)

- V1 E.5 placement toggle (`HANDOVER-FIX.md` §10 items 10–11; `PLAN-FIX.md` §14 DD-I;
  R127–R129): VOID. Code already hosts maps + ring (`ridgefill_config.h:84-91`
  PROJ_PLACE/PROJ_RING, `ridgefill_projector.h:plan_maps/take_operands/upload_rank`,
  `ridgefill_layer.h:ring_copy/ring_next`). F builds host+ring only; R127–R129 collapse to
  "host+ring numbers recorded" (see F.9).
- V2 E.6 streaming arm (`HANDOVER-FIX.md` §10 item 11; `PLAN-FIX.md` §15 DD-J; R130–R132):
  VOID — that arm IS the shipped ring. No `rad_memcpy_async`/lane-1 arm, no R130–R132.
- V3 Template bake (`HANDOVER-FIX.md` §10 item 12 `--emit`, `PLAN-FIX.md` §16; R133/R134):
  RETIRED by `PLAN-FIX.md` §17 + `HANDOVER-FIX.md` §11 (PACKAGING): folder + snippet +
  `--override-chat-template`. F.10 = R150 merge/guard only. `tools/ridgefill_template.py:341-390`
  already implements merge/check/strip (no `--emit` — do not add it).
- V4 Per-request speed (`REFUTATION-2-marker.md` §0 table "speed request", §1 mode-token
  speed/quality/test; `PLAN-FIX.md` §11.4 speed column): VOID. Binding: ON=quality only,
  server-wide `speed|quality` unchanged (`arch/ridgefill_config.h:89` has no REQUEST yet).
  Current spec json has dials at 60/61/62 + erase-only test token, not §1's 62-token+mode layout.
- V5 `PLAN-FIX.md` §11.2 "`ridgefill_mask` already runs ahead of layer 0" — actually runs in
  `arch/qwen4exp_ridgefill.cpp:213` `approximate_step` via `mask_rows:182` after prologue/layers 0..S-1
  (impl `notes/impl.md` §3). F detection op must stay where `mask_rows` is callable for views;
  move only if views need layer-0 inputs.
- V6 DD-H early-exit projector (`PLAN-FIX.md` §11.6, R112 kernel): CONDITIONAL ONLY — build iff
  R111 fails on projector term. Default: do not build.

## 1. F.2 Template + ids (mostly built; wire + prove)

Files: `tools/ridgefill_template.py:69` `load_spec`, `:247` `build_block`, `:332/341/353/375`
snippet/merge/check/strip; `tools/template_identity.py:216` `build_suite`, `:275`
`resolve_marker`, `:335/392/471` suite/render/diff, `:548` `reply_format_entry`;
spec `worker-context/ridgefill-marker-spec.json`; engine `core/server/oai.cpp:655-680`,
`core/config.cpp:105,145` (verify items 11–13).
Work: (a) confirm spec json is the single source (pattern rule "alternate…" +
3 dial tables + switch ridgefill/on + token_ids 7 entries); (b) `merge` the stock template,
`check` proves merged == block+base byte-for-byte; (c) document `--override-chat-template`
+ completions-client marker text in README; (d) declare-refusal when container eos ≠ 248044
(see F.4). Engine reliance: kwargs unfiltered (item 11), override path (item 12),
server defaults (item 13), 7 marker pieces single-token + eos 248044 == text_config.eos
(item 14).
Tests: `tests/test_ridgefill_template.py` (generate/diff/idempotence/refusal — CPU-only);
`tests/test_template_identity.py` fake-radiance (CPU-only); GPU: R119 suite
(`/tokenize` stock-vs-merged 0 diffs), R120 reply-format line identical. Unblocks R119/R120,
gates R101+ (no ON traffic before R119/R120 per HANDOVER §10.8).
Size: ~40 lines docs + ~60 tests. GPU: R119/R120 one short server pair, ~15 min (basis:
`notes/stageb.md` §3 session1 15 min for serve+compare).

## 2. F.3a Kernels: detection + `kv_ridgefill_seq` (first, blocks all)

Files: `kernels/ridgefill.h:109` `RidgeFillMask`, `:49` MK_* enum, `ridgefill_mask_window/ridgefill_mask_from`;
`kernels/rows.cpp:42-44` pMask/oMask, `:73-92` schema text, `:279/297` ROWs;
`kernels/host_ref.cpp:ridgefill_mask_parse/ridgefill_mask_host`; `kernels/mask.hip:ridgefill_mask_kernel`;
`arch/ridgefill_declare.h:83` `decl_late_group` (model for seq group), `:42-48` kv groups.
New state: LINEAR `kv_ridgefill_seq` [1,1,8+] i32, bound to layer S via `decl_late_group`
(same call as `kv_ridgefill_applied/kv_ridgefill_rho`), fields `{mode,shift,pos_end,R,L_next,R_next,
share_idx,alpha_idx,T_slot,malformed,...}`; zeroed at admission, snapshotted per checkpoint
(engine `core/mem/kv.cpp:1449-1486`, `abi/rad_runtime.h:33-36` — NOT in F.1's 14 lines; re-verify
at build, risk §5).
New operands (host-row-first: parse+oracle in `host_ref.cpp`, device in `mask.hip`):
`ridgefill_mask` += in `cu` full + `ctx_lens` + `ahead` extent (`praw(token_ids+n_tok,I32,n_ahead)`,
`RAD_NONE` when 0, per PLAN-FIX §11.5) + inout `seq` slots; detection rule per prefill seq:
`ctx_lens==0 && q_len>=64` compare `token_ids[cu[s]..+64)` to 3 patterns (60 fixed + dials +
248044 end); write mode/shift else OFF; malformed (wrong pattern/unknown dial/non-EOS end/
T out of bound per §11.12 table) ⇒ OFF + `malformed++`. Mask rule per seq: OFF ⇒ rows exact;
ON ⇒ quality class rule in its bulk window (host `b` slot rule comes in F.4/F.7).
Engine reliance: quantum-aligned first chunk ≥64 rows (`scheduler.cpp:677-688`, cited REFUTATION-2
§2 — not re-verified in F.1; assert in test). CPU: `tests/kernel_test.cpp` R105 cases
(marker at 0 vs offset-1 vs 63-token prefix vs wrong dial vs non-EOS end vs q_len<64 vs
second-chunk no-redetect) host≡device. Unblocks R105, R118 counters.
Size: ~200 kernel + ~150 test. GPU: kernel_test gpu leg only (~10 min).

## 3. F.3b `ridgefill_views` (shifted batch views, one op/step)

Files: create `kernels/views.hip` + parse/oracle in `kernels/ridgefill.h` + `kernels/host_ref.cpp` +
schema in `kernels/rows.cpp:73` table; issue from `arch/qwen4exp_ridgefill.cpp:182` `mask_rows`
neighbour; consume in `arch/ridgefill_layer.h:88-174` gdn fns, `:242` `masked_layer`,
`arch/ridgefill_fill.h:65/101` `qsa_keys/attn_kv`.
Op `ridgefill_views`: in `cu`, `ctx_lens`, `seq` slots, real `block_table/seqused/slot_mapping/
positions/rope_pos/nc`; out PERSIST `v_block_table` (rows shifted 16 entries = 64/4),
`v_seqused` (−64), `v_rope_pos` [3,T] (`positions−shift`, marker rows clamped 0),
`v_nc` (−16). Host-row-first. Device derives `b_slot` later (F.7); this item writes views
for shift∈{0,64} only. Planner unchanged.
Engine reliance (all F.1-verified): attn causality from seqused/q_len/row + marker rows write 0
(item 1–2 `r4d_attn_prefill_h256_gqa6.hip:122-161,:435/:443-445`); QSA p0 consecutive + rope-plane
mc path (items 3–4 `r4d_qsa_work_bf16.hip:69-93,:109-123`); rope consumers via rope_pos1/mc
(item 10 `rad_arch.h:935-941`, rope_mc item 9 `:741-751`); GDN chunk/cumsum (item 5
`r4d_gdn_conv_w4_h128_bf16.hip:44,:145-178`); PLE EOS/zero cold (item 7 `docs/OPS.md:888-897`,
boundary id item 8 `qwen4exp_fp8.cpp:962`).
Tests: static views-vs-Python-model (CPU-only oracle); mutants: work takes shifted p0;
score takes absolute tables; `nc` not shifted → each caught. Unblocks R102 mechanism, R108.
Size: ~180 + ~120. GPU: none for unit; R102/R108 later.

## 4. F.3c `ridgefill_zero_rows` (erase GDN + PLE inputs)

Files: create `kernels/zero.hip`; same schema/parse pattern as F.3b; call sites:
`arch/ridgefill_layer.h:133` `gdn_prefill_front` (BEFORE `gdn_conv_prep`, not the scan —
HANDOVER §10.6 warning) per late GDN layer (36), and PLE `gv` before `ple_conv`
(engine `arch/common/rad_block_ple.h:453-541` — F.1 §2.3 bonus CONFIRMED).
Op `ridgefill_zero_rows`: in `seq` slots + `cu/ctx_lens`; inout `q/k/v` cols (GDN) or `gv` (PLE);
zeros rows `[cu[s],cu[s]+64)` for ON seqs with `ctx==0`. Operand per layer (36 launches) +
1 PLE ≈ 38 tiny launches/step (REFUTATION-2 §0 cost ≈0.15–0.25 ms).
Engine reliance: GDN zero→silu→l2-norm zero (item 6 `:636-639` eps compiled in);
PLE n-gram EOS + conv-zero (item 7); marker never out_ids (`batch.cpp:1006-1013` cited
REFUTATION-2 §3f — not in F.1 lines; assert in R102 test).
Tests: host≡device byte tests (CPU-only); positive controls for R102 (skip-GDN / skip-PLE /
non-EOS-last each must break floor). Unblocks R102.
Size: ~100 + ~80. GPU: R102 gate group (see §9).

## 5. F.3d `ridgefill_readback` + host plan rule input (DD-G, on for this deployment)

Files: create `kernels/readback.hip`; pinned host buffer allocated at open via
`abi/rad_device.h` (same pattern as `arch/ridgefill_dump.h:54` `dump_read` + hazard instrument
`PLAN-FIX.md` §5.4); read on NEXT step for logging+planning only.
Op `ridgefill_readback`: in `seq` slots (ON + last-prefill-seq only); out host buffer
`{pos_end,R,L_next,R_next}` per live ON slot where `L_next=min(remaining,budget)` clamped to
checkpoint+quantum (`scheduler.cpp:661-688`), `R_next=R−L_next` or capped. `n_ahead` arrives as
ahead-token extent (PLAN-FIX §11.5), never ranged param (`abi/rad_abi.h:54`,
`core/runtime/issue.cpp:811-814` per `notes/impl.md` §1).
Host rule lives in F.4 (planner), this item only supplies records + `top_level` capture
(sizing declares `engine_bringup.cpp:612-680` = stager `min_rows−1`, `:2563-2566`).
Tests: unit on record computation vs Python scheduler rule (CPU-only); engine R114 log-line
counts. Unblocks R111/R113/R114.
Size: ~120 + ~80. GPU: R114 counting (~10 min) + R111/R113 loads.

## 6. F.4 Arch: `request` mode + derive + batch COPY + MTP views

Files: `arch/ridgefill_config.h:89` Mode enum + `:238` `read_config` (add `request` to
`RADIANCE_RIDGEFILL` choice; add `RADIANCE_RIDGEFILL_READBACK=on|off` default off, `RADIANCE_RIDGEFILL_READBACK_FORCE`
debug); `arch/ridgefill_plan.h:44` PlanMode + `:47-68` PlanIn/Config/Pass + `:88` `plan_pass`
(add request rule §11.5: recordable `n_tok<=top_level` or READBACK off ⇒ stock; else masked iff
ON prediction matches keyed `n_ahead`/`n_tok-DT`; lean NEVER — static mutant); `arch/qwen4exp_ridgefill.cpp:106`
`derive` (fill PlanIn incl. `top_level` from sizing declares; READBACK refused by name when none ran),
`:182` `mask_rows` (issue detection+views+readback), `:213` `approximate_step`, `:242` `log_pass`
(add `plan=stock|masked,reason=...` fields for R114), `:332` `step`; `arch/ridgefill_declare.h:42-48`
RidgeFill struct (add `kv_ridgefill_seq`, view PERSIST bufs, readback host buf, `top_level`, eos id check),
`:83` `decl_late_group` (declare `kv_ridgefill_seq` same call), `:127-130` rho/views declares;
`arch/ridgefill_declare_masked.h:57` `decl_masked` + `:161` `decl_selected` (refuse `request` if container
eos ≠ 248044 — F.1 §2.2: compare `text_config.eos_token_id`, NOT tokenizer_config `eos_token`;
refuse READBACK=on with no sizing declare; marker-template-but-no-projector guard for F.10/R150);
`arch/ridgefill_layer.h:19/28` slot helpers (reuse for `kv_ridgefill_seq`), `:88-174` GDN fns (take view COPY),
`:242` `masked_layer`; `arch/ridgefill_fill.h:65/101` (same views); projector untouched (host+ring only).
Batch COPY: F.1 relies on "no arch code calls `rad_batch()`" — `grep rad_batch arch/` here returns
empty (verified 2026-10-05); assert that grep in the static test (HANDOVER §10.4). Pass key hashes
REAL batch (`ctx.cpp:979-1013`, not in F.1 — re-verify); view op is recorded ⇒ replay rebuilds.
MTP: same shifted views in history/draft passes (plugin owns them per REFUTATION-2 §3h) —
DEPENDS on Stage D (`final`/MTP passes in progress); F.4 builds the hook point, R109 runs after D.
Engine reliance: RadBatch has no host identity (`abi/rad_runtime.h:43-226`, REFUTATION-2 §6 —
not in F.1 lines, re-verify); armed⇒live `ctx.cpp:1262`, audit fail `ctx.cpp:1287-1298` (same).
Tests: R52″ truth table + R107 (CPU-only, `tests/arch_static_test.cpp` pattern `:941`);
mutants: lean allowed in request; detection on non-first chunk; views on real batch; readback used
at `n_tok<=top_level` — each caught. Unblocks R101/R103/R104/R107/R114.
Size: ~250 arch + ~200 static. GPU: none (static host) until §9 gates.

## 7. F.7 Dials (R115–118,123,125) — after erase works

Files: `kernels/ridgefill.h:109` RidgeFillMask + `host_ref.cpp:ridgefill_mask_parse/host` + `mask.hip` (read
`share_idx/T_slot` from `kv_ridgefill_seq`; device-derived `b_slot=e−roundup_G(T_slot−n_ahead)`);
`kernels/rows.cpp:57` pCorrect + `arch/ridgefill_declare.h:115-117` (add OPTIONAL `alpha_slot` operand
to `ridgefill_state_correct`: `s=alpha_param×table[alpha_idx]×ρ`; absent ≡ today — `kernel_test`
host≡device both forms, CPU-only); `arch/ridgefill_layer.h:152` `gdn_prefill_scans` (per-slot-T≠server-tail
⇒ end-variant: one scan, rho whole chunk, apply after — current `STRADDLE=end` path
`arch/ridgefill_config.h:96-104` + `ridgefill_plan.h` split flag); `tools/ridgefill_template.py:69` (dials from spec
single source — already: `_dial_lines`, unknown value ⇒ no marker ⇒ OFF).
Host rule: issue `end` whenever `request` on and server tail not the only allowed T (device writes
`t_matches`, host can't read in time — log it). Malformed ⇒ OFF + `malformed` counter (F.3a).
R123: `ridgefill_split`/`ridgefill_proj` kwargs render no marker (template: unknown-kwarg ⇒ OFF + counter).
R125: readback prediction uses `T_slot` (`R_next+L_next>T_slot` else tail ⇒ stop matching).
Tests: R115 per-T end-vs-split NLL (last-512 CI∋0, whole-tail ≤+0.01); R116 share ordering;
R117 alpha-0 bits-identical/correction-off + 0.5 between; R118 malformed⇒stock+counter (all GPU).
Size: ~150 + ~120. GPU: R115–118 group ~90 min.

## 8. F.8 Compat suite (R119–124) + F.9 residual (R126) + F.10 merge guard (R150)

F.8 files: `tools/ridgefill_template.py:341/353/375` merge/check/strip (BUILT — add `--diff` alias if
missing + idempotence/refusal cases); `tools/template_identity.py:216/335/392/471/548`
suite/render/diff/replyfmt (BUILT); base template `worker-context/container-chat-template.jinja`.
R119: `/tokenize` suite (system/user/assistant, tools+tool_calls, thinking on/off+budget,
reasoning_effort, multimodal, add_generation_prompt:false) stock-vs-merged 0 diffs;
R120: startup reply-format line identical; R121: marker-as-user-content/mid-prompt/tool-result/
63-token-completion/offset-1 ⇒ OFF, completion offset-0 ⇒ ON; R122: OFF decode ms/step = stock;
R124: tool runs fresh-clone+container-only, idempotent, refuses `ridgefill`-symbol containers.
RUN R119/R120 BEFORE any ON request on prod server (HANDOVER §10.8). CPU: template unit tests;
GPU: R119–122 ~45 min.
F.9 (R126): R77/R80 rerun with `RADIANCE_RIDGEFILL=request`, no ON traffic (resident experts/arena/
decode C=1/8/32 vs stock) + DD-I(ii) host+ring measured once (IS the config: decode 0±noise,
ON 16K TTFT recorded). GPU ~60 min. (R127–132 VOID per §0 — record host+ring numbers only.)
F.10 (R150 only): `merge` snippet into operator template; WITHOUT `ridgefill` ⇒ byte-identical over R119
suite + reply line unchanged; merged + NO valid projector ⇒ startup refuses by name (new guard in
`decl_selected`, §6); with projector ⇒ serves; no-snippet ⇒ never triggers. GPU ~20 min.

## 9. Gates in order + dependency graph + sizes

Order (HANDOVER §10.5): R101 STOP ⇒ R102 (+3 positive: skip-GDN/skip-PLE/non-EOS-last break floor;
1 negative: shift-0 large KL) ⇒ R105 ⇒ R103 ⇒ R104 ⇒ R107 ⇒ R106* ⇒ R109* ⇒ R110 ⇒ R114 ⇒
R108 soak ⇒ R111 (paired stock, same boot; mixed + OFF-only) ⇒ R113 (force on/off byte-identical +
TTFT moves; negative ctrl non-staging ⇒ RAD_E_STATE, READBACK-off clean) ⇒ R112 ONLY if R111 red.
Then R115–118/123/125 ⇒ R119–124 (R119/120 first) ⇒ R126 ⇒ R150 ⇒ R108 soak.
* R106 needs Stage C (snapshot carries `kv_ridgefill_seq`; `kv.cpp:1459-1478` re-verify); R109 needs Stage D.
Graph: F.2 ∥ F.3a → F.3b+F.3c ∥ F.3d → F.4 → gates; F.7 after R102; F.8 ∥ F.7 (R119/120 gates ON
traffic); F.9/F.10 last. Critical path: F.3a→F.3b→F.3c→F.4→R101→R102→R103/104→R111→R113.
Sizes (lines): F.2 100 · F.3a 350 · F.3b 300 · F.3c 180 · F.3d 200 · F.4 450 · F.5 200 ·
F.7 270 · F.8 150 · F.9/F.10 120. Total ≈ 2.3 k (+ tests ≈ 1 k). GPU minima: R101 30 · R102 60 ·
R103/104 45 · R105/R107/R114 20 · R106* 30 · R109* 30 · R110/R122 20 · R108 120 · R111 90 ·
R113 60 · R115–118 90 · R119–124 45 · R126 60 · R150 20 ≈ 11 h + 2 k-soak machine time.

## 10. Stop rules + residual risks

- R101 red (no-marker ≠ stock bytes/3 boots/KL floor) = switch not inert ⇒ STOP, run nothing else
  (HANDOVER §10.5/§10.6). Trigger: views/planes/zeroing leak into marker-free path.
- R102 red (erase≠marker-free at exact-vs-exact floor 1.15e-7, `notes/impl.md` §4 R94/R35′ value)
  or any positive control NOT breaking floor ⇒ STOP. Trigger: wrong view per consumer (first suspect
  QSA work absolute-vs-shifted split, `r4d_qsa_work_bf16.hip:79-101` item 3); GDN zeroing after
  `gdn_conv_prep` (`ridgefill_layer.h:133` order); non-EOS last token; PLE `gv` vs `gvn` operand
  (`rad_block_ple.h:516-536` F.1 §2.3).
- R113 negative control NOT failing on non-staging server (`--placement all_vram`) ⇒ precondition
  analysis wrong ⇒ STOP, unsafe to document (HANDOVER §10.6). Trigger: `top_level`/staging detect
  wrong (`engine_bringup.cpp:2563-2572` not in F.1 — re-verify before F.4).
- R112 iff R111 red (paired OFF ΔTTFT CI ∌ 0 at R1 ±1% spread): decompose projector vs probes via
  `--profile-ops`; build DD-H `ridgefill_proj_gemm_bias` only if projector term > noise (else probes-only
  fix). R111 first-chunk misses (one stock chunk/ON request) and tie false-positives (OFF middle
  chunks match ON `n_ahead=max_tok`) cost TIME only — never output (mask-only approximation +
  no lean in request, PLAN-FIX §11.5 proof; R52″ pins it).
- Unverified-at-F.1 lines to re-verify at build (STOP if differ): `kv.cpp:1449-1486` (seq slots
  snapshotted), `scheduler.cpp:661-688` (quantum/chunk rule), `ctx.cpp:979-1013/1257-1331/1287-1298`
  (key/tape/audit), `batch.cpp:24-53/1006-1013/1033/1066-1080` (max_ctx/n_out/seq_ids/n_ahead),
  `abi/rad_runtime.h:43-226/170-190/313-330` (identity/rope_pos/lanes), `rad_arch.h` rope helpers
  beyond `:935-941`, `oai.cpp:1445-1447/1525/1648` (special-token parse, no BOS), R100 bound
  `T<2C−G` for dial T (`ridgefill_declare.h:142` check_mode).
- Batch-shape noise (NOT byte-identity ON-vs-marker-free): chunk ends at real 1984+2048k, last-chunk
  M differs, 64 rows dense→sparse at ≤2051 tokens (QSA.md:70), dense/sparse switch reads unshifted
  `max_ctx_len` (`rad_block_attn_gated_fp8.h:502-504` — F.1 item via `ridgefill_declare_masked.h` usage,
  re-verify). Bar = ident.sh rule + KL floor with 64-rebased ref (R102).
