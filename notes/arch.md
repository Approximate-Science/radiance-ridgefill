# notes/arch.md — ARCH lane lab notes (arch plugin, top-level CMake, static test)

Provenance: radiance `140987f` (v1.0.8), plugin commits `dd93a7f`, `77e52a6` (2026-10-04). Host:
Fedora g++ 16.1.1, cmake 4.4.2. Container build: image `radiance-build` (sha256 `335138ad…`, ROCm 7.2.4,
g++-14, ninja). No GPU and no model were used in this lane.

## 1. Status

| row | state | evidence |
|---|---|---|
| R4/R5 (arch half) | **verified** host-only and in the build image (no kva.so installed yet) | §3 |
| R9 | **built** out of tree inside `radiance-build` with HIP (gfx1201) | §2 |
| R10 | **tested**: v1.0.7 source refused, both versions named | §3 |
| R11 (arch half) | **tested**: `ctest -LE gpu` runs `arch_static` host-only, green | §3 |
| R31 (declare half) | **tested** in the static test (fake builder); not yet run against the engine | §3 |
| R6/R7 | graph/issue identity **tested statically** vs the in-tree plugin; the engine runs need the container | §3 |

## 2. Build commands (exact)

Host-only radiance (once; outside its tree, ~4 min at -j8):
```
cmake -S ~/projects/inference/radiance -B <repo>/build-radiance-host -DRAD_WITH_HIP=OFF \
      -DRAD_WITH_FFMPEG=OFF -DRAD_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=<repo>/build-radiance-host/install
cmake --build <repo>/build-radiance-host -j 8 && cmake --install <repo>/build-radiance-host
```
Plugins host-only (R11):
```
cmake -S <repo> -B <repo>/build-host -DCMAKE_PREFIX_PATH=<repo>/build-radiance-host/install \
      -DRADIANCE_SRC=~/projects/inference/radiance -DRAD_WITH_HIP=OFF
cmake --build <repo>/build-host -j 8 && (cd <repo>/build-host && ctest -LE gpu --output-on-failure)
```
Plugins with HIP, out of tree in the build image (R9):
```
docker run --rm -v ~/projects/inference/radiance:/rsrc:ro -v ~/projects/inference/radiance-kva:/kva -w /kva \
  radiance-build bash -c 'cmake -S /kva -B /kva/build-hip -G Ninja -DCMAKE_PREFIX_PATH=/stage/opt/radiance \
    -DRADIANCE_SRC=/rsrc && cmake --build /kva/build-hip && ctest --test-dir /kva/build-hip -LE gpu --output-on-failure'
```
The plugin served from `<repo>/home` must be the IMAGE-built one: the host build needs GLIBCXX newer than the
runtime image's libstdc++ (the image build tops out at GLIBCXX_3.4.31). Installed with
`cp build-hip/radiance_home/architectures/qwen4exp_fp8.so home/architectures/` (only my target; `cmake --install
build-hip` would also install the KERNELS lane's kva.so, their call).

## 3. Pasted outputs

Build image configure (2026-10-04 17:24):
```
-- radiance 1.0.8: RADIANCE_SRC and /stage/opt/radiance/bin/radiance agree
-- radiance-kva: device rows ON, GPU targets 'gfx1201', installs into /kva/home
[10/12] Linking CXX shared module radiance_home/architectures/qwen4exp_fp8.so
2/2 Test #3: arch_static ......................   Passed    0.42 sec
```
(`kva_kernels_host` was red in both builds at the time: the KERNELS lane's stub test, mid-change; not this lane.)

R4/R5 arch half, in the build image with the KVA home first:
```
I plugin /stage/opt/radiance/share/radiance/architectures/qwen4exp_fp8.so is shadowed by /plugins/architectures/qwen4exp_fp8.so, which comes first on the search path
   3 qwen4exp_kva     0.1.0    architecture  qwen4exp @ (unquantised)
     /plugins/architectures/qwen4exp_fp8.so
```
Exactly one plugin claims `qwen4exp`; no refusal line. Same result host-only.

R10 (radiance v1.0.7 via `git archive v1.0.7` into the scratchpad; the radiance tree untouched), configure exit 1:
```
CMake Error at CMakeLists.txt:61 (message):
  RADIANCE_SRC is radiance 1.0.7 and the installed radiance is not:
  .../build-radiance-host/install/bin/radiance
  carries version string(s) '1.0.8;0.46.1'.  Check RADIANCE_SRC out at the
  installed release, or point CMAKE_PREFIX_PATH at an install of 1.0.7.
```
R11 arch half (`build-host/tests/arch_static_test`): `183659 check(s) over 15 case(s)`, all ok. Cases: off
graph = in-tree graph (TP1, TP2 both ranks, max_spec 0 and 3); off with every kva.* tensor held = in-tree graph;
off step = in-tree step, operand for operand, prefill (64 tok, 2 seq) and decode (2 seq), zero device-API calls;
rad-convert view declares every held copy with the planned shape/dtype/shard/access/group and no op; speed
declares only the selected correction copy (shipped vs swap) and 2 × 3 `kva_state_correct` ops; quality
declares only the selected row table and one `kva_rowsel` (cap 512, M ≤ 2048, rule from the switch); refusals:
tail 4096 > 2048 (both numbers printed), tail 256 < 512, quality without table, kva_state_correct unresolved
(names kva.so), kva_rowsel unresolved, a mode with no projector, a projector with a hole (names
kva.proj.6.weight), unknown switch values (allowed values printed); cap = ceil(share·max_tok/64)·64 ≤ max_tok.

**Negative controls on the test itself** (mutants, not committed): a step that drops the logits (n_out → 0)
fails `off_step_issues_the_in_tree_sequence`; an off declare with one extra weight fails
`off_declares_exactly_the_in_tree_graph` (4 configurations). Both restored; suite green again.

## 4. Decisions and their cost

- **R10 is not "project VERSION vs radiance_VERSION"**: `find_package(radiance)`'s version is the ABI number
  (15.0.0 at v1.0.8; radiance `CMakeLists.txt:400-429`), so that comparison would refuse every build. The
  release lives only in the engine binary (`RAD_VERSION`, `core/CMakeLists.txt:25`): configure requires the
  installed `bin/radiance` to carry RADIANCE_SRC's `project(VERSION)` string (skipped, and said, for a
  headers-only install), AND every installed `abi/` and `arch/` header to be byte-identical to RADIANCE_SRC's
  (the actual dependency of the included .cpp; also refuses a right-tag checkout with local edits). Cost: one
  `file(STRINGS)` over a ~10 MB binary and ~40 SHA256s at configure (<1 s).
- **off declares nothing extra — not even name maps** — so a container holding kva.* serves exactly stock
  in `off` (placement included: declaring the projector would place 1.26 GiB a rank and move experts).
- **Only the selected copy is declared when serving**; `RADIANCE_KVA_DECLARE=all` is rad-convert's view.
  REQUIRED for the Stage 2.3 append and any later append (Stage 6 refit): an `--in-place` append rewrites the
  directory from the planned weights only (`core/format/radfile.cpp:699-718`, `tools/rad_convert.cpp:741-786`),
  so a convert run without it writes only the selected copies, and a later run would DROP the unselected ones
  from the directory (bytes stay; `.pre-append` restores). The sidecar notes' §7 command needs
  `-e RADIANCE_KVA_DECLARE=all` added to its `docker run`.
- **An undeclared tensor in the container is harmless at load**: bring-up walks the PROGRAM's declared weights
  and looks each up by name (`core/engine_bringup.cpp:1719-1736`); only rad-info iterates container entries
  (`tools/rad_info.cpp:197-234`). So unselected copies cost disk only.
- **Every mode but off refuses at declare** ("not implemented in this build") after declaring and checking
  everything: nothing unimplemented returns success. Stage 3 removes it for plumb/speed.
- **Kernel ops declared at Stage 2.2** (2 × `kva_state_correct` per late GDN layer, `kva_rowsel` once) only so
  "kva.so missing" is a startup refusal (R31). Param names/types match the KERNELS lane's schemas
  (`kernels/rows.cpp:35-46`). `kva_rho_update` is not declared yet: it needs the in-tree `w_a_log`/`w_dt_bias`
  handles and the `gdn_ab` liveness decision (Stage 5, §6). With kva.so absent entirely, the real builder's
  validate fails first ("no such op", `core/build/rad_builder.cpp:640-644`) and our refusal follows; with the
  schema present but no row for the machine, ours alone.
- **Sizing declares**: geometry read from `g_model[rank]` (the engine runs every sizing declare after the real
  one, `core/engine_bringup.cpp:616-660`; refused with RAD_E_STATE if not); refusals skipped; `cap` kept at the
  real declare's (also a capacity param in kva.so, `rows.cpp:19-22`); own thread_local scratch and own name
  pool (sizing declares run in parallel threads). R15's grep will hit `static thread_local Kva probe_kva`:
  declare-only scratch, same pattern as the in-tree `qwen4exp_fp8.cpp:301`.
- **Alpha is refused outside [0, 1]** (tcc's documented range, KVA-FACTS §3); widening it is a one-line change.
- **Quality refuses without the selected table even under RADIANCE_KVA_ROWSEL=all**: uniform rule, and the
  kva_rowsel op always has its weight operand. The R35 run selects `RADIANCE_KVA_ROWSEL_TABLE=all`.
- The served recipe (`data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe`) has no catch-all rule, so kva.* stay in the
  checkpoint dtype (bf16 / f32), which is what the declare states.

## 5. Q answers (read from code, not measured)

**Q13 — block output buffers.** Both blocks write their output into the SAME buffer they read their input from:
`x` (= `a_x.x`), bf16 `[max_tok, n_embd]` (`qwen4exp_fp8.cpp:726`; wires `gw.h = aw.h = m.a_x` at :914, :926).
GDN: `out.step(c, w.o, w.h.x, T)` (`rad_block_gdn_fp8.h:510`); attention: `o.step(c, w.attn, w.h.x, T)`
(`rad_block_attn_gated_fp8.h:559`). Both projections are RAD_SHARD_COL (`gdn:376-377`, `attn:419-420`), and the
all-reduce rides in the hc write (`qwen4exp_fp8.cpp:328-331, 1048, 1081`; `rad_block_hc.h:378-397`), so at TP2
`x` holds THIS RANK's PARTIAL sum between the out-proj and the write. That is consistent with the quality path:
gather_rows(x → y_R) gathers partials and `ar_hc_write` over the cap rows sums them. bf16, so `gather_rows`
takes it directly (OPS.md:776-800). The bf16 points before the quantisers, if ever needed: `gdn_o` (`a_go.x`,
`[T, n_head_v, head_v]`, gated in place by `gdn_gated_rmsnorm` :497-500, then int8 codes by `q_o` :507) and
`attn_out` (`a_attn.x`, `[T, q_dim]`, gated in place by `gate_quant_fp8` :555-557). The MoE output is NOT
a plain buffer on this model: the routed rows stay unsummed in `moe_edn` `[rows*top_k, n_embd]` and the hc
write gathers + reduces them (`ar_gather_hc_write`, `qwen4exp_fp8.cpp:1117-1128`, `rad_block_hc.h:381-391`).
Liveness: `x` and `b_h` are read by ops declared after the whole in-tree graph (the projector, gathers), so
both need `rad_buf_concurrent` (D4); `x`'s last in-tree use is `op_gather` (:1149-1153).

**Q11 — GDN a/b.** `w.ab` = buffer `gdn_ab`, bf16 `[max_tok, 2·n_head_v]` with n_head_v THIS rank's value heads
(`qwen4exp_fp8.cpp:790`, divided at :394). Written by `op_ab`, a bf16 `gemm_nt` over the block input's bf16
plane (`rad_block_gdn_fp8.h:289-292`, issued :416) with weight `blk.L.ssm_ab` = concat(in_proj_a, in_proj_b)
along dim 0, row-sharded in two parts `{n_head_v, n_head_v}` (:223, :232-234). So columns `[0, H)` = **a** (the
decay input) and `[H, 2H)` = **b** (beta), per rank, head order = the rank's value heads. Consumed as
`bcol(w.ab, 0, H, T)` → conv_prep's `a` and `bcol(w.ab, H, H, T)` → `b_gate` (:481; OPS.md:622).
`A_log` = weight `blk.L.ssm_a_log` f32 `[n_head_v]` RAD_SHARD_ROW, handle `l.gdn.w_a_log` (:241-242);
`dt_bias` = `blk.L.ssm_dt.bias` f32 `[n_head_v]` RAD_SHARD_ROW, handle `l.gdn.w_dt_bias` (:243-244); both bf16 in
the checkpoint, widened at load. g = −exp(A_log)·softplus(a + dt_bias) is computed INSIDE gdn_conv_prep, and
`gdn_gdec` (`[T, n_head_v]` f32, :800) holds its in-chunk cumulative form, not per-token g — so kva_rho_update
should recompute g from the `a` columns as planned. Check conv_prep's `softplus_thr` default against torch's
(threshold 20) when writing the rho reference. `gdn_ab` is an in-tree transient: rho declared after the graph
needs `rad_buf_concurrent(gdn_ab)` (192 KiB at 2048 rows).

## 6. Stage 3 fill path — what to issue, in order (served config: int8 trunk, rotated experts, bf16 indexer)

Projected activation: the in-tree linears declare NO quantiser of their own, because the hc read writes their
int8 codes (`codes_i8`, `qwen4exp_fp8.cpp:732-733`; `rad_fp8.h:812-817`). So the fill needs its OWN
`quant_act_i8g` (M range 1..max_tok, n = n_embd, group 128, dtype bf16; OPS.md:271, 294; libr4d
`r4d_rows.cpp:4404`) after each projector GEMM, and hands the linears an ActFP8 {x, q8, s8, q8_fed = true}.
(E4M3 trunk → `quant_act_fp8`; bf16 container → none: mirror LinearFP8's three paths.)

Per approximate chunk, after layer S−1's ffn `hc_write` (stock layers 0..S−1 untouched), for each L ≥ S:
1. `gemm_nt_bias(brows(b_h,T), W=proj_w[L], bias=proj_b[L], res=NONE) → activation` — libr4d
   `gemm_bf16_nt_bias_tiled`, any M ≥ 1, K % 8, bias bf16 or f32 (`r4d_rows.cpp:3016, 3175-3183`): **Q2 answered
   in code, resolves at every band**; verify with `--debug-graph`. Then the quantiser.
2. GDN layer (`rad_block_gdn_fp8.h`): `in.step(c, act, w.in.x, T)` :415 → `op_ab` with `brows(act.x,T)` :416 →
   `op_conv_prep` :478-485 → `op_kkt` :486-488 → `op_scan` :489-494 (h0 = ht = kv_cache(kv_state, L), state_idx).
   Skip :497-513 (gated norm, quant, out-proj, all-reduce).
3. Attention layer: indexer FIRST, its front kept together (`rad_qsa.h`): `proj.step(c, act, w.qk, T)` :324 →
   `op_work` :372-381 → `op_bkey` :382-385 → `op_tail` :386-390 (with the kv-batch lookups :345-348, `work`
   :363, `bpos` :369-371); skip q-prep :332-342 and score/select :416-426. Then `kp.step` :452, `vp.step` :453,
   `op_k_norm` :476-477, `op_rope_k` :479 (rope_posmc), `op_kv_store` :482-484 — the UNFUSED path, because the
   fused prologue needs the q|gate GEMM; at T = 2048 > qk_fuse_rows = 64 (`rad_fp8.h:179-181`) stock takes the
   unfused path too.
4. No hc read/write, no MoE for L ≥ S; then stock `mixer.read` and, if `n_out > 0`, gather + logits.

**Arena-aliasing rule for any hand-issued sequence**: transients whose DECLARED op ranges are disjoint may share
bytes, and the issue check only tests range membership, not runtime order. The indexer's buffers are declared
after the attention block's ops (`qwen4exp_fp8.cpp:1049` then :1065), so e.g. `qsa_qk` and `attn_k` have
disjoint ranges and may alias. Safe: keep each block's ops contiguous and the indexer front (proj, q-prep,
work, bkey, tail) together; never interleave two blocks.

**How plumb is byte-identical to stock**: plumb runs every layer's stock control flow; for L ≥ S the block
prefix above is issued by the fill functions with `act` = the REAL `a_x` the stock hc read just wrote, and the
block's tail is issued by hand after it (GDN: :497-513; attention: stock `qsa.step` whole, then fill K/V
(kp, vp, k_norm, rope_k, kv_store), then qg :451, q_norm :474-475, rope_q :478, attention :502-547, gate
:549-557, o :559, ar :560-561). Every op then sees the stock operands; GDN's sequence is the stock sequence
exactly, the attention's differs only by moving the K/V ops ahead of the q path inside the block (fused vs
unfused prologue is byte-identical by the in-tree's own r4d_selftest claim, `rad_block_attn_gated_fp8.h:261-267`).
The static test can assert plumb == stock issue-for-issue for GDN layers and as a multiset for attention ones;
R14 checks the bytes.

**Design proposal for the orchestrator (not implemented; deviates from PLAN D4's `kva_bi`)**: use `a_x` itself as
the projector's output buffer (x ← GEMM, then x.q8/x.s8 ← quantiser). The fill then issues the stock op handles
with the STOCK wiring (`l.gdn.step`'s own `w.h`), which removes every operand substitution from the fill and makes
quality mode simpler: scatter x_R into `x`, quantise, call the stock `GdnFP8::step` / `qsa.step` + `attn.step`
over all rows unchanged (they read `w.h` = a_x, which a plugin cannot redirect otherwise), gather y_R from `x`.
Cost: `x`, `x.q8`, `x.s8` marked concurrent (≈16 MiB at 2048 rows, live nearly all program anyway) vs a separate
`kva_bi` + codes (≈16 MiB more). Risk: none found; the in-tree blocks already treat `x` as scratch between the
read and the write.

## 7. Stages 3-5 (2026-10-04, commits 35adbe3, bb582ea, 8b4266e, 387405b, ca8b01f)

Status: **implemented and statically tested**; nothing below is measured. Engine gates are the orchestrator's.

- **a_x instead of kva_bi (approved deviation from PLAN D4).** The projector writes the model's block input `x`
  (`a_x.x`) and the in-tree `QuantFP8` writes x's codes (int8 on the served formats); every in-tree op then runs on
  its stock operands. `b_h`, `x`, `x.q8`, `x.s8` are `rad_buf_concurrent`. Cost: arena MiB to be read off
  `--debug-placement` (orchestrator's Stage 3 run); expected ≈ 0, since all four are live nearly the whole step anyway.
- **Approximate decision** (`qwen4exp_kva.cpp approximate()`): mode != off, have_proj, !enc, draft_pass == 0,
  n_seq_decode == 0 (via batch_split, so a phase-only batch counts), n_seq == 1, n_spec == 0, n_ahead >= T. Each is
  either a pass-key field or fixed at declare (R15). Static test: the five non-approximate shapes issue the stock
  sequence exactly.
- **Speed late layer** = projector, codes, then the stock block filtered to its cache-writing handles (§6), with
  Stage 4's undo before conv_prep and apply after the scan. **Plumb** = the stock layer with the block issued through
  the same pieces (kva_fill.h) fed the real `x`: equal to stock issue for issue except two moves inside each late
  attention block (indexer q-prep/score/select after its block-key half; K/V before the q path). Plumb declares no
  op and no correction.
- **kva.st is replicated, not ROW-sharded** (deviation from PLAN D6, found at the first TP2 load: the loader has no
  ROW share of a rank-3 weight, `core/format/share.cpp:56`). Each rank holds [48,128,128] (3 MiB a layer, 54 MiB a
  rank) and passes its heads as a weight row slice (offset rank·24·128·128, rows 24; `issue.cpp:661-698` narrows
  and bounds-checks it).
- **kva_state_correct operands** (KERNELS schema 141015c): state, state_idx, applied, applied_idx, C, ND?, nd_idx? —
  each index is its own group's `state_index` at its own pitch, so no group relies on another's slot numbers.
- **Quality layer** (`quality_layer`): projector over all rows; the layer's connection read issued over `h_R` into
  `x_R`/`inj_R` (codes absent: libr4d's hc_read takes q/scale as optional, `r4d_hc_bf16.hip:3351-3374`);
  scatter_rows(x_R → x); codes; the whole block over all rows (stock `qsa.step`+`attn.step`; delta net: project,
  undo, scan, kva_rho_update, apply with ND, tail); gather_rows(x → y_R); the connection write into `h_R` at cap rows
  (decided with T = n_tok, as the block decided its all-reduce); the ffn read over h_R, ONE `MoeFP8::pass(cap, 0,
  cap)` with its routing report, the ffn write (all decided with T = cap). Once per chunk: kva_rowsel (token ids,
  positions, score → rows_idx [cap], mask [n_tok]) and gather_rows(b_h → h_R). Buffers (plugin, concurrent):
  rows_idx [cap] i32, mask [max_tok] i32, h_R [cap,10240], x_R/y_R [cap,2560], inj_R [cap,4] bf16; plus the in-tree
  `gdn_ab` concurrent (rho reads its a columns). At cap 512: h_R 10 MiB, x_R+y_R 5 MiB. kv_kva_rho [heads,1,2] f32
  only when a correction is held.
- **The quality oracle is the in-tree code**: the static test builds each late layer's expected issues by running
  `HyperConn::read/write` and `MoeFP8::pass` on scratch contexts and swapping only the stream operand, at TP1 and
  TP2. 13 mutants across Stages 3-5, each caught.
- **Padding rows** (k < cap): rows_idx −1 → zero rows in h_R; the connection read of a zero row is 0 (norm of zero
  times rsqrt(eps)), the scatter skips it, MoE on a zero row adds 0. They cost compute and add to the routing counts.
- **RADIANCE_KVA_ST=refit with no kva.str.* held** = speed without correction: have_st false, no group, no ops,
  no issues (static test `the_correction_is_declared_and_issued_only_when_held`).
- **Log**: rank 0 prints one `radiance: qwen4exp_kva: kva: approximate step (<mode>, <n> tokens, <ahead> ahead)` per
  approximate step. **Dump**: RADIANCE_KVA_DUMP=<dir> (rank 0, syncs, debug only): boundary.p<P>.npy (f32
  [n_tok,10240]) + boundary.jsonl every approximate chunk (R17); rows.jsonl in quality mode (R39 format,
  notes/sidecar.md §6).
- Greps (arch/): R15 — one hit, `static thread_local Kva probe_kva` (declare-only sizing scratch, same as the
  in-tree's); R27 empty; R28 — `tail = 2048` only, commented as the method's operating choice (PLAN D9). Every
  getenv is in a declare-time function.

## Capture (Stage 6 step 1) — file layout, written before the code

Two debug switches, read at declare. Both SYNCHRONISE the stream and copy device memory to pageable host memory;
neither changes the issue sequence of the step it observes (only host copies are added), and with both unset no
device→host copy exists anywhere in the plugin. "Single-sequence prefill chunk" = `n_seq == 1`, `n_seq_decode == 0`,
`n_spec == 0`, not an encoder or draft pass; any `n_ahead`. `P` = the absolute position of the chunk's first row
(`positions[0]`). `H16` = FNV-1a 64 of the chunk's int32 token ids, 16 hex digits — it keeps two prompts' chunks at
the same `P` apart without any host counter. bf16 tensors are stored as `'<u2'` raw bits: f32 = `(u2 as u32) << 16`.

### `RADIANCE_KVA_CAPTURE=<dir>` — the projector's fitting data (mode must be `off`; rank 0 only)

On the served formats the stock connection read writes BOTH the bf16 block input `x` (`a_x.x`, always — it is a
required output of `hc_read`) and its int8 codes, from the same values in the same kernel. The bf16 consumers
(`op_ab`, the indexer projection) read `x`; the int8 linears read the codes. **`x` is the projector's target**, read
right after layer L's `hc_mix` read and before the block overwrites it with its output. Per chunk:

| file | dtype, shape | content |
|---|---|---|
| `<dir>/chunk.p<P>.h<H16>.boundary.npy` | `'<u2'` [R, hc·n_embd] = [R, 10240] | `b_h` entering layer S (after layer S−1's ffn write) at the captured rows |
| `<dir>/chunk.p<P>.h<H16>.bi.<L>.npy`, L = S … n_layer−1 | `'<u2'` [R, n_embd] = [R, 2560] | `x` right after layer L's connection read, same rows |
| `<dir>/chunk.p<P>.h<H16>.rows.npy` | `'<i4'` [R] | chunk-relative indices of the captured rows: every i with (P + i) % 8 == 0, ascending |
| `<dir>/chunk.p<P>.h<H16>.ids.npy` | `'<i4'` [n_tok] | ALL of the chunk's token ids |
| `<dir>/chunk.p<P>.h<H16>.pos.npy` | `'<i4'` [n_tok] | ALL of the chunk's positions (1-D; on a text pass the rotary position equals it) |
| `<dir>/capture.jsonl` | one line a chunk | `{"prefix": "chunk.p<P>.h<H16>", "chunk_start": P, "n_tok": n, "stride": 8, "rows": R, "split": S, "layers": [S, …, 47], "hidden": 2560, "hc": 4, "dtype": "bf16"}` |

S is the lowest `kva.proj.*` layer the container holds, else its `kva.split` metadata; capture refuses if neither.
Stride 8 from the rows whose position is a multiple of 8 (tcc's capture). At 2048 rows: R = 256, 5 MiB + 24 × 1.25
MiB = 35 MiB a chunk. Not captured: tcc's `final_multi_hidden` (the `final` map feeds only MTP, out of v1).

**HC check.** `bi.S` is the engine's own `hc_read` of exactly the captured `boundary` rows, so
`HC_read(boundary) == bi.S` holds in the engine by construction; the offline check of a CPU HC implementation needs
layer S's connection weights from the container: `blk.S.attn_hc.norm.weight` (bf16), `blk.S.attn_hc_down.weight`
and `blk.S.attn_hc_up.weight` (E4M3 rows with an f32 scale per group of 128 / 80 on the served recipe — dequantise
from `rad-info -v` planes), eps 1e-6, `1 + w` gain (rad_block_hc.h:12-18 for the formula). Comparing
`bi.L` across the captured layers with the fit's targets needs nothing more.

### `RADIANCE_KVA_CAPTURE_STATE=<dir>` — the correction's fitting data (any mode; every rank)

At the end of every single-sequence prefill chunk, each rank copies its late delta-net layers' recurrent state slot
(this rank's value heads), after the layer's scan and BEFORE any correction apply (in speed/quality with a held
correction the copy is taken between the scan and the apply; run the fit's speed arm with `RADIANCE_KVA_ST=refit`
on a container without `kva.str.*` and nothing is applied anyway). Per (chunk, rank):

| file | dtype, shape | content |
|---|---|---|
| `<dir>/state.p<P>.h<H16>.r<rank>.npy` | `'<f4'` [n_late_gdn, H_local, 128 (V), 128 (K)] | layers in ascending order (24,25,26,28,…,46); heads `[rank·H_local, (rank+1)·H_local)` of the model's 48 — the contiguous split of the delta net's own weights; the state's own [heads, V, K] layout (`kv_gdn_state`, = tcc's) |
| `<dir>/state.jsonl` | one line a (chunk, rank) | `{"file", "chunk_start": P, "last_position": P + n_tok − 1, "n_tok", "rank", "world", "heads": [lo, hi], "layers": [...], "approximate": bool, "mode": "off|plumb|speed|quality", "applied_before_copy": false}` |

28 MB a chunk a rank. **Needs kva.so's `kva_state_read`** (KERNELS afce51b): no ABI call returns a KV-pool pointer
(RADIANCE-FACTS §5), so the slot is copied into the plugin buffer `kva_state_copy` [1, H, 128, 128] f32 (declared
only with this switch) and read from there; the capture refuses at declare, by name, when kva.so does not serve it.
Schema: params `M` (range n_seq, issued at 1) `n_head` `sd0` `sd1`; operands `state` RAD_KV(kv_gdn_state, L),
`state_idx` (that group's own slot row), `out`.

### Implemented (commit in the report): what to know when running it

- **Only passes issued live are captured.** A replayed pass does not call `step()` (radiance
  `core/runtime/ctx.cpp:1258-1278`; a prefill key is recorded the second time it is seen and played after). In the
  Stage 3 KL run every bulk chunk logged its line (67/67), i.e. prefill was issued live there, but nothing guarantees
  it for many same-length prompts. **Check `capture.jsonl` / `state.jsonl` line counts against the chunks sent; if
  short, rerun with `--profile-ops`** (recording off whenever profiling is on, `ctx.cpp:120`).
- `RADIANCE_KVA_CAPTURE` refuses at declare unless the mode is `off`. Rank 1 issues the stock step untouched; rank 0
  issues the stock sequence (static test: identical) with the reads interleaved. Reads are full-chunk copies,
  subsampled on the host (≈280 MiB device→host per 2048-row chunk; debug).
- `RADIANCE_KVA_CAPTURE_STATE` works in any mode, on every rank. In speed/quality the copy sits immediately after
  each late layer's `gdn_chunk_scan` and before its `kva_state_correct(apply)` (static test pins the position); on an
  exact or plumb chunk all late layers are copied after the step. One `kva_state_read` issue per copy.
- Static tests (arch_static_test, 29 cases): the capture's read positions (b_h before layer S's first issue, `x` one
  issue after each late connection read), the file names/shapes/dtypes and jsonl fields, rank 1 and two-sequence
  steps untouched, the state copy's operands and position, refusals. Two mutants (read `x` after the block; copy the
  state after the apply) each caught.

## 8. Open / for other lanes

- SIDECAR/orchestrator: add `-e RADIANCE_KVA_DECLARE=all` to every rad-convert run with the KVA home (§4).
- KERNELS: `kva_kernels_host` was red at 17:24 (stub test vs rows in progress).
- Stage 5: in-tree weight handles under a sizing declare live in the inaccessible function-local `probe_model`
  (`qwen4exp_fp8.cpp:301`). Proposal: reuse `g_model[rank]`'s handles (the weight declaration sequence does
  not depend on max_tok, so indices agree) and add a static-test case asserting the sizing declare's weight
  list equals the real one name-for-name.
- Quality-mode hc read/write on compacted rows: `HyperConn::read/write` always address `w.h` = b_h and `w.x` =
  a_x (`rad_block_hc.h:360-397`), so "hc read on h_R" means issuing `op_read`/`op_write` by hand with that
  operand list, or keeping h_R in b_h rows [0, cap) after copying b_h → a plugin `h_S` for the projector.
