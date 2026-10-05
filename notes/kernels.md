# KERNELS lane notes — `kva.so` (kva_rowsel, kva_rho_update, kva_state_correct, kva_state_read)

Owner: KERNELS lane. Files: `kernels/` (CMakeLists.txt, kva.h, rows.cpp, host_ref.cpp, rowsel.hip,
rho.hip, state_correct.hip), `tests/kernel_test.cpp`, `tests/rho_ref.py`, `tests/fixtures/rho_numpy.txt`,
this file. Radiance `140987f` (v1.0.8).

**Stage A (PLAN-FIX v2, 2026-10-04, last section) replaced `kva_rowsel` by `kva_mask` and added `kva_select`,
`kva_drop_rows` and optional `bounds` operands; §1-§2 below describe the library before it and are kept as history.**

## 1. The three ops — schemas and operand order (for the ARCH lane)

Operands are POSITIONAL, in this order. `?` = optional (pass `RAD_NONE`). Params in `RAD_PARAMS` order
do not matter to the engine, but every listed param is REQUIRED (except where noted).

| op | params | operands (in order) |
|---|---|---|
| `kva_rowsel` | `M` (RAD_RANGE, rows of the chunk) · `cap` (int, CAPACITY role) · `share` (f64, 0..1) · `seed` (int) · `mode` (str: `class`/`random`/`all`) | 0 `token_ids` i32 [n] (`praw(batch->token_ids, RAD_I32, n)`, n = n_tok) · 1 `positions` i32 [n] (`praw(batch->positions, RAD_I32, n)`; a component-major [c, n] operand is also accepted, row 0 read at its strides) · 2 `score` **weight** f32 [vocab] (`kva.rowsel.score`) · 3 `rows_idx` out i32 [cap] · 4 `mask` out i32 [n] |
| `kva_rho_update` | `M` (RAD_RANGE, rows) · `n_head` (int, this rank's GDN value heads) | 0 `a` bf16 or f32 [n, n_head], read at its own row AND column strides (the in-tree layout puts the a columns first: `bcol(w.ab, 0, H, n)`, radiance `arch/common/rad_block_gdn_fp8.h:481`) · 1 `mask` i32 [n] (kva_rowsel's mask) · 2 `A_log` **weight** f32 [n_head] · 3 `dt_bias` **weight** f32 [n_head] · 4 `ND` inout f32, the LINEAR group `kv_kva_rho` layer cache `[n_states, n_head, 1, 2]` (or any trailing dims whose product is 2) · 5 `state_idx` i32 [1, pitch] (`praw2(st_idx, RAD_I32, 1, st_w)`; exactly ONE sequence) |
| `kva_state_correct` | `M` (RAD_RANGE, n_seq) · `mode` (str: `undo`/`apply`) · `alpha` (f64; read in apply, ignored in undo, still required) · `n_head` · `sd0` · `sd1` (ints: GDN state per head is sd0×sd1) | 0 `state` inout f32 `RAD_KV(kv_gdn_state, L)` [n_states, n_head, sd0, sd1] (strides read off the operand) · 1 `state_idx` i32 [n_seq, pitch] — **kv_gdn_state's** `state_index` (column 0 used) · 2 `applied` inout f32 `RAD_KV(kv_kva_applied, L)` [n_states, n_head, 1, 1] · 3 `applied_idx` i32 [n_seq, pitch] — **kv_kva_applied's** `state_index` · 4 `C` **weight** f32 [n_head, sd0, sd1] (`kva.st.L`) · 5 `ND`? f32 `RAD_KV(kv_kva_rho, L)` · 6 `nd_idx`? i32 [n_seq, pitch] — **kv_kva_rho's** `state_index`. ND and nd_idx are present or absent TOGETHER (speed mode: `RAD_NONE, RAD_NONE` → rho = 1; one without the other is RAD_E_INVAL). All three index arrays must name the same n_seq rows (else RAD_E_SHAPE). |
| `kva_state_read` | `M` (RAD_RANGE, n_seq) · `n_head` · `sd0` · `sd1` | 0 `state` in f32 `RAD_KV(kv_gdn_state, L)` [n_states, n_head, sd0, sd1] (strides read off the operand; padded slots fine) · 1 `state_idx` i32 [n_seq, pitch] (column 0; a slot outside the pool, or negative, reads ZEROS) · 2 `out` out f32, written densely as [n_seq, n_head, sd0, sd1] from its first element (any contiguous operand at least that big; smaller is RAD_E_SHAPE) |

Semantics (also in each schema's doc string, `rad-schemas kva.so`):

- `kva_rowsel`: a row MATCHES when `score[token_ids[i]]` is finite (an id outside `[0, vocab)` does not
  match). `k = rint(share × matches)` — round half to EVEN, exactly Python's `round()` in `fnlev/rules.py`.
  `class`: row kept iff it matches and fewer than k matching rows precede it by (score desc, row asc).
  `random`: same k, over ALL n rows, ranked by `kva_row_hash(seed, absolute position)` ascending (ties by
  row), the position read from `positions`. Pass `batch->positions` (the token's index in its sequence,
  `abi/rad_runtime.h:51`), NOT `batch->rope_pos`: rope_pos is the [3, n] rotary position, which runs behind the
  index after an image (`abi/rad_runtime.h:170-185`). Only random mode reads `positions`.
  `all`: k = n, ranked by row. At most `min(k, cap, rows_idx extent)` rows are kept — the best ones
  (the truncation case). `rows_idx` = kept rows ascending, `-1` padded to its extent; `mask[i] = 1` for
  rows NOT kept (approximated), 0 for kept. Rows are CHUNK-relative (0..n-1).
- `kva_rho_update`: per head, sequentially over the n rows: `g = -exp(A_log)·softplus(a + dt_bias)`
  (softplus threshold 20), `D = e^g·D + 1`, `N = e^g·N + mask[t]`, carried in the ND slot across chunks
  (zeroed at admission by the engine). The slot is `state_idx[0]`; a slot outside the pool writes nothing.
- `kva_state_correct`: per (sequence, head), each pool is addressed through ITS OWN index: state slot =
  `state_idx[s][0]`, applied slot = `applied_idx[s][0]`, ND slot = `nd_idx[s][0]` (the KV manager gives every
  stateful group the same slot today, radiance `core/mem/kv.cpp:478-484, 676-691`, but that is not a contract).
  A sequence with ANY slot outside its pool (or negative) is skipped. Pools may have different depths.
  `undo`: `state -= applied·C; applied = 0`. `apply`: `s = alpha × (ND ? kva_rho(N, D) : 1)` with
  `kva_rho = D > 0 ? clamp(N/D, 0, 1) : 1`; `state += s·C; applied = s`. **When the scale is 0 nothing is
  added** (so `alpha = 0` leaves every state bit alone, incl. the sign of a −0.0 — R22).

- `kva_state_read`: a straight copy, `out[s, h, i, j] = state[slot_s, h, i, j]`; for the Stage 6 correction refit
  (no ABI call returns a KV-pool pointer). Two sequences may name one slot (it is a read).

Issue order the plan wants per late GDN layer L (HANDOVER Stage 4.2 / 5.3): `kva_state_correct(undo)`
before the layer's `gdn_conv_prep`; after `gdn_chunk_scan`: `kva_rho_update` (quality mode only), then
`kva_state_correct(apply)` with ND present (quality) or `RAD_NONE` (speed). `mode` is a fixed parameter,
so undo and apply are two declared ops per layer.

## 2. Fixture format for R33 (`tests/fixtures/rowsel_quick9.json`, written by the SIDECAR lane)

The SIDECAR lane committed the fixture (caf4dc1, `tools/rows_compare.py fixture`) before my proposed format
landed, in its own format `kva-rowsel-fixture-1`; the test reads THAT format (no change asked of them):

```json
{ "format": "kva-rowsel-fixture-1", "window": 2048, "share": 0.25, "classes": ["cap","mixed","piece"],
  "vocab": 248320, "kept_ids": [..1757 ids..], "kept_scores": [..their f32 scores..],
  "score_sha256": "...", "rule": "...",
  "docs": [ { "doc": "ppl/8k/0", "token_ids": [..2048..], "matches": 475, "k": 119, "rows": [..119..] }, ... ] }
```

The test rebuilds a [vocab] f32 table (kept ids → score, every other id −inf), runs `mode=class`, `seed=0`,
`share` from the file, `cap = ceil(share × window / 64) × 64` (PLAN D12's capacity, 512), and checks per doc:
kept rows == `rows`, kept count == `k`, rows_idx ascending + `-1` padded, mask 0 exactly on kept rows; in the gpu
group also device == host. Env `KVA_ROWSEL_FIXTURE` gives the path (CTest sets it); absent file → SKIP by name.

## 3. Build and test commands

```
# host-only (no ROCm): against the host-only radiance install the ARCH lane built
cmake -S kernels -B build-kernels-host -DCMAKE_PREFIX_PATH=$PWD/build-radiance-host/install
cmake --build build-kernels-host -j8 && ctest --test-dir build-kernels-host -LE gpu
# HIP, in the radiance-build image (rootless docker: run as container root, no --user)
docker run --rm --security-opt label=disable -v $PWD:/kva -w /kva radiance-build sh -c \
  'cmake -S kernels -B build-kernels-hip -G Ninja -DCMAKE_PREFIX_PATH=/stage/opt/radiance && cmake --build build-kernels-hip'
# device cases: all of /dev/dri must be passed (one render node alone -> HSA_STATUS_ERROR_OUT_OF_RESOURCES);
# the card is picked with ROCR_VISIBLE_DEVICES mapped from its PCI bus id via rocminfo's GPU agent order,
# under flock ~/AI-Work/radiance-kva-plugin-20261004/gpu.lock, kernel log checked before/after.
```

## 4. Log

### Stage 1 — declared, every launch a stub (R8)
(outputs pasted below at the Stage 1 commit)

2026-10-04, radiance 140987f. Host-only build (no ROCm, against `build-radiance-host/install`):
```
$ ./build-kernels-host/kernel_test build-kernels-host/radiance_home/kernels/kva.so host
  ok   descriptions_cover_schemas
  kernel kva_rowsel_host (kva, host domain) refused op 'kva_rowsel': unsupported
  kernel kva_rho_update_host (kva, host domain) refused op 'kva_rho_update': unsupported
  kernel kva_state_correct_host (kva, host domain) refused op 'kva_state_correct': unsupported
  ok   stub_refuses
  SKIP described_operands_launch: every row in this domain is still a stub
39 check(s), group host
$ ctest --test-dir build-kernels-host      -> kva_kernels_host Passed, kva_kernels_gpu Skipped (no HIP)
$ rad-info --plugins --home build-kernels-host/radiance_home:<host install>/share/radiance
   0 kva              0.1.0    kernels          3 kernel(s),   3 schema(s), built for host
$ rad-schemas kva.so
  kva_rho_update     params M:int n_head:int
                     operands a:in mask:in A_log:weight dt_bias:weight ND:inout state_idx:in
  kva_rowsel         params M:int cap:int share:f64 seed:int mode:str
                     operands token_ids:in score:weight rows_idx:out mask:out
  kva_state_correct  params M:int mode:str alpha:f64 n_head:int sd0:int sd1:int
                     operands state:inout applied:inout state_idx:in C:weight ND:in?
```
HIP build (radiance-build image, ROCm 7.2.4), device group on the card at PCI 0000:13:00.0:
```
ROCR_VISIBLE_DEVICES=1
  device 0 of 1 visible, PCI 0000:13:00.0
  ok   descriptions_cover_schemas
  kernel kva_rowsel_device (kva, device domain) refused op 'kva_rowsel': unsupported
  kernel kva_rho_update_device (kva, device domain) refused op 'kva_rho_update': unsupported
  kernel kva_state_correct_device (kva, device domain) refused op 'kva_state_correct': unsupported
  ok   stub_refuses
  SKIP described_operands_launch: every row in this domain is still a stub
39 check(s), group gpu
kernel log clean before and after (no amdgpu MES/SMU/timeout/reset lines)
$ rad-info --plugins (HIP build)  ->  kva 0.1.0 kernels 6 kernel(s), 3 schema(s), built for gfx1201 host
$ grep -nE 'hipMalloc|getenv|hipDeviceSynchronize|hipStreamSynchronize' kernels/   -> empty (R21)
```
The engine reports a refusing launch as `kernel <name> (<plugin>, ..., <domain> domain) refused op '<op>'
at ...: unsupported` and fails the step (radiance `core/runtime/issue.cpp:1197-1222`, `abort_step`); the
test prints the same shape so R8's "logged by name" is visible without a model.

### Host rows, device rows, tests (Stages 2-4 of this lane)

2026-10-04, radiance 140987f, ROCm 7.2.4 (radiance-build image), card PCI 0000:13:00.0 (ROCR_VISIBLE_DEVICES
mapped from its BDFID), kernel log checked clean before and after every GPU run.

Host group (host-only build, no ROCm):
```
  ok   descriptions_cover_schemas
  SKIP stub_refuses: no stubbed row remains in this domain (R8 retired: every row implemented)
  ok   described_operands_launch
  ok   rowsel_class_semantics
  ok   rowsel_random_and_all_semantics
  ok   refuses_bad_operands
  ok   state_correct_semantics
  ok   rho_semantics
  ok   rho_in_unit_interval
  rho vs NumPy: max |rho diff| 1.55e-06, max rel N/D diff 2.15e-06
  ok   rho_matches_numpy_reference
  ppl/8k/0   n 2048  k 119  == fnlev.rules
  ppl/8k/1   n 2048  k 101  == fnlev.rules
  ppl/8k/2   n 2048  k 78  == fnlev.rules
  ppl/16k/0  n 2048  k 127  == fnlev.rules
  ppl/16k/1  n 2048  k 144  == fnlev.rules
  ppl/16k/2  n 2048  k 140  == fnlev.rules
  ppl/16k/3  n 2048  k 124  == fnlev.rules
  ppl/32k/0  n 2048  k 94  == fnlev.rules
  ppl/32k/1  n 2048  k 101  == fnlev.rules
  ok   rowsel_matches_fnlev_rules
450 check(s), group host
```
Device group (HIP build), the same binary's `gpu` cases:
```
card 0000:13:00.0 (BDFID 4864); kernel log clean before
ROCR_VISIBLE_DEVICES=1
  device 0 of 1 visible, PCI 0000:13:00.0
  ok   descriptions_cover_schemas
  SKIP stub_refuses: no stubbed row remains in this domain (R8 retired: every row implemented)
  ok   described_operands_launch
  ok   refuses_bad_operands
  rho vs NumPy: max |rho diff| 1.55e-06, max rel N/D diff 2.15e-06
  ok   rho_matches_numpy_reference
  ppl/8k/0   n 2048  k 119  == fnlev.rules == host row
  ppl/8k/1   n 2048  k 101  == fnlev.rules == host row
  ppl/8k/2   n 2048  k 78  == fnlev.rules == host row
  ppl/16k/0  n 2048  k 127  == fnlev.rules == host row
  ppl/16k/1  n 2048  k 144  == fnlev.rules == host row
  ppl/16k/2  n 2048  k 140  == fnlev.rules == host row
  ppl/16k/3  n 2048  k 124  == fnlev.rules == host row
  ppl/32k/0  n 2048  k 94  == fnlev.rules == host row
  ppl/32k/1  n 2048  k 101  == fnlev.rules == host row
  ok   rowsel_matches_fnlev_rules
  undo  alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  apply alpha 0.7 ND 1: 10031712 state bytes, 0 differ (max abs diff 0)
  undo  alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  apply alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  apply alpha 0.0 ND 1: 10031712 state bytes, 0 differ (max abs diff 0)
  ok   state_correct_device_matches_host
  90 configurations: device rows_idx and mask == host
  ok   rowsel_device_matches_host
  device vs host: max |rho diff| 2.68e-07, max rel N/D diff 7.67e-07
  ok   rho_device_matches_host
495 check(s), group gpu
kernel log clean after; exit 0
```
ctest: host-only build `kva_kernels_host` Passed, `kva_kernels_gpu` Skipped (no HIP); HIP build `ctest -LE gpu`
Passed 1/1; `ctest -L gpu` on the card Passed 1/1. Also verified under the ARCH lane's top-level CMake, host-only
(`-DRAD_WITH_HIP=OFF`) and HIP (docker): `kva` + `kernel_test` build, `kva_kernels_host` passes.

Numbers (all from the outputs above):
- R20 `kva_state_correct`: host vs device on 6 slots × 24 heads × 128×128 f32 with padded slot/head/row strides,
  4 sequences (one with slot −1, a junk speculation column in state_idx), nonzero applied scales, 5 steps
  (undo, apply α0.7+ND with some N>D, undo, apply α1, apply α0+ND): **0 of 10,031,712 state bytes differ in every
  step; applied identical** (max abs diff 0).
- R33 `kva_rowsel`: 9/9 quick docs' first 2048-token windows: host == fnlev.rules, device == host == fnlev.rules
  (k 78–144 rows, 3.8–7.0% of the window). 90 random configurations (n ∈ {1, 7, 300, 2048, 8192}, class/random/all,
  cap ∈ {3, 512, n} incl. k > cap truncation, share {0.25, 1}, ids outside the table, heavy ties): device rows_idx
  and mask == host. Random mode: exactly k rows, same rows twice for a seed, different rows for another seed.
- R34 `kva_rho_update`: vs the NumPy transcription of st_hook.py (closed-form cumsum, 6 heads, 3 chunks carried,
  slow head A_log −4, fast head A_log 4, one row past softplus' threshold): max |Δrho| 1.55e-6, max rel ΔN/ΔD
  2.15e-6 (host and device alike). Device vs host (24 heads × 2048 rows × 2 chunks, bf16 a as a column slice of
  pitch 48, exact shares 0.056 / 0.5 / 0): max |Δrho| 2.68e-7, max rel ΔN/ΔD 7.67e-7. rho == 1 to the bit when no
  row is exact (host and device; N and D take identical operations). Raw N/D ∈ [0, 1] over 20 random cases ×
  3 chunks (each step is monotone and mask ≤ 1, so N ≤ D holds exactly, not just within tolerance).
- R21: `grep -rnE 'hipMalloc|getenv|hipDeviceSynchronize|hipStreamSynchronize' kernels/` empty. The
  REQUIREMENTS command was `grep -nE … kernels/` (no `-r`: exits 2 "Is a directory"); fixed to `-rnE` in
  REQUIREMENTS.md (commit 03702f0).
- R27 grep over kernels/, tests/kernel_test.cpp, tests/rho_ref.py: empty. R28 grep `\b(48|24|2560|10240|128|2048)\b`
  over kernels/*: empty (no model number in the kernels; all extents come from operands/params).

Negative controls — 8 mutants, each built and run, all caught (scratch copies, nothing committed):
```
M1 host rowsel ties by HIGHER row        -> FAIL rowsel_class_semantics, rowsel_matches_fnlev_rules
M2 host k rounds half up (not to even)   -> FAIL rowsel_class_semantics, rowsel_matches_fnlev_rules
M3 host apply adds even at scale 0       -> FAIL state_correct_semantics (the -0.0 / alpha 0 check)
M4 host rho ignores the mask             -> FAIL rho_semantics, rho_matches_numpy_reference
M5 host state ignores row padding        -> FAIL state_correct_semantics
D1 device state ignores row padding      -> FAIL state_correct_device_matches_host (4 of 5 steps; alpha 0 adds nothing)
D2 device rowsel drops the -1 fill       -> FAIL rowsel_device_matches_host, rowsel_matches_fnlev_rules
D3 device rowsel ties by HIGHER row      -> FAIL rowsel_device_matches_host, rowsel_matches_fnlev_rules
```

Code objects (gfx1201, read with llvm-readelf --notes, final build): rowsel LDS 33,792 B, kernarg 80 B; rho
kernarg 120 B (368 B before `blockDim` was replaced by the launch constant: the implicit args went away);
state_correct kernarg 192 B (struct only, no implicit args); every kernel: private
segment 0, `uses_dynamic_stack: false`, 0 VGPR/SGPR spills, no hostcall buffer; fat binary uncompressed
(`__CLANG_OFFLOAD_BUNDLE__` magic, 3 bundles). That is the engine's launch contract (docs/PLUGIN.md §5).

### Tool checks (radiance-plugin skill §1)
- `rad-info --plugins`: host-only build `kva 0.1.0 kernels 3 kernel(s), 3 schema(s), built for host`; HIP build on
  the card `6 kernel(s), 3 schema(s), built for gfx1201 host`, AQL backend sees gfx1201, kva not left out.
- `rad-schemas kva.so`: the three schemas as in §1.
- `opd_shape`: every row has one; `descriptions_cover_schemas` checks one description per schema operand and a
  refusal one past the end; `described_operands_launch` launches every row on its own description (RAD_OK).
- `rad-kbench --op kva_` (replay): loads kva, `0 of 3 kva rows reached` — the shipped fixtures were recorded from
  other models and contain no kva op, so there is nothing to replay. DOES NOT APPLY as an oracle check even with a
  container (`-m`): the oracle must be a separate plugin — rad-kbench skips every row of the `--ref` plugin
  (`tools/rad_kbench.cpp:1990`), so `--ref kva` would skip kva's device rows too, and libref has no kva rows. The
  host-vs-device oracle check lives in `tests/kernel_test` instead (R20/R33/R34). Cost of making kbench usable:
  move the host rows into a second `kvaref.so` that declares `rad_plugin_reference` — not done (the plan fixes one
  kva.so, and a reference library is refused for serving, so the host rows would no longer be a fallback).
- Scratch hook: none needed (rowsel's keys are LDS; the other two use nothing). `describe`: none (each op is one
  launch per layer per prefill chunk; prefill passes are rarely recorded).
- What is NOT yet exercised: the device rows launched through the ENGINE's own queues (the tests use the HIP
  runtime). The static code-object check above is the evidence until the ARCH lane's Stage 4/5 runs issue them.

### Decisions and their cost
- `cap` carries `RAD_PROLE_CAPACITY`: it is derived from max_tok at declare, and a sizing declare at a smaller
  max_tok would otherwise be refused (radiance `core/build/rad_builder.cpp:751-790`). Cost: the kernel writes
  `min(cap, rows_idx extent)` entries, so "keep the cap best" is bounded by the buffer as well.
- `alpha` is REQUIRED in both modes (PLAN §5 lists it plain); undo ignores it. Cost: the undo ops carry a dead
  parameter. Making it optional would let an apply op resolve without one.
- Nothing is added when the scale is 0 (both modes). Cost: one compare; buys R22's byte identity at alpha 0
  (x + 0 would turn a −0.0 state element into +0.0).
- Undo is `state − s·C`, not a restore of a saved copy (D7; accepted by the orchestrator 2026-10-04). It is exact only up to one rounding per element per
  apply/undo pair (≤ 1 ulp of the state, derived, not measured), accumulating at most once per approximate chunk.
  tcc restored a saved copy; matching it bit for bit would need a second LINEAR group the size of the GDN state
  (24 heads × 128×128 f32 = 1.5 MiB per late GDN layer per sequence per rank, ×18 layers = 27 MiB/seq/rank) and a
  copy op. Not done; negligible next to the approximation it corrects.
- `kva_rho_update` takes exactly one sequence (state_idx rows == 1, else RAD_E_SHAPE): the op is issued only on
  single-sequence approximate steps (PLAN §3). A multi-sequence form would need cu_seqlens.
- Random mode keys on the ABSOLUTE position (orchestrator decision, 2026-10-04): the first version hashed the
  chunk-relative row, so one seed kept the same in-chunk offsets in every chunk — periodic in the chunk, rejected
  as the control. Fix: a `positions` operand right after `token_ids` (schema change, operand order above) and
  `kva_row_hash(seed, positions[t])`. Test `rowsel_random_keys_on_position`: two chunks of the same tokens at
  positions 0.. and 2048.. keep exactly k = 256 rows each at different offsets; same seed + positions → same
  rows; [3, n] positions read like [n]; class mode unaffected. Mutant M6 (hash the row again) fails it.
- The truncation count (k − cap) is not an output (not in the schema). With `cap = ceil(share × max_tok / 64) × 64`
  and n ≤ max_tok, k = rint(share × matches) ≤ cap always, so it can only happen when `kva.rowsel.cap` is set
  below that; the debug dump HANDOVER 5.1 mentions has nothing to report at the default.
- Non-finite handling is defined, not left to chance: a non-finite score (−inf, +inf, NaN) or an id outside the
  table is not a match; rho is 1 while D ≤ 0 and a NaN ratio clamps to 0, identically on both sides.
- `-ffp-contract=off` is added for the plugin's HOST sources (rad_add_plugin sets it for device code only), so the
  oracle rounds `x + s·c` twice like the device does; R20's 0-byte result depends on it.
- `tests/kernel_test.cpp` is ~1000 lines, over the ~400 guideline: one binary with two ctest entries (host / gpu)
  as the brief asks; it splits cleanly into a host file and a gpu file if a reviewer wants it.

### `positions` operand for kva_rowsel (orchestrator decision, 2026-10-04)
Host-only build: host group 462 checks, all cases ok (new `rowsel_random_keys_on_position`; quick9 9/9 unchanged:
class mode does not read positions). Mutant M6 (hash the chunk row instead of the position) → FAIL
rowsel_random_keys_on_position. HIP build: compiles, `ctest -LE gpu` passes. **Device group NOT re-run for this
change**: the GPU lock was held by the orchestrator's reservation (`flock … sleep 86400`, 17:40) when I got
there; I did not wait on it or break it. The device row's change is one line (the hash key read through
`positions`, same as the host row); `rowsel_device_matches_host` now passes positions at 4096.. and every other
case as [3, n], so the next gpu run checks it. Run: `ctest --test-dir build-kernels-hip -L gpu` on the card.

### Per-group slot indices for kva_state_correct (orchestrator decision, 2026-10-04)
Operands are now `state, state_idx, applied, applied_idx, C, ND?, nd_idx?` (each KV operand followed by its own
group's index, as gdn_recurrent_update pairs `state` with `state_idx`). Host-only build: host group 499 checks, all
cases ok, incl. new `state_correct_separate_slots` (state slot 3 / applied slot 1 / ND slot 5 in pools of depth
6 / 4 / 7, a second sequence whose applied slot is outside its pool is skipped; exact hand-computed values; undo
restores the state bytes) and new refusals (ND without nd_idx → RAD_E_INVAL; index row counts differ →
RAD_E_SHAPE). Mutants M7 (applied through the state index) and M8 (ND through the state index) → FAIL
state_correct_separate_slots. HIP build compiles, `ctest -LE gpu` passes; state_correct kernarg 240 B, no
dynamic stack. `state_correct_device_matches_host` now runs every step twice — shared slots, then own slots
(state 3 0 5 2, applied 1 3 0 + one outside its 4-slot pool, ND 5 6 2 0 in a 7-slot pool) — comparing state,
applied and ND bytes. **Device group NOT run** (GPU lock held for engine runs). Command for the orchestrator:

    cd ~/projects/inference/radiance-kva && docker run --rm --security-opt label=disable \
      --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined \
      -e ROCR_VISIBLE_DEVICES=<GPU-agent index of the card> -v $PWD:/kva -w /kva radiance-build \
      ctest --test-dir build-kernels-hip -L gpu --output-on-failure

(index = position of the card's BDFID among rocminfo's GPU agents; 0000:13:00.0 = BDFID 4864 was index 1 today.)
Bug caught on the way: an edit dropped kva_rho_parse's `state_idx` assignment; `described_operands_launch`
segfaulted on it before anything was committed.

### kva_state_read (orchestrator request for Stage 6, 2026-10-04)
Schema exactly as requested (params M, n_head, sd0, sd1; operands state, state_idx, out). Host row + device row
(state_correct.hip, one workgroup per (head, sequence), kernarg 104 B, no LDS, no dynamic stack). Never a stub:
implemented directly; the generic cases (descriptions_cover_schemas, stub_refuses, described_operands_launch) now
iterate it too. Host group 566 checks, all ok: `state_read_semantics` (padded strides, slots 2 / −1 / 9-past-a-
4-slot-pool / 2 again → exact copies and zero rows, state untouched) and refusals (out too small → SHAPE, bf16 out
→ DTYPE, missing index → INVAL, n_head mismatch → SHAPE). Mutants M9 (row padding ignored) and M10 (no pool bound
→ reads past the pool) → FAIL state_read_semantics. `state_read_device_matches_host` (24 heads × 128×128, padded
strides, slots 4 / −1 / 1 / 7-past-pool / 4) compares the whole output bitwise and checks the out-of-pool rows are
zero — **not run: the GATES worker holds the GPU lock**. HIP build compiles, `ctest -LE gpu` passes. Device command
as in the previous entry (`ctest --test-dir build-kernels-hip -L gpu --output-on-failure` in radiance-build with
all of /dev/dri and ROCR_VISIBLE_DEVICES set to the card's GPU-agent index).

## Stage A — the device mask (PLAN-FIX v2, 2026-10-04, late evening)

Plan: `~/AI-Work/radiance-kva-plugin-20261004/fix-246/` (PLAN-FIX v2, REQUIREMENTS-FIX R46/R98). Binding schemas:
`notes/impl.md` §1 (IMPL lane). Radiance `140987f`, ROCm 7.2.4 (radiance-build image), card PCI 0000:13:00.0.
Files now: `kernels/` (CMakeLists.txt, kva.h, rows.cpp, host_ref.cpp, mask.hip [was rowsel.hip], select.hip [new],
rho.hip, state_correct.hip), `tests/kernel_test.cpp`.

Commits: c377613 (rho / state_correct `bounds`), 9f87144 (`kva_mask` replaces `kva_rowsel`), 790a711 (`kva_select`,
`kva_drop_rows`, R98 case), 007c967 (test harness: zero-row operands present and empty).

### A.1 Schemas — exactly impl.md §1, no deviation

`rad-schemas kva.so` (host-only build):
```
kva -- 6 ops
  kva_drop_rows      params M:int top_k:int                          operands mask:in ids:inout
  kva_mask           params M:int share:f64 seed:int mode:str        operands cu_last:in token_ids:in positions:in score:weight? mask:out bounds:out
  kva_rho_update     params M:int n_head:int                         operands a:in mask:in A_log:weight dt_bias:weight ND:inout state_idx:in bounds:in?
  kva_select         params M:int                                    operands mask:in x_src:in q_src:in? s_src:in? x:inout q:inout? s:inout?
  kva_state_correct  params M:int mode:str alpha:f64 n_head:int sd0:int sd1:int
                     operands state:inout state_idx:in applied:inout applied_idx:in C:weight ND:in? nd_idx:in? bounds:in?
  kva_state_read     params M:int n_head:int sd0:int sd1:int         operands state:in state_idx:in out:out
```
(columns re-wrapped; names, order, roles and optionality as printed.) Every param is REQUIRED and has role
RAD_PROLE_NONE. `kva_rowsel` is deleted entirely (schema, rows, host/device code, tests); its `cap` (the only
CAPACITY param) went with it.

### A.2 Decisions and their cost (what the arch side can trip)

- **kva_mask, cu_last out of order.** The host row refuses s < 0, s > e, e > n with RAD_E_INVAL. The device row
  reads cu_last in device memory and cannot return a code without a synchronize (R21), so it does the safe thing:
  clamps e into [0, n], s into [0, e], and takes the window EMPTY -- every mask row 0 (exact, the plain model),
  bounds {s, s, s, e} clamped. Test `mask_device_fail_safe`. Cost: a wrong cu_last on the device is silent (all
  rows exact, so slower, never wrong numbers).
- **kva_mask device row: b <= 8192** (`KVA_MASK_MAX_ROWS`, renamed from KVA_ROWSEL_MAX_ROWS, same value; nothing
  outside kernels/ used it). The row's constraint is `M <= 8192` (band selection, M issued at b) and the launch
  also refuses token_ids' extent b > 8192 with RAD_E_SHAPE (the window, <= b rows, keeps one 4-byte key per row in
  LDS: 33,792 B). The mask's n is NOT bounded (tested at n = 9000). The host row has no limit.
- **kva_mask, required operands.** cu_last [>= 2], token_ids, positions (rank 1 or 2, last extent >= b, even in
  modes that never read it -- the schema does not mark it optional), mask, bounds [>= 4] are required; `score` is
  optional and RAD_E_INVAL when absent in class/random; `share` in [0, 1] and `seed` required in every mode.
  b = 0 (token_ids of zero rows) works as long as its data pointer is non-null: a null pointer is "absent" under
  the ABI's rule and is refused RAD_E_INVAL.
- **kva_select.** Exactly `mask[i] == 1` copies (other values copy nothing). Each pair rank 2, last stride 1,
  same dtype and shape[1], >= n rows each; a half-present pair is RAD_E_INVAL. Whole-byte dtypes only: a sub-byte
  dtype (i4, fp4, ...) is refused RAD_E_DTYPE (a row of them need not start on a byte boundary under every
  pitch). The device row copies each pair in the widest word (16, 4 or 1 bytes) its two addresses, two pitches and
  row width are multiples of, picked at parse; a bf16 row moves in 16-byte loads. One workgroup per mask row;
  an exact row costs one 4-byte mask read. A destination may be its own source (host uses memmove).
- **kva_drop_rows.** top_k >= 0 (missing or negative: RAD_E_INVAL); ids rank 2, last stride 1, rows >= n,
  shape[1] >= top_k (else RAD_E_SHAPE). top_k 0 writes nothing.
- **bounds on rho / state_correct.** i32, dense, >= 1 (rho) / >= 2 (state_correct) elements, else
  DTYPE / STRIDE / SHAPE. Read INSIDE the kernel on the device row (never on the host). rho runs over
  [clamp(bounds[0], 0, n), n) with n = `a`'s extent; state_correct writes nothing when bounds[1] <= bounds[0], in
  both modes and for every sequence. Absent = the old op, bit for bit (every pre-Stage-A case unchanged).
- **rho host vs device is not bitwise** (the device's expf/log1pf are not glibc's; max |Δrho| 2.4e-7 here), so
  "rho with bounds, device == host" is checked as: device == device-on-the-slice BITWISE (7/7 bounds incl. -3, 0,
  2048, 4000 on n = 2048) plus device vs host within the R34 tolerance (1e-5). Every other op is bytewise.
- **R98:** `grep -rn RAD_PROLE_SEQ_CHUNK kernels/` -> empty (exit 1); case `params_carry_no_role` asserts every
  param of every schema has role in {NONE, CAPACITY} and in fact NONE, and that the schema count is 6.

### A.3 Tests (R46/R98)

New host cases: `params_carry_no_role`, `mask_class_semantics` (s = 4 decode prefix + 2-row tail, ties, every
non-finite score, id past the table, k half to even, rows outside W carry the best id), `mask_window_clamping`
(b < s, b > e, b == s, s == e == n, b = 0, bulk + tail; none = 1 exactly on W, all = 0, bounds bytes),
`mask_random_semantics` (W = [100, 900) of 1100 rows: kept rows == the k smallest (hash(seed, position), row) by
a transcription of kva_row_hash; seed and positions matter; [3, b] positions == [b]; class ignores positions),
`mask_refuses_bad_window`, `select_copies_masked_rows` (bf16 x, i8 q, f32 s, padded and different pitches, more
destination rows than n, mask values 2 and -1, EVERY destination byte checked against a 0xA5 sentinel, sources
untouched, absent q/s), `drop_rows_semantics`, `select_drop_refuse_bad_operands`, `rho_bounds_semantics`
(== the op on the slice [s, n), carried ND, clamp below 0 / at or past n), `state_correct_bounds` (empty and
inverted bounds change no byte in either mode; non-empty == no bounds). `refuses_bad_operands` re-pointed at
kva_mask (+ bounds refusals of rho / state_correct). R33 re-pointed: `mask_matches_fnlev_rules` (class mode, window
[0, 2048) of each doc == fixture rows, count == k; and the same ids at rows [64, 2112) of a 2112-row step, cu_last
{64, 2112}, rows before 64 the table's best id: kept == rows + 64, rows < 64 all 0). Generic cases iterate all
six ops; the described operands use the engine tools' IDX_CU fill for cu_last ({0, M}).
New gpu cases: `mask_device_matches_host` (192 configurations: n in {1, 7, 300, 2048, 8192, 9000} x 4 window
layouts x 4 modes x share {0.25, 1}, random ids/positions/score with heavy ties, [b] and [3, b] positions; mask
AND bounds bytes equal), `mask_device_fail_safe`, `rho_bounds_device_matches_host`,
`state_correct_bounds_device_matches_host`, `select_device_matches_host` (9 configurations, 7.58 MB of destination
bytes, n up to 8192, bf16 / i8 / E4M3 / f32, all three copy words, absent pairs), `drop_rows_device_matches_host`
(12 configurations, top_k 0..300).

Host group, host-only build (`ctest -LE gpu`: `kva_kernels_host Passed`, 100% tests passed out of 1):
```
  ok   descriptions_cover_schemas
  SKIP stub_refuses: no stubbed row remains in this domain (R8 retired: every row implemented)
  ok   described_operands_launch
  ok   params_carry_no_role
  ok   mask_class_semantics
  ok   mask_window_clamping
  ok   mask_random_semantics
  ok   mask_refuses_bad_window
  ok   refuses_bad_operands
  ok   state_read_semantics
  ok   state_correct_separate_slots
  ok   state_correct_semantics
  ok   rho_semantics
  ok   rho_in_unit_interval
  ok   select_copies_masked_rows
  ok   drop_rows_semantics
  ok   select_drop_refuse_bad_operands
  ok   rho_bounds_semantics
  ok   state_correct_bounds
  rho vs NumPy: max |rho diff| 1.55e-06, max rel N/D diff 2.15e-06
  ok   rho_matches_numpy_reference
  ppl/8k/0   n 2048  k 119  == fnlev.rules (window [0, 2048) and [64, 2112))
  ppl/8k/1   n 2048  k 101  == fnlev.rules (window [0, 2048) and [64, 2112))
  ppl/8k/2   n 2048  k 78  == fnlev.rules (window [0, 2048) and [64, 2112))
  ppl/16k/0  n 2048  k 127  == fnlev.rules (window [0, 2048) and [64, 2112))
  ppl/16k/1  n 2048  k 144  == fnlev.rules (window [0, 2048) and [64, 2112))
  ppl/16k/2  n 2048  k 140  == fnlev.rules (window [0, 2048) and [64, 2112))
  ppl/16k/3  n 2048  k 124  == fnlev.rules (window [0, 2048) and [64, 2112))
  ppl/32k/0  n 2048  k 94  == fnlev.rules (window [0, 2048) and [64, 2112))
  ppl/32k/1  n 2048  k 101  == fnlev.rules (window [0, 2048) and [64, 2112))
  ok   mask_matches_fnlev_rules
767 check(s), group host
```
HIP build in radiance-build: compiles, `ctest -LE gpu` 100% passed (1/1). Device group (kernel log clean before
and after every GPU run; card mapped from BDFID 4864 inside the container):
```
card 0000:13:00.0 (BDFID 4864) -> ROCR_VISIBLE_DEVICES=1
  device 0 of 1 visible, PCI 0000:13:00.0
  ok   descriptions_cover_schemas
  SKIP stub_refuses: no stubbed row remains in this domain (R8 retired: every row implemented)
  ok   described_operands_launch
  ok   params_carry_no_role
  ok   refuses_bad_operands
  ok   select_drop_refuse_bad_operands
  rho vs NumPy: max |rho diff| 1.55e-06, max rel N/D diff 2.15e-06
  ok   rho_matches_numpy_reference
  ppl/8k/0   n 2048  k 119  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ppl/8k/1   n 2048  k 101  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ppl/8k/2   n 2048  k 78  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ppl/16k/0  n 2048  k 127  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ppl/16k/1  n 2048  k 144  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ppl/16k/2  n 2048  k 140  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ppl/16k/3  n 2048  k 124  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ppl/32k/0  n 2048  k 94  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ppl/32k/1  n 2048  k 101  == fnlev.rules (window [0, 2048) and [64, 2112)) == host row
  ok   mask_matches_fnlev_rules
  shared slots undo  alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  shared slots apply alpha 0.7 ND 1: 10031712 state bytes, 0 differ (max abs diff 0)
  shared slots undo  alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  shared slots apply alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  shared slots apply alpha 0.0 ND 1: 10031712 state bytes, 0 differ (max abs diff 0)
  own slots   undo  alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  own slots   apply alpha 0.7 ND 1: 10031712 state bytes, 0 differ (max abs diff 0)
  own slots   undo  alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  own slots   apply alpha 1.0 ND 0: 10031712 state bytes, 0 differ (max abs diff 0)
  own slots   apply alpha 0.0 ND 1: 10031712 state bytes, 0 differ (max abs diff 0)
  ok   state_correct_device_matches_host
  7864320 out bytes, 0 differ; out-of-pool rows all zero: yes
  ok   state_read_device_matches_host
  192 configurations (17 layouts with a window, 10 with s > 0): device mask and bounds == host
  ok   mask_device_matches_host
  ok   mask_device_fail_safe
  device vs host: max |rho diff| 2.68e-07, max rel N/D diff 7.67e-07
  ok   rho_device_matches_host
  7 bounds: device == device-on-slice bitwise 7/7; device vs host max |rho diff| 2.38e-07
  ok   rho_bounds_device_matches_host
  8 runs (undo/apply x absent/empty/inverted/non-empty bounds): device == host bytewise
  ok   state_correct_bounds_device_matches_host
  9 configurations, 7578006 destination bytes: device == host bytewise
  ok   select_device_matches_host
  12 configurations: device ids == host bytewise
  ok   drop_rows_device_matches_host
944 check(s), group gpu
exit 0
$ ctest --test-dir build-kernels-hip -L gpu      (final run, HEAD 007c967)
1/1 Test #2: kva_kernels_gpu ..................   Passed    1.81 sec
100% tests passed, 0 tests failed out of 1
```
The first device run (before 007c967) aborted in `rho_bounds_device_matches_host` with std::length_error: the test
harness sized a zero-row tensor with explicit strides to a negative byte count (the case slices to zero rows when
bounds start at or past n). Every case before it had passed; fixed in the harness, not the op, and re-run as above.

### A.4 Negative controls — 11 mutants, each built and run in a scratch copy (nothing committed), all caught
```
H1 host mask ignores s (window from row 0)          -> FAIL mask_class_semantics, mask_window_clamping,
                                                         mask_random_semantics, mask_matches_fnlev_rules
H2 host select also copies a mask==0 row            -> FAIL select_copies_masked_rows
H3 host drop_rows writes top_k+1 columns            -> FAIL drop_rows_semantics
H4 host rho ignores bounds                          -> FAIL rho_bounds_semantics
H5 host state_correct ignores empty bounds          -> FAIL state_correct_bounds
H6 host mask k rounds half up                       -> FAIL mask_class_semantics, mask_matches_fnlev_rules
D1 device mask ties by higher row                   -> FAIL mask_device_matches_host, mask_matches_fnlev_rules
D2 device select also copies a mask==0 row          -> FAIL select_device_matches_host
D3 device drop_rows writes top_k+1 columns          -> FAIL drop_rows_device_matches_host
D4 device rho ignores bounds                        -> FAIL rho_bounds_device_matches_host
D5 device state_correct ignores empty bounds        -> FAIL state_correct_bounds_device_matches_host
```

### A.5 Static checks
- R21 `grep -rnE 'hipMalloc|getenv|hipDeviceSynchronize|hipStreamSynchronize' kernels/` -> empty.
- R27 portability grep over kernels/ and tests/kernel_test.cpp -> empty. R28 `\b(48|24|2560|10240|128|2048)\b` over
  kernels/* -> empty.
- R98 `grep -rn RAD_PROLE_SEQ_CHUNK kernels/` -> empty.
- Code objects (gfx1201, `.hip_fatbin` split into its 4 bundles, `llvm-readelf --notes`): kva_mask_kernel LDS
  33,792 B, kernarg 104 B; kva_select_kernel kernarg 160 B; kva_drop_kernel 40 B; kva_rho_kernel 128 B (+8 for
  bounds); kva_correct_kernel 248 B (+8); kva_state_read_kernel 104 B. Every kernel: private segment 0,
  `uses_dynamic_stack: false`, 0 SGPR/VGPR spills, no hostcall buffer, no hidden (implicit) arguments.
- `rad-info --plugins`: host-only build `kva 0.1.0 kernels 6 kernel(s), 6 schema(s), built for host`; HIP build
  (in the image, no card) `kva 0.1.0 kernels 12 kernel(s), 6 schema(s), built for gfx1201 host`.
- NOT exercised: the device rows through the ENGINE's queues (the tests use the HIP runtime); the arch side's
  Stage A runs are the first engine launches of kva_mask / kva_select / kva_drop_rows.
