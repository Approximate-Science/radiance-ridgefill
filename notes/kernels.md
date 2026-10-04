# KERNELS lane notes — `kva.so` (kva_rowsel, kva_rho_update, kva_state_correct)

Owner: KERNELS lane. Files: `kernels/` (CMakeLists.txt, kva.h, rows.cpp, host_ref.cpp, rowsel.hip,
rho.hip, state_correct.hip), `tests/kernel_test.cpp`, `tests/rho_ref.py`, `tests/fixtures/rho_numpy.txt`,
this file. Radiance `140987f` (v1.0.8).

## 1. The three ops — schemas and operand order (for the ARCH lane)

Operands are POSITIONAL, in this order. `?` = optional (pass `RAD_NONE`). Params in `RAD_PARAMS` order
do not matter to the engine, but every listed param is REQUIRED (except where noted).

| op | params | operands (in order) |
|---|---|---|
| `kva_rowsel` | `M` (RAD_RANGE, rows of the chunk) · `cap` (int, CAPACITY role) · `share` (f64, 0..1) · `seed` (int) · `mode` (str: `class`/`random`/`all`) | 0 `token_ids` i32 [n] (praw of `batch->token_ids`, n = n_tok) · 1 `score` **weight** f32 [vocab] (`kva.rowsel.score`) · 2 `rows_idx` out i32 [cap] · 3 `mask` out i32 [n] |
| `kva_rho_update` | `M` (RAD_RANGE, rows) · `n_head` (int, this rank's GDN value heads) | 0 `a` bf16 or f32 [n, n_head], any row pitch (e.g. `bcol(w.ab, 0, H, n)`) · 1 `mask` i32 [n] (kva_rowsel's mask) · 2 `A_log` **weight** f32 [n_head] · 3 `dt_bias` **weight** f32 [n_head] · 4 `ND` inout f32, the LINEAR group `kv_kva_rho` layer cache `[n_states, n_head, 1, 2]` (or any trailing dims whose product is 2) · 5 `state_idx` i32 [1, pitch] (`praw2(st_idx, RAD_I32, 1, st_w)`; exactly ONE sequence) |
| `kva_state_correct` | `M` (RAD_RANGE, n_seq) · `mode` (str: `undo`/`apply`) · `alpha` (f64; read in apply, ignored in undo, still required) · `n_head` · `sd0` · `sd1` (ints: GDN state per head is sd0×sd1) | 0 `state` inout f32 `RAD_KV(kv_gdn_state, L)` [n_states, n_head, sd0, sd1] (strides read off the operand) · 1 `applied` inout f32 `RAD_KV(kv_kva_applied, L)` [n_states, n_head, 1, 1] · 2 `state_idx` i32 [n_seq, pitch] (column 0 used) · 3 `C` **weight** f32 [n_head, sd0, sd1] (`kva.st.L`) · 4 `ND`? f32 (as above; absent in speed mode → rho = 1) |

Semantics (also in each schema's doc string, `rad-schemas kva.so`):

- `kva_rowsel`: a row MATCHES when `score[token_ids[i]]` is finite (an id outside `[0, vocab)` does not
  match). `k = rint(share × matches)` — round half to EVEN, exactly Python's `round()` in `fnlev/rules.py`.
  `class`: row kept iff it matches and fewer than k matching rows precede it by (score desc, row asc).
  `random`: same k, over ALL n rows, ranked by `kva_row_hash(seed, row)` ascending (ties by row).
  `all`: k = n, ranked by row. At most `min(k, cap, rows_idx extent)` rows are kept — the best ones
  (the truncation case). `rows_idx` = kept rows ascending, `-1` padded to its extent; `mask[i] = 1` for
  rows NOT kept (approximated), 0 for kept. Rows are CHUNK-relative (0..n-1).
- `kva_rho_update`: per head, sequentially over the n rows: `g = -exp(A_log)·softplus(a + dt_bias)`
  (softplus threshold 20), `D = e^g·D + 1`, `N = e^g·N + mask[t]`, carried in the ND slot across chunks
  (zeroed at admission by the engine). The slot is `state_idx[0]`; a slot outside the pool writes nothing.
- `kva_state_correct`: per (sequence, head), slot = `state_idx[s][0]` (outside the pool: skipped).
  `undo`: `state -= applied·C; applied = 0`. `apply`: `s = alpha × (ND ? kva_rho(N, D) : 1)` with
  `kva_rho = D > 0 ? clamp(N/D, 0, 1) : 1`; `state += s·C; applied = s`. **When the scale is 0 nothing is
  added** (so `alpha = 0` leaves every state bit alone, incl. the sign of a −0.0 — R22).

Issue order the plan wants per late GDN layer L (HANDOVER Stage 4.2 / 5.3): `kva_state_correct(undo)`
before the layer's `gdn_conv_prep`; after `gdn_chunk_scan`: `kva_rho_update` (quality mode only), then
`kva_state_correct(apply)` with ND present (quality) or `RAD_NONE` (speed). `mode` is a fixed parameter,
so undo and apply are two declared ops per layer.

## 2. Fixture format for R33 (SIDECAR lane writes `tests/fixtures/rowsel_quick9.json`)

The test reads the path from env `KVA_ROWSEL_FIXTURE` (CTest sets it to `tests/fixtures/rowsel_quick9.json`)
and SKIPs with a reason when the file is absent. Exact format (plain JSON, numbers only, no NaN/Infinity):

```json
{
  "format": "kva-rowsel-quick9/1",
  "source": "free text: rules.py commit, freq table sha256, tokenizer, date",
  "cases": [
    {
      "name": "<doc id>/chunk0",
      "share": 0.25,
      "cap": 512,
      "token_ids": [ 1234, 56, ... ],
      "scores":    [ 11.53125, null, ... ],
      "expected_rows": [ 3, 17, ... ]
    }
  ]
}
```

- `token_ids`: the chunk's n ids (the doc's first 2048 tokens for R33; any n ≤ 8192 works).
- `scores`: one entry per row: the f32 value `kva.rowsel.score[token_ids[i]]` from the sidecar table
  (write `float(np.float32(x))` so it round-trips exactly), or `null` where the score is −inf (class not
  kept). The test rebuilds a score table from these (id → score) and checks the same id never has two scores.
- `expected_rows`: `fnlev.rules.Rules().rows({"rule":"class56","share":0.25}, ids, P=n, name)` on exactly
  this window — ascending, chunk-relative. The kernel runs `mode=class`, `seed=0`, the case's `share`/`cap`.
- The test checks host row == `expected_rows` (host group) and device row == host row == `expected_rows`
  (gpu group), plus `mask` consistency and `-1` padding.

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
