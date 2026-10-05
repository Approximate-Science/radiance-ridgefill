# notes/impl.md — IMPL lane: PLAN-FIX v2 Stage A (device mask, row-exact tail, stager lever, guard)

Owner: IMPL lane (2026-10-04, late evening). Plan: `~/AI-Work/radiance-kva-plugin-20261004/fix-246/`
(PLAN-FIX v2, REQUIREMENTS-FIX R45–R99, HANDOVER-FIX v2). radiance `140987f` (v1.0.8), read only.
Decisions in force (orchestrator, plan defaults): DD-F device mask replaces compaction (deleted, not
flagged); DD-B stager lever behind `RADIANCE_KVA_STAGE`; DD-C both straddle variants
(`RADIANCE_KVA_STRADDLE=split|end`); DD-E forward on a version mismatch. DD-A (hazard counter,
`kva.ckpt_floor`) is Stage C and DD-D (`final`) Stage D: neither is built here.

## 1. kva.so schemas for Stage A (the arch side issues exactly these)

Operands are POSITIONAL. `?` = optional (`RAD_NONE`). Every param is required. No param carries
`RAD_PROLE_SEQ_CHUNK` (R98) and, with `cap` gone, none carries `RAD_PROLE_CAPACITY` either.

**A kernel learns a per-step host number only through an operand's EXTENT.** The engine hands a
kernel the ranged parameter at its BAND's upper bound, not the issued value (radiance
`abi/rad_abi.h:54` "ranges collapsed", `core/runtime/issue.cpp:811-814`); an operand's row count is
per issue and is what a recorded pass replays. So `kva_mask` reads the bulk end `b` as the extent of
its `token_ids` operand, and `M` is issued at `b` only to select the band.

| op | params | operands (in order) |
|---|---|---|
| `kva_mask` (replaces `kva_rowsel`) | `M` (range, issued at b) · `share` f64 · `seed` int · `mode` str `none`/`class`/`random`/`all` | 0 `cu_last` in i32 [2] = `praw(cu_seqlens + n_seq−1, I32, 2)` → {s, e} · 1 `token_ids` in i32 [b] (rows [0, b) of the step; **its extent is b**) · 2 `positions` in i32 [b] or component-major [c, b] (row 0 at its strides; random mode only reads it) · 3 `score` **weight** f32 [vocab], optional (class/random need it) · 4 `mask` out i32 [n] (n = its extent = n_tok; every row written) · 5 `bounds` out i32 [4] |
| `kva_select` | `M` (range, rows) | 0 `mask` in i32 [n] (n = its extent) · 1 `x_src` in [n, w0] · 2 `q_src`? in [n, w1] · 3 `s_src`? in [n, w2] · 4 `x` inout [n, w0] · 5 `q`? inout [n, w1] · 6 `s`? inout [n, w2] |
| `kva_drop_rows` | `M` (range, rows) · `top_k` int | 0 `mask` in i32 [n] · 1 `ids` inout i32 [n, ≥top_k] |
| `kva_rho_update` | unchanged: `M` · `n_head` | unchanged 0–5 (`a` [n, n_head] at its strides, `mask`, `A_log`w, `dt_bias`w, `ND` inout, `state_idx` [1, pitch]) · **6 `bounds`? in i32 [≥1]** |
| `kva_state_correct` | unchanged: `M` · `mode` · `alpha` · `n_head` · `sd0` · `sd1` | unchanged 0–6 (`state`, `state_idx`, `applied`, `applied_idx`, `C`w, `ND`?, `nd_idx`?) · **7 `bounds`? in i32 [≥2]** |
| `kva_state_read` | unchanged | unchanged |

Semantics:

- `kva_mask`: s = cu_last[0], e = cu_last[1]; b = extent of `token_ids`; b′ = min(max(b, s), e).
  The window is W = [s, b′). `mask[i] = 0` for every row outside W. Inside W: `none` → 1 (every row
  approximated: speed mode); `all` → 0 (every row exact: plumb, and quality's all-rows oracle);
  `class` → a row MATCHES when `score[token_ids[i]]` is finite (an id outside [0, vocab) does not);
  k = rint(share × matches in W), half to even (Python's `round`, fnlev/rules.py); a row is kept
  (mask 0) iff it matches and fewer than k matching rows of W rank before it by (score descending,
  row ascending); every other row of W is 1. `random` → the same k, over ALL rows of W, ranked by
  `kva_row_hash(seed, positions[i])` ascending, ties by row; kept rows 0, others 1. This is
  `kva_rowsel`'s rule run over W instead of the whole chunk, with no cap and no `rows_idx`.
  `bounds = {s, b′, b′, e}` (the GDN split's two `cu` pairs, PLAN-FIX §4). Refused: s < 0, s > e,
  e > n.
- `kva_select`: for every row i < n with `mask[i] == 1`, row i of each present destination is
  overwritten with row i of its source, byte for byte; nothing else is written. Each (src, dst) pair
  is present or absent together, has one dtype and one row width (shape[1]); row pitches come off
  each tensor (last stride 1); both have ≥ n rows. Dtype-agnostic (bf16 x, int8 or E4M3 codes, f32
  scales).
- `kva_drop_rows`: for every row i < n with `mask[i] == 1`, `ids[i, 0..top_k) = −1`; nothing else is
  written (columns ≥ top_k and unmasked rows untouched). ids rows ≥ n, row pitch off the tensor.
- `kva_rho_update` + `bounds`: the recurrence runs over rows [lo, n) with lo = clamp(bounds[0], 0,
  n) and n = `a`'s extent; absent = [0, n), i.e. today. (The arch side issues `a` at b rows for the
  split variant and at n_tok rows for the end-of-chunk variant, with bounds = kva_mask's.)
- `kva_state_correct` + `bounds`: when present and `bounds[1] <= bounds[0]` the op writes nothing in
  either mode (the step's last sequence had no bulk row, so its correction stays as the previous
  chunk end left it); otherwise exactly today's op. **Deviation from PLAN-FIX §7**, which gave
  bounds to the rho update only: without it a sequence that never had a bulk row (possible only with
  two prefill chunks in one step, where the host cannot see s) would get alpha·C added by `apply`
  (rho is 1 while D is 0).
