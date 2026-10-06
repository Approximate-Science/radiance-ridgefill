# notes/investigate.md — why radiance's RidgeFill numbers looked worse than tcc's (INVESTIGATE lane, 2026-10-04)

Question (Dylan): "How is it possible that it is this much worse? It's not strictly due to different measurements.
Something else is happening." Governing metric: per-doc mean NLL delta vs the same engine's exact run, on the SAME
tokens and the SAME scoring window as tcc (last 512 tokens). Labbook: `~/AI-Work/radiance-kva-plugin-20261004/labbook`,
hypotheses `INV-*` (seq 89 onward). Scripts: `tools/investigate/`. Raw outputs: `data/investigate/` (git-ignored).
Provenance: radiance 140987f (v1.0.8), container `qwen3.8-next-flash-fp8-iq4r-moe.rad` (post append #2), boot
`75e3e39b-…`, plugin homes `data/home-f60f893` (KL runs) and `data/refit/home-3af4eaa` (state captures, sha256
c408596e… / 34062e78…). tcc files: `~/AI-Work/kva-flashnext-iterate/tests/results/`.

## 1. Answer

Nothing else is happening that the data can see. On identical tokens and window radiance matches tcc within tcc's own
run-to-run noise on every arm; the "much worse" headline was the scoring window (2,047 tail positions vs tcc's last 512).
The residual the orchestrator found (+st removes −0.015 on radiance vs −0.026 on tcc) came from pairing tcc's big-nost
with a big-st from ANOTHER boot; tcc's same-server pair gives −0.019, and tcc's identical big-st arm moved −0.0073
between its two boots. The correction itself is applied correctly on the engine (state-level check, with a negative
control), its orientation and head order are right, and the shipped C fits radiance's own state error as well as or
better than a radiance fit. A powered same-docs re-run on tcc's 13 final-tier 16k/32k docs (§6) puts radiance's
fill and fill+st slightly BELOW tcc's and the +st effect at −0.0163 vs tcc −0.0196 (paired +0.0032 [−0.004, +0.010]).
Over all 19 matched docs, any real +st shortfall is below 0.009 nats (95% upper bound); the point estimate is 0.003.

## 2. The window conclusion, re-verified (independent of the orchestrator)

- `.rows` columns: per-doc `exp(mean col1)` / `exp(mean col2)` reproduce the report's `by_source` candidate / reference
  ppl for all 14 arms (0 mismatches); col 2 is byte-identical across arms (the stored reference). Doc order = the
  reference's `kld.json` docs (row0 0, 2047, …); row i predicts token `score_from + i + 1`, so the last 512 rows are
  exactly tcc's 512 scored tokens.
- Tokens: our last-512 ids equal tcc's stored `ids` for 16k/0–3 and 32k/0–1; the 8k docs differ (ours cut to 8,192,
  tcc's 9,216, P 6,144 vs 7,168), so only 6 docs are matched.
- tcc stores exactly 512 logprobs per doc in every ppl.jsonl on disk (31 files under tests/results, plus the research
  repo copies) — tcc's early-tail damage cannot be compared from stored data.
- Same tokens, last 512 (6 docs): fill radiance +0.0426 vs tcc +0.0414 (big-nost, iter-02).

## 3. The +st residual is tcc noise (INV-tcc-noise, post-hoc — numbers seen before registering)

| +st effect (st − fill), 6 matched docs, last 512 | mean | per doc |
|---|---|---|
| radiance (deterministic, same boot) | **−0.0154** | −0.025 −0.014 −0.002 −0.012 −0.024 −0.015 |
| tcc same server (iter-02 big-nost → big-st) | −0.0189 | −0.009 −0.021 −0.028 −0.001 −0.024 −0.031 |
| tcc cross boot (iter-02 big-nost → iter-04 big-st) = the "−0.026" | −0.0262 | |
| tcc same arm twice (big-st iter-04 − iter-02) | **−0.0073** (sd 0.0105/doc) | |

Paired radiance − tcc: same server +0.0035 [−0.008, +0.016]; cross boot +0.0108 [−0.0016, +0.023]. Other tcc readings:
9-doc quick tier same server −0.0124 (radiance 9 docs, last 512: −0.0143); final tier 19 docs −0.0151 [−0.021, −0.009];
history −0.012 … −0.029 (research HANDBACK.md:22). Quality: radiance's exact rows buy MORE over +st than tcc's on the same
boot (−0.0234 vs −0.0180); the quality gap (+0.0038 vs −0.0028) is inherited from the low iter-04 big-st anchor and sits
inside tcc's stated ±1% cross-boot exact uncertainty.

## 4. Is the correction right on the engine? (INV-orient, INV-cfit, INV-apply, INV-apply-neg — all registered first)

**Orientation / heads** (state_err.py on the REFIT lane's captures, 14 +st prompts, 55 chunk ends, tail 512):
cos(radiance mean error, shipped C) 0.927 (0.88–0.98 per layer) vs transposed V↔K 0.010, heads shifted by one 0.018,
tiled V-head order 0.026. Code agrees: the libr4d scan addresses the state `v*128 + k` ([heads,V,K]) and value head h reads
key head h/3 (HF grouped), = tcc's. Confirmed.

**Does the shipped C fit radiance?** Share of chunk-end state error energy removed (all layers):
shipped 24.7%, leave-one-prompt-out radiance mean 20.2%, in-sample 26.5%; best scalar for shipped alpha* 0.966.
Per layer 5–36% (L44 best, L42 worst). The shipped C (62 ends) generalises better than a 14-prompt radiance fit; R44's
"refit = shipped" NLL agrees. Confirmed. Caveat: radiance S_pred comes from the refit projector (the only capture).

**On-engine apply** (new GPU session `tools/investigate/session.sh`, 4 arms, PROJ=refit, tail 512, capture home):
| arm | layer-24 tail-end state error energy vs alpha 0 | layers improved |
|---|---|---|
| shipped C, alpha 1 | **0.506** | 18 / 18 (mean 0.851) |
| shipped C, alpha 0.5 | 0.636 | 18 / 18 (mean 0.883) |
| halves swapped, alpha 1 (negative control) | 1.0005 | 0 / 18 (mean 1.113) |
Layer 24 is the clean read: its tail inputs come from exact layers 0–23 in every arm. Undo: first chunk end bitwise
equal; later chunk ends differ by 0.8e-4–3.9e-4 of |C| with cos(diff, C) −0.004…+0.007 — a failed undo would be ≈ C·M,
a large fraction of |C| pointing along C. The alpha-0 run is bitwise identical to the REFIT lane's earlier no-correction
capture at all 110 approximate chunk ends (determinism across sessions and container appends). Two registered sub-criteria missed AS WRITTEN and are
reported as such: (a) "max element diff ≤ 1e-3 × rms" read 0.006–0.11 because the state is heavy-tailed
(max|S|/rms 76–358); (b) alpha 0.5 is linear to 1.2%, not 0.1%, because the scan stages the state in bf16 in LDS
(`r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip`, `Smem::S` unsigned short). Neither is a fault.

## 5. Where the loss sits on radiance (tail_profile.py; 9 docs, 512-token quarters of the 2,047 scored rows)

| arm | rows 0–510 | 511–1022 | 1023–1534 | 1535–2046 (tcc's window) |
|---|---|---|---|---|
| fill | +0.097 | +0.057 | +0.054 | +0.038 |
| fill + st | +0.069 | +0.041 | +0.039 | +0.024 |
| quality 25% | +0.045 | +0.020 | +0.022 | +0.007 |
The first quarter carries 2.5× the last quarter's loss; +st's gain is 2× larger there (−0.028 vs −0.014). That is why
the 2,047-row headline (+6.4 / +4.4 / +2.4%) reads so much worse than tcc's last-512 numbers (+4.3 / +2.8 / +0.2%).

## 6. Final-tier check (INV-final13, registered before the run)

Radiance on tcc's 13 final-tier 16k + 32k docs (same tokens, `data/investigate/final13.jsonl`, 131 bulk chunks =
131 approximate steps), new same-boot stock reference `data/investigate/kld-ref-final13`, home-f60f893, speed with
alpha 0 (fill) and 1 (fill + st); tcc's final tier ran exact / ours-nost / ours on ONE server. Last 512 tokens:

| | radiance | tcc (same server) | radiance − tcc, paired [95% CI] |
|---|---|---|---|
| fill | +0.0374 | +0.0455 | −0.0082 [−0.0167, +0.0006] |
| fill + st | +0.0211 | +0.0260 | −0.0049 [−0.0146, +0.0038] |
| +st effect | −0.0163 [−0.020, −0.013] | −0.0196 [−0.026, −0.013] | +0.0032 [−0.0041, +0.0104] |

Pooled with §3's 6 docs (19 matched docs, same-server tcc pairs): +st effect radiance −0.0160 vs tcc −0.0193, paired
+0.0033 [−0.0029, +0.0094]. Radiance's absolute fill and fill+st losses are LOWER than tcc's on these docs; the same runs
scored on all 2,047 rows read +0.0594 / +0.0374 — the window again. The fill sub-criterion (±0.008) was exceeded by
0.0002 in radiance's favour.

## 7. Ruled out, and how

| hypothesis | verdict | evidence |
|---|---|---|
| correction on the wrong layers / heads / rank slice / V-K transposed | ruled out | §4 orientation cosines; swap control; layer-24 tail-end error −49% |
| undo failing, applied at the wrong time | ruled out | §4 chunk-end diffs at rounding level, uncorrelated with C |
| state layout / precision (f32 radiance vs bf16 tcc state) | no effect found | layout = [heads,V,K] both; tcc's state is bf16 (ROOM-ST.md); radiance f32 storage, bf16 in the scan's matmuls |
| C fits radiance worse | ruled out | §4 alpha* 0.97, shipped ≥ LOO refit |
| tail length / chunk ends | not applicable on the matched docs | P = N − 2048 on both engines for 16k/32k |
| something in the tail (QSA on projected indexer keys) | no sign | fill identical on matched docs (+0.0426 vs +0.0414) |
| a late-position-only C would help | refuted (negative result) | late_c.py: at ends ≥ 6,144, LOO fit on ends ≥ 4,096 removes 20.5% vs all ends 20.6% vs shipped 25.4% |

## 8. Files

`tools/investigate/`: rescore.py (alignment helpers), st_gap.py (§3), state_err.py (§4 fit/orientation), late_c.py,
session.sh (§4 GPU captures), apply_check.py (§4), tail_profile.py (§5), kl_session.sh + final13_compare.py (§6).
`data/investigate/`: st_gap.json, state_err_sterm_refitproj.json, apply_check.json, undo_check.json, late_c.json,
tail_profile.json, state-{st1,st0,swap1,sthalf}/ (3.7 GB each, deletable), session logs, final13 corpus/reference/runs.

## 9. Next (if more certainty is wanted)

- The only place a real difference could still hide is the early tail (rows 0–1,535), which tcc never stored. One tcc
  run with prompt logprobs over the whole 2,048-token tail on the 6 quick + 13 final docs (fill and +st arms, two
  repeats for its noise) would settle it; it needs tcc's server on both cards (~1 h). Not run: no evidence points there.
- Levers, not gap explanations: the shipped C removes 25% of the chunk-end state error energy and ~half of layer 24's
  tail-end error; a C fitted on more prompts is the only C-side lever left (late-position-only fitting: no gain).
- Housekeeping: `data/investigate/state-*` (4 × 3.7 GB) and `kld-ref-final13` (13 GB) are deletable once read.
  The brief's `data/home-3af4eaa` does not exist; the capture home is `data/refit/home-3af4eaa`.
