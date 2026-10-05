# notes/staged.md -- Stage D on `stage-e`: the MTP `final` map and multimodal skip-and-resume

Continued in the Stage E session after the merge-readiness report (orchestrator, 2026-10-05: "code + static tests
while others hold the GPUs"). Rows: R69-R71, R74-R76, R97 (fix-246/REQUIREMENTS-FIX.md).

## 1. What is built (commits on `stage-e`)
| commit | what |
|---|---|
| fd78fc9 | `tools/kva_projector.py final`: a folder plus the MTP final map [w, w+1] from the source it was fitted on (pytest 9 passed). R69's folder form (the container sidecar route is retired, A′) |
| 8750c10 | arch: `RADIANCE_KVA_FINAL` on/off; with MTP (max_spec > 0) and a folder holding the map, `kva_final` + `kva_gemm_nt_bias` declared; the map streams through the ring as hc row blocks after the late layers; predicted rows into b_h through the device mask (masked/straddle) or a row copy (lean) BEFORE the epilogue. Static: `the_final_map_is_held_and_declared_only_with_mtp`, `with_mtp_the_bulk_rows_take_the_predicted_final_stream_before_the_epilogue` (R71) |
| 6a68dde | (Stage E) one ring slot: the final blocks ride the same slot; a bf16 block's size even with int8 maps; h_S kept with the final map |
| 69ae174 | R74's static case on the published container's picture geometry; `tools/media_ident.py` (R76), `tools/mtp_accept.py` (R70, R97); README: how media steps are handled |

## 2. Multimodal (Dylan's decision, PLAN-FIX §6.3): media steps stock, text after them approximated
- The rule was already in derive (`arch/qwen4exp_kva.cpp:109-110`): eligible only with no encoder pass, no draft pass,
  no media rows and no mixed rotary components. The text chunks after an image need no plugin state: every pass is
  decided from its own keyed fields, and the fill and the projected layers take their rotary operands through the
  in-tree helpers (`rope_posmc`, `rope_pos1`; `arch/kva_fill.h:84,107`, `arch/kva_layer.h:327`,
  `arch/qwen4exp_kva.cpp:143`). The row mask reads index positions (`mask_rows`), which is the bookkeeping position.
- **The default serving configuration is the media one**: `--mm-max-patches` is `auto` (16384 with a vision tower,
  radiance docs/GUIDE.md:890) and the published container carries one, with interleaved M-RoPE `11 11 10`. So
  `rope_pos` is set on every pass of every server measured in this project (components equal on text passes), and the
  approximate path has always run at the rotary operands -- the byte-identity rows (R144, S1) were taken that way.
- R74 static (69ae174, `media_steps_run_stock_and_the_text_after_an_image_approximates_at_its_rotary_positions`):
  that geometry and a two-block vision tower with a patch budget; encoder pass, media rows (mixed or not) and mixed
  components issue the in-tree step op for op in speed and quality; a text chunk after an image is approximated and
  every in-tree op it issues takes the position operand the stock step gives that op. Mutants M1-M6
  (`evidence/stagee/scripts/mutate_media.py`): media rows / mixed / encoder approximated, filled k / projected q
  rotated at the index, the indexer's work list without the rotary planes.
- R76 engine and R70/R97: session D1 (`evidence/stagee/scripts/d1.sh`, queued 18:58Z on gpu.lock, home 69ae174).
  Predictions registered first (labbook seq 430-431: HD-R76-plumb-media, HD-R70-final).
- R75 (post-image tail NLL on 5 image + ≥ 12K-text docs) needs that corpus: asked the orchestrator for its location.

## 3. Results -- D1 (2026-10-05 20:55-21:03Z, frozen home 406e746 = stage-e with main + Stage C merged: ctest -LE gpu 3/3;
evidence/staged/d1/session.log; int8 only)
- **R76 GREEN -- plumb after an image = off, byte for byte.** One picture (radiance's tests/data/media/red.png) then quick
  doc 3 cut to 48,000 characters: 12,155 prompt tokens, greedy 64: off and plumb (FORCE_STREAM) text sha256 c6cfeba6…
  both; 4 approximate passes logged in plumb and quality (chunks 2-5: the chunk with the picture's rows ran stock, the
  last chunk is the exact tail). This build returns no logprobs (the server refuses them), so text is the comparison.
  quality's text differs (04681d98…), as an approximation may.
- **R70 -- the final map's gain is small, and the deficit it was built for does not show here.** MTP depth 3, 16K
  prompts (tools/speed.py's builder), 256 greedy tokens, 5 reps, median tokens/step: stock 2.265; int8 + final
  **2.339 (1.033x)**; int8, RADIANCE_KVA_FINAL=off **2.297 (1.014x)**. final and off produce byte-identical texts in all
  5 reps (the drafts change speed, never text), so final vs off is a paired comparison: final ≥ off in 5/5 reps, +1.8%
  tokens a step. Against stock the continuations differ (KVA changes the text), so the ratio to stock compares
  different texts. R9V's 0.895x deficit without the map is not reproduced at T 2048 (the draft head's recent history is
  the exact tail). HD-R70-final: REFUTED (off is not in 0.85-0.92x; final − off = 0.019x < 0.03).
  Cost of the map: the slot is a bf16 block (50 MiB instead of 25.4 MiB VRAM a rank), +200 MiB host, +210 MB a pass
  over the link. Whether to ship it is Dylan's call; the data says +1.8% drafted tokens a step for ~25 MiB more VRAM.
- **R97 -- not isolated by this instrument.** /stats link h2d a request (mover + stager): stock 125-137 GiB, KVA 43-83
  GiB (approximate passes skip the late layers' experts), so the +213 MiB of a first exact pass after an approximate one
  is inside the per-request totals' spread. Decode after the first request: ~7 ms/token in all three arms.
- (mutant paths above: the scripts are `evidence/stagee/scripts/mutant_*.py` since the GPU-time audit renamed them)

## 4. Results -- D2 (2026-10-05 21:02-21:15Z, home 406e746; evidence/staged/d2/; int8 only, quality-bf16 skipped)
- **R75 -- no sign that text after a picture is approximated worse; the instrument is coarse.** tools/media_corpus.py
  (seed 75: bar/line/flow/table/pie, quick docs 3-7), 4 cuts each (44-56K characters, 11.6-15.0K tokens): 20 picture +
  20 text-only twin prompts, last-row candidates through the engine's buffer probe (`-vv`, RADIANCE_DUMP_BUF
  sampler.cand_idx/cand_val), quality int8 vs exact: **top-1 agreement 14/20 picture, 14/20 text-only; top-20 KL
  over the surviving candidates 0.052 (picture) vs 0.297 (text-only), max 0.37 / 2.65.** 190 approximate passes.
  Limits, stated: in 15 of 20 rows of each kind stock's top-1 is the same special token (id 248046), so most
  agreement is template-driven (where it is not: 1/5 picture, 1/5 text); and the container's default top-p 0.95 left a
  single candidate in 22 of 40 reference rows (the planes hold only what survives the request's top-p). A sharper
  run would send top_p 1.0 and a prompt whose first answer token is content. HD-R75-media-band: picture within the
  text-only band on top-1 (equal) and below it on KL -- not killed; weak evidence.
- **Mutants: 34/34 caught** (beside D2, on src-406e746): mutant_slot S1-S10 (one slot, host correction, int8 slot and
  h_S), mutant_final F1-F8, mutant_ring R1-R5 + R7, mutant_media M1-M6 + X1-X4 (straddle line, the merge's two fixes);
  X5 (final map on by default) caught on the working tree after 6cc3f5e.
