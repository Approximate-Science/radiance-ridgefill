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

## 3. Results
(pending D1)
