# DRAFT (not sent) -- request to radiance: a runtime knob for the heat engine's credit cap

**To:** radiance (codeberg.org/StillDeadcode/radiance), issue tracker. **From:** RidgeFill (Dylan Johnston).
**Status:** draft for Dylan; nothing filed.

## Request
A way to set `HeatParams::credit_cap` at startup, for example `--heat-credit-cap N` (0 = uncapped) or
`RAD_PLACE_CREDIT_CAP`, and ideally a cap per pass kind: one for decode steps, another for prefill chunks.
Better still, a way for an architecture to say how much a histogram it reports should count, e.g. a
`weight` (or `rows_computed`) field in `RadRouting` that `HeatEngine::observe` scales by before the cap.

## Why
1.4.0 credits each layer's histogram at most 512 selections a step (core/place/heat.cpp `observe`). That is
right for stock traffic: it keeps a prompt from swamping the answer's resident set (decode hit rate 77.3% ->
79.0% in your replay). An architecture plugin that computes different layers for different numbers of rows,
however, loses the signal it depends on. RidgeFill runs layers 24-47 of a long prompt's bulk chunks only for
a few exact rows. Through 1.3.0 the heat engine then moved ~2,550 expert slots a rank toward layers 0-23
over ~15-18 long prompts, which is where RidgeFill's sustained 2x prefill comes from (RidgeFill
notes/ramp.md). Under the cap, both halves of a bulk chunk are over 512, so they are credited the same and
the cache stays put. Measured with the same 32K ramp on two R9700s:
- 1.3.0: quality 23.8 -> 16.5 s, layers 0-23 resident 8,616 -> 10,319;
- 1.4.0: flat at ~22 s, both halves shrinking.

The plugin works around it today by reporting an all-zero histogram for the late layers of such passes
(RidgeFill commit 00546fe). That is coarser than what it needs, which is "this histogram stands for N of
the chunk's M rows".

## Smallest change that would serve
`RadRouting` gains `float credit_weight` (0 = 1.0, so existing plugins are unchanged). `observe` multiplies
the histogram's credit by it *after* the cap. An architecture that ran a layer for E of T rows sets E/T.
