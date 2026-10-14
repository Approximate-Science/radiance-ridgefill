# DRAFT (not sent) -- request to radiance: let sustained long-prompt traffic move the expert cache again

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

A second 1.4.0 rule matters more: "NO SWAPS BEHIND A PASS THE STAGER ARMED" (core/engine.cpp). Every chunk of
a long prompt is armed, so prompts back to back give the mover about one unarmed step each: 64 promotions a
request, against 272 on 1.3.0. Later in the ramp it reaches 1,044 a request, 96-97% of them readmits.
Stock radiance 1.5.0 also gets slower over the same ramp (32K prompts back to back: 22.86 -> 28.67 s, staged
bytes 279 -> 351 GiB a prompt) where 1.3.0's stock did not. A plugin-side workaround (reporting an all-zero
histogram for the late layers of such passes, RidgeFill branch heat-credit-1.5.0) fixes the ranking but not
the move rate, and measured no prefill gain on 1.5.0 (RidgeFill notes/rebase-1.5.0.md).

Measured on two R9700s (Gen5 x16 + Gen4 x4), shipped defaults plus the flashnext serving flags.

## Smallest changes that would serve
1. Some swaps behind armed passes, e.g. a small per-step budget that the stager's own copies do not
   saturate, or swaps at a prompt's chunk boundaries. Without this, a sustained run of long prompts cannot
   adapt the cache at all, with or without RidgeFill.
2. `RadRouting` gains `float credit_weight` (0 = 1.0, so existing plugins are unchanged), and `observe`
   multiplies the histogram's credit by it *after* the cap. An architecture that ran a layer for E of T rows
   sets E/T.
