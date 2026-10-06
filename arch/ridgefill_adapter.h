/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* ridgefill_adapter.h -- the ONE interface between the RidgeFill core and a model (notes/adapter-split-spec.md §1.3).
 *
 * The core (every ridgefill_*.h but this file's users' adapters) is compiled into each adapter's .so and
 * names no model type: what it needs of the model it reads here, filled once per rank by the
 * adapter's `adapter_of` after the in-tree declare: facts, and hooks for whatever the model owns.
 * Nothing speculative: every field is read by the core's declare or step.
 */
#ifndef RIDGEFILL_ADAPTER_H
#define RIDGEFILL_ADAPTER_H

#include <arch/rad_arch.h>

#include "ridgefill_plan.h"   /* Pass, Path: what late_block is told */

#include <cstdint>
#include <vector>

namespace ridgefill {

struct RidgeFill;        /* ridgefill_declare.h: the core's per-rank declare state, which the hooks fill */
struct StateDump;  /* ridgefill_dump.h: RADIANCE_RIDGEFILL_CAPTURE_STATE's per-step copies */

/* This rank's recurrent state per late layer, as the correction ops see it; {0,0,0} = none. `first` is
 * this rank's first head of the model's `n_head_all`: the rank's heads are [first, first + n_head), which
 * is NOT rank * n_head when the heads do not divide evenly (radiance 1.1.0 at three ranks: 18, 15, 15 of
 * 48, the extra heads on the rank without attention, which is the LAST rank). */
struct StateShape { int64_t n_head = 0, sd0 = 0, sd1 = 0, first = 0, n_head_all = 0; };

struct RidgeFillAdapter {
    /* identity: the refusal prefix, the projector manifest's "adapter" string, and the in-tree .so the
     * release guard forwards to (it shadows that file by stem, arch/CMakeLists.txt). */
    const char* log_name = nullptr;
    const char* match_name = nullptr;
    const char* shadow_so = nullptr;
    /* geometry. `wide` is the stream that feeds the projector (hc * n_embd on a hyper-connection
     * model); `tile` the delta net's chunk, which the bulk end is rounded to -- 1 when no recurrent
     * block makes that check vacuous; `split_lo` the lowest split whose stream has seen every
     * injected input (the PLE layer + 1). */
    int64_t n_layer = 0, n_embd = 0, n_vocab_all = 0, wide = 0;
    int64_t n_vocab = 0;     /* this rank's logits columns (vocab-sharded under TP) */
    int64_t tile = 1, split_lo = 0;
    /* the stager probes ride behind the gate-up of layer S - probe_depth, so streaming needs that
     * many routed layers below the split. top_k == 0 = a dense model: no probes, no drop arm. */
    int64_t probe_depth = 0, top_k = 0, n_expert = 0, n_ff_exp = 0;
    /* the measured tail facts (KVA-FACTS §5): a fit measured at 512 rows is not trusted below it */
    int64_t min_tail = 0, default_tail = 0;
    uint32_t act_dtype = RAD_BF16;  /* the stream and block-input buffers' element type */
    const char* dtype = "bf16";     /* the same, as the op params name it (gemm, cast, add) */
    StateShape state;
    /* per layer, length n_layer: attention (not recurrent); the block takes a pre-normed input (the
     * fill hands it one, check_fill); the block runs the MoE calibration tap the core never issues;
     * the FFN is routed (probes, streaming, the drop arm). */
    std::vector<uint8_t> full, ext_in, calibrated, routed;
    /* per attention layer: what it lacks for speed's tail-only straddle (nullptr = nothing), and the
     * rows its sparse attention keeps exact (the straddle must reach past them, derive) */
    std::vector<const char*> straddle_lack;
    std::vector<int64_t> qsa_exact_to;
    /* buffers the core issues against: the stream, the block input and the code pair its producer
     * writes (whichever dtype that is), the logits. The projector maps' dtype is not a model fact:
     * it is read from the folder's manifest (bf16 or int8). */
    rad_buf buf_stream = 0, buf_x = 0, buf_x_q = 0, buf_x_s = 0, buf_logits = 0;
    rad_buf buf_route_ids = 0;   /* the routed FFN's expert ids, which ridgefill_drop_rows rewrites; 0 = dense */
    /* hooks; nullptr = the capability is absent and the core skips it */
    int (*declare_model)(RadBuilder*, const RadBuildCtx*, RidgeFill&) = nullptr;          /* the fill's quantiser */
    int (*declare_codes)(RadBuilder*, const RadBuildCtx*, RidgeFill&) = nullptr;          /* RidgeFill::xp's code pair */
    const char* (*decl_state_ops)(RadBuilder*, const RadBuildCtx*, RidgeFill&) = nullptr; /* correction ops: the
                                                                     missing op's name, or nullptr */
    /* The step's late-layer issues, which the core's drivers (ridgefill_layer.h) order around the projection.
     * conn: the connection in front of the block (ffn false) or the FFN, its read or its write, over
     * rows [r0, r0 + rows) of a T-row step. late_block: the block over the path's rows -- LEAN its
     * cache-writing pieces only, MASKED the whole block with the correction spliced in, STRADDLE /
     * DECODERS the rows [r0, r0 + rows) whole and the rest lean. ffn: the feed-forward over rows
     * [r0, to) (to < 0: all), dropping the rows `mask` marks with `drop` when both are set. */
    void (*conn)(RadCtx*, int64_t li, bool ffn, bool write, int64_t T, int64_t r0, int64_t rows) = nullptr;
    void (*late_block)(RadCtx*, const RidgeFill&, int64_t li, const RadBatch*, const Pass&, StateDump*, Path,
                       int64_t r0, int64_t rows) = nullptr;
    void (*ffn)(RadCtx*, const RidgeFill&, int64_t li, const RadBatch*, int64_t r0, int64_t to, rad_op drop,
                rad_buf mask) = nullptr;
    /* The rest of the step (ridgefill_step.h), verbatim copies of the in-tree step's pieces: prologue (embedding
     * .. rope table) and epilogue (last connection, logits) around the layers; stock_layer one exact
     * layer below S, carrying the stager probes when `probes` (the routed layer S - probe_depth of a
     * streaming pass); stock_step the in-tree step itself, for every pass RidgeFill leaves alone. */
    void (*prologue)(RadCtx*, const RadBatch*) = nullptr;
    void (*stock_layer)(RadCtx*, const RidgeFill&, int64_t li, const RadBatch*, bool probes) = nullptr;
    void (*epilogue)(RadCtx*, const RadBatch*) = nullptr;
    void (*stock_step)(RadCtx*, const RadBatch*) = nullptr;
    /* the debug captures (RADIANCE_RIDGEFILL_CAPTURE / _CAPTURE_STATE), which interleave with the model's own
     * blocks; null = the adapter keeps none and the core skips them */
    void (*capture_step)(RadCtx*, const RidgeFill&, const RadBatch*) = nullptr;
    void (*finish_state)(RadCtx*, const RidgeFill&, const RadBatch*, StateDump&, bool approx) = nullptr;
    void (*capture_mixed)(RadCtx*, const RidgeFill&, const RadBatch*, bool approx) = nullptr;
};

}  // namespace ridgefill

#endif
