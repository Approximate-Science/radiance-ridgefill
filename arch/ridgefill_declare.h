/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* ridgefill_declare.h -- the RidgeFill plugin's declare side: its state, the ops it adds to the in-tree graph,
 * and every refusal a mode makes before anything runs (R31). The fitted tensors come from the
 * projector folder (ridgefill_projector.h), as RAW operands.
 *
 * Everything here runs at declare only. step() reads the RidgeFill a rank's real declare filled and
 * nothing else that can change (R15/R99).
 */
#ifndef RIDGEFILL_DECLARE_H
#define RIDGEFILL_DECLARE_H

#include "ridgefill_adapter.h"
#include "ridgefill_config.h"
#include "ridgefill_int8.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ridgefill {

using namespace rad::arch;

struct RidgeFill {
    /* The model's facts (ridgefill_adapter.h), filled first by the declare: everything below that the core
     * reads of the model it reads here. */
    RidgeFillAdapter ad{};
    Config  cfg{};
    /* Its own name pool: declared names must outlive declare (rad_arch.h's Names), and a sizing
     * declare runs on its own thread, so it must not append to the real model's pool. */
    Names   nm{""};
    int64_t split = -1;          /* S: the lowest projected layer, -1 when no projector is held */
    /* G: the recurrent block's chunk (the adapter's tile fact). Every bulk end lands on it, so a split
     * scan's two halves keep the single conv-prep/kkt pass's tiles (PLAN-FIX §4). */
    int64_t tile = 0;
    bool    have_proj = false, have_st = false, have_rowsel = false;
    /* KL mode (max_out_rows > 0) asks for logits on bulk rows, which are not the model's there:
     * it serves stock unless RADIANCE_RIDGEFILL_SCORE_BULK says the caller scores the tail only (R73). */
    bool    out_rows_ok = true;
    /* The folder's tensors on this rank (ridgefill_projector.h): RAW operands, RAD_NONE where absent. */
    std::vector<RadOperand> proj_w, proj_b;   /* [n_layer]: none below S */
    /* THE INT8 PROJECTOR (ridgefill_int8.h, R79): the maps' scales, the stream's int8 codes and scales
     * (quantised once a pass, read by every late layer's GEMM), the quantiser and the bias add. */
    bool int8 = false;
    std::vector<RadOperand> proj_s;           /* [n_layer]: none below S, none for bf16 maps */
    rad_buf b_q8 = 0, b_s8 = 0;
    rad_op  op_quant8 = 0, op_bias = 0;
    std::vector<RadOperand> st;               /* [n_layer]: this rank's heads; none below S and on attention */
    RadOperand score = RAD_NONE;
    /* The staging ring (host placement, ridgefill_projector.h plan_maps): per late layer the host map's
     * row block and the VRAM slot it is copied into, and the copy op. */
    std::vector<RadOperand> ring_src, ring_dst;
    rad_op op_ring = 0;
    int64_t ring_end = 0;                     /* blocks the ring moves a pass: the late layers' indices, then
                                               * the final map's hc blocks (n_layer .. n_layer + hc) */
    /* THE MTP `final` MAP (ridgefill_final.h, DD-D): with MTP on, the predicted final stream of the bulk rows,
     * computed from the layer-S stream in hc column blocks streamed through the ring, written into the
     * trunk's stream before the epilogue. */
    bool want_final = false;
    std::vector<RadOperand> final_w, final_b; /* [hc]: each block's map rows and bias, in its ring slot */
    rad_buf b_final = 0;
    rad_op  op_final = 0;
    /* ridgefill_state_correct per late delta-net layer (undo, apply) and ridgefill_rho_update (quality). */
    std::vector<rad_op> op_undo, op_apply, op_rho;
    /* What each sequence's late delta-net layer had added to its state at its last approximate
     * chunk end (PLAN D7): one f32 a head, zeroed by the engine at admission. */
    rad_kvgroup kv_applied = 0;
    /* Quality mode's running decay sums N and D per head (ridgefill_rho_update), same lifetime. */
    rad_kvgroup kv_rho = 0;
    /* DD-A's branch-hazard instrument (ridgefill_hazard.h): one slot a sequence, bound to one late layer
     * so checkpoints snapshot it, and the op that counts and records. */
    rad_kvgroup kv_meta = 0;
    int         meta_layer = -1;
    rad_op      op_hazard = 0;
    rad_op      hazard_first = 0;             /* the op as core_declare_first declared it, before the graph */
    /* The projector GEMM per late layer, and the quantiser that writes a projected block input's
     * codes as the connection read would have. */
    std::vector<rad_op> op_proj;              /* [n_layer]: 0 below S */
    QuantFP8 quant{};
    /* THE MASKED PATH (PLAN-FIX §3): which rows of the step a late layer approximates (`mask`,
     * device data), where the last sequence's bulk lies (`bounds` = {s, b, b, e}), the layer-S
     * stream every projector reads (`h_S`), and the projected block input with its codes (`xp`). */
    rad_buf b_mask = 0, b_bounds = 0, b_hs = 0;
    ActFP8  xp{};
    rad_op  op_mask = 0, op_cast = 0, op_select = 0, op_drop = 0;
    bool    mask_scored = false;              /* ridgefill_mask was declared with the score table */
    /* THE STAGER LEVER (notes/impl.md §2): the zero expert offsets and the scratch rows the late
     * layers' gate-up probes read and write. */
    rad_buf b_zeros = 0, b_probe = 0;
    int64_t n_zeros = 0;
    /* THE TAIL-ONLY STRADDLE (A.1) needs every late attention layer on its per-row sparse gated form
     * (rad_block_attn_gated_fp8.h:542-558): declared here, decided per pass from the keyed context
     * against the largest exactness bound. */
    bool    straddle_layers = false;
    int64_t qsa_exact_to = 0;
    std::string dump_dir;                     /* RADIANCE_RIDGEFILL_DUMP, empty when unset */
    /* Stage 6 captures (debug; notes/arch.md "Capture"): RADIANCE_RIDGEFILL_CAPTURE and
     * RADIANCE_RIDGEFILL_CAPTURE_STATE, and the copy op the state capture needs. */
    std::string capture_dir, state_dir, logits_dir;   /* debug dumps, read at declare */
    rad_buf     b_state = 0;
    rad_op      op_state_read = 0;
};

static RidgeFill g_ridgefill[MAX_RANKS];

/* The refusals a serving mode needs before anything is issued (R31). Each names the number or the
 * tensor that refused it. */
static int check_mode(const RidgeFill& k, int64_t max_tok) {
    const Config& c = k.cfg;
    const char* mode = kModeNames[c.mode];
    /* A TAIL LONGER THAN A STEP (REFUTATION §4, R100). n_ahead is capped at the step C =
     * max_tok (radiance core/sched/batch.cpp:1066-1080), so a full chunk only PROVES that row i
     * has (n_tok-1-i) + C tokens after it: rows [0, n_tok - ceil_G(T - C)) are bulk and the rest run
     * exact (derive()) -- lower coverage, never a wrong row. That leaves a bulk row only while
     * ceil_G(T - C) < C, i.e. T <= 2C - G; above it no chunk could ever be approximated, so the
     * mode is refused rather than served as an expensive no-op. The operator route that needs no
     * plugin change: --checkpoint-interval below --max-num-batched-tokens (README, R84). */
    const int64_t G = k.ad.tile;
    if (c.tail > 2 * max_tok - G) {
        std::fprintf(stderr, "radiance: %s: ridgefill.tail is %lld tokens and the largest step "
                             "is %lld (--max-num-batched-tokens); the scheduler never reports more "
                             "than %lld prompt tokens ahead, so only rows followed by %lld - %lld "
                             "more tokens inside the chunk are provably bulk, and with the delta "
                             "net's %lld-row tile none is once the tail exceeds %lld (2 x %lld - %lld). "
                             "Lower the tail or raise the step.\n", g_log_name,
                     (long long)c.tail, (long long)max_tok, (long long)max_tok, (long long)c.tail,
                     (long long)max_tok, (long long)G, (long long)(2 * max_tok - G),
                     (long long)max_tok, (long long)G);
        return RAD_E_UNSUPPORTED;
    }
    if (c.tail < c.min_tail) {
        std::fprintf(stderr, "radiance: %s: ridgefill.tail is %lld tokens; the shortest exact "
                             "tail this method was measured at is %lld\n", g_log_name,
                     (long long)c.tail, (long long)c.min_tail);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* What the late layers assume of the model (its adapter's facts), refused by name if it ever stops
 * holding. */
static int check_fill(const RidgeFill& k) {
    const RidgeFillAdapter& a = k.ad;
    if (k.split < a.split_lo) {
        std::fprintf(stderr, "radiance: %s: the projector starts at layer %lld and the "
                             "n-gram embedding enters the stream at layer %lld; a split at or below "
                             "it would predict from a stream that never received it\n", g_log_name,
                     (long long)k.split, (long long)(a.split_lo - 1));
        return RAD_E_UNSUPPORTED;
    }
    for (int64_t l = k.split - 1; l < a.n_layer; ++l) {
        /* The masked path issues each late layer's FFN itself (the adapter's ffn hook), and that copy has
         * no calibration tap: a calibration run serves through the in-tree path only. */
        if (a.calibrated[(size_t)l]) {
            std::fprintf(stderr, "radiance: %s: layer %lld runs the MoE calibration tap, "
                                 "which RidgeFill's late layers do not issue; calibrate with "
                                 "RADIANCE_RIDGEFILL=off\n", g_log_name, (long long)l);
            return RAD_E_UNSUPPORTED;
        }
        if (l < k.split || a.ext_in[(size_t)l]) continue;
        std::fprintf(stderr, "radiance: %s: layer %lld's block owns its input norm, and "
                             "the fill hands blocks their input already normed\n", g_log_name, (long long)l);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* The lean fill's ops: one projector GEMM per late layer and the quantiser that writes the block
 * input's codes as the connection read would (QuantFP8 takes int8 or E4M3 off the model, nothing
 * for a bf16 one). The lean fill writes the model's `x` itself (the a_x decision, notes/arch.md §6);
 * the masked path issues the same handles over h_S -> x_P. Declared after the whole in-tree graph,
 * so every buffer they touch takes the whole program (rad_buf_concurrent, PLAN D4). Under a sizing
 * declare the in-tree buffer handles are the real declare's, which name the same buffers. */
/* THE INT8 PROJECTOR'S OPS (R79): libr4d's quant_act_i8g over the layer-S stream (once a pass, into
 * the plugin's codes and scales), ridgefill.so's forward of the engine's int8 gemm_nt_q per late layer (its
 * maps in the stored form ridgefill_int8.h relaid them into), and the in-tree row-broadcast add for the
 * bias, which gemm_nt_q has no operand for. Cost: the codes buffer, [max_tok, hc*n] int8 + scales
 * (~21 MiB at 2,048 rows), in the activation arena. */
static int decl_fill_i8(RadBuilder* b, const RadBuildCtx* ctx, RidgeFill& k) {
    const RidgeFillAdapter& a = k.ad;
    const int64_t wide = a.wide, n = a.n_embd, max_tok = ctx->max_tok;
    k.b_q8 = decl_b(b, k.nm.f("ridgefill_stream_q8"), RAD_I8, {max_tok, wide});
    k.b_s8 = decl_b(b, k.nm.f("ridgefill_stream_s8"), RAD_F32, {max_tok, wide / kI8Group});
    if (!k.b_q8 || !k.b_s8) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_buf_concurrent(b, k.b_q8));
    RAD_ARCH_TRY(rad_buf_concurrent(b, k.b_s8));
    k.op_quant8 = RAD_OP(b, "quant_act_i8g", RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("n", wide),
                                                       RAD_INT("group", kI8Group), RAD_STR("dtype", a.dtype)),
                         RAD_NOWEIGHTS);
    k.op_bias = RAD_OP(b, "add", RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("n", n),
                                            RAD_STR("dtype", a.dtype)), RAD_NOWEIGHTS);
    for (int64_t l = k.split; l < a.n_layer; ++l)
        k.op_proj[(size_t)l] = rw(b, RAD_OP(b, "ridgefill_gemm_nt_q",
                                            RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("N", n),
                                                       RAD_INT("K", wide), RAD_INT("group", kI8Group),
                                                       RAD_STR("dtype", "i8a8")),
                                            RAD_NOWEIGHTS),
                                  {k.b_q8, k.b_s8}, {a.buf_x});
    const char* missing = !k.op_quant8 ? "quant_act_i8g" : !k.op_bias ? "add"
                        : !k.op_proj[(size_t)k.split] ? "ridgefill_gemm_nt_q" : nullptr;
    if (!missing || ctx->shape_probe) return RAD_OK;
    std::fprintf(stderr, "radiance: %s: no kernel serves the int8 projector's %s (N %lld, K %lld): "
                         "ridgefill.so offers ridgefill_gemm_nt_q only when libr4d is loaded\n", g_log_name, missing,
                 (long long)n, (long long)wide);
    return RAD_E_UNSUPPORTED;
}

/* THE STAGING RING'S COPY: libr4d's strided row copy (cast bf16 -> bf16, r4d_p2p_copy2d -- bytes moved,
 * no value converted, so int8 codes ride it as rows of bf16 pairs), one row block a layer: the bf16
 * map and its bias, [n + 1, hc*n], or the int8 map's stored codes, scales and bias in fewer rows. */
static int decl_ring(RadBuilder* b, const RadBuildCtx* ctx, RidgeFill& k) {
    const int64_t wide = k.ad.wide;
    k.op_ring = RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, k.ad.n_embd + 1), RAD_INT("n", wide),
                                             RAD_STR("from", "bf16"), RAD_STR("to", "bf16")), RAD_NOWEIGHTS);
    if (k.op_ring || ctx->shape_probe) return RAD_OK;
    std::fprintf(stderr, "radiance: %s: no kernel serves the staging ring's copy (cast bf16, "
                         "n %lld): the projector is streamed from host memory and cannot run without it\n", g_log_name,
                 (long long)wide);
    return RAD_E_UNSUPPORTED;
}

static int decl_fill(RadBuilder* b, const RadBuildCtx* ctx, RidgeFill& k) {
    const RidgeFillAdapter& a = k.ad;
    for (rad_buf h : { a.buf_stream, a.buf_x, a.buf_x_q, a.buf_x_s })
        if (h) RAD_ARCH_TRY(rad_buf_concurrent(b, h));
    if (a.declare_model) RAD_ARCH_TRY(a.declare_model(b, ctx, k));
    if (k.int8) {
        RAD_ARCH_TRY(decl_fill_i8(b, ctx, k));
        return decl_ring(b, ctx, k);
    }
    const int64_t wide = a.wide, n = a.n_embd, max_tok = ctx->max_tok;
    for (int64_t l = k.split; l < a.n_layer; ++l) {
        /* ridgefill.so's forward of the engine's gemm_nt_bias row (kernels/forward.cpp): the projector is
         * plugin memory, not a weight, so it rides as an IN operand. */
        const rad_op h = rw(b, RAD_OP(b, "ridgefill_gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("N", n),
                                           RAD_INT("K", wide), RAD_STR("dtype", a.dtype)),
                                RAD_NOWEIGHTS),
                            {a.buf_stream}, {a.buf_x});
        if (!h && !ctx->shape_probe) {
            std::fprintf(stderr, "radiance: %s: no kernel serves the projector "
                                 "(ridgefill_gemm_nt_bias, N %lld, K %lld, %s): ridgefill.so offers it only "
                                 "when libr4d (device) or libref (host) is loaded\n", g_log_name,
                         (long long)n, (long long)wide, a.dtype);
            return RAD_E_UNSUPPORTED;
        }
        k.op_proj[(size_t)l] = h;
    }
    return decl_ring(b, ctx, k);
}

}  /* namespace ridgefill */

#include "ridgefill_projector.h"
#include "ridgefill_final.h"
#include "ridgefill_declare_masked.h"

#endif /* RIDGEFILL_DECLARE_H */
