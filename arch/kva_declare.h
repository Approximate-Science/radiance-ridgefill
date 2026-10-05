/* kva_declare.h -- the KVA plugin's declare side: its state, the ops it adds to the in-tree graph,
 * and every refusal a mode makes before anything runs (R31). The fitted tensors come from the
 * projector folder (kva_projector.h), as RAW operands.
 *
 * Everything here runs at declare only. step() reads the Kva a rank's real declare filled and
 * nothing else that can change (R15/R99).
 */
#ifndef QWEN4EXP_KVA_DECLARE_H
#define QWEN4EXP_KVA_DECLARE_H

#include "kva_config.h"
#include "kva_int8.h"

#include <algorithm>
#include <string>
#include <vector>

namespace qwen4exp_kva {

using namespace rad::arch;

struct Kva {
    Config  cfg{};
    /* Its own name pool: declared names must outlive declare (rad_arch.h's Names), and a sizing
     * declare runs on its own thread, so it must not append to the real model's pool. */
    Names   nm{""};
    int64_t split = -1;          /* S: the lowest projected layer, -1 when no projector is held */
    /* G: the delta net's chunk (GdnFP8::Config::chunk). Every bulk end lands on it, so a split
     * scan's two halves keep the single conv-prep/kkt pass's tiles (PLAN-FIX §4). */
    int64_t tile = 0;
    bool    have_proj = false, have_st = false, have_rowsel = false;
    /* KL mode (max_out_rows > 0) asks for logits on bulk rows, which are not the model's there:
     * it serves stock unless RADIANCE_KVA_SCORE_BULK says the caller scores the tail only (R73). */
    bool    out_rows_ok = true;
    /* The folder's tensors on this rank (kva_projector.h): RAW operands, RAD_NONE where absent. */
    std::vector<RadOperand> proj_w, proj_b;   /* [n_layer]: none below S */
    /* THE INT8 PROJECTOR (kva_int8.h, R79): the maps' scales, the stream's int8 codes and scales
     * (quantised once a pass, read by every late layer's GEMM), the quantiser and the bias add. */
    bool int8 = false;
    std::vector<RadOperand> proj_s;           /* [n_layer]: none below S, none for bf16 maps */
    rad_buf b_q8 = 0, b_s8 = 0;
    rad_op  op_quant8 = 0, op_bias = 0;
    std::vector<RadOperand> st;               /* [n_layer]: this rank's heads; none below S and on attention */
    RadOperand score = RAD_NONE;
    /* The staging ring (host placement, kva_projector.h plan_maps): per late layer the host map's
     * row block and the VRAM slot it is copied into, and the copy op. */
    std::vector<RadOperand> ring_src, ring_dst;
    rad_op op_ring = 0;
    int64_t ring_end = 0;                     /* blocks the ring moves a pass: the late layers' indices, then
                                               * the final map's hc blocks (n_layer .. n_layer + hc) */
    /* THE MTP `final` MAP (kva_final.h, DD-D): with MTP on, the predicted final stream of the bulk rows,
     * computed from the layer-S stream in hc column blocks streamed through the ring, written into the
     * trunk's stream before the epilogue. */
    bool want_final = false;
    std::vector<RadOperand> final_w, final_b; /* [hc]: each block's map rows and bias, in its ring slot */
    rad_buf b_final = 0;
    rad_op  op_final = 0;
    /* kva_state_correct per late delta-net layer (undo, apply) and kva_rho_update (quality). */
    std::vector<rad_op> op_undo, op_apply, op_rho;
    /* What each sequence's late delta-net layer had added to its state at its last approximate
     * chunk end (PLAN D7): one f32 a head, zeroed by the engine at admission. */
    rad_kvgroup kv_applied = 0;
    /* Quality mode's running decay sums N and D per head (kva_rho_update), same lifetime. */
    rad_kvgroup kv_rho = 0;
    /* DD-A's branch-hazard instrument (kva_hazard.h): one slot a sequence, bound to one late layer
     * so checkpoints snapshot it, and the op that counts and records. */
    rad_kvgroup kv_meta = 0;
    int         meta_layer = -1;
    rad_op      op_hazard = 0;
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
    bool    mask_scored = false;              /* kva_mask was declared with the score table */
    /* THE STAGER LEVER (notes/impl.md §2): the zero expert offsets and the scratch rows the late
     * layers' gate-up probes read and write. */
    rad_buf b_zeros = 0, b_probe = 0;
    int64_t n_zeros = 0;
    /* THE TAIL-ONLY STRADDLE (A.1) needs every late attention layer on its per-row sparse gated form
     * (rad_block_attn_gated_fp8.h:503-519): declared here, decided per pass from the keyed context
     * against the largest exactness bound. */
    bool    straddle_layers = false;
    int64_t qsa_exact_to = 0;
    std::string dump_dir;                     /* RADIANCE_KVA_DUMP, empty when unset */
    /* Stage 6 captures (debug; notes/arch.md "Capture"): RADIANCE_KVA_CAPTURE and
     * RADIANCE_KVA_CAPTURE_STATE, and the copy op the state capture needs. */
    std::string capture_dir, state_dir, logits_dir;   /* debug dumps, read at declare */
    rad_buf     b_state = 0;
    rad_op      op_state_read = 0;
};

static Kva g_kva[MAX_RANKS];

/* A LINEAR group of [heads, 1, inner] f32 a sequence, bound to every late delta-net layer: one
 * slot per sequence the engine zeroes at admission, keeps for the sequence's life and snapshots
 * with every checkpoint (kv.cpp:1449-1486). The tape audit fails a step whose issues depend on host
 * state outside the pass key, so per-sequence state lives on the device, here (PLAN D7). */
static rad_kvgroup decl_late_group(RadBuilder* b, const qwen4exp_fp8::Model& m, Kva& k,
                                   const char* name, int64_t inner) {
    RadKVGroupDecl d{};
    d.kind         = RAD_KV_LINEAR;
    d.dtype        = RAD_F32;
    d.n_head_kv    = m.gcfg.n_head_v;
    d.state_dim[0] = 1;
    d.state_dim[1] = inner;
    const rad_kvgroup g = rad_decl_kv_group(b, k.nm.f("%s", name), &d);
    for (int64_t l = k.split; g && l < m.g.n_layer; ++l)
        if (!m.layers[(size_t)l].full && rad_bind_layer_kv(b, (int)l, g) < 0) return 0;
    return g;
}

/* The correction's and the decay sums' kva.so ops, declared per late delta-net layer. A handle
 * that comes back null means no kernel library serves the op: kva.so is not on the search path, or
 * declines this machine -- reported by name in decl_selected (R31). */
static const char* decl_kernel_ops(RadBuilder* b, const qwen4exp_fp8::Model& m,
                                   const RadBuildCtx* ctx, Kva& k) {
    const Config& c = k.cfg;
    const char* missing = nullptr;
    const bool corrects = k.have_st && (c.mode == MODE_SPEED || c.mode == MODE_QUALITY);
    if (corrects && !(k.kv_applied = decl_late_group(b, m, k, "kv_kva_applied", 1)))
        return "kv_kva_applied";
    if (corrects && c.mode == MODE_QUALITY && !(k.kv_rho = decl_late_group(b, m, k, "kv_kva_rho", 2)))
        return "kv_kva_rho";
    /* The decay sums read the in-tree a|b buffer from an op declared after the graph. */
    if (k.kv_rho && rad_buf_concurrent(b, m.b_ab) < 0) return "gdn_ab";
    for (int64_t l = k.split; corrects && l < m.g.n_layer; ++l) {
        const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
        if (lay.full) continue;
        for (int apply = 0; apply < 2; ++apply) {
            const rad_op h = RAD_OP(b, "kva_state_correct",
                RAD_PARAMS(RAD_RANGE("M", 1, m.g.max_seqs), RAD_STR("mode", apply ? "apply" : "undo"),
                           RAD_F64("alpha", c.alpha), RAD_INT("n_head", m.gcfg.n_head_v),
                           RAD_INT("sd0", m.gcfg.head_v), RAD_INT("sd1", m.gcfg.head_k)),
                RAD_NOWEIGHTS);
            (apply ? k.op_apply : k.op_undo)[(size_t)l] = h;
            if (!h) missing = "kva_state_correct";
        }
        /* The decay sums read the layer's own a|b columns and its own A_log / dt_bias. Under a
         * sizing declare these in-tree handles are the real declare's, which name the same weights
         * (the in-tree declare declares them in the same order at every max_tok; the static test
         * checks the weight lists agree). */
        if (!k.kv_rho) continue;
        k.op_rho[(size_t)l] = rw(b, RAD_OP(b, "kva_rho_update",
            RAD_PARAMS(RAD_RANGE("M", 1, ctx->max_tok), RAD_INT("n_head", m.gcfg.n_head_v)),
            RAD_WEIGHTS(lay.gdn.w_a_log, lay.gdn.w_dt_bias)), {lay.gdn.w.ab, k.b_mask, k.b_bounds}, {});
        if (!k.op_rho[(size_t)l]) missing = "kva_rho_update";
    }
    return missing;
}

/* The refusals a serving mode needs before anything is issued (R31). Each names the number or the
 * tensor that refused it. */
static int check_mode(const Kva& k, const qwen4exp_fp8::Model& m) {
    const Config& c = k.cfg;
    const char* mode = kModeNames[c.mode];
    const int64_t max_tok = m.g.max_tok;
    /* A TAIL LONGER THAN A STEP (REFUTATION §4, R100). n_ahead is capped at the step C =
     * max_tok (radiance core/sched/batch.cpp:1066-1080), so a full chunk only PROVES that row i
     * has (n_tok-1-i) + C tokens after it: rows [0, n_tok - ceil_G(T - C)) are bulk and the rest run
     * exact (derive()) -- lower coverage, never a wrong row. That leaves a bulk row only while
     * ceil_G(T - C) < C, i.e. T <= 2C - G; above it no chunk could ever be approximated, so the
     * mode is refused rather than served as an expensive no-op. The operator route that needs no
     * plugin change: --checkpoint-interval below --max-num-batched-tokens (README, R84). */
    const int64_t G = m.gcfg.chunk;
    if (c.tail > 2 * max_tok - G) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: kva.tail is %lld tokens and the largest step "
                             "is %lld (--max-num-batched-tokens); the scheduler never reports more "
                             "than %lld prompt tokens ahead, so only rows followed by %lld - %lld "
                             "more tokens inside the chunk are provably bulk, and with the delta "
                             "net's %lld-row tile none is once the tail exceeds %lld (2 x %lld - %lld). "
                             "Lower the tail or raise the step.\n",
                     (long long)c.tail, (long long)max_tok, (long long)max_tok, (long long)c.tail,
                     (long long)max_tok, (long long)G, (long long)(2 * max_tok - G),
                     (long long)max_tok, (long long)G);
        return RAD_E_UNSUPPORTED;
    }
    if (c.tail < c.min_tail) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: kva.tail is %lld tokens; the shortest exact "
                             "tail this method was measured at is %lld\n",
                     (long long)c.tail, (long long)c.min_tail);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* What the late layers assume of the in-tree model, refused by name if it ever stops holding. */
static int check_fill(const qwen4exp_fp8::Model& m, const Kva& k) {
    if (m.ple_layer >= k.split) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: the projector starts at layer %lld and the "
                             "n-gram embedding enters the stream at layer %lld; a split at or below "
                             "it would predict from a stream that never received it\n",
                     (long long)k.split, (long long)m.ple_layer);
        return RAD_E_UNSUPPORTED;
    }
    for (int64_t l = k.split - 1; l < m.g.n_layer; ++l) {
        const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
        /* The masked path issues each late layer's MoE pass itself (kva_moe.h), and that copy has
         * no calibration tap: a calibration run serves through the in-tree path only. */
        if (lay.mlp.op_gram_gu) {
            std::fprintf(stderr, "radiance: qwen4exp_kva: layer %lld runs the MoE calibration tap, "
                                 "which KVA's late layers do not issue; calibrate with "
                                 "RADIANCE_KVA=off\n", (long long)l);
            return RAD_E_UNSUPPORTED;
        }
        if (l < k.split || (lay.full ? lay.attn.ext_in : lay.gdn.ext_in)) continue;
        std::fprintf(stderr, "radiance: qwen4exp_kva: layer %lld's block owns its input norm, and "
                             "the fill hands blocks their input already normed\n", (long long)l);
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
 * the plugin's codes and scales), kva.so's forward of the engine's int8 gemm_nt_q per late layer (its
 * maps in the stored form kva_int8.h relaid them into), and the in-tree row-broadcast add for the
 * bias, which gemm_nt_q has no operand for. Cost: the codes buffer, [max_tok, hc*n] int8 + scales
 * (~21 MiB at 2,048 rows), in the activation arena. */
static int decl_fill_i8(RadBuilder* b, const Geom& g, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx, Kva& k) {
    const int64_t wide = m.hccfg.hc * g.n_embd;
    k.b_q8 = decl_b(b, k.nm.f("kva_stream_q8"), RAD_I8, {g.max_tok, wide});
    k.b_s8 = decl_b(b, k.nm.f("kva_stream_s8"), RAD_F32, {g.max_tok, wide / kI8Group});
    if (!k.b_q8 || !k.b_s8) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_buf_concurrent(b, k.b_q8));
    RAD_ARCH_TRY(rad_buf_concurrent(b, k.b_s8));
    k.op_quant8 = RAD_OP(b, "quant_act_i8g", RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", wide),
                                                       RAD_INT("group", kI8Group), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS);
    k.op_bias = RAD_OP(b, "add", RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                            RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS);
    for (int64_t l = k.split; l < g.n_layer; ++l)
        k.op_proj[(size_t)l] = rw(b, RAD_OP(b, "kva_gemm_nt_q",
                                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                                       RAD_INT("K", wide), RAD_INT("group", kI8Group),
                                                       RAD_STR("dtype", "i8a8")),
                                            RAD_NOWEIGHTS),
                                  {k.b_q8, k.b_s8}, {m.a_x.x});
    const char* missing = !k.op_quant8 ? "quant_act_i8g" : !k.op_bias ? "add"
                        : !k.op_proj[(size_t)k.split] ? "kva_gemm_nt_q" : nullptr;
    if (!missing || ctx->shape_probe) return RAD_OK;
    std::fprintf(stderr, "radiance: qwen4exp_kva: no kernel serves the int8 projector's %s (N %lld, K %lld): "
                         "kva.so offers kva_gemm_nt_q only when libr4d is loaded\n", missing,
                 (long long)g.n_embd, (long long)wide);
    return RAD_E_UNSUPPORTED;
}

/* THE STAGING RING'S COPY: libr4d's strided row copy (cast bf16 -> bf16, r4d_p2p_copy2d -- bytes moved,
 * no value converted, so int8 codes ride it as rows of bf16 pairs), one row block a layer: the bf16
 * map and its bias, [n + 1, hc*n], or the int8 map's stored codes, scales and bias in fewer rows. */
static int decl_ring(RadBuilder* b, const Geom& g, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx, Kva& k) {
    const int64_t wide = m.hccfg.hc * g.n_embd;
    k.op_ring = RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, g.n_embd + 1), RAD_INT("n", wide),
                                             RAD_STR("from", "bf16"), RAD_STR("to", "bf16")), RAD_NOWEIGHTS);
    if (k.op_ring || ctx->shape_probe) return RAD_OK;
    std::fprintf(stderr, "radiance: qwen4exp_kva: no kernel serves the staging ring's copy (cast bf16, "
                         "n %lld): the projector is streamed from host memory and cannot run without it\n",
                 (long long)wide);
    return RAD_E_UNSUPPORTED;
}

static int decl_fill(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx, Kva& k) {
    Geom g = m.g;
    g.max_tok = ctx->max_tok;
    for (rad_buf h : { m.b_h, m.a_x.x, m.a_x.cq(), m.a_x.cs() })
        if (h) RAD_ARCH_TRY(rad_buf_concurrent(b, h));
    RAD_ARCH_TRY(k.quant.declare(b, g, m.a_x, g.n_embd));
    if (k.int8) {
        RAD_ARCH_TRY(decl_fill_i8(b, g, m, ctx, k));
        return decl_ring(b, g, m, ctx, k);
    }
    const int64_t wide = m.hccfg.hc * g.n_embd;
    for (int64_t l = k.split; l < g.n_layer; ++l) {
        /* kva.so's forward of the engine's gemm_nt_bias row (kernels/forward.cpp): the projector is
         * plugin memory, not a weight, so it rides as an IN operand. */
        const rad_op h = rw(b, RAD_OP(b, "kva_gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                           RAD_INT("K", wide), RAD_STR("dtype", g.dtype)),
                                RAD_NOWEIGHTS),
                            {m.b_h}, {m.a_x.x});
        if (!h && !ctx->shape_probe) {
            std::fprintf(stderr, "radiance: qwen4exp_kva: no kernel serves the projector "
                                 "(kva_gemm_nt_bias, N %lld, K %lld, %s): kva.so offers it only "
                                 "when libr4d (device) or libref (host) is loaded\n",
                         (long long)g.n_embd, (long long)wide, g.dtype);
            return RAD_E_UNSUPPORTED;
        }
        k.op_proj[(size_t)l] = h;
    }
    return decl_ring(b, g, m, ctx, k);
}

}  /* namespace qwen4exp_kva */

#include "kva_projector.h"
#include "kva_final.h"
#include "kva_declare_masked.h"

#endif /* QWEN4EXP_KVA_DECLARE_H */
