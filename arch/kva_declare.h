/* kva_declare.h -- the KVA plugin's declare side: its state, the fitted tensors, the ops it adds
 * to the in-tree graph, and every refusal a mode makes before anything runs (R31).
 *
 * Everything here runs at declare only. step() reads the Kva a rank's real declare filled and
 * nothing else that can change (R15/R99).
 */
#ifndef QWEN4EXP_KVA_DECLARE_H
#define QWEN4EXP_KVA_DECLARE_H

#include "kva_config.h"

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
    std::vector<rad_weight> proj_w, proj_b;   /* [n_layer]: 0 below S */
    std::vector<rad_weight> st;               /* [n_layer]: 0 below S and on attention layers */
    rad_weight score = 0;
    /* kva_state_correct per late delta-net layer (undo, apply) and kva_rho_update (quality). */
    std::vector<rad_op> op_undo, op_apply, op_rho;
    /* What each sequence's late delta-net layer had added to its state at its last approximate
     * chunk end (PLAN D7): one f32 a head, zeroed by the engine at admission. */
    rad_kvgroup kv_applied = 0;
    /* Quality mode's running decay sums N and D per head (kva_rho_update), same lifetime. */
    rad_kvgroup kv_rho = 0;
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
    std::string dump_dir;                     /* RADIANCE_KVA_DUMP, empty when unset */
    /* Stage 6 captures (debug; notes/arch.md "Capture"): RADIANCE_KVA_CAPTURE and
     * RADIANCE_KVA_CAPTURE_STATE, and the copy op the state capture needs. */
    std::string capture_dir, state_dir;
    rad_buf     b_state = 0;
    rad_op      op_state_read = 0;
};

static Kva g_kva[MAX_RANKS];

/* `name` declared when the model holds it; 0 when it does not. The name map comes first because a
 * checkpoint is searched through it (rad_weight_encoding); a container is searched by the declared
 * name, which is the same string. A map for an absent tensor is inert: name maps are read only for
 * declared weights (radiance core/format/checkpoint.cpp:525-541). */
static rad_weight decl_held(RadBuilder* b, Names& nm, const char* name, uint32_t dtype,
                            std::initializer_list<int64_t> shape, int shard, RadWeightGroup grp) {
    const char* d = nm.f("%s", name);
    if (map_copy(b, d, nm.ckpt("%s", name)) < 0) return 0;
    RadEncoding e{};
    if (!weight_enc(b, name, &e)) return 0;
    return decl_w(b, d, dtype, shape, RAD_ACCESS_PER_TOKEN, shard, grp, 1);
}

/* One projector copy (`kva.proj` or `kva.projr`): [n_embd, hc*n_embd] bf16 and its bias, replicated
 * on every rank (PLAN D5), PER_TOKEN so the planner keeps it resident (HANDOVER Stage 2.2). S is
 * the lowest layer held, and every layer from S up must be held with its bias -- a projector with
 * a hole is refused rather than run with one layer computed exactly by accident. */
static int decl_projector(RadBuilder* b, Names& nm, const qwen4exp_fp8::Model& m,
                          const char* base, Kva& k) {
    const int64_t n = m.g.n_embd, wide = m.hccfg.hc * m.g.n_embd;
    for (int64_t l = 0; l < m.g.n_layer; ++l) {
        char wn[96], bn[96];
        std::snprintf(wn, sizeof wn, "%s.%lld.weight", base, (long long)l);
        std::snprintf(bn, sizeof bn, "%s.%lld.bias", base, (long long)l);
        const rad_weight w = decl_held(b, nm, wn, RAD_BF16, {n, wide}, RAD_SHARD_NONE, grp_layer((int)l));
        const rad_weight bias = decl_held(b, nm, bn, RAD_BF16, {n}, RAD_SHARD_NONE, grp_layer((int)l));
        if (k.split < 0 && (w || bias)) k.split = l;
        if (k.split < 0) continue;
        if (!w || !bias) {
            std::fprintf(stderr, "radiance: qwen4exp_kva: the projector '%s' starts at layer %lld "
                                 "and has no %s; every layer from S to the last needs its weight "
                                 "and bias\n", base, (long long)k.split, w ? bn : wn);
            return RAD_E_INVAL;
        }
        k.proj_w[(size_t)l] = w;
        k.proj_b[(size_t)l] = bias;
    }
    k.have_proj = k.split >= 0;
    return RAD_OK;
}

/* One correction copy (`kva.st`, `kva.stswap` or `kva.str`): [value heads, head_v, head_k] f32 a
 * late delta-net layer -- the state's own layout, kv_gdn_state -- for ALL of the model's value heads.
 * REPLICATED, NOT ROW-SHARDED (deviation from PLAN D6, found at the first TP2 load): the loader
 * defines no ROW share of a rank-3 weight (radiance core/format/share.cpp:56), so each rank holds
 * the whole tensor (3 MiB a layer, 54 MiB a rank over 18 layers) and hands kva_state_correct its
 * own heads as a row slice of it at issue (correction_heads in kva_layer.h): rank r's value heads
 * are [r*H, (r+1)*H), the contiguous split the delta net's own weights take.
 * All of the late delta-net layers or none. */
static int decl_correction(RadBuilder* b, Names& nm, const qwen4exp_fp8::Model& m,
                           const char* base, int64_t from, Kva& k) {
    int held = 0, want = 0;
    for (int64_t l = from; l < m.g.n_layer; ++l) {
        if (m.layers[(size_t)l].full) continue;
        char sn[96];
        std::snprintf(sn, sizeof sn, "%s.%lld", base, (long long)l);
        k.st[(size_t)l] = decl_held(b, nm, sn, RAD_F32,
                                    {m.gcfg.n_head_v * m.g.world, m.gcfg.head_v, m.gcfg.head_k},
                                    RAD_SHARD_NONE, grp_layer((int)l));
        held += k.st[(size_t)l] != 0;
        ++want;
    }
    if (held && held != want) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: the correction '%s' covers %d of the %d "
                             "delta-net layers from %lld up; it is all of them or none\n",
                     base, held, want, (long long)from);
        return RAD_E_INVAL;
    }
    k.have_st = held > 0;
    return RAD_OK;
}

static rad_weight decl_score(RadBuilder* b, Names& nm, const qwen4exp_fp8::Model& m,
                             const char* name) {
    return decl_held(b, nm, name, RAD_F32, {m.g.n_vocab_all}, RAD_SHARD_NONE, grp_model());
}

/* rad-convert's view (RADIANCE_KVA_DECLARE=all): every copy the model holds, and nothing else --
 * no ops, no completeness checks, no refusals, because converting is not serving; the serving
 * declare checks what it selects. */
static int decl_every_copy(RadBuilder* b, const qwen4exp_fp8::Model& m, Kva& k) {
    const int64_t n = m.g.n_embd, wide = m.hccfg.hc * m.g.n_embd;
    char name[96];
    for (int64_t l = 0; l < m.g.n_layer; ++l) {
        const RadWeightGroup grp = grp_layer((int)l);
        for (const char* p : { "kva.proj", "kva.projr" }) {
            std::snprintf(name, sizeof name, "%s.%lld.weight", p, (long long)l);
            decl_held(b, k.nm, name, RAD_BF16, {n, wide}, RAD_SHARD_NONE, grp);
            std::snprintf(name, sizeof name, "%s.%lld.bias", p, (long long)l);
            decl_held(b, k.nm, name, RAD_BF16, {n}, RAD_SHARD_NONE, grp);
        }
        if (m.layers[(size_t)l].full) continue;
        for (const char* s : { "kva.st", "kva.stswap", "kva.str" }) {
            std::snprintf(name, sizeof name, "%s.%lld", s, (long long)l);
            decl_held(b, k.nm, name, RAD_F32,
                      {m.gcfg.n_head_v * m.g.world, m.gcfg.head_v, m.gcfg.head_k},
                      RAD_SHARD_NONE, grp);
        }
    }
    for (const char* t : { "kva.rowsel.score", "kva.rowsel.score_none", "kva.rowsel.score_all" })
        decl_score(b, k.nm, m, t);
    return RAD_OK;
}

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
                RAD_WEIGHTS(k.st[(size_t)l]));
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
    if (!k.have_proj) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: mode %s needs the projector '%s.L.weight' "
                             "and the model holds none of it\n", mode, c.proj);
        return RAD_E_UNSUPPORTED;
    }
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
    if (c.tail < kMinTail) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: kva.tail is %lld tokens; the shortest exact "
                             "tail this method was measured at is %lld\n",
                     (long long)c.tail, (long long)kMinTail);
        return RAD_E_INVAL;
    }
    if (c.mode == MODE_QUALITY && !k.have_rowsel) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: mode quality selects exact rows from the "
                             "table '%s' and the model does not hold it\n", c.score);
        return RAD_E_UNSUPPORTED;
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
static int decl_fill(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx, Kva& k) {
    Geom g = m.g;
    g.max_tok = ctx->max_tok;
    for (rad_buf h : { m.b_h, m.a_x.x, m.a_x.cq(), m.a_x.cs() })
        if (h) RAD_ARCH_TRY(rad_buf_concurrent(b, h));
    RAD_ARCH_TRY(k.quant.declare(b, g, m.a_x, g.n_embd));
    const int64_t wide = m.hccfg.hc * g.n_embd;
    for (int64_t l = k.split; l < g.n_layer; ++l) {
        const rad_op h = rw(b, RAD_OP(b, "gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                           RAD_INT("K", wide), RAD_STR("dtype", g.dtype)),
                                RAD_WEIGHTS(k.proj_w[(size_t)l], k.proj_b[(size_t)l])),
                            {m.b_h}, {m.a_x.x});
        if (!h && !ctx->shape_probe) {
            std::fprintf(stderr, "radiance: qwen4exp_kva: no kernel serves the projector "
                                 "(gemm_nt_bias, N %lld, K %lld, %s)\n",
                         (long long)g.n_embd, (long long)wide, g.dtype);
            return RAD_E_UNSUPPORTED;
        }
        k.op_proj[(size_t)l] = h;
    }
    return RAD_OK;
}

}  /* namespace qwen4exp_kva */

#include "kva_declare_masked.h"

#endif /* QWEN4EXP_KVA_DECLARE_H */
