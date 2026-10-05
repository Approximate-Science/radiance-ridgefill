/* qwen4exp_adapter.h -- the qwen4exp side of the core/adapter split (notes/adapter-split-spec.md §1.2):
 * every fact the core reads, taken from the in-tree model after its declare. Included by
 * qwen4exp_kva.cpp after <qwen4exp_fp8/qwen4exp_fp8.cpp>, so `qwen4exp_fp8::Model` is complete here.
 */
#ifndef QWEN4EXP_ADAPTER_H
#define QWEN4EXP_ADAPTER_H

#include "kva_adapter.h"

namespace qwen4exp_kva {

/* A LINEAR group of [heads, 1, inner] f32 a sequence, bound to every late delta-net layer: one
 * slot per sequence the engine zeroes at admission, keeps for the sequence's life and snapshots
 * with every checkpoint (kv.cpp:1449-1486). The tape audit fails a step whose issues depend on host
 * state outside the pass key, so per-sequence state lives on the device, here (PLAN D7). */
inline rad_kvgroup decl_late_group(RadBuilder* b, const qwen4exp_fp8::Model& m, Kva& k,
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

/* The correction's and the decay sums' kva.so ops, declared per late delta-net layer: the adapter's
 * decl_state_ops hook, because the decay sums read the delta net's own a|b columns and A_log / dt_bias.
 * A handle that comes back null means no kernel library serves the op: kva.so is not on the search
 * path, or declines this machine -- reported by name in decl_selected (R31). */
inline const char* decl_state_ops(RadBuilder* b, const RadBuildCtx* ctx, Kva& k) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[ctx->rank];
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

/* The block input's code pair, written by the fill as the connection read would (QuantFP8 takes int8 or
 * E4M3 off the model's a_x, nothing for a bf16 one): the adapter's declare_model hook. */
inline int declare_model(RadBuilder* b, const RadBuildCtx* ctx, Kva& k) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[ctx->rank];
    Geom g = m.g;
    g.max_tok = ctx->max_tok;
    return k.quant.declare(b, g, m.a_x, g.n_embd);
}

/* The projected block input's code pair, mirroring the model's a_x (int8 when the trunk is fed int8
 * codes, E4M3 otherwise, none for a bf16 model), so kva_select can copy a projected row's codes over
 * an exact row's and no linear re-quantises anything: the declare_codes hook. */
inline int declare_codes(RadBuilder* b, const RadBuildCtx* ctx, Kva& k) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[ctx->rank];
    Geom g = m.g;
    g.max_tok = ctx->max_tok;
    if (m.a_x.cq()) RAD_ARCH_TRY(k.xp.declare_qs(b, k.nm, g, "kva_x_proj", g.n_embd, 0, m.a_x.q8_fed));
    k.xp.q8_fed = m.a_x.q8_fed;
    return RAD_OK;
}

/* What a late attention layer lacks for speed's tail-only straddle -- the per-row sparse attention (the
 * indexer's selection and its sequence map) and the fused sparse attention + gate -- or nullptr. */
inline const char* straddle_missing(const AttnGatedFP8& a) {
    if (!a.qsa_sel) return "the indexer's selection (qsa_sel)";
    if (!a.qsa_sequ) return "the indexer's sequence map (qsa_sequ)";
    if (!a.op_attn_gq) return "the fused sparse attention + gate (attn_paged_gate_quant)";
    return nullptr;
}

/* ---- the step hooks: the in-tree layer's pieces (qwen4exp_fp8.cpp:1407-1425) over the rows the core's
 * driver names. Each reads this rank's model, the one the declare filled. */

/* hc_mix in front of the block, hc_ffn in front of the MoE; a write also takes the debug residual dump. */
inline void conn(RadCtx* c, int64_t li, bool ffn, bool write, int64_t T, int64_t r0, int64_t rows) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const HyperConn& h = ffn ? l.hc_ffn : l.hc_mix;
    if (!write) { h.read(c, T, r0, rows); return; }
    h.write(c, T, r0, rows);
    dbg_resid(c, (int)li, ffn ? "ffn" : "mix", m.g.n_embd, m.b_h, m.a_x.x);
}

inline void late_block(RadCtx* c, const Kva& k, int64_t li, const RadBatch* batch, const Pass& p,
                       StateDump* sd, Path path, int64_t r0, int64_t rows) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    if (path == PATH_LEAN) {   /* the cache writers only: K/V and the indexer keys, or the corrected scan */
        if (l.full) {
            qsa_keys(c, l.qsa, l.attn.w.h, batch);
            attn_kv(c, l.attn, batch);
            return;
        }
        gdn_project(c, l.gdn, rows);
        correct(c, k, m, li, k.op_undo[(size_t)li], batch, false);
        gdn_scan(c, l.gdn, batch);
        read_state(c, k, m, li, batch, sd);
        correct(c, k, m, li, k.op_apply[(size_t)li], batch, false);
        return;
    }
    if (path == PATH_MASKED) {
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch); l.attn.step(c, batch); }
        else        gdn_masked(c, k, m, li, batch, p, sd);
        return;
    }
    if (l.full)                     attn_rows(c, l, batch, r0, rows);
    else if (path == PATH_STRADDLE) gdn_straddle(c, k, m, li, batch, p, sd);
    else                            gdn_decoders(c, k, m, li, batch, p, sd);
}

inline void ffn(RadCtx* c, const Kva&, int64_t li, const RadBatch* batch, int64_t r0, int64_t to, rad_op drop,
                rad_buf mask) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    moe_layer(c, m.layers[(size_t)li].mlp, MoeArm{ drop, mask, {} }, batch, r0, to);
}

/* The fit facts the method was measured at on this model (KVA-FACTS §5). */
constexpr int64_t kAdapterMinTail = 512, kAdapterDefaultTail = 2048;

/* Rank `m`'s facts. Read after qwen4exp_fp8::declare has filled `m`; the buffers are this rank's. */
inline KvaAdapter adapter_of(const qwen4exp_fp8::Model& m) {
    KvaAdapter a;
    a.log_name = "qwen4exp_kva";
    a.match_name = "qwen4exp";
    a.shadow_so = "qwen4exp_fp8.so";
    a.n_layer = m.g.n_layer;
    a.n_embd = m.g.n_embd;
    a.n_vocab_all = m.g.n_vocab_all;
    a.world = m.g.world;
    a.wide = m.hccfg.hc * m.g.n_embd;
    a.tile = m.gcfg.chunk;
    a.split_lo = m.ple_layer + 1;  /* -1 (no PLE) gives 0: no constraint */
    a.probe_depth = 3;             /* the probes ride behind layer S-3's gate-up (notes/impl.md §2) */
    a.top_k = m.moecfg.top_k;
    a.buf_route_ids = m.b_eids;
    for (const qwen4exp_fp8::Layer& l : m.layers) {
        a.n_expert = std::max(a.n_expert, l.mlp.c.n_expert);
        a.n_ff_exp = std::max(a.n_ff_exp, l.mlp.c.n_ff_exp);
        a.full.push_back(l.full ? 1 : 0);
        a.ext_in.push_back((l.full ? l.attn.ext_in : l.gdn.ext_in) ? 1 : 0);
        a.calibrated.push_back(l.mlp.op_gram_gu ? 1 : 0);
        a.routed.push_back(l.mlp.c.n_expert > 0 ? 1 : 0);
        a.straddle_lack.push_back(l.full ? straddle_missing(l.attn) : nullptr);
        a.qsa_exact_to.push_back(l.full ? l.attn.qsa_exact_to : 0);
    }
    a.min_tail = kAdapterMinTail;
    a.default_tail = kAdapterDefaultTail;
    a.act_dtype = m.g.act_dtype;
    a.dtype = m.g.dtype;
    a.state = {m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k};
    a.buf_stream = m.b_h;
    a.buf_x = m.a_x.x;
    a.buf_x_q = m.a_x.cq();        /* the pair the connection read writes: int8 when q8_fed, else E4M3 */
    a.buf_x_s = m.a_x.cs();
    a.buf_logits = m.b_logits;
    a.declare_model = &declare_model;
    a.declare_codes = &declare_codes;
    a.conn = &conn;
    a.late_block = &late_block;
    a.ffn = &ffn;
    a.decl_state_ops = &decl_state_ops;
    return a;
}

}  // namespace qwen4exp_kva

#endif
