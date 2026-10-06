/* qwen4exp_adapter.h -- the qwen4exp side of the core/adapter split (notes/adapter-split-spec.md §1.2):
 * every fact the core reads, taken from the in-tree model after its declare (adapter_of, at the end),
 * and every hook body: the declare-side ops only a gated delta net and a routed MoE have, then the
 * step-side pieces that issue this model's blocks. Included by qwen4exp_ridgefill.cpp after
 * <qwen4exp_fp8/qwen4exp_fp8.cpp> and the core, so `qwen4exp_fp8::Model` and `ridgefill::RidgeFill` are complete.
 *
 * Every hook reads g_model[rank] -- the model the REAL declare filled -- also under a sizing declare,
 * exactly as the plugin did before the split: the in-tree declare writes a sizing declare's model to
 * scratch, and the layer schedule and head counts read here do not depend on max_tok.
 */
#ifndef QWEN4EXP_ADAPTER_H
#define QWEN4EXP_ADAPTER_H

#include "ridgefill_adapter.h"

namespace qwen4exp_ridgefill {

/* A LINEAR group of [heads, 1, inner] f32 a sequence, bound to every late delta-net layer: one
 * slot per sequence the engine zeroes at admission, keeps for the sequence's life and snapshots
 * with every checkpoint (kv.cpp:1449-1486). The tape audit fails a step whose issues depend on host
 * state outside the pass key, so per-sequence state lives on the device, here (PLAN D7). */
inline rad_kvgroup decl_late_group(RadBuilder* b, const qwen4exp_fp8::Model& m, RidgeFill& k,
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

/* The correction's and the decay sums' ridgefill.so ops, declared per late delta-net layer: the adapter's
 * decl_state_ops hook, because the decay sums read the delta net's own a|b columns and A_log / dt_bias.
 * A handle that comes back null means no kernel library serves the op: ridgefill.so is not on the search
 * path, or declines this machine -- reported by name in decl_selected (R31). */
inline const char* decl_state_ops(RadBuilder* b, const RadBuildCtx* ctx, RidgeFill& k) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[ctx->rank];
    const Config& c = k.cfg;
    const char* missing = nullptr;
    const bool corrects = k.have_st && (c.mode == MODE_SPEED || c.mode == MODE_QUALITY);
    if (corrects && !(k.kv_applied = decl_late_group(b, m, k, "kv_ridgefill_applied", 1)))
        return "kv_ridgefill_applied";
    if (corrects && c.mode == MODE_QUALITY && !(k.kv_rho = decl_late_group(b, m, k, "kv_ridgefill_rho", 2)))
        return "kv_ridgefill_rho";
    /* The decay sums read the in-tree a|b buffer from an op declared after the graph. */
    if (k.kv_rho && rad_buf_concurrent(b, m.b_ab) < 0) return "gdn_ab";
    for (int64_t l = k.split; corrects && l < m.g.n_layer; ++l) {
        const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
        if (lay.full) continue;
        for (int apply = 0; apply < 2; ++apply) {
            const rad_op h = RAD_OP(b, "ridgefill_state_correct",
                RAD_PARAMS(RAD_RANGE("M", 1, m.g.max_seqs), RAD_STR("mode", apply ? "apply" : "undo"),
                           RAD_F64("alpha", c.alpha), RAD_INT("n_head", m.gcfg.n_head_v),
                           RAD_INT("sd0", m.gcfg.head_v), RAD_INT("sd1", m.gcfg.head_k)),
                RAD_NOWEIGHTS);
            (apply ? k.op_apply : k.op_undo)[(size_t)l] = h;
            if (!h) missing = "ridgefill_state_correct";
        }
        /* The decay sums read the layer's own a|b columns and its own A_log / dt_bias. Under a
         * sizing declare these in-tree handles are the real declare's, which name the same weights
         * (the in-tree declare declares them in the same order at every max_tok; the static test
         * checks the weight lists agree). */
        if (!k.kv_rho) continue;
        k.op_rho[(size_t)l] = rw(b, RAD_OP(b, "ridgefill_rho_update",
            RAD_PARAMS(RAD_RANGE("M", 1, ctx->max_tok), RAD_INT("n_head", m.gcfg.n_head_v)),
            RAD_WEIGHTS(lay.gdn.w_a_log, lay.gdn.w_dt_bias)), {lay.gdn.w.ab, k.b_mask, k.b_bounds}, {});
        if (!k.op_rho[(size_t)l]) missing = "ridgefill_rho_update";
    }
    return missing;
}

/* The block input's code pair, written by the fill as the connection read would (QuantFP8 takes int8 or
 * E4M3 off the model's a_x, nothing for a bf16 one): the adapter's declare_model hook. */
inline int declare_model(RadBuilder* b, const RadBuildCtx* ctx, RidgeFill& k) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[ctx->rank];
    Geom g = m.g;
    g.max_tok = ctx->max_tok;
    return k.quant.declare(b, g, m.a_x, g.n_embd);
}

/* The projected block input's code pair, mirroring the model's a_x (int8 when the trunk is fed int8
 * codes, E4M3 otherwise, none for a bf16 model), so ridgefill_select can copy a projected row's codes over
 * an exact row's and no linear re-quantises anything: the declare_codes hook. */
inline int declare_codes(RadBuilder* b, const RadBuildCtx* ctx, RidgeFill& k) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[ctx->rank];
    Geom g = m.g;
    g.max_tok = ctx->max_tok;
    if (m.a_x.cq()) RAD_ARCH_TRY(k.xp.declare_qs(b, k.nm, g, "ridgefill_x_proj", g.n_embd, 0, m.a_x.q8_fed));
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

/* ---- the step hooks: the in-tree layer's pieces (qwen4exp_fp8.cpp:1465-1485) over the rows the core's
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

/* Layer li's block on an approximate pass. LEAN issues only what writes the caches (the attention's K/V
 * and indexer keys, or the delta net's projections and its corrected scan): nothing reads a bulk row's
 * block output, so none is computed. MASKED runs the in-tree block over every row, the delta net's last
 * sequence corrected. STRADDLE and DECODERS run the window [r0, r0 + rows) whole -- the tail, or the
 * decoders -- and the rest lean, which is where speed mode's saving beside other rows comes from. */
inline void late_block(RadCtx* c, const RidgeFill& k, int64_t li, const RadBatch* batch, const Pass& p,
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

/* Layer li's MoE over rows [r0, to), issued by hand (qwen4exp_moe.h) so that, given the drop op and the
 * mask, the bulk rows' routing slots are emptied: a dropped slot's expert is neither staged nor read. */
inline void ffn(RadCtx* c, const RidgeFill&, int64_t li, const RadBatch* batch, int64_t r0, int64_t to, rad_op drop,
                rad_buf mask) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    moe_layer(c, m.layers[(size_t)li].mlp, MoeArm{ drop, mask, {} }, batch, r0, to);
}

/* ---- the exact layers and the step around them: copies of the in-tree step's pieces, issued by the core's
 * approximate_step (ridgefill_step.h) and the debug captures. */

/* qwen4exp_fp8.cpp:1450-1463, verbatim: embedding, media rows, PLE hash, the stream's first value,
 * the rope table. */
inline void prologue(RadCtx* c, qwen4exp_fp8::Model& m, const RadBatch* batch) {
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    if (m.op_embed_ar) RAD_ISSUE_N(c, m.op_embed_ar, T * m.g.n_embd, brows(m.a_x.x, T), RAD_NONE);
    if (m.ple_layer >= 0) m.ple.ids(c, batch);
    m.enter.step(c, T);
    if (m.op_rope_cs && T <= qk_fuse_rows(m.g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));
}

/* qwen4exp_fp8.cpp:1465-1485 for one layer -- with the MoE issued through `arm` when one is given
 * (layer S-3 of a streaming pass: stock rows, the stager probes behind its gate-up GEMM). */
inline void layer(RadCtx* c, qwen4exp_fp8::Model& m, int64_t li, const RadBatch* batch,
                  const MoeArm* arm = nullptr) {
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const int64_t T = batch->n_tok;
    if (li == m.ple_layer) m.ple.step(c, batch);
    l.hc_mix.read(c, T, 0, T);
    if (l.full) { l.qsa.step(c, l.attn.w.h, batch); l.attn.step(c, batch); }
    else        l.gdn.step(c, batch);
    l.hc_mix.write(c, T, 0, T);
    dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_h, m.a_x.x);
    l.hc_ffn.read(c, T, 0, T);
    if (arm) moe_layer(c, l.mlp, *arm, batch);
    else     l.mlp.step(c, batch);
    l.hc_ffn.write(c, T, 0, T);
    dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_h, m.a_x.x);
}

/* qwen4exp_fp8.cpp:1487-1496: the 97th connection and, when the chunk asks for them, logits. On an
 * approximate chunk they are requested only on exact rows in production (PLAN-FIX §6.2). */
inline void epilogue(RadCtx* c, qwen4exp_fp8::Model& m, const RadBatch* batch) {
    const int64_t T = batch->n_tok;
    m.mixer.read(c, T, 0, T);
    if (batch->n_out <= 0) return;
    RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.a_x.x, T),
                praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.b_hout, batch->n_out));
    RAD_ISSUE_N(c, m.op_logits, batch->n_out, brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                brows(m.b_logits, batch->n_out));
}

/* THE STAGER LEVER (notes/impl.md §2): behind layer S-3's gate-up GEMM, while the stager holds layers
 * S-3 and S-2 in its two buffers and has released neither, probe the gate-up of every layer from S-1
 * to the last. Each probe makes the stager start the layer after it, and a start that finds both
 * buffers held STREAMS that layer: from here on no late layer is staged whole, and each reads only
 * the experts its exact rows route to. Layer S-1 itself, which runs every row, is still staged when
 * the real pass reaches layer S-2. Correctness never depends on any of it: a probe moves no number. */
inline MoeArm probe_arm(RadCtx* c, const RidgeFill& k, const qwen4exp_fp8::Model& m) {
    MoeArm arm;
    arm.after_gate_up = [c, &k, &m] {
        for (int64_t x = k.split - 1; x < m.g.n_layer; ++x)
            moe_probe(c, m.layers[(size_t)x].mlp, k.b_zeros, k.b_probe);
    };
    return arm;
}

/* The hooks over them: this rank's model, as the declare filled it. stock_layer carries the probes on the
 * one layer the core names (S - probe_depth of a streaming pass); every other layer is the in-tree one. */
inline void prologue_hook(RadCtx* c, const RadBatch* batch) { prologue(c, qwen4exp_fp8::g_model[rad_rank(c)], batch); }
inline void epilogue_hook(RadCtx* c, const RadBatch* batch) { epilogue(c, qwen4exp_fp8::g_model[rad_rank(c)], batch); }
inline void stock_layer(RadCtx* c, const RidgeFill& k, int64_t li, const RadBatch* batch, bool probes) {
    qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    if (!probes) { layer(c, m, li, batch); return; }
    const MoeArm arm = probe_arm(c, k, m);
    layer(c, m, li, batch, &arm);
}

/* RADIANCE_RIDGEFILL_CAPTURE (mode off, rank 0): the stock step, issued through the same pieces as
 * every other path here (the static test holds them to the in-tree step), with host copies of the
 * stream entering layer S and of every late layer's block input `x`, read right after its
 * connection read and before the block writes its output over it. */
inline void capture_step(RadCtx* c, const RidgeFill& k, const RadBatch* batch) {
    qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    const int64_t T = batch->n_tok, n = m.g.n_embd;
    Capture cap;
    const bool ok = capture_begin(c, batch, k.capture_dir, &cap);
    if (!ok) std::fprintf(stderr, "radiance: qwen4exp_ridgefill: RADIANCE_RIDGEFILL_CAPTURE: device read failed\n");
    prologue(c, m, batch);
    for (int64_t li = 0; li < m.g.n_layer; ++li) {
        const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
        if (ok && li == k.split) capture_rows(c, cap, "boundary", m.b_h, T, m.hccfg.hc * n);
        if (li == m.ple_layer) m.ple.step(c, batch);
        l.hc_mix.read(c, T, 0, T);
        if (ok && li >= k.split) {
            capture_rows(c, cap, "bi." + std::to_string(li), m.a_x.x, T, n);
            cap.layers.push_back((int)li);
        }
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch); l.attn.step(c, batch); }
        else        l.gdn.step(c, batch);
        l.hc_mix.write(c, T, 0, T);
        dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_h, m.a_x.x);
        l.hc_ffn.read(c, T, 0, T);
        l.mlp.step(c, batch);
        l.hc_ffn.write(c, T, 0, T);
        dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_h, m.a_x.x);
    }
    epilogue(c, m, batch);
    if (ok) capture_end(cap, k.split, n, m.hccfg.hc);
}

/* RADIANCE_RIDGEFILL_CAPTURE_STATE: whatever late delta-net layer the step did not already copy (an
 * exact chunk copies here, after the step; nothing touches a layer's state after its scan), then
 * one file for this (chunk, rank). */
inline void finish_state(RadCtx* c, const RidgeFill& k, const RadBatch* batch, StateDump& sd, bool approx) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    for (int64_t li = k.split; li < m.g.n_layer; ++li)
        if (!m.layers[(size_t)li].full &&
            std::find(sd.layers.begin(), sd.layers.end(), (int)li) == sd.layers.end())
            read_state(c, k, m, li, batch, &sd);
    state_end(c, k.state_dir, batch, sd, m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k, rad_rank(c),
              m.g.world, approx, kModeNames[k.cfg.mode]);
}

/* R61 -- RADIANCE_RIDGEFILL_CAPTURE_STATE ON A MIXED STEP (decoders beside a prefill chunk): after the
 * step, every sequence's late delta-net states, so the decoders' slots of two runs of one arrangement
 * (RidgeFill and off) can be compared byte for byte; the prefill's slot is the positive control. */
inline void capture_mixed(RadCtx* c, const RidgeFill& k, const RadBatch* batch, bool approx) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    std::vector<int> layers;
    for (int64_t li = k.split; li < m.g.n_layer; ++li)
        if (!m.layers[(size_t)li].full) layers.push_back((int)li);
    std::vector<float> data;
    for (int64_t seq = 0; seq < batch->n_seq; ++seq)
        for (int li : layers)
            if (!copy_state(c, k, m, li, batch, seq, &data)) return;
    mixed_state_end(c, k.state_dir, batch, data, layers, {m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k},
                    rad_rank(c), approx, kModeNames[k.cfg.mode]);
}

/* The fit facts the method was measured at on this model (KVA-FACTS §5). */
constexpr int64_t kAdapterMinTail = 512, kAdapterDefaultTail = 2048;

/* Rank `m`'s facts. Read after qwen4exp_fp8::declare has filled `m`; the buffers are this rank's. Once
 * a declare (a few hundred bytes of per-layer arrays), never on the step path. */
inline RidgeFillAdapter adapter_of(const qwen4exp_fp8::Model& m) {
    RidgeFillAdapter a;
    a.log_name = "qwen4exp_ridgefill";
    a.match_name = "qwen4exp";
    a.shadow_so = "qwen4exp_fp8.so";
    a.n_layer = m.g.n_layer;
    a.n_embd = m.g.n_embd;
    a.n_vocab_all = m.g.n_vocab_all;
    a.n_vocab = m.g.n_vocab;
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
    a.prologue = &prologue_hook;
    a.stock_layer = &stock_layer;
    a.epilogue = &epilogue_hook;
    a.stock_step = &qwen4exp_fp8::step;
    a.capture_step = &capture_step;
    a.finish_state = &finish_state;
    a.capture_mixed = &capture_mixed;
    a.decl_state_ops = &decl_state_ops;
    return a;
}

}  // namespace qwen4exp_ridgefill

#endif
