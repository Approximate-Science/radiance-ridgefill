/* kva_layer.h -- one late layer of an approximate pass: the masked layer (PLAN-FIX §3, §4, §8) and
 * the lean fill (the pure all-bulk case), with the correction's issues they share.
 *
 * THE MASKED LAYER runs every in-tree op over all n_tok rows with the stock handles and lets the
 * DEVICE decide which rows use the projection: the connection read writes the exact block input for
 * every row, kva_select overwrites the rows the mask marks with the projected input and its codes,
 * the block runs whole, and the MoE drops the marked rows' routing slots. Decode rows and every
 * other sequence are therefore computed exactly as stock computes them, at stock GEMM shapes.
 */
#ifndef QWEN4EXP_KVA_LAYER_H
#define QWEN4EXP_KVA_LAYER_H

namespace qwen4exp_kva {

using namespace rad::arch;

enum Path { PATH_STOCK = 0, PATH_LEAN, PATH_MASKED };
static const char* const kPathNames[] = { "stock", "lean", "masked" };

/* What step() decided for this pass, from keyed batch fields and declare-time config only (R99). */
struct Pass {
    int     path  = PATH_STOCK;
    int64_t b     = 0;       /* the bulk END, an absolute row of the step (PLAN-FIX §2) */
    int64_t s_lb  = 0;       /* a host LOWER bound of the last sequence's first row */
    bool    split = false;   /* b < n_tok: the last sequence's chunk straddles the bulk end */
    bool    alt   = false;   /* the late down GEMMs go through the alternate handles (§6.1) */
};

/* This rank's value heads of the replicated correction (kva_declare.h's decl_correction): a row
 * slice of the weight, which the issue path narrows and bounds-checks like a buffer slice
 * (radiance core/runtime/issue.cpp:661-698). */
inline RadOperand correction_heads(const qwen4exp_fp8::Model& m, rad_weight w) {
    const GdnFP8::Config& g = m.gcfg;
    RadOperand o = RAD_W(w);
    o.offset = (int64_t)m.g.rank * g.n_head_v * g.head_v * g.head_k;
    o.rows   = g.n_head_v;
    return o;
}

/* A group's slot row for the step's LAST sequence (index row n_seq-1, one row): its own
 * state_index at its own pitch. The approximated sequence is always the last entry (n_ahead > 0
 * means it is a non-final prefill chunk, radiance core/sched/batch.cpp:1066-1080). */
inline RadOperand last_slot(const RadBatch* batch, rad_kvgroup g) {
    const RadKVGroupBatch* kb = kv_batch(batch, g);
    const int64_t pitch = kb && kb->state_index_pitch > 0 ? kb->state_index_pitch : 1;
    return praw2(kb ? kb->state_index + (batch->n_seq - 1) * pitch : nullptr, RAD_I32, 1, pitch);
}

/* THE +st CORRECTION (Stage 4, PLAN D7) on the last sequence, M = 1: `undo` before the layer's
 * scan takes back what the previous approximate chunk end added (nothing at the first: the slot is
 * zeroed at admission); `apply` at the bulk end adds alpha*C (times rho in quality) and records the
 * scale. ND only in quality (rho = 1 in speed). `bounds` on the masked path: a step whose last
 * sequence has no bulk row leaves the correction as it was (notes/impl.md §1). */
inline void correct(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li, rad_op op,
                    const RadBatch* batch, bool masked) {
    if (!op) return;
    const int L = (int)li;
    RAD_ISSUE_N(c, op, 1,
                kv_cache(m.kv_state, L), last_slot(batch, m.kv_state),
                kv_cache(k.kv_applied, L), last_slot(batch, k.kv_applied),
                correction_heads(m, k.st[(size_t)li]),
                k.kv_rho ? kv_cache(k.kv_rho, L) : RAD_NONE,
                k.kv_rho ? last_slot(batch, k.kv_rho) : RAD_NONE,
                masked ? brows(k.b_bounds, 2) : RAD_NONE);
}

/* The decay sums of the rows the mask approximated, per head, over rows [s, rows) of the step
 * (kva_rho_update; s = bounds[0] on the device). */
inline void decay_sums(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch, int64_t rows) {
    const rad_op op = k.op_rho[(size_t)li];
    if (!op) return;
    const GdnFP8& d = m.layers[(size_t)li].gdn;
    RAD_ISSUE_N(c, op, rows, bcol(d.w.ab, 0, d.cfg.n_head_v, rows), brows(k.b_mask, rows),
                RAD_W(d.w_a_log), RAD_W(d.w_dt_bias), kv_cache(k.kv_rho, (int)li),
                last_slot(batch, k.kv_rho), brows(k.b_bounds, 1));
}

/* RADIANCE_KVA_CAPTURE_STATE: layer li's state slot of this step's one sequence, copied by
 * kva_state_read into the plugin's buffer and from there to the host. Debug: synchronises. */
inline void read_state(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch, StateDump* sd) {
    if (!sd || !k.op_state_read) return;
    const GdnFP8::Config& g = m.gcfg;
    const int64_t n = g.n_head_v * g.head_v * g.head_k;
    RAD_ISSUE_N(c, k.op_state_read, 1, kv_cache(m.kv_state, (int)li), last_slot(batch, m.kv_state),
                brows(k.b_state, 1));
    const size_t at = sd->data.size();
    sd->data.resize(at + (size_t)n);
    if (!dump_read(c, sd->data.data() + at, rad_buf_ptr(c, k.b_state), n * 4)) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_CAPTURE_STATE: device read failed\n");
        sd->data.resize(at);
        return;
    }
    sd->layers.push_back((int)li);
}

/* ---------------------------------------------------------------- the delta net, masked */

/* rad_block_gdn_fp8.h:450-475, verbatim: the decode half over sequences [0, D), tokens [0, DT). */
inline void gdn_decode_half(RadCtx* c, const GdnFP8& d, const RadBatch* batch, int64_t D, int64_t DT) {
    if (D <= 0) return;
    const RadKVGroupBatch* st = kv_batch(batch, d.kv_state);
    const RadKVGroupBatch* cv = kv_batch(batch, d.kv_conv);
    const int64_t st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* st_idx = st ? st->state_index : nullptr;
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;
    const int64_t H = d.cfg.n_head_v, conv_dim = d.cfg.conv_dim(), v_dim = d.cfg.v_dim();
    const int32_t qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode : batch->max_q_len;
    const auto& w = d.w;
    if (d.op_cr && qd <= d.cr_rows) {
        RAD_ISSUE_N(c, d.op_cr, qd,
                    bcol(w.in.x, 0, conv_dim, DT), RAD_W(d.w_conv), RAD_NONE, kv_cache(d.kv_conv, d.layer),
                    praw(cv_idx, RAD_I32, D), bcol(w.ab, 0, H, DT), bcol(w.ab, H, H, DT),
                    RAD_W(d.w_a_log), RAD_W(d.w_dt_bias), kv_cache(d.kv_state, d.layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    bcol(w.in.x, conv_dim, v_dim, DT), RAD_W(d.w_out_norm), brows(w.o.x, DT),
                    w.o.cq() ? brows(w.o.cq(), DT) : RAD_NONE, w.o.cq() ? brows(w.o.cs(), DT) : RAD_NONE);
        return;
    }
    RAD_ISSUE_N(c, d.op_conv_update, qd,
                bcol(w.in.x, 0, conv_dim, DT), RAD_W(d.w_conv), RAD_NONE, kv_cache(d.kv_conv, d.layer),
                praw(cv_idx, RAD_I32, D), praw(batch->num_accepted, RAD_I32, D),
                praw(batch->cu_seqlens, RAD_I32, D + 1), brows(w.q, DT), brows(w.k, DT), brows(w.v, DT));
    RAD_ISSUE_N(c, d.op_recur, qd,
                brows(w.q, DT), brows(w.k, DT), brows(w.v, DT), bcol(w.ab, 0, H, DT), bcol(w.ab, H, H, DT),
                RAD_W(d.w_a_log), RAD_W(d.w_dt_bias), kv_cache(d.kv_state, d.layer),
                praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                praw(batch->cu_seqlens, RAD_I32, D + 1),
                bcol(w.in.x, conv_dim, v_dim, DT), RAD_W(d.w_out_norm), brows(w.o.x, DT),
                w.o.cq() ? brows(w.o.cq(), DT) : RAD_NONE, w.o.cq() ? brows(w.o.cs(), DT) : RAD_NONE);
}

/* rad_block_gdn_fp8.h:489-494: the chunked scan over the sequences `cu` names, whose state rows
 * are `st` (n entries). `cu` is a raw batch pointer or the plugin's `bounds`. */
inline void gdn_scan_over(RadCtx* c, const GdnFP8& d, int64_t T, RadOperand cu, RadOperand st) {
    const auto& w = d.w;
    RAD_ISSUE(c, d.op_scan, brows(w.q, T), brows(w.k, T), brows(w.v, T), brows(w.kkt, T),
              brows(w.gdec, T), brows(w.beta, T), kv_cache(d.kv_state, d.layer), cu,
              brows(w.o.x, T), kv_cache(d.kv_state, d.layer), st);
}

/* rad_block_gdn_fp8.h:477-488: conv window and kkt, ONCE over every prefill sequence -- both
 * anchor their 64-row tiles at each sequence's start, which a split at b - s = 0 mod 64 keeps. */
inline void gdn_prefill_front(RadCtx* c, const GdnFP8& d, const RadBatch* batch, int64_t D) {
    const RadKVGroupBatch* cv = kv_batch(batch, d.kv_conv);
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;
    const int64_t T = batch->n_tok, P = batch->n_seq - D, H = d.cfg.n_head_v;
    const auto& w = d.w;
    RAD_ISSUE(c, d.op_conv_prep,
              bcol(w.in.x, 0, d.cfg.conv_dim(), T), RAD_W(d.w_conv), RAD_NONE, RAD_W(d.w_a_log),
              RAD_W(d.w_dt_bias), kv_cache(d.kv_conv, d.layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
              bcol(w.ab, 0, H, T), bcol(w.ab, H, H, T), praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
              praw(batch->ctx_lens + D, RAD_I32, P),
              brows(w.q, T), brows(w.k, T), brows(w.v, T), brows(w.gdec, T), brows(w.beta, T));
    RAD_ISSUE(c, d.op_kkt, brows(w.k, T), brows(w.beta, T), brows(w.gdec, T),
              praw(batch->cu_seqlens + D, RAD_I32, P + 1), brows(w.kkt, T));
}

/* The prefill half's scans with the last sequence corrected (PLAN-FIX §4, DD-C): the other prefill
 * sequences as stock; then the last one -- split at the bulk end (scan [s, b), rho, apply, scan
 * [b, e)) or, for a whole bulk chunk and for STRADDLE=end, one scan with rho and apply after it. */
inline void gdn_prefill_scans(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                              const RadBatch* batch, const Pass& p, StateDump* sd, int64_t D) {
    const GdnFP8& d = m.layers[(size_t)li].gdn;
    const RadKVGroupBatch* st = kv_batch(batch, d.kv_state);
    const int64_t st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* st_idx = st ? st->state_index : nullptr;
    const int64_t T = batch->n_tok, S = batch->n_seq, P = S - D;
    if (P > 1)
        gdn_scan_over(c, d, T, praw(batch->cu_seqlens + D, RAD_I32, P),
                      praw2(st_idx ? st_idx + D * st_w : nullptr, RAD_I32, P - 1, st_w));
    const RadOperand last = praw2(st_idx ? st_idx + (S - 1) * st_w : nullptr, RAD_I32, 1, st_w);
    const bool split = p.split && k.cfg.straddle == STRADDLE_SPLIT;
    gdn_scan_over(c, d, T, split ? brow_slice(k.b_bounds, 0, 2, 1)
                                 : praw(batch->cu_seqlens + S - 1, RAD_I32, 2), last);
    read_state(c, k, m, li, batch, sd);   /* before the apply: S_pred, uncorrected */
    decay_sums(c, k, m, li, batch, split || !p.split ? p.b : T);
    correct(c, k, m, li, k.op_apply[(size_t)li], batch, true);
    if (split) gdn_scan_over(c, d, T, brow_slice(k.b_bounds, 2, 2, 1), last);
}

/* GdnFP8::step (rad_block_gdn_fp8.h:397-514) with the last sequence's scan corrected. Its block
 * input is normed by the caller (ext_in, check_fill), so there is no norm here. */
inline void gdn_masked(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch, const Pass& p, StateDump* sd) {
    const GdnFP8& d = m.layers[(size_t)li].gdn;
    const int64_t T = batch->n_tok, v_dim = d.cfg.v_dim(), conv_dim = d.cfg.conv_dim();
    d.in.step(c, d.w.h, d.w.in.x, T);
    RAD_ISSUE_N(c, d.op_ab, T, brows(d.w.h.x, T), RAD_W(d.w_ab), brows(d.w.ab, T));
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    gdn_decode_half(c, d, batch, D, DT);
    const int64_t PT = T - DT;
    correct(c, k, m, li, k.op_undo[(size_t)li], batch, true);
    gdn_prefill_front(c, d, batch, D);
    gdn_prefill_scans(c, k, m, li, batch, p, sd, D);
    RAD_ISSUE_N(c, d.op_gnorm, PT, brow_slice(d.w.o.x, DT, PT, v_dim),
                bcol_at(d.w.in.x, DT, conv_dim + v_dim, conv_dim, v_dim, PT), RAD_W(d.w_out_norm),
                brow_slice(d.w.o.x, DT, PT, v_dim));
    d.q_o.step(c, T, DT, PT);
    d.out.step(c, d.w.o, d.w.h.x, T);
    if (d.op_ar && !ar_taken(d.g, T, d.ar_out, d.ar_out_take))
        RAD_ISSUE_N(c, d.op_ar, T * d.g.n_embd, brows(d.w.h.x, T), RAD_NONE);
    if (d.op_add) RAD_ISSUE(c, d.op_add, brows(d.w.x, T), brows(d.w.h.x, T), brows(d.w.x, T));
}

/* ---------------------------------------------------------------- the layer */

/* The projected block input for the bulk superset [s_lb, b): the projector over the layer-S stream
 * h_S, its codes from the plugin's own quantiser (never re-quantised from bf16, so exact rows keep
 * the connection read's bytes), then kva_select puts the marked rows over `x` and its codes. */
inline void project_masked(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                           const Pass& p, int64_t T) {
    const int64_t r0 = p.s_lb, rows = p.b - p.s_lb, n = m.g.n_embd, wide = m.hccfg.hc * n;
    const ActFP8& x = m.a_x;
    RAD_ISSUE_N(c, k.op_proj[(size_t)li], rows, brow_slice(k.b_hs, r0, rows, wide),
                RAD_W(k.proj_w[(size_t)li]), RAD_W(k.proj_b[(size_t)li]), RAD_NONE,
                brow_slice(k.xp.x, r0, rows, n));
    if (k.quant.op)
        RAD_ISSUE_N(c, k.quant.op, rows, brow_slice(k.xp.x, r0, rows, n),
                    brow_slice(k.xp.cq(), r0, rows, n), brow_slice(k.xp.cs(), r0, rows, n / RAD_FP8_BLOCK));
    const bool codes = k.xp.cq() != 0;
    RAD_ISSUE_N(c, k.op_select, T, brows(k.b_mask, T), brows(k.xp.x, T),
                codes ? brows(k.xp.cq(), T) : RAD_NONE, codes ? brows(k.xp.cs(), T) : RAD_NONE,
                brows(x.x, T), codes ? brows(x.cq(), T) : RAD_NONE, codes ? brows(x.cs(), T) : RAD_NONE);
}

/* A MASKED LATE LAYER (PLAN-FIX §8): the in-tree layer (qwen4exp_fp8.cpp:1407-1425) with the
 * projection selected in after the connection read, the delta net's last sequence corrected, and
 * the MoE issued by hand (drop the bulk rows' slots; down GEMM on the alternate when the pass says
 * so). Plumb declares no projector and no drop: it is the stock layer through the same path. */
inline void masked_layer(RadCtx* c, const Kva& k, qwen4exp_fp8::Model& m, int64_t li,
                         const RadBatch* batch, const Pass& p, StateDump* sd) {
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const int64_t T = batch->n_tok;
    l.hc_mix.read(c, T, 0, T);
    if (k.op_select) project_masked(c, k, m, li, p, T);
    if (l.full) { l.qsa.step(c, l.attn.w.h, batch); l.attn.step(c, batch); }
    else        gdn_masked(c, k, m, li, batch, p, sd);
    l.hc_mix.write(c, T, 0, T);
    dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_h, m.a_x.x);
    l.hc_ffn.read(c, T, 0, T);
    moe_layer(c, l.mlp, MoeArm{ p.alt ? k.alt_dn[(size_t)li] : l.mlp.op_dn, k.op_drop, k.b_mask },
              batch);
    l.hc_ffn.write(c, T, 0, T);
    dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_h, m.a_x.x);
}

/* ---------------------------------------------------------------- the lean fill */

/* The projector writes the block input `x` from the stream entering layer S, for every row. */
inline void project(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li, int64_t T) {
    RAD_ISSUE_N(c, k.op_proj[(size_t)li], T, brows(m.b_h, T), RAD_W(k.proj_w[(size_t)li]),
                RAD_W(k.proj_b[(size_t)li]), RAD_NONE, brows(m.a_x.x, T));
}

/* x's codes, as the connection read would have written them: QuantFP8::step without its
 * matvec-only row guard, because an int8 linear always reads the codes. */
inline void quantise(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t T) {
    const int64_t n = m.g.n_embd;
    if (k.quant.op)
        RAD_ISSUE(c, k.quant.op, brow_slice(m.a_x.x, 0, T, n), brow_slice(m.a_x.cq(), 0, T, n),
                  brow_slice(m.a_x.cs(), 0, T, n / RAD_FP8_BLOCK));
}

/* A LEAN late layer (speed, every row bulk): the projection and its codes, then only the
 * cache-writing pieces. No connection read or write and no MoE: `b_h` stays the layer-S stream for
 * every projector. */
inline void fill_layer(RadCtx* c, const Kva& k, qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch, StateDump* sd) {
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const int64_t T = batch->n_tok;
    project(c, k, m, li, T);
    quantise(c, k, m, T);
    if (l.full) {
        qsa_keys(c, l.qsa, l.attn.w.h, batch);
        attn_kv(c, l.attn, batch);
    } else {
        gdn_project(c, l.gdn, T);
        correct(c, k, m, li, k.op_undo[(size_t)li], batch, false);
        gdn_scan(c, l.gdn, batch);
        read_state(c, k, m, li, batch, sd);
        correct(c, k, m, li, k.op_apply[(size_t)li], batch, false);
    }
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_LAYER_H */
