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
 * the MoE issued by hand (the bulk rows' slots dropped). Plumb declares no projector and no drop: it
 * is the stock layer through the same path. */
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
    moe_layer(c, l.mlp, MoeArm{ k.op_drop, k.b_mask, {} }, batch);
    l.hc_ffn.write(c, T, 0, T);
    dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_h, m.a_x.x);
}

/* ---------------------------------------------------------------- the straddle: tail-only blocks */

/* The bulk rows' block input, lean style: the projector over the layer-S stream rows [0, b) -- b_h's
 * bulk rows stay that stream, since only the tail rows are written from here on -- into `x` and its
 * codes. */
inline void project_bulk(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li, int64_t b) {
    const int64_t n = m.g.n_embd;
    RAD_ISSUE_N(c, k.op_proj[(size_t)li], b, brows(m.b_h, b), RAD_W(k.proj_w[(size_t)li]),
                RAD_W(k.proj_b[(size_t)li]), RAD_NONE, brows(m.a_x.x, b));
    if (k.quant.op)
        RAD_ISSUE_N(c, k.quant.op, b, brows(m.a_x.x, b), brows(m.a_x.cq(), b),
                    brow_slice(m.a_x.cs(), 0, b, n / RAD_FP8_BLOCK));
}

/* rad_block_gdn_fp8.h:497-513 over the tail rows [r0, T): the delta net's output for them alone. */
inline void gdn_tail_rows(RadCtx* c, const GdnFP8& d, int64_t T, int64_t r0) {
    const int64_t rows = T - r0, n = d.g.n_embd, conv_dim = d.cfg.conv_dim(), v_dim = d.cfg.v_dim();
    RAD_ISSUE_N(c, d.op_gnorm, rows, brow_slice(d.w.o.x, r0, rows, v_dim),
                bcol_at(d.w.in.x, r0, conv_dim + v_dim, conv_dim, v_dim, rows), RAD_W(d.w_out_norm),
                brow_slice(d.w.o.x, r0, rows, v_dim));
    d.q_o.step(c, T, r0, rows);
    d.out.step(c, d.w.o, d.w.h.x, T, r0, rows);
    if (d.op_ar && !ar_taken(d.g, T, d.ar_out, d.ar_out_take))
        RAD_ISSUE_N(c, d.op_ar, rows * n, brow_slice(d.w.h.x, r0, rows, n), RAD_NONE);
    if (d.op_add)
        RAD_ISSUE_N(c, d.op_add, rows, brow_slice(d.w.x, r0, rows, n), brow_slice(d.w.h.x, r0, rows, n),
                    brow_slice(d.w.x, r0, rows, n));
}

/* The delta net of a straddling chunk: projections, conv window and the scans over every row (the
 * state needs them all), split at b around the correction; the output for the tail rows only. */
inline void gdn_straddle(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                         const RadBatch* batch, const Pass& p, StateDump* sd) {
    const GdnFP8& d = m.layers[(size_t)li].gdn;
    gdn_project(c, d, batch->n_tok);
    correct(c, k, m, li, k.op_undo[(size_t)li], batch, true);
    gdn_prefill_front(c, d, batch, 0);
    gdn_prefill_scans(c, k, m, li, batch, p, sd, 0);
    gdn_tail_rows(c, d, batch->n_tok, p.b);
}

/* The attention of a straddling chunk (rad_block_attn_gated_fp8.h:442-563): the indexer whole (block
 * keys AND every row's selection -- both cheap), K/V of every row, the query path over every row (an
 * M-RoPE position plane cannot be column-sliced), then the attention, gate and output projection
 * for the tail rows only, through the per-row sparse gated form, whose rows are independent
 * queries (:487-519; the plan admits this path only when every late layer takes that form). */
inline void attn_straddle(RadCtx* c, const qwen4exp_fp8::Layer& l, const RadBatch* batch, int64_t r0) {
    const AttnGatedFP8& a = l.attn;
    const int64_t T = batch->n_tok, rows = T - r0, n = a.g.n_embd, qw = a.g.q_dim(), hd = a.g.head_dim;
    l.qsa.step(c, a.w.h, batch);
    attn_kv(c, a, batch);
    a.qg.step(c, a.w.h, a.w.qg, T);
    RAD_ISSUE_N(c, a.op_q_norm, T * a.g.n_head, bcol(a.w.qg, 0, hd, T), RAD_W(a.w_q_norm), brows(a.w.q, T));
    RAD_ISSUE(c, a.op_rope_q, brows(a.w.q, T), rope_posmc(a.g, batch, T));
    const int64_t chunk = a.qsa_gq_rows();
    for (int64_t off = r0; off < T; off += chunk) {
        const int64_t r = (T - off) < chunk ? (T - off) : chunk;
        RAD_ISSUE_N(c, a.op_attn_gq, 1, brow_slice(a.w.q, off, r, qw), kv_cache(a.kv, a.layer),
                    brow_slice(a.qsa_sel, off, r, a.qsa_topk + 1), brow_slice(a.qsa_sequ, off, r, 1),
                    RAD_NONE, RAD_NONE, RAD_NONE, brow_slice(a.w.attn.x, off, r, qw),
                    bcol_at(a.w.qg, off, a.g.n_head * 2 * hd, hd, hd, r),
                    brow_slice(a.w.attn.cq(), off, r, qw), brow_slice(a.w.attn.cs(), off, r, fp8_blocks(qw)));
    }
    a.o.step(c, a.w.attn, a.w.h.x, T, r0, rows);
    if (a.op_ar && !ar_taken(a.g, T, a.ar_out, a.ar_out_take))
        RAD_ISSUE_N(c, a.op_ar, rows * n, brow_slice(a.w.h.x, r0, rows, n), RAD_NONE);
    if (a.op_add)
        RAD_ISSUE_N(c, a.op_add, rows, brow_slice(a.w.x, r0, rows, n), brow_slice(a.w.h.x, r0, rows, n),
                    brow_slice(a.w.x, r0, rows, n));
}

/* A STRADDLING LATE LAYER (A.1, speed): the bulk rows [0, b) get the lean pieces (projection, K/V,
 * indexer keys, the delta net's recurrence), the tail rows [b, n) the whole in-tree layer -- their
 * connection read, block output, connection write, feed-forward read, MoE and write -- issued over
 * that row range with the in-tree helpers' own r0/rows (rad_block_hc.h:360-397, rad_fp8.h:915,
 * MoeFP8::pass). Nothing reads a bulk row's late block output, so none is computed. */
inline void straddle_layer(RadCtx* c, const Kva& k, qwen4exp_fp8::Model& m, int64_t li,
                           const RadBatch* batch, const Pass& p, StateDump* sd) {
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const int64_t T = batch->n_tok, r0 = p.b, rows = T - p.b;
    l.hc_mix.read(c, T, r0, rows);
    project_bulk(c, k, m, li, p.b);
    if (l.full) attn_straddle(c, l, batch, r0);
    else        gdn_straddle(c, k, m, li, batch, p, sd);
    l.hc_mix.write(c, T, r0, rows);
    dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_h, m.a_x.x);
    l.hc_ffn.read(c, T, r0, rows);
    moe_layer(c, l.mlp, MoeArm{}, batch, r0);
    l.hc_ffn.write(c, T, r0, rows);
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
