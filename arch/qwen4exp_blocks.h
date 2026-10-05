/* qwen4exp_blocks.h -- the qwen4exp adapter's late-block pieces (notes/adapter-split-spec.md §1.2): the
 * delta net's and the attention's in-tree issues over a row window, with the +st correction spliced
 * into the last sequence's scan. Verbatim copies of the in-tree blocks' step() pieces (cited per
 * function, radiance 140987f), so the static oracle compares them issue for issue. The core's layer
 * drivers (kva_layer.h) call them; nothing here names the projector.
 */
#ifndef QWEN4EXP_BLOCKS_H
#define QWEN4EXP_BLOCKS_H

namespace qwen4exp_kva {

using namespace rad::arch;

/* A group's slot row for sequence `seq` of the step (one index row): its own state_index at its own
 * pitch. */
inline RadOperand slot_row(const RadBatch* batch, rad_kvgroup g, int64_t seq) {
    const RadKVGroupBatch* kb = kv_batch(batch, g);
    const int64_t pitch = kb && kb->state_index_pitch > 0 ? kb->state_index_pitch : 1;
    return praw2(kb ? kb->state_index + seq * pitch : nullptr, RAD_I32, 1, pitch);
}

/* The slot row of the step's LAST sequence (index row n_seq-1). The approximated sequence is always
 * the last entry (n_ahead > 0 means it is a non-final prefill chunk, radiance
 * core/sched/batch.cpp:1066-1080). */
inline RadOperand last_slot(const RadBatch* batch, rad_kvgroup g) {
    return slot_row(batch, g, batch->n_seq - 1);
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
                k.st[(size_t)li],   /* this rank's value heads, copied at declare */
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

/* Layer li's state slot of sequence `seq` of the step, copied by kva_state_read into the plugin's
 * buffer and appended to `out` on the host. Debug: synchronises. */
inline bool copy_state(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch, int64_t seq, std::vector<float>* out) {
    const GdnFP8::Config& g = m.gcfg;
    const int64_t n = g.n_head_v * g.head_v * g.head_k;
    RAD_ISSUE_N(c, k.op_state_read, 1, kv_cache(m.kv_state, (int)li), slot_row(batch, m.kv_state, seq),
                brows(k.b_state, 1));
    const size_t at = out->size();
    out->resize(at + (size_t)n);
    if (dump_read(c, out->data() + at, rad_buf_ptr(c, k.b_state), n * 4)) return true;
    std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_CAPTURE_STATE: device read failed\n");
    out->resize(at);
    return false;
}

/* RADIANCE_KVA_CAPTURE_STATE: layer li's state slot of this step's one sequence. */
inline void read_state(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch, StateDump* sd) {
    if (!sd || !k.op_state_read) return;
    if (copy_state(c, k, m, li, batch, batch->n_seq - 1, &sd->data)) sd->layers.push_back((int)li);
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

/* rad_block_gdn_fp8.h:505-513 over rows [r0, r0 + rows): the output projection of the normed,
 * quantised delta-net output, its all-reduce and residual add. */
inline void gdn_out_rows(RadCtx* c, const GdnFP8& d, int64_t T, int64_t r0, int64_t rows) {
    const int64_t n = d.g.n_embd;
    d.out.step(c, d.w.o, d.w.h.x, T, r0, rows);
    if (d.op_ar && !ar_taken(d.g, T, d.ar_out, d.ar_out_take))
        RAD_ISSUE_N(c, d.op_ar, rows * n, brow_slice(d.w.h.x, r0, rows, n), RAD_NONE);
    if (d.op_add)
        RAD_ISSUE_N(c, d.op_add, rows, brow_slice(d.w.x, r0, rows, n), brow_slice(d.w.h.x, r0, rows, n),
                    brow_slice(d.w.x, r0, rows, n));
}

/* rad_block_gdn_fp8.h:497-513 over the tail rows [r0, T): the delta net's output for them alone. */
inline void gdn_tail_rows(RadCtx* c, const GdnFP8& d, int64_t T, int64_t r0) {
    const int64_t rows = T - r0, conv_dim = d.cfg.conv_dim(), v_dim = d.cfg.v_dim();
    RAD_ISSUE_N(c, d.op_gnorm, rows, brow_slice(d.w.o.x, r0, rows, v_dim),
                bcol_at(d.w.in.x, r0, conv_dim + v_dim, conv_dim, v_dim, rows), RAD_W(d.w_out_norm),
                brow_slice(d.w.o.x, r0, rows, v_dim));
    d.q_o.step(c, T, r0, rows);
    gdn_out_rows(c, d, T, r0, rows);
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
 * queries (:487-519; the plan admits this path only when every late layer takes that form). The
 * rows are [r0, r0 + rows): the straddle's tail, or the decoders' rows. */
inline void attn_rows(RadCtx* c, const qwen4exp_fp8::Layer& l, const RadBatch* batch, int64_t r0,
                      int64_t rows) {
    const AttnGatedFP8& a = l.attn;
    const int64_t T = batch->n_tok, end = r0 + rows, n = a.g.n_embd, qw = a.g.q_dim(), hd = a.g.head_dim;
    l.qsa.step(c, a.w.h, batch);
    attn_kv(c, a, batch);
    a.qg.step(c, a.w.h, a.w.qg, T);
    RAD_ISSUE_N(c, a.op_q_norm, T * a.g.n_head, bcol(a.w.qg, 0, hd, T), RAD_W(a.w_q_norm), brows(a.w.q, T));
    RAD_ISSUE(c, a.op_rope_q, brows(a.w.q, T), rope_posmc(a.g, batch, T));
    const int64_t chunk = a.qsa_gq_rows();
    for (int64_t off = r0; off < end; off += chunk) {
        const int64_t r = (end - off) < chunk ? (end - off) : chunk;
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

/* The delta net beside decoders: projections over every row (the bulk rows' scan needs them), the
 * in-tree decode half for the decoders (rad_block_gdn_fp8.h:450-475; it writes their normed, quantised
 * output itself), the corrected scan of the one prefill sequence, and the output projection for the
 * decoder rows alone. */
inline void gdn_decoders(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                         const RadBatch* batch, const Pass& p, StateDump* sd) {
    const GdnFP8& d = m.layers[(size_t)li].gdn;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    gdn_project(c, d, batch->n_tok);
    gdn_decode_half(c, d, batch, D, DT);
    correct(c, k, m, li, k.op_undo[(size_t)li], batch, true);
    gdn_prefill_front(c, d, batch, D);
    gdn_prefill_scans(c, k, m, li, batch, p, sd, D);
    gdn_out_rows(c, d, batch->n_tok, 0, DT);
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_BLOCKS_H */
