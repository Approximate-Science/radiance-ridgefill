/* qwen4exp_fill.h -- the LEAN fill's pieces: the in-tree blocks' own op handles, issued by hand.
 *
 * The lean fill serves the one shape with no exact row at all -- a pure prefill step of ONE
 * sequence whose whole chunk is bulk, in speed mode (PLAN-FIX §3) -- and runs only the part of each
 * late block that writes a cache: the delta net's input projections and recurrence, the indexer's
 * block-key half, and the attention's K/V path. The in-tree GdnFP8 / QsaIndexer / AttnGatedFP8
 * step() functions take the whole batch and run the whole block, so the pieces are spelled out
 * here, each a verbatim copy of the in-tree lines it cites (radiance 140987f), with the SAME
 * handles and the SAME operands -- the stock wiring, including `w.h` (= the model's `x`) as the
 * input, because the projector writes its prediction there (notes/arch.md §6, the a_x decision).
 * Every other approximate shape takes the masked path (kva_layer.h), which runs the blocks whole.
 *
 * Every function assumes that one-sequence prefill step (n_seq_decode == 0, n_seq == 1); the
 * in-tree mixed-step offsets (D, DT) are therefore 0 and dropped. Each block's ops stay contiguous
 * -- transients with disjoint declared op ranges may share bytes (notes/arch.md §6).
 */
#ifndef QWEN4EXP_FILL_H
#define QWEN4EXP_FILL_H

namespace qwen4exp_kva {

using namespace rad::arch;

/* ---------------------------------------------------------------- the delta net */

/* rad_block_gdn_fp8.h:415-416 -- [q|k|v|z] and a|b from the block input. */
inline void gdn_project(RadCtx* c, const GdnFP8& d, int64_t T) {
    d.in.step(c, d.w.h, d.w.in.x, T);
    RAD_ISSUE_N(c, d.op_ab, T, brows(d.w.h.x, T), RAD_W(d.w_ab), brows(d.w.ab, T));
}

/* rad_block_gdn_fp8.h:478-494 -- the conv window and the recurrent state, which is all a filled
 * layer keeps. The scan reads and writes the state in place (h0 = ht = this layer's slot). */
inline void gdn_scan(RadCtx* c, const GdnFP8& d, const RadBatch* batch) {
    const RadKVGroupBatch* st = kv_batch(batch, d.kv_state);
    const RadKVGroupBatch* cv = kv_batch(batch, d.kv_conv);
    const int32_t* st_idx = st ? st->state_index : nullptr;
    const int64_t  st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;
    const int64_t T = batch->n_tok, P = batch->n_seq, H = d.cfg.n_head_v;
    const int64_t conv_dim = d.cfg.conv_dim();
    RAD_ISSUE(c, d.op_conv_prep,
              bcol(d.w.in.x, 0, conv_dim, T), RAD_W(d.w_conv), RAD_NONE, RAD_W(d.w_a_log),
              RAD_W(d.w_dt_bias), kv_cache(d.kv_conv, d.layer),
              praw(batch->cu_seqlens, RAD_I32, P + 1),
              bcol(d.w.ab, 0, H, T), bcol(d.w.ab, H, H, T),
              praw(cv_idx, RAD_I32, P), praw(batch->ctx_lens, RAD_I32, P),
              brows(d.w.q, T), brows(d.w.k, T), brows(d.w.v, T),
              brows(d.w.gdec, T), brows(d.w.beta, T));
    RAD_ISSUE(c, d.op_kkt,
              brows(d.w.k, T), brows(d.w.beta, T), brows(d.w.gdec, T),
              praw(batch->cu_seqlens, RAD_I32, P + 1), brows(d.w.kkt, T));
    RAD_ISSUE(c, d.op_scan,
              brows(d.w.q, T), brows(d.w.k, T), brows(d.w.v, T), brows(d.w.kkt, T),
              brows(d.w.gdec, T), brows(d.w.beta, T),
              kv_cache(d.kv_state, d.layer), praw(batch->cu_seqlens, RAD_I32, P + 1),
              brows(d.w.o.x, T), kv_cache(d.kv_state, d.layer),
              praw2(st_idx, RAD_I32, P, st_w));
}

/* ---------------------------------------------------------------- the QSA indexer */

/* rad_qsa.h:324 and :345-390 -- the projection and the block-key half: the work list, the pooled
 * block keys and the tail ring. Stateful, so a filled layer must run it (rad_qsa.h:65-71). */
inline void qsa_keys(RadCtx* c, const QsaIndexer& q, const ActFP8& in, const RadBatch* batch) {
    if (!q.on()) return;
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const int64_t nw = (q.c.heads + 1) * q.c.head_dim, koff = q.c.heads * q.c.head_dim;
    q.proj.step(c, in, q.w.qk, T);
    const RadKVGroupBatch* bkb = kv_batch(batch, q.kv_bk);
    const RadKVGroupBatch* atb = kv_batch(batch, q.c.kv_attn);
    const RadKVGroupBatch* tlb = kv_batch(batch, q.c.kv_tail);
    if (!bkb || !atb || !tlb) return;
    const int64_t tw = tlb->state_index_pitch > 0 ? tlb->state_index_pitch : 1;
    const int64_t work = (T + q.c.ratio - 1) / q.c.ratio + S;
    const bool mc = q.g.rope_mc && batch->rope_pos;
    RadOperand bpos = q.g.rope_mc ? RAD_B_COL(q.w.bpos, 0, mc ? 3 : 1) : brows(q.w.bpos, work);
    if (q.g.rope_mc) bpos.rows = work;
    RAD_ISSUE_N(c, q.op_work, T,
                bcol_at(q.w.qk, 0, nw, koff, q.c.head_dim, T),
                praw(batch->positions, RAD_I32, T), praw(batch->cu_seqlens, RAD_I32, S + 1),
                kv_cache(q.c.kv_tail, q.layer), praw2(tlb->state_index, RAD_I32, S, tw),
                praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                mc ? praw2(batch->rope_pos, RAD_I32, 3, T) : RAD_NONE,
                brows(q.w.stage, work * q.c.ratio), brows(q.w.page, work), bpos, brows(q.w.nc, S));
    RAD_ISSUE_N(c, q.op_bkey, work,
                brows(q.w.stage, work * q.c.ratio), RAD_W(q.w_kn),
                brows(q.w.page, work), bpos, kv_cache(q.kv_bk, q.layer));
    RAD_ISSUE_N(c, q.op_tail, T,
                bcol_at(q.w.qk, 0, nw, koff, q.c.head_dim, T),
                praw(batch->positions, RAD_I32, T), praw(batch->cu_seqlens, RAD_I32, S + 1),
                kv_cache(q.c.kv_tail, q.layer), praw2(tlb->state_index, RAD_I32, S, tw));
}

/* ---------------------------------------------------------------- the gated attention */

/* rad_block_attn_gated_fp8.h:452-453, :476-477, :479, :482-484 -- k and v, k's norm and rotation,
 * the paged store. The UNFUSED form: the fused prologue needs the q|gate projection, which a filled
 * layer does not compute; the two are byte-identical (the in-tree's r4d_selftest claim, :261-267),
 * and at more than qk_fuse_rows (64) tokens the stock step takes this form too. */
inline void attn_kv(RadCtx* c, const AttnGatedFP8& a, const RadBatch* batch) {
    const RadKVGroupBatch* kvb = kv_batch(batch, a.kv);
    const int64_t T = batch->n_tok;
    a.kp.step(c, a.w.h, a.w.k, T);
    a.vp.step(c, a.w.h, a.w.v, T);
    RAD_ISSUE_N(c, a.op_k_norm, T * a.g.n_head_kv, brows(a.w.k, T), RAD_W(a.w_k_norm), brows(a.w.k, T));
    RAD_ISSUE(c, a.op_rope_k, brows(a.w.k, T), rope_posmc(a.g, batch, T));
    RAD_ISSUE(c, a.op_kv_store, brows(a.w.k, T), brows(a.w.v, T),
              praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T), kv_cache(a.kv, a.layer));
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_FILL_H */
