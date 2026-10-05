/* kva_layer.h -- one late layer of an approximate pass: the masked layer (PLAN-FIX §3, §4, §8) and
 * the lean fill (the pure all-bulk case). The blocks' own issues and the correction spliced into them
 * are the adapter's (its conn / late_block / ffn hooks, kva_adapter.h); what stays here is the
 * projection, the ring and the drivers that order them.
 *
 * THE MASKED LAYER runs every in-tree op over all n_tok rows with the stock handles and lets the
 * DEVICE decide which rows use the projection: the connection read writes the exact block input for
 * every row, kva_select overwrites the rows the mask marks with the projected input and its codes,
 * the block runs whole, and the MoE drops the marked rows' routing slots. Decode rows and every
 * other sequence are therefore computed exactly as stock computes them, at stock GEMM shapes.
 */
#ifndef KVA_LAYER_H
#define KVA_LAYER_H

namespace kva {

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

/* ---------------------------------------------------------------- the staging ring (DD-L) */

/* ONE SLOT (Stage E): block li's map, host block -> the VRAM slot, on the second lane, issued right after
 * block li - 1's GEMM -- the slot's last reader, which lane 0 has already been handed: lane 1 waits for
 * lane 0 first. Each copy overlaps the rest of the layer after its projector (attention or the delta net,
 * the MoE). A second slot would let it overlap the GEMM too, for one more block of VRAM a card, which every
 * request pays in resident experts (notes/stagee.md §12-§14). */
inline void ring_copy(RadCtx* c, const Kva& k, int64_t li) {
    rad_lane_join(c, 0, 1);
    rad_lane(c, 1);
    RAD_ISSUE_N(c, k.op_ring, k.ring_src[(size_t)li].rows, k.ring_src[(size_t)li], k.ring_dst[(size_t)li]);
    rad_lane(c, 0);
}

/* Before block li's GEMM: lane 0 waits for its copy. Nothing without the ring. */
inline void ring_wait(RadCtx* c, const Kva& k) {
    if (k.op_ring) rad_lane_join(c, 1, 0);
}

/* After block li's last reader (its GEMM, and the bias add on the int8 path): block li + 1's copy into the
 * same slot -- after the last layer, the final map's blocks (MTP). */
inline void ring_after(RadCtx* c, const Kva& k, int64_t li) {
    if (k.op_ring && li + 1 < k.ring_end) ring_copy(c, k, li + 1);
}

/* THE PROJECTOR over rows [r0, r0 + rows) of the layer-S stream `src` into `dst`: the bf16 GEMM, or
 * with an int8 folder (R79) the stream's int8 codes, the int8 GEMM and the bias. Every late layer
 * projects the SAME rows of the same stream (h_S on the masked path; b_h's bulk rows, which nothing
 * writes, on the lean and straddle paths), so the codes are made once a pass, at layer S. */
inline void project_rows(RadCtx* c, const Kva& k, int64_t li, rad_buf src, int64_t r0, int64_t rows,
                         int64_t wide, RadOperand dst) {
    const size_t L = (size_t)li;
    ring_wait(c, k);
    if (!k.int8) {
        RAD_ISSUE_N(c, k.op_proj[L], rows, brow_slice(src, r0, rows, wide), k.proj_w[L], k.proj_b[L], RAD_NONE, dst);
        ring_after(c, k, li);
        return;
    }
    const int64_t groups = wide / kI8Group;
    if (li == k.split)
        RAD_ISSUE_N(c, k.op_quant8, rows, brow_slice(src, r0, rows, wide), brow_slice(k.b_q8, r0, rows, wide),
                    brow_slice(k.b_s8, r0, rows, groups));
    RAD_ISSUE_N(c, k.op_proj[L], rows, brow_slice(k.b_q8, r0, rows, wide), brow_slice(k.b_s8, r0, rows, groups),
                k.proj_w[L], k.proj_s[L], dst, RAD_NONE, RAD_NONE);
    RAD_ISSUE_N(c, k.op_bias, rows, dst, k.proj_b[L], dst);
    ring_after(c, k, li);
}

/* The projected block input for the bulk superset [s_lb, b): the projector over the layer-S stream
 * h_S, its codes from the plugin's own quantiser (never re-quantised from bf16, so exact rows keep
 * the connection read's bytes), then kva_select puts the marked rows over `x` and its codes. */
inline void project_masked(RadCtx* c, const Kva& k, int64_t li, const Pass& p, int64_t T) {
    const KvaAdapter& a = k.ad;
    const int64_t r0 = p.s_lb, rows = p.b - p.s_lb, n = a.n_embd;
    /* int8 without the MTP map keeps no h_S: the codes are made at layer S from b_h, which still holds the
     * layer-S stream there (its connection write comes after this), and every later layer reads the codes */
    project_rows(c, k, li, k.b_hs ? k.b_hs : a.buf_stream, r0, rows, a.wide, brow_slice(k.xp.x, r0, rows, n));
    if (k.quant.op)
        RAD_ISSUE_N(c, k.quant.op, rows, brow_slice(k.xp.x, r0, rows, n),
                    brow_slice(k.xp.cq(), r0, rows, n), brow_slice(k.xp.cs(), r0, rows, n / RAD_FP8_BLOCK));
    const bool codes = k.xp.cq() != 0;
    RAD_ISSUE_N(c, k.op_select, T, brows(k.b_mask, T), brows(k.xp.x, T),
                codes ? brows(k.xp.cq(), T) : RAD_NONE, codes ? brows(k.xp.cs(), T) : RAD_NONE,
                brows(a.buf_x, T), codes ? brows(a.buf_x_q, T) : RAD_NONE, codes ? brows(a.buf_x_s, T) : RAD_NONE);
}

/* A MASKED LATE LAYER (PLAN-FIX §8): the model's layer in its own order (the adapter's conn /
 * late_block / ffn hooks) with the projection selected in after the connection read, the recurrent
 * block's last sequence corrected, and the FFN's bulk-row slots dropped. Plumb declares no projector
 * and no drop: it is the stock layer through the same path. */
inline void masked_layer(RadCtx* c, const Kva& k, int64_t li, const RadBatch* batch, const Pass& p,
                         StateDump* sd) {
    const KvaAdapter& a = k.ad;
    const int64_t T = batch->n_tok;
    a.conn(c, li, false, false, T, 0, T);
    if (k.op_select) project_masked(c, k, li, p, T);
    a.late_block(c, k, li, batch, p, sd, PATH_MASKED, 0, T);
    a.conn(c, li, false, true, T, 0, T);
    a.conn(c, li, true, false, T, 0, T);
    a.ffn(c, k, li, batch, 0, -1, k.op_drop, k.b_mask);
    a.conn(c, li, true, true, T, 0, T);
}

/* ---------------------------------------------------------------- the straddle: tail-only blocks */

/* The bulk rows' block input, lean style: the projector over the layer-S stream rows [0, b) -- b_h's
 * bulk rows stay that stream, since only the tail rows are written from here on -- into `x` and its
 * codes. */
inline void project_bulk(RadCtx* c, const Kva& k, int64_t li, int64_t b) {
    const KvaAdapter& a = k.ad;
    const int64_t n = a.n_embd;
    project_rows(c, k, li, a.buf_stream, 0, b, a.wide, brows(a.buf_x, b));
    if (k.quant.op)
        RAD_ISSUE_N(c, k.quant.op, b, brows(a.buf_x, b), brows(a.buf_x_q, b),
                    brow_slice(a.buf_x_s, 0, b, n / RAD_FP8_BLOCK));
}

/* A STRADDLING LATE LAYER (A.1, speed): the bulk rows [0, b) get the lean pieces (projection and the
 * blocks' cache writers), the tail rows [b, n) the whole in-tree layer -- their connection read, block
 * output, connection write, feed-forward read, FFN and write -- issued over that row range by the
 * adapter's hooks with the in-tree helpers' own r0/rows. Nothing reads a bulk row's late block output,
 * so none is computed. */
inline void straddle_layer(RadCtx* c, const Kva& k, int64_t li, const RadBatch* batch, const Pass& p,
                           StateDump* sd) {
    const KvaAdapter& a = k.ad;
    const int64_t T = batch->n_tok, r0 = p.b, rows = T - p.b;
    a.conn(c, li, false, false, T, r0, rows);
    project_bulk(c, k, li, p.b);
    a.late_block(c, k, li, batch, p, sd, PATH_STRADDLE, r0, rows);
    a.conn(c, li, false, true, T, r0, rows);
    a.conn(c, li, true, false, T, r0, rows);
    a.ffn(c, k, li, batch, r0, -1, 0, 0);
    a.conn(c, li, true, true, T, r0, rows);
}

/* ---------------------------------------------------------------- speed beside decoders */

/* The bulk rows' block input [r0, T), lean style: the projector over the layer-S stream (b_h's rows
 * there stay that stream: only the decoder rows are written from here on) into `x` and its codes --
 * through project_rows, so the ring and an int8 folder serve this path as every other. */
inline void project_beside(RadCtx* c, const Kva& k, int64_t li, int64_t r0, int64_t T) {
    const KvaAdapter& a = k.ad;
    const int64_t rows = T - r0, n = a.n_embd;
    project_rows(c, k, li, a.buf_stream, r0, rows, a.wide, brow_slice(a.buf_x, r0, rows, n));
    if (k.quant.op)
        RAD_ISSUE_N(c, k.quant.op, rows, brow_slice(a.buf_x, r0, rows, n), brow_slice(a.buf_x_q, r0, rows, n),
                    brow_slice(a.buf_x_s, r0, rows, n / RAD_FP8_BLOCK));
}

/* A LATE LAYER OF SPEED BESIDE DECODERS (Stage B, Dylan's decision 2026-10-05): the bulk rows [DT, n)
 * get the lean pieces; the decoder rows [0, DT) the whole in-tree layer over that range -- connection
 * read/write, block, feed-forward read, MoE, write -- with the in-tree helpers' own r0/rows. The
 * decoders' dense GEMMs run at M = DT, as in a decode-only step, so their bytes may differ from a
 * 2,048-row step's within ident.sh's ksplit-from-M class. Nothing reads a bulk row's late output. */
inline void decoders_layer(RadCtx* c, const Kva& k, int64_t li, const RadBatch* batch, const Pass& p,
                           StateDump* sd) {
    const KvaAdapter& a = k.ad;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t T = batch->n_tok;
    a.conn(c, li, false, false, T, 0, DT);
    project_beside(c, k, li, DT, T);
    a.late_block(c, k, li, batch, p, sd, PATH_DECODERS, 0, DT);
    a.conn(c, li, false, true, T, 0, DT);
    a.conn(c, li, true, false, T, 0, DT);
    a.ffn(c, k, li, batch, 0, DT, 0, 0);
    a.conn(c, li, true, true, T, 0, DT);
}

/* ---------------------------------------------------------------- the lean fill */

/* The projector writes the block input `x` from the stream entering layer S, for every row. */
inline void project(RadCtx* c, const Kva& k, int64_t li, int64_t T) {
    project_rows(c, k, li, k.ad.buf_stream, 0, T, k.ad.wide, brows(k.ad.buf_x, T));
}

/* x's codes, as the connection read would have written them: QuantFP8::step without its
 * matvec-only row guard, because an int8 linear always reads the codes. */
inline void quantise(RadCtx* c, const Kva& k, int64_t T) {
    const KvaAdapter& a = k.ad;
    const int64_t n = a.n_embd;
    if (k.quant.op)
        RAD_ISSUE(c, k.quant.op, brow_slice(a.buf_x, 0, T, n), brow_slice(a.buf_x_q, 0, T, n),
                  brow_slice(a.buf_x_s, 0, T, n / RAD_FP8_BLOCK));
}

/* A LEAN late layer (speed, every row bulk): the projection and its codes, then only the
 * cache-writing pieces. No connection read or write and no MoE: `b_h` stays the layer-S stream for
 * every projector. */
inline void fill_layer(RadCtx* c, const Kva& k, int64_t li, const RadBatch* batch, const Pass& p,
                       StateDump* sd) {
    const int64_t T = batch->n_tok;
    project(c, k, li, T);
    quantise(c, k, T);
    k.ad.late_block(c, k, li, batch, p, sd, PATH_LEAN, 0, T);
}

}  /* namespace kva */

#endif /* KVA_LAYER_H */
