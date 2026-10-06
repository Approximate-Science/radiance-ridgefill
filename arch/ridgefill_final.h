/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* ridgefill_final.h -- the MTP `final` map (Stage D, PLAN-FIX DD-D): what the drafting head reads for a bulk row.
 *
 * WHY. The MTP head's history pass gathers the trunk's final stream `b_h` (rad_block_mtp_hc.h:252-256). On an
 * approximate pass a bulk row's late layers never ran (lean) or ran on projected inputs (masked), so its `b_h`
 * row is not the final stream the head was trained on, and acceptance drops (R9V: 0.895x tokens a step). The
 * fitted `final` map predicts that stream from the layer-S stream: y = h_S W^T + bias, [hc*n] from [hc*n].
 *
 * HOW, STREAMED LIKE THE PROJECTOR. The map (wide x wide bf16, wide = hc*n; 200 MiB at hc*n = 10240) lives in
 * the host block as hc row blocks
 * of [n + 1, hc*n] -- a bf16 projector block's shape -- and rides the staging ring after the last late layer
 * (ridgefill_projector.h plan_final; ring indices n_layer .. n_layer + hc). Block i's GEMM writes columns i*n .. i*n + n
 * of `ridgefill_final`. Then the predicted rows go into `b_h` BEFORE the epilogue's connection read: through the
 * device mask (ridgefill_select, mask 1 rows only -- exact rows, decode rows and other sequences keep their own
 * stream) on the masked and straddle paths, by a plain row copy on the lean path, where every row is bulk.
 *
 * Declared only with RADIANCE_RIDGEFILL_FINAL=on (default off: not in the release, R70's +1.8% drafted tokens a step is
 * not worth its VRAM), MTP on (max_spec > 0), a projecting mode and the folder holding the map. Cost: hc more
 * ring copies and GEMMs an approximate pass -- +210 MB a pass a rank over the link (the bf16 projector moves 1.26 GB, int8 0.64 GB) and +17% of the projector's MACs -- 200 MiB
 * more host-mapped memory a rank, the ring's one slot sized for a bf16 block even with an int8 folder (50 MiB),
 * h_S kept even with an int8 folder, and `ridgefill_final` [max_tok, hc*n] in the arena.
 */
#ifndef RIDGEFILL_FINAL_H
#define RIDGEFILL_FINAL_H

namespace ridgefill {

/* ridgefill_layer.h: lane 0 waits for the slot's block; after its GEMM, lane 1 copies block j + 1 into the slot. */
inline void ring_wait(RadCtx* c, const RidgeFill& k);
inline void ring_after(RadCtx* c, const RidgeFill& k, int64_t li);

/* The predicted final stream's buffer and its GEMM (the engine's gemm_nt_bias, forwarded by ridgefill.so, the map
 * block as an IN operand). Every buffer it touches takes the whole program (rad_buf_concurrent). */
static const char* decl_final(RadBuilder* b, const RadBuildCtx* ctx, RidgeFill& k) {
    const RidgeFillAdapter& a = k.ad;
    const int64_t n = a.n_embd, wide = a.wide;
    k.b_final = decl_b(b, k.nm.f("ridgefill_final"), a.act_dtype, {ctx->max_tok, wide});
    if (!k.b_final || rad_buf_concurrent(b, k.b_final) < 0) return "a buffer";
    k.op_final = rw(b, RAD_OP(b, "ridgefill_gemm_nt_bias",
                              RAD_PARAMS(RAD_RANGE("M", 1, ctx->max_tok), RAD_INT("N", n), RAD_INT("K", wide),
                                         RAD_STR("dtype", a.dtype)),
                              RAD_NOWEIGHTS),
                    {k.b_hs, a.buf_stream}, {k.b_final});
    return k.op_final ? nullptr : "ridgefill_gemm_nt_bias (the final map)";
}

/* After the last late layer, before the epilogue: the final map over the bulk rows' layer-S stream (h_S on the
 * masked path; b_h's bulk rows, which nothing wrote, on the lean and straddle paths), then into b_h. */
inline void final_stream(RadCtx* c, const RidgeFill& k, const RadBatch* batch, const Pass& p) {
    if (!k.op_final) return;
    const RidgeFillAdapter& a = k.ad;
    const int64_t n = a.n_embd, wide = a.wide, hc = wide / n, T = batch->n_tok;   /* hc stream copies */
    const bool lean = p.path == PATH_LEAN, masked = p.path == PATH_MASKED;
    const rad_buf src = masked ? k.b_hs : a.buf_stream;
    const int64_t r0 = masked ? p.s_lb : 0, rows = lean ? T : p.b - r0;
    for (int64_t i = 0; i < hc; ++i) {
        ring_wait(c, k);
        RAD_ISSUE_N(c, k.op_final, rows, brow_slice(src, r0, rows, wide), k.final_w[(size_t)i],
                    k.final_b[(size_t)i], RAD_NONE, bcol_at(k.b_final, r0, wide, i * n, n, rows));
        ring_after(c, k, a.n_layer + i);
    }
    if (lean)
        RAD_ISSUE_N(c, k.op_cast, T, brows(k.b_final, T), brows(a.buf_stream, T));
    else
        RAD_ISSUE_N(c, k.op_select, T, brows(k.b_mask, T), brows(k.b_final, T), RAD_NONE, RAD_NONE,
                    brows(a.buf_stream, T), RAD_NONE, RAD_NONE);
}

}  /* namespace ridgefill */

#endif /* RIDGEFILL_FINAL_H */
