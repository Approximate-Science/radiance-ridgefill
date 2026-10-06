/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* qwen4exp_moe.h -- a routed layer's MoE, issued by hand on an approximate pass (PLAN-FIX §3, §6.1).
 *
 * WHY A COPY. MoeFP8::pass (radiance 1.1.1 7001841, arch/common/rad_block_moe_fp8.h:1473-1663) has no hook
 * for the two things an approximate pass needs: the routing slots of bulk rows dropped
 * (`ridgefill_drop_rows`, ids := -1) between the top-k and the scatter, and the stager probes issued
 * right after one layer's gate-up GEMM (notes/impl.md §2). Everything else below is that function
 * verbatim, with the same handles and operands; tests/arch_static_test.cpp holds it to the in-tree
 * pass issue for issue (R53'). Dropped slots cost nothing downstream: moe_scatter ignores an id
 * outside the rank's experts, the grouped GEMMs bound their work by the live count, and the gather
 * gives an unclaimed slot zero weight (libr4d/r4d_moe.hip:477-479, 909-916, 1580-1586).
 *
 * Omitted, and refused at declare instead (ridgefill_declare.h check_fill): the calibration tap.
 */
#ifndef QWEN4EXP_MOE_H
#define QWEN4EXP_MOE_H

#include <functional>

namespace qwen4exp_ridgefill {

using namespace rad::arch;

/* THE GROUPED GEMMS' WEIGHT TABLES (rad_block_moe_fp8.h:1542-1599). A quantised layer passes its experts
 * as two tables by parity (`moe_gemm_q`), or -- at four ranks, where the extra fp8 block of a five-block
 * expert goes round all four (MoeFP8::Config::ff_lo4, radiance 1.0.10) -- as four by number mod 4
 * (`moe_gemm_q_mod4`), class q holding ceil((n_reg - q) / 4) experts. `n` is how many class q holds. */
inline int64_t moe_class_len(const MoeFP8& e, int q) {
    return e.ncls == 4 ? (e.n_reg + 3 - q) / 4 : e.n_reg / 2;
}

/* The gate/up GEMM over rows [r0, r0 + rows) of the quantised input, into `out` (rows * top_k slots), with
 * `ids`/`offsets` its sorted slots and per-expert offsets: the in-tree issue in either table form. The
 * probe passes zeros for both, which leaves every expert empty. */
inline void moe_gate_up_q(RadCtx* c, const MoeFP8& e, int64_t r0, int64_t rows, RadOperand sorted,
                          RadOperand offsets, RadOperand out) {
    const int64_t n = e.g.n_embd;
    const bool rot = e.c.expert_rot != 0;
    const auto& w = e.w;
    const RadOperand x = brow_slice(rot ? w.hr.q : w.h.q, r0, rows, n);
    const RadOperand s = brow_slice(rot ? w.hr.s : w.h.s, r0, rows, n / RAD_FP8_BLOCK);
    if (e.ncls == 4)
        RAD_ISSUE_N(c, e.op_gu, rows, x, s,
                    RAD_WTAB(e.w_gu[0], moe_class_len(e, 0)), RAD_WTAB(e.w_gus[0], moe_class_len(e, 0)),
                    sorted, offsets,
                    RAD_WTAB(e.w_gu[1], moe_class_len(e, 1)), RAD_WTAB(e.w_gus[1], moe_class_len(e, 1)),
                    RAD_WTAB(e.w_gu[2], moe_class_len(e, 2)), RAD_WTAB(e.w_gus[2], moe_class_len(e, 2)),
                    RAD_WTAB(e.w_gu[3], moe_class_len(e, 3)), RAD_WTAB(e.w_gus[3], moe_class_len(e, 3)),
                    out);
    else
        RAD_ISSUE_N(c, e.op_gu, rows, x, s,
                    RAD_WTAB(e.w_gu[0], moe_class_len(e, 0)), RAD_WTAB(e.w_gus[0], moe_class_len(e, 0)),
                    sorted, offsets,
                    RAD_WTAB(e.w_gu[1], moe_class_len(e, 1)), RAD_WTAB(e.w_gus[1], moe_class_len(e, 1)),
                    out);
}

/* What differs from the in-tree pass: the drop op with the mask it reads (0 = no row dropped),
 * and what to issue right after the gate-up GEMM (empty = nothing). */
struct MoeArm {
    rad_op                drop = 0;
    rad_buf               mask = 0;
    std::function<void()> after_gate_up;
};

/* THE STAGER PROBE (notes/impl.md §2): layer e's gate-up GEMM issued over one token with every
 * expert offset zero. No expert owns a row, so the kernel reads no weight and only writes zeros into
 * its output rows -- the plugin's own (libr4d/r4d_moe.hip:909-916) -- while its HANDLE is the
 * layer's first expert op, which is all the prefill stager reacts to (stager.cpp:114-152). */
inline void moe_probe(RadCtx* c, const MoeFP8& e, rad_buf zeros, rad_buf out) {
    const int64_t ne = e.c.n_expert, k = e.c.top_k, n = e.g.n_embd;
    if (e.c.expert_bf16)
        RAD_ISSUE_N(c, e.op_gu, 1, brow_slice(e.w.h.x, 0, 1, n), RAD_WTAB(e.w_gu[0], ne),
                    brows(zeros, k), brows(zeros, ne + 1), brows(out, k));
    else
        moe_gate_up_q(c, e, 0, 1, brows(zeros, k), brows(zeros, e.n_reg + 1), brows(out, k));
}

/* The router, the top-k and the sort, with the drop between the last two. Neither FUSED form has a
 * point between them, so a pass that drops rows always takes the router and the pair; a pass that does
 * not drops nothing and keeps the stock choice, which leaves plumb's issue identical to stock's at every
 * row count. Returns whether the fused router wrote the shared gate (rad_block_moe_fp8.h:1499-1527). */
inline bool moe_route(RadCtx* c, const MoeFP8& e, const MoeArm& arm, RadOperand ecnt, int64_t r0,
                      int64_t rows) {
    const int64_t ne = e.c.n_expert, ne_all = e.c.n_expert_all, k = e.c.top_k, n = e.g.n_embd;
    const auto& w = e.w;
    if (!arm.drop && e.op_router_fused && rows <= e.router_fused_rows) {
        RAD_ISSUE_N(c, e.op_router_fused, rows, brow_slice(w.h.x, r0, rows, n),
                    RAD_W(e.w_router), e.router_fused_gate ? RAD_W(e.w_sgate) : RAD_NONE,
                    brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt,
                    e.router_fused_gate ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE);
        return e.router_fused_gate;
    }
    RAD_ISSUE_N(c, e.op_router, rows, brow_slice(w.h.x, r0, rows, n), RAD_W(e.w_router),
                brow_slice(w.logits, r0, rows, ne_all));
    if (!arm.drop && e.op_topk_scatter && rows <= e.topk_scatter_rows) {
        RAD_ISSUE_N(c, e.op_topk_scatter, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
        return false;
    }
    RAD_ISSUE_N(c, e.op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),
                brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));
    if (arm.drop)
        RAD_ISSUE_N(c, arm.drop, rows, brow_slice(arm.mask, r0, rows, 1),
                    brow_slice(w.ids, r0, rows, k));
    RAD_ISSUE_N(c, e.op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    return false;
}

/* rad_block_moe_fp8.h:1542-1599: the two grouped GEMMs, the shared arm beside them (its gate already
 * written when `gate_done`), the protected experts after -- with the arm's probes right behind the
 * gate-up GEMM. */
inline void moe_experts(RadCtx* c, const MoeFP8& e, const MoeArm& arm, int64_t T, int64_t r0,
                        int64_t rows, bool gate_done) {
    const int64_t ne = e.c.n_expert, k = e.c.top_k, n = e.g.n_embd;
    const bool shared = e.op_sgate != 0;
    const auto& w = e.w;
    if (e.c.expert_bf16)
        RAD_ISSUE_N(c, e.op_gu, rows, brow_slice(w.h.x, r0, rows, n), RAD_WTAB(e.w_gu[0], ne),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), brows(w.egu, rows * k));
    else
        moe_gate_up_q(c, e, r0, rows, brows(w.sorted, rows * k), brows(w.eoff, e.n_reg + 1),
                      brows(w.egu, rows * k));
    if (arm.after_gate_up) arm.after_gate_up();
    if (shared) e.shared_up(c, T, r0, rows, gate_done);
    e.gq_exp.step(c, w.egu, rows * k, 0, rows * k);
    if (shared) e.gq_sh.step(c, w.sgu, T, r0, rows);
    if (e.c.expert_bf16)
        RAD_ISSUE_N(c, e.op_dn, rows, brows(w.eff.x, rows * k), RAD_WTAB(e.w_dn[0], ne),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), brows(w.edn, rows * k));
    else if (e.ncls == 4)
        RAD_ISSUE_N(c, e.op_dn, rows, brows(w.eff.q, rows * k), brows(w.eff.s, rows * k),
                    RAD_WTAB(e.w_dn[0], moe_class_len(e, 0)), RAD_WTAB(e.w_dns[0], moe_class_len(e, 0)),
                    brows(w.sorted, rows * k), brows(w.eoff, e.n_reg + 1),
                    RAD_WTAB(e.w_dn[1], moe_class_len(e, 1)), RAD_WTAB(e.w_dns[1], moe_class_len(e, 1)),
                    RAD_WTAB(e.w_dn[2], moe_class_len(e, 2)), RAD_WTAB(e.w_dns[2], moe_class_len(e, 2)),
                    RAD_WTAB(e.w_dn[3], moe_class_len(e, 3)), RAD_WTAB(e.w_dns[3], moe_class_len(e, 3)),
                    brows(w.edn, rows * k));
    else
        RAD_ISSUE_N(c, e.op_dn, rows, brows(w.eff.q, rows * k), brows(w.eff.s, rows * k),
                    RAD_WTAB(e.w_dn[0], moe_class_len(e, 0)), RAD_WTAB(e.w_dns[0], moe_class_len(e, 0)),
                    brows(w.sorted, rows * k), brows(w.eoff, e.n_reg + 1),
                    RAD_WTAB(e.w_dn[1], moe_class_len(e, 1)), RAD_WTAB(e.w_dns[1], moe_class_len(e, 1)),
                    brows(w.edn, rows * k));
    if (e.op_pgu) {
        const int64_t np = (int64_t)e.prot.size();
        const RadOperand poff = brow_slice(w.eoff, e.n_reg, np + 1, 1);
        RAD_ISSUE_N(c, e.op_pgu, rows, brow_slice(w.h.x, r0, rows, n), RAD_WTAB(e.w_pgu[0], np),
                    brows(w.sorted, rows * k), poff, brows(e.b_pgu, rows * k));
        e.gq_prot.step(c, e.b_pgu, rows * k, 0, rows * k);
        RAD_ISSUE_N(c, e.op_pdn, rows, brows(e.pff.x, rows * k), RAD_WTAB(e.w_pdn[0], np),
                    brows(w.sorted, rows * k), poff, brows(w.edn, rows * k));
    }
    if (shared) e.sh_down.step(c, w.sff, w.sout, T, r0, rows);
}

/* MoeFP8::pass with the arm's substitutions (rad_block_moe_fp8.h:1473-1663). */
inline void moe_pass(RadCtx* c, const MoeFP8& e, const MoeArm& arm, int64_t T, int64_t r0,
                     int64_t rows) {
    const int64_t ne = e.c.n_expert, k = e.c.top_k, n = e.g.n_embd;
    const auto& w = e.w;
    if (!e.ext_in) e.nq_h.step(c, T, r0, rows);
    int32_t* const counts = rad_route_counts(c, e.layer, ne);
    const RadOperand ecnt = counts ? praw(counts, RAD_I32, ne) : brows(w.ecnt, ne);
    const bool gate_done = moe_route(c, e, arm, ecnt, r0, rows);
    if (e.op_hq)
        RAD_ISSUE_N(c, e.op_hq, rows, brow_slice(w.h.x, r0, rows, n), brow_slice(w.hr.q, r0, rows, n),
                    brow_slice(w.hr.s, r0, rows, n / RAD_FP8_BLOCK));
    moe_experts(c, e, arm, T, r0, rows, gate_done);
    const bool fold = e.gfold;
    if (!gather_taken(e.g, T, e.gather_out_rows, e.gather_out_rows6))
        RAD_ISSUE_N(c, e.op_gather, rows, brows(w.edn, rows * k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), fold ? brow_slice(w.sout, r0, rows, n) : RAD_NONE,
                    fold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE, brow_slice(w.h.x, r0, rows, n));
    if (e.op_ar && !ar_taken(e.g, e.ar_out_take != kArOutNorm ? T : rows, e.ar_out, e.ar_out_take))
        RAD_ISSUE_N(c, e.op_ar, rows * n, brow_slice(w.h.x, r0, rows, n), RAD_NONE);
    if (e.op_add)
        RAD_ISSUE_N(c, e.op_add, rows, brow_slice(w.x, r0, rows, n), brow_slice(w.h.x, r0, rows, n),
                    brow_slice(w.x, r0, rows, n));
}

/* MoeFP8::step (rad_block_moe_fp8.h:1710-1743) over rows [from, to) (to < 0: the step's end): the
 * passes and the routing report the heat engine reads -- with dropped slots or an exact-rows-only
 * range, the counts are the exact rows' alone, so bulk rows no longer register as heat. */
inline void moe_layer(RadCtx* c, const MoeFP8& e, const MoeArm& arm, const RadBatch* batch,
                      int64_t from = 0, int64_t to = -1) {
    const int64_t T = batch->n_tok, end = to < 0 ? T : to;
    for (int64_t r0 = from; r0 < end; r0 += e.c.rows)
        moe_pass(c, e, arm, T, r0, (end - r0 < e.c.rows) ? (end - r0) : e.c.rows);
    moe_debug_weights(c, e.layer, e.c.top_k, e.w.ids, e.w.ew);
    RadRouting r{};
    r.expert_ids    = (const int32_t*)rad_buf_ptr(c, e.w.ids);
    r.expert_w      = nullptr;
    const int32_t* const counts = rad_route_counts(c, e.layer, e.c.n_expert);
    r.expert_count  = counts ? counts : (const int32_t*)rad_buf_ptr(c, e.w.ecnt);
    r.sorted_tok    = (const int32_t*)rad_buf_ptr(c, e.w.sorted);
    r.expert_offset = (const int32_t*)rad_buf_ptr(c, e.w.eoff);
    r.top_k         = e.c.top_k;
    r.n_expert      = e.c.n_expert;
    rad_route_report(c, e.layer, &r);
}

}  /* namespace qwen4exp_ridgefill */

#endif /* QWEN4EXP_MOE_H */
