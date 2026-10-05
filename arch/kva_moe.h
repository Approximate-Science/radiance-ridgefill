/* kva_moe.h -- a routed layer's MoE, issued by hand on an approximate pass (PLAN-FIX §3, §6.1).
 *
 * WHY A COPY. MoeFP8::pass (radiance 140987f, arch/common/rad_block_moe_fp8.h:1341-1496) has no hook
 * for the two things an approximate pass needs: the routing slots of bulk rows dropped
 * (`kva_drop_rows`, ids := -1) between the top-k and the scatter, and the stager probes issued
 * right after one layer's gate-up GEMM (notes/impl.md §2). Everything else below is that function
 * verbatim, with the same handles and operands; tests/arch_static_test.cpp holds it to the in-tree
 * pass issue for issue (R53'). Dropped slots cost nothing downstream: moe_scatter ignores an id
 * outside the rank's experts, the grouped GEMMs bound their work by the live count, and the gather
 * gives an unclaimed slot zero weight (libr4d/r4d_moe.hip:477-479, 909-916, 1580-1586).
 *
 * Omitted, and refused at declare instead (kva_declare.h check_fill): the calibration tap.
 */
#ifndef QWEN4EXP_KVA_MOE_H
#define QWEN4EXP_KVA_MOE_H

#include <functional>

namespace qwen4exp_kva {

using namespace rad::arch;

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
    const int64_t ne = e.c.n_expert, k = e.c.top_k, n = e.g.n_embd, half = e.n_reg / 2;
    const bool rot = e.c.expert_rot != 0;
    const auto& w = e.w;
    if (e.c.expert_bf16)
        RAD_ISSUE_N(c, e.op_gu, 1, brow_slice(w.h.x, 0, 1, n), RAD_WTAB(e.w_gu[0], ne),
                    brows(zeros, k), brows(zeros, ne + 1), brows(out, k));
    else
        RAD_ISSUE_N(c, e.op_gu, 1, brow_slice(rot ? w.hr.q : w.h.q, 0, 1, n),
                    brow_slice(rot ? w.hr.s : w.h.s, 0, 1, n / RAD_FP8_BLOCK),
                    RAD_WTAB(e.w_gu[0], half), RAD_WTAB(e.w_gus[0], half), brows(zeros, k),
                    brows(zeros, e.n_reg + 1), RAD_WTAB(e.w_gu[1], half), RAD_WTAB(e.w_gus[1], half),
                    brows(out, k));
}

/* The top-k and the sort, with the drop between them. The FUSED form has no point between the two,
 * so a pass that drops rows always takes the pair; a pass that does not drops nothing and keeps the
 * stock choice, which leaves plumb's issue identical to stock's at every row count. */
inline void moe_route(RadCtx* c, const MoeFP8& e, const MoeArm& arm, RadOperand ecnt, int64_t r0,
                      int64_t rows) {
    const int64_t ne = e.c.n_expert, ne_all = e.c.n_expert_all, k = e.c.top_k;
    const auto& w = e.w;
    if (!arm.drop && e.op_topk_scatter && rows <= e.topk_scatter_rows) {
        RAD_ISSUE_N(c, e.op_topk_scatter, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
        return;
    }
    RAD_ISSUE_N(c, e.op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),
                brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));
    if (arm.drop)
        RAD_ISSUE_N(c, arm.drop, rows, brow_slice(arm.mask, r0, rows, 1),
                    brow_slice(w.ids, r0, rows, k));
    RAD_ISSUE_N(c, e.op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
}

/* rad_block_moe_fp8.h:1404-1438: the two grouped GEMMs, the shared arm beside them, the protected
 * experts after -- with the arm's probes right behind the gate-up GEMM. */
inline void moe_experts(RadCtx* c, const MoeFP8& e, const MoeArm& arm, int64_t T, int64_t r0,
                        int64_t rows) {
    const int64_t ne = e.c.n_expert, k = e.c.top_k, n = e.g.n_embd, half = e.n_reg / 2;
    const bool rot = e.c.expert_rot != 0, shared = e.op_sgate != 0;
    const auto& w = e.w;
    if (e.c.expert_bf16)
        RAD_ISSUE_N(c, e.op_gu, rows, brow_slice(w.h.x, r0, rows, n), RAD_WTAB(e.w_gu[0], ne),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), brows(w.egu, rows * k));
    else
        RAD_ISSUE_N(c, e.op_gu, rows, brow_slice(rot ? w.hr.q : w.h.q, r0, rows, n),
                    brow_slice(rot ? w.hr.s : w.h.s, r0, rows, n / RAD_FP8_BLOCK),
                    RAD_WTAB(e.w_gu[0], half), RAD_WTAB(e.w_gus[0], half),
                    brows(w.sorted, rows * k), brows(w.eoff, e.n_reg + 1),
                    RAD_WTAB(e.w_gu[1], half), RAD_WTAB(e.w_gus[1], half), brows(w.egu, rows * k));
    if (arm.after_gate_up) arm.after_gate_up();
    if (shared) e.shared_up(c, T, r0, rows);
    e.gq_exp.step(c, w.egu, rows * k, 0, rows * k);
    if (shared) e.gq_sh.step(c, w.sgu, T, r0, rows);
    if (e.c.expert_bf16)
        RAD_ISSUE_N(c, e.op_dn, rows, brows(w.eff.x, rows * k), RAD_WTAB(e.w_dn[0], ne),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), brows(w.edn, rows * k));
    else
        RAD_ISSUE_N(c, e.op_dn, rows, brows(w.eff.q, rows * k), brows(w.eff.s, rows * k),
                    RAD_WTAB(e.w_dn[0], half), RAD_WTAB(e.w_dns[0], half),
                    brows(w.sorted, rows * k), brows(w.eoff, e.n_reg + 1),
                    RAD_WTAB(e.w_dn[1], half), RAD_WTAB(e.w_dns[1], half), brows(w.edn, rows * k));
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

/* MoeFP8::pass with the arm's substitutions (rad_block_moe_fp8.h:1341-1496). */
inline void moe_pass(RadCtx* c, const MoeFP8& e, const MoeArm& arm, int64_t T, int64_t r0,
                     int64_t rows) {
    const int64_t ne = e.c.n_expert, ne_all = e.c.n_expert_all, k = e.c.top_k, n = e.g.n_embd;
    const auto& w = e.w;
    if (!e.ext_in) e.nq_h.step(c, T, r0, rows);
    int32_t* const counts = rad_route_counts(c, e.layer, ne);
    const RadOperand ecnt = counts ? praw(counts, RAD_I32, ne) : brows(w.ecnt, ne);
    RAD_ISSUE_N(c, e.op_router, rows, brow_slice(w.h.x, r0, rows, n), RAD_W(e.w_router),
                brow_slice(w.logits, r0, rows, ne_all));
    moe_route(c, e, arm, ecnt, r0, rows);
    if (e.op_hq)
        RAD_ISSUE_N(c, e.op_hq, rows, brow_slice(w.h.x, r0, rows, n), brow_slice(w.hr.q, r0, rows, n),
                    brow_slice(w.hr.s, r0, rows, n / RAD_FP8_BLOCK));
    moe_experts(c, e, arm, T, r0, rows);
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

/* MoeFP8::step (rad_block_moe_fp8.h:1529-1576): the passes and the routing report the heat engine
 * reads -- with dropped slots, the counts are the exact rows' alone, so bulk rows no longer
 * register as heat. */
inline void moe_layer(RadCtx* c, const MoeFP8& e, const MoeArm& arm, const RadBatch* batch) {
    const int64_t T = batch->n_tok;
    for (int64_t r0 = 0; r0 < T; r0 += e.c.rows)
        moe_pass(c, e, arm, T, r0, (T - r0 < e.c.rows) ? (T - r0) : e.c.rows);
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

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_MOE_H */
