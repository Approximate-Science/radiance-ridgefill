/* kva_hazard.h -- DD-A's exact branch-hazard instrument (PLAN-FIX §5.4; Stage C, R65/R68).
 *
 * A request resuming from a prefix-cache checkpoint that a LONGER request wrote may find some of its
 * exact-tail positions were approximated by that producer (the branch-shorter case, §5.2). Each
 * sequence carries a LINEAR meta slot -- zeroed at admission, snapshotted with every checkpoint because
 * the group is bound to a late layer (radiance core/mem/kv.cpp:1449-1486) -- holding its last
 * approximated position; kva_hazard (kva.so) records it on every approximate pass and, on a pass whose
 * last sequence still has tail ahead (span = T - n_ahead > 0, keyed), counts the tail positions before
 * the pass that the restored slot says were approximated, once, into a host-mapped counter. Rank 0 logs
 * the counter when it moves, on a LATER step: it is read for the log only, never for an issue (R99).
 * Speed and quality only; plumb, which approximates nothing, and off declare none of it.
 */
#ifndef KVA_HAZARD_H
#define KVA_HAZARD_H

namespace kva {

using namespace rad::arch;

/* After the step: count (span > 0) and, on an approximate pass, record. The last bulk position comes
 * from kva_mask's bounds {s, b'} on the masked, straddle and decoders paths, and is the chunk's last
 * row on the lean path, whose whole chunk is bulk ({s, e} itself). */
static void hazard_issue(RadCtx* c, const Kva& k, const RadBatch* batch, const Pass& p) {
    if (!k.op_hazard || !g_hazard_dev[rad_rank(c)] || batch->enc || batch->draft_pass) return;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t span = k.cfg.tail - batch->n_ahead;
    const bool approx = p.path != PATH_STOCK;
    if (batch->n_seq <= D || (!approx && span <= 0)) return;
    const RadOperand cu_last = praw(batch->cu_seqlens + batch->n_seq - 1, RAD_I32, 2);
    const RadOperand bounds = !approx ? RAD_NONE : p.path == PATH_LEAN ? cu_last : brows(k.b_bounds, 2);
    RAD_ISSUE_N(c, k.op_hazard, 1, cu_last, praw(batch->positions, RAD_I32, batch->n_tok), bounds,
                span > 0 ? praw(batch->positions, RAD_I32, span) : RAD_NONE,   /* its extent only */
                kv_cache(k.kv_meta, k.meta_layer), last_slot(batch, k.kv_meta),
                praw(g_hazard_dev[rad_rank(c)], RAD_F32, 1));
}

/* Rank 0, host only: the counter as the device left it some steps ago, said when it moved. */
static void hazard_log(RadCtx* c) {
    const int r = rad_rank(c);
    if (r != 0 || !g_hazard_dev[r]) return;
    const float* h = (const float*)rad_dev_host_ptr(g_hazard_dev[r]);
    if (!h || *h == g_hazard_logged[r]) return;
    std::fprintf(stderr, "radiance: %s: kva: hazard %lld positions (total %lld)\n", g_log_name,   /* tools/hazard_rate.py HAZARD_LOG_RE */
                 (long long)(*h - g_hazard_logged[r]), (long long)*h);
    g_hazard_logged[r] = *h;
}

}  /* namespace kva */

#endif /* KVA_HAZARD_H */
