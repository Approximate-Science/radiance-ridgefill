/* qwen4exp_adapter.h -- the qwen4exp side of the core/adapter split (notes/adapter-split-spec.md §1.2):
 * every fact the core reads, taken from the in-tree model after its declare. Included by
 * qwen4exp_kva.cpp after <qwen4exp_fp8/qwen4exp_fp8.cpp>, so `qwen4exp_fp8::Model` is complete here.
 */
#ifndef QWEN4EXP_ADAPTER_H
#define QWEN4EXP_ADAPTER_H

#include "kva_adapter.h"

namespace qwen4exp_kva {

using kva::KvaAdapter;
using kva::StateShape;

/* The fit facts the method was measured at on this model (KVA-FACTS §5). */
constexpr int64_t kAdapterMinTail = 512, kAdapterDefaultTail = 2048;

/* Rank `m`'s facts. Read after qwen4exp_fp8::declare has filled `m`; the buffers are this rank's. */
inline KvaAdapter adapter_of(const qwen4exp_fp8::Model& m) {
    KvaAdapter a;
    a.log_name = "qwen4exp_kva";
    a.match_name = "qwen4exp";
    a.shadow_so = "qwen4exp_fp8.so";
    a.n_layer = m.g.n_layer;
    a.n_embd = m.g.n_embd;
    a.n_vocab_all = m.g.n_vocab_all;
    a.wide = m.hccfg.hc * m.g.n_embd;
    a.tile = m.gcfg.chunk;
    a.split_lo = m.ple_layer + 1;  /* -1 (no PLE) gives 0: no constraint */
    a.probe_depth = 3;             /* the probes ride behind layer S-3's gate-up (notes/impl.md §2) */
    a.top_k = m.moecfg.top_k;
    for (const qwen4exp_fp8::Layer& l : m.layers) {
        a.n_expert = std::max(a.n_expert, l.mlp.c.n_expert);
        a.n_ff_exp = std::max(a.n_ff_exp, l.mlp.c.n_ff_exp);
        a.full.push_back(l.full ? 1 : 0);
        a.ext_in.push_back((l.full ? l.attn.ext_in : l.gdn.ext_in) ? 1 : 0);
        a.calibrated.push_back(l.mlp.op_gram_gu ? 1 : 0);
        a.routed.push_back(l.mlp.c.n_expert > 0 ? 1 : 0);
    }
    a.min_tail = kAdapterMinTail;
    a.default_tail = kAdapterDefaultTail;
    a.act_dtype = m.g.act_dtype;
    a.state = {m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k};
    a.buf_stream = m.b_h;
    a.buf_x = m.a_x.x;
    a.buf_x_q = m.a_x.cq();        /* the pair the connection read writes: int8 when q8_fed, else E4M3 */
    a.buf_x_s = m.a_x.cs();
    a.buf_logits = m.b_logits;
    return a;
}

}  // namespace qwen4exp_kva

#endif
