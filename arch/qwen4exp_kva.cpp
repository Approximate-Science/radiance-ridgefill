/* qwen4exp_kva -- KVA / RidgeFill over radiance's in-tree Qwen4-Exp plugin.
 *
 * WHAT THIS FILE IS. The in-tree arch/qwen4exp_fp8/qwen4exp_fp8.cpp, #included whole with its
 * exports suppressed (PLAN D2, the pattern radiance's own tests/arch_test.cpp:14,281-283 uses), and
 * this plugin's exports in their place: arch id `qwen4exp`, quant "", plugin name qwen4exp_kva,
 * built as qwen4exp_fp8.so so it shadows the installed file by stem (PLAN D1, arch/CMakeLists.txt).
 *
 * `off` -- and every container with no kva.* weights -- is the included declare and step with
 * NOTHING added: no weight, no buffer, no op, not even a name map. Byte-identical to stock is the
 * contract (R6, R7), and tests/arch_static_test.cpp checks the declared graph and the issued
 * sequence against the in-tree plugin's.
 *
 * WHAT IS DONE: the fitted tensors are declared, optional, by the names tools/kva_sidecar.py writes
 * (kva_config.h lists them and the switches that pick a copy); S is the lowest projected layer the
 * model holds; every refusal a mode needs is made at declare, by name (R31; kva_declare.h). The
 * fill path (Stage 3): an approximate chunk runs the stock prologue and layers 0..S-1, then for each
 * late layer the projector into the block input `x`, its codes, and only the block's cache-writing
 * pieces (kva_fill.h); `plumb` runs every late layer exactly through those same pieces, fed the real
 * block input. NOT YET: the correction (Stage 4) and the exact rows (Stage 5) -- quality refuses at
 * declare until then, because nothing unimplemented returns success.
 *
 * Included by tests/arch_static_test.cpp too, which defines RAD_ARCH_NO_EXPORTS itself; then
 * neither plugin's exports are emitted and the test calls both namespaces directly.
 */
#ifndef RAD_ARCH_NO_EXPORTS
#define RAD_ARCH_NO_EXPORTS 1
#define QWEN4EXP_KVA_EXPORTS 1
#endif
#include <qwen4exp_fp8/qwen4exp_fp8.cpp>

#include "kva_declare.h"
#include "kva_dump.h"
#include "kva_fill.h"

namespace qwen4exp_kva {

using namespace rad::arch;

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    RAD_ARCH_TRY(qwen4exp_fp8::declare(b, meta, ctx));

    /* A SIZING DECLARE WRITES SCRATCH, as the included declare does. Its geometry is read from
     * g_model, which the real declare filled: the engine runs every sizing declare after it
     * (radiance core/engine_bringup.cpp:616-660), and the layer schedule and head counts read here
     * do not depend on max_tok. */
    static thread_local Kva probe_kva;
    Kva& k = ctx->shape_probe ? probe_kva : g_kva[ctx->rank];
    k = Kva{};
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[ctx->rank];
    if (m.layers.empty()) return RAD_E_STATE;
    RAD_ARCH_TRY(read_config(meta, m.g.max_tok, &k.cfg));
    if (!k.cfg.declare_all && k.cfg.mode == MODE_OFF) return RAD_OK;
    const char* dump = std::getenv("RADIANCE_KVA_DUMP");
    k.dump_dir = dump ? dump : "";

    k.nm = Names(ctx->scope ? ctx->scope : "");
    k.proj_w.assign(m.layers.size(), 0);
    k.proj_b.assign(m.layers.size(), 0);
    k.st.assign(m.layers.size(), 0);
    k.op_undo.assign(m.layers.size(), 0);
    k.op_apply.assign(m.layers.size(), 0);
    if (k.cfg.declare_all) return decl_every_copy(b, m, k);
    return decl_selected(b, m, ctx, k);
}

/* ================================================================== step */

/* AN APPROXIMATE STEP IS ONE PREFILL CHUNK OF ONE SEQUENCE WITH AT LEAST T PROMPT TOKENS AFTER IT
 * (PLAN §3, D3). Every field read here is in the pass key (radiance core/runtime/ctx.cpp:979-1013)
 * or fixed at declare, so a replayed pass issues what a fresh one would (R15). Anything else --
 * a mixed step, two prefills, a draft pass, the last chunks -- is the stock step. */
static bool approximate(const Kva& k, const RadBatch* b) {
    int64_t n_seq_decode = 0, n_tok_decode = 0;
    batch_split(b, &n_seq_decode, &n_tok_decode);
    return k.cfg.mode != MODE_OFF && k.have_proj && !b->enc && b->draft_pass == 0 &&
           n_seq_decode == 0 && b->n_seq == 1 && b->n_spec == 0 && b->n_ahead >= k.cfg.tail;
}

/* qwen4exp_fp8.cpp:1391-1404, verbatim: embedding, media rows, PLE hash, the stream's first value,
 * the rope table. */
static void prologue(RadCtx* c, qwen4exp_fp8::Model& m, const RadBatch* batch) {
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    if (m.op_embed_ar) RAD_ISSUE_N(c, m.op_embed_ar, T * m.g.n_embd, brows(m.a_x.x, T), RAD_NONE);
    if (m.ple_layer >= 0) m.ple.ids(c, batch);
    m.enter.step(c, T);
    if (m.op_rope_cs && T <= qk_fuse_rows(m.g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));
}

/* qwen4exp_fp8.cpp:1407-1425 for one layer, with the mixer's block issued by `block`. */
template <class Block>
static void layer(RadCtx* c, qwen4exp_fp8::Model& m, int64_t li, const RadBatch* batch, Block block) {
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const int64_t T = batch->n_tok;
    if (li == m.ple_layer) m.ple.step(c, batch);
    l.hc_mix.read(c, T, 0, T);
    block(l);
    l.hc_mix.write(c, T, 0, T);
    dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_h, m.a_x.x);
    l.hc_ffn.read(c, T, 0, T);
    l.mlp.step(c, batch);
    l.hc_ffn.write(c, T, 0, T);
    dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_h, m.a_x.x);
}

/* The stock block (qwen4exp_fp8.cpp:1415-1416), and plumb's: the same ops through the fill's
 * pieces, fed the real block input, so a fill piece that issues anything wrong shows up as a
 * difference from stock (R14). */
static void stock_block(RadCtx* c, const qwen4exp_fp8::Layer& l, const RadBatch* batch) {
    if (l.full) { l.qsa.step(c, l.attn.w.h, batch); l.attn.step(c, batch); }
    else        l.gdn.step(c, batch);
}

static void plumb_block(RadCtx* c, const qwen4exp_fp8::Layer& l, const RadBatch* batch) {
    if (l.full) {
        qsa_keys(c, l.qsa, l.attn.w.h, batch);
        qsa_select(c, l.qsa, batch);
        attn_kv(c, l.attn, batch);
        attn_tail(c, l.attn, batch);
    } else {
        gdn_project(c, l.gdn, batch->n_tok);
        gdn_scan(c, l.gdn, batch);
        gdn_tail(c, l.gdn, batch->n_tok);
    }
}

/* A filled late layer: the projector writes the block input `x` from the stream entering layer S,
 * the quantiser writes its codes as the connection read would (QuantFP8::step without its
 * matvec-only guard: an int8 linear always reads the codes), then only the cache-writing pieces.
 * No connection read or write and no MoE: `b_h` stays the layer-S stream for every projector. */
static void fill_layer(RadCtx* c, const Kva& k, qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch) {
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const int64_t T = batch->n_tok, n = m.g.n_embd;
    RAD_ISSUE_N(c, k.op_proj[(size_t)li], T, brows(m.b_h, T), RAD_W(k.proj_w[(size_t)li]),
                RAD_W(k.proj_b[(size_t)li]), RAD_NONE, brows(m.a_x.x, T));
    if (k.quant.op)
        RAD_ISSUE(c, k.quant.op, brow_slice(m.a_x.x, 0, T, n), brow_slice(m.a_x.cq(), 0, T, n),
                  brow_slice(m.a_x.cs(), 0, T, n / RAD_FP8_BLOCK));
    if (l.full) {
        qsa_keys(c, l.qsa, l.attn.w.h, batch);
        attn_kv(c, l.attn, batch);
    } else {
        gdn_project(c, l.gdn, T);
        gdn_scan(c, l.gdn, batch);
    }
}

/* qwen4exp_fp8.cpp:1428-1437: the 97th connection and, when the chunk asks for them, logits. On a
 * filled chunk those rows are not the model's (KL mode scores only the exact tail). */
static void epilogue(RadCtx* c, qwen4exp_fp8::Model& m, const RadBatch* batch) {
    const int64_t T = batch->n_tok;
    m.mixer.read(c, T, 0, T);
    if (batch->n_out <= 0) return;
    RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.a_x.x, T),
                praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.b_hout, batch->n_out));
    RAD_ISSUE_N(c, m.op_logits, batch->n_out, brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                brows(m.b_logits, batch->n_out));
}

static void approximate_step(RadCtx* c, const Kva& k, const RadBatch* batch) {
    const int rank = rad_rank(c);
    qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    prologue(c, m, batch);
    for (int64_t li = 0; li < k.split; ++li)
        layer(c, m, li, batch, [&](const qwen4exp_fp8::Layer& l) { stock_block(c, l, batch); });
    if (rank == 0 && !k.dump_dir.empty())
        dump_boundary(c, k.dump_dir, m.b_h, m.hccfg.hc * m.g.n_embd, batch);
    for (int64_t li = k.split; li < m.g.n_layer; ++li) {
        if (k.cfg.mode == MODE_PLUMB)
            layer(c, m, li, batch, [&](const qwen4exp_fp8::Layer& l) { plumb_block(c, l, batch); });
        else
            fill_layer(c, k, m, li, batch);
    }
    epilogue(c, m, batch);
    /* ONE LINE AN APPROXIMATE STEP, rank 0: scripts/grade.sh counts them against the bulk-chunk
     * count (R18), which is what proves no bulk chunk silently ran exact. */
    if (rank == 0)
        std::fprintf(stderr, "radiance: qwen4exp_kva: kva: approximate step (%s, %lld tokens, %lld "
                             "ahead)\n", kModeNames[k.cfg.mode], (long long)batch->n_tok,
                     (long long)batch->n_ahead);
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Kva& k = g_kva[rad_rank(c)];
    if (approximate(k, batch)) approximate_step(c, k, batch);
    else                       qwen4exp_fp8::step(c, batch);
}

/* The in-tree probe's answers hold here: the draft depth is the model's, and this declare writes
 * scratch under shape_probe exactly as the included one does. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    return qwen4exp_fp8::probe(meta, out);
}

}  /* namespace qwen4exp_kva */

#ifdef QWEN4EXP_KVA_EXPORTS
RAD_ARCH_PROBE(qwen4exp_kva)
RAD_ARCH_PLUGIN(qwen4exp_kva, "qwen4exp", "", "0.1.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next) with KVA / RidgeFill prefill: the in-tree "
                "qwen4exp_fp8 plugin plus projected late-layer cache fill (modes off, plumb, "
                "speed, quality)")
#endif
