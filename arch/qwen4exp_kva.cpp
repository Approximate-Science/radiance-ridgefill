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
 * block input. The correction (Stage 4): kva_state_correct undo before each late delta-net layer's
 * scan and apply after it, approximate chunks only. Quality mode (Stage 5): kva_rowsel picks rows
 * per chunk, their own stream runs exactly through every late layer on cap compacted rows (the
 * connections and the MoE), the blocks run over all rows, and the correction is scaled by the
 * decay sums kva_rho_update keeps.
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
    for (auto [env, dir] : { std::pair<const char*, std::string*>{"RADIANCE_KVA_DUMP", &k.dump_dir},
                             {"RADIANCE_KVA_CAPTURE", &k.capture_dir},
                             {"RADIANCE_KVA_CAPTURE_STATE", &k.state_dir} }) {
        const char* v = std::getenv(env);
        *dir = v ? v : "";
    }
    /* The projector's fitting data comes from EXACT runs only (HANDOVER Stage 6.1). */
    if (!k.capture_dir.empty() && k.cfg.mode != MODE_OFF) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_CAPTURE records exact runs and the "
                             "mode is %s; run it with RADIANCE_KVA=off\n", kModeNames[k.cfg.mode]);
        return RAD_E_INVAL;
    }
    const bool capturing = !k.capture_dir.empty() || !k.state_dir.empty();
    if (!k.cfg.declare_all && k.cfg.mode == MODE_OFF && !capturing) return RAD_OK;

    k.nm = Names(ctx->scope ? ctx->scope : "");
    k.proj_w.assign(m.layers.size(), 0);
    k.proj_b.assign(m.layers.size(), 0);
    k.st.assign(m.layers.size(), 0);
    k.op_undo.assign(m.layers.size(), 0);
    k.op_apply.assign(m.layers.size(), 0);
    if (k.cfg.declare_all) return decl_every_copy(b, m, k);
    if (k.cfg.mode == MODE_OFF) RAD_ARCH_TRY(capture_split(b, m, meta, k));
    else                        RAD_ARCH_TRY(decl_selected(b, m, ctx, k));
    if (!k.state_dir.empty()) RAD_ARCH_TRY(decl_state_read(b, m, ctx, k));
    return RAD_OK;
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

/* THE +st CORRECTION (Stage 4, PLAN D7): `undo` before the layer's conv/scan takes back what the
 * previous approximate chunk end added (nothing, at the first: the slot is zeroed at admission);
 * `apply` after the scan adds alpha * C and records the scale. Approximate chunks only, so the exact
 * tail starts from S_pred(P) + alpha*C and no approximate chunk reads a corrected state. ND is absent
 * in speed mode (rho = 1). Not issued when the model holds no correction. */
/* This rank's value heads of the replicated correction (kva_declare.h's decl_correction): a row
 * slice of the weight, which the issue path narrows and bounds-checks like a buffer slice
 * (radiance core/runtime/issue.cpp:661-698). */
static RadOperand correction_heads(const qwen4exp_fp8::Model& m, rad_weight w) {
    const GdnFP8::Config& g = m.gcfg;
    RadOperand o = RAD_W(w);
    o.offset = (int64_t)m.g.rank * g.n_head_v * g.head_v * g.head_k;
    o.rows   = g.n_head_v;
    return o;
}

/* A group's slot rows for this step's sequences: its own state_index, at its own pitch. */
static RadOperand slots(const RadBatch* batch, rad_kvgroup g) {
    const RadKVGroupBatch* kb = kv_batch(batch, g);
    const int64_t pitch = kb && kb->state_index_pitch > 0 ? kb->state_index_pitch : 1;
    return praw2(kb ? kb->state_index : nullptr, RAD_I32, batch->n_seq, pitch);
}

/* RADIANCE_KVA_CAPTURE_STATE: layer li's state slot of this step's one sequence, copied by
 * kva_state_read into the plugin's buffer and from there to the host. Debug: synchronises. */
static void read_state(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch, StateDump* sd) {
    if (!sd || !k.op_state_read) return;
    const GdnFP8::Config& g = m.gcfg;
    const int64_t n = g.n_head_v * g.head_v * g.head_k;
    RAD_ISSUE_N(c, k.op_state_read, 1, kv_cache(m.kv_state, (int)li), slots(batch, m.kv_state),
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

/* kva_state_correct's operands (kernels/rows.cpp): every KV operand followed by ITS OWN group's
 * slot rows, so no group has to share another's slot numbers. ND and its index come in quality
 * mode only (Stage 5); speed passes neither and rho is 1. */
static void correct(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li, rad_op op,
                    const RadBatch* batch, bool with_nd) {
    if (!op) return;
    const int L = (int)li;
    RAD_ISSUE_N(c, op, batch->n_seq,
                kv_cache(m.kv_state, L), slots(batch, m.kv_state),
                kv_cache(k.kv_applied, L), slots(batch, k.kv_applied),
                correction_heads(m, k.st[(size_t)li]),
                with_nd ? kv_cache(k.kv_rho, L) : RAD_NONE,
                with_nd ? slots(batch, k.kv_rho) : RAD_NONE);
}

/* The projector writes the block input `x` from the stream entering layer S, for every row. */
static void project(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li, int64_t T) {
    RAD_ISSUE_N(c, k.op_proj[(size_t)li], T, brows(m.b_h, T), RAD_W(k.proj_w[(size_t)li]),
                RAD_W(k.proj_b[(size_t)li]), RAD_NONE, brows(m.a_x.x, T));
}

/* x's codes, as the connection read would have written them: QuantFP8::step without its
 * matvec-only row guard, because an int8 linear always reads the codes. */
static void quantise(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t T) {
    const int64_t n = m.g.n_embd;
    if (k.quant.op)
        RAD_ISSUE(c, k.quant.op, brow_slice(m.a_x.x, 0, T, n), brow_slice(m.a_x.cq(), 0, T, n),
                  brow_slice(m.a_x.cs(), 0, T, n / RAD_FP8_BLOCK));
}

/* A filled late layer (speed): the projection and its codes, then only the cache-writing pieces.
 * No connection read or write and no MoE: `b_h` stays the layer-S stream for every projector. */
static void fill_layer(RadCtx* c, const Kva& k, qwen4exp_fp8::Model& m, int64_t li,
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
        read_state(c, k, m, li, batch, sd);   /* before the apply: S_pred, uncorrected */
        correct(c, k, m, li, k.op_apply[(size_t)li], batch, false);
    }
}

/* The decay sums of the rows this chunk approximated, per head, after the layer's scan
 * (kernels/rows.cpp kva_rho_update: one sequence, its own slot row). */
static void decay_sums(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, int64_t li,
                       const RadBatch* batch) {
    const rad_op op = k.op_rho[(size_t)li];
    if (!op) return;
    const GdnFP8& d = m.layers[(size_t)li].gdn;
    const int64_t T = batch->n_tok;
    RAD_ISSUE_N(c, op, T, bcol(d.w.ab, 0, d.cfg.n_head_v, T), brows(k.b_mask, T),
                RAD_W(d.w_a_log), RAD_W(d.w_dt_bias), kv_cache(k.kv_rho, (int)li),
                slots(batch, k.kv_rho));
}

/* A quality-mode late layer (PLAN §3, D13): every row's block input is projected, except the
 * selected rows', which come exactly from their own stream h_R through this layer's connection read
 * (scattered over the projection); the WHOLE block then runs over all rows, as tcc's engine did;
 * its output is gathered back to the selected rows, written into h_R, and their feed-forward runs
 * on the cap compacted rows alone. The block decides its all-reduce with T = n_tok, so the write
 * behind it asks the same T; the feed-forward pass and its write both decide with T = cap. */
static void quality_layer(RadCtx* c, const Kva& k, qwen4exp_fp8::Model& m, int64_t li,
                          const RadBatch* batch, StateDump* sd) {
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const HyperConn& mix = l.hc_mix;
    const int64_t T = batch->n_tok, cap = k.cfg.cap;
    project(c, k, m, li, T);
    RAD_ISSUE_N(c, mix.op_read, cap, brows(k.b_hr, cap), RAD_W(mix.w_norm), RAD_W(mix.w_down),
                RAD_W(mix.w_up), mix.c.inject ? RAD_W(mix.w_inj) : RAD_NONE, brows(k.b_xr, cap),
                mix.c.inject ? brows(k.b_injr, cap) : RAD_NONE,
                RAD_NONE, RAD_NONE, RAD_NONE, RAD_NONE);
    RAD_ISSUE_N(c, k.op_scatter_x, cap, brows(k.b_xr, cap), brows(k.b_rows, cap), brows(m.a_x.x, T));
    quantise(c, k, m, T);
    if (l.full) {
        l.qsa.step(c, l.attn.w.h, batch);
        l.attn.step(c, batch);
    } else {
        gdn_project(c, l.gdn, T);
        correct(c, k, m, li, k.op_undo[(size_t)li], batch, false);
        gdn_scan(c, l.gdn, batch);
        read_state(c, k, m, li, batch, sd);
        decay_sums(c, k, m, li, batch);
        correct(c, k, m, li, k.op_apply[(size_t)li], batch, k.kv_rho != 0);
        gdn_tail(c, l.gdn, T);
    }
    RAD_ISSUE_N(c, k.op_gather_y, cap, brows(m.a_x.x, T), brows(k.b_rows, cap), brows(k.b_yr, cap));
    const rad_op write = mix.takes_ar(T) ? mix.ar_write(T) : mix.op_write;
    if (write) RAD_ISSUE_N(c, write, cap, brows(k.b_yr, cap), brows(k.b_injr, cap), brows(k.b_hr, cap));
    hc_read_from(c, l.hc_ffn, k.b_hr, cap);
    moe_rows(c, l.mlp, cap);
    hc_write_into(c, l.hc_ffn, k.b_hr, cap, cap);
}

/* Once a quality chunk, after layer S-1: which rows run exact (kva_rowsel, ranked within the chunk,
 * PLAN D11), and their wide stream, compacted. */
static void select_rows(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m, const RadBatch* batch) {
    const int64_t T = batch->n_tok, cap = k.cfg.cap;
    RAD_ISSUE_N(c, k.op_rowsel, T, praw(batch->token_ids, RAD_I32, T),
                praw(batch->positions, RAD_I32, T), RAD_W(k.score), brows(k.b_rows, cap),
                brows(k.b_mask, T));
    RAD_ISSUE_N(c, k.op_gather_h, cap, brows(m.b_h, T), brows(k.b_rows, cap), brows(k.b_hr, cap));
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

static void approximate_step(RadCtx* c, const Kva& k, const RadBatch* batch, StateDump* sd) {
    const int rank = rad_rank(c);
    qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    prologue(c, m, batch);
    for (int64_t li = 0; li < k.split; ++li)
        layer(c, m, li, batch, [&](const qwen4exp_fp8::Layer& l) { stock_block(c, l, batch); });
    if (rank == 0 && !k.dump_dir.empty())
        dump_boundary(c, k.dump_dir, m.b_h, m.hccfg.hc * m.g.n_embd, batch);
    if (k.cfg.mode == MODE_QUALITY) {
        select_rows(c, k, m, batch);
        if (rank == 0 && !k.dump_dir.empty()) dump_rows(c, k.dump_dir, k.b_rows, k.cfg.cap, batch);
    }
    for (int64_t li = k.split; li < m.g.n_layer; ++li) {
        if (k.cfg.mode == MODE_PLUMB)
            layer(c, m, li, batch, [&](const qwen4exp_fp8::Layer& l) { plumb_block(c, l, batch); });
        else if (k.cfg.mode == MODE_QUALITY)
            quality_layer(c, k, m, li, batch, sd);
        else
            fill_layer(c, k, m, li, batch, sd);
    }
    epilogue(c, m, batch);
    /* ONE LINE AN APPROXIMATE STEP, rank 0: scripts/grade.sh counts them against the bulk-chunk
     * count (R18), which is what proves no bulk chunk silently ran exact. */
    if (rank == 0)
        std::fprintf(stderr, "radiance: qwen4exp_kva: kva: approximate step (%s, %lld tokens, %lld "
                             "ahead)\n", kModeNames[k.cfg.mode], (long long)batch->n_tok,
                     (long long)batch->n_ahead);
}

/* ONE SEQUENCE'S PREFILL CHUNK, whatever is ahead of it: what the captures record. */
static bool single_prefill(const RadBatch* b) {
    int64_t n_seq_decode = 0, n_tok_decode = 0;
    batch_split(b, &n_seq_decode, &n_tok_decode);
    return !b->enc && b->draft_pass == 0 && n_seq_decode == 0 && b->n_seq == 1 && b->n_spec == 0;
}

/* RADIANCE_KVA_CAPTURE (mode off, rank 0): the stock step, issued through the same pieces as
 * every other path here (the static test holds them to the in-tree step), with host copies of the
 * stream entering layer S and of every late layer's block input `x`, read right after its
 * connection read and before the block writes its output over it. */
static void capture_step(RadCtx* c, const Kva& k, const RadBatch* batch) {
    qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    const int64_t T = batch->n_tok, n = m.g.n_embd;
    Capture cap;
    const bool ok = capture_begin(c, batch, k.capture_dir, &cap);
    if (!ok) std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_CAPTURE: device read failed\n");
    prologue(c, m, batch);
    for (int64_t li = 0; li < m.g.n_layer; ++li) {
        if (ok && li == k.split) capture_rows(c, cap, "boundary", m.b_h, T, m.hccfg.hc * n);
        layer(c, m, li, batch, [&](const qwen4exp_fp8::Layer& l) {
            if (ok && li >= k.split) {
                capture_rows(c, cap, "bi." + std::to_string(li), m.a_x.x, T, n);
                cap.layers.push_back((int)li);
            }
            stock_block(c, l, batch);
        });
    }
    epilogue(c, m, batch);
    if (ok) capture_end(cap, k.split, n, m.hccfg.hc);
}

/* RADIANCE_KVA_CAPTURE_STATE: whatever late delta-net layer the step did not already copy (an
 * exact or plumb chunk copies here, after the step; nothing touches a layer's state after its
 * scan), then one file for this (chunk, rank). */
static void finish_state(RadCtx* c, const Kva& k, const RadBatch* batch, StateDump& sd, bool approx) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    for (int64_t li = k.split; li < m.g.n_layer; ++li)
        if (!m.layers[(size_t)li].full &&
            std::find(sd.layers.begin(), sd.layers.end(), (int)li) == sd.layers.end())
            read_state(c, k, m, li, batch, &sd);
    state_end(c, k.state_dir, batch, sd, m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k, rad_rank(c),
              m.g.world, approx, kModeNames[k.cfg.mode]);
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Kva& k = g_kva[rad_rank(c)];
    const bool approx = approximate(k, batch);
    const bool capture = !k.capture_dir.empty() && rad_rank(c) == 0 && single_prefill(batch);
    const bool states = !k.state_dir.empty() && single_prefill(batch);
    StateDump sd;
    if (approx)       approximate_step(c, k, batch, states ? &sd : nullptr);
    else if (capture) capture_step(c, k, batch);
    else              qwen4exp_fp8::step(c, batch);
    if (states) finish_state(c, k, batch, sd, approx);
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
