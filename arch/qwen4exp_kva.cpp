/* qwen4exp_kva -- KVA / RidgeFill over radiance's in-tree Qwen4-Exp plugin.
 *
 * WHAT THIS FILE IS. The in-tree arch/qwen4exp_fp8/qwen4exp_fp8.cpp, #included whole with its
 * exports suppressed (PLAN D2, the pattern radiance's own tests/arch_test.cpp:14,281-283 uses), and
 * this plugin's exports in their place: arch id `qwen4exp`, quant "", plugin name qwen4exp_kva,
 * built as qwen4exp_fp8.so so it shadows the installed file by stem (PLAN D1, arch/CMakeLists.txt).
 *
 * `off` -- the default, and every container with no kva.* weights -- is the included declare and
 * step with NOTHING added: no weight, no buffer, no op, not even a name map. Byte-identical to stock
 * is the contract (R6, R7), and tests/arch_static_test.cpp checks the declared graph and the issued
 * sequence against the in-tree plugin's.
 *
 * WHAT AN APPROXIMATE PASS IS (PLAN-FIX v2). Any trunk pass whose LAST entry is a non-final prefill
 * chunk with bulk rows -- rows followed by at least T' prompt tokens (T rounded up to the delta
 * net's tile) -- whatever else rides in the step. Layers 0..S-1 run stock; from S on, each late
 * layer either takes the LEAN fill (speed mode, a one-sequence step whose whole chunk is bulk: only
 * the cache-writing pieces, kva_fill.h) or the MASKED layer (every other shape: the in-tree layer
 * over all rows with the device mask choosing which rows use the projection, kva_layer.h). The
 * routed down GEMMs from layer S-1 go through weightless alternate handles so the expert stager
 * streams the late layers' routed experts instead of staging each whole (§6.1).
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
#include "kva_moe.h"
#include "kva_layer.h"

namespace qwen4exp_kva {

using namespace rad::arch;

/* The container's kva.mode is not a switch (PLAN-FIX §6.5): said once, by the real declare of
 * rank 0, so an operator who converted with --set kva.mode=... learns why nothing changed. */
static void note_meta_mode(const Config& c, const RadBuildCtx* ctx) {
    if (!c.meta_mode || ctx->shape_probe || ctx->rank != 0) return;
    std::fprintf(stderr, "radiance: qwen4exp_kva: the container's kva.mode=%s is ignored; the mode "
                         "comes from RADIANCE_KVA only (now %s)\n", c.meta_mode, kModeNames[c.mode]);
}

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
    RAD_ARCH_TRY(read_config(meta, &k.cfg));
    note_meta_mode(k.cfg, ctx);
    for (auto [name, dir] : { std::pair<const char*, std::string*>{"RADIANCE_KVA_DUMP", &k.dump_dir},
                              {"RADIANCE_KVA_CAPTURE", &k.capture_dir},
                              {"RADIANCE_KVA_CAPTURE_STATE", &k.state_dir} }) {
        const char* v = std::getenv(name);
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
    k.tile = m.gcfg.chunk;
    k.out_rows_ok = ctx->max_out_rows == 0 || k.cfg.score_bulk;
    for (auto* v : { &k.proj_w, &k.proj_b, &k.st })
        v->assign(m.layers.size(), 0);
    for (auto* v : { &k.op_undo, &k.op_apply, &k.op_rho, &k.op_proj, &k.alt_dn })
        v->assign(m.layers.size(), 0);
    if (k.cfg.declare_all) return decl_every_copy(b, m, k);
    if (k.cfg.mode == MODE_OFF) RAD_ARCH_TRY(capture_split(b, m, meta, k));
    else                        RAD_ARCH_TRY(decl_selected(b, m, ctx, k));
    if (!k.state_dir.empty()) RAD_ARCH_TRY(decl_state_read(b, m, ctx, k));
    return RAD_OK;
}

/* ================================================================== step */

/* WHAT THIS PASS IS (PLAN-FIX §2): a function of keyed batch fields (radiance
 * core/runtime/ctx.cpp:989-995) and declare-time config only, so a replayed pass issues what a
 * fresh one would (R15/R99). The host never needs the last sequence's first row s, which is device
 * data: computing a row exactly is always correct, so every op runs over all rows and only the
 * device mask decides which rows use the projection; the host knows the bulk END b and a lower
 * bound s_lb of s, which bound the projector's rows. Media steps, encoder and draft passes and the
 * last chunks of a prompt run stock; KL mode too unless SCORE_BULK says the tail alone is scored. */
static Pass derive(const Kva& k, const RadBatch* batch) {
    Pass p;
    const Config& c = k.cfg;
    if (c.mode == MODE_OFF || !k.have_proj || !k.out_rows_ok) return p;
    if (batch->enc || batch->draft_pass != 0 || batch->n_mm_rows > 0 || rope_mixed(batch) ||
        batch->n_ahead <= 0)
        return p;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t n_tok = batch->n_tok, Pn = batch->n_seq - D;
    if (Pn <= 0) return p;
    /* b - s = 0 mod G: a non-final chunk starts and ends on the scheduler's quantum, a multiple of
     * the delta net's chunk (scheduler.cpp:666-688, geometry.cpp:62-108), and n_tok - b is rounded
     * to G here -- so the split scan's two halves keep the single conv-prep/kkt pass's tiles. The
     * exact tail is therefore T .. T+G-1 rows. */
    const int64_t G = k.tile;
    int64_t b = batch->n_ahead >= c.tail ? n_tok
                                         : n_tok - (c.tail - batch->n_ahead + G - 1) / G * G;
    if (c.force_split) b = n_tok - c.force_split;
    b = std::min(std::max<int64_t>(b + c.shift_b, 0), n_tok);
    const int64_t q_prefill = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_prefill
                                                              : batch->max_q_len;
    const int64_t s_lb = std::max(DT, n_tok - q_prefill);
    if (b <= s_lb) return p;
    p.b = b;
    p.s_lb = s_lb;
    p.split = b < n_tok;
    p.path = c.mode == MODE_SPEED && D == 0 && Pn == 1 && b == n_tok ? PATH_LEAN : PATH_MASKED;
    p.alt = p.path == PATH_MASKED &&
            (c.force_alt || (c.stage == STAGE_AUTO && n_tok - (b - s_lb) <= c.stage_rows));
    return p;
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

/* qwen4exp_fp8.cpp:1407-1425 for one layer -- with the MoE on `arm` when one is given (layer S-1
 * of a pass on the alternates: stock rows, the alternate down handle). */
static void layer(RadCtx* c, qwen4exp_fp8::Model& m, int64_t li, const RadBatch* batch,
                  const MoeArm* arm = nullptr) {
    const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
    const int64_t T = batch->n_tok;
    if (li == m.ple_layer) m.ple.step(c, batch);
    l.hc_mix.read(c, T, 0, T);
    if (l.full) { l.qsa.step(c, l.attn.w.h, batch); l.attn.step(c, batch); }
    else        l.gdn.step(c, batch);
    l.hc_mix.write(c, T, 0, T);
    dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_h, m.a_x.x);
    l.hc_ffn.read(c, T, 0, T);
    if (arm) moe_layer(c, l.mlp, *arm, batch);
    else     l.mlp.step(c, batch);
    l.hc_ffn.write(c, T, 0, T);
    dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_h, m.a_x.x);
}

/* qwen4exp_fp8.cpp:1428-1437: the 97th connection and, when the chunk asks for them, logits. On an
 * approximate chunk they are requested only on exact rows in production (PLAN-FIX §6.2). */
static void epilogue(RadCtx* c, qwen4exp_fp8::Model& m, const RadBatch* batch) {
    const int64_t T = batch->n_tok;
    m.mixer.read(c, T, 0, T);
    if (batch->n_out <= 0) return;
    RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.a_x.x, T),
                praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.b_hout, batch->n_out));
    RAD_ISSUE_N(c, m.op_logits, batch->n_out, brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                brows(m.b_logits, batch->n_out));
}

/* Once a masked pass, after layer S-1 (PLAN-FIX §3.1-2): the device mask and bounds, then the
 * layer-S stream of the bulk superset copied into h_S, which every projector reads from here on
 * (b_h's bulk rows become a stream nobody reads). */
static void mask_rows(RadCtx* c, const Kva& k, qwen4exp_fp8::Model& m, const RadBatch* batch,
                      const Pass& p) {
    const int64_t T = batch->n_tok;
    RAD_ISSUE_N(c, k.op_mask, p.b, praw(batch->cu_seqlens + batch->n_seq - 1, RAD_I32, 2),
                praw(batch->token_ids, RAD_I32, p.b), praw(batch->positions, RAD_I32, p.b),
                k.mask_scored ? RAD_W(k.score) : RAD_NONE, brows(k.b_mask, T), brows(k.b_bounds, 4));
    if (!k.op_cast) return;
    const int64_t wide = m.hccfg.hc * m.g.n_embd, rows = p.b - p.s_lb;
    RAD_ISSUE_N(c, k.op_cast, rows, brow_slice(m.b_h, p.s_lb, rows, wide),
                brow_slice(k.b_hs, p.s_lb, rows, wide));
}

static void approximate_step(RadCtx* c, const Kva& k, const RadBatch* batch, const Pass& p,
                             StateDump* sd) {
    const int rank = rad_rank(c);
    qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    prologue(c, m, batch);
    const int64_t last_stock = k.split - 1;
    for (int64_t li = 0; li < last_stock; ++li) layer(c, m, li, batch);
    if (last_stock >= 0) {
        const MoeArm alt{ k.alt_dn[(size_t)last_stock], 0, 0 };
        layer(c, m, last_stock, batch, p.alt ? &alt : nullptr);
    }
    if (rank == 0 && !k.dump_dir.empty())
        dump_boundary(c, k.dump_dir, m.b_h, m.hccfg.hc * m.g.n_embd, batch);
    if (p.path == PATH_MASKED) {
        mask_rows(c, k, m, batch, p);
        if (rank == 0 && !k.dump_dir.empty())
            dump_mask(c, k.dump_dir, k.b_mask, k.b_bounds, batch, p.b, p.s_lb);
    }
    for (int64_t li = k.split; li < m.g.n_layer; ++li) {
        if (p.path == PATH_LEAN) fill_layer(c, k, m, li, batch, sd);
        else                     masked_layer(c, k, m, li, batch, p, sd);
    }
    epilogue(c, m, batch);
}

/* ONE LINE AN APPROXIMATE STEP, rank 0: scripts/grade.sh counts them against the bulk-chunk count
 * (R18/R57), which is what proves no bulk chunk silently ran exact; the fields are every keyed
 * number the decision read. */
static void log_pass(const Kva& k, const RadBatch* batch, const Pass& p) {
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    std::fprintf(stderr, "radiance: qwen4exp_kva: kva: approximate step (%s, %lld tokens, %lld ahead, "
                         "b %lld, s_lb %lld, D %lld, Pn %lld, ckpt %d, %s, stage %s%s)\n",
                 kModeNames[k.cfg.mode], (long long)batch->n_tok, (long long)batch->n_ahead,
                 (long long)p.b, (long long)p.s_lb, (long long)D, (long long)(batch->n_seq - D),
                 batch->n_checkpoints, kPathNames[p.path], p.alt ? "alt" : "stock",
                 p.split ? (k.cfg.straddle == STRADDLE_SPLIT ? ", split" : ", end") : "");
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
        const qwen4exp_fp8::Layer& l = m.layers[(size_t)li];
        if (ok && li == k.split) capture_rows(c, cap, "boundary", m.b_h, T, m.hccfg.hc * n);
        if (li == m.ple_layer) m.ple.step(c, batch);
        l.hc_mix.read(c, T, 0, T);
        if (ok && li >= k.split) {
            capture_rows(c, cap, "bi." + std::to_string(li), m.a_x.x, T, n);
            cap.layers.push_back((int)li);
        }
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch); l.attn.step(c, batch); }
        else        l.gdn.step(c, batch);
        l.hc_mix.write(c, T, 0, T);
        dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_h, m.a_x.x);
        l.hc_ffn.read(c, T, 0, T);
        l.mlp.step(c, batch);
        l.hc_ffn.write(c, T, 0, T);
        dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_h, m.a_x.x);
    }
    epilogue(c, m, batch);
    if (ok) capture_end(cap, k.split, n, m.hccfg.hc);
}

/* RADIANCE_KVA_CAPTURE_STATE: whatever late delta-net layer the step did not already copy (an
 * exact chunk copies here, after the step; nothing touches a layer's state after its scan), then
 * one file for this (chunk, rank). */
static void finish_state(RadCtx* c, const Kva& k, const RadBatch* batch, StateDump& sd, bool approx) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    for (int64_t li = k.split; li < m.g.n_layer; ++li)
        if (!m.layers[(size_t)li].full &&
            std::find(sd.layers.begin(), sd.layers.end(), (int)li) == sd.layers.end())
            read_state(c, k, m, li, batch, &sd);
    state_end(c, k.state_dir, batch, sd, m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k, rad_rank(c),
              m.g.world, approx, kModeNames[k.cfg.mode]);
}

/* A one-sequence step whose chunk does not end on the delta net's tile cannot be split exactly;
 * the scheduler never cuts one (derive's note), so this names a broken invariant rather than a
 * shape to serve. Forced splits are R47's debug arm and may be off the tile on purpose. */
static bool misaligned(const Kva& k, const RadBatch* batch, const Pass& p) {
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    return p.path == PATH_MASKED && batch->n_seq - D == 1 && !k.cfg.force_split &&
           (p.b - DT) % k.tile != 0;
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Kva& k = g_kva[rad_rank(c)];
    const Pass p = derive(k, batch);
    const bool approx = p.path != PATH_STOCK;
    if (approx && misaligned(k, batch, p)) {
        rad_step_fail(c, "qwen4exp_kva: the approximated chunk does not end on the delta net's "
                         "chunk tile, which the scheduler's quantum guarantees; refusing the step "
                         "rather than splitting a tile");
        return;
    }
    const bool capture = !k.capture_dir.empty() && rad_rank(c) == 0 && single_prefill(batch);
    const bool states = !k.state_dir.empty() && single_prefill(batch);
    StateDump sd;
    if (approx)       approximate_step(c, k, batch, p, states ? &sd : nullptr);
    else if (capture) capture_step(c, k, batch);
    else              qwen4exp_fp8::step(c, batch);
    if (approx && rad_rank(c) == 0) log_pass(k, batch, p);
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
RAD_ARCH_PLUGIN(qwen4exp_kva, "qwen4exp", "", "0.2.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next) with KVA / RidgeFill prefill: the in-tree "
                "qwen4exp_fp8 plugin plus projected late-layer cache fill (modes off, plumb, "
                "speed, quality; RADIANCE_KVA, default off)")
#endif
