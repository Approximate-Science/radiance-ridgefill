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
 * over all rows with the device mask choosing which rows use the projection, kva_layer.h). On a
 * masked pass the expert stager is steered, with zero-row probes of the late layers' gate-up GEMMs,
 * to stream the late layers' routed experts instead of staging each layer whole (notes/impl.md §2). The engine release
 * is checked at open and a mismatch forwards to the engine's own architecture (kva_guard.h).
 *
 * Included by tests/arch_static_test.cpp too, which defines RAD_ARCH_NO_EXPORTS itself; then
 * neither plugin's exports are emitted and the test calls both namespaces directly.
 */
#ifndef RAD_ARCH_NO_EXPORTS
#define RAD_ARCH_NO_EXPORTS 1
#define QWEN4EXP_KVA_EXPORTS 1
#endif
#include <qwen4exp_fp8/qwen4exp_fp8.cpp>

#include "kva_plan.h"
#include "kva_declare.h"
#include "kva_dump.h"
#include "kva_fill.h"
#include "kva_moe.h"
#include "kva_layer.h"
#include "kva_hazard.h"
#include "kva_guard.h"

namespace qwen4exp_kva {

using namespace rad::arch;

/* The in-tree file this plugin is built to shadow (arch/CMakeLists.txt's stem): the guard forwards to
 * it on a release mismatch. */
constexpr const char* kShadowSo = "qwen4exp_fp8.so";

static_assert((int)MODE_OFF == PLAN_OFF && (int)MODE_PLUMB == PLAN_PLUMB && (int)MODE_SPEED == PLAN_SPEED &&
              (int)MODE_QUALITY == PLAN_QUALITY, "kva_config.h's Mode and kva_plan.h's PlanMode share one order");

/* The container's kva.mode is not a switch (PLAN-FIX §6.5): said once, by the real declare of
 * rank 0, so an operator who converted with --set kva.mode=... learns why nothing changed. */
static void note_meta_mode(const Config& c, const RadBuildCtx* ctx) {
    if (!c.meta_mode || ctx->shape_probe || ctx->rank != 0) return;
    std::fprintf(stderr, "radiance: qwen4exp_kva: the container's kva.mode=%s is ignored; the mode "
                         "comes from RADIANCE_KVA only (now %s)\n", c.meta_mode, kModeNames[c.mode]);
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (g_forward.declare) return g_forward.declare(b, meta, ctx);
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
                              {"RADIANCE_KVA_CAPTURE_STATE", &k.state_dir},
                              {"RADIANCE_KVA_DUMP_LOGITS", &k.logits_dir} }) {
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
    if (k.cfg.mode == MODE_OFF && !capturing) return RAD_OK;

    k.nm = Names(ctx->scope ? ctx->scope : "");
    k.tile = m.gcfg.chunk;
    k.out_rows_ok = ctx->max_out_rows == 0 || k.cfg.score_bulk;
    for (auto* v : { &k.proj_w, &k.proj_b, &k.proj_s, &k.st })
        v->assign(m.layers.size(), RAD_NONE);
    for (auto* v : { &k.op_undo, &k.op_apply, &k.op_rho, &k.op_proj })
        v->assign(m.layers.size(), 0);
    if (k.cfg.mode == MODE_OFF) RAD_ARCH_TRY(capture_split(b, m, meta, k));
    else                        RAD_ARCH_TRY(decl_selected(b, meta, m, ctx, k));
    if (!k.state_dir.empty()) RAD_ARCH_TRY(decl_state_read(b, m, ctx, k));
    return RAD_OK;
}

/* ================================================================== step */

/* WHAT THIS PASS IS: kva_plan.h's rule over this pass's keyed fields (radiance core/runtime/ctx.cpp:
 * 989-995) and the declare's config, so a replayed pass issues what a fresh one would (R15/R99).
 * The host never needs the last sequence's first row s (device data); it knows the bulk END b and a
 * lower bound s_lb. Media steps, encoder and draft passes and the last chunks of a prompt run stock;
 * KL mode too unless SCORE_BULK says the tail alone is scored. */
static Pass derive(const Kva& k, const RadBatch* batch) {
    const Config& c = k.cfg;
    PlanIn in;
    in.mode = c.mode;   /* Mode and PlanMode share their order: off, plumb, speed, quality */
    in.eligible = k.have_proj && k.out_rows_ok && !batch->enc && batch->draft_pass == 0 &&
                  batch->n_mm_rows <= 0 && !rope_mixed(batch);
    batch_split(batch, &in.n_seq_decode, &in.n_tok_decode);
    in.n_tok = batch->n_tok;
    in.n_seq = batch->n_seq;
    in.q_prefill = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_prefill : batch->max_q_len;
    in.n_ahead = batch->n_ahead;
    in.n_checkpoints = batch->n_checkpoints;
    /* The probes ride in layer S-3's MoE (approximate_step), so the lever needs three routed layers
     * below S. */
    in.stream_ok = k.split >= 3 && (c.stage == STAGE_AUTO || c.force_stream);
    /* rad_block_attn_gated_fp8.h:502-504: past the exactness bound every late attention layer takes
     * its per-row sparse form, whose rows the straddle can restrict. */
    const int64_t reach = (int64_t)batch->max_ctx_len + batch->max_q_len;
    in.straddle_ok = k.straddle_layers && (k.qsa_exact_to <= 0 || reach > k.qsa_exact_to);
    PlanConfig pc;
    pc.tail = c.tail;
    pc.tile = k.tile;
    pc.force_split = c.force_split;
    pc.shift_b = c.shift_b;
    pc.stage_rows = c.stage_rows;
    pc.min_bulk_rows = c.min_bulk_rows;
    pc.ckpt_floor = c.ckpt_floor;
    pc.force_stream = c.force_stream;
    pc.mask_step = c.mask_step;
    return plan_pass(in, pc);
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

/* qwen4exp_fp8.cpp:1407-1425 for one layer -- with the MoE issued through `arm` when one is given
 * (layer S-3 of a streaming pass: stock rows, the stager probes behind its gate-up GEMM). */
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

/* Once a masked pass, ahead of the layers (PLAN-FIX §3.1): the device mask, the bounds, and the zero
 * expert offsets the stager probes read. It reads only the batch, so it may run this early. */
static void mask_rows(RadCtx* c, const Kva& k, const RadBatch* batch, const Pass& p) {
    RAD_ISSUE_N(c, k.op_mask, p.b, praw(batch->cu_seqlens + batch->n_seq - 1, RAD_I32, 2),
                praw(batch->token_ids, RAD_I32, p.b), praw(batch->positions, RAD_I32, p.b),
                k.mask_scored ? k.score : RAD_NONE, brows(k.b_mask, batch->n_tok),
                brows(k.b_bounds, 4), brows(k.b_zeros, k.n_zeros));
}

/* After layer S-1: the layer-S stream of the bulk superset copied into h_S, which every projector
 * reads from here on (b_h's bulk rows become a stream nobody reads). */
static void copy_stream(RadCtx* c, const Kva& k, qwen4exp_fp8::Model& m, const Pass& p) {
    if (!k.op_cast || !k.b_hs) return;   /* int8 without the MTP map keeps no h_S (kva_layer.h project_masked) */
    const int64_t wide = m.hccfg.hc * m.g.n_embd, rows = p.b - p.s_lb;
    RAD_ISSUE_N(c, k.op_cast, rows, brow_slice(m.b_h, p.s_lb, rows, wide),
                brow_slice(k.b_hs, p.s_lb, rows, wide));
}

/* THE STAGER LEVER (notes/impl.md §2): behind layer S-3's gate-up GEMM, while the stager holds layers
 * S-3 and S-2 in its two buffers and has released neither, probe the gate-up of every layer from S-1
 * to the last. Each probe makes the stager start the layer after it, and a start that finds both
 * buffers held STREAMS that layer: from here on no late layer is staged whole, and each reads only
 * the experts its exact rows route to. Layer S-1 itself, which runs every row, is still staged when
 * the real pass reaches layer S-2. Correctness never depends on any of it: a probe moves no number. */
static MoeArm probe_arm(RadCtx* c, const Kva& k, const qwen4exp_fp8::Model& m) {
    MoeArm arm;
    arm.after_gate_up = [c, &k, &m] {
        for (int64_t x = k.split - 1; x < m.g.n_layer; ++x)
            moe_probe(c, m.layers[(size_t)x].mlp, k.b_zeros, k.b_probe);
    };
    return arm;
}

static void approximate_step(RadCtx* c, const Kva& k, const RadBatch* batch, const Pass& p,
                             StateDump* sd) {
    const int rank = rad_rank(c);
    qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    prologue(c, m, batch);
    if (k.op_ring) ring_copy(c, k, k.split);   /* layer S's map lands during layers 0 .. S-1 */
    if (p.path != PATH_LEAN) mask_rows(c, k, batch, p);   /* masked, straddle, decoders: zeros + bounds */
    const MoeArm probes = probe_arm(c, k, m);
    for (int64_t li = 0; li < k.split; ++li)
        layer(c, m, li, batch, p.stream && li == k.split - 3 ? &probes : nullptr);
    if (rank == 0 && !k.dump_dir.empty())
        dump_boundary(c, k.dump_dir, m.b_h, m.hccfg.hc * m.g.n_embd, batch);
    if (p.path == PATH_MASKED) {
        copy_stream(c, k, m, p);
        if (rank == 0 && !k.dump_dir.empty())
            dump_mask(c, k.dump_dir, k.b_mask, k.b_bounds, batch, p.b, p.s_lb);
    }
    for (int64_t li = k.split; li < m.g.n_layer; ++li) {
        if (p.path == PATH_LEAN)          fill_layer(c, k, m, li, batch, sd);
        else if (p.path == PATH_STRADDLE) straddle_layer(c, k, m, li, batch, p, sd);
        else if (p.path == PATH_DECODERS) decoders_layer(c, k, m, li, batch, p, sd);
        else                              masked_layer(c, k, m, li, batch, p, sd);
    }
    final_stream(c, k, m, batch, p);   /* MTP: the bulk rows' predicted final stream, before the epilogue reads b_h */
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
                 batch->n_checkpoints, kPathNames[p.path], p.stream ? "stream" : "stock",
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

/* R61 -- RADIANCE_KVA_CAPTURE_STATE ON A MIXED STEP (decoders beside a prefill chunk): after the
 * step, every sequence's late delta-net states, so the decoders' slots of two runs of one arrangement
 * (KVA and off) can be compared byte for byte; the prefill's slot is the positive control. */
static void capture_mixed(RadCtx* c, const Kva& k, const RadBatch* batch, bool approx) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
    std::vector<int> layers;
    for (int64_t li = k.split; li < m.g.n_layer; ++li)
        if (!m.layers[(size_t)li].full) layers.push_back((int)li);
    std::vector<float> data;
    for (int64_t seq = 0; seq < batch->n_seq; ++seq)
        for (int li : layers)
            if (!copy_state(c, k, m, li, batch, seq, &data)) return;
    mixed_state_end(c, k.state_dir, batch, data, layers, {m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k},
                    rad_rank(c), approx, kModeNames[k.cfg.mode]);
}

/* A one-sequence step whose chunk does not end on the delta net's tile cannot be split exactly;
 * the scheduler never cuts one (derive's note), so this names a broken invariant rather than a
 * shape to serve. Forced splits are R47's debug arm and may be off the tile on purpose. */
static bool misaligned(const Kva& k, const RadBatch* batch, const Pass& p) {
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    return p.path != PATH_LEAN && batch->n_seq - D == 1 &&
           !k.cfg.force_split && (p.b - DT) % k.tile != 0;
}

static void step(RadCtx* c, const RadBatch* batch) {
    if (g_forward.step) { g_forward.step(c, batch); return; }
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
    const bool mixed_states = !k.state_dir.empty() && batch->phase == RAD_PHASE_MIXED;
    StateDump sd;
    if (approx)       approximate_step(c, k, batch, p, states ? &sd : nullptr);
    else if (capture) capture_step(c, k, batch);
    else              qwen4exp_fp8::step(c, batch);
    if (approx && rad_rank(c) == 0) log_pass(k, batch, p);
    hazard_issue(c, k, batch, p);
    hazard_log(c);
    if (states) finish_state(c, k, batch, sd, approx);
    if (mixed_states) capture_mixed(c, k, batch, approx);
    if (!k.logits_dir.empty() && batch->n_out > 0 && batch->draft_pass == 0) {
        const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rad_rank(c)];
        dump_logits(c, k.logits_dir, batch, m.b_logits, m.g.n_vocab, rad_rank(c));
    }
}

/* The in-tree probe's answers hold here: the draft depth is the model's, and this declare writes
 * scratch under shape_probe exactly as the included one does. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (g_forward.probe) return g_forward.probe(meta, out);
    return qwen4exp_fp8::probe(meta, out);
}

}  /* namespace qwen4exp_kva */

#ifdef QWEN4EXP_KVA_EXPORTS
extern "C" int rad_plugin_open(void) { return qwen4exp_kva::open_guard(qwen4exp_kva::kShadowSo); }
extern "C" void rad_plugin_close(void) { qwen4exp_kva::free_uploads(); }
RAD_ARCH_PROBE(qwen4exp_kva)
RAD_ARCH_PLUGIN(qwen4exp_kva, "qwen4exp", "", "0.2.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next) with KVA / RidgeFill prefill: the in-tree "
                "qwen4exp_fp8 plugin plus projected late-layer cache fill (modes off, plumb, "
                "speed, quality; RADIANCE_KVA, default off)")
#endif
