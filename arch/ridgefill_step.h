/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* ridgefill_step.h -- the core's declare body and step: what a pass is, and the order its pieces run in.
 *
 * Model-free: everything the model owns -- its in-tree step, the exact layers, the embedding and the
 * logits, the debug captures that interleave with its blocks -- is a RidgeFillAdapter hook, and every number
 * a fact. The adapter's own declare runs the in-tree declare and fills RidgeFill::ad; its step forwards here.
 */
#ifndef RIDGEFILL_STEP_H
#define RIDGEFILL_STEP_H

#include <cstdlib>
#include <string>
#include <utility>

namespace ridgefill {

using namespace rad::arch;

static_assert((int)MODE_OFF == PLAN_OFF && (int)MODE_PLUMB == PLAN_PLUMB && (int)MODE_SPEED == PLAN_SPEED &&
              (int)MODE_QUALITY == PLAN_QUALITY, "ridgefill_config.h's Mode and ridgefill_plan.h's PlanMode share one order");

/* ================================================================== declare */

/* The container's ridgefill.mode is not a switch (PLAN-FIX §6.5): said once, by the real declare of
 * rank 0, so an operator who converted with --set ridgefill.mode=... learns why nothing changed. */
static void note_meta_mode(const Config& c, const RadBuildCtx* ctx) {
    if (!c.meta_mode || ctx->shape_probe || ctx->rank != 0) return;
    std::fprintf(stderr, "radiance: %s: the container's ridgefill.mode=%s is ignored; the mode "
                         "comes from RADIANCE_RIDGEFILL only (now %s)\n", g_log_name, c.meta_mode, kModeNames[c.mode]);
}

/* BEFORE THE MODEL'S GRAPH: the hazard op (ridgefill_hazard.h), in the modes that may issue it. radiance's
 * prefill stager stages a routed layer ahead only up to the highest op a pass of that kind issued before
 * (core/place/stager.cpp will_issue), and a stock pass issues this op after the model's whole step.
 * Declared after the graph, its handle carried that bound past the draft head's routed layer, and every
 * trunk pass of more than 1,024 tokens staged the head's experts for nothing: 50 layer stages against
 * stock's 49, +2.5% prefill (notes/stock-path-cost.md). Declared first, it is never a pass's highest op.
 * The adapter calls this before its model's declare and hands the handle to core_declare in
 * RidgeFill::hazard_first; when the projector then proves unusable the op is never issued, which is no
 * error (radiance abi/rad_builder.h rad_op_resolved). */
static rad_op core_declare_first(RadBuilder* b) {
    const char* v = env("RADIANCE_RIDGEFILL");
    const int mode = v ? pick(v, { "off", "plumb", "speed", "quality" }) : MODE_OFF;
    if (mode != MODE_SPEED && mode != MODE_QUALITY) return 0;
    return RAD_OP(b, "ridgefill_hazard", RAD_PARAMS(RAD_RANGE("M", 1, 1)), RAD_NOWEIGHTS);
}

/* Everything RidgeFill declares after the model's own graph, for the RidgeFill whose `ad` the adapter just filled.
 * `off` with no capture declares nothing at all: the in-tree graph, byte for byte (R6, R7). */
static int core_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx, RidgeFill& k) {
    g_log_name = k.ad.log_name;   /* a test's declare runs without rad_plugin_open */
    RAD_ARCH_TRY(read_config(meta, &k.cfg, k.ad.min_tail, k.ad.default_tail));
    note_meta_mode(k.cfg, ctx);
    for (auto [name, dir] : { std::pair<const char*, std::string*>{"RADIANCE_RIDGEFILL_DUMP", &k.dump_dir},
                              {"RADIANCE_RIDGEFILL_CAPTURE", &k.capture_dir},
                              {"RADIANCE_RIDGEFILL_CAPTURE_STATE", &k.state_dir},
                              {"RADIANCE_RIDGEFILL_DUMP_LOGITS", &k.logits_dir} }) {
        const char* v = std::getenv(name);
        *dir = v ? v : "";
    }
    /* The projector's fitting data comes from EXACT runs only (HANDOVER Stage 6.1). */
    if (!k.capture_dir.empty() && k.cfg.mode != MODE_OFF) {
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_CAPTURE records exact runs and the "
                             "mode is %s; run it with RADIANCE_RIDGEFILL=off\n", g_log_name, kModeNames[k.cfg.mode]);
        return RAD_E_INVAL;
    }
    const bool capturing = !k.capture_dir.empty() || !k.state_dir.empty();
    if (k.cfg.mode == MODE_OFF && !capturing) return RAD_OK;

    k.nm = Names(ctx->scope ? ctx->scope : "");
    k.tile = k.ad.tile;
    k.out_rows_ok = ctx->max_out_rows == 0 || k.cfg.score_bulk;
    for (auto* v : { &k.proj_w, &k.proj_b, &k.proj_s, &k.st })
        v->assign((size_t)k.ad.n_layer, RAD_NONE);
    for (auto* v : { &k.op_undo, &k.op_apply, &k.op_rho, &k.op_proj })
        v->assign((size_t)k.ad.n_layer, 0);
    if (k.cfg.mode == MODE_OFF) RAD_ARCH_TRY(capture_split(b, meta, k));
    else                        RAD_ARCH_TRY(decl_selected(b, meta, ctx, k));
    if (!k.state_dir.empty()) RAD_ARCH_TRY(decl_state_read(b, ctx, k));
    return RAD_OK;
}

/* ================================================================== step */

/* The stager lever needs `probe_depth` routed layers below S: the probes ride in layer S - probe_depth's
 * FFN (approximate_step). A dense model (no routed layer, probe_depth 0) never streams. */
inline bool can_stream(const RidgeFill& k) {
    const RidgeFillAdapter& a = k.ad;
    if (a.probe_depth <= 0 || k.split < a.probe_depth) return false;
    int64_t routed = 0;
    for (int64_t l = 0; l < k.split; ++l) routed += a.routed[(size_t)l] ? 1 : 0;
    return routed >= a.probe_depth;
}

/* WHAT THIS PASS IS: ridgefill_plan.h's rule over this pass's keyed fields (radiance core/runtime/ctx.cpp:
 * 989-995) and the declare's config, so a replayed pass issues what a fresh one would (R15/R99).
 * The host never needs the last sequence's first row s (device data); it knows the bulk END b and a
 * lower bound s_lb. Media steps, encoder and draft passes and the last chunks of a prompt run stock;
 * KL mode too unless SCORE_BULK says the tail alone is scored. */
static Pass derive(const RidgeFill& k, const RadBatch* batch) {
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
    in.stream_ok = can_stream(k) && (c.stage == STAGE_AUTO || c.force_stream);
    /* Past the exactness bound every late attention layer takes its per-row sparse form, whose rows the
     * straddle can restrict (the gated attention's own rule, rad_block_attn_gated_fp8.h:541-543). */
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

/* Once a masked pass, ahead of the layers (PLAN-FIX §3.1): the device mask, the bounds, and the zero
 * expert offsets the stager probes read. It reads only the batch, so it may run this early. */
static void mask_rows(RadCtx* c, const RidgeFill& k, const RadBatch* batch, const Pass& p) {
    RAD_ISSUE_N(c, k.op_mask, p.b, praw(batch->cu_seqlens + batch->n_seq - 1, RAD_I32, 2),
                praw(batch->token_ids, RAD_I32, p.b), praw(batch->positions, RAD_I32, p.b),
                k.mask_scored ? k.score : RAD_NONE, brows(k.b_mask, batch->n_tok),
                brows(k.b_bounds, 4), brows(k.b_zeros, k.n_zeros));
}

/* After layer S-1: the layer-S stream of the bulk superset copied into h_S, which every projector
 * reads from here on (the stream's bulk rows become a stream nobody reads). */
static void copy_stream(RadCtx* c, const RidgeFill& k, const Pass& p) {
    if (!k.op_cast || !k.b_hs) return;   /* int8 without the MTP map keeps no h_S (ridgefill_layer.h project_masked) */
    const int64_t wide = k.ad.wide, rows = p.b - p.s_lb;
    RAD_ISSUE_N(c, k.op_cast, rows, brow_slice(k.ad.buf_stream, p.s_lb, rows, wide),
                brow_slice(k.b_hs, p.s_lb, rows, wide));
}

/* AN APPROXIMATE PASS: the model's exact layers below S (one of them carrying the stager probes when
 * the pass streams), then the late layers on the pass's path, the final map, the model's epilogue. */
static void approximate_step(RadCtx* c, const RidgeFill& k, const RadBatch* batch, const Pass& p,
                             StateDump* sd) {
    const RidgeFillAdapter& a = k.ad;
    const int rank = rad_rank(c);
    a.prologue(c, batch);
    if (k.op_ring) ring_copy(c, k, k.split);   /* layer S's map lands during layers 0 .. S-1 */
    if (p.path != PATH_LEAN) mask_rows(c, k, batch, p);   /* masked, straddle, decoders: zeros + bounds */
    for (int64_t li = 0; li < k.split; ++li)
        a.stock_layer(c, k, li, batch, p.stream && li == k.split - a.probe_depth);
    if (rank == 0 && !k.dump_dir.empty())
        dump_boundary(c, k.dump_dir, a.buf_stream, a.wide, batch);
    if (p.path == PATH_MASKED) {
        copy_stream(c, k, p);
        if (rank == 0 && !k.dump_dir.empty())
            dump_mask(c, k.dump_dir, k.b_mask, k.b_bounds, batch, p.b, p.s_lb);
    }
    for (int64_t li = k.split; li < a.n_layer; ++li) {
        if (p.path == PATH_LEAN)          fill_layer(c, k, li, batch, p, sd);
        else if (p.path == PATH_STRADDLE) straddle_layer(c, k, li, batch, p, sd);
        else if (p.path == PATH_DECODERS) decoders_layer(c, k, li, batch, p, sd);
        else                              masked_layer(c, k, li, batch, p, sd);
    }
    final_stream(c, k, batch, p);   /* MTP: the bulk rows' predicted final stream, before the epilogue reads it */
    a.epilogue(c, batch);
}

/* ONE LINE AN APPROXIMATE STEP, rank 0: scripts/grade.sh counts them against the bulk-chunk count
 * (R18/R57), which is what proves no bulk chunk silently ran exact; the fields are every keyed
 * number the decision read. */
static void log_pass(const RidgeFill& k, const RadBatch* batch, const Pass& p) {
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    std::fprintf(stderr, "radiance: %s: ridgefill: approximate step (%s, %lld tokens, %lld ahead, "
                         "b %lld, s_lb %lld, D %lld, Pn %lld, ckpt %d, %s, stage %s%s)\n", g_log_name,
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

/* A one-sequence step whose chunk does not end on the recurrent block's tile cannot be split
 * exactly; the scheduler never cuts one (derive's note), so this names a broken invariant rather than a
 * shape to serve. Forced splits are R47's debug arm and may be off the tile on purpose. */
static bool misaligned(const RidgeFill& k, const RadBatch* batch, const Pass& p) {
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    return p.path != PATH_LEAN && batch->n_seq - D == 1 &&
           !k.cfg.force_split && (p.b - DT) % k.tile != 0;
}

/* The step: the approximate pass, the capture, or the model's own step, then the instruments. Every
 * branch is decided from keyed fields and declare-time state (R15/R99). */
static void core_step(RadCtx* c, const RadBatch* batch) {
    const RidgeFill& k = g_ridgefill[rad_rank(c)];
    const RidgeFillAdapter& a = k.ad;
    const Pass p = derive(k, batch);
    const bool approx = p.path != PATH_STOCK;
    if (approx && misaligned(k, batch, p)) {
        const std::string why = std::string(g_log_name) + ": the approximated chunk does not end on the "
                                "delta net's chunk tile, which the scheduler's quantum guarantees; refusing "
                                "the step rather than splitting a tile";
        rad_step_fail(c, why.c_str());
        return;
    }
    const bool capture = !k.capture_dir.empty() && rad_rank(c) == 0 && single_prefill(batch) && a.capture_step;
    const bool states = !k.state_dir.empty() && single_prefill(batch) && a.finish_state;
    const bool mixed_states = !k.state_dir.empty() && batch->phase == RAD_PHASE_MIXED && a.capture_mixed;
    StateDump sd;
    if (approx)       approximate_step(c, k, batch, p, states ? &sd : nullptr);
    else if (capture) a.capture_step(c, k, batch);
    else              a.stock_step(c, batch);
    if (approx && rad_rank(c) == 0) log_pass(k, batch, p);
    hazard_issue(c, k, batch, p);
    hazard_log(c);
    if (states) a.finish_state(c, k, batch, sd, approx);
    if (mixed_states) a.capture_mixed(c, k, batch, approx);
    if (!k.logits_dir.empty() && batch->n_out > 0 && batch->draft_pass == 0)
        dump_logits(c, k.logits_dir, batch, a.buf_logits, a.n_vocab, rad_rank(c));
}

}  /* namespace ridgefill */

#endif /* RIDGEFILL_STEP_H */
