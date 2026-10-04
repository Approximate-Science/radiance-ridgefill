/* kva_declare.h -- the KVA plugin's declare side: its state, the fitted tensors, the ops it adds
 * to the in-tree graph, and every refusal a mode makes before anything runs (R31).
 *
 * Everything here runs at declare only. step() reads the Kva a rank's real declare filled and
 * nothing else that can change (R15).
 */
#ifndef QWEN4EXP_KVA_DECLARE_H
#define QWEN4EXP_KVA_DECLARE_H

#include "kva_config.h"

#include <string>
#include <vector>

namespace qwen4exp_kva {

using namespace rad::arch;

struct Kva {
    Config  cfg{};
    /* Its own name pool: declared names must outlive declare (rad_arch.h's Names), and a sizing
     * declare runs on its own thread, so it must not append to the real model's pool. */
    Names   nm{""};
    int64_t split = -1;          /* S: the lowest projected layer, -1 when no projector is held */
    bool    have_proj = false, have_st = false, have_rowsel = false;
    std::vector<rad_weight> proj_w, proj_b;   /* [n_layer]: 0 below S */
    std::vector<rad_weight> st;               /* [n_layer]: 0 below S and on attention layers */
    rad_weight score = 0;
    /* kva_state_correct per late delta-net layer (Stage 4 issues them), and kva_rowsel (Stage 5).
     * Declared now because "kva.so is missing" has to be a refusal at startup (R31), and the only
     * way to ask whether an op resolves is to declare it. */
    std::vector<rad_op> op_undo, op_apply;
    rad_op  op_rowsel = 0;
    /* What each sequence's late delta-net layer had added to its state at its last approximate
     * chunk end (PLAN D7): one f32 a head, zeroed by the engine at admission. */
    rad_kvgroup kv_applied = 0;
    /* Quality mode's running decay sums N and D per head (kva_rho_update), same lifetime. */
    rad_kvgroup kv_rho = 0;
    /* The fill (Stage 3): a projector GEMM per late layer, and the quantiser that writes the
     * projected block input's codes. */
    std::vector<rad_op> op_proj;              /* [n_layer]: 0 below S */
    QuantFP8 quant{};
    /* Quality mode (Stage 5): the selected rows, their compacted streams, the row moves, and the
     * decay sums per late delta-net layer. */
    rad_buf b_rows = 0, b_mask = 0, b_hr = 0, b_xr = 0, b_injr = 0, b_yr = 0;
    rad_op  op_gather_h = 0, op_scatter_x = 0, op_gather_y = 0;
    std::vector<rad_op> op_rho;               /* [n_layer]: late delta-net layers, have_st only */
    std::string dump_dir;                     /* RADIANCE_KVA_DUMP, empty when unset */
    /* Stage 6 captures (debug; notes/arch.md "Capture"): RADIANCE_KVA_CAPTURE and
     * RADIANCE_KVA_CAPTURE_STATE, and the copy op the state capture needs. */
    std::string capture_dir, state_dir;
    rad_buf     b_state = 0;
    rad_op      op_state_read = 0;
};

static Kva g_kva[MAX_RANKS];

/* `name` declared when the model holds it; 0 when it does not. The name map comes first because a
 * checkpoint is searched through it (rad_weight_encoding); a container is searched by the declared
 * name, which is the same string. A map for an absent tensor is inert: name maps are read only for
 * declared weights (radiance core/format/checkpoint.cpp:525-541). */
static rad_weight decl_held(RadBuilder* b, Names& nm, const char* name, uint32_t dtype,
                            std::initializer_list<int64_t> shape, int shard, RadWeightGroup grp) {
    const char* d = nm.f("%s", name);
    if (map_copy(b, d, nm.ckpt("%s", name)) < 0) return 0;
    RadEncoding e{};
    if (!weight_enc(b, name, &e)) return 0;
    return decl_w(b, d, dtype, shape, RAD_ACCESS_PER_TOKEN, shard, grp, 1);
}

/* One projector copy (`kva.proj` or `kva.projr`): [n_embd, hc*n_embd] bf16 and its bias, replicated
 * on every rank (PLAN D5), PER_TOKEN so the planner keeps it resident (HANDOVER Stage 2.2). S is
 * the lowest layer held, and every layer from S up must be held with its bias -- a projector with
 * a hole is refused rather than run with one layer computed exactly by accident. */
static int decl_projector(RadBuilder* b, Names& nm, const qwen4exp_fp8::Model& m,
                          const char* base, Kva& k) {
    const int64_t n = m.g.n_embd, wide = m.hccfg.hc * m.g.n_embd;
    for (int64_t l = 0; l < m.g.n_layer; ++l) {
        char wn[96], bn[96];
        std::snprintf(wn, sizeof wn, "%s.%lld.weight", base, (long long)l);
        std::snprintf(bn, sizeof bn, "%s.%lld.bias", base, (long long)l);
        const rad_weight w = decl_held(b, nm, wn, RAD_BF16, {n, wide}, RAD_SHARD_NONE, grp_layer((int)l));
        const rad_weight bias = decl_held(b, nm, bn, RAD_BF16, {n}, RAD_SHARD_NONE, grp_layer((int)l));
        if (k.split < 0 && (w || bias)) k.split = l;
        if (k.split < 0) continue;
        if (!w || !bias) {
            std::fprintf(stderr, "radiance: qwen4exp_kva: the projector '%s' starts at layer %lld "
                                 "and has no %s; every layer from S to the last needs its weight "
                                 "and bias\n", base, (long long)k.split, w ? bn : wn);
            return RAD_E_INVAL;
        }
        k.proj_w[(size_t)l] = w;
        k.proj_b[(size_t)l] = bias;
    }
    k.have_proj = k.split >= 0;
    return RAD_OK;
}

/* One correction copy (`kva.st`, `kva.stswap` or `kva.str`): [value heads, head_v, head_k] f32 a
 * late delta-net layer -- the state's own layout, kv_gdn_state -- for ALL of the model's value heads.
 * REPLICATED, NOT ROW-SHARDED (deviation from PLAN D6, found at the first TP2 load): the loader
 * defines no ROW share of a rank-3 weight (radiance core/format/share.cpp:56), so each rank holds
 * the whole tensor (3 MiB a layer, 54 MiB a rank over 18 layers) and hands kva_state_correct its
 * own heads as a row slice of it at issue (correction_heads in qwen4exp_kva.cpp): rank r's value
 * heads are [r*H, (r+1)*H), the contiguous split the delta net's own weights take.
 * All of the late delta-net layers or none. */
static int decl_correction(RadBuilder* b, Names& nm, const qwen4exp_fp8::Model& m,
                           const char* base, int64_t from, Kva& k) {
    int held = 0, want = 0;
    for (int64_t l = from; l < m.g.n_layer; ++l) {
        if (m.layers[(size_t)l].full) continue;
        char sn[96];
        std::snprintf(sn, sizeof sn, "%s.%lld", base, (long long)l);
        k.st[(size_t)l] = decl_held(b, nm, sn, RAD_F32,
                                    {m.gcfg.n_head_v * m.g.world, m.gcfg.head_v, m.gcfg.head_k},
                                    RAD_SHARD_NONE, grp_layer((int)l));
        held += k.st[(size_t)l] != 0;
        ++want;
    }
    if (held && held != want) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: the correction '%s' covers %d of the %d "
                             "delta-net layers from %lld up; it is all of them or none\n",
                     base, held, want, (long long)from);
        return RAD_E_INVAL;
    }
    k.have_st = held > 0;
    return RAD_OK;
}

static rad_weight decl_score(RadBuilder* b, Names& nm, const qwen4exp_fp8::Model& m,
                             const char* name) {
    return decl_held(b, nm, name, RAD_F32, {m.g.n_vocab_all}, RAD_SHARD_NONE, grp_model());
}

/* rad-convert's view (RADIANCE_KVA_DECLARE=all): every copy the model holds, and nothing else --
 * no ops, no completeness checks, no refusals, because converting is not serving; the serving
 * declare checks what it selects. */
static int decl_every_copy(RadBuilder* b, const qwen4exp_fp8::Model& m, Kva& k) {
    const int64_t n = m.g.n_embd, wide = m.hccfg.hc * m.g.n_embd;
    char name[96];
    for (int64_t l = 0; l < m.g.n_layer; ++l) {
        const RadWeightGroup grp = grp_layer((int)l);
        for (const char* p : { "kva.proj", "kva.projr" }) {
            std::snprintf(name, sizeof name, "%s.%lld.weight", p, (long long)l);
            decl_held(b, k.nm, name, RAD_BF16, {n, wide}, RAD_SHARD_NONE, grp);
            std::snprintf(name, sizeof name, "%s.%lld.bias", p, (long long)l);
            decl_held(b, k.nm, name, RAD_BF16, {n}, RAD_SHARD_NONE, grp);
        }
        if (m.layers[(size_t)l].full) continue;
        for (const char* s : { "kva.st", "kva.stswap", "kva.str" }) {
            std::snprintf(name, sizeof name, "%s.%lld", s, (long long)l);
            decl_held(b, k.nm, name, RAD_F32,
                      {m.gcfg.n_head_v * m.g.world, m.gcfg.head_v, m.gcfg.head_k},
                      RAD_SHARD_NONE, grp);
        }
    }
    for (const char* t : { "kva.rowsel.score", "kva.rowsel.score_none", "kva.rowsel.score_all" })
        decl_score(b, k.nm, m, t);
    return RAD_OK;
}

/* THE APPLIED SCALE LIVES ON THE DEVICE, NOT THE HOST (PLAN D7): the tape audit fails a step whose
 * issues depend on host state that is not in the pass key, and "was a correction applied at this
 * sequence's last chunk end" is per-sequence, per-layer state. A LINEAR group gives each sequence
 * one slot the engine zeroes at admission and keeps for the sequence's life: [heads, 1, 1] f32,
 * bound to every late delta-net layer. kva_state_correct takes it with its own slot rows. */
static int decl_applied(RadBuilder* b, const qwen4exp_fp8::Model& m, Kva& k) {
    RadKVGroupDecl d{};
    d.kind         = RAD_KV_LINEAR;
    d.dtype        = RAD_F32;
    d.n_head_kv    = m.gcfg.n_head_v;
    d.state_dim[0] = 1;
    d.state_dim[1] = 1;
    k.kv_applied = rad_decl_kv_group(b, k.nm.f("kv_kva_applied"), &d);
    if (!k.kv_applied) return RAD_E_INVAL;
    for (int64_t l = k.split; l < m.g.n_layer; ++l)
        if (!m.layers[(size_t)l].full) RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, k.kv_applied));
    return RAD_OK;
}

/* The ops the mode issues from kva.so, declared per PLAN §5's schemas. A handle that comes back
 * null means no kernel library serves the op: kva.so is not on the search path, or declines this
 * machine. Refused here, by name, rather than at the first approximate chunk -- or, worse, served
 * without the correction (R31). */
static int decl_kernel_ops(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx,
                           Kva& k) {
    const bool probe = ctx->shape_probe != 0;
    const int64_t max_tok = ctx->max_tok;
    const Config& c = k.cfg;
    const char* missing = nullptr;
    const bool corrects = k.have_st && (c.mode == MODE_SPEED || c.mode == MODE_QUALITY);
    if (corrects) RAD_ARCH_TRY(decl_applied(b, m, k));
    for (int64_t l = k.split; corrects && l < m.g.n_layer; ++l) {
        if (m.layers[(size_t)l].full) continue;
        for (int apply = 0; apply < 2; ++apply) {
            const rad_op h = RAD_OP(b, "kva_state_correct",
                RAD_PARAMS(RAD_RANGE("M", 1, m.g.max_seqs), RAD_STR("mode", apply ? "apply" : "undo"),
                           RAD_F64("alpha", c.alpha), RAD_INT("n_head", m.gcfg.n_head_v),
                           RAD_INT("sd0", m.gcfg.head_v), RAD_INT("sd1", m.gcfg.head_k)),
                RAD_WEIGHTS(k.st[(size_t)l]));
            (apply ? k.op_apply : k.op_undo)[(size_t)l] = h;
            if (!h) missing = "kva_state_correct";
        }
    }
    if (c.mode == MODE_QUALITY) {
        k.op_rowsel = rw(b, RAD_OP(b, "kva_rowsel",
            RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("cap", c.cap),
                       RAD_F64("share", c.share), RAD_INT("seed", c.seed),
                       RAD_STR("mode", kRowselNames[c.rowsel])),
            RAD_WEIGHTS(k.score)), {}, {k.b_rows, k.b_mask});
        if (!k.op_rowsel) missing = "kva_rowsel";
    }
    /* The decay sums read the layer's own a|b columns and its own A_log / dt_bias. Under a sizing
     * declare these in-tree handles are the real declare's, which name the same weights (the
     * in-tree declare declares them in the same order at every max_tok; the static test checks
     * the weight lists agree). */
    for (int64_t l = k.split; k.kv_rho && l < m.g.n_layer; ++l) {
        const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
        if (lay.full) continue;
        k.op_rho[(size_t)l] = rw(b, RAD_OP(b, "kva_rho_update",
            RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("n_head", m.gcfg.n_head_v)),
            RAD_WEIGHTS(lay.gdn.w_a_log, lay.gdn.w_dt_bias)), {lay.gdn.w.ab, k.b_mask}, {});
        if (!k.op_rho[(size_t)l]) missing = "kva_rho_update";
    }
    if (missing && !probe) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: mode %s issues '%s' and no kernel library "
                             "serves it -- kva.so is missing from $RADIANCE_HOME or declines this "
                             "machine. Refusing rather than serving without it.\n",
                     kModeNames[c.mode], missing);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* The refusals a serving mode needs before anything is issued (R31). Each names the number or the
 * tensor that refused it. */
static int check_mode(const Kva& k, int64_t max_tok) {
    const Config& c = k.cfg;
    const char* mode = kModeNames[c.mode];
    if (!k.have_proj) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: mode %s needs the projector '%s.L.weight' "
                             "and the model holds none of it\n", mode, c.proj);
        return RAD_E_UNSUPPORTED;
    }
    /* n_ahead is capped at max_tok (radiance core/sched/batch.cpp:1066-1080), so a tail longer
     * than one step could never be satisfied and no chunk would ever be approximated. */
    if (c.tail > max_tok) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: kva.tail is %lld tokens and the largest step "
                             "is %lld (--max-num-batched-tokens); a chunk is approximated only "
                             "when %lld prompt tokens follow it, and the scheduler never reports "
                             "more than %lld. Lower the tail or raise the step.\n",
                     (long long)c.tail, (long long)max_tok, (long long)c.tail, (long long)max_tok);
        return RAD_E_UNSUPPORTED;
    }
    if (c.tail < kMinTail) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: kva.tail is %lld tokens; the shortest exact "
                             "tail this method was measured at is %lld\n",
                     (long long)c.tail, (long long)kMinTail);
        return RAD_E_INVAL;
    }
    if (c.mode == MODE_QUALITY && !k.have_rowsel) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: mode quality selects exact rows from the "
                             "table '%s' and the model does not hold it\n", c.score);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

static void note_config(RadBuilder* b, const Kva& k) {
    const Config& c = k.cfg;
    rad_note(b, "KVA: mode %s from layer %lld, tail %lld; projector %s, correction %s (alpha %g), "
                "row table %s, rows %s share %g cap %lld seed %lld",
             kModeNames[c.mode], (long long)k.split, (long long)c.tail, c.proj,
             k.have_st ? c.st : "absent", c.alpha, k.have_rowsel ? c.score : "absent",
             kRowselNames[c.rowsel], c.share, (long long)c.cap, (long long)c.seed);
}

/* What the fill assumes of the in-tree model, refused by name if it ever stops holding. */
static int check_fill(const qwen4exp_fp8::Model& m, const Kva& k) {
    if (m.ple_layer >= k.split) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: the projector starts at layer %lld and the "
                             "n-gram embedding enters the stream at layer %lld; a split at or below "
                             "it would predict from a stream that never received it\n",
                     (long long)k.split, (long long)m.ple_layer);
        return RAD_E_UNSUPPORTED;
    }
    for (int64_t l = k.split; l < m.g.n_layer; ++l) {
        const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
        if (lay.full ? lay.attn.ext_in : lay.gdn.ext_in) continue;
        std::fprintf(stderr, "radiance: qwen4exp_kva: layer %lld's block owns its input norm, and "
                             "the fill hands blocks their input already normed\n", (long long)l);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* The fill's own ops: one projector GEMM per late layer, writing the model's block input `x`
 * itself (the a_x decision, notes/arch.md §6 -- every in-tree op then runs on its stock operands),
 * and the quantiser that writes x's codes as the connection read would (QuantFP8 takes int8 or E4M3
 * off the model, nothing for a bf16 one). Both are declared after the whole in-tree graph, outside
 * the planned op ranges of the stream `b_h` they read and the `x` they write, so those buffers get
 * the whole program (rad_buf_concurrent, PLAN D4). Plumb declares none of it: it feeds the real `x`.
 * Under a sizing declare the in-tree buffer handles are the real declare's, which name the same
 * buffers: the in-tree declare declares them in the same order at every max_tok. */
static int decl_fill(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx, Kva& k) {
    if (!ctx->shape_probe) RAD_ARCH_TRY(check_fill(m, k));
    k.op_proj.assign(m.layers.size(), 0);
    if (k.cfg.mode == MODE_PLUMB) return RAD_OK;
    Geom g = m.g;
    g.max_tok = ctx->max_tok;
    for (rad_buf h : { m.b_h, m.a_x.x, m.a_x.cq(), m.a_x.cs() })
        if (h) RAD_ARCH_TRY(rad_buf_concurrent(b, h));
    RAD_ARCH_TRY(k.quant.declare(b, g, m.a_x, g.n_embd));
    const int64_t wide = m.hccfg.hc * g.n_embd;
    for (int64_t l = k.split; l < g.n_layer; ++l) {
        const rad_op h = rw(b, RAD_OP(b, "gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                           RAD_INT("K", wide), RAD_STR("dtype", g.dtype)),
                                RAD_WEIGHTS(k.proj_w[(size_t)l], k.proj_b[(size_t)l])),
                            {m.b_h}, {m.a_x.x});
        if (!h && !ctx->shape_probe) {
            std::fprintf(stderr, "radiance: qwen4exp_kva: no kernel serves the projector "
                                 "(gemm_nt_bias, N %lld, K %lld, %s)\n",
                         (long long)g.n_embd, (long long)wide, g.dtype);
            return RAD_E_UNSUPPORTED;
        }
        k.op_proj[(size_t)l] = h;
    }
    return RAD_OK;
}

static rad_kvgroup decl_rho(RadBuilder* b, const qwen4exp_fp8::Model& m, Kva& k) {
    RadKVGroupDecl d{};
    d.kind         = RAD_KV_LINEAR;
    d.dtype        = RAD_F32;
    d.n_head_kv    = m.gcfg.n_head_v;
    d.state_dim[0] = 1;
    d.state_dim[1] = 2;            /* N, D */
    const rad_kvgroup g = rad_decl_kv_group(b, k.nm.f("kv_kva_rho"), &d);
    for (int64_t l = k.split; g && l < m.g.n_layer; ++l)
        if (!m.layers[(size_t)l].full && rad_bind_layer_kv(b, (int)l, g) < 0) return 0;
    return g;
}

/* QUALITY MODE'S EXACT ROWS (Stage 5, PLAN §3, D11-D14). Every compacted issue is at M = cap,
 * never at the selected count, which is device data the host never reads (D12): rows_idx is
 * -1 padded, a gather reads a zero row for -1 and a scatter skips it. The exact rows' wide stream
 * lives in its own buffer `h_R` [cap, hc*n_embd] across the late layers -- b_h must stay the
 * layer-S stream for every projector -- with its block input `x_R`, write gains `inj_R` and block
 * output `y_R`. All plugin-owned and whole-program (declared after the in-tree graph, PLAN D4), as
 * is the in-tree `gdn_ab` the decay sums read. The decay sums' group exists only with a correction
 * to scale. */
static int decl_quality(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx, Kva& k) {
    k.op_rho.assign(m.layers.size(), 0);
    if (k.cfg.mode != MODE_QUALITY) return RAD_OK;
    const int64_t cap = k.cfg.cap, n = m.g.n_embd, hc = m.hccfg.hc;
    const uint32_t act = m.g.act_dtype;
    k.b_rows = decl_b(b, k.nm.f("kva_rows_idx"), RAD_I32, {cap});
    k.b_mask = decl_b(b, k.nm.f("kva_mask"),     RAD_I32, {ctx->max_tok});
    k.b_hr   = decl_b(b, k.nm.f("kva_h_rows"),   act,     {cap, hc * n});
    k.b_xr   = decl_b(b, k.nm.f("kva_x_rows"),   act,     {cap, n});
    k.b_injr = decl_b(b, k.nm.f("kva_inj_rows"), act,     {cap, hc});
    k.b_yr   = decl_b(b, k.nm.f("kva_y_rows"),   act,     {cap, n});
    for (rad_buf h : { k.b_rows, k.b_mask, k.b_hr, k.b_xr, k.b_injr, k.b_yr, m.b_ab }) {
        if (!h) return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_buf_concurrent(b, h));
    }
    auto rows_op = [&](const char* op, int64_t width) {
        return RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, cap), RAD_INT("n", width),
                                        RAD_STR("dtype", m.g.dtype)), RAD_NOWEIGHTS);
    };
    k.op_gather_h  = rw(b, rows_op("gather_rows", hc * n), {m.b_h, k.b_rows}, {k.b_hr});
    k.op_scatter_x = rw(b, rows_op("scatter_rows", n), {k.b_xr, k.b_rows, m.a_x.x}, {m.a_x.x});
    k.op_gather_y  = rw(b, rows_op("gather_rows", n), {m.a_x.x, k.b_rows}, {k.b_yr});
    if (!ctx->shape_probe && (!k.op_gather_h || !k.op_scatter_x || !k.op_gather_y)) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: quality mode moves rows with gather_rows and "
                             "scatter_rows, and no kernel serves them at %lld rows\n", (long long)cap);
        return RAD_E_UNSUPPORTED;
    }
    if (k.have_st && !(k.kv_rho = decl_rho(b, m, k))) return RAD_E_INVAL;
    return RAD_OK;
}

/* WHERE THE LATE LAYERS START FOR A CAPTURE, which in `off` mode nothing else has asked: the lowest
 * projector layer the container holds -- what every other mode calls S -- else its `kva.split`
 * metadata. Probing adds only name maps, which are inert. */
static int capture_split(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadModelMeta* meta, Kva& k) {
    for (int64_t l = 0; l < m.g.n_layer && k.split < 0; ++l) {
        char name[96];
        std::snprintf(name, sizeof name, "kva.proj.%lld.weight", (long long)l);
        if (map_copy(b, k.nm.f("%s", name), k.nm.ckpt("%s", name)) < 0) return RAD_E_INVAL;
        RadEncoding e{};
        if (weight_enc(b, name, &e)) k.split = l;
    }
    if (k.split < 0) k.split = rad_meta_geti(meta, "kva.split", -1);
    if (k.split <= m.ple_layer || k.split >= m.g.n_layer) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: a capture needs the split layer S: the container "
                             "holds no kva.proj.* and its kva.split is %lld\n", (long long)k.split);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* THE STATE CAPTURE'S COPY. No ABI call returns a KV-pool pointer (RADIANCE-FACTS §5), so a late
 * layer's slot is copied by kva.so's kva_state_read into this rank's [1, heads, V, K] buffer and
 * read from there. Declared only when RADIANCE_KVA_CAPTURE_STATE is set. */
static int decl_state_read(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx, Kva& k) {
    const GdnFP8::Config& g = m.gcfg;
    k.b_state = decl_b(b, k.nm.f("kva_state_copy"), RAD_F32, {1, g.n_head_v, g.head_v, g.head_k});
    if (!k.b_state) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_buf_concurrent(b, k.b_state));
    k.op_state_read = rw(b, RAD_OP(b, "kva_state_read",
                                RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n_head", g.n_head_v),
                                           RAD_INT("sd0", g.head_v), RAD_INT("sd1", g.head_k)),
                                RAD_NOWEIGHTS), {}, {k.b_state});
    if (!k.op_state_read && !ctx->shape_probe) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_CAPTURE_STATE copies the delta-net "
                             "state with kva_state_read, and no kernel library serves it -- kva.so is "
                             "missing or too old\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* The selected set, its ops and its refusals. Under a sizing declare the cap is the real
 * declare's -- a fixed op parameter that followed max_tok would cost the arena its level -- and
 * nothing is refused: the real declare already decided. */
static int decl_selected(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx,
                         Kva& k) {
    RAD_ARCH_TRY(decl_projector(b, k.nm, m, k.cfg.proj, k));
    /* plumb issues no correction, so it does not place one */
    if (k.have_proj && k.cfg.mode != MODE_PLUMB)
        RAD_ARCH_TRY(decl_correction(b, k.nm, m, k.cfg.st, k.split, k));
    if (k.cfg.mode == MODE_QUALITY) k.score = decl_score(b, k.nm, m, k.cfg.score);
    k.have_rowsel = k.score != 0;
    if (ctx->shape_probe) k.cfg.cap = g_kva[ctx->rank].cfg.cap;
    else RAD_ARCH_TRY(check_mode(k, m.g.max_tok));
    RAD_ARCH_TRY(decl_fill(b, m, ctx, k));
    RAD_ARCH_TRY(decl_quality(b, m, ctx, k));
    RAD_ARCH_TRY(decl_kernel_ops(b, m, ctx, k));
    note_config(b, k);
    return RAD_OK;
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_DECLARE_H */
