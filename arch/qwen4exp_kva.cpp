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
 * WHAT IS DONE (Stage 2.2): the fitted tensors are declared, optional, by the names
 * tools/kva_sidecar.py writes (kva_config.h lists them and the switches that pick a copy); S is
 * the lowest projected layer the model holds; every refusal a mode needs is made at declare, by
 * name (R31). WHAT IS NOT: the fill path (Stage 3), the correction (Stage 4) and the exact rows
 * (Stage 5). Until they exist every mode but `off` refuses at declare -- a mode that served stock
 * output while saying `speed` would be the silent fallback HANDOVER §2.1 forbids.
 *
 * Included by tests/arch_static_test.cpp too, which defines RAD_ARCH_NO_EXPORTS itself; then
 * neither plugin's exports are emitted and the test calls both namespaces directly.
 */
#ifndef RAD_ARCH_NO_EXPORTS
#define RAD_ARCH_NO_EXPORTS 1
#define QWEN4EXP_KVA_EXPORTS 1
#endif
#include <qwen4exp_fp8/qwen4exp_fp8.cpp>

#include "kva_config.h"

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
 * late delta-net layer, row-sharded by head (PLAN D6) -- the state's own layout, kv_gdn_state.
 * All of the late delta-net layers or none. */
static int decl_correction(RadBuilder* b, Names& nm, const qwen4exp_fp8::Model& m,
                           const char* base, int64_t from, Kva& k) {
    int held = 0, want = 0;
    for (int64_t l = from; l < m.g.n_layer; ++l) {
        if (m.layers[(size_t)l].full) continue;
        char sn[96];
        std::snprintf(sn, sizeof sn, "%s.%lld", base, (long long)l);
        k.st[(size_t)l] = decl_held(b, nm, sn, RAD_F32,
                                    {m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k},
                                    RAD_SHARD_ROW, grp_layer((int)l));
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
            decl_held(b, k.nm, name, RAD_F32, {m.gcfg.n_head_v, m.gcfg.head_v, m.gcfg.head_k},
                      RAD_SHARD_ROW, grp);
        }
    }
    for (const char* t : { "kva.rowsel.score", "kva.rowsel.score_none", "kva.rowsel.score_all" })
        decl_score(b, k.nm, m, t);
    return RAD_OK;
}

/* The ops the mode issues from kva.so, declared per PLAN §5's schemas. A handle that comes back
 * null means no kernel library serves the op: kva.so is not on the search path, or declines this
 * machine. Refused here, by name, rather than at the first approximate chunk -- or, worse, served
 * without the correction (R31). */
static int decl_kernel_ops(RadBuilder* b, const qwen4exp_fp8::Model& m, bool probe, Kva& k) {
    const Config& c = k.cfg;
    const char* missing = nullptr;
    const bool corrects = k.have_st && (c.mode == MODE_SPEED || c.mode == MODE_QUALITY);
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
        k.op_rowsel = RAD_OP(b, "kva_rowsel",
            RAD_PARAMS(RAD_RANGE("M", 1, m.g.max_tok), RAD_INT("cap", c.cap),
                       RAD_F64("share", c.share), RAD_INT("seed", c.seed),
                       RAD_STR("mode", kRowselNames[c.rowsel])),
            RAD_WEIGHTS(k.score));
        if (!k.op_rowsel) missing = "kva_rowsel";
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

/* The selected set, its kernel ops and its refusals. Under a sizing declare the cap is the real
 * declare's -- it is an op parameter, and a fixed parameter that follows max_tok is refused by the
 * arena's level check -- and nothing is refused: the real declare already decided. */
static int decl_selected(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx,
                         Kva& k) {
    RAD_ARCH_TRY(decl_projector(b, k.nm, m, k.cfg.proj, k));
    if (k.have_proj) RAD_ARCH_TRY(decl_correction(b, k.nm, m, k.cfg.st, k.split, k));
    k.score = decl_score(b, k.nm, m, k.cfg.score);
    k.have_rowsel = k.score != 0;
    if (ctx->shape_probe) k.cfg.cap = g_kva[ctx->rank].cfg.cap;
    else RAD_ARCH_TRY(check_mode(k, m.g.max_tok));
    RAD_ARCH_TRY(decl_kernel_ops(b, m, ctx->shape_probe != 0, k));
    note_config(b, k);
    /* NOTHING UNIMPLEMENTED RETURNS SUCCESS. Stages 3-5 replace this with the step paths. */
    if (!ctx->shape_probe) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: mode %s is not implemented in this build "
                             "(the fill path is Stage 3); run with RADIANCE_KVA=off\n",
                     kModeNames[k.cfg.mode]);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
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
    RAD_ARCH_TRY(read_config(meta, m.g.max_tok, &k.cfg));
    if (!k.cfg.declare_all && k.cfg.mode == MODE_OFF) return RAD_OK;

    k.nm = Names(ctx->scope ? ctx->scope : "");
    k.proj_w.assign(m.layers.size(), 0);
    k.proj_b.assign(m.layers.size(), 0);
    k.st.assign(m.layers.size(), 0);
    k.op_undo.assign(m.layers.size(), 0);
    k.op_apply.assign(m.layers.size(), 0);
    if (k.cfg.declare_all) return decl_every_copy(b, m, k);
    return decl_selected(b, m, ctx, k);
}

static void step(RadCtx* c, const RadBatch* batch) {
    qwen4exp_fp8::step(c, batch);
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
