/* kva_declare_masked.h -- the masked path's declarations (PLAN-FIX §3, §6.1), the selected set's
 * declare, and the debug captures' declarations. Included by kva_declare.h only.
 */
#ifndef QWEN4EXP_KVA_DECLARE_MASKED_H
#define QWEN4EXP_KVA_DECLARE_MASKED_H

namespace qwen4exp_kva {

/* THE ALTERNATE DOWN HANDLE (PLAN-FIX §6.1, DD-B): layer l's routed down GEMM declared a second
 * time, after the whole graph, with the SAME params as the in-tree op_dn (rad_block_moe_fp8.h:
 * 1033-1043; bf16 experts :891-898) and NO weights. The approximate pass issues it with the layer's
 * own weight tables (legal: issue.cpp:136-138 serves a table the declaration did not name, with a
 * one-time ~2 KiB allocation per table at the first approximate pass, :233-246). Because the
 * prefill stager learns a layer's span from DECLARED weight uses only (stager.cpp:21-55), this
 * handle is no layer's last op, so a staged layer's buffer is never released before the pass ends
 * and every later layer streams its routed experts instead of being staged whole (stager.cpp:93).
 * Its reads/writes are op_dn's, which keeps the MoE scratch live to here: without them the planner
 * could lay another buffer over `eff`/`edn` (HANDOVER-FIX §9). Kernel selection is by the dtype
 * string (r4d_rows.cpp:2471-2503), so the weightless declaration resolves to op_dn's row. */
static rad_op decl_alt_down(RadBuilder* b, const MoeFP8& e, int64_t max_tok) {
    const MoeFP8::Config& c = e.c;
    const int64_t rows = std::min(max_tok, c.rows);   /* a sizing declare's pass, as its own block */
    if (c.expert_bf16)
        return rw(b, RAD_OP(b, "moe_gemm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", e.g.n_embd),
                                 RAD_INT("K", c.n_ff_exp), RAD_INT("n_expert", c.n_expert),
                                 RAD_INT("top_k", c.top_k), RAD_STR("a_order", "sorted"),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS),
                  {e.w.eff.x, e.w.sorted, e.w.eoff}, {e.w.edn});
    const bool sliced = c.ff_hi[0] > 0;   /* each parity's width on this rank, as declare's wq */
    const int64_t k0 = sliced ? c.ff_hi[0] - c.ff_lo[0] : c.n_ff_exp;
    const int64_t k1 = sliced ? c.ff_hi[1] - c.ff_lo[1] : c.n_ff_exp;
    return rw(b, RAD_OP(b, "moe_gemm_q",
                  RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", e.g.n_embd), RAD_INT("K", k0),
                             RAD_INT("n_expert", e.n_reg), RAD_INT("top_k", c.top_k),
                             RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("a_order", "sorted"),
                             RAD_STR("dtype", MoeFP8::kServed[e.fmt[1]]),
                             RAD_INT("N_odd", e.g.n_embd), RAD_INT("K_odd", k1),
                             RAD_INT("parts", 1)),
                  RAD_NOWEIGHTS),
              {e.w.eff.q, e.w.eff.s, e.w.sorted, e.w.eoff}, {e.w.edn});
}

/* The projected block input and its codes: the same code pair the connection read writes into `x`
 * (int8 when the trunk is fed int8 codes, E4M3 otherwise, none for a bf16 model), so kva_select can
 * copy a projected row's codes over an exact row's and no linear re-quantises anything. */
static int decl_projected(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx, Kva& k) {
    Geom g = m.g;
    g.max_tok = ctx->max_tok;
    const int64_t n = g.n_embd;
    k.b_hs = decl_b(b, k.nm.f("kva_h_stream"), g.act_dtype, {g.max_tok, m.hccfg.hc * n});
    k.xp.x = decl_b(b, k.nm.f("kva_x_proj"), g.act_dtype, {g.max_tok, n});
    if (!k.b_hs || !k.xp.x) return RAD_E_INVAL;
    if (m.a_x.cq()) RAD_ARCH_TRY(k.xp.declare_qs(b, k.nm, g, "kva_x_proj", n, 0, m.a_x.q8_fed));
    k.xp.q8_fed = m.a_x.q8_fed;
    for (rad_buf h : { k.b_hs, k.xp.x, k.xp.cq(), k.xp.cs(), m.b_eids })
        if (h) RAD_ARCH_TRY(rad_buf_concurrent(b, h));
    const int64_t wide = m.hccfg.hc * n;
    /* Every buffer these three touch takes the whole program (concurrent here or in decl_fill), so
     * they state no read/write sets. */
    k.op_cast = RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", wide),
                                             RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                       RAD_NOWEIGHTS);
    k.op_select = RAD_OP(b, "kva_select", RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok)), RAD_NOWEIGHTS);
    k.op_drop = RAD_OP(b, "kva_drop_rows",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("top_k", m.moecfg.top_k)),
                       RAD_NOWEIGHTS);
    return RAD_OK;
}

/* THE MASKED PATH'S DECLARATIONS. Every mode but off: plumb runs the late layers exactly through
 * the same path (mask all 0, no projector), which is what makes it the oracle for the split scan
 * (R47) and the alternate handles (R94). `kva_mask`'s mode is the mode's row rule: plumb keeps every
 * row exact, speed approximates the whole window, quality keeps its class (or random/all) rows. */
static const char* decl_masked(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx,
                               Kva& k) {
    const Config& c = k.cfg;
    const bool project = c.mode != MODE_PLUMB;
    k.b_mask   = decl_b(b, k.nm.f("kva_mask"), RAD_I32, {ctx->max_tok});
    k.b_bounds = decl_b(b, k.nm.f("kva_bounds"), RAD_I32, {4});
    if (!k.b_mask || !k.b_bounds) return "a buffer";
    for (rad_buf h : { k.b_mask, k.b_bounds })
        if (rad_buf_concurrent(b, h) < 0) return "a buffer";
    if (project && decl_projected(b, m, ctx, k) < 0) return "a buffer";
    const char* rule = c.mode == MODE_PLUMB ? "all" : c.mode == MODE_SPEED ? "none"
                                                     : kRowselNames[c.rowsel];
    const bool scored = !std::strcmp(rule, "class") || !std::strcmp(rule, "random");
    k.mask_scored = scored;
    const rad_weight w[1] = { k.score };
    k.op_mask = rad_decl_op(b, "kva_mask",
                            RAD_PARAMS(RAD_RANGE("M", 1, ctx->max_tok), RAD_F64("share", c.share),
                                       RAD_INT("seed", c.seed), RAD_STR("mode", rule)),
                            scored ? w : nullptr, scored ? 1 : 0);
    if (!k.op_mask) return "kva_mask";
    if (project && (!k.op_select || !k.op_drop)) return !k.op_select ? "kva_select" : "kva_drop_rows";
    if (project && !k.op_cast) return "cast";
    k.alt_dn.assign(m.layers.size(), 0);
    for (int64_t l = std::max<int64_t>(k.split - 1, 0); l < m.g.n_layer; ++l)
        if (!(k.alt_dn[(size_t)l] = decl_alt_down(b, m.layers[(size_t)l].mlp, ctx->max_tok)))
            return "moe_gemm_q (the alternate down handle)";
    return nullptr;
}

static void note_config(RadBuilder* b, const Kva& k) {
    const Config& c = k.cfg;
    rad_note(b, "KVA: mode %s from layer %lld, tail %lld, tile %lld; projector %s, correction %s "
                "(alpha %g), row table %s, rows %s share %g seed %lld; stage %s (exact rows <= %lld), "
                "straddle %s%s%s",
             kModeNames[c.mode], (long long)k.split, (long long)c.tail, (long long)k.tile, c.proj,
             k.have_st ? c.st : "absent", c.alpha, k.have_rowsel ? c.score : "absent",
             kRowselNames[c.rowsel], c.share, (long long)c.seed, kStageNames[c.stage],
             (long long)c.stage_rows, kStraddleNames[c.straddle],
             k.out_rows_ok ? "" : "; KL mode serves stock (RADIANCE_KVA_SCORE_BULK unset)",
             c.force_split || c.shift_b || c.force_alt ? "; DEBUG switches set" : "");
}

/* The gate-only switches, said loudly at declare so no measured run carries one unknowingly. A
 * forced split off the tile is accepted: it is R47's negative control (its bytes must move). */
static void note_debug(const Kva& k) {
    const Config& c = k.cfg;
    if (c.force_split)
        std::fprintf(stderr, "radiance: qwen4exp_kva: DEBUG RADIANCE_KVA_FORCE_SPLIT=%lld: every "
                             "approximate chunk's bulk ends %lld rows before its end%s\n",
                     (long long)c.force_split, (long long)c.force_split,
                     c.force_split % k.tile ? " -- OFF the delta net's tile, a negative control" : "");
    if (c.shift_b)
        std::fprintf(stderr, "radiance: qwen4exp_kva: DEBUG RADIANCE_KVA_SHIFT_B=%lld\n",
                     (long long)c.shift_b);
    if (c.force_alt)
        std::fprintf(stderr, "radiance: qwen4exp_kva: DEBUG RADIANCE_KVA_FORCE_ALT=1\n");
}

/* The selected set, its ops and its refusals. Under a sizing declare nothing is refused: the real
 * declare already decided. */
static int decl_selected(RadBuilder* b, const qwen4exp_fp8::Model& m, const RadBuildCtx* ctx,
                         Kva& k) {
    const bool probe = ctx->shape_probe != 0;
    RAD_ARCH_TRY(decl_projector(b, k.nm, m, k.cfg.proj, k));
    /* plumb issues no correction, so it does not place one */
    if (k.have_proj && k.cfg.mode != MODE_PLUMB)
        RAD_ARCH_TRY(decl_correction(b, k.nm, m, k.cfg.st, k.split, k));
    if (k.cfg.mode == MODE_QUALITY) k.score = decl_score(b, k.nm, m, k.cfg.score);
    k.have_rowsel = k.score != 0;
    if (!probe) RAD_ARCH_TRY(check_mode(k, m));
    if (!probe) RAD_ARCH_TRY(check_fill(m, k));
    if (k.cfg.mode != MODE_PLUMB) RAD_ARCH_TRY(decl_fill(b, m, ctx, k));
    const char* missing = decl_masked(b, m, ctx, k);
    if (!missing) missing = decl_kernel_ops(b, m, ctx, k);
    if (missing && !probe) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: mode %s issues '%s' and no kernel library "
                             "serves it -- kva.so is missing from $RADIANCE_HOME or declines this "
                             "machine. Refusing rather than serving without it.\n",
                     kModeNames[k.cfg.mode], missing);
        return RAD_E_UNSUPPORTED;
    }
    if (!probe) note_debug(k);
    note_config(b, k);
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

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_DECLARE_MASKED_H */
