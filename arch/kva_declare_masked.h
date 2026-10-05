/* kva_declare_masked.h -- the masked path's declarations (PLAN-FIX §3, §6.1), the selected set's
 * declare, and the debug captures' declarations. Included by kva_declare.h only.
 */
#ifndef QWEN4EXP_KVA_DECLARE_MASKED_H
#define QWEN4EXP_KVA_DECLARE_MASKED_H

namespace qwen4exp_kva {

/* THE STAGER PROBES' BUFFERS (notes/impl.md §2): expert offsets that are all zero -- kva_mask writes
 * them every masked pass -- and the rows a probe's gate-up GEMM writes its zeros into, sized for the
 * widest routed layer. Both take the whole program. */
static int decl_probes(RadBuilder* b, Kva& k) {
    const KvaAdapter& a = k.ad;
    if (a.top_k == 0) return RAD_OK;   /* dense: nothing is staged, so nothing to steer */
    k.n_zeros = std::max(a.n_expert + 1, a.top_k);
    k.b_zeros = decl_b(b, k.nm.f("kva_zero_offsets"), RAD_I32, {k.n_zeros});
    k.b_probe = decl_b(b, k.nm.f("kva_probe_rows"), RAD_BF16, {a.top_k, 2 * a.n_ff_exp});
    if (!k.b_zeros || !k.b_probe) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_buf_concurrent(b, k.b_zeros));
    return rad_buf_concurrent(b, k.b_probe);
}

/* The projected block input and its codes: the same code pair the connection read writes into `x`
 * (int8 when the trunk is fed int8 codes, E4M3 otherwise, none for a bf16 model), so kva_select can
 * copy a projected row's codes over an exact row's and no linear re-quantises anything. */
static int decl_projected(RadBuilder* b, const RadBuildCtx* ctx, Kva& k) {
    const KvaAdapter& a = k.ad;
    const int64_t n = a.n_embd, wide = a.wide, max_tok = ctx->max_tok;
    /* h_S, the layer-S stream every projector reads on a masked pass (40 MiB at 2,048 rows, paid by every
     * request in resident experts): not with int8 maps and no MTP map, whose codes are made from b_h at
     * layer S (kva_layer.h project_masked) */
    if (!k.int8 || k.want_final) {
        k.b_hs = decl_b(b, k.nm.f("kva_h_stream"), a.act_dtype, {max_tok, wide});
        if (!k.b_hs) return RAD_E_INVAL;
    }
    k.xp.x = decl_b(b, k.nm.f("kva_x_proj"), a.act_dtype, {max_tok, n});
    if (!k.xp.x) return RAD_E_INVAL;
    if (a.declare_codes) RAD_ARCH_TRY(a.declare_codes(b, ctx, k));
    for (rad_buf h : { k.b_hs, k.xp.x, k.xp.cq(), k.xp.cs(), a.buf_route_ids })
        if (h) RAD_ARCH_TRY(rad_buf_concurrent(b, h));
    /* Every buffer these three touch takes the whole program (concurrent here or in decl_fill), so
     * they state no read/write sets. */
    k.op_cast = RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("n", wide),
                                             RAD_STR("from", a.dtype), RAD_STR("to", a.dtype)),
                       RAD_NOWEIGHTS);
    k.op_select = RAD_OP(b, "kva_select", RAD_PARAMS(RAD_RANGE("M", 1, max_tok)), RAD_NOWEIGHTS);
    k.op_drop = RAD_OP(b, "kva_drop_rows",
                       RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("top_k", a.top_k)),
                       RAD_NOWEIGHTS);
    return RAD_OK;
}

/* THE MASKED PATH'S DECLARATIONS. Every mode but off: plumb runs the late layers exactly through
 * the same path (mask all 0, no projector), which is what makes it the oracle for the split scan
 * (R47) and the stager probes (R94). `kva_mask`'s mode is the mode's row rule: plumb keeps every
 * row exact, speed approximates the whole window, quality keeps its class (or random/all) rows. */
static const char* decl_masked(RadBuilder* b, const RadBuildCtx* ctx, Kva& k) {
    const Config& c = k.cfg;
    const bool project = c.mode != MODE_PLUMB;
    k.b_mask   = decl_b(b, k.nm.f("kva_mask"), RAD_I32, {ctx->max_tok});
    k.b_bounds = decl_b(b, k.nm.f("kva_bounds"), RAD_I32, {4});
    if (!k.b_mask || !k.b_bounds) return "a buffer";
    for (rad_buf h : { k.b_mask, k.b_bounds })
        if (rad_buf_concurrent(b, h) < 0) return "a buffer";
    if (project && decl_projected(b, ctx, k) < 0) return "a buffer";
    const char* rule = c.mode == MODE_PLUMB ? "all" : c.mask_step ? "step"
                                                     : c.mode == MODE_SPEED ? "none"
                                                     : kRowselNames[c.rowsel];
    const bool scored = !std::strcmp(rule, "class") || !std::strcmp(rule, "random");
    k.mask_scored = scored;
    k.op_mask = RAD_OP(b, "kva_mask",
                       RAD_PARAMS(RAD_RANGE("M", 1, ctx->max_tok), RAD_F64("share", c.share),
                                  RAD_INT("seed", c.seed), RAD_STR("mode", rule)),
                       RAD_NOWEIGHTS);
    if (!k.op_mask) return "kva_mask";
    if (project && (!k.op_select || !k.op_drop)) return !k.op_select ? "kva_select" : "kva_drop_rows";
    if (project && !k.op_cast) return "cast";
    const bool straddle = c.tail_only && c.mode == MODE_SPEED;
    int64_t lacking = -1;
    const char* what = nullptr;
    for (int64_t l = k.split; l < k.ad.n_layer; ++l) {
        if (!k.ad.full[(size_t)l]) continue;
        const char* miss = k.ad.straddle_lack[(size_t)l];
        if (miss && lacking < 0) { lacking = l; what = miss; }
        k.qsa_exact_to = std::max(k.qsa_exact_to, k.ad.qsa_exact_to[(size_t)l]);
    }
    k.straddle_layers = straddle && lacking < 0;
    /* Otherwise the downgrade shows only in each step log's path field: say it once, at startup. */
    if (straddle && lacking >= 0)
        std::fprintf(stderr, "radiance: %s: KVA: late attention layer %lld lacks %s, so speed mode "
                             "takes the masked path for straddle chunks: slower, same output class\n", g_log_name,
                     (long long)lacking, what);
    return decl_probes(b, k) < 0 ? "a buffer" : nullptr;
}

static void note_config(RadBuilder* b, const Kva& k) {
    const Config& c = k.cfg;
    char rows[24] = "any";
    if (c.stage_rows != INT64_MAX) std::snprintf(rows, sizeof rows, "<= %lld", (long long)c.stage_rows);
    rad_note(b, "KVA: mode %s from layer %lld, tail %lld, tile %lld; projector %s (%s, streamed from host), correction %s "
                "(alpha %g), row table %s, rows %s share %g seed %lld; stage %s (exact rows %s), "
                "approximates >= %lld bulk rows, checkpoint floor %lld, straddle %s%s%s",
             kModeNames[c.mode], (long long)k.split, (long long)c.tail, (long long)k.tile,
             g_loaded.folder.place.dir.c_str(), k.int8 ? "int8" : "bf16", k.have_st ? "held" : "absent",
             c.alpha, k.have_rowsel ? kScoreNames[c.rowsel_table] : "absent",
             kRowselNames[c.rowsel], c.share, (long long)c.seed, kStageNames[c.stage],
             rows, (long long)c.min_bulk_rows, (long long)c.ckpt_floor, kStraddleNames[c.straddle],
             k.out_rows_ok ? "" : "; KL mode serves stock (RADIANCE_KVA_SCORE_BULK unset)",
             c.force_split || c.shift_b || c.force_stream || c.mask_step ? "; DEBUG switches set" : "");
}

/* The gate-only switches, said loudly at declare so no measured run carries one unknowingly. A
 * forced split off the tile is accepted: it is R47's negative control (its bytes must move). */
static void note_debug(const Kva& k) {
    const Config& c = k.cfg;
    if (c.force_split)
        std::fprintf(stderr, "radiance: %s: DEBUG RADIANCE_KVA_FORCE_SPLIT=%lld: every "
                             "approximate chunk's bulk ends %lld rows before its end%s\n", g_log_name,
                     (long long)c.force_split, (long long)c.force_split,
                     c.force_split % k.tile ? " -- OFF the delta net's tile, a negative control" : "");
    if (c.shift_b)
        std::fprintf(stderr, "radiance: %s: DEBUG RADIANCE_KVA_SHIFT_B=%lld\n", g_log_name,
                     (long long)c.shift_b);
    if (c.force_stream)
        std::fprintf(stderr, "radiance: %s: DEBUG RADIANCE_KVA_FORCE_STREAM=1\n", g_log_name);
    if (c.mask_step)
        std::fprintf(stderr, "radiance: %s: DEBUG RADIANCE_KVA_MASK=all: every row of a "
                             "masked pass before its bulk end is approximated, decoders included -- "
                             "a negative control, never a served configuration\n", g_log_name);
}

/* What the folder lets this mode run (kva_projector.h): false = serve stock, already said. Plumb
 * reads no fitted tensor (its mask keeps every row exact) and so holds none. */
static bool take_folder(RadBuilder* b, const RadModelMeta* meta, Kva& k) {
    const Loaded& l = load_folder(meta, b, k.ad);
    const Config& c = k.cfg;
    if (!l.usable) return false;
    if (c.mode == MODE_QUALITY && !tensor(l.folder, kScoreNames[c.rowsel_table])) {
        std::fprintf(stderr, "radiance: %s: KVA: mode quality selects exact rows from the "
                             "table '%s', and the projector %s holds none; serving stock\n", g_log_name,
                     kScoreNames[c.rowsel_table], l.folder.place.dir.c_str());
        return false;
    }
    k.split = l.split;
    k.have_proj = true;
    k.have_st = l.has_st && c.mode != MODE_PLUMB;
    k.have_rowsel = c.mode == MODE_QUALITY;
    k.int8 = l.int8 && c.mode != MODE_PLUMB;
    return true;
}

/* This rank's copies (the real declare only) handed to the issue sites. */
static int take_upload(const RadBuildCtx* ctx, Kva& k) {
    if (!upload_rank(g_loaded, k.ad, k.cfg, ctx->rank, k.want_final)) return RAD_E_DEVICE;
    const kva::Upload& u = g_upload[ctx->rank];
    if (k.cfg.mode == MODE_PLUMB) return RAD_OK;
    k.proj_w = u.proj_w;
    k.proj_b = u.proj_b;
    k.proj_s = u.proj_s;
    k.st = u.st;
    k.score = u.score;
    k.ring_src = u.ring_src;
    k.ring_dst = u.ring_dst;
    k.final_w = u.final_w;
    k.final_b = u.final_b;
    return RAD_OK;
}

/* DD-A's branch-hazard instrument, the declare half (the issue half and the log: kva_hazard.h). */
/* One f32 counter a rank, host-mapped so the host reads what the device adds without a copy op.
 * Allocated at the first real declare and kept for the process (4 bytes). */
static void* g_hazard_dev[MAX_RANKS];
static float g_hazard_logged[MAX_RANKS];

static int decl_hazard(RadBuilder* b, const RadBuildCtx* ctx, Kva& k) {
    if (k.cfg.mode != MODE_SPEED && k.cfg.mode != MODE_QUALITY) return RAD_OK;
    for (int64_t l = k.split; l < k.ad.n_layer && k.meta_layer < 0; ++l)
        if (!k.ad.full[(size_t)l]) k.meta_layer = (int)l;
    if (k.meta_layer < 0) return RAD_OK;
    RadKVGroupDecl d{};
    d.kind = RAD_KV_LINEAR;
    d.dtype = RAD_F32;
    d.n_head_kv = 1;
    d.state_dim[0] = 1;
    d.state_dim[1] = 2;   /* {last approximated position + 1, positions counted below} */
    k.kv_meta = rad_decl_kv_group(b, k.nm.f("kv_kva_meta"), &d);
    if (!k.kv_meta || rad_bind_layer_kv(b, k.meta_layer, k.kv_meta) < 0) return RAD_E_INVAL;
    k.op_hazard = RAD_OP(b, "kva_hazard", RAD_PARAMS(RAD_RANGE("M", 1, 1)), RAD_NOWEIGHTS);
    if (!k.op_hazard) return ctx->shape_probe ? RAD_OK : RAD_E_UNSUPPORTED;
    if (!ctx->shape_probe && !g_hazard_dev[ctx->rank]) {
        g_hazard_dev[ctx->rank] = rad_dev_alloc(sizeof(float), RAD_MEM_HOST_MAPPED);
        float* h = g_hazard_dev[ctx->rank] ? (float*)rad_dev_host_ptr(g_hazard_dev[ctx->rank]) : nullptr;
        if (!h) return RAD_E_NOMEM;
        *h = 0.0f;
    }
    return RAD_OK;
}

/* The selected set, its ops and its refusals. Under a sizing declare nothing is refused or copied:
 * the real declare already decided, and its handles are the ones issued. With no usable projector
 * nothing at all is declared: the engine serves the in-tree graph. */
static int decl_selected(RadBuilder* b, const RadModelMeta* meta, const qwen4exp_fp8::Model& m,
                         const RadBuildCtx* ctx, Kva& k) {
    const bool probe = ctx->shape_probe != 0;
    if (!take_folder(b, meta, k)) return RAD_OK;
    /* THE MTP final map (kva_final.h): only when this deployment drafts, the mode projects, the folder holds
     * it and RADIANCE_KVA_FINAL is not off. Decided from declare-time numbers only. */
    k.want_final = ctx->max_spec > 0 && (k.cfg.mode == MODE_SPEED || k.cfg.mode == MODE_QUALITY) &&
                   g_loaded.has_final && k.cfg.final_on;
    k.ring_end = k.ad.n_layer + (k.want_final ? k.ad.wide / k.ad.n_embd : 0);   /* + hc final blocks */
    if (!probe) RAD_ARCH_TRY(check_mode(k, m.g.max_tok));
    if (!probe) RAD_ARCH_TRY(check_fill(k));
    if (!probe) RAD_ARCH_TRY(take_upload(ctx, k));
    if (k.cfg.mode != MODE_PLUMB) RAD_ARCH_TRY(decl_fill(b, ctx, k));
    const char* missing = decl_masked(b, ctx, k);
    if (!missing && k.ad.decl_state_ops) missing = k.ad.decl_state_ops(b, ctx, k);
    if (!missing && decl_hazard(b, ctx, k) != RAD_OK) missing = "kva_hazard";
    if (!missing && k.want_final) missing = decl_final(b, m, ctx, k);
    if (missing && !probe) {
        std::fprintf(stderr, "radiance: %s: mode %s issues '%s' and no kernel library "
                             "serves it -- kva.so is missing from $RADIANCE_HOME or declines this "
                             "machine. Refusing rather than serving without it.\n", g_log_name,
                     kModeNames[k.cfg.mode], missing);
        return RAD_E_UNSUPPORTED;
    }
    if (!probe) note_debug(k);
    note_config(b, k);
    return RAD_OK;
}

/* WHERE THE LATE LAYERS START FOR A CAPTURE, which in `off` mode nothing else has asked: the
 * projector folder's split -- what every other mode calls S -- else RADIANCE_KVA_CAPTURE_SPLIT (a
 * capture fits a projector, so there may be no folder yet). Never the container: nothing kva.* is
 * read from the model file. Declares nothing. */
static int capture_split(RadBuilder* b, const RadModelMeta* meta, Kva& k) {
    const Loaded& l = load_folder(meta, b, k.ad);
    k.split = l.usable ? l.split : -1;
    const char* v = env("RADIANCE_KVA_CAPTURE_SPLIT");
    if (!l.usable && v && !parse_int(v, &k.split)) k.split = -1;
    if (k.split < k.ad.split_lo || k.split >= k.ad.n_layer) {
        std::fprintf(stderr, "radiance: %s: a capture needs the split layer S: there is no "
                             "usable projector folder and RADIANCE_KVA_CAPTURE_SPLIT is %s\n", g_log_name,
                     v ? v : "unset");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* THE STATE CAPTURE'S COPY. No ABI call returns a KV-pool pointer (RADIANCE-FACTS §5), so a late
 * layer's slot is copied by kva.so's kva_state_read into this rank's [1, heads, V, K] buffer and
 * read from there. Declared only when RADIANCE_KVA_CAPTURE_STATE is set. */
static int decl_state_read(RadBuilder* b, const RadBuildCtx* ctx, Kva& k) {
    const StateShape& g = k.ad.state;
    k.b_state = decl_b(b, k.nm.f("kva_state_copy"), RAD_F32, {1, g.n_head, g.sd0, g.sd1});
    if (!k.b_state) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_buf_concurrent(b, k.b_state));
    k.op_state_read = rw(b, RAD_OP(b, "kva_state_read",
                                RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n_head", g.n_head),
                                           RAD_INT("sd0", g.sd0), RAD_INT("sd1", g.sd1)),
                                RAD_NOWEIGHTS), {}, {k.b_state});
    if (!k.op_state_read && !ctx->shape_probe) {
        std::fprintf(stderr, "radiance: %s: RADIANCE_KVA_CAPTURE_STATE copies the delta-net "
                             "state with kva_state_read, and no kernel library serves it -- kva.so is "
                             "missing or too old\n", g_log_name);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_DECLARE_MASKED_H */
