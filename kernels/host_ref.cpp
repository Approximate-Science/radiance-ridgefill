/* host_ref.cpp -- the operand checks both rows share, and the host rows: the oracles.
 *
 * The host rows are written to be read, not to be fast: plain loops, one element at a time, the
 * op's definition (rows.cpp's schema docs) transcribed. The device rows are tested against them
 * (tests/kernel_test.cpp). Everything here is compiled -ffp-contract=off, so `x + s*c` rounds
 * twice exactly as the device code does.
 */
#include "kva.h"

#include <cmath>
#include <cstring>
#include <vector>

/* ================================================================== operand checks */

static bool dense(const RadTensor* t) { return rad_tensor_is_contiguous(t); }

/* The product of the extents after the first two: what one (slot, head) of a LINEAR cache holds. */
static int64_t per_head(const RadTensor* t) {
    int64_t n = 1;
    for (uint32_t i = 2; i < t->rank; ++i) n *= t->shape[i];
    return n;
}

/* The element stride between N and D in an ND cache: the stride of its one trailing axis of
 * extent 2 (the group may be declared [.., 1, 2] or [.., 2, 1]). 0 if the layout is not a pair. */
static int64_t pair_stride(const RadTensor* t) {
    if (t->rank < 3 || per_head(t) != 2) return 0;
    for (uint32_t i = 2; i < t->rank; ++i)
        if (t->shape[i] == 2) return t->stride[i];
    return 0;
}

/* state_idx [n_seq] or [n_seq, pitch]: column 0 of each row is the sequence's slot. */
static int index_rows(const RadTensor* t, int64_t* rows, int64_t* pitch) {
    if (t->dtype != RAD_I32) return RAD_E_DTYPE;
    if (t->rank < 1 || t->rank > 2 || t->stride[t->rank - 1] != 1) return RAD_E_STRIDE;
    *rows = t->shape[0];
    *pitch = t->rank == 2 ? t->stride[0] : 1;
    return RAD_OK;
}

static int parse_mode(const char* s, const char* const* names, int n) {
    for (int i = 0; s && i < n; ++i)
        if (std::strcmp(s, names[i]) == 0) return i;
    return -1;
}

extern "C" int kva_rowsel_parse(const RadArgs* a, KvaRowsel* g) {
    const RadTensor* tok = rad_arg_in(a, RS_TOKENS);
    const RadTensor* pos = rad_arg_in(a, RS_POS);
    const RadTensor* score = rad_arg_in(a, RS_SCORE);
    const RadTensor* rows = rad_arg_in(a, RS_ROWS);
    const RadTensor* mask = rad_arg_in(a, RS_MASK);
    if (!tok || !pos || !score || !rows || !mask) return RAD_E_INVAL;
    if (tok->dtype != RAD_I32 || pos->dtype != RAD_I32 || score->dtype != RAD_F32 ||
        rows->dtype != RAD_I32 || mask->dtype != RAD_I32) return RAD_E_DTYPE;
    if (!dense(tok) || !dense(score) || !dense(rows) || !dense(mask)) return RAD_E_STRIDE;
    static const char* const modes[] = { "class", "random", "all" };
    long long cap = -1, seed = 0;
    g->mode = parse_mode(rad_args_gets(a, "mode"), modes, 3);
    g->share = rad_args_getf_or(a, "share", NAN);
    if (g->mode < 0 || !rad_args_geti(a, "cap", &cap) || !rad_args_geti(a, "seed", &seed) ||
        !(g->share >= 0.0 && g->share <= 1.0)) return RAD_E_INVAL;
    g->n = rad_tensor_numel(tok);
    g->vocab = rad_tensor_numel(score);
    /* `cap` is a CAPACITY: written by the operand's extent, bounded by the parameter. */
    g->cap = cap < rad_tensor_numel(rows) ? cap : rad_tensor_numel(rows);
    /* Positions: [n], or component-major [c, n] whose row 0 is the index (RadBatch::rope_pos's
     * layout); token t is at t * the last axis' stride either way. */
    if (pos->rank < 1 || pos->rank > 2) return RAD_E_SHAPE;
    g->pos_stride = pos->stride[pos->rank - 1];
    if (cap < 0 || rad_tensor_numel(mask) < g->n || pos->shape[pos->rank - 1] < g->n)
        return RAD_E_SHAPE;
    g->tokens = (const int32_t*)tok->data;
    g->positions = (const int32_t*)pos->data;
    g->score = (const float*)score->data;
    g->rows = (int32_t*)rows->data;
    g->mask = (int32_t*)mask->data;
    g->seed = seed;
    return RAD_OK;
}

extern "C" int kva_rho_parse(const RadArgs* a, KvaRho* g) {
    const RadTensor* av = rad_arg_in(a, RH_A);
    const RadTensor* mask = rad_arg_in(a, RH_MASK);
    const RadTensor* alog = rad_arg_in(a, RH_ALOG);
    const RadTensor* dtb = rad_arg_in(a, RH_DTBIAS);
    const RadTensor* nd = rad_arg_in(a, RH_ND);
    const RadTensor* sidx = rad_arg_in(a, RH_SIDX);
    if (!av || !mask || !alog || !dtb || !nd || !sidx) return RAD_E_INVAL;
    if ((av->dtype != RAD_BF16 && av->dtype != RAD_F32) || mask->dtype != RAD_I32 ||
        alog->dtype != RAD_F32 || dtb->dtype != RAD_F32 || nd->dtype != RAD_F32)
        return RAD_E_DTYPE;
    int64_t n_seq = 0, pitch = 0;
    const int rc = index_rows(sidx, &n_seq, &pitch);
    if (rc != RAD_OK) return rc;
    /* `a` is read at its own strides: a column slice of the a|b buffer, or any other layout of it. */
    if (av->rank != 2 || !dense(mask) || !dense(alog) || !dense(dtb)) return RAD_E_STRIDE;
    long long heads = 0;
    if (!rad_args_geti(a, "n_head", &heads)) return RAD_E_INVAL;
    g->n = av->shape[0];
    g->n_head = av->shape[1];
    g->nd_inner = pair_stride(nd);
    /* One sequence: the op runs only on single-sequence prefill steps (PLAN §3). */
    if (heads != g->n_head || rad_tensor_numel(mask) < g->n || rad_tensor_numel(alog) != heads ||
        rad_tensor_numel(dtb) != heads || nd->shape[1] != heads || g->nd_inner == 0 || n_seq != 1)
        return RAD_E_SHAPE;
    g->a = av->data;
    g->a_pitch = av->stride[0];
    g->a_col = av->stride[1];
    g->a_bf16 = av->dtype == RAD_BF16;
    g->mask = (const int32_t*)mask->data;
    g->a_log = (const float*)alog->data;
    g->dt_bias = (const float*)dtb->data;
    g->nd = (float*)nd->data;
    g->nd_slot = nd->stride[0];
    g->nd_head = nd->stride[1];
    g->n_states = nd->shape[0];
    g->state_idx = (const int32_t*)sidx->data;
    return RAD_OK;
}

extern "C" int kva_correct_parse(const RadArgs* a, KvaCorrect* g) {
    const RadTensor* st = rad_arg_in(a, SC_STATE);
    const RadTensor* sidx = rad_arg_in(a, SC_STATE_IDX);
    const RadTensor* ap = rad_arg_in(a, SC_APPLIED);
    const RadTensor* aidx = rad_arg_in(a, SC_APPLIED_IDX);
    const RadTensor* c = rad_arg_in(a, SC_C);
    const RadTensor* nd = rad_arg_in(a, SC_ND);        /* optional, with its index */
    const RadTensor* nidx = rad_arg_in(a, SC_ND_IDX);
    if (!st || !sidx || !ap || !aidx || !c || !nd != !nidx) return RAD_E_INVAL;
    if (st->dtype != RAD_F32 || ap->dtype != RAD_F32 || c->dtype != RAD_F32 ||
        (nd && nd->dtype != RAD_F32)) return RAD_E_DTYPE;
    static const char* const modes[] = { "undo", "apply" };
    long long heads = 0, sd0 = 0, sd1 = 0;
    g->apply = parse_mode(rad_args_gets(a, "mode"), modes, 2);
    const double alpha = rad_args_getf_or(a, "alpha", NAN);
    if (g->apply < 0 || !rad_args_geti(a, "n_head", &heads) || !rad_args_geti(a, "sd0", &sd0) ||
        !rad_args_geti(a, "sd1", &sd1) || (g->apply && !std::isfinite(alpha))) return RAD_E_INVAL;
    int64_t ap_rows = 0, nd_rows = 0;
    int rc = index_rows(sidx, &g->n_seq, &g->st_pitch);
    if (rc == RAD_OK) rc = index_rows(aidx, &ap_rows, &g->ap_pitch);
    if (rc == RAD_OK && nidx) rc = index_rows(nidx, &nd_rows, &g->nd_pitch);
    if (rc != RAD_OK) return rc;
    if (ap_rows != g->n_seq || (nidx && nd_rows != g->n_seq)) return RAD_E_SHAPE;
    if (st->rank != 4 || st->shape[1] != heads || st->shape[2] != sd0 || st->shape[3] != sd1 ||
        c->rank != 3 || c->shape[0] != heads || c->shape[1] != sd0 || c->shape[2] != sd1 ||
        ap->rank < 2 || ap->shape[1] != heads || per_head(ap) != 1 ||
        (nd && (nd->shape[1] != heads || pair_stride(nd) == 0))) return RAD_E_SHAPE;
    g->state = (float*)st->data;
    g->st_slot = st->stride[0]; g->st_head = st->stride[1];
    g->st_row = st->stride[2];  g->st_col = st->stride[3];
    g->applied = (float*)ap->data;
    g->ap_slot = ap->stride[0]; g->ap_head = ap->stride[1];
    g->c = (const float*)c->data;
    g->c_head = c->stride[0]; g->c_row = c->stride[1]; g->c_col = c->stride[2];
    g->nd = nd ? (const float*)nd->data : nullptr;
    g->nd_slot = nd ? nd->stride[0] : 0;
    g->nd_head = nd ? nd->stride[1] : 0;
    g->nd_inner = nd ? pair_stride(nd) : 0;
    g->st_idx = (const int32_t*)sidx->data;
    g->ap_idx = (const int32_t*)aidx->data;
    g->nd_idx = nidx ? (const int32_t*)nidx->data : nullptr;
    g->st_states = st->shape[0];
    g->ap_states = ap->shape[0];
    g->nd_states = nd ? nd->shape[0] : 0;
    g->n_head = heads; g->sd0 = sd0; g->sd1 = sd1;
    g->alpha = (float)alpha;
    return RAD_OK;
}

/* ================================================================== kva_rowsel */

extern "C" int kva_rowsel_host(const RadArgs* a, RadStream) {
    KvaRowsel g{};
    const int rc = kva_rowsel_parse(a, &g);
    if (rc != RAD_OK) return rc;
    std::vector<float> key((size_t)g.n);      /* class: the score, -inf where the row does not match */
    std::vector<uint32_t> hash((size_t)g.n);  /* random: the row's rank key */
    int64_t matches = 0;
    for (int64_t i = 0; i < g.n; ++i) {
        const int32_t id = g.tokens[i];
        const float s = id >= 0 && id < g.vocab ? g.score[id] : -INFINITY;
        key[(size_t)i] = std::isfinite(s) ? s : -INFINITY;
        hash[(size_t)i] = kva_row_hash(g.seed, (uint32_t)g.positions[i * g.pos_stride]);
        matches += std::isfinite(s) ? 1 : 0;
    }
    const int64_t k = g.mode == KVA_MODE_ALL ? g.n : (int64_t)std::rint(g.share * (double)matches);
    const int64_t limit = k < g.cap ? k : g.cap;
    int64_t out = 0;
    for (int64_t i = 0; i < g.n; ++i) {
        int64_t ahead = 0;   /* rows that rank before row i */
        bool eligible = true;
        if (g.mode == KVA_MODE_ALL) {
            ahead = i;
        } else if (g.mode == KVA_MODE_CLASS) {
            eligible = key[(size_t)i] > -INFINITY;
            for (int64_t j = 0; j < g.n; ++j)
                ahead += key[(size_t)j] > key[(size_t)i] ||
                         (key[(size_t)j] == key[(size_t)i] && j < i);
        } else {
            for (int64_t j = 0; j < g.n; ++j)
                ahead += hash[(size_t)j] < hash[(size_t)i] ||
                         (hash[(size_t)j] == hash[(size_t)i] && j < i);
        }
        const bool kept = eligible && ahead < limit;
        g.mask[i] = kept ? 0 : 1;
        if (kept) g.rows[out++] = (int32_t)i;
    }
    for (int64_t i = out; i < g.cap; ++i) g.rows[i] = -1;
    return RAD_OK;
}

/* ================================================================== kva_rho_update */

extern "C" int kva_rho_host(const RadArgs* a, RadStream) {
    KvaRho g{};
    const int rc = kva_rho_parse(a, &g);
    if (rc != RAD_OK) return rc;
    const int32_t slot = g.state_idx[0];
    if (slot < 0 || slot >= g.n_states) return RAD_OK;   /* no slot: nothing to carry */
    for (int64_t h = 0; h < g.n_head; ++h) {
        float* nd = g.nd + slot * g.nd_slot + h * g.nd_head;
        float n = nd[0], d = nd[g.nd_inner];
        const float decay = expf(g.a_log[h]);
        for (int64_t t = 0; t < g.n; ++t) {
            const int64_t at = t * g.a_pitch + h * g.a_col;
            const float av = g.a_bf16 ? kva_bf16_to_f32(((const uint16_t*)g.a)[at])
                                      : ((const float*)g.a)[at];
            const float e = expf(-decay * kva_softplus(av + g.dt_bias[h]));
            d = e * d + 1.0f;
            n = e * n + (float)g.mask[t];
        }
        nd[0] = n;
        nd[g.nd_inner] = d;
    }
    return RAD_OK;
}

/* ================================================================== kva_state_correct */

extern "C" int kva_correct_host(const RadArgs* a, RadStream) {
    KvaCorrect g{};
    const int rc = kva_correct_parse(a, &g);
    if (rc != RAD_OK) return rc;
    for (int64_t s = 0; s < g.n_seq; ++s) {
        int64_t st_slot = 0, ap_slot = 0, nd_slot = 0;
        if (!kva_correct_slots(&g, s, &st_slot, &ap_slot, &nd_slot)) continue;
        for (int64_t h = 0; h < g.n_head; ++h) {
            float* applied = g.applied + ap_slot * g.ap_slot + h * g.ap_head;
            float scale = *applied;
            if (g.apply) {
                const float* nd = g.nd ? g.nd + nd_slot * g.nd_slot + h * g.nd_head : nullptr;
                scale = g.alpha * (nd ? kva_rho(nd[0], nd[g.nd_inner]) : 1.0f);
            }
            float* state = g.state + st_slot * g.st_slot + h * g.st_head;
            const float* c = g.c + h * g.c_head;
            for (int64_t i = 0; i < g.sd0 && scale != 0.0f; ++i)
                for (int64_t j = 0; j < g.sd1; ++j) {
                    float& x = state[i * g.st_row + j * g.st_col];
                    const float sc = scale * c[i * g.c_row + j * g.c_col];
                    x = g.apply ? x + sc : x - sc;
                }
            *applied = g.apply ? scale : 0.0f;
        }
    }
    return RAD_OK;
}
