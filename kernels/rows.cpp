/* rows.cpp -- the kva kernel library's op schemas, its row table and its operand descriptions.
 *
 * THREE NEW OPS, none of them in docs/OPS.md, so this plugin FIXES their schemas (the first plugin
 * in hierarchy order to declare an op does) and nothing else in the engine knows them. Each has a
 * host row -- plain C++, the oracle the device row is tested against (tests/kernel_test.cpp) --
 * and, in a HIP build, a device row. The plan they implement is PLAN.md §5 of the KVA plugin build.
 *
 * Every row carries an `opd_shape`, so a tool can call it cold instead of skipping it by name. No
 * row has a scratch hook (rowsel keeps its keys in LDS; the other two need nothing), tunables, a
 * layout hook or a `describe`: each is one fixed launch issued once per layer per prefill chunk.
 */
#include "kva.h"

#include <cstring>
#include <initializer_list>

#define P_INT(k) { (k), RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }
#define P_STR(k) { (k), RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE }
#define P_F64(k) { (k), RAD_P_F64, RAD_REQUIRED, RAD_PROLE_NONE }
/* A CAPACITY (rad_abi.h): the kernel writes by its operands' extents and treats the value as a
 * bound, so a sizing declare at a smaller max_tok may hand it a smaller one. `cap` is derived from
 * max_tok at declare (PLAN D12), which is exactly the case the role exists for. */
#define P_CAP(k) { (k), RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY }

#define OPD(n)   { (n), RAD_OPD_IN, 0 }
#define OPD_O(n) { (n), RAD_OPD_IN, 1 }
#define OUT(n)   { (n), RAD_OPD_OUT, 0 }
#define INOUT(n) { (n), RAD_OPD_INOUT, 0 }
#define WGT(n)   { (n), RAD_OPD_WEIGHT, 0 }

#define ARR(a) (a), (int)(sizeof(a) / sizeof((a)[0]))

/* ================================================================== schemas */

static const RadParamSpec pRowsel[] = { P_INT("M"), P_CAP("cap"), P_F64("share"), P_INT("seed"),
                                        P_STR("mode") };
static const RadOperandSpec oRowsel[] = { OPD("token_ids"), WGT("score"), OUT("rows_idx"),
                                          OUT("mask") };

static const RadParamSpec pRho[] = { P_INT("M"), P_INT("n_head") };
static const RadOperandSpec oRho[] = { OPD("a"), OPD("mask"), WGT("A_log"), WGT("dt_bias"),
                                       INOUT("ND"), OPD("state_idx") };

static const RadParamSpec pCorrect[] = { P_INT("M"), P_STR("mode"), P_F64("alpha"),
                                         P_INT("n_head"), P_INT("sd0"), P_INT("sd1") };
static const RadOperandSpec oCorrect[] = { INOUT("state"), INOUT("applied"), OPD("state_idx"),
                                           WGT("C"), OPD_O("ND") };

static const RadOpSchema kSchemas[] = {
{ "kva_rowsel", ARR(pRowsel), ARR(oRowsel),
  "Which rows of ONE prefill chunk run exact (KVA quality mode). M = n rows (token_ids' extent). "
  "A row matches when score[token_ids[i]] is finite (an id outside the table does not match). "
  "k = rint(share * matches), half to even (Python's round). mode class: row i is kept iff it "
  "matches and fewer than k matching rows rank before it by (score descending, row ascending); "
  "mode random: the same k over ALL rows, ranked by a 32-bit hash of (seed, row) ascending; mode "
  "all: k = n, ranked by row. At most min(cap, rows_idx extent) rows are kept (the best). "
  "rows_idx = kept rows ascending, -1 padded; mask[i] = 1 for every row NOT kept (approximated). "
  "i32 ids, rows and mask; f32 score table." },
{ "kva_rho_update", ARR(pRho), ARR(oRho),
  "The decayed share of approximated rows in each GDN head's state, carried across one "
  "sequence's chunks (KVA quality mode). Per head h, sequentially over the n rows of `a` "
  "[n, n_head] (read at its own strides, e.g. a column slice of the a|b buffer): g = -exp(A_log[h]) * "
  "softplus(a[t,h] + dt_bias[h]); D = e^g * D + 1; N = e^g * N + mask[t]. ND is a LINEAR slot "
  "[n_states, n_head, ...] holding (N, D) per head as two f32; the slot is state_idx[0] (one "
  "sequence); a slot outside the pool writes nothing. rho = clamp(N/D, 0, 1) is read by "
  "kva_state_correct. a bf16 or f32; mask i32; A_log, dt_bias, ND f32." },
{ "kva_state_correct", ARR(pCorrect), ARR(oCorrect),
  "Apply or undo the GDN terminal-state correction (KVA +st) on each sequence's slot. M = n_seq "
  "(state_idx's rows; column 0 is the slot; a slot outside the pool is skipped). Per (sequence, "
  "head): undo: state -= applied * C; applied = 0. apply: s = alpha * (ND ? clamp(N/D, 0, 1) : 1) "
  "(1 also while D is 0); state += s * C; applied = s. Nothing is added when the scale is 0, so "
  "alpha 0 leaves the state's bits alone. state [n_states, n_head, sd0, sd1] f32 with the strides "
  "the operand carries (linear slots may be padded); applied [n_states, n_head, ...] one f32 a "
  "head; C [n_head, sd0, sd1] f32; ND as kva_rho_update's, optional. `alpha` is read in apply "
  "mode and ignored in undo." },
};

/* ================================================================== operand descriptions
 *
 * What a tool needs to call a row cold. The numbers the geometry does not carry -- the vocabulary
 * size, the state pool's slot count -- are this description's to choose, the way libr4d's
 * kv_store description chooses its block count: big enough that an index is not the identity. */

static RadOpdDesc opd(uint32_t dtype, std::initializer_list<int64_t> dims,
                      uint32_t fill = RAD_FILL_NORMAL) {
    RadOpdDesc d{};
    d.dtype = dtype;
    d.fill = fill;
    d.idx_const = -1;
    for (int64_t e : dims) d.shape[d.rank++] = e;
    return d;
}

static RadOpdDesc opd_idx(std::initializer_list<int64_t> dims, int64_t idx_max,
                          uint32_t flags = 0) {
    RadOpdDesc d = opd(RAD_I32, dims, RAD_FILL_INDEX);
    d.idx_max = idx_max;
    d.flags = flags;
    return d;
}

static RadOpdDesc opd_absent() {
    RadOpdDesc d{};
    d.idx_const = -1;
    d.flags = RAD_OPD_F_ABSENT;
    return d;
}

/* Checks the whole list first (a missing parameter shows as a non-positive extent, which is
 * RAD_E_UNSUPPORTED: the geometry does not carry what the description needs), then hands out one. */
static int pick(RadOpdDesc* out, int operand, std::initializer_list<RadOpdDesc> list) {
    for (const RadOpdDesc& d : list) {
        if (d.flags & RAD_OPD_F_ABSENT) continue;
        for (uint32_t i = 0; i < d.rank; ++i)
            if (d.shape[i] <= 0) return RAD_E_UNSUPPORTED;
    }
    if (!out || operand < 0 || operand >= (int)list.size()) return RAD_E_SHAPE;
    *out = list.begin()[operand];
    return RAD_OK;
}

static int64_t param(const RadParam* p, int n_p, const char* key) {
    return rad_param_getdim(p, n_p, key, 0);
}

static int shape_rowsel(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t n = param(p, n_p, "M"), cap = param(p, n_p, "cap");
    const int64_t vocab = 4 * n;   /* ids repeat a little, so ties and repeated tokens occur */
    return pick(out, operand, {
        opd_idx({ n }, vocab),
        opd(RAD_F32, { vocab }),
        opd_idx({ cap }, n),
        opd_idx({ n }, 2),
    });
}

static int shape_rho(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t n = param(p, n_p, "M"), heads = param(p, n_p, "n_head");
    const int64_t slots = 4;
    return pick(out, operand, {
        opd(RAD_BF16, { n, heads }),
        opd_idx({ n }, 2),
        opd(RAD_F32, { heads }),
        opd(RAD_F32, { heads }),
        opd(RAD_F32, { slots, heads, 1, 2 }, RAD_FILL_SIGMOID),
        opd_idx({ 1, 1 }, slots),
    });
}

static int shape_correct(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t n_seq = param(p, n_p, "M"), heads = param(p, n_p, "n_head");
    const int64_t sd0 = param(p, n_p, "sd0"), sd1 = param(p, n_p, "sd1");
    const int64_t slots = n_seq + 2;   /* slack, so the slot index is not the identity */
    const bool apply = std::strcmp(rad_param_gets(p, n_p, "mode", ""), "apply") == 0;
    return pick(out, operand, {
        opd(RAD_F32, { slots, heads, sd0, sd1 }),
        opd(RAD_F32, { slots, heads, 1, 1 }),
        /* Distinct: two sequences on one slot would make the update a race. */
        opd_idx({ n_seq, 1 }, slots, RAD_OPD_F_IDX_UNIQUE),
        opd(RAD_F32, { heads, sd0, sd1 }),
        apply ? opd(RAD_F32, { slots, heads, 1, 2 }, RAD_FILL_SIGMOID) : opd_absent(),
    });
}

/* ================================================================== the row table */

static const RadConstraint cRowselHost[] = { RAD_CIN("mode", "class random all") };
static const RadConstraint cRowselDevice[] = { RAD_CIN("mode", "class random all"),
                                               RAD_CLE("M", KVA_ROWSEL_MAX_ROWS) };
static const RadConstraint cCorrect[] = { RAD_CIN("mode", "undo apply") };

#define ROW(nm, opname, what, shp, dts, dom, cons, fn, shapefn)                                  \
    { nm, opname, "kva", what, shp, dts, dom, 0, ARR(cons), nullptr, 0, nullptr, nullptr,         \
      nullptr, fn, nullptr, nullptr, nullptr, shapefn, nullptr, nullptr, nullptr }
#define ROW_NC(nm, opname, what, shp, dts, dom, fn, shapefn)                                     \
    { nm, opname, "kva", what, shp, dts, dom, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,        \
      nullptr, fn, nullptr, nullptr, nullptr, shapefn, nullptr, nullptr, nullptr }

static const RadKernelInfo kKernels[] = {
ROW("kva_rowsel_host", "kva_rowsel", "chunk row selection, O(n^2) rank counting, the oracle",
    "any n and cap", "i32 ids / rows / mask, f32 score", RAD_DOMAIN_HOST, cRowselHost,
    kva_rowsel_host, shape_rowsel),
ROW_NC("kva_rho_update_host", "kva_rho_update", "per-head decayed approximated share, the oracle",
       "any n and n_head", "bf16 or f32 a, i32 mask, f32 A_log / dt_bias / ND", RAD_DOMAIN_HOST,
       kva_rho_host, shape_rho),
ROW("kva_state_correct_host", "kva_state_correct", "GDN state apply / undo, the oracle",
    "any slot strides", "f32 state / applied / C / ND, i32 slots", RAD_DOMAIN_HOST, cCorrect,
    kva_correct_host, shape_correct),
#ifdef KVA_HAVE_HIP
ROW("kva_rowsel_device", "kva_rowsel", "chunk row selection in one workgroup, keys in LDS",
    "n up to the LDS key budget (KVA_ROWSEL_MAX_ROWS), any cap", "i32 ids / rows / mask, f32 score",
    RAD_DOMAIN_DEVICE, cRowselDevice, kva_rowsel_device, shape_rowsel),
ROW_NC("kva_rho_update_device", "kva_rho_update", "one thread a head, sequential over the rows",
       "any n and n_head", "bf16 or f32 a, i32 mask, f32 A_log / dt_bias / ND", RAD_DOMAIN_DEVICE,
       kva_rho_device, shape_rho),
ROW("kva_state_correct_device", "kva_state_correct", "one workgroup per (sequence, head)",
    "any slot strides", "f32 state / applied / C / ND, i32 slots", RAD_DOMAIN_DEVICE, cCorrect,
    kva_correct_device, shape_correct),
#endif
};

/* ================================================================== plugin exports */

#ifndef KVA_BUILD_TARGET
#define KVA_BUILD_TARGET "host"
#endif

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "kva",
    "0.1.0",
    "KVA / RidgeFill prefill ops: kva_rowsel (exact rows of a chunk), kva_rho_update (decayed "
    "approximated share per GDN head), kva_state_correct (GDN terminal-state correction). A host "
    "row (the oracle) and, in a HIP build, a device row each.",
    KVA_BUILD_TARGET
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

extern "C" int rad_kernel_schema_count(void) { return (int)(sizeof kSchemas / sizeof kSchemas[0]); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    return i >= 0 && i < rad_kernel_schema_count() ? &kSchemas[i] : nullptr;
}
extern "C" int rad_kernel_count(void) { return (int)(sizeof kKernels / sizeof kKernels[0]); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return i >= 0 && i < rad_kernel_count() ? &kKernels[i] : nullptr;
}
