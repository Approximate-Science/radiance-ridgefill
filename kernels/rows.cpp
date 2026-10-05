/* rows.cpp -- the kva kernel library's op schemas, its row table and its operand descriptions.
 *
 * NEW OPS, none of them in docs/OPS.md, so this plugin FIXES their schemas (the first plugin in
 * hierarchy order to declare an op does) and nothing else in the engine knows them. The fitted
 * tensors (projector, correction, score table) are IN operands, not weights: they live in memory
 * the arch plugin fills from the projector folder (PACKAGING.md §0), so nothing here asks the
 * container for them. kva_gemm_nt_bias's rows are the engine's own gemm_nt_bias rows (forward.cpp). Each has a
 * host row -- plain C++, the oracle the device row is tested against (tests/kernel_test.cpp) --
 * and, in a HIP build, a device row. The plan they implement is PLAN.md §5 of the KVA plugin build
 * and PLAN-FIX v2 Stage A (the device mask; schemas in notes/impl.md §1).
 *
 * Every row carries an `opd_shape`, so a tool can call it cold instead of skipping it by name. No
 * row has a scratch hook (kva_mask keeps its keys in LDS; the others need nothing), tunables, a
 * layout hook or a `describe`: each is one fixed launch issued once per layer per prefill chunk.
 * No parameter carries a role (R98): nothing here is a sequence chunk or a capacity.
 */
#include "kva.h"

#include <cstring>
#include <initializer_list>

#define P_INT(k) { (k), RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }
#define P_STR(k) { (k), RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE }
#define P_F64(k) { (k), RAD_P_F64, RAD_REQUIRED, RAD_PROLE_NONE }

#define OPD(n)   { (n), RAD_OPD_IN, 0 }
#define OPD_O(n) { (n), RAD_OPD_IN, 1 }
#define OUT(n)   { (n), RAD_OPD_OUT, 0 }
#define OUT_O(n) { (n), RAD_OPD_OUT, 1 }
#define INOUT(n) { (n), RAD_OPD_INOUT, 0 }
#define INOUT_O(n) { (n), RAD_OPD_INOUT, 1 }
#define WGT(n)   { (n), RAD_OPD_WEIGHT, 0 }
#define WGT_O(n) { (n), RAD_OPD_WEIGHT, 1 }

#define ARR(a) (a), (int)(sizeof(a) / sizeof((a)[0]))

/* ================================================================== schemas */

/* M is issued at the bulk end b only to select the band: the engine hands a kernel a ranged
 * parameter at its band's upper bound (rad_abi.h "ranges collapsed"), so b itself is read as
 * token_ids' extent, which is per issue and is what a recorded pass replays. */
static const RadParamSpec pMask[] = { P_INT("M"), P_F64("share"), P_INT("seed"), P_STR("mode") };
static const RadOperandSpec oMask[] = { OPD("cu_last"), OPD("token_ids"), OPD("positions"),
                                        OPD_O("score"), OUT("mask"), OUT("bounds"), OUT_O("zeros") };

static const RadParamSpec pSelect[] = { P_INT("M") };
static const RadOperandSpec oSelect[] = { OPD("mask"), OPD("x_src"), OPD_O("q_src"), OPD_O("s_src"),
                                          INOUT("x"), INOUT_O("q"), INOUT_O("s") };

static const RadParamSpec pDrop[] = { P_INT("M"), P_INT("top_k") };
static const RadOperandSpec oDrop[] = { OPD("mask"), INOUT("ids") };

static const RadParamSpec pRho[] = { P_INT("M"), P_INT("n_head") };
static const RadOperandSpec oRho[] = { OPD("a"), OPD("mask"), WGT("A_log"), WGT("dt_bias"),
                                       INOUT("ND"), OPD("state_idx"), OPD_O("bounds") };

static const RadParamSpec pCorrect[] = { P_INT("M"), P_STR("mode"), P_F64("alpha"),
                                         P_INT("n_head"), P_INT("sd0"), P_INT("sd1") };
/* Each KV operand is followed by its own group's slot index, as gdn_recurrent_update pairs
 * `state` with `state_idx`: the groups need not share a slot per sequence. */
static const RadOperandSpec oCorrect[] = { INOUT("state"), OPD("state_idx"), INOUT("applied"),
                                           OPD("applied_idx"), OPD("C"), OPD_O("ND"),
                                           OPD_O("nd_idx"), OPD_O("bounds") };

/* libr4d's gemm_nt_bias (r4d_rows.cpp:1519-1529, docs/OPS.md:343) with `b` and `bias` as inputs. */
static const RadParamSpec pGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype"),
                                      { "act", RAD_P_STR, RAD_OPTIONAL, RAD_PROLE_NONE } };
static const RadOperandSpec oGemm[] = { OPD("a"), OPD("b"), OPD("bias"), OPD_O("res"), OUT("y") };

static const RadParamSpec pStateRead[] = { P_INT("M"), P_INT("n_head"), P_INT("sd0"), P_INT("sd1") };
static const RadOperandSpec oStateRead[] = { OPD("state"), OPD("state_idx"), OUT("out") };

static const RadOpSchema kSchemas[] = {
{ "kva_mask", ARR(pMask), ARR(oMask),
  "Which rows of the step's LAST sequence run exact (KVA, the device mask). cu_last [2] = {s, e} "
  "(cu_seqlens + n_seq - 1, read on the device); b = token_ids' extent (rows [0, b) of the step, "
  "the bulk end; M is issued at b only to pick the band); b' = min(max(b, s), e); the window is "
  "W = [s, b'). mask [n] (n = its extent) is written on every row: 0 outside W. Inside W, mode "
  "none: 1 (every row approximated); all: 0 (every row exact); step (a debug control): 1 on every "
  "row of [0, b') when W is not empty -- the step's other sequences included -- else as none; "
  "class: a row matches when "
  "score[token_ids[i]] is finite (an id outside the table does not); k = rint(share * matches in "
  "W), half to even (Python's round); a row is kept (0) iff it matches and fewer than k matching "
  "rows of W rank before it by (score descending, row ascending), else 1; random: the same k over "
  "ALL rows of W, ranked by a 32-bit hash of (seed, positions[i]) ascending, ties by row. bounds "
  "[4] = {s, b', b', e}. Refused (host row): s < 0, s > e, e > n; the device row, which reads "
  "cu_last on the device and cannot refuse, takes W empty inside the clamped [s, e] instead "
  "(every row 0). score is optional and required by class and random. `positions` is the "
  "batch's token index in its sequence (RadBatch::positions), [b] or component-major [c, b] "
  "(row 0 is read, at the operand's strides); only random mode reads it. i32 cu_last, ids, "
  "positions, mask and bounds; f32 score table. Optional `zeros` i32: every element written 0 (the "
  "zero expert offsets the arch side's stager probes read, notes/impl.md)." },
{ "kva_select", ARR(pSelect), ARR(oSelect),
  "Approximated rows take their source's values (KVA, the device mask). n = mask's extent. For "
  "every row i < n with mask[i] == 1, row i of each present destination is overwritten with row i "
  "of its source, byte for byte; nothing else is written (rows with any other mask value, rows "
  "past n, row padding). Pairs (x_src, x), (q_src, q), (s_src, s): x required, q and s optional, "
  "each pair present or absent together, one dtype and one row width (shape[1]) per pair, any "
  "dtype of whole bytes (bf16 activations, int8 or E4M3 codes, f32 scales); rank 2, last stride "
  "1, row pitches off each tensor, at least n rows each. i32 mask." },
{ "kva_drop_rows", ARR(pDrop), ARR(oDrop),
  "Approximated rows select nothing (KVA, the device mask). n = mask's extent. For every row "
  "i < n with mask[i] == 1, ids[i, 0 .. top_k) = -1; nothing else is written (columns >= top_k, "
  "rows with any other mask value, rows past n). ids i32 [>= n, >= top_k], last stride 1, row "
  "pitch off the tensor; top_k >= 0. i32 mask." },
{ "kva_rho_update", ARR(pRho), ARR(oRho),
  "The decayed share of approximated rows in each GDN head's state, carried across one "
  "sequence's chunks (KVA quality mode). Per head h, sequentially over the n rows of `a` "
  "[n, n_head] (read at its own strides, e.g. a column slice of the a|b buffer): g = -exp(A_log[h]) * "
  "softplus(a[t,h] + dt_bias[h]); D = e^g * D + 1; N = e^g * N + mask[t]. ND is a LINEAR slot "
  "[n_states, n_head, ...] holding (N, D) per head as two f32; the slot is state_idx[0] (one "
  "sequence); a slot outside the pool writes nothing. rho = clamp(N/D, 0, 1) is read by "
  "kva_state_correct. Optional `bounds` i32 [>=1] (kva_mask's): the recurrence runs over rows "
  "[clamp(bounds[0], 0, n), n) only; absent = every row. a bf16 or f32; mask i32; A_log, "
  "dt_bias, ND f32." },
{ "kva_state_correct", ARR(pCorrect), ARR(oCorrect),
  "Apply or undo the GDN terminal-state correction (KVA +st) on each sequence's slots. M = n_seq "
  "(state_idx's rows). Every KV operand has its own group's index after it -- state_idx, "
  "applied_idx, nd_idx, each [n_seq] or [n_seq, pitch] with column 0 the slot -- and a sequence "
  "with any slot outside its pool is skipped. ND and nd_idx are present or absent together. Per (sequence, "
  "head): undo: state -= applied * C; applied = 0. apply: s = alpha * (ND ? clamp(N/D, 0, 1) : 1) "
  "(1 also while D is 0); state += s * C; applied = s. Nothing is added when the scale is 0, so "
  "alpha 0 leaves the state's bits alone. state [n_states, n_head, sd0, sd1] f32 with the strides "
  "the operand carries (linear slots may be padded); applied [n_states, n_head, ...] one f32 a "
  "head; C [n_head, sd0, sd1] f32, dense (any rank of that many elements); ND as kva_rho_update's, optional. `alpha` is read in apply "
  "mode and ignored in undo. Optional `bounds` i32 [>=2] (kva_mask's {s, b', ...}): when "
  "bounds[1] <= bounds[0] (the step's last sequence had no bulk row) the op writes nothing, in "
  "either mode; otherwise it is the op without bounds." },
{ "kva_state_read", ARR(pStateRead), ARR(oStateRead),
  "Copy each sequence's GDN state slot out (KVA correction refit captures). M = n_seq "
  "(state_idx's rows; column 0 is the slot). out[s, h, i, j] = state[slot_s, h, i, j]; a slot "
  "outside the pool reads zeros. state [n_states, n_head, sd0, sd1] f32 at the strides the "
  "operand carries (linear slots may be padded); out f32, written densely from its first element "
  "as [n_seq, n_head, sd0, sd1] (any contiguous operand at least that big)." },
{ "kva_gemm_nt_bias", ARR(pGemm), ARR(oGemm),
  "gemm_nt_bias (docs/OPS.md) with the weight `b` [N, K] and `bias` [N] as IN operands -- memory "
  "the caller owns, e.g. a projector loaded from a folder rather than the container: y = res + "
  "act(a @ b^T + bias), rounded to bf16 after the biased product, after the activation and after "
  "the residual. Its rows are the engine's own gemm_nt_bias rows (libr4d's device row, libref's "
  "host row), offered only when that library is loaded." },
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

static int shape_mask(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t n = param(p, n_p, "M");
    const int64_t vocab = 4 * n;   /* ids repeat a little, so ties and repeated tokens occur */
    return pick(out, operand, {
        opd_idx({ 2 }, n, RAD_OPD_F_IDX_CU),             /* {0, M}: one sequence, all bulk */
        opd_idx({ n }, vocab),
        opd_idx({ n }, 64 * n, RAD_OPD_F_IDX_UNIQUE),   /* distinct, as a sequence's positions are */
        opd(RAD_F32, { vocab }),
        opd_idx({ n }, 2),
        opd_idx({ 4 }, n),
        opd_idx({ 8 }, 4),
    });
}

/* Row widths are this description's to choose: a bf16 activation row, odd-width int8 codes and two
 * f32 scales, so the 16-, 1- and 4-byte copy words are all exercised. */
static int shape_select(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t n = param(p, n_p, "M");
    const RadOpdDesc x = opd(RAD_BF16, { n, 64 }), q = opd(RAD_I8, { n, 66 });
    const RadOpdDesc scales = opd(RAD_F32, { n, 2 });
    return pick(out, operand, { opd_idx({ n }, 2), x, q, scales, x, q, scales });
}

/* Three columns past top_k, which the op must leave alone; ids index a KV of 4n rows. */
static int shape_drop(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t n = param(p, n_p, "M"), top_k = rad_param_getdim(p, n_p, "top_k", -1);
    return pick(out, operand, { opd_idx({ n }, 2), opd_idx({ n, top_k >= 0 ? top_k + 3 : 0 }, 4 * n) });
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
        opd_idx({ 4 }, n),   /* kva_mask's bounds: a first row inside the chunk */
    });
}

static int shape_correct(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t n_seq = param(p, n_p, "M"), heads = param(p, n_p, "n_head");
    const int64_t sd0 = param(p, n_p, "sd0"), sd1 = param(p, n_p, "sd1");
    const int64_t slots = n_seq + 2;   /* slack, so the slot index is not the identity */
    const bool apply = std::strcmp(rad_param_gets(p, n_p, "mode", ""), "apply") == 0;
    /* Distinct slots: two sequences on one slot would make the update a race. */
    const RadOpdDesc index = opd_idx({ n_seq, 1 }, slots, RAD_OPD_F_IDX_UNIQUE);
    return pick(out, operand, {
        opd(RAD_F32, { slots, heads, sd0, sd1 }),
        index,
        opd(RAD_F32, { slots, heads, 1, 1 }),
        index,
        opd(RAD_F32, { heads, sd0, sd1 }),
        apply ? opd(RAD_F32, { slots, heads, 1, 2 }, RAD_FILL_SIGMOID) : opd_absent(),
        apply ? index : opd_absent(),
        opd_absent(),   /* bounds: a drawn pair would write nothing half the time */
    });
}

static int shape_state_read(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t n_seq = param(p, n_p, "M"), heads = param(p, n_p, "n_head");
    const int64_t sd0 = param(p, n_p, "sd0"), sd1 = param(p, n_p, "sd1");
    const int64_t slots = n_seq + 2;
    return pick(out, operand, {
        opd(RAD_F32, { slots, heads, sd0, sd1 }),
        opd_idx({ n_seq, 1 }, slots),   /* a read: two sequences may name one slot */
        opd(RAD_F32, { n_seq, heads, sd0, sd1 }),
    });
}

/* ================================================================== the row table */

static const RadConstraint cMaskHost[] = { RAD_CIN("mode", "none class random all step") };
static const RadConstraint cMaskDevice[] = { RAD_CIN("mode", "none class random all step"),
                                             RAD_CLE("M", KVA_MASK_MAX_ROWS) };
static const RadConstraint cCorrect[] = { RAD_CIN("mode", "undo apply") };

#define ROW(nm, opname, what, shp, dts, dom, cons, fn, shapefn)                                  \
    { nm, opname, "kva", what, shp, dts, dom, 0, ARR(cons), nullptr, 0, nullptr, nullptr,         \
      nullptr, fn, nullptr, nullptr, nullptr, shapefn, nullptr, nullptr, nullptr }
#define ROW_NC(nm, opname, what, shp, dts, dom, fn, shapefn)                                     \
    { nm, opname, "kva", what, shp, dts, dom, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,        \
      nullptr, fn, nullptr, nullptr, nullptr, shapefn, nullptr, nullptr, nullptr }

static const RadKernelInfo kKernels[] = {
ROW("kva_mask_host", "kva_mask", "last-sequence window mask, O(w^2) rank counting, the oracle",
    "any n and b", "i32 cu_last / ids / mask / bounds, f32 score", RAD_DOMAIN_HOST, cMaskHost,
    kva_mask_host, shape_mask),
ROW_NC("kva_select_host", "kva_select", "masked row copy, bytewise, the oracle",
       "any n and row widths", "any whole-byte dtype per pair, i32 mask", RAD_DOMAIN_HOST,
       kva_select_host, shape_select),
ROW_NC("kva_drop_rows_host", "kva_drop_rows", "masked top-k id drop, the oracle",
       "any n and top_k", "i32 mask / ids", RAD_DOMAIN_HOST, kva_drop_host, shape_drop),
ROW_NC("kva_rho_update_host", "kva_rho_update", "per-head decayed approximated share, the oracle",
       "any n and n_head", "bf16 or f32 a, i32 mask, f32 A_log / dt_bias / ND", RAD_DOMAIN_HOST,
       kva_rho_host, shape_rho),
ROW("kva_state_correct_host", "kva_state_correct", "GDN state apply / undo, the oracle",
    "any slot strides", "f32 state / applied / C / ND, i32 slots", RAD_DOMAIN_HOST, cCorrect,
    kva_correct_host, shape_correct),
ROW_NC("kva_state_read_host", "kva_state_read", "GDN state slot copy-out, the oracle",
       "any slot strides", "f32 state / out, i32 slots", RAD_DOMAIN_HOST, kva_state_read_host,
       shape_state_read),
#ifdef KVA_HAVE_HIP
ROW("kva_mask_device", "kva_mask", "last-sequence window mask in one workgroup, keys in LDS",
    "b up to the LDS key budget (KVA_MASK_MAX_ROWS), any n", "i32 cu_last / ids / mask / bounds, f32 score",
    RAD_DOMAIN_DEVICE, cMaskDevice, kva_mask_device, shape_mask),
ROW_NC("kva_select_device", "kva_select", "one workgroup per row, 16 / 4 / 1-byte words",
       "any n and row widths", "any whole-byte dtype per pair, i32 mask", RAD_DOMAIN_DEVICE,
       kva_select_device, shape_select),
ROW_NC("kva_drop_rows_device", "kva_drop_rows", "one workgroup per row",
       "any n and top_k", "i32 mask / ids", RAD_DOMAIN_DEVICE, kva_drop_device, shape_drop),
ROW_NC("kva_rho_update_device", "kva_rho_update", "one thread a head, sequential over the rows",
       "any n and n_head", "bf16 or f32 a, i32 mask, f32 A_log / dt_bias / ND", RAD_DOMAIN_DEVICE,
       kva_rho_device, shape_rho),
ROW("kva_state_correct_device", "kva_state_correct", "one workgroup per (sequence, head)",
    "any slot strides", "f32 state / applied / C / ND, i32 slots", RAD_DOMAIN_DEVICE, cCorrect,
    kva_correct_device, shape_correct),
ROW_NC("kva_state_read_device", "kva_state_read", "one workgroup per (head, sequence), a copy",
       "any slot strides", "f32 state / out, i32 slots", RAD_DOMAIN_DEVICE, kva_state_read_device,
       shape_state_read),
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
    "KVA / RidgeFill prefill ops: kva_mask (which rows of the last sequence run exact, on the "
    "device), kva_select (approximated rows take their source rows), kva_drop_rows (approximated "
    "rows select no ids), kva_rho_update (decayed approximated share per GDN head), "
    "kva_state_correct (GDN terminal-state correction), kva_state_read (GDN state slot copy-out "
    "for the correction refit), kva_gemm_nt_bias (the engine's gemm_nt_bias with its weight as an "
    "input). A host row (the oracle) and, in a HIP build, a device row each.",
    KVA_BUILD_TARGET
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

extern "C" int rad_kernel_schema_count(void) { return (int)(sizeof kSchemas / sizeof kSchemas[0]); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    return i >= 0 && i < rad_kernel_schema_count() ? &kSchemas[i] : nullptr;
}
/* This library's own rows, then the forwarded gemm rows (forward.cpp), which exist only when their
 * source library is loaded -- counted at the first call, which the loader makes after it has
 * dlopened every plugin. */
static const int kOwnRows = (int)(sizeof kKernels / sizeof kKernels[0]);
extern "C" int rad_kernel_count(void) { return kOwnRows + kva_forward_count(); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i >= 0 && i < kOwnRows) return &kKernels[i];
    return kva_forward_at(i - kOwnRows);
}
/* A forwarded row keeps its source's promise; this library's own rows make none (abi/rad_abi.h:591). */
extern "C" int rad_kernel_concurrent(int i) { return i >= kOwnRows ? kva_forward_concurrent(i - kOwnRows) : 0; }
