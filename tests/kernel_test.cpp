/* kernel_test.cpp -- the kva kernel library, seen the way the engine sees it: dlopen'd, its rows
 * found by op and domain, called through RadArgs with the operands in schema order.
 *
 *   kernel_test <path/to/kva.so> host [lib.so ...]   cases that need no card (host rows, the oracles)
 *   kernel_test <path/to/kva.so> gpu  [lib.so ...]   device rows against the host rows (needs a ROCm device)
 *
 * The trailing libraries are loaded FIRST, as the engine's loader has every plugin mapped before it
 * reads kva.so's rows: libr4d and libref, whose gemm_nt_bias rows kva_gemm_nt_bias forwards to
 * (kernels/forward.cpp). Without them that op has no row, which one case checks.
 *
 * A case that cannot run prints SKIP and its reason and asserts nothing. The binary exits 77
 * (ctest's skip code) when nothing was checked at all, 1 on any failure, 0 otherwise -- a binary
 * that checked nothing is not a binary that passed (radiance tests/rad_test.h, same rule).
 */
#include <rad_abi.h>
#include <rad_plugin.h>

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef KVA_TEST_HIP
#include <hip/hip_runtime_api.h>
#endif

/* ================================================================== harness */

static int g_checks = 0, g_failures = 0;
static const char* g_case = "";
static bool g_skipped = false;

static void fail_at(const char* file, int line, const std::string& what) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL %s\n    %s:%d: %s\n", g_case, file, line, what.c_str());
}
#define CHECK(c) \
    do { ++g_checks; if (!(c)) fail_at(__FILE__, __LINE__, "CHECK(" #c ")"); } while (0)
#define CHECK_EQ(a, b)                                                                       \
    do { ++g_checks; long long _a = (long long)(a), _b = (long long)(b);                     \
         if (_a != _b) fail_at(__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b "): " +            \
                               std::to_string(_a) + " vs " + std::to_string(_b)); } while (0)
#define REQUIRE(c) \
    do { ++g_checks; if (!(c)) { fail_at(__FILE__, __LINE__, "REQUIRE(" #c ")"); return; } } while (0)

static void skip(const char* why) {
    g_skipped = true;
    std::fprintf(stderr, "  SKIP %s: %s\n", g_case, why);
}

struct Case { const char* name; const char* group; void (*fn)(); };
static std::vector<Case>& cases() { static std::vector<Case> v; return v; }
struct Reg { Reg(const char* n, const char* g, void (*f)()) { cases().push_back({ n, g, f }); } };
#define TEST(name, group) \
    static void name(); static Reg reg_##name(#name, group, name); static void name()

/* ================================================================== the plugin */

static int (*g_kernel_count)(void);
static const RadKernelInfo* (*g_kernel_at)(int);
static int (*g_schema_count)(void);
static const RadOpSchema* (*g_schema_at)(int);
static std::string g_group;
static std::vector<void*> g_preloaded;   /* the trailing libraries, in argument order */

static const RadKernelInfo* find_row(const char* op, int domain) {
    for (int i = 0; i < g_kernel_count(); ++i) {
        const RadKernelInfo* k = g_kernel_at(i);
        if (k->domain == domain && std::strcmp(k->op, op) == 0) return k;
    }
    return nullptr;
}

static const RadOpSchema* find_schema(const char* op) {
    for (int i = 0; i < g_schema_count(); ++i)
        if (std::strcmp(g_schema_at(i)->op, op) == 0) return g_schema_at(i);
    return nullptr;
}

static int group_domain() { return g_group == "gpu" ? RAD_DOMAIN_DEVICE : RAD_DOMAIN_HOST; }

/* ================================================================== seeded draws
 * splitmix64 and our own uniform/normal: <random>'s distributions are implementation-defined, so a
 * case drawn through them is a different case on another standard library. */

struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    int64_t below(int64_t n) { return n > 0 ? (int64_t)(next() % (uint64_t)n) : 0; }
    float normal() {
        const double u = uniform() + 1e-12, v = uniform();
        return (float)(std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v));
    }
};

/* ================================================================== tensors
 * A host buffer and its descriptor. Strides are elements; an empty stride list means packed. The
 * buffer covers the furthest element the strides reach, so a padded layout really is padded. */

struct Buf {
    std::vector<unsigned char> bytes;
    RadTensor t{};
    Buf() = default;
    /* A copy owns its bytes, so its descriptor points at them and not at the original's. */
    Buf(const Buf& o) : bytes(o.bytes), t(o.t) { t.data = o.t.data ? bytes.data() : nullptr; }
    Buf& operator=(const Buf& o) {
        bytes = o.bytes;
        t = o.t;
        t.data = o.t.data ? bytes.data() : nullptr;
        return *this;
    }
};

static Buf make(uint32_t dtype, std::vector<int64_t> shape, std::vector<int64_t> stride = {}) {
    Buf b;
    b.t.dtype = dtype;
    b.t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i) b.t.shape[i] = shape[i];
    if (stride.empty()) rad_tensor_pack(&b.t);
    else for (size_t i = 0; i < stride.size(); ++i) b.t.stride[i] = stride[i];
    int64_t last = 0;
    bool empty = false;
    for (uint32_t i = 0; i < b.t.rank; ++i) {
        last += (b.t.shape[i] - 1) * b.t.stride[i];
        empty = empty || b.t.shape[i] == 0;
    }
    /* A zero-extent operand is PRESENT and empty (one spare byte keeps its data non-null), as an
     * engine operand of zero rows is; a null data pointer would read as absent. */
    b.bytes.assign(empty ? 1 : (size_t)rad_dtype_bytes(dtype, b.t.rank ? last + 1 : 0), 0);
    b.t.data = b.bytes.data();
    return b;
}

static float getf(const Buf& b, int64_t i) { return rad_load_f32(b.bytes.data(), b.t.dtype, i); }
static void setf(Buf& b, int64_t i, float v) { rad_store_f32(b.bytes.data(), b.t.dtype, i, v); }
static int32_t geti(const Buf& b, int64_t i) {
    int32_t v; std::memcpy(&v, b.bytes.data() + 4 * i, 4); return v;
}
static void seti(Buf& b, int64_t i, int32_t v) { std::memcpy(b.bytes.data() + 4 * i, &v, 4); }

static RadParam pint(const char* k, long long v) { return RadParam{ k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
static RadParam pstr(const char* k, const char* v) { return RadParam{ k, RAD_P_STR, 0, 0, v, 0.0 }; }
static RadParam pf64(const char* k, double v) { return RadParam{ k, RAD_P_F64, 0, 0, nullptr, v }; }

/* ================================================================== running a row */

static int launch(const RadKernelInfo* row, const std::vector<RadTensor>& t,
                  const std::vector<RadParam>& p, RadStream stream, void* scratch = nullptr,
                  int64_t scratch_bytes = 0) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size();
    a.p = p.data(); a.n_p = (int)p.size();
    a.world_size = 1;
    a.scratch = scratch; a.scratch_bytes = scratch_bytes;
    return row->launch(&a, stream);
}

/* The scratch a row's hook asks for at these operands (0 without a hook). */
static int64_t scratch_of(const RadKernelInfo* row, const std::vector<RadTensor>& t, const std::vector<RadParam>& p) {
    if (!row->scratch) return 0;
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size();
    a.p = p.data(); a.n_p = (int)p.size();
    a.world_size = 1;
    return row->scratch(&a);
}

/* A host row on host buffers. A null Buf is an absent optional operand (null data). */
static int run_host(const RadKernelInfo* row, std::vector<Buf*> opds, const std::vector<RadParam>& p) {
    std::vector<RadTensor> t(opds.size());
    for (size_t i = 0; i < opds.size(); ++i) if (opds[i]) t[i] = opds[i]->t;
    return launch(row, t, p, nullptr);
}

#ifdef KVA_TEST_HIP
static bool hip_ok(hipError_t e, const char* what) {
    if (e == hipSuccess) return true;
    fail_at(__FILE__, __LINE__, std::string(what) + ": " + hipGetErrorString(e));
    return false;
}

/* A device row on device copies of the buffers: every buffer is copied in, the row runs on its own
 * stream, and every buffer is copied back (inputs come back unchanged, which is also checked by
 * the cases that compare them). */
static int run_device(const RadKernelInfo* row, std::vector<Buf*> opds, const std::vector<RadParam>& p) {
    std::vector<RadTensor> t(opds.size());
    std::vector<void*> dev(opds.size(), nullptr);
    hipStream_t stream = nullptr;
    int rc = RAD_E_DEVICE;
    if (!hip_ok(hipStreamCreate(&stream), "hipStreamCreate")) return rc;
    bool ok = true;
    for (size_t i = 0; i < opds.size() && ok; ++i) {
        if (!opds[i]) continue;
        ok = hip_ok(hipMalloc(&dev[i], opds[i]->bytes.size() + 1), "hipMalloc") &&
             hip_ok(hipMemcpy(dev[i], opds[i]->bytes.data(), opds[i]->bytes.size(),
                              hipMemcpyHostToDevice), "copy in");
        t[i] = opds[i]->t;
        t[i].data = dev[i];
    }
    void* scratch = nullptr;
    const int64_t scratch_bytes = ok ? scratch_of(row, t, p) : 0;
    if (scratch_bytes > 0) ok = hip_ok(hipMalloc(&scratch, (size_t)scratch_bytes), "hipMalloc scratch");
    if (ok) {
        rc = launch(row, t, p, (RadStream)stream, scratch, scratch_bytes);
        ok = hip_ok(hipStreamSynchronize(stream), "kernel");
    }
    if (scratch) (void)hipFree(scratch);
    for (size_t i = 0; i < opds.size() && ok; ++i)
        if (opds[i]) ok = hip_ok(hipMemcpy(opds[i]->bytes.data(), dev[i], opds[i]->bytes.size(),
                                           hipMemcpyDeviceToHost), "copy out");
    for (void* d : dev) if (d) (void)hipFree(d);
    (void)hipStreamDestroy(stream);
    return ok ? rc : RAD_E_DEVICE;
}

static bool have_device() {
    static int state = -1;
    if (state < 0) {
        int n = 0;
        state = hipGetDeviceCount(&n) == hipSuccess && n > 0;
        if (state) {
            char bus[64] = "?";
            (void)hipDeviceGetPCIBusId(bus, sizeof bus, 0);
            std::fprintf(stderr, "  device 0 of %d visible, PCI %s\n", n, bus);
        }
    }
    return state == 1;
}
#endif

/* The run the group calls for: the host row on host memory, or the device row on device memory.
 * Returns false (having printed SKIP) when this build or machine cannot run the group. */
static bool group_runnable() {
    if (g_group != "gpu") return true;
#ifdef KVA_TEST_HIP
    if (have_device()) return true;
    skip("no ROCm device visible");
#else
    skip("built without HIP: no device rows");
#endif
    return false;
}

static int run_group(const RadKernelInfo* row, std::vector<Buf*> opds, const std::vector<RadParam>& p) {
#ifdef KVA_TEST_HIP
    if (row->domain == RAD_DOMAIN_DEVICE) return run_device(row, opds, p);
#endif
    return run_host(row, opds, p);
}

/* ================================================================== described operands
 * Operands built from a row's own opd_shape hook -- what a tool calling the row cold would build. */

/* `m` is the geometry's M: a cumulative-lengths operand is [0 .. M] spread evenly, as the engine's
 * tools fill one (the batch's shape, not a draw). */
static Buf from_desc(const RadOpdDesc& d, int64_t prev_extent, int64_t m, Rng& r) {
    std::vector<int64_t> shape(d.shape, d.shape + d.rank);
    Buf b = make(d.dtype, shape);
    const int64_t n = rad_tensor_numel(&b.t);
    const int64_t hi = d.idx_max > 0 ? d.idx_max : prev_extent;
    std::vector<int32_t> perm;
    if (d.flags & RAD_OPD_F_IDX_UNIQUE)
        for (int64_t i = 0; i < hi; ++i) perm.push_back((int32_t)i);
    for (int64_t i = 0; i < n; ++i) {
        if (d.fill == RAD_FILL_INDEX && (d.flags & RAD_OPD_F_IDX_CU)) {
            seti(b, i, (int32_t)(n > 1 ? i * m / (n - 1) : m));
        } else if (d.fill == RAD_FILL_INDEX && (d.flags & RAD_OPD_F_IDX_UNIQUE)) {
            const int64_t j = i + r.below((int64_t)perm.size() - i);
            std::swap(perm[(size_t)i], perm[(size_t)j]);
            seti(b, i, perm[(size_t)i]);
        } else if (d.fill == RAD_FILL_INDEX) {
            seti(b, i, (int32_t)(d.idx_const >= 0 ? d.idx_const : r.below(hi)));
        } else if (d.fill == RAD_FILL_SIGMOID) {
            setf(b, i, (float)(0.01 + 0.98 * r.uniform()));
        } else {
            setf(b, i, r.normal());
        }
    }
    return b;
}

/* The geometry each op is described and exercised at in the generic cases. */
static std::vector<RadParam> described_params(const char* op) {
    if (!std::strcmp(op, "kva_mask"))
        return { pint("M", 64), pf64("share", 0.25), pint("seed", 7), pstr("mode", "class") };
    if (!std::strcmp(op, "kva_select")) return { pint("M", 37) };
    if (!std::strcmp(op, "kva_drop_rows")) return { pint("M", 37), pint("top_k", 5) };
    if (!std::strcmp(op, "kva_rho_update")) return { pint("M", 48), pint("n_head", 4) };
    if (!std::strcmp(op, "kva_state_read"))
        return { pint("M", 3), pint("n_head", 2), pint("sd0", 8), pint("sd1", 8) };
    return { pint("M", 3), pstr("mode", "apply"), pf64("alpha", 1.0), pint("n_head", 2),
             pint("sd0", 8), pint("sd1", 8) };
}

/* Builds a row's operands from its description; false if the description declined. */
static bool described_operands(const RadKernelInfo* row, const std::vector<RadParam>& p,
                               std::vector<Buf>& store, std::vector<Buf*>& opds) {
    const RadOpSchema* sc = find_schema(row->op);
    if (!sc || !row->opd_shape) return false;
    Rng r{ 1234 };
    store.assign((size_t)sc->n_operands, Buf{});
    opds.assign((size_t)sc->n_operands, nullptr);
    int64_t prev = 0;
    for (int i = 0; i < sc->n_operands; ++i) {
        RadOpdDesc d{};
        if (row->opd_shape(p.data(), (int)p.size(), i, &d) != RAD_OK) return false;
        if (d.flags & RAD_OPD_F_ABSENT) continue;
        store[(size_t)i] = from_desc(d, prev, rad_param_getdim(p.data(), (int)p.size(), "M", 0), r);
        opds[(size_t)i] = &store[(size_t)i];
        prev = d.shape[0];
    }
    return true;
}

static const char* const kOps[] = { "kva_mask", "kva_select", "kva_drop_rows", "kva_rho_update",
                                    "kva_state_correct", "kva_state_read", "kva_hazard" };

/* Rows whose launch is still a stub (R8). Emptied as each row was implemented (all six were stubs
 * at the Stage 1 commit, 43bfeda); with none left the case skips and says R8 is retired. */
static const char* const kStubbed[] = { nullptr };

static bool is_stub(const char* name) {
    for (const char* s : kStubbed) if (s && !std::strcmp(s, name)) return true;
    return false;
}

/* ================================================================== generic cases */

/* Every op has a schema, a row in this group's domain, and a description that covers exactly the
 * schema's operands (one more is refused, which is how a drifted hook shows). */
TEST(descriptions_cover_schemas, "both") {
    if (!group_runnable()) return;
    for (const char* op : kOps) {
        const RadOpSchema* sc = find_schema(op);
        const RadKernelInfo* row = find_row(op, group_domain());
        REQUIRE(sc != nullptr);
        REQUIRE(row != nullptr);
        REQUIRE(row->opd_shape != nullptr);
        const std::vector<RadParam> p = described_params(op);
        for (int i = 0; i < sc->n_operands; ++i) {
            RadOpdDesc d{};
            CHECK_EQ(row->opd_shape(p.data(), (int)p.size(), i, &d), RAD_OK);
        }
        RadOpdDesc d{};
        CHECK_EQ(row->opd_shape(p.data(), (int)p.size(), sc->n_operands, &d), RAD_E_SHAPE);
    }
}

/* R8: a stubbed launch refuses with RAD_E_UNSUPPORTED, reported by name the way the engine
 * reports a refusing kernel (core/runtime/issue.cpp abort_step). */
TEST(stub_refuses, "both") {
    if (!group_runnable()) return;
    int stubs = 0;
    for (const char* op : kOps) {
        const RadKernelInfo* row = find_row(op, group_domain());
        REQUIRE(row != nullptr);
        if (!is_stub(row->name)) continue;
        ++stubs;
        std::vector<Buf> store;
        std::vector<Buf*> opds;
        const std::vector<RadParam> p = described_params(op);
        REQUIRE(described_operands(row, p, store, opds));
        const int rc = run_group(row, opds, p);
        std::fprintf(stderr, "  kernel %s (kva, %s domain) refused op '%s': %s\n", row->name,
                     row->domain == RAD_DOMAIN_HOST ? "host" : "device", op, rad_strerror(rc));
        CHECK_EQ(rc, RAD_E_UNSUPPORTED);
    }
    if (stubs == 0) skip("no stubbed row remains in this domain (R8 retired: every row implemented)");
}

/* Every implemented row accepts the operands its own description builds. */
TEST(described_operands_launch, "both") {
    if (!group_runnable()) return;
    int implemented = 0;
    for (const char* op : kOps) {
        const RadKernelInfo* row = find_row(op, group_domain());
        REQUIRE(row != nullptr);
        if (is_stub(row->name)) continue;
        ++implemented;
        std::vector<Buf> store;
        std::vector<Buf*> opds;
        const std::vector<RadParam> p = described_params(op);
        REQUIRE(described_operands(row, p, store, opds));
        CHECK_EQ(run_group(row, opds, p), RAD_OK);
    }
    if (implemented == 0) skip("every row in this domain is still a stub");
}

/* R98: no parameter of any schema carries RAD_PROLE_SEQ_CHUNK (the scheduler would cut steps on
 * it), and with `cap` gone none carries RAD_PROLE_CAPACITY either. */
TEST(params_carry_no_role, "both") {
    if (!group_runnable()) return;
    for (int i = 0; i < g_schema_count(); ++i) {
        const RadOpSchema* sc = g_schema_at(i);
        for (int j = 0; j < sc->n_params; ++j) {
            const int role = sc->params[j].role;
            CHECK(role == RAD_PROLE_NONE || role == RAD_PROLE_CAPACITY);   /* R98 as written */
            CHECK_EQ(role, RAD_PROLE_NONE);                                 /* and in fact none */
        }
    }
    CHECK_EQ(g_schema_count(), (int)(sizeof kOps / sizeof kOps[0]) + 2);   /* + the two forwarded GEMMs */
}

/* ================================================================== op runners
 * Each builds one op's operands from plain vectors, runs the given row (host or device, by its
 * domain), and hands back the outputs. Output buffers start as a sentinel, so a row that leaves an
 * element unwritten shows as a mismatch rather than as a lucky zero. */

static const int32_t kSentinel = 0x5A5A5A5A;

/* One kva_mask call. token_ids is `ids` (so b is its extent), the mask has n rows, cu_last is
 * {s, e}; row i's position is first_pos + i, laid out component-major [components, b] when
 * components > 1 (row 0 the index, the rest junk). A null table passes `score` absent. */
struct MaskCall {
    std::vector<int32_t> ids;
    int64_t n = 0;
    int32_t s = 0, e = 0;
    const std::vector<float>* table = nullptr;
    double share = 0.25;
    long long seed = 0;
    const char* mode = "class";
    int32_t first_pos = 0;
    int64_t components = 1;
    int64_t n_zeros = 0;   /* > 0: pass the optional `zeros` output, this many elements */
};

struct MaskOut { int rc = 0; std::vector<int32_t> mask, bounds, zeros; };

static MaskOut run_mask(const RadKernelInfo* row, const MaskCall& c) {
    const int64_t b = (int64_t)c.ids.size(), vocab = c.table ? (int64_t)c.table->size() : 1;
    Buf cu = make(RAD_I32, { 2 }), tok = make(RAD_I32, { b }), score = make(RAD_F32, { vocab });
    Buf mask = make(RAD_I32, { c.n }), bounds = make(RAD_I32, { 4 });
    Buf pos = c.components > 1 ? make(RAD_I32, { c.components, b }) : make(RAD_I32, { b });
    seti(cu, 0, c.s);
    seti(cu, 1, c.e);
    for (int64_t i = 0; i < b; ++i) seti(tok, i, c.ids[(size_t)i]);
    for (int64_t i = 0; i < c.components * b; ++i)
        seti(pos, i, i < b ? c.first_pos + (int32_t)i : 7777 + (int32_t)i);
    for (int64_t i = 0; c.table && i < vocab; ++i) setf(score, i, (*c.table)[(size_t)i]);
    for (int64_t i = 0; i < c.n; ++i) seti(mask, i, kSentinel);
    for (int64_t i = 0; i < 4; ++i) seti(bounds, i, kSentinel);
    Buf zeros = make(RAD_I32, { c.n_zeros > 0 ? c.n_zeros : 1 });
    for (int64_t i = 0; i < c.n_zeros; ++i) seti(zeros, i, kSentinel);
    MaskOut o;
    o.rc = run_group(row, { &cu, &tok, &pos, c.table ? &score : nullptr, &mask, &bounds,
                            c.n_zeros > 0 ? &zeros : nullptr },
                     { pint("M", b), pf64("share", c.share), pint("seed", c.seed), pstr("mode", c.mode) });
    for (int64_t i = 0; i < c.n; ++i) o.mask.push_back(geti(mask, i));
    for (int64_t i = 0; i < 4; ++i) o.bounds.push_back(geti(bounds, i));
    for (int64_t i = 0; i < c.n_zeros; ++i) o.zeros.push_back(geti(zeros, i));
    return o;
}

/* The rows of [lo, hi) the mask keeps exact (0), and how many rows of the whole mask are 1. */
static std::vector<int32_t> zero_rows(const MaskOut& o, int64_t lo, int64_t hi) {
    std::vector<int32_t> k;
    for (int64_t i = lo; i < hi; ++i) if (o.mask[(size_t)i] == 0) k.push_back((int32_t)i);
    return k;
}

static int64_t ones(const MaskOut& o) {
    int64_t c = 0;
    for (int32_t m : o.mask) c += m == 1;
    return c;
}

/* Every mask row 0 or 1, every row outside [s, end) 0, bounds {s, end, end, e}. */
static bool mask_shaped(const MaskOut& o, int32_t s, int32_t end, int32_t e) {
    for (size_t i = 0; i < o.mask.size(); ++i)
        if (o.mask[i] != 0 && (o.mask[i] != 1 || (int64_t)i < s || (int64_t)i >= end)) return false;
    return o.bounds == std::vector<int32_t>{ s, end, end, e };
}

/* kva_select's operands: sources filled with a pattern, destinations with a sentinel byte, so a
 * byte the op should not write shows. A pair whose source is an empty Buf is passed absent. */
struct SelectRun { Buf mask, xs, qs, ss, x, q, s; };

static void fill_pattern(Buf& b, int tag) {
    for (size_t i = 0; i < b.bytes.size(); ++i) b.bytes[i] = (unsigned char)(tag * 31 + i * 7 + 1);
}

static void fill_sentinel(Buf& b) { std::fill(b.bytes.begin(), b.bytes.end(), (unsigned char)0xA5); }

static int run_select(const RadKernelInfo* row, SelectRun& k) {
    auto opd = [](Buf& b) { return b.t.data ? &b : nullptr; };
    return run_group(row, { &k.mask, &k.xs, opd(k.qs), opd(k.ss), &k.x, opd(k.q), opd(k.s) },
                     { pint("M", k.mask.t.shape[0]) });
}

/* After kva_select: every destination byte is its source's on rows i < n with mask[i] == 1 inside
 * the row width, and what it was before everywhere else (padding, rows past n, other mask values). */
static bool selected_exactly(const Buf& src, const Buf& before, const Buf& after,
                             const std::vector<int32_t>& mask) {
    const int64_t bytes = rad_dtype_bytes(src.t.dtype, 1), width = src.t.shape[1] * bytes;
    const int64_t src_pitch = src.t.stride[0] * bytes, dst_pitch = before.t.stride[0] * bytes;
    if (after.bytes.size() != before.bytes.size()) return false;
    for (size_t at = 0; at < after.bytes.size(); ++at) {
        const int64_t row = (int64_t)at / dst_pitch, col = (int64_t)at % dst_pitch;
        const bool copied = row < (int64_t)mask.size() && mask[(size_t)row] == 1 && col < width;
        if (after.bytes[at] != (copied ? src.bytes[(size_t)(row * src_pitch + col)] : before.bytes[at]))
            return false;
    }
    return true;
}

static Buf mask_buf(const std::vector<int32_t>& mask) {
    Buf m = make(RAD_I32, { (int64_t)mask.size() });
    for (size_t i = 0; i < mask.size(); ++i) seti(m, (int64_t)i, mask[i]);
    return m;
}

static int run_drop(const RadKernelInfo* row, Buf& mask, Buf& ids, long long top_k) {
    return run_group(row, { &mask, &ids }, { pint("M", mask.t.shape[0]), pint("top_k", top_k) });
}

/* After kva_drop_rows: -1 exactly on columns < top_k of rows i < n with mask[i] == 1, every other
 * element (columns >= top_k, padding, rows past n) as before. */
static bool dropped_exactly(const Buf& before, const Buf& after, const std::vector<int32_t>& mask,
                            int64_t top_k) {
    const int64_t pitch = before.t.stride[0];
    for (int64_t at = 0; at < (int64_t)after.bytes.size() / 4; ++at) {
        const int64_t row = at / pitch, col = at % pitch;
        const bool dropped = row < (int64_t)mask.size() && mask[(size_t)row] == 1 && col < top_k;
        if (geti(after, at) != (dropped ? -1 : geti(before, at))) return false;
    }
    return true;
}

struct RhoRun {
    int64_t n = 0, heads = 0, pitch = 0, col = 1, slots = 4;
    bool bf16 = false;
    int32_t slot = 0;
    std::vector<float> a;            /* [n, heads], row-major dense; placed at `pitch` */
    std::vector<int32_t> mask;
    std::vector<float> a_log, dt_bias;
    std::vector<float> nd;           /* [slots, heads, 2] (N, D); updated in place by run_rho */
    std::vector<int32_t> bounds;     /* kva_mask's bounds operand; empty = absent */
};

/* `bounds_as` replaces c.bounds by an arbitrary operand (the refusal cases). */
static int run_rho(const RadKernelInfo* row, RhoRun& c, const Buf* bounds_as = nullptr) {
    const int64_t pitch = c.pitch ? c.pitch : c.heads;
    Buf a = make(c.bf16 ? RAD_BF16 : RAD_F32, { c.n, c.heads }, { pitch, c.col });
    Buf mask = make(RAD_I32, { c.n }), alog = make(RAD_F32, { c.heads });
    Buf dtb = make(RAD_F32, { c.heads }), nd = make(RAD_F32, { c.slots, c.heads, 1, 2 });
    Buf sidx = make(RAD_I32, { 1, 1 });
    for (int64_t t = 0; t < c.n; ++t) {
        seti(mask, t, c.mask[(size_t)t]);
        for (int64_t h = 0; h < c.heads; ++h)
            setf(a, t * pitch + h * c.col, c.a[(size_t)(t * c.heads + h)]);
    }
    for (int64_t h = 0; h < c.heads; ++h) {
        setf(alog, h, c.a_log[(size_t)h]);
        setf(dtb, h, c.dt_bias[(size_t)h]);
    }
    for (size_t i = 0; i < c.nd.size(); ++i) setf(nd, (int64_t)i, c.nd[i]);
    seti(sidx, 0, c.slot);
    Buf bounds = make(RAD_I32, { (int64_t)c.bounds.size() });
    for (size_t i = 0; i < c.bounds.size(); ++i) seti(bounds, (int64_t)i, c.bounds[i]);
    if (bounds_as) bounds = *bounds_as;
    Buf* bounds_opd = bounds_as || !c.bounds.empty() ? &bounds : nullptr;
    const int rc = run_group(row, { &a, &mask, &alog, &dtb, &nd, &sidx, bounds_opd },
                             { pint("M", c.n), pint("n_head", c.heads) });
    for (size_t i = 0; i < c.nd.size(); ++i) c.nd[i] = getf(nd, (int64_t)i);
    return rc;
}

/* A random rho case: realistic gate ranges (A_log in [-4, 2.5], dt_bias in [-3, 1]). */
static RhoRun random_rho(Rng& r, int64_t n, int64_t heads, double exact_share, bool bf16) {
    RhoRun c;
    c.n = n; c.heads = heads; c.bf16 = bf16; c.slot = 2;
    for (int64_t i = 0; i < n * heads; ++i) {
        const float v = 2.0f * r.normal();
        c.a.push_back(bf16 ? rad_bf16_to_f32(rad_f32_to_bf16(v)) : v);
    }
    for (int64_t t = 0; t < n; ++t) c.mask.push_back(r.uniform() < exact_share ? 0 : 1);
    for (int64_t h = 0; h < heads; ++h) {
        c.a_log.push_back((float)(-4.0 + 6.5 * r.uniform()));
        c.dt_bias.push_back((float)(-3.0 + 4.0 * r.uniform()));
    }
    c.nd.assign((size_t)(c.slots * heads * 2), 0.0f);
    return c;
}

/* Rows [s, n) of a case alone: what kva_rho_update with bounds {s, ..} must equal. */
static RhoRun rho_slice(const RhoRun& c, int64_t s) {
    RhoRun out = c;
    out.n = c.n - s;
    out.a.erase(out.a.begin(), out.a.begin() + s * c.heads);
    out.mask.erase(out.mask.begin(), out.mask.begin() + s);
    out.bounds.clear();
    return out;
}

static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static float nd_rho(const RhoRun& c, int64_t h) {
    const float n = c.nd[(size_t)((c.slot * c.heads + h) * 2)], d = c.nd[(size_t)((c.slot * c.heads + h) * 2 + 1)];
    if (!(d > 0.0f)) return 1.0f;
    const float q = n / d;
    return q < 0.0f ? 0.0f : (q > 1.0f ? 1.0f : q);
}

/* The state-correction operands at a padded layout: rows of the head padded by `row_pad`, heads by
 * `head_pad`, slots by `slot_pad`, so a row that assumed a dense pool writes into the padding. */
struct CorrectRun {
    int64_t slots, heads, sd0, sd1;
    Buf state, sidx, applied, aidx, c, nd, nidx;   /* each pool with its own slot index */
};

/* One sequence's slot per row in column 0, and a speculation column the op must not read. */
static Buf slot_index(const std::vector<int32_t>& seq_slots) {
    Buf b = make(RAD_I32, { (int64_t)seq_slots.size(), 2 });
    for (size_t s = 0; s < seq_slots.size(); ++s) {
        seti(b, (int64_t)(2 * s), seq_slots[s]);
        seti(b, (int64_t)(2 * s + 1), 999);
    }
    return b;
}

/* All three pools `slots` deep and indexed alike unless the caller replaces a pool or an index. */
static CorrectRun correct_operands(int64_t slots, int64_t heads, int64_t sd0, int64_t sd1,
                                   const std::vector<int32_t>& seq_slots, int64_t pad) {
    CorrectRun k{ slots, heads, sd0, sd1, {}, {}, {}, {}, {}, {}, {} };
    const int64_t row = sd1 + pad, head = sd0 * row + pad, slot = heads * head + pad;
    k.state = make(RAD_F32, { slots, heads, sd0, sd1 }, { slot, head, row, 1 });
    k.applied = make(RAD_F32, { slots, heads, 1, 1 });
    k.c = make(RAD_F32, { heads, sd0, sd1 });
    k.nd = make(RAD_F32, { slots, heads, 1, 2 });
    k.sidx = k.aidx = k.nidx = slot_index(seq_slots);
    return k;
}

/* `bounds` is kva_mask's bounds operand; empty = absent. */
static int run_correct(const RadKernelInfo* row, CorrectRun& k, const char* mode, double alpha,
                       bool with_nd, const std::vector<int32_t>& bounds = {}) {
    Buf b = make(RAD_I32, { (int64_t)bounds.size() });
    for (size_t i = 0; i < bounds.size(); ++i) seti(b, (int64_t)i, bounds[i]);
    return run_group(row, { &k.state, &k.sidx, &k.applied, &k.aidx, &k.c,
                            with_nd ? &k.nd : nullptr, with_nd ? &k.nidx : nullptr,
                            bounds.empty() ? nullptr : &b },
                     { pint("M", k.sidx.t.shape[0]), pstr("mode", mode), pf64("alpha", alpha),
                       pint("n_head", k.heads), pint("sd0", k.sd0), pint("sd1", k.sd1) });
}

/* kva_state_read on k.state through `index`; `out` starts as a sentinel so an unwritten element
 * shows. Returns the launch status. */
static int run_state_read(const RadKernelInfo* row, CorrectRun& k, Buf& index, Buf& out) {
    out = make(RAD_F32, { index.t.shape[0], k.heads, k.sd0, k.sd1 });
    for (int64_t i = 0; i < rad_tensor_numel(&out.t); ++i) setf(out, i, 12345.0f);
    return run_group(row, { &k.state, &index, &out },
                     { pint("M", index.t.shape[0]), pint("n_head", k.heads), pint("sd0", k.sd0),
                       pint("sd1", k.sd1) });
}

/* Element (slot, head, i, j) of the state, through its strides. */
static int64_t st_at(const CorrectRun& k, int64_t s, int64_t h, int64_t i, int64_t j) {
    return s * k.state.t.stride[0] + h * k.state.t.stride[1] + i * k.state.t.stride[2] + j;
}

/* ================================================================== host semantics */

/* Hand-built step: four rows of other sequences (s = 4), the last sequence's ten bulk rows [4, 14)
 * and a two-row tail [14, 16). The window has ties, a non-finite score of every kind, an id past
 * the table and k at .5; the rows outside it carry the table's best id, so a row that ranked from
 * row 0 instead of s, or past b', would keep them. */
TEST(mask_class_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_mask", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    const std::vector<float> table = { 5.0f, 3.0f, -INFINITY, 5.0f, 1.0f, NAN, INFINITY, 2.0f };
    MaskCall c;
    /* window rows j:          0  1  2  3  4  5  6  7  8  9   matches: j 0 1 3 4 5 9 (six) */
    c.ids = { 0, 0, 0, 0,      0, 1, 2, 3, 4, 0, 9, 5, 6, 7 };   /* b = 14 */
    c.n = 16; c.s = 4; c.e = 16; c.table = &table;
    struct Want { double share; std::vector<int32_t> window; } wants[] = {
        { 0.5,        { 0, 1, 1, 0, 1, 0, 1, 1, 1, 1 } },   /* k 3: the three 5.0s, by row */
        { 0.75,       { 0, 0, 1, 0, 1, 0, 1, 1, 1, 1 } },   /* 4.5 -> 4 (half to even): + the 3.0 */
        { 0.25,       { 0, 1, 1, 0, 1, 1, 1, 1, 1, 1 } },   /* 1.5 -> 2 */
        { 1.0 / 12.0, { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } },   /* 0.5 -> 0 */
        { 5.0 / 12.0, { 0, 1, 1, 0, 1, 1, 1, 1, 1, 1 } },   /* 2.5 -> 2 */
        { 1.0,        { 0, 0, 1, 0, 0, 0, 1, 1, 1, 0 } },   /* every match */
    };
    for (const Want& w : wants) {
        c.share = w.share;
        const MaskOut o = run_mask(row, c);
        std::vector<int32_t> want(16, 0);
        std::copy(w.window.begin(), w.window.end(), want.begin() + 4);
        CHECK_EQ(o.rc, RAD_OK);
        CHECK(o.mask == want);
        CHECK(o.bounds == (std::vector<int32_t>{ 4, 14, 14, 16 }));
    }
}

/* The optional `zeros` output is written 0 on every element, whatever the window -- the stager
 * probes' expert offsets (notes/impl.md) -- and its absence changes nothing else. */
TEST(mask_writes_zeros_when_asked, "host") {
    const RadKernelInfo* row = find_row("kva_mask", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    MaskCall c;
    c.ids = std::vector<int32_t>(64, 3);
    c.n = 80; c.s = 16; c.e = 80; c.mode = "none";
    const MaskOut plain = run_mask(row, c);
    c.n_zeros = 513;
    const MaskOut with = run_mask(row, c);
    CHECK_EQ(with.rc, RAD_OK);
    CHECK(with.zeros == std::vector<int32_t>(513, 0));
    CHECK(with.mask == plain.mask && with.bounds == plain.bounds);
}

/* b' = min(max(b, s), e): a bulk end before s is an empty window, one past e stops at e; an empty
 * last sequence (s == e) and s == b are empty windows; none writes 1 exactly on W, all writes 0. */
TEST(mask_window_clamping, "host") {
    const RadKernelInfo* row = find_row("kva_mask", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    struct Win { int64_t b, n; int32_t s, e, end; } wins[] = {
        { 3, 10, 5, 9, 5 },     /* b < s: empty, bounds {5, 5, 5, 9} */
        { 12, 12, 2, 8, 8 },    /* b > e: W = [2, 8) */
        { 5, 10, 5, 9, 5 },     /* b == s: empty */
        { 6, 6, 6, 6, 6 },      /* s == e == n: the last sequence has no row */
        { 10, 10, 0, 10, 10 },  /* one sequence, every row bulk */
        { 7, 12, 3, 12, 7 },    /* bulk then a tail [7, 12) */
        { 0, 9, 0, 9, 0 },      /* b = 0: token_ids of zero rows, an empty window */
        { 0, 9, 3, 9, 3 },
    };
    for (const Win& w : wins) {
        MaskCall c;
        c.ids.assign((size_t)w.b, 1);
        c.n = w.n; c.s = w.s; c.e = w.e; c.mode = "none";
        const MaskOut none = run_mask(row, c);
        CHECK_EQ(none.rc, RAD_OK);
        CHECK(mask_shaped(none, w.s, w.end, w.e));
        CHECK_EQ(ones(none), w.end - w.s);
        c.mode = "all";
        const MaskOut all = run_mask(row, c);
        CHECK(mask_shaped(all, w.s, w.end, w.e) && ones(all) == 0);
        c.mode = "step";   /* R54's negative control: from row 0 when the window is not empty */
        const MaskOut step = run_mask(row, c);
        std::vector<int32_t> want((size_t)w.n, 0);
        for (int64_t i = w.end > w.s ? 0 : w.s; i < w.end; ++i) want[(size_t)i] = 1;
        CHECK_EQ(step.rc, RAD_OK);
        CHECK(step.mask == want && step.bounds == none.bounds);
    }
}

/* One kva_hazard call on a 4-slot meta pool; `meta` in and out, the count accumulated. The last
 * sequence is [s, e) of the step; positions are first_pos + row. bounds absent when b <= 0. */
struct HazardCall {
    int32_t s = 0, e = 64, first_pos = 0, slot = 1;
    int64_t span = 0, b = 0;
    float   m0 = 0, m1 = 0, count = 0;
};
struct HazardOut { int rc = 0; float m0 = 0, m1 = 0, count = 0; };

static HazardOut run_hazard(const RadKernelInfo* row, const HazardCall& c) {
    const int64_t n = c.e > 1 ? c.e : 1;
    Buf cu = make(RAD_I32, { 2 }), pos = make(RAD_I32, { n }), bounds = make(RAD_I32, { 4 });
    Buf span = make(RAD_I32, { c.span > 0 ? c.span : 1 }), meta = make(RAD_F32, { 4, 1, 1, 2 });
    Buf idx = make(RAD_I32, { 1, 1 }), cnt = make(RAD_F32, { 1 });
    seti(cu, 0, c.s); seti(cu, 1, c.e);
    for (int64_t i = 0; i < n; ++i) seti(pos, i, c.first_pos + (int32_t)i);
    seti(bounds, 0, c.s); seti(bounds, 1, (int32_t)c.b); seti(bounds, 2, (int32_t)c.b); seti(bounds, 3, c.e);
    for (int64_t i = 0; i < 8; ++i) setf(meta, i, 0.0f);
    setf(meta, 2 * 1, c.m0); setf(meta, 2 * 1 + 1, c.m1);   /* slot 1 */
    seti(idx, 0, c.slot);
    setf(cnt, 0, c.count);
    HazardOut o;
    o.rc = run_group(row, { &cu, &pos, c.b > 0 ? &bounds : nullptr, c.span > 0 ? &span : nullptr, &meta, &idx, &cnt },
                     { pint("M", 1) });
    o.m0 = getf(meta, 2); o.m1 = getf(meta, 3); o.count = getf(cnt, 0);
    return o;
}

/* DD-A -- THE BRANCH HAZARD (PLAN-FIX §5.2): a producer approximated up to position 4,095 of a 6,144
 * token request (T 2,048); a consumer resumes from its checkpoint at P = 4,096 and ends at N2 = 5,000:
 * min(N1 - N2, T - (N2 - P)) = 1,144 of its tail positions were approximated -- counted once, and the
 * slot remembers it; the consumer's next pass counts nothing again. A request's own earlier bulk lies
 * before its own tail (0). A pass with bounds records its last bulk position; without the span it
 * counts nothing; a slot outside the pool is left alone; cu_last out of order is refused. */
TEST(hazard_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_hazard", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    HazardCall c;
    c.s = 0; c.e = 904; c.first_pos = 4096; c.span = 2048; c.m0 = 4096;   /* final chunk, n_ahead 0 */
    HazardOut o = run_hazard(row, c);
    CHECK_EQ(o.rc, RAD_OK);
    CHECK_EQ(o.count, 1144.0f);
    CHECK_EQ(o.m1, 4096.0f);
    c.m1 = o.m1; c.count = o.count; c.first_pos = 4096 + 904;            /* a later pass of the same request */
    CHECK_EQ(run_hazard(row, c).count, 1144.0f);
    HazardCall own;                                                       /* no branch: own bulk ends before own tail */
    own.e = 2048; own.first_pos = 4096; own.span = 2048; own.m0 = 4096;
    CHECK_EQ(run_hazard(row, own).count, 0.0f);
    HazardCall rec;                                                       /* an approximate pass, whole bulk */
    rec.e = 2048; rec.first_pos = 2048; rec.b = 2048;
    o = run_hazard(row, rec);
    CHECK(o.count == 0.0f && o.m0 == 4096.0f);
    rec.b = 0;                                                            /* bounds empty-window: no record */
    CHECK_EQ(run_hazard(row, rec).m0, 0.0f);
    HazardCall none = c;
    none.span = 0; none.count = 0; none.m1 = 0;
    CHECK_EQ(run_hazard(row, none).count, 0.0f);
    HazardCall out = c;
    out.slot = 9; out.count = 0; out.m1 = 0;
    o = run_hazard(row, out);
    CHECK(o.count == 0.0f && o.m1 == 0.0f);
    HazardCall bad = c;
    bad.s = 10; bad.e = 5;
    CHECK_EQ(run_hazard(row, bad).rc, RAD_E_INVAL);
}

/* transcribed from kva.h's kva_row_hash, to pin the random rule's key (seed, absolute position). */
static uint32_t test_mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static uint32_t test_row_hash(long long seed, uint32_t position) {
    const unsigned long long s = (unsigned long long)seed;
    const uint32_t mixed = test_mix32((uint32_t)s ^ test_mix32((uint32_t)(s >> 32)));
    return test_mix32(mixed ^ (position * 0x9E3779B9U + 0x7F4A7C15U));
}

/* Random mode over W = [100, 900) of a 1100-row step: exactly k = rint(share x matches in W) rows
 * kept, and they are the k smallest (hash(seed, position), row) of W; same seed and positions ->
 * same rows; another seed or other positions -> other rows; [3, b] positions read like [b]; class
 * mode does not read positions. */
TEST(mask_random_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_mask", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    std::vector<float> table(64, -INFINITY);
    for (int i = 0; i < 64; i += 2) table[(size_t)i] = (float)(i % 7);
    MaskCall c;
    for (int i = 0; i < 900; ++i) c.ids.push_back((i * 37) % 64);   /* half the rows match */
    c.n = 1100; c.s = 100; c.e = 1000; c.table = &table; c.mode = "random"; c.seed = 3;
    c.first_pos = 3000;
    const MaskOut a = run_mask(row, c);
    CHECK_EQ(a.rc, RAD_OK);
    CHECK(mask_shaped(a, 100, 900, 1000));
    std::vector<std::pair<uint32_t, int32_t>> keys;
    for (int32_t i = 100; i < 900; ++i) keys.push_back({ test_row_hash(3, (uint32_t)(3000 + i)), i });
    std::sort(keys.begin(), keys.end());
    std::vector<int32_t> want;
    for (size_t j = 0; j < 100; ++j) want.push_back(keys[j].second);   /* 400 matches x 0.25 */
    std::sort(want.begin(), want.end());
    CHECK(zero_rows(a, 100, 900) == want);
    CHECK(run_mask(row, c).mask == a.mask);
    MaskCall seed = c, moved = c, planes = c, cls = c, cls_moved = c;
    seed.seed = 4;
    moved.first_pos = 5000;
    planes.components = 3;
    cls.mode = cls_moved.mode = "class";
    cls_moved.first_pos = 5000;
    CHECK(zero_rows(run_mask(row, seed), 100, 900) != want);
    CHECK(zero_rows(run_mask(row, moved), 100, 900) != want);
    CHECK(run_mask(row, planes).mask == a.mask);
    CHECK(run_mask(row, cls).mask == run_mask(row, cls_moved).mask);
}

/* cu_last out of order is refused by the host row (the device row's fail-safe is a gpu case). */
TEST(mask_refuses_bad_window, "host") {
    const RadKernelInfo* row = find_row("kva_mask", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    MaskCall c;
    c.ids.assign(8, 0);
    c.n = 10; c.mode = "none";
    struct Cu { int32_t s, e; int rc; } cus[] = {
        { -1, 5, RAD_E_INVAL }, { 6, 5, RAD_E_INVAL }, { 2, 11, RAD_E_INVAL },
        { 10, 10, RAD_OK }, { 0, 0, RAD_OK }, { 0, 10, RAD_OK } };
    for (const Cu& cu : cus) {
        c.s = cu.s; c.e = cu.e;
        CHECK_EQ(run_mask(row, c).rc, cu.rc);
    }
}

/* Refusals by name; the parse is shared with the device row. */
TEST(refuses_bad_operands, "both") {
    if (!group_runnable()) return;
    const RadKernelInfo* mk = find_row("kva_mask", group_domain());
    const RadKernelInfo* rh = find_row("kva_rho_update", group_domain());
    const RadKernelInfo* sc = find_row("kva_state_correct", group_domain());
    REQUIRE(mk && rh && sc);
    const std::vector<float> table = { 1.0f, 2.0f };
    MaskCall mc;
    mc.ids = { 0, 1, 0, 1 };
    mc.n = 4; mc.s = 0; mc.e = 4; mc.table = &table;
    MaskCall bad_mode = mc, bad_share = mc, no_score = mc, no_score_rand = mc, no_score_none = mc,
             no_score_all = mc;
    bad_mode.mode = "classy";
    bad_share.share = 1.5;
    no_score.table = no_score_rand.table = no_score_none.table = no_score_all.table = nullptr;
    no_score_rand.mode = "random";
    no_score_none.mode = "none";
    no_score_all.mode = "all";
    CHECK_EQ(run_mask(mk, bad_mode).rc, RAD_E_INVAL);
    CHECK_EQ(run_mask(mk, bad_share).rc, RAD_E_INVAL);
    CHECK_EQ(run_mask(mk, no_score).rc, RAD_E_INVAL);       /* class needs the table */
    CHECK_EQ(run_mask(mk, no_score_rand).rc, RAD_E_INVAL);  /* random counts matches */
    CHECK_EQ(run_mask(mk, no_score_none).rc, RAD_OK);
    CHECK_EQ(run_mask(mk, no_score_all).rc, RAD_OK);
    Buf cu = make(RAD_I32, { 2 }), cu1 = make(RAD_I32, { 1 }), cu_f = make(RAD_F32, { 2 });
    Buf tok = make(RAD_I32, { 4 }), pos = make(RAD_I32, { 4 }), short_pos = make(RAD_I32, { 3 });
    Buf pos16 = make(RAD_BF16, { 4 }), score = make(RAD_F32, { 2 }), score16 = make(RAD_BF16, { 2 });
    Buf mask = make(RAD_I32, { 4 }), bounds = make(RAD_I32, { 4 }), bounds3 = make(RAD_I32, { 3 });
    seti(cu, 1, 4);
    const std::vector<RadParam> p = { pint("M", 4), pf64("share", 0.5), pint("seed", 0),
                                      pstr("mode", "class") };
    std::vector<RadParam> no_seed = p;
    no_seed.erase(no_seed.begin() + 2);
    CHECK_EQ(run_group(mk, { &cu, &tok, &pos, &score, &mask, &bounds }, p), RAD_OK);
    CHECK_EQ(run_group(mk, { &cu, &tok, &pos, &score, &mask, &bounds }, no_seed), RAD_E_INVAL);
    CHECK_EQ(run_group(mk, { nullptr, &tok, &pos, &score, &mask, &bounds }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(mk, { &cu, nullptr, &pos, &score, &mask, &bounds }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(mk, { &cu, &tok, nullptr, &score, &mask, &bounds }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(mk, { &cu, &tok, &pos, &score, nullptr, &bounds }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(mk, { &cu, &tok, &pos, &score, &mask, nullptr }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(mk, { &cu_f, &tok, &pos, &score, &mask, &bounds }, p), RAD_E_DTYPE);
    CHECK_EQ(run_group(mk, { &cu, &tok, &pos16, &score, &mask, &bounds }, p), RAD_E_DTYPE);
    CHECK_EQ(run_group(mk, { &cu, &tok, &pos, &score16, &mask, &bounds }, p), RAD_E_DTYPE);
    CHECK_EQ(run_group(mk, { &cu1, &tok, &pos, &score, &mask, &bounds }, p), RAD_E_SHAPE);
    CHECK_EQ(run_group(mk, { &cu, &tok, &short_pos, &score, &mask, &bounds }, p), RAD_E_SHAPE);
    CHECK_EQ(run_group(mk, { &cu, &tok, &pos, &score, &mask, &bounds3 }, p), RAD_E_SHAPE);
    CHECK_EQ(run_group(rh, {}, { pint("M", 8), pint("n_head", 4) }), RAD_E_INVAL);
    CorrectRun k = correct_operands(3, 2, 4, 4, { 0 }, 0);
    CHECK_EQ(run_correct(sc, k, "redo", 1.0, false), RAD_E_INVAL);
    const std::vector<RadParam> cp = { pint("M", 1), pstr("mode", "apply"), pf64("alpha", 1.0),
                                       pint("n_head", 2), pint("sd0", 4), pint("sd1", 4) };
    std::vector<RadParam> bad_heads = cp;
    bad_heads[3] = pint("n_head", 3);
    Buf two_rows = slot_index({ 0, 1 });
    CHECK_EQ(run_group(sc, { &k.state, &k.sidx, &k.applied, &k.aidx, &k.c, nullptr, nullptr },
                       bad_heads), RAD_E_SHAPE);
    CHECK_EQ(run_group(sc, { &k.state, &k.sidx, &k.applied, &two_rows, &k.c, nullptr, nullptr },
                       cp), RAD_E_SHAPE);                   /* applied_idx names 2 sequences, state_idx 1 */
    CHECK_EQ(run_group(sc, { &k.state, &k.sidx, &k.applied, &k.aidx, &k.c, &k.nd, nullptr }, cp),
             RAD_E_INVAL);                                  /* ND without its index */
    CHECK_EQ(run_group(sc, { &k.state, nullptr, &k.applied, &k.aidx, &k.c, nullptr, nullptr }, cp),
             RAD_E_INVAL);
    const RadKernelInfo* sr = find_row("kva_state_read", group_domain());
    REQUIRE(sr != nullptr);
    const std::vector<RadParam> rp = { pint("M", 1), pint("n_head", 2), pint("sd0", 4), pint("sd1", 4) };
    std::vector<RadParam> rp_heads = rp;
    rp_heads[1] = pint("n_head", 3);
    Buf small_out = make(RAD_F32, { 31 }), out = make(RAD_F32, { 1, 2, 4, 4 });
    Buf out16 = make(RAD_BF16, { 1, 2, 4, 4 });
    CHECK_EQ(run_group(sr, { &k.state, &k.sidx, &small_out }, rp), RAD_E_SHAPE);   /* 31 < 32 */
    CHECK_EQ(run_group(sr, { &k.state, &k.sidx, &out16 }, rp), RAD_E_DTYPE);
    CHECK_EQ(run_group(sr, { &k.state, nullptr, &out }, rp), RAD_E_INVAL);
    CHECK_EQ(run_group(sr, { &k.state, &k.sidx, &out }, rp_heads), RAD_E_SHAPE);
    /* The optional bounds operand of kva_rho_update and kva_state_correct. */
    Buf bounds1 = make(RAD_I32, { 1 }), bounds_f32 = make(RAD_F32, { 4 });
    Buf bounds_gap = make(RAD_I32, { 2 }, { 2 });
    const std::vector<Buf*> sc_base = { &k.state, &k.sidx, &k.applied, &k.aidx, &k.c, nullptr, nullptr };
    std::vector<Buf*> sc_opds = sc_base;
    sc_opds.push_back(&bounds1);
    CHECK_EQ(run_group(sc, sc_opds, cp), RAD_E_SHAPE);      /* needs bounds[1] too */
    sc_opds.back() = &bounds_f32;
    CHECK_EQ(run_group(sc, sc_opds, cp), RAD_E_DTYPE);
    sc_opds.back() = &bounds_gap;
    CHECK_EQ(run_group(sc, sc_opds, cp), RAD_E_STRIDE);
    Rng rr{ 5 };
    RhoRun rho = random_rho(rr, 8, 2, 0.5, false);
    CHECK_EQ(run_rho(rh, rho, &bounds_f32), RAD_E_DTYPE);
    CHECK_EQ(run_rho(rh, rho, &bounds_gap), RAD_E_STRIDE);
    CHECK_EQ(run_rho(rh, rho, &bounds1), RAD_OK);           /* one element is all it reads */
}

/* A straight copy of each sequence's slot through the padded strides; a negative slot and one past
 * the pool read zeros; two sequences may read one slot; the state is left alone. */
TEST(state_read_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_state_read", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    CorrectRun k = correct_operands(4, 2, 2, 3, { 0 }, 3);
    for (int64_t s = 0; s < 4; ++s) for (int64_t h = 0; h < 2; ++h) for (int64_t e = 0; e < 6; ++e)
        setf(k.state, st_at(k, s, h, e / 3, e % 3), (float)(100 * s + 10 * h + e) + 0.5f);
    const CorrectRun before = k;
    Buf index = slot_index({ 2, -1, 9, 2 }), out;
    CHECK_EQ(run_state_read(row, k, index, out), RAD_OK);
    for (int64_t s = 0; s < 4; ++s) for (int64_t h = 0; h < 2; ++h) for (int64_t e = 0; e < 6; ++e)
        CHECK(getf(out, (s * 2 + h) * 6 + e) ==
              (s == 1 || s == 2 ? 0.0f : (float)(200 + 10 * h + e) + 0.5f));
    CHECK(k.state.bytes == before.state.bytes);
}

/* Each pool through its own index: state slot 3, applied slot 1, ND slot 5 -- pools of different
 * depths -- and a second sequence whose applied slot is outside its pool, which is skipped. A row
 * that addressed any pool through another pool's index changes a slot this case checks. */
TEST(state_correct_separate_slots, "host") {
    const RadKernelInfo* row = find_row("kva_state_correct", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    CorrectRun k = correct_operands(6, 1, 2, 2, { 3, 4 }, 1);
    k.applied = make(RAD_F32, { 4, 1, 1, 1 });
    k.nd = make(RAD_F32, { 7, 1, 1, 2 });
    k.aidx = slot_index({ 1, 4 });
    k.nidx = slot_index({ 5, 0 });
    for (int64_t s = 0; s < 6; ++s) for (int64_t e = 0; e < 4; ++e)
        setf(k.state, st_at(k, s, 0, e / 2, e % 2), (float)(10 * s + e));
    for (int64_t e = 0; e < 4; ++e) setf(k.c, e, (float)(0.5 + 0.25 * e));
    for (int64_t s = 0; s < 4; ++s) setf(k.applied, s, 7.0f);
    for (int64_t s = 0; s < 7; ++s) { setf(k.nd, 2 * s, 4.0f); setf(k.nd, 2 * s + 1, 4.0f); }   /* rho 1 */
    setf(k.nd, 10, 1.0f);   /* ND slot 5: N 1, D 4 -> rho 0.25 */
    const CorrectRun before = k;
    CHECK_EQ(run_correct(row, k, "apply", 0.5, true), RAD_OK);   /* scale 0.5 x 0.25 */
    for (int64_t s = 0; s < 6; ++s) for (int64_t e = 0; e < 4; ++e)
        CHECK(getf(k.state, st_at(k, s, 0, e / 2, e % 2)) ==
              (float)(10 * s + e + (s == 3 ? 0.125 * (0.5 + 0.25 * e) : 0.0)));
    CHECK(getf(k.applied, 1) == 0.125f);
    CHECK(getf(k.applied, 0) == 7.0f && getf(k.applied, 2) == 7.0f && getf(k.applied, 3) == 7.0f);
    CHECK(k.nd.bytes == before.nd.bytes);
    CHECK_EQ(run_correct(row, k, "undo", 1.0, false), RAD_OK);   /* through applied slot 1 */
    CHECK(k.state.bytes == before.state.bytes);                  /* exact: short binary fractions */
    CHECK(getf(k.applied, 1) == 0.0f && getf(k.applied, 3) == 7.0f);
}

TEST(state_correct_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_state_correct", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    /* Sequences on slots 1, none, 0; slot 2 is nobody's. Values are short binary fractions, so
     * every expected value below is exact and computed in double. */
    CorrectRun k = correct_operands(3, 2, 2, 3, { 1, -1, 0 }, 3);
    for (int64_t s = 0; s < 3; ++s) for (int64_t h = 0; h < 2; ++h)
        for (int64_t i = 0; i < 2; ++i) for (int64_t j = 0; j < 3; ++j)
            setf(k.state, st_at(k, s, h, i, j), (float)(1 + 100 * s + 10 * h + 3 * i + j));
    for (int64_t h = 0; h < 2; ++h) for (int64_t e = 0; e < 6; ++e)
        setf(k.c, h * 6 + e, (float)(0.5 * (h + 1) + 0.125 * e));
    const float applied0[6] = { 0.5f, 0.0f, 0.25f, 2.0f, 1.0f, 1.0f };   /* [slot][head] */
    for (int e = 0; e < 6; ++e) setf(k.applied, e, applied0[e]);
    const CorrectRun before = k;
    CHECK_EQ(run_correct(row, k, "undo", 9.0, false), RAD_OK);
    for (int64_t s = 0; s < 3; ++s) for (int64_t h = 0; h < 2; ++h) {
        const double sc = s == 2 ? 0.0 : applied0[s * 2 + h];
        for (int64_t i = 0; i < 2; ++i) for (int64_t j = 0; j < 3; ++j)
            CHECK(getf(k.state, st_at(k, s, h, i, j)) ==
                  (float)(getf(before.state, st_at(k, s, h, i, j)) - sc * getf(k.c, h * 6 + i * 3 + j)));
        CHECK(getf(k.applied, s * 2 + h) == (s == 2 ? applied0[s * 2 + h] : 0.0f));
    }
    /* apply, alpha 0.5, with ND: slot 0 rho 0.5 / clamp to 1; slot 1 D = 0 -> 1 / clamp to 0. */
    const float nd[12] = { 1, 2, 3, 2,  0, 0, -1, 4,  7, 7, 7, 7 };
    for (int e = 0; e < 12; ++e) setf(k.nd, e, nd[e]);
    const CorrectRun undone = k;
    CHECK_EQ(run_correct(row, k, "apply", 0.5, true), RAD_OK);
    const double scale[3][2] = { { 0.25, 0.5 }, { 0.5, 0.0 }, { 0.0, 0.0 } };
    for (int64_t s = 0; s < 3; ++s) for (int64_t h = 0; h < 2; ++h) {
        for (int64_t i = 0; i < 2; ++i) for (int64_t j = 0; j < 3; ++j)
            CHECK(getf(k.state, st_at(k, s, h, i, j)) ==
                  (float)(getf(undone.state, st_at(k, s, h, i, j)) + scale[s][h] * getf(k.c, h * 6 + i * 3 + j)));
        CHECK(getf(k.applied, s * 2 + h) == (s == 2 ? applied0[s * 2 + h] : (float)scale[s][h]));
    }
    CHECK(k.state.bytes != undone.state.bytes);
    /* alpha 0: every state bit kept, -0.0 included; applied cleared. */
    CorrectRun z = undone;
    setf(z.state, st_at(z, 0, 0, 0, 0), -0.0f);
    const CorrectRun z0 = z;
    CHECK_EQ(run_correct(row, z, "apply", 0.0, false), RAD_OK);
    CHECK(z.state.bytes == z0.state.bytes);
    CHECK(getf(z.applied, 0) == 0.0f && getf(z.applied, 3) == 0.0f);
}

TEST(rho_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_rho_update", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    Rng r{ 11 };
    /* No decay (exp(-1000) is 0, so e = 1): D counts rows, N counts approximated rows. */
    RhoRun flat = random_rho(r, 5, 2, 0.0, false);
    flat.mask = { 1, 0, 1, 1, 0 };
    flat.a_log = { -1000.0f, -1000.0f };
    flat.nd[(size_t)(2 * 2 * 2)] = 2.0f; flat.nd[(size_t)(2 * 2 * 2 + 1)] = 3.0f;   /* slot 2, head 0 */
    CHECK_EQ(run_rho(row, flat), RAD_OK);
    CHECK(flat.nd[8] == 5.0f && flat.nd[9] == 8.0f);      /* head 0: N 2+3, D 3+5 */
    CHECK(flat.nd[10] == 3.0f && flat.nd[11] == 5.0f);    /* head 1: from zero */
    /* Total forgetting (exp(100) overflows, e = 0): only the last row is left. */
    RhoRun sharp = random_rho(r, 6, 1, 0.0, false);
    sharp.mask = { 1, 1, 1, 1, 1, 0 };
    sharp.a_log = { 100.0f };
    CHECK_EQ(run_rho(row, sharp), RAD_OK);
    CHECK(nd_rho(sharp, 0) == 0.0f);                       /* last row exact: rho 0, not a bug */
    /* No exact row: N == D to the bit, so rho is exactly 1, across two chunks. */
    RhoRun ones = random_rho(r, 300, 8, 0.0, false);
    CHECK_EQ(run_rho(row, ones), RAD_OK);
    CHECK_EQ(run_rho(row, ones), RAD_OK);
    for (int64_t h = 0; h < 8; ++h) CHECK(nd_rho(ones, h) == 1.0f);
    /* A slot outside the pool writes nothing. */
    RhoRun away = random_rho(r, 10, 2, 0.5, false);
    away.slot = -1;
    const std::vector<float> nd0 = away.nd;
    CHECK_EQ(run_rho(row, away), RAD_OK);
    CHECK(away.nd == nd0);
    /* The same rows as a column slice of a wider buffer, interleaved with other columns, and as
     * f32 instead of bf16, give the same bits. */
    RhoRun dense = random_rho(r, 64, 6, 0.2, true), sliced = dense, mixed = dense, wide = dense;
    sliced.pitch = 12;
    mixed.pitch = 12;
    mixed.col = 2;
    wide.bf16 = false;
    CHECK_EQ(run_rho(row, dense), RAD_OK);
    CHECK_EQ(run_rho(row, sliced), RAD_OK);
    CHECK_EQ(run_rho(row, mixed), RAD_OK);
    CHECK_EQ(run_rho(row, wide), RAD_OK);
    CHECK(sliced.nd == dense.nd && mixed.nd == dense.nd && wide.nd == dense.nd);
}

/* N <= D always (each step is monotone in its inputs and mask <= 1), so rho in [0, 1] before any
 * clamp. Checked raw over random gates and masks. */
TEST(rho_in_unit_interval, "host") {
    const RadKernelInfo* row = find_row("kva_rho_update", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    Rng r{ 21 };
    for (int trial = 0; trial < 20; ++trial) {
        RhoRun c = random_rho(r, 1 + r.below(700), 8, r.uniform(), false);
        for (int chunk = 0; chunk < 3; ++chunk) CHECK_EQ(run_rho(row, c), RAD_OK);
        for (int64_t h = 0; h < 8; ++h) {
            const float n = c.nd[(size_t)((2 * 8 + h) * 2)], d = c.nd[(size_t)((2 * 8 + h) * 2 + 1)];
            CHECK(n >= 0.0f && n <= d && d >= 1.0f);
        }
    }
}

/* kva_select copies exactly the mask==1 rows of a bf16 x, int8 q and f32 s pair -- padded row
 * pitches that differ between source and destination, destinations with more rows than n, mask
 * values other than 0 and 1 -- and no other byte; the sources are left alone; absent q and s
 * pairs leave x's copy unchanged. */
TEST(select_copies_masked_rows, "host") {
    const RadKernelInfo* row = find_row("kva_select", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    const std::vector<int32_t> mask = { 1, 0, 1, 1, 2, 0, -1, 1, 0 };   /* only 1 copies */
    const int64_t n = (int64_t)mask.size();
    SelectRun k{ mask_buf(mask),
                 make(RAD_BF16, { n, 5 }, { 6, 1 }), make(RAD_I8, { n + 1, 7 }),
                 make(RAD_F32, { n, 2 }, { 3, 1 }),
                 make(RAD_BF16, { n + 2, 5 }, { 8, 1 }), make(RAD_I8, { n, 7 }, { 9, 1 }),
                 make(RAD_F32, { n + 3, 2 }) };
    fill_pattern(k.xs, 1); fill_pattern(k.qs, 2); fill_pattern(k.ss, 3);
    fill_sentinel(k.x); fill_sentinel(k.q); fill_sentinel(k.s);
    const SelectRun before = k;
    CHECK_EQ(run_select(row, k), RAD_OK);
    CHECK(selected_exactly(k.xs, before.x, k.x, mask));
    CHECK(selected_exactly(k.qs, before.q, k.q, mask));
    CHECK(selected_exactly(k.ss, before.s, k.s, mask));
    CHECK(k.x.bytes != before.x.bytes);
    CHECK(k.xs.bytes == before.xs.bytes && k.qs.bytes == before.qs.bytes && k.ss.bytes == before.ss.bytes);
    SelectRun x_only = before;
    x_only.qs = x_only.ss = x_only.q = x_only.s = Buf{};
    CHECK_EQ(run_select(row, x_only), RAD_OK);
    CHECK(x_only.x.bytes == k.x.bytes);
}

/* kva_drop_rows writes -1 on exactly the top_k first columns of the mask==1 rows (a padded pitch,
 * columns past top_k, a row past n, mask values other than 0 and 1 all untouched); top_k 0 writes
 * nothing; top_k equal to the width is accepted. */
TEST(drop_rows_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_drop_rows", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    const std::vector<int32_t> mask = { 1, 0, 1, 2, -1, 1 };
    Buf m = mask_buf(mask), ids = make(RAD_I32, { 7, 8 }, { 10, 1 });
    for (int64_t i = 0; i < (int64_t)ids.bytes.size() / 4; ++i) seti(ids, i, (int32_t)(100 + i));
    const Buf before = ids;
    CHECK_EQ(run_drop(row, m, ids, 5), RAD_OK);
    CHECK(dropped_exactly(before, ids, mask, 5));
    CHECK(ids.bytes != before.bytes);
    Buf none = before, full = before;
    CHECK_EQ(run_drop(row, m, none, 0), RAD_OK);
    CHECK(none.bytes == before.bytes);
    CHECK_EQ(run_drop(row, m, full, 8), RAD_OK);
    CHECK(dropped_exactly(before, full, mask, 8));
}

/* Refusals of the two row-copy ops, by name; the parse is shared with the device rows. */
TEST(select_drop_refuse_bad_operands, "both") {
    if (!group_runnable()) return;
    const RadKernelInfo* sl = find_row("kva_select", group_domain());
    const RadKernelInfo* dr = find_row("kva_drop_rows", group_domain());
    REQUIRE(sl && dr);
    Buf m = mask_buf({ 1, 0, 1 }), m_f32 = make(RAD_F32, { 3 });
    Buf x = make(RAD_BF16, { 3, 4 }), x_f32 = make(RAD_F32, { 3, 4 }), x_wide = make(RAD_BF16, { 3, 5 });
    Buf x_short = make(RAD_BF16, { 2, 4 }), x_gap = make(RAD_BF16, { 3, 4 }, { 8, 2 });
    Buf x_rank3 = make(RAD_BF16, { 3, 4, 1 }), x_i4 = make(RAD_I4, { 3, 4 }), q = make(RAD_I8, { 3, 6 });
    const std::vector<RadParam> p = { pint("M", 3) };
    CHECK_EQ(run_group(sl, { &m, &x, nullptr, nullptr, &x, nullptr, nullptr }, p), RAD_OK);
    CHECK_EQ(run_group(sl, { nullptr, &x, nullptr, nullptr, &x, nullptr, nullptr }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(sl, { &m, nullptr, nullptr, nullptr, &x, nullptr, nullptr }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(sl, { &m, &x, nullptr, nullptr, nullptr, nullptr, nullptr }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(sl, { &m, &x, &q, nullptr, &x, nullptr, nullptr }, p), RAD_E_INVAL);   /* half a pair */
    CHECK_EQ(run_group(sl, { &m, &x, nullptr, nullptr, &x, &q, nullptr }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(sl, { &m_f32, &x, nullptr, nullptr, &x, nullptr, nullptr }, p), RAD_E_DTYPE);
    CHECK_EQ(run_group(sl, { &m, &x, nullptr, nullptr, &x_f32, nullptr, nullptr }, p), RAD_E_DTYPE);
    CHECK_EQ(run_group(sl, { &m, &x_i4, nullptr, nullptr, &x_i4, nullptr, nullptr }, p), RAD_E_DTYPE);
    CHECK_EQ(run_group(sl, { &m, &x, nullptr, nullptr, &x_wide, nullptr, nullptr }, p), RAD_E_SHAPE);
    CHECK_EQ(run_group(sl, { &m, &x, nullptr, nullptr, &x_short, nullptr, nullptr }, p), RAD_E_SHAPE);
    CHECK_EQ(run_group(sl, { &m, &x_short, nullptr, nullptr, &x, nullptr, nullptr }, p), RAD_E_SHAPE);
    CHECK_EQ(run_group(sl, { &m, &x_rank3, nullptr, nullptr, &x_rank3, nullptr, nullptr }, p), RAD_E_SHAPE);
    CHECK_EQ(run_group(sl, { &m, &x, nullptr, nullptr, &x_gap, nullptr, nullptr }, p), RAD_E_STRIDE);
    Buf ids = make(RAD_I32, { 3, 8 }), ids_short = make(RAD_I32, { 2, 8 }), ids_f32 = make(RAD_F32, { 3, 8 });
    Buf ids_gap = make(RAD_I32, { 3, 8 }, { 16, 2 }), ids_flat = make(RAD_I32, { 24 });
    CHECK_EQ(run_drop(dr, m, ids, 8), RAD_OK);
    CHECK_EQ(run_drop(dr, m, ids, 9), RAD_E_SHAPE);          /* top_k past the width */
    CHECK_EQ(run_drop(dr, m, ids, -1), RAD_E_INVAL);
    CHECK_EQ(run_group(dr, { &m, &ids }, { pint("M", 3) }), RAD_E_INVAL);   /* no top_k */
    CHECK_EQ(run_group(dr, { nullptr, &ids }, { pint("M", 3), pint("top_k", 2) }), RAD_E_INVAL);
    CHECK_EQ(run_drop(dr, m, ids_short, 2), RAD_E_SHAPE);
    CHECK_EQ(run_drop(dr, m, ids_flat, 2), RAD_E_SHAPE);
    CHECK_EQ(run_drop(dr, m, ids_f32, 2), RAD_E_DTYPE);
    CHECK_EQ(run_drop(dr, m_f32, ids, 2), RAD_E_DTYPE);
    CHECK_EQ(run_drop(dr, m, ids_gap, 2), RAD_E_STRIDE);
}

/* kva_rho_update with bounds {s, ..} equals the op without bounds on rows [s, n) alone, bit for
 * bit, with N and D carried in from an earlier chunk; bounds[0] clamps into [0, n] (below 0: every
 * row; at or past n: N and D written back untouched). */
TEST(rho_bounds_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_rho_update", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    Rng r{ 71 };
    RhoRun base = random_rho(r, 40, 3, 0.3, true);
    for (size_t i = 0; i < base.nd.size(); ++i) base.nd[i] = 0.25f + 0.5f * (float)(i % 5);
    for (int32_t s : { 0, 1, 17, 39 }) {
        RhoRun bounded = base, slice = rho_slice(base, s);
        bounded.bounds = { s, 33, 33, 40 };
        CHECK_EQ(run_rho(row, bounded), RAD_OK);
        CHECK_EQ(run_rho(row, slice), RAD_OK);
        CHECK(same_bits(bounded.nd, slice.nd));
        CHECK(!same_bits(bounded.nd, base.nd));
    }
    RhoRun below = base, plain = base;
    below.bounds = { -5 };
    CHECK_EQ(run_rho(row, below), RAD_OK);
    CHECK_EQ(run_rho(row, plain), RAD_OK);
    CHECK(same_bits(below.nd, plain.nd));
    for (int32_t s : { 40, 41, 1 << 30 }) {
        RhoRun past = base;
        past.bounds = { s, 0 };
        CHECK_EQ(run_rho(row, past), RAD_OK);
        CHECK(same_bits(past.nd, base.nd));
    }
}

/* kva_state_correct with bounds: an empty bulk (bounds[1] <= bounds[0]) changes no byte of state
 * or applied in either mode; a non-empty one is exactly the op without bounds. */
TEST(state_correct_bounds, "host") {
    const RadKernelInfo* row = find_row("kva_state_correct", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    Rng r{ 81 };
    CorrectRun k = correct_operands(3, 2, 2, 3, { 1, -1, 0 }, 3);
    for (size_t i = 0; i < k.state.bytes.size() / 4; ++i) setf(k.state, (int64_t)i, r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&k.c.t); ++i) setf(k.c, i, r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&k.applied.t); ++i) setf(k.applied, i, 0.5f + r.uniform());
    for (int64_t i = 0; i < rad_tensor_numel(&k.nd.t); ++i) setf(k.nd, i, 1.0f + (float)(i % 3));
    for (const char* mode : { "undo", "apply" }) {
        const bool nd = !std::strcmp(mode, "apply");
        for (const std::vector<int32_t>& empty : { std::vector<int32_t>{ 5, 5, 5, 9 },
                                                   std::vector<int32_t>{ 7, 3, 3, 9 },
                                                   std::vector<int32_t>{ 0, 0 } }) {
            CorrectRun t = k;
            CHECK_EQ(run_correct(row, t, mode, 0.5, nd, empty), RAD_OK);
            CHECK(t.state.bytes == k.state.bytes && t.applied.bytes == k.applied.bytes);
        }
        CorrectRun with = k, without = k;
        CHECK_EQ(run_correct(row, with, mode, 0.5, nd, { 2, 6, 6, 9 }), RAD_OK);
        CHECK_EQ(run_correct(row, without, mode, 0.5, nd), RAD_OK);
        CHECK(with.state.bytes == without.state.bytes && with.applied.bytes == without.applied.bytes);
        CHECK(with.state.bytes != k.state.bytes);
    }
}

/* ================================================================== fixtures */

/* tests/rho_ref.py's output: the reference's closed-form rho (st_hook.py) over three chunks. */
struct RhoFixture { int64_t heads = 0; RhoRun base; std::vector<RhoRun> chunks;
                    std::vector<std::vector<float>> n, d, rho; };

static bool read_list(FILE* f, const char* name, std::vector<float>& v) {
    char word[64];
    long long count = 0;
    if (std::fscanf(f, "%63s %lld", word, &count) != 2 || std::strcmp(word, name)) return false;
    v.assign((size_t)count, 0.0f);
    for (float& x : v) if (std::fscanf(f, "%f", &x) != 1) return false;
    return true;
}

static bool load_rho_fixture(RhoFixture& fx) {
    const char* path = std::getenv("KVA_RHO_FIXTURE");
    FILE* f = path ? std::fopen(path, "r") : nullptr;
    if (!f) return false;
    long long heads = 0, chunks = 0, seed = 0;
    int version = 0;
    bool ok = std::fscanf(f, "kva-rho-fixture %d heads %lld chunks %lld seed %lld", &version, &heads,
                          &chunks, &seed) == 4 && version == 1;
    fx.heads = heads;
    ok = ok && read_list(f, "A_log", fx.base.a_log) && read_list(f, "dt_bias", fx.base.dt_bias);
    for (long long c = 0; ok && c < chunks; ++c) {
        long long index = 0, rows = 0;
        RhoRun run = fx.base;
        std::vector<float> mask, n, d, rho;
        ok = std::fscanf(f, " chunk %lld rows %lld", &index, &rows) == 2 &&
             read_list(f, "a", run.a) && read_list(f, "mask", mask) && read_list(f, "N", n) &&
             read_list(f, "D", d) && read_list(f, "rho", rho);
        run.n = rows; run.heads = heads;
        for (float m : mask) run.mask.push_back((int32_t)m);
        fx.chunks.push_back(run); fx.n.push_back(n); fx.d.push_back(d); fx.rho.push_back(rho);
    }
    std::fclose(f);
    return ok;
}

/* R34's reference leg: the row against the NumPy transcription, chunk by chunk, ND carried. */
TEST(rho_matches_numpy_reference, "both") {
    if (!group_runnable()) return;
    RhoFixture fx;
    if (!load_rho_fixture(fx)) { skip("KVA_RHO_FIXTURE unset or unreadable (tests/rho_ref.py writes it)"); return; }
    const RadKernelInfo* row = find_row("kva_rho_update", group_domain());
    REQUIRE(row != nullptr);
    std::vector<float> nd((size_t)(4 * fx.heads * 2), 0.0f);
    double worst_rho = 0, worst_rel = 0;
    for (size_t c = 0; c < fx.chunks.size(); ++c) {
        RhoRun run = fx.chunks[c];
        run.nd = nd;
        CHECK_EQ(run_rho(row, run), RAD_OK);
        nd = run.nd;
        for (int64_t h = 0; h < fx.heads; ++h) {
            const double n = nd[(size_t)((run.slot * fx.heads + h) * 2)];
            const double d = nd[(size_t)((run.slot * fx.heads + h) * 2 + 1)];
            worst_rel = std::max({ worst_rel, std::fabs(n - fx.n[c][(size_t)h]) / fx.n[c][(size_t)h],
                                   std::fabs(d - fx.d[c][(size_t)h]) / fx.d[c][(size_t)h] });
            worst_rho = std::max(worst_rho, std::fabs((double)nd_rho(run, h) - fx.rho[c][(size_t)h]));
        }
    }
    std::fprintf(stderr, "  rho vs NumPy: max |rho diff| %.3g, max rel N/D diff %.3g\n", worst_rho, worst_rel);
    CHECK(worst_rho <= 1e-5);
    CHECK(worst_rel <= 1e-4);
}

/* A JSON reader for the R33 fixture: objects, arrays, numbers, strings, null. */
struct Json {
    enum Kind { NUL, NUM, STR, ARR, OBJ } kind = NUL;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;
    const Json* get(const char* key) const {
        for (const auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

struct JsonReader {
    const char* p;
    bool ok = true;
    void space() { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p; }
    bool eat(char c) { space(); if (*p != c) return false; ++p; return true; }
    std::string text() {
        std::string s;
        if (!eat('"')) { ok = false; return s; }
        while (*p && *p != '"') { if (*p == '\\' && p[1]) ++p; s += *p++; }
        ok = ok && eat('"');
        return s;
    }
    Json value() {
        Json v;
        space();
        if (!ok || !*p) { ok = false; return v; }
        if (*p == '{' || *p == '[') {
            const bool object = *p++ == '{';
            v.kind = object ? Json::OBJ : Json::ARR;
            if (eat(object ? '}' : ']')) return v;
            do {
                if (object) { std::string key = text(); ok = ok && eat(':'); v.obj.push_back({ key, value() }); }
                else v.arr.push_back(value());
            } while (ok && eat(','));
            ok = ok && eat(object ? '}' : ']');
        } else if (*p == '"') {
            v.kind = Json::STR;
            v.str = text();
        } else if (!std::strncmp(p, "null", 4)) {
            p += 4;
        } else {
            char* end = nullptr;
            v.kind = Json::NUM;
            v.num = std::strtod(p, &end);
            ok = end != p;
            p = end;
        }
        return v;
    }
};

static std::vector<int32_t> ints(const Json* j) {
    std::vector<int32_t> v;
    if (j) for (const Json& x : j->arr) v.push_back((int32_t)x.num);
    return v;
}

/* R33: kva_mask in class mode over window [0, 2048) of each quick doc keeps exactly fnlev.rules'
 * rows (the SIDECAR lane's tools/rows_compare.py fixture, format kva-rowsel-fixture-1); and the
 * same 2048 ids placed at rows [64, 2112) of a 2112-row step (cu_last {64, 2112}, the rows before
 * them the table's best id) keep those rows + 64, with every row before 64 left at 0. */
TEST(mask_matches_fnlev_rules, "both") {
    if (!group_runnable()) return;
    const char* path = std::getenv("KVA_ROWSEL_FIXTURE");
    FILE* f = path ? std::fopen(path, "rb") : nullptr;
    if (!f) { skip("KVA_ROWSEL_FIXTURE unset or absent (SIDECAR lane: tools/rows_compare.py fixture)"); return; }
    std::string body;
    char chunk[65536];
    for (size_t got; (got = std::fread(chunk, 1, sizeof chunk, f)) > 0;) body.append(chunk, got);
    std::fclose(f);
    JsonReader rd{ body.c_str() };
    const Json fx = rd.value();
    REQUIRE(rd.ok && fx.get("format") && fx.get("format")->str == "kva-rowsel-fixture-1");
    const RadKernelInfo* row = find_row("kva_mask", group_domain());
    const RadKernelInfo* host = find_row("kva_mask", RAD_DOMAIN_HOST);
    REQUIRE(row && host && fx.get("vocab") && fx.get("kept_ids") && fx.get("kept_scores") &&
            fx.get("docs") && fx.get("share"));
    std::vector<float> table((size_t)fx.get("vocab")->num, -INFINITY);
    const std::vector<int32_t> kept = ints(fx.get("kept_ids"));
    int32_t best = kept.empty() ? 0 : kept[0];
    for (size_t i = 0; i < kept.size(); ++i) {
        table[(size_t)kept[i]] = (float)fx.get("kept_scores")->arr[i].num;
        if (table[(size_t)kept[i]] > table[(size_t)best]) best = kept[i];
    }
    for (const Json& doc : fx.get("docs")->arr) {
        const std::vector<int32_t> ids = ints(doc.get("token_ids")), want = ints(doc.get("rows"));
        const int32_t w = (int32_t)ids.size(), shift = 64;
        MaskCall plain;
        plain.ids = ids;
        plain.n = w; plain.s = 0; plain.e = w; plain.table = &table;
        plain.share = fx.get("share")->num; plain.mode = "class";
        MaskCall moved = plain;
        moved.ids.assign((size_t)shift, best);
        moved.ids.insert(moved.ids.end(), ids.begin(), ids.end());
        moved.n = moved.e = w + shift; moved.s = shift;
        std::vector<int32_t> want_moved;
        for (int32_t r : want) want_moved.push_back(r + shift);
        const MaskOut a = run_mask(row, plain), m = run_mask(row, moved);
        CHECK(a.rc == RAD_OK && m.rc == RAD_OK);
        CHECK(mask_shaped(a, 0, w, w) && mask_shaped(m, shift, w + shift, w + shift));
        CHECK_EQ(zero_rows(a, 0, w).size(), (size_t)doc.get("k")->num);
        if (zero_rows(a, 0, w) != want) fail_at(__FILE__, __LINE__, doc.get("doc")->str + ": rows differ from fnlev.rules");
        if (zero_rows(m, shift, w + shift) != want_moved || zero_rows(m, 0, shift).size() != (size_t)shift)
            fail_at(__FILE__, __LINE__, doc.get("doc")->str + ": shifted window differs from fnlev.rules + 64");
        if (row != host) CHECK(a.mask == run_mask(host, plain).mask && m.mask == run_mask(host, moved).mask);
        std::fprintf(stderr, "  %-10s n %d  k %zu  == fnlev.rules (window [0, %d) and [64, %d))%s\n",
                     doc.get("doc")->str.c_str(), w, want.size(), w, w + shift,
                     row != host ? " == host row" : "");
    }
}

/* ================================================================== per-rank extents (TP1 / TP2 / TP4)
 * The correction, the state read and the decay sums are per value head, and tensor parallelism hands
 * rank r the contiguous heads [r * H/W, (r + 1) * H/W) (the delta net's own split; the folder's st.L is
 * sliced the same way, kva_projector.h plan_rank). So a rank's row at its extent -- 24 heads at TP2, 12
 * at TP4 -- must compute, byte for byte, the slice of what the row computes over all 48 heads at TP1.
 * The host rows here; the device rows against them at each extent are the "gpu" cases above. */

/* Heads [h0, h0 + n) of `full` copied into `part` (a CorrectRun of n heads), every pool and C. */
static void copy_heads(const CorrectRun& full, CorrectRun& part, int64_t h0) {
    for (int64_t s = 0; s < part.slots; ++s)
        for (int64_t h = 0; h < part.heads; ++h) {
            for (int64_t i = 0; i < part.sd0; ++i)
                for (int64_t j = 0; j < part.sd1; ++j)
                    setf(part.state, st_at(part, s, h, i, j), getf(full.state, st_at(full, s, h0 + h, i, j)));
            setf(part.applied, s * part.heads + h, getf(full.applied, s * full.heads + h0 + h));
            for (int w = 0; w < 2; ++w) setf(part.nd, (s * part.heads + h) * 2 + w, getf(full.nd, (s * full.heads + h0 + h) * 2 + w));
        }
    for (int64_t h = 0; h < part.heads; ++h)
        for (int64_t e = 0; e < part.sd0 * part.sd1; ++e)
            setf(part.c, h * part.sd0 * part.sd1 + e, getf(full.c, (h0 + h) * full.sd0 * full.sd1 + e));
}

/* Element differences between heads [h0, h0 + n) of `full` and all of `part`: state, applied, ND. */
static int64_t heads_differ(const CorrectRun& full, const CorrectRun& part, int64_t h0) {
    int64_t n = 0;
    for (int64_t s = 0; s < part.slots; ++s)
        for (int64_t h = 0; h < part.heads; ++h) {
            for (int64_t i = 0; i < part.sd0; ++i)
                for (int64_t j = 0; j < part.sd1; ++j)
                    n += getf(part.state, st_at(part, s, h, i, j)) != getf(full.state, st_at(full, s, h0 + h, i, j));
            n += getf(part.applied, s * part.heads + h) != getf(full.applied, s * full.heads + h0 + h);
            for (int w = 0; w < 2; ++w)
                n += getf(part.nd, (s * part.heads + h) * 2 + w) != getf(full.nd, (s * full.heads + h0 + h) * 2 + w);
        }
    return n;
}

TEST(each_ranks_heads_compute_their_slice_of_tp1, "host") {
    const RadKernelInfo* corr = find_row("kva_state_correct", RAD_DOMAIN_HOST);
    const RadKernelInfo* read = find_row("kva_state_read", RAD_DOMAIN_HOST);
    const RadKernelInfo* rho = find_row("kva_rho_update", RAD_DOMAIN_HOST);
    REQUIRE(corr && read && rho);
    const int64_t H = 48, D = 128;   /* the model's value heads and state width */
    Rng r{ 4848 };
    CorrectRun full = correct_operands(6, H, D, D, { 4, -1, 1, 5 }, 8);
    for (size_t i = 0; i < full.state.bytes.size() / 4; ++i) setf(full.state, (int64_t)i, r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&full.c.t); ++i) setf(full.c, i, 0.01f * r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&full.applied.t); ++i) setf(full.applied, i, 0.5f + r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&full.nd.t); i += 2) {
        const float d = (float)(1.0 + 50.0 * r.uniform());
        setf(full.nd, i, (float)(d * 1.2 * r.uniform()));
        setf(full.nd, i + 1, d);
    }
    struct Step { const char* mode; double alpha; bool nd; };
    const Step steps[] = { { "undo", 1.0, false }, { "apply", 0.7, true }, { "apply", 1.0, false } };
    Buf index = slot_index({ 4, -1, 1, 5 });
    /* TP1: every step over all 48 heads, then the read. */
    CorrectRun tp1 = full;
    for (const Step& st : steps) CHECK_EQ(run_correct(corr, tp1, st.mode, st.alpha, st.nd), RAD_OK);
    Buf read1;
    CHECK_EQ(run_state_read(read, tp1, index, read1), RAD_OK);
    RhoRun rho1 = random_rho(r, 512, H, 0.1, true);
    rho1.pitch = 2 * H;
    CHECK_EQ(run_rho(rho, rho1), RAD_OK);
    for (int world : { 2, 4 }) {
        const int64_t hw = H / world;
        int64_t corr_diff = 0, read_diff = 0, rho_diff = 0;
        for (int rank = 0; rank < world; ++rank) {
            const int64_t h0 = rank * hw;
            CorrectRun part = correct_operands(6, hw, D, D, { 4, -1, 1, 5 }, 8);
            copy_heads(full, part, h0);
            for (const Step& st : steps) CHECK_EQ(run_correct(corr, part, st.mode, st.alpha, st.nd), RAD_OK);
            corr_diff += heads_differ(tp1, part, h0);
            Buf readp;
            CHECK_EQ(run_state_read(read, part, index, readp), RAD_OK);
            for (int64_t q = 0; q < 4; ++q)
                for (int64_t h = 0; h < hw; ++h)
                    for (int64_t e = 0; e < D * D; ++e)
                        read_diff += getf(readp, (q * hw + h) * D * D + e) != getf(read1, (q * H + h0 + h) * D * D + e);
            /* the decay sums: the rank's heads of a (its half of a|b), A_log, dt_bias and ND */
            RhoRun rp = rho1;
            rp.heads = hw; rp.pitch = 2 * hw;
            rp.a.clear(); rp.a_log.clear(); rp.dt_bias.clear();
            for (int64_t t = 0; t < rho1.n; ++t)
                for (int64_t h = 0; h < hw; ++h) rp.a.push_back(rho1.a[(size_t)(t * H + h0 + h)]);
            for (int64_t h = 0; h < hw; ++h) {
                rp.a_log.push_back(rho1.a_log[(size_t)(h0 + h)]);
                rp.dt_bias.push_back(rho1.dt_bias[(size_t)(h0 + h)]);
            }
            rp.nd.assign((size_t)(rp.slots * hw * 2), 0.0f);
            CHECK_EQ(run_rho(rho, rp), RAD_OK);
            for (int64_t s = 0; s < rp.slots; ++s)
                for (int64_t h = 0; h < hw; ++h)
                    for (int w = 0; w < 2; ++w)
                        rho_diff += rp.nd[(size_t)((s * hw + h) * 2 + w)] != rho1.nd[(size_t)((s * H + h0 + h) * 2 + w)];
        }
        CHECK_EQ(corr_diff, 0);
        CHECK_EQ(read_diff, 0);
        CHECK_EQ(rho_diff, 0);
        std::fprintf(stderr, "  TP%d (%lld heads a rank): correct %lld, read %lld, decay sums %lld elements differ "
                     "from TP1's slices\n", world, (long long)hw, (long long)corr_diff, (long long)read_diff,
                     (long long)rho_diff);
    }
}

/* ================================================================== device against host */

/* R20: the device row and the host row on the same random operands -- every rank count's per-rank
 * value heads (48 at TP1, 24 at TP2, 12 at TP4) at the model's 128 x 128 state, padded
 * slot / head / row strides, nonzero applied scales, a skipped sequence -- agree to the bit, over
 * the whole buffers (padding included). */
static void state_correct_device_vs_host(const RadKernelInfo* dev, const RadKernelInfo* host, int64_t heads) {
    Rng r{ (uint64_t)(31 + heads) };
    CorrectRun k = correct_operands(6, heads, 128, 128, { 4, -1, 1, 5 }, 8);
    for (size_t i = 0; i < k.state.bytes.size() / 4; ++i) setf(k.state, (int64_t)i, r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&k.c.t); ++i) setf(k.c, i, 0.01f * r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&k.applied.t); ++i) setf(k.applied, i, 0.5f + r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&k.nd.t); i += 2) {
        const float d = (float)(1.0 + 50.0 * r.uniform());
        setf(k.nd, i, (float)(d * 1.2 * r.uniform()));   /* some N > D: the clamp is exercised */
        setf(k.nd, i + 1, d);
    }
    struct Step { const char* mode; double alpha; bool nd; } steps[] = {
        { "undo", 1.0, false }, { "apply", 0.7, true }, { "undo", 1.0, false },
        { "apply", 1.0, false }, { "apply", 0.0, true } };
    /* Then the same buffers with every pool on its own index and depth: state slots 3 0 5 2,
     * applied 1 3 0 and one outside its 4-slot pool, ND 5 6 2 0 in a 7-slot pool. */
    CorrectRun sep = k;
    sep.applied = make(RAD_F32, { 4, heads, 1, 1 });
    sep.nd = make(RAD_F32, { 7, heads, 1, 2 });
    for (int64_t i = 0; i < rad_tensor_numel(&sep.applied.t); ++i) setf(sep.applied, i, 0.5f + r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&sep.nd.t); i += 2) {
        setf(sep.nd, i + 1, (float)(1.0 + 50.0 * r.uniform()));
        setf(sep.nd, i, (float)(getf(sep.nd, i + 1) * 1.2 * r.uniform()));
    }
    sep.sidx = slot_index({ 3, 0, 5, 2 });
    sep.aidx = slot_index({ 1, 3, 0, 4 });
    sep.nidx = slot_index({ 5, 6, 2, 0 });
    for (int pass = 0; pass < 2; ++pass)
        for (const Step& st : steps) {
            CorrectRun& h = pass ? sep : k;
            CorrectRun on_dev = h;
            CHECK_EQ(run_correct(host, h, st.mode, st.alpha, st.nd), RAD_OK);
            CHECK_EQ(run_correct(dev, on_dev, st.mode, st.alpha, st.nd), RAD_OK);
            size_t differ = 0;
            for (size_t i = 0; i < h.state.bytes.size(); ++i) differ += h.state.bytes[i] != on_dev.state.bytes[i];
            CHECK_EQ(differ, 0);
            CHECK(h.applied.bytes == on_dev.applied.bytes && h.nd.bytes == on_dev.nd.bytes);
            std::fprintf(stderr, "  %2lld heads %s %-5s alpha %.1f ND %d: %zu state bytes, %zu differ (max abs diff %s)\n",
                         (long long)heads, pass ? "own slots  " : "shared slots", st.mode, st.alpha, (int)st.nd,
                         h.state.bytes.size(), differ, differ ? ">0" : "0");
        }
}

/* At every rank count's per-rank extent: TP1's 48 value heads, TP2's 24, TP4's 12. */
TEST(state_correct_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_state_correct", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_state_correct", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    for (int64_t heads : { 48, 24, 12 }) state_correct_device_vs_host(dev, host, heads);
}

/* kva_state_read, device vs host: model-sized heads at padded strides, a negative slot, a slot past
 * the pool and a repeated slot; the outputs agree to the bit and the out-of-pool rows are zeros. */
static void state_read_device_vs_host(const RadKernelInfo* dev, const RadKernelInfo* host, int64_t heads) {
    Rng r{ (uint64_t)(61 + heads) };
    CorrectRun k = correct_operands(6, heads, 128, 128, { 0 }, 8);
    for (size_t i = 0; i < k.state.bytes.size() / 4; ++i) setf(k.state, (int64_t)i, r.normal());
    Buf index = slot_index({ 4, -1, 1, 7, 4 }), out_host, out_dev;
    CHECK_EQ(run_state_read(host, k, index, out_host), RAD_OK);
    CHECK_EQ(run_state_read(dev, k, index, out_dev), RAD_OK);
    size_t differ = 0, nonzero_out_of_pool = 0;
    const int64_t per_seq = heads * 128 * 128;
    for (size_t i = 0; i < out_host.bytes.size(); ++i) differ += out_host.bytes[i] != out_dev.bytes[i];
    for (int64_t s : { 1, 3 }) for (int64_t e = 0; e < per_seq; ++e)
        nonzero_out_of_pool += getf(out_dev, s * per_seq + e) != 0.0f;
    CHECK_EQ(differ, 0);
    CHECK_EQ(nonzero_out_of_pool, 0);
    std::fprintf(stderr, "  %2lld heads: %zu out bytes, %zu differ; out-of-pool rows all zero: %s\n",
                 (long long)heads, out_host.bytes.size(), differ, nonzero_out_of_pool ? "no" : "yes");
}

TEST(state_read_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_state_read", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_state_read", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    for (int64_t heads : { 48, 24, 12 }) state_read_device_vs_host(dev, host, heads);
}

/* A random window layout over an n-row step, the bulk end b kept within the device row's LDS
 * budget: whole step, a window inside, b before s (empty), b past e. */
static MaskCall random_window(Rng& r, int64_t n, int layout) {
    MaskCall c;
    const int64_t limit = 8192;   /* KVA_MASK_MAX_ROWS: b above it is a refused geometry */
    int64_t s = layout == 0 ? 0 : r.below(n / 2 + 1), e = n, b = std::min(n, limit);
    if (layout == 1) b = s + r.below(std::max<int64_t>(std::min(e, limit) - s, 0) + 1);
    if (layout >= 2) e = s + r.below(n - s + 1);
    if (layout == 2) b = s - r.below(s + 1);
    if (layout == 3) b = e + 1 + r.below(50);
    c.ids.assign((size_t)std::min(std::max<int64_t>(b, 1), limit), 0);
    for (int32_t& id : c.ids) id = (int32_t)r.below(1010) - 5;   /* a few ids outside the table */
    c.n = n; c.s = (int32_t)s; c.e = (int32_t)e;
    return c;
}

/* kva_mask's device leg: device == host on mask and bounds, byte for byte, over random windows
 * (s > 0, b before s, b past e, tails), every mode, heavy ties, ids outside the table, n up to
 * past the LDS budget (the mask is not bounded by it, only b is). */
TEST(mask_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_mask", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_mask", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 41 };
    std::vector<float> table(1000);
    const float levels[] = { -INFINITY, -INFINITY, 1.0f, 2.0f, 2.5f, 3.0f };   /* ties everywhere */
    for (float& v : table) v = levels[r.below(6)];
    int runs = 0, windows = 0, shifted = 0;
    for (int64_t n : { 1, 7, 300, 2048, 8192, 9000 })
        for (int layout = 0; layout < 4; ++layout) {
            MaskCall c = random_window(r, n, layout);
            c.table = &table; c.seed = 12345; c.first_pos = 4096;
            for (const char* mode : { "none", "class", "random", "all", "step" })
                for (double share : { 0.25, 1.0 }) {
                    c.mode = mode; c.share = share; c.components = runs % 2 ? 3 : 1;
                    c.n_zeros = runs % 3 ? 0 : 513;
                    const MaskOut h = run_mask(host, c), d = run_mask(dev, c);
                    CHECK(h.rc == RAD_OK && d.rc == RAD_OK);
                    CHECK(d.mask == h.mask && d.bounds == h.bounds && d.zeros == h.zeros);
                    ++runs;
                }
            const MaskOut h = run_mask(host, c);
            windows += h.bounds[1] > h.bounds[0];
            shifted += h.bounds[0] > 0 && h.bounds[1] > h.bounds[0];
        }
    MaskCall c = random_window(r, 4000, 1);
    c.table = &table; c.mode = "random"; c.seed = 99; c.first_pos = 2048;
    MaskCall next = c;
    next.first_pos = 6144;
    const MaskOut once = run_mask(dev, c), again = run_mask(dev, c), other = run_mask(dev, next);
    CHECK(once.mask == again.mask);                       /* deterministic: seed + positions */
    CHECK(ones(other) == ones(once) && other.mask != once.mask);
    std::fprintf(stderr, "  %d configurations (%d layouts with a window, %d with s > 0): device mask and bounds == host\n",
                 runs, windows, shifted);
}

/* kva_hazard's device leg: device == host on meta and count over branch, continuation and record
 * shapes. */
TEST(hazard_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_hazard", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_hazard", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 7 };
    int runs = 0;
    for (int i = 0; i < 64; ++i) {
        HazardCall c;
        c.e = 1 + (int32_t)r.below(2048); c.s = (int32_t)r.below((uint64_t)c.e);
        c.first_pos = (int32_t)r.below(40000); c.span = (int64_t)r.below(3000);
        c.b = r.below(2) ? c.s + (int64_t)r.below((uint64_t)(c.e - c.s + 1)) : 0;
        c.m0 = (float)r.below(45000); c.m1 = (float)r.below(40000); c.count = (float)r.below(100);
        const HazardOut h = run_hazard(host, c), d = run_hazard(dev, c);
        CHECK(h.rc == RAD_OK && d.rc == RAD_OK && h.m0 == d.m0 && h.m1 == d.m1 && h.count == d.count);
        ++runs;
    }
    std::fprintf(stderr, "  %d hazard configurations: device meta and count == host\n", runs);
}

/* What the device row does with operands the host row refuses: cu_last out of order (it cannot
 * return what it reads on the device) -> the window empty inside the clamped [s, e], every mask row
 * 0, bounds {s, s, s, e} clamped; a bulk end past the LDS budget is refused at launch. */
TEST(mask_device_fail_safe, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_mask", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_mask", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    const std::vector<float> table(8, 1.0f);
    struct Bad { int32_t s, e; std::vector<int32_t> bounds; } bads[] = {
        { -3, 10, { 0, 0, 0, 10 } }, { 12, 8, { 8, 8, 8, 8 } }, { 2, 21, { 2, 2, 2, 16 } } };
    for (const Bad& bad : bads) {
        MaskCall c;
        c.ids.assign(14, 1);
        c.n = 16; c.s = bad.s; c.e = bad.e; c.table = &table; c.mode = "none";
        const MaskOut d = run_mask(dev, c);
        CHECK_EQ(d.rc, RAD_OK);
        CHECK(d.mask == std::vector<int32_t>(16, 0) && d.bounds == bad.bounds);
        CHECK_EQ(run_mask(host, c).rc, RAD_E_INVAL);
    }
    MaskCall big;
    big.ids.assign(8193, 1);
    big.n = 8193; big.e = 8193; big.table = &table;
    CHECK_EQ(run_mask(dev, big).rc, RAD_E_SHAPE);
    CHECK_EQ(run_mask(host, big).rc, RAD_OK);
}

/* R34's device leg (at 48, 24 and 12 heads: TP1, TP2, TP4 a rank): device vs host on model-sized gates (bf16 a as a column slice of the a|b
 * buffer, carried over two chunks); rho within 1e-5, N and D reported. */
TEST(rho_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_rho_update", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_rho_update", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 51 };
    double worst_rho = 0, worst_rel = 0;
    for (int64_t heads : { 48, 24, 12 })   /* TP1, TP2, TP4's value heads a rank */
    for (double exact : { 0.056, 0.5, 0.0 }) {
        RhoRun h = random_rho(r, 2048, heads, exact, true);
        h.pitch = 2 * heads;   /* a is the first half of the rank's a|b columns */
        RhoRun d = h;
        for (int chunk = 0; chunk < 2; ++chunk) {
            CHECK_EQ(run_rho(host, h), RAD_OK);
            CHECK_EQ(run_rho(dev, d), RAD_OK);
        }
        for (int64_t k = 0; k < heads; ++k) {
            worst_rho = std::max(worst_rho, (double)std::fabs(nd_rho(h, k) - nd_rho(d, k)));
            for (int w = 0; w < 2; ++w) {
                const double a = h.nd[(size_t)((2 * heads + k) * 2 + w)], b = d.nd[(size_t)((2 * heads + k) * 2 + w)];
                worst_rel = std::max(worst_rel, std::fabs(a - b) / std::max(std::fabs(a), 1e-30));
            }
            if (exact == 0.0) CHECK(nd_rho(d, k) == 1.0f);   /* no exact row: exactly 1 */
            CHECK(nd_rho(d, k) >= 0.0f && nd_rho(d, k) <= 1.0f);
        }
    }
    std::fprintf(stderr, "  device vs host: max |rho diff| %.3g, max rel N/D diff %.3g\n", worst_rho, worst_rel);
    CHECK(worst_rho <= 1e-5);
}

/* kva_rho_update with bounds on the device: equal to the device row run on the slice [s, n) bit
 * for bit (same kernel, same arithmetic), and to the host row within the rho tolerance (expf and
 * log1pf are the device library's, not the host's, so host vs device is not bitwise for rho). */
TEST(rho_bounds_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_rho_update", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_rho_update", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 91 };
    RhoRun base = random_rho(r, 2048, 24, 0.1, true);
    base.pitch = 48;
    for (size_t i = 0; i < base.nd.size(); ++i) base.nd[i] = 0.5f + (float)(i % 7);
    double worst_rho = 0;
    int bitwise = 0;
    for (int32_t s : { -3, 0, 5, 1000, 2047, 2048, 4000 }) {
        RhoRun d = base, h = base, slice = rho_slice(base, s < 0 ? 0 : (s > 2048 ? 2048 : s));
        d.bounds = h.bounds = { s, 2048, 2048, 2048 };
        CHECK_EQ(run_rho(dev, d), RAD_OK);
        CHECK_EQ(run_rho(host, h), RAD_OK);
        CHECK_EQ(run_rho(dev, slice), RAD_OK);
        CHECK(same_bits(d.nd, slice.nd));
        bitwise += same_bits(d.nd, slice.nd);
        for (int64_t k = 0; k < 24; ++k)
            worst_rho = std::max(worst_rho, (double)std::fabs(nd_rho(h, k) - nd_rho(d, k)));
    }
    std::fprintf(stderr, "  7 bounds: device == device-on-slice bitwise %d/7; device vs host max |rho diff| %.3g\n",
                 bitwise, worst_rho);
    CHECK(worst_rho <= 1e-5);
}

/* kva_state_correct with bounds on the device: host and device agree to the byte for absent,
 * empty, inverted and non-empty bounds in both modes, and an empty bulk leaves every byte alone. */
TEST(state_correct_bounds_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_state_correct", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_state_correct", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 101 };
    CorrectRun k = correct_operands(6, 24, 128, 128, { 4, -1, 1, 5 }, 8);
    for (size_t i = 0; i < k.state.bytes.size() / 4; ++i) setf(k.state, (int64_t)i, r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&k.c.t); ++i) setf(k.c, i, 0.01f * r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&k.applied.t); ++i) setf(k.applied, i, 0.5f + r.normal());
    for (int64_t i = 0; i < rad_tensor_numel(&k.nd.t); i += 2) {
        setf(k.nd, i + 1, (float)(1.0 + 50.0 * r.uniform()));
        setf(k.nd, i, (float)(getf(k.nd, i + 1) * 1.2 * r.uniform()));
    }
    const std::vector<std::vector<int32_t>> variants = { {}, { 9, 9, 9, 12 }, { 10, 4, 4, 12 },
                                                         { 3, 9, 9, 12 } };
    int runs = 0;
    for (const char* mode : { "undo", "apply" })
        for (const std::vector<int32_t>& b : variants) {
            const bool nd = !std::strcmp(mode, "apply");
            CorrectRun h = k, d = k;
            CHECK_EQ(run_correct(host, h, mode, 0.7, nd, b), RAD_OK);
            CHECK_EQ(run_correct(dev, d, mode, 0.7, nd, b), RAD_OK);
            CHECK(h.state.bytes == d.state.bytes && h.applied.bytes == d.applied.bytes);
            const bool empty = !b.empty() && b[1] <= b[0];
            CHECK((d.state.bytes == k.state.bytes) == empty);
            CHECK((d.applied.bytes == k.applied.bytes) == empty);
            ++runs;
        }
    std::fprintf(stderr, "  %d runs (undo/apply x absent/empty/inverted/non-empty bounds): device == host bytewise\n", runs);
}

/* kva_select's device leg: device == host on every destination byte, over random masks (a few
 * non-0/1 values), padded pitches, rows past n, every copy word (16-byte bf16 rows, odd-width
 * int8 / E4M3 codes, 3-wide f32 scales) and absent q / s pairs; and exactly the mask==1 rows. */
TEST(select_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_select", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_select", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 111 };
    int runs = 0;
    size_t bytes = 0;
    for (int64_t n : { 1, 300, 8192 })
        for (int variant = 0; variant < 3; ++variant) {
            std::vector<int32_t> mask((size_t)n);
            const double exact = r.uniform();
            for (int32_t& v : mask) v = r.uniform() < 0.02 ? 2 : (r.uniform() < exact ? 0 : 1);
            const uint32_t codes = variant == 1 ? RAD_F8E4M3 : RAD_I8;
            const int64_t x_w = 64 * (1 + variant), q_w = 33 + 16 * variant;
            SelectRun k{ mask_buf(mask), make(RAD_BF16, { n, x_w }, { x_w + 8, 1 }),
                         make(codes, { n, q_w }), make(RAD_F32, { n + 1, 3 }),
                         make(RAD_BF16, { n + 1, x_w }), make(codes, { n, q_w }, { q_w + 5, 1 }),
                         make(RAD_F32, { n, 3 }, { 4, 1 }) };
            fill_pattern(k.xs, 4 + variant); fill_pattern(k.qs, 5); fill_pattern(k.ss, 6);
            fill_sentinel(k.x); fill_sentinel(k.q); fill_sentinel(k.s);
            if (variant == 2) k.qs = k.ss = k.q = k.s = Buf{};
            SelectRun h = k, d = k;
            CHECK_EQ(run_select(host, h), RAD_OK);
            CHECK_EQ(run_select(dev, d), RAD_OK);
            CHECK(d.x.bytes == h.x.bytes && d.q.bytes == h.q.bytes && d.s.bytes == h.s.bytes);
            CHECK(selected_exactly(k.xs, k.x, d.x, mask));
            if (variant != 2) CHECK(selected_exactly(k.qs, k.q, d.q, mask) && selected_exactly(k.ss, k.s, d.s, mask));
            bytes += d.x.bytes.size() + d.q.bytes.size() + d.s.bytes.size();
            ++runs;
        }
    std::fprintf(stderr, "  %d configurations, %zu destination bytes: device == host bytewise\n", runs, bytes);
}

/* kva_drop_rows' device leg: device == host on every ids element over random masks, top_k from 0
 * to past a workgroup's width, padded pitches, rows past n. */
TEST(drop_rows_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_drop_rows", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_drop_rows", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 121 };
    int runs = 0;
    for (int64_t n : { 1, 300, 8192 })
        for (int64_t top_k : { 0, 1, 7, 300 }) {
            std::vector<int32_t> mask((size_t)n);
            for (int32_t& v : mask) v = r.uniform() < 0.02 ? 2 : (r.uniform() < 0.3 ? 0 : 1);
            Buf m = mask_buf(mask), ids = make(RAD_I32, { n + 1, top_k + 4 }, { top_k + 7, 1 });
            for (int64_t i = 0; i < (int64_t)ids.bytes.size() / 4; ++i) seti(ids, i, (int32_t)r.below(1 << 20));
            Buf h = ids, d = ids;
            CHECK_EQ(run_drop(host, m, h, top_k), RAD_OK);
            CHECK_EQ(run_drop(dev, m, d, top_k), RAD_OK);
            CHECK(d.bytes == h.bytes);
            CHECK(dropped_exactly(ids, d, mask, top_k));
            ++runs;
        }
    std::fprintf(stderr, "  %d configurations: device ids == host bytewise\n", runs);
}

/* ================================================================== kva_gemm_nt_bias (R140)
 * The forwarded rows: present exactly when their source library is loaded, carrying the source
 * row's hooks, and writing the bytes gemm_nt_bias writes with the weight as a plain input. */

/* gemm_nt_bias's row in `domain` from the preloaded library named `plugin`, or null. */
static const RadKernelInfo* source_gemm(const char* plugin, int domain) {
    for (void* h : g_preloaded) {
        auto info = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
        auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        if (!info || !count || !at || std::strcmp(info()->name, plugin) != 0) continue;
        for (int i = 0; i < count(); ++i)
            if (at(i)->domain == domain && !std::strcmp(at(i)->op, "gemm_nt_bias")) return at(i);
    }
    return nullptr;
}

static const char* gemm_source() { return g_group == "gpu" ? "libr4d" : "libref"; }

TEST(gemm_forward_offers_no_row_without_its_source, "host") {
    if (!g_preloaded.empty()) { skip("source libraries preloaded; the kva_kernels_alone run checks this"); return; }
    CHECK(find_schema("kva_gemm_nt_bias") != nullptr);
    CHECK(find_row("kva_gemm_nt_bias", RAD_DOMAIN_HOST) == nullptr);
    CHECK(find_row("kva_gemm_nt_bias", RAD_DOMAIN_DEVICE) == nullptr);
    CHECK(find_schema("kva_gemm_nt_q") != nullptr);
    CHECK(find_row("kva_gemm_nt_q", RAD_DOMAIN_HOST) == nullptr);
    CHECK(find_row("kva_gemm_nt_q", RAD_DOMAIN_DEVICE) == nullptr);
}

TEST(gemm_forward_is_the_source_row, "both") {
    if (!group_runnable()) return;
    const RadKernelInfo* src = source_gemm(gemm_source(), group_domain());
    if (!src) { skip("the source library is not preloaded"); return; }
    const RadKernelInfo* row = find_row("kva_gemm_nt_bias", group_domain());
    REQUIRE(row != nullptr);
    CHECK(row->launch == src->launch && row->describe == src->describe);
    CHECK(row->init == src->init && row->fini == src->fini && row->scratch == src->scratch);
    CHECK(row->constraints == src->constraints && row->n_constraints == src->n_constraints);
    CHECK(row->opd_shape == src->opd_shape && row->priority == src->priority);
    CHECK(row->layout == nullptr && row->relayout == nullptr && row->n_tunables == 0);
}

/* A, W, bias (bf16 or f32) and res drawn at random; the forwarded row with W and bias as plain
 * tensors against the source row, every output byte. */
static void gemm_once(const RadKernelInfo* fwd, const RadKernelInfo* src, int64_t M, int64_t N,
                      int64_t K, bool bias_f32, bool res, Rng& r, size_t* bytes) {
    Buf a = make(RAD_BF16, { M, K }), w = make(RAD_BF16, { N, K });
    Buf bias = make(bias_f32 ? RAD_F32 : RAD_BF16, { N }), rs = make(RAD_BF16, { M, N });
    for (Buf* b : { &a, &w, &bias, &rs })
        for (int64_t i = 0; i < rad_tensor_numel(&b->t); ++i) setf(*b, i, r.normal());
    Buf y1 = make(RAD_BF16, { M, N }), y2 = y1;
    fill_sentinel(y1); fill_sentinel(y2);
    const std::vector<RadParam> p = { pint("M", M), pint("N", N), pint("K", K), pstr("dtype", "bf16") };
    CHECK_EQ(run_group(src, { &a, &w, &bias, res ? &rs : nullptr, &y1 }, p), RAD_OK);
    CHECK_EQ(run_group(fwd, { &a, &w, &bias, res ? &rs : nullptr, &y2 }, p), RAD_OK);
    CHECK(y1.bytes == y2.bytes);
    *bytes += y2.bytes.size();
}

TEST(gemm_forward_matches_its_source_bytewise, "both") {
    if (!group_runnable()) return;
    const RadKernelInfo* src = source_gemm(gemm_source(), group_domain());
    if (!src) { skip("the source library is not preloaded"); return; }
    const RadKernelInfo* fwd = find_row("kva_gemm_nt_bias", group_domain());
    REQUIRE(fwd != nullptr);
    /* The projector's own shape on the card (N 2560, K 10240); a small one for the naive host row. */
    const bool gpu = g_group == "gpu";
    const int64_t N = gpu ? 2560 : 48, K = gpu ? 10240 : 136;
    Rng r{ 140 };
    size_t bytes = 0;
    int runs = 0;
    for (int64_t M : { 1, 64, gpu ? 2048 : 130 })
        for (int variant = 0; variant < 2; ++variant, ++runs)
            gemm_once(fwd, src, M, N, K, variant == 1, variant == 1, r, &bytes);
    std::fprintf(stderr, "  %d runs (M 1, 64, %d; N %lld, K %lld; bf16 and f32 bias, with and without res), "
                 "%zu output bytes: forwarded == %s's gemm_nt_bias bytewise\n", runs, gpu ? 2048 : 130,
                 (long long)N, (long long)K, bytes, gemm_source());
}

/* ================================================================== kva_gemm_nt_q (R79's int8 projector)
 * libr4d's int8 gemm_nt_q device rows (dtype i8a8), forwarded with their layout hooks: every one of
 * them, and nothing else; and on the card the forwarded rows, fed planes stored by their own
 * relayout hook, compute what libref's gemm_nt_q computes from the canonical planes. */

/* The device rows of `op` in the preloaded library `plugin` whose constraints admit dtype i8a8. */
static std::vector<const RadKernelInfo*> source_i8_rows(const char* plugin) {
    std::vector<const RadKernelInfo*> out;
    for (void* h : g_preloaded) {
        auto info = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
        auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        if (!info || !count || !at || std::strcmp(info()->name, plugin) != 0) continue;
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* k = at(i);
            if (k->domain != RAD_DOMAIN_DEVICE || std::strcmp(k->op, "gemm_nt_q") || k->n_tunables) continue;
            for (int c = 0; c < k->n_constraints; ++c)
                if (k->constraints[c].op == RAD_C_IN && !std::strcmp(k->constraints[c].key, "dtype") &&
                    std::strstr(k->constraints[c].sval, "i8a8"))
                    out.push_back(k);
        }
    }
    return out;
}

static std::vector<const RadKernelInfo*> forwarded_i8_rows() {
    std::vector<const RadKernelInfo*> out;
    for (int i = 0; i < g_kernel_count(); ++i)
        if (!std::strcmp(g_kernel_at(i)->op, "kva_gemm_nt_q")) out.push_back(g_kernel_at(i));
    return out;
}

TEST(int8_forward_is_libr4ds_int8_rows_with_their_hooks, "both") {
    const std::vector<const RadKernelInfo*> src = source_i8_rows("libr4d");
    if (src.empty()) { skip("libr4d is not preloaded"); return; }
    const std::vector<const RadKernelInfo*> fwd = forwarded_i8_rows();
    REQUIRE(fwd.size() == src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        CHECK(fwd[i]->domain == RAD_DOMAIN_DEVICE);
        CHECK(fwd[i]->launch == src[i]->launch && fwd[i]->describe == src[i]->describe);
        CHECK(fwd[i]->init == src[i]->init && fwd[i]->fini == src[i]->fini && fwd[i]->scratch == src[i]->scratch);
        CHECK(fwd[i]->constraints == src[i]->constraints && fwd[i]->priority == src[i]->priority);
        CHECK(fwd[i]->layout == src[i]->layout && fwd[i]->relayout == src[i]->relayout);
        CHECK(fwd[i]->unrelayout == src[i]->unrelayout && fwd[i]->layout != nullptr);
    }
}

/* Whether `row` admits M (an LE constraint on "M"). */
static bool admits_m(const RadKernelInfo* row, int64_t M) {
    for (int c = 0; c < row->n_constraints; ++c)
        if (row->constraints[c].op == RAD_C_LE && !std::strcmp(row->constraints[c].key, "M") &&
            M > row->constraints[c].ival)
            return false;
    return true;
}

/* `plane` relaid by `row`'s hooks as operand `opd` (2 = codes, 3 = scale) of encoding i8*bf16[1x128]. */
static Buf stored(const RadKernelInfo* row, const std::vector<RadParam>& p, int opd, const Buf& plane) {
    const RadEncoding enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    const int sel = opd == 2 ? 0 : 1;
    RadTensor geom = plane.t;
    geom.data = nullptr;
    RadLayout L{};
    if (row->layout(p.data(), (int)p.size(), opd, &enc, &sel, &geom, 1, &L) != RAD_OK) return Buf{};
    Buf out = plane;
    out.bytes.assign((size_t)L.bytes, 0);
    out.t.data = out.bytes.data();
    if (row->relayout(p.data(), (int)p.size(), opd, &enc, &sel, &plane.t, 1, out.bytes.data(), L.bytes) != RAD_OK)
        return Buf{};
    return out;
}

TEST(int8_forward_on_stored_planes_matches_libref_on_canonical_ones, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* ref = nullptr;
    for (void* h : g_preloaded) {
        auto info = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
        auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        if (!info || !count || !at || std::strcmp(info()->name, "libref")) continue;
        for (int i = 0; i < count(); ++i)
            if (at(i)->domain == RAD_DOMAIN_HOST && !std::strcmp(at(i)->op, "gemm_nt_q")) ref = at(i);
    }
    const std::vector<const RadKernelInfo*> fwd = forwarded_i8_rows();
    if (!ref || fwd.empty()) { skip("libr4d and libref are not both preloaded"); return; }
    const int64_t N = 2560, K = 10240;   /* the projector's own shape */
    Rng r{ 79 };
    Buf w = make(RAD_I8, { N, K }), ws = make(RAD_BF16, { N, K / 128 });
    for (int64_t i = 0; i < N * K; ++i) w.bytes[(size_t)i] = (unsigned char)(int8_t)(r.below(255) - 127);
    for (int64_t i = 0; i < N * K / 128; ++i) setf(ws, i, (float)(1e-3 * (0.5 + r.uniform())));
    double worst = 0;
    int runs = 0;
    for (const RadKernelInfo* row : fwd)
        for (int64_t M : { 1, 64, 2048 }) {
            if (!admits_m(row, M)) continue;
            const std::vector<RadParam> p = { pint("M", M), pint("N", N), pint("K", K), pint("group", 128),
                                              pstr("dtype", "i8a8") };
            Buf a = make(RAD_I8, { M, K }), as = make(RAD_F32, { M, K / 128 });
            for (int64_t i = 0; i < M * K; ++i) a.bytes[(size_t)i] = (unsigned char)(int8_t)(r.below(255) - 127);
            for (int64_t i = 0; i < M * K / 128; ++i) setf(as, i, (float)(0.01 * (0.5 + r.uniform())));
            Buf sw = stored(row, p, 2, w), sws = stored(row, p, 3, ws);
            REQUIRE(!sw.bytes.empty() && !sws.bytes.empty());
            Buf y_ref = make(RAD_BF16, { M, N }), y_dev = y_ref;
            fill_sentinel(y_dev);
            CHECK_EQ(run_group(ref, { &a, &as, &w, &ws, &y_ref }, p), RAD_OK);
            CHECK_EQ(run_group(row, { &a, &as, &sw, &sws, &y_dev }, p), RAD_OK);
            /* int32 within a 128-K block on both sides, the 80 block sums folded in f32 in different
             * orders, one bf16 rounding of the result each: two bf16 ulps of the larger output, plus an
             * absolute 2^-9 of the output's RMS for the outputs a fold cancels to near zero (a wrong
             * layout or scale errs by the RMS itself). */
            double rms = 0;
            for (int64_t i = 0; i < M * N; ++i) rms += (double)getf(y_ref, i) * getf(y_ref, i);
            rms = std::sqrt(rms / (double)(M * N));
            double run_worst = 0;
            int64_t at = -1, over = 0;
            for (int64_t i = 0; i < M * N; ++i) {
                const double x = getf(y_ref, i), y = getf(y_dev, i);
                const double tol = std::ldexp(std::max(std::fabs(x), std::fabs(y)), -7) + std::ldexp(rms, -9);
                if (std::fabs(x - y) / tol > run_worst) { run_worst = std::fabs(x - y) / tol; at = i; }
                over += std::fabs(x - y) > tol;
            }
            std::fprintf(stderr, "    %s M %lld: worst %.3g of the bound at [%lld, %lld] (ref %.6g, device %.6g, rms %.4g), "
                         "%lld of %lld over\n", row->name, (long long)M, run_worst, (long long)(at / N), (long long)(at % N),
                         at >= 0 ? getf(y_ref, at) : 0.0, at >= 0 ? getf(y_dev, at) : 0.0, rms, (long long)over,
                         (long long)(M * N));
            CHECK_EQ(over, 0);
            worst = std::max(worst, run_worst);
            ++runs;
        }
    std::fprintf(stderr, "  %d runs (%zu forwarded int8 rows, M 1/64/2048 as each admits; N %lld, K %lld): "
                 "stored-plane device == libref canonical within %.3g of the 2-ulp bound\n", runs, fwd.size(),
                 (long long)N, (long long)K, worst);
    CHECK(runs >= 3);
}

/* ================================================================== main */

int main(int argc, char** argv) {
    if (argc < 3 || (std::strcmp(argv[2], "host") && std::strcmp(argv[2], "gpu"))) {
        std::fprintf(stderr, "usage: %s <kva.so> host|gpu [source.so ...]\n", argv[0]);
        return 2;
    }
    g_group = argv[2];
    for (int i = 3; i < argc; ++i) {
        void* s = dlopen(argv[i], RTLD_NOW | RTLD_LOCAL);
        if (!s) { std::fprintf(stderr, "cannot load %s: %s\n", argv[i], dlerror()); return 1; }
        g_preloaded.push_back(s);
    }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "cannot load %s: %s\n", argv[1], dlerror()); return 1; }
    g_kernel_count = (int (*)(void))dlsym(h, "rad_kernel_count");
    g_kernel_at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    g_schema_count = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
    g_schema_at = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
    auto abi = (uint32_t (*)(void))dlsym(h, "rad_plugin_abi_version");
    if (!g_kernel_count || !g_kernel_at || !g_schema_count || !g_schema_at || !abi) {
        std::fprintf(stderr, "%s is not a kernel plugin\n", argv[1]);
        return 1;
    }
    if (abi() != RAD_ABI_VERSION) {
        std::fprintf(stderr, "%s reports ABI %u, this test was built against %u\n", argv[1], abi(),
                     RAD_ABI_VERSION);
        return 1;
    }
    for (const Case& c : cases()) {
        if (std::strcmp(c.group, "both") && g_group != c.group) continue;
        g_case = c.name;
        g_skipped = false;
        const int before = g_failures;
        c.fn();
        if (g_failures == before && !g_skipped) std::fprintf(stderr, "  ok   %s\n", c.name);
    }
    if (g_failures) { std::fprintf(stderr, "%d failure(s)\n", g_failures); return 1; }
    if (g_checks == 0) { std::fprintf(stderr, "nothing was checked\n"); return 77; }
    std::fprintf(stderr, "%d check(s), group %s\n", g_checks, g_group.c_str());
    return 0;
}
