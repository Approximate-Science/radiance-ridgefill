/* kernel_test.cpp -- the kva kernel library, seen the way the engine sees it: dlopen'd, its rows
 * found by op and domain, called through RadArgs with the operands in schema order.
 *
 *   kernel_test <path/to/kva.so> host   cases that need no card (host rows, the oracles)
 *   kernel_test <path/to/kva.so> gpu    device rows against the host rows (needs a ROCm device)
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
    for (uint32_t i = 0; i < b.t.rank; ++i) last += (b.t.shape[i] - 1) * b.t.stride[i];
    b.bytes.assign((size_t)rad_dtype_bytes(dtype, b.t.rank ? last + 1 : 0), 0);
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
                  const std::vector<RadParam>& p, RadStream stream) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size();
    a.p = p.data(); a.n_p = (int)p.size();
    a.world_size = 1;
    return row->launch(&a, stream);
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
    if (ok) {
        rc = launch(row, t, p, (RadStream)stream);
        ok = hip_ok(hipStreamSynchronize(stream), "kernel");
    }
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

static Buf from_desc(const RadOpdDesc& d, int64_t prev_extent, Rng& r) {
    std::vector<int64_t> shape(d.shape, d.shape + d.rank);
    Buf b = make(d.dtype, shape);
    const int64_t n = rad_tensor_numel(&b.t);
    const int64_t hi = d.idx_max > 0 ? d.idx_max : prev_extent;
    std::vector<int32_t> perm;
    if (d.flags & RAD_OPD_F_IDX_UNIQUE)
        for (int64_t i = 0; i < hi; ++i) perm.push_back((int32_t)i);
    for (int64_t i = 0; i < n; ++i) {
        if (d.fill == RAD_FILL_INDEX && (d.flags & RAD_OPD_F_IDX_UNIQUE)) {
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
    if (!std::strcmp(op, "kva_rowsel"))
        return { pint("M", 64), pint("cap", 16), pf64("share", 0.25), pint("seed", 7),
                 pstr("mode", "class") };
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
        store[(size_t)i] = from_desc(d, prev, r);
        opds[(size_t)i] = &store[(size_t)i];
        prev = d.shape[0];
    }
    return true;
}

static const char* const kOps[] = { "kva_rowsel", "kva_rho_update", "kva_state_correct",
                                    "kva_state_read" };

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

/* ================================================================== op runners
 * Each builds one op's operands from plain vectors, runs the given row (host or device, by its
 * domain), and hands back the outputs. Output buffers start as a sentinel, so a row that leaves an
 * element unwritten shows as a mismatch rather than as a lucky zero. */

static const int32_t kSentinel = 0x5A5A5A5A;

struct RowselOut { int rc = 0; std::vector<int32_t> rows, mask; };

/* Positions are first_pos, first_pos + 1, ... (one sequence's chunk). components > 1 lays them out
 * component-major [components, n] like RadBatch::rope_pos, row 0 the index and the rest junk. */
static RowselOut run_rowsel(const RadKernelInfo* row, const std::vector<int32_t>& ids,
                            const std::vector<float>& table, int64_t cap, double share,
                            long long seed, const char* mode, int32_t first_pos = 0,
                            int64_t components = 1) {
    const int64_t n = (int64_t)ids.size();
    Buf tok = make(RAD_I32, { n }), score = make(RAD_F32, { (int64_t)table.size() });
    Buf rows = make(RAD_I32, { cap }), mask = make(RAD_I32, { n });
    Buf pos = components > 1 ? make(RAD_I32, { components, n }) : make(RAD_I32, { n });
    for (int64_t i = 0; i < components * n; ++i) seti(pos, i, i < n ? first_pos + (int32_t)i : 7777 + (int32_t)i);
    for (int64_t i = 0; i < n; ++i) { seti(tok, i, ids[(size_t)i]); seti(mask, i, kSentinel); }
    for (size_t i = 0; i < table.size(); ++i) setf(score, (int64_t)i, table[i]);
    for (int64_t i = 0; i < cap; ++i) seti(rows, i, kSentinel);
    RowselOut o;
    o.rc = run_group(row, { &tok, &pos, &score, &rows, &mask },
                     { pint("M", n), pint("cap", cap), pf64("share", share), pint("seed", seed),
                       pstr("mode", mode) });
    for (int64_t i = 0; i < cap; ++i) o.rows.push_back(geti(rows, i));
    for (int64_t i = 0; i < n; ++i) o.mask.push_back(geti(mask, i));
    return o;
}

/* The kept rows (rows_idx without its -1 padding). */
static std::vector<int32_t> kept_rows(const RowselOut& o) {
    std::vector<int32_t> k;
    for (int32_t r : o.rows) if (r >= 0) k.push_back(r);
    return k;
}

/* rows_idx ascending then -1 padded, and mask 0 exactly on the kept rows. */
static bool rowsel_consistent(const RowselOut& o) {
    size_t i = 0;
    for (; i < o.rows.size() && o.rows[i] >= 0; ++i)
        if ((i && o.rows[i] <= o.rows[i - 1]) || o.rows[i] >= (int32_t)o.mask.size()) return false;
    const size_t kept = i;
    for (; i < o.rows.size(); ++i) if (o.rows[i] != -1) return false;
    size_t zeros = 0;
    for (int32_t m : o.mask) {
        if (m != 0 && m != 1) return false;
        zeros += m == 0;
    }
    for (size_t j = 0; j < kept; ++j) if (o.mask[(size_t)o.rows[j]] != 0) return false;
    return zeros == kept;
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

/* Hand-built chunk: ties, a non-finite score of every kind, an id past the table, and k at .5. */
TEST(rowsel_class_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_rowsel", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    const std::vector<float> table = { 5.0f, 3.0f, -INFINITY, 5.0f, 1.0f, NAN, INFINITY, 2.0f };
    /* rows:                         0  1  2  3  4  5  6  7  8  9   matches: 0 1 3 4 5 9 (six) */
    const std::vector<int32_t> ids = { 0, 1, 2, 3, 4, 0, 9, 5, 6, 7 };
    const RowselOut half = run_rowsel(row, ids, table, 4, 0.5, 0, "class");   /* k = 3 */
    CHECK_EQ(half.rc, RAD_OK);
    CHECK(half.rows == (std::vector<int32_t>{ 0, 3, 5, -1 }));   /* the three 5.0s, by position */
    CHECK(half.mask == (std::vector<int32_t>{ 0, 1, 1, 0, 1, 0, 1, 1, 1, 1 }));
    const RowselOut more = run_rowsel(row, ids, table, 4, 0.75, 0, "class");  /* 4.5 -> 4 */
    CHECK(more.rows == (std::vector<int32_t>{ 0, 1, 3, 5 }));
    const RowselOut cut = run_rowsel(row, ids, table, 2, 0.75, 0, "class");   /* k 4 > cap 2 */
    CHECK(cut.rows == (std::vector<int32_t>{ 0, 3 }));
    CHECK(rowsel_consistent(cut));
    const RowselOut quarter = run_rowsel(row, ids, table, 4, 0.25, 0, "class");  /* 1.5 -> 2 */
    CHECK(quarter.rows == (std::vector<int32_t>{ 0, 3, -1, -1 }));
    const RowselOut none = run_rowsel(row, ids, table, 4, 1.0 / 12.0, 0, "class");  /* 0.5 -> 0 */
    CHECK(none.rows == (std::vector<int32_t>{ -1, -1, -1, -1 }));
    CHECK(none.mask == std::vector<int32_t>(10, 1));
    const RowselOut two = run_rowsel(row, ids, table, 4, 5.0 / 12.0, 0, "class");  /* 2.5 -> 2 */
    CHECK(two.rows == (std::vector<int32_t>{ 0, 3, -1, -1 }));
}

TEST(rowsel_random_and_all_semantics, "host") {
    const RadKernelInfo* row = find_row("kva_rowsel", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    std::vector<float> table(100, -INFINITY);
    std::vector<int32_t> ids(100);
    for (int i = 0; i < 100; ++i) { ids[(size_t)i] = i; if (i % 5 < 3) table[(size_t)i] = 1.0f; }
    /* 60 matches, share 0.25: k = 15, drawn from all 100 rows. */
    const RowselOut a = run_rowsel(row, ids, table, 32, 0.25, 7, "random");
    const RowselOut b = run_rowsel(row, ids, table, 32, 0.25, 7, "random");
    const RowselOut other = run_rowsel(row, ids, table, 32, 0.25, 8, "random");
    const RowselOut cut = run_rowsel(row, ids, table, 10, 0.25, 7, "random");
    CHECK_EQ(a.rc, RAD_OK);
    CHECK_EQ(kept_rows(a).size(), 15);
    CHECK(rowsel_consistent(a));
    CHECK(a.rows == b.rows);                       /* deterministic for a seed */
    CHECK(kept_rows(other) != kept_rows(a));       /* and the seed matters */
    CHECK_EQ(kept_rows(cut).size(), 10);           /* k 15 > cap 10: the 10 best of the 15 */
    for (int32_t r : kept_rows(cut))
        CHECK(std::find(a.rows.begin(), a.rows.end(), r) != a.rows.end());
    const RowselOut all = run_rowsel(row, { 3, 1, 4, 1, 5 }, table, 8, 0.25, 0, "all");
    CHECK(all.rows == (std::vector<int32_t>{ 0, 1, 2, 3, 4, -1, -1, -1 }));
    CHECK(all.mask == std::vector<int32_t>(5, 0));
    const RowselOut all_cut = run_rowsel(row, std::vector<int32_t>(12, 2), table, 8, 0.25, 0, "all");
    CHECK(all_cut.rows == (std::vector<int32_t>{ 0, 1, 2, 3, 4, 5, 6, 7 }));
    CHECK(all_cut.mask == (std::vector<int32_t>{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 }));
}

/* The random control keys on the ABSOLUTE position: two chunks of one prompt with the same tokens
 * keep the same count but different offsets; the same seed and positions keep the same rows; the
 * index row of a component-major [3, n] positions operand reads like [n]; class mode ignores it. */
TEST(rowsel_random_keys_on_position, "host") {
    const RadKernelInfo* row = find_row("kva_rowsel", RAD_DOMAIN_HOST);
    REQUIRE(row != nullptr);
    std::vector<float> table(64, -INFINITY);
    for (int i = 0; i < 64; i += 2) table[(size_t)i] = (float)(i % 7);
    std::vector<int32_t> ids(2048);
    for (int i = 0; i < 2048; ++i) ids[(size_t)i] = (i * 37) % 64;   /* half the rows match */
    const RowselOut first = run_rowsel(row, ids, table, 512, 0.25, 3, "random", 0);
    const RowselOut second = run_rowsel(row, ids, table, 512, 0.25, 3, "random", 2048);
    const RowselOut again = run_rowsel(row, ids, table, 512, 0.25, 3, "random", 2048);
    const RowselOut planes = run_rowsel(row, ids, table, 512, 0.25, 3, "random", 2048, 3);
    CHECK_EQ(kept_rows(first).size(), 256);    /* 1024 matches x 0.25, every chunk */
    CHECK_EQ(kept_rows(second).size(), 256);
    CHECK(kept_rows(first) != kept_rows(second));   /* not the same in-chunk offsets */
    CHECK(second.rows == again.rows && second.mask == again.mask);
    CHECK(planes.rows == second.rows);
    CHECK(rowsel_consistent(first) && rowsel_consistent(second));
    CHECK(run_rowsel(row, ids, table, 512, 0.25, 3, "class", 0).rows ==
          run_rowsel(row, ids, table, 512, 0.25, 3, "class", 2048).rows);
}

/* Refusals by name; the parse is shared with the device row. */
TEST(refuses_bad_operands, "both") {
    if (!group_runnable()) return;
    const RadKernelInfo* rs = find_row("kva_rowsel", group_domain());
    const RadKernelInfo* rh = find_row("kva_rho_update", group_domain());
    const RadKernelInfo* sc = find_row("kva_state_correct", group_domain());
    REQUIRE(rs && rh && sc);
    const std::vector<float> table = { 1.0f, 2.0f };
    CHECK_EQ(run_rowsel(rs, { 0, 1 }, table, 2, 0.5, 0, "classy").rc, RAD_E_INVAL);
    CHECK_EQ(run_rowsel(rs, { 0, 1 }, table, 2, 1.5, 0, "class").rc, RAD_E_INVAL);
    Buf tok = make(RAD_I32, { 4 }), score16 = make(RAD_BF16, { 2 }), rows = make(RAD_I32, { 2 });
    Buf mask = make(RAD_I32, { 4 }), short_mask = make(RAD_I32, { 3 }), score = make(RAD_F32, { 2 });
    Buf pos = make(RAD_I32, { 4 }), short_pos = make(RAD_I32, { 3 }), pos16 = make(RAD_BF16, { 4 });
    const std::vector<RadParam> p = { pint("M", 4), pint("cap", 2), pf64("share", 0.5),
                                      pint("seed", 0), pstr("mode", "class") };
    CHECK_EQ(run_group(rs, { &tok, &pos, &score16, &rows, &mask }, p), RAD_E_DTYPE);
    CHECK_EQ(run_group(rs, { &tok, &pos16, &score, &rows, &mask }, p), RAD_E_DTYPE);
    CHECK_EQ(run_group(rs, { &tok, &pos, &score, nullptr, &mask }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(rs, { &tok, nullptr, &score, &rows, &mask }, p), RAD_E_INVAL);
    CHECK_EQ(run_group(rs, { &tok, &pos, &score, &rows, &short_mask }, p), RAD_E_SHAPE);
    CHECK_EQ(run_group(rs, { &tok, &short_pos, &score, &rows, &mask }, p), RAD_E_SHAPE);
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

/* R33: on each quick doc's first chunk the row keeps exactly fnlev.rules' rows (the SIDECAR lane's
 * tools/rows_compare.py fixture, format kva-rowsel-fixture-1). */
TEST(rowsel_matches_fnlev_rules, "both") {
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
    const RadKernelInfo* row = find_row("kva_rowsel", group_domain());
    const RadKernelInfo* host = find_row("kva_rowsel", RAD_DOMAIN_HOST);
    REQUIRE(row && host && fx.get("vocab") && fx.get("kept_ids") && fx.get("kept_scores") &&
            fx.get("docs") && fx.get("share") && fx.get("window"));
    std::vector<float> table((size_t)fx.get("vocab")->num, -INFINITY);
    const std::vector<int32_t> kept = ints(fx.get("kept_ids"));
    for (size_t i = 0; i < kept.size(); ++i) table[(size_t)kept[i]] = (float)fx.get("kept_scores")->arr[i].num;
    const double share = fx.get("share")->num;
    /* PLAN D12's capacity at a chunk of `window` rows: share x window rounded up to 64. */
    const int64_t cap = (int64_t)std::ceil(share * fx.get("window")->num / 64.0) * 64;
    for (const Json& doc : fx.get("docs")->arr) {
        const std::vector<int32_t> ids = ints(doc.get("token_ids")), want = ints(doc.get("rows"));
        const RowselOut got = run_rowsel(row, ids, table, cap, share, 0, "class");
        CHECK_EQ(got.rc, RAD_OK);
        CHECK(rowsel_consistent(got));
        CHECK_EQ(kept_rows(got).size(), (size_t)doc.get("k")->num);
        if (kept_rows(got) != want) fail_at(__FILE__, __LINE__, doc.get("doc")->str + ": rows differ from fnlev.rules");
        if (row != host) CHECK(got.rows == run_rowsel(host, ids, table, cap, share, 0, "class").rows);
        std::fprintf(stderr, "  %-10s n %zu  k %zu  == fnlev.rules%s\n", doc.get("doc")->str.c_str(),
                     ids.size(), want.size(), row != host ? " == host row" : "");
    }
}

/* ================================================================== device against host */

/* R20: the device row and the host row on the same random operands -- model-sized heads, padded
 * slot / head / row strides, nonzero applied scales, a skipped sequence -- agree to the bit, over
 * the whole buffers (padding included). */
TEST(state_correct_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_state_correct", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_state_correct", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 31 };
    CorrectRun k = correct_operands(6, 24, 128, 128, { 4, -1, 1, 5 }, 8);
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
    sep.applied = make(RAD_F32, { 4, 24, 1, 1 });
    sep.nd = make(RAD_F32, { 7, 24, 1, 2 });
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
            std::fprintf(stderr, "  %s %-5s alpha %.1f ND %d: %zu state bytes, %zu differ (max abs diff %s)\n",
                         pass ? "own slots  " : "shared slots", st.mode, st.alpha, (int)st.nd,
                         h.state.bytes.size(), differ, differ ? ">0" : "0");
        }
}

/* kva_state_read, device vs host: model-sized heads at padded strides, a negative slot, a slot past
 * the pool and a repeated slot; the outputs agree to the bit and the out-of-pool rows are zeros. */
TEST(state_read_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_state_read", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_state_read", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 61 };
    CorrectRun k = correct_operands(6, 24, 128, 128, { 0 }, 8);
    for (size_t i = 0; i < k.state.bytes.size() / 4; ++i) setf(k.state, (int64_t)i, r.normal());
    Buf index = slot_index({ 4, -1, 1, 7, 4 }), out_host, out_dev;
    CHECK_EQ(run_state_read(host, k, index, out_host), RAD_OK);
    CHECK_EQ(run_state_read(dev, k, index, out_dev), RAD_OK);
    size_t differ = 0, nonzero_out_of_pool = 0;
    const int64_t per_seq = 24 * 128 * 128;
    for (size_t i = 0; i < out_host.bytes.size(); ++i) differ += out_host.bytes[i] != out_dev.bytes[i];
    for (int64_t s : { 1, 3 }) for (int64_t e = 0; e < per_seq; ++e)
        nonzero_out_of_pool += getf(out_dev, s * per_seq + e) != 0.0f;
    CHECK_EQ(differ, 0);
    CHECK_EQ(nonzero_out_of_pool, 0);
    std::fprintf(stderr, "  %zu out bytes, %zu differ; out-of-pool rows all zero: %s\n",
                 out_host.bytes.size(), differ, nonzero_out_of_pool ? "no" : "yes");
}

/* R33's device leg: the device row equals the host row on chunks with many ties, non-matching
 * rows, every mode, truncation (k > cap), tiny and maximal n. */
TEST(rowsel_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_rowsel", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_rowsel", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 41 };
    std::vector<float> table(1000);
    const float levels[] = { -INFINITY, -INFINITY, 1.0f, 2.0f, 2.5f, 3.0f };   /* ties everywhere */
    for (float& s : table) s = levels[r.below(6)];
    const int64_t sizes[] = { 1, 7, 300, 2048, 8192 };
    const char* modes[] = { "class", "random", "all" };
    int cases_run = 0;
    for (int64_t n : sizes) {
        std::vector<int32_t> ids((size_t)n);
        for (int32_t& id : ids) id = (int32_t)r.below(1010) - 5;   /* a few ids outside the table */
        for (const char* mode : modes)
            for (int64_t cap : { (int64_t)3, (int64_t)512, n })
                for (double share : { 0.25, 1.0 }) {
                    const int64_t comps = cases_run % 2 ? 3 : 1;   /* every other case [3, n] */
                    const RowselOut h = run_rowsel(host, ids, table, cap, share, 12345, mode, 4096, comps);
                    const RowselOut d = run_rowsel(dev, ids, table, cap, share, 12345, mode, 4096, comps);
                    CHECK_EQ(d.rc, RAD_OK);
                    CHECK(d.rows == h.rows && d.mask == h.mask);
                    CHECK(rowsel_consistent(d));
                    ++cases_run;
                }
    }
    std::vector<int32_t> ids(2048);
    for (int32_t& id : ids) id = (int32_t)r.below(1000);
    const RowselOut once = run_rowsel(dev, ids, table, 512, 0.25, 99, "random", 2048);
    const RowselOut again = run_rowsel(dev, ids, table, 512, 0.25, 99, "random", 2048);
    const RowselOut next = run_rowsel(dev, ids, table, 512, 0.25, 99, "random", 4096);
    CHECK(once.rows == again.rows && once.mask == again.mask);   /* deterministic: seed + positions */
    CHECK(kept_rows(next).size() == kept_rows(once).size() && kept_rows(next) != kept_rows(once));
    std::fprintf(stderr, "  %d configurations: device rows_idx and mask == host\n", cases_run);
}

/* R34's device leg: device vs host on model-sized gates (bf16 a as a column slice of the a|b
 * buffer, carried over two chunks); rho within 1e-5, N and D reported. */
TEST(rho_device_matches_host, "gpu") {
    if (!group_runnable()) return;
    const RadKernelInfo* dev = find_row("kva_rho_update", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* host = find_row("kva_rho_update", RAD_DOMAIN_HOST);
    REQUIRE(dev && host);
    Rng r{ 51 };
    double worst_rho = 0, worst_rel = 0;
    for (double exact : { 0.056, 0.5, 0.0 }) {
        RhoRun h = random_rho(r, 2048, 24, exact, true);
        h.pitch = 48;
        RhoRun d = h;
        for (int chunk = 0; chunk < 2; ++chunk) {
            CHECK_EQ(run_rho(host, h), RAD_OK);
            CHECK_EQ(run_rho(dev, d), RAD_OK);
        }
        for (int64_t k = 0; k < 24; ++k) {
            worst_rho = std::max(worst_rho, (double)std::fabs(nd_rho(h, k) - nd_rho(d, k)));
            for (int w = 0; w < 2; ++w) {
                const double a = h.nd[(size_t)((2 * 24 + k) * 2 + w)], b = d.nd[(size_t)((2 * 24 + k) * 2 + w)];
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

/* ================================================================== main */

int main(int argc, char** argv) {
    if (argc != 3 || (std::strcmp(argv[2], "host") && std::strcmp(argv[2], "gpu"))) {
        std::fprintf(stderr, "usage: %s <kva.so> host|gpu\n", argv[0]);
        return 2;
    }
    g_group = argv[2];
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
