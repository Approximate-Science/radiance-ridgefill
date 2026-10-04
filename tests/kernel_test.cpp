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

static const char* const kOps[] = { "kva_rowsel", "kva_rho_update", "kva_state_correct" };

/* Rows whose launch is still a stub (R8). Emptied as each row is implemented; the case then skips. */
static const char* const kStubbed[] = {
    "kva_rowsel_host", "kva_rho_update_host", "kva_state_correct_host",
    "kva_rowsel_device", "kva_rho_update_device", "kva_state_correct_device",
};

static bool is_stub(const char* name) {
    for (const char* s : kStubbed) if (!std::strcmp(s, name)) return true;
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
