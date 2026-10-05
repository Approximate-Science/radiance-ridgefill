/* forward.cpp -- kva_gemm_nt_bias and kva_gemm_nt_q: the engine's own gemm_nt_bias and int8
 * gemm_nt_q rows, offered again under ops whose weight operands are IN operands (PACKAGING.md §0,
 * REFUTATION-3 §2.2).
 *
 * WHY. The projector lives in plugin-owned memory filled from the projector folder at declare, not
 * in the container, so it has no weight handle: libr4d's GEMMs refuse a declare without one for
 * every WEIGHT position (core/build/rad_builder.cpp:676-705). These ops have the same parameters
 * and operands with the weights as plain inputs, and their rows ARE the source library's rows: the
 * same launch, describe, scratch and init hooks, the same constraints -- so the kernel that runs,
 * and every byte it writes, is the one the source op would have run. Nothing is computed here.
 *
 * WHERE THE ROWS COME FROM. The public kernel ABI of the libraries the engine already loaded
 * (rad_plugin_info, rad_kernel_count/at, rad_kernel_concurrent; abi/rad_abi.h:515-597): the device
 * rows from "libr4d", the bf16 host row (the oracle) from "libref". The loader dlopens every plugin
 * of every home before it commits any (core/plugin/loader.cpp:563-575), and commits read this
 * library's row count, so by the first rad_kernel_count the sources are mapped; each is found by
 * walking the loaded objects (dl_iterate_phdr) and re-opened with RTLD_NOLOAD -- never loaded by
 * this library. The handle is kept, so a source the loader later unloads stays mapped.
 *
 * A SOURCE THAT IS NOT LOADED OFFERS NO ROW: the op then resolves to nothing, the arch plugin
 * refuses KVA by name and the engine serves stock. A row with tunables is never forwarded (the
 * tuning cache keys rows by name). The bf16 rows have no layout hooks and none is accepted: the
 * projector is read as the folder stores it. The int8 rows KEEP their layout and relayout hooks:
 * the engine consults a row's hooks only for WEIGHT operands (rad_builder.cpp:510-545,
 * ctx.cpp:470-490, oracle.cpp:463), which this op has none of, so here they are inert -- and they
 * are how the arch plugin turns the folder's canonical int8 planes into the stored form these
 * kernels read, once, at load (arch/kva_int8.h). The file never carries a library's layout.
 * libref's int8 row is not forwarded: it reads canonical planes, and the operands are stored ones.
 */
#include "kva.h"

#include <dlfcn.h>
#include <link.h>

#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

struct Source {
    const char* plugin;   /* rad_plugin_info()->name */
    int         domain;
    const char* src_op;   /* the source op whose rows are forwarded */
    const char* dtype;    /* null, or the dtype a row's constraints must admit */
    bool        hooks;    /* the rows keep their layout hooks (else a row with them is not taken) */
    const char* op;       /* the op the rows are offered under */
    const char* name;     /* the forwarded rows' name prefix in this library */
};

const Source kSources[] = {
    { "libr4d", RAD_DOMAIN_DEVICE, "gemm_nt_bias", nullptr, false, "kva_gemm_nt_bias", "kva_gemm_nt_bias_r4d" },
    { "libref", RAD_DOMAIN_HOST,   "gemm_nt_bias", nullptr, false, "kva_gemm_nt_bias", "kva_gemm_nt_bias_ref" },
    { "libr4d", RAD_DOMAIN_DEVICE, "gemm_nt_q",    "i8a8",  true,  "kva_gemm_nt_q",    "kva_gemm_nt_q_r4d" },
};
constexpr int kMaxRows = 8;

RadKernelInfo g_rows[kMaxRows];
int           g_concurrent[kMaxRows];
char          g_computes[kMaxRows][200];
char          g_names[kMaxRows][96];
int           g_count = 0;
std::once_flag g_once;

/* Whether row `k`'s constraints admit `dtype` (an IN constraint on the key "dtype" naming it). */
bool admits(const RadKernelInfo* k, const char* dtype) {
    for (int i = 0; i < k->n_constraints; ++i) {
        const RadConstraint& c = k->constraints[i];
        if (c.op != RAD_C_IN || !c.key || std::strcmp(c.key, "dtype") || !c.sval) continue;
        const size_t n = std::strlen(dtype);
        for (const char* p = c.sval; (p = std::strstr(p, dtype)); p += n)
            if ((p == c.sval || p[-1] == ' ') && (p[n] == 0 || p[n] == ' ')) return true;
    }
    return false;
}

/* Whether `k` is a row `s` forwards. */
bool wanted(const Source& s, const RadKernelInfo* k) {
    if (!k || k->domain != s.domain || !k->op || std::strcmp(k->op, s.src_op) != 0) return false;
    if (k->n_tunables > 0 || (!s.hooks && (k->layout || k->relayout))) return false;
    return !s.dtype || admits(k, s.dtype);
}

/* The source's row, renamed onto this library's op; every hook is the source's (the layout hooks
 * only where the source keeps them). */
void add_row(const Source& s, const RadKernelInfo* k, int concurrent) {
    RadKernelInfo r = *k;
    std::snprintf(g_names[g_count], sizeof g_names[g_count], "%s%s%s", s.name, s.hooks ? "_" : "",
                  s.hooks ? k->name : "");
    std::snprintf(g_computes[g_count], sizeof g_computes[g_count],
                  "%s's %s, forwarded: the same kernel with its weight operands as IN operands", s.plugin,
                  k->name);
    r.name = g_names[g_count];
    r.op = s.op;
    r.computes = g_computes[g_count];
    if (!s.hooks) r.unrelayout = nullptr;
    r.replaces = nullptr;
    g_concurrent[g_count] = concurrent;
    g_rows[g_count++] = r;
}

/* Every row of `s` in the library behind handle `h`; how many were taken. */
int take_rows(void* h, const Source& s) {
    auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    auto conc = (int (*)(int))dlsym(h, "rad_kernel_concurrent");
    if (!count || !at) return 0;
    for (int i = 0; i < g_count; ++i)
        if (!std::strncmp(g_rows[i].name, s.name, std::strlen(s.name))) return 0;   /* first source wins */
    int taken = 0;
    for (int i = 0; i < count() && g_count < kMaxRows; ++i) {
        const RadKernelInfo* k = at(i);
        if (!wanted(s, k)) continue;
        add_row(s, k, conc ? conc(i) != 0 : 0);
        ++taken;
        if (!s.hooks) break;   /* the bf16 op: the source's first row, under the source's one name */
    }
    return taken;
}

int visit(struct dl_phdr_info* info, size_t, void*) {
    if (!info->dlpi_name || !info->dlpi_name[0] || g_count == kMaxRows) return 0;
    void* h = dlopen(info->dlpi_name, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (!h) return 0;
    auto plugin = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
    const RadPluginInfo* p = plugin ? plugin() : nullptr;
    int kept = 0;
    for (const Source& s : kSources)
        if (p && p->name && std::strcmp(p->name, s.plugin) == 0) kept += take_rows(h, s);
    if (!kept) dlclose(h);
    return 0;
}

void resolve() { dl_iterate_phdr(visit, nullptr); }

}  /* namespace */

extern "C" int kva_forward_count(void) {
    std::call_once(g_once, resolve);
    return g_count;
}

extern "C" const RadKernelInfo* kva_forward_at(int i) {
    return i >= 0 && i < kva_forward_count() ? &g_rows[i] : nullptr;
}

extern "C" int kva_forward_concurrent(int i) {
    return i >= 0 && i < kva_forward_count() ? g_concurrent[i] : 0;
}
