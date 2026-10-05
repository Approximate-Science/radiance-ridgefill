/* forward.cpp -- kva_gemm_nt_bias: the engine's own gemm_nt_bias rows, offered again under an op
 * whose weight `b` and `bias` are IN operands (PACKAGING.md §0, REFUTATION-3 §2.2).
 *
 * WHY. The projector lives in plugin-owned memory filled from the projector folder at declare, not
 * in the container, so it has no weight handle: libr4d's gemm_nt_bias refuses a declare without one
 * for every WEIGHT position (core/build/rad_builder.cpp:676-705). This op has the same parameters
 * and operands with the two weights as plain inputs, and its rows ARE the source library's rows:
 * the same launch, describe, scratch and init hooks, the same constraints -- so the kernel that
 * runs, and every byte it writes, is the one gemm_nt_bias would have run. Nothing is computed here.
 *
 * WHERE THE ROWS COME FROM. The public kernel ABI of the libraries the engine already loaded
 * (rad_plugin_info, rad_kernel_count/at, rad_kernel_concurrent; abi/rad_abi.h:515-597): the device
 * row from "libr4d", the host row (the oracle) from "libref". The loader dlopens every plugin of
 * every home before it commits any (core/plugin/loader.cpp:563-575), and commits read this
 * library's row count, so by the first rad_kernel_count the sources are mapped; each is found by
 * walking the loaded objects (dl_iterate_phdr) and re-opened with RTLD_NOLOAD -- never loaded by
 * this library. The handle is kept, so a source the loader later unloads stays mapped.
 *
 * A SOURCE THAT IS NOT LOADED OFFERS NO ROW: the op then resolves to nothing, the arch plugin
 * refuses KVA by name and the engine serves stock. A row with layout hooks is not forwarded (an IN
 * operand is never relaid out), nor is one with tunables (the tuning cache keys rows by name).
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
    const char* name;     /* the forwarded row's name in this library */
};

const Source kSources[] = {
    { "libr4d", RAD_DOMAIN_DEVICE, "kva_gemm_nt_bias_r4d" },
    { "libref", RAD_DOMAIN_HOST,   "kva_gemm_nt_bias_ref" },
};
constexpr int kMaxRows = (int)(sizeof kSources / sizeof kSources[0]);

RadKernelInfo g_rows[kMaxRows];
int           g_concurrent[kMaxRows];
char          g_computes[kMaxRows][160];
int           g_count = 0;
std::once_flag g_once;

/* `info`'s gemm_nt_bias row in `domain` from the library behind handle `h`, or null; its
 * rad_kernel_concurrent answer into *concurrent. */
const RadKernelInfo* source_row(void* h, int domain, int* concurrent) {
    auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    auto conc = (int (*)(int))dlsym(h, "rad_kernel_concurrent");
    if (!count || !at) return nullptr;
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || k->domain != domain || !k->op || std::strcmp(k->op, "gemm_nt_bias") != 0) continue;
        if (k->layout || k->relayout || k->n_tunables > 0) return nullptr;
        *concurrent = conc ? conc(i) != 0 : 0;
        return k;
    }
    return nullptr;
}

/* The source's row, renamed onto this library's op; every hook is the source's. */
void add_row(const Source& s, const RadKernelInfo* k, int concurrent) {
    RadKernelInfo r = *k;
    std::snprintf(g_computes[g_count], sizeof g_computes[g_count],
                  "%s's %s, forwarded: the same kernel with b and bias as IN operands", s.plugin,
                  k->name);
    r.name = s.name;
    r.op = "kva_gemm_nt_bias";
    r.computes = g_computes[g_count];
    r.unrelayout = nullptr;
    r.replaces = nullptr;
    g_concurrent[g_count] = concurrent;
    g_rows[g_count++] = r;
}

int visit(struct dl_phdr_info* info, size_t, void*) {
    if (!info->dlpi_name || !info->dlpi_name[0] || g_count == kMaxRows) return 0;
    void* h = dlopen(info->dlpi_name, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (!h) return 0;
    auto plugin = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
    const RadPluginInfo* p = plugin ? plugin() : nullptr;
    bool kept = false;
    for (const Source& s : kSources) {
        if (!p || !p->name || std::strcmp(p->name, s.plugin) != 0) continue;
        bool taken = false;
        for (int i = 0; i < g_count; ++i) taken = taken || std::strcmp(g_rows[i].name, s.name) == 0;
        int concurrent = 0;
        const RadKernelInfo* k = taken ? nullptr : source_row(h, s.domain, &concurrent);
        if (k) { add_row(s, k, concurrent); kept = true; }
    }
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
