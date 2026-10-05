/* kva_int8.h -- the int8 projector (Stage E, R79): the folder's canonical int8 planes turned into
 * the stored form the engine's own int8 GEMM reads, once per rank at load.
 *
 * THE FILE IS CANONICAL, THE CARD IS NOT. An int8 folder holds each map in the encoding the
 * container's int8 trunk uses, i8*bf16[1x128] (tools/kva_projector.py int8): codes i8 [n, wide]
 * row-major and a bf16 scale per 128 columns of a row. The GEMM that reads them is libr4d's int8
 * gemm_nt_q, which wants its own arrangement (fragment-order codes, tile-major scales) and says so
 * through its layout/relayout hooks (abi/rad_abi.h:154-201). kva.so forwards those rows as
 * kva_gemm_nt_q WITH their hooks (kernels/forward.cpp); this file finds them in the loaded kva.so
 * and runs relayout on the folder's planes before the upload. So the folder never carries a
 * library's layout, and a libr4d that changes its arrangement changes it here too.
 *
 * Every forwarded row must describe the same stored bytes (one upload serves whichever row the
 * engine picks for a step's M); rows that disagree, or no row at all, refuse the int8 folder by
 * name and the engine serves stock.
 */
#ifndef QWEN4EXP_KVA_INT8_H
#define QWEN4EXP_KVA_INT8_H

#include <dlfcn.h>
#include <link.h>

#include <string>
#include <vector>

namespace qwen4exp_kva {

/* The int8 encoding's scale group (i8*bf16[1x128]): libr4d's int8 rows and quant_act_i8g read 128. */
constexpr int64_t kI8Group = 128;

/* Tests hand the rows in here instead of a loaded kva.so (tests/arch_static_test.cpp). */
static const std::vector<const RadKernelInfo*>* g_i8_rows_for_test = nullptr;

/* kva.so's kva_gemm_nt_q rows that carry a relayout, from the library behind `h`. */
inline void kva_rows_of(void* h, std::vector<const RadKernelInfo*>* out) {
    auto info = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
    auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    const RadPluginInfo* p = info ? info() : nullptr;
    if (!p || !p->name || std::strcmp(p->name, "kva") || !count || !at) return;
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && !std::strcmp(k->op, "kva_gemm_nt_q") && k->layout && k->relayout) out->push_back(k);
    }
}

inline int visit_kva(struct dl_phdr_info* info, size_t, void* out) {
    auto* rows = (std::vector<const RadKernelInfo*>*)out;
    if (!info->dlpi_name || !info->dlpi_name[0] || !rows->empty()) return 0;
    void* h = dlopen(info->dlpi_name, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (!h) return 0;
    kva_rows_of(h, rows);
    if (rows->empty()) dlclose(h);   /* kept open otherwise: the row pointers live in it */
    return 0;
}

/* The int8 GEMM rows the engine can pick for the projector. */
inline std::vector<const RadKernelInfo*> i8_rows() {
    if (g_i8_rows_for_test) return *g_i8_rows_for_test;
    std::vector<const RadKernelInfo*> rows;
    dl_iterate_phdr(visit_kva, &rows);
    return rows;
}

/* One map's stored form: codes (operand b) and scale (operand b_scale). */
struct I8Stored { std::vector<unsigned char> codes, scale; };

inline RadTensor plane(uint32_t dtype, int64_t rows, int64_t cols, const void* data) {
    RadTensor t{};
    t.dtype = dtype;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    rad_tensor_pack(&t);
    t.data = (void*)data;
    return t;
}

/* The stored form every row describes for operand `opd` (2 codes, 3 scale); empty `why` on success. */
inline RadLayout stored_layout(const std::vector<const RadKernelInfo*>& rows, const RadParam* p, int n_p,
                               int opd, const RadEncoding& enc, const RadTensor& geom, std::string* why) {
    RadLayout first{};
    const int sel = opd == 2 ? 0 : 1;
    for (size_t i = 0; i < rows.size() && why->empty(); ++i) {
        RadLayout L{};
        if (rows[i]->layout(p, n_p, opd, &enc, &sel, &geom, 1, &L) != RAD_OK) {
            *why = std::string("the int8 GEMM ") + rows[i]->name + " does not read i8*bf16[1x128] at this shape";
        } else if (i == 0) {
            first = L;
        } else if (L.bytes != first.bytes || std::strcmp(L.tag ? L.tag : "", first.tag ? first.tag : "")) {
            *why = std::string("the int8 GEMM rows store the map differently (") + (first.tag ? first.tag : "?") +
                   " vs " + (L.tag ? L.tag : "?") + "); one upload cannot serve both";
        }
    }
    return first;
}

/* The [n, k] map `codes` / `scale` in the stored form; false and `why` when it cannot be made. */
inline bool relayout_i8(const std::vector<const RadKernelInfo*>& rows, int64_t n, int64_t k,
                        const void* codes, const void* scale, I8Stored* out, std::string* why) {
    if (rows.empty()) {
        *why = "no kernel library offers the int8 GEMM (kva_gemm_nt_q: kva.so forwards libr4d's int8 "
               "gemm_nt_q rows only when libr4d is loaded)";
        return false;
    }
    const RadParam p[] = { RAD_INT("M", 1), RAD_INT("N", n), RAD_INT("K", k), RAD_INT("group", kI8Group),
                           RAD_STR("dtype", "i8a8") };
    const int n_p = (int)(sizeof p / sizeof p[0]);
    const RadEncoding enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, kI8Group);
    const RadTensor c = plane(RAD_I8, n, k, codes), s = plane(RAD_BF16, n, k / kI8Group, scale);
    const RadLayout lc = stored_layout(rows, p, n_p, 2, enc, c, why);
    const RadLayout ls = why->empty() ? stored_layout(rows, p, n_p, 3, enc, s, why) : RadLayout{};
    if (!why->empty()) return false;
    out->codes.assign((size_t)lc.bytes, 0);
    out->scale.assign((size_t)ls.bytes, 0);
    const int sc = 0, ss = 1;
    if (rows[0]->relayout(p, n_p, 2, &enc, &sc, &c, 1, out->codes.data(), lc.bytes) != RAD_OK ||
        rows[0]->relayout(p, n_p, 3, &enc, &ss, &s, 1, out->scale.data(), ls.bytes) != RAD_OK) {
        *why = std::string("the int8 GEMM ") + rows[0]->name + "'s relayout refused the map";
        return false;
    }
    return true;
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_INT8_H */
