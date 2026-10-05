/* select.hip -- kva_select and kva_drop_rows' device rows: one workgroup per row of the mask,
 * which returns at once unless the row is approximated (mask 1).
 *
 * kva_select is a dtype-agnostic byte copy; each pair is copied in the widest word its addresses,
 * pitches and row width allow (KvaCopy::word, picked at parse), so a bf16 activation row moves in
 * 16-byte loads. A workgroup of an exact row costs one 4-byte read of the mask.
 */
#include <hip/hip_runtime.h>

#include <climits>

#include "kva.h"

enum { KVA_ROW_THREADS = 256 };

template <typename Word>
__device__ static void copy_row(const KvaCopy& p, int64_t row) {
    const Word* src = (const Word*)(p.src + row * p.src_pitch);
    Word* dst = (Word*)(p.dst + row * p.dst_pitch);
    const int64_t words = p.row_bytes / (int64_t)sizeof(Word);
    for (int64_t w = threadIdx.x; w < words; w += KVA_ROW_THREADS) dst[w] = src[w];
}

__global__ void __launch_bounds__(KVA_ROW_THREADS) kva_select_kernel(KvaSelect g) {
    const int64_t row = blockIdx.x;
    if (g.mask[row] != 1) return;
    for (int k = 0; k < KVA_SELECT_PAIRS; ++k) {
        const KvaCopy& p = g.pair[k];
        if (p.word == 16) copy_row<uint4>(p, row);
        if (p.word == 4) copy_row<uint32_t>(p, row);
        if (p.word == 1) copy_row<uint8_t>(p, row);
    }
}

extern "C" int kva_select_device(const RadArgs* a, RadStream s) {
    KvaSelect g{};
    const int rc = kva_select_parse(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.n == 0) return RAD_OK;
    if (g.n > INT_MAX) return RAD_E_SHAPE;
    kva_select_kernel<<<(unsigned)g.n, KVA_ROW_THREADS, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

__global__ void __launch_bounds__(KVA_ROW_THREADS) kva_drop_kernel(KvaDrop g) {
    const int64_t row = blockIdx.x;
    if (g.mask[row] != 1) return;
    int32_t* ids = g.ids + row * g.pitch;
    for (int64_t j = threadIdx.x; j < g.top_k; j += KVA_ROW_THREADS) ids[j] = -1;
}

extern "C" int kva_drop_device(const RadArgs* a, RadStream s) {
    KvaDrop g{};
    const int rc = kva_drop_parse(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.n == 0 || g.top_k == 0) return RAD_OK;
    if (g.n > INT_MAX) return RAD_E_SHAPE;
    kva_drop_kernel<<<(unsigned)g.n, KVA_ROW_THREADS, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
