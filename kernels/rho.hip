/* rho.hip -- kva_rho_update's device row: one thread a head, sequential over the chunk's rows.
 *
 * The recurrence is sequential in t by definition and the work is n * n_head multiply-adds, so
 * one thread per head is the whole kernel; consecutive heads are consecutive columns of `a`, so a
 * wave's loads of one row are contiguous. Same arithmetic, in the same order, as the host row.
 */
#include <hip/hip_runtime.h>

#include "kva.h"

enum { KVA_RHO_THREADS = 64 };

__global__ void __launch_bounds__(KVA_RHO_THREADS) kva_rho_kernel(KvaRho g) {
    const int64_t h = (int64_t)blockIdx.x * KVA_RHO_THREADS + threadIdx.x;
    if (h >= g.n_head) return;
    const int32_t slot = g.state_idx[0];
    if (slot < 0 || slot >= g.n_states) return;
    float* nd = g.nd + slot * g.nd_slot + h * g.nd_head;
    float n = nd[0], d = nd[g.nd_inner];
    const float decay = expf(g.a_log[h]);
    const float bias = g.dt_bias[h];
    for (int64_t t = 0; t < g.n; ++t) {
        const int64_t at = t * g.a_pitch + h * g.a_col;
        const float av = g.a_bf16 ? kva_bf16_to_f32(((const uint16_t*)g.a)[at])
                                  : ((const float*)g.a)[at];
        const float e = expf(-decay * kva_softplus(av + bias));
        d = e * d + 1.0f;
        n = e * n + (float)g.mask[t];
    }
    nd[0] = n;
    nd[g.nd_inner] = d;
}

extern "C" int kva_rho_device(const RadArgs* a, RadStream s) {
    KvaRho g{};
    const int rc = kva_rho_parse(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.n == 0 || g.n_head == 0) return RAD_OK;
    const unsigned blocks = (unsigned)((g.n_head + KVA_RHO_THREADS - 1) / KVA_RHO_THREADS);
    kva_rho_kernel<<<blocks, KVA_RHO_THREADS, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
