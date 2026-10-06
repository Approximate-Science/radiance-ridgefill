/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* rho.hip -- ridgefill_rho_update's device row: one thread a head, sequential over the chunk's rows.
 *
 * The recurrence is sequential in t by definition and the work is n * n_head multiply-adds, so
 * one thread per head is the whole kernel; consecutive heads are consecutive columns of `a`, so a
 * wave's loads of one row are contiguous. Same arithmetic, in the same order, as the host row.
 */
#include <hip/hip_runtime.h>

#include "ridgefill.h"

enum { RIDGEFILL_RHO_THREADS = 64 };

__global__ void __launch_bounds__(RIDGEFILL_RHO_THREADS) ridgefill_rho_kernel(RidgeFillRho g) {
    const int64_t h = (int64_t)blockIdx.x * RIDGEFILL_RHO_THREADS + threadIdx.x;
    if (h >= g.n_head) return;
    const int32_t slot = g.state_idx[0];
    if (slot < 0 || slot >= g.n_states) return;
    float* nd = g.nd + slot * g.nd_slot + h * g.nd_head;
    float n = nd[0], d = nd[g.nd_inner];
    const float decay = expf(g.a_log[h]);
    const float bias = g.dt_bias[h];
    for (int64_t t = ridgefill_rho_first_row(&g); t < g.n; ++t) {
        const int64_t at = t * g.a_pitch + h * g.a_col;
        const float av = g.a_bf16 ? ridgefill_bf16_to_f32(((const uint16_t*)g.a)[at])
                                  : ((const float*)g.a)[at];
        const float e = expf(-decay * ridgefill_softplus(av + bias));
        d = e * d + 1.0f;
        n = e * n + (float)g.mask[t];
    }
    nd[0] = n;
    nd[g.nd_inner] = d;
}

extern "C" int ridgefill_rho_device(const RadArgs* a, RadStream s) {
    RidgeFillRho g{};
    const int rc = ridgefill_rho_parse(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.n == 0 || g.n_head == 0) return RAD_OK;
    const unsigned blocks = (unsigned)((g.n_head + RIDGEFILL_RHO_THREADS - 1) / RIDGEFILL_RHO_THREADS);
    ridgefill_rho_kernel<<<blocks, RIDGEFILL_RHO_THREADS, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
