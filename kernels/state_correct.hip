/* state_correct.hip -- the device rows that touch a GDN state slot: kva_state_correct (apply /
 * undo the correction) and kva_state_read (copy the slot out), one workgroup per (head, sequence)
 * each. kva_state_correct addresses each pool (state, applied, ND) through its own slot index
 * (kva_correct_slots).
 *
 * Each workgroup adds (apply) or subtracts (undo) scale * C over one head's sd0 x sd1 state, at
 * the strides the operands carry: a linear slot may be padded, so nothing here assumes a dense
 * pool. `x + s*c` rounds twice (-ffp-contract=off), exactly as the host row, so the two agree to
 * the bit.
 */
#include <hip/hip_runtime.h>

#include <climits>

#include "kva.h"

enum { KVA_CORRECT_THREADS = 256 };

__global__ void __launch_bounds__(KVA_CORRECT_THREADS) kva_correct_kernel(KvaCorrect g) {
    const int64_t h = blockIdx.x, s = blockIdx.y;
    int64_t st_slot = 0, ap_slot = 0, nd_slot = 0;
    if (!kva_correct_has_bulk(&g)) return;   /* the bounds are device memory: read here, not at launch */
    if (!kva_correct_slots(&g, s, &st_slot, &ap_slot, &nd_slot)) return;   /* whole workgroup */
    float* applied = g.applied + ap_slot * g.ap_slot + h * g.ap_head;
    float scale = *applied;
    if (g.apply) {
        const float* nd = g.nd ? g.nd + nd_slot * g.nd_slot + h * g.nd_head : nullptr;
        scale = g.alpha * (nd ? kva_rho(nd[0], nd[g.nd_inner]) : 1.0f);
    }
    __syncthreads();   /* every thread has read `applied` before thread 0 rewrites it */
    float* state = g.state + st_slot * g.st_slot + h * g.st_head;
    const float* c = g.c + h * g.c_head;
    const int cols = (int)g.sd1, cells = (int)(g.sd0 * g.sd1);
    for (int e = threadIdx.x; e < cells && scale != 0.0f; e += KVA_CORRECT_THREADS) {
        const int i = e / cols, j = e - i * cols;
        float* x = state + i * g.st_row + j * g.st_col;
        const float sc = scale * c[i * g.c_row + j * g.c_col];
        *x = g.apply ? *x + sc : *x - sc;
    }
    if (threadIdx.x == 0) *applied = g.apply ? scale : 0.0f;
}

extern "C" int kva_correct_device(const RadArgs* a, RadStream s) {
    KvaCorrect g{};
    const int rc = kva_correct_parse(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.n_seq == 0 || g.n_head == 0) return RAD_OK;
    if (g.sd0 * g.sd1 > INT_MAX || g.n_head > INT_MAX || g.n_seq > 65535) return RAD_E_SHAPE;
    const dim3 grid((unsigned)g.n_head, (unsigned)g.n_seq);
    kva_correct_kernel<<<grid, KVA_CORRECT_THREADS, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* kva_state_read: a straight copy of one head of one sequence's slot into the dense output, zeros
 * for a slot outside the pool. */
__global__ void __launch_bounds__(KVA_CORRECT_THREADS) kva_state_read_kernel(KvaStateRead g) {
    const int64_t h = blockIdx.x, s = blockIdx.y;
    const int32_t slot = g.idx[s * g.idx_pitch];
    const bool in_pool = slot >= 0 && slot < g.n_states;
    const float* state = in_pool ? g.state + slot * g.st_slot + h * g.st_head : g.state;
    float* out = g.out + (s * g.n_head + h) * g.sd0 * g.sd1;
    const int cols = (int)g.sd1, cells = (int)(g.sd0 * g.sd1);
    for (int e = threadIdx.x; e < cells; e += KVA_CORRECT_THREADS) {
        const int i = e / cols, j = e - i * cols;
        out[e] = in_pool ? state[i * g.st_row + j * g.st_col] : 0.0f;
    }
}

extern "C" int kva_state_read_device(const RadArgs* a, RadStream s) {
    KvaStateRead g{};
    const int rc = kva_state_read_parse(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.n_seq == 0 || g.n_head == 0) return RAD_OK;
    if (g.sd0 * g.sd1 > INT_MAX || g.n_head > INT_MAX || g.n_seq > 65535) return RAD_E_SHAPE;
    const dim3 grid((unsigned)g.n_head, (unsigned)g.n_seq);
    kva_state_read_kernel<<<grid, KVA_CORRECT_THREADS, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
