/* mask.hip -- kva_mask's device row: one workgroup ranks the window of the step's last sequence.
 *
 * The window W = [s, b') is read here, from cu_last in device memory (kva_mask_window); the host
 * knows only b (token_ids' extent) and n (the mask's). Each thread owns a contiguous run of window
 * rows; the rank of a row is counted against every other window row's key in LDS (O(w^2) reads at
 * worst, cut short once a row is known to be out); at the window sizes the engine cuts that is a
 * few microseconds and needs no sort. The keys are 4 bytes a row, which is what bounds b
 * (KVA_MASK_MAX_ROWS, kva.h). Rows outside W are written 0 by every thread in a stride.
 */
#include <hip/hip_runtime.h>

#include "kva.h"

/* How many window rows rank before row i, counting no further than `limit` (a row at or past it
 * is out). */
__device__ static int class_ahead(const float* key, int w, int i, int limit) {
    const float ki = key[i];
    int ahead = 0;
    for (int j = 0; j < w && ahead < limit; ++j) ahead += key[j] > ki || (key[j] == ki && j < i);
    return ahead;
}

__device__ static int hash_ahead(const uint32_t* key, int w, int i, int limit) {
    const uint32_t ki = key[i];
    int ahead = 0;
    for (int j = 0; j < w && ahead < limit; ++j) ahead += key[j] < ki || (key[j] == ki && j < i);
    return ahead;
}

/* The sum of every thread's count (read after a barrier; the caller syncs before rewriting). */
__device__ static int workgroup_total(const int* count) {
    int total = 0;
    for (int i = 0; i < KVA_MASK_THREADS; ++i) total += count[i];
    return total;
}

__global__ void __launch_bounds__(KVA_MASK_THREADS) kva_mask_kernel(KvaMask g) {
    __shared__ union { float score[KVA_MASK_MAX_ROWS]; uint32_t hash[KVA_MASK_MAX_ROWS]; } key;
    __shared__ int count[KVA_MASK_THREADS];
    const int t = threadIdx.x;
    int64_t win[3];
    kva_mask_window(g.cu_last[0], g.cu_last[1], g.b, g.n, win);   /* false: empty, all exact */
    const int64_t s = kva_mask_from(g.mode, win), end = win[1];
    if (t == 0) {
        g.bounds[0] = (int32_t)win[0]; g.bounds[1] = (int32_t)end;
        g.bounds[2] = (int32_t)end; g.bounds[3] = (int32_t)win[2];
    }
    for (int64_t i = t; i < g.n; i += KVA_MASK_THREADS)
        if (i < s || i >= end) g.mask[i] = 0;
    for (int64_t i = t; i < g.n_zeros; i += KVA_MASK_THREADS) g.zeros[i] = 0;

    const int w = (int)(end - s);
    const int seg = (w + KVA_MASK_THREADS - 1) / KVA_MASK_THREADS;
    const int lo = min(w, t * seg), hi = min(w, lo + seg);
    const bool ranked = g.mode == KVA_MODE_CLASS || g.mode == KVA_MODE_RANDOM;
    int matches = 0;
    for (int j = lo; j < hi && ranked; ++j) {
        const float sc = kva_mask_score(&g, s + j);
        matches += sc > -INFINITY ? 1 : 0;
        if (g.mode == KVA_MODE_CLASS) key.score[j] = sc;
        if (g.mode == KVA_MODE_RANDOM)
            key.hash[j] = kva_row_hash(g.seed, (uint32_t)g.positions[(s + j) * g.pos_stride]);
    }
    count[t] = matches;
    __syncthreads();   /* keys and counts visible to every thread */
    const int k = (int)rint(g.share * (double)workgroup_total(count));
    for (int j = lo; j < hi; ++j) {
        bool kept = g.mode == KVA_MODE_ALL;
        if (g.mode == KVA_MODE_CLASS)
            kept = key.score[j] > -INFINITY && class_ahead(key.score, w, j, k) < k;
        if (g.mode == KVA_MODE_RANDOM) kept = hash_ahead(key.hash, w, j, k) < k;
        g.mask[s + j] = kept ? 0 : 1;
    }
}

extern "C" int kva_mask_device(const RadArgs* a, RadStream s) {
    KvaMask g{};
    const int rc = kva_mask_parse(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.b > KVA_MASK_MAX_ROWS) return RAD_E_SHAPE;   /* the window (<= b rows) must fit LDS */
    kva_mask_kernel<<<1, KVA_MASK_THREADS, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
