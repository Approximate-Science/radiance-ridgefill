/* rowsel.hip -- kva_rowsel's device row: one workgroup ranks the whole chunk.
 *
 * Each thread owns a contiguous run of rows, so the kept rows come out in row order from one
 * prefix count over the threads. The rank of a row is counted against every other row's key in
 * LDS (O(n^2) reads at worst, cut short once a row is known to be out); at the chunk sizes the
 * engine cuts that is a few microseconds and needs no sort. The keys are 4 bytes a row, which is
 * what bounds n (KVA_ROWSEL_MAX_ROWS, kva.h).
 */
#include <hip/hip_runtime.h>

#include "kva.h"

/* How many rows rank before row i, counting no further than `limit` (a row at or past it is out). */
__device__ static int class_ahead(const float* key, int n, int i, int limit) {
    const float ki = key[i];
    int ahead = 0;
    for (int j = 0; j < n && ahead < limit; ++j) ahead += key[j] > ki || (key[j] == ki && j < i);
    return ahead;
}

__device__ static int hash_ahead(const uint32_t* key, int n, int i, int limit) {
    const uint32_t ki = key[i];
    int ahead = 0;
    for (int j = 0; j < n && ahead < limit; ++j) ahead += key[j] < ki || (key[j] == ki && j < i);
    return ahead;
}

/* Sum of count[0, upto) and of all of count, over the workgroup's per-thread counts. */
__device__ static void prefix(const int* count, int upto, int* before, int* total) {
    *before = 0;
    *total = 0;
    for (int i = 0; i < KVA_ROWSEL_THREADS; ++i) {
        *before += i < upto ? count[i] : 0;
        *total += count[i];
    }
}

__global__ void __launch_bounds__(KVA_ROWSEL_THREADS) kva_rowsel_kernel(KvaRowsel g) {
    __shared__ union { float score[KVA_ROWSEL_MAX_ROWS]; uint32_t hash[KVA_ROWSEL_MAX_ROWS]; } key;
    __shared__ int count[KVA_ROWSEL_THREADS];
    const int t = threadIdx.x, n = (int)g.n;
    const int seg = (n + KVA_ROWSEL_THREADS - 1) / KVA_ROWSEL_THREADS;
    const int lo = min(n, t * seg), hi = min(n, lo + seg);

    int matches = 0;
    for (int i = lo; i < hi; ++i) {
        const int32_t id = g.tokens[i];
        const float s = id >= 0 && id < g.vocab ? g.score[id] : -INFINITY;
        matches += isfinite(s) ? 1 : 0;
        if (g.mode == KVA_MODE_CLASS) key.score[i] = isfinite(s) ? s : -INFINITY;
        if (g.mode == KVA_MODE_RANDOM)
            key.hash[i] = kva_row_hash(g.seed, (uint32_t)g.positions[(int64_t)i * g.pos_stride]);
    }
    count[t] = matches;
    __syncthreads();
    int unused = 0, total = 0;
    prefix(count, 0, &unused, &total);
    __syncthreads();   /* `count` is rewritten below */

    const int64_t k = g.mode == KVA_MODE_ALL ? g.n : (int64_t)rint(g.share * (double)total);
    const int limit = (int)(k < g.cap ? k : g.cap);
    int kept_here = 0;
    for (int i = lo; i < hi; ++i) {
        bool kept = i < limit;
        if (g.mode == KVA_MODE_CLASS)
            kept = key.score[i] > -INFINITY && class_ahead(key.score, n, i, limit) < limit;
        if (g.mode == KVA_MODE_RANDOM) kept = hash_ahead(key.hash, n, i, limit) < limit;
        g.mask[i] = kept ? 0 : 1;
        kept_here += kept ? 1 : 0;
    }
    count[t] = kept_here;
    __syncthreads();
    int out = 0, kept_total = 0;
    prefix(count, t, &out, &kept_total);
    for (int i = lo; i < hi; ++i)
        if (g.mask[i] == 0) g.rows[out++] = i;   /* this thread's own writes, read back */
    for (int64_t i = kept_total + t; i < g.cap; i += KVA_ROWSEL_THREADS) g.rows[i] = -1;
}

extern "C" int kva_rowsel_device(const RadArgs* a, RadStream s) {
    KvaRowsel g{};
    const int rc = kva_rowsel_parse(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.n > KVA_ROWSEL_MAX_ROWS) return RAD_E_SHAPE;
    kva_rowsel_kernel<<<1, KVA_ROWSEL_THREADS, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
