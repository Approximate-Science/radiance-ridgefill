/* hazard.hip -- kva_hazard's device row: one thread runs kva.h's rule on the step's last sequence
 * (a handful of loads and two stores a pass). cu_last is device memory, so an out-of-order window
 * counts nothing rather than refusing (the host row refuses it). */
#include <hip/hip_runtime.h>

#include "kva.h"

__global__ void kva_hazard_kernel(KvaHazard g) {
    if (g.cu_last[0] < 0 || g.cu_last[0] > g.cu_last[1]) return;
    kva_hazard_step(&g);
}

extern "C" int kva_hazard_device(const RadArgs* a, RadStream s) {
    KvaHazard g{};
    const int rc = kva_hazard_parse(a, &g);
    if (rc != RAD_OK) return rc;
    kva_hazard_kernel<<<1, 1, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
