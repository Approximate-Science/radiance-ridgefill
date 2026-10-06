/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* hazard.hip -- ridgefill_hazard's device row: one thread runs ridgefill.h's rule on the step's last sequence
 * (a handful of loads and two stores a pass). cu_last is device memory, so an out-of-order window
 * counts nothing rather than refusing (the host row refuses it). */
#include <hip/hip_runtime.h>

#include "ridgefill.h"

__global__ void ridgefill_hazard_kernel(RidgeFillHazard g) {
    if (g.cu_last[0] < 0 || g.cu_last[0] > g.cu_last[1]) return;
    ridgefill_hazard_step(&g);
}

extern "C" int ridgefill_hazard_device(const RadArgs* a, RadStream s) {
    RidgeFillHazard g{};
    const int rc = ridgefill_hazard_parse(a, &g);
    if (rc != RAD_OK) return rc;
    ridgefill_hazard_kernel<<<1, 1, 0, (hipStream_t)s>>>(g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
