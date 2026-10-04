/* rho.hip -- the device row of the op. STAGE 1: declared, not implemented; the launch refuses with
 * RAD_E_UNSUPPORTED, which the engine reports by kernel and op name. */
#include <hip/hip_runtime.h>

#include "kva.h"

extern "C" int kva_rho_device(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }
