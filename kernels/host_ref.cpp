/* host_ref.cpp -- the host rows of the kva ops: the oracles the device rows are tested against.
 *
 * STAGE 1: declared, not implemented. Every launch refuses with RAD_E_UNSUPPORTED, which the
 * engine reports by kernel and op name (core/runtime/issue.cpp abort_step) -- nothing
 * unimplemented returns success.
 */
#include "kva.h"

extern "C" int kva_rowsel_host(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }
extern "C" int kva_rho_host(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }
extern "C" int kva_correct_host(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }
