/* ridgefill_core.h -- the whole RidgeFill core, in the order its headers lean on each other. An adapter includes
 * this after radiance's installed arch headers (and, if it wraps an in-tree architecture, after that
 * source), then its own files; tests/adapter_core_test.cpp includes nothing else of the plugin.
 */
#ifndef RIDGEFILL_CORE_H
#define RIDGEFILL_CORE_H

#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>   /* QuantFP8 / ActFP8: the fill's quantiser and the projected codes */

#include "ridgefill_plan.h"
#include "ridgefill_declare.h"    /* + config, int8, adapter, projector, final, declare_masked */
#include "ridgefill_dump.h"
#include "ridgefill_layer.h"
#include "ridgefill_hazard.h"
#include "ridgefill_guard.h"
#include "ridgefill_step.h"

#endif /* RIDGEFILL_CORE_H */
