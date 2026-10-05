/* kva_core.h -- the whole KVA core, in the order its headers lean on each other. An adapter includes
 * this after radiance's installed arch headers (and, if it wraps an in-tree architecture, after that
 * source), then its own files; tests/adapter_core_test.cpp includes nothing else of the plugin.
 */
#ifndef KVA_CORE_H
#define KVA_CORE_H

#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>   /* QuantFP8 / ActFP8: the fill's quantiser and the projected codes */

#include "kva_plan.h"
#include "kva_declare.h"    /* + config, int8, adapter, projector, final, declare_masked */
#include "kva_dump.h"
#include "kva_layer.h"
#include "kva_hazard.h"
#include "kva_guard.h"
#include "kva_step.h"

#endif /* KVA_CORE_H */
