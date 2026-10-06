/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* qwen4exp_ridgefill -- RidgeFill over radiance's in-tree Qwen4-Exp plugin.
 *
 * WHAT THIS FILE IS. The in-tree arch/qwen4exp_fp8/qwen4exp_fp8.cpp, #included whole with its
 * exports suppressed (PLAN D2, the pattern radiance's own tests/arch_test.cpp:14,281-283 uses), and
 * this plugin's exports in their place: arch id `qwen4exp`, quant "", plugin name qwen4exp_ridgefill,
 * built as qwen4exp_fp8.so so it shadows the installed file by stem (PLAN D1, arch/CMakeLists.txt).
 *
 * `off` -- the default, and every container with no ridgefill.* weights -- is the included declare and
 * step with NOTHING added: no weight, no buffer, no op, not even a name map. Byte-identical to stock
 * is the contract (R6, R7), and tests/arch_static_test.cpp checks the declared graph and the issued
 * sequence against the in-tree plugin's.
 *
 * WHAT AN APPROXIMATE PASS IS (PLAN-FIX v2). Any trunk pass whose LAST entry is a non-final prefill
 * chunk with bulk rows -- rows followed by at least T' prompt tokens (T rounded up to the delta
 * net's tile) -- whatever else rides in the step. Layers 0..S-1 run stock; from S on, each late
 * layer either takes the LEAN fill (speed mode, a one-sequence step whose whole chunk is bulk: only
 * the cache-writing pieces, qwen4exp_fill.h) or the MASKED layer (every other shape: the in-tree layer
 * over all rows with the device mask choosing which rows use the projection, ridgefill_layer.h). On a
 * masked pass the expert stager is steered, with zero-row probes of the late layers' gate-up GEMMs,
 * to stream the late layers' routed experts instead of staging each layer whole (notes/impl.md §2).
 * The engine release is checked at open and a mismatch forwards to the engine's own architecture
 * (ridgefill_guard.h).
 *
 * CORE AND ADAPTER (docs/ADDING-A-MODEL.md). All of the above that is not Qwen4-Exp's is the core
 * (arch/ridgefill_*.h, namespace ridgefill, which names no model); this file is the scaffolding, the thin
 * declare/step/probe and the exports, and qwen4exp_adapter.h / _blocks.h / _fill.h / _moe.h are
 * the model's half: its facts and the hooks that issue its blocks.
 *
 * Included by tests/arch_static_test.cpp too, which defines RAD_ARCH_NO_EXPORTS itself; then
 * neither plugin's exports are emitted and the test calls both namespaces directly.
 */
#ifndef RAD_ARCH_NO_EXPORTS
#define RAD_ARCH_NO_EXPORTS 1
#define QWEN4EXP_RIDGEFILL_EXPORTS 1
#endif
#include <qwen4exp_fp8/qwen4exp_fp8.cpp>

/* THE CORE (namespace ridgefill; it names no model: tests/core_purity.cmake) ... */
#include "ridgefill_core.h"

/* ... THEN THIS MODEL'S ADAPTER, which sees the core's names as its own, so the hooks and the static
 * test spell them qwen4exp_ridgefill::. */
namespace qwen4exp_ridgefill { using namespace ridgefill; }
#include "qwen4exp_fill.h"
#include "qwen4exp_moe.h"
#include "qwen4exp_blocks.h"
#include "qwen4exp_adapter.h"

namespace qwen4exp_ridgefill {

using namespace rad::arch;

/* The in-tree file this plugin is built to shadow (arch/CMakeLists.txt's stem): the guard forwards to
 * it on a release mismatch. */
constexpr const char* kShadowSo = "qwen4exp_fp8.so";

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (g_forward.declare) return g_forward.declare(b, meta, ctx);
    RAD_ARCH_TRY(qwen4exp_fp8::declare(b, meta, ctx));

    /* A SIZING DECLARE WRITES SCRATCH, as the included declare does. Its geometry is read from
     * g_model, which the real declare filled: the engine runs every sizing declare after it
     * (radiance core/engine_bringup.cpp:616-660), and the layer schedule and head counts read here
     * do not depend on max_tok. */
    static thread_local RidgeFill probe_ridgefill;
    RidgeFill& k = ctx->shape_probe ? probe_ridgefill : g_ridgefill[ctx->rank];
    k = RidgeFill{};
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[ctx->rank];
    if (m.layers.empty()) return RAD_E_STATE;
    k.ad = adapter_of(m);
    return core_declare(b, meta, ctx, k);
}

/* ================================================================== step */

static void step(RadCtx* c, const RadBatch* batch) {
    if (g_forward.step) { g_forward.step(c, batch); return; }
    core_step(c, batch);
}

/* The in-tree probe's answers hold here: the draft depth is the model's, and this declare writes
 * scratch under shape_probe exactly as the included one does. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (g_forward.probe) return g_forward.probe(meta, out);
    return qwen4exp_fp8::probe(meta, out);
}

}  /* namespace qwen4exp_ridgefill */

#ifdef QWEN4EXP_RIDGEFILL_EXPORTS
extern "C" int rad_plugin_open(void) {
    ridgefill::g_log_name = "qwen4exp_ridgefill";   /* before the guard's first line */
    return qwen4exp_ridgefill::open_guard(qwen4exp_ridgefill::kShadowSo);
}
extern "C" void rad_plugin_close(void) { qwen4exp_ridgefill::free_uploads(); }
RAD_ARCH_PROBE(qwen4exp_ridgefill)
RAD_ARCH_PLUGIN(qwen4exp_ridgefill, "qwen4exp", "", "0.1.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next) with RidgeFill prefill: the in-tree "
                "qwen4exp_fp8 plugin plus projected late-layer cache fill (modes off, plumb, "
                "speed, quality; RADIANCE_RIDGEFILL, default off)")
#endif
