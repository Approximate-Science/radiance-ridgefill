/* kva_plan.h -- WHAT AN APPROXIMATE PASS IS, decided from numbers alone (model-agnostic).
 *
 * A pure function: the keyed batch fields the decision may read (radiance core/runtime/ctx.cpp:989-995)
 * and the declare-time configuration, in; the pass plan out. No model type, no device data, no
 * host state that a recorded pass would not replay (R15/R99). A model adapter fills PlanIn from its
 * RadBatch and its declare (qwen4exp_kva.cpp derive()); tests/arch_static_test.cpp holds it to a
 * hand-computed truth table (R52').
 *
 * THE PATHS, cheapest first, and the rule that picks one (PLAN-FIX §2, A.1):
 *   lean      speed, one prefill sequence, no decoders, the whole chunk bulk: only the late blocks'
 *             cache-writing pieces run.
 *   straddle  speed, one prefill sequence, no decoders, the chunk straddles the bulk end b: the bulk
 *             rows [0, b) get the lean pieces and the full late blocks run over the tail rows [b, n)
 *             only (the adapter needs every late attention layer on its per-row sparse form).
 *   decoders  speed, one prefill sequence whose chunk is all bulk, decoders beside it: the bulk rows get
 *             the lean pieces and the full late blocks run over the decoder rows [0, DT) only --
 *             their dense GEMMs at M = DT, the shape a decode-only step gives them (ident.sh's
 *             ksplit-from-M class; Dylan, 2026-10-05: byte-identity to off not required here). Needs
 *             the per-row sparse attention (straddle_ok) and the stager lever (the decoders'
 *             experts stream).
 *   masked    any other shape: every late block over all rows, the device mask choosing which rows use
 *             the projection. Taken only while the pass can STREAM its late experts (few exact rows,
 *             the stager lever on): measured on this deployment, a masked pass that has to stage its
 *             late layers costs more than the exact step (Stage A session 3: quality with staging
 *             0.86x), and a pass with many exact rows streams every expert over the link (A.1
 *             profile: a 1,024-row tail's late MoE 453 ms against 86 ms staged). So:
 *   stock     whatever the mode, when no cheaper plan exists -- an opted-in request is never served
 *             slower than stock, and never less exact than its mode promises. That includes a pass
 *             with fewer bulk rows than `min_bulk_rows`: an approximate pass has a fixed cost (with
 *             a host-placed projector, streaming every late layer's map), which a 64-row checkpoint
 *             remainder cannot repay -- and the decoders riding that step pay it too (Stage B).
 * plumb (the oracle mode) always takes the masked path.
 */
#ifndef QWEN4EXP_KVA_PLAN_H
#define QWEN4EXP_KVA_PLAN_H

#include <algorithm>
#include <cstdint>

namespace kva {

using namespace rad::arch;

enum Path { PATH_STOCK = 0, PATH_LEAN, PATH_MASKED, PATH_STRADDLE, PATH_DECODERS };
static const char* const kPathNames[] = { "stock", "lean", "masked", "straddle", "decoders" };
enum PlanMode { PLAN_OFF = 0, PLAN_PLUMB, PLAN_SPEED, PLAN_QUALITY };

/* The keyed numbers of one pass, and whether the adapter can serve each shape. */
struct PlanIn {
    int     mode = PLAN_OFF;
    bool    eligible = false;        /* projector held, not KL-without-SCORE_BULK, not media/enc/draft */
    int64_t n_tok = 0, n_seq = 0, n_seq_decode = 0, n_tok_decode = 0;
    int64_t q_prefill = 0;           /* the longest prefill query of the pass */
    int64_t n_ahead = 0;
    int64_t n_checkpoints = 0;       /* checkpoints this pass writes (keyed) */
    bool    stream_ok = false;       /* the stager lever is on and has the layers it needs */
    bool    straddle_ok = false;     /* every late attention layer takes its per-row sparse form */
};

/* Declare-time numbers (kva_config.h). */
struct PlanConfig {
    int64_t tail = 2048;             /* T */
    int64_t tile = 64;               /* G: the delta net's chunk */
    int64_t force_split = 0, shift_b = 0;
    int64_t stage_rows = INT64_MAX;  /* exact rows a masked pass may carry and still stream: any */
    int64_t ckpt_floor = 0;          /* T_ck: the last rows of a checkpoint-writing chunk kept exact */
    int64_t min_bulk_rows = 0;       /* fewer bulk rows than this: the stock step (a pass's fixed
                                      * cost, e.g. a host-placed projector's stream, outweighs them) */
    bool    force_stream = false;
    bool    mask_step = false;       /* debug: every row before b approximated (R54's control) */
};

/* What step() does with the pass. */
struct Pass {
    int     path   = PATH_STOCK;
    int64_t b      = 0;       /* the bulk END, an absolute row of the step */
    int64_t s_lb   = 0;       /* a host LOWER bound of the last sequence's first row */
    bool    split  = false;   /* b < n_tok: the last sequence's chunk straddles the bulk end */
    bool    stream = false;   /* the late layers stream their routed experts (notes/impl.md §2) */
};

/* The bulk end (PLAN-FIX §2): rows followed by at least T' tokens, T rounded up to the tile; a capped
 * n_ahead proves only itself (R100). b - s = 0 mod G because a non-final chunk starts and ends on the
 * scheduler's quantum, a multiple of G. */
inline int64_t bulk_end(const PlanIn& in, const PlanConfig& c) {
    int64_t b = in.n_ahead >= c.tail ? in.n_tok
                                     : in.n_tok - (c.tail - in.n_ahead + c.tile - 1) / c.tile * c.tile;
    /* DD-A's floor: a chunk that writes a checkpoint keeps its last T_ck rows exact, so a branch that
     * resumes there has at least T_ck exact rows before it (PLAN-FIX §5.2). */
    if (in.n_checkpoints > 0 && c.ckpt_floor > 0)
        b = std::min(b, in.n_tok - (c.ckpt_floor + c.tile - 1) / c.tile * c.tile);
    if (c.force_split) b = in.n_tok - c.force_split;
    return std::min(std::max<int64_t>(b + c.shift_b, 0), in.n_tok);
}

inline Pass plan_pass(const PlanIn& in, const PlanConfig& c) {
    Pass p;
    const int64_t Pn = in.n_seq - in.n_seq_decode;
    if (in.mode == PLAN_OFF || !in.eligible || in.n_ahead <= 0 || Pn <= 0) return p;
    const int64_t b = bulk_end(in, c);
    const int64_t s_lb = std::max(in.n_tok_decode, in.n_tok - in.q_prefill);
    if (b <= s_lb) return p;
    if (in.mode != PLAN_PLUMB && b - s_lb < c.min_bulk_rows) return p;
    p.b = b;
    p.s_lb = c.mask_step ? 0 : s_lb;   /* the projector then covers every row the mask can mark */
    p.split = b < in.n_tok;
    const bool one = in.n_seq_decode == 0 && Pn == 1;
    if (in.mode == PLAN_SPEED && one && !p.split) { p.path = PATH_LEAN; return p; }
    if (in.mode == PLAN_SPEED && one && in.straddle_ok) { p.path = PATH_STRADDLE; return p; }
    if (in.mode == PLAN_SPEED && Pn == 1 && in.n_seq_decode > 0 && !p.split && in.straddle_ok &&
        in.stream_ok && !c.mask_step) {
        p.path = PATH_DECODERS;
        p.stream = true;
        return p;
    }
    const int64_t exact_rows = in.n_tok - (b - p.s_lb);
    p.stream = in.stream_ok && (c.force_stream || exact_rows <= c.stage_rows);
    if (p.stream || in.mode == PLAN_PLUMB) p.path = PATH_MASKED;
    return p.path == PATH_MASKED ? p : Pass{};
}

}  /* namespace kva */

/* The adapter and its static test still name these qwen4exp_kva:: (the split moves them, the names
 * stay). */
namespace qwen4exp_kva { using namespace kva; }

#endif /* QWEN4EXP_KVA_PLAN_H */
