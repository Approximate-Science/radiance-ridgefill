/* kva_adapter.h -- the ONE interface between the KVA core and a model (notes/adapter-split-spec.md §1.3).
 *
 * The core (every kva_*.h but this file's users' adapters) is compiled into each adapter's .so and
 * names no model type: what it needs of the model it reads here, filled once per rank by the
 * adapter's `adapter_of` after the in-tree declare. Facts only, in this step of the split; the hooks
 * (late_block, ffn, conn, ...) join as the code that calls them moves into the core, so no field
 * exists before something reads it.
 */
#ifndef KVA_ADAPTER_H
#define KVA_ADAPTER_H

#include <arch/rad_arch.h>

#include <cstdint>
#include <vector>

namespace kva {

struct Kva;   /* kva_declare.h: the core's per-rank declare state, which the hooks fill */

/* This rank's recurrent state per late layer, as the correction ops see it; {0,0,0} = none. */
struct StateShape { int64_t n_head = 0, sd0 = 0, sd1 = 0; };

struct KvaAdapter {
    /* identity: the refusal prefix, the projector manifest's "adapter" string, and the in-tree .so the
     * release guard forwards to (it shadows that file by stem, arch/CMakeLists.txt). */
    const char* log_name = nullptr;
    const char* match_name = nullptr;
    const char* shadow_so = nullptr;
    /* geometry. `wide` is the stream that feeds the projector (hc * n_embd on a hyper-connection
     * model); `tile` the delta net's chunk, which the bulk end is rounded to -- 1 when no recurrent
     * block makes that check vacuous; `split_lo` the lowest split whose stream has seen every
     * injected input (the PLE layer + 1). */
    int64_t n_layer = 0, n_embd = 0, n_vocab_all = 0, wide = 0;
    int64_t world = 1;       /* tensor-parallel ranks: the folder holds every rank's state heads */
    int64_t tile = 1, split_lo = 0;
    /* the stager probes ride behind the gate-up of layer S - probe_depth, so streaming needs that
     * many routed layers below the split. top_k == 0 = a dense model: no probes, no drop arm. */
    int64_t probe_depth = 0, top_k = 0, n_expert = 0, n_ff_exp = 0;
    /* the measured tail facts (KVA-FACTS §5): a fit measured at 512 rows is not trusted below it */
    int64_t min_tail = 0, default_tail = 0;
    uint32_t act_dtype = RAD_BF16;  /* the stream and block-input buffers' element type */
    const char* dtype = "bf16";     /* the same, as the op params name it (gemm, cast, add) */
    StateShape state;
    /* per layer, length n_layer: attention (not recurrent); the block takes a pre-normed input (the
     * fill hands it one, check_fill); the block runs the MoE calibration tap the core never issues;
     * the FFN is routed (probes, streaming, the drop arm). */
    std::vector<uint8_t> full, ext_in, calibrated, routed;
    /* per attention layer: what it lacks for speed's tail-only straddle (nullptr = nothing), and the
     * rows its sparse attention keeps exact (the straddle must reach past them, derive) */
    std::vector<const char*> straddle_lack;
    std::vector<int64_t> qsa_exact_to;
    /* buffers the core issues against: the stream, the block input and the code pair its producer
     * writes (whichever dtype that is), the logits. The projector maps' dtype is not a model fact:
     * it is read from the folder's manifest (bf16 or int8). */
    rad_buf buf_stream = 0, buf_x = 0, buf_x_q = 0, buf_x_s = 0, buf_logits = 0;
    rad_buf buf_route_ids = 0;   /* the routed FFN's expert ids, which kva_drop_rows rewrites; 0 = dense */
    /* hooks; nullptr = the capability is absent and the core skips it */
    int (*declare_model)(RadBuilder*, const RadBuildCtx*, Kva&) = nullptr;          /* the fill's quantiser */
    int (*declare_codes)(RadBuilder*, const RadBuildCtx*, Kva&) = nullptr;          /* Kva::xp's code pair */
    const char* (*decl_state_ops)(RadBuilder*, const RadBuildCtx*, Kva&) = nullptr; /* correction ops: the
                                                                     missing op's name, or nullptr */
};

}  // namespace kva

#endif
