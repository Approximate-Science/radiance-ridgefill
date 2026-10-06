# Adding a model (dense, MoE or hybrid): one adapter, no core change

The RidgeFill core (`arch/ridgefill_*.h`, namespace `ridgefill`) names no model type. It reads the model through ONE
struct, `ridgefill::RidgeFillAdapter` (arch/ridgefill_adapter.h): facts plus hooks, each field's comment its contract.
qwen4exp is the worked example: arch/qwen4exp_adapter.h, qwen4exp_blocks.h, qwen4exp_fill.h,
qwen4exp_moe.h, qwen4exp_ridgefill.cpp.

0. Read radiance docs/PLUGIN.md §11: your adapter is ONE .so claiming (arch_id, quant) and
   shadowing the in-tree stem on $RADIANCE_HOME. Copy qwen4exp_ridgefill.cpp's scaffolding
   (`RAD_ARCH_NO_EXPORTS` + the `#include` of your in-tree arch .cpp, then `#include "ridgefill_core.h"`,
   the declare/step/probe thins that call `core_declare` / `core_step`, the `RAD_ARCH_PLUGIN` exports;
   the declare calls `core_declare_first` BEFORE your in-tree declare and hands its handle to
   `core_declare` in `RidgeFill::hazard_first`, so the hazard op never sorts after your draft head)
   and arch/CMakeLists.txt's `rad_add_plugin` entry (stem = the shadowed .so's name; pass the same
   name as `shadow_so` and to `open_guard`, so a release mismatch forwards to it).
1. Write `<model>_adapter.h` with `adapter_of(model)` filling every fact from your in-tree Model
   after its declare: geometry (`n_layer`, `n_embd`, `wide`, `tile`, `split_lo`, `world`), the
   per-layer arrays (`full`, `ext_in`, `calibrated`, `routed`, `straddle_lack`, `qsa_exact_to`),
   the recurrent `state` shape ({0,0,0} = none), the MoE facts (`top_k` 0 = dense: no probes, no
   drop arm), the buffers, and your measured `min_tail` / `default_tail`.
2. Implement the hooks; a null hook is a capability the core skips: `declare_model` (the fill's
   quantiser), `declare_codes` (the projected input's code pair, mirroring your block input's),
   `decl_state_ops` (your correction ops; null with no recurrent state), and for the step `conn`
   (your connection read/write over a row window), `late_block` (your blocks on the LEAN / MASKED /
   STRADDLE / DECODERS paths -- qwen4exp_blocks.h is the pattern, verbatim copies of the in-tree
   issues) and `ffn` (your MoE with the drop arm, or your dense MLP).
3. Reuse qwen4exp_moe.h as-is if your FFN is MoeFP8; otherwise write your own lean pieces (the
   lean fill needs exactly your blocks' cache-writing ops).
4. The projector folder needs no schema code: the core checks every shape against your facts and
   takes the maps' dtype (bf16 or int8; int8 is what ships) from the manifest. Its "adapter" string
   is your `match_name`.
5. Static test: copy tests/arch_static_test.cpp's oracle pattern -- off must declare and issue
   exactly what your in-tree plugin does; add your derive truth-table rows, your projector
   refusals, and -- while you write adapter_of -- a case asserting each fact against the direct
   read it replaces (the split carried one, the_adapter_facts_match_the_model, until its step 7).
6. List what your adapter COPIES from radiance in `arch/<adapter>.copies` (one `<adapter file> <in-tree
   file>` pair a line; arch/qwen4exp.copies is the example). scripts/update_radiance.sh reads it to say,
   for each new radiance release, which of your files to port -- the static oracle says whether you must
   (and CI, .github/workflows/radiance-watch.yml, says it daily against radiance's newest release).
7. Fit the projector from exact captures (RADIANCE_RIDGEFILL=off + RADIANCE_RIDGEFILL_CAPTURE; notes/refit.md),
   then gate: `ctest -LE gpu`, off ≡ stock (scripts/ident.sh), and the KL rows (scripts/grade.sh).

The step around the late layers is the core's too (arch/ridgefill_step.h): the adapter adds prologue /
stock_layer / epilogue / stock_step (copies of its in-tree step's pieces) and, if it keeps the debug
captures, capture_step / finish_state / capture_mixed. tests/adapter_core_test.cpp is a whole
adapter in one file -- a toy dense model -- and the smallest example of the contract.
