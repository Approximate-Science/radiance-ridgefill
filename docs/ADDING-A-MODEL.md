# Adding a model (dense, MoE or hybrid): one adapter, no core change

The KVA core (`arch/kva_*.h`, namespace `kva`) names no model type. It reads the model through ONE
struct, `kva::KvaAdapter` (arch/kva_adapter.h): facts plus hooks, each field's comment its contract.
qwen4exp is the worked example: arch/qwen4exp_adapter.h, qwen4exp_blocks.h, qwen4exp_fill.h,
qwen4exp_moe.h, qwen4exp_kva.cpp.

0. Read radiance docs/PLUGIN.md §11: your adapter is ONE .so claiming (arch_id, quant) and
   shadowing the in-tree stem on $RADIANCE_HOME. Copy qwen4exp_kva.cpp's scaffolding
   (`RAD_ARCH_NO_EXPORTS` + the `#include` of your in-tree arch .cpp, the `RAD_ARCH_PLUGIN` exports)
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
   refusals, and a facts-vs-model case (`the_adapter_facts_match_the_model`).
6. Fit the projector from exact captures (RADIANCE_KVA=off + RADIANCE_KVA_CAPTURE; notes/refit.md),
   then gate: `ctest -LE gpu`, off ≡ stock (scripts/ident.sh), and the KL rows (scripts/grade.sh).

Status (2026-10-05): the split is in progress on branch `split` (notes/split.md). The step-side
hooks for the exact layers and the debug captures (prologue / stock_layer / epilogue / stock_step /
capture_*) arrive with kva_step.h; until then derive/step live in qwen4exp_kva.cpp.
