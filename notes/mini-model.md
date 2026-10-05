# notes/mini-model.md -- the no-GPU correctness run: the mini model, and the wall it hits

Status: **steps 1-2 built and green; CPU serving (step 3) is blocked in STOCK radiance 1.0.8**, not
in this plugin, by nine ops the qwen4exp architecture declares that only libr4d -- the DEVICE
kernel library, which a host-only build does not even produce -- has rows for. The task's
contingency ("if an op has no host row so CPU serving fails, report exactly which op and stop
there") is what happened. `scripts/mini_cpu.sh` is the one-command run: it prints PASS/FAIL per
check with the evidence line, and stops at the wall.

## 1. What was built (radiance 1.0.8, host-only, no GPU anywhere)

| thing | where (job folder, git-ignored) | time |
|---|---|---|
| radiance host-only (`RAD_WITH_HIP=OFF`, `-DRAD_WITH_FFMPEG=OFF`, Release, install prefix inside the tree) | `<job>/build-radiance-host/` | ~4 min (12 cores) |
| this repo's plugins host-only (`-DRADIANCE_SRC=/var/home/dylan/projects/inference/radiance`) | `<job>/build-host/` (`radiance_home/{architectures/qwen4exp_fp8.so,kernels/kva.so}`) | ~2.5 min |
| `ctest -LE gpu` on the plugin build | 3/3 passed (`arch_static_test`, `kernel_test`, `rho_ref`) | 1.8 s |
| the Python tier (`pytest tests -q`, needs no build) | 201 passed, 33 skipped (machine-local data skips, each names its env var) | 12 s |
| mini checkpoint (`tools/mini_model.py`) | `<job>/mini/ckpt/` | 4.5 s |
| `rad-convert --plan-only` (namespace gate) | fails only on the op wall below | ~3 s |
| serving (`--debug-accept-reference-kernels --tp 1 --max-num-batched-tokens 512 --max-model-len 8192`) | exits 1 at declare, both homes | ~2 s |

`kva.so` itself is host-clean: `rad-info --plugins` lists it "7 kernel(s), 7 schema(s), built for
host", and `kva_gemm_nt_bias` forwards **libref's HOST row** when libr4d is absent
(kernels/forward.cpp), so the plugin's own kernel side is not the blocker. The blocker is the
in-tree architecture it must include.

## 2. The mini model (`tools/mini_model.py`): exact dimensions

An HF-format checkpoint with the published container's source config (the `Qwen/Qwen3.8-Flash-Next`
config that radiance-kva's data/stub records) scaled down, and **the real tokenizer files**
(tokenizer.json: 248077 tokens / 247587 merges; marker ids 248044-248057 all exist).

| kept (the architecture's identity) | value |
|---|---|
| per-head dims | 24 query + 2 KV heads of 256 (partial rotary 64); 16 key / 48 value delta-net heads of 128, conv width 4, chunk tile 64; 4 indexer heads of 128 |
| layer pattern | 3 x GatedDeltaNet + 1 x gated attention, `full_attention_interval` 4 |
| gated residual | `hc_count` 4, `hc_lowrank` 320 |
| PLE | `ngram` 3, `heads_per_ngram` 8 -> 16 hash heads of 160 (`ple_embed_dim` 2560), 128 shards, `ple_layer_ids` [2] -> layer 1 |
| MTP head | one full-attention layer + routed experts, `mtp_num_hidden_layers` 1 |
| vocab | 248320 (`vocab_size`), real tokenizer, chat template |
| `moe_intermediate_size` | 640 (per-expert width kept whole: 5 x 128-col blocks, so the recipe's fwht128/group-64 grids fit) |

| cut | real -> mini | why / what failed |
|---|---|---|
| hidden | 2560 -> **256** | the smallest multiple of the recipe's 128-column scale blocks; 128 was not tried once 256 met every constraint, and the plan gate passed 256 (no tensor/shape complaints at all) |
| layers | 48 -> **8** | two full 4-groups; PLE at layer 1, full attention at 3 and 7 |
| experts / top-k | 512/10 -> **16/4** | 16 is the smallest the code accepts without any expert-parallel constraint at tp 1; top-k 4 ≤ experts |
| PLE n-gram table | 20e6 base -> **1000** (`ngram_vocab_size_base`) | the real table is 51B rows; the mini's is 16826 rows (16 primes over 999, padded to 16896, 128 shards of 132), per-head geometry untouched |
| vision tower | dropped | the plugin's paths never touch it; serving is text-only |
| weights | seeded random, std = 1/sqrt(fan-in) | norm gains zero (the reference's `x_hat * (1 + w)` init), delta-net gated output norm ones, `A_log` = log(0.05), `dt_bias` 0 -> activations stay finite |

`model.safetensors` is 0.507 GiB bf16, 365 tensors -- the same namespace as the real checkpoint
(read off the stub's safetensors headers), verified shape by shape by `rad-convert --plan-only`
walking every name map before the op wall. Container size: **not reached** (the convert dies at
declare); projected ≤ 0.6 GiB (the only sizeable planes are the 248320-row embed + lm_head).

The quantisation: `scripts/mini-qwen4exp.recipe` is the published container's recipe
(`qwen4exp-w4nl64-i8-hc8m.recipe`, per /var/home/dylan/models/rad/README.md) with ONE deviation --
the routed experts quantise by `rtn`, not `gptq`, because the published rule needs `$CALIB` (ten
million tokens of Grams recorded from the real model) and a random-weights mini model has none.
The grid is kept rule-for-rule (w4nl codebook, fwht128 rotation, E4M3 scale per 64 under the fixed
2^-13), so the ENCODING the container would hold -- what `rad-info -v` prints and the projector
folder's `encodings` fingerprint compares -- is the same w4nl64a8h as the published container's.

## 3. What the mini run CAN and CANNOT test

CAN and does (no GPU): the host-only builds of both trees; `ctest -LE gpu` (the graph/issue
oracle against the in-tree plugin, the kva.so host rows against the fixture, rho's numpy oracle);
the whole Python tier; the plugin's own host kernel path (`kva.so` built for host, its
`kva_gemm_nt_bias` forwarding libref's host row); the checkpoint/recipe namespace gates.

CANNOT (blocked before any of it runs): live CPU serving of ANY qwen4exp model -- stock home or
plugin home, `RADIANCE_KVA` off/speed/quality, `--tp 1` or `--tp 2` (the same declare runs, so
host-only tp 2 is untestable and would not pass either); the mini `.rad` container (rad-convert
dies at the same declare); the projector folder and its loader (`tools/dev/mini_projector.py` --
unwritable without a container to fingerprint, and untestable without serving); every
determinism/byte-identical check of step 3-4. Timing and real-model quality were never the mini
run's to test; correctness-by-live-serving is the loss.

## 4. The wall: the exact ops, and why each is there

`rad_arch_declare` fails host-only because `Registry::validate` returns `RAD_E_NOSCHEMA` for an op
no loaded plugin declares (core/plugin/schema.cpp:106), the builder records it as a declare error
(core/build/rad_builder.cpp:630), and `rad_arch_declare returned invalid argument` follows. With
the bf16 checkpoint (no recipe) the engine names seven ops; with the recipe's int8 trunk +
rotated experts, `rad-convert --plan-only` additionally dies at layer 0 on two more:

| op | declared by | host-side status |
|---|---|---|
| `router_topk_scatter` | every MoeFP8 layer, unconditionally (rad_block_moe_fp8.h:729) | the unfused pair `router_topk` + `moe_scatter` EXISTS in libavx/libref, but the arch declares the fused op unconditionally -- "0 leaves the pair" only works on a builder where an unresolved op returns null, and the real builder makes NOSCHEMA fatal |
| `qk_norm_rope_gate` | every gated-attention layer once a rope table exists (rad_block_attn_gated_fp8.h:273) | same story: the fused prologue probe; the unfused `rmsnorm` + `rope` pair exists on host |
| `qsa_work`, `qsa_block_key`, `qsa_tail_store`, `qsa_score`, `qsa_select` | every full-attention layer with an indexer (rad_qsa.h; my mini keeps the real indexer config) | **no host implementation anywhere in the tree** -- the indexer chain exists only as libr4d device kernels |
| `quant_act_i8g` | the trunk's activation quantiser when the trunk linears are int8 (rad_fp8.h:234,813) | libavx/libref have `quant_act_i8` only |
| `gated_had_quant_fp8` | the hyper-connection read's rotated E4M3 twin when the experts are rotated (rad_fp8.h:572) | host has `gated_had_quant_i8` only |

So the qwen4exp architecture (the in-tree one this plugin #includes whole) was written against the
device library: five of the nine are FUSIONS whose unfused halves exist on the host, two more are
int8/E4M3 twins of ops the host has, and the QSA indexer chain has no host form at all. GUIDE.md
§1.1's "host backend is a real implementation" holds for the core, scheduler, tokeniser and server,
and radiance's own host tests pass -- but they use stub builders (tests/arch_test.cpp) that never
resolve ops against the kernel registry, which is why this never surfaced.

## 5. What would unblock it (for Dylan's decision; NOT done here)

1. Minimal, in radiance: make the three unconditional fused declares (`router_topk_scatter`,
   `qk_norm_rope_gate`, and the two quant twins) tolerate absence on the real builder -- either a
   "probe first, declare second" (ask the registry before `RAD_OP`, as the comments already imply)
   or host rows for the fusions composed of the existing host halves. That alone gets a NO-INDEXER
   mini (indexer_n_heads 0 -> dense attention, which the arch supports) serving on CPU; the
   plugin's `speed`/`quality` masked and lean-fill paths would then be exercisable end to end.
2. The QSA chain needs real host rows (`qsa_*`) or a no-indexer config accepted for the mini --
   which loses the straddle path's "per-row sparse gated form" but keeps masked/quality.
3. Then: `tools/dev/mini_projector.py` (the manifest/fingerprint code is a direct port of
   tools/kva_projector.py's, reading the mini container), split at layer 4 (half of 8, a group
   boundary, past PLE's layer 1), and the step-3/4 checks as specified.

Nothing in this repo was changed to work around the wall: the plugin, the kernel library, the
tests and the tools are untouched except for the new `tools/mini_model.py`,
`scripts/mini-qwen4exp.recipe`, `scripts/mini_cpu.sh` and this note.
## GPU-side conversion attempts (orchestrator, 2026-10-05) -- PARKED

Dylan parked the mini model with per-request ON/OFF ("stop investing"); the tooling stays. What the two
GPU-side `rad-convert` runs (runtime image `stilldeadcode/radiance:1.0.8`, both cards visible) showed:
1. The checkpoint kept the real `vision_config`, so radiance declared a vision tower: "the vision tower
   emits 2560-wide rows and the model embeds 256-wide tokens". Fixed in `tools/mini_model.py` (cccc80e).
2. With the tower gone, declare failed on every hyper-connection weight (`blk.*.attn_hc_up/down`,
   `ffn_hc_up/down`, `output_hc_*`, `mtp.*_hc_*`): "takes 2 planes ... and no op reads it" -- libr4d's
   hyper-connection ops do not resolve at hidden 256, so nothing reads those weights. A mini that serves on
   the GPUs needs the REAL hidden size (2560): ~3 GiB, mostly the 248,320 x 2560 embedding and head.
Resume: `MINI hidden = 2560` in mini_model.py, re-run `convert.sh` from `~/models/rad/test/mini-qwen4exp/`
(GPU queue: `gpuq.sh`), then write the mini projector tool (not started).
