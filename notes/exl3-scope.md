# notes/exl3-scope.md -- what it would take to run RidgeFill on the EXL3 container (scope only, no code)

Date 2026-10-13. Inputs: this branch (rebase-1.4.0, radiance 1.4.0 dc339b4), radiance-exl3 main e0f1d69
(`qwen4exp_exl3`, binaries = release dry run 5263e0a), the EXL3 container
`models-boot/radiance-exl3/qwen3.8-flash-next-exl3.rad`.

## Why the EXL3 container never gets RidgeFill today

radiance picks ONE architecture per container by `(arch_id, quant)`. The EXL3 container says `quant_method exl3`,
so the engine takes `qwen4exp_exl3` (`("qwen4exp", "exl3")`); RidgeFill's plugin claims `("qwen4exp", "")` and is
never asked. With both homes installed and `RADIANCE_RIDGEFILL=quality` set, the EXL3 server logs no RidgeFill
line at all (notes/rebase-1.4.0.md, session A).

## What the EXL3 architecture is

`architectures/qwen4exp_exl3/qwen4exp_exl3.cpp` is a copy of radiance's `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`
with ONE block swapped: the routed MoE is `MoeEXL3` (`rad_block_moe_exl3.h`, a 2,007-line copy of
`rad_block_moe_fp8.h`). Attention, delta net, HC, PLE, QSA and the MTP head are radiance's installed headers,
unchanged. So everything RidgeFill copies or calls outside the MoE (`qwen4exp_blocks.h`, `qwen4exp_fill.h`, the
HC read/write, the prologue/epilogue) applies to the EXL3 model as it is.

Two things in that file shape the work:
- `namespace rad::arch { using MoeFP8 = MoeEXL3; }` (line 149), so the installed MTP head composes EXL3 experts.
  In any translation unit that includes it, `MoeFP8` IS `MoeEXL3`: RidgeFill's `qwen4exp_moe.h` (written against
  MoeFP8's members: `w_gus`, `ncls`, the libr4d operand order) does not compile there, and one TU cannot hold
  both MoE blocks.
- The prefill stager is decided per setup (`exl3_stage.h`, `EXL3_PREFILL_STAGE=auto`): where staging does not
  pay, the plugin turns it off declaratively (a never-issued expert reader after the language model). RidgeFill's
  quality-mode lever (zero-row probes that make the stager STREAM the late layers, `probe_arm`) assumes staging
  is on.

## The projector: fitted on the baseline, unchecked on EXL3

The int8 folder (`ridgefill-projector-qwen3.8-flash-next-i8`) was fitted on radiance 1.0.8 from exact captures of
the BASELINE container (`qwen3.8-next-flash-fp8-iq4r-moe.rad`, 312 prompts / 4,044,768 tokens; notes/refit.md).
Its fingerprint names 489 weights around and after the split (attention, shared expert, delta net, HC, router)
and 3 anchors; the EXL3 container byte-copies all of them, and `rad-info -v` shows **489/489 encodings equal**.
The routed experts are NOT fingerprinted, and they are exactly what differs (24,576 late routed-expert tensors:
iq4r `affine u4 + fp8 scales /fwht128` in the baseline, EXL3 `u4 trellis + suh/svh /h128` here). So an EXL3
adapter would load this folder with **0 warnings** -- the loader cannot tell; only a gate can.

Expectation, not measured: the maps predict late-layer block inputs from the layer-24 stream, and both containers
quantise the same bf16 experts (mean KL vs bf16: baseline 0.0537, EXL3 0.0467 on ref64, EXL3 model card). The
containers differ from each other by about their quantisation noise, which is well inside the projector's own
fit error (held-out mean block-input cosine 0.75, notes/refit.md §4). There is a precedent: the earlier map was
fitted on another engine's captures (tcc's), whose late block inputs differ from radiance's by cosine 0.91-0.98
(notes/refit.md §2). On radiance's held rows it scored 0.7501 against 0.7536 for the refit on radiance's own
captures (the folder that ships today). The correction C (the delta-net terminal-state offset) is the more sensitive piece. So the
folder probably transfers, and the gate below decides.

## Options

**A. A RidgeFill adapter for `qwen4exp_exl3` (recommended)** -- docs/ADDING-A-MODEL.md as written: one more .so,
`arch/qwen4exp_exl3_ridgefill.cpp`, `#include`s radiance-exl3's `qwen4exp_exl3.cpp` with `RAD_ARCH_NO_EXPORTS`,
builds under the stem `qwen4exp_exl3` so it shadows the EXL3 plugin on `$RADIANCE_HOME`, registers
`("qwen4exp", "exl3")`, and forwards to the EXL3 .so on a release mismatch (the guard's `shadow_so`).
The core is unchanged. Work:
1. Scaffolding + CMake: a second source input (`RADIANCE_EXL3_SRC`, its `include/exl3_abi.h`), a pin for the
   radiance-exl3 commit next to `RADIANCE_VERSION`, and a guard that also checks the EXL3 plugin it forwards to.
2. The adapter files over either model: `qwen4exp_adapter.h` / `blocks.h` / `fill.h` read
   `qwen4exp_fp8::Model`; make them a template (or a namespace alias per TU) over the Model type.
3. `qwen4exp_exl3_moe.h`: MoeEXL3::pass with the drop arm between top-k and scatter (the EXL3 block keeps the
   same router -> top-k -> scatter split, so the drop goes where it goes today), the three EXL3 gate_up forms
   (prefill passes take the plain `gate_up(a_rot 0)` + `down`; GD / prep-split only at <= 16 rows), the 1.4.0
   prefetch calls, the calibration tap and the route report; and the probe arm re-thought for
   `EXL3_PREFILL_STAGE` (with staging off there is nothing to steer: dropped rows simply read no pooled expert;
   with staging on, a zero-row probe on the EXL3 gate_up op with its own operand positions).
4. Static oracle for the new adapter: `off` issues exactly what `qwen4exp_exl3` issues, every approximate path
   op for op; fixtures with EXL3 expert tensors; derive truth-table rows; an `arch/qwen4exp_exl3.copies` whose
   sources live in radiance-exl3 (update_radiance.sh reads only radiance today, so it needs a second repo).
5. GPU gates on the EXL3 container (the quality gate below), plus prefill speed off / quality / speed at 16K and
   32K, and the stock-path cost (<= ~1% on prompts it does not approximate, as on the baseline).

**B. The EXL3 MoE inside RidgeFill's qwen4exp adapter (not recommended).** It still needs a second .so (one
`RAD_ARCH_PLUGIN` claims one key, and the fp8 and EXL3 MoE cannot share a TU because of the alias above), and it
would carry its own copy of `MoeEXL3` and depend on exl3r4d's ops: a fork of radiance-exl3 to keep in step on
every EXL3 change. It costs A's work plus that copy, then every EXL3 change twice.

## The quality gate

The same protocol RidgeFill passed on the baseline (scripts/grade.sh, R100), on the EXL3 container against its OWN
exact reference:
- `off` == the EXL3 plugin alone: the six ident hashes (temp 0 / 0.7 / 1.0 x two questions) equal;
- an exact KL reference recorded on the EXL3 container (quick9 corpus, measurement flags);
- int8 quality at T 2,560 and speed at T 2,048: 67 approximate steps each; dNLL vs exact on the last 512 tokens
  of the 9 docs, paired bootstrap 95% CI. Pass = no worse than the baseline's own result with the same folder
  (quality +0.0010 [-0.0135, +0.0153], speed +0.0242 [+0.0040, +0.0446]): the quality CI includes 0 and the
  speed point sits inside the baseline's interval;
- needle 72/72 for both modes; the package's fresh-engine e2e cases.
Fail = refit on the EXL3 container: the same capture (312 prompts, ~48 GPU min + 14 state prompts ~1.5 min),
solve, int8 conversion, package, then the gate again.

## Effort (hours of focused work; GPU time inside)

| piece | hours |
|---|---|
| 1 scaffolding, CMake, second pin, guard | 4-6 |
| 2 adapter files over either Model | 3-5 |
| 3 EXL3 MoE copy with drop arm, prefetch, staging/probe decision | 8-12 |
| 4 static oracle + fixtures + .copies + update script for a second repo | 8-12 |
| 5 GPU gates (ident, reference, KL rows, needle, e2e, prefill speed, stock-path) + write-up | 6-8 |
| **total with the existing projector** | **29-43** |
| refit if the gate fails (capture ~1 GPU h, solve, int8, package, re-gate) | +6-10 |

Option B: A plus a maintained MoeEXL3 copy, roughly +6-10 h now and a second port on every EXL3 change.
