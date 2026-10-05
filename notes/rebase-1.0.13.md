# notes/rebase-1.0.13.md -- the plugin on radiance 1.0.13 (account-B worker, 2026-10-05 from ~21:30Z)

Branch `rebase-1.0.13` from main f704bd9 (= the merged split), worktree `radiance-kva-wt-b`.
radiance 1.0.13 = commit **d0f639bd** (tag object 53282fa); 1.0.8 = 140987f. Source:
`<radiance-kva>/data/radiance-src-1.0.13` (read-only; every abi/ and arch/ file and tests/rad_test.h
byte-identical to `git show v1.0.13:<file>` in the radiance checkout -- checked, 0 differ).

## What 1.0.9-1.0.13 changed that the plugin sees (`git diff v1.0.8 v1.0.13 -- abi/ arch/`)

No abi/ change (ABI 15.0.0 at both). In arch/:

| in-tree change | release | what it means here | done |
|---|---|---|---|
| rad_block_moe_fp8.h: at four ranks the routed experts are four classes by number mod 4 (`Config::ff_lo4`, `MoeFP8::ncls`), declared as four tables and issued through the new `moe_gemm_q_mod4` (gate-up and down) | 1.0.10 | qwen4exp_moe.h copies MoeFP8::pass's GEMM issues (to put the drop and the probes in): both GEMMs and the probe must take the four-table form when `ncls == 4` | ported (87415fd) |
| rad_block_hc.h / rad_fp8.h: four-rank fused all-reduce + hc write (`ar_exact_write_takes`, `ar_exact_write_rows`, gather_taken/ar_taken through it) | 1.0.10 | none: the adapter calls `hc_mix/hc_ffn.read/write`, `gather_taken`, `ar_taken` -- the in-tree methods -- so the new decisions come with them, the same T on both sides | nothing to port |
| qwen4exp_fp8.cpp declare: four-rank shares (ff_lo4, `shared_place`), three-rank vocabulary spans (`rad_weight_shard_span` on embedding / lm_head), geom_from `arch_splits_ff` | 1.0.10, 1.0.13 | declare-side only; the plugin runs the in-tree declare first and `off` is held to it by the oracle | nothing to port |
| qwen4exp_fp8.cpp `step()` | -- | **byte-identical** between 1.0.8 and 1.0.13, so the adapter's prologue / layer / epilogue copies hold | citations moved to 1.0.13's lines (1450-1463, 1465-1485, 1487-1496) |
| rad_block_mtp_hc.h / rad_block_mtp_fp8.h: a history pass that drafts nothing (draft_pass < 0, n_draft_out 0) stops after the head's attention -- no experts, no lm_head | 1.0.13 | none in code: head passes are never approximated (derive: draft passes run stock) and go to the in-tree step; the final map (off by default) writes `b_h` on the TRUNK pass before the head reads it, unchanged | pinned by a new static case |
| rad_arch.h: `vocab_per_rank` (block-aligned ceil where the world does not divide), geom_from flag | 1.0.13 | n_vocab fact reads `m.g.n_vocab`, which is this rank's either way | nothing to port |
| rad_block_ple.h: n-gram placement (device gather) | 1.0.10/11 | comments + buffer domain wording only; the adapter calls `ple.ids`/`ple.step` | nothing to port |
| rad_block_gdn_fp8.h, rad_block_attn_gated_fp8.h, rad_qsa.h | -- | unchanged: the blocks/fill copies and their line citations hold | stamps say "1.0.8 through 1.0.13" |
| libr4d / libref rows kva.so forwards (gemm_nt_bias, gemm_nt_q i8a8, quant_act_i8g, cast, add) | -- | no diff line mentions them; kva_kernels_host passes against 1.0.13's libref | -- |

1.0.12's conv-state slot fix and the --num-speculative-tokens 0 audit fix are core-side (core/); the
plugin touches neither the conv slot's width nor the all-reduce argument struct.

## Per file (87415fd, b970e11)

- **arch/qwen4exp_moe.h** -- `moe_class_len(e, q)` (class q's table length: ceil((n_reg - q) / 4) at four
  classes, n_reg / 2 by parity) and `moe_gate_up_q` (the quantised gate-up GEMM in either form, also
  used by the probe with zero offsets); `moe_experts`' down GEMM gains the four-table branch. The drop
  is unchanged. Header citation 1.0.13 lines 1408-1584 / 1631-1664.
- **arch/qwen4exp_adapter.h, qwen4exp_blocks.h, qwen4exp_fill.h** -- citations/stamps only.
- **tests/arch_static_test.cpp** -- `probe_of` reads the GEMM's output as its LAST operand (8 by parity,
  12 at four classes). Three new cases:
  - `at_tp4_the_masked_path_issues_the_four_class_moe_with_its_drop`: every rank of four, quality and
    streaming plumb, `== masked_expected` (the in-tree step + the masked substitutions); every layer's
    gate-up and down are `moe_gemm_q_mod4`; one drop a late layer in quality; the probe present.
  - `at_tp4_a_layer_with_plain_experts_takes_unequal_class_tables`: two experts plain bf16 in every
    layer -> 510 quantised, classes 128/128/127/127; the only shape where the class lengths differ.
  - `an_mtp_history_pass_is_the_in_tree_head_whether_or_not_it_drafts`: off/speed/quality, n_draft_out
    0 and 1: the plugin's pass == the in-tree pass; lm_head and experts issued only when it drafts.
  - Negative controls (each caught): gate-up always two tables; down always two tables; every class
    n_reg / 4 (caught by the unequal-classes case only -- 512 divides by 4).
- **CMakeLists.txt** (comment: ABI 15.0.0 at both releases -- the release-string check is what tells
  them apart), **scripts/ident.sh** (provenance), **docs/release/PLUGIN-README.md** (1.0.8 -> 1.0.13).
- The build needs no source change to point at 1.0.13: CMake reads the release from RADIANCE_SRC's
  project(VERSION), requires the installed engine to carry it and every installed abi/arch header to
  be byte-identical to RADIANCE_SRC's; the guard's KVA_RADIANCE_VERSION comes from the same variable.
  This branch does NOT build against 1.0.8 (MoeFP8::ncls is 1.0.10's).

## Host build and checks

```
cmake -S <radiance-kva>/data/radiance-src-1.0.13 -B build-radiance-1.0.13-host -DRAD_WITH_HIP=OFF \
      -DRAD_WITH_FFMPEG=OFF -DRAD_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PWD/build-radiance-1.0.13-host/install      # ~12 min, -j4 nice 19
cmake -S . -B build-host-1.0.13 -DCMAKE_PREFIX_PATH=$PWD/build-radiance-1.0.13-host/install \
      -DRADIANCE_SRC=<radiance-kva>/data/radiance-src-1.0.13 -DRAD_WITH_HIP=OFF
```
Configure: "radiance 1.0.13: RADIANCE_SRC and .../bin/radiance agree"; headers identical.
ctest -LE gpu **5/5**; arch_static **66 cases / 1,144,062 checks** (63 before + 3); adapter_core 5 / 115;
purity gate green; pytest tests **209 passed / 33 skipped**.

## Not done yet (GPU; waits for Stage E's 1.0.13 build image and 1.0.13 stock baselines)

One gpuq session: frozen home of this branch built in the 1.0.13 image (RK_BUILD_IMAGE /
RK_RADIANCE_SRC = the 1.0.13 source), served on the 1.0.13 runtime image (RK_IMAGE -- scripts/common.sh
still defaults to stilldeadcode/radiance:1.0.8); off ident vs the NEW 1.0.13 R3; int8 quality + speed
KL rows vs the NEW 1.0.13 exact reference; rows will differ from 1.0.8's -- report the paired dNLL vs
exact (R100's protocol).

## The next release costs one command: scripts/update_radiance.sh + CI (orchestrator ~22:10Z, Dylan)

`scripts/update_radiance.sh <tag> [--host-only] [--ci] [--gpu-smoke]` -- source (git archive, read-only,
verified when reused), the ABI number vs the pin, a host-only radiance install + the plugin against it,
ctest -LE gpu + pytest, then (not with --host-only) the device build in `radiance-build:<release>` and
`tools/package.py` into `dist/`. Exit 0 COMPATIBLE / 1 INCOMPATIBLE / 2 INFRASTRUCTURE. Every run ends
with the WARNINGS report: each file of radiance's abi/, arch/common/, arch/qwen4exp_fp8/ changed since the
pin (`RADIANCE_VERSION`: `1.0.13 d0f639bd...`) and which adapter file copies it (`arch/qwen4exp.copies`).
A failing oracle case prints the adapter files whose copied source changed and its first difference with
OP NAMES (arch_static_test's differ_at now names the ops from the case's builder). `--gpu-smoke` writes and
queues one gpuq session (stock vs off ident on the release's runtime image, an exact KL reference for it,
int8 quality T2560 + speed T2048 last-512 vs exact). `scripts/frozen_home.sh` gained `RK_HOME_TAG` so one
commit keeps a device home per release. CI: `.github/workflows/radiance-watch.yml` (README "CI: radiance
release watch").

Proven locally (2026-10-05, this machine: cmake 4.4.2, g++ 16.1.1, Python 3.14 -- the workflow's own
toolchain is ubuntu-24.04's cmake 3.28 / g++-14 / Python 3.12, whose pinned wheels exist: numpy 2.5.3,
torch 2.11.0+cpu, transformers 5.18.0, safetensors 0.8.0, pytest 9.1.1 all publish cp312 / py3 wheels):

| run | the workflow's steps | exit | what it said |
|---|---|---|---|
| v1.0.13 | tag resolved by `git ls-remote` on codeberg (newest = v1.0.13 = d0f639bd), `git clone --filter=blob:none` (3 s), `--host-only --ci` | **0** | COMPATIBLE: ABI 15 = pin, 66 cases / 1,144,062 checks, adapter_core 5 / 115, pytest 209 / 33; WARNINGS none; report.md written. The Codeberg tag's archive == data/radiance-src-1.0.13 (tar --compare) |
| v1.0.8 | `--host-only --ci` | **1** | INCOMPATIBLE: compile errors in arch/qwen4exp_moe.h:28/41/120 (`MoeFP8` has no member `ncls` -- the four-class MoE is 1.0.10's) and the two static cases that read it; copied sources changed for qwen4exp_adapter.h, qwen4exp_moe.h; nine ::warning:: files |
| simulated release (scratch `git clone --shared` at v1.0.13 + one commit: GdnFP8::step issues its a\|b projection before its input projection; MoeFP8::pass requantises before routing) | `--host-only --ci` | **1** | INCOMPATIBLE: 12 oracle cases fail -- every approximate path; every `off` case passes (it IS the in-tree code) -- each pointing at qwen4exp_blocks.h, qwen4exp_fill.h, qwen4exp_moe.h, first difference "op 134 (gemm_nt_q) vs 135 (gemm_nt)" (the delta net's two projections, swapped); WARNINGS: rad_block_gdn_fp8.h -- copied by blocks.h, fill.h; rad_block_moe_fp8.h -- copied by moe.h |

Not provable here: the workflow on a hosted runner (no forge access from this session); its first run is
the check that the apt/pip steps install as written.
