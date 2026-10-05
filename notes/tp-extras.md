# notes/tp-extras.md -- TP4's kernel extents and a one-card TP1 run (account-B worker, 2026-10-05)

Branch `tp-extras` from main (rebased onto a4d8f16, release 0.1.0): 7dff421 (tests), 7d32b02 (scripts).

## What TP changes for the plugin

- The correction, the state read and the decay sums are per VALUE HEAD, and TP hands rank r the contiguous
  heads [r*48/W, (r+1)*48/W): 48 a rank at TP1, 24 at TP2, 12 at TP4 (the folder's st.L is sliced the same
  way, kva_projector.h plan_rank).
- The int8 projector GEMM is NOT sharded: every rank holds the whole map and computes the whole block input,
  N = n_embd 2560 from K = hc*n 10240, at any TP -- TP4's per-rank GEMM is TP1's.
- kva_hazard has no head extent.

## Host (7dff421; ctest -LE gpu 5/5, arch_static 67 / 1,144,174, pytest 209 / 33)

- kernel_test `each_ranks_heads_compute_their_slice_of_tp1` (host): the host rows at TP2's and TP4's
  extents equal, element for element, the matching head slice of the 48-head computation -- state,
  applied scale, ND, the read, the decay sums: 0 differ at both widths. Negative control (each rank vs the
  slice one head over): 4.7M diffs, FAIL.
- kernel_test's device-vs-host cases (state_correct, state_read, rho) loop over 48, 24, 12 heads.
- arch_static `every_rank_at_every_tp_declares_the_full_width_int8_projector`: every rank of TP1/2/4 declares
  kva_gemm_nt_q N 2560 K 10240 on each late layer, the quantiser over 10240, and the correction at
  n_head 48 / world on the three late delta-net layers.

## Scripts (7d32b02)

- `scripts/card_index.sh [pci]`: ROCR_VISIBLE_DEVICES index from rocminfo's GPU agents (BDFID): 03:00.0 -> 0,
  13:00.0 -> 1 on this box.
- `scripts/kernels_gpu_check.sh <commit> [pci]`: frozen_home build (no GPU), then one gpuq session running
  that build's `ctest -L gpu` with one card visible; requires "device 0 of 1 visible, PCI <pci>".
- `scripts/tp1_check.sh [pci]` (under gpuq, RK_PLUGIN_HOME set): stock --tp 1 startup = the fit check (stop,
  exit 3, printing the engine's budget and reason if refused); TP1 R3; off ident == it; a TP1 exact KL
  reference recorded in the session; int8 quality T2560 last-512 paired vs it.

Predictions: labbook seq 475 (HTP-kernels-extents), 476 (HTP1-fit-and-off), 477 (HTP1-quality).

## Results (boot 75e3e39b; frozen home 7d32b02 built in radiance-build:1.0.13, host suite green in the image)

### Kernel device legs -- PASS (gpuq kernels-gpu-7d32b02, 22:45:30Z; evidence/tpx/kernels-gpu-7d32b02.log)
"device 0 of 1 visible, PCI 0000:13:00.0". state_correct device == host BITWISE at 48, 24 and 12 heads (shared and
own slots; undo / apply at alpha 1, 0.7, 0; ND on and off: 20.1 / 10.0 / 5.0 MB of state, 0 bytes differ);
state_read bitwise at all three (out-of-pool rows zero); rho max |diff| 5.36e-07 (bound 1e-5); the int8 projector
GEMM at N 2560 K 10240 (the per-rank shape at every TP): both forwarded i8a8 rows within 0.801 of the 2-ulp bound,
0 outputs over; gemm_nt_bias forward bytewise; hazard, mask, select, drop: ok. 1,554 checks, ctest -L gpu 100%.
Labbook HTP-kernels-extents CONFIRMED (records/verdict after seq 477).
Ordering: this ran BEFORE Stage E's release session, not after it as asked -- E aborted its 0.2.0 release waiter
at 22:45:41Z and re-queued it at the same queue time as 0.1.0 (22:45:46Z); in that gap this ticket was the oldest
and took the GPUs at 22:45:30Z for under a minute. gpu.lock was free (E's job was waiting, not running): nothing
overlapped.

### TP1 on one card -- DOES NOT FIT with the profile's memory flags (gpuq tp1-7d32b02, 23:30:12Z;
### evidence/tpx/tp1-20261005T233012Z)
A first attempt at 23:29:45Z stopped itself before starting anything: the release session's last container
(radiance-kva-speed) was still being torn down as the queue moved on. Re-queued at 23:30:12Z.
Stock `exact`, RK_FLAGS with only --tp 1 (expert_tiered, --host-pool-mib 12288, --gpu-headroom-mib 3072,
--expert-vs-cache-ratio 0.82, fp8 KV, 49152 context), on 13:00.0 alone ("AQL backend, 1 device(s) [0:gfx1201]").
The engine's own budget: claimable 27.59 GiB; static weights 6.33 GiB (TP2: 4.03 a card; the token embedding
1.18 GiB whole); elastic 21.26 -> experts 15.74 GiB (6,074 slots of 2.42 MiB) + KV 3.83 GiB (44% of the 8 x 49152
worst case) + prefill staging 1.69 GiB; host pinned pool 12 GiB full; the n-gram table 47.68 GiB in pinned RAM.
**"DID NOT FIT: 13411 x blk.*.ffn_gate_up_exps.*.weight (31.65 GiB): the host pool is full and no
--weights-disk-tier was given."** At TP2 the two cards' slabs plus the host pool hold every expert (each card a
half-width slice); at TP1 one card's slab plus 12 GiB of host pool leave 31.65 GiB of experts with no tier. So
the TP1 check stopped there by design (exit 3): no TP1 R3, no off ident, no TP1 reference, no quality run.
Labbook HTP1-fit-and-off REFUTED (the fit half); HTP1-quality INCONCLUSIVE (not run).
What would make it start (not tried -- outside the profile's memory flags, Dylan's call): --host-pool-mib of
about 44 GiB (12 + 31.65; with the 47.68 GiB n-gram copy that is ~92 GiB pinned of 123 GiB RAM), or a
--weights-disk-tier for the remainder (experts then read from SSD on every miss: a slow TP1).
