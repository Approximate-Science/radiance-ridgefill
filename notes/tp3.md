# notes/tp3.md -- RidgeFill at three ranks (radiance 1.1.1's any-world-size serve)

**Status: built, host-tested and kernel-tested; NOT run end to end.** A real run needs three GPUs of one
architecture (radiance core/engine_bringup.cpp refuses `--tp N` with fewer devices); the development machine has
two R9700s. Commits: the plugin (4e7c7d1, was 048a7f1), the kernel tests (cc0dadc, was ec415d3).

## What radiance 1.1.1 does at three ranks (data/radiance-src-1.1.1/docs/TP3.md)

- **Attention world.** Two KV heads over three ranks: attention is served on the largest multiple of the KV head
  count under the world (2); rank 2 is **attention-zero** -- its attention block is one `fill` of zeros on the
  output wire, no projections, no KV cache (`rad_block_attn_gated_fp8.h` 258-265, 482-489).
- **Uneven delta-net heads.** 16 K heads split 6 / 5 / 5, the extra first to the rank without attention
  (`qwen4exp_fp8.cpp` 412-443, `remainder_place`); V = 3 x K, so ranks 0 / 1 / 2 hold V heads [18, 33), [33, 48),
  [0, 18) -- not `rank * 16`.
- **MoE** keeps the two-parity tables at three ranks (whole 128-column blocks by rotation).

## What the plugin does (048a7f1)

- The adapter's `StateShape` gains `first` / `n_head_all` from the in-tree split (`v_first` when uneven): the
  projector's per-head correction slice, the folder check and the state dump take the rank's real heads.
- The attention-zero rank: the approximate path's attention rows take the in-tree `fill` (no KV write), and
  `straddle_missing` answers "nothing lacking" there, so every rank takes the same path and issues the same
  all-reduce sizes (otherwise rank 2 alone would take the masked path and the reduce would hang).
- The projector itself is not sharded (every rank holds the whole map, arch_static's
  `every_rank_at_every_tp_declares_the_full_width_int8_projector`); nothing else changes.

## Tests

| test | result |
|---|---|
| static oracle (arch_static): TP3 in the raw-operand, int8 and straddle cases; masked path on every rank at its own uneven heads vs the in-tree A_log span; same path and all-reduce sizes on every rank for masked / lean / straddle / decoders | 71 cases / 1,160,080 checks; four mutants (each fix undone) caught |
| kernel_test host: each rank's correction, state read and decay sums at [18,33) [33,48) [0,18) == the slices of TP1's 48-head run | 0 elements differ; mutants caught (a table gap: FAIL; comparing one head over: 4.7M differ) |
| kernel_test device legs at 18 and 15 heads (device vs host) on one gfx1201 (0000:13:00.0) | **PASS**: state_correct / state_read 0 bytes differ at 48/24/18/15/12 heads; rho max diff 5.4e-7; 1,798 checks (evidence/tpx/kernels-gpu-ec415d3.log) |
| ctest -LE gpu, pytest (on main e45dd03) | 5/5; 228 passed, 33 skipped |

## To run it when three cards exist

```sh
radiance --model <stock .rad> --tp 3 --tp-wire exact ... (radiance docs/TP3.md's row) with RADIANCE_HOME=<plugin>:<engine share>
```
Gate it as TP2 was: `off` ident == stock ident at TP3, a TP3 exact KL reference (TP3 sums partials in another
order than TP2), int8 quality T2560 scored against it, then stock-path and decode ABCA.
