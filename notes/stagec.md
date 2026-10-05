# notes/stagec.md -- Stage C: prefix cache on (HANDOVER-FIX §4, PLAN-FIX §5, R62-R68)

Worker: account-B Opus 5.5, worktree `radiance-kva-wt-b`, branch `stage-c` (from stage-b 0fda876, main
db0548e merged in at 14ccb67). radiance `140987f` (v1.0.8), read only. Deployed placement: projector in
host memory (Dylan, 15:40Z; the VRAM placement is being removed on stage-e). Decisions in force: DD-A =
(D-a) with the hazard counter on and the floor `kva.ckpt_floor` = 0 by default (Dylan decides T_ck from
R65/R66).

## 1. What was built (code + static + mutants first)

| commit | what |
|---|---|
| `23d0a40` | `RADIANCE_KVA_CKPT_FLOOR` (T_ck): a chunk that writes a checkpoint (`n_checkpoints`, keyed) keeps its last T_ck rows exact, rounded up to the tile (`kva_plan.h` bulk_end). `scripts/common.sh` `RK_CACHE_DIR`: fnserve.sh's cache flags (host tier 4 GiB, disk tier under the mounted dir) in place of `--no-prefix-cache`. `tools/turns.py`: append-only two-turn conversations (R62, R64) |
| `b47f7f7` | kva.so `kva_hazard` (schema, host row, device row in `hazard.hip`, kernel tests): the branch-hazard count of PLAN-FIX §5.4 on the step's last sequence |
| `45471f3` | arch: `kv_kva_meta` (LINEAR, one slot of two floats a sequence, bound to the first late delta-net layer so every checkpoint snapshots it), the op, a 4-byte host-mapped counter per rank, issue at the end of every pass that approximates (record) or still has tail ahead (count), rank-0 log when the counter moves (`kva_hazard.h`) |
| `2b6ac05` | the log line in `tools/hazard_rate.py`'s contract: `kva: hazard <n> positions (total <m>)` |
| `041ab13` | `tools/branch_send.py`: sends `tools/branch_corpus.py`'s triples (A, B, C) with records in hazard_rate's format |

**The hazard rule (kva.h `kva_hazard_step`).** The pass's last sequence starts at P with q rows and
n_ahead more to come, so its exact tail starts at P + q + n_ahead − T: `before` = (T − n_ahead) − q of
those tail positions lie before this pass, i.e. came from the prefix cache. Any of them at or below the
slot's last approximated position was approximated by the request that wrote the snapshot: counted,
once (the slot remembers how far it counted). Then an approximate pass records its last bulk position.
A request's own bulk always ends before its own tail, so outside a branch the count is 0. The host
passes only keyed numbers (T − n_ahead as an operand extent); P, q and the slot are device data.
Simplifications vs PLAN-FIX §5.4, named: (1) only the step's LAST sequence is checked (the resuming
request of R65's corpus is always the last entry; a resume that shares a step behind another prefill
is not counted); (2) in quality mode the slot records the window's last position, so class rows inside
the window (which ran exact) count as hazard rows -- a superset, as PLAN's oracle `min(N1−N2,
T−(N2−P))` is; (3) a request that resumes from ANOTHER consumer's snapshot after that consumer counted
inherits its "counted" mark (rare: a branch of a branch).

**Static**: 56 arch cases / 680,943 checks; kernel host 851 checks (`hazard_semantics`: the branch
formula min(N1−N2, T−(N2−P)) = 1,144 for N1 6,144 / N2 5,000 / P 4,096, counted once, continuation 0,
record-only pass, out-of-pool slot, refusal). Mutants (scratch copy, `evidence/stageC/mutants-*.txt`):
F1-F5 (floor) and H1-H8 (hazard wiring) all caught.
