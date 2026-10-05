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

## 2. Session C1 (17:55-18:13Z; `evidence/stageC/session1.log`, `sessions/c1.sh`; home `data/home-041ab13`:
## arch dabbe9f5…, kva.so d550da69…; projector in host memory; boot 75e3e39b…)
- **kva.so device leg** (card 0000:13:00.0): 64 `kva_hazard` configurations device meta and count == host;
  1,158 GPU checks, ctest gpu 1/1.
- **R6/R7 green** at 041ab13: off ident = R3's six hashes.
- **R62 GREEN** (`tools/turns.py`, 10 append-only conversations 8K-32K, cache servers): turn-2 `cache_n`
  is stock's in every conversation for quality AND speed (8,192 / 12,288 / ... / 32,768; 8,192 for the
  9,216 one); 89 of each mode's 90 approximate steps wrote a checkpoint; `kva: hazard` lines: 0 (the
  append-only negative half of R68).
- **R64 as written cannot hold, by construction** (not a plugin defect): quality with the cache vs
  quality without it give different turn-2 answers (8 of 8 non-empty differ; 2 are empty in every run).
  With the cache, turn 2 keeps turn 1's own exact tail (positions N1 − T ... N1, e.g. 6,144-8,191)
  exact -- that is D-a's point; a no-cache recompute of the longer turn-2 prompt approximates those same
  positions, because they now lie before ITS tail. Same tokens, different rows exact. The cached answer
  has MORE exact rows. A meaningful R64 compares each against exact (KL), not against each other.
- **R65 / R68, first corpus (12 triples; 8 skipped as too short)**: the device counter is right on all 36
  requests -- the one B that resumed at its predicted checkpoint (`branch/003/B`, cache_n 14,336 = P)
  logged **1,696 positions = the oracle min(N1−N2, T−(N2−P)) = 1,696**; the other 35 have true hazard 0
  and logged 0. But the corpus formed only 1 branch: (a) 20 triples cycle over 9 documents, so later
  A's resumed from an earlier same-document A (`hazard_rate.py`'s superset rule flags those 6 A's;
  producer and consumer lengths differ by < 40 tokens, so the cached tail rows were exact in the producer
  -- true hazard 0, as the device says); (b) 10 of 12 B's hit an earlier checkpoint than the corpus
  predicted (engine checkpoint retention), so none of their tail came from cache. `hazard_rate.py
  --require-match` therefore reports DISAGREE (its superset vs the device's exact count). Rerun planned
  with one document per triple and enough `--checkpoint-slots`.

**R64 redefined (orchestrator, 18:25Z, agreeing with the reading above).** Turn 2 with the cache and
turn 2 without it are EACH scored against exact (KL / NLL on the same positions); bar: cached KL <=
no-cache KL within noise (the cache must never make a follow-up worse than recomputing it), and both
within the quality-mode budget already accepted. Reason: with the cache, turn 2 resumes from turn 1's
snapshot, whose own exact tail stayed exact; a no-cache recompute of the longer turn-2 prompt
approximates those positions -- so the two differ by design and identity is the wrong bar.
**R65 corpus bar (orchestrator)**: >= 8 real branches before R65/R66 are read.

## 3. Session C2 (19:08Z-; home 041ab13; host placement) -- two Stage B items + the R65 rerun
- **The 32K C = 8 decoder stall** (twice in Stage B session 8, speed and quality): the same arm shape
  (interleaved, 16K + 32K, C 0/1/4/8, 3 reps, server log followed, conc.py printing the decoders' errors
  and queue counts on a stall) on STOCK and on quality-host: **all reps completed on both** (24 + 24).
  Not reproduced. Tally: 2 stalls in ~150 plugin reps of that shape, 0 in ~120 stock reps -- too rare to
  call a degradation or to rule one out; the instrumentation stays in every conc session.
- **R60 on path A: unreachable, so unaffected.** With `--num-speculative-tokens 3` the stock scheduler
  never co-batches decode with prefill: 0 MIXED steps in either server (85 prefill-only, ~480 decode-only),
  so speculating decoders never ride beside a prefill chunk and speed took the lean path on all 75
  approximate steps. Per-decoder acceptance (separate requests, 5 rounds x 4): off 3,347 / 5,298 = 0.6317,
  speed 3,344 / 5,298 = 0.6312. (Stage B session 4's batched spec texts were taken the same way.)
- **R65 GREEN at T_ck 0** (corpus `branches2.jsonl`: one document per triple, `--min-prompt 8192`, server
  `--checkpoint-slots 64 --checkpoint-policy keep-all`): **9 real branches** (every B resumed at its predicted
  checkpoint), and on every one the device counter equals the oracle min(N1−N2, T−(N2−P)) and the
  records-side count exactly:

| branch | N1 | N2 | P (cache_n) | oracle | records | device |
|---|---|---|---|---|---|---|
| 000 | 9,384 | 4,787 | 4,096 | 1,357 | 1,357 | 1,357 |
| 001 | 9,393 | 6,561 | 6,144 | 1,631 | 1,631 | 1,631 |
| 002 | 9,401 | 6,569 | 6,144 | 1,623 | 1,623 | 1,623 |
| 003 | 16,570 | 15,094 | 14,336 | 1,290 | 1,290 | 1,290 |
| 004 | 16,549 | 2,532 | 2,048 | 1,564 | 1,564 | 1,564 |
| 005 | 16,554 | 8,530 | 8,192 | 1,710 | 1,710 | 1,710 |
| 006 | 16,553 | 12,564 | 12,288 | 1,772 | 1,772 | 1,772 |
| 007 | 32,927 | 16,725 | 16,384 | 1,707 | 1,707 | 1,707 |
| 008 | 32,952 | 4,466 | 4,096 | 1,678 | 1,678 | 1,678 |

  The 18 A (producer) and C (append-only) requests: no tail overlap in the records, device count 0.
  Not yet measured: the T_ck = 512 variant (session C3, queued), the greedy-answer agreement, and the
  negative control (zeroing kv_kva_meta after a restore needs a debug switch that is not built).
- **R68 GREEN**: `tools/hazard_rate.py` (superset rule from `timings`) flags exactly the 9 B's and none of
  the A/C's on this corpus, and with unnamed lines paired in order (fix above) its per-request cross-check
  with the device AGREEs (14,332 positions both). On the append-only R62 traffic: 0 flagged, 0 logged.

## 4. Stage C row table (stage-c; plugin code at 041ab13 = frozen home `data/home-041ab13`)

| row | state | evidence |
|---|---|---|
| R62 | **green**: turn-2 `cache_n` = stock's, 10/10 conversations, quality and speed | C1 |
| R63 | **not run** (snapshot/restore of the correction state at ≤ 1 ulp needs a capture of the pre-apply state after a restore; R62/R65 show restores working end to end) | -- |
| R64 | **redefined** (each vs exact, cached ≤ no-cache); the original form shown unsatisfiable by design (C1); the new form's session `sessions/c4.sh` + `tools/turns_kld_corpus.py` written, **not run** | C1, c4.sh |
| R65 | **green at T_ck 0** on 9 real branches (device = oracle = records); T_ck 512 variant, answer agreement and the negative control **not run** | C2 (+C3 if it ran) |
| R66 | **not run** unless C3 ran (session written: exact / speed T_ck 0, 512, 1,024 / quality 0, 512, 16K + 32K, cache servers) | c3.sh |
| R68 | **green**: superset flags exactly the 9 branches, per-request AGREE with the device after the in-order pairing fix | C2 |
| static | 56 arch cases / 680,943 checks, kernel host 851 + GPU 1,158 (incl. kva_hazard device == host); mutants F1-F5, H1-H8 caught | §1, C1 |
| R6/R7 | **green** at 041ab13 | C1 |
