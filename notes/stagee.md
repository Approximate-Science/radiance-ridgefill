# notes/stagee.md -- Stage E: the projector's VRAM cost, the int8 projector, untested combinations

Worker: account-B Opus 5.5, worktree `radiance-kva-wt-e`, branch `stage-e` off `0e25c16` (main after A′'s R144
re-run). radiance `140987f` (v1.0.8), read only. Plan: `fix-246/HANDOVER-FIX.md` §6, `PLAN-FIX.md` §6.4, §14-§17 and
"DECISIONS RECORDED", `PACKAGING.md`, `REFUTATION-3-selfload.md` §2.2, `REQUIREMENTS-FIX.md` R77-R80 (+ v1 R85-R91).
Model: the PUBLISHED file `~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad` (sha256 `0af5e962…4d20`, never written);
projector `~/models/rad/projector/` by discovery unless a row says `RADIANCE_KVA_PROJECTOR`. Boot
`75e3e39b-cc5b-49de-8087-a4791f372a92`. Image `stilldeadcode/radiance:1.0.8`, `RK_FLAGS` of `scripts/common.sh`
(`--gpu-headroom-mib 3072`). Every engine arm under `flock gpu.lock`; scripts in `evidence/stagee/scripts/`.

## 0. Predictions registered before the runs (labbook ids)

| id | row | prediction | kill |
|---|---|---|---|
| HE-R80-slots | R80 | resident slab slots a rank: off = exact ±0.3%; vram bf16 −6…−7.5% (≈ −1,100); host ring −0.5…−1% (128 MiB) | vram within 3% of exact, or host ring > 2% down |
| HE-held-2k | R80 | the pure VRAM cost (FORCE_SPLIT=2048: projector held, every pass stock), 2,048-token prompt: vram +8…+20% TTFT vs exact (A.1: +223 ms at 2,112); host ring 0…+3% | vram within ±3% of exact |
| HE-held-1k | R78 | 1,024-token prompt (one unstaged chunk): vram +0…+6%, host ring ±2% | vram > +10% |
| HE-R77-dec | R77 | decode ms/step at C = 1: off = exact ±1%; vram bf16 +1…+4%; host ring ±1%. C = 32: vram +2…+6% | vram C=1 > +8% or < 0 beyond noise |
| HE-R79-kl | R79 | int8 projector vs bf16, paired per doc, last 512 at T 2560 and whole tail at T 2048: |ΔNLL| ≤ 0.002, CI includes 0 | CI excludes 0 or |Δ| > 0.005 |
| HE-R79-slots | R79 | int8 recovers ≈ half the vram slots lost (≈ +550 a rank vs bf16) | < +400 |
| HE-held-1slot | §14 | held (FORCE_SPLIT 2048) vs stock, 2K settled, matched state, after 6a68dde: int8 +0.3..+1.0%, bf16 +0.8..+1.6%; slab slots a rank int8 −60..−100, bf16 −110..−160 | int8 > +1.5%, or bf16 not below +2.6% |
| HE-1slot-ttft | §14 | ON TTFT 16K/32K: one slot within ±1.5% of S3's two slots per dtype; int8 ≤ bf16 + 1% | > +3% either way |
| HD-R76-plumb-media | R76 | image + ~12K text, greedy 64: plumb text (and logprobs) IDENTICAL to off; ≥ 3 approximate passes after the image | differs, or no approximate pass |
| HD-R70-final | R70 | MTP depth 3, 16K, 256 greedy, median of 5, tokens/step vs stock: final 0.95..1.0x, FINAL=off 0.85..0.92x, int8+final within 0.02 of bf16+final | final < 0.93x, or final − off < 0.03 |

## 1. What was built (commits on `stage-e`)

| commit | what |
|---|---|
| `915494a` | kva.so: `kva_gemm_nt_q` = libr4d's int8 `gemm_nt_q` rows (dtype `i8a8`: `gemm_i8a8_nt_m16`, `gemm_i8a8_tiled`) forwarded with b / b_scale as IN operands, **layout/relayout/unrelayout hooks kept** (the engine consults a row's hooks only for WEIGHT operands -- `rad_builder.cpp:510-545`, `ctx.cpp:470-490`, `oracle.cpp:463` -- so they are inert here, and they are what the arch calls at load). Device only: libref's int8 row reads canonical planes, the operands are stored ones. bf16 forward unchanged (first row, same name). |
| `ab3f57a` | arch: an int8 folder (manifest `"projector": {"dtype": "i8"}`), `arch/kva_int8.h` finds kva.so's `kva_gemm_nt_q` rows in the loaded objects and relayouts each map's canonical planes through their hooks before the per-rank upload (all rows must describe the same stored bytes, else refused by name); `project_rows` (kva_layer.h) replaces the three projector issue sites: bf16 = the old issue, int8 = `quant_act_i8g` over the layer-S stream ONCE a pass (at layer S; every late layer projects the same rows), then per layer `kva_gemm_nt_q` + the in-tree row-broadcast `add` for the bias. Host placement + int8 = serving stock, said by name (the ring moves bf16 rows). |
| `d8f19eb` | `tools/kva_projector.py int8 --from <bf16 folder> --out <dir>`: i8*bf16[1x128] (the container trunk's own encoding: absmax/127 per 128 of a row, bf16 scale, codes half-to-even against the ROUNDED scale), canonical planes only; other files copied byte for byte; the source manifest's sha256 recorded. |
| `255fad6` | R87 static TP4 cases; `scripts/ident_long.sh` (R89's instrument) |

**The int8 folder** `data/projector-qwen38fn-int8` (from `~/models/rad/projector`, i.e. data/projector-qwen38fn):
29 files, 698,753,818 B (0.651 GiB; bf16 folder 1.228 GiB); 24 maps, worst |w − dequant| = 0.39% of a map's
max |w|; kva.json sha256 `5b699e27e88d…85ae`; 0644. Built in 5.8 s. Selected with
`RADIANCE_KVA_PROJECTOR=<folder>` (the switch; defaults unchanged).

## 2. Static results (HIP build in radiance-build, `ctest -LE gpu` 3/3)
- `arch_static_test`: 52 cases, 1,148,836 checks, all ok. New: `an_int8_folder_uploads_its_maps_in_the_gemms_stored_form`
  (ops/params/buffers, 2 relayouts a layer, copies of the hook's byte size), `the_int8_projector_quantises_the_stream_once_then_gemm_and_bias_a_layer`
  (the int8 pass = the bf16 pass with the projection swapped, op for op with handles normalised by name, on lean,
  tail-only straddle, masked T2048/T1984 and a masked mixed step with s_lb 65), `an_int8_folder_with_host_placement_serves_stock`
  (graph == stock), `an_int8_folder_without_one_agreed_stored_form_is_refused_by_name` (no row; rows that disagree;
  dtype i4 = cannot run → stock). TP4 added to off == in-tree, raw-operand head slices (12 a rank), straddle issues.
- `kernel_test host` (libr4d + libref preloaded): `int8_forward_is_libr4ds_int8_rows_with_their_hooks` ok; alone:
  no `kva_gemm_nt_q` row. pytest `tests/test_kva_projector.py` 8 passed (2 new: the rounding rule, the folder).
- Mutants (scratch copy, `evidence/stagee/scripts/mutate_int8.py`, output `evidence/stagee/mutate_int8.out`): 9/9
  caught -- I1 quantise every layer, I2 no bias, I3 scales at the codes' offset (caught only after the s_lb 65 case
  was added), I4 canonical planes uploaded, I5 int8 + host allowed, I6 disagreeing rows accepted, I7 bias before the
  GEMM, I8 manifest dtype ignored, I9 scale operand per row.

## R91 -- external draft model: REFUSED BY THE ENGINE, by name (no plugin claim)
`docker run stilldeadcode/radiance:1.0.8 --model … --tp 2 --draft-model /models/draft` with the plugin home and
`RADIANCE_KVA=quality` (no GPU attached; it fails at flag parsing, before any plugin loads), exit 2
(`evidence/stagee/r91/draft-model.log`):
`E --draft-model is not implemented. This engine loads its drafter from the model container, not from a second one:
merge it at conversion time with rad-convert --draft-model DIR and pass the container to --model. An MTP head that
ships inside the target checkpoint needs neither flag.` (radiance `core/config.cpp:415-430`).

## 3. Session e1 -- what the projector's VRAM costs a request (R80 / R77 / R78), 2026-10-05 11:46-14:22Z
Home `data/home-7ec3603` (arch d877e61f…, kva.so e73dc71b… = A′'s code), published model, projector by discovery,
boot 75e3e39b…, RK_FLAGS (headroom 3072), `--max-num-seqs 8`, `--debug-placement`. Quality ON servers (T 2048).
Warmed TTFT (A.1's protocol: one RK_REPS=2 pass of 1K/2K/4K/8K discarded, then RK_REPS=7, median of the last 5;
rounds a+b pooled where both exist). **held** = `RADIANCE_KVA_FORCE_SPLIT=2048` (b = n_tok − 2048 ≤ 0 on every pass:
the projector is loaded and EVERY pass runs the stock step -- what a request that is not approximated pays, i.e. an
OFF request on a Stage F server). Logs/JSON `evidence/stagee/e1/`, table `evidence/stagee/scripts/table.py e1`.

| arm (prompt_ms, n = reps) | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---|---|---|---|
| exact (stock) | 816 (10) | 1,232 (10) | 2,527 (10) | 4,999 (10) |
| off | 818 = 1.00x [+3: −11, +17] | 1,232 = 1.00x | 2,532 = 1.00x | 5,009 = 1.00x [+11: −2, +34] |
| vram bf16, ON (qv) | 928 = **0.88x** (+112 [+99, +122]) | 1,432 = **0.86x** (+200) | 2,592 = 0.97x (+65) | 4,488 = **1.11x** (−511) |
| vram bf16, held | 922 = **0.88x** (+107) | 1,430 = **0.86x** (+198) | 2,940 = **0.86x** (+413) | 5,815 = **0.86x** (+817) |
| host ring, ON (qh) | 779 = 1.05x (−37) | 1,449 = **0.85x** (+217) | 2,827 = **0.89x** (+301) | 5,114 = 0.98x (+116) |
| host ring, held | 777 = 1.05x (−39) | 1,451 = **0.85x** (+218) | 3,014 = **0.84x** (+487) | 5,954 = **0.84x** (+955) |

Placement (both ranks; `--debug-placement` lines in each serve log): slab slots exact 16,880 / 16,928 · off 16,883 /
16,930 · vram 15,765 / 15,812 (−1,115, −6.6%; already held 1.50 GiB vs 306 MiB) · host ring 16,713 / 16,761 (−167,
−1.0%; held 434 MiB = correction + row table + 2 × 50 MiB ring slots; 1,200.5 MiB host-mapped a rank). Prefill
staging buffers 186.1 (stock) / 213.0 (vram) / 190.1 MiB (host). Approximate steps: ON 44 a server (4K/8K prompts),
held 0. Kernel log clean.

Caveats (named): qv-a's TTFT overlapped a scratch compile of mine (load 2.48 at its start; qv-a and qv-b agree to
<1%, both pooled); qv-a waited 1,160 s + 510 s in quiet() behind my own queued GPU-test wrapper (the old pgrep -f
pattern; fixed to process names in arm.sh) -- still a quiet host (load 0.28); qv-held-a's measured label was lost to
a preflight false positive (another worker's labbook command line matched "radiance"), so qv-held = round b only;
qh-held-b not run (stopped for the decode priority); e1 was killed once by the tool's 2-hour background limit at
qv-a (partial files in e1/killed-qv-a/) and resumed as e1b.

**Reading (R80 / R78): the short-prompt penalty is NOT the projector's VRAM.** Held (every pass stock), host ring --
which keeps 99% of stock's resident experts -- costs the same as vram: 0.85x / 0.84x / 0.84x at 2K/4K/8K vs vram's
0.86x. And stock with 1,228 MiB less VRAM (ediag below) decodes at stock speed. So A.1's mechanism ("−6.7% slab ⇒
bigger staging buffers ⇒ +1.3 GB h2d a staged chunk") does not explain it: something an ON server does on stock
passes costs ~14% of prefill whatever the placement. OPEN (first item on resume): separate it with held TTFT arms
for plumb (masked declare, no projector, no correction groups), speed (correction group only), quality without the
row table, and a quality server with the declare but no `rad_buf_concurrent` widening (if needed, a debug switch).
1,024-token prompts (unstaged chunk): vram 0.88x, host 1.05x -- R78 (±2%) red for vram, host faster than stock (?).

**R77 decode from e1 is WITHDRAWN as steady state**: e1's cstep2 windows ran right after the TTFT prefill, inside the
post-start/post-prefill transient (e1's own stock reads 12.75 ms at C=1 there vs 10.70 settled, below).

## 4. ediag -- decode warm-up vs steady state (2026-10-05 15:00-15:12Z; home d8f19eb; one gpu.lock for the session)
Decode-only servers, `--max-num-seqs 8`, no prefill before cstep2; C = 1, 8 twice: window 1 = cold (right after
boot), window 2 = settled. Labbook HE-dec-vram-slots / HE-dec-host-alloc registered before.

| arm | C=1 cold | C=8 cold | **C=1 settled** | **C=8 settled** |
|---|---|---|---|---|
| stock | 23.04 | 21.15 | **10.70** | **14.32** |
| stock, headroom 4300 (1,228 MiB less, slab −1,056) | 24.06 | 21.79 | 10.71 | 14.36 |
| quality, vram | 24.06 | 21.64 | 10.71 | 14.36 |
| quality, host + ring | 12.23 | 14.64 | **11.30 (+5.6%)** | **15.46 (+8.0%)** |
| quality, host, RING=0 | 22.88 | 21.11 | 10.71 | 14.22 |
| plumb, host (nothing host-mapped) | 12.18 | 14.57 | 10.69 | 14.19 |

- **HE-dec-vram-slots REFUTED**: 1,056 fewer resident slots cost nothing settled; vram placement decodes as stock.
- **HE-dec-host-alloc REFUTED**: the same host-mapped allocation without the ring decodes as stock; the steady-state
  cost (+5.6% / +8.0%, one boot; Stage B independently +8.7% at C=8) comes with the RING (2 × 50 MiB VRAM slots, the
  declared `cast` op, or lane use -- none issued on a decode pass). `--profile-ops` (cold, C=1, diagnostic only):
  host+ring vs stock +3.4% (rank 1) / +5% (rank 0) spread uniformly over the dense ops (gemm_nt_q, gemm_nt,
  hc_read, all-reduces), routed-expert GEMMs equal (1,737 vs 1,743 ms) -- no single op.
- Warm-up cost after start: every arm, stock included, decodes at 12-24 ms/step in its first ~6 s window.
- **ediag2 (15:16-15:29Z, 5 interleaved boots, 3 windows; window 3 = settled), C=1 / C=8:** stock 10.67 / 14.18 and
  10.67 / 14.19; host + ring **10.65 / 14.19** and **10.67 / 14.19**; host RING=0 10.63 / 14.19. **Host placement
  settles to stock**: ediag's +5.6% / +8.0% (one boot, window 2) was a LONGER WARM-UP TAIL at C=8 (window 2: 16.4
  vs 14.2), not a steady-state cost. Labbook `stageE-dec-settled-c1` (median 10.68, range 10.63-10.71, n 10);
  HE-dec-vram-slots and HE-dec-host-alloc both REFUTED.
- **R77 conclusion: steady-state decode on an ON server = stock within ±0.4% for both placements; warm-up after start
  is 12-24 ms/step in the first ~6 s window for every server (stock included), host's C=8 tail one window longer.**
  No plugin fix is needed for decode. (C = 32: e1/d-* servers, run in ediag/ediag2.)

## 5. The int8 device leg (R79's oracle) -- green after a bound fix
First run (e2/try1/kernel-gpu.log): 2 of 5 runs over a 2-bf16-ulp bound (worst 1.47x) on outputs the 80 f32 block
sums cancel toward zero. Bound now 2 ulp + 2^-9 × RMS (5b991e1); re-run on card 0000:13:00.0
(evidence/stagee/devleg-green.log): `kva_gemm_nt_q_r4d_gemm_i8a8_nt_m16` M 1 / 64 and `_tiled` M 1 / 64 / 2048 vs
libref's gemm_nt_q on canonical planes, 0 of 5.57 M outputs over, worst 0.80 of the bound. A negative control
(canonical scale plane fed un-relaid) is built in /tmp/stagee-mut/neg but NOT yet run (needs a quiet GPU slot).

## 6. Row state at this checkpoint (2026-10-05 ~15:20Z)
| row | state |
|---|---|
| R80 | **measured** (bf16 vram, host ring; §3): slab −6.6% / −1.0%, held 1.50 GiB / 434 MiB a card; `h_S`/`x_P` inside the arena (+~100 MiB vs stock); speed+final SKIPPED (no `final` map until Stage D). int8 arms not run yet (e3) |
| R77 | **green at C = 1, 8 (steady state = stock ±0.4%, both placements; §4)**; warm-up stated; C=32 servers in ediag2 |
| R78 | **red as measured**: 1,024 tokens vram 0.88x; host 1.05x; plus the ON-server prefill penalty on stock passes (~0.85x 2K-8K, both placements) -- cause open (§3) |
| R79 | int8 BUILT + static (52 cases, 9/9 mutants) + device leg green; KL arms (e2) and VRAM/TTFT arms (e3) not run |
| R85, R86, R88, R89, R90 | scripts ready (e4, e5), not run |
| R87 | static TP4 green (255fad6); engine TP1 not attempted |
| R91 | **refused by the engine by name** (§R91) |
| MUST-FIX (Stage B) | vram placement refuses at `--max-num-batched-tokens 8192 --max-num-seqs 10` (host pool full) where stock starts -- not started |

### Chain state at the checkpoint (all detached with `nohup setsid`, each session under ONE gpu.lock)
- ediag stopped after `d-exact` at 15:16Z (my edit of arm2.sh under a running arm; the arm's measurements completed,
  its stop did not -- container stopped by hand, logs saved). Lesson: never edit a script a session is executing.
- `run_diag2.sh` → ediag2 (running from 15:17Z): y-exact-1 / y-qh-1 / y-qh-noring / y-qh-2 / y-exact-2 (decode, 3
  windows) then R77's 32-sequence servers d-off / d-sv / d-qv / d-qh (into e1/).
- `run_diag3.sh` → ediag3 after ediag2: held TTFT for stock / plumb / speed+host / quality+host (which part of the
  declare costs stock passes ~15%), then `--profile-ops` of 2048-token prefills, stock vs quality+host held.
- Both runners skip if their session already logged its end line; rerunning them is safe.

## 7. Resume here
1. `evidence/stagee/ediag.session.log` (d-* 32-seq decode servers, written into e1/) and `ediag2.session.log`.
2. Separate the ON-server prefill penalty (§3 OPEN) and the ring's decode cost (§4); fix plugin-side; frozen home; re-measure.
3. Stage B's must-fix (auto placement decided at declare, loud log line; test their 8192/10 config).
4. Then e2 (int8 KL, R79), e3 (int8 table), e4 (R85/R86/R89/R90), e5 (R88). Every session: ONE `flock gpu.lock` for the
   whole session (per-arm locks starve behind other lanes' session-long locks), launched with `nohup setsid`, no compile
   while any session runs, no waiter whose command line contains a quiet() pattern word.

## 8. DYLAN'S DECISION (2026-10-05 ~15:45Z, via the orchestrator): VRAM placement is REMOVED
The projector lives in host RAM and is streamed through the staging ring; that is the only placement. Why (history
kept here): vram placement refused to start at `--max-num-batched-tokens 8192 --max-num-seqs 10` where stock
starts (Stage B: "DID NOT FIT 461 gate_up experts: host pool full") -- its 1.2 GiB a card pushes ~1,100 experts into
the pinned pool -- and the plugin has no budget ABI at declare to choose safely (§ the must-fix message). Measured
trade it removes: vram ON TTFT 6% / 10% faster than host at 16K / 32K (A′ R148); settled decode equal (§4); the
stock-pass prefill penalty equal for both (§3).
Work list (in order, after the held-prefill penalty which stays first):
1. Delete the vram path and the zero-copy (RING=0) path; `RADIANCE_KVA_PROJ_PLACE` / `RADIANCE_KVA_PROJ_RING` become
   retired switches refused by name ("the projector is always streamed from host memory through the staging ring").
2. int8 THROUGH THE RING: per layer the stored codes + stored scales + bias laid out as rows of `hc·n` bf16 (codes
   1,280 rows, scales 20, bias 1 → 1,301 rows), copied by the same `cast` bf16→bf16 (libr4d `r4d_p2p_copy2d`, a pure
   byte copy -- no value passes through a float), slots 2 × 25.4 MiB; the GEMM's codes/scale/bias = slot offsets.
3. Option C (correction + row table to host memory) only if it measures free on ON TTFT (paired, warmed).
4. kva_config.h comments, README, notes: nothing documents vram placement.
5. Tests: B's 8192/10 config starts and serves; retired-switch refusals + mutants; off ≡ stock.
6. Measure int8-ring vs bf16-ring: ON TTFT 9K/16K/32K warmed + paired; bytes a pass and copy time per rank.

### Ring-only placement: COMMITTED 0003294 (static 51 cases green, 1,124,919 checks); not yet: mutants
(`evidence/stagee/scripts/mutate_ring.py`, R1-R7), frozen home, engine checks (list below). Was:
Ring-only placement + int8 through the ring: `arch/kva_config.h` (PLACE/RING retired, refused by name),
`arch/kva_declare.h` (`decl_ring`, always declared, int8 too), `arch/kva_projector.h` (RowBlock: bf16 [n+1] rows /
int8 stored codes + scales + bias at the hook's sizes; host block + 2 VRAM slots only; upload key mode × table),
`arch/kva_declare_masked.h` (int8+host refusal removed), `tests/arch_static_test.cpp` (run_step sets the ring's lanes
and copies aside for the in-tree oracle -- `Run::all` keeps them; raw-operand, host-memory, int8-upload cases check
the host row blocks' bytes; retired-switch refusals for PLACE/RING; the int8 comparison names ring copies by layer),
README switch table + "Where the projector lives", folder README template. NEXT: build (`/tmp/stagee-build.sh`) and
static-test once no session is measuring; mutants (PLACE/RING not refused; int8 scale/bias offsets in the slot);
frozen home; engine: off ident, hostring bf16 KL bytes = A′'s hostring rows, int8 ring KL (R79), Stage B's 8192/10
config starts and serves, int8-ring vs bf16-ring TTFT 9K/16K/32K + bytes/pass + copy time per rank.

## 9. ediag3 (15:41-~16:05Z, home d8f19eb) -- the stock-pass prefill penalty is a SERVER STATE, not a fixed cost
Held TTFT (every pass the stock step), same protocol as e1, last-5 medians (1K / 2K / 4K / 8K ms):
stock 814 / 1,233 / 2,529 / 5,010 · plumb held 820 / 1,237 / 2,541 / 5,029 (+0.3%) · speed+host held 830 / 1,260 /
2,603 / 5,136 (+2.2 / +2.9 / +2.5%) · quality+host held 831 / 1,261 / 2,597 / 5,137 (+2.3 / +2.7 / +2.5%).
e1's quality+host held (home 7ec3603, 13:38Z) was 777 / 1,451 / 3,014 / 5,954 (+18 / +19 / +19%) with every rep
flat. The rep sequences explain the gap: ediag3's z-qh-held STARTED in e1's slow state (warm-up 2K 1,452 / 1,448,
4K 2,916) and dropped to the fast one inside its warm-up (4K 2,759, then 2K 1,260 for all measured reps); e1's ON
servers (qv, qh, qv-held, qh-held, both rounds) never left the slow state in ~3 min of reps. Stock servers start
fast. So an ON server can sit in a slow prefill state (+16..19%) that some boots leave within seconds and others
not for minutes; the steady fast state still costs +2.3..2.9% on stock passes when the projector is held (plumb,
which holds nothing, +0.3%). OPEN: what the slow state is and what flips it (prime suspects: the expert mover's
initial placement / heat with the plugin's already-held VRAM, or host-side state) -- the ediag3 prefill profiles
(pp-exact vs pp-qh-held, --profile-ops) are the first evidence; e1's table above is therefore a slow-state table.
**ediag3 prefill profiles (diagnostic only; `profcmp.py`, 2048-token prefills, fresh servers):** stock 1,289 ms,
quality+host held 1,520 ms (+18%: the slow state, caught). Rank 1: EVERY kernel slower -- rmsnorm 7.8 → 22.5 ms,
rope 4.3 → 13.7, gdn_gated_rmsnorm 8.6 → 24.8, gdn_chunk_scan 23.4 → 42.4, gemm_nt_q 153 → 215, moe_gemm_q 316 → 375
(total +20.6%); rank 0's extra time is waiting in the all-reduces (ar_gather_hc_write +421 ms, ar_hc_write +164). Small
kernels 2-3x slower on one card is contention from concurrent device activity on rank 1, not plugin work (a held pass
issues the stock step). Prime suspect: the expert mover still migrating experts on rank 1 (its slab is smaller by the
plugin's held VRAM; the heat engine moves 8 units a dispatch) until it converges -- the fast state. Next: the mover's
copy counters / RADIANCE_LOG_STEPS on a slow vs fast server, and whether a stock server with the same held VRAM
(headroom +128 MiB / +1,228 MiB) shows the same slow start.

## 10. Resume order (written 16:01Z; supersedes §7's list)
1. Mutants for 0003294: `python3 evidence/stagee/scripts/mutate_ring.py` (R1-R7) -- compiles: only when no radiance-kva
   container is up (`/tmp/stagee-gapbuild.sh` pattern) or between my own sessions.
2. Frozen home of the ring-only commit: `RK_RADIANCE_SRC=~/projects/inference/radiance scripts/frozen_home.sh <HEAD>`.
3. `flock gpu.lock env HOME_E=<that home> sh evidence/stagee/scripts/ering.sh` (detached, nohup setsid): off ident,
   retired-switch refusals, bf16-ring KL bytes vs A′'s R144 rows, int8-ring KL (R79), Stage B's 8192/10 config,
   int8-ring vs bf16-ring TTFT 9K/16K/32K + 16K profiles (the ring's `cast`: 24 calls a pass; the stream copy into h_S
   is the other `cast`, 1 call a masked pass).
4. `flock gpu.lock env HOME_E=<home> sh evidence/stagee/scripts/ediag4.sh`: the slow prefill state -- stock with the
   same VRAM taken (headroom +128 / +1,228 MiB), RADIANCE_LOG_STEPS through the transition.
5. Then e3 (int8 short-prompt table, now ring-only: drop qv arms), e4 (R85/R86/R89/R90), e5 (R88) -- each under ONE lock.

## 11. Resume session S1 (2026-10-05 16:26-16:48Z, one gpu.lock; home `data/home-bc7e742` = ring-only 0003294 + tools; arch 1dad9831…, kva.so f0911d4a…; boot 75e3e39b…; evidence/stagee/ering/)
- Mutants (`mutate_ring.py`, `ering/mutate_ring.out`): **7/7 caught** -- R1/R2 retired switches not refused, R3 int8
  scales at the slot start, R4 int8 bias at the bf16 block's place, R5 canonical codes uploaded, R6 copy into the other
  slot, R7 bias at the codes' start.
- **off ident = R3** (R6/R7 for the ring-only code). `RADIANCE_KVA_PROJ_PLACE=vram` and `RADIANCE_KVA_PROJ_RING=0`:
  startup refused, "is retired: the projector is always streamed from host memory through the staging ring …".
- **bf16 through the ring, rows byte-IDENTICAL to A′'s R144 rows**: plumb FORCE_STREAM, speed T2048, quality T2048,
  quality T2560 (67 approximate steps each).
- **Stage B's config `--max-num-batched-tokens 8192 --max-num-seqs 10`: starts and serves** in quality and speed (a 9K
  prompt, 8 concurrent decoders); pinned pool 9,860 / 9,813 slots of ~10,240 (≈0.4 GiB margin) with the ring's 128 MiB.
- **R79 quality -- int8 through the ring holds the bf16 projector's level** (paired int8 − bf16 dNLL, same boot):
  quality T2560 last 512 **−0.00109 [−0.00400, +0.00215]**; quality T2048 whole tail **+0.00101 [−0.00111, +0.00336]**;
  quality T2048 last 512 +0.00169 [−0.00146, +0.00493]; speed T2048 last 512 +0.00041 [−0.00450, +0.00527]. int8
  headline +0.00103 [−0.01348, +0.01529] vs exact (bf16 +0.00212). Labbook `stageE-r79-int8-minus-bf16`, HE-R79-kl
  CONFIRMED. R79's TTFT half: S3.
- Option C (correction + row table to host memory) NOT done: it would save 28 MiB a card (the ring's slots are 100 MiB)
  and make every approximated chunk zero-copy ~54 MiB of correction (2 ops × 18 layers × 1.5 MiB) over the link --
  unlikely to measure free; kept in VRAM.
- Earlier stock-vs-plugin warm-up evidence (e1, ediag3 warm-up reps): 5 stock/off boots and plumb-held run 2K at
  1,227-1,234 ms from the first request; every e1 quality server (vram and host, ON and held) started at 1,426-1,453 ms;
  ediag3's quality-host-held started slow and turned fast mid-warm-up, its speed-host-held started fast. S2 settles it.

## 12. S2 -- the slow prefill state is engine/machine state, NOT the plugin (2026-10-05 17:09-17:52Z, home bc7e742)
`tools/settle.py` on fresh boots (52b8145/bc7e742): 15 interleaved cycles of 1K/2K/4K/8K prefills from /health, /stats
(mover, link, passes) polled every second; evidence/stagee/s2/ (settle-*.json/.txt, serve logs). 2K / 8K ms:
| boot | 2K first → steps → settled | 8K | 
|---|---|---|
| stock-1, off-1, stock-2, off-2 | 1,366 → **1,250** at ~90-99 s | 4,966 → 5,060 at the same step |
| quality held-1/-2, speed held-1/-2 (identical) | 1,600 → 1,405 at ~30 s → **1,283** at ~112 s | 5,877 → 5,080 → 5,190-5,225 |
| **stock-3, off-3** | **1,600 → 1,548, never faster in 150 s** | 5,771 → 5,712 |
- The levels are discrete and reproducible to the millisecond within a configuration (the engine is deterministic for
  a request sequence); each step coincides with a ~700-800-promotion burst of the expert mover. Card temperature or
  clock at boot does not predict the state (exact-1 fast at 65 °C, exact-3 slow at 50 °C).
- **So the e1/A.1 "ON server 0.85x" was a cross-state comparison** (stock boots that happened to be fast vs plugin boots
  that were slow), and ediag3's slow-state profile (every rank-1 kernel 1.3-3x slower) is an engine/machine state that
  stock servers enter too.
- **Held cost in MATCHED states** (every pass the stock step, projector held): fast 1,283 vs 1,251 (+2.6%), mid 1,405
  vs 1,366 (+2.9%), 8K +2.5-3%; slow ≈ level. Routed-expert hit rate trails stock by 1-1.5 points throughout (0.767 vs
  0.776 at start, 0.922 vs 0.932 at 135 s) -- consistent with the ring's 128 MiB of VRAM (−167 slab slots, −1%). S3
  opens with stock at headroom +128 MiB to test exactly that.
- **Protocol (binding from here): pair only servers in the same state** -- interleave, classify each server by its 2K
  level from a settle trace (S3 adds a 2K probe to every warm-up), and report state-mismatched pairs as such.

## 13. Dropped from the plan (orchestrator's pace cut, 2026-10-05 ~17:20Z) and why
- e3 (int8 short-prompt VRAM table, vram-int8 / bf16-held rounds): only mattered for VRAM placement, which is removed.
- e1's remaining arms (qh-held-b, 32-seq d-* for vram): VRAM placement removed; settled decode = stock is shown (§4).
- R86 (`--kv-cache-dtype bf16`), R88 (48K-160K contexts), R89 (long-prompt determinism × 3 boots), R90
  (`--deterministic`), R87 engine TP1: not needed to ship this stage -- SKIPPED by decision (their scripts e4/e5 and
  corpora stay in evidence/stagee/ and data/stagee/ for a later pass). R85 (production wire wht6) kept minimal: plumb
  byte identity + quality T2560 KL vs stock under wht6, batched into S3's lock.
- 9K in the int8-ring vs bf16-ring TTFT: 16K and 32K only.

## 14. The held cost is the plugin's VRAM -- reduced (6a68dde); S4 re-measures (orchestrator's decision ~18:00Z)
**S3's front controls** (17:58Z-, home bc7e742, evidence/stagee/s3/session.log; settle.py from /health, settled 2K):
stock at headroom 3200 (= the ring's 128 MiB taken away) 1,377.7 → **1,270.4** (+1.6% over S2's stock 1,250); stock at
4300 (1,228 MiB, the old vram maps) stuck slow (1,562); quality held int8 stuck slow (1,584); **plumb held (masked
declare, nothing uploaded) 1,367.5 → 1,251.7 = stock**. So the held +2.6% is VRAM taken from resident experts, not
declarations or per-pass work: ring 2 × 50 MiB (+ correction 27 + row table 1) + arena h_S 40 + x_P ~16 ≈ 212 MiB bf16.
**Not accepted as a cost** (Dylan: requests that do not use KVA lose nothing). Built in **6a68dde**:
- (a) the ring's slot follows the loaded folder's row block: 50 MiB bf16, 25.4 MiB int8 (a bf16 block with the MTP map).
- (b) each rank's correction heads and the row table live in the host block (zero-copy reads, approximate passes only).
- (c) instead of half-layer blocks: **one slot**. Layer l+1's copy is issued right after layer l's GEMM (and the int8
  bias add), its last reader, and overlaps the rest of layer l (attention or delta net + MoE); same VRAM as half
  blocks, no GEMM split, and the copy loses only the GEMM's own time to overlap (< 1 ms at 2,048 rows).
- int8 maps without the MTP map declare no h_S: the codes are made from b_h at layer S, before any late layer writes it.
- VRAM a rank now: **bf16 ≈ 106 MiB** (slot 50 + h_S 40 + x_P 16), **int8 ≈ 62 MiB** (slot 25.4 + codes 20 + x_P 16);
  was ≈ 212 / ≈ 156.
- (d) **NOT achievable plugin-side** (read rad_bufplan.cpp:80-85, rad_builder.cpp:1427-1471, rad_buf_concurrent
  rad_builder.cpp:835): a transient's lifetime is [first_def, last_use] over op indices in DECLARATION order, and the
  planner packs non-overlapping intervals. The plugin's ops are declared after the in-tree graph but issued
  interleaved with it, so any narrow plugin lifetime lies after every in-tree op's index: the planner would alias it
  with in-tree transients that are live at the same time on the device -- unsafe. Lane-1 buffers (the ring slot) must
  be whole-program (rad_buf_concurrent). Getting the remaining ~60-100 MiB to zero needs the in-tree graph re-declared in
  issue order around the plugin's ops (an engine change; not proposed) -- the remaining cost is what S4 measures.
- Option C's earlier objection (§11: zero-copy correction reads ~54 MiB a chunk over the link) is what S4's ON TTFT
  tests; if it costs, the correction can ride the ring with its layer's block (+1.5 MiB slot) instead.
- Static tests follow (ring order wait → GEMM → copy next; one slot; host-placed correction/row table; int8 slot 1,301
  rows and no h_S). Mutants: evidence/stagee/scripts/mutate_slot.py S1-S10 (new), mutate_final F4 now drops
  `ring_wait`, mutate_ring R6 retired (two slots).
- Predictions registered before S4 (labbook seq 428-429): **HE-held-1slot** (held vs stock 2K settled: int8 +0.3..+1.0%,
  bf16 +0.8..+1.6%; kill int8 > +1.5% or bf16 not below +2.6%), **HE-1slot-ttft** (one slot within ±1.5% of S3's two
  slots per dtype; int8 ≤ bf16 + 1%; kill > +3%).
- **S4** (evidence/stagee/scripts/s4.sh, queued 18:45Z on gpu.lock, `HOME_E=6a68dde`): frozen home + host tests;
  correctness (off ident = R3, bf16 rows = A′ R144, int8 rows = S1's, B's 8192/10 starts); held settle vs stock and ON
  TTFT 16K/32K bf16/int8/stock, two rounds interleaved; --profile-ops copy time per rank; mutants last.

## 15. S3 results (2026-10-05 18:05-19:09Z, home bc7e742 = the TWO-slot ring; evidence/stagee/s3/; labbook seq 433-435)
**R79's speed half -- int8 through the ring is FASTER than bf16** (quality T2048, warmed, median of 7, ratio to the same
round's stock 9,981 / 19,393 ms; tables: `evidence/stagee/scripts/s4table.py evidence/stagee/s3`):
| arm | 16K | 32K | slab slots r0/r1 (stock 16,847/16,894) |
|---|---|---|---|
| bf16 a / b | 0.865x / 0.859x | 0.663x / 0.658x | −187 / −162 |
| int8 a / b | 0.836x / 0.839x | (lost) / 0.628x | — / −163 |
- **Copy time per rank** (`--profile-ops`, one 16K prefill, 336 ring copies = 14 approximate passes × 24 layers; a
  profiled run is synchronised, so this is each copy alone): rank 0 bf16 0.97 ms, int8 0.49 ms a copy; **rank 1 bf16
  8.77 ms, int8 3.94 ms** (rank 1 sits on PCIe Gen4 x4 on this machine: ~6-7 GB/s). Bytes a pass a rank: bf16 24 ×
  2,561 × 20,480 B = 1.259 GB, int8 24 × 1,301 × 20,480 B = 0.639 GB. Rank 1's copy time a pass, bf16 210 ms vs int8
  95 ms, is what int8 saves where the copy is not hidden behind the layer.
- **R79 VERDICT: keep int8** -- KL at the bf16 level (§11: every paired CI includes 0), half the link bytes and the
  ring slot, 2.5-4.7% faster ON TTFT. It is the likely shipping folder; the default stays the folder the user points
  at (no default changed here).
- int8-a's 32K sample was lost to a measurement-hygiene slip: an interactive command naming a path outside the repo
  that matches the preflight's squatter pattern ran at the moment of a check (the preflight fails closed on a process
  it cannot read). Since then every command run during MY samples carries the repo path, which my preflight skips;
  during another lane's samples (whose preflight skips only its own repo) no command of mine names the pattern.
- **R85 (production wire `--tp-wire wht6`), minimal: GREEN.** plumb rows byte-identical to stock under wht6. Quality
  T2560 paired vs stock under wht6: last 512 −0.0025 [−0.0145, +0.0103]; whole tail (last 2,047) +0.0102 [+0.0025,
  +0.0176] -- the same metric on the exact wire (S1's rows, plumb = stock) is +0.0021 [−0.0126, +0.0157] and +0.0131
  [+0.0020, +0.0235]: the production wire adds nothing to KVA's cost.
- The held-cost controls of this session are in §14.

## 16. main merged into stage-e (b316096, 2026-10-05 ~19:25Z) -- merge-readiness
- main had moved (Stage B merged: 185e31b R56's min-bulk-rows gate, 1252304 R53' mixed set, 173dfc4 speed beside
  decoders, Stage C/F tools, release docs). `git merge-tree` showed one textual conflict (kva_config.h, two switch docs
  added at the same line: both kept) and the build would have hidden two semantic ones, fixed in the merge:
  - `RADIANCE_KVA_MIN_BULK_ROWS` defaulted on the placement (`c->place == PLACE_HOST`); host is the only placement
    here, so the default is 1,024 unconditionally; B's test no longer sets the retired PROJ_PLACE (refused by name).
  - B's speed-beside-decoders path projected with its own bf16-only GEMM after the two-slot `ring_next`: with an
    int8 folder it would have run a bf16 GEMM over int8 codes. It now goes through `project_rows` (`project_beside`),
    and the int8-vs-bf16 issue comparison covers that path too.
- main untracks the `evidence` symlink (1a53237), so the merge deleted this worktree's shared link: recreated at once
  from `git show 1a53237^:evidence` (no session had started; nothing written in between).
- Verification: S4 builds its frozen home from b316096 (the host suite, static case set included, runs in that build;
  falls back to 6a68dde if it fails) and measures everything on it; its mutants run on that source.

## 17. S4 correctness half on 5a3115b (2026-10-05 19:58-20:18Z, boot 75e3e39b; evidence/stagee/s4/session.log)
- Frozen home **5a3115b** (= 6a68dde's reductions + main merged + the static fixes + the straddle line): `ctest -LE gpu`
  3/3 (arch_static with every case: R74 media, straddle downgrade, int8 vs bf16 on the decoders path, B's set). arch
  14b7d304…, kva.so ff82ad62…. (S4's first attempt at 19:33Z and D1's at 19:34Z aborted at this step on 6a68dde/
  b316096: two cases read the pre-6a68dde log text, and the merge's 1,024-row gate sent small static chunks to stock.)
- **off ident = R3.** **KL rows byte-identical**: bf16 speed and quality T2048 = A′'s R144 rows (one slot, correction
  and row table in host memory, merged code); int8 speed and quality T2048 = S1's int8-ring rows (codes from b_h at
  layer S, no h_S). 67 approximate steps each.
- **Plugin VRAM a rank (startup line): bf16 50.0 MiB, int8 25.4 MiB** -- the ring's one slot, nothing else uploaded;
  host-mapped 1,228 / 637 MiB (maps, correction, row table). Arena: h_S 40 MiB (bf16 or MTP only), x_P + codes ~15 MiB,
  int8 stream codes 20 MiB.
- **B's 8192/10 config starts and serves** (quality, bf16: the larger slot): 9K prompt 7,253 ms, 8 decoders; pinned
  pool 9,782 / 9,735 slots.
- GPU-time audit (orchestrator ~20:15Z) applied from here: KL identity needs 2 runs (bf16 quality + int8 speed) next
  time; mutants run beside a correctness-only session (D2), never under timing or as a session's tail -- S4's tail
  finds no mutant scripts (renamed `mutant_*.py`); int8 arms first in D1/D2, bf16 arms skipped if
  `evidence/stagee/INT8_ONLY` exists (Dylan's decision pending).

## 18. DYLAN'S DECISION (2026-10-05 ~20:30Z, via the orchestrator): int8 is the only projector going forward
- **Shipped folder: `data/projector-qwen38fn-int8`** (shared data dir, `<radiance-kva>/data/projector-qwen38fn-int8`):
  28 files, 698,753,818 bytes; manifest `kva.json` sha256 **5b699e27e88d2e27cb546c174c6cb6f257e20433555e96b37fe17781aa3a85ae**
  (it lists every file's sha256; built by `tools/kva_projector.py int8 --from data/projector-qwen38fn`, d8f19eb).
  With the MTP final map (Stage D, `tools/kva_projector.py final`): `data/projector-qwen38fn-int8-final`, 29 files,
  908,489,957 bytes, `kva.json` sha256 fd6f3a28b0b9344117004d04d471be4d1e762991669ea7a3ca7c58b058e7cf1e -- the folder for
  deployments that draft (`--num-speculative-tokens` > 0), if R70 (D1) is green; without MTP the map is not declared.
- bf16 results already measured stay as history (§11, §15, S4's round a). Every bf16 arm not yet run is dropped: S4's
  bf16-a was stopped mid-sample at 20:31:52Z and held-bf16-b, bf16-b, p-bf16 are stopped at their start (a watcher,
  /tmp/stagee-skipbf16.py, kills the serve/settle/speed process whose stdout is that arm's file -- S4's own script
  is not edited while it runs; each such arm logs "serve FAILED" / "speed.sh FAILED"); D1's bf16+final and D2's
  quality-bf16 are skipped by `evidence/stagee/INT8_ONLY`. KL identity from here: int8 quality + int8 speed.
- The bf16 code path stays (it is the int8 builder's input); nothing in the release tests bf16.
