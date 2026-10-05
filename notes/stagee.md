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

### Work in progress at the job limit (UNCOMMITTED in the worktree, not yet compiled -- ediag3 was measuring)
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
