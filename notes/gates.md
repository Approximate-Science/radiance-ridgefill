# notes/gates.md — GATES lane: engine-level gates for Stages 3–5 (GPU, full model)

Owner: GATES lane (2026-10-04). Every engine run below uses the frozen plugin home
`data/home-f60f893` (git archive of plugin commit **f60f893**; `qwen4exp_fp8.so` sha256 `3ed08f7d…`,
`kva.so` sha256 `f1cd5dae…`), unless a row says otherwise.

## Provenance (applies to every row unless stated)

| item | value |
|---|---|
| plugin commit | f60f893 (engine runs, `RK_PLUGIN_HOME=data/home-f60f893`) |
| radiance | 140987f (v1.0.8), runtime image `stilldeadcode/radiance:1.0.8` `sha256:34ec6b01…` |
| build image | `radiance-build` `sha256:335138ad…` (ROCm 7.2.4) |
| container | `qwen3.8-next-flash-fp8-iq4r-moe.rad`, pre-append sha256 `0af5e96244e8…ceaa4d20` (121,969,901,568 B), appended in place with 87 `kva.*` weights: `ls -l` = **123,364,379,720 B** |
| flags (RK_FLAGS) | `--tp 2 --kv-cache-dtype fp8 --tp-wire exact --max-num-batched-tokens 2048 --no-prefix-cache --num-speculative-tokens 0 --max-model-len 49152 --gpu-headroom-mib 3072 --placement expert_tiered --host-pool-mib 12288 --expert-vs-cache-ratio 0.82`; KL runs add `--max-num-seqs 1`, serves `--max-num-seqs 8` |
| boot id | `75e3e39b-cc5b-49de-8087-a4791f372a92` (= `evidence/stage0/boot_id`, the KL reference's boot), checked before every KL/speed run |
| KL reference | `data/kld/ref-stage0`, corpus `corpus/quick9.jsonl` (9 docs, Σ bulk chunks 67) |
| GPU lock | every engine sequence under `flock ~/AI-Work/radiance-kva-plugin-20261004/gpu.lock` |

Script changes in this lane (path-limited commits): `scripts/common.sh` gains `RK_DOCKER_EXTRA` (extra docker
args, one per word; used for the `RADIANCE_KVA_DUMP` writable mount) and passes `RADIANCE_PROFILE_EVERY`,
`RADIANCE_DEBUG_ROUTING` through; new `scripts/send_doc.py` (one prefill-only request from a document, ids
via the served tokenizer), `scripts/profile_steps.py` (per-step op lists from a `--profile-ops` log, by
differencing consecutive cumulative tables), `scripts/boundary_cos.py` (R17 cosine vs a tcc capture).

## Run log (UTC 2026-10-04 23:47 → ; boot 75e3e39b… throughout)

GPU lock: queued at 23:16 UTC behind another session's Qwen3-8B capture sweep (`kva-capture-sweep-qwen3-8b`, card 13,
launched 23:15:56 under the same lock); acquired 23:47:41. **Profiled runs** (never a TTFT source): `r16-profile-speed`,
`r17-dump-speed`, `profile-speed-st`, Stage 5 `profile-quality`. Dump runs: `r17-dump-speed`, Stage 5 R39 run.

### Gate 1 — KERNELS device test (R20 device half) — PASS
`docker run --rm --security-opt label=disable --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
-e ROCR_VISIBLE_DEVICES=1 -v <repo>:/kva -w /kva radiance-build sh -c 'cmake --build build-kernels-hip && ctest
--test-dir build-kernels-hip -L gpu --output-on-failure -V'` → `evidence/stage4/kernel-device-test.log`.
Built from kernels at **afce51b** (HEAD of `kernels/`, working tree clean; `ninja: no work to do`), i.e. f60f893's kernels
plus `kva_state_read`. rocminfo GPU agents: 0 = BDFID 768 (0000:03:00.0), **1 = BDFID 4864 (0000:13:00.0)**, 2 = gfx1036.
```
device 0 of 1 visible, PCI 0000:13:00.0
ok descriptions_cover_schemas / SKIP stub_refuses (R8 retired: every row implemented) / ok described_operands_launch
ok refuses_bad_operands / ok rho_matches_numpy_reference (max |rho diff| 1.55e-06) / ok rowsel_matches_fnlev_rules
state_correct shared + own slots, undo/apply, alpha 0/0.7/1, ND 0/1: 10031712 state bytes, 0 differ (max abs diff 0) x10
ok state_correct_device_matches_host / ok state_read_device_matches_host / ok rowsel_device_matches_host
device vs host: max |rho diff| 2.68e-07 -> ok rho_device_matches_host
546 check(s), group gpu        1/1 Test #2: kva_kernels_gpu ... Passed 1.99 sec
```
Kernel log clean before and after.

### Gate 2 — Stage 3 regression on home-f60f893 — PASS (byte-identical)
`RADIANCE_KVA_ST=refit scripts/grade.sh speed data/kld/ref-stage0 evidence/stage3/kld-fill-f60f893.json` (RK_EXPECT_APPROX=67)
→ 67 approximate steps; ppl ratio 1.0636, top-1 87.21%, KL mean 0.0818. `cmp` with `evidence/stage3/kld-fill.json.rows`
(home-35adbe3): **identical** (sha256 `26b31cee…`). Stage 4/5 code did not change the fill.

### Gate 3 — R16 (late layers skipped on approximate chunks) — GREEN; R26 projector half — GREEN
`RADIANCE_KVA_ST=refit RADIANCE_PROFILE_EVERY=1 RADIANCE_LOG_STEPS=1 scripts/serve.sh speed --profile-ops`, one request:
`scripts/send_doc.py --jsonl <quick ppl.jsonl> --name ppl/16k/0 --length 16384` (16,384 ids, sha256 `2b0f007d…`), log
`evidence/stage3/r16-profile-speed.log`, per-step lists `scripts/profile_steps.py … --json evidence/stage3/r16-steps.json`.
Steps 0–6 logged `kva: approximate step (speed, 2048 tokens, 2048 ahead)`, step 7 (ctx 14336) exact — 7 of 8 chunks, as T=2048
predicts. Approximate step 2, both ranks identical op sets:
```
layers <24 (614 issues): ... moe_gemm_q 48, router_topk 24, moe_scatter 24, attn_paged_gate_quant 6, hc_read 48,
                         ar_hc_write 24, ar_gather_hc_write 24, gdn_* 18 each, qsa_score/select 6 ...
layers >=24 (193 issues): gdn_chunk_scan 18, gdn_conv_prep 18, gdn_kkt_solve 18, gemm_nt 24 (op_ab / indexer proj),
                         gemm_nt_bias 24 (projector), gemm_nt_q 30 (GDN in 18 + attn k,v 12), quant_act_i8g 24 (fill codes),
                         kv_store 6, rmsnorm 6, rope 6, qsa_work 6, qsa_block_key 6, qsa_tail_store 6,
                         hc_read 1 (= the 97th connection, mixer.read, declared after layer 47)
forbidden on layers >=24 (moe*, router*, attn_paged*, hc_write, ar_hc_write, ar_gather_hc_write): NONE on either rank
exact step 7, layers >=24 (1098 issues): moe_gemm_q 96, router_topk 24, attn_paged_gate_quant 12, ar_gather_hc_write 48, ...
per late layer (rank 0 and rank 1 alike):
  GDN L24..26,28..30,...,46: gemm_nt_q gemm_nt gdn_conv_prep gdn_kkt_solve gdn_chunk_scan gemm_nt_bias
  attn L27,31,...,47:       gemm_nt_q gemm_nt_q rmsnorm rope kv_store gemm_nt qsa_work qsa_block_key qsa_tail_store gemm_nt_bias
```
(Per-layer lists are in declared order; the plugin's projector GEMM is declared after the in-tree graph, so it sorts last.)
Projector `gemm_nt_bias` ×24 per approximate step on **both** ranks (rank 0 1,737 µs/call, rank 1 1,464–1,538 µs/call).

### Gate 4 — R17 boundary equivalence — mean 0.986 (pass on the brief's ≥0.98-mean criterion; per-row criterion NOT met by 11% of rows)
Source text found (≈10 min): held docs are `plan-held.json` in the KVA research repo's private inputs
(`kva.data.INPUTS`, read via `qfn/steps.py capture_plan()` → `kva.data.calib_docs("held")`); `capture/held/<name>` holds the
RAW-text captures (chat copies live in `capture/chatheld`). Flash-Next tokenizer (`tokenizers`, no special tokens) vs every
stride-8 capture id: held0-rust 512/512, held1-code 603/603, heldc0 880/880, heldc1 858/858, heldc2 881/881 match.
Run: `RADIANCE_KVA_ST=refit RADIANCE_KVA_DUMP=/dump RK_DOCKER_EXTRA="-v <repo>/data/r17/dump:/dump" scripts/serve.sh speed
--profile-ops` (profiling so no pass is replayed from tape), `send_doc.py --name heldc0` (7,036 ids; the served ids equal every
capture id), log `evidence/stage3/r17-dump-speed.log`: chunks 0 and 2048 approximate (dumped), 4096 and 6144 exact.
`scripts/boundary_cos.py --dump data/r17/dump --capture heldc0/capture_0039{7,8,9}.pt --ids evidence/stage3/r17-ids.json`
→ `evidence/stage3/r17-boundary-cos.json`:

| chunk | rows (stride 8) | min | mean | median | p01 | norm ratio (median) |
|---|---|---|---|---|---|---|
| 0 (capture_00397) | 256 | 0.8183 (pos 1024) | 0.9842 | 0.9866 | 0.9305 | 1.0018 |
| 2048 (capture_00398) | 256 | 0.8660 (pos 3064) | 0.9880 | 0.9903 | 0.9477 | 1.0018 |
| pooled | 512 | 0.8183 | **0.9861** | 0.9887 | — | 11.3% of rows < 0.98 |

Reading: layers 0–23 run stock in speed mode, so this measures radiance's w4nl64-i8 container vs tcc's MXFP4-FP8-GPTQ engine at
the split, not plugin wiring (plumb == exact bytes already proved the wiring). Projector input matches the fit distribution on
average (norms equal to 0.2%); a tail of rows sits at 0.82–0.98.

### Gate 5 — Q6 arena cost (`--debug-placement`) — measured
`scripts/serve.sh <off|speed|quality> --debug-placement` (speed/quality with `RADIANCE_KVA_ST=shipped`), logs
`evidence/stage5/q6-placement-{off,speed,quality}.log`. Per rank (rank 0; rank 1 within 2 MiB / 50 units):

| | off | speed | quality |
|---|---|---|---|
| declared ops / buffers / KV groups | 1942 / 99 / 8 | 2003 / 99 / 9 | 2025 / 105 / 10 |
| activation arena from the buffer plan | 657.43 MiB | 657.43 MiB (**+0**) | 672.44 MiB (**+15.0**: h_R, x_R, y_R, rows, mask) |
| prefill staging (2 × one layer's non-resident experts) | 373.91 MiB | 426.18 MiB (+52.3) | 426.85 MiB |
| arena device total (I[0]) | 937.18 MiB | 989.45 MiB | 1005.14 MiB |
| static weights | 4.03 GiB | 5.25 GiB (+1.22: kva.proj 1.17 GiB, kva.st 54 MiB) | 5.25 GiB (+ rowsel.score 970 KiB) |
| expert budget | 19.93 GiB | 18.65 GiB | 18.63 GiB |
| resident expert units (gate_up), rank 0 / rank 1 | 16,793 / 16,840 | 15,707 / 15,755 (−6.5%) | 15,696 / 15,744 |

So making `b_h`, `x`, `x.q8`, `x.s8` concurrent costs **0 MiB** of arena (as ARCH predicted); quality adds 15 MiB. The real VRAM
cost of KVA is the replicated projector + correction (1.22 GiB a rank), which displaces ~1,090 resident expert units a rank
and grows the prefill staging by 52 MiB (fewer resident experts → larger non-resident set per layer).

### Gate 6 — Stage 4
(a) **R22 — GREEN.** `RADIANCE_KVA_ST=shipped RADIANCE_KVA_ALPHA=0 grade.sh speed … evidence/stage4/kld-alpha0.json`: 67 steps;
`.rows` **byte-identical** to `kld-fill-f60f893.json.rows`.
(b) **R23 — GREEN** (H4-st-paired confirmed; H4-st-nll refuted, above band). `RADIANCE_KVA_ST=shipped grade.sh speed …
evidence/stage4/kld-st.json`: 67 steps; ppl ratio 1.0442, top-1 87.46%, KL mean 0.0770, p99 0.578.
`tools/paired.py kld-fill-f60f893.json kld-st.json`:
```
ppl/16k/0 -0.022922  ppl/16k/1 -0.021070  ppl/16k/2 -0.018624  ppl/16k/3 -0.016324  ppl/32k/0 -0.025444
ppl/32k/1 -0.019282  ppl/8k/0  -0.010449  ppl/8k/1  -0.008509  ppl/8k/2  -0.023423
mean difference B - A: -0.018450  (95% CI [-0.021717, -0.014785])      all 9 docs improve
```
vs exact: +0.0432 [0.0277, 0.0602] nats (ratio +4.42%; band +1.5…+4%).
(c) **R24 — GREEN** (H4-swap confirmed). `RADIANCE_KVA_ST=swap … evidence/stage4/kld-swap.json`: ppl ratio 1.0717, top-1 86.27%,
KL 0.0982. Paired vs fill: **+0.0075 [+0.0037, +0.0113]** (8 of 9 docs worse) — swapped halves hurt, so the head order is right
and the correction is head-specific.
(d) **H4-st-ttft — confirmed by decomposition.** Same session, fill first: `serve.sh speed` (ST=refit) + `speed.sh fill-f60f893`,
then (ST=shipped) `speed.sh speed-st` → `evidence/stage4/speed-{fill-f60f893,speed-st}.json`:

| length | fill (this session) | fill+st | Δ | exact (stage 0) | speedup fill / fill+st |
|---|---|---|---|---|---|
| 9,216 | 5,421.8 (reps 4,767–5,703) | 4,871.2 | −10% (noise) | 5,923.2 | 1.09x / 1.22x |
| 16,384 | 6,356.5 | 6,373.9 | +0.27% | 10,002.6 | 1.57x / 1.57x |
| 32,768 | 10,106.5 | 10,116.6 | +0.10% | 19,788.6 | 1.96x / 1.96x |

TTFT CIs (unpaired, n=5) are −5…+8% wide: within each length every arm's reps fall monotonically (heat engine re-placing experts
for the doc mix), and 9,216 fill swung 4,767–5,703 ms. So the TTFT cannot resolve 1%. Decided on the op-level decomposition
(`RADIANCE_PROFILE_EVERY=1 serve.sh speed --profile-ops`, ST=shipped, same 16K prompt, `evidence/stage4/profile-speed-st.log`):
`kva_state_correct` ×36 per approximate chunk (18 undo + 18 apply) = **0.98–0.99 ms per chunk on rank 0, 1.7–2.0 ms on rank 1**
vs ~800 ms per chunk ⇒ ≤ 0.3% of TTFT. Everything else in the approximate step's op list is unchanged vs R16.

### Gate 7 — Stage 5
(a) **R41 — GREEN.** `RADIANCE_KVA_ST=shipped RADIANCE_KVA_ROWSEL_TABLE=none grade.sh quality … evidence/stage5/kld-quality-none.json`:
67 steps; ppl ratio 1.0442, top-1 87.46%; `.rows` **byte-identical** to `evidence/stage4/kld-st.json.rows` (speed + st). With no
row selected the whole compaction path (rowsel → gather → hc on h_R → MoE on cap padding rows → rho = 1) leaves the bytes alone.
(b) **R35 — GREEN (stop rule cleared).** `RADIANCE_KVA_ST=shipped RADIANCE_KVA_ROWSEL=all RADIANCE_KVA_SHARE=1
RADIANCE_KVA_ROWSEL_TABLE=all grade.sh quality … evidence/stage5/kld-quality-all.json`: 67 steps; KL mean **1.15e-7** (p99 6.9e-7,
max 1.2e-6) = the exact-vs-exact floor, top-1 **100%**, ppl ratio **1.0000**; `.rows` **byte-identical to the exact run**
(`evidence/stage0/kld-exact-vs-ref.json.rows`, sha256 `7be218f0…`) — stronger than the row asks (GEMM-shape noise allowed).
(c) **R36 — GREEN** (Q14 partly). `RADIANCE_KVA_ST=shipped RADIANCE_PROFILE_EVERY=1 RADIANCE_LOG_STEPS=1 RADIANCE_DEBUG_ROUTING=30
scripts/serve.sh quality --profile-ops`, the same 16K prompt, log `evidence/stage5/profile-quality.log`, steps
`evidence/stage5/profile-quality-steps.json`. Steps 0–6 `kva: approximate step (quality, …)`, step 7 exact. Profile tables are
cumulative and the last one also folds in the decode step, so only tables 2–6 (one approximate step each) are read.
Approximate step 2, late layers (both ranks): moe_gemm_q 48, router_topk 24, moe_scatter 24, hc_read 49, ar_hc_write 24,
ar_gather_hc_write 24, gather_rows 25 (24 block outputs + b_h→h_R), scatter_rows 24, kva_rowsel 1, kva_rho_update 18,
kva_state_correct 36, the full GDN and attention blocks (gdn_* 18, attn_paged_gate_quant 6, qsa_score/select 6, …), projector 24.
M per op, read two ways:
- `rad_route_counts` (RADIANCE_DEBUG_ROUTING=30, steps 0–2): **5,120 placements = 512 rows × top-10** on every approximate chunk
  (an exact chunk is 2,048 × 10) ⇒ late MoE at M = cap = 512:
  ```
  routing layer 30 step 0: 153 experts of 512 used, 5120 placements, max 389 at e9; first 24 = [385,385,386,385,386,388,385,385,385,389,1,0,...]
  routing layer 30 step 1: 162 experts of 512 used, 5120 placements, max 390 at e5
  routing layer 30 step 2: 162 experts of 512 used, 5120 placements, max 377 at e4
  ```
- per-call time, late vs early layers on the SAME approximate step (rank 0 / rank 1, µs): moe_gemm_q 317/916, 396/1106 (0.35);
  router_topk 26/39, 27/102; hc_read 228/686, 408/847; ar_hc_write 507/1758 (rank 1) ⇒ connection + MoE at a quarter of the
  rows. Blocks at n_tok: gdn_chunk_scan 191/197, gdn_kkt_solve 33/35, gemm_nt_q 228/247, attn_paged_gate_quant 1336/1310,
  qsa_score 57/59 (rank 0) ⇒ equal ⇒ M = n_tok.
**Finding (Q14 / speed): the cap's padding rows route as a block.** In ppl/16k/0 chunk 0 kva_rowsel kept k = 127 rows (= the R33
fixture's k), so 385 of the 512 cap rows are zero padding; each zero row routes to the same top-10 (experts 0–9, ~385 placements
each). The real rows touch ~143–152 other experts. Output is unaffected (a zero row's MoE output is 0; R41 is byte-identical),
but the late MoE does ~3× the rows it needs and pins experts 0–9 hot in every late layer. Possible lever (Stage 8 / ARCH): an
M = k issue is not allowed (host never reads k), but the padding could be routed to nothing (mask in the router) or the cap
lowered toward the observed max k.
