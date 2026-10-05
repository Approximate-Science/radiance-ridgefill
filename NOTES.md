# NOTES — radiance-kva lab notebook

Provenance on every table: plugin commit, radiance `140987f` (v1.0.8), container sha256, flags, date.
Container: `qwen3.8-next-flash-fp8-iq4r-moe.rad` from HF `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`
(LFS sha256 `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20`, 121,969,901,568 B).

## Current state (headline numbers)

| Metric / Question | Latest Measured Numbers | Protocol | Source (file + section) & Commit |
|---|---|---|---|
| **(a) Quality headline** (R100) | **ΔNLL: +0.00212**<br>95% CI: [−0.01255, +0.01565]<br>ppl ratio: 1.0021<br>KL mean: 0.0368<br>top-1: 0.9156 (91.56%) | Quality mode, T 2560, last 512 tokens, paired per doc vs exact (9 docs quick9), bootstrap 95% CI, same boot (`75e3e39b…`), `RADIANCE_KVA_SCORE_BULK=1`, default guard unlimited.<br>*(Protocol/source agreement: Stage A [notes/impl.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/impl.md) and Stage A′ R144 [notes/aprime.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/aprime.md) match to the byte. In Stage A.1, the 64-row guard default ran 0 approx steps [exact, ΔNLL 0]; relaxing `STAGE_ROWS=512` or using A′ R144 unlimited guard restored +0.00212).* | [notes/aprime.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/aprime.md) §R144 Session 1 (lines 309–310), commit `0d75987`; also [notes/impl.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/impl.md) §4 Session 2 (line 445), commit `68d5d92` |
| **(b) Warmed TTFT speed & quality** (Published model file + projector folder) | **Prompt lengths: 9,216 / 16,384 / 32,768 ms**<br>• Exact baseline: 5,662 / 5,662 · 9,742 / 9,751 · 19,045 / 19,058<br>• **Quality**: 4,773 / 4,738 = **1.19x** · 6,816 / 6,797 = **1.43x** · 10,385 / 10,404 = **1.83x**<br>  (diff vs exact: −907 ms [−1,020, −793] / −2,940 ms [−3,119, −2,757] / −8,661 ms [−8,771, −8,364])<br>• **Speed**: 4,038 / 4,034 = **1.40x** · 5,245 / 5,244 = **1.86x** · 8,958 / 8,966 = **2.13x**<br>  (diff vs exact: −1,626 ms [−1,689, −1,569] / −4,501 ms / −10,090 ms) | Published pristine model file (`0af5e962…4d20`), folder beside model by discovery, warmed server (A.1 session 3 protocol: 1 pass of RK_REPS=2 across 3 lengths discarded; RK_REPS=7 reads reps 3–7; quiet host 1-min load < 2.5), settled medians a / b vs exact. Quality T 2048 (default unlimited guard), speed T 2048.<br>*(Protocol contrast: Stage A fresh-server protocol gave quality 0.91x / 1.07x / 1.38x and speed 1.22x / 1.52x / 1.97x; A.1 warmed protocol on appended file with 64-row guard gave quality 1.10x / 1.42x / 1.82x [9,216 slowed by guard fallback] and speed 1.41x / 1.85x / 2.12x).* | [notes/aprime.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/aprime.md) §R144 Session 2 (lines 316–332), commit `0d75987` |
| **(c) Speed mode quality** (last 512 & whole tail) | • **Last 512 tokens**: **ΔNLL: +0.02382 [+0.00292, +0.04388]**, KL: 0.0672, top-1: 0.8874 (88.74%)<br>  *(Also reported as +0.0238 [+2.4%] in NOTES.md §scoring-window; on 6 matched docs: +0.0272; on 13 final-tier docs: +0.0211)*<br>• **Whole tail (2,047 tokens)**: **ΔNLL: +0.0432 [0.028, 0.060]**, ppl ratio: 1.0442, top-1: 87.46%, KL mean: 0.0770 / p99 0.578<br>  *(On quick9-off1024 tail-only straddle: +0.04432, KL 0.0759, top-1 0.8809; on 13 final-tier docs: +0.0374)* | Speed mode (lean fill + correction st), T 2048, paired per doc vs exact on quick9 (9 docs), bootstrap 95% CI, same boot (`75e3e39b…`), `RADIANCE_KVA_SCORE_BULK=1`. Rows byte-identical across Stages 4, 5, A, A.1, and A′/R144. | Last 512: [notes/impl.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/impl.md) §4 Session 2 (line 447), commit `68d5d92`<br>Whole tail: [NOTES.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/NOTES.md) §Stages 3–5 (line 92), commit `f60f893`; [notes/impl.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/impl.md) §4 Session 2 (line 433), commit `68d5d92` |
| **(d) Projector placement costs** (VRAM vs Host Ring) | • **Memory / VRAM per card**:<br>  - VRAM placement: held 1.50 / 1.53 GiB (1,228.1 MiB uploaded by declare); slab slots rank 0: 15,742 / 15,765 (R141 serving) / 18,371 (R144)<br>  - Host Ring placement (`RADIANCE_KVA_PROJ_PLACE=host` + DD-L staging ring): held 434 / 430 MiB (R141) / 459.62 MiB (R144); 128.0 MiB VRAM + 1200.5 MiB host-mapped; slab slots rank 0: 16,713 / 16,716 (**+6.2%**, buys back ~970 slab slots a rank, −0.88 GiB card memory) / 19,296 (R144)<br>• **TTFT cost (16,384 / 32,768 ms, last-5 median)**:<br>  - VRAM: 9,657 / 13,536 ms<br>  - Host Ring: 10,230 / 14,942 ms (+573 ms [+175, +904] [**+6%**] at 16K; +1,405 ms [+22, +2,905] [**+10%**] at 32K; vs exact: +252 ms [−27, +423] [+2.5%, CI includes 0] at 16K, −4,978 ms [−5,863, −3,934] [−25%] at 32K)<br>  - Ring PCIe traffic: exactly 1.26 GB/pass/rank (24 × 2561 × 10240 × 2 B)<br>  *(Host zero-copy rejected: 27,096 ms at 16K, 54,104 ms at 32K, 2.7x exact)* | [notes/aprime.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/aprime.md) R141 (memory accounting) & R148 Session 3 (interleaved TTFT, fresh server, last-5 median pooled, n = 10, quick ppl prompts). | [notes/aprime.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/aprime.md) §4 Session 3 (lines 183–207) and §R144 (lines 312–313), commits `d3262dc`, `0d75987` |
| **(e) Investigation conclusion** (Radiance vs tcc) | • **Scoring window mismatch resolved**: Initial "much worse" gap was evaluating all 2,047 tail rows vs tcc's last 512. Rows 0–510 carry 2.5× the loss of rows 1535–2046 (+0.097 vs +0.038 fill; +0.069 vs +0.024 fill+st).<br>• **Identical window (last 512)**: Radiance matches tcc within tcc run-to-run noise on every arm.<br>  - 6 matched quick docs fill: radiance +0.0426 vs tcc +0.0414.<br>  - +st residual (−0.015 vs −0.026): proved to be tcc cross-boot noise (tcc same-server −0.019; tcc same arm across boots shifted −0.0073).<br>  - 13 final-tier docs (same server): fill radiance +0.0374 vs tcc +0.0455; fill+st radiance +0.0211 vs tcc +0.0260; +st effect −0.0163 [−0.020, −0.013] vs tcc −0.0196 [−0.026, −0.013] (paired diff +0.0032 [−0.0041, +0.0104]).<br>  - 19 pooled docs: +st effect −0.0160 vs −0.0193 (paired diff +0.0033 [−0.0029, +0.0094]; shortfall < 0.009 nats).<br>• **Correction verified on-engine**: cos(error, shipped C) = 0.927 vs 0.010 V↔K transposed; shipped C removes 24.7% error energy (alpha* 0.966); layer-24 tail error energy reduced to 0.506 (alpha 1, 18/18 layers improved) / 0.636 (alpha 0.5); negative control (swapped halves) 1.0005 (0/18 improved). | Evaluated on 6 matched quick docs, 13 final-tier 16k/32k docs, and 19 pooled docs, same boot, last 512 rows. On-engine GPU state captures with negative control ([tools/investigate/session.sh](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/tools/investigate/session.sh)). | [notes/investigate.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/investigate.md) §1–6 (lines 11–102), commit `f60f893` (KL) / `3af4eaa` (captures) |

## Run-wide deviations from HANDOVER (decided 2026-10-04)
- `--gpu-headroom-mib 3072` on every run (production default 96): Dylan's display is on the R9700 at
  0000:03:00.0 and he asked for headroom on that card. The flag is per-engine, so both cards keep 3 GiB.
  Applied to reference and candidate alike; absolute TTFT is not comparable with the author's 96-MiB numbers.

## Stage 0

Provenance for Stage 0–3 below: radiance `140987f` (v1.0.8); runtime image `stilldeadcode/radiance:1.0.8`
(`sha256:34ec6b01800f…`); compiler image `radiance-build` (built locally from 140987f, `sha256:335138adc1a7…`);
boot id `75e3e39b-cc5b-49de-8087-a4791f372a92`; date 2026-10-04 (times CDT). Flags = `scripts/common.sh` RK_FLAGS:
`--tp 2 --kv-cache-dtype fp8 --tp-wire exact --max-num-batched-tokens 2048 --no-prefix-cache
--num-speculative-tokens 0 --max-model-len 49152 --gpu-headroom-mib 3072 --placement expert_tiered
--host-pool-mib 12288 --expert-vs-cache-ratio 0.82` (+ `--max-num-seqs 8` serving, `1` in KL mode).

### Container (R-pre)
Downloaded 16:57–17:32 with radiance `scripts/hfget.sh`; `sha256sum` = `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20`
= the HF LFS oid (MATCH; `evidence/stage0/container.sha256`). Metadata: `evidence/stage0/rad-info-meta.txt`.
Load: 65.99 GiB placed per run; experts 19.93 GiB resident a rank (16,847 slab slots vs 7,783 pinned-pool slots);
both cards settle at 3.09 / 3.17 GiB free (the headroom).

### R3 — reproducibility: GREEN
`scripts/ident.sh` on three boots of the stock engine (`evidence/stage0/ident-exact-boot{1,2,3}.txt`), identical:
```
temp 0     dc7115567e4d85e4  baba87f4bfdebdca
temp 0.7   74624bdf06eb6837  bc36faa6cefaa6fe
temp 1.0   aa87d500fb3831d1  fc9a051f9310d638
```

### R1 — stock TTFT baseline: GREEN (`evidence/stage0/speed-exact.json`, boot 3, median of 5 after 1 warm-up)
| prompt | median ms | min | max | prefill tok/s |
|---|---|---|---|---|
| 9,216 (exact tail 3,072) | 5,923.2 | 5,904.9 | 5,974.5 | 1,556 |
| 16,384 | 10,002.6 | 9,937.1 | 10,005.6 | 1,638 |
| 32,768 | 19,788.6 | 19,483.0 | 19,925.3 | 1,656 |
For scale: tcc's exact engine took 9.91 / 16.21 / 32.75 s at the same lengths (KVA-FACTS §6.1) — radiance stock is
already ~1.65x faster, so KVA's ratios here are not comparable to tcc's ratios one for one.

### R2 — KL reference: GREEN
`scripts/grade.sh record data/kld/ref-stage0 corpus/quick9.jsonl` → 18,423 positions, 9 docs, 107 s, 8.6 GB.
Corpus: `tools/kld_corpus.py`, docs cut to 3×8,192 / 4×16,384 / 2×32,768 tokens, `score_from = N − 2048`; server
`/tokenize` gives exactly those counts. Σ bulk chunks = 67.
**Noise floor** (stock vs reference, `evidence/stage0/kld-exact-vs-ref.json`): KL mean 1.15e-7, max 1.21e-6, top-1
1.0000, ppl ratio 1.000000 (the reference stores f16 log-probs; the engine is deterministic).

## Stage 1–2 (engine-side checks; code-side rows are in notes/arch.md, notes/kernels.md)
- **Q12 answered: NO** — `rad-convert --reuse --in-place` needs the full source namespace (config, tokenizer, every
  declared tensor). Way through: a header-only stub of `Qwen/Qwen3.8-Flash-Next@de4b8e4d` (131 sparse shards holding
  only the real safetensors headers, 23 MB on disk) + the sidecar shard (`notes/sidecar.md` §7).
- Append (17:56, `evidence/stage2/append.log`): `--in-place: 51489 weight(s), 113.47 GiB, kept where they were …; 87
  added past its end`; file now 123,364,379,720 B; restore record `…rad.pre-append` (256 B). Plugin home used:
  `data/append-home` (frozen copy). Plan-vs-container gate after: planned 51,576, held 51,576, new 0, dropped 0,
  changed 0 → PASS. Metadata diff = exactly the 13 `kva.*` keys. Recipe unchanged.
- **R13 GREEN**: `tools/kva_sidecar.py verify` on the appended metadata → PASS (7 source hashes OK).
- **R12 GREEN**: stock engine on the appended container → ident identical to R3 (`evidence/stage2/ident-exact-appended.txt`).
- **R7 GREEN** (Stage 2): plugin `off` (home `data/append-home`) → ident identical to R3; log shows
  `qwen4exp_fp8.so is shadowed by /plugins/architectures/qwen4exp_fp8.so` (R5 evidence).
- Docker findings: `--network host` hides every GPU under rootless Docker (each flag tried alone); the runtime image
  has no video/render group names; `--security-opt label=disable` needed.

## Stage 3 (plugin home `data/home-35adbe3`, built from `git archive 35adbe3`)
> Note: Superseded — R19 TTFT numbers below used the fresh-server protocol (superseded by the warmed-server protocol in Stage A.1; see [notes/impl.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/impl.md)).
- **R6 GREEN**: `--debug-graph` stock vs plugin-off: 116,057 vs 116,058 lines; sorted diff (minus a timing line and the
  shadow line) empty. Unsorted diff shows only rank-interleaved log lines and an unordered summary table.
- **R14 GREEN** (stronger than ident, whose prompts are too short to trigger any approximate chunk): KL mode, plumb,
  `RADIANCE_KVA_ST=refit` → 67 approximate steps; KL mean 1.15e-7 / max 1.21e-6 / top-1 1.0 / ppl ratio 1.0 —
  identical to exact-vs-exact. Also answers **Q15**: `--max-num-seqs 1` + 2048-cut docs ⇒ approximate-step count ==
  bulk-chunk count (67).
- Engine refusal found and fixed: `'kva.st.24' at --tp 2: a ROW shard of a rank-3 weight is not defined` → ARCH
  8b4266e replicates kva.st (54 MiB a rank) and slices this rank's heads at issue (deviation from PLAN D6).
- **R18 (fill only, no correction)** — `evidence/stage3/kld-fill.json`, 67 approximate steps:
  KL mean 0.0818, p99 0.624, top-1 87.21%, ppl 8.1771 vs 7.6878 → **ratio 1.0636**; paired per-doc ΔNLL **+0.0617
  [+0.0441, +0.0796]** (labbook `exact-stage0..fill-stage3`). Positive control: KL 0.08 vs floor 1e-7.
  Per doc ΔNLL: 8k +0.024/+0.026/+0.052, 16k +0.086/+0.073/+0.089/+0.049, 32k +0.047/+0.109.
  Prediction H3-fill-nll (+3…+6%) **refuted, just above band**; stop rule (>+8% or top-1 <85%) not met → continue,
  Stage 6 refit pulled forward (PLAN Q4).
- **R19 (fill only)** — `evidence/stage3/speed-fill.json`, same boot: 4,928 / 6,419 / 10,163 ms median →
  **1.20x / 1.56x / 1.95x**. H3-fill-ttft (1.3–1.7x at 16K) **confirmed**. 150 approximate steps logged = 6 requests ×
  (3 + 7 + 15) bulk chunks.

## Stages 3–5 engine gates (GATES lane; plugin home `data/home-f60f893`; full detail `notes/gates.md`)
> Note: Superseded — TTFT numbers below used the fresh-server protocol (Stage A.1 and Stage A′ R144 introduced the warmed-server protocol); compaction quality was superseded by Stage A's device mask (see [Stage A](#stage-a), [Stage A.1](#stage-a1), and [notes/impl.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/impl.md)).
Same boot as the KL reference throughout; kernel log clean; profiled runs never used for TTFT.

| arm | ppl ratio | ΔNLL vs exact [95% CI] | top-1 | KL mean / p99 | TTFT 9,216 / 16,384 / 32,768 ms (speedup) |
|---|---|---|---|---|---|
| exact | 1.0000 | — | 100% | 1.15e-7 / 6.9e-7 | 5,923 / 10,003 / 19,789 |
| fill | 1.0636 | +0.0617 [0.044, 0.080] | 87.21% | 0.0818 / 0.624 | 5,422 / 6,357 / 10,107 (1.09 / 1.57 / 1.96x) |
| fill + st | 1.0442 | +0.0432 [0.028, 0.060] | 87.46% | 0.0770 / 0.578 | 4,871 / 6,374 / 10,117 (1.22 / 1.57 / 1.96x) |
| swapped heads | 1.0717 | +0.0692 [0.050, 0.089] | 86.27% | 0.0982 / 0.730 | — |
| quality (class 25%) | 1.0238 | +0.0236 [0.012, 0.036] | 88.32% | 0.0677 / 0.506 | 7,441 / 10,972 / 20,687 (**0.80 / 0.91 / 0.96x**) |
| random rows | 1.0287 | +0.0283 [0.016, 0.041] | 88.39% | 0.0655 / 0.504 | — |
| all rows | 1.0000 | 0 (byte-identical) | 100% | 1.15e-7 / 6.9e-7 | — |

Green: R16, R17 (mean cosine 0.986, median 0.989, min 0.818; 11% of rows < 0.98 — passes the mean criterion, not the
per-row wording), R20 device (546 checks, 0 differ), R22, R23, R24, R26 (projector half), R35, R36, R37, R38, R39
(Jaccard 0.917), R41; Q6 (concurrent buffers 0 MiB; quality +15 MiB; projector 1.22 GiB static a rank = −1,090 resident
experts, −6.5%); Q14 (late MoE at M=512 touches ~150–160 of 512 experts per layer; ~385 cap rows are zero padding
routed to experts 0–9). **Red: R40 — quality mode is slower than exact.** Measured host→device per KL run: exact 428 GiB,
speed+st 291, quality 516 (flat in rows selected: none 515, all 529) → the expert stager copies every late layer's pooled
experts on a >1,024-token step even though the compacted pass routes ~30% of experts. Under investigation (ARCH).
Verdicts: H4-st-paired confirmed (−0.0185 [−0.0218, −0.0147], 9/9 docs improve); H4-st-nll refuted (+4.42%, just above
+1.5…+4%); H4-swap confirmed (worse, +0.0075 [0.0036, 0.0113]); H4-st-ttft confirmed on op timing (0.99 ms/chunk,
≤0.3%); H5-quality-vs-speed confirmed (−0.0197 [−0.0250, −0.0141]); H5-quality-nll refuted (+2.38% > +1.5%);
H5-class-vs-random refuted on size (−0.0047 [−0.0094, −0.0005], CI excludes 0 so R37 passes); H5-rows-share confirmed
(5.61%); H5-jaccard refuted high (0.917 > 0.9); H5-quality-ttft refuted (0.91x).

## Stage 6 — retune on radiance (REFIT lane; details notes/refit.md; evidence evidence/stage6/)
- **R42 GREEN**: projector refit from radiance captures (151 docs raw+chat, 248,892 + 249,312 rows, chat share 0.5,
  lambda 0.03 = shipped's), captures read by fit.py's Sums unchanged; HC identity check 0.99996. Held-out block-input
  cosine on radiance captures: refit 0.7536 vs shipped 0.7501, paired over 24 layers +0.0035 [0.0029, 0.0041]
  (H6-refit-cos refuted on size: band +0.005…+0.04). Appended as `kva.projr.*` (append #1, 20:37, kva.mode=off).
- **R43 GREEN (counts)**: correction refit with the refit projector, 14 prompts (13 shipped minus sterm4 + 2 clean b2),
  55 chunk ends per layer per rank, tail 512; refit vs shipped C cosine 0.92 (0.001 with halves swapped — head order
  verified); appended as `kva.str.*` (append #2, 20:43).
- **R44 — shipped weights KEPT.** Paired per-doc tail NLL, refit − shipped, same boot, plugin home-f60f893:
  fill −0.0028 [−0.0060, +0.0005]; fill+st +0.0007 [−0.0018, +0.0030]; quality (25%) +0.00005 [−0.0030, +0.0030].
  No CI excludes 0 → a refit kept on cosine alone would fail R44; shipped stays the default (`RADIANCE_KVA_PROJ/ST=shipped`).
  Reading: the radiance-vs-tcc gap (+6.4% vs +4.3% fill) is not the projector's fit; radiance's own late-layer inputs
  differ from tcc's by 0.06–0.09 cosine and the shipped map already predicts them almost as well as a radiance fit.
- **Share sweep** (quality, refit weights; speed cost not measurable until the Stage A rework lands):
  | share | ppl ratio | top-1 | KL mean |
  |---|---|---|---|
  | 0.10 | 1.0334 | 87.86% | 0.0719 |
  | 0.25 | 1.0239 | 88.35% | 0.0672 |
  | 0.50 | 1.0164 | 88.54% | 0.0626 |
  0.10→0.25 −0.0093 [−0.0132, −0.0056]; 0.25→0.50 −0.0073 [−0.0109, −0.0040]. Default stays 0.25 (Dylan's call).

## Why the numbers looked "much worse" than tcc's (2026-10-04 ~21:15) — scoring-window mismatch + one residual
> Note: Resolved — the +st residual below was proved to be tcc cross-boot noise, and GDN correction was verified on-engine (see [Current state](#current-state-headline-numbers) and [notes/investigate.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/investigate.md)).
tcc's quick tier (`~/AI-Work/kva-flashnext-iterate/tests/results/`, same 9 ppl docs) scored the LAST 512 tokens
(`scored_tokens: 512`, `nll_all` / `nll_novel`), against an exact run from another boot. Our KL runs score the whole exact
tail (`score_from = N − 2048`, 2,047 positions). The first ~1,500 tail positions sit right after the approximated region
and are hurt far more than the last 512, so our headline ratios were not comparable with tcc's (or with HANDOVER §6's
bands, which came from tcc's last-512 numbers). Rescored from our `.rows` files (col 1 = candidate NLL, col 2 = reference):
| arm | ours, last 2,047 | ours, last 512 | tcc quick tier, last 512 (all 9 docs) |
|---|---|---|---|
| fill | +0.0617 (+6.4%) | +0.0381 (+3.9%) | +0.0362 (+3.7%) big-nost |
| fill + st | +0.0432 (+4.4%) | +0.0238 (+2.4%) | +0.0190 … +0.0237 (iter-04 / iter-02 big-st) |
| quality 25% | +0.0236 (+2.4%) | +0.0065 (+0.65%) | +0.0031 (+0.31%) |
On the 6 docs whose token ranges are identical in both (16k + 32k; tcc's 8k docs were 9,216 tokens, ours cut to 8,192),
last 512: fill ours +0.0426 vs tcc +0.0414 (same); fill+st ours +0.0272 vs tcc +0.0152 (**ours 0.012 nats worse**);
quality ours +0.0038 vs tcc −0.0028 (0.007 worse; tcc's cross-boot noise ≈ ±0.004–0.006 on such means).
Conclusion: the projector fill behaves exactly as on tcc's engine; the residual is that the GDN correction (and so
quality mode built on it) removes less of the loss on radiance (−0.015 vs tcc −0.026 on those 6 docs). Under investigation.

## Stage A

Full detail in [notes/impl.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/impl.md) (§1–5). Commits `42ac7fd`…`68d5d92` (final `b6979d4`); boot `75e3e39b…`; homes `data/home-{0fd3a65,68d5d92,b6979d4}`.
- **Built & mechanisms**: DD-F device mask (in-place `kva_mask`, `kva_select`, `kva_drop_rows`) replaces compaction; zero-row probes behind layer S-3 replace plan DD-B (weightless alternate not implementable in radiance v1.0.8, §2).
- **Green static & kernel rows**: R46, R98 (kernel host 737, gpu 945 checks); R52', R72, R93, R53' single-prefill (static 40 cases, 21 mutants caught).
- **Green correctness & policy gates**: R94 (probes forced stream, 3/3 boots byte-identical); R47 (forced split, negative control catastrophic); R35' (quality all-rows byte-identical); R45 (mask rule matches 5/5 prompt lengths); R81 (env-only mode switch); R83 (engine 1.0.9 forwards to stock, KVA off); R84 (tail refusal on small batched tokens); R73 (KL serves stock unless SCORE_BULK=1); R49 (offset split +0.0040 [-0.0016, +0.0097]); R50 (split kept, beats end by -0.0088 [-0.0112, -0.0065] whole tail); R51 (+2048 worse, CI excludes 0; -64 not worse); R36' (routed expert counts match fixture); R95 mechanism (rank-1 wait 21.7 -> ~10.5 ms/layer; mover copies 518.6 -> 287 GiB); R99 soak (400 mixed requests, 0 audit failures).
- **Red / measured rows**: R48 refuted at 9,216 (speed 1.22x vs band 1.35–1.50x; fresh server 16K 1.52x, 32K 1.97x); R95 TTFT (fresh server: quality 16K 1.07x under band, 32K 1.38x over band; 9,216 is 0.91x, slower than exact); R96 (streaming beats stock staging at all measured exact-row counts 64–1,984; no crossover); DD-B (alternate handle refused).
- **Key quality numbers (paired vs exact, 9 docs)**:
  - Quality R100 headline (T 2560, last 512): dNLL +0.00212 [-0.01255, +0.01565], KL 0.0368, top-1 0.9156 (ppl ratio 1.0021).
  - Quality T 2048: whole tail (2,047) +0.02282 [+0.01155, +0.03464], last 512 +0.00576 [-0.01307, +0.02245], KL 0.0592, top-1 0.8950.
  - Speed T 2048: last 512 +0.02382 [+0.00292, +0.04388], KL 0.0672, top-1 0.8874 (whole tail identical to Stage 4 speed+st: +0.0432).
- **Key TTFT numbers (fresh-server protocol, median of 5 after 1 warm-up, 9,216 / 16,384 / 32,768 ms)**:
  - Exact: 5,994 / 10,014 / 20,083; speed: 4,915 (1.22x) / 6,573 (1.52x) / 10,222 (1.97x); quality: 6,584 (0.91x) / 9,342 (1.07x) / 14,510 (1.38x).

## Stage A.1

Full detail in [notes/impl.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/impl.md) (§Stage A.1). Commits `5102d9c`, `1a524d8`; home `data/home-1a524d8`; boot `75e3e39b…`.
- **Built**: Speed straddling chunk (tail rows [b, n) only for late blocks; K/V, indexer, GDN whole step); quality guard (`STAGE_ROWS` default 64).
- **Green static & correctness**: 13/14 mutants caught (A12 equivalent on served shapes); 42 static cases; R94/R47 plumb identical to exact; regression speed/quality T 2048 identical to Stage A; R6/R7 ident = R3.
- **Two-path KL oracle (speed off1024, whole tail)**: tail-only vs masked paired diff +0.00018 [-0.00148, +0.00158] (HA1-kl-oracle green).
- **Quality gain on off1024**: A.1 (straddles exact) vs Stage A (straddles masked): whole tail -0.00755 [-0.01187, -0.00323] (better).
- **Instrument update**: Fresh-server protocol discarded (approximate reps fall over 7 reps; heat engine); replaced by warmed-server protocol (session 3: RK_REPS=2 warm-up across all 3 lengths discarded, RK_REPS=7 reads reps 3–7; quiet host 1-min load < 2.5).
- **R48 confirmed on warmed server**: Speed 9,216 = **1.41x** (in 1.35–1.50x band); 16K = **1.85x**; 32K = **2.12x**. Tail-only straddle saves -788 ms [-889, -680] vs running exact (refuting the handoff's ~1.34x ceiling).
- **Quality TTFT warmed**: 9,216 1.10x (5,138 ms) / 16K 1.42x (6,888 ms) / 32K 1.82x (10,478 ms), all CIs exclude 0.
- **Red / open findings**: Guard premise refuted on warmed server (streaming beats stock step at 512/1,024/1,984 exact rows by ~220–240 ms; guard off saves -331 ms [-551, -108] at 9,216, giving 1.18x); R100 default ran 0 approx steps (exact) due to 64-row guard (STAGE_ROWS=512 restores +0.00212 headline at 1.04x / 1.24x / 1.36x); ON server static VRAM (5.25 vs 4.03 GiB) costs ~+220 ms on stock chunks.
- **Key TTFT numbers (warmed server, settled medians a / b, 9,216 / 16,384 / 32,768 ms)**:
  - Exact: 5,672 / 5,672 · 9,747 / 9,752 · 19,052 / 19,053.
  - Speed (tail-only): 4,041 / 4,025 (1.41x) · 5,271 / 5,259 (1.85x) · 8,997 / 8,987 (2.12x).
  - Quality (guard 64): 5,138 / 5,139 (1.10x) · 6,888 / 6,884 (1.42x) · 10,478 / 10,478 (1.82x); guard off: 4,807 (1.18x) at 9,216.

## Stage A′ (projector folder) and R144

Full detail in [notes/aprime.md](file:///home/dylan/AI-Work/scratch/agent-workers/55fcdebb-014a-4bf5-acde-fa3085f0c2a4/repo/notes/aprime.md). Commits `3d69eda`…`02b674e`, `0d75987`; homes `data/home-{d3262dc,0d75987}`; boot `75e3e39b…`.
- **Built**: Projector folder loader replaces container append; `kva_gemm_nt_bias` forwards libr4d device row (R140); discovery logic (R149); manifest verification with refuse/warn split (R142); per-rank allocation slicing correction heads (27 MiB vs 54 MiB, R141); staging ring for host placement (DD-L, R148); guard default changed to unlimited (commit `0d75987`).
- **Green gates before R144**: R140 (980 GPU checks, libr4d match bytewise; plumb x3 identical); R149 (symlink, inode, dir mount; env wins; file-only -> stock); R142 (wrong dims/tokenizer/missing file refused by name -> stock graph & R3 ident; 3-warning variant runs identical bytes); R141 (held +1229.75 MiB vram / +128 MiB host ring; resident experts −0.02…+0.15% vram, +6.2% host ring); R143 (plumb, speed, quality KL rows identical to append; TTFT within 1%, CIs include 0); R148 (staging ring default: +6%/+10% vs vram TTFT, buys back ~970 slab slots; zero-copy rejected at 2.7x exact).
- **R144 re-run on PUBLISHED model file** (sha256 `0af5e962…4d20`, 121,969,901,568 B; folder `data/projector-qwen38fn`; commit `0d75987`):
  - Clean stock execution: off mode and missing folder serve stock (debug graph 0 lines diff; ident = R3).
  - KL correctness: plumb, speed, quality T2048, quality T2560, host ring identical to append runs; no `kva.*` weights read from container.
  - **R100 headline verified**: Quality T 2560 last 512: dNLL +0.00212 [−0.01255, +0.01565], ppl ratio 1.0021, KL 0.0368, top-1 0.9156.
  - **Warmed TTFT on published file (Session 2, medians a / b, 9,216 / 16,384 / 32,768 ms)**:
    - Exact: 5,662 / 5,662 · 9,742 / 9,751 · 19,045 / 19,058.
    - Quality: 4,773 / 4,738 (**1.19x**) · 6,816 / 6,797 (**1.43x**) · 10,385 / 10,404 (**1.83x**); diff: −907 / −2,940 / −8,661 ms.
    - Speed: 4,038 / 4,034 (**1.40x**) · 5,245 / 5,244 (**1.86x**) · 8,958 / 8,966 (**2.13x**); diff: −1,626 / −4,501 / −10,090 ms.
    - All approximate arms faster than exact at every length (CIs exclude 0); quality 9,216 is −383 ms faster than A.1 (guard trade taken).
