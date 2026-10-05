# NOTES — radiance-kva lab notebook

Provenance on every table: plugin commit, radiance `140987f` (v1.0.8), container sha256, flags, date.
Container: `qwen3.8-next-flash-fp8-iq4r-moe.rad` from HF `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`
(LFS sha256 `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20`, 121,969,901,568 B).

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
