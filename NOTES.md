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
