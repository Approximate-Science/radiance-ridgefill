# Sidecar lane — lab notes (tools/, tests/test_*.py, tests/fixtures/)

Date 2026-10-04 (local evening). Plugin repo commits: `c211153` (sidecar tool), `caf4dc1` (corpus, fixture,
rows_compare). radiance `140987f` (v1.0.8), read only. Container not yet downloaded/appended, so no container
sha256 applies to anything here. Python: the research venv (numpy 2.5.3, safetensors 0.8.0, torch
2.11.0+rocm, transformers 5.18.0, pytest 9.1.1); `tools/requirements.txt` pins these. Nothing here touched a
GPU.

## 1. Inputs verified (HANDOVER §3)

```
$ cd ~/AI-Work/kva-flashnext-hf/model/ && sha256sum -c SHA256SUMS
kva-big-s24.safetensors: OK
st/kva-big-s24-his-s24.done.json: OK
st/kva-big-s24-his-s24.rank0.pt: OK
st/kva-big-s24-his-s24.rank1.pt: OK
```
(`kva-big-s24.safetensors` there is a symlink to the HANDOVER path `~/AI-Work/kva/qfn/proj/kva-big-s24.safetensors`.)

- **Projector**: 25 tensors, `layer.24`..`layer.47` [2560, 10241] BF16 + `final` [10240, 10241] BF16. Header
  metadata: split 24, lambda 0.03, train_rows 497846.0, fit "centred ridge, unpenalised bias", checkpoint
  `67c6f9ca…`.
- **Correction `.pt`** (each rank): `{"sum": {L: f32 [24,128,128]}, "count": {L: 62}}`, nothing else; layers
  24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46 (the 18 late GDN layers) in both files.
- **Unigram table**: one tensor `logfreq` [248320] f32 = **natural-log frequency, already a log**
  (metadata `smoothing: add-one`, 1,147,875 tokens, 99 docs, 77,096 seen ids, vocab 248320); range
  [−14.149, −3.384], all finite. So rarity = −logfreq. Every unseen id shares the maximum rarity 14.149, so
  those rank first, ties by position (faithful to the rule).
- **Vocab**: the model has **248,320** rows (`text_config.vocab_size` in the tcc checkpoint's config.json and
  in the source repo's — fetched `Qwen/Qwen3.8-Flash-Next/config.json`, 4.7 KB). The tokenizer has 248,077 ids:
  248,044 base + 33 added (248,044..248,076, all special/added → never kept). Ids 248,077..248,319 are padding
  rows → −inf. Freq table length = 248,320 = model vocab (asserted at build).
- **Tokenizer identity with the container (checked by hash, not assumed)**: the container README says its
  vocab was baked from `Qwen/Qwen3.8-Flash-Next/tokenizer.json`. HF lists that file's LFS sha256 as
  `0997f410…29b9f3` = sha256 of the tcc checkpoint's `tokenizer.json` (the `--tokenizer` I build with);
  `tokenizer_config.json` git blob `5de744b3` is equal too. So the score table is indexed by the container's
  own ids.

## 2. Sidecar build (Stage 2 step 1)

```
$ tools/kva_sidecar.py build --proj ~/AI-Work/kva/qfn/proj/kva-big-s24.safetensors \
    --st .../st/kva-big-s24-his-s24.rank0.pt .../rank1.pt \
    --freq ~/AI-Work/kva/pred/tcc-qwen38-flash-next-mxfp4-fp8-gptq-freq.safetensors \
    --tokenizer <models-boot>/tcc-qwen38-flash-next-mxfp4-fp8-gptq --out data/sidecar
split 24; 24 projector layers; 18 correction layers; rowsel: 167288 of 248320 vocab rows kept (248077 tokenizer ids)
```
7 s. Output in `data/sidecar/` (git-ignored, /var/home filesystem):

| file | size | note |
|---|---|---|
| `kva-sidecar.safetensors` | 1,374,648,720 B (1.28 GiB) | **sha256 `05c4e088dc9a011041f547329673877b9ad10e273dcf58624ec99c786098ef46`**, 87 tensors; byte-identical across rebuilds |
| `model.safetensors.index.json` | 4.2 KB | weight_map names only the shard |
| `rad-convert-set.txt` | 712 B | the header metadata as `key=value` lines, for `rad-convert --set` |

Tensors (names are the CHECKPOINT tensor names the ARCH lane name-maps; identity logical names recommended so
`rad-info -v` shows exactly these, as R12 expects):

| name | shape / dtype | count | source |
|---|---|---|---|
| `kva.proj.L.weight` | [2560, 10240] bf16 | 24 (L 24..47) | `layer.L[:, :10240]` |
| `kva.proj.L.bias` | [2560] bf16 | 24 | `layer.L[:, 10240]` |
| `kva.st.L` | [48, 128, 128] f32 | 18 | `cat(rank0.sum/count, rank1.sum/count)`, rank 0's heads first |
| `kva.stswap.L` | [48, 128, 128] f32 | 18 | halves swapped (R24 negative control) |
| `kva.rowsel.score` | [248320] f32 | 1 | −logfreq if class ∈ {cap, mixed, piece}, else −inf |
| `kva.rowsel.score_none` | [248320] f32 | 1 | all −inf (R41) |
| `kva.rowsel.score_all` | [248320] f32 | 1 | all 0: every id a match, ties by position (R35 all-rows) |

Header metadata (also in `rad-convert-set.txt`):
```
kva.format=kva-sidecar-1
kva.rowsel.classes=cap,mixed,piece
kva.rowsel.share=0.25
kva.split=24
kva.src.config.sha256=238b0a0024b5cfb7ebc1df0c8d122c01c1b24e0843b2c9395740eabc43637e91
kva.src.freq.sha256=4f1b719a59fdaa6ddfc7e06502b817c5a96e6a00bca060136d110e672059f97a
kva.src.proj.sha256=c9db6064eb275d3166df8139239959dfdeb64f2778707a2f42215164999cb9cd
kva.src.st0.sha256=38e4b550dfc1419922feccb59f75f758be7e8f2468c84b51a30712914b0b5dd3
kva.src.st1.sha256=af517cf760f7baef3f514c48ec4ab36d31effa444641cd6a0f87fdae2e384a18
kva.src.tokenizer_config.sha256=b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27
kva.src.tokenizer_json.sha256=0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
```
(`config` = the tokenizer dir's config.json, read only for the vocab size.)

Independent check against the source files (one-off script, not the tool's code path): all 24
`kva.proj.L.weight/.bias` equal the source slices; all 18 `kva.st.L`/`kva.stswap.L` equal the rank halves in
the stated order; `score` keeps 167,288 ids, none of the 33 specials nor the 243 padding rows; kept scores in
[6.540, 14.149]; `score_none` all −inf; `score_all` all 0.

`verify` (R13 against the sidecar's own header; the container form waits for the append):
```
$ tools/kva_sidecar.py verify data/sidecar/kva-sidecar.safetensors --proj … --st … … --freq … --tokenizer …
OK        kva.src.config.sha256 238b0a00…   (7 hashes OK)
          kva.format = kva-sidecar-1 / kva.split = 24 / kva.rowsel.share = 0.25 / kva.rowsel.classes = cap,mixed,piece
verify data/sidecar/kva-sidecar.safetensors: PASS
```
For the container: `rad-info --meta <rad> > evidence/stage2/rad-info-meta.txt` inside `radiance-build`, then
`tools/kva_sidecar.py verify evidence/stage2/rad-info-meta.txt <same source flags>` (a `.rad` target also works
when `--rad-info` names a rad-info runnable on the host path).

**Bug found and fixed, with its regression test**: the first two builds of the same inputs had different
sha256s. Only the header differed: safetensors (Rust) writes `__metadata__` in hash order. The tool now
rewrites the header with sorted metadata, same length, in place
(`test_two_builds_of_the_same_inputs_are_byte_identical` fails on the old code, passes now).

## 3. Tests

`python -m pytest tests/` (no env): **44 passed, 3 skipped** (the three need real data). With
`KVA_SIDECAR=data/sidecar/kva-sidecar.safetensors KVA_TOKENIZER=<checkpoint dir> KVA_RESEARCH_ROOT=<research repo>`:
**47 passed**. Covered: projector weight/bias split, bf16 kept, `final` not converted, st = sum/count with
rank-0 heads first, swap variant, score = rarity for kept classes and −inf for dropped classes / specials /
padding, score_none / score_all, metadata hashes, index, set file, verify pass and fail (rank files swapped),
verify on saved `rad-info --meta` text, refusals (wrong freq length, missing input named), reproducible bytes;
classifier cases (" The" cap, "The" cap, "ing" piece, " the" not kept, "3.14" number not kept, "x2" mixed,
punct/space/newline not kept, a lone UTF-8 byte reads U+FFFD and is "mixed" — faithful quirk); half-to-even
rounding; ties by position; the transcription equals `fnlev/rules.py` over the WHOLE vocab (env-gated); the
real sidecar's layout and the four named tokens (env-gated); kld_corpus cutting and its unstable-boundary
search; rows_compare dump matching and refusals; the fixture's rows follow from its own kept scores and from
the sidecar table.

## 4. R33 fixture — `tests/fixtures/rowsel_quick9.json` (format defined here; notes/kernels.md did not exist)

The KERNELS lane already points ctest at it (`KVA_ROWSEL_FIXTURE`, kernels/CMakeLists.txt) and their schema
text says `k = rint(share * matches), half to even` — consistent. One-line JSON, 131 KB:

```
{"format": "kva-rowsel-fixture-1", "window": 2048, "share": 0.25, "classes": ["cap","mixed","piece"],
 "score_tensor": "kva.rowsel.score", "score_sha256": "<sha256 of the full f32 LE score table>",
 "vocab": 248320, "kept_ids": [1757 ids, ascending], "kept_scores": [their f32 scores],
 "ppl": "ppl.jsonl", "ppl_sha256": "1ed15cb2…", "rule": "…",
 "docs": [{"doc": "ppl/8k/0", "token_ids": [2048 ids], "matches": 475, "k": 119,
           "rows": [119 window-relative rows, ascending], "random_count": 119}, … 9 docs]}
```
- Score table for the test: a [vocab] f32 array of −inf with `kept_ids` set to `kept_scores` (every id the
  windows use that is not listed is −inf). This makes the host test self-contained (no 1.3 GB file in CI);
  `score_sha256` lets a test with `KVA_SIDECAR` confirm the full table is the one the fixture came from.
  (Deviation from "ids + rows only": +~45 KB, so R33's host part runs in a fresh clone.)
- `rows` are fnlev.rules' own output (`Rules.rows({"rule":"class56","share":0.25,"classes":[…]}, ids, 2048,
  doc)`); the generator refuses if the sidecar table + `kva_rules.select_rows` disagree (they agree on all 9).
- `random_count` = k (count-matched control; the device hash's rows are its own, R33 checks count + determinism).
- Per doc (matches → k): 475→119, 404→101, 312→78, 507→127, **578→144 (144.5, half-to-even; `roundf` gives
  145)**, 559→140, 497→124, 376→94, 405→101. k ≤ 144 < cap 512, so no truncation case here.

Regenerate: `tools/rows_compare.py fixture --ppl <quick/ppl.jsonl> --out tests/fixtures/rowsel_quick9.json
--tokenizer <checkpoint> --sidecar data/sidecar/kva-sidecar.safetensors --freq <freq> --fnlev-root <research repo>`.

## 5. KL corpus (Stage 0 step 6) — `corpus/quick9.jsonl` (git-ignored) + `.manifest.json`

`tools/kld_corpus.py --ppl <quick/ppl.jsonl> --tokenizer <tcc checkpoint> --out corpus/quick9.jsonl`
(corpus sha256 `35d7b9d1…7d`, ppl.jsonl sha256 `1ed15cb2…bb`):

| doc | N original | N (cut) | score_from | bulk chunks | cut moved | re-encoded ids == original prefix |
|---|---|---|---|---|---|---|
| ppl/8k/0, 8k/1, 8k/2 | 9,216 | 8,192 | 6,144 | 3 each | 0 | yes |
| ppl/16k/0..3 | 16,384 | 16,384 | 14,336 | 7 each | 0 | yes |
| ppl/32k/0, 32k/1 | 32,768 | 32,768 | 30,720 | 15 each | 0 | yes |

**Σ bulk chunks = 67** = the approximate-step count R18 must log. No boundary needed moving (decode→encode
round-trip stable at every cut). The suite's own `prompt_tokens` (9216/16384/32768) equal this tokenizer's
counts.
- Format: raw PG-19 text, **no chat formatting** (no `<|im_start|>`; KVA-FACTS §7: the projector was fit at
  chat share 0.5, so raw text is the format the paper's ppl numbers used).
- Scoring differs from the original harness: it scored the **last 512 tokens** of 9,216/16,384/32,768-token
  docs; ours scores positions `[N−2048, N−1)` = **2,047 predictions** over the last chunk, and the 8k docs are
  cut to 8,192 (their last 1,024 tokens dropped). Numbers are not comparable with the tcc table one to one.
- Check for the orchestrator: the engine re-tokenises `prompt` with its own baked tokenizer (same tokenizer.json
  by hash, §1); confirm in the first KL report that each doc scores 2,047 positions (else N differed).

## 6. R39 — engine dump format and `tools/rows_compare.py`

Dump the ARCH lane should write (`RADIANCE_KVA_DUMP=<dir>` → `<dir>/rows.jsonl`), one line per approximate chunk:
```
{"chunk_start": <absolute position of the chunk's first row>, "n_tok": <rows>,
 "rows_idx": [chunk-relative selected rows, ascending; -1 padding allowed, dropped by the reader],
 "token_ids": [the chunk's n_tok ids]}
```
Chunks are matched to corpus docs by their token ids at `chunk_start` (no doc id needed from the engine; any
order). `compare --dump` reports per doc: whole-prompt rows (fnlev.rules at P = N − 2048), in-chunk rows,
Jaccard, in-chunk share of bulk rows, `dump_equals_simulated` (the engine's rows vs the same rule run here per
chunk — an end-to-end R33 check) and `chunks_missing` (bulk chunks with no record = ran exact or not dumped).

**Offline preview (simulated chunks, not the engine — Q10/R39 prediction input):**
`rows_compare.py compare --corpus corpus/quick9.jsonl --simulate …`

| doc | P | whole | in-chunk | both | Jaccard | in-chunk share |
|---|---|---|---|---|---|---|
| ppl/8k/0 | 6144 | 355 | 355 | 347 | 0.9559 | 5.78% |
| ppl/8k/1 | 6144 | 299 | 299 | 294 | 0.9671 | 4.87% |
| ppl/8k/2 | 6144 | 246 | 246 | 231 | 0.8851 | 4.00% |
| ppl/16k/0 | 14336 | 906 | 906 | 848 | 0.8797 | 6.32% |
| ppl/16k/1 | 14336 | 971 | 971 | 899 | 0.8619 | 6.77% |
| ppl/16k/2 | 14336 | 887 | 887 | 862 | 0.9452 | 6.19% |
| ppl/16k/3 | 14336 | 881 | 881 | 853 | 0.9384 | 6.15% |
| ppl/32k/0 | 30720 | 1494 | 1493 | 1413 | 0.8977 | 4.86% |
| ppl/32k/1 | 30720 | 1660 | 1660 | 1591 | 0.9202 | 5.40% |

Mean Jaccard **0.917** (HANDOVER §6 band 0.5–0.9: slightly above — a finding, not tuned); rows selected
**5.61%** of bulk rows (band 4–8%; tcc 5.6%). This is the rule run in Python on the real docs; the engine
number replaces it once the dump exists.

## 7. Stage 2 step 3 — the append, and Q12 (read from `tools/rad_convert.cpp:180-900` and `core/format/`, not run)

**Q12 answer (UNVERIFIED reading, verify with `--plan-only -v` once the container + plugin exist): NO —
`--reuse --in-place` does not accept a sidecar-only namespace.** Four reasons in the code:
1. `Checkpoint::open` needs `config.json` beside the shards and refuses without it
   (`core/format/checkpoint.cpp:170-176`); the declare's meta, and the container's rewritten header meta
   (`rad_convert.cpp:694-708`), come from the INPUT's config, not from the `--reuse` container (only its
   `radiance.*` keys carry over, `:403-417`).
2. Every non-optional weight the declare names must resolve to a checkpoint tensor in the input
   (`rad_convert.cpp:596-606` → `ckpt_resolve`, `checkpoint.cpp:569-573` "it needs checkpoint tensor …, which is
   not in <dir>"). The `--reuse` container is consulted only AFTER planning, by name (`rad_convert.cpp:762-786`).
3. The vocab section is rebuilt from the input dir's `tokenizer.json` or `--tokenizer` (`:803-826`).
4. **Silent hazard**: an appended header names only this run's plan (`radfile.h:255-265`, `add_existing` at
   `rad_convert.cpp:770-786`); nothing checks that every old weight was carried. A weight the run does not plan
   would vanish from the container without an error (its bytes stay; `<rad>.pre-append` restores). Weights
   whose planned encoding differs ARE refused (`:770-779`).
Also: without `--recipe`, every unnamed weight plans as its checkpoint dtype, so the container's quantised
weights would mismatch and be refused — the container's recipe must be passed.

**Proposed way through (orchestrator / Dylan to decide; nothing downloaded):** a STUB namespace of the source
checkpoint. The source is `Qwen/Qwen3.8-Flash-Next` (bf16, 131 shards, 360 GB; public, the HF tree API answers
without a token). Build a dir with its real `config.json`, `generation_config.json`, `preprocessor_config.json`,
`video_preprocessor_config.json`, `tokenizer.json` (small files), its `model.safetensors.index.json` extended
with the 87 `kva.*` names → `kva-sidecar.safetensors` (symlink/copy of ours), and 131 SPARSE shard files whose
first bytes are the real safetensors headers (HTTP range read: probed shard 2, header 160 B, 1 tensor
`model.language_model.layers.0.mlp.experts.gate_up_proj` BF16 [512,1280,2560]) and whose data is a hole
(apparent 360 GB, ~0 disk on /var/home). In-place reused weights are planned from names/dtypes/shapes only and
copied by offset (`add_existing`), so stub data is never read; only the 87 new weights are read (from our real
shard). Risks to check first: (a) the gptq rule's options include `calib=calib/w4nl-calib` — if the quantiser
checks that path when the recipe is compiled (`rad_convert.cpp:446`), an empty dir at that relative path may
be needed (the options string must stay identical); (b) the recipe text must equal the container's
(`rad-info --recipe` vs `~/models/rad/qwen4exp-w4nl64-i8-hc8m.recipe`); (c) the safetensors loader must accept
holes (it mmaps; to confirm). Alternatives: the plugin loads the sidecar file itself (RADIANCE-FACTS §8,
UNVERIFIED, not documented), or a full re-convert (impossible: the expert calibration Grams are unpublished).

**The command** (inside `radiance-build`; `STUB` as above; `home/` = the plugin home the ARCH/KERNELS lanes
install, holding `architectures/qwen4exp_fp8.so` (KVA) and `kernels/kva.so`):
```
docker run --rm \
  -v "$(readlink -f ~/models/rad)":/models \
  -v ~/projects/inference/radiance-kva:/kva:ro -v "$STUB":/stub:ro \
  radiance-build /stage/opt/radiance/bin/rad-convert /stub \
    --reuse /models/qwen3.8-next-flash-fp8-iq4r-moe.rad --in-place \
    --recipe /models/qwen4exp-w4nl64-i8-hc8m.recipe \
    --home /kva/home:/stage/opt/radiance/share/radiance \
    $(sed 's/^/--set /' ~/projects/inference/radiance-kva/data/sidecar/rad-convert-set.txt) \
    --set kva.mode=quality --set kva.tail=2048 \
    --plan-only -v 2>&1 | tee evidence/stage2/plan-only.log
```
(`-o` defaults to the `--reuse` file under `--in-place`; `tokenizer.json` in the stub dir replaces the
original `--tokenizer` flag — same file by hash.) Drop `--plan-only` for the real append; the container must
not be open in any engine (`begin_append` refuses, `radfile.cpp:673-680`). Disk: +1.28 GiB on the model SSD.

**What `--plan-only -v` must show:**
- the KVA `qwen4exp_fp8.so` from `/kva/home` shadowing the installed one, `kva.so` loaded, no refusal;
- `source /stub (T tensors)`, `arch qwen4exp`;
- `plan N weight(s): …` with **N = (weights in `rad-info -v <rad>`) + 87**;
- per-weight debug lines for the new ones, all "as is": `kva.proj.L.weight bf16 50.0 MiB`,
  `kva.proj.L.bias bf16 5.0 KiB`, `kva.st.L` and `kva.stswap.L f32 3.0 MiB`, `kva.rowsel.score{,_none,_all}
  f32 970 KiB`;
- `--plan-only: about ~115 GiB would be written` — this sums EVERY planned weight, reused ones included; the
  real append writes 1.28 GiB.
`--plan-only` cannot show new vs reused vs dropped itself, so the gate is a diff of the plan's name+encoding
list against `rad-info -v`: new = exactly the 87 `kva.*`, dropped = 0, changed encoding = 0. After the real
append the last line must read `--in-place: <container count> weight(s) … kept where they were …; 87 added
past its end`, and `<rad>.pre-append` must exist.

## 8. Decisions and their cost

- **Controls in one container** (orchestrator): `kva.stswap.*` + `score_none` + `score_all` = 54 MiB + 2 MiB
  more on disk. ARCH lane: declare the swap/none/all variants only when an env/meta switch selects them,
  otherwise they cost VRAM on every run (`kva.st*` RAD_SHARD_ROW: 27 MiB per rank for the swap set).
- **`score_all` added** (not in the brief's list; HANDOVER Stage 2.1 names `--all-rows`, R35 needs an all-kept
  table): 970 KiB. With `RADIANCE_KVA_ROWSEL=all` the op ignores scores anyway; this table makes the R35 run
  exercise the class path with every row a match.
- **Transcribed, not imported** rules: the tool runs without the research repo; cost = a duplicated 25 lines,
  guarded by the whole-vocab equality test and by the fixture generator's refusal on any disagreement.
- **f32 score table**: −logfreq of an f32 is exact, so ranking and ties equal the original's fp64 ranking.
- **Kernel rounding**: `k = round(share × matches)` is Python's half-to-even; the fixture holds a case
  (144.5 → 144). A `roundf` kernel fails R33 on ppl/16k/1.
- **Lone UTF-8 byte tokens count as "mixed" (kept)**: the original decodes per-token bytes with
  `errors="replace"`; kept faithful, not "fixed".
- **File mode**: safetensors writes the shard 0600. Readable by the owner and by root in Docker; chmod 644 if a
  container runs under another uid.

Rejected arms: none (nothing measured on the engine in this lane).

## 9. Open for other lanes / the orchestrator
- ARCH: name-map `kva.*` identically (logical = checkpoint name); dump format §6; gate controls behind switches.
- KERNELS: fixture §4 (self-contained table; optional `KVA_SIDECAR` cross-check against `score_sha256`).
- Orchestrator: Q12 (§7) decides how Stage 2.3 can run at all; R18 count = 67; KL report must show 2,047
  scored positions per doc.
