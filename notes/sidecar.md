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
$ tools/ridgefill_sidecar.py build --proj ~/AI-Work/kva/qfn/proj/kva-big-s24.safetensors \
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
| `ridgefill.proj.L.weight` | [2560, 10240] bf16 | 24 (L 24..47) | `layer.L[:, :10240]` |
| `ridgefill.proj.L.bias` | [2560] bf16 | 24 | `layer.L[:, 10240]` |
| `ridgefill.st.L` | [48, 128, 128] f32 | 18 | `cat(rank0.sum/count, rank1.sum/count)`, rank 0's heads first |
| `ridgefill.stswap.L` | [48, 128, 128] f32 | 18 | halves swapped (R24 negative control) |
| `ridgefill.rowsel.score` | [248320] f32 | 1 | −logfreq if class ∈ {cap, mixed, piece}, else −inf |
| `ridgefill.rowsel.score_none` | [248320] f32 | 1 | all −inf (R41) |
| `ridgefill.rowsel.score_all` | [248320] f32 | 1 | all 0: every id a match, ties by position (R35 all-rows) |

Header metadata (also in `rad-convert-set.txt`):
```
ridgefill.format=ridgefill-sidecar-1
ridgefill.rowsel.classes=cap,mixed,piece
ridgefill.rowsel.share=0.25
ridgefill.split=24
ridgefill.src.config.sha256=238b0a0024b5cfb7ebc1df0c8d122c01c1b24e0843b2c9395740eabc43637e91
ridgefill.src.freq.sha256=4f1b719a59fdaa6ddfc7e06502b817c5a96e6a00bca060136d110e672059f97a
ridgefill.src.proj.sha256=c9db6064eb275d3166df8139239959dfdeb64f2778707a2f42215164999cb9cd
ridgefill.src.st0.sha256=38e4b550dfc1419922feccb59f75f758be7e8f2468c84b51a30712914b0b5dd3
ridgefill.src.st1.sha256=af517cf760f7baef3f514c48ec4ab36d31effa444641cd6a0f87fdae2e384a18
ridgefill.src.tokenizer_config.sha256=b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27
ridgefill.src.tokenizer_json.sha256=0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
```
(`config` = the tokenizer dir's config.json, read only for the vocab size.)

Independent check against the source files (one-off script, not the tool's code path): all 24
`ridgefill.proj.L.weight/.bias` equal the source slices; all 18 `ridgefill.st.L`/`ridgefill.stswap.L` equal the rank halves in
the stated order; `score` keeps 167,288 ids, none of the 33 specials nor the 243 padding rows; kept scores in
[6.540, 14.149]; `score_none` all −inf; `score_all` all 0.

`verify` (R13 against the sidecar's own header; the container form waits for the append):
```
$ tools/ridgefill_sidecar.py verify data/sidecar/kva-sidecar.safetensors --proj … --st … … --freq … --tokenizer …
OK        ridgefill.src.config.sha256 238b0a00…   (7 hashes OK)
          ridgefill.format = ridgefill-sidecar-1 / ridgefill.split = 24 / ridgefill.rowsel.share = 0.25 / ridgefill.rowsel.classes = cap,mixed,piece
verify data/sidecar/kva-sidecar.safetensors: PASS
```
For the container: `rad-info --meta <rad> > evidence/stage2/rad-info-meta.txt` inside `radiance-build`, then
`tools/ridgefill_sidecar.py verify evidence/stage2/rad-info-meta.txt <same source flags>` (a `.rad` target also works
when `--rad-info` names a rad-info runnable on the host path).

**Bug found and fixed, with its regression test**: the first two builds of the same inputs had different
sha256s. Only the header differed: safetensors (Rust) writes `__metadata__` in hash order. The tool now
rewrites the header with sorted metadata, same length, in place
(`test_two_builds_of_the_same_inputs_are_byte_identical` fails on the old code, passes now).

## 3. Tests

`python -m pytest tests/` (no env): **44 passed, 3 skipped** (the three need real data). With
`RIDGEFILL_SIDECAR=data/sidecar/kva-sidecar.safetensors RIDGEFILL_TOKENIZER=<checkpoint dir> RIDGEFILL_RESEARCH_ROOT=<research repo>`:
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

The KERNELS lane already points ctest at it (`RIDGEFILL_ROWSEL_FIXTURE`, kernels/CMakeLists.txt) and their schema
text says `k = rint(share * matches), half to even` — consistent. One-line JSON, 131 KB:

```
{"format": "ridgefill-rowsel-fixture-1", "window": 2048, "share": 0.25, "classes": ["cap","mixed","piece"],
 "score_tensor": "ridgefill.rowsel.score", "score_sha256": "<sha256 of the full f32 LE score table>",
 "vocab": 248320, "kept_ids": [1757 ids, ascending], "kept_scores": [their f32 scores],
 "ppl": "ppl.jsonl", "ppl_sha256": "1ed15cb2…", "rule": "…",
 "docs": [{"doc": "ppl/8k/0", "token_ids": [2048 ids], "matches": 475, "k": 119,
           "rows": [119 window-relative rows, ascending], "random_count": 119}, … 9 docs]}
```
- Score table for the test: a [vocab] f32 array of −inf with `kept_ids` set to `kept_scores` (every id the
  windows use that is not listed is −inf). This makes the host test self-contained (no 1.3 GB file in CI);
  `score_sha256` lets a test with `RIDGEFILL_SIDECAR` confirm the full table is the one the fixture came from.
  (Deviation from "ids + rows only": +~45 KB, so R33's host part runs in a fresh clone.)
- `rows` are fnlev.rules' own output (`Rules.rows({"rule":"class56","share":0.25,"classes":[…]}, ids, 2048,
  doc)`); the generator refuses if the sidecar table + `ridgefill_rules.select_rows` disagree (they agree on all 9).
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

Dump the ARCH lane should write (`RADIANCE_RIDGEFILL_DUMP=<dir>` → `<dir>/rows.jsonl`), one line per approximate chunk:
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

**Q12 answer: NO (read in the code; the §7.4 plan-only run is consistent with it) —
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

**Way through, APPROVED by the orchestrator and DONE (2026-10-04 17:28–17:40): a header-only stub of the source
checkpoint.** The plan-only run below confirms the reading: with the full namespace the plan covers every old
weight; nothing in the stub's data regions is read.

### 7.1 The stub — `data/stub/` (git-ignored, /var/home), built by `tools/stub_checkpoint.py`
```
$ tools/stub_checkpoint.py --repo Qwen/Qwen3.8-Flash-Next --revision main --out data/stub \
      --extra data/sidecar/kva-sidecar.safetensors
Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540: 13 small files, 131 shards to stub
wrote data/stub: 131 header-only shards (335.3 GiB apparent), extra tensors {'kva-sidecar.safetensors': 87}
```
13 s. **Disk: 23 MB real, 336 GB apparent** (sparse; anything that copies it must use `--sparse`/reflinks, or it
expands to 336 GB). Every non-safetensors file of the repo downloaded as is (configs, tokenizer, vocab, merges,
chat template, index, README, LICENSE). Each shard = its real header (two HTTP range reads; header + data
length checked against the repo's file size) + a hole. The index gains the 87 `ridgefill.*` names →
`kva-sidecar.safetensors`, a relative symlink to `../sidecar/kva-sidecar.safetensors` (resolves inside the
`/ridgefill` mount). Revision pinned in `data/stub/stub-manifest.json` with every header's sha256.

### 7.2 Checks (evidence/stage2/)
- **Container sha256** (orchestrator, `evidence/stage0/container.sha256`): `0af5e962…4d20` = the published LFS
  sha256. All runs below are on the pristine container.
- **Tokenizer**: `sha256sum data/stub/tokenizer.json` = `0997f410…29b9f3` = the file the container's vocab was
  baked from (README "How this file was made") = the tokenizer the score table was built with.
  `tokenizer_config.json` git blob `5de744b3` equal too.
- **config.json vs `rad-info --meta`** (`config-vs-meta.txt`): of the container's 180 meta keys, 166 equal
  the stub's config.json / generation_config.json / preprocessor configs flattened the radiance way (text_config
  aliased, `generation.*`, `preprocessor.*`, `video_preprocessor.*`); the rest are rad-convert's derived keys
  (n_layers 48, n_embd 2560, n_vocab 248320, …, `radiance.encoder vision`). **0 differ, 0 container keys
  unexplained.** Four stub keys absent from the meta are `null` in config.json (`pad_token_id`,
  `mtp_use_hidden_state_from_layer`, plain and text_config), which radiance's flatten skips. arch: model_type
  `qwen4_exp` → `qwen4exp`; header "created by rad-convert Oct 2 2026 from Qwen/Qwen3.8-Flash-Next".
- **Recipe**: `rad-info --recipe` is NOT byte-equal to `~/models/rad/qwen4exp-w4nl64-i8-hc8m.recipe`
  (`recipe.diff`, 108 lines): the container stores the parsed rules — no comments, single spaces, `$CALIB`
  expanded. Normalised (comments stripped, whitespace collapsed, `$CALIB` → `calib/w4nl-calib`) the file equals
  the stored 26 rules exactly (`recipe-file-normalised.txt` vs `rad-info-recipe.txt`: `diff` empty).
- **`calib=` (read in the code)**: no directory is needed. `Compiled::build` only parses option strings
  (`rad_convert.cpp:67-127`); gptq's `encoding` hook never reads `calib` (`libquant/lq_registry.cpp:101-118`);
  the directory is opened only in `gptq_quantize` (`:131`, `gptq_factor`/`calib_gram`), which runs only for a
  weight being WRITTEN, never for one reused in place. BUT the recipe file says `calib=$CALIB` and an unset
  variable is refused at parse (`core/format/recipe.cpp:37-56`), so the run sets `CALIB=calib/w4nl-calib` —
  the literal the container recorded, so the options string (`options_text`, sorted) matches. No empty dir made.
- **R13 negative control**: `ridgefill_sidecar.py verify evidence/stage2/rad-info-meta.txt …` on the pristine
  container → `FAIL (11 problem(s))` (7 hashes + 4 keys MISSING), exit 1 (`verify-pre-append.txt`).

### 7.3 Plugin home used (`home/`)
`ls home/*`: only the arch `.so` was installed (by the ARCH lane); `ridgefill.so` was missing. Built + installed both
from **committed `fc90feb`** (`git archive HEAD` into `build-sidecar-hip/src`, sources unedited) inside
`radiance-build`: configure says `radiance 1.0.8: RADIANCE_SRC and /stage/opt/radiance/bin/radiance agree`,
`device rows ON, GPU targets 'gfx1201', installs into /ridgefill/home`; log `build-sidecar-hip/build.log`.
`home/architectures/qwen4exp_fp8.so` sha256 `eb6396a9…`, `home/kernels/ridgefill.so` `73229f27…`; both `.comment`
Ubuntu GCC 14.2 + AMD clang 22 (roc-7.2.4), i.e. the radiance-build toolchain, HIP configured. ridgefill.so is the
KERNELS lane's committed Stage 1 (device rows refuse; no device code object yet: "0 fat binaries, gfx1201
host"). The KERNELS/ARCH lanes will reinstall newer builds over these; the plan only needs the declare.

### 7.4 `--plan-only -v` — RUN, exit 0 (`evidence/stage2/plan-only.log`, 51,732 lines, 4.5 s, no GPU)
```
docker run --rm -e RADIANCE_RIDGEFILL_DECLARE=all -e CALIB=calib/w4nl-calib \
  -v "$(readlink -f ~/models/rad)":/models:ro -v ~/projects/inference/radiance-kva:/ridgefill:ro \
  radiance-build /stage/opt/radiance/bin/rad-convert /ridgefill/data/stub \
    --reuse /models/qwen3.8-next-flash-fp8-iq4r-moe.rad --in-place \
    --recipe /models/qwen4exp-w4nl64-i8-hc8m.recipe \
    --home /ridgefill/home:/stage/opt/radiance/share/radiance \
    $(sed 's/^/--set /' data/sidecar/rad-convert-set.txt | tr '\n' ' ') --set ridgefill.mode=quality --set ridgefill.tail=2048 \
    --plan-only -v > evidence/stage2/plan-only.log 2>&1
```
Key lines:
```
I source   /ridgefill/data/stub (1745 tensors)
I arch     qwen4exp
I device: built with HIP but no device is visible -- using the host backend
D loader.cpp:488  plugin 0: ridgefill 0.1.0 (/ridgefill/home/kernels/ridgefill.so) -- 6 kernels, 3 schemas, 0 fat binaries, gfx1201 host
I plugin /stage/opt/radiance/share/radiance/architectures/qwen4exp_fp8.so is shadowed by /ridgefill/home/architectures/qwen4exp_fp8.so, which comes first on the search path
D loader.cpp:488  plugin 4: qwen4exp_ridgefill 0.1.0 (/ridgefill/home/architectures/qwen4exp_fp8.so) -- 0 kernels, 0 schemas, 0 fat binaries, portable
I recipe   26 rule(s)
D rad_builder.cpp:1532  declare: 0 device band(s) and 265 host band(s) resolved to nothing; run with --debug-graph for the full list
I plan     51576 weight(s): 50603 quantised by the recipe, 973 kept as the checkpoint holds them
D rad_convert.cpp:659    ridgefill.proj.24.weight      bf16    50.00 MiB  as is
D rad_convert.cpp:659    ridgefill.st.24               f32      3.00 MiB  as is
D rad_convert.cpp:659    ridgefill.rowsel.score        f32    970.00 KiB  as is
I --plan-only: about 114.74 GiB would be written; nothing was written to /models/qwen3.8-next-flash-fp8-iq4r-moe.rad.
```
51,576 = 51,489 (container entries, `rad-info.txt`) + 87. "About 114.74 GiB" counts every planned weight; the
real append writes 1.28 GiB of new weights + ~19 MB of new tables. The 265 unresolved host bands are the
device-only ops on a GPU-less run (no dead op, else exit 1); the container's original convert had the same view.
The `/models` mount was read-only, so this run could not have written.

**`-e RADIANCE_RIDGEFILL_DECLARE=all` is on every rad-convert command here** (ARCH lane, notes/arch.md §4): under it
the plugin (committed `arch/qwen4exp_ridgefill.cpp` `decl_every_copy`) declares, all optional, `ridgefill.proj.L.{weight,bias}`
and `ridgefill.projr.L.{weight,bias}` for every layer, `ridgefill.st.L` / `ridgefill.stswap.L` / `ridgefill.str.L` for every delta-net
layer, and `ridgefill.rowsel.score`, `score_none`, `score_all`. The shipped shard's 87 tensors are a subset (so
`score_all` stays); absent optional names (projr, str, layers < 24) are skipped by the planner. The gate below
proves the match: every one of the 87 was planned (`expected_missing 0`), nothing else new.

### 7.5 Name + encoding diff gate — **PASS** (`tools/plan_diff.py`; `plan-diff.txt`, `plan-diff.json`)
```
$ tools/plan_diff.py --plan evidence/stage2/plan-only.log --container evidence/stage2/rad-info-v.txt \
      --expect-new data/sidecar/kva-sidecar.safetensors --out evidence/stage2/plan-diff.json
new                  87  ridgefill.proj.24.bias, ridgefill.proj.24.weight, ridgefill.proj.25.bias, ...
dropped               0
changed               0
unexpected_new        0
expected_missing      0
planned 51576, held 51489: PASS
```
"changed" compares encoding AND quantiser + options for all 51,489 held weights (e.g. an expert:
`affine:codes=u4[1x1],scale=fp8_e4m3[1x64],scale.1=f32[*x*],table=f32{1x16}/fwht128` + `gptq
block2=*x*,calib=calib/w4nl-calib,cd=3,…` on both sides; the protected `blk.47.ffn_down_exps.445.weight`
`bf16` + `cast dtype=bf16` on both). Sizes are not compared (rad-info prints stored bytes incl. plane padding,
the plan prints raw plane bytes: 825.31 vs 825.07 KiB for one expert). Shape/group mismatches the plan cannot
show are refused loudly by the writer (`rad_convert.cpp:762-779`), not dropped silently.

### 7.6 The REAL append — orchestrator only, after the Stage 0 baselines, on the pristine container
Preconditions: (1) `docker ps` shows NO engine container and `pgrep -af 'radiance|rad-'` shows nothing — the
writer's "open in another process" refusal (`radfile.cpp:673-680`) scans /proc inside rad-convert's own
container and **cannot see an engine running in another container**; (2) `sha256sum` of the container =
`0af5e962…4d20` (or trust stage0's check if the file has not been opened rw since); (3) model SSD free space
≥ 2 GB (33 GB free at 17:38); (4) the same `home/` and `data/sidecar` shard as the gate (rerun §7.4 + §7.5 if
either changed since — `data/sidecar/kva-sidecar.safetensors` sha256 `05c4e088…`).
```
cd ~/projects/inference/radiance-kva
docker run --rm -e RADIANCE_RIDGEFILL_DECLARE=all -e CALIB=calib/w4nl-calib \
  -v "$(readlink -f ~/models/rad)":/models -v ~/projects/inference/radiance-kva:/ridgefill:ro \
  radiance-build /stage/opt/radiance/bin/rad-convert /ridgefill/data/stub \
    --reuse /models/qwen3.8-next-flash-fp8-iq4r-moe.rad --in-place \
    --recipe /models/qwen4exp-w4nl64-i8-hc8m.recipe \
    --home /ridgefill/home:/stage/opt/radiance/share/radiance \
    $(sed 's/^/--set /' data/sidecar/rad-convert-set.txt | tr '\n' ' ') --set ridgefill.mode=quality --set ridgefill.tail=2048 \
    -v > evidence/stage2/append.log 2>&1; echo "exit $?"
```
(Identical to §7.4 minus `--plan-only` and with `/models` writable.) Expect the last lines:
`wrote /models/…rad: 51576 weight(s), …` and `--in-place: 51489 weight(s), …, kept where they were in …; 87
added past its end`, then `/models/…rad.pre-append` (256 B) beside the container.

After it (evidence/stage2/, `-after` suffix):
1. `rad-info -v` → `rad-info-v-after.txt`; `tools/plan_diff.py --plan evidence/stage2/plan-only.log
   --container evidence/stage2/rad-info-v-after.txt` (no `--expect-new`) must PASS: the container holds exactly
   the plan (51,576, none new, none dropped, none changed).
2. `rad-info --meta` → `rad-info-meta-after.txt`; `diff` with `rad-info-meta.txt` must show only the 13 added
   `ridgefill.*` keys (11 from the set file + `ridgefill.mode`, `ridgefill.tail`); then `tools/ridgefill_sidecar.py verify
   evidence/stage2/rad-info-meta-after.txt <the build's source flags>` must PASS (R13).
3. `rad-info --recipe` → must equal `rad-info-recipe.txt` (diff empty). The header's "created by" gains
   "; extended in place by rad-convert … from data/stub" (expected).
4. R12: `rad-info -v` lists `ridgefill.proj.24.weight … ridgefill.st.46`; R7 again with `RADIANCE_RIDGEFILL=off`.
Note `ridgefill.mode=quality` becomes the container's default: until the plugin implements quality mode, every serve
with the RidgeFill home must set `RADIANCE_RIDGEFILL=off|speed|plumb` explicitly (the plugin refuses unimplemented modes at
declare — loud, not silent). A later append can re-set it (see the next note).

**Every later append (Stage 6 refit) must re-pass ALL ridgefill `--set` keys** — the shipped set file, the refit set
file, `ridgefill.mode`, `ridgefill.tail` — because only `radiance.*` keys carry over from the old header
(`rad_convert.cpp:403-416`) and the new meta is the stub's config + this run's `--set`s. Rebuild the stub with
both `--extra` shards, `-e RADIANCE_RIDGEFILL_DECLARE=all`, and gate with `--expect-new` = the refit shard.

### 7.7 Restoring the pristine container from `.pre-append`
What it holds (`core/format/radfile.cpp:1318-1338`): the old `RadFileHeader` byte for byte (248 B,
`abi/rad_format.h:38-60`) followed by the old file length as a native int64 (x86: little-endian) — 256 B
total. Why that is enough: an original container's tables sit in front of the blob (strings at 248 … vocab
ending ≈ 19.3 MB, data from 20,971,520 — `rad-info.txt`); the append writes every new byte at or past the old
end rounded up to 2 MiB (`append_base_`, `radfile.cpp:684`) and leaves the old tables and blob untouched, so
the old header + truncation to the old length gives the old file back byte for byte (the comment at
`radfile.cpp:1318-1319` states it). The writer also undoes itself on any error it catches (`abort()`,
`radfile.cpp:577-590`: old header back, truncate). No radiance tool does the restore; by hand:
```
RAD=$(readlink -f ~/models/rad)/qwen3.8-next-flash-fp8-iq4r-moe.rad; BAK=$RAD.pre-append
docker ps; pgrep -af 'radiance|rad-'                       # nothing may have the file open
H=$(( $(stat -c %s "$BAK") - 8 )); test "$H" -eq 248       # the header size of this format version
OLD=$(od -An -t d8 -j "$H" -N 8 "$BAK" | tr -d ' '); test "$OLD" -eq 121969901568
cmp -n 8 "$BAK" "$RAD"                                     # same magic + version as the live header
dd if="$BAK" of="$RAD" bs="$H" count=1 conv=notrunc,fsync  # the old header back
truncate -s "$OLD" "$RAD"                                  # the appended tail off
sha256sum "$RAD"                                           # MUST print 0af5e96244e8…ceaa4d20
rm "$BAK"                                                  # only after the hash matches
```
If rad-convert was killed hard BEFORE `.pre-append` existed, the old header is still in place (it is written
last); the file is just longer: `truncate -s 121969901568 "$RAD"` and check the sha256. The sha256 takes ~4 min.

## 8. Decisions and their cost

- **Controls in one container** (orchestrator): `ridgefill.stswap.*` + `score_none` + `score_all` = 54 MiB + 2 MiB
  more on disk. ARCH lane: declare the swap/none/all variants only when an env/meta switch selects them,
  otherwise they cost VRAM on every run (`ridgefill.st*` RAD_SHARD_ROW: 27 MiB per rank for the swap set).
- **`score_all` added** (not in the brief's list; HANDOVER Stage 2.1 names `--all-rows`, R35 needs an all-kept
  table): 970 KiB. With `RADIANCE_RIDGEFILL_ROWSEL=all` the op ignores scores anyway; this table makes the R35 run
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
- ARCH: name-map `ridgefill.*` identically (logical = checkpoint name); dump format §6; gate controls behind switches.
- KERNELS: fixture §4 (self-contained table; optional `RIDGEFILL_SIDECAR` cross-check against `score_sha256`).
- Orchestrator: Q12 (§7) decides how Stage 2.3 can run at all; R18 count = 67; KL report must show 2,047
  scored positions per doc.
