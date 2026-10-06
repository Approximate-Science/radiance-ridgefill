# notes/refit.md — REFIT lane (Stage 6 fitting side: projector + correction refit on radiance)

Provenance: radiance `140987f` (v1.0.8, read only); capture code = plugin commit `3af4eaa` (ARCH capture mode +
KERNELS `ridgefill_state_read`); refit code = commits in §9; container `~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad`
(the Stage 2 append, 87 `ridgefill.*`); research code `~/projects/research/kva` @ `cba2d91`, imported read-only through
`RIDGEFILL_RESEARCH_ROOT`; Python = its venv (torch 2.11.0+rocm, numpy 2.5.3, tokenizers 0.23.2). Date 2026-10-04.

## 0. Status

| step | state |
|---|---|
| 1 inputs | **done**: 151 training docs (raw + chat = 302 prompts, 3,984,554 tokens), 5 held docs (10 prompts, 60,214 tokens), 14 +st prompts (134,081 tokens, 55 chunk ends at tail 512); overlap re-checked (§1) |
| 2 capture format | **implemented + tested** on synthetic captures in the plugin's layout (`tests/test_refit_capture.py`, 7 pass); Sums fed directly (§2) |
| 3 capture driver | **ran**: 312 projector prompts (4,044,768 tokens) + 14 exact +st prompts, 48 min under the lock, every chunk captured (§6) |
| 4 projector solve | **measured**: lambda 0.03, 248,892 raw + 249,312 chat rows; held-out bi refit 0.7536 vs shipped 0.7501 on the same radiance rows (§4) |
| 5 correction fit | **measured**: 14 prompts, 55 chunk ends per layer per rank (R43 counts met); head order checked (§5) |
| 7 append prep | append #1 (projr) **done by the orchestrator** 20:37; **append #2 (ridgefill.str) prepared**: shard verify PASS, plan diff PASS (18 new, 0 dropped, 0 changed); command in §7 |

## 1. Inputs (step 1) — `tools/refit/docs.py build` → `data/refit/prompts/` (git-ignored)

```
$ RIDGEFILL_RESEARCH_ROOT=~/projects/research/kva .venv/bin/python tools/refit/docs.py build \
    --plans ~/AI-Work/kva-flashnext-iterate/bigcap/plans --tokenizer data/stub/tokenizer.json \
    --overlap ~/projects/research/kva/docs/flash-next/run-20261004/bigcap/overlap.json --out data/refit/prompts
train: 302 prompts, 3984554 tokens (raw 151, chat 151, cut 0) -> data/refit/prompts/train.jsonl
held: 10 prompts, 60214 tokens (raw 5, chat 5, cut 0) -> data/refit/prompts/held.jsonl
sterm: 14 prompts, 134081 tokens (raw 14, chat 0, cut 2) -> data/refit/prompts/sterm.jsonl
sterm: 55 approximate chunk ends at tail 512, chunk 2048
```
sha256: `train.jsonl` 42c2cbf4…db742, `held.jsonl` 47102673…e8d5, `sterm.jsonl` a47232a9…10d7,
`manifest.json` 7ea1e409…2e17 (holds every prompt's key, text sha256 and token count; the doc table is the appendix).
Tokenizer = `data/stub/tokenizer.json` sha256 `0997f410…` = the file the container's vocab was baked from (and tcc's).

- **Training docs = the shipped fit's own docs** (`report-big-s24.json` add_stats = FN-BIGCAP b0 + b1): b0 = FN-CAP2's
  71 overlap-clean calibration docs (the 28 suite-overlapping `trainc*` docs already excluded by FN-BIGCAP's
  50-gram check, `overlap.json` "excluded"; the tool refuses any doc on that list), b1 = 80 impact-corpus docs
  (web-001/web-036 excluded there). So the refit differs from the shipped fit in the engine, not the data.
- **Each doc raw AND chat** (brief). The shipped fit had chat copies of only 22 of b0's 71 (FN-CAP2's time box) +
  b1's 80: 151 raw + 102 chat. Chat share 0.5 by row weighting either way. Chat text = `qfn.steps.chat_text(text, i)`,
  i = index in FN-CAP2's capture order (99 calibration docs in plan order, then the 5 held): **checked equal** to the
  49 chat texts FN-BIGCAP stored (PLAN-b4 `chatold`); b1's chat texts taken from PLAN-b1 as captured; b1 raw texts
  checked against their manifest sha256.
- **Held = the shipped fit's held set**: heldc0, heldc1, heldc2, held0-rust, held1-code, raw + chat (shipped: 7,479
  rows on tcc; here 60,214 tokens → ~7,530 rows at stride 8, radiance captures every chunk incl. < 256 rows).
- **+st prompts**: plan-sterm minus **sterm4** (protocol 1.1 exclusion list `datasets/calibration-exclude.json`; my
  re-check below confirms it shares 779 50-grams with the suite's raw LongBench pool), plus two clean impact docs of
  FN-BIGCAP batch b2 (in neither the projector's training nor its held set), cut to 10,240 tokens:
  `impact-code-016`, `impact-prose-016`. **Why more prompts:** radiance approximates whole chunks only (a chunk
  with ≥ T tokens after it), so an N-token prompt gives floor((N − 512)/2048) chunk ends; the 13 original prompts
  give **49** (< R43's 50; tcc clipped a chunk to end at N − 512 and got 62). Minus sterm4: 12 prompts, 47 ends.
  +2 docs (one prompt of margin over the minimum): **14 prompts, 55 chunk ends per layer**.
- **Overlap re-check** (`docs.py overlap`, FN-BIGCAP's own `qfn.bigcap_overlap.check`, word 50-grams): the 13
  plan-sterm texts + 5 held + 2 added docs vs `~/AI-Work/kva-flashnext-tests-data` (356 files incl. the locked set,
  118,620,725 grams): **only `sterm-sterm4` overlaps** (lcc 41, repobench-p 287, repobench-p_e 41, … 779 total).
- Longest prompt: 30,486 tokens; none cut at the 49,151 limit (served `--max-model-len 49152` − 1).

## 2. Capture format → the research code (step 2) — `tools/refit/convert.py`

The plugin writes notes/arch.md "Capture" (checked against `arch/ridgefill_dump.h` @ 3af4eaa): per chunk
`chunk.p<P>.h<H16>.{boundary,bi.L,rows,ids,pos}.npy` + a `capture.jsonl` line. `convert.read_chunk` gives tcc's
Capture record (`boundary_24` bf16 [R,10240], `block_input_L` bf16 [R,2560] L = 24..47, `positions` int64 [R],
`input_ids` int32 [R], `stride` 8 — dtypes as in tcc's held capture files, checked on `capture/held/heldc0`).
**Training rows are fed to `qfn.fit.Sums.add` directly** (one call per document, chunks concatenated): Sums only
indexes the record dict, so writing `.pt` files first would double the disk writes for nothing. **Held docs are
written as `capture_NNNNN.pt`** (`convert.write_pt`) because `qfn.fit.Held` / `fit.records` glob files.
`final_multi_hidden` is not captured (the `final` map feeds only MTP): the Layout is `fit.Layout("plain")` with
`final = False`, so the solve fits layer.24..47 only.

HC identity check (`convert.py check`, and `hc_identity_check` in the fit report): `qfn.hc.mix(boundary_24,
layer 24's read weights from tcc's bf16 HC file)` vs radiance's `block_input_24`. tcc's own captures gave
0.9999978. Radiance's connection weights are E4M3 per group (served recipe), so < 1 measures the read-weight
numerics difference. **Measured on the first filed capture** (held0-rust raw, 537 rows):
`{"split": 24, "hc_identity_cos": 0.99996543}` (`data/refit/hc-check-held0-rust.json`) — the captured tensors are
the boundary and the layer-24 block input; the 3.5e-5 gap is radiance's stored connection weights vs tcc's bf16 ones.

**Engine shift, same tokens** (one-off diagnostic, radiance capture vs tcc's capture of held0-rust, 512 matched rows,
ids equal): raw — boundary_24 cos 0.9866, block_input_24 0.9716, block_input_31 0.9241, block_input_47 0.9109;
chat — 0.9896 / 0.9798 / 0.9552 / 0.9431. The late block inputs the shipped map was fitted to differ from
radiance's by ~0.06–0.09 in cosine, which is the room the refit has.

## 3. Capture driver (step 3) — `tools/refit/capture.py`

`run`: tokenizer probe first (two texts from manifest.json through `/tokenize` must give docs.py's ids); then each
prompt as ids to `/v1/completions` (max_tokens 1, temperature 0; prompt_tokens must equal the ids, cached 0). Its
records = the new jsonl lines, which must be exactly one per chunk (per rank for states), keyed by first position +
FNV-1a of the chunk's ids **recomputed in Python** (hash checked against a C++ copy of `chunk_key`: both
`60a834bd84e29423` on a test vector). Files go to `<store>/<set>/<format>/<source>-<name>/` (+ jsonl lines +
meta.json last, via `.tmp` + rename). `acc --follow`: qfn.fit.Sums per format in qfn.bigcap's checkpoint format
(`data/refit/sums/acc-{raw,chat}-s24.pt`), restored with `bigcap.restore`; a doc's capture dir is deleted only after a
checkpoint holding it is written (every 40 docs); row counts checked against the capture lines.

**Replayed passes.** A recorded pass replays without calling the plugin's step() (ARCH, ctx.cpp:1258-1278). The
sessions set `RADIANCE_DEBUG_ARGSHA=1`: `Ctx::prepare` turns recording off under it (`core/runtime/ctx.cpp:120`)
and its digest code only runs on draft pass 1 (`issue.cpp:819-866`; MTP is off, so never). Cost ≈ 0 (the live issue
path, ~1,850 host issues a step). Rejected: `--profile-ops` (same effect, but restores a sync around every launch,
`core/engine.cpp:818-821`, "several times slower"). The driver's count check is the gate either way.

## 4. Projector solve (step 4) — `tools/refit/fit_projector.py`

Sums → `mixed()` (w = 0.5·n_raw/(0.5·n_chat), qfn.fit's rule) → `qfn.fit.solve` (RidgeFit, lambda grid 0.001…0.3 =
the shipped grid) → lambda by held-out mean block-input cosine → `kva-radiance-s24.safetensors` (layer.24..47
[2560,10241] bf16, bias last; no `final`) → `report-radiance-s24.json`. The shipped projector is scored by
`fit.his_metrics` on the SAME radiance held rows; both cosines are recomputed from the bf16 files.

```
$ fit_projector.py --sums data/refit/sums --held data/refit/store/held --ckpt <tcc checkpoint> \
    --shipped ~/AI-Work/kva/qfn/proj/kva-big-s24.safetensors --out data/refit/proj --threads 10
20:27:42 held rows 7532; HC identity at layer 24: cos 0.999961
20:27:51 shipped projector on radiance held rows: {'bi': 0.75014, 'bi_gdn': 0.74993, 'bi_attn': 0.75077, 'kdir': 0.88927, 'ikdir': 0.90957, 'vrel': 0.15808}
20:32:24 lambda 0.001: {'bi': 0.75295, ...}   lambda 0.003: 0.75305   lambda 0.01: 0.75333
20:32:24 lambda 0.03:  {'bi': 0.75363, 'bi_gdn': 0.75352, 'bi_attn': 0.75395, 'kdir': 0.89088, 'ikdir': 0.91085, 'vrel': 0.15580}
20:32:24 lambda 0.1: 0.75240   lambda 0.3: 0.74573
20:33:05 solved in 310s; lambda 0.03; refit file on held rows: {'bi': 0.75363, 'bi_gdn': 0.75352, 'bi_attn': 0.75394, 'kdir': 0.89088, 'ikdir': 0.91085, 'vrel': 0.15580}
```
| | shipped (tcc fit) | refit (radiance) |
|---|---|---|
| held-out bi, radiance captures (7,532 rows) | **0.7501** | **0.7536** (+0.0035) |
| held-out bi, tcc captures (shipped report, 7,479 rows) | 0.7525 | — |
| kdir / ikdir / vrel (radiance rows) | 0.8893 / 0.9096 / 0.1581 | 0.8909 / 0.9109 / 0.1558 |
| lambda; rows raw / chat (weighted) | 0.03; 248,923 / 176,866 (497,846) | 0.03; 248,892 / 249,312 (497,784; chat weight 0.998) |
| HC identity at 24 (instrument) | 0.9999978 (tcc) | 0.9999608 (radiance) |

Per layer (refit − shipped, bi): every one of the 24 layers improves, +0.0012 (L24) … +0.0079 (L32); paired over layers
**+0.00349 [0.00292, 0.00410]** (`labbook compare heldout-bi-shipped-radiance heldout-bi-refit-radiance --paired`;
records seq after H6-refit-cos). Registered prediction H6-refit-cos was a gain in [+0.005, +0.04]: the direction holds,
the size is below the band (verdict is the orchestrator's). Reading: the shipped map loses only 0.0024 of held-out
cosine moving from tcc's activations to radiance's (0.7525 → 0.7501) even though radiance's late block inputs differ
from tcc's by 0.06–0.09 cosine on the same tokens (§2) — the map is near the S24 ceiling of a linear predictor on
either engine. So the refit is not expected to close most of R18's +6.4% vs +4.3% gap; R44's paired NLL decides.
Output `data/refit/proj/kva-radiance-s24.safetensors` sha256 `bedf9d77…f1b5` (1,258,416,408 B), report
`report-radiance-s24.json` (lambda grid scores, per-layer bi both files, doc keys, rows, checkpoint fingerprint).

## 5. Correction fit (step 5) — `tools/refit/fit_correction.py`

Pairs speed-run records (`approximate: true`) with the exact-run record of the same chunk key, rank and last
position; C_L,r = mean(S_exact − S_pred), f64 sums, written as `kva-radiance-s24-st.rank{0,1}.pt` in st_hook's
`{"sum","count"}` format (rank r = heads [24r, 24r+24), checked per record) → `ridgefill_sidecar.py build --names refit
--st`. Refuses < 50 chunk ends or < 13 prompts (R43). Report: relative state error, share of the error energy the
constant removes, cosine with the shipped C.

`session.sh speed` (after append #1; lock 01:38:40Z → 01:40:36Z; kernel log clean; `RADIANCE_RIDGEFILL=speed`,
`RADIANCE_RIDGEFILL_PROJ=refit`, `RADIANCE_RIDGEFILL_ST=refit` with no `ridgefill.str.*` held = no correction, `RADIANCE_RIDGEFILL_TAIL=512`):
14 prompts in 1.2 min, every chunk captured; the engine logged **55 approximate steps = the planned 55 chunk ends at
tail 512** (41 at tail 2048, so the env took effect); records: speed run 55 approximate + 15 tail per rank, exact
run 70 `off` per rank.
```
$ fit_correction.py --exact data/refit/state-exact --pred data/refit/state-speed --out data/refit/st --shipped <shipped rank0/1 .pt>
rank 0: 55 chunk ends, 14 prompts; rel error 0.192..0.624, constant removes 0.050..0.374 of the error energy
rank 1: 55 chunk ends, 14 prompts; rel error 0.180..0.547, constant removes 0.059..0.411 of the error energy
```
Per layer (rank 0 / rank 1, rel error |S_exact − S_pred|/|S_exact| ; share of error energy the constant removes,
in sample): L24 0.19/0.18 ; 0.12/0.35 … L44 0.55/0.53 ; 0.37/0.38 … L46 0.62/0.49 ; 0.10/0.16 (all 18 layers in
`report-radiance-st.json`). **Head order / sanity vs the shipped C** (one-off check): cosine refit vs shipped, same
rank: mean 0.923 (min 0.850) rank 0, 0.931 (min 0.838) rank 1; halves SWAPPED: mean 0.001 (max 0.018) — the
rank-0-first order is right, and the refit C points where tcc's did. Norm ratio refit/shipped 0.93–1.07 per layer.
Files: `data/refit/st/kva-radiance-s24-st.rank0.pt` sha256 `c80d2b5b…006d`, `.rank1.pt` `6d776ca7…da21`.
R43's "passes R20/R22 with the new weights" is an engine check after append #2 (GATES / orchestrator).

## 6. GPU sessions — `tools/refit/session.sh exact|speed` (whole serve → capture → stop under `flock gpu.lock`)

Frozen home `data/refit/home-3af4eaa` (git archive 3af4eaa, built in `radiance-build`; ctest 2/2 host tests pass):
`qwen4exp_fp8.so` c408596e…9e35, `ridgefill.so` 34062e78…000d. Serve = `scripts/serve.sh off|speed` (RK_FLAGS unchanged:
`--gpu-headroom-mib 3072` etc.), capture dir mounted rw via `RK_DOCKER_EXTRA`, kernel log checked before, between and
after (window from the lock). Recording off via `RADIANCE_DEBUG_ARGSHA=1` (§3).

`session.sh exact`, lock held 00:39:13Z → released 01:27:20Z (48 min), log `data/refit/session-exact-20261004T233238Z.log`:
- preflight OK (03:00.0 144 MiB / 13:00.0 70 MiB before serve); kernel log clean before, between and after.
- tokenizer probe: 2 texts tokenise identically through the served `/tokenize`.
- activations: 312 prompts, 4,044,768 tokens in 45.2 min (~1,490 tok/s incl. the capture's syncs and ~280 MiB
  device→host per chunk; exact TTFT alone is ~1,640 tok/s), **every chunk's record present** (count + key check per
  prompt, no replay). acc kept pace (40-doc checkpoints, ~15 s each); final sums: raw 151 docs 248,892 rows, chat
  151 docs 249,312 rows; all training capture dirs deleted after checkpoints.
- exact states: 14 +st prompts, 134,081 tokens, 1.5 min; records = chunks × 2 ranks for every prompt
  (`data/refit/state-exact/`, 3.7 GiB, f32 [18, 24, 128, 128] per chunk per rank).

Disk (`data/refit`, /var/home, after step 5 + append #2 prep): **24.2 GiB** — sums 11.0, proj 2.2 (incl. held .pt),
sidecar-projr 1.2, sidecar-full 1.2, held captures 1.0, exact states 3.7, speed states 3.7, stubs 2 × 23 MiB real
(335 GiB apparent each, sparse). Nothing on the model SSD. Deletable once Stage 6 is decided: the state captures
(7.4 GiB) and sidecar-projr (1.2 GiB); keep the sums (refit at a new lambda / share without recapture).

## 7. Appends (step 7) — prepared, NOT run

`tools/refit/append_prep.sh projr` (no GPU, container mounted read-only; outputs in `data/refit/append-projr/`):
```
verify .../sidecar-projr/kva-sidecar-refit.safetensors: PASS          (ridgefill.src.projr.sha256 bedf9d77… OK)
wrote data/refit/stub-projr: 131 header-only shards (335.3 GiB apparent), extra tensors
      {'kva-sidecar.safetensors': 87, 'kva-sidecar-refit.safetensors': 48}   (revision de4b8e4d…, as data/stub)
I plugin .../qwen4exp_fp8.so is shadowed by /ridgefill/data/refit/home-3af4eaa/architectures/qwen4exp_fp8.so
I plan     51624 weight(s): 50603 quantised by the recipe, 1021 kept as the checkpoint holds them
D rad_convert.cpp:659    ridgefill.projr.24.weight    bf16    50.00 MiB  as is
new                  48  ridgefill.projr.24.bias, ridgefill.projr.24.weight, ridgefill.projr.25.bias, ...
dropped 0   changed 0   unexpected_new 0   expected_missing 0
planned 51624, held 51576: PASS
```
Refit shard `data/refit/sidecar-projr/kva-sidecar-refit.safetensors` sha256 `575149a9…5297` (48 tensors, 1.17 GiB);
`set.txt` = the shipped set file ∪ the refit one (13 keys, `ridgefill.format` shared, no conflicts).

### 7.1 Append #1 — the projector

**The real append #1 — orchestrator only.** Preconditions: no engine container (`docker ps`, `pgrep -af
'radiance|rad-'`: rad-convert's open-file check cannot see another container); the container is still
123,364,379,720 B (= `rad-info-v-before.txt`'s state; else rerun append_prep); model SSD ≥ 2 GB free;
**save the Stage 2 restore record first** — the writer renames its new `.pre-append` over the old one
(`core/format/radfile.cpp:1321-1336`), and the Stage 2 record is the way back to the pristine container:
```
cd ~/projects/inference/radiance-kva
cp -p ~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad.pre-append evidence/stage6/pre-append.stage2   # 256 B
docker run --rm --security-opt label=disable -e RADIANCE_RIDGEFILL_DECLARE=all -e CALIB=calib/w4nl-calib \
  -v "$(readlink -f ~/models/rad)":/models -v "$(pwd -P)":/ridgefill:ro \
  radiance-build /stage/opt/radiance/bin/rad-convert /ridgefill/data/refit/stub-projr \
    --reuse /models/qwen3.8-next-flash-fp8-iq4r-moe.rad --in-place \
    --recipe /models/qwen4exp-w4nl64-i8-hc8m.recipe \
    --home /ridgefill/data/refit/home-3af4eaa:/stage/opt/radiance/share/radiance \
    $(sed 's/^/--set /' data/refit/append-projr/set.txt | tr '\n' ' ') --set ridgefill.mode=quality --set ridgefill.tail=2048 \
    -v > data/refit/append-projr/append.log 2>&1; echo "exit $?"
```
(= the plan-only run minus `--plan-only`, `/models` writable.) Expect `--in-place: 51576 weight(s), …, kept where they
were …; 48 added past its end`. After, in `data/refit/append-projr/`:
1. `rad-info -v` → `rad-info-v-after.txt`; `tools/plan_diff.py --plan plan-only.log --container rad-info-v-after.txt`
   (no `--expect-new`) must PASS (planned 51,624 = held 51,624).
2. `rad-info --meta` → `rad-info-meta-after.txt`; `diff` with `rad-info-meta-before.txt` shows only
   `ridgefill.src.projr.sha256` added; `tools/ridgefill_sidecar.py verify rad-info-meta-after.txt --names refit --proj
   data/refit/proj/kva-radiance-s24.safetensors` PASS (R13, refit) and the shipped verify (sidecar notes §7.6 2.) PASS.
3. Restore to post-Stage-2: the new `.pre-append`; to pristine: `evidence/stage6/pre-append.stage2` (same procedure,
   notes/sidecar.md §7.7 — both stay valid: an append writes only past the old end, plus the header).
**Done by the orchestrator 20:37** with one change: `--set ridgefill.mode=off` (no surprise defaults: every serve passes
the mode, scripts/serve.sh does). Plan diff after PASS (51,624 = 51,624); shipped + refit verify PASS; Stage 2 restore
record saved as `evidence/stage6/pre-append.stage2`.

### 7.2 Append #2 — the correction (`append_prep.sh full`, prepared, NOT run)

```
verify .../sidecar-full/kva-sidecar-refit.safetensors: PASS   (projr bedf9d77…, str0 c80d2b5b…, str1 6d776ca7… OK)
wrote data/refit/stub-full: ... extra tensors {'kva-sidecar.safetensors': 87, 'kva-sidecar-refit.safetensors': 66}
I plan     51642 weight(s): 50603 quantised by the recipe, 1039 kept as the checkpoint holds them
D rad_convert.cpp:659    ridgefill.str.24        f32       3.00 MiB  as is
new                  18  ridgefill.str.24, ridgefill.str.25, ridgefill.str.26, ridgefill.str.28, ridgefill.str.29, ridgefill.str.30 ...
dropped 0   changed 0   unexpected_new 0   expected_missing 0
planned 51642, held 51624: PASS
```
Full refit shard `data/refit/sidecar-full/kva-sidecar-refit.safetensors` sha256 `cae9d399…78db` (66 tensors); its
48 `ridgefill.projr.*` are **byte-identical** to append #1's shard (checked tensor by tensor), so the in-place writer
reuses them by name and only the 18 `ridgefill.str.*` (54 MiB) are new. `append-full/set.txt` = 14 keys (the shipped set
+ `ridgefill.src.projr/str0/str1.sha256`); with `ridgefill.mode` / `ridgefill.tail` on the command line it re-passes every `ridgefill.*`
key the container holds (meta diff: only `ridgefill.src.str0/str1.sha256` are added).
Preconditions: no engine container; container = 124,642,761,992 B (the post-append-#1 state the plan was made
against; else rerun `append_prep.sh full`); model SSD ≥ 2 GB free; keep a copy of the current `.pre-append`
(append #1's record, the way back to post-Stage-2) — the writer replaces it:
```
cd ~/projects/inference/radiance-kva
cp -p ~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad.pre-append evidence/stage6/pre-append.append1   # 256 B
docker run --rm --security-opt label=disable -e RADIANCE_RIDGEFILL_DECLARE=all -e CALIB=calib/w4nl-calib \
  -v "$(readlink -f ~/models/rad)":/models -v "$(pwd -P)":/ridgefill:ro \
  radiance-build /stage/opt/radiance/bin/rad-convert /ridgefill/data/refit/stub-full \
    --reuse /models/qwen3.8-next-flash-fp8-iq4r-moe.rad --in-place \
    --recipe /models/qwen4exp-w4nl64-i8-hc8m.recipe \
    --home /ridgefill/data/refit/home-3af4eaa:/stage/opt/radiance/share/radiance \
    $(sed 's/^/--set /' data/refit/append-full/set.txt | tr '\n' ' ') --set ridgefill.mode=off --set ridgefill.tail=2048 \
    -v > data/refit/append-full/append.log 2>&1; echo "exit $?"
```
Expect `--in-place: 51624 weight(s), …, kept where they were …; 18 added past its end`. After, in
`data/refit/append-full/`: `rad-info -v` → `rad-info-v-after.txt` and `tools/plan_diff.py --plan plan-only.log
--container rad-info-v-after.txt` PASS (51,642 = 51,642); `rad-info --meta` diff vs `rad-info-meta-before.txt` =
only `ridgefill.src.str0/str1.sha256` added; `tools/ridgefill_sidecar.py verify rad-info-meta-after.txt --names refit --proj
data/refit/proj/kva-radiance-s24.safetensors --st data/refit/st/kva-radiance-s24-st.rank0.pt
data/refit/st/kva-radiance-s24-st.rank1.pt` PASS (R13 refit) and the shipped verify PASS. Then serving the refit set =
`RADIANCE_RIDGEFILL_PROJ=refit RADIANCE_RIDGEFILL_ST=refit` (R44 runs: shipped vs refit × speed/quality, same boot).

## 8. Decisions and their cost

- **Same docs as the shipped fit, each raw + chat**: 3.98 M tokens to capture (~45–55 GPU min est.) instead of the
  shipped 3.4 M; buys an engine-only comparison. Chat share stays 0.5 by weighting.
- **sterm4 excluded, two b2 docs added** (§1): R43's 50 chunk ends need ≥ 1 more prompt anyway on radiance's whole
  chunks; cost: the +st prompt set is not byte-identical to the shipped fit's.
- **No `final` target**: saves 0.8 GiB of sums per format and a 10240×10241 map nobody reads (radiance's RidgeFill path has
  no MTP); the file is not loadable by tcc's engine (its loader requires `final`).
- **Sums fed in memory, held as .pt**: one conversion path (`read_chunk`), no duplicate disk writes for 4 M tokens.
- **ridgefill_sidecar.py `--names refit` takes `--st` optionally** (extended within its refit option, as the brief allows):
  the correction fit needs `ridgefill.projr` in the container and must run with NO `ridgefill.str` (RADIANCE_RIDGEFILL_ST=refit then
  = no correction), and an in-place append cannot replace a weight, so append #1 = projector alone, append #2 = the
  full refit set (projector bytes identical, reused by name). Tested (`tests/test_refit_sidecar.py`).
- **acc runs beside the capture** (CPU, 6 threads, ~12 GiB of fp64 sums + temporaries) so the GPU lock is released
  when the last prompt is filed; disk stays bounded (≤ 40 docs of captures, ~35 MiB a 2048-token chunk).

## 9. Commits

- `270b6ba` tools/refit (docs, convert, capture, fit_projector, fit_correction, session.sh), tests/test_refit_*,
  ridgefill_sidecar `--st` optional for refit.
- `60919e9` append_prep.sh + these notes; later commit: append_prep physical paths (a stub symlink computed between
  /home and /var/home spellings pointed nowhere inside the /ridgefill mount; fixed with `pwd -P` / `readlink -f`).

## Appendix — prompt list (text sha256, first 16 hex; tokens after any cut)

**train** (302 prompts, 3,984,554 tokens)

| doc | raw sha256[:16] | raw tokens | chat sha256[:16] | chat tokens |
|---|---|---|---|---|
| fncap2-trainc58 | 920b094537a9ca42 | 11915 | e0767804157cb034 | 11938 |
| fncap2-trainc19 | dba0b86388966378 | 11950 | 53870ec7e5032977 | 11972 |
| fncap2-trainc38 | 78d285b4b110d7fa | 11534 | 66baaab3e308ebe0 | 11555 |
| fncap2-trainc98 | 6c4c5050fc07e31f | 10184 | f167d2a9268496b9 | 10207 |
| fncap2-trainc6 | fe9e15f633f0f624 | 11908 | ecaf67b20bd393a8 | 11930 |
| fncap2-trainc32 | 7391187e657bbd20 | 9273 | 9721441ae9a4b842 | 9295 |
| fncap2-trainc72 | 522e42a96d731c71 | 11957 | af53689328cb1ce9 | 11978 |
| fncap2-trainc21 | 3b9b376632cf2c96 | 11406 | 07e336006b7ec110 | 11430 |
| fncap2-trainc13 | ff47296feb9dd87a | 10815 | 3366ffc387faf02e | 10837 |
| fncap2-trainc39 | 87f60eebb7601a6c | 11760 | a6f77fbc3a64444d | 11782 |
| fncap2-trainc94 | e6ea6013c4e3313b | 22232 | b9cefc9822209571 | 22254 |
| fncap2-trainc9 | 96cb9d41757aaf90 | 10840 | dd8d3568291ae655 | 10861 |
| fncap2-trainc8 | fb9cc6068a54c8c6 | 11502 | 95ebaf08983bdb2d | 11523 |
| fncap2-trainc22 | 8f40e78a676933e7 | 11747 | f8309cf640bc85b8 | 11770 |
| fncap2-trainc75 | c82b21846605d75e | 11708 | 2e933777581081e4 | 11729 |
| fncap2-trainc1 | 1c55e63e07846e33 | 10961 | 0a911f84fa90449e | 10983 |
| fncap2-trainc89 | 165df302ba4b0ee8 | 11856 | 3b98e6ae8285aa8f | 11877 |
| fncap2-trainc47 | 9fb52839b7a1d13b | 1887 | 78269dace284f0a1 | 1910 |
| fncap2-trainc25 | 5008303fe6691003 | 11824 | 3d5ba24c080ea1c3 | 11846 |
| fncap2-trainc27 | b89a2aa3a70619fb | 11407 | cf7702a87c60383e | 11428 |
| fncap2-trainc37 | 5bb3ac52d6c04956 | 11699 | 68b16371e40bd5be | 11720 |
| fncap2-trainc84 | 6182534ba78e4ec8 | 8816 | 118b31fc2741cf16 | 8839 |
| fncap2-trainc100 | 637341ea6d09595a | 11637 | 1302e032b4105a50 | 11658 |
| fncap2-trainc36 | 58065dc1f36bd2e2 | 11708 | aa28ffa208505645 | 11729 |
| fncap2-trainc17 | a1fa4d29b12fda80 | 10744 | 54bc0069306294c9 | 10766 |
| fncap2-trainc50 | b9deb1a69e1bdcfe | 22482 | 257fe21d443baf42 | 22504 |
| fncap2-trainc10 | 4bac873e1fa34f4a | 22453 | f5f4f4daad5705d8 | 22475 |
| fncap2-trainc80 | 2716277b32973d8a | 11250 | 0c49141e102ede29 | 11273 |
| fncap2-trainc11 | 2056d9648561974b | 11603 | 21924051ee4cc90b | 11625 |
| fncap2-trainc28 | c38f5697c56d7464 | 10815 | 92f99477f4e6457c | 10837 |
| fncap2-trainc88 | f00960d9ae84a0a1 | 11120 | fa223cae2ea437b8 | 11143 |
| fncap2-trainc29 | 6a7686e77a8a7532 | 11819 | 2a5fb67014a73f8f | 11841 |
| fncap2-trainc15 | f6afc3786fc0f63d | 11526 | fce39af8d7b95dac | 11547 |
| fncap2-trainc91 | 3fb2b3827365c494 | 11644 | 17f2d5ce199e91a8 | 11665 |
| fncap2-trainc87 | c31ce8202d450074 | 10846 | 90ee03cac7b917e7 | 10868 |
| fncap2-trainc99 | 94efa585f178ac7a | 11075 | 5660c83a59ac61f7 | 11097 |
| fncap2-trainc24 | 68f5ebe231873891 | 10094 | 4b60acc5aa660b16 | 10116 |
| fncap2-trainc63 | 399de4b50df1812e | 2698 | 6ceda79bc68ff155 | 2720 |
| fncap2-trainc48 | 8a91a38840d10acf | 30465 | 1029abcc0591d923 | 30486 |
| fncap2-trainc33 | 133fab0732c1150f | 11689 | f112c8b0cb381836 | 11711 |
| fncap2-trainc70 | 7a24a6297b94ac78 | 10236 | 361491cfae58e783 | 10258 |
| fncap2-trainc74 | 81461035a41a90a2 | 14259 | 6596fb98a2565cff | 14280 |
| fncap2-trainc14 | f0cfc3271919132b | 11611 | 2bb0469d6e2063ab | 11633 |
| fncap2-trainc95 | 211f1e1056c978bf | 12005 | f559d5fb2e00b9d1 | 12026 |
| fncap2-trainc18 | 1a42db65f05e8f4f | 11631 | fb8ed18425608805 | 11653 |
| fncap2-trainc61 | 64a500bcc20da4bf | 4571 | 63a5b46ec0fa92bd | 4593 |
| fncap2-trainc2 | 35f24c2e2bebe000 | 11659 | 05393d8d1aae70e2 | 11683 |
| fncap2-trainc16 | 0ec2a45b6272417c | 11203 | 360eff46f821f110 | 11225 |
| fncap2-trainc30 | 9e4e0d0afc36d94d | 11563 | fbdabf06c9481b2a | 11585 |
| fncap2-trainc62 | 93e0c76b7d9e3270 | 22802 | ac1d3dda52375cba | 22826 |
| fncap2-trainc76 | 88e0bf094964b686 | 10944 | 4626f55ba8ae16e7 | 10965 |
| fncap2-trainc101 | 0b32889a0e0764dc | 2028 | fde8ee8913673c2a | 2049 |
| fncap2-trainc46 | 1863fcf53154f6f1 | 11951 | d057f9f910713f2f | 11972 |
| fncap2-trainc60 | cf94a583a32aeb3c | 10784 | 08add6721c587b5d | 10808 |
| fncap2-trainc23 | 295b2ce01d235297 | 11814 | d5536aa76200f150 | 11835 |
| fncap2-trainc67 | 4ff4d241b53bf8b4 | 12656 | 2b0dd6a4d8bad979 | 12677 |
| fncap2-trainc43 | 781c6f2ec26244fe | 5553 | fcac5a871a761a8d | 5574 |
| fncap2-trainc4 | 7eb764a3e52657bd | 11816 | 5f3741afa14b119f | 11840 |
| fncap2-trainc26 | ad98a282b8c02648 | 11433 | 7222fca3c2b7b3fe | 11455 |
| fncap2-trainc5 | 08c385c7da23eda2 | 9328 | bbfca7c21afa19cf | 9349 |
| fncap2-trainc44 | 8b9af0f70c8a272e | 12000 | 053ff708df6baa7f | 12024 |
| fncap2-trainc51 | d83ec5989a03e935 | 9576 | 23bb1a85505c30c0 | 9597 |
| fncap2-trainc53 | 380aadd9d8000bf1 | 24295 | 8de882869f103ae7 | 24316 |
| fncap2-trainc66 | f333c9fd82903745 | 1699 | ade259ab20e7a6f2 | 1722 |
| fncap2-trainc42 | aac49261209d7e16 | 9992 | 99c960615ad8ea03 | 10013 |
| fncap2-trainc85 | 1ddd150d037cfa03 | 11900 | 061230575972c961 | 11921 |
| fncap2-trainc55 | 314c4eb3eafc2940 | 11925 | 9167f49044905b33 | 11949 |
| fncap2-trainc92 | 97822fc0e37e25f2 | 9864 | d940601660bcee15 | 9885 |
| fncap2-trainc71 | 87eb6f19b3a62adb | 10245 | 6c436217b74b5787 | 10268 |
| fncap2-trainc64 | 81638fb1018a587a | 22251 | fdc494d981e65bf2 | 22273 |
| fncap2-trainc77 | 3fe2169e938565ca | 11209 | 28aed99268323c21 | 11230 |
| impact-code-000 | abed89e18d054eec | 18584 | c21efe11a9a48c47 | 18605 |
| impact-prose-000 | 2398551b4e2670f6 | 18933 | 408d748ae80e4cc7 | 18956 |
| impact-rust-000 | 9e3407341ba8f904 | 11885 | 56965867baa282d5 | 11906 |
| impact-techdocs-000 | 3c52dc3a1051c9c6 | 11660 | b453898698d6125e | 11681 |
| impact-web-000 | 7fd17797d8b8865e | 10698 | 0342217850140bea | 10722 |
| impact-code-001 | bf017154e2d50986 | 13168 | 1ee25731dbc1ad8f | 13189 |
| impact-prose-001 | 8d6860138a79cd3e | 19014 | 89f708fc8357c463 | 19035 |
| impact-rust-001 | 04b60b7a00bcf26b | 16271 | 60a773392acafa3d | 16292 |
| impact-techdocs-001 | 67079d26b3713046 | 11563 | 9911c654aeffdf61 | 11584 |
| impact-web-002 | 8ef4eafe4d26d033 | 11206 | 474c16a108f9b43a | 11227 |
| impact-code-002 | a3d7741c8ba97c24 | 13747 | 71c1eda6eb20ece6 | 13770 |
| impact-prose-002 | 59efbe093db81e5c | 12517 | e1927a543d6858e5 | 12538 |
| impact-rust-002 | 6af7af33edec32b9 | 13583 | d01b22cee2e2ec5e | 13606 |
| impact-techdocs-002 | a0e352d9011e6aff | 15306 | 2c9201beeb826f9d | 15329 |
| impact-web-003 | 5b8e0f6b30179607 | 9778 | d3812db5188a104c | 9800 |
| impact-code-003 | 3e96f8d107954364 | 19264 | bd1452d996153139 | 19285 |
| impact-prose-003 | 126991695bd3a94c | 13834 | 2df8f3a68a8e10a5 | 13855 |
| impact-rust-003 | 7db4c69d40c76bb7 | 15335 | 6c47eaeae7b888fd | 15356 |
| impact-techdocs-003 | 6e064a2e21c848ea | 11244 | fa0df9ee131670f4 | 11265 |
| impact-web-004 | f7306e394165f62e | 10296 | 49243b1eef44b687 | 10318 |
| impact-code-004 | ce344e179716223e | 18228 | dee212e3a4f7fbe5 | 18249 |
| impact-prose-004 | 5326d1d7293f4bdb | 14663 | 4a7669d9d7f7a32c | 14686 |
| impact-rust-004 | 6bd17accd8df0b09 | 12968 | 4a2ba6d8a48fbb3d | 12989 |
| impact-techdocs-004 | fc181f545d13d49b | 12167 | b13e21859f2d8d41 | 12188 |
| impact-web-005 | cc7e58e447fdf62e | 13481 | e2f3761c97860c14 | 13505 |
| impact-code-005 | 910cf61496233b23 | 19383 | 2178f635d6ce30ef | 19404 |
| impact-prose-005 | e16f254d5a7b524f | 12897 | 4277483dc3632bb6 | 12918 |
| impact-rust-005 | 4d641e304d51c801 | 19406 | a708d3d86f7c4093 | 19427 |
| impact-techdocs-005 | 4e075c44b09214b5 | 15737 | 9b0b68ac498f1c92 | 15758 |
| impact-web-006 | 63901e45e3e6e9f5 | 9122 | e9d4ab21a75e8cd7 | 9144 |
| impact-code-006 | 771f1106de3ee07a | 14552 | ff2020240d1c3be9 | 14575 |
| impact-prose-006 | 388663b148e64433 | 12772 | 8d4083159ad35a60 | 12793 |
| impact-rust-006 | 75c975d378fab1e1 | 11007 | 8fb26a603c7a6ecc | 11030 |
| impact-techdocs-006 | 030b7d8f50862225 | 13733 | 3dda5e48a0e33247 | 13756 |
| impact-web-007 | 1a58f52272b5d039 | 14871 | 5b51f49b74d4532e | 14892 |
| impact-code-007 | d86e7c52703660cc | 13947 | e142879f2afa9d1f | 13968 |
| impact-prose-007 | c515dda878dc1c75 | 14778 | 4044a53d6c87531c | 14799 |
| impact-rust-007 | 98be33f4b2f76488 | 19261 | ad1337bd5bb10058 | 19282 |
| impact-techdocs-007 | cc072450b3090d29 | 16696 | ebfb5d5b9241fd19 | 16717 |
| impact-web-008 | 40ae6d90ff04f2a3 | 10523 | 67786ecada4003e5 | 10544 |
| impact-code-008 | 0f3d1889564ffea4 | 10644 | 3cd1161acc39cc41 | 10665 |
| impact-prose-008 | 8d016571c22bfca2 | 18873 | 08da9c814e5a6fe6 | 18896 |
| impact-rust-008 | 0b18170a9332788c | 10335 | 7204e9597b770a83 | 10356 |
| impact-techdocs-008 | 0f9acef927829432 | 11271 | 578f0d9b2cdde6f7 | 11291 |
| impact-web-009 | 764a9dfd99ae4713 | 11497 | 183e23e60f8c4c0f | 11520 |
| impact-code-009 | b7d4ee847cde5247 | 14109 | f4dd4408d41fb0a8 | 14130 |
| impact-prose-009 | 7804d1dab3ec9158 | 13654 | d4ef82931ac37bd9 | 13675 |
| impact-rust-009 | 1f5d889ff05e9df5 | 19734 | fcc2a6b340d8d534 | 19755 |
| impact-techdocs-009 | 0d9b078ff250f5b7 | 18838 | b03cc34b33283356 | 18859 |
| impact-web-010 | 066f453a4f3478a8 | 8798 | ef319755171f93ca | 8820 |
| impact-code-010 | 48339e4c7314fb0a | 16646 | d577a3210810aeab | 16669 |
| impact-prose-010 | 321f2677472dbff5 | 18468 | 901eae3ce696d780 | 18489 |
| impact-rust-010 | b2a51ea769c514af | 11249 | 80bcf60a1748fd7f | 11272 |
| impact-techdocs-010 | bc517408e62b90fd | 12196 | 6cfe12fd4049eb4b | 12219 |
| impact-web-011 | ac4cbb8776cc4cc3 | 12889 | 654d5522ff39abcb | 12911 |
| impact-code-011 | 0792a3526c97095f | 13917 | 16c92f830945341c | 13938 |
| impact-prose-011 | c0485e7ffcf9e3cd | 12605 | 9eb04f5eeeca6e47 | 12626 |
| impact-rust-011 | 927d9ac79e1f93d9 | 17130 | 24f4af91cc6b5c95 | 17151 |
| impact-techdocs-011 | ce7a242f8e95bf98 | 16562 | f8a91300f1b466c6 | 16583 |
| impact-web-012 | 91a0026885fd54c7 | 15440 | a01f03d26c81a860 | 15462 |
| impact-code-012 | 2fd159a3e93c9ac3 | 16393 | 292e6d545c60228b | 16414 |
| impact-prose-012 | 32efceffc8160965 | 11602 | 2a66e1abfa31f6c1 | 11625 |
| impact-rust-012 | 3a1594aa94777a4f | 15914 | aec139a6c0c1ee61 | 15935 |
| impact-techdocs-012 | bc52f449fd7117b8 | 15602 | 8d3ffb4a686b54b7 | 15623 |
| impact-web-013 | db0f23df95c5e0b7 | 12034 | 27aa91c5c5c04d70 | 12058 |
| impact-code-013 | 585a3fd4551dbb8b | 18413 | c11156b323ccd91e | 18434 |
| impact-prose-013 | 79e5c829dbcc5957 | 13516 | 83e3296b1d5454f3 | 13537 |
| impact-rust-013 | 3164ab81587d789f | 14324 | d1db780faeba11cb | 14345 |
| impact-techdocs-013 | b18daeb69e365ad0 | 19124 | 424ea3281f1f93c4 | 19145 |
| impact-web-014 | 2e66765eb257f3af | 19098 | 9cf4d0e3d7b99bb6 | 19119 |
| impact-code-014 | 7993e84cdd43afad | 15517 | 348216260734b5a0 | 15540 |
| impact-prose-014 | 923e2d3f11d4621e | 14686 | a1f3908a315a1736 | 14707 |
| impact-rust-014 | 9505bd24894c412c | 12622 | 93269e876cad1993 | 12645 |
| impact-techdocs-014 | a68eaf698cdf9628 | 11802 | c8505d969e5254ef | 11825 |
| impact-web-015 | 5a5dfb9780153616 | 10599 | 6a3c90dc9acb4b3f | 10620 |
| impact-code-015 | 103c860e538c1fb8 | 19548 | 5a8b2ae17e591a27 | 19569 |
| impact-prose-015 | e31faa159faec4d1 | 12209 | 626eca58c9275625 | 12230 |
| impact-rust-015 | 38dbab682972c348 | 18505 | a52a758d1cc9ae63 | 18526 |
| impact-techdocs-015 | d956bb72f2933ace | 16977 | ad6702ca68d7f121 | 16998 |
| impact-web-016 | ac76b2b1d463f724 | 9564 | 65ce9f726be2c6aa | 9586 |

**held** (10 prompts, 60,214 tokens)

| doc | raw sha256[:16] | raw tokens | chat sha256[:16] | chat tokens |
|---|---|---|---|---|
| fncap2-held0-rust | 396c62bfd2eacd1e | 4293 | 3617b9732a09da72 | 4314 |
| fncap2-held1-code | aba2d481a89308bb | 4821 | e2ec5b1d02c548e4 | 4845 |
| fncap2-heldc0 | 55bee35a9c924c7b | 7036 | 7903d0b59d7dda0c | 7057 |
| fncap2-heldc1 | a17a53e4979fa2ea | 6857 | a7d520300a46bf18 | 6878 |
| fncap2-heldc2 | b89549aa878b257d | 7046 | bd3a9123122108a1 | 7067 |

**sterm** (14 prompts, 134,081 tokens)

| doc | raw sha256[:16] | raw tokens | chat sha256[:16] | chat tokens |
|---|---|---|---|---|
| sterm-sterm0 | f441626eda868e9e | 9316 |  |  |
| sterm-sterm1 | 271dbe438dde4c75 | 8928 |  |  |
| sterm-sterm2 | 219866cd8ff012b6 | 9437 |  |  |
| sterm-sterm3 | ee375d45b0735999 | 10016 |  |  |
| sterm-sterm5 | a3af7f128370f6cd | 10189 |  |  |
| sterm-sterm6 | 176b35731c1d7679 | 10274 |  |  |
| sterm-sterm7 | 950d94b44ad5914a | 9223 |  |  |
| sterm-sterm8 | 1a306604cf1d9437 | 9907 |  |  |
| sterm-sterm9 | e08bfda5f064896b | 9793 |  |  |
| sterm-sterm10 | 6784eb9fb1da2320 | 9954 |  |  |
| sterm-sterm11 | aeba865a8b4acce7 | 9892 |  |  |
| sterm-sterm12 | 3d637f13b11ae8fa | 6672 |  |  |
| impact-code-016 | bc239631f55d2969 | 10240 |  |  |
| impact-prose-016 | 4dcf9a93e3551942 | 10240 |  |  |
