# notes/aprime.md -- Stage A′: the projector folder loader (PACKAGING.md §0-§7, R140-R149)

Worker: account-B Opus 5.5, worktree `radiance-kva-wt-aprime`, branch `stage-aprime` off `05a9db0`
(A.1 static + the template tool). radiance `140987f` (v1.0.8), read only. Boot `75e3e39b-cc5b-49de-8087-a4791f372a92`.
Container: `~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad`, STILL APPENDED (124,720,163,232 B, ridgefill.* weights and
keys inside) -- the loader declares none of them; R144 (restore to `0af5e962…4d20`) is the orchestrator's, after which
the gates re-run on the pristine file.

## 1. What was built (commits)

| commit | what |
|---|---|
| `3d69eda` | ridgefill.so: `ridgefill_gemm_nt_bias` (R140) + the fitted tensors as IN operands (`ridgefill_mask` score, `ridgefill_state_correct` C) |
| `1c9e763` | arch: folder discovery (R149), manifest + fingerprint with the DD-K refuse/warn split (R142), per-rank upload (R141), `RADIANCE_RIDGEFILL_PROJ_PLACE` |
| `232b8f4` | `tools/ridgefill_projector.py build` + tests; the append route (`ridgefill_sidecar.py`, `stub_checkpoint.py`, `plan_diff.py`) `git mv`'d to `tools/dev/` with a README |
| `9bba8fb` | README: the switch table and intro for the folder |
| `d3262dc` | the staging ring for host placement (DD-L, `RADIANCE_RIDGEFILL_PROJ_RING`, default 1 pending R148), a sizing declare copies nothing |

### 1.1 ridgefill_gemm_nt_bias (kernels/forward.cpp)
Schema = libr4d's `gemm_nt_bias` (`r4d_rows.cpp:1519-1529`) with `b`, `bias` as IN. Its rows ARE the engine's rows:
at the first `rad_kernel_count` (the loader calls it in `commit_`, after it has dlopened every plugin of every home,
`loader.cpp:563-575`) ridgefill.so walks the loaded objects (`dl_iterate_phdr`), re-opens each with `RTLD_NOLOAD`, asks
`rad_plugin_info()->name`, and copies libr4d's DEVICE `gemm_nt_bias` row and libref's HOST row, renamed onto
`ridgefill_gemm_nt_bias`: same `launch`, `describe`, `init/fini`, `scratch`, constraints, `opd_shape`; `rad_kernel_concurrent`
answers libr4d's own answer for it (ridgefill.so now exports the hook; its own rows answer 0, as before when it had none).
libr4d's launch reads operands positionally with no kind check (`r4d_vit_bf16.hip:313-380`) and `hipLaunchKernel`
resolves its stub through the engine's global registry (`aql.cpp:2061-2075`), so the kernel and its bytes are
gemm_nt_bias's. No source loaded ⇒ no row ⇒ `RAD_OP` null ⇒ the arch refuses startup by name (the existing R31
rule for a missing kernel library; an engine without libr4d cannot serve on this card anyway). A row with
layout hooks or tunables is not forwarded. **Cost:** none measured yet (R143 TTFT).
`ridgefill_state_correct`'s C is now read as dense `[heads, sd0, sd1]` of any rank (an IN operand arrives as RAD_P_T2).

### 1.2 The folder (arch/ridgefill_folder.h, ridgefill_json.h)
Discovery at the first real declare (process-wide, once, mutex): `$RADIANCE_RIDGEFILL_PROJECTOR` (explicit: nothing else
is tried) → `dirname(--model as typed)/projector` if the typed path is the mapped file (device+inode) →
`dirname(mapped file)/projector`. The mapped file = the first `/proc/self/maps` path whose first 4 bytes are
`RAD_MAGIC`. Read: `ridgefill.json` format 1, every listed file mmapped and sha256'd (one thread a file), safetensors
headers parsed, tensors keyed by their own names; a missing / corrupt file or a duplicate tensor refuses the
folder naming the file. Mode `off` never looks.

### 1.3 The match (arch/ridgefill_match.h; Dylan's DD-K split, binding)
Cannot run ⇒ folder REFUSED by name, nothing declared, the in-tree graph: adapter/arch_id; every `model.meta` key
(the container's own values as the engine hands them to plugins); the canonical tokenizer hash; then the
qwen4exp shapes (split a late layer above the n-gram layer; proj.L.weight `[n_embd, hc·n_embd]` + bias for every
L ≥ S; corrections all-or-none `[heads_all, V, K]`; row tables `[n_vocab_all]`); quality without the selected row
table. Another variant ⇒ WARNING naming both sides, RidgeFill runs: `model.encodings` (rad_weight_encoding +
rad_enc_format of every non-expert weight of layers S−1…47, 489 names), `model.anchors` (sha256 of the planes of
`blk.23.attn_hc_norm`, `blk.23.ffn_hc_norm`, `blk.24.attn_hc_norm`), the model name.
**The tokenizer hash is canonical, not the section's bytes**: each token's text, NUL, type byte in id order,
then the merge table. Why: an in-place append rebuilds the vocab section with new string-blob offsets, so a byte
hash would differ between the appended and the published file for the same tokenizer. tools/ridgefill_projector.py
computes the same; both tests assert the same two constants for the same tiny container.

### 1.4 Memory (arch/ridgefill_projector.h)
Each rank's real declare copies what its mode reads into ONE `rad_dev_alloc(RAD_MEM_DEVICE)` block (256-byte
aligned pieces), `rad_memcpy_async` from the mapping on a plugin stream + `rad_stream_sync`: speed = 24 maps +
biases + this rank's correction heads; quality adds the selected row table; plumb copies nothing (it reads no
fitted tensor). **Deviation from the append (better, not worse): each rank copies only ITS value heads of the
correction** (`[rank·H, rank·H + H)` of `[48, 128, 128]`, 27 MiB) instead of the replicated 54 MiB the append
needed (the loader has no ROW share of a rank-3 weight) -- the kernel reads exactly the bytes it read before.
`RADIANCE_RIDGEFILL_PROJ_PLACE=host`: the maps + biases in one `RAD_MEM_HOST_MAPPED` block per rank, written on the
host, read through `rad_dev_device_ptr`; correction and row table stay in VRAM. Uploads are keyed by
(mode, place, row table): a process serves one configuration, so they are made once per rank. Freed in
`rad_plugin_close`. An allocation failure on a rank refuses startup by name (the ranks cannot disagree).
Sizing declares copy nothing; tools without RADIANCE_RIDGEFILL never look.

### 1.4b The staging ring (DD-L, host placement only; arch/ridgefill_layer.h ring_copy/ring_next)
The host block holds each late layer's map + bias as ONE `[n+1, hc·n]` bf16 row block (bias = the first n elements of
the last row); two VRAM slots of that size (2 × 50.0 MiB a card) take turns, `slot[L % 2]`. Per approximate pass:
after the prologue, `join(0→1)`, lane 1 copies layer S (in-tree `cast` bf16→bf16, libr4d's strided row copy, M = n+1
rows), lane 0 resumes; before layer L's projector GEMM: `join(1→0)` (L's copy landed), `join(0→1)` (the slot's last
reader, layer L−1's GEMM, two layers back for this slot, is already handed to lane 0), lane 1 copies L+1. So each map
crosses the link once a chunk and each copy overlaps a layer. The GEMM operands are then the slot pointers (fixed per
layer: tape-safe). `RADIANCE_RIDGEFILL_PROJ_RING=0` = zero-copy (the GEMM reads the host map in place). Cost: 100 MiB of
VRAM a card in host mode only, 26 extra issues a pass (one copy + joins per late layer).

### 1.5 Config
Container metadata no longer read for ridgefill.tail / ridgefill.rowsel.share / ridgefill.rowsel.seed (the defaults 2048 / 0.25 / 0
equal what the append set). `RADIANCE_RIDGEFILL_PROJ`, `_ST`, `_DECLARE` refused by name ("retired with the container
append"). New: `RADIANCE_RIDGEFILL_PROJECTOR`, `RADIANCE_RIDGEFILL_PROJ_PLACE=vram|host`. `RADIANCE_RIDGEFILL_ROWSEL_TABLE`
selects `score|score_none|score_all` in the folder.

## 2. The folder as built (data/projector-qwen38fn, git-ignored)

```
$ tools/ridgefill_projector.py build --proj ~/AI-Work/kva/qfn/proj/kva-big-s24.safetensors \
    --st ~/AI-Work/kva-flashnext-hf/model/st/kva-big-s24-his-s24.rank{0,1}.pt \
    --freq ~/AI-Work/kva/pred/tcc-qwen38-flash-next-mxfp4-fp8-gptq-freq.safetensors --tokenizer data/stub \
    --container ~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad --rad-info-v evidence/stage2/rad-info-v.txt \
    --spec ~/AI-Work/radiance-kva-plugin-20261004/worker-context/kva-marker-spec.json --out data/projector-qwen38fn
split 24; 24 projector layers; 18 correction layers; rowsel 167288 of 248320 rows kept (248077 tokenizer ids)
merged: data/projector-qwen38fn/chat_template.jinja = 2853-byte ridgefill block + .template-source.jinja (8952 bytes, unchanged)
wrote data/projector-qwen38fn: 29 files, 1318066885 bytes (1.228 GiB); model qwen4exp, vocab 17f26a1adbf6..., 489 encodings, 3 anchors
```
18.5 s. Layout: `ridgefill.json`, `proj.L24.safetensors … proj.L47.safetensors` (each `proj.L.weight [2560, 10240]` bf16 +
`proj.L.bias [2560]` bf16, 52.4 MB), `correction.safetensors` (`st.L [48,128,128]` f32 × 18, 56.6 MB),
`rowsel.safetensors` (`score`, `score_none`, `score_all` f32 [248320]), `chat_template.jinja`, `README.md`.
No `final.safetensors` (MTP map: Stage D). ridgefill.json sha256 `$(sha256sum ridgefill.json)` see §4.
Checks: all 69 tensors byte-identical to `data/sidecar/kva-sidecar.safetensors` (the appended bytes); the template
extracted from the container equals `worker-context/container-chat-template.jinja` (sha256). The model's header
name is empty, so its name is the arch id `qwen4exp` (the engine does the same, `engine_bringup.cpp:358`).

## 3. Static results (host build, build-host; HIP build in radiance-build: `ctest -LE gpu` 3/3)
- `arch_static_test`: 45 cases, 680,577 checks, all ok. New: off never looks for the folder; a mode with no usable
  folder (none, other arch, other dims, other tokenizer, misshapen map, hole in the maps, split at the n-gram
  layer, half a correction, quality without a row table) declares the in-tree graph exactly and says why;
  another variant (encoding, anchor, name) warns ×3 and runs; retired switches refused; RAW operands trace to
  their tensors' copies (TP1, TP2 both ranks: head slices); host placement; reading a folder and five damaged forms;
  discovery by env and by a MAPPED model file in the test process; the container hashes; the JSON reader.
  Every case starts with no folder (a per-case reset wrapper), so no case leaks its folder into the next.
- `kernel_test host` (libr4d + libref preloaded): forwarded row == libref's hooks; 6 shapes bytewise equal to
  libref's gemm_nt_bias; `ridgefill_kernels_alone` (nothing preloaded): no ridgefill_gemm_nt_bias row.
- pytest: 146 passed, 11 skipped (the skips need real data).
- Symbol check: every `rad_*` the arch .so imports is exported by the 1.0.8 runtime binary.

## 3b. Mutation run (scratch copy `/tmp/aprime/mut`, `/tmp/aprime/mutate.py`, not committed) -- every mutant caught
```
A1 off looks for the folder                         -> off_with_a_projector_folder_still_declares_the_in_tree_graph
A2 metadata mismatch not refused                    -> a_mode_without_a_usable_projector_serves_the_in_tree_graph
A3 tokenizer not checked                            -> a_mode_without_a_usable_projector_serves_the_in_tree_graph
A4 an encoding difference refuses (instead of warns)-> a_projector_fitted_on_another_variant_warns_and_runs
A5 correction not sliced by rank                    -> the_raw_operands_point_at_their_tensors_copies
A6 host placement ignored                           -> host_placement_puts_the_maps_in_host_mapped_memory, the_staging_ring_...
A7 ring: no wait for the copy                       -> the_staging_ring_copies_each_map_a_layer_ahead_on_lane_1
A8 ring: slot overwritten before lane 0 is past it  -> the_staging_ring_copies_each_map_a_layer_ahead_on_lane_1
A9 corrupt file accepted                            -> a_folder_is_read_and_a_damaged_one_refused_by_name
A10 no resolved-path fallback                       -> the_folder_is_found_by_the_env_then_beside_the_mapped_model
A11 env not exclusive                               -> the_folder_is_found_by_the_env_then_beside_the_mapped_model
A12 sizing declare uploads                          -> a_sizing_declare_matches_the_real_one (after adding its "sizing alone" block)
A13 quality without a row table served              -> a_mode_without_a_usable_projector_serves_the_in_tree_graph
A14 hole in the maps accepted                       -> a_mode_without_a_usable_projector_serves_the_in_tree_graph
A15 retired switches accepted                       -> the_container_routes_switches_are_refused_by_name
A16 canonical vocab skips the type byte             -> the_container_hashes_are_the_builders (+ every folder case: the hash refuses)
A17 forward offers a row with no source loaded      -> ridgefill_kernels_alone (gemm_forward_offers_no_row_without_its_source)
```

## 3c. The pristine file fingerprints the same (checked BEFORE R144, without touching the file)
The append left the original tables in place, so the published file is readable through its original header,
`evidence/stage6/pre-append.stage2` (old length 121,969,901,568). Read that way (tools/ridgefill_projector.py's `Rad`
with the old header): arch/name, all 11 manifest metadata values, the canonical tokenizer hash, the three anchors and
the chat-template source hash all EQUAL the folder's manifest (built from the appended file); no `ridgefill.*` metadata.
So after R144 the folder should match with 0 warnings. **For R144: the `.pre-append` beside the container is append
#2's record (it restores the 124,642,761,992-byte append-#1 state), not the published file's -- use
`evidence/stage6/pre-append.stage2`.**

## 4. Engine and device gates (radiance 140987f, image stilldeadcode/radiance:1.0.8, boot 75e3e39b…, RK_FLAGS)

### R140 device leg (2026-10-05 06:16Z, card 0000:13:00.0 = GPU agent 1, build-hip of d3262dc's tree; evidence/aprime/r140-kernel-gpu.log)
`ctest -L gpu` with libr4d + libref preloaded: 980 checks, all ok (every Stage A device case still green), plus
`gemm_forward_is_the_source_row` and `gemm_forward_matches_its_source_bytewise`: 6 runs at the projector's own
shape (M 1 / 64 / 2048, N 2560, K 10240; bf16 and f32 bias; with and without res), 21,637,120 output bytes,
**forwarded == libr4d's gemm_nt_bias bytewise**. Kernel log clean before and after.

### Session 1 -- R143 bytes, R140 plumb x3, R141 (home data/home-d3262dc: arch 504cd89c…, ridgefill.so e73dc71b…; folder ridgefill.json 73ace68e…; 2026-10-05 06:16-06:27Z; evidence/aprime/session1.log)
Every run: `RADIANCE_RIDGEFILL_PROJECTOR=/projector` (the folder bind-mounted), `RADIANCE_RIDGEFILL_SCORE_BULK=1`, quick9 vs
`data/kld/ref-stage0`, still on the APPENDED container. Startup, every boot: `RidgeFill: projector /projector (found
$RADIANCE_RIDGEFILL_PROJECTOR=/projector) matches qwen4exp: arch ok, metadata 11/11, tokenizer ok, encodings 489/489,
anchors 3/3, 0 warning(s); split 24, correction held, 1257.0 MiB in 28 files`. Comparison = stageA1's rows (A.1
worker, home-1a524d8 = this branch's base plugin on the appended container, same boot):

| gate | run | approx steps | rows vs the append run |
|---|---|---|---|
| R140 / R143 | plumb, FORCE_STREAM=1, boot 1 / 2 / 3 | 67 masked, stage stream | **IDENTICAL** to stageA1/r94-plumb-stream (×3) |
| R143 | speed T2048 | 67 lean | **IDENTICAL** to stageA1/speed-t2048 |
| R143 | quality T2048, STAGE_ROWS 4096 | 67 masked, stage stream | **IDENTICAL** to stageA1/quality-t2048 |

R141 (budget report of the same runs, per card): **already held 306.25 MiB (append) -> 1.50 GiB (folder)**, +1229.75 MiB
against the 1228.1 MiB each rank's declare reports copying (within the report's 0.01-GiB print; run-to-run noise of
this line is ~30 MiB: plumb, which copies nothing, read 335.59 vs 306.23 MiB). The appended arm carried the same
bytes in "static vram" (4.42 -> 3.19 GiB). **Resident experts: 18,347 -> 18,375 slab slots rank 0 (+0.15%), 18,394 ->
18,422 rank 1** (the correction is now a per-rank head slice: 27 MiB less than the replicated append). Plumb holds
0 MiB (reads no fitted tensor). Kernel log clean. Labbook: HAp-R143-bytes, HAp-R141-held confirmed.

### Session 3 -- TTFT: folder vs append (R143), host placement (R148), budgets (R141) (2026-10-05 07:06-08:0xZ; evidence/aprime/session3.log, speed-*.json, ttft-reps.json)
`scripts/serve.sh` + `scripts/speed.sh`, RK_LENGTHS 16384 32768, RK_REPS 7 after one warm-up, quick ppl prompts,
RK_FLAGS (prefix cache off), one fresh server per arm, arms interleaved (exact, append-q, folder-q, host-ring,
host-zero-copy, append-s, folder-s, folder-q-b, append-q-b, host-ring-b, exact-b). Append arms = home-1a524d8 on the
appended container (its container weights); folder arms = home-d3262dc + the folder, same file. Quality reps keep
falling through all 7 reps on every arm (the heat engine re-placing for exact-row routing, A.1's note), so the
number read is the median of the LAST 5 reps of each server, a and b pooled (n = 10; labbook records
aprime-ttft-*). Default settings (no STAGE_ROWS override).

| arm (prompt_ms, last-5 median, pooled) | 16,384 | 32,768 |
|---|---|---|
| exact (stock engine) | 9,979 | 19,920 |
| quality, append | 9,569 | 13,420 |
| quality, **folder** (vram) | 9,657 | 13,536 |
| quality, folder, host + staging ring (default host) | 10,230 | 14,942 |
| quality, folder, host zero-copy (`RADIANCE_RIDGEFILL_PROJ_RING=0`, 3 reps) | **27,096** | **54,104** |
| speed, append | 7,554 | 10,668 |
| speed, **folder** | 7,522 | 10,691 |

- **R143 TTFT green**: folder − append, quality +88 ms [−294, +477] (16K), +116 ms [−1,451, +1,425] (32K); speed
  −32 ms [−466, +410], +23 ms [−1,212, +1,048] -- every CI includes 0, medians within 1%; rep-by-rep the two
  trajectories overlap (e.g. last rep 9,166 vs 9,156 ms). HAp-R143-ttft confirmed.
- **R148**: host zero-copy is 2.7× exact (+17.4 s [+17.1, +17.7] at 16K, +40.6 s at 32K vs vram) -- the GEMM's
  per-M-tile re-reads of the host map over the link, as feared, and worse than registered (HAp-R148-host REFUTED on
  magnitude: predicted 10-40% slower). **The DD-L staging ring is therefore built and is the host default**:
  vs vram +573 ms [+175, +904] (+6%) at 16K, +1,405 ms [+22, +2,905] (+10%) at 32K; **vs exact +252 ms [−27,
  +423] (+2.5%, CI includes 0) at 16K, −4,978 ms [−5,863, −3,934] (−25%) at 32K.** The README says host ON
  chunks at 16K run about as fast as exact (not faster) on this box. PCIe bytes: the ring moves exactly 24 × (2561
  × 10240 × 2 B) = 1.26 GB per approximate pass per rank (one read of each map); a `--profile-ops` byte count was
  NOT run (the zero-copy timing settles the choice; the ring's bytes are fixed by construction).
- **R141 (serving arms)**: already held 306.25 MiB (append) -> 1.50 / 1.53 GiB (folder vram) -> 434 / 430 MiB (host
  ring: correction + row table + 2 × 50 MiB slots = 128 MiB a rank); resident expert slab slots rank 0: append
  15,745 · folder vram 15,742 / 15,765 (−0.02% / +0.13%) · **host ring 16,713 / 16,716 (+6.2%)** · exact 16,883.
  Host placement buys back ~970 slab slots a rank (−0.88 GiB of each card used by RidgeFill) for +6-10% ON TTFT.
- Kernel log clean (session3.log: before 0, after 0).

### Session 2 -- no projector == stock, R142 refusals, R149 discovery (home-d3262dc; 2026-10-05 08:10Z; evidence/aprime/session2.log, g-*.graph)
`--debug-graph` (declare runs, then the engine exits; R6's method: sorted dump, durations normalised, the plugin's
own lines and the shadow line out), quality mode unless said, on the appended container:

| run | RidgeFill line | graph vs the stock engine |
|---|---|---|
| off (plugin home) | (the container's ridgefill.mode=off is ignored -- appended file only) | **identical** (0 lines) |
| quality, no folder (model dir mounted, no projector/) | `RidgeFill: no projector folder (looked at /models/projector, ...); serving stock` | **identical** |
| quality, R142 wrong dims (manifest n_embd 4096) | `REFUSED, it cannot run on this model: metadata 'n_embd' is '2560' in this model and '4096' in the projector's [arch ok, metadata 10/11, tokenizer ok, encodings 489/489, anchors 3/3]; serving stock` | **identical** |
| quality, R142 wrong tokenizer hash | `REFUSED ... the tokenizer differs (vocab sha256 17f26a1a… here, 0000… in the projector's model)`; serving stock | **identical** |
| quality, R142 missing file (proj.L30 removed) | `REFUSED: the manifest lists proj.L30.safetensors, which is missing or unreadable in /projector; serving stock` | **identical** |
| quality, the folder | `matches qwen4exp: ... 0 warning(s)`, rank 0/1 hold 1228.1 MiB | 641 lines: the RidgeFill ops, `ridgefill_gemm_nt_bias  N=2560 K=10240 dtype=bf16 / M <= 2048 -> ridgefill_gemm_nt_bias_r4d (ridgefill, prio 50) / matched M >= 1, K % 8 == 0, dtype in {bf16}` (R140: the forwarded libr4d row is what the engine selects) |

R149 (declare-time discovery, same `--debug-graph` runs):
- **HF-cache layout** `/hf/snapshots/rev1/model.rad -> ../../blobs/0af5e962…` with `projector/` beside the symlink:
  `found beside --model /hf/snapshots/rev1/model.rad (same device and inode as the mapped /hf/blobs/0af5e962…)`.
- **resolved fallback**: `projector/` only beside the blob: `found beside the resolved model file /hf/blobs/0af5e962…`.
- **Docker directory mount** (`/srv/model/{model.rad, projector/}`): `found beside --model /srv/model/model.rad`.
All three: `matches qwen4exp ... 0 warning(s)` and both ranks hold 1228.1 MiB.
(The session stopped before the file-only and env-wins cases on a preflight false positive -- another worker's
`labbook` command line contained "radiance"; those and the rest re-ran as session 2b below.)

### Session 2b -- the rest of session 2 (2026-10-05 08:30-08:42Z; evidence/aprime/session2.log, session2b.out)
- **R149 file-only mount** (`-v model.rad:/m/model.rad` only): `RidgeFill: no projector folder (looked at /m/projector,
  /m/projector); serving stock`; graph **identical** to stock. (The doubled path in the message -- typed dir ==
  resolved dir -- is fixed in the commit after this session; cosmetic.)
- **R149 env wins**: `RADIANCE_RIDGEFILL_PROJECTOR=/other` (the R142 warn variant) with the real folder ALSO beside the model:
  `projector /other (found $RADIANCE_RIDGEFILL_PROJECTOR=/other) matches ... 3 warning(s)`.
- **No folder == stock, ident**: quality, model dir with no projector/: `scripts/ident.sh` **EQUALS R3** (the six
  hashes of evidence/stage0/ident-exact-boot1.txt).
- **R142 wrong-model folder, ident**: refused (n_embd) -> ident **EQUALS R3**.
- **R142 warning case runs**: the warn variant (anchor `blk.23.attn_hc_norm.weight` hash, `blk.24.ssm_inz.weight`
  encoding and the model name edited in ridgefill.json; tensors untouched) -> three WARNING lines, each naming both sides:
  `tensor blk.23.attn_hc_norm.weight hashes 1c74a996… here and ffff… in the model the projector was fitted on`,
  `this model is named 'qwen4exp' and the projector was fitted on 'qwen3.8-next-flash-bf16'`,
  `weight blk.24.ssm_inz.weight is i8*bf16[1x128] here; the projector was fitted on i4*bf16[1x128]/fwht128`
  (each "-- it runs, but was fitted on another variant"); quality KL 67 approximate steps, rows **IDENTICAL** to
  r143-quality-t2048.
- **R148 correctness**: host + ring and host zero-copy, quality T2048 STAGE_ROWS 4096: rows **IDENTICAL** to the vram
  run (both, 67 approximate steps). Ring rank holds 128.0 MiB VRAM + 1200.5 MiB host-mapped; zero-copy 27.9 + 1200.1.
- Kernel log clean (before 0, after 0).

## 5. Status before R144

| row | state | evidence |
|---|---|---|
| R140 | **green**: kernel host (libref) + device (libr4d, projector shape, M 1/64/2048) bytewise; the engine selects `ridgefill_gemm_nt_bias_r4d`; plumb ×3 boots byte-identical to the container-weight plumb; no source loaded ⇒ no row (ridgefill_kernels_alone) ⇒ refused by name (static). Not run: an engine with libr4d hidden (it cannot serve on this card without libr4d at all) | §4 R140, session 1, session 2 |
| R149 | **green**: HF-cache symlink via cmdline + inode, resolved fallback, Docker directory mount, file-only mount ⇒ stock (graph identical), env wins | sessions 2, 2b |
| R142 | **green**: wrong dims / wrong tokenizer / missing file refused by name, graph identical to stock, ident = R3 (dims); the right folder logs the five-check line (11/11, ok, 489/489, 3/3); the warning class runs with named warnings; static covers the other cannot-run cases | sessions 1, 2, 2b; §3 |
| R141 | **green**: already held +1229.75 MiB (folder) / +128 MiB (host ring) per card; resident experts −0.02…+0.15% vs the append (vram), +6.2% (host); sizing declares copy nothing (static) | sessions 1, 3 |
| R143 | **green**: plumb, speed, quality KL rows byte-identical to the append runs; TTFT 16K/32K within noise (all CIs include 0) | sessions 1, 3 |
| R148 | **measured**: zero-copy 2.7× exact ⇒ the DD-L staging ring built (default for host), +6% / +10% vs vram, level with exact at 16K, −25% at 32K; bytes identical (ring and zero-copy) | sessions 2b, 3 |
| R144 | **NOT run** (the orchestrator's): restore with `evidence/stage6/pre-append.stage2`; the published file pre-checked to fingerprint identically (§3c) | -- |

## R144 -- re-run on the PUBLISHED model file (2026-10-05, after the orchestrator's restore and merge 87994fe)

**Provenance.** Model `~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad`, 121,969,901,568 B, sha256
`0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20` (the orchestrator's restore check; size and the
`RAD1` magic re-read at session start; no ridgefill.* weights or keys). Projector at Dylan's user layout
`~/models/rad/projector/` (28 files + ridgefill.json, 0644; ridgefill.json identical to data/projector-qwen38fn's), found by
DISCOVERY (serve.sh mounts the model directory at /models). Plugin home `data/home-0d75987` (arch f44a4bbb…, ridgefill.so
e73dc71b… -- ridgefill.so unchanged since A′), commit 0d75987 = merge + the guard default below. Boot
75e3e39b-cc5b-49de-8087-a4791f372a92 (KL references valid). Image stilldeadcode/radiance:1.0.8, RK_FLAGS
(--gpu-headroom-mib 3072). Scripts: evidence/aprime/scripts/r144_s{1,2}.sh; logs evidence/aprime/r144/.

### The guard default (commit 0d75987, before the runs)
`RADIANCE_RIDGEFILL_STAGE_ROWS` default 64 -> unlimited (every masked pass streams), per Stage A.1's warmed engine results
(notes/impl.md): inside an ON server streaming beats the stock step by 220-240 ms a straddling chunk at 512 / 1,024 /
1,984 exact rows, the 64-row guard cost 331 ms at 9,216 and ran every T 2560 chunk exact. The env keeps the
threshold. Static: 47 cases (truth table under an explicit 64; `the_default_streams_every_masked_pass` for the default);
guard mutants G1-G6 all caught (evidence/aprime/scripts/mutate_guard.out). Then 02b674e: a capture with no folder
takes S from `RADIANCE_RIDGEFILL_CAPTURE_SPLIT`, never the container's `ridgefill.split` (48 cases; debug path only, not
exercised below).

### Session 1 (09:14-09:33Z; evidence/aprime/r144/session1.log)
`--debug-graph` vs the stock engine (sorted, durations normalised, plugin + shadow lines out):
| run | RidgeFill line | differing lines |
|---|---|---|
| off, folder visible beside the model | (none: off never looks; no ridgefill.mode key in the published file) | **0** |
| quality, folder hidden (empty dir over /models/projector) | `no projector folder (looked at /models/projector); serving stock` | **0** |
| quality, only the file mounted (R149) | `no projector folder (looked at /m/projector); serving stock` | **0** |
| quality, R142 wrong dims / wrong tokenizer / missing file | REFUSED by name, as on the appended file; serving stock | **0 / 0 / 0** |
| quality, folder by discovery | `projector /models/projector (found beside --model /models/qwen3.8-next-flash-fp8-iq4r-moe.rad (same device and inode as the mapped …)) matches qwen4exp: arch ok, metadata 11/11, tokenizer ok, encodings 489/489, anchors 3/3, 0 warning(s)` | 641 (the RidgeFill ops) |

ident.sh = **R3's six hashes** for: off with the folder visible (R7), quality with the folder hidden, quality with the
wrong-dims folder (refused).

KL (quick9 vs data/kld/ref-stage0, SCORE_BULK=1, folder by discovery unless said, 67 approximate steps each):
| run | rows vs |
|---|---|
| plumb FORCE_STREAM, boots 1 and 2 | **IDENTICAL** to the appended-file run (evidence/aprime/r140-plumb-stream-1) |
| speed T2048 | **IDENTICAL** to evidence/aprime/r143-speed-t2048 |
| quality T2048 (default guard now) | **IDENTICAL** to evidence/aprime/r143-quality-t2048 (STAGE_ROWS 4096 there) |
| quality T2560 (default guard: 67 masked, stream, split) | **IDENTICAL** to Stage A's R100 run (evidence/stageA/quality-t2560) |
| quality T2048, host + ring | **IDENTICAL** to r143-quality-t2048 |
| quality T2048, R142 warn variant (env) | **IDENTICAL** to r143-quality-t2048; 3 named WARNINGs |
No difference anywhere: the guard default changes only passes with > 64 exact rows, which the T2048 arms already ran
streamed (STAGE_ROWS 4096 on the appended file) and which T2560 now runs approximate again (A.1's default ran it exact).

**R100 headline (quality, T 2560, last 512, paired vs exact, 9 docs): dNLL +0.00212 [−0.01255, +0.01565], ppl ratio
1.0021, KL 0.0368, top-1 0.9156** -- Stage A's headline to the byte.

R141 (same logs): already held 306.25 / 331.59 MiB (plumb: copies nothing) -> 1.50-1.53 GiB (speed / quality) ->
459.62 MiB (host ring); slab slots rank 0: plumb 19,530 · quality 18,371 · host ring 19,296.
Kernel log clean before/after. Labbook: HAp-R144-bytes, HAp-R144-nofolder, HAp-R144-headline confirmed.

### Session 2 -- warmed TTFT (10:08Z end; A.1's session-3 protocol: each server warmed by one RK_REPS=2 pass of all
### three lengths, then RK_REPS=7 read reps 3-7; quiet host = 1-min load < 2.5 and no compiler/test process before
### every label; arms exact / quality / speed twice; evidence/aprime/r144/session2.log, speed-w-*.json)
Published file, folder by discovery, home-0d75987 (guard default unlimited), quality T 2048, speed T 2048.
Settled medians a / b, and pooled vs exact (labbook aprime-r144-ttft-w-*):

| arm | 9,216 | 16,384 | 32,768 |
|---|---|---|---|
| exact | 5,662 / 5,662 | 9,742 / 9,751 | 19,045 / 19,058 |
| quality | 4,773 / 4,738 = **1.19x** | 6,816 / 6,797 = **1.43x** | 10,385 / 10,404 = **1.83x** |
| speed | 4,038 / 4,034 = **1.40x** | 5,245 / 5,244 = **1.86x** | 8,958 / 8,966 = **2.13x** |

quality − exact: −907 ms [−1,020, −793] / −2,940 [−3,119, −2,757] / −8,661 [−8,771, −8,364]; speed − exact: −1,626
[−1,689, −1,569] / −4,501 / −10,090. Against A.1 (appended file, 64-row guard, same protocol): quality 9,216 −383 ms
[−545, −218] faster (the guard's trade, now taken), 16K/32K within 1%; speed within 0.5%; exact within 0.2%. Every
approximate arm faster than exact at every length, CIs excluding 0. Kernel log clean. HAp-R144-ttft confirmed.

### Nothing reads ridgefill.* from the model file
Plugin: no `ridgefill.*` weight is declared (decl_held/decl_projector/decl_correction/decl_score/decl_every_copy deleted in
A′; static `a_mode_without_a_usable_projector_serves_the_in_tree_graph` holds the declare to the in-tree weights
with ridgefill.* entries present); since 02b674e the capture split no longer falls back to `ridgefill.split`; the only `ridgefill.*`
key still looked up is `ridgefill.mode`, read solely to log once that it is ignored (R81) -- the published file has none.
ridgefill.so declares no weights. Tools: `tools/ridgefill_projector.py` reads the model file for the fingerprint only (metadata,
tokenizer, template, 3 anchors); `scripts/mask_rule.py` and `tools/rows_compare.py` read `ridgefill.rowsel.*` from the
sidecar file data/sidecar (dev measurement aids, not the model); the append route lives in tools/dev (retired).
