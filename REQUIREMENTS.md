# REQUIREMENTS — radiance-kva (copy into the plugin repo root; keep every row's output pasted under it)

Rule: a row is green only when its check has been RUN and its output is pasted below the table in `NOTES.md`
with the commit hash of the plugin, the radiance commit, the container sha256 and the date. "Implemented" is not
green. A check that could not run is marked SKIPPED with the reason, never passed.

Paths below assume the repo at `~/projects/inference/radiance-kva`, radiance at `~/projects/inference/radiance`
(commit `140987f`), and `$RH` = the RADIANCE_HOME string with the plugin's home first. `<rad>` = the container.

| id | requirement | stage | the check that fails if it is not met |
|---|---|---|---|
| R1 | Stock engine serves the container; baseline recorded | 0 | `scripts/speed.sh exact` writes `evidence/stage0/speed.json` with median TTFT at 9216/16384/32768 tokens (tail at 9216 is 3,072, HANDOVER Stage 0.5); `dmesg | grep -iE 'amdgpu.*(MES|SMU)'` is empty for the run window |
| R2 | KL reference recorded, same boot | 0 | `radiance --model <rad> --max-num-seqs 1 --kld-record evidence/stage0/ref --kld-corpus corpus/quick9.jsonl` exits 0 and `evidence/stage0/ref` is non-empty; every doc in `quick9.jsonl` is a multiple of 2048 tokens (`tools/kld_corpus.py` asserts it) |
| R3 | Stock output is reproducible | 0 | `scripts/ident.sh` run 3× gives identical hashes (if not, record the noise floor from `--kld-ref` of exact vs exact and use it in R14/R18) |
| R4 | Both plugins load and are listed | 1 | `rad-info --plugins --home $RH` lists `qwen4exp_kva` (architecture, claims `qwen4exp`) and `kva` (kernels, declares `kva_state_correct`); no "refused" line |
| R5 | Shadowing works, in-tree plugin not loaded twice | 1 | the same output shows exactly one plugin claiming `qwen4exp`; engine starts with `RADIANCE_HOME=$RH` |
| R6 | `off` mode graph is identical to stock | 1 | `diff <(radiance … --debug-graph) <(RADIANCE_HOME=<stock> radiance … --debug-graph)` is empty except plugin name/version lines |
| R7 | `off` mode output is identical to stock | 1–7 | `scripts/ident.sh` hashes under the KVA plugin (`RADIANCE_KVA=off`) equal the stock hashes from R3 |
| R8 | Nothing unimplemented returns success | 1 | with the op's launch stubbed, issuing it returns `RAD_E_UNSUPPORTED` and the engine logs it by name (unit test `tests/kernel_test` case `stub_refuses`) |
| R9 | Builds out of tree against an installed radiance | 1,7 | `cmake -S . -B /tmp/oot -DCMAKE_PREFIX_PATH=<install> -DRADIANCE_SRC=<src> && cmake --build /tmp/oot` in a clean dir, inside the Docker `build` stage |
| R10 | Refuses a mismatched radiance source | 1 | configure with `RADIANCE_SRC` at a different tag fails with a message naming both versions |
| R11 | Builds and tests with no GPU | 1,7 | `cmake … -DRAD_WITH_HIP=OFF && ctest -LE gpu` green on the host-only build (static arch test + kernel host row test) |
| R12 | Sidecar weights appear in the container and are optional | 2 | `rad-info -v <rad>` lists `kva.proj.24.weight … kva.st.46`; a container WITHOUT them still starts and `ident.sh` equals R3 |
| R13 | Projector weight integrity | 2 | `tools/kva_sidecar.py --verify <rad>` recomputes sha256 of the source `.safetensors`/`.pt` and compares with the container metadata |
| R14 | `plumb` mode is byte-identical to stock | 3 | `RADIANCE_KVA=plumb scripts/ident.sh` hashes equal R3 (or KL = 0 vs R2 if R3 was noisy) |
| R15 | Approximation decision uses keyed fields only | 3 | `grep -nE 'static .*(count|last|prev)|thread_local' arch/*.cpp` shows no mutable per-request host state; a 2,000-request soak with `RADIANCE_LOG_STEPS=1` shows no `RAD_E_STATE` tape-audit failure |
| R16 | Approximate chunks really skip the late layers | 3 | `--profile-ops` on a 16K prompt: no `moe_gemm_q`, `attn_paged`, `hc_write` issues for layers ≥ S on chunks with `n_ahead >= T`; per-layer op list pasted |
| R17 | Boundary equivalence | 3 | debug dump of `b_h` at layer S on the ids of one tcc capture file: cosine ≥ 0.98 per row vs `boundary_24` (record the actual value; < 0.98 → stop and report) |
| R18 | Fill quality measured with controls | 3 | `--kld-ref evidence/stage0/ref` with `score_from = N − T`: mean KL, NLL ratio and top-1 vs exact on the quick set, per-doc paired, bootstrap CI; positive control KL(fill) ≫ noise floor (R3); `--max-num-seqs 1`; the plugin's approximate-step count in the log equals Σ_docs (N_d − T)/2048 (otherwise some bulk chunks ran exact and the row is red); prediction registered in `labbook` BEFORE the run |
| R19 | Speed measured | 3 | `scripts/speed.sh speed` → median of 5 after warm-up at 9216/16384/32768, same boot as R1, `--no-prefix-cache`, MTP off; table with speedup vs R1 |
| R20 | New op: host row ≡ device row | 4 | `tests/kernel_test` runs both rows on random operands with padded slot strides and a nonzero flag; max abs diff 0 (f32 add is exact for equal inputs) |
| R21 | New op: no allocation / getenv / sync on the launch path | 4 | `grep -rnE 'hipMalloc|getenv|hipDeviceSynchronize|hipStreamSynchronize' kernels/` is empty |
| R22 | Correction neutral at alpha = 0 | 4 | `RADIANCE_KVA_ALPHA=0` output byte-identical to Stage 3 (`ident.sh`) |
| R23 | Correction helps (positive control) | 4 | paired per-doc NLL, correction on vs off, CI excludes 0 in the improving direction; prediction registered first |
| R24 | Head-order check (negative control) | 4 | a sidecar built with the two rank halves swapped does NOT beat correction off (CI includes 0 or worsens) |
| R33 | `kva_rowsel` host ≡ device, and ≡ the Python rule | 5 | `tests/kernel_test` case `rowsel`: on the 9 quick docs' first chunks, device `rows_idx` equals the host row and equals `fnlev.rules` applied to the same 2048-token window (share 0.25, kept classes cap/mixed/piece, ties by position); `mask` consistent; `−1` padding; random mode gives exactly k rows, deterministic for a seed |
| R34 | `kva_rho_update` host ≡ device ≡ reference | 5 | `tests/kernel_test` case `rho`: device vs host max abs diff ≤ 1e-5 on dumped `a` columns, A_log, dt_bias, mask; vs a NumPy transcription of `st_hook.py:25-34, 229-276`; rho = 1 when mask is all ones (no exact rows); rho ∈ [0,1] |
| R35 | Quality mode with all rows selected reproduces stock | 5 | `RADIANCE_KVA_ROWSEL=all` with `kva.rowsel.share=1` and an all-kept score table: KL vs the Stage 0 reference ≤ the exact-vs-exact noise floor (R3), top-1 ≥ 99% on the tail (GEMM-shape noise is documented, byte-identity is not required) |
| R36 | Quality mode really runs the exact-row path only on the compacted rows | 5 | `--profile-ops` on a 16K prompt: late `moe_gemm_q` issued at M = cap (not n_tok); `hc_read`/`hc_write` for layers ≥ S at M = cap; attention and GDN blocks at M = n_tok; `rad_route_counts` per layer pasted |
| R37 | Class rows beat count-matched random rows (positive control for the selector) | 5 | paired per-doc NLL, `class` vs `random` (same share, same seed policy), bootstrap CI excludes 0 in the improving direction; prediction registered first |
| R38 | Quality mode improves on speed mode | 5 | paired per-doc NLL, quality vs speed (both with correction), CI excludes 0; quality-mode NLL ratio vs exact and top-1 recorded |
| R39 | In-chunk selection vs whole-prompt selection | 5 | dump of `rows_idx` per chunk for the 9 docs (debug env) vs `fnlev.xsuite` row lists at P = N − T: Jaccard per doc recorded; if mean Jaccard < 0.5, report before claiming the paper's row |
| R40 | Quality-mode speed measured | 5 | `scripts/speed.sh quality` at 9216/16384/32768, same boot as R1; table with speedup vs exact and vs speed mode; expert sweep per late layer noted (Q14) |
| R41 | Row-aware correction neutral when nothing is selected | 5 | a score table keeping no ids (`--none`) → `kva_rho_update` yields rho = 1 and the output is byte-identical to speed mode |
| R42 | Projector refit from radiance captures | 6 | capture files read by the existing `fit.py` sums unchanged; `report-*.json` written with lambda, rows, held-out cosine; sidecar `kva-radiance-s24` builds and passes R13 |
| R43 | Correction refit from radiance state captures | 6 | `C_L` recomputed from ≥ 13 prompts; per-layer count ≥ 50 chunk ends; passes R20/R22 with the new weights |
| R44 | Shipped vs refit decided on paired NLL, same boot | 6 | table: shipped / refit × {speed, quality} NLL ratio vs exact with CI, TTFT; the kept set named in `NOTES.md` with the number that chose it; a refit kept on cosine alone is a failed row |
| R25 | Mixed steps fall back to exact and stay correct | 7 | `scripts/tiercross.sh` pattern: a long prefill beside a decoding session; the decoding session's text equals its solo run byte for byte |
| R26 | Both TP ranks do their share | 7 | `--profile-ops` per rank on an approximate chunk shows projector GEMMs and fills on both ranks; no rank idle |
| R27 | Portability greps clean | 7 | `grep -rnE '/var/home|/home/dylan|13:00\.0|0000:|blackbox|R9700|gfx1201' --include='*.cpp' --include='*.h' --include='*.hip' --include='*.py' --include='CMakeLists.txt' .` is not literally empty: every hit is a comment, help text, test fixture or lab note; none is a path/device default in code (gfx target only via `RAD_GPU_TARGETS`). Current hit classes (8 hits, all in `tests/` or `tools/`): the `--gpu-targets` help string in `tools/package.py` ("e.g. gfx1201", an example, not a default); `gfx1201` inside `tests/test_package.py`'s fake-ELF fixture bytes, parsed targets and asserts; and the machine-local default tokenizer path and its docstring in `tests/test_template_identity.py` (an opt-in fixture, overridable via `KVA_TEST_TOKENIZER`) |
| R28 | No machine constants | 7 | `grep -nE '\b(48|24|2560|10240|128|2048)\b' arch/*.cpp kernels/*` — every hit is read from meta/operands or justified in a comment naming the hardware/model fact |
| R29 | Docker: build in the `build` stage, run in the runtime image with the plugin mounted | 7 | `docker/build.sh -r 140987f --target build`, plugin build inside it, then `docker run … -v plugins:/plugins -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance stilldeadcode/radiance:1.0.8 …` serves and R7 holds |
| R30 | One-command install documented and true | 7 | a fresh clone + `README.md` steps only, in a scratch dir, reaches R4 |
| R31 | Declines what it cannot serve | 7 | with `kva.tail` > `max_tok`, declare refuses with a message naming both numbers; with `kva.*` weights present but `kva.so` missing, `speed` mode refuses at startup by name (not a silent run without correction) |
| R32 | NOTES.md is the lab notebook | all | every measured table, every rejected arm with its number, every stage's pasted outputs; a reader can find the latest numbers without the chat |
