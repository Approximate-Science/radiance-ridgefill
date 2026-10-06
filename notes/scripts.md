# Notes — scripts lane (measurement scripts, tools/speed.py, tools/paired.py)

Written by the scripts-lane worker, 2026-10-04. All engine facts below were read from the
radiance tree at tag v1.0.8, commit 140987f (read-only); nothing here was run against a GPU
or a real engine (no docker, no radiance in this worker's sandbox). Everything was verified
by `sh -n`/`bash -n`, pytest on the pure Python parts, and dry-runs with a fake
`docker`/`curl`/`journalctl` and a fake HTTP server shaped like the real endpoints.

## Flags verified in radiance core/config.cpp (flag table, lines 31-190)

Every flag in `RK_FLAGS` (scripts/common.sh) exists in the flag table:

| flag | line | value |
|---|---|---|
| `--tp` | 37 | 2 |
| `--tp-wire` | 38 | exact (the flag's documented default mode) |
| `--max-num-batched-tokens` | 40 | 2048 |
| `--max-model-len` | 42 | 49152 |
| `--gpu-headroom-mib` | 44 | 3072 (measurement shape; fnserve.sh's 96 is production) |
| `--expert-vs-cache-ratio` | 51 | 0.82 (mirrored from fnserve.sh) |
| `--kv-cache-dtype` | 56 | fp8 |
| `--host-pool-mib` | 57 | 12288 (mirrored from fnserve.sh) |
| `--placement` | 71 | expert_tiered (mirrored from fnserve.sh) |
| `--no-prefix-cache` | 85 | - |
| `--num-speculative-tokens` | 89 | 0 |
| `--kld-record` / `--kld-ref` / `--kld-corpus` / `--kld-out` | 174 / 177 / 180 / 181 | grade.sh |

Why the three fnserve.sh flags are mirrored: the container's routed experts do not fit both
cards' VRAM, so Flash-Next needs `--placement expert_tiered` and a pinned host pool to start
at all; fnserve.sh's ratio 0.82 is the production split (at `--max-model-len 49152` it only
widens the expert side). No flag name had to be reported as missing.

Also mirrored into the docker prefix: `--ulimit memlock=-1` (deploy/compose/common.yaml
comment: the pinned host pool needs it). `GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100`
are already image env (docs/DOCKER.md "Run it").

## Endpoint shapes (from GUIDE.md and core/server/)

- `POST /tokenize` takes `{"prompt": str, "add_special_tokens": bool}` (prompt form defaults
  add_special_tokens=true, which would silently prepend BOS; speed.py passes false).
  Response `{"count": N, "max_model_len": M, "tokens": [ids]}` — core/server/admin.cpp:119-196.
- `POST /v1/completions` accepts `"prompt": [int, ...]` (list-of-ids) directly —
  core/server/oai.cpp:1549-1568 (`push_ids`; "token arrays must contain integers only"). This
  is NOT documented in GUIDE.md §5.7 (the table only shows the string forms), so speed.py
  sends ids and verifies `usage.prompt_tokens == target` on every response (a server that
  added BOS to an id prompt would fail loudly there).
- `POST /v1/completions` response: `timings.prompt_ms` / `cache_n` / `prompt_n`
  (GUIDE.md §5.9, lines ~1236-1269), `usage.prompt_tokens`,
  `usage.prompt_tokens_details.cached_tokens` (scripts/fnpf.sh reads the same field).
- `GET /health` (200), `GET /server_info` (GUIDE.md §6.2, lines ~1300), `GET /v1/models`.
- KL-mode report shape: `all` and `by_source` summaries with `kld.{mean,median,p90,p95,p99,
  p99_9,p99_99,max}`, `top1_agreement`, `ppl.{candidate,reference,ratio}`; `.rows` = [positions, 4]
  f32 in reference row order = (KL, candidate NLL, reference NLL, top-1 agreement) —
  core/kld.cpp:562-579 (report), 586-593 (rows). Corpus JSONL: one {"prompt", "score_from",
  "source"} per line (core/kld.cpp:209-242).

## Decisions taken (with the reason)

- `--max-num-seqs` is NOT in `RK_FLAGS` (it differs per use): serve.sh pins 8
  (fnserve.sh's default), grade.sh passes 1 on EVERY KL command (the scheduler-top-up
  control, HANDOVER §5 Stage 0 step 6).
- grade.sh's `record` runs the stock engine (exact home) by definition, same RK_FLAGS as the
  candidates: the noise floor is exact-vs-exact under the measurement configuration.
  TOOLS.md's "reference with a bf16 KV cache" advice is for quantisation measurement and is
  deliberately NOT followed here.
- speed.py's nonces: tokenized server-side with `add_special_tokens: false`, so the prompt
  length is exact by construction and no BOS is silently prepended (consistent with the KL
  corpus tokenization, core/kld.cpp:222: encode(..., add_special=false, parse_special=true)).
- preflight (a) excludes the check's own PROCESS TREE (the script, its launcher, every
  ancestor), the repo's own tooling (cmdline contains the repo path), the PIDs inside
  radiance-ridgefill-* containers (`docker top`), and any cmdline that merely QUOTES the check
  pattern (no real engine carries a `|` in its name). Verified live: a fake `vllm-serve`
  squatter is caught and named; the harness/agent wrappers are not.
- preflight (b) counts a card as discrete when `mem_info_vram_total > 8 GiB` — verified live
  on this machine: skips the 2 GiB card, reads both R9700s (0000:03:00.0, 0000:13:00.0).
- paired.py's `.rows` fallback REQUIRES the corpus lines to carry `"tokens"` (the token count
  the doc was cut to) because n = token count is not recoverable from prompt text without a
  tokenizer; this is a contract on tools/kld_corpus.py's output (its lane). Missing → error,
  named, with the line number.
- The exact-tail note in the speed evidence: tail = 2048 + (L % 2048) for L not a multiple of
  2048 (9216 → 3072), exactly 2048 otherwise; guarded for L < 2 chunks.

## What was tested, and how (verbs per WORKER-RULES)

- Implemented: scripts/{common,serve,stop,preflight,ident,speed,grade}.sh, tools/speed.py,
  tools/paired.py, tests/test_paired.py, tests/test_speed.py.
- Checked: `sh -n` and `bash -n` on every script (pass). shellcheck NOT AVAILABLE in this
  sandbox — still to run on a machine that has it.
- Tested (pytest, 19 cases): paired.py report path (ln of ppl.candidate), .rows+corpus
  fallback, rows-size mismatch, corpus without "tokens", source mismatch, bootstrap
  reproducibility (seed 0) and CI bracketing, CLI end-to-end and refusal paths; speed.py doc
  loader, prompt builder (exact lengths, nonce leading, cut), tail notes at 9216/16384/32768.
- Dry-run verified against the REAL host /sys and journalctl (no docker daemon in the
  sandbox, so docker/curl were fakes on PATH):
  - preflight.sh: provenance line with both PCI ids, temps, sclk; catches a fake `vllm`
    squatter; catches fake amdgpu MES/SMU kernel lines; passes on the live machine.
  - serve.sh: refuses on an existing radiance-ridgefill-* container; preflight before start;
    assembles the full docker line (prefix + RK_FLAGS + port + extra args, RADIANCE_RIDGEFILL*
    pass-through, plugin home only for non-exact); waits for /health; records container id,
    image digest, model size and the full command one-arg-per-line in the .cmd evidence
    file. Paths with spaces survive the POSIX arg materialisation.
  - stop.sh: stops/removes, waits for VRAM below the ceiling (verified against the live
    cards: 143 MiB / 70 MiB).
  - grade.sh record: ref dir refusal, corpus mount, KL flags, log to <ref_dir>.log, dies
    named when kld.json did not appear; candidate: plugin home per mode, ref ro mount, out
    dir rw mount, approximate-step count printed, RK_EXPECT_APPROX mismatch exits non-zero
    with both numbers; overwrite refusals for out/.rows/.log.
  - speed.py against a fake server with the real request/response shapes: exact prompt
    lengths verified on every response, cached-token check, warm-up + reps, median/min/max,
    per-rep preflight lines and /server_info provenance in the evidence JSON.
- NOT tested (needs the real engine): the docker GPU run itself, the real /health of a
  served radiance, KL-mode record/ref against a real .rad, and the plugin's
  `ridgefill: approximate step` log lines (faked in the dry-run).

## Open items for other lanes

- tools/kld_corpus.py must emit `"tokens": N` per corpus line (paired.py's rows fallback and
  the count checks rely on it).
- The `.cmd` evidence files and the speed JSON carry the provenance the §7 protocol asks
  for (image digest, model size, flags, preflight lines); the plugin commit is not in them
  (it is not known to the scripts) — it goes in NOTES.md by hand per measurement.