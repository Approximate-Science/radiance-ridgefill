# notes/release-session.md -- the final release session (2026-10-05/06, boot 75e3e39b)

Plugin built from main **a4d8f16**, version **0.1.0**, against radiance **1.0.13** (radiance-build:1.0.13, runtime
stilldeadcode/radiance:1.0.13). Scripts: branch `release-run` (scripts/release_session.sh, evidence/release-final.sh,
evidence/release-decode.sh). Evidence: evidence/release/ (session.log, per-arm files), labbook seq 486-496.

## Config ("the user's config", orchestrator)
radiance 1.0.13's own `deploy/compose/flashnext.yaml` flags: `--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len
200000 --placement expert_tiered --host-pool-mib 12288 --expert-vs-cache-ratio 0.82 --kv-cache-dtype fp8
--prefix-cache-host-mib 4096 --prefix-cache-dir /kvcache --prefix-cache-disk-mib 131072 --num-speculative-tokens 3
--max-num-batched-tokens 2048`. Deviations, stated:
- `--gpu-headroom-mib 3072` instead of 96: the display runs on card 0000:03:00.0 (Dylan's rule).
- The disk tier was effectively **32 GiB**, not 128: scripts/common.sh's Stage C cache profile appends its own cache
  flags whenever RK_CACHE_DIR is set and the engine takes the last value ("written on to disk (32.00 GiB)" in every
  server log). The busiest server used 28 GB (needle round b), so the cap was not reached. Fixed for later runs (0ac930e).
- common.yaml's docker settings: mirrored except network_mode host (it hides the GPUs under rootless docker), user
  1000:1000, group_add, api-key/restart/healthcheck.
- Each server got a fresh prefix-cache dir under ~/.cache/radiance-kva-release/<server>/, deleted as it stopped.
  Radiance's pure defaults do not load the model on 2 x 32 GB (notes/r1013.md).

## Packages (tools/package.py, dist/ on the Crucial path)
- `radiance-kva-0.1.0.tar.gz` sha256 694a2c5b580c2cd926c72d4323fe2e9270d7d598f22741bd7a254fcf4cb5d481; VERSION.json:
  plugin 0.1.0, commit a4d8f16, radiance 1.0.13, gfx1201; the arch .so's release strings are 0.1.0 and 1.0.13.
- `projector-qwen3.8-flash-next-i8.tar.gz` sha256 45d0632c8c385b08d867d64a5a10af7adfc3042ded83e33dcc7de4f708a7a4d5
  (byte-identical across two package runs). 37 files' SHA256SUMS verified after extraction.
- Frozen home a4d8f16: host tests 5/5.

## Fresh-engine gate (scripts/e2e_fresh.sh, stock image, published model sha256 0af5e962...aa4d20 checked)
Cases 0-6 PASS, 7 SKIPPED (per-request KVA parked):
- no projector = stock and off = stock (ident hashes equal);
- quality and speed: projector line with 0 warnings, approximation logged, faster than stock at 16K (cold: 10,274 /
  7,790 vs 11,024 ms);
- a corrupted projector is refused by name ("proj8.L24.safetensors is corrupt") and serves stock;
- RADIANCE_KVA_PROJ_PLACE=vram is refused by name at startup.
The first attempt aborted at case 5: the gate's glob found only bf16 map files (fixed in f57f016).

## Headline (S4 protocol: settle.py warm-up then speed.sh 7 reps, median; two rounds, int8 folder, T 2048)
| | 16K | 32K |
|---|---|---|
| stock | 12,258 / 12,269 ms | 24,058 / 24,069 ms |
| quality | 8,210 / 8,231 ms = **0.670x / 0.671x** | 11,454 / 11,472 ms = **0.476x / 0.477x** |
| speed | 6,237 / 6,247 ms = **0.509x / 0.509x** | 9,451 / 9,453 ms = **0.393x / 0.393x** |
- Decode at C = 1 with MTP, 16K prompt + 256 tokens in one request (tools/mtp_accept.py, 3 reps):
  - stock 11.1-12.0 ms/token (2.31 tokens/step);
  - quality 7.7-9.7 ms/token (2.30 tokens/step).
  The decode check below shows this gap is not a stable cost.
- Decode mechanism check (tools/decode_check.py, 3 reps; medians, ms/token):
  - right after a 16K prompt: stock 8.87, quality 9.37, although a stock request moved 125-138 GiB of experts and a
    quality request 50-82 GiB;
  - settled (≥ 60 s idle after a long prompt, then a ~200-token prompt + 256 tokens; identical texts in 2 of 3 reps):
    stock 8.18, quality 7.99.
  So **decode is equal**. The first request after boot is slower on both (14.7 / 15.7 ms/token).

## Needle (tools/needle.py build --chat; 3 seeds x 16K/32K x depths 0.10/0.30/0.50/0.70/0.85 + 0.98 x single/4-key)
- 72 items a server: 60 with the needle inside the approximated bulk, 12 tail controls. max_tokens 32.
- Run on round b's servers after a stock smoke gate (6/6).
- **stock 72/72, quality 72/72, speed 72/72**; McNemar p = 1 for both comparisons (no disagreement).
- The first attempt sent raw prompts (no chat template): replies opened a reasoning block or continued the text.
  Invalid, kept as evidence/release/needle-raw-INVALID.
- The needle prompts share their filler prefix, so later items hit the prefix cache up to the needle. The needle and
  everything after it are computed in each request.

## R64 redefined (tools/turn2.py; 5 conversations, ~12K-token turn 1 + 64-token answer + follow-up)
- Servers: MTP off and --profile-ops.
- Turn 2's first answer row vs exact (mode off), KL and first-token agreement:

| arm | cached tokens | KL per conversation | mean KL | first-token agreement |
|---|---|---|---|---|
| quality, prefix cache | 10,240 | 0.230, 0.098, 0.048, 0.077, 0.045 | **0.0995** | 2/5 |
| quality, --no-prefix-cache | 0 | 0.100, 0.171, 0.035, 0.067, 0.052 | **0.0847** | 4/5 |

- Cached = 1.17x no-cache: within the registered bar (≤ 1.25x). With n = 5 the difference is one conversation's
  (conv 0), and the first-token agreement leans against the cache. Not enough to call either way.
- The no-cache arm's first run had the cache on (common.sh's cache profile drops --no-prefix-cache) and was re-run.
- Greedy answers usually leave exact within the first tokens, so `logit_compare`'s "until the first input difference"
  means compare different row counts. The first-row KL is the like-for-like number.

## Process notes
- The aborted 0.2.0 run (evidence/release-aborted-0.2.0) was stopped for the version change (0.1.0 is compiled into the
  .so).
- The first 0.1.0 run filled /var/mnt/qwen-storage: the prefix-cache disk tier was written under evidence/, so the
  needle writes failed (ENOSPC). Caches now go on the root disk.
