# notes/stock-path-cost.md -- what prompts RidgeFill does not approximate cost on a RidgeFill server (2026-10-06)

> **Fixed on branch fix/stockpath-prefetch (not merged yet):** the extra prefill stage was the hazard op's handle.
> With it removed, 1,600 / 2,000 tokens cost +0.8-0.9% like 512 / 1,024 -- notes/stockpath-fix.md. The
> measurements below are kept as made.

Prompts below ~2K tokens are not approximated and run the stock path. The rule: such requests lose no speed against
stock, with a ~1% residual accepted. Three figures disagreed:
- BetterBench: 0.97x at 1.6K real tokens on both modes, "outside noise", one sample set per server, servers in
  different states.
- The release docs: "+0.9% (2K) / +1.2% (8K) settled".
- notes/mixed-traffic.md: ~150-token chats +0.5-0.6% TTFT on fresh servers.

This run measures 512 / 1,024 / 1,600 / 2,000 tokens on fresh servers in ABCA order, so server-to-server drift is
measured too.

## Answer
- **The cost is real, and it is ~2.5% at 1,600 and 2,000 tokens, which is over the 1% allowance.** BetterBench's 0.97x
  was right. The release docs' "+0.9% (2K)" does not describe the release config (see "Why the figures disagreed").
- **512 and 1,024 tokens cost ~0.9%.** That is at the allowance, not clearly inside it.
- **Drift is negligible.** quality2 reproduces quality to within 0.2% at every length, so these are not
  server-state effects.
- **Where the time goes, from rank 0's /stats (measured):**
  1. **On every length: VRAM displacement.**
     - The RidgeFill servers have 47 fewer expert slab slots a rank (15,664 vs 15,711) and ~60 fewer resident
       experts on rank 0.
     - Each request streams 1.0-1.2% more expert bytes, and the resident share of routed experts is 0.0015-0.0025
       lower.
     - This matches the ~0.9% at 512 / 1,024 tokens, where it is the only difference.
  2. **On steps of more than 1,024 tokens only: one extra layer staged.**
     - Above 1,024 tokens, the engine's prefill stager copies each routed layer's pooled experts onto the card
       ahead of use.
     - Every RidgeFill request at 1,600 / 2,000 tokens made **50 layer stages**, against stock's **49**: 36 of 36
       RidgeFill requests, 12 of 12 stock, at each length.
     - Per request that is **+208 staged units** (7,243 vs 7,035) and **+505 MiB** across the link (+3.0%).
     - Of those 208 units, ~50-60 follow from the displacement above. The other ~150 are about one layer's pooled
       experts (7,035 / 49 = 144 on average).
     - The radiance 1.0.13 stager starts each of its 49 layers at most once per armed pass (core/place/stager.cpp
       `started_`), so a 50th stage means a second armed pass or a re-stage. **Which one is not identifiable from
       these counters.** Pinning it needs `RADIANCE_LOG_STEPS` or a debug build: not run (one session was the
       budget).
  3. **Plugin host time: not measured.** The plugin logs only approximate steps; a stock-path step writes no line.
     The engine's per-request counters match between arms: 97 expert dispatches everywhere, and 2,304 dispatched ops
     at 1,600 / 2,000 tokens (6,230 vs 6,228 at 512 / 1,024). So the stock path issues the same work.

## What ran
- One gpuq session, 09:36:54-09:47:34Z (**10.7 GPU minutes**), boot 75e3e39b, repo 712adda.
- Evidence is in `~/AI-Work/radiance-kva-plugin-20261004/evidence/stockpath-20261006/`:
  - scripts: stockpath.sh, analyze.py, and ramp.py (the mixed run's, byte-identical, sha256 8524aa23…);
  - `stockpath-<server>.jsonl` (raw, with rank 0's /stats before and after every request) and **`stockpath.csv`**
    (one row per request, every server);
  - tables.md (all tables, including per-round values), session.log, and the serve logs.
- Labbook: HSTOCK-path was registered first (seq 533). The per-server, per-length series are recorded as
  `stockpath-<server>-<len>`.
- Config was exactly as in the mixed run:
  - radiance 1.0.13 flashnext profile, `--gpu-headroom-mib 3072`, MTP 3, prefix cache on, a fresh cache dir per server
    under `~/.cache/radiance-kva-release/stockpath-<server>/`, removed afterwards;
  - packages `dist-ridgefill-0.1.0-r2` (754f1ebb…, 88a46a39…), extracted fresh, 39 SHA256SUMS files verified;
  - the projector was found by discovery.
- Servers in order:
  - a throwaway stock server (one 512-token request, 3 min);
  - then **quality, speed, stock, quality2**, each fresh. Each ran the same sequence: 2 warm-ups (1,024, 2,000), then
    12 rounds of 512 / 1,024 / 1,600 / 2,000 tokens, with the order rotated by one each round (every length sits in
    every position 3 times).
- Requests used ramp.py's natural-prose token-id builder with a unique leading nonce (the same prompt for the same
  request number on every server), `/v1/completions`, max_tokens 1.
- Recorded per request: timings.prompt_ms, prompt_tokens (exact on all 200), cached tokens (0 on all 200),
  approximate-step lines (**0 on all 200**, so no length is dropped), and rank 0's /stats.
- Kernel log: 0 amdgpu MES/SMU/timeout/reset lines before and after every server.
- The n-gram table went to disk (auto) on every server, with 76.1-78.2 GiB available at placement.
- Statistics: each arm is compared with stock **paired by request** (same round and length), as the median of the 12
  per-round ratios with a bootstrap 95% CI (10,000 resamples). Drift is quality2 / quality the same way.
- Verdict rule:
  - **within 1%**: the CI's upper bound is at most +1%;
  - **over 1%**: the CI's lower bound is above +1% and the median is beyond the drift;
  - **not resolvable**: anything else.

## Prefill time vs stock (paired, median [95% CI], 12 pairs per length)
| tokens | quality / stock | speed / stock | quality2 / stock | drift quality2 / quality | verdict |
|---|---|---|---|---|---|
| 512 | **+0.92%** [+0.76, +1.21] | **+0.97%** [+0.86, +1.23] | +0.83% [+0.65, +1.04] | -0.19% [-0.37, -0.00] | not resolvable (at the 1% line) |
| 1,024 | **+0.89%** [+0.78, +1.16] | **+0.92%** [+0.86, +1.14] | +0.95% [+0.86, +0.99] | -0.01% [-0.14, +0.10] | not resolvable (at the 1% line; quality2 inside) |
| 1,600 | **+2.59%** [+2.56, +2.65] | **+2.57%** [+2.55, +2.63] | +2.57% [+2.51, +2.61] | -0.04% [-0.08, +0.01] | **over 1%** |
| 2,000 | **+2.46%** [+2.42, +2.50] | **+2.46%** [+2.43, +2.50] | +2.46% [+2.43, +2.49] | -0.03% [-0.04, +0.03] | **over 1%** |

- **In absolute terms:** +7 ms (512), +10 ms (1,024), +38 ms (1,600) and +34 ms (2,000) per request.
- Medians of prompt_ms (ms; quality / speed / stock / quality2):
  - 512: 782 / 783 / 775 / 781;
  - 1,024: 1,024 / 1,024 / 1,015 / 1,024;
  - 1,600: 1,473 / 1,475 / 1,437 / 1,473;
  - 2,000: 1,405 / 1,405 / 1,371 / 1,405.
- **1,600 is bimodal on every server** (~1,380 or ~1,560 ms, depending on the round's prompt), and some 1,600-token
  prompts take longer than 2,000-token ones. Pairing by request removes this.
- **Two transient outliers on quality, not on quality2:**
  - round 5 at 1,600: 1,725 ms against 1,563 on the others;
  - round 10 at 2,000: 2,673 ms against ~1,401.
  - The medians are unaffected.

## Expert traffic per request, rank 0 (median over 12 rounds; full table in tables.md)
| tokens | server | layer stages | staged units | streamed GiB | link h2d GiB | resident share | resident after |
|---|---|---|---|---|---|---|---|
| 512 | stock | 0 | 0 | 6.83 | 0.83 | 0.8067 | 18,022 |
| 512 | quality | 0 | 0 | 6.91 | 0.99 | 0.8042 | 17,962 |
| 1,024 | stock | 0 | 0 | 7.18 | 0.08 | 0.8250 | 18,744 |
| 1,024 | quality | 0 | 0 | 7.26 | 0.08 | 0.8230 | 18,699 |
| 1,600 | stock | **49** | 7,034 | 9.64 | 16.62 | 0.7845 | 18,027 |
| 1,600 | quality | **50** | 7,240 | 9.76 | 17.11 | 0.7818 | 17,965 |
| 2,000 | stock | **49** | 7,035 | 9.88 | 16.62 | 0.7831 | 18,022 |
| 2,000 | quality | **50** | 7,243 | 9.99 | 17.12 | 0.7807 | 17,962 |

Speed and quality2 match quality to within 3 units.

- **Streamed bytes, paired:** RidgeFill / stock = 1.009-1.012 at every length, all CIs above 1.
- **Startup budgets** (serve logs, quality vs stock):
  - "already held" +28.5 MiB (the 25.4 MiB staging-ring slot and code);
  - activation arena 940 vs 897 MiB (+43 MiB);
  - experts 18.53 vs 18.59 GiB;
  - slab slots 15,664 / 15,712 vs 15,711 / 15,759.
  - The RidgeFill servers also backed 7 of 8 prefix-cache checkpoint slots, against stock's 6 (58 MiB each).
- **Time tracks the bytes moved:**
  - At 512 / 1,024 tokens, +1.0-1.2% streamed bytes go with +0.9% time.
  - At 1,600 / 2,000 tokens, the link carries +2.3% (h2d + streamed: 27.11 vs 26.50 GiB) and time is +2.5%.
  - Prefill at these sizes is bound by expert traffic over PCIe, so every extra byte shows up in TTFT.

## Why the figures disagreed
- **BetterBench's 0.97x at 1.6K:** confirmed. This run measures +2.6% prefill time at 1,600 tokens, i.e. ~0.975x.
- **"+0.9% (2K) / +1.2% (8K) settled"** is notes/stagee.md §14's HE-held-1slot S4 measurement. Its setup differs
  from the release config in several ways:
  - it ran before the 1.0.13 rebase (2026-10-05 ~20:54Z), so on radiance 1.0.8;
  - it used the then-current RK_FLAGS (MTP 0), not the flashnext profile;
  - it ran on settled servers;
  - its arm was `RADIANCE_RIDGEFILL_FORCE_SPLIT=2048` (projector held, every pass stock).
  - It was never re-measured on the release config.
  - This run shows that, at 2,000 tokens on 1.0.13 with the profile, the extra stage adds ~1.6 points on top of the
    displacement's ~0.9.
  - Whether 1.0.8's stager behaved the same was not checked.
- **The mixed run's +0.5-0.6% TTFT on ~150-token chats** fits the displacement-only regime here (+0.9% at 512),
  measured on a different quantity (client TTFT).

## What this means
- **The no-degradation rule fails for single-step prompts of 1,025-2,048 tokens** (~2.5%, ~35 ms). It is at the 1%
  line below 1,025 tokens (~0.9%).
- **The docs should say:** "≈ +0.9% up to 1K tokens, ≈ +2.5% at 1.6-2K tokens (measured 2026-10-06, fresh servers)".
  Drop the "+0.9% (2K) / +1.2% (8K)" figure. PLUGIN-README.md:114 and :128 carry the old wording; not edited here.
- **8K is not measured here.** Prompts over 2K are approximated on most chunks. Whether their exact chunks pay the
  extra stage is open.
- **Two levers, both unbuilt:**
  - The extra stage (~1.6 of the 2.5 points) is the larger part and may be removable. The first step is to find which
    pass makes it (`RADIANCE_LOG_STEPS=1`, stock vs quality, a few 2,000-token prompts, ~6 GPU min).
  - The displacement (~0.9 points) is the plugin's VRAM, already cut to one int8 ring slot. notes/stagee.md §14 names
    a half-layer slot as the only lever left (~-0.2%).
- **Not tested:** settled servers (after long-prompt traffic the residency shifts; notes/mixed-traffic.md), prompts
  of 2-8K tokens, and concurrency.

## Prediction (HSTOCK-path, seq 533) against the result
| prediction | result |
|---|---|
| every length +0.0..+1.5%, CI upper ≤ +2% | **missed** at 1,600 / 2,000 (+2.5-2.6%, CIs [+2.42, +2.65]); met at 512 / 1,024 (+0.9%, CI ≤ +1.23) |
| drift quality2 / quality within ±1% | met: -0.19 / -0.01 / -0.04 / -0.03% |
| 0 approximate steps at all four lengths | met: 0 on 200 of 200 |
| mechanism: derive() plus ~0.3% displacement only | **partly refuted**: displacement is measured (+1.0-1.2% bytes), but an extra layer stage per >1,024-token step was not foreseen |
| kill: quality and speed > +2% with CI above +1% beyond drift | **reached** at 1,600 and 2,000 |
