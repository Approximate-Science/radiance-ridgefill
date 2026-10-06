# notes/ramp.md -- does RidgeFill's long-prompt prefill settle, where, and why (2026-10-06)

The release headline (quality 2.10x, speed 2.55x at 32K) was the median of a series that was still falling. BetterBench
and the content probe measured ~1.2-1.45x / ~1.3-1.7x on fresh servers. This run measures the whole curve.

## What ran
- One gpuq session, 07:48:41-08:19:52Z (**31.2 GPU minutes**), boot 75e3e39b, repo c1bdb12. Evidence:
  `~/AI-Work/radiance-kva-plugin-20261004/evidence/ramp-20261006/` (session.log, ramp.sh, ramp.py, analyze.py,
  `ramp-<arm>.jsonl` raw, **`ramp-<arm>.csv` per request**, tables.md, serve logs). Labbook HRAMP-ramp, seq 503-510.
- Release config as scripts/release_session.sh: radiance 1.0.13 flashnext profile, `--gpu-headroom-mib 3072`, MTP 3,
  prefix cache on (4 GiB host + 131,072 MiB disk), fresh cache dir per server under
  `~/.cache/radiance-kva-release/ramp-<server>/`, removed after (20-21 GB each).
- Packages: `dist-ridgefill-0.1.0` (7f83cfc1…, df7eef04…), extracted fresh. Every SHA256SUMS verified (39 files).
  Projector mounted at /models/projector and found by discovery ("matches" line in the serve logs).
- Servers in order:
  1. throwaway stock: one 512-token request, stopped at 3 min;
  2. quality, speed, stock: each fresh, with **30 consecutive 32,768-token requests then 5 at 16,384**, back to back, no
     warm-up.
  - Requests use tools/speed.py's builder (natural prose RK_DOCS, `Measurement nonce {n}.`, token ids, max_tokens 1),
    and every request had 0 cached tokens.
  - Per request: timings.prompt_ms, wall, rank 0's /stats before/after, approximate-step log lines, MemAvailable/SwapFree.
- Kernel log: 0 amdgpu MES/SMU/timeout/reset lines before and after every server.
- Memory at each server start: 82.0-82.8 GiB available, swap 16.3/16.4 GiB used.
- **Confound (not controlled): the n-gram table's auto placement.**
  - The engine puts the n-gram table in RAM only if 79.69 GiB is available at placement.
  - quality (77.77 GiB), speed (77.60) and the throwaway stock (79.66) put it on **disk**.
  - The measured stock server had enough and put it in **RAM**.
  - The release session had the same split (exact-b RAM; quality-b, speed-a and speed-b disk).
  - BetterBench finding 2 measured no disk reads from the disk-placed table during serving.

## The three series (32,768 tokens, prompt_ms in s)
| # | quality | speed | stock | quality / stock same # | speed / stock same # |
|---|---|---|---|---|---|
| 1 | 20.20 | 14.69 | 21.64 | 1.07x | 1.47x |
| 2 | 20.22 | 14.70 | 22.02 | 1.09x | 1.50x |
| 3 | 19.70 | 14.24 | 22.17 | 1.13x | 1.56x |
| 5 | 17.99 | 12.78 | 21.58 | 1.20x | 1.69x |
| 10 | 13.93 | 9.85 | 21.14 | 1.52x | 2.15x |
| 13 | 11.98 | 8.40 | 20.98 | 1.75x | 2.50x |
| 15 | 10.80 | 7.73 | 20.83 | 1.93x | 2.70x |
| 17 | 10.20 | 7.51 | 20.62 | 2.02x | 2.75x |
| 18 | 10.06 | 7.42 | 20.46 | 2.03x | 2.76x |
| 20 | 9.98 | 7.20 | 20.44 | 2.05x | 2.84x |
| 25 | 9.96 | 7.25 | 20.44 | 2.05x | 2.82x |
| 30 | 9.92 | 7.34 | 20.44 | 2.06x | 2.78x |

All 30 rows are in tables.md and the CSVs.

- quality falls ~0.85 s a request from #3 to #14, then flattens.
- speed falls the same way. Its last 10 creep up 2% (7.20 -> 7.34 s).
- stock falls too, by less: 21.64 -> 20.44 s (-5.5%).

16,384 tokens, the 5 requests right after the 30 long ones (s):
- quality 5.50 5.53 5.52 5.54 5.55;
- speed 4.24 4.32 4.39 4.48 4.56 (rising 7.5%: not flat);
- stock 10.16 10.07 10.25 10.26 10.27.

## Plateau and speedups
Plateau = the first request within 2% of the arm's final-10 median. "Stays from" is the first request after which every
later request stays inside that band.

| arm | final-10 median 32K | plateau (stays from) | prefill time spent to reach it | settled 32K vs stock | 16K median vs stock |
|---|---|---|---|---|---|
| quality | 9.96 s | #18 (#18) | 255 s of prefill over 17 requests | **2.05x** | 5.53 s = **1.85x** |
| speed | 7.27 s | #18 (#18) | 183 s | **2.81x** | 4.39 s = **2.33x** |
| stock | 20.44 s | #11 (#15) | | 1.00x | 10.25 s |

- **Fresh server, long requests 1-3, vs stock's same request:**
  - quality 1.07x / 1.09x / 1.13x;
  - speed 1.47x / 1.50x / 1.56x;
  - vs stock's settled 20.44 s: quality 1.01 / 1.01 / 1.04x, speed 1.39 / 1.39 / 1.44x.
- **The release numbers are reachable in steady state, after ~15-18 consecutive 32K prompts:**
  - quality 2.05x settled vs the published 2.10x (-2%);
  - speed 2.81x vs 2.55x (above it);
  - quality first reaches 2.0x at #17; speed passes 2.55x at #14.
  - At 16K, after the long run: 1.85x / 2.33x vs the published 1.49x / 1.97x.
- **The release session was right for the wrong reasons.**
  - Its stock was 24.06 s against 20.44 s here (+18%). This stock and the BetterBench probe's stock (22.2 / 22.5 s on
    its first long requests) were both faster. Cause not known.
  - Its quality reps (13.23 -> 10.19 s) are this run's #10-#17 segment: still falling, as BetterBench found.
  - Against its own stock, settled quality would have been ~2.4x. The slow stock and the unsettled quality offset each
    other.

## Why: the expert cache adapting to RidgeFill's routing
Rank 0's /stats per 32K request:

| arm # | promotions | readmits | routed | resident share | stream (miss) GiB | link h2d GiB | resident slots L0-23 / L24-47 |
|---|---|---|---|---|---|---|---|
| quality 1 | 224 | 6 | 198,027 | 0.723 | 128.9 | 142.1 | 8,718 / 8,710 |
| quality 5 | 1,241 | 960 | 215,690 | 0.807 | 97.7 | 117.8 | 9,613 / 7,828 |
| quality 10 | 1,254 | 941 | 215,641 | 0.905 | 47.8 | 80.4 | 10,580 / 7,181 |
| quality 15 | 272 | 272 | 215,525 | 0.963 | 18.6 | 55.9 | 11,155 / 6,593 |
| quality 20-30 | 15-64 | 8-53 | ~215,500 | 0.973 | 13.4-13.7 | ~49.8 | ~11,265 / ~6,490 |
| speed 1 | 224 | 6 | 140,587 | 0.725 | 90.9 | 147.3 | 8,718 / 8,710 |
| speed 20-30 | 5-39 | 5-31 | ~158,000 | 0.985 | 5.3-5.5 | 57-59 | ~11,200 / ~6,550 |
| stock 1 | 224 | 5 | 257,470 | 0.722 | 168.3 | 264.1 | 8,725 / 8,763 |
| stock 20-30 | ~980 | ~977 | ~274,700 | 0.916 | ~53.8 | ~241 | 9,833 / 7,998 |

- **Fresh state.** A fresh server's resident set (the container profile's ranking) holds about as many experts of
  layers 24-47 as of 0-23. RidgeFill computes layers 24-47 only for the exact rows and the tail, so it routes few
  experts there, and the heat-based mover moves the slots to layers 0-23.
- **The rate is capped.** radiance's `HeatParams::moves_per_dispatch = 8` per slab class
  (core/place/heat.h:87-105) gives ~1,241 swaps per 32K request, from request 3 to about request 13. Requests 1-2 make
  only 224 / 272.
- **About 2,550 slots move on rank 0.** Then the swaps stop (quality 15-64 a request) and prefill is flat.
- **Correlations.** Spearman of prompt_ms over the 30 requests:
  - quality: resident share -0.96, stream bytes +0.96, h2d +0.96, L24-47 resident +0.89;
  - speed: -0.99 / +0.99 / +1.00 / +0.94.
  - These are monotonic series, so the request index correlates as well (-0.99 / -0.88). The causal link rests on
    the mechanism and the size of the change: stream bytes fall 129 -> 13 GiB (quality) and 91 -> 5 GiB (speed) while
    prompt time halves.
- **Stock adapts too, less.**
  - Its residency also shifts (L0-23 8,725 -> 9,833) and its stream bytes fall 168 -> 54 GiB.
  - But its total link traffic stays ~241 GiB: ~980 swaps a request, with readmits ≈ promotions, so it is thrashing
    at steady state. Its time falls only 5.5%.
  - RidgeFill's working set fits the slab once rebalanced; stock's does not.
- **Not the cause:** free memory (Spearman -0.18 / +0.18 on the RidgeFill arms), and approximate steps (15 per 32K
  request, constant from #1).

## What this means for the docs
- **Steady state.** On a server that keeps serving long prompts, the release figures hold: quality ~2.05x, speed
  ~2.8x at 32K, after ~15-18 long prompts (~4 min of prefill).
- **Fresh server.** Its first 1-3 long prompts get quality ~1.1x and speed ~1.5x.
- **Between the two:** it scales with how far the expert cache has moved.
- **Untested: mixed traffic.**
  - Heat decays per layer dispatch (0.98) and decode steps are dispatches too. Short prompts and decode run every layer
    stock-like, so interleaved traffic may pull the residency back toward balanced.
  - This could explain BetterBench's flat ~1.45x at 47K after 376 short requests, but that is not shown here.
  - Next test: long prompts interleaved with a few hundred decode tokens or short requests.
- **Open:** why the release session's stock was 18% slower (24.06 s) than this one, and the n-gram RAM/disk confound
  above.

## Prediction (HRAMP-ramp, seq 503): confirmed on the core, two sub-predictions missed
| prediction | result |
|---|---|
| quality 18-22 s -> plateau 8.5-11 s at #15-30 | 20.20 -> 9.96 s at #18 |
| speed 12-15 s -> 7-9 s by #25 | 14.69 -> 7.27 s at #18 |
| settled quality 2.0-2.8x, speed 2.5-3.3x | 2.05x, 2.81x |
| resident share / stream bytes, Spearman 0.8 or more | -0.96 / +0.96 (quality), -0.99 / +0.99 (speed) |
| stock flat within 5% | missed: -5.5% (it adapts too) |
| fresh requests 1-3: quality 1.1-1.4x, speed 1.5-1.9x | low: 1.07 / 1.09 / 1.13x and 1.47 / 1.50 / 1.56x |
