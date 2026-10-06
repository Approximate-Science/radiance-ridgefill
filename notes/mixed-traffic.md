# notes/mixed-traffic.md -- does RidgeFill's expert-cache shift slow decode, and what do long prompts get under mixed traffic? (2026-10-06)

notes/ramp.md found that steady 32K traffic moves rank 0's resident expert slots from layers 24-47 to 0-23
(8,718/8,710 -> ~11,265/6,490), and that is what makes the settled speedups. Two questions followed:
1. After that shift, are decode and short requests slower than on a stock server that saw the same traffic? The rule:
   requests that do not use the approximation lose no speed against stock (a ~1% residual is accepted).
2. With long prompts interleaved with short chats, what speedup do the long prompts get?

## Answers
- **The no-degradation rule holds after a long-prompt burst and under mixed traffic.**
  - Fresh server (A): quality -0.3% and speed -0.5% against stock. That is the plugin's residual when nothing is
    approximated, inside the 1% allowance.
  - After the burst (C) and in mixed traffic (D), RidgeFill decodes **faster** than stock: +3.3% to +3.9%.
  - Under 8-way concurrency (E) no loss was shown, but the test was underpowered: quality's median is -1.7% with a CI
    reaching +0.5%.
- **Decode undoes the shift quickly.** Ten short chats after the burst brought quality's split from 11,263/6,470 back
  to 9,794/8,849, the same as stock's 9,800/8,871.
- **Under mixed traffic, long prompts get about the fresh-server speedup, not the settled one.**
  - Cycles 6-15 of D: quality **1.24x** [1.21, 1.27], speed **1.70x** [1.68, 1.74] against stock's D. The long-only
    plateau in this run was 2.05x / 2.79x.
  - The cache does not settle in between. Each 32K prompt moves it a little toward layers 0-23 and the next six chats
    move it back.

## What ran
- One gpuq session, 08:29:27-09:19:28Z (**50.0 GPU minutes**), boot 75e3e39b, repo 1209d26.
- Evidence is in `~/AI-Work/radiance-kva-plugin-20261004/evidence/mixed-20261006/`:
  - scripts: mixed.sh, mixed.py, analyze.py, and ramp.py (used for the throwaway server);
  - `mixed-<arm>.jsonl` (raw) and **`mixed-<arm>.csv`** (one row per request);
  - tables.md, session.log, and the serve logs.
- Labbook: hypotheses HMIX-decode and HMIX-mixed were registered first (seq 511-512). The per-arm series are seq
  513-530. No verdicts have been recorded yet.
- Config was exactly as in the ramp run:
  - radiance 1.0.13 flashnext profile, `--gpu-headroom-mib 3072`, MTP 3, prefix cache on, a fresh cache dir per server
    under `~/.cache/radiance-kva-release/mixed-<server>/`, removed afterwards (25 GB each);
  - `dist-ridgefill-0.1.0` packages (7f83cfc1…, df7eef04…), extracted fresh, 39 SHA256SUMS files verified;
  - the projector was found by discovery.
- Servers in order:
  - a throwaway stock server (one 512-token request, 3 min);
  - then quality, speed and stock, each fresh, each running the same seeded sequence back to back:
    - **A**: 10 short chats;
    - **B**: 20 x 32,768-token prompts (the ramp protocol);
    - **C**: the same 10 chats as A, with a new tag;
    - **D**: 15 cycles of [one 32K prompt, then 6 short chats];
    - **E**: 8 concurrent short chats, twice.
- Short chat:
  - a BetterBench v1 corpus conversation behind a fixed system prompt that starts with a per-request tag (no
    prefix-cache hit: 0 cached tokens on all 378 short requests);
  - prompts were 119-248 tokens (median 151), so a little shorter than the ~200 asked for;
  - streamed `/v1/chat/completions` with `max_tokens 256`, `ignore_eos` (exactly 256 tokens each), temperature 0, and
    `enable_thinking false`.
- Decode tok/s is the server's `timings.predicted_per_second`: decode tokens over first-to-last token, TTFT excluded.
  The client's own clock gives the same figure (median ratio 1.000).
- Kernel log: 0 amdgpu MES/SMU/timeout/reset lines before and after every server.
- **The n-gram table went to disk (auto) on all four servers.** Available memory at placement was 77.5-79.6 GiB, below
  the 79.69 GiB threshold, so the ramp run's RAM/disk confound is absent here.
- Short requests made 0 approximate-step lines on every arm. Each 32K prompt made 15 on the RidgeFill arms.
- Statistics:
  - medians use a bootstrap 95% CI (10,000 resamples);
  - an arm is compared with stock **paired by request** (the same conversation and tag), as the median of the per-pair
    ratios with a bootstrap CI;
  - "violates" means the median is below -1% and the whole CI is below 0.

## Decode: RidgeFill vs stock (paired by request)
| phase | pairs | quality vs stock | speed vs stock | violates? | TTFT quality / speed vs stock | answers identical to stock |
|---|---|---|---|---|---|---|
| A fresh | 10 | **-0.3%** [-0.5, -0.1] | **-0.5%** [-0.7, -0.2] | no (inside 1%) | +0.5% / +0.6% | 10/10, 9/10 |
| C after B | 10 | **+3.4%** [+0.7, +10.4] | **+3.3%** [+0.7, +9.9] | no (faster) | -4.3% / -4.9% | 10/10, 10/10 |
| D mixed | 90 | **+3.3%** [+2.6, +4.6] | **+3.9%** [+2.7, +6.5] | no (faster) | -6.0% / -7.1% | 90/90, 90/90 |
| E 8-concurrent, per request | 16 | **-1.7%** [-2.8, +0.5] | **-0.5%** [-1.1, +5.6] | no (CI includes 0) | +14% / -8% | 8/16, 7/16 |
| E aggregate, batch 1 / 2 | 2 | -4.5% / -1.7% | -0.4% / +7.5% | n = 2, no CI | | |

Medians of decode tok/s [95% CI]:

| phase | quality | speed | stock |
|---|---|---|---|
| A | 116.2 [81.0, 134.9] | 115.9 [80.7, 134.8] | 116.7 [81.3, 134.8] |
| C | 113.2 [86.3, 133.3] | 112.5 [85.6, 131.7] | 108.0 [81.2, 125.4] |
| D | 111.1 [105.5, 114.4] | 111.8 [105.8, 115.5] | 107.6 [101.6, 112.6] |
| E per request | 18.6 [18.3, 19.5] | 19.5 [18.8, 20.1] | 19.0 [18.6, 19.7] |
| E aggregate (batch 1 / 2) | 133.4 / 125.1 | 139.1 / 136.9 | 139.7 / 127.3 |

- **Wide CIs within an arm, tight pairs.** Per-request decode ranges from 42 to 173 tok/s, depending on the
  conversation and how much it streams (4-56 GiB per 256-token answer). Pairing removes that spread, which is
  why the paired CIs are narrow.
- **C, request by request.** RidgeFill's lead is largest on the first requests after the burst (C1 +17% / +24%, C3
  +14%) and is about 0 by C7-C9 (+0.2%, +1.1%, -1.1%).
- **Does stock's decode shift after B? Yes, a little, and more under mixed traffic.**
  - C vs A on the same conversations: stock -0.8% [-14.4, +6.7], quality +0.0%, speed +0.1%.
  - D median over A median (unpaired, different conversations): stock -7.8%, quality -4.4%, speed -3.5%.
  - Per short request in D, stock streams 12.6 GiB against RidgeFill's 11.3, and its resident share is 0.953 against
    0.958.
  - **Likely mechanism, not proven:** stock's full late-layer prefill leaves prefill-hot experts in the layer 24-47 slots,
    and decode has to displace them. RidgeFill's prefill barely heats those layers, so decode's experts win them back
    sooner.
- **E is underpowered.**
  - It has only 2 batches. With batching, identical answers drop to 7-8 of 16, so MTP acceptance and the token paths
    differ between servers.
  - Quality's per-request median (-1.7%) is past the 1% allowance, but its CI includes 0, so no loss is shown.
  - Aggregate decode at 8-way concurrency (~125-140 tok/s) is no higher than one stream (~110-140). Decode here is
    bound by expert streaming on every arm, stock included.
- **Unexplained:** speed's A5 answer differs from stock's and quality's (math-proof-medium; 0 approximate steps; MTP
  acceptance 0.820 vs 0.815). The same conversation matches at C5, and every other sequential short answer (A, C, D: 109 of 110
  on speed, 110 of 110 on quality) is byte-identical to stock's. Nondeterminism in a fresh server's first requests is likely but not shown.

## Long prompts: burst vs mixed (32,768 tokens; speedup = stock's same request / arm's)
| arm | B1 | B16-20 median | D cycles 1-5 | **D cycles 6-15 median [CI]** | **D 6-15 speedup vs stock D [CI]** |
|---|---|---|---|---|---|
| quality | 17.99 s (1.20x) | 9.97 s (2.05x) | 16.6-17.6 s | **17.35 s** [16.88, 17.60] | **1.24x** [1.21, 1.27] |
| speed | 12.94 s (1.67x) | 7.31 s (2.79x) | 12.1-12.6 s | **12.57 s** [12.34, 12.74] | **1.70x** [1.68, 1.74] |
| stock | 21.58 s | 20.40 s | 21.4-21.5 s | **21.44 s** [21.37, 21.48] | 1.00x |

- **B reproduces the ramp run**: plateau 2.05x / 2.79x from about B15, against 2.05x / 2.81x in notes/ramp.md.
- **B1 is faster than the ramp run's #1** (quality 17.99 s vs 20.20 s). Phase A's decode had already moved residency
  to 9,871 / 8,788.
- **D gives about the post-decode fresh speedup from cycle 1 onward.** Cycles 1-15 run 1.20x-1.30x (quality) and
  1.64x-1.78x (speed), with no trend.
- Stock also gives back its own ramp gain in D: 21.44 s against 20.40 s settled (+5%).

## Where the slot split sits (rank 0, resident experts in layers 0-23 / 24-47)
| point | quality | speed | stock |
|---|---|---|---|
| server start (ramp run) | 8,718 / 8,710 | 8,718 / 8,710 | 8,725 / 8,763 |
| end of A (10 chats) | 9,871 / 8,788 | 9,866 / 8,794 | 9,890 / 8,802 |
| end of B (20 x 32K) | **11,263 / 6,470** | 11,197 / 6,542 | 9,834 / 7,997 |
| end of C (10 chats) | **9,794 / 8,849** | 9,789 / 8,855 | 9,800 / 8,871 |
| D, after each 32K prompt (cycles 1-15, range) | 9,446-9,754 / 7,729-8,039 | 9,421-9,768 / 7,698-8,042 | 9,260-9,535 / 8,050-8,329 |
| D, after each cycle's 6 chats (range) | 9,831-10,151 / 8,476-8,797 | 9,850-10,169 / 8,460-8,782 | 9,762-10,091 / 8,568-8,904 |
| end of E | 9,953 / 8,719 | 9,947 / 8,705 | 9,975 / 8,706 |

- **Decode moves the cache much faster than prefill.** A short request makes ~1,140 promotions (median 11-13 GiB
  streamed); a 32K prompt in D makes ~265. Decode's own preferred split is about 9,900 / 8,800, not the profile's balance.
- **In D, one 32K prompt pulls RidgeFill to ~9,650 / ~7,850.** Six chats push it back to ~10,000 / ~8,650.
- **The burst-trained state (~11,260 / ~6,470) never comes back under this mix.** Per 32K prompt in D, quality streams
  92.8 GiB (resident share 0.818), against 13.6 GiB at the B plateau and 110.8 GiB at B1.

## What this means
- **No-degradation rule:** after a long-prompt burst and under mixed traffic, short requests and decode on RidgeFill are
  not slower than stock; they are 3-4% faster.
  - On a fresh server the residual is -0.3% / -0.5% (CI inside 1%).
  - 8-way concurrency showed no loss, but the test was too small to rule out ~2% on quality.
  - A larger E (e.g. 10+ batches, ABAB servers) would settle that.
- **Docs: the 32K speedup depends on the traffic.**
  - Long prompts back to back: ~2.05x / ~2.8x after ~15 of them.
  - Long prompts interleaved with chat (here 1 long : 6 short x 256 tokens): ~**1.24x / 1.70x**, which is about the fresh
    figure, from the first cycle.
  - The headline 2.10x / 2.55x should not be quoted without saying which.
- **Not tested:** other ratios (e.g. 3 long : 1 short), longer idle gaps, and long prompts that also decode many tokens.

## Predictions (registered seq 511-512) against results (verdicts left to the orchestrator)
| prediction | result |
|---|---|
| HMIX-decode: A within +/-1% | **met**: -0.3% / -0.5% |
| HMIX-decode: C 1-5% slower than stock, CI below 0.99 | **refuted**: +3.4% / +3.3% (faster) |
| HMIX-decode: D 0-3% slower | **refuted**: +3.3% / +3.9% (faster) |
| HMIX-decode: E within 2% | met on the medians (-1.7% / -0.5%); aggregate -4.5% / -1.7% on quality (n = 2) |
| HMIX-decode: stock's C within 3% of its A | met: -0.8% (CI -14 to +7) |
| HMIX-mixed: D 6-15 quality 1.3-1.8x, speed 1.8-2.4x | **missed low**: 1.24x / 1.70x. Kill (<= 1.15x / 1.55x) not reached: near fresh, not fully reset |
| HMIX-mixed: L0-23 during D 9,500-10,800 | met: 9,446-11,082 (quality), mostly ~9,600-10,150 |
