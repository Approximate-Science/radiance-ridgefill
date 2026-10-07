# notes/rebase-1.2.3.md -- the plugin on radiance 1.2.3 (05af4db, seen upstream 2026-10-07 13:23Z)

1.2.3 changes only the DFlash2 drafter (MXFP4 linears, candidates rescored against the target's lm_head) and
`rad_fp8.h`'s MXFP4 helpers: nothing the adapter copies, nothing on Flash-Next's path (it drafts with MTP). Host:
`update_radiance.sh v1.2.3 --host-only` COMPATIBLE (ABI 15, 71 cases, pytest 228/33). Pin bf9d1de on top of the
1.2.2 branch (keeps its YaRN note). Images: stilldeadcode/radiance:1.2.3 pulled; radiance-build:1.2.3 built (moved
:latest). Home bf9d1de: qwen4exp_fp8.so 6037ccce (carries "1.2.3"), ridgefill.so 8c5b4ada (= 1.2.2's).

## Gate (data/lead-20261006/gate123, 13:29-14:08Z): PASS -- labbook H123-* confirmed

| check | result |
|---|---|
| off == stock; ref-r1.2.3 == ref-r1.2.2; rows == G13, 67/67; e2e 0-6 | **PASS** (plugin 31878533, projector 203b5a20) |
| stock-path (ABCA pooled) | quality +0.72% [+0.58, +0.77], speed +0.74% [+0.60, +0.81], quality2 +0.74%; drift +0.04% |
| decode vs stock | 0.9945 / 0.9960 / 0.9944, 30/30 identical |

## The stock-path cost across releases, and what moves it (corrects notes/rebase-1.2.2.md's last section)

| release | startup "already held", MiB: stock / plugin servers | stock-path cost, quality / speed |
|---|---|---|
| 1.1.1 | 347 / 371-376 | +0.76% / +0.74% |
| 1.2.0 | 347 / 371-376 | +0.74% / +0.74% |
| 1.2.1 | **376** / 375-376 | +0.35% / +0.40% |
| 1.2.2 | 343 / 372-376, quality **405** | +1.20% / +0.77% |
| 1.2.3 | 347 / 376 | +0.72% / +0.74% |

A plugin server holds 25-30 MiB more at startup than stock on every release, plus its ~39 MiB of buffers; the
flex expert cache gets what is left, and the steady cost is **~+0.74%** (r3's level, inside the ~1% rule). The two
outliers are single servers whose held reading came out ~29 MiB high: 1.2.1's stock server (cost looked lower)
and 1.2.2's first quality server (cost looked higher). So: no release changed the plugin's cost; a gate's
stock-path number moves by ~+-0.4 points with one server's held reading, and a gate should print held per server.
