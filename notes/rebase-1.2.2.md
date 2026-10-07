# notes/rebase-1.2.2.md -- the plugin on radiance 1.2.2 (ac79f4d, seen upstream 2026-10-07 11:43Z): NOT MERGED

**Status: correct. The gate's stock-path reading (quality +1.20%) is engine-side measurement variance, not a plugin
change ("Correction" below): held-normalised the plugin costs ~+0.5%, as on 1.0.13-1.2.1. Kept on this branch for
Dylan's decision; main stays on 1.2.1.**

## What 1.2.2 changed, and what it means here
| change | here |
|---|---|
| static YaRN (#37, #38): `yarn_from` in qwen4exp declare, rotary ops through `decl_rotary` (also `rad_qsa.h`) | the adapter issues the engine's own rotary op handles (op_rope_cs, op_bkey, op_rope_q/k), so YaRN reaches the approximate path; the projector's quality under YaRN is unmeasured (release README says so). Without a YaRN config nothing changes (reference byte-identical) |
| MoE prefill rewrite (#35), "by rule, bit-identical"; MoE decode down kernel (#34) | reference byte-identical to 1.2.1's: confirmed bit-identical. Stock prefill 7-8% faster at 512 / 1,024 tokens |
| uneven-world VRAM (#36), adaptive draft window (#24), metrics | engine-side |
| abi/rad_builder.h +4, abi/rad_yarn.h (new); ABI number 15 | ridgefill.so rebuilds to 8c5b4ada (was a63c29bc); its output is unchanged (rows == G13) |

Host: `update_radiance.sh v1.2.2 --host-only` COMPATIBLE (71 oracle cases, pytest 228/33). Pin 76929c1; images
stilldeadcode/radiance:1.2.2 (pulled), radiance-build:1.2.2 (built; moved :latest).

## Gate (data/lead-20261006/gate122, 11:50-12:29Z)
| check | result |
|---|---|
| off == stock; ref-r1.2.2 == ref-r1.2.1; rows == G13, 67/67; e2e 0-6 | **PASS** |
| decode vs stock | 0.9927 / 0.9970 / 0.9948, 30/30 identical |
| stock-path (ABCA pooled) | quality **+1.20%** [+1.14, +1.22], speed +0.77%, quality2 +0.76%; drift q2/q -0.39% (largest seen) |

Where the extra cost goes: RidgeFill servers stream 1.2-1.4% more expert bytes than stock per request (1.2.1:
+0.4%) because their runtime resident share is 0.22-0.36 points lower (1.2.1: 0.1). The VRAM plan is identical
(14,994 slab slots on every server), so the difference is in which experts the mover keeps resident as requests
run -- something about the plugin's stock path steers 1.2.2's mover slightly differently. Next: compare the heat
engine's promotion / routing-evidence counters (serve logs, /stats) between a stock and an off server on 1.2.2.

## Cause, measured from the gate's own /stats (no GPU run)

| `experts.flex_capacity`, card 0 | 1.2.1 | 1.2.2 |
|---|---|---|
| stock | 4,107 MiB | 4,147 MiB |
| speed | 4,081 MiB | 4,081 MiB |
| quality | 4,081 MiB | 4,040 MiB |

Startup "already held": quality 404.89 MiB vs stock 343.02 on 1.2.2 (1.2.1: 375.56 vs 376.36). Two effects: 1.2.2 gives
stock 40 MiB more flex expert residency that the plugin servers do not get, and charges ~41 MiB of quality-mode
VRAM against flex that 1.2.1 did not (speed is unchanged). Quality ends 107 MiB (~44 expert units) short of stock,
streams 1.2-1.4% more, and prefills short prompts +1.2% slower. Plugin-side candidates: find what quality allocates
that 1.2.2 now measures as held (masked-path / row-table buffers?) and move it into the activation arena the engine
lends to experts at small steps, or shrink it; then re-gate the stock path.

## Correction: the cause is 1.2.2's startup measurement, not the plugin

The table above reads as if 1.2.2 charged quality-mode VRAM against the expert cache. It does not. Per server,
"already held" (the engine's startup measurement) plus flex is nearly constant:

| server | 1.2.1 held / flex / sum (MiB) | 1.2.2 held / flex / sum (MiB) |
|---|---|---|
| stock | 376.4 / 4107 / 4483 | **343.0** / 4147 / 4490 |
| speed | 375.6 / 4081 / 4457 | 371.5 / 4081 / 4453 |
| quality | 375.6 / 4081 / 4457 | **404.9** / 4040 / 4445 |
| quality2 | 375.5 / 4081 / 4457 | 375.6 / 4071 / 4447 |

The plugin's own footprint (stock's sum minus a plugin server's) is 26-45 MiB on both releases. On 1.2.2 the held
measurement varies 343-405 MiB from server to server (1.2.1: 375.5-376.4), and each server's flex expert cache is
what remains, so stock-path cost tracks it: quality (4040) +1.20%, quality2 (4071) +0.76%, speed (4081) +0.77%,
against a stock server that happened to measure low. Held-normalised, the plugin costs ~+0.5%. Worth reporting
upstream: 1.2.2's startup held measurement moves a server's expert cache by up to ~60 MiB, about +-1% on
short-prompt prefill, stock included. To confirm: a stock-path rerun with more servers (or held-matched servers).
