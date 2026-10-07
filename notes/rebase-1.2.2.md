# notes/rebase-1.2.2.md -- the plugin on radiance 1.2.2 (ac79f4d, seen upstream 2026-10-07 11:43Z): NOT MERGED

**Status: correct, but its stock-path cost reads at the ~1% line (quality +1.20% pooled, +0.98% over the ABCA pair),
up from +0.35-0.42% on 1.2.1. Held on this branch until Dylan decides or the cause is found; main stays on 1.2.1.**

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
