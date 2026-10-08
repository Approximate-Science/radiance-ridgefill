# notes/rebase-1.3.0.md -- the plugin on radiance 1.3.0 (89cee7c, seen upstream 2026-10-08 16:33Z)

1.3.0: llama-server's request fields and /props, Anthropic's Messages API, OpenAI's Responses API, logprobs;
`--model` takes a Hugging Face repo id; `--tp 3 --tp-wire wht6`; CPU-only serving again; `--kernels` names the
kernel libraries to load; a model the loaded libraries cannot fully serve refuses at startup; `--vram-kv-mib` and
`--vram-weights-mib` are gone (the split is automatic); `--embedding-placement ram` is the default; the pinned host
pool gets huge pages where shmem THP is off. `abi/rad_abi.h` and `abi/rad_sample.h` change by a comment and one
sampler flag (`RAD_SP_LOGPROBS`), ABI 15. No file the adapter copies changed, and nothing in this repo passes the
removed flags. Host: COMPATIBLE (ctest 5/5, pytest 228/33). Pin 4426e85; images stilldeadcode/radiance:1.3.0
(pulled), radiance-build:1.3.0 (built; moved :latest); home 4426e85: qwen4exp_fp8.so 509535ea (carries "1.3.0"),
ridgefill.so 8c5b4ada (unchanged).

## Gate (data/lead-20261006/gate130, 16:40-17:24Z): correctness PASS, stock-path at +0.95%
| check | result |
|---|---|
| off == stock; ref-r1.3.0 == ref-r1.2.4; rows == G13, 67/67; e2e 0-6 | **PASS** (plugin e5158528, projector 203b5a20) |
| decode vs stock | 0.9941 / 0.9946 / 0.9911, 30/30 identical |
| stock-path (all four arms started quiet, load 1.4-2.4) | quality +0.97% [+0.86, +1.28], speed +1.09% [+0.98, +1.21], quality2 +1.11% [+0.95, +1.33] |
| repeat on fresh servers (gate130b, 17:30-17:37Z) | quality3 / stock2 **+0.94%** [+0.76, +1.05], quality3 / stock3 **+0.96%** [+0.82, +1.05]; stock3 / stock2 -0.09% |

Labbook: H130-ident/-ref/-rows/-e2e/-decode confirmed; H130-stockpath refuted (predicted pooled <= +1.0%; speed
and quality2 read +1.09 / +1.11%); H130-stockrepeat confirmed.

## The stock-path cost moved from ~+0.74% to ~+0.95%
- **Not one server.** The repeat put two fresh stock servers around a fresh quality server. The stock servers agree
  within 0.1% pooled; every plugin arm of both sessions reads +0.92..+1.00% pooled against them (8 pairings).
  gate130's single stock server ran ~0.7% fast at 512 tokens, which is what pushed its short-length ratios to
  +1.3-1.8%.
- **Same plugin, faster engine (likely, not proven).** 1.3.0 prefills ~7% faster than 1.2.4 (stock, 2,000 tokens:
  1,313 -> 1,221 ms). Likely cause: the token embedding now lives in RAM, and the ~0.6 GiB it frees holds more
  experts (flex capacity at 2,000 tokens 4.1 -> 4.7 GiB, resident share 0.796 -> 0.815). The plugin's cost is
  what it was: every plugin server holds ~26 MiB more VRAM at startup (347 vs 373 MiB), so its expert cache is
  ~55-65 MiB smaller and it stages +63..79 units a long prompt more than stock (1.2.4: +55..60), with the same 49
  stages. In ms at 2,000 tokens that is ~10 ms on 1.2.4 and ~12 ms on 1.3.0, over a smaller denominator.
- It sits at the edge of the accepted ~1% residual: pooled medians are below +1%, the upper CI bounds +1.01..+1.15%.
