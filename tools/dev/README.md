# tools/dev -- the retired container-append route (dev only)

Before A' (2026-10-05) the fitted tensors were appended INTO the served `.rad` with
`rad-convert --reuse --in-place` (notes/sidecar.md §7). That is retired for users: Dylan's flow runs the
STOCK published model file and the plugin loads `<model dir>/projector/` (tools/ridgefill_projector.py builds it,
arch/ridgefill_projector.h loads it). The plugin no longer declares any `ridgefill.*` weight a container holds, and
refuses `RADIANCE_RIDGEFILL_DECLARE`, `RADIANCE_RIDGEFILL_PROJ` and `RADIANCE_RIDGEFILL_ST` by name.

Kept here, unchanged, for the record and for anyone who needs to reproduce the Stage 2-6 evidence:

| file | what it did |
|---|---|
| `ridgefill_sidecar.py` | the sidecar shard + `rad-convert --set` file (+ `verify`, R13). Its source readers are what `tools/ridgefill_projector.py` uses, so the folder's tensors are the appended bytes |
| `stub_checkpoint.py` | the header-only stub of the source checkpoint that `rad-convert --reuse --in-place` needs |
| `plan_diff.py` | the append's name + encoding gate (`--plan-only` log against `rad-info -v`) |

To append again you need a plugin home built from a commit that still declares container weights under
`RADIANCE_RIDGEFILL_DECLARE=all` (the last one is `05a9db0`: `scripts/frozen_home.sh 05a9db0`), then the
commands of notes/sidecar.md §7.4-§7.6. To undo an append, notes/sidecar.md §7.7 (`.pre-append`).
