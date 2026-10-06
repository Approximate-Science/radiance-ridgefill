# Troubleshooting RidgeFill

**Fastest path: run the report and read its first line.** It reads the server log, checks the plugin and
projector files against their hashes, and names the problem with a code from the tables below.

```sh
# from the unpacked plugin folder (the script ships beside architectures/ and kernels/)
python3 ridgefill_report.py --container <your radiance container>      # Docker
python3 ridgefill_report.py --log <server stderr file>                 # native
# add --projector <model dir>/projector to verify the projector's files, --url http://127.0.0.1:<port> for /version
```

The report needs only Python 3.8+ (standard library), changes nothing, and replaces your home directory, user
name and host name with `<home>`, `<user>`, `<host>`. Paste its whole output into a bug report
(the source repository's CONTRIBUTING.md describes what a good report holds).

## Is RidgeFill running?

Three lines in the server log answer it, in this order:

1. `plugin …/qwen4exp_fp8.so is shadowed by <plugin dir>/architectures/qwen4exp_fp8.so` -- the engine found the
   plugin first on `RADIANCE_HOME`. **No such line: the plugin never loaded** (put its directory first on
   `RADIANCE_HOME`; a directory given only as `--radiance-home` is invisible to plugins).
2. `RidgeFill: projector … matches qwen4exp: … 0 warning(s)` -- the projector was found and fits this model. A
   `REFUSED` or `no projector folder` line here instead means the server runs **stock** (codes RF-101 to RF-108).
3. `ridgefill: approximate step (…)` -- a prompt chunk was approximated. Only prompts longer than the exact tail
   (`RADIANCE_RIDGEFILL_TAIL`, default 2048) have chunks to approximate. One line is logged per chunk the engine
   issues live; radiance replays a recorded pass without logging it, so later identical chunks may log nothing.

## Symptoms

- **The server does not start.** The last `radiance: qwen4exp_ridgefill:` line names the reason (configuration
  codes RF-3xx, RF-202, RF-401/402). Startup refusals are deliberate: RidgeFill never starts half-configured.
- **No speed-up.** Check line 3 above. Short prompts are not approximated by design; mode `off` approximates
  nothing; RF-201 means this build is for another radiance release and RidgeFill is off.
- **Output differs from stock.** Modes `quality` and `speed` approximate by design (README: measured quality).
  Mode `off` and the no-projector case must be byte-identical to stock: if they are not, that is a bug -- report it.
- **`amdgpu … [gfxhub] page fault` in the kernel log as a server stops.** Seen with the stock engine too (radiance
  1.0.13 and 1.1.1, the flashnext profile, after long prompts): every such burst on the maintainer's machine came
  within 3 s of a container stop. Results measured before the stop are not affected. A fault while the server is
  serving is a different matter: report it.
- **The GPU hangs or resets.** Check the report's *Kernel log* section. An amdgpu MES/SMU/reset line means the GPU
  or driver failed; results after one are not trustworthy. Retry with `RADIANCE_RIDGEFILL=off` to see whether it
  is RidgeFill-specific.

## Message codes

The codes are assigned by `ridgefill_report.py` (its `CATALOG`), not printed by the plugin: match your log line
against the middle column. A test keeps the catalog complete: every message the plugin can print has an entry.

### Status (all is well)

| Code | Level | The log line contains | What it means | What to do |
|---|---|---|---|---|
| <a id="rf-001"></a>`RF-001` | ok | `RidgeFill: projector … matches ` | RidgeFill loaded and the projector matches this model. | — |
| <a id="rf-002"></a>`RF-002` | ok | `RidgeFill: rank N holds the projector` | The projector is in host memory on this rank. | — |
| <a id="rf-003"></a>`RF-003` | ok | `ridgefill: approximate step` | A prompt chunk was approximated (one line a live step; a replayed recorded pass logs none). | — |
| <a id="rf-004"></a>`RF-004` | info | `ridgefill: hazard N positions` | A branched conversation resumed approximated positions from the prefix cache. | Informational. See README 'Prefix cache'. |
| <a id="rf-005"></a>`RF-005` | info | `RidgeFill: rank N holds nothing (plumb` | Mode plumb: the plugin path runs without fitted tensors (a test mode). | Use speed or quality to serve. |
| <a id="rf-006"></a>`RF-006` | ok | `plugin qwen4exp_fp8.so is shadowed by ` | The engine found RidgeFill's architecture first on the plugin search path. | — |

### Projector

| Code | Level | The log line contains | What it means | What to do |
|---|---|---|---|---|
| <a id="rf-101"></a>`RF-101` | warn | `RidgeFill: no projector folder (looked at` | No projector found; the server runs stock. | Put the projector folder at <model dir>/projector or set RADIANCE_RIDGEFILL_PROJECTOR (inside the container if you use Docker: mount it). |
| <a id="rf-102"></a>`RF-102` | error | `RidgeFill: projector … REFUSED, it cannot run on this model` | The projector was fitted for a different model (tokenizer, geometry or tensor names differ); serving stock. | Use the projector made for this model file (README 'Requirements' names the model and its SHA256). |
| <a id="rf-103"></a>`RF-103` | error | `RidgeFill: projector … REFUSED: ` | The projector folder is damaged or incomplete (a file is missing, corrupt or not the one the manifest names); serving stock. | Download the projector again and check it against its SHA256SUMS. |
| <a id="rf-104"></a>`RF-104` | warn | `RidgeFill: WARNING: projector … fitted on another variant` | The projector runs, but was fitted on a variant of this model (quantisation or weight anchors differ). | Expect quality to differ from the published numbers; use the projector for this exact file if you can. |
| <a id="rf-105"></a>`RF-105` | error | `selects exact rows from the table … holds none; serving stock` | Mode quality needs the projector's row table and this projector has none; serving stock. | Use the shipped projector, or RADIANCE_RIDGEFILL=speed. |
| <a id="rf-106"></a>`RF-106` | error | `RidgeFill: rank N could not allocate … for the projector` | Not enough host memory (or VRAM for the staging slot) for the projector on this rank. | Free RAM (the projector maps ~640 MiB a rank) or lower other host-memory users (pinned pools, n-gram table). |
| <a id="rf-107"></a>`RF-107` | error | `RidgeFill: rank N: the projector upload failed` | Copying the projector to the GPU failed. | Check the kernel log for amdgpu errors (in this report) and GPU health. |
| <a id="rf-108"></a>`RF-108` | error | `RidgeFill: rank N cannot load the int8 projector` | The int8 projector files could not be read on this rank. | Download the projector again; check SHA256SUMS. |

### Engine release, plugin search path, GPU

| Code | Level | The log line contains | What it means | What to do |
|---|---|---|---|---|
| <a id="rf-201"></a>`RF-201` | warn | `built against radiance … forwarding to the engine` | This plugin was built for another radiance release; it hands every request to the engine's own architecture, so RidgeFill is OFF. | Use the RidgeFill build for your radiance release, or build one (docs/REBASING.md). |
| <a id="rf-202"></a>`RF-202` | error | `built against radiance … no in-tree architectures/` | Engine release mismatch, and the engine's own architecture is not on RADIANCE_HOME behind the plugin, so startup fails. | Set RADIANCE_HOME=<plugin dir>:<engine share dir> (plugin first). A home given only as --radiance-home is invisible to plugins. |
| <a id="rf-203"></a>`RF-203` | error | `radiance: …: cannot load …: ` | The engine's own architecture file could not be loaded by the plugin's release guard. | Check the engine install and RADIANCE_HOME. |
| <a id="rf-204"></a>`RF-204` | error | `exports no rad_arch_declare/step` | A file named like the engine's architecture is not an architecture plugin. | Check RADIANCE_HOME. |
| <a id="rf-205"></a>`RF-205` | error | `plugin 'ridgefill' declines this machine` | ridgefill.so does not serve this GPU, so RidgeFill's kernels are missing. | RidgeFill ships for the GPU targets in VERSION.json (gfx1201). Other GPUs need a build for them. |
| <a id="rf-206"></a>`RF-206` | error | `kernel plugin 'ridgefill' (has no device code \| declares it was built for)` | ridgefill.so has no code for this GPU. | As RF-205. |

### Configuration

| Code | Level | The log line contains | What it means | What to do |
|---|---|---|---|---|
| <a id="rf-301"></a>`RF-301` | error | `: … is '…'; it takes ` | A RADIANCE_RIDGEFILL* variable has a value it does not take; startup refused. | Fix the value named in the line (README 'Configuration'). |
| <a id="rf-302"></a>`RF-302` | error | `is retired with the container append` | A retired switch (RADIANCE_RIDGEFILL_PROJ, _ST or _DECLARE) is set; startup refused. | Unset it. |
| <a id="rf-303"></a>`RF-303` | error | `is retired: the projector is always streamed` | A retired switch (RADIANCE_RIDGEFILL_PROJ_PLACE or _PROJ_RING) is set; startup refused. | Unset it. |
| <a id="rf-304"></a>`RF-304` | error | `ridgefill.tail is N tokens and the largest step is` | The exact tail is too long for the step size; startup refused. | Lower RADIANCE_RIDGEFILL_TAIL or raise --max-num-batched-tokens. |
| <a id="rf-305"></a>`RF-305` | error | `ridgefill.tail is N tokens; the shortest exact tail` | The exact tail is shorter than the method was measured at; startup refused. | Raise RADIANCE_RIDGEFILL_TAIL. |
| <a id="rf-306"></a>`RF-306` | warn | `the container's ridgefill.mode=… is ignored` | The mode comes from RADIANCE_RIDGEFILL only. | Set RADIANCE_RIDGEFILL. |
| <a id="rf-307"></a>`RF-307` | error | `RADIANCE_RIDGEFILL_MASK=all approximates rows and mode plumb` | A debug switch that cannot combine with mode plumb. | Unset RADIANCE_RIDGEFILL_MASK. |
| <a id="rf-308"></a>`RF-308` | error | `the row share is …; it takes (0, 1]` | RADIANCE_RIDGEFILL_SHARE is out of range; startup refused. | Use a value in (0, 1]. |
| <a id="rf-309"></a>`RF-309` | error | `the projector starts at layer … n-gram embedding enters` | The projector's split layer cannot serve this model. | Use the projector made for this model. |
| <a id="rf-310"></a>`RF-310` | error | `runs the MoE calibration tap` | An engine calibration run with RidgeFill on. | Calibrate with RADIANCE_RIDGEFILL=off. |
| <a id="rf-311"></a>`RF-311` | error | `block owns its input norm` | The engine's model build is one this plugin release cannot serve. | Open an issue with this report. |

### Kernels

| Code | Level | The log line contains | What it means | What to do |
|---|---|---|---|---|
| <a id="rf-401"></a>`RF-401` | error | `no kernel serves the (int8 projector's \| staging ring's copy \| projector (ridgefill_gemm)` | A kernel the projector needs is missing: the engine's kernel libraries (libr4d) did not load. | Keep the engine's own share directory on RADIANCE_HOME after the plugin directory. |
| <a id="rf-402"></a>`RF-402` | error | `no kernel library serves it -- ridgefill.so is missing` | ridgefill.so is not on RADIANCE_HOME, or declines this GPU. | Install kernels/ridgefill.so beside architectures/qwen4exp_fp8.so in the plugin directory. |
| <a id="rf-403"></a>`RF-403` | info | `so speed mode takes the masked path` | Speed mode uses its slower (masked) path for chunks that straddle the tail. | Informational. |

### Runtime

| Code | Level | The log line contains | What it means | What to do |
|---|---|---|---|---|
| <a id="rf-501"></a>`RF-501` | error | `does not end on the delta net's chunk tile` | An internal invariant broke (the scheduler cut a chunk RidgeFill cannot approximate); the step was refused. | Open an issue with this report and the full log. |

### Development switches

| Code | Level | The log line contains | What it means | What to do |
|---|---|---|---|---|
| <a id="rf-901"></a>`RF-901` | warn | `DEBUG RADIANCE_RIDGEFILL_` | A debug switch is set; output is not the served configuration. | Unset RADIANCE_RIDGEFILL_FORCE_*/SHIFT_B/MASK. |
| <a id="rf-902"></a>`RF-902` | error | `RADIANCE_RIDGEFILL_(CAPTURE \| DUMP)*: (device read failed \| cannot (append to \| write))` | A capture/dump tool (development only) could not read or write. | Check the dump directory; unset the switch to serve. |
| <a id="rf-903"></a>`RF-903` | error | `(a capture needs the split layer \| RADIANCE_RIDGEFILL_CAPTURE records exact runs \| RADIANCE_RIDGEFILL_CAPTURE_STATE copies the delta-net state)` | A capture tool (development only) is misconfigured. | Unset RADIANCE_RIDGEFILL_CAPTURE*. |

