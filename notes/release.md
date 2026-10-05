# notes/release.md — cutting a radiance-kva release

The two tools this repo's release flow runs, in order:

1. **`tools/package.py`** builds the distribution from the frozen artifacts of one commit.
2. **`scripts/e2e_fresh.sh`** is the release gate: it verifies the *packaged* artifacts end
   to end on a **fresh** stock engine, the exact shape of PLAN-FIX's FINAL DELIVERABLE
   (a stock `stilldeadcode/radiance:1.0.8` runtime image, the stock published `.rad`
   container (sha256 `0af5e962…4d20`), a clean `RADIANCE_HOME` holding only the packaged
   plugin, the projector folder unpacked from the package). Nothing is uploaded; the gate
   proves the box is ready to go. This release selects the mode **server-wide only**
   (`RADIANCE_KVA=off|quality|speed`); per-request ON/OFF is PARKED
   (`notes/future/per-request.md`, decided 2026-10-05), so the dist carries **no
   chat-template package by default**.

## The packaging inputs (all frozen by earlier steps)

| input | what it is | who made it |
|---|---|---|
| `--home` | `architectures/qwen4exp_fp8.so` + `kernels/kva.so` | `scripts/frozen_home.sh <commit>` (builds from ONE commit in the radiance-build image, runs the host tests first) |
| `--projector` | the projector folder: 29 files incl. `kva.json`; its own `projector.dtype` field (`bf16` or `int8`) is what names the package | `tools/kva_projector.py build` (writes and hashes every file), or `tools/kva_projector.py int8 --from <bf16 folder> --out <dir>` (the int8 build) |
| `--with-template` | opt IN to the `kva-chat-template` package. Default **off**: per-request KVA is parked, and this release selects the mode server-wide | the 2026-10-05 decision (`notes/future/per-request.md`) |
| `--template-spec` | `kva-marker-spec.json` — required **only with `--with-template`** | the marker spec (worker-context; the template tool holds no constants) |
| `--base-template` | the model's own chat template — required **only with `--with-template`** | extracted from the served container (`tools/kva_projector.py` verifies the merge round-trip when it builds the folder) |
| `--version`, `--commit` | the plugin version and the commit the home was built from | the release notes |
| `--abi-version` | the radiance ABI the plugin was linked against | `find_package`'s version, if you want it in `VERSION.json` (it is not recorded in the `.so`) |

`VERSION.json` also carries the radiance release the plugin was built against and the GPU
targets, and records **how each field was obtained**: the release string is read from the
arch `.so` (the NUL-delimited `KVA_RADIANCE_VERSION` that `arch/CMakeLists.txt` compiles
in; a bare `d.d.d` string, so version tags like `GLIBCXX_3.4.32` don't match), the targets
from the `gfx…` strings in the binaries. Pass `--radiance-version` / `--gpu-targets` only
to disambiguate or to record what a host-only build cannot carry.

### What lands in `dist/`

```
radiance-kva-<version>/                   the plugin home + README.md + LICENSE + VERSION.json + SHA256SUMS
projector-qwen3.8-flash-next-<dtype>/     the projector folder, hashes re-verified against its kva.json
                                          BEFORE copying (a mismatch refuses naming the file); named by
                                          the dtype the manifest itself carries in projector.dtype —
                                          bf16 or int8, never hard-coded
                                          + LICENSE (the repo's) + SHA256SUMS
kva-chat-template/                        ONLY with --with-template (default off; per-request KVA is
                                          parked, notes/future/per-request.md): kva_template.py + the
                                          spec + a pre-merged chat_template.jinja (built by
                                          `kva_template.py merge`, re-verified by `check`)
                                          + a README (merge your own; --override-chat-template)
                                          + LICENSE (the repo's) + SHA256SUMS
<each>.tar.gz                             deterministic tarballs (sorted names, fixed mtime/uid/gid,
                                          gzip mtime 0): two runs over the same inputs give identical bytes
SHA256SUMS                                every tarball (two; three with --with-template)
```

Every file is mode 0644; every refusal names the missing thing; nothing is written outside
`--out` (and `--out` must not overlap an input). Every package now carries the repo's
Apache-2.0 LICENSE (which landed in 7d52e70, after this note first said packaging refused to
run without one); the projector loader only hashes the files kva.json lists, so the extra
file changes nothing on the load path. One thing still to fix **before** the first real
release: the release commit must be merged (not a worktree) so the recorded commit is fetchable.

## Running the gate

```sh
tools/package.py --home data/home-<short> --projector data/projector-qwen38fn \
    --out dist/ --version <x.y.z> --commit <sha>

# the chat-template package is NOT built by default (per-request KVA is parked);
# opt in explicitly when testing that code path:
#   --with-template --template-spec kva-marker-spec.json --base-template <model chat template>

RK_DIST=$PWD/dist RK_MODEL=<the published .rad> RK_E2E_WORK=/tmp/rk-e2e \
    scripts/e2e_fresh.sh
```

The gate finds the projector package by its `projector-qwen3.8-flash-next-<dtype>.tar.gz`
pattern (whichever dtype the dist carries, bf16 or int8) and demands the optional
`kva-chat-template.tar.gz` only when `RK_E2E_STAGE_F=1` asks for its case.

The gate never copies the 114 GiB model: `<work>/model/<basename>` is a **symlink** to it
(host-side reference only), and every container mounts the model's **real directory** at
`/models` (a symlink inside a bind mount would dangle; docker does not rewrite targets)
with the projector folder bind-mounted a second time at `/models/projector` — the path the
loader's discovery order finds beside `--model`. The model's sha256 is checked against the
published `0af5e962…4d20` (override: `RK_E2E_MODEL_SHA256`; empty string skips it). Each
container runs with the same `RK_FLAGS` as every measurement script (`scripts/common.sh`),
`scripts/preflight.sh` in front of it, and is stopped and removed in an EXIT/INT/TERM trap —
a failed gate leaves no server behind. The verdict and every piece of evidence land in
`<work>/e2e-report.json`; the exit status is 0 only when no case failed.

### What each case proves

| case | configuration | what a pass proves |
|---|---|---|
| 0 | the stock image, no plugin, no projector | the baseline: `ident.sh`'s six hashes and a 16,384-token TTFT that every "≡ stock" case is compared against, taken with the same image and flags |
| 1 | the packaged plugin, `RADIANCE_KVA=quality`, an empty dir at `/models/projector` (it shadows any real `projector/` the model dir may hold) | the no-projector refuse path: the plugin logs `KVA: no projector folder …; serving stock` and the server is **byte-identical** to stock |
| 2 | projector mounted, `RADIANCE_KVA=off` | `off` never looks at the folder and stays stock, byte for byte |
| 3 | `RADIANCE_KVA=quality` | the packaged projector loads (the `0 warning(s)` line), the pass actually **approximates** — the `kva: approximate step (quality, …` lines the plugin logs on every approximate chunk — and a 16,384-token prompt (tools/speed.py's prompt builder, exact length, cache checked) is **faster** than stock's TTFT |
| 4 | `RADIANCE_KVA=speed` | the same, in speed mode |
| 5 | a scratch projector copy with one byte flipped in a `proj.L*.safetensors` | a corrupt folder is **refused by name** (the file's manifest hash), the engine serves stock, byte-identical |
| 6 | `RADIANCE_KVA_PROJ_PLACE=vram` (a retired switch — the projector is always streamed from host RAM now) | the engine **refuses at startup naming the switch**: the container exits before `/health` ever answers, so an old command line cannot silently run something else |
| 7 | `--override-chat-template` with `"chat_template_kwargs": {"kva": "on"}` | **PARKED** with the per-request feature (`notes/future/per-request.md`): SKIPPED with that reason by default; with `RK_E2E_STAGE_F=1` **and** a dist built with `--with-template` it runs and expects the request to succeed and the log to show the marker detected |

Cases 1, 2 and 5 are the identity demands (off ≡ stock, no-projector ≡ stock, a
wrong projector refused by name while serving stock); case 6 proves a retired switch
is refused by name at startup; 3 and 4 prove the packaged projector actually engages
(the approximation is logged) and pays; 7 is the per-request opt-in, parked for a future
update. The projector is always streamed from host RAM, so no case (and no switch) can
place it in VRAM anymore.

## What is still pending

- **Per-request ON/OFF (Stage F) — PARKED** (Dylan, 2026-10-05; `notes/future/per-request.md`).
  This release selects the mode server-wide only (`RADIANCE_KVA=off|quality|speed`), so the
  chat-template package is not built by default: `package.py --with-template` keeps the code
  path (and its tests) alive behind the flag, and case 7 stays SKIPPED behind
  `RK_E2E_STAGE_F=1`, needing a dist built with the flag. Resumption order: rebuild on the
  core/adapter split, then follow `notes/stagef-plan.md` from F.2. Note for whoever lands it:
  the startup guard (PACKAGING §0) refuses an override template that contains the marker when
  no usable projector is loaded — check how that interacts with an `RADIANCE_KVA=off` arm
  before flipping the default, since `off` never loads the folder.
- **The kva:on quality numbers**: the FINAL DELIVERABLE also asks that per-request on
  equals the quality numbers; that belongs to the parked feature's own gates (grade.sh vs
  the reference), not to this fresh-engine gate.
- **Uploading**: nothing here uploads anything. When Dylan is ready, the plugin tarball and
  the projector folder go to the HF repos (PACKAGING §3-§4: `Dyluhn/kva-projector-<model>`
  holds exactly the `projector/` contents), the gate run recorded beside them.
- **Determinism note**: the tarballs are byte-deterministic, so a release can be re-verified
  bit for bit — record the `dist/SHA256SUMS` digests (two packages; three with
  `--with-template`) in the release notes.