# Rebasing RidgeFill onto a new radiance release

Each RidgeFill build targets one radiance release (`RADIANCE_VERSION`: `<version> <commit>`). On any other
release the plugin turns itself off and serves stock (log code RF-201, docs/TROUBLESHOOTING.md). Moving it to a
new release is one command, and anyone can run it and open a pull request with the result.

## What you need

No GPU for the verdict. Linux with `git`, `cmake` >= 3.21, `g++` 14 (C++20), `binutils`, Python >= 3.12 and

```sh
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt   # torch: the CPU build is enough
```

## 1. Ask whether the release fits

```sh
RK_PYTHON=.venv/bin/python scripts/update_radiance.sh v1.2.0 --host-only
```

The first run clones radiance into `data/radiance` (or set `RK_RADIANCE_REPO` to a checkout you have; it is only
read). The script builds that release host-only, builds the plugin against it, and runs the static oracle -- every
path of the plugin held op for op against that release's own in-tree architecture -- plus the kernel host rows
and pytest. It ends with one verdict and its exit code:

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | COMPATIBLE | builds and every host test passes against the new release |
| 1 | INCOMPATIBLE | a build error, an ABI number change, or an oracle case that no longer matches |
| 2 | INFRASTRUCTURE | the answer could not be computed (clone, toolchain, Python packages); never "compatible" |

Everything it wrote is in `data/update-<release>/`: `summary.txt` (the verdict and each failure),
`report.md` (every radiance file under `abi/`, `arch/common/`, `arch/qwen4exp_fp8/` that changed since the pinned
release, and which of our files copies it), and the build and test logs.

## 2a. COMPATIBLE: pin it

Change the one line in `RADIANCE_VERSION` to `<version> <full commit of the tag>`. Every image name, source
directory and default in the scripts derives from it (`scripts/common.sh`). Run the command again: it now
reports against the new pin, with no changed files. Commit (`pin radiance <version>`) and open a pull request.

## 2b. INCOMPATIBLE: port the change

`summary.txt` names each failing case with its first difference and the adapter files it points at. Those are
the files in `arch/` that carry a **copy** of radiance's in-tree code (`arch/qwen4exp.copies` maps each copy to
its radiance source). Read the radiance diff for that source (`report.md` lists it), carry the change into the
copy, and rerun until the verdict is COMPATIBLE. Rules:

- Never edit radiance. RidgeFill is plugin-side only; the engine is a dependency.
- Keep each copy textually close to its source: the next rebase diffs against it.
- A changed message string goes into `tools/ridgefill_report.py`'s `CATALOG` and `docs/TROUBLESHOOTING.md`;
  `tests/test_ridgefill_report.py` fails until it does.

Worked example: radiance 1.1.0 fused the MoE router (`router_gemm_topk_scatter`); the plugin's hand-issued MoE
copy had to follow, and 12 oracle cases said so until it did (`notes/rebase-master.md`).

## 3. GPU validation (maintainers, or contributors with the hardware)

The host verdict says the plugin issues exactly what the engine does. The GPU gates say it still computes and
runs as it did: `off` byte-identical to stock, the approximated output's KL against an exact reference, the
package's fresh-engine end-to-end cases, the speed of requests that do not approximate. They need two `gfx1201`
cards, the stock model, the ROCm build image `radiance-build:<release>` (radiance's `docker/build.sh -r <commit>
--target build`), and the KL corpus (`corpus/`, maintainer data):

```sh
RK_MODEL=<the stock .rad> scripts/update_radiance.sh v1.2.0 --gpu-smoke   # ident + KL, ~20 GPU minutes
```

If you cannot run them, say so in the pull request; the maintainer runs them before merging.

## What goes in the pull request

- The radiance release and its commit; the `summary.txt` verdict (paste it).
- For INCOMPATIBLE releases: which radiance change broke which case, and how the copy now follows it.
- Whether the GPU gates ran, and their result lines if they did.
