# Contributing to RidgeFill

## Reporting a problem

1. Run the report on the machine that serves, against the server's log:
   `python3 ridgefill_report.py --container <name>` (or `--log <file>`; add `--projector <dir>`). It ships in the
   plugin folder and in `tools/` here. Its first line usually names the problem and the fix
   ([docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)).
2. If that does not solve it, open an issue with the **Bug report** template and paste the whole report. Add the
   exact command or `docker run` line, what you expected, and whether the same happens with
   `RADIANCE_RIDGEFILL=off`. An issue without the report will be asked for it first.

The report replaces your home directory, user name and host name. It does not look for tokens or keys: read it
before you post it.

## A new radiance release

Follow [docs/REBASING.md](docs/REBASING.md): one command gives the verdict, and a compatible release is a
one-line pull request. Open a **New radiance release** issue if you only want to flag it.

## Pull requests

- **One concern per pull request.** A rebase, a fix and a cleanup are three.
- **Tests:** `ctest -LE gpu` in your build directory and `python3 -m pytest -q tests` must pass. A bug fix comes
  with a test that fails before it. Kernel or arch changes that touch GPU code need the GPU gates; say whether
  you ran them (the maintainer runs them otherwise).
- **Plugin-side only.** Never change radiance; carry its changes into our copies (docs/REBASING.md).
- **Portable.** No paths, host names, card addresses or machine sizes in source or scripts: they come from
  arguments or documented environment variables. `grep -rnE '/home/|/var/home|/mnt/' arch kernels tools scripts`
  stays empty for anything a user runs.
- **Messages.** A new or changed log line goes into `tools/ridgefill_report.py`'s `CATALOG` with its cause and
  fix, and into docs/TROUBLESHOOTING.md; `tests/test_ridgefill_report.py` checks both.
- **Commit messages say why.** The diff says what.
- **License.** Contributions are under Apache-2.0, as the project is. Keep the credit lines (NOTICE,
  CITATION.cff) as they are.
