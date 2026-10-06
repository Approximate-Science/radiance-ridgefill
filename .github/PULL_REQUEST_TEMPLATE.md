<!-- One concern per pull request. CONTRIBUTING.md has the rules. -->

### What and why

### Checks

- [ ] `ctest -LE gpu` passes (paste the summary line)
- [ ] `python3 -m pytest -q tests` passes
- [ ] a bug fix comes with a test that fails without it
- [ ] new or changed log lines are in `tools/ridgefill_report.py` CATALOG and docs/TROUBLESHOOTING.md
- [ ] no machine-specific paths, hosts or sizes (CONTRIBUTING.md "Portable")
- [ ] GPU gates: ran (results below) / not run (the maintainer runs them)

### Rebase only

- radiance release and commit:
- `scripts/update_radiance.sh <tag> --host-only` verdict (paste summary.txt):
