"""Read-only access to the RidgeFill research code the refit reuses unchanged (qfn.fit Sums / Held / solve, ridgefill.ridge,
qfn.hc, qfn.bigcap's checkpoint format, qfn.steps' chat frame). Its repo root comes from RIDGEFILL_RESEARCH_ROOT, so no
path is typed in this repo (R27); nothing here writes to it."""
import os
import sys
from pathlib import Path


def root():
    """The research repo root, put on sys.path once; refuses by name when it is not set or not that repo."""
    value = os.environ.get("RIDGEFILL_RESEARCH_ROOT")
    if not value:
        raise SystemExit("RIDGEFILL_RESEARCH_ROOT is not set: the RidgeFill research repo (qfn/, ridgefill/, b0/) the refit imports "
                         "read-only")
    path = Path(value).resolve()
    if not (path / "qfn" / "fit.py").is_file():
        raise SystemExit(f"RIDGEFILL_RESEARCH_ROOT={value}: no qfn/fit.py there")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
