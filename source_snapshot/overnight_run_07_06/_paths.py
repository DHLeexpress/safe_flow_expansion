"""sys.path bootstrap: reuse overnight_run_2026-07-01 (di_grid_viz), overnight_run_today/src (dynamics.Env),
overnight_run_2026-06-28 (best_area_mode4.json), the repo root (cfm_mppi.*)."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RUN_01 = os.path.join(ROOT, "overnight_run_2026-07-01")
RUN_TODAY_SRC = os.path.join(ROOT, "overnight_run_today", "src")
RUN_28 = os.path.join(ROOT, "overnight_run_2026-06-28")
IEEE_SRC = os.path.join(ROOT, "ieee_compact_polytope_verifier_package", "src")
BEST_CONFIG = os.path.join(RUN_28, "best_area_mode4.json")

for _p in (HERE, RUN_01, RUN_TODAY_SRC, RUN_28, IEEE_SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
