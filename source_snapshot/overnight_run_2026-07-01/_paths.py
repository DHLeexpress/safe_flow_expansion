"""Shared sys.path bootstrap for overnight_run_2026-07-01 (windowed FM + safe expansion restart).

Import first from every module so bare-name imports resolve:
  - `cfm_mppi.*` (planner, polytope)         -> repo ROOT
  - `dynamics/flow_policy/safeflow/...`      -> overnight_run_today/src
  - di_grid / polytope_explainer helpers     -> overnight_run_2026-06-28
  - the compact SOCP plotting API            -> ieee_compact_polytope_verifier_package/src
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                       # cfm_mppi repo root
RUN_TODAY_SRC = os.path.join(ROOT, "overnight_run_today", "src")       # dynamics/flow_policy/safeflow/descriptors
RUN_28 = os.path.join(ROOT, "overnight_run_2026-06-28")               # di_grid.py, polytope_explainer.py, best_area_mode4.json
IEEE_SRC = os.path.join(ROOT, "ieee_compact_polytope_verifier_package", "src")

BEST_CONFIG = os.path.join(RUN_28, "best_area_mode4.json")

for _p in (HERE, RUN_TODAY_SRC, RUN_28, IEEE_SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
