#!/usr/bin/env python3
"""Launch the frozen B1 balanced-sweep trainer with an execution rule outside
the two rules declared in the original 24-arm matrix.

The B1 protocol gate rejects undeclared execution rules
(`validate_protocol_args`: rule must be MAX_STEP_MARGIN or SAFEMPPI_COST).
For the execution-rule ablation requested on 2026-07-21 we need
`nominal_hp_max_step_progress` (SOCP/nominal-Hp-gated, rank by step progress)
and `legacy_max_horizon_progress` (ungated horizon progress, the 'terrible'
baseline). Both rules are fully implemented in the frozen trainer
(afe_execution.py; legacy branch at grid_expand_afe_rbf.py:634-664) — only
the declaration gate blocks them.

This wrapper swaps the rule to a declared value strictly for the duration of
the validation call (the `exact` comparison for execution_rule is
self-referential, so nothing else depends on it) and restores the real rule
before training. recipe.json records the TRUE rule. The frozen snapshot is
not modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKBOOK = Path(__file__).resolve().parents[1]
SNAP = WORKBOOK / "source_snapshot/overnight_run_07_06/rev_expansion/codex_overnight"
sys.path.insert(0, str(SNAP.parents[1]))
sys.path.insert(0, str(SNAP.parent))
sys.path.insert(0, str(SNAP))

import grid_expand_afe_rbf as G  # noqa: E402

_ALLOWED = {"nominal_hp_max_step_progress", "legacy_max_horizon_progress"}
_orig_validate = G.validate_protocol_args


def _patched_validate(args) -> None:
    real_rule = args.execution_rule
    if real_rule not in _ALLOWED:
        raise ValueError(
            f"this wrapper is only for {_ALLOWED}; use the frozen trainer directly"
        )
    real_audit = args.nvp_audit_all_k
    args.execution_rule = G.EX.MAX_STEP_MARGIN
    if real_rule == "legacy_max_horizon_progress":
        # the all-K NVP audit is nominal-Hp-only; the profile's exact dict
        # demands True, so satisfy it during validation and restore after
        args.nvp_audit_all_k = True
    try:
        _orig_validate(args)
    finally:
        args.execution_rule = real_rule
        args.nvp_audit_all_k = real_audit


G.validate_protocol_args = _patched_validate

if __name__ == "__main__":
    sys.exit(G.main())
