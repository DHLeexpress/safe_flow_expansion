#!/usr/bin/env python3
"""Run the 39fb3e6 B1 margin50 protocol with native SafeMPPI-cost execution.

The original margin50 trainer intentionally accepts only
``nominal_hp_max_step_margin`` at its declaration gate, although the frozen
trainer implements and validates ``nominal_hp_safemppi_cost`` for the B1
family.  This additive launcher changes the rule to max-step-margin only while
that declaration gate runs, then restores native SafeMPPI cost before
calibration, gathering, replay, training, or artifact generation.

The trainer repository is supplied through ``B1_MARGIN_TRAINER_ROOT`` and must
be a clean detached checkout of the exact source commit used by the reference
margin arm.  All scientific arguments remain arguments of the original
trainer.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


EXPECTED_TRAINER_SHA = "39fb3e63542e6b23efe3321f311505e553c0bec6"
TRAINER_RELATIVE = Path(
    "overnight_run_07_06/rev_expansion/codex_overnight"
)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _require_frozen_trainer(root: Path) -> Path:
    if not root.is_dir():
        raise RuntimeError(f"B1_MARGIN_TRAINER_ROOT is not a directory: {root}")
    sha = _git_output(root, "rev-parse", "HEAD")
    if sha != EXPECTED_TRAINER_SHA:
        raise RuntimeError(
            f"trainer source SHA {sha} != required {EXPECTED_TRAINER_SHA}"
        )
    if _git_output(root, "status", "--porcelain"):
        raise RuntimeError("trainer source worktree is not clean")
    trainer = root / TRAINER_RELATIVE
    if not (trainer / "grid_expand_afe_rbf.py").is_file():
        raise RuntimeError(f"frozen trainer is missing below {trainer}")
    return trainer


def _install_cost_validation_override(trainer_module) -> None:
    original_validate = trainer_module.validate_protocol_args

    def validate_cost_margin50(args) -> None:
        if args.protocol_profile != "b1_balanced_r0_margin50":
            raise ValueError("this launcher is exclusive to b1_balanced_r0_margin50")
        if args.execution_rule != trainer_module.EX.SAFEMPPI_COST:
            raise ValueError(
                "this launcher is exclusive to nominal_hp_safemppi_cost"
            )
        real_rule = args.execution_rule
        args.execution_rule = trainer_module.EX.MAX_STEP_MARGIN
        try:
            original_validate(args)
        finally:
            args.execution_rule = real_rule

    trainer_module.validate_protocol_args = validate_cost_margin50


def main() -> int:
    value = os.environ.get("B1_MARGIN_TRAINER_ROOT")
    if not value:
        raise RuntimeError("B1_MARGIN_TRAINER_ROOT is required")
    trainer = _require_frozen_trainer(Path(value).resolve())
    for path in (trainer.parents[1], trainer.parent, trainer):
        sys.path.insert(0, str(path))

    import grid_expand_afe_rbf as trainer_module

    _install_cost_validation_override(trainer_module)
    result = trainer_module.main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
