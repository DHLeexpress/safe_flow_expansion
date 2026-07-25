from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_kazuki_absolute_coefficient_grid import COEFFICIENTS, GAMMAS


def test_absolute_grid_is_exactly_four_by_four() -> None:
    assert COEFFICIENTS == (0.0, 1.0, 2.0, 3.0)
    assert len({(goal, safe) for goal in COEFFICIENTS for safe in COEFFICIENTS}) == 16


def test_absolute_grid_uses_requested_gammas() -> None:
    assert GAMMAS == (0.1, 1.0)


def test_retained_absolute_grid_is_complete_and_hashed() -> None:
    root = ROOT / "provenance/paper_baselines/kazuki_absolute_grid_m10"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "KAZUKI_ABSOLUTE_COEFFICIENT_GRID_COMPLETE"
    assert manifest["M_per_cell"] == 10
    assert manifest["gammas"] == [0.1, 1.0]
    assert len(manifest["outputs"]) == 16
    for entry in manifest["outputs"]:
        path = root / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
