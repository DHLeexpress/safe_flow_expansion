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


def test_retained_alpha_grid_is_complete_common_random_number_screen() -> None:
    root = ROOT / "provenance/paper_baselines/kazuki_alpha_fine_grid_m10"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "KAZUKI_ALPHA_COEFFICIENT_GRID_COMPLETE"
    assert manifest["M_per_cell"] == 10
    assert manifest["gammas"] == [0.1, 1.0]
    assert manifest["alphas"] == [0.1, 0.5, 1.0, 2.0]
    assert manifest["goal_coefficients"] == [0.0, 0.01, 0.05, 0.1]
    assert manifest["safe_coefficients"] == [0.5, 1.0, 1.5, 2.0]
    assert len(manifest["outputs"]) == 64
    assert "coefficients excluded for common random numbers" in manifest[
        "seed_contract"
    ]
    for entry in manifest["outputs"]:
        path = root / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_alpha_grid_figures_do_not_claim_a_timeout_local_minimum() -> None:
    sidecar = json.loads(
        (ROOT / "assets/paper/kazuki_alpha_fine_grid.json").read_text()
    )
    assert sidecar["status"] == "KAZUKI_ALPHA_FINE_GRID_FIGURES_COMPLETE"
    assert not sidecar["selected_modes"]["0.1"]["has_timeout_local_minimum"]
    assert not sidecar["selected_modes"]["1.0"]["has_timeout_local_minimum"]


def test_retained_alpha34_wall_grid_is_exactly_eight_arms() -> None:
    root = ROOT / "provenance/paper_baselines/kazuki_alpha34_wall_grid_m10"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "KAZUKI_ALPHA_COEFFICIENT_GRID_COMPLETE"
    assert manifest["M_per_cell"] == 10
    assert manifest["alphas"] == [3.0, 4.0]
    assert manifest["goal_coefficients"] == [0.0, 1.0]
    assert manifest["safe_coefficients"] == [3.0, 4.0]
    assert manifest["gammas"] == [0.1, 0.5, 1.0]
    assert len(manifest["outputs"]) == 8
    for entry in manifest["outputs"]:
        path = root / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_wall_search_is_diagnostic_only_and_uses_fifty_seeds() -> None:
    root = (
        ROOT
        / "provenance/paper_baselines/kazuki_alpha4_wg0_ws3_wall_search_m50"
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["M_per_cell"] == 50
    assert manifest["alphas"] == [4.0]
    assert manifest["goal_coefficients"] == [0.0]
    assert manifest["safe_coefficients"] == [3.0]
    assert manifest["gammas"] == [0.1, 0.5, 1.0]
    assert len(manifest["outputs"]) == 1
    sidecar = json.loads(
        (ROOT / "assets/paper/kazuki_alpha34_wall_grid.json").read_text()
    )
    assert sidecar["selected_wall_candidate"]["search_M"] == 50
    assert sidecar["selected_wall_candidate"]["wall_fraction_0p6"] < 0.2
