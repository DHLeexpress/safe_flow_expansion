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


def test_normalized_ws_comparison_is_hashed_and_explicit_about_raw_scale() -> None:
    root = ROOT / "provenance/paper_baselines/kazuki_alpha4_wg0_ws6_m10"
    source_manifest = json.loads((root / "manifest.json").read_text())
    assert source_manifest["alphas"] == [4.0]
    assert source_manifest["goal_coefficients"] == [0.0]
    assert source_manifest["safe_coefficients"] == [6.0]
    assert source_manifest["gammas"] == [0.1, 0.5, 1.0]
    assert source_manifest["M_per_cell"] == 10
    assert len(source_manifest["outputs"]) == 1
    entry = source_manifest["outputs"][0]
    assert hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest() == entry[
        "sha256"
    ]

    gallery = json.loads(
        (
            ROOT
            / "provenance/b1_current_best/gallery_shared_v5/gallery_manifest.json"
        ).read_text()
    )
    assert gallery["canonical_plot_recipe"] == "scripts/build_b1_shared_galleries.py"
    assert gallery["guidance"]["normalization"] == (
        "paper_w_s = raw_safe_coefficient / 6"
    )
    assert gallery["guidance"]["arms"] == [
        {"paper_w_s": 0.5, "raw_safe_coefficient": 3.0},
        {"paper_w_s": 1.0, "raw_safe_coefficient": 6.0},
    ]
    assert gallery["layout"]["rows"] == [
        "SafeMPPI ID",
        "pretrained OOD",
        "ours r15 OOD",
        "CFM-MPPI normalized ws=0.5",
        "CFM-MPPI normalized ws=1.0",
    ]
    assert gallery["zoom"]["paired_raw_bank"]["pretrained_collision_count"] >= 5
    assert gallery["zoom"]["paired_raw_bank"]["ours_collision_count"] == 0
    assert gallery["zoom"]["selected_contexts"]["paper_ws0.5"][
        "guidance_cosine"
    ] < 0
    assert gallery["zoom"]["selected_contexts"]["paper_ws1.0"][
        "guidance_cosine"
    ] < 0
