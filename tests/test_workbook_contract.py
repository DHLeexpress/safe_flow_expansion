from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_package_verifier_passes():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_package.py")],
        cwd=ROOT,
        check=True,
    )


def test_b1_current_best_recipe():
    recipe = json.loads((ROOT / "configs" / "b1_current_best_recipe.json").read_text())
    assert recipe["source_git_commit"] == "63ebefa7877c0b923c1c7cdea19228302dd6a0ca"
    assert recipe["execution_rule"] == "nominal_hp_safemppi_cost"
    assert recipe["conditioning_schema"] == "low7_closest_boundary_tie_mean"
    assert recipe["kernel"] == "RBF"
    assert recipe["gp_cap"] == 512
    assert recipe["adaptive_ess_target"] == 0.25
    assert recipe["negative_alpha"] == 0.01
    assert recipe["replay_window"] == 2
    assert recipe["optimizer_steps_formula"] == "ceil(|eligible D+|/batch)"
    assert recipe["freeze_visual_encoder"] is True
    assert recipe["no_fallback"] is True


def test_static_teacher_contract():
    teacher = json.loads((ROOT / "configs" / "safemppi_static_teacher.json").read_text())
    planner = teacher["planner"]
    assert planner["centroid_gain"] == 0.2
    assert planner["centroid_smooth"] == 0.25
    assert planner["centroid_eps"] == 0.15
    assert planner["smooth_weight"] == 0.12
    assert planner["safety_margin"] == 0.0
    assert planner["barrier_extra_margin"] == 0.0
    assert teacher["scene"]["planning_margin_m"] == 0.0
    assert teacher["scene"]["robot_radius_m"] == 0.0
    assert teacher["semantics"]["centroid_active_for_static_obstacles"] is True

    base = json.loads(
        (ROOT / "source_snapshot" / "overnight_run_2026-06-28" / "best_area_mode4.json").read_text()
    )["config"]
    assert base["centroid_gain"] == planner["centroid_gain"]
    assert base["centroid_smooth"] == planner["centroid_smooth"]
    assert base["centroid_eps"] == planner["centroid_eps"]
    assert base["safety_margin"] == planner["safety_margin"]

    grid_scene = (
        ROOT / "source_snapshot" / "overnight_run_07_06" / "grid_scene.py"
    ).read_text()
    assert 'cfg["polytope_area_sampling"] = False' in grid_scene
    assert 'cfg["urgency_size_diff"] = False' in grid_scene
    assert 'cfg["noise_sigma"] = [0.5 * (noise_var_mult ** 0.5)] * 2' in grid_scene
    assert 'cfg["centroid_gain"]' not in grid_scene

    safemppi = (
        ROOT / "source_snapshot" / "cfm_mppi" / "safegpc_adapter" / "safemppi.py"
    ).read_text()
    assert "barrier_extra_margin: float = 0.0" in safemppi
    assert "if self.config.centroid_gain > 0.0 and poly is not None:" in safemppi


def test_checkpoint_hashes():
    assert sha256_file(ROOT / "checkpoints" / "b1_balanced_pretrained.pt") == (
        "524c9c0a4fd071221ac509b9d8e6fbbfb85fdf1811aa04160317f2a9e2d3ef90"
    )
    assert sha256_file(ROOT / "checkpoints" / "b1_current_best_r19.pt") == (
        "60c155472f5ed0e4a1d53581857f09aead7924f8ce11e8e3adf890d5a57fc079"
    )


def test_b1_clearance_is_split_by_outcome():
    payload = json.loads(
        (ROOT / "provenance" / "b1_current_best" / "clearance_breakdown.json").read_text()
    )
    row = next(
        item for item in payload["rows"]
        if item["round"] == 20 and item["gamma"] == 0.1
    )
    assert row["all"]["n"] == 50
    assert row["success_only"]["n"] == 45
    assert row["failure_only"]["n"] == 5
    assert abs(row["all"]["mean"] - 0.04909099576696931) < 1e-12
    assert abs(row["success_only"]["mean"] - 0.054879736618134445) < 1e-12


def test_current_assets_exist():
    expected = (
        "assets/data/full_space_all_gamma_trajectory_overlay.png",
        "assets/polytopes/low7_exact_polytope_replay.mp4",
        "assets/polytopes/safemppi_polytope_gammas.gif",
        "assets/results/b1_current_best/report.png",
        "assets/results/b1_current_best/selected_raw_m50_gallery.png",
        "assets/results/b1_current_best/selected_expansion.mp4",
        "assets/results/b1_current_best/b1_current_best_5x3_gallery.png",
        "assets/results/b1_current_best/b1_current_best_5x3_gallery.pdf",
    )
    for relative in expected:
        assert (ROOT / relative).is_file(), relative
