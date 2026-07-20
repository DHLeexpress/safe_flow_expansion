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


def test_workbook_verifier_passes():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_package.py")],
        cwd=ROOT,
        check=True,
    )


def test_recipe_is_exact_selected_phase_c_arm():
    recipe = json.loads((ROOT / "configs" / "recipe.json").read_text())
    assert recipe["source_git_commit"] == "e63ebd80e5fa4f712a1a7cf590ec74c116768873"
    assert recipe["kernel"] == "RBF"
    assert recipe["lengthscale"] == 0.2256188330740796
    assert recipe["rollout_replicas"] == 8
    assert (recipe["K"], recipe["B"], recipe["T"]) == (16, 4, 300)
    assert recipe["replay_window"] == 2
    assert recipe["gp_replay_window"] == 2
    assert recipe["optimizer_steps_per_round"] == 32
    assert recipe["demo_frac"] == 0.25
    assert recipe["negative_alpha"] == 0.0
    assert recipe["freeze_visual_encoder"] is True
    assert recipe["no_fallback"] is True


def test_checkpoint_hashes():
    assert sha256_file(ROOT / "checkpoints" / "low7_pretrained_checkpoint.pt") == (
        "7ae44f773b3f5fe36579c4101542e495119cf6e348f622f5edbfedaa2855a46c"
    )
    assert sha256_file(ROOT / "checkpoints" / "phase_c_selected_r25.pt") == (
        "ab6ce3c39671554ef114234c464f23cc18828ea751a4bbb5547beb59793c1b54"
    )
    assert sha256_file(ROOT / "checkpoints" / "b1_balanced_pretrained.pt") == (
        "524c9c0a4fd071221ac509b9d8e6fbbfb85fdf1811aa04160317f2a9e2d3ef90"
    )
    assert sha256_file(ROOT / "checkpoints" / "b1_current_best_r19.pt") == (
        "60c155472f5ed0e4a1d53581857f09aead7924f8ce11e8e3adf890d5a57fc079"
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


def test_b1_clearance_is_explicitly_split_by_outcome():
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


def test_paper_assets_exist():
    expected = (
        "assets/data/full_space_all_gamma_trajectory_overlay.png",
        "assets/polytopes/low7_exact_polytope_replay.mp4",
        "assets/polytopes/safemppi_polytope_gammas.gif",
        "assets/results/report.png",
        "assets/results/selected_raw_m50_gallery.png",
        "assets/results/selected_expansion_diagnostic.mp4",
        "assets/results/b1_current_best/report.png",
        "assets/results/b1_current_best/selected_raw_m50_gallery.png",
        "assets/results/b1_current_best/selected_expansion.mp4",
        "assets/results/b1_current_best/b1_current_best_5x3_gallery.png",
        "assets/results/b1_current_best/b1_current_best_5x3_gallery.pdf",
    )
    for relative in expected:
        assert (ROOT / relative).is_file(), relative
