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


def test_paper_assets_exist():
    expected = (
        "assets/data/full_space_all_gamma_trajectory_overlay.png",
        "assets/polytopes/low7_exact_polytope_replay.mp4",
        "assets/polytopes/safemppi_polytope_gammas.gif",
        "assets/results/report.png",
        "assets/results/selected_raw_m50_gallery.png",
        "assets/results/selected_expansion_diagnostic.mp4",
    )
    for relative in expected:
        assert (ROOT / relative).is_file(), relative
