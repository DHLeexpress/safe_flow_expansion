from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

_ANALYSIS = Path(__file__).resolve().parent
_ROOT = _ANALYSIS.parent
_REV = _ROOT.parent
_WORK = _REV.parent
_COLLIDING = {
    "_paths", "grid_feats", "grid_metrics", "grid_metrics2", "grid_rollout",
    "grid_scene", "grid_hp_expt", "grid_expand_hardtail", "di_grid_viz",
    "afe_core", "grid_expand_afe2", "afe2_scene_profiles", "afe2_calibration",
    "validate_afe2_pair", "verifier_polytope",
}


@contextmanager
def _validator_modules():
    names = {name for name in sys.modules if name in _COLLIDING}
    saved = {name: sys.modules.pop(name) for name in names}
    old_path = list(sys.path)
    sys.path[:0] = [str(_ANALYSIS), str(_ROOT), str(_REV), str(_WORK)]
    try:
        validator = importlib.import_module("validate_afe2_pair")
        scene_module = importlib.import_module("afe2_scene_profiles")
        yield validator, scene_module
    finally:
        for name in _COLLIDING:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = old_path


def _valid_round(scene_module):
    profile = scene_module.CODEX_RADIUS1_V1
    scene = scene_module.scene_snapshot(scene_module.build_scene(profile), profile)
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    eps = []
    viz = []
    start = np.asarray(scene["start_state"], dtype=np.float32)[:2]
    for gamma in gammas:
        eps.append({
            "gamma": gamma,
            "path": start[None].copy(),
            "status": "nvp",
            "term_t": 0,
            "steps": 0,
        })
        segments = np.repeat(start[None, None, :], 64 * 10, axis=0).reshape(64, 10, 2)
        viz.append({
            "t": 0,
            "gamma": gamma,
            "segsK": segments,
            "drawn": list(range(8)),
            "y": [0] * 8,
            "exec_y": [0] * 8,
            "terminal_rescue": [False] * 8,
            "terminal_tau": [None] * 8,
            "sel": -1,
        })
    recipe = {"reference_recipe": {"gammas": gammas, "K": 64, "B": 8}}
    db = {
        "round": 1,
        "scene": scene,
        "goal": np.asarray(scene["goal"]),
        "x0": np.asarray(scene["start_state"]),
        "eps": eps,
        "viz": viz,
        "train_ids": np.zeros(0, dtype=np.int64),
    }
    return db, recipe, scene


def test_round_promotion_requires_complete_gamma_and_kb_streams() -> None:
    with _validator_modules() as (validator, scene_module):
        db, recipe, scene = _valid_round(scene_module)
        validator._validate_viz_round(db, recipe, scene, np.zeros(0, np.int8), "arm", 1)

        missing = copy.deepcopy(db)
        missing["eps"].pop()
        with pytest.raises(RuntimeError, match="exact gamma sweep"):
            validator._validate_viz_round(
                missing, recipe, scene, np.zeros(0, np.int8), "arm", 1
            )

        wrong_budget = copy.deepcopy(db)
        wrong_budget["viz"][0]["drawn"] = list(range(7))
        with pytest.raises(RuntimeError, match="K/B semantics"):
            validator._validate_viz_round(
                wrong_budget, recipe, scene, np.zeros(0, np.int8), "arm", 1
            )


def test_completion_source_commit_is_bound() -> None:
    with _validator_modules() as (validator, _):
        recipe = {
            "scene": {"sha256": "s" * 64},
            "source_checkpoint_sha256": "c" * 64,
            "source_git_commit": "a" * 40,
        }
        complete = {
            "scene_sha256": "s" * 64,
            "checkpoint_sha256": "c" * 64,
            "source_git_commit": "b" * 40,
        }
        with pytest.raises(RuntimeError, match="source commit"):
            validator._validate_complete_identity(complete, recipe, "arm")


def test_pair_validator_accepts_only_locked_continuous_ess_artifact() -> None:
    with _validator_modules() as (validator, _):
        spans = np.linspace(0.02, 0.20, 31, dtype=np.float64)
        pools = np.stack([np.linspace(0.0, span, 64) for span in spans])
        solution = validator.BC.solve_beta(pools)
        contract = {
            "name": "test_contract",
            "checkpoint_file_sha256": "a" * 64,
            "checkpoint_model_state_sha256": "b" * 64,
        }
        contract_sha = validator._canonical_json_sha256(contract)
        calibration = {
            "status": validator.BC.SUCCESS_STATUS,
            "chosen": solution["beta"],
            "ess_target": validator.BC.ESS_TARGET,
            "ess_tolerance": validator.BC.ESS_TOLERANCE,
            "solver": validator.BC.SOLVER,
            "acquisition": validator.BC.ACQUISITION,
            "pool_weighting": validator.BC.POOL_WEIGHTING,
            "solution": solution,
            "failure_reason": None,
            "n_pools": len(pools),
            "sigma_pool_sha256": validator.BC.sigma_pool_sha256(pools),
            "checkpoint_sha256": "a" * 64,
            "checkpoint_model_sha256": "b" * 64,
            "checkpoint_contract": contract,
            "checkpoint_contract_sha256": contract_sha,
            "scene_sha256": "c" * 64,
            "source_git_commit": "d" * 40,
            "lam": 1.0,
            "K": 64,
            "B": 8,
            "seed": 20260716,
        }
        recipe = {
            "source_checkpoint_sha256": calibration["checkpoint_sha256"],
            "source_checkpoint_model_sha256": calibration["checkpoint_model_sha256"],
            "source_checkpoint_contract": contract,
            "source_checkpoint_contract_sha256": contract_sha,
            "source_git_commit": calibration["source_git_commit"],
            "scene": {"sha256": calibration["scene_sha256"]},
            "lam": calibration["lam"],
            "K": calibration["K"],
            "B": calibration["B"],
            "seed": calibration["seed"],
            "beta": calibration["chosen"],
            "beta_calibration": calibration,
            "beta_calibration_sha256": "e" * 64,
        }

        validator._validate_checkpoint_contract(recipe)
        validator._validate_beta_calibration(recipe)
        calibration["solution"]["achieved"]["ess_med"] = 0.5
        with pytest.raises(RuntimeError, match="continuous-ESS"):
            validator._validate_beta_calibration(recipe)


def test_corrupt_report_cannot_be_promoted(tmp_path: Path) -> None:
    with _validator_modules() as (validator, _):
        corrupt = tmp_path / "report.png"
        corrupt.write_bytes(b"not-a-png")
        with pytest.raises(RuntimeError, match="decodable image"):
            validator._validate_png(corrupt)


def test_launcher_rejects_missing_args_and_contaminated_output(tmp_path: Path) -> None:
    launcher = _ROOT / "run_afe2_pair.sh"
    missing = subprocess.run([str(launcher)], capture_output=True, text=True)
    assert missing.returncode == 2
    assert "usage:" in missing.stderr

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.txt").write_text("stale")
    contaminated = subprocess.run(
        [str(launcher), "codex_radius1_v1", str(checkpoint), "0" * 64, str(output)],
        capture_output=True,
        text=True,
    )
    assert contaminated.returncode == 2
    assert "new or empty output root" in contaminated.stderr
