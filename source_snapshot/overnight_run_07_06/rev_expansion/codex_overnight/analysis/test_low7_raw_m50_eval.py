from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_results"))

import low7_raw_m50_eval as EV


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_run(
    root: Path,
    rounds: int = 23,
    artifact_profile: str = "full",
    compact_checkpoint_every: int = 10,
) -> Path:
    run = root / "run"
    run.mkdir(parents=True)
    recipe = {
        "algorithm": "afe_rbf_low7_signed_execution_sweep_v1",
        "arm": "afe",
        "single_arm": True,
        "rounds": rounds,
        "T": EV.T,
        "nfe": EV.NFE,
        "reach": EV.REACH,
        "gammas": list(EV.GAMMAS),
        "source_git_commit": "c" * 40,
        "scene": {
            "sha256": "a" * 64,
            "profile": {"name": "low7_radius1_canonical_v1"},
        },
        "source_checkpoint_sha256": "b" * 64,
        "source_checkpoint_model_sha256": "d" * 64,
        "source_checkpoint_contract_sha256": "e" * 64,
        "no_curriculum": True,
        "no_anchor": True,
        "no_prox": True,
        "no_fallback": True,
        "artifact_profile": artifact_profile,
        "compact_checkpoint_every": compact_checkpoint_every,
    }
    (run / "recipe.json").write_text(json.dumps(recipe))
    (run / "probe.jsonl").write_text("{}\n")
    required = EV.expected_inventory(
        rounds, artifact_profile, compact_checkpoint_every
    )
    for relative in required - {"recipe.json", "probe.jsonl"}:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    inventory = {relative: _sha(run / relative) for relative in required}
    complete = {
        "status": "COMPLETE",
        "algorithm": recipe["algorithm"],
        "completed_round": rounds,
        "scene_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_model_sha256": "d" * 64,
        "checkpoint_contract_sha256": "e" * 64,
        "source_git_commit": "c" * 40,
        "artifact_sha256": inventory,
    }
    (run / "COMPLETE.json").write_text(json.dumps(complete))
    return run


def _rate(count: int, n: int) -> dict:
    return {
        "count": count,
        "n": n,
        "estimate": count / n,
        "wilson95": list(EV.wilson95(count, n)),
    }


def _metric_row(
    round_i: int,
    gamma: float | None,
    n: int,
    eval_profile: str = EV.DEFAULT_EVAL_PROFILE,
) -> dict:
    profile = EV.resolve_evaluation_profile(eval_profile)
    return {
        "metric_version": profile.metric_version,
        "caption": profile.caption,
        "evaluation_profile": profile.name,
        "mode": "raw",
        "method": "afe_rbf",
        "algorithm": "afe_rbf_low7_signed_execution_sweep_v1",
        "round": round_i,
        "scope": "pooled" if gamma is None else "gamma",
        "gamma": gamma,
        "M_per_gamma": profile.m,
        "n": n,
        "binary": {
            "SR": _rate(0, n),
            "CR": _rate(0, n),
            "timeout": _rate(n, n),
            "V_safe": _rate(0, n),
            "V_full": _rate(0, n),
        },
        "minimum_clearance": {
            "n": n,
            "mean": 0.1,
            "bootstrap95": [0.1, 0.1],
            "values": [0.1] * n,
        },
        "successful_time_to_goal": {
            "n": 0,
            "mean": None,
            "bootstrap95": [None, None],
            "values": [],
        },
        "route_modes": {
            "early_first_10_steps": {
                "balance": 0.0,
            },
            "closest_obstacle_approach": {
                "balance": 0.0,
            },
            "closest_obstacle_approach_success_only": {
                "coverage_weighted_balance": 0.0,
            },
        },
    }


def _ranking_row(
    round_i: int, sr: float, cr: float, timeout: float, clearance: float
) -> dict:
    return {
        "mode": "raw",
        "scope": "pooled",
        "round": round_i,
        "gamma": None,
        "binary": {
            "SR": {"estimate": sr},
            "CR": {"estimate": cr},
            "timeout": {"estimate": timeout},
        },
        "minimum_clearance": {"mean": clearance},
    }


def test_variable_round_schedule_has_r0_every_ten_and_final():
    assert EV.evaluation_rounds(1) == (0, 1)
    assert EV.evaluation_rounds(23) == (0, 10, 20, 23)
    assert EV.evaluation_rounds(40) == (0, 10, 20, 30, 40)
    with pytest.raises(ValueError, match="at least one"):
        EV.evaluation_rounds(0)


def test_v2_smoke_profile_uses_m10_and_every_stored_round():
    profile = EV.resolve_evaluation_profile(EV.V2_SMOKE_EVAL_PROFILE)
    assert profile.m == 10
    assert profile.caption == (
        "stored checkpoints re-evaluated on the same raw M=10/gamma seed bank"
    )
    assert EV.evaluation_rounds(10, profile) == tuple(range(11))
    bank, metadata = EV.build_noise_bank(
        "low7_radius1_canonical_v1", 2, profile
    )
    canonical, _ = EV.build_noise_bank("low7_radius1_canonical_v1", 2)
    assert bank.shape == (len(EV.GAMMAS), 10, EV.T, 2)
    assert np.array_equal(bank, canonical[:, :10])
    assert metadata["evaluation_profile"] == profile.name
    assert "canonical M=50" in metadata["cross_profile_pairing"]


def test_completed_rbf_validator_authenticates_all_but_selects_schedule(tmp_path: Path):
    run = _fake_run(tmp_path)
    contract = EV.validate_completed_run(run, "low7_radius1_canonical_v1")
    assert contract["evaluation_rounds"] == [0, 10, 20, 23]
    assert sorted(contract["selected_checkpoints"]) == [0, 10, 20, 23]
    assert contract["authenticated_artifact_count"] == len(EV.expected_inventory(23))
    assert contract["final_checkpoint_alias"]["path"].endswith("final.pt")


def test_v2_smoke_validator_selects_every_authenticated_checkpoint(tmp_path: Path):
    run = _fake_run(tmp_path, rounds=10)
    contract = EV.validate_completed_run(
        run,
        "low7_radius1_canonical_v1",
        EV.V2_SMOKE_EVAL_PROFILE,
    )
    assert contract["evaluation_profile"] == EV.V2_SMOKE_EVAL_PROFILE
    assert contract["evaluation_rounds"] == list(range(11))
    assert sorted(contract["selected_checkpoints"]) == list(range(11))


def test_compact_sweep_inventory_keeps_video_rounds_and_omits_dstore(tmp_path: Path):
    expected = EV.expected_inventory(23, "sweep_compact")
    assert "dstore.pt" not in expected
    assert "viz_db/round1.pt" in expected
    assert "viz_db/round10.pt" in expected
    assert "viz_db/round20.pt" in expected
    assert "viz_db/round11.pt" not in expected
    assert "ckpt_0.pt" in expected
    assert "ckpt_10.pt" in expected
    assert "ckpt_20.pt" in expected
    assert "ckpt_23.pt" in expected
    assert "ckpt_11.pt" not in expected
    run = _fake_run(tmp_path, artifact_profile="sweep_compact")
    contract = EV.validate_completed_run(run, "low7_radius1_canonical_v1")
    assert contract["authenticated_artifact_count"] == len(expected)


def test_compact_v2_inventory_can_store_every_checkpoint_without_dstore(
    tmp_path: Path,
):
    expected = EV.expected_inventory(10, "sweep_compact", compact_checkpoint_every=1)
    assert "dstore.pt" not in expected
    assert {f"ckpt_{round_i}.pt" for round_i in range(11)} <= expected
    run = _fake_run(
        tmp_path,
        rounds=10,
        artifact_profile="sweep_compact",
        compact_checkpoint_every=1,
    )
    contract = EV.validate_completed_run(
        run,
        "low7_radius1_canonical_v1",
        EV.V2_SMOKE_EVAL_PROFILE,
    )
    assert contract["evaluation_rounds"] == list(range(11))


def test_missing_final_round_checkpoint_never_substitutes_final_pt(tmp_path: Path):
    run = _fake_run(tmp_path)
    (run / "ckpt_23.pt").unlink()
    assert (run / "final.pt").is_file()
    with pytest.raises(FileNotFoundError, match="inventoried RBF artifact is missing"):
        EV.validate_completed_run(run, "low7_radius1_canonical_v1")


def test_noise_pairing_has_no_arm_or_round_key_and_is_reproducible():
    parameters = inspect.signature(EV.paired_seed).parameters
    assert "arm" not in parameters
    assert "round_i" not in parameters
    first = EV.paired_seed("low7_radius1_canonical_v1", 0.3, 7)
    assert first == EV.paired_seed("low7_radius1_canonical_v1", 0.3, 7)
    assert first == EV.paired_seed(
        "low7_radius1_canonical_v1", 0.3, 7, EV.V2_SMOKE_EVAL_PROFILE
    )
    assert first != EV.paired_seed("low7_radius1_canonical_v1", 0.3, 8)
    bank_a, metadata_a = EV.build_noise_bank("low7_radius1_canonical_v1", 2)
    bank_b, metadata_b = EV.build_noise_bank("low7_radius1_canonical_v1", 2)
    assert np.array_equal(bank_a, bank_b)
    assert metadata_a == metadata_b
    assert metadata_a["independence"].endswith("arm and checkpoint round")


def test_post_hoc_ranking_obeys_declared_lexicographic_order():
    rows = [
        _ranking_row(0, 0.50, 0.00, 0.50, 1.0),
        _ranking_row(10, 0.60, 0.30, 0.10, 1.0),
        _ranking_row(20, 0.60, 0.10, 0.30, 1.0),
        _ranking_row(30, 0.60, 0.10, 0.20, 0.0),
        _ranking_row(40, 0.60, 0.10, 0.20, 0.1),
        _ranking_row(50, 0.60, 0.10, 0.20, 0.1),
    ]
    best, ranking = EV.select_best_round(rows)
    assert best == 40
    assert [entry["round"] for entry in ranking[:2]] == [40, 50]


def test_raw_metric_grid_has_only_m50_gamma_and_pooled_rows():
    rounds = EV.evaluation_rounds(23)
    rows = []
    for round_i in rounds:
        rows.extend(_metric_row(round_i, gamma, EV.M) for gamma in EV.GAMMAS)
        rows.append(_metric_row(round_i, None, EV.M * len(EV.GAMMAS)))
    EV._authenticate_metric_grid(rows, rounds)
    assert len(rows) == len(rounds) * (len(EV.GAMMAS) + 1)
    rows[0]["mode"] = "verified"
    with pytest.raises(RuntimeError, match="non-raw"):
        EV._authenticate_metric_grid(rows, rounds)


def test_v2_smoke_metric_grid_authenticates_m10_every_round():
    profile = EV.resolve_evaluation_profile(EV.V2_SMOKE_EVAL_PROFILE)
    rounds = EV.evaluation_rounds(3, profile)
    rows = []
    for round_i in rounds:
        rows.extend(
            _metric_row(round_i, gamma, profile.m, profile.name)
            for gamma in EV.GAMMAS
        )
        rows.append(
            _metric_row(
                round_i, None, profile.m * len(EV.GAMMAS), profile.name
            )
        )
    EV._authenticate_metric_grid(rows, rounds, eval_profile=profile)
    assert len(rows) == 4 * (len(EV.GAMMAS) + 1)


def test_aggregate_reports_wilson_and_bootstrap_cis_from_raw_rows():
    raw = []
    for index in range(EV.M):
        success = index < 20
        cr = 20 <= index < 35
        timeout = index >= 35
        raw.append(
            {
                "success": success,
                "cr": cr,
                "timeout": timeout,
                "minimum_clearance": -0.1 + index / 100,
                "time_to_goal": 3.0 + index / 10 if success else None,
                "v_safe": index % 2 == 0,
                "v_full": index % 5 == 0,
                "route_mode_early": 1 if index % 2 == 0 else -1,
                "route_mode_closest": 1 if index % 2 == 0 else -1,
            }
        )
    row = EV.aggregate_metrics(
        raw,
        round_i=10,
        gamma=0.3,
        scope="gamma",
        scene_profile="low7_radius1_canonical_v1",
        algorithm="afe_rbf_low7_signed_execution_sweep_v1",
    )
    assert row["caption"] == EV.REPORT_CAPTION
    assert row["binary"]["SR"]["count"] == 20
    assert row["binary"]["CR"]["count"] == 15
    assert row["binary"]["timeout"]["count"] == 15
    assert row["binary"]["V_safe"]["count"] == 25
    assert row["binary"]["V_full"]["count"] == 10
    assert len(row["binary"]["SR"]["wilson95"]) == 2
    assert len(row["minimum_clearance"]["bootstrap95"]) == 2
    assert row["successful_time_to_goal"]["n"] == 20


def test_gallery_draws_small_state_dots():
    fig, ax = plt.subplots()
    profile = SimpleNamespace(start=(0.3, 0.3), goal=(4.7, 4.7))
    env = SimpleNamespace(obstacles=torch.empty((0, 3)), r_robot=0.1)
    path = np.stack((np.linspace(0.3, 4.0, 21), np.linspace(0.3, 4.0, 21)), axis=1)
    EV._draw_scene(
        ax,
        profile,
        env,
        [path] * len(EV.GALLERY_INDICES),
        0.3,
        "test",
        ["SR"] * len(EV.GALLERY_INDICES),
    )
    dot_lines = [line for line in ax.lines if line.get_marker() == "."]
    assert len(dot_lines) == len(EV.GALLERY_INDICES)
    assert all(line.get_markersize() == pytest.approx(1.3) for line in dot_lines)
    plt.close(fig)


def test_true_eval_curve_renders_validity_with_other_metrics(tmp_path: Path):
    rounds = (0, 10)
    rows = []
    for round_i in rounds:
        rows.extend(_metric_row(round_i, gamma, EV.M) for gamma in EV.GAMMAS)
        rows.append(_metric_row(round_i, None, EV.M * len(EV.GAMMAS)))
    outputs = EV._render_curves(tmp_path, rows, rounds, best_round=0)
    assert [path.suffix for path in outputs] == [".png", ".pdf"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_render_existing_evaluation_is_additive_and_writes_report_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    evaluation = tmp_path / "evaluation"
    presentation = tmp_path / "presentation"
    evaluation.mkdir()
    rounds = (0, 1)
    rows = []
    for round_i in rounds:
        rows.extend(
            _metric_row(
                round_i,
                gamma,
                EV.resolve_evaluation_profile(EV.V2_SMOKE_EVAL_PROFILE).m,
                EV.V2_SMOKE_EVAL_PROFILE,
            )
            for gamma in EV.GAMMAS
        )
        rows.append(
            _metric_row(
                round_i,
                None,
                EV.resolve_evaluation_profile(EV.V2_SMOKE_EVAL_PROFILE).m
                * len(EV.GAMMAS),
                EV.V2_SMOKE_EVAL_PROFILE,
            )
        )
    (evaluation / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (evaluation / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "evaluation_profile": EV.V2_SMOKE_EVAL_PROFILE,
                "rounds": list(rounds),
            }
        )
    )
    manifest = evaluation / "EVALUATION_COMPLETE.json"
    manifest.write_text(json.dumps({"status": "AUTHENTICATED_TEST_DELIVERY"}))
    before = {path.name: _sha(path) for path in evaluation.iterdir()}
    monkeypatch.setattr(
        EV,
        "validate_output",
        lambda root: {"status": "AUTHENTICATED_TEST_DELIVERY"},
    )
    monkeypatch.setattr(
        EV,
        "git_state",
        lambda: {
            "commit": "f" * 40,
            "parent": "e" * 40,
            "tracked_dirty": False,
            "untracked_runtime_sources": [],
        },
    )
    payload = EV.render_existing_evaluation(evaluation, presentation)
    assert payload["post_hoc_best_round"] == 0
    assert (presentation / "report.png").stat().st_size > 0
    assert (presentation / "report.pdf").stat().st_size > 0
    assert (presentation / "PRESENTATION_COMPLETE.json").is_file()
    assert before == {path.name: _sha(path) for path in evaluation.iterdir()}


def test_v2_smoke_renders_a_fixed_index_gallery_for_every_round(tmp_path: Path):
    profile = EV.resolve_evaluation_profile(EV.V2_SMOKE_EVAL_PROFILE)
    rounds = (0, 1)
    rows = []
    cell_dir = tmp_path / "cells" / "raw" / "afe_rbf"
    cell_dir.mkdir(parents=True)
    path = np.stack(
        (np.linspace(0.3, 4.0, 8), np.linspace(0.3, 4.0, 8)), axis=1
    )
    for round_i in rounds:
        for gamma in EV.GAMMAS:
            rows.append(_metric_row(round_i, gamma, profile.m, profile.name))
            paths = np.empty(profile.m, dtype=object)
            for index in range(profile.m):
                paths[index] = path
            np.savez_compressed(
                cell_dir / f"r{round_i:03d}_g{gamma:.1f}.npz",
                paths=paths,
                status=np.asarray(["timeout"] * profile.m),
                outcome=np.asarray(["timeout"] * profile.m),
                rollout_index=np.arange(profile.m, dtype=np.int32),
            )
        rows.append(
            _metric_row(
                round_i, None, profile.m * len(EV.GAMMAS), profile.name
            )
        )
    scene = SimpleNamespace(start=(0.3, 0.3), goal=(4.7, 4.7))
    env = SimpleNamespace(obstacles=torch.empty((0, 3)), r_robot=0.1)
    outputs, manifest_path = EV._render_galleries(
        tmp_path, scene, env, rows, rounds, best_round=0, eval_profile=profile
    )
    assert (tmp_path / "round_galleries" / "raw_m10_round_000.png").is_file()
    assert (tmp_path / "round_galleries" / "raw_m10_round_001.png").is_file()
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["per_round"] == [0, 1]
    assert manifest["M"] == 10
    assert manifest["caption"] == profile.caption


def test_required_report_caption_is_exact():
    assert EV.REPORT_CAPTION == (
        "stored checkpoints re-evaluated on the same raw M=50/gamma seed bank"
    )
    assert EV.M == 50
    assert EV.TEMP == 1.0
    assert EV.NFE == 8
    assert "afe_rbf_low7_signed_execution_sweep_v1" in EV.SUPPORTED_ALGORITHMS


def test_worker_validity_is_preserved_while_raw_outcomes_are_disjoint():
    episode = {
        "status": "reached",
        "route_mode_early": 1,
        "route_mode_closest": -1,
    }
    worker = {
        "status": "reached",
        "success": True,
        "collision": False,
        "oob": False,
        "cr": False,
        "nvp": False,
        "timeout": False,
        "v_safe": True,
        "v_full": False,
        "minimum_clearance": 0.2,
        "steps": 21,
        "time_to_goal": 2.1,
    }
    normalized = EV.normalize_trajectory_metrics(episode, worker, 0.1)
    assert normalized["outcome"] == "SR"
    assert normalized["v_safe"] is True
    assert normalized["v_full"] is False
    assert "nvp" not in normalized
