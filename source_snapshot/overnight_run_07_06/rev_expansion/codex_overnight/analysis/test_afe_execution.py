from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parents[1]
_REV = _HERE.parent
_WORK = _REV.parent
sys.path[:0] = [str(_HERE), str(_REV), str(_WORK)]

import afe_execution as EX


def _patch_nominal_hp(monkeypatch, hp, calls):
    obstacle_token = object()

    def mode1_config():
        return {
            "barrier_activation_radius": 2.75,
            "polytope_nbase": 24,
            "predict_gain": 0.4,
        }

    def planner_obstacles(env):
        calls["env"] = env
        return obstacle_token

    def polytope_hp(center, obstacles, *, sensing, n_base, predict_gain):
        calls.update(
            center=np.asarray(center),
            obstacles=obstacles,
            sensing=sensing,
            n_base=n_base,
            predict_gain=predict_gain,
        )
        return hp, (None, None, None)

    monkeypatch.setattr(EX.GS, "mode1_config", mode1_config)
    monkeypatch.setattr(EX.GS, "planner_obstacles", planner_obstacles)
    monkeypatch.setattr(EX.GF, "polytope_HP", polytope_hp)
    return obstacle_token


def _inputs(count=3):
    state = np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    controls = np.zeros((count, 2, 2), dtype=np.float32)
    env = SimpleNamespace(dt=0.1, goal=np.asarray((1.0, 0.0)))
    return state, controls, env


def test_exact_nominal_builder_and_deterministic_selectors(monkeypatch) -> None:
    calls = {}
    obstacle_token = _patch_nominal_hp(
        monkeypatch,
        lambda points: 1.0 - 0.5 * np.asarray(points)[:, 0],
        calls,
    )
    state, controls, env = _inputs()
    segments = np.asarray([
        [[0.2, 0.0]],
        [[0.4, 0.0]],
        [[0.4, 0.0]],
    ])
    results = [{"y": 1}, {"y": 1}, {"y": 1}]
    candidate_ids = [7, 3, 9]

    by_progress = EX.nominal_hp_max_step_progress(
        state, controls, results, 0.5, env,
        segments=segments, candidate_ids=candidate_ids,
    )
    by_margin = EX.nominal_hp_max_step_margin(
        state, controls, results, 0.5, env,
        segments=segments, candidate_ids=candidate_ids,
    )

    assert by_progress["chosen"]["candidate_id"] == 3
    assert by_margin["chosen"]["candidate_id"] == 7
    assert calls["env"] is env
    assert calls["obstacles"] is obstacle_token
    assert calls["center"] == pytest.approx((0.0, 0.0))
    assert calls["sensing"] == 2.75
    assert calls["n_base"] == 24
    assert calls["predict_gain"] == 0.4


def test_margin_only_selector_does_not_use_progress_as_a_tie_break(monkeypatch) -> None:
    calls = {}
    _patch_nominal_hp(monkeypatch, lambda points: np.ones(len(points)), calls)
    state, controls, env = _inputs(count=2)
    segments = np.asarray([[[0.8, 0.0]], [[-0.2, 0.0]]])

    selection = EX.nominal_hp_max_step_margin_only(
        state,
        controls,
        [{"y": 1}, {"y": 1}],
        0.5,
        env,
        segments=segments,
        candidate_ids=[9, 3],
    )

    assert selection["per_candidate"][0]["step_progress"] > 0.0
    assert selection["per_candidate"][1]["step_progress"] < 0.0
    assert selection["chosen"]["candidate_id"] == 3


def test_exec_y_gate_preserves_terminal_prefix_and_negative_progress(monkeypatch) -> None:
    calls = {}
    _patch_nominal_hp(
        monkeypatch,
        lambda points: 1.0 - 0.5 * np.abs(np.asarray(points)[:, 0]),
        calls,
    )
    state, controls, env = _inputs()
    segments = np.asarray([
        [[-0.2, 0.0]],  # eligible despite negative one-step progress
        [[0.5, 0.0]],   # certified terminal prefix is execution-eligible
        [[2.0, 0.0]],   # full positive but outside the gamma level set
    ])
    results = [
        {"y": 1, "exec_y": 1},
        {"y": 0, "exec_y": 1},
        {"y": 1, "exec_y": 1},
    ]

    selection = EX.nominal_hp_max_step_progress(
        state, controls, results, 0.5, env, segments=segments
    )

    assert selection["chosen"]["candidate_id"] == 1
    assert selection["per_candidate"][0]["step_progress"] < 0.0
    assert selection["per_candidate"][0]["eligible"] is True
    assert selection["counts"] == {
        "candidates": 3,
        "full_socp_positive": 2,
        "execution_verifier_positive": 3,
        "nominal_hp_eligible": 2,
    }
    assert selection["per_candidate"][1]["full_socp_positive"] is False
    assert selection["per_candidate"][1]["execution_verifier_positive"] is True
    assert selection["per_candidate"][1]["eligible"] is True
    assert selection["per_candidate"][2]["nominal_hp_step_margin"] == pytest.approx(-0.5)


def test_margin_uses_hp_at_current_state_and_accepts_tolerance(monkeypatch) -> None:
    calls = {}
    _patch_nominal_hp(
        monkeypatch,
        lambda points: 2.0 - np.asarray(points)[:, 0],
        calls,
    )
    state, controls, env = _inputs(count=2)
    segments = np.asarray([
        [[1.0 + 0.5 * EX.NOMINAL_HP_TOLERANCE, 0.0]],
        [[1.0 + 2.0 * EX.NOMINAL_HP_TOLERANCE, 0.0]],
    ])

    selection = EX.nominal_hp_max_step_margin(
        state, controls, [{"y": 1}, {"y": 1}], 0.5, env,
        segments=segments,
    )

    assert selection["nominal_hp_at_state"] == 2.0
    assert selection["counts"]["nominal_hp_eligible"] == 1
    assert selection["chosen"]["candidate_id"] == 0


def test_no_eligible_candidate_fails_closed_without_fallback(monkeypatch) -> None:
    calls = {}
    _patch_nominal_hp(
        monkeypatch,
        lambda points: 1.0 - np.asarray(points)[:, 0],
        calls,
    )
    state, controls, env = _inputs(count=2)
    segments = np.asarray([[[0.8, 0.0]], [[0.9, 0.0]]])

    selection = EX.nominal_hp_max_step_progress(
        state, controls, [{"y": 1}, {"y": 0}], 0.1, env,
        segments=segments,
    )

    assert selection["chosen"] is None
    assert selection["failure"] == "no_exec_verified_nominal_hp_step"
    assert selection["counts"]["full_socp_positive"] == 1
    assert selection["counts"]["execution_verifier_positive"] == 1
    assert selection["counts"]["nominal_hp_eligible"] == 0


def test_segments_are_computed_from_candidate_controls(monkeypatch) -> None:
    calls = {}
    _patch_nominal_hp(monkeypatch, lambda points: np.ones(len(points)), calls)
    state, controls, env = _inputs(count=1)
    controls[0, 0, 0] = 1.0

    selection = EX.nominal_hp_max_step_progress(
        state, controls, [{"y": 1}], 0.5, env
    )

    assert selection["chosen"]["step_progress"] == pytest.approx(0.005)


def test_safemppi_cost_matches_frozen_double_integrator_objective(monkeypatch) -> None:
    monkeypatch.setattr(EX.GS, "mode1_config", lambda: {
        "horizon": 2,
        "running_goal_weight": 1.0,
        "terminal_goal_weight": 2.0,
        "control_weight": 3.0,
        "smooth_weight": 4.0,
        "progress_weight": 5.0,
        "soft_clearance_weight": 6.0,
        "safety_margin": 0.0,
        "barrier_extra_margin": 0.0,
    })
    monkeypatch.setattr(
        EX.GS, "planner_obstacles", lambda env: np.empty((0, 3), dtype=np.float32)
    )
    state = np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    controls = np.asarray([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    positions = np.asarray([[[0.1, 0.0], [0.2, 0.0]]], dtype=np.float32)
    env = SimpleNamespace(goal=np.asarray((1.0, 0.0)))

    costs = EX.safemppi_plan_costs(state, controls, positions, env)

    # running 1.45 + effort 3 + smooth 8 + progress -1.5 + terminal 1.28
    assert costs == pytest.approx([12.23])


def test_safemppi_cost_selector_ranks_only_after_existing_gates(monkeypatch) -> None:
    calls = {}
    obstacle_token = _patch_nominal_hp(
        monkeypatch, lambda points: np.ones(len(points)), calls
    )
    original_config = EX.GS.mode1_config
    monkeypatch.setattr(EX.GS, "mode1_config", lambda: {
        **original_config(),
        "horizon": 2,
        "safety_margin": 0.0,
        "barrier_extra_margin": 0.0,
    })
    monkeypatch.setattr(
        EX.GS,
        "planner_obstacles",
        lambda env: np.empty((0, 3), dtype=np.float32),
    )
    state, controls, env = _inputs(count=2)
    segments = np.asarray([
        [[0.2, 0.0], [0.8, 0.0]],
        [[0.1, 0.0], [-0.2, 0.0]],
    ])

    selection = EX.nominal_hp_safemppi_cost(
        state,
        controls,
        [{"y": 1}, {"y": 1}],
        0.5,
        env,
        segments=segments,
        candidate_ids=[8, 3],
    )

    assert obstacle_token is not None
    assert selection["chosen"]["candidate_id"] == 8
    assert selection["per_candidate"][0]["safemppi_cost"] < (
        selection["per_candidate"][1]["safemppi_cost"]
    )
