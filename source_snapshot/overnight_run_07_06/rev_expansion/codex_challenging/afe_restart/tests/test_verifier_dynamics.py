from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afe_restart.config import DynamicsConfig, SamplingConfig, VerifierConfig
from afe_restart.dynamics import execute_first_action, planned_positions, rollout_plan, step_state
from afe_restart import verifier as verifier_module


def fake_env(obstacles: np.ndarray | None = None):
    return SimpleNamespace(
        obstacles=np.empty((0, 3), dtype=np.float64) if obstacles is None else obstacles,
        r_robot=0.0,
        goal=np.asarray([4.5, 4.5], dtype=np.float64),
    )


def install_accepting_socp(monkeypatch: pytest.MonkeyPatch, seen: dict) -> None:
    face = SimpleNamespace(
        feasible=True,
        m=1.0,
        kind="artificial",
        a=np.asarray([1.0, 0.0]),
    )

    def certify_window(path, *args, **kwargs):
        seen["path"] = np.asarray(path).copy()
        seen["kwargs"] = dict(kwargs)
        return True, [face], [], 2.5

    def check_certificate(faces, trajectory, alpha, include_start=False):
        seen["centered"] = np.asarray(trajectory).copy()
        seen["alpha"] = np.asarray(alpha).copy()
        seen["include_start"] = include_start
        return True, 0.25, 10

    monkeypatch.setattr(verifier_module.VP, "certify_window", certify_window)
    monkeypatch.setattr(verifier_module.VP, "check_certificate", check_certificate)


def test_exact_double_integrator_transition_and_eleven_state_rollout() -> None:
    state = np.asarray([1.0, 2.0, 0.3, -0.2])
    action = np.asarray([0.4, -0.6])
    expected = np.asarray([1.032, 1.977, 0.34, -0.26])
    np.testing.assert_allclose(step_state(state, action, dt=0.1), expected, atol=1e-12)

    plan = np.repeat(action[None], 10, axis=0)
    states = rollout_plan(state, plan)
    assert states.shape == (11, 4)
    assert planned_positions(state, plan).shape == (11, 2)
    np.testing.assert_allclose(states[1], expected, atol=1e-12)
    np.testing.assert_allclose(execute_first_action(state, plan), states[1], atol=1e-12)


def test_plan_shape_is_exactly_h10() -> None:
    with pytest.raises(ValueError, match="shape"):
        rollout_plan(np.zeros(4), np.zeros((9, 2)))
    with pytest.raises(ValueError, match="shape"):
        rollout_plan(np.zeros(4), np.zeros((11, 2)))


def test_verifier_receives_current_plus_all_ten_predictions(monkeypatch) -> None:
    seen: dict = {}
    install_accepting_socp(monkeypatch, seen)
    state = np.asarray([0.5, 0.5, 0.2, 0.1])
    controls = np.zeros((10, 2))

    result = verifier_module.verify_plan(state, controls, fake_env(), gamma=0.5)

    assert result.positions.shape == (11, 2)
    assert seen["path"].shape == (11, 2)
    np.testing.assert_array_equal(seen["path"][0], state[:2])
    np.testing.assert_array_equal(seen["path"], result.positions)
    np.testing.assert_allclose(seen["centered"][0], np.zeros(2), atol=0.0)
    assert seen["alpha"].shape == (11,)
    assert seen["include_start"] is False
    assert result.safe and result.in_bounds and result.socp_ok


def test_bounds_are_strict_and_socp_is_still_evaluated(monkeypatch) -> None:
    seen: dict = {}
    install_accepting_socp(monkeypatch, seen)
    # The current point is in-bounds, but the very first prediction is x=5.09.
    state = np.asarray([4.99, 2.0, 1.0, 0.0])
    result = verifier_module.verify_plan(state, np.zeros((10, 2)), fake_env(), gamma=0.4)

    assert "path" in seen
    assert result.socp_ok
    assert not result.in_bounds
    assert not result.safe
    assert result.bounds_margin_m < 0.0


def test_negative_progress_does_not_change_the_safety_label(monkeypatch) -> None:
    seen: dict = {}
    install_accepting_socp(monkeypatch, seen)
    state = np.asarray([0.5, 0.5, -0.2, -0.2])
    result = verifier_module.verify_plan(state, np.zeros((10, 2)), fake_env(), gamma=0.7)

    assert result.progress_m < 0.0
    assert result.safe
    assert result.safe == (result.in_bounds and result.socp_ok)


def test_physical_clearance_is_diagnostic_not_an_extra_label(monkeypatch) -> None:
    seen: dict = {}
    install_accepting_socp(monkeypatch, seen)
    obstacle = np.asarray([[0.5, 0.5, 0.2]], dtype=np.float64)
    result = verifier_module.verify_plan(
        np.asarray([0.5, 0.5, 0.0, 0.0]),
        np.zeros((10, 2)),
        fake_env(obstacle),
        gamma=0.3,
    )

    assert result.physical_clearance_m < 0.0
    # This mocked certificate intentionally establishes that clearance is
    # telemetry: the production SOCP is responsible for the actual rejection.
    assert result.safe


def test_result_arrays_cannot_be_mutated(monkeypatch) -> None:
    seen: dict = {}
    install_accepting_socp(monkeypatch, seen)
    result = verifier_module.verify_plan(np.ones(4), np.zeros((10, 2)), fake_env(), gamma=1.0)
    with pytest.raises(ValueError):
        result.positions[0, 0] = 123.0


def test_actual_socp_with_radius_1p2_obstacle_is_deterministic() -> None:
    start = np.asarray([0.5, 0.5], dtype=np.float64)
    obstacle = np.asarray([[2.5, 2.5, 1.2]], dtype=np.float64)
    env = fake_env(obstacle)
    controls = np.zeros((10, 2), dtype=np.float64)
    first = verifier_module.verify_plan(np.r_[start, 0.0, 0.0], controls, env, gamma=0.5)
    second = verifier_module.verify_plan(np.r_[start, 0.0, 0.0], controls, env, gamma=0.5)

    assert first.safe and first.socp_ok
    assert first.safe == second.safe
    assert first.certificate_residual == second.certificate_residual
    assert first.face_margin_m == second.face_margin_m
    np.testing.assert_array_equal(first.positions, second.positions)

    inside_giant = verifier_module.verify_plan(
        np.asarray([2.5, 2.5, 0.0, 0.0]), controls, env, gamma=0.5)
    assert inside_giant.in_bounds
    assert not inside_giant.socp_ok
    assert not inside_giant.safe
    assert inside_giant.physical_clearance_m < 0.0


def test_config_rejects_contract_drift() -> None:
    with pytest.raises(ValueError, match="H=10"):
        DynamicsConfig(horizon=8)
    with pytest.raises(ValueError, match="verifier_budget"):
        SamplingConfig(candidate_count=4, verifier_budget=5)
    with pytest.raises(ValueError, match="maximum_face_margin"):
        VerifierConfig(minimum_face_margin=0.1, maximum_face_margin=0.01)
