from __future__ import annotations

import math

import numpy as np
import pytest

import afe_restart.stage2_planned_demos as stage2
from afe_restart.fallback import SafeMPPIBackup
from afe_restart.stage2_planned_demos import (
    DemoRunConfig,
    make_parser,
    mode_paired_sweep_seed_schedule,
    summarize_smoothness_sweep_cell,
)


def _episode(
    *,
    success: bool,
    mode: str,
    kind: str,
    plan: np.ndarray,
    clearance: float,
    requested_mode: str = "R-first",
) -> dict[str, object]:
    return {
        "gamma": 0.4,
        "seed": 1,
        "requested_mode": requested_mode,
        "success": success,
        "direction_class": mode,
        "dead_reason": None if success else "no_certified_plan",
        "steps": 1,
        "queries": 2,
        "safe_queries": 1,
        "query_acceptance": 0.5,
        "min_clearance_m": clearance,
        "wall_seconds": 0.25,
        "selected_query_indices": np.asarray([0], dtype=np.int64),
        "query_plans": np.asarray([plan], dtype=np.float32),
        "query_kinds": [kind],
    }


def test_noise_variance_multiplier_reaches_safemppi_and_keeps_legacy_default() -> None:
    legacy = SafeMPPIBackup(max_debug_candidates=0)
    low_noise = SafeMPPIBackup(max_debug_candidates=0, noise_var_mult=1.0)

    assert legacy.noise_var_mult == 3.0
    assert np.allclose(legacy.adapter.config.noise_sigma, [0.5 * math.sqrt(3.0)] * 2)
    assert np.allclose(low_noise.adapter.config.noise_sigma, [0.5, 0.5])
    with pytest.raises(ValueError, match="finite and positive"):
        SafeMPPIBackup(noise_var_mult=0.0)


def test_stage2_noise_and_prescribed_sweep_defaults() -> None:
    args = make_parser().parse_args(["sweep"])
    assert args.noise_var_mult == 3.0
    assert tuple(args.sweep_smooth_weights) == (32.0, 64.0, 128.0)
    assert tuple(args.sweep_noise_var_mults) == (1.0, 2.0, 3.0)
    assert DemoRunConfig().noise_var_mult == 3.0
    with pytest.raises(ValueError, match="finite and positive"):
        DemoRunConfig(noise_var_mult=float("nan"))


def test_sweep_freezes_a_mode_paired_per_gamma_seed_schedule(monkeypatch) -> None:
    def fake_order(gamma, config):
        offset = int(round(10 * gamma))
        return [100 + offset, 200 + offset], {
            "matched_hint_count": 2,
            "ordered_requested_modes": ["R-first", "U-first"],
        }

    monkeypatch.setattr(stage2, "candidate_seed_order", fake_order)
    schedule, _ = mode_paired_sweep_seed_schedule(
        (0.1, 0.2), DemoRunConfig(max_candidate_seeds_per_gamma=2), 2
    )

    assert schedule == {
        "0.1": [
            {"seed": 101, "requested_mode": "R-first"},
            {"seed": 201, "requested_mode": "U-first"},
        ],
        "0.2": [
            {"seed": 102, "requested_mode": "R-first"},
            {"seed": 202, "requested_mode": "U-first"},
        ],
    }
    with pytest.raises(ValueError, match="positive even"):
        mode_paired_sweep_seed_schedule((0.1,), DemoRunConfig(), 1)


def test_sweep_metrics_use_only_successful_classified_training_targets() -> None:
    smooth = np.full((10, 2), [0.2, 0.2], dtype=np.float32)
    jagged = np.empty((10, 2), dtype=np.float32)
    jagged[::2] = [1.0, -1.0]
    jagged[1::2] = [-1.0, 1.0]
    failed = np.ones((10, 2), dtype=np.float32)
    episodes = [
        _episode(
            success=True,
            mode="R-first",
            kind="weighted_mean",
            plan=smooth,
            clearance=0.2,
        ),
        _episode(
            success=True,
            mode="U-first",
            kind="debug_candidate",
            plan=jagged,
            clearance=0.1,
            requested_mode="U-first",
        ),
        _episode(
            success=False,
            mode="unclassified",
            kind="debug_candidate",
            plan=failed,
            clearance=-0.1,
        ),
    ]

    metrics = summarize_smoothness_sweep_cell(episodes)

    assert metrics["attempts"] == 3
    assert metrics["successes"] == 2
    assert metrics["requested_R-first_attempts"] == 2
    assert metrics["requested_U-first_attempts"] == 1
    assert metrics["R-first_successes"] == 1
    assert metrics["U-first_successes"] == 1
    assert metrics["fail_closed"] == 1
    assert metrics["training_targets"] == 2
    assert metrics["cost_selected_target_share"] == 0.5
    assert metrics["debug_target_share"] == 0.5
    assert metrics["target_kind_counts"] == {
        "weighted_mean": 1,
        "internal_best": 0,
        "debug_candidate": 1,
    }
    assert metrics["mean_adjacent_action_jump"] == pytest.approx(1.0)
    assert metrics["saturated_action_coordinate_share"] == pytest.approx(0.5)
    assert metrics["mean_min_clearance_m_success"] == pytest.approx(0.15)
    assert metrics["mean_successful_steps"] == 1.0
    assert metrics["mean_time_to_goal_s_success"] == pytest.approx(0.1)
