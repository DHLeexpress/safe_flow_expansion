from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_results"))

import low7_raw_m50_eval as RAW
import low7_support_sweep_eval as EV


def rate(value):
    return {"estimate": value, "count": int(value * 70), "n": 70, "wilson95": [0, 1]}


def metric(round_i, gamma, *, sr, cr, timeout, clearance, c):
    return {
        "round": round_i,
        "scope": "pooled" if gamma is None else "gamma",
        "gamma": gamma,
        "binary": {
            "SR": rate(sr), "CR": rate(cr), "timeout": rate(timeout),
            "V_safe": rate(0.5), "V_full": rate(0.4),
        },
        "minimum_clearance": {"mean": clearance},
        "successful_route_coverage": {"C": c},
    }


def round_rows(round_i, *, sr, cr, timeout, clearance, c):
    rows = [
        metric(round_i, gamma, sr=sr, cr=cr, timeout=timeout, clearance=clearance, c=c)
        for gamma in RAW.GAMMAS
    ]
    rows.append(metric(
        round_i, None, sr=sr, cr=cr, timeout=timeout, clearance=clearance, c=c
    ))
    return rows


def test_screen_bank_exactly_reuses_reviewed_m10_crn():
    support, support_meta = RAW.build_noise_bank(
        EV.SCENE, 20, EV.SCREEN_PROFILE
    )
    reviewed, reviewed_meta = RAW.build_noise_bank(
        EV.SCENE, 20, RAW.V2_SMOKE_EVAL_PROFILE
    )
    assert np.array_equal(support, reviewed)
    assert support_meta["sha256"] == reviewed_meta["sha256"]


def test_m50_holdout_is_not_the_m10_prefix():
    holdout, metadata = EV.holdout_noise_bank(EV.SCENE, 20)
    screen, _ = RAW.build_noise_bank(EV.SCENE, 20, EV.SCREEN_PROFILE)
    assert holdout.shape == (7, 50, 300, 20)
    assert not np.array_equal(holdout[:, :10], screen)
    assert metadata["disjoint_from_screen_prefix"] is True


def test_b1_screen_and_holdout_have_the_declared_sizes_and_disjoint_banks():
    screen, screen_meta = RAW.build_noise_bank(
        EV.SCENE, 20, EV.B1_SCREEN_PROFILE
    )
    holdout, holdout_meta = EV.holdout_noise_bank(
        EV.SCENE, 20, profile=EV.B1_HOLDOUT_PROFILE, study="b1"
    )
    assert screen.shape == (7, 10, 300, 20)
    assert holdout.shape == (7, 50, 300, 20)
    assert holdout_meta["screen_sha256"] == screen_meta["sha256"]
    assert not np.array_equal(holdout[:, :10], screen)
    assert EV.B1_SCREEN_PROFILE.checkpoint_stride == 1


def test_successful_route_coverage_uses_all_attempts_as_denominator():
    rows = [
        {"success": True, "route_mode_closest": 1},
        {"success": True, "route_mode_closest": 1},
        {"success": True, "route_mode_closest": -1},
        {"success": True, "route_mode_closest": 0},
        *({"success": False, "route_mode_closest": -1} for _ in range(6)),
    ]
    result = EV.successful_route_coverage(rows, 10)
    assert result["n_success_U"] == 2
    assert result["n_success_R"] == 1
    assert result["n_success_ambiguous"] == 1
    assert result["C"] == 0.2


def test_selection_prioritizes_J_then_declared_ties():
    rows = []
    rows += round_rows(5, sr=0.9, cr=0.05, timeout=0.05, clearance=0.2, c=0.2)
    rows += round_rows(10, sr=0.7, cr=0.1, timeout=0.2, clearance=0.3, c=0.4)
    rows += round_rows(15, sr=0.8, cr=0.1, timeout=0.1, clearance=0.1, c=0.4)
    best, ranking = EV.select_round(rows, (5, 10, 15))
    assert best == 15
    assert [row["round"] for row in ranking] == [15, 10, 5]


def test_coarse_schedule_and_local_window_are_exact():
    assert EV.COARSE_ROUNDS == (0, *range(5, 101, 5))
    coarse_best = 55
    local = [
        value for value in range(max(0, coarse_best - 2), min(100, coarse_best + 2) + 1)
        if value not in EV.COARSE_ROUNDS
    ]
    assert local == [53, 54, 56, 57]


def test_pareto_frontier_keeps_only_nondominated_SR_J_points():
    scores = [
        {"round": 1, "SR": 0.5, "J": 0.3},
        {"round": 2, "SR": 0.6, "J": 0.2},
        {"round": 3, "SR": 0.7, "J": 0.4},
        {"round": 4, "SR": 0.65, "J": 0.5},
    ]
    assert [row["round"] for row in EV.pareto_frontier(scores)] == [4, 3]
