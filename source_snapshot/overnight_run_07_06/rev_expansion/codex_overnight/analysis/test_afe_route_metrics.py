from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_route_metrics as RM


def test_canonical_diagonal_sign_labels_up_and_right_sides() -> None:
    points = np.asarray([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    signed = RM.signed_cross_track(points, start=(0.0, 0.0), goal=(4.0, 4.0))
    labels = RM.classify_cross_track(signed, ambiguity_band=0.05)

    assert signed.tolist() == pytest.approx([1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0])
    assert labels.tolist() == [RM.MODE_U, RM.MODE_R, RM.MODE_AMBIGUOUS]


def test_generic_oriented_line_and_fixed_ambiguity_band() -> None:
    labels = RM.classify_plan_endpoints(
        [[1.0, 0.11], [1.0, -0.11], [1.0, 0.09]],
        start=(0.0, 0.0),
        goal=(2.0, 0.0),
        ambiguity_band=0.1,
    )

    assert labels.tolist() == [RM.MODE_U, RM.MODE_R, RM.MODE_AMBIGUOUS]


def test_summary_excludes_ambiguous_from_resolved_fractions() -> None:
    summary = RM.summarize_modes([RM.MODE_U, RM.MODE_U, RM.MODE_R, 0])

    assert summary["total_count"] == 4
    assert summary["resolved_count"] == 3
    assert summary["u_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["r_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["balance"] == pytest.approx(2.0 / 3.0)
    assert summary["resolved_fraction"] == pytest.approx(0.75)
    assert summary["coverage_weighted_balance"] == pytest.approx(0.5)
    assert summary["binary_entropy"] == pytest.approx(0.9182958340544896)
    assert summary["ambiguous_fraction"] == pytest.approx(0.25)


def test_all_ambiguous_summary_is_finite_and_explicit() -> None:
    summary = RM.summarize_modes([0, 0])

    assert summary["resolved_count"] == 0
    assert summary["u_fraction"] == 0.0
    assert summary["r_fraction"] == 0.0
    assert summary["balance"] == 0.0
    assert summary["resolved_fraction"] == 0.0
    assert summary["coverage_weighted_balance"] == 0.0
    assert summary["binary_entropy"] == 0.0
    assert summary["ambiguous_fraction"] == 1.0


def test_closest_approach_supports_multiple_obstacles_and_batch_shape() -> None:
    trajectories = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.8], [2.0, 2.0]],
            [[0.0, 0.0], [1.0, -0.8], [3.9, 4.0]],
        ]
    )
    labels, closest = RM.classify_trajectories_at_closest_approach(
        trajectories,
        start=(0.0, 0.0),
        goal=(4.0, 4.0),
        obstacle_centers=[[1.0, 1.0], [4.0, 4.0]],
        obstacle_radii=[0.1, 0.05],
        ambiguity_band=0.05,
    )

    assert closest["points"].shape == (2, 2)
    assert closest["time_index"].tolist() == [1, 2]
    assert closest["obstacle_index"].tolist() == [0, 1]
    assert closest["clearance"].tolist() == pytest.approx([0.1, 0.05])
    assert labels.tolist() == [RM.MODE_R, RM.MODE_U]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: RM.signed_cross_track([[0.0, 0.0]], start=(1.0, 1.0), goal=(1.0, 1.0)),
            "distinct",
        ),
        (lambda: RM.classify_cross_track([0.0], ambiguity_band=-0.1), "nonnegative"),
        (lambda: RM.summarize_modes([2]), "labels"),
    ],
)
def test_invalid_metric_inputs_fail_closed(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
