import numpy as np
import pytest

from paper_results import b1_indexed_video_suite as V


def test_markup_bank_contains_control_moderate_and_paper_horizon_equivalent():
    assert V.MARKUP_CANDIDATES == (1.01, 1.05, 1.09)
    exact = 1.01 ** (79.0 / 9.0)
    assert exact == pytest.approx(1.09127, rel=2.0e-5)
    assert 1.05 ** 9 == pytest.approx(1.551328, rel=1.0e-6)


def test_markup_selection_uses_best_nonperfect_success_rate():
    rows = [
        {"markup": 1.01, "sr": 0.0},
        {"markup": 1.05, "sr": 0.2},
        {"markup": 1.09, "sr": 0.4},
    ]
    assert V.select_kazuki_markup(rows)["markup"] == pytest.approx(1.09)


def test_halfspace_clipping_returns_expected_square():
    normals = np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=float)
    bounds = np.ones(4)
    polygon = V.clip_halfspaces(normals, bounds, box=(-2, 2))
    assert len(polygon) == 4
    assert polygon[:, 0].min() == pytest.approx(-1.0)
    assert polygon[:, 0].max() == pytest.approx(1.0)
    assert polygon[:, 1].min() == pytest.approx(-1.0)
    assert polygon[:, 1].max() == pytest.approx(1.0)


def test_nominal_levels_implement_hp_recurrence():
    nominal = {
        "A": np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=float),
        "b": np.ones(4),
        "margins": np.ones(4),
    }
    polygons = V.nominal_level_polygons(nominal, gamma=0.5, horizon=2)
    assert np.abs(polygons[0]).max() == pytest.approx(0.5)
    assert np.abs(polygons[1]).max() == pytest.approx(0.75)


def test_verifier_levels_implement_beta_recurrence():
    faces = [
        {"a": np.asarray((1, 0)), "m": 1.0, "feasible": True},
        {"a": np.asarray((-1, 0)), "m": 1.0, "feasible": True},
        {"a": np.asarray((0, 1)), "m": 1.0, "feasible": True},
        {"a": np.asarray((0, -1)), "m": 1.0, "feasible": True},
    ]
    polygons = V.verifier_level_polygons(faces, np.zeros(2), gamma=0.5, horizon=2)
    assert np.abs(polygons[0]).max() == pytest.approx(0.5)
    assert np.abs(polygons[1]).max() == pytest.approx(0.75)
