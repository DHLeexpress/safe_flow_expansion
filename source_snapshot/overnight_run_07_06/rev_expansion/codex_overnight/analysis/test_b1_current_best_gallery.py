from __future__ import annotations

import importlib.util
from pathlib import Path
import inspect

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "paper_results" / "b1_current_best_gallery.py"
SPEC = importlib.util.spec_from_file_location("b1_current_best_gallery", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_low_high_are_predeclared_endpoints_not_metric_selected():
    summaries = {
        0.03: {"balance": 1.0, "route_value_std": 0.4},
        0.1: {"balance": 0.8, "route_value_std": 0.3},
        0.3: {"balance": 0.4, "route_value_std": 0.2},
        0.9: {"balance": 0.0, "route_value_std": 0.1},
    }
    assert MODULE.choose_low_high(summaries) == (0.03, 0.9)


def test_route_summary_counts_both_modes():
    up = np.asarray([[0.3, 0.3], [2.0, 2.4], [4.7, 4.7]])
    right = np.asarray([[0.3, 0.3], [2.4, 2.0], [4.7, 4.7]])
    summary = MODULE.route_summary([up, right])
    assert summary["u_count"] == 1
    assert summary["r_count"] == 1
    assert summary["balance"] == 1.0


def test_failure_classifier_marks_collision_not_success():
    class TensorLike:
        def __init__(self, value):
            import torch
            self.value = torch.as_tensor(value, dtype=torch.float32)

        def detach(self):
            return self.value.detach()

    class Env:
        goal = TensorLike([4.7, 4.7])
        obstacles = TensorLike([[2.5, 2.5, 1.0]])
        r_robot = 0.0

    path = np.asarray([[0.3, 0.3], [2.5, 2.5]])
    assert MODULE.classify_path(path, Env(), reach=0.15) == "CR"


def test_kazuki_adapter_exposes_additive_conditioning_schema():
    import kazuki_baseline

    parameter = inspect.signature(kazuki_baseline.kazuki_deploy).parameters[
        "conditioning_schema"
    ]
    assert parameter.default is None


def test_reused_kazuki_round_trip(tmp_path):
    paths = np.empty(2, dtype=object)
    paths[0] = np.asarray([[0.3, 0.3], [1.0, 1.2]], dtype=np.float32)
    paths[1] = np.asarray([[0.3, 0.3], [1.2, 1.0]], dtype=np.float32)
    np.savez_compressed(
        tmp_path / "kazuki_ws_0.1.npz", paths=paths,
        outcomes=np.asarray(["TO", "CR"]),
    )
    loaded_paths, outcomes = MODULE.load_reused_kazuki(tmp_path, 0.1)
    assert outcomes == ["TO", "CR"]
    assert np.array_equal(loaded_paths[0], paths[0])
