from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent), str(ROOT.parent.parent)]

import kazuki_baseline as KB


def test_refinement_cost_is_exact_b1_safemppi_adapter(monkeypatch) -> None:
    expected = np.asarray([1.25, 2.5], dtype=np.float64)
    calls = []

    def exact_cost(state, controls, positions, env):
        calls.append((state, controls, positions, env))
        return expected

    monkeypatch.setattr(KB.EX, "safemppi_plan_costs", exact_cost)
    state = np.zeros(4, dtype=np.float32)
    controls = torch.zeros((2, 10, 2), dtype=torch.float32)
    positions = torch.zeros_like(controls)
    env = object()

    actual = KB.refinement_cost_batch(state, controls, positions, env)

    assert actual.cpu().numpy() == pytest.approx(expected)
    assert len(calls) == 1
    assert calls[0][3] is env


def test_all_three_refinement_rankings_use_one_frozen_scorer(monkeypatch) -> None:
    calls = []

    def scorer(state, controls, positions, env):
        calls.append(tuple(controls.shape))
        return torch.arange(len(controls), dtype=controls.dtype, device=controls.device)

    monkeypatch.setattr(KB, "refinement_cost_batch", scorer)
    monkeypatch.setattr(KB, "N_ELITE", 2)
    monkeypatch.setattr(KB, "N_COPY", 3)
    policy = SimpleNamespace(u_max=1.0)
    state = np.zeros(4, dtype=np.float32)
    controls = torch.zeros((4, 10, 2), dtype=torch.float32)
    goal = torch.ones(2)
    obstacles = torch.zeros((1, 2))
    radii = torch.ones(1)

    result = KB.flow_mppi_refine(
        policy, state, goal, obstacles, radii, 0.1, controls, None,
        object(), "cpu",
    )

    assert result.shape == (10, 2)
    assert calls == [(4, 10, 2), (6, 10, 2), (2, 10, 2)]
