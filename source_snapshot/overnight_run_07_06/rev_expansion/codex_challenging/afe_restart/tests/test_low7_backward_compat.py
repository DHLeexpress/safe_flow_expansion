from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
import torch

import grid_hp_expt as challenging_hp


def _load_overnight_hp() -> ModuleType:
    """Load the sibling implementation without shadowing ``grid_hp_expt``."""

    path = (
        Path(__file__).resolve().parents[2].parent
        / "codex_overnight"
        / "grid_hp_expt.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_afe_restart_test_codex_overnight_grid_hp_expt", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load sibling HP implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    # The historical module creates its default output directory at import
    # time.  Suppress that unrelated side effect in this contract test.
    with patch.object(os, "makedirs"):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=("challenging", "overnight"))
def hp_module(request: pytest.FixtureRequest) -> ModuleType:
    if request.param == "challenging":
        return challenging_hp
    return _load_overnight_hp()


def _assert_same_state_dict(
    expected: torch.nn.Module, actual: torch.nn.Module
) -> None:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name])


def test_closest_boundary_vector_and_low7_keep_gamma_last() -> None:
    features = challenging_hp.GF
    state = np.asarray((0.0, 0.0, 0.4, -0.2), dtype=np.float64)
    goal = np.asarray((4.0, 3.0), dtype=np.float64)
    obstacles = np.asarray(
        (
            # Closer center, but 0.55 m inflated-boundary clearance.
            (0.7, 0.0, 0.05),
            # Farther center, but the closest boundary at 0.30 m.
            (1.0, 0.0, 0.60),
        ),
        dtype=np.float64,
    )

    boundary = features.closest_boundary_vector(
        state[:2], obstacles, r_robot=0.1
    )
    # Physical clearance 1.0 - 0.60 - 0.10 = 0.30, normalized by
    # SENSING=2.0, in the world-frame direction of the selected center.
    np.testing.assert_allclose(boundary, (0.15, 0.0), atol=1.0e-7)

    condition = features.low7(
        state, goal, 0.7, obstacles, r_robot=0.1
    )
    np.testing.assert_allclose(
        condition,
        (0.8, 0.6, 0.2, -0.1, 0.15, 0.0, 0.7),
        atol=1.0e-7,
    )
    assert condition.shape == (7,)
    assert condition.dtype == np.float32
    assert condition[-1] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("raw_condition_dim", "conditioning_schema", "ctx_dim", "trunk_in"),
    (
        (5, "low5", 37, 89),
        (7, "low7_closest_boundary", 39, 91),
    ),
)
def test_low5_and_low7_context_and_trunk_dimensions(
    hp_module: ModuleType,
    raw_condition_dim: int,
    conditioning_schema: str,
    ctx_dim: int,
    trunk_in: int,
) -> None:
    policy = hp_module.GridHPFlowPolicy(
        repr_dim=32,
        grid_hw=(32, 32),
        trunk_hidden=(160, 96),
        enc_depth=3,
        raw_condition_dim=raw_condition_dim,
        conditioning_schema=conditioning_schema,
    )
    grid = torch.zeros(2, 3, 32, 32)
    condition = torch.arange(
        2 * raw_condition_dim, dtype=torch.float32
    ).reshape(2, raw_condition_dim)
    history = torch.zeros(2, policy.K_hist, 2)

    context = policy.ctx_from(grid, condition, history)
    assert policy.ctx_dim == ctx_dim
    assert context.shape == (2, ctx_dim)
    assert policy.trunk[0].in_features == trunk_in
    torch.testing.assert_close(context[:, :raw_condition_dim], condition)

    config = policy.config()
    assert config["raw_condition_dim"] == raw_condition_dim
    assert config["conditioning_schema"] == conditioning_schema
    assert config["ctx_dim"] == ctx_dim


@pytest.mark.parametrize(
    ("raw_condition_dim", "conditioning_schema"),
    ((5, "low5"), (7, "low7_closest_boundary")),
)
def test_save_load_preserves_conditioning_contract(
    hp_module: ModuleType,
    raw_condition_dim: int,
    conditioning_schema: str,
    tmp_path: Path,
) -> None:
    policy = hp_module.GridHPFlowPolicy(
        repr_dim=8,
        grid_hw=(16, 12),
        trunk_hidden=(12,),
        enc_depth=2,
        raw_condition_dim=raw_condition_dim,
        conditioning_schema=conditioning_schema,
    )
    checkpoint_path = tmp_path / "policy.pt"
    hp_module.save_hp(policy, checkpoint_path, extra={"witness": "round-trip"})

    restored, checkpoint = hp_module.load_hp(checkpoint_path)

    assert checkpoint["witness"] == "round-trip"
    assert restored.raw_condition_dim == raw_condition_dim
    assert restored.conditioning_schema == conditioning_schema
    assert restored.ctx_dim == raw_condition_dim + 32
    assert restored.trunk[0].in_features == 20 + raw_condition_dim + 32 + 32
    assert restored.config()["schema_version"] == (
        "w8sg-hp-v3-low7-closest-boundary"
        if raw_condition_dim == 7
        else "w8sg-hp-v2-low5-only"
    )
    _assert_same_state_dict(policy, restored)


def test_old_checkpoint_without_new_config_fields_defaults_to_low5(
    hp_module: ModuleType, tmp_path: Path
) -> None:
    policy = hp_module.GridHPFlowPolicy(
        repr_dim=8,
        grid_hw=(16, 12),
        trunk_hidden=(12,),
        enc_depth=2,
    )
    old_config = dict(policy.config())
    # The codex_overnight checkpoints produced before this change omitted the
    # first three fields as well as the two new conditioning declarations.
    for field in (
        "schema_version",
        "raw_start_goal",
        "use_gru",
        "raw_condition_dim",
        "conditioning_schema",
    ):
        old_config.pop(field, None)
    checkpoint_path = tmp_path / "old_low5_policy.pt"
    torch.save(
        {"state_dict": policy.state_dict(), "config": old_config},
        checkpoint_path,
    )

    restored, _ = hp_module.load_hp(checkpoint_path)

    assert restored.raw_condition_dim == 5
    assert restored.conditioning_schema == "low5"
    assert restored.ctx_dim == 37
    assert restored.trunk[0].in_features == 89
    assert restored.config()["schema_version"] == "w8sg-hp-v2-low5-only"
    _assert_same_state_dict(policy, restored)


def test_low7_boundary_columns_reach_the_velocity_field() -> None:
    torch.manual_seed(9)
    policy = challenging_hp.GridHPFlowPolicy(
        repr_dim=8,
        grid_hw=(16, 12),
        trunk_hidden=(12,),
        enc_depth=2,
        raw_condition_dim=7,
        conditioning_schema="low7_closest_boundary",
    )
    grid = torch.zeros(3, 3, 16, 12)
    low7 = torch.randn(3, 7, requires_grad=True)
    history = torch.zeros(3, policy.K_hist, 2)
    context = policy.ctx_from(grid, low7, history)
    x = torch.randn(3, policy.d)
    tau = torch.full((3,), 0.5)

    policy(x, tau, context).square().mean().backward()

    assert low7.grad is not None
    assert bool((low7.grad[:, 4:6].abs().sum(dim=0) > 0).all())
