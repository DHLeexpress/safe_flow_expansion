from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_results"))

import afe_context as CX
import afe_core as AC
from afe2_scene_profiles import build_scene, get_scene_profile, scene_snapshot
import grid_expand_afe2 as AFE2
import grid_expand_afe_ensemble as ENS
import grid_expand_afe_rbf as RBF
import grid_hp_expt as HP
import afe_m20_eval as EV


def _env():
    return build_scene(get_scene_profile("low7_radius1_canonical_v1"))


def _policy():
    return HP.GridHPFlowPolicy(
        repr_dim=32,
        grid_hw=(32, 32),
        trunk_hidden=(160, 96),
        enc_depth=3,
        raw_condition_dim=7,
        conditioning_schema=CX.LOW7_SCHEMA,
    )


def _cfg(**overrides):
    values = dict(
        conditioning_schema=CX.LOW7_SCHEMA,
        raw_condition_dim=7,
        gammas=(0.1,),
        taskspace_epsilon=1.0e-6,
        K=2,
        B=1,
        T=1,
        reach=0.15,
        seed=910,
        nfe=1,
        temp=1.0,
        s=0.9,
        beta=0.1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gamma_wire_mapping_preserves_declared_keys_and_rejects_collisions() -> None:
    mapping = CX.declared_gamma_storage_map((0.3, 0.4, 0.7))

    assert CX.canonical_declared_gamma(np.float32(0.3), mapping) == 0.3
    assert CX.canonical_declared_gamma(np.float32(0.4), mapping) == 0.4
    assert CX.canonical_declared_gamma(np.float32(0.7), mapping) == 0.7
    with pytest.raises(ValueError, match="not unique"):
        CX.declared_gamma_storage_map((0.3, np.float32(0.3)))


def test_giant_scene_and_boundary_vector_orientation() -> None:
    env = _env()
    snapshot = scene_snapshot(env, get_scene_profile("low7_radius1_canonical_v1"))
    obstacles = np.asarray(snapshot["obstacles"])
    giant = obstacles[np.all(np.isclose(obstacles[:, :2], (2.5, 2.5)), axis=1)]
    assert giant.shape == (1, 3)
    assert giant[0, 2] == pytest.approx(1.0)
    assert snapshot["start_state"] == pytest.approx([0.3, 0.3, 0.0, 0.0])
    assert snapshot["goal"] == pytest.approx([4.7, 4.7])

    left = CX.build_context(
        np.asarray((1.25, 2.5, 0.0, 0.0)), env.goal.numpy(), 0.3, [], env,
        CX.LOW7_SCHEMA,
    ).low5
    right = CX.build_context(
        np.asarray((3.75, 2.5, 0.0, 0.0)), env.goal.numpy(), 0.3, [], env,
        CX.LOW7_SCHEMA,
    ).low5
    assert left.shape == right.shape == (7,)
    assert left[4] > 0.0 and abs(left[5]) < 1.0e-7
    assert right[4] < 0.0 and abs(right[5]) < 1.0e-7
    assert left[-1] == right[-1] == pytest.approx(0.3)


def test_tie_mean_schema_removes_canonical_start_obstacle_order_bias() -> None:
    env = _env()
    state = env.x0.numpy().astype(np.float32)
    legacy = CX.build_context(
        state, env.goal.numpy(), 0.1, [], env, CX.LOW7_SCHEMA
    ).low5
    tie_mean = CX.build_context(
        state, env.goal.numpy(), 0.1, [], env, CX.LOW7_TIE_SCHEMA
    ).low5

    assert legacy[4] != pytest.approx(legacy[5])
    assert tie_mean[4] == pytest.approx(tie_mean[5], abs=1.0e-8)


def test_policy_and_trainability_contract_is_exact_low7() -> None:
    policy = _policy()
    contract = CX.policy_contract(policy)
    assert (contract.raw_condition_dim, contract.ctx_dim, contract.trunk_input_dim) == (
        7, 39, 91
    )
    assert sum(parameter.numel() for parameter in policy.parameters()) == 70_308
    state = ENS.configure_policy_trainability(policy, True)
    assert state["frozen"]
    assert all(name.startswith("enc_grid.") for name in state["frozen"])
    assert any(name.startswith("trunk.") for name in state["trainable"])
    assert any(name.startswith("head.") for name in state["trainable"])


def test_gather_audit_store_replay_and_embedding_share_low7() -> None:
    env = _env()
    cfg = _cfg()
    policy = _policy().eval()
    state = env.x0.numpy().astype(np.float32)
    episode = RBF._episode(state, 0.1, 0, 0, env, cfg)
    grids, conditions, histories = RBF._context_arrays([episode], env, cfg)
    exact = CX.build_context(state, env.goal.numpy(), 0.1, [], env, CX.LOW7_SCHEMA)
    np.testing.assert_array_equal(conditions[0], exact.low5)

    audit = AC.build_audit_contexts(
        env, (0.1,), n_pos=1, conditioning_schema=CX.LOW7_SCHEMA
    )
    assert all(row["low5"].shape == (7,) for row in audit)
    assert all(row["conditioning_schema"] == CX.LOW7_SCHEMA for row in audit)

    store = AC.DStore(conditioning_schema=CX.LOW7_SCHEMA, condition_dim=7)
    sid = store.add_step_ctx(state, grids[0], conditions[0], histories[0], (1, 0, 0))
    store.add_query(
        sid,
        np.zeros((10, 2), np.float32),
        {
            "y": 1, "margin": 1.0, "resid": 0.0, "prog": 0.1, "d0": 1.0,
            "exec_y": 1, "exec_prog": 0.1, "exec_margin": 1.0,
            "terminal_hit": False, "terminal_rescue": False,
            "terminal_tau": None, "terminal_prog": None, "terminal_resid": None,
            "terminal_reason": None, "terminal_reverify": False,
        },
        0.2, 0.1, 1, np.zeros((10, 2), np.float32),
    )
    _, replay_condition, _, _, _ = store.sample_pos(
        4, np.random.default_rng(1)
    )
    assert replay_condition.shape == (4, 7)
    np.testing.assert_array_equal(replay_condition[0].numpy(), exact.low5)
    features = AFE2.embed_queries(policy, store, cfg, "cpu")
    assert features.shape == (1, 32)


def test_endpoint_raw_evaluation_calls_shared_low7_builder(monkeypatch) -> None:
    env = _env()
    cfg = _cfg(T=1, nfe=1, temp=1.0, scene_profile="low7_radius1_canonical_v1")
    policy = _policy().eval()
    calls = []
    original = EV.CX.build_context

    def wrapped(*args, **kwargs):
        calls.append(args[5])
        return original(*args, **kwargs)

    monkeypatch.setattr(EV.CX, "build_context", wrapped)
    episodes = EV.run_raw_batch(policy, env, cfg, "cpu")
    assert len(episodes) == EV.M
    assert calls and set(calls) == {CX.LOW7_SCHEMA}


def test_store_rejects_low5_or_adapter_substitution() -> None:
    store = AC.DStore(conditioning_schema=CX.LOW7_SCHEMA, condition_dim=7)
    with pytest.raises(RuntimeError, match=r"expected \(7,\)"):
        store.add_step_ctx(
            np.zeros(4), np.zeros((3, 32, 32)), np.zeros(5),
            np.zeros((16, 2)), (1, 0, 0),
        )
    policy = _policy()
    policy.conditioning_schema = CX.LOW5_SCHEMA
    with pytest.raises(RuntimeError, match="dimensions"):
        CX.policy_contract(policy)


def test_any_socp_error_is_excluded_from_uncertainty_and_training_memory() -> None:
    assert RBF.query_has_socp_error({"reason": "socp_error", "terminal_reason": None})
    assert RBF.query_has_socp_error({"reason": "unsafe", "terminal_reason": "socp_error"})
    assert not RBF.query_has_socp_error({"reason": "unsafe", "terminal_reason": "unsafe"})
