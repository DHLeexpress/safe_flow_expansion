from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parents[1]
_REV = _HERE.parent
_WORK = _REV.parent
_COLLIDING = {
    "_paths", "grid_feats", "grid_metrics", "grid_metrics2", "grid_rollout",
    "grid_scene", "grid_hp_expt", "grid_expand_hardtail", "di_grid_viz",
    "afe_core", "grid_expand_afe2", "afe2_scene_profiles", "afe2_calibration",
    "verifier_polytope",
}


@contextmanager
def _isolated_overnight_modules():
    names = {name for name in sys.modules if name in _COLLIDING}
    saved = {name: sys.modules.pop(name) for name in names}
    old_path = list(sys.path)
    sys.path[:0] = [str(_HERE), str(_REV), str(_WORK)]
    try:
        ac = importlib.import_module("afe_core")
        afe2 = importlib.import_module("grid_expand_afe2")
        yield ac, afe2
    finally:
        for name in _COLLIDING:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = old_path


@pytest.fixture
def afe2_modules():
    with _isolated_overnight_modules() as modules:
        yield modules


def test_terminal_prefix_rescues_execution_without_relabeling_full_window(
    monkeypatch, afe2_modules
) -> None:
    AC, _ = afe2_modules
    calls: list[int] = []

    def always_certified(state, controls, env, gamma, n_theta):
        calls.append(len(controls))
        return True, 0.25, 0.0

    monkeypatch.setattr(AC.GM2, "window_socp_stats", always_certified)
    env = SimpleNamespace(dt=1.0)
    state = np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float32)
    controls = np.zeros((10, 2), dtype=np.float32)

    result = AC.verify_plan_with_terminal(
        state,
        controls,
        env,
        gamma=0.5,
        goal_np=np.asarray((1.0, 0.0)),
        reach=0.15,
    )

    assert result["y"] == 0  # the full plan leaves the box and is not a D+ sample
    assert result["reason"] == "oob"
    assert result["exec_y"] == 1
    assert result["terminal_rescue"] is True
    assert result["terminal_tau"] == 1
    assert calls == [1]  # full OOB is cheap; only the terminal prefix reaches SOCP


def test_full_positive_remains_training_positive_and_needs_no_prefix_reverify(
    monkeypatch, afe2_modules
) -> None:
    AC, _ = afe2_modules
    calls: list[int] = []

    def always_certified(state, controls, env, gamma, n_theta):
        calls.append(len(controls))
        return True, 0.4, 0.0

    monkeypatch.setattr(AC.GM2, "window_socp_stats", always_certified)
    env = SimpleNamespace(dt=0.1)
    state = np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float32)
    controls = np.zeros((3, 2), dtype=np.float32)

    result = AC.verify_plan_with_terminal(
        state,
        controls,
        env,
        gamma=0.5,
        goal_np=np.asarray((0.1, 0.0)),
        reach=0.05,
    )

    assert result["y"] == 1
    assert result["exec_y"] == 1
    assert result["terminal_hit"] is True
    assert result["terminal_rescue"] is False
    assert result["terminal_reverify"] is False
    assert result["exec_prog"] == result["terminal_prog"]
    assert result["exec_prog"] > result["prog"]
    assert calls == [3]


def test_no_goal_hit_cannot_rescue_a_rejected_full_window(monkeypatch, afe2_modules) -> None:
    AC, _ = afe2_modules
    def reject(state, controls, env, gamma, n_theta):
        return False, float("nan"), -0.1

    monkeypatch.setattr(AC.GM2, "window_socp_stats", reject)
    env = SimpleNamespace(dt=0.1)
    state = np.asarray((1.0, 1.0, 0.0, 0.0), dtype=np.float32)
    controls = np.zeros((10, 2), dtype=np.float32)

    result = AC.verify_plan_with_terminal(
        state,
        controls,
        env,
        gamma=0.5,
        goal_np=np.asarray((4.5, 4.5)),
        reach=0.15,
    )

    assert result["y"] == 0
    assert result["exec_y"] == 0
    assert result["terminal_hit"] is False
    assert result["terminal_reverify"] is False


def test_terminal_execution_does_not_leak_into_positive_replay(
    monkeypatch, afe2_modules
) -> None:
    AC, AFE2 = afe2_modules
    class Policy:
        def sample_window(self, grid, low5, hist, n, temp, nfe):
            controls = torch.zeros((n, 10, 2), dtype=torch.float32)
            controls[0, 0, 0] = 0.1
            controls[1, 0, 0] = -0.1
            return controls

    def features(policy, controls, grid, low5, hist, s):
        return torch.eye(2, dtype=torch.float32)

    def labels(state, controls, env, gamma, goal, reach, n_theta):
        rescue = bool(controls[0, 0] > 0)
        return {
            "y": int(not rescue),
            "margin": float("nan") if rescue else 0.3,
            "resid": -0.1 if rescue else 0.0,
            "prog": -1.0 if rescue else 0.5,
            "d0": 1.0,
            "reason": "socp_fail" if rescue else "ok",
            "exec_y": 1,
            "exec_prog": 1.0 if rescue else 0.5,
            "exec_margin": 0.2 if rescue else 0.3,
            "terminal_prog": 1.0 if rescue else None,
            "terminal_resid": 0.0 if rescue else None,
            "terminal_hit": rescue,
            "terminal_tau": 1 if rescue else None,
            "terminal_rescue": rescue,
            "terminal_reason": "ok" if rescue else None,
            "terminal_reverify": rescue,
            "n_socp_solve": 1,
            "verifier_seconds": 0.001,
        }

    monkeypatch.setattr(AFE2.AC, "frozen_feat", features)
    monkeypatch.setattr(AFE2.AC, "verify_plan_with_terminal", labels)
    env = SimpleNamespace(
        obstacles=torch.zeros((0, 3), dtype=torch.float32),
        r_robot=0.0,
        goal=torch.tensor((1.0, 1.0)),
        dt=0.1,
    )
    cfg = SimpleNamespace(
        K=2,
        B=2,
        beta=0.2,
        temp=1.0,
        nfe=1,
        s=0.9,
        n_theta=1,
        reach=0.15,
    )
    store = AC.DStore()
    blr = AC.BLRSigma(dim=2, lam=1.0)

    best, stats = AFE2.acquire_and_execute(
        Policy(),
        blr,
        env,
        cfg,
        st=np.asarray((1.0, 1.0, 0.0, 0.0), dtype=np.float32),
        hist=[],
        g=0.5,
        store=store,
        round_i=1,
        ep=0,
        t=0,
        device="cpu",
        collect=True,
    )

    assert best is not None
    assert len(store) == 2
    assert store.n_pos() == 1
    assert stats["n_pos"] == 1
    assert stats["n_exec_pos"] == 2
    assert stats["sig_span"] >= 0.0 and stats["sig_iqr"] >= 0.0
    assert len(stats["feature_cosine_distance_q"]) == 3
    assert stats["selected_terminal_rescue"] is True
    store.mark_executed(best[1])
    store.validate_execution_witnesses()
    assert store.q_y[best[1]] == 0
    assert store.q_exec_y[best[1]] == 1
    assert store.q_terminal_rescue[best[1]] == 1
    assert store.q_terminal_tau[best[1]] == 1
    assert store.q_prog[best[1]] == -1.0
    assert store.q_exec_prog[best[1]] == 1.0
    assert all(store.q_y[qid] == 1 for qid in store.pos_ids)


def test_named_rng_stream_is_repeatable_and_restores_global_state(afe2_modules) -> None:
    AC, AFE2 = afe2_modules
    np.random.seed(1234)
    torch.manual_seed(5678)
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    seed = AFE2.named_seed(910, "gather", 3, 2, 7)
    with AC.isolated_random_state(seed):
        first_numpy = np.random.standard_normal(5)
        first_torch = torch.randn(5)
    assert np.array_equal(torch.random.get_rng_state().numpy(), torch_before.numpy())
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]

    np.random.standard_normal(20)
    torch.randn(20)
    with AC.isolated_random_state(seed):
        second_numpy = np.random.standard_normal(5)
        second_torch = torch.randn(5)
    assert np.array_equal(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)


def test_legacy_claude_checkpoint_contract_is_explicit_and_profile_bound(
    afe2_modules, monkeypatch,
) -> None:
    _, AFE2 = afe2_modules
    import codex_challenging.afe_restart.policy as checkpoint_policy

    policy = AFE2.HP.GridHPFlowPolicy(
        repr_dim=32, grid_hw=(32, 32), trunk_hidden=(160, 96)
    )
    checkpoint = {
        "config": policy.config(),
        "data": "druni_",
        "per_gamma_cap": 0,
        "best_val": 1.0101,
    }
    monkeypatch.setattr(
        checkpoint_policy,
        "model_state_hash",
        lambda _: AFE2.CLAUDE_LEGACY_MODEL_SHA256,
    )
    model_hash, contract, digest = AFE2.validate_checkpoint_contract(
        "claude_grid_v1",
        policy,
        checkpoint,
        AFE2.CLAUDE_LEGACY_CHECKPOINT_SHA256,
    )
    assert contract["name"] == "legacy_a32uni_forensic_v2"
    assert contract["checkpoint_model_state_sha256"] == model_hash
    assert AFE2._canonical_json_sha256(contract) == digest
    wrong_trunk = dict(checkpoint)
    wrong_trunk["config"] = dict(checkpoint["config"], trunk_hidden=[128, 64])
    with pytest.raises(RuntimeError, match="legacy uncapped druni_"):
        AFE2.validate_checkpoint_contract(
            "claude_grid_v1",
            policy,
            wrong_trunk,
            AFE2.CLAUDE_LEGACY_CHECKPOINT_SHA256,
        )

    with pytest.raises(RuntimeError, match="legacy uncapped druni_"):
        AFE2.validate_checkpoint_contract(
            "claude_grid_v1", policy, checkpoint, "a" * 64
        )

    monkeypatch.setattr(checkpoint_policy, "model_state_hash", lambda _: "f" * 64)
    with pytest.raises(RuntimeError, match="legacy uncapped druni_"):
        AFE2.validate_checkpoint_contract(
            "claude_grid_v1",
            policy,
            checkpoint,
            AFE2.CLAUDE_LEGACY_CHECKPOINT_SHA256,
        )


def test_codex_checkpoint_contract_requires_both_allowlist_and_promotion_witness(
    afe2_modules,
) -> None:
    _, AFE2 = afe2_modules
    from codex_challenging.afe_restart.policy import model_state_hash

    policy = AFE2.HP.GridHPFlowPolicy(repr_dim=32, grid_hw=(32, 32))
    config = policy.config()
    config.update(
        schema_version="w8sg-hp-v2-low5-only",
        raw_start_goal=False,
        use_gru=False,
    )
    checkpoint = {
        "config": config,
        "stage_schema": "afe_fresh_pretrain_v1",
        "fresh_from_scratch": True,
        "endpoint_free": True,
        "expansion_promotion": True,
        "id_mode_diversity_gate_passed": True,
        "id_evaluation_temperature": 1.0,
        "id_evaluation_uncertainty_tilting": False,
        "model_state_sha256": model_state_hash(policy),
        "source_query_hash_digest": "b" * 64,
        "source_manifest": "/sealed/stage2/manifest.json",
        "id_metrics_sha256": "c" * 64,
        "frozen_feature_snapshot": False,
    }
    documented_hash = next(iter(AFE2.CODEX_PROMOTED_CHECKPOINTS))
    model_hash, contract, digest = AFE2.validate_checkpoint_contract(
        "codex_radius1_v1", policy, checkpoint, documented_hash
    )
    assert contract["name"] == "fresh_stage3_promoted_v1"
    assert contract["checkpoint_model_state_sha256"] == model_hash
    assert AFE2._canonical_json_sha256(contract) == digest
    radius04_hash, radius04_contract, _ = AFE2.validate_checkpoint_contract(
        "codex_radius04_v1", policy, checkpoint, documented_hash
    )
    assert radius04_hash == model_hash
    assert radius04_contract["name"] == "fresh_stage3_promoted_v1"

    unpromoted = dict(checkpoint, expansion_promotion=False)
    with pytest.raises(RuntimeError, match="promoted only"):
        AFE2.validate_checkpoint_contract(
            "codex_radius1_v1", policy, unpromoted, documented_hash
        )
    with pytest.raises(RuntimeError, match="documented promoted"):
        AFE2.validate_checkpoint_contract(
            "codex_radius1_v1", policy, checkpoint, "d" * 64
        )


def test_query_context_archive_preserves_embedding_inputs_in_float32(afe2_modules) -> None:
    AC, _ = afe2_modules
    store = AC.DStore()
    grid = np.linspace(-1.0, 1.0, 3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)
    hist = np.linspace(-0.7, 0.7, AC.GF.K_HIST * 2, dtype=np.float32).reshape(
        AC.GF.K_HIST, 2
    )
    sid = store.add_step_ctx(
        np.zeros(4, np.float32), grid, np.zeros(5, np.float32), hist, (1, 2, 3)
    )
    assert store.ctx_hp[sid].dtype == np.float32
    assert store.ctx_hist[sid].dtype == np.float32
    assert np.array_equal(store.ctx_hp[sid], grid[2:3])
    assert np.array_equal(store.ctx_hist[sid], hist)


def test_legacy_task_box_tolerance_is_explicitly_locked(afe2_modules) -> None:
    AC, AFE2 = afe2_modules
    eps = float(AC.GM.EPS_TASK)
    assert AFE2.REFERENCE_RECIPE["taskspace_epsilon"] == eps
    inside_legacy_band = np.asarray([[-0.5 * eps, 2.5]], dtype=np.float32)
    outside_legacy_band = np.asarray([[-1.5 * eps, 2.5]], dtype=np.float32)
    assert AC.GM.in_taskspace(inside_legacy_band)
    assert not AC.GM.in_taskspace(outside_legacy_band)
