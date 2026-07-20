from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import random
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import afe_context as CX
import afe_core as AC
import afe_demo_support as DS
import grid_expand_afe_rbf as RBF
from codex_challenging.afe_restart.scene import make_id_scene


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(3, 4)
        self.head = torch.nn.Linear(4, 2)
        self.u_max = 1.0
        self.d = 2

    def module_groups(self):
        return {"trunk": self.trunk, "head": self.head}

    def ctx_from(self, grid, low, hist):
        del grid, hist
        return low[:, :1]

    def _expand_ctx(self, context, count):
        assert len(context) == count
        return context

    def forward(self, values, time, context):
        del time
        return self.head(torch.nn.functional.silu(self.trunk(
            torch.cat((values, context), dim=1)
        )))

    def cfm_loss(self, controls, context, weights=None):
        count = len(controls)
        x1 = controls.reshape(count, self.d)
        x0 = torch.randn_like(x1)
        tau = torch.rand(count).clamp_(1.0e-4, 1.0)
        mixed = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
        per = ((self(mixed, tau, context) - (x1 - x0)) ** 2).mean(dim=1)
        if weights is not None:
            per = per * weights
        return per.mean()


def make_store(count=256):
    store = AC.DStore()
    gammas = (0.1, 0.5)
    for index in range(count):
        context_id = len(store.ctx_meta)
        query_round = 1 + (index % 2)
        gamma = gammas[index % len(gammas)]
        store.ctx_meta.append((query_round, index % 8, index))
        store.ctx_hp.append(np.zeros((1, 32, 32), np.float32))
        store.ctx_low5.append(np.asarray((1, 0, 0, 0, gamma), np.float32))
        store.ctx_hist.append(np.zeros((16, 2), np.float32))
        query_id = len(store.q_sid)
        store.q_sid.append(context_id)
        store.q_round.append(query_round)
        store.q_gamma.append(gamma)
        store.q_U.append(np.asarray([[0.2, -0.1]], np.float32))
        store.pos_ids.append(query_id)
    return store


def cfg(steps=16, demo_frac=0.0):
    return SimpleNamespace(
        replay_window=2,
        replay_sampling="round_gamma_replica_context",
        replay_loss_weighting="gamma_episode_context_query_equal_mass",
        gammas=(0.1, 0.5),
        optimizer_steps_per_round=steps,
        demo_frac=demo_frac,
        grad_clip=1.0,
        seed=910,
    )


class FakeDemo:
    def __init__(self):
        self.rows_requested = []

    def sample_original_rows(self, pair_count, rng):
        del rng
        rows = np.arange(pair_count, dtype=np.int64)
        return rows, {
            "pair_count": pair_count,
            "gamma_counts": {"0.1": (pair_count + 1) // 2, "0.5": pair_count // 2},
            "unique_trajectories": pair_count,
        }

    def paired_batch(self, rows, device):
        self.rows_requested.extend(int(value) for value in rows)
        count = 2 * len(rows)
        grid = torch.zeros(count, 3, 32, 32, device=device)
        low = torch.zeros(count, 5, device=device)
        hist = torch.zeros(count, 16, 2, device=device)
        controls = torch.zeros(count, 1, 2, device=device)
        controls[0::2, 0] = torch.tensor((0.3, -0.2), device=device)
        controls[1::2, 0] = torch.tensor((-0.2, 0.3), device=device)
        return grid, low, hist, controls


def support_args(**overrides):
    values = {
        "protocol_profile": "v3_support_sweep",
        "scene_profile": "low7_radius1_canonical_v1",
        "rounds": 100,
        "rollout_replicas": 8,
        "K": 16,
        "B": 4,
        "T": 300,
        "M_eval": 0,
        "batch": 128,
        "afe_steps": 0,
        "afe_lr": 1.0e-5,
        "gp_cap": 512,
        "gp_lam": 1.0e-2,
        "acquisition_mode": "sequential",
        "adaptive_ess_target": 0.5,
        "adaptive_beta_contexts_per_gamma": 64,
        "adaptive_beta_equalize_gammas": True,
        "replay_window": 2,
        "replay_sampling": "round_gamma_replica_context",
        "replay_update_mode": "fixed_macro_steps_exact_epoch",
        "replay_loss_weighting": "gamma_episode_context_query_equal_mass",
        "gp_replay_window": 2,
        "gp_replay_sampling": "round_gamma_replica_context",
        "lengthscale_multiplier": 1.0,
        "negative_alpha": 0.0,
        "execution_rule": "nominal_hp_max_step_margin",
        "conditioning_schema": CX.LOW7_SCHEMA,
        "freeze_visual_encoder": True,
        "skip_training_probes": True,
        "calibration_replicas": 8,
        "calibration_control_steps": 4,
        "sweep_compact_artifacts": True,
        "compact_checkpoint_every": 1,
        "route_metric_steps": 10,
        "route_ambiguity_band": 0.05,
        "nvp_audit_all_k": False,
        "optimizer_steps_per_round": 16,
        "demo_frac": 0.125,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reflection_is_an_exact_involution_for_all_coordinate_objects():
    rng = np.random.default_rng(4)
    for shape in ((2,), (10, 2), (16, 2)):
        value = rng.standard_normal(shape).astype(np.float32)
        assert np.array_equal(DS.reflect_xy(DS.reflect_xy(value)), value)
    state = rng.standard_normal(4).astype(np.float32)
    assert np.array_equal(DS.reflect_state(DS.reflect_state(state)), state)
    flat = torch.randn(7, 20)
    assert torch.equal(DS.reflect_action_tensor(DS.reflect_action_tensor(flat)), flat)


def test_reflected_context_is_canonically_recomputed_and_grid_equivariant():
    env = make_id_scene(start=np.asarray((0.7, 1.3), np.float32),
                        goal=np.asarray((4.7, 4.7), np.float32))
    state = np.asarray((0.7, 1.3, 0.2, -0.1), np.float32)
    goal = np.asarray((4.7, 4.7), np.float32)
    hist = np.arange(32, dtype=np.float32).reshape(16, 2) / 50.0
    original = CX.build_context(state, goal, 0.4, hist, env, CX.LOW7_SCHEMA)
    reflected = CX.build_context(
        DS.reflect_state(state), DS.reflect_xy(goal), 0.4, DS.reflect_xy(hist),
        env, CX.LOW7_SCHEMA,
    )
    assert np.array_equal(reflected.grid, DS.reflect_polar_grid(original.grid))
    assert np.array_equal(reflected.low5[:2], original.low5[:2][::-1])
    assert np.array_equal(reflected.low5[2:4], original.low5[2:4][::-1])
    assert np.array_equal(reflected.low5[4:6], original.low5[4:6][::-1])
    assert reflected.low5[-1] == original.low5[-1]
    assert np.array_equal(reflected.hist, original.hist[..., ::-1])


@pytest.mark.parametrize("gamma", (0.1, 0.4, 1.0))
def test_nominal_scene_full_verifier_is_reflection_invariant(gamma):
    env = make_id_scene(start=np.asarray((0.5, 0.5), np.float32),
                        goal=np.asarray((4.5, 4.5), np.float32))
    state = np.asarray((0.5, 0.5, 0.0, 0.0), np.float32)
    plan = np.tile(np.asarray((0.3, 0.4), np.float32), (10, 1))
    goal = np.asarray((4.5, 4.5), np.float64)
    original = AC.verify_plan(state, plan, env, gamma, goal)
    reflected = AC.verify_plan(
        DS.reflect_state(state), DS.reflect_xy(plan), env, gamma, goal
    )
    assert original["y"] == reflected["y"]
    assert original["reason"] == reflected["reason"]
    assert original["prog"] == pytest.approx(reflected["prog"], abs=1.0e-7)
    if np.isfinite(original["margin"]):
        assert original["margin"] == pytest.approx(reflected["margin"], abs=1.0e-7)


def test_authenticated_reference_proves_train_validation_separation_and_linkage():
    reviewed_recipe = Path(
        "/home/dohyun/projects/afe2_runs/"
        "low7_rbf_v2_lineage_mass_giant_dad39e6/run/recipe.json"
    )
    recipe = json.loads(reviewed_recipe.read_text())
    reference, provenance = DS.load_authenticated_demo_reference(
        recipe["source_checkpoint"], recipe["source_checkpoint_sha256"],
        load_tensors=False,
    )
    assert reference is None
    assert provenance.pair_leakage == 0
    assert provenance.train_pairs > 0 and provenance.validation_pairs > 0
    assert provenance.train_windows > 0 and provenance.validation_windows > 0
    assert provenance.source_checkpoint_sha256 == recipe["source_checkpoint_sha256"]
    assert (
        provenance.source_checkpoint_model_sha256
        == recipe["source_checkpoint_model_sha256"]
    )
    audit = provenance.fixed_symmetry_audit
    assert audit["subset_train_only"] is True
    assert audit["reflection_involution"] is True
    assert len(audit["nominal_verifier_invariance"]) == len(DS.GAMMAS)


def test_macro_partition_has_exact_coverage_and_near_equal_mass():
    ids = list(range(257))
    raw = np.linspace(1.0, 4.0, len(ids))
    raw /= raw.sum()
    mass = dict(zip(ids, raw))
    batches, masses, residual = DS.partition_epoch_by_mass(ids, mass, 32)
    flat = [value for batch in batches for value in batch]
    assert len(flat) == len(set(flat)) == len(ids)
    assert set(flat) == set(ids)
    assert len(batches) == 32 and all(batches)
    assert sum(masses) == pytest.approx(1.0)
    assert residual <= max(raw)


def test_demo_original_sampling_balances_gamma_and_trajectory_hierarchy():
    hierarchy = {
        gamma: {
            trajectory: np.asarray([1000 * gamma_index + 10 * trajectory + row for row in range(3)])
            for trajectory in range(5 + gamma_index)
        }
        for gamma_index, gamma in enumerate(DS.GAMMAS)
    }
    demo = object.__new__(DS.DemoReference)
    demo.train_rows_by_gamma_trajectory = hierarchy
    rows, diagnostics = demo.sample_original_rows(701, np.random.default_rng(17))
    assert len(rows) == 701
    assert max(diagnostics["gamma_counts"].values()) - min(
        diagnostics["gamma_counts"].values()
    ) <= 1
    assert max(
        row["draw_count_spread"]
        for row in diagnostics["trajectory_balance_by_gamma"].values()
    ) <= 1
    allowed = set(np.concatenate([
        rows_for_trajectory
        for per_gamma in hierarchy.values()
        for rows_for_trajectory in per_gamma.values()
    ]))
    assert set(rows).issubset(allowed)


def test_support_update_exact_steps_coverage_pairing_and_store_isolation():
    store = make_store()
    before = copy.deepcopy(store.__dict__)
    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-5)
    demo = FakeDemo()
    torch.manual_seed(7)
    result = DS.update_round_support(
        policy, optimizer, store, cfg(16, 0.125), torch.device("cpu"),
        np.random.default_rng(9), 2, demo,
    )
    assert result["optimizer_steps"] == 16
    assert result["optimizer_draws"] == 256
    assert result["n_distinct"] == 256
    assert result["replay_epoch_coverage"] == 1.0
    assert result["replay_duplicate_draws"] == 0
    assert result["demo_original_count"] == result["demo_reflected_count"]
    assert sum(result["demo_batch_sizes"]) == result["demo_examples"]
    assert len(demo.rows_requested) == result["demo_original_count"]
    assert store.__dict__.keys() == before.keys()
    for key in store.__dict__:
        left, right = store.__dict__[key], before[key]
        if isinstance(left, list) and left and isinstance(left[0], np.ndarray):
            assert all(np.array_equal(a, b) for a, b in zip(left, right))
        else:
            assert left == right


def test_demo_zero_path_is_bitwise_equivalent_and_never_accesses_reference():
    class Bomb:
        def __getattribute__(self, name):
            raise AssertionError(f"demo reference was accessed: {name}")

    store = make_store()
    first = TinyPolicy()
    second = copy.deepcopy(first)
    opt_first = torch.optim.Adam(first.parameters(), lr=1.0e-5)
    opt_second = torch.optim.Adam(second.parameters(), lr=1.0e-5)
    torch.manual_seed(123)
    state = torch.random.get_rng_state()
    out_first = DS.update_round_support(
        first, opt_first, store, cfg(16, 0.0), torch.device("cpu"),
        np.random.default_rng(11), 2, None,
    )
    end_first = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    out_second = DS.update_round_support(
        second, opt_second, store, cfg(16, 0.0), torch.device("cpu"),
        np.random.default_rng(11), 2, Bomb(),
    )
    assert torch.equal(end_first, torch.random.get_rng_state())
    assert out_first["drawn_ids"] == out_second["drawn_ids"]
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_empty_dplus_performs_no_demo_only_update():
    store = make_store(0)
    policy = TinyPolicy()
    before = copy.deepcopy(policy.state_dict())
    result = DS.update_round_support(
        policy, torch.optim.Adam(policy.parameters()), store, cfg(16, 0.25),
        torch.device("cpu"), np.random.default_rng(1), 2, FakeDemo(),
    )
    assert result is None
    for name, value in policy.state_dict().items():
        assert torch.equal(value, before[name])


def test_support_profile_locks_only_declared_optimizer_and_demo_matrix():
    for steps in (16, 32):
        for demo_frac in (0.0, 0.125, 0.25):
            RBF.validate_protocol_args(support_args(
                optimizer_steps_per_round=steps, demo_frac=demo_frac
            ))
    source = inspect.getsource(RBF.run)
    assert '"execution_rule": cfg.execution_rule' in source
    for name, value in (
        ("optimizer_steps_per_round", 24),
        ("demo_frac", 0.2),
        ("nvp_audit_all_k", True),
        ("rollout_replicas", 16),
        ("execution_rule", "nominal_hp_max_step_progress"),
    ):
        with pytest.raises(ValueError):
            RBF.validate_protocol_args(support_args(**{name: value}))


def test_preflight_changes_only_round_count():
    RBF.validate_protocol_args(support_args(
        protocol_profile="v3_support_preflight", rounds=1
    ))
    with pytest.raises(ValueError, match="rounds"):
        RBF.validate_protocol_args(support_args(
            protocol_profile="v3_support_preflight", rounds=2
        ))


def test_disabling_all_k_audit_cannot_change_training_or_rng_state(monkeypatch):
    forbidden = {
        "policy", "gp", "store", "optimizer", "beta", "rng", "acquisition_rng"
    }
    assert not forbidden & set(inspect.signature(RBF.run_all_k_nvp_audit_only).parameters)

    class Executor:
        def map(self, function, tasks, chunksize):
            del function, chunksize
            return [
                (episode_id, candidate_id, {
                    "reason": "socp_fail", "n_socp_solve": 1,
                    "verifier_seconds": 0.01,
                })
                for episode_id, candidate_id, *_ in tasks
            ]

    monkeypatch.setattr(
        RBF.EX,
        "select_nominal_hp_execution",
        lambda *args, **kwargs: {
            "counts": {"nominal_hp_eligible": 0, "execution_verifier_positive": 0},
            "chosen": None,
        },
    )
    policy = TinyPolicy()
    store = make_store(32)
    gp_state = np.arange(9, dtype=np.float64)
    beta = 0.25
    acquisition = [0, 2]
    execution = {"chosen": None, "counts": {"nominal_hp_eligible": 0}}
    model_before = copy.deepcopy(policy.state_dict())
    store_before = copy.deepcopy(store.__dict__)
    gp_before = gp_state.copy()
    np.random.seed(91)
    random.seed(92)
    torch.manual_seed(93)
    np_before = copy.deepcopy(np.random.get_state())
    random_before = random.getstate()
    torch_before = torch.random.get_rng_state()
    selected_result = {
        "reason": "socp_fail", "n_socp_solve": 1, "verifier_seconds": 0.01
    }
    audit, _ = RBF.run_all_k_nvp_audit_only(
        Executor(),
        episode_id=0,
        state=np.zeros(4, np.float32),
        candidate_controls=np.zeros((4, 10, 2), np.float32),
        gamma=0.5,
        query_rows=[(0, 0, selected_result), (2, 1, selected_result)],
        execution_selection={
            "counts": {"nominal_hp_eligible": 0, "execution_verifier_positive": 0}
        },
        execution_rule="nominal_hp_max_step_margin",
        env=object(),
    )
    assert audit["status"] == "audit_only_no_execution_no_storage"
    assert beta == 0.25 and acquisition == [0, 2] and execution["chosen"] is None
    assert np.array_equal(gp_state, gp_before)
    for name, value in policy.state_dict().items():
        assert torch.equal(value, model_before[name])
    assert store.__dict__.keys() == store_before.keys()
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert random.getstate() == random_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)
