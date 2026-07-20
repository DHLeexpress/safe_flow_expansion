"""Integration invariants for Stage 05 accounting.

These tests intentionally stay tiny and mocked: the expensive flow sampler and
SOCP verifier are covered by their focused tests.  Here we guard metrics that
are consumed directly by the final validity report.
"""
from __future__ import annotations

import argparse
import copy
import json
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from afe_restart.config import (
    AFEConfig,
    SamplingConfig,
    clean_method_absence_manifest,
)
from afe_restart.controller import PlannedWindowAFEController
from afe_restart.policy import (
    FrozenFeatureModel,
    ledger_common_random_arrays,
    model_state_hash,
)
from afe_restart.stage5_expand import (
    CHECKPOINT_SCHEMA,
    FULL_REPLAY_DESCRIPTION,
    PROXIMAL_OBJECTIVE_FORMULA,
    RUN_DEFAULTS,
    _query_round_summary,
    _require_usable_proximal_solve,
    _restore_resume_state,
    _restore_saved_run_config,
)
from afe_restart.stage4_baseline import (
    SafeMPPIExpertSettings,
    build_sealed_final_test_bank,
)
from afe_restart.scene import verifier_spec_fingerprint
from afe_restart.schemas import QuerySource
from afe_restart.store import VerificationStore
from afe_restart.uncertainty import CumulativeLinearUncertainty
from afe_restart.validity import aggregate_independent_training_seed_audits


def test_fallback_frequency_is_bounded_fraction_of_control_decisions() -> None:
    """A fail-closed decision is a fallback attempt, but not an executed action."""

    episodes = [
        SimpleNamespace(
            actions=[],
            traces=[object()],
            fallback_steps=1,
            success=False,
            fail_closed=True,
            cache_hits=0,
        ),
        SimpleNamespace(
            actions=[],
            traces=[object()],
            fallback_steps=1,
            success=False,
            fail_closed=True,
            cache_hits=0,
        ),
    ]
    store = SimpleNamespace(records=())

    summary = _query_round_summary(episodes, store, before_queries=0)

    assert summary["executed_steps"] == 0
    assert summary["control_decisions"] == 2
    assert summary["fallback_steps"] == 2
    assert summary["fallback_frequency"] == 1.0
    assert 0.0 <= summary["fallback_frequency"] <= 1.0


def test_query_acceptance_is_flow_only_and_backup_is_reported_separately() -> None:
    records = (
        SimpleNamespace(source=QuerySource.FLOW, safe=False),
        SimpleNamespace(source=QuerySource.FLOW, safe=True),
        SimpleNamespace(source=QuerySource.SAFEMPPI_BACKUP, safe=True),
    )
    episode = SimpleNamespace(
        actions=[np.zeros(2)],
        traces=[object()],
        fallback_steps=1,
        success=True,
        fail_closed=False,
        cache_hits=0,
    )
    summary = _query_round_summary(
        [episode], SimpleNamespace(records=records), before_queries=0
    )

    assert summary["new_total_full_verifier_calls"] == 3
    assert summary["new_verifier_calls"] == 2
    assert summary["query_acceptance"] == 0.5
    assert summary["backup_verifier_calls"] == 1
    assert summary["backup_acceptance"] == 1.0


def test_query_summary_reports_afe_selectivity_without_replay_weighting() -> None:
    trace = SimpleNamespace(
        candidate_sigmas=np.array([0.01, 0.02, 0.04]),
        acquisition_probabilities=np.array([0.2, 0.3, 0.5]),
        acquisition_ess=1.0 / (0.2**2 + 0.3**2 + 0.5**2),
        queried=(
            SimpleNamespace(
                plan_kind="flow", acquisition_sigma=0.04,
            ),
            SimpleNamespace(
                plan_kind="backup", acquisition_sigma=0.02,
            ),
        ),
    )
    episode = SimpleNamespace(
        actions=[], traces=[trace], fallback_steps=0, success=False,
        fail_closed=False, cache_hits=0,
    )
    summary = _query_round_summary(
        [episode], SimpleNamespace(records=()), before_queries=0,
    )
    diagnostic = summary["acquisition_diagnostics"]

    assert diagnostic["decision_count"] == 1
    assert diagnostic["candidate_sigma_span"]["median"] == pytest.approx(0.03)
    assert diagnostic["effective_sample_size_fraction"]["median"] < 1.0
    assert diagnostic["gibbs_probability_max_min_ratio"]["median"] == 2.5
    assert diagnostic["queried_flow_sigma"]["mean"] == pytest.approx(0.04)
    assert diagnostic["queried_flow_sigma_midrank"]["mean"] == pytest.approx(5 / 6)


def test_ledger_cfm_common_randomness_is_query_keyed_not_order_keyed() -> None:
    records = [
        SimpleNamespace(query_hash=character * 64)
        for character in ("1", "2", "3")
    ]
    x0, tau = ledger_common_random_arrays(
        records, round_seed=91, dimension=20
    )
    reverse_x0, reverse_tau = ledger_common_random_arrays(
        list(reversed(records)), round_seed=91, dimension=20
    )

    np.testing.assert_array_equal(x0, reverse_x0[::-1])
    np.testing.assert_array_equal(tau, reverse_tau[::-1])
    changed_x0, changed_tau = ledger_common_random_arrays(
        records, round_seed=92, dimension=20
    )
    assert not np.array_equal(x0, changed_x0)
    assert not np.array_equal(tau, changed_tau)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3, 3)


def _resume_fixture():
    torch.manual_seed(7)
    phi0_model = TinyModel()
    frozen = FrozenFeatureModel.from_pretrained(phi0_model, expected_dim=32)
    current = copy.deepcopy(phi0_model)
    with torch.no_grad():
        current.layer.weight.add_(0.25)
    store = VerificationStore(CumulativeLinearUncertainty(lambda_=0.01))
    no_positive_solver = {
        "positive_count": 0,
        "total_record_count": 0,
        "optimizer_steps": 0,
        "stopping_reason": "no_positive_records",
        "converged": False,
        "sampling": "uniform_full_positive_pass_seeded_reshuffle",
        "trace": [],
    }
    history = [
        {
            "round": 0,
            "matrix": {"observations": 0},
            "model_hash": model_state_hash(current),
            "query": None,
            "solver": None,
        },
        *[
            {
                "round": round_index,
                "matrix": {"observations": 0},
                "model_hash": model_state_hash(current),
                "query": {},
                "solver": copy.deepcopy(no_positive_solver),
            }
            for round_index in (1, 2)
        ],
    ]
    run_config = dict(RUN_DEFAULTS)
    recipe = {
        "method": "planned-window AFE",
        "arm": "full",
        "acquisition": "afe",
        "acquisition_mode": "afe",
        "progress_ranking": True,
        "eligibility_mode": "full",
        "replay_eligibility": "full_safe",
        "runtime_safety_claim": True,
        "uncertainty_tilting": True,
        "ordinary_audit_untilted": True,
        "sampling_temperature": 1.0,
        "visualization_temperature": 0.5,
        "feature_time": 0.9,
        "frozen_feature_hash": frozen.state_hash,
        "source_model_hash": frozen.state_hash,
        "audit_bank_fingerprint": "fixed-bank",
        "audit_bank_role": "round_monitoring",
        "verifier_spec_fingerprint": "f" * 64,
        "run_config": run_config,
        "legacy_mechanisms": clean_method_absence_manifest(),
        "prox_eta": run_config["prox_eta"],
        "learning_rate": run_config["learning_rate"],
        "solver_max_steps": run_config["solver_max_steps"],
        "solver_min_steps": run_config["solver_min_steps"],
        "update_norm_limit": run_config["update_norm_limit"],
        "solver": {
            "max_steps": run_config["solver_max_steps"],
            "max_steps_role": "numerical cap, not a fixed scientific update count",
            "min_steps": run_config["solver_min_steps"],
            "min_steps_role": "minimum before tolerance-based convergence",
            "relative_loss_tolerance": run_config["relative_loss_tolerance"],
            "gradient_tolerance": run_config["gradient_tolerance"],
            "update_norm_limit": run_config["update_norm_limit"],
            "reported_optimizer_steps_are_numerical_outcomes": True,
        },
        "update_objective": {
            "formula": PROXIMAL_OBJECTIVE_FORMULA,
            "proximal_reference": "theta_n captured at expansion-round entry",
            "proximal_reference_is_data_or_replay_anchoring": False,
            "legacy_anchor_or_recovery_data": False,
        },
        "replay": FULL_REPLAY_DESCRIPTION,
    }
    payload = {
        "afe_schema": CHECKPOINT_SCHEMA,
        "round": 2,
        "recipe": recipe,
        "history": history,
        "frozen_feature_hash": frozen.state_hash,
        "frozen_feature_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in frozen.model.state_dict().items()
        },
        "current_model_hash": model_state_hash(current),
        "verification_store_state": store.state_dict(),
    }
    return current, frozen, payload


def test_resume_restores_original_phi0_and_cumulative_store_not_current_theta() -> None:
    current, original_frozen, payload = _resume_fixture()

    restored, store, history, round_index, recipe = _restore_resume_state(
        current,
        payload,
        audit_bank_fingerprint="fixed-bank",
        verifier_spec="f" * 64,
    )

    assert restored.state_hash == original_frozen.state_hash
    assert restored.state_hash != model_state_hash(current)
    assert store.query_count == store.uncertainty.count == 0
    assert round_index == 2
    assert history[-1]["round"] == 2
    assert recipe["run_config"]["ridge_lambda"] == 0.01


def test_resume_rejects_model_or_audit_source_mismatch() -> None:
    current, _frozen, payload = _resume_fixture()
    bad_model = copy.deepcopy(current)
    with torch.no_grad():
        bad_model.layer.bias.add_(1.0)
    with pytest.raises(RuntimeError, match="current-model hash"):
        _restore_resume_state(
            bad_model,
            payload,
            audit_bank_fingerprint="fixed-bank",
            verifier_spec="f" * 64,
        )
    with pytest.raises(RuntimeError, match="audit bank"):
        _restore_resume_state(
            current,
            payload,
            audit_bank_fingerprint="different-bank",
            verifier_spec="f" * 64,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda recipe: recipe["legacy_mechanisms"].__setitem__(
                "demo_frac", "present"
            ),
            "absence manifest",
        ),
        (
            lambda recipe: recipe["update_objective"].__setitem__(
                "formula", "legacy anchor"
            ),
            "prescribed proximal objective",
        ),
        (
            lambda recipe: recipe.__setitem__("acquisition_mode", "uniform"),
            "clean Full arm",
        ),
    ],
)
def test_resume_rejects_ad_hoc_or_nonfull_recipe(mutate, message) -> None:
    current, _frozen, payload = _resume_fixture()
    mutate(payload["recipe"])
    with pytest.raises(RuntimeError, match=message):
        _restore_resume_state(
            current,
            payload,
            audit_bank_fingerprint="fixed-bank",
            verifier_spec="f" * 64,
        )


def test_max_step_proximal_solve_emits_stuck_and_cannot_checkpoint(tmp_path) -> None:
    solver = {
        "positive_count": 3,
        "total_record_count": 3,
        "optimizer_steps": 12,
        "stopping_reason": "max_steps",
        "converged": False,
        "sampling": "uniform_full_positive_pass_seeded_reshuffle",
        "trace": [{"positive_coverage": 1.0, "projected_to_update_bound": False}],
    }
    with pytest.raises(RuntimeError, match="stopping_reason='max_steps'"):
        _require_usable_proximal_solve(
            solver,
            label="test Full",
            round_index=4,
            output_dir=tmp_path,
        )
    stuck_path = tmp_path / "logs/round_004_solver_STUCK.json"
    stuck = json.loads(stuck_path.read_text())
    assert stuck["status"] == "STUCK_UNUSABLE_PROXIMAL_SOLVE"
    assert stuck["stopping_reason"] == "max_steps"
    assert stuck["usable_for_checkpoint"] is False
    assert not (tmp_path / "checkpoints/round_004.pt").exists()


def test_resume_inherits_saved_protocol_and_rejects_explicit_drift() -> None:
    saved = dict(RUN_DEFAULTS)
    saved["candidate_count"] = 96
    args = argparse.Namespace(**RUN_DEFAULTS)
    _restore_saved_run_config(args, saved)
    assert args.candidate_count == 96

    conflicting = argparse.Namespace(**RUN_DEFAULTS)
    conflicting.candidate_count = 128
    with pytest.raises(RuntimeError, match="cannot change scientific option"):
        _restore_saved_run_config(conflicting, saved)


class ConstantFeatures:
    state_hash = "fixed"

    def encode(self, _context, plans):
        features = np.zeros((len(plans), 32), dtype=np.float64)
        features[:, 0] = 1.0
        return features


class CountingEmptyBackup:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def propose(self, *_args, **_kwargs):
        return [], {}


class ControllerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class TinyEnvironment:
    dt = 0.1
    T = 1
    r_robot = 0.1
    x0 = torch.tensor([0.5, 0.5, 0.0, 0.0])
    goal = torch.tensor([4.5, 4.5])
    obstacles = torch.empty((0, 3))


def test_controller_resets_backup_once_at_each_episode_entry(monkeypatch) -> None:
    backup = CountingEmptyBackup()
    model = ControllerModel()
    config = AFEConfig(sampling=SamplingConfig(candidate_count=1, verifier_budget=1))

    def plans(*_args, **_kwargs):
        return np.zeros((1, 10, 2), dtype=np.float32)

    def verifier(*_args, **_kwargs):
        return SimpleNamespace(
            safe=False,
            in_bounds=False,
            socp_ok=False,
            physical_clearance_m=-1.0,
            certificate_residual=-1.0,
            face_margin_m=-1.0,
            start_goal_distance_m=5.0,
            terminal_goal_distance_m=5.0,
            progress_m=0.0,
        )

    monkeypatch.setattr("afe_restart.controller.sample_plans", plans)
    # The hash/schema path requires a real QueryContext; reuse the smallest
    # finite numerical form after confirming reset happened before sampling.
    from afe_restart.schemas import QueryContext
    monkeypatch.setattr(
        "afe_restart.controller.context_from_state",
        lambda state, goal, _gamma, _actions, env, **_kwargs: QueryContext(
            np.zeros((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.asarray(state, dtype=np.float64),
            verifier_spec_fingerprint(env, goal),
        ),
    )
    controller = PlannedWindowAFEController(
        model,
        ConstantFeatures(),
        VerificationStore(CumulativeLinearUncertainty(lambda_=0.01)),
        config=config,
        backup=backup,
        verifier_fn=verifier,
        device="cpu",
        fallback_verifier_budget=1,
    )
    controller.run_episode(TinyEnvironment(), 0.5, seed=1)
    controller.run_episode(TinyEnvironment(), 0.5, seed=1)
    assert backup.reset_calls == 2


def _sealed_audit(validity: tuple[float, float], fingerprint: str = "sealed"):
    return {
        "context_bank_fingerprint": fingerprint,
        "context_bank_role": "sealed_final_test",
        "temperature": 1.0,
        "uncertainty_tilting": False,
        "per_gamma": [
            {"gamma": 0.1, "validity_mass": validity[0], "progress_validity": validity[0] / 2},
            {"gamma": 1.0, "validity_mass": validity[1], "progress_validity": validity[1] / 2},
        ],
    }


def test_multiseed_validity_uses_trained_model_as_replication_unit() -> None:
    result = aggregate_independent_training_seed_audits({
        11: _sealed_audit((0.2, 0.5)),
        22: _sealed_audit((0.4, 0.7)),
        33: _sealed_audit((0.6, 0.9)),
    })
    assert result["independent_training_seed_count"] == 3
    assert result["replication_unit"] == "independently_trained_model"
    assert result["plan_samples_pooled_across_training_seeds"] is False
    gamma01 = result["per_gamma"][0]
    assert gamma01["validity"]["mean"] == pytest.approx(0.4)
    assert gamma01["validity"]["method"] == "student_t_across_independent_training_seed_estimates"


def test_multiseed_validity_rejects_monitoring_or_mismatched_banks() -> None:
    monitoring = _sealed_audit((0.2, 0.4))
    monitoring["context_bank_role"] = "round_monitoring"
    with pytest.raises(ValueError, match="sealed final-test"):
        aggregate_independent_training_seed_audits({
            1: monitoring,
            2: _sealed_audit((0.3, 0.5)),
        })
    with pytest.raises(ValueError, match="same fingerprinted"):
        aggregate_independent_training_seed_audits({
            1: _sealed_audit((0.2, 0.4), "a"),
            2: _sealed_audit((0.3, 0.5), "b"),
        })


def test_sealed_bank_includes_start_and_distinct_seed_interiors(monkeypatch) -> None:
    upper = np.asarray(
        [[0.5, 0.5], [1.2, 1.4], [1.8, 2.7], [2.6, 3.7], [4.5, 4.5]],
        dtype=np.float32,
    )
    lower = upper[:, ::-1].copy()

    observed_planner_settings = []

    def episode(*, seed, config, **_kwargs):
        observed_planner_settings.append((
            config.smooth_weight, config.noise_var_mult, config.retreat_weight,
        ))
        path = upper if seed % 2 == 0 else lower
        states = np.column_stack((path, np.zeros((len(path), 2), dtype=np.float32)))
        actions = np.zeros((len(path) - 1, 2), dtype=np.float32)
        return {
            "seed": seed,
            "success": True,
            "path": path,
            "states": states,
            "executed_actions": actions,
            "steps": len(actions),
        }

    monkeypatch.setattr("afe_restart.stage4_baseline.run_expert_rollout", episode)
    env = SimpleNamespace(x0=torch.tensor([0.5, 0.5, 0.0, 0.0]))
    bank, _provenance = build_sealed_final_test_bank(
        env,
        device=torch.device("cpu"),
        seed0=100,
        interior_contexts_per_mode=2,
        candidate_limit=4,
        planner_settings=SafeMPPIExpertSettings(23.0, 1.5, 0.2),
    )
    assert bank.role == "sealed_final_test"
    assert len(bank) == 5
    assert bank[0].expert_mode == "deployment_start"
    assert bank[0].source_step == 0
    seeds = [row.expert_seed for row in bank if row.expert_seed >= 0]
    assert len(seeds) == len(set(seeds)) == 4
    assert observed_planner_settings == [(23.0, 1.5, 0.2)] * 4
