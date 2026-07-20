from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import afe_context as CX
import grid_expand_afe_rbf as RBF


def _v2_args(**overrides):
    values = {
        "protocol_profile": "v2_smoke",
        "scene_profile": "low7_radius1_canonical_v1",
        "rounds": 10,
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
        "replay_update_mode": "one_epoch_without_replacement",
        "replay_loss_weighting": "query_uniform",
        "gp_replay_window": 2,
        "gp_replay_sampling": "round_gamma_replica_context",
        "lengthscale_multiplier": 1.0,
        "negative_alpha": 0.0,
        "execution_rule": "nominal_hp_max_step_margin_only",
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
        "balanced_r0_delivery": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_v2_smoke_contract_accepts_only_the_declared_end_to_end_recipe() -> None:
    RBF.validate_protocol_args(_v2_args())


def test_v2_lineage_mass_smoke_locks_weighting_execution_and_nvp_audit() -> None:
    RBF.validate_protocol_args(_v2_args(
        protocol_profile="v2_lineage_mass_smoke",
        replay_loss_weighting="gamma_episode_context_query_equal_mass",
        execution_rule="nominal_hp_max_step_margin",
        nvp_audit_all_k=True,
    ))
    for name, value in (
        ("replay_loss_weighting", "query_uniform"),
        ("execution_rule", "nominal_hp_max_step_margin_only"),
        ("nvp_audit_all_k", False),
    ):
        args = _v2_args(
            protocol_profile="v2_lineage_mass_smoke",
            replay_loss_weighting="gamma_episode_context_query_equal_mass",
            execution_rule="nominal_hp_max_step_margin",
            nvp_audit_all_k=True,
        )
        setattr(args, name, value)
        with pytest.raises(ValueError, match=name):
            RBF.validate_protocol_args(args)


@pytest.mark.parametrize("gp_cap", [512, 1024])
@pytest.mark.parametrize("ess_target", [0.25, 0.5])
@pytest.mark.parametrize("alpha", [0.0, 0.001, 0.01])
@pytest.mark.parametrize(
    "execution_rule", [RBF.EX.MAX_STEP_MARGIN, RBF.EX.SAFEMPPI_COST]
)
def test_b1_balanced_sweep_accepts_only_declared_scientific_arms(
    gp_cap, ess_target, alpha, execution_rule
) -> None:
    RBF.validate_protocol_args(_v2_args(
        protocol_profile="b1_balanced_r0_sweep",
        rounds=20,
        replay_loss_weighting="gamma_episode_context_query_equal_mass",
        execution_rule=execution_rule,
        nvp_audit_all_k=True,
        gp_cap=gp_cap,
        adaptive_ess_target=ess_target,
        negative_alpha=alpha,
        balanced_r0_delivery="qualified.json",
    ))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("gp_cap", 256),
        ("adaptive_ess_target", 0.75),
        ("negative_alpha", 0.1),
        ("execution_rule", RBF.EX.MAX_STEP_PROGRESS),
        ("rollout_replicas", 16),
    ],
)
def test_b1_balanced_sweep_rejects_undeclared_variants(name, value) -> None:
    args = _v2_args(
        protocol_profile="b1_balanced_r0_sweep",
        rounds=20,
        replay_loss_weighting="gamma_episode_context_query_equal_mass",
        execution_rule=RBF.EX.MAX_STEP_MARGIN,
        nvp_audit_all_k=True,
        balanced_r0_delivery="qualified.json",
    )
    setattr(args, name, value)
    with pytest.raises(ValueError, match=("B1" if name in {
        "gp_cap", "adaptive_ess_target", "negative_alpha", "execution_rule"
    } else name)):
        RBF.validate_protocol_args(args)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("K", 64),
        ("B", 8),
        ("rollout_replicas", 2),
        ("afe_lr", 1.0e-4),
        ("replay_sampling", "query_uniform"),
        ("replay_update_mode", "fixed_steps_with_replacement"),
        ("replay_loss_weighting", "gamma_episode_context_query_equal_mass"),
        ("execution_rule", "nominal_hp_max_step_progress"),
        ("compact_checkpoint_every", 10),
    ],
)
def test_v2_smoke_contract_rejects_silent_recipe_drift(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        RBF.validate_protocol_args(_v2_args(**{name: value}))


def test_v1_contract_remains_backward_compatible() -> None:
    RBF.validate_protocol_args(SimpleNamespace(
        protocol_profile="v1", K=64, B=8, batch=128
    ))
    with pytest.raises(ValueError, match="first RBF study"):
        RBF.validate_protocol_args(SimpleNamespace(
            protocol_profile="v1", K=16, B=8, batch=128
        ))


@pytest.mark.parametrize(
    ("counts", "errors", "expected"),
    [
        (
            {"nominal_hp_eligible": 1, "execution_verifier_positive": 1},
            0,
            "selected_B_acquisition_miss",
        ),
        (
            {"nominal_hp_eligible": 0, "execution_verifier_positive": 2},
            0,
            "all_K_nominal_hp_gate_failure",
        ),
        (
            {"nominal_hp_eligible": 0, "execution_verifier_positive": 0},
            0,
            "finite_K_no_execution_candidate",
        ),
        (
            {"nominal_hp_eligible": 0, "execution_verifier_positive": 0},
            1,
            "indeterminate_socp_error",
        ),
        (
            {"nominal_hp_eligible": 0, "execution_verifier_positive": 2},
            1,
            "indeterminate_socp_error",
        ),
    ],
)
def test_all_k_nvp_audit_classifier_is_mutually_exclusive(
    counts, errors, expected
) -> None:
    assert RBF.classify_nvp_all_k(counts, errors) == expected
