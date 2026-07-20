from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from afe_restart.ablations import (
    AblationArm,
    MatchedProtocol,
    OfflineBoundsReplayView,
    ablation_spec,
    arm_manifest,
    assert_matched_protocols,
    training_view,
)
from afe_restart.config import AFEConfig, FeatureConfig, SamplingConfig
from afe_restart.controller import PlannedWindowAFEController
from afe_restart.dynamics import rollout_plan
from afe_restart.fallback import BackupProposal
from afe_restart.proximal_update import ProximalConfig, solve_proximal_update
from afe_restart.scene import make_ood_scene
from afe_restart.store import VerificationStore
from afe_restart.uncertainty import CumulativeLinearUncertainty
from afe_restart.verifier import PlanVerification
import afe_restart.stage6_ablations as stage6


class TinyFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.d = 20
        self.T = 10
        self.u_max = 1.0

    def ctx_from(self, grid, low5, hist):
        del low5, hist
        return torch.zeros(grid.shape[0], 1, device=grid.device) + self.weight

    @staticmethod
    def _expand_ctx(ctx, count):
        return ctx[None].expand(count, -1) if ctx.ndim == 1 else ctx.expand(count, -1)

    def forward(self, x, tau, ctx):
        del tau, ctx
        return torch.zeros_like(x) + self.weight


class IndexedFeatures:
    def encode(self, context, plans):
        del context
        rows = []
        for index, plan in enumerate(np.asarray(plans)):
            z = np.zeros(32, dtype=np.float64)
            z[index % 32] = 1.0
            z[(index + 9) % 32] = 0.1 + abs(float(plan[0, 0]))
            z /= np.linalg.norm(z)
            rows.append(z)
        return np.asarray(rows)


class NeverUsedBackup:
    def propose(self, *args, **kwargs):
        del args, kwargs
        plan = np.zeros((10, 2), dtype=np.float32)
        return [BackupProposal(plan, "test_backup", None)], {"proposal_count": 1}


def _config() -> AFEConfig:
    return AFEConfig(
        sampling=SamplingConfig(
            candidate_count=4,
            verifier_budget=2,
            beta=0.2,
            expansion_temperature=1.0,
            audit_temperature=1.0,
            visualization_temperature=0.5,
            nfe=1,
        ),
        features=FeatureConfig(ridge_lambda=0.01),
    )


def _verification(state, plan, env, gamma, *, in_bounds, socp_ok, progress):
    states = rollout_plan(state, plan)
    positions = states[:, :2].copy()
    states.setflags(write=False)
    positions.setflags(write=False)
    d0 = float(np.linalg.norm(positions[0] - env.goal.numpy()))
    return PlanVerification(
        safe=bool(in_bounds and socp_ok),
        in_bounds=bool(in_bounds),
        socp_ok=bool(socp_ok),
        bounds_margin_m=0.1 if in_bounds else -0.1,
        physical_clearance_m=0.2,
        face_margin_m=0.2 if socp_ok else -np.inf,
        certificate_residual=0.1 if socp_ok else -0.1,
        certificate_worst_step=10,
        progress_m=float(progress),
        start_goal_distance_m=d0,
        terminal_goal_distance_m=max(0.0, d0 - float(progress)),
        gamma=float(gamma),
        states=states,
        positions=positions,
    )


def _controller(spec, verifier):
    model = TinyFlow()
    store = VerificationStore(CumulativeLinearUncertainty(lambda_=0.01))
    controller = PlannedWindowAFEController(
        model,
        IndexedFeatures(),
        store,
        config=_config(),
        backup=NeverUsedBackup(),
        verifier_fn=verifier,
        device="cpu",
        fallback_verifier_budget=1,
        acquisition_mode=spec.acquisition_mode,
        progress_ranking=spec.progress_ranking,
        eligibility_mode=spec.eligibility_mode,
    )
    return controller, store


def _protocol() -> MatchedProtocol:
    return MatchedProtocol(
        seed=5,
        candidate_count=4,
        verifier_budget=2,
        fallback_verifier_budget=1,
        beta=0.2,
        backup_smooth_weight=8.0,
        backup_noise_var_mult=3.0,
        backup_retreat_weight=1.0,
        rounds=1,
        episodes_per_gamma=1,
        episode_max_steps=1,
        expansion_temperature=1.0,
        nfe=1,
        ridge_lambda=0.01,
        prox_eta=0.1,
        learning_rate=1e-3,
        microbatch=2,
        solver_max_steps=1,
        solver_min_steps=1,
        update_norm_limit=0.2,
        relative_loss_tolerance=0.0,
        gradient_tolerance=0.0,
        audit_plans_per_context=2,
        audit_progress_threshold=0.1,
        eval_rollouts=2,
    )


def test_minus_afe_probabilities_are_uniform_while_real_sigma_is_logged() -> None:
    spec = ablation_spec(AblationArm.MINUS_AFE)

    def accept(state, plan, env, gamma, **kwargs):
        del kwargs
        return _verification(
            state, plan, env, gamma, in_bounds=True, socp_ok=True,
            progress=float(np.asarray(plan).sum()),
        )

    controller, store = _controller(spec, accept)
    first = controller.run_episode(make_ood_scene(), 0.5, seed=50, max_steps=1)
    second = controller.run_episode(make_ood_scene(), 0.5, seed=51, max_steps=1)
    for result in (first, second):
        trace = result.traces[0]
        np.testing.assert_allclose(trace.acquisition_probabilities, 0.25)
        assert np.all(np.isfinite(trace.candidate_sigmas))
        assert np.all(trace.candidate_sigmas > 0.0)
    # The real sigma is not replaced by the zero acquisition score.
    assert not np.allclose(second.traces[0].candidate_sigmas, 0.0)
    assert store.query_count == store.uncertainty.count == 4


def test_minus_progress_selects_first_verified_safe_not_max_progress() -> None:
    spec = ablation_spec(AblationArm.MINUS_PROGRESS)
    calls = 0

    def increasing_progress(state, plan, env, gamma, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return _verification(
            state, plan, env, gamma, in_bounds=True, socp_ok=True,
            progress=float(calls),
        )

    controller, store = _controller(spec, increasing_progress)
    result = controller.run_episode(make_ood_scene(), 0.5, seed=77, max_steps=1)
    assert [record.progress_value for record in store.records] == [1.0, 2.0]
    assert result.traces[0].selected_query_hash == store.records[0].query_hash
    assert store.records[0].executed
    assert not store.records[1].executed


def test_minus_socp_uses_bounds_view_but_retains_failed_actual_certificate() -> None:
    spec = ablation_spec(AblationArm.MINUS_SOCP)

    def reject_socp(state, plan, env, gamma, **kwargs):
        del kwargs
        return _verification(
            state, plan, env, gamma, in_bounds=True, socp_ok=False,
            progress=float(np.asarray(plan).sum()),
        )

    controller, store = _controller(spec, reject_socp)
    result = controller.run_episode(make_ood_scene(), 0.5, seed=91, max_steps=1)
    assert len(result.actions) == 1  # offline simulation trace only
    assert not result.runtime_safety_claim
    assert not result.traces[0].runtime_safety_claim
    assert result.traces[0].selected_query_hash is not None
    assert result.traces[0].selected_actual_full_safe is False
    assert all(record.safety.strict_bounds for record in store.records)
    assert all(not record.safety.socp_certified for record in store.records)
    assert all(not record.safe and not record.executed for record in store.records)

    view = training_view(store.records, spec)
    assert isinstance(view, OfflineBoundsReplayView)
    assert len(view) == 2
    assert all(not row.actual_socp_certified for row in view)
    assert all(not row.source_executed for row in view)

    def simple_loss(model, batch, *, generator):
        del generator
        target = torch.tensor(
            np.mean([np.asarray(row.plan).mean() for row in batch]),
            dtype=model.weight.dtype,
        )
        return (model.weight - target).square()

    solved = solve_proximal_update(
        controller.model,
        view,
        simple_loss,
        ProximalConfig(
            eta=0.1,
            learning_rate=1e-3,
            batch_size=2,
            max_steps=1,
            min_steps=1,
            update_norm_limit=0.2,
            relative_loss_tolerance=0.0,
            gradient_tolerance=0.0,
            seed=3,
        ),
    )
    assert solved.positive_count == 2


def test_controls_have_matched_declared_and_real_one_step_query_counts() -> None:
    protocol = assert_matched_protocols([_protocol(), _protocol(), _protocol()])
    assert protocol.scheduled_flow_query_budget == 2
    counts = []

    def accept(state, plan, env, gamma, **kwargs):
        del kwargs
        return _verification(
            state, plan, env, gamma, in_bounds=True, socp_ok=True, progress=0.1,
        )

    for arm in AblationArm:
        controller, store = _controller(ablation_spec(arm), accept)
        result = controller.run_episode(make_ood_scene(), 0.5, seed=123, max_steps=1)
        counts.append((result.verifier_calls, store.query_count, store.uncertainty.count))
    assert counts == [(2, 2, 2)] * 3


def test_matched_control_recipe_embeds_seed_and_backup_audit_protocol() -> None:
    protocol = _protocol()
    manifest = arm_manifest(ablation_spec(AblationArm.MINUS_AFE), protocol)
    assert manifest["matched_protocol"]["seed"] == 5
    assert manifest["matched_protocol"]["backup_smooth_weight"] == 8.0
    assert manifest["matched_protocol"]["backup_noise_var_mult"] == 3.0
    assert manifest["matched_protocol"]["backup_retreat_weight"] == 1.0
    assert manifest["matched_protocol"]["audit_plans_per_context"] == 2


def test_stage6_arm_requires_promoted_stage3_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"placeholder")
    args = SimpleNamespace(
        outdir=tmp_path / "out",
        device="cpu",
        checkpoint=checkpoint,
    )
    monkeypatch.setattr(
        stage6.HP, "load_hp", lambda *_args, **_kwargs: (object(), {})
    )

    def reject(_model, _payload):
        raise RuntimeError("promotion gate sentinel")

    monkeypatch.setattr(stage6, "require_promoted_fresh_pretrain", reject)
    with pytest.raises(RuntimeError, match="promotion gate sentinel"):
        stage6._run_arm(
            args,
            ablation_spec(AblationArm.MINUS_AFE),
            _protocol(),
            checkpoint_sha256="0" * 64,
            full_reference={},
        )
