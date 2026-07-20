from __future__ import annotations

import numpy as np
import torch

from afe_restart.config import AFEConfig, FeatureConfig, SamplingConfig
from afe_restart.controller import PlannedWindowAFEController
from afe_restart.dynamics import rollout_plan
from afe_restart.fallback import BackupProposal
from afe_restart.scene import make_ood_scene
from afe_restart.store import VerificationStore
from afe_restart.uncertainty import CumulativeLinearUncertainty
from afe_restart.verifier import PlanVerification


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


class FixedFeatures:
    def encode(self, context, plans):
        del context
        plans = np.asarray(plans)
        rows = []
        for index, plan in enumerate(plans):
            z = np.zeros(32, dtype=np.float64)
            z[index % 32] = 1.0
            z[(index + 7) % 32] = 0.25 + abs(float(plan[0, 0]))
            z /= np.linalg.norm(z)
            rows.append(z)
        return np.asarray(rows)


class OneBackup:
    def __init__(self, plan):
        self.plan = np.asarray(plan, dtype=np.float32)

    def propose(self, *args, **kwargs):
        del args, kwargs
        return [BackupProposal(self.plan, "test_backup", None)], {"proposal_count": 1}


def verification(state, plan, env, gamma, *, safe, progress):
    states = rollout_plan(state, plan)
    positions = states[:, :2].copy()
    states.setflags(write=False)
    positions.setflags(write=False)
    d0 = float(np.linalg.norm(positions[0] - env.goal.numpy()))
    return PlanVerification(
        safe=safe,
        in_bounds=safe,
        socp_ok=safe,
        bounds_margin_m=0.1 if safe else -0.1,
        physical_clearance_m=0.2,
        face_margin_m=0.2 if safe else -np.inf,
        certificate_residual=0.1 if safe else -0.1,
        certificate_worst_step=10,
        progress_m=progress,
        start_goal_distance_m=d0,
        terminal_goal_distance_m=d0 - progress,
        gamma=gamma,
        states=states,
        positions=positions,
    )


def config() -> AFEConfig:
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


def make_controller(verifier_fn, backup) -> tuple[PlannedWindowAFEController, VerificationStore]:
    model = TinyFlow()
    store = VerificationStore(CumulativeLinearUncertainty(lambda_=0.01))
    controller = PlannedWindowAFEController(
        model,
        FixedFeatures(),
        store,
        config=config(),
        backup=backup,
        verifier_fn=verifier_fn,
        device="cpu",
        fallback_verifier_budget=1,
    )
    return controller, store


def test_flow_query_batch_identity_and_exact_first_action() -> None:
    def accept(state, plan, env, gamma, **kwargs):
        del kwargs
        return verification(state, plan, env, gamma, safe=True, progress=float(plan.sum()))

    controller, store = make_controller(accept, OneBackup(np.zeros((10, 2))))
    result = controller.run_episode(make_ood_scene(), 0.5, seed=8, max_steps=1)

    assert len(store.records) == store.uncertainty.count == 2
    assert store.batch_sizes == (2,)
    assert sum(record.executed for record in store.records) == 1
    executed = next(record for record in store.records if record.executed)
    np.testing.assert_array_equal(result.actions[0], executed.plan[0])
    assert executed.query_hash == executed.generated_hash == executed.verifier_input_hash
    assert result.traces[0].candidate_plans.shape == (4, 10, 2)
    assert not result.traces[0].fallback_used


def test_same_verifier_backup_and_fail_closed() -> None:
    calls = 0
    backup_plan = np.full((10, 2), 0.1, dtype=np.float32)

    def backup_only(state, plan, env, gamma, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        safe = calls == 3  # two flow verifier calls, then one backup call
        return verification(state, plan, env, gamma, safe=safe, progress=0.2 if safe else -0.1)

    controller, store = make_controller(backup_only, OneBackup(backup_plan))
    result = controller.run_episode(make_ood_scene(), 0.5, seed=2, max_steps=1)
    assert result.fallback_steps == 1
    assert not result.fail_closed
    np.testing.assert_array_equal(result.actions[0], backup_plan[0])
    assert store.records[-1].source.value == "safemppi_backup"
    assert store.records[-1].safe and store.records[-1].executed

    def reject(state, plan, env, gamma, **kwargs):
        del kwargs
        return verification(state, plan, env, gamma, safe=False, progress=-0.1)

    controller, store = make_controller(reject, OneBackup(backup_plan))
    result = controller.run_episode(make_ood_scene(), 0.5, seed=2, max_steps=1)
    assert result.fail_closed
    assert len(result.actions) == 0
    assert len(result.states) == 1
    assert store.query_count == store.uncertainty.count == 3
    assert not any(record.executed for record in store.records)

