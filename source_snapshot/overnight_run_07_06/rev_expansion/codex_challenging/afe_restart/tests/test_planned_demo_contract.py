from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from afe_restart.dynamics import step_state
from afe_restart.fallback import BackupProposal
from afe_restart.scene import verifier_spec_fingerprint
from afe_restart.schemas import QueryContext, query_content_hash
from afe_restart.stage2_planned_demos import DemoRunConfig, run_expert_rollout


class TinyEnvironment:
    def __init__(self, goal: tuple[float, float]) -> None:
        self.x0 = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)
        self.goal = torch.tensor(goal, dtype=torch.float32)
        self.obstacles = torch.empty((0, 3), dtype=torch.float32)
        self.r_robot = 0.1
        self.dt = 0.1


class FixedBackup:
    def __init__(self, plans: list[np.ndarray]) -> None:
        self.plans = plans
        self.calls = 0

    def propose(self, state, goal, env, gamma, *, seed, device):
        self.calls += 1
        return [
            BackupProposal(plan, f"proposal_{index}", True)
            for index, plan in enumerate(self.plans)
        ], {"proposal_count": len(self.plans)}


class KindBackup:
    def __init__(self, proposals: list[tuple[np.ndarray, str]]) -> None:
        self.proposals = proposals

    def propose(self, state, goal, env, gamma, *, seed, device):
        return [
            BackupProposal(plan, kind, True) for plan, kind in self.proposals
        ], {"proposal_count": len(self.proposals)}


def tiny_context(state, goal, gamma, controls, env) -> QueryContext:
    history = np.zeros((3, 2), dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32).reshape(-1, 2)
    if len(controls):
        history[-min(3, len(controls)) :] = controls[-3:]
    return QueryContext(
        grid=np.asarray([[state[0], state[1]]], dtype=np.float32),
        low5=np.asarray([*state, gamma], dtype=np.float32),
        hist=history,
        verifier_state=np.asarray(state, dtype=np.float64),
        verifier_spec_fingerprint=verifier_spec_fingerprint(env, goal),
    )


def verifier_with_progress(progress_by_action: dict[tuple[float, float], float], safe=True):
    seen: list[np.ndarray] = []

    def verify(state, plan, env, gamma, *, goal):
        seen.append(np.asarray(plan).copy())
        action = tuple(np.asarray(plan[0], dtype=float).round(6))
        progress = progress_by_action.get(action, -1.0)
        return SimpleNamespace(
            safe=bool(safe),
            in_bounds=bool(safe),
            socp_ok=bool(safe),
            bounds_margin_m=0.5,
            physical_clearance_m=0.4,
            face_margin_m=0.3,
            certificate_residual=0.2,
            certificate_worst_step=5,
            progress_m=progress,
            start_goal_distance_m=1.0,
            terminal_goal_distance_m=1.0 - progress,
        )

    return verify, seen


def test_exact_verified_plan_is_training_target_and_only_first_action_executes() -> None:
    lower_progress = np.full((10, 2), [0.2, 0.0], dtype=np.float32)
    higher_progress = np.full((10, 2), [0.0, 0.3], dtype=np.float32)
    expected_next = step_state(np.asarray([0.5, 0.5, 0.0, 0.0]), higher_progress[0])
    env = TinyEnvironment(tuple(expected_next[:2]))
    backup = FixedBackup([lower_progress, higher_progress])
    verify, seen = verifier_with_progress({(0.2, 0.0): 0.1, (0.0, 0.3): 0.5})
    config = DemoRunConfig(max_steps=1, reach_m=1.0e-6, max_proposals_per_step=2)

    episode = run_expert_rollout(
        env=env,
        gamma=0.4,
        seed=17,
        device=torch.device("cpu"),
        config=config,
        backup=backup,
        verify_fn=verify,
        context_fn=tiny_context,
    )

    assert episode["success"]
    assert len(seen) == 2
    assert np.array_equal(seen[0], lower_progress)
    assert np.array_equal(seen[1], higher_progress)
    assert np.array_equal(episode["training_plans"][0], higher_progress)
    assert np.array_equal(episode["executed_actions"][0], higher_progress[0])
    assert np.allclose(episode["states"][1], expected_next)
    # Executing the full open-loop window would produce a different state and
    # is therefore observably not what the receding-horizon generator did.
    full_window_state = np.asarray([0.5, 0.5, 0.0, 0.0])
    for action in higher_progress:
        full_window_state = step_state(full_window_state, action)
    assert not np.allclose(episode["states"][1], full_window_state)

    selected_index = int(episode["selected_query_indices"][0])
    context = episode["contexts"][0]
    identity = query_content_hash(context, 0.4, episode["training_plans"][0])
    assert episode["training_hashes"] == [identity]
    assert episode["query_hashes"][selected_index] == identity


def test_no_safe_plan_fails_closed_without_action_or_target() -> None:
    plan = np.full((10, 2), [1.0, 0.0], dtype=np.float32)
    env = TinyEnvironment((4.5, 4.5))
    backup = FixedBackup([plan])
    verify, seen = verifier_with_progress({(1.0, 0.0): 1.0}, safe=False)
    config = DemoRunConfig(max_steps=3, reach_m=0.1, max_proposals_per_step=1)

    episode = run_expert_rollout(
        env=env,
        gamma=0.7,
        seed=9,
        device=torch.device("cpu"),
        config=config,
        backup=backup,
        verify_fn=verify,
        context_fn=tiny_context,
    )

    assert len(seen) == 1
    assert not episode["success"]
    assert episode["dead_reason"] == "no_certified_plan"
    assert episode["steps"] == 0
    assert episode["training_plans"].shape == (0, 10, 2)
    assert episode["executed_actions"].shape == (0, 2)
    assert episode["states"].shape == (1, 4)
    assert np.array_equal(episode["states"][0], env.x0.numpy())


def test_raw_debug_rollout_cannot_bypass_safe_expert_smoothness_selection() -> None:
    expert = np.full((10, 2), [0.2, 0.0], dtype=np.float32)
    debug = np.empty((10, 2), dtype=np.float32)
    debug[::2] = [1.0, -1.0]
    debug[1::2] = [-1.0, 1.0]
    expected_next = step_state(np.asarray([0.5, 0.5, 0.0, 0.0]), expert[0])
    env = TinyEnvironment(tuple(expected_next[:2]))
    backup = KindBackup(
        [(expert, "weighted_mean"), (debug, "debug_candidate")]
    )
    verify, seen = verifier_with_progress(
        {(0.2, 0.0): 0.1, (1.0, -1.0): 0.9}
    )

    episode = run_expert_rollout(
        env=env,
        gamma=0.4,
        seed=18,
        device=torch.device("cpu"),
        config=DemoRunConfig(max_steps=1, reach_m=1.0e-6, max_proposals_per_step=2),
        backup=backup,
        verify_fn=verify,
        context_fn=tiny_context,
    )

    assert episode["success"]
    assert len(seen) == 2  # diagnostic proposal remains a real verifier query
    assert np.array_equal(episode["training_plans"][0], expert)
    selected = int(episode["selected_query_indices"][0])
    assert episode["query_kinds"][selected] == "weighted_mean"


def test_safe_debug_only_step_fails_closed_without_execution_or_target() -> None:
    debug = np.full((10, 2), [0.4, 0.0], dtype=np.float32)
    env = TinyEnvironment((4.5, 4.5))
    backup = KindBackup([(debug, "debug_candidate")])
    verify, seen = verifier_with_progress({(0.4, 0.0): 0.8}, safe=True)

    episode = run_expert_rollout(
        env=env,
        gamma=0.4,
        seed=19,
        device=torch.device("cpu"),
        config=DemoRunConfig(max_steps=1, reach_m=0.1, max_proposals_per_step=1),
        backup=backup,
        verify_fn=verify,
        context_fn=tiny_context,
    )

    assert len(seen) == 1  # it remains a fully observed audit query
    assert episode["dead_reason"] == "no_certified_cost_selected_plan"
    assert episode["steps"] == 0
    assert episode["training_plans"].shape == (0, 10, 2)
    assert episode["executed_actions"].shape == (0, 2)
