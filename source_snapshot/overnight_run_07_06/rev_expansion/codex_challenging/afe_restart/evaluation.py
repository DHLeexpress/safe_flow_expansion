"""Ordinary-flow rollout evaluation, separate from verifier-assisted control."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch

from .dynamics import step_state
from .policy import sample_plans
from .scene import GIANT_CENTER, context_from_state


@dataclass(frozen=True)
class RolloutResult:
    gamma: float
    seed: int
    temperature: float
    states: np.ndarray
    actions: np.ndarray
    reached: bool
    collision: bool
    out_of_bounds: bool
    timeout: bool
    min_clearance_m: float
    path_length_m: float
    time_to_goal_s: float | None
    detour_mode: str

    @property
    def success(self) -> bool:
        return self.reached and not self.collision and not self.out_of_bounds


def detour_mode(path: np.ndarray, *, deadband: float = 0.05) -> str:
    """Global giant-obstacle homotopy: upper-left, lower-right, or unresolved."""
    positions = np.asarray(path, dtype=np.float64)
    near = np.linalg.norm(positions - GIANT_CENTER[None], axis=1) <= 1.9
    if not bool(near.any()):
        return "unresolved"
    score = float(np.median(positions[near, 1] - positions[near, 0]))
    if score > deadband:
        return "upper-left"
    if score < -deadband:
        return "lower-right"
    return "diagonal"


def local_plan_mode(
    state: np.ndarray,
    goal: np.ndarray,
    positions: np.ndarray,
    *,
    cross_track_threshold_m: float = 0.03,
) -> str:
    """Preregistered local valid-coverage bin for an H=10 audit plan."""
    origin = np.asarray(state, dtype=np.float64)[:2]
    direct = np.asarray(goal, dtype=np.float64)[:2] - origin
    displacement = np.asarray(positions, dtype=np.float64)[-1, :2] - origin
    scale = max(float(np.linalg.norm(direct)), 1.0e-12)
    signed_cross_track = float(direct[0] * displacement[1] - direct[1] * displacement[0]) / scale
    if signed_cross_track > cross_track_threshold_m:
        return "left-of-goal-ray"
    if signed_cross_track < -cross_track_threshold_m:
        return "right-of-goal-ray"
    return "goal-ray"


def _clearance(path: np.ndarray, env) -> tuple[float, np.ndarray]:
    obstacles = env.obstacles.detach().cpu().numpy()
    if len(obstacles) == 0:
        values = np.full(len(path), np.inf)
    else:
        values = (
            np.linalg.norm(path[:, None] - obstacles[None, :, :2], axis=2)
            - obstacles[None, :, 2]
            - float(env.r_robot)
        ).min(axis=1)
    return float(values.min()), values


@torch.inference_mode()
def rollout_ordinary_flow(
    model: torch.nn.Module,
    env,
    gamma: float,
    *,
    seed: int,
    temperature: float = 1.0,
    nfe: int = 8,
    max_steps: int | None = None,
    reach: float = 0.2,
) -> RolloutResult:
    """Evaluate the model distribution without sigma tilting or safety filter."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    state = env.x0.detach().cpu().numpy().astype(np.float64)
    goal = env.goal.detach().cpu().numpy().astype(np.float64)
    states = [state.copy()]
    actions: list[np.ndarray] = []
    collision = out_of_bounds = False
    limit = int(env.T if max_steps is None else max_steps)
    for _step in range(limit):
        context = context_from_state(state, goal, gamma, actions, env)
        plan = sample_plans(
            model,
            context,
            1,
            temperature=temperature,
            nfe=nfe,
            generator=generator,
        )[0]
        action = plan[0].astype(np.float64)
        state = step_state(state, action, float(env.dt))
        actions.append(action.astype(np.float32))
        states.append(state.copy())
        position = state[:2]
        out_of_bounds = bool(np.any(position < 0.0) or np.any(position > 5.0))
        _, instant = _clearance(position[None], env)
        collision = bool(instant[0] < 0.0)
        if out_of_bounds or collision or np.linalg.norm(position - goal) < reach:
            break
    path = np.asarray(states, dtype=np.float32)[:, :2]
    reached = bool(np.linalg.norm(path[-1] - goal) < reach)
    minimum, _ = _clearance(path, env)
    successful_time = (len(actions) * float(env.dt)) if reached and not collision and not out_of_bounds else None
    return RolloutResult(
        gamma=float(gamma),
        seed=int(seed),
        temperature=float(temperature),
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 2),
        reached=reached,
        collision=collision,
        out_of_bounds=out_of_bounds,
        timeout=not reached and not collision and not out_of_bounds and len(actions) >= limit,
        min_clearance_m=minimum,
        path_length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        time_to_goal_s=successful_time,
        detour_mode=detour_mode(path),
    )


def summarize_rollouts(results: Iterable[RolloutResult]) -> dict[str, object]:
    rows = list(results)
    if not rows:
        raise ValueError("cannot summarize an empty rollout set")
    successes = [row for row in rows if row.success]
    mode_counts: dict[str, int] = {}
    for row in successes:
        mode_counts[row.detour_mode] = mode_counts.get(row.detour_mode, 0) + 1
    return {
        "n": len(rows),
        "success_rate": sum(row.success for row in rows) / len(rows),
        "collision_rate": sum(row.collision for row in rows) / len(rows),
        "out_of_bounds_rate": sum(row.out_of_bounds for row in rows) / len(rows),
        "timeout_rate": sum(row.timeout for row in rows) / len(rows),
        "mean_min_clearance_m": float(np.mean([row.min_clearance_m for row in rows])),
        "mean_success_clearance_m": (
            float(np.mean([row.min_clearance_m for row in successes])) if successes else None
        ),
        "mean_time_to_goal_s": (
            float(np.mean([row.time_to_goal_s for row in successes])) if successes else None
        ),
        "mean_path_length_m": float(np.mean([row.path_length_m for row in rows])),
        "mode_counts_successes": dict(sorted(mode_counts.items())),
        "successful_mode_coverage": len(mode_counts),
    }
