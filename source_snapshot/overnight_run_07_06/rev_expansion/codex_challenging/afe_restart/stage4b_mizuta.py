#!/usr/bin/env python3
"""Stage 04B: verifier-free Mizuta/CFM-MPPI baseline on the clean OOD scene.

The generation and FlowMPPI equations follow ``reference/kazuki_baseline.py``.
Only task adapters change: the radius-1.2 restart scene, endpoint-free restart
contexts, the fresh Stage-03 checkpoint contract, and one collision radius per
obstacle.  Scientific rollouts always use source temperature 1.0.  Temperature
0.5 is saved separately for gallery rendering and never supplies metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import grid_hp_expt as HP

from .config import GAMMAS
from .deps import sha256_file
from .dynamics import step_state
from .evaluation import detour_mode
from .policy import (
    context_tensors,
    model_state_hash,
    require_promoted_fresh_pretrain,
)
from .scene import GIANT_CENTER, GOAL, START, context_from_state, make_ood_scene


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "stage_results/03_pretrain/data/checkpoint_best.pt"
DEFAULT_OUTDIR = PACKAGE_ROOT / "stage_results/04b_mizuta"
REFERENCE_SOURCE = PACKAGE_ROOT.parent / "reference/kazuki_baseline.py"
SWEEP_SCHEMA = "afe_mizuta_low_guidance_sweep_v1"
ROLLOUT_SCHEMA = "afe_mizuta_rollouts_v1"
MANIFEST_SCHEMA = "afe_mizuta_stage4b_v1"
ALGORITHM_CONTRACT_SCHEMA = "afe_mizuta_reference_adapter_v1"
SCIENTIFIC_TEMPERATURE = 1.0
GALLERY_TEMPERATURE = 0.5
GUIDANCE_EQUATION = (
    "v_guided = v_base + goal_guidance_coef * normalized_goal_gradient + "
    "safe_coef * normalized_cbf_gradient * markup"
)


@dataclass(frozen=True)
class MizutaConfig:
    """Faithful CFM-MPPI constants plus one bounded guidance setting."""

    tag: str
    safe_coef: float
    collision_weight: float
    goal_weight: float = 2.0
    goal_guidance_coef: float = 0.1
    cbf_alpha: float = 1.0
    worst_obstacles: int = 5
    collision_beta: float = 20.0
    collision_time_base: float = 1.0
    collision_time_decay: float = 0.99
    control_consistency_weight: float = 0.1
    mppi_lambda: float = 0.1
    mppi_sigma: float = 0.2
    collision_margin: float = 0.05
    markup: float = 1.01
    nfe: int = 8
    warm_tau: float = 0.75
    n_samples: int = 200
    n_elite: int = 10
    n_copies: int = 200

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError("Mizuta config tag cannot be empty")
        nonnegative = (
            self.safe_coef,
            self.collision_weight,
            self.goal_weight,
            self.goal_guidance_coef,
            self.collision_margin,
            self.collision_time_base,
            self.control_consistency_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in nonnegative):
            raise ValueError("Mizuta weights and collision margin must be finite and nonnegative")
        positive = (
            self.cbf_alpha,
            self.collision_beta,
            self.mppi_lambda,
            self.mppi_sigma,
            self.markup,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("Mizuta scale parameters must be finite and positive")
        if self.worst_obstacles <= 0 or self.nfe <= 0:
            raise ValueError("worst_obstacles and nfe must be positive")
        if not 0.0 < self.collision_time_decay <= 1.0:
            raise ValueError("collision_time_decay must lie in (0,1]")
        if not 0.0 < self.warm_tau < 1.0:
            raise ValueError("warm_tau must lie in (0,1)")
        if min(self.n_samples, self.n_elite, self.n_copies) <= 0:
            raise ValueError("Mizuta sample counts must be positive")
        if self.n_elite > self.n_samples:
            raise ValueError("n_elite cannot exceed n_samples")

    @property
    def low_guidance_admissible(self) -> bool:
        return bool(
            0.0 < self.safe_coef <= 0.04
            and self.collision_weight <= 4.0
            and self.goal_guidance_coef <= 0.2
            and math.isclose(self.goal_weight, 2.0)
            and math.isclose(self.cbf_alpha, 1.0)
            and self.worst_obstacles == 5
            and math.isclose(self.collision_beta, 20.0)
            and math.isclose(self.collision_time_base, 1.0)
            and math.isclose(self.collision_time_decay, 0.99)
            and math.isclose(self.control_consistency_weight, 0.1)
            and math.isclose(self.mppi_lambda, 0.1)
            and math.isclose(self.mppi_sigma, 0.2)
            and math.isclose(self.collision_margin, 0.05)
            and math.isclose(self.markup, 1.01)
            and self.nfe == 8
            and math.isclose(self.warm_tau, 0.75)
            and self.n_samples == 200
            and self.n_elite == 10
            and self.n_copies == 200
        )


LOW_GUIDANCE_SWEEP = (
    MizutaConfig("lg005", safe_coef=0.005, collision_weight=0.5, goal_guidance_coef=0.025),
    MizutaConfig("lg010", safe_coef=0.010, collision_weight=1.0, goal_guidance_coef=0.050),
    MizutaConfig("lg020", safe_coef=0.020, collision_weight=2.0, goal_guidance_coef=0.100),
    MizutaConfig("lg040", safe_coef=0.040, collision_weight=4.0, goal_guidance_coef=0.200),
)
TUNING_GAMMAS = (0.1, 0.5, 1.0)


def algorithm_contract() -> dict[str, Any]:
    """Machine-readable boundary between the reference port and task adapters."""

    return {
        "schema_version": ALGORITHM_CONTRACT_SCHEMA,
        "reference_source": str(REFERENCE_SOURCE.resolve()),
        "guidance_equation": GUIDANCE_EQUATION,
        "goal_guidance_term_count": 1,
        "safety_guidance_term_count": 1,
        "global_gradient_norm_matching": True,
        "flowmppi_top_elite_refit": True,
        "generation_verifier_free": True,
        "generation_socp_free": True,
        "safety_filter_used": False,
        "scene_adapter": "radius-1.2 restart OOD scene",
        "context_adapter": "live state + live executed history + actual gamma",
        "collision_radius_adapter": "one true radius per obstacle + robot + 0.05m",
        "preregistered_low_guidance_configs": [
            asdict(config) for config in LOW_GUIDANCE_SWEEP
        ],
        "fixed_reference_fields": [
            "cbf_alpha",
            "worst_obstacles",
            "collision_beta",
            "collision_time_base",
            "collision_time_decay",
            "control_consistency_weight",
            "mppi_lambda",
            "mppi_sigma",
            "collision_margin",
            "markup",
            "nfe",
            "warm_tau",
            "n_samples",
            "n_elite",
            "n_copies",
        ],
        "bounded_adapter_fields": [
            "safe_coef",
            "collision_weight",
            "goal_weight",
            "goal_guidance_coef",
        ],
        "scientific_temperature": SCIENTIFIC_TEMPERATURE,
        "gallery_temperature": GALLERY_TEMPERATURE,
        "legacy_artifact_reuse": False,
        "resume_supported": False,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_sha256(value: Any) -> str:
    canonical = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
    return digest


def _require_clean_output_dir(outdir: Path) -> None:
    """Reject resume/stale-artifact mixing without deleting user files."""

    if not outdir.exists():
        return
    retained = [item for item in outdir.rglob("*") if item.is_file() or item.is_symlink()]
    if retained:
        preview = ", ".join(str(item.relative_to(outdir)) for item in retained[:3])
        raise RuntimeError(
            "Stage 04B requires an empty output directory and never resumes or "
            f"retains legacy artifacts; found {preview}"
        )


def _configure_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.split(",")[0].strip() != "1" or device.index not in (None, 0):
        raise RuntimeError(
            "Stage 04B is assigned to physical GPU 1: launch with "
            "CUDA_VISIBLE_DEVICES=1 and use --device cuda:0"
        )
    torch.cuda.set_device(device)
    return device


def collision_radii(env, margin: float) -> np.ndarray:
    """Return each obstacle's own radius plus robot radius and tuning margin."""

    if margin < 0.0 or not math.isfinite(float(margin)):
        raise ValueError("collision margin must be finite and nonnegative")
    obstacles = env.obstacles.detach().cpu().numpy()
    radii = obstacles[:, 2].astype(np.float64) + float(env.r_robot) + float(margin)
    if radii.ndim != 1 or len(radii) != len(obstacles) or not np.isfinite(radii).all():
        raise ValueError("invalid per-obstacle collision radii")
    return radii


def describe_ood_scene(env) -> dict[str, Any]:
    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64)
    giant_matches = np.flatnonzero(
        np.linalg.norm(obstacles[:, :2] - GIANT_CENTER.astype(np.float64)[None], axis=1)
        <= 1.0e-7
    )
    if len(giant_matches) != 1:
        raise RuntimeError(f"expected one giant obstacle at {GIANT_CENTER.tolist()}")
    giant_index = int(giant_matches[0])
    if not math.isclose(float(obstacles[giant_index, 2]), 1.2, abs_tol=1.0e-7):
        raise RuntimeError("Stage 04B requires the radius-1.2 giant OOD scene")
    payload = {
        "name": "radius_1.2_giant_obstacle_OOD",
        "start": np.asarray(env.x0.detach().cpu().numpy())[:2].tolist(),
        "goal": np.asarray(env.goal.detach().cpu().numpy())[:2].tolist(),
        "robot_radius": float(env.r_robot),
        "obstacles": obstacles.tolist(),
        "giant_index": giant_index,
        "giant_center": GIANT_CENTER.tolist(),
        "giant_radius": float(obstacles[giant_index, 2]),
        "per_obstacle_radius_model": True,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def double_integrator_rollout(
    state: np.ndarray | torch.Tensor, controls: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable DI rollout for controls shaped ``[B,H,2]``."""

    if controls.ndim != 3 or controls.shape[-1] != 2:
        raise ValueError(f"controls must have shape [B,H,2], got {tuple(controls.shape)}")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    batch, horizon, _ = controls.shape
    initial = torch.as_tensor(state, dtype=controls.dtype, device=controls.device)
    if initial.shape != (4,) or not bool(torch.isfinite(initial).all()):
        raise ValueError("state must be finite with shape [4]")
    position = initial[:2].expand(batch, 2).clone()
    velocity = initial[2:].expand(batch, 2).clone()
    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    for step in range(horizon):
        action = controls[:, step]
        position = position + dt * velocity + 0.5 * dt * dt * action
        velocity = velocity + dt * action
        positions.append(position)
        velocities.append(velocity)
    return torch.stack(positions, dim=1), torch.stack(velocities, dim=1)


def cbf_reward(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radii: torch.Tensor | np.ndarray,
    config: MizutaConfig,
) -> torch.Tensor:
    """Mizuta CBF reward using the five worst per-obstacle violations."""

    if positions.shape != velocities.shape or positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions and velocities must share shape [B,H,2]")
    if obstacle_xy.ndim != 2 or obstacle_xy.shape[1] != 2 or len(obstacle_xy) == 0:
        raise ValueError("obstacle_xy must be nonempty with shape [O,2]")
    radii = torch.as_tensor(
        obstacle_radii, dtype=positions.dtype, device=positions.device
    ).reshape(-1)
    if len(radii) != len(obstacle_xy):
        raise ValueError("one collision radius is required per obstacle")
    delta = positions.unsqueeze(2) - obstacle_xy[None, None]
    barrier = (delta**2).sum(dim=-1) - radii[None, None] ** 2
    barrier_dot = 2.0 * (delta * velocities.unsqueeze(2)).sum(dim=-1)
    cbf = torch.clamp(barrier_dot + config.cbf_alpha * barrier, max=0.0)
    k = min(config.worst_obstacles, len(obstacle_xy))
    worst = torch.topk(cbf, k=k, dim=2, largest=False).values
    weights = torch.arange(
        k, 0, -1, dtype=positions.dtype, device=positions.device
    )[None, None]
    return (worst * weights).sum(dim=(1, 2))


def goal_reward(positions: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return -torch.linalg.vector_norm(positions[:, -1] - goal[None], dim=1)


def stage_cost_batch(
    positions: torch.Tensor,
    controls: torch.Tensor,
    goal: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radii: torch.Tensor | np.ndarray,
    config: MizutaConfig,
    previous_controls: torch.Tensor | None = None,
) -> torch.Tensor:
    """Faithful FlowMPPI running, terminal, proximity, and consistency cost."""

    if positions.shape != controls.shape or positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions and controls must share shape [B,H,2]")
    radii = torch.as_tensor(
        obstacle_radii, dtype=positions.dtype, device=positions.device
    ).reshape(-1)
    if len(radii) != len(obstacle_xy):
        raise ValueError("one collision radius is required per obstacle")
    horizon = positions.shape[1]
    goal_distance = torch.linalg.vector_norm(positions - goal[None, None], dim=2)
    distances = torch.linalg.vector_norm(
        positions.unsqueeze(2) - obstacle_xy[None, None], dim=3
    )
    proximity = torch.clamp(
        torch.exp(-config.collision_beta * (distances - radii[None, None])),
        max=1.0,
    ).sum(dim=2)
    time_weight = config.collision_weight * (
        config.collision_time_base
        + config.collision_time_decay
        ** torch.arange(horizon, dtype=positions.dtype, device=positions.device)
    )[None]
    cost = (config.goal_weight * goal_distance + time_weight * proximity).sum(dim=1)
    cost = cost + config.goal_weight * torch.linalg.vector_norm(
        positions[:, -1] - goal[None], dim=1
    )
    if previous_controls is not None:
        if previous_controls.shape != controls.shape[1:]:
            raise ValueError("previous_controls must have shape [H,2]")
        cost = cost + config.control_consistency_weight * (
            (controls - previous_controls[None]) ** 2
        ).sum(dim=(1, 2))
    return cost


def _ode_times(config: MizutaConfig, *, warm: bool) -> tuple[float, ...]:
    full = tuple(index / config.nfe for index in range(config.nfe + 1))
    if not warm:
        return full
    selected = tuple(value for value in full if value >= config.warm_tau - 1.0e-12)
    if len(selected) < 2:
        raise ValueError("warm_tau leaves fewer than two ODE knots")
    return selected


def guided_generate(
    policy: torch.nn.Module,
    encoded_context: torch.Tensor,
    state: np.ndarray,
    goal: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radii: torch.Tensor,
    dt: float,
    latent: torch.Tensor,
    times: Sequence[float],
    config: MizutaConfig,
) -> torch.Tensor:
    """Guided Euler flow from Mizuta ``run_CFM`` with global norm matching."""

    count = latent.shape[0]
    horizon = int(policy.T)
    if latent.shape != (count, int(policy.d)) or int(policy.d) != 2 * horizon:
        raise ValueError("latent/policy dimensions do not encode an Hx2 control window")
    if len(times) < 2 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("ODE times must contain at least two increasing knots")
    context = policy._expand_ctx(encoded_context, count)
    markup = (
        config.markup
        ** torch.arange(
            horizon - 1, -1, -1, dtype=latent.dtype, device=latent.device
        )
    )[None, :, None]
    value = latent
    for left, right in zip(times, times[1:]):
        tau_value = float(left)
        tau = torch.full(
            (count,), tau_value, dtype=value.dtype, device=value.device
        ).clamp(1.0e-4, 1.0)
        with torch.no_grad():
            velocity_field = policy(value, tau, context)
        endpoint = (
            value + (1.0 - tau_value) * velocity_field
        ).detach().requires_grad_(True)
        controls = endpoint.reshape(count, horizon, 2) * float(policy.u_max)
        positions, velocities = double_integrator_rollout(state, controls, dt)
        cbf_total = cbf_reward(
            positions, velocities, obstacle_xy, obstacle_radii, config
        ).sum()
        goal_total = goal_reward(positions, goal).sum()
        cbf_gradient = torch.autograd.grad(
            cbf_total, endpoint, retain_graph=True
        )[0]
        goal_gradient = torch.autograd.grad(goal_total, endpoint)[0]
        field_norm = torch.linalg.vector_norm(velocity_field)
        cbf_gradient = cbf_gradient * field_norm / (
            torch.linalg.vector_norm(cbf_gradient) + 1.0e-8
        )
        goal_gradient = goal_gradient * field_norm / (
            torch.linalg.vector_norm(goal_gradient) + 1.0e-8
        )
        guidance = (
            config.goal_guidance_coef * goal_gradient.reshape(count, horizon, 2)
            + config.safe_coef * cbf_gradient.reshape(count, horizon, 2) * markup
        )
        guided_field = velocity_field + guidance.reshape(count, -1)
        value = value + (float(right) - tau_value) * guided_field
    return value.detach()


@torch.no_grad()
def flow_mppi_refine(
    policy: torch.nn.Module,
    state: np.ndarray,
    goal: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radii: torch.Tensor,
    dt: float,
    generated_controls: torch.Tensor,
    previous_controls: torch.Tensor | None,
    config: MizutaConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Top-elite perturb/refit step from Mizuta FlowMPPI."""

    positions, _ = double_integrator_rollout(state, generated_controls, dt)
    costs = stage_cost_batch(
        positions,
        generated_controls,
        goal,
        obstacle_xy,
        obstacle_radii,
        config,
        previous_controls,
    )
    elite_count = min(config.n_elite, len(generated_controls))
    elite_indices = torch.topk(costs, k=elite_count, largest=False).indices
    elites = generated_controls[elite_indices]
    perturbed = elites.repeat_interleave(config.n_copies, dim=0)
    noise = torch.randn(
        perturbed.shape,
        dtype=perturbed.dtype,
        device=perturbed.device,
        generator=generator,
    )
    perturbed = (perturbed + config.mppi_sigma * noise).clamp(
        -float(policy.u_max), float(policy.u_max)
    )
    perturbed_positions, _ = double_integrator_rollout(state, perturbed, dt)
    perturbed_costs = stage_cost_batch(
        perturbed_positions,
        perturbed,
        goal,
        obstacle_xy,
        obstacle_radii,
        config,
        previous_controls,
    ).reshape(elite_count, config.n_copies)
    baseline = perturbed_costs.min(dim=1, keepdim=True).values
    weights = torch.softmax(
        -(perturbed_costs - baseline) / config.mppi_lambda, dim=1
    )
    refit = (
        weights[:, :, None, None]
        * perturbed.reshape(elite_count, config.n_copies, *perturbed.shape[1:])
    ).sum(dim=1)
    refit_positions, _ = double_integrator_rollout(state, refit, dt)
    refit_costs = stage_cost_batch(
        refit_positions,
        refit,
        goal,
        obstacle_xy,
        obstacle_radii,
        config,
        previous_controls,
    )
    return refit[int(torch.argmin(refit_costs))]


def transition_clearance(
    state: np.ndarray,
    action: np.ndarray,
    env,
) -> tuple[float, bool]:
    """Exact continuous-transition clearance against every true obstacle radius."""

    current = np.asarray(state, dtype=np.float64)
    control = np.asarray(action, dtype=np.float64)
    if current.shape != (4,) or control.shape != (2,):
        raise ValueError("transition state/action must have shapes [4] and [2]")
    dt = float(env.dt)
    candidate_times = {0.0, dt}
    for axis in range(2):
        if abs(control[axis]) > 1.0e-15:
            turning = -current[axis + 2] / control[axis]
            if 0.0 < turning < dt:
                candidate_times.add(float(turning))
    bound_times = np.asarray(sorted(candidate_times), dtype=np.float64)
    bound_points = (
        current[None, :2]
        + bound_times[:, None] * current[None, 2:]
        + 0.5 * bound_times[:, None] ** 2 * control[None]
    )
    in_bounds = bool(((bound_points >= 0.0) & (bound_points <= 5.0)).all())

    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64)
    minimum = math.inf
    quadratic = 0.5 * control
    for obstacle in obstacles:
        offset = current[:2] - obstacle[:2]
        velocity = current[2:]
        # Stationary points of ||offset + velocity*t + quadratic*t^2||^2.
        coefficients = np.asarray(
            (
                2.0 * float(quadratic @ quadratic),
                3.0 * float(velocity @ quadratic),
                float(2.0 * (offset @ quadratic) + velocity @ velocity),
                float(offset @ velocity),
            ),
            dtype=np.float64,
        )
        nonzero = np.flatnonzero(np.abs(coefficients) > 1.0e-15)
        roots: list[float] = []
        if len(nonzero):
            for root in np.roots(coefficients[int(nonzero[0]) :]):
                if abs(float(root.imag)) <= 1.0e-10 and 0.0 < float(root.real) < dt:
                    roots.append(float(root.real))
        times = np.asarray((0.0, dt, *roots), dtype=np.float64)
        points = (
            offset[None]
            + times[:, None] * velocity[None]
            + times[:, None] ** 2 * quadratic[None]
        )
        distance = float(np.linalg.norm(points, axis=1).min())
        minimum = min(minimum, distance - float(obstacle[2]) - float(env.r_robot))
    return float(minimum), in_bounds


def deploy_mizuta(
    policy: torch.nn.Module,
    env,
    gamma: float,
    *,
    config: MizutaConfig,
    seed: int,
    temperature: float,
    max_steps: int,
    reach_m: float,
) -> dict[str, Any]:
    """Run one verifier-free receding-horizon CFM-MPPI episode."""

    if temperature <= 0.0 or max_steps <= 0 or reach_m <= 0.0:
        raise ValueError("temperature, max_steps, and reach_m must be positive")
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float32)
    obstacle_xy = torch.as_tensor(obstacles[:, :2], dtype=torch.float32, device=device)
    radii_np = collision_radii(env, config.collision_margin)
    obstacle_radii = torch.as_tensor(radii_np, dtype=torch.float32, device=device)
    goal_np = env.goal.detach().cpu().numpy().astype(np.float64)
    goal = torch.as_tensor(goal_np, dtype=torch.float32, device=device)
    state = env.x0.detach().cpu().numpy().astype(np.float64)
    actions: list[np.ndarray] = []
    path: list[np.ndarray] = [state[:2].astype(np.float32)]
    minimum_clearance, _ = transition_clearance(
        state, np.zeros(2, dtype=np.float64), env
    )
    collision = False
    out_of_bounds = False
    reached = float(np.linalg.norm(state[:2] - goal_np)) < reach_m
    previous_latent: torch.Tensor | None = None
    previous_controls: torch.Tensor | None = None
    started = time.perf_counter()
    policy.eval()
    for _step in range(max_steps):
        if reached or collision or out_of_bounds:
            break
        query_context = context_from_state(state, goal_np, float(gamma), actions, env)
        grid, low5, history = context_tensors(query_context, device)
        with torch.no_grad():
            encoded = policy.ctx_from(grid, low5, history).squeeze(0)
        if previous_latent is None:
            latent = temperature * torch.randn(
                config.n_samples,
                int(policy.d),
                dtype=encoded.dtype,
                device=device,
                generator=generator,
            )
            times = _ode_times(config, warm=False)
        else:
            fresh = temperature * torch.randn(
                config.n_samples,
                int(policy.d),
                dtype=encoded.dtype,
                device=device,
                generator=generator,
            )
            latent = config.warm_tau * previous_latent[None].expand_as(fresh) + (
                1.0 - config.warm_tau
            ) * fresh
            times = _ode_times(config, warm=True)
        endpoint = guided_generate(
            policy,
            encoded,
            state,
            goal,
            obstacle_xy,
            obstacle_radii,
            float(env.dt),
            latent,
            times,
            config,
        )
        generated = (
            endpoint.reshape(config.n_samples, int(policy.T), 2)
            * float(policy.u_max)
        ).clamp(-float(policy.u_max), float(policy.u_max))
        selected = flow_mppi_refine(
            policy,
            state,
            goal,
            obstacle_xy,
            obstacle_radii,
            float(env.dt),
            generated,
            previous_controls,
            config,
            generator,
        )
        action = selected[0].detach().cpu().numpy().astype(np.float64)
        segment_margin, segment_in_bounds = transition_clearance(state, action, env)
        next_state = step_state(state, action, dt=float(env.dt))
        minimum_clearance = min(minimum_clearance, segment_margin)
        collision = segment_margin < 0.0
        out_of_bounds = not segment_in_bounds
        actions.append(action.astype(np.float32))
        path.append(next_state[:2].astype(np.float32))
        state = next_state
        reached = float(np.linalg.norm(state[:2] - goal_np)) < reach_m
        shifted = torch.cat((selected[1:], selected[-1:]), dim=0)
        previous_latent = (shifted / float(policy.u_max)).reshape(-1).detach()
        previous_controls = shifted.detach()
    path_array = np.asarray(path, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32).reshape(-1, 2)
    success = bool(reached and not collision and not out_of_bounds)
    timeout = bool(
        not success and not collision and not out_of_bounds and len(actions) >= max_steps
    )
    failure_reason = (
        None
        if success
        else "collision"
        if collision
        else "out_of_bounds"
        if out_of_bounds
        else "timeout"
    )
    return {
        "method": "CFM-MPPI* / Mizuta low guidance",
        "gamma": float(gamma),
        "seed": int(seed),
        "temperature": float(temperature),
        "config_tag": config.tag,
        "path": path_array,
        "actions": action_array,
        "success": success,
        "reached": bool(reached),
        "collision": bool(collision),
        "out_of_bounds": bool(out_of_bounds),
        "timeout": timeout,
        "failure_reason": failure_reason,
        "min_clearance_m": float(minimum_clearance),
        "path_length_m": float(
            np.linalg.norm(np.diff(path_array.astype(np.float64), axis=0), axis=1).sum()
        ),
        "endpoint_distance_m": float(np.linalg.norm(path_array[-1] - goal_np)),
        "goal_progress_m": float(
            np.linalg.norm(path_array[0] - goal_np) - np.linalg.norm(path_array[-1] - goal_np)
        ),
        "rollout_duration_s": len(action_array) * float(env.dt),
        "time_to_goal_s": len(action_array) * float(env.dt) if success else None,
        "wall_time_s": time.perf_counter() - started,
        "detour_mode": detour_mode(path_array),
        "verifier_calls": 0,
        "safety_filter_used": False,
        "obstacle_radius_model": "per-obstacle",
    }


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> dict[str, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson counts require 0 <= successes <= positive trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Wilson confidence must lie in (0,1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    probability = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (probability + z_squared / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z_squared / (4.0 * trials**2)
    ) / denominator
    return {
        "confidence": float(confidence),
        "low": max(0.0, center - half),
        "high": min(1.0, center + half),
    }


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("cannot summarize empty Mizuta rollouts")
    count = len(records)
    success_rows = [row for row in records if bool(row["success"])]
    successes = len(success_rows)
    collisions = sum(bool(row["collision"]) for row in records)
    out_of_bounds = sum(bool(row["out_of_bounds"]) for row in records)
    timeouts = sum(bool(row["timeout"]) for row in records)
    modes: dict[str, int] = {}
    failures: dict[str, int] = {}
    for row in success_rows:
        mode = str(row["detour_mode"])
        modes[mode] = modes.get(mode, 0) + 1
    for row in records:
        if not bool(row["success"]):
            reason = str(row.get("failure_reason", "unknown"))
            failures[reason] = failures.get(reason, 0) + 1
    return {
        "n": count,
        "successes": successes,
        "success_rate": successes / count,
        "success_rate_wilson_95": wilson_interval(successes, count),
        "collisions": collisions,
        "collision_rate": collisions / count,
        "collision_rate_wilson_95": wilson_interval(collisions, count),
        "out_of_bounds_count": out_of_bounds,
        "out_of_bounds_rate": out_of_bounds / count,
        "out_of_bounds_rate_wilson_95": wilson_interval(out_of_bounds, count),
        "timeouts": timeouts,
        "timeout_rate": timeouts / count,
        "timeout_rate_wilson_95": wilson_interval(timeouts, count),
        "mean_min_clearance_m": float(
            np.mean([float(row["min_clearance_m"]) for row in records])
        ),
        "mean_success_clearance_m": (
            float(np.mean([float(row["min_clearance_m"]) for row in success_rows]))
            if success_rows
            else None
        ),
        "mean_time_to_goal_s": (
            float(np.mean([float(row["time_to_goal_s"]) for row in success_rows]))
            if success_rows
            else None
        ),
        "mean_rollout_duration_s": float(
            np.mean([float(row["rollout_duration_s"]) for row in records])
        ),
        "mean_path_length_m": float(
            np.mean([float(row["path_length_m"]) for row in records])
        ),
        "mean_goal_progress_m": float(
            np.mean([float(row["goal_progress_m"]) for row in records])
        ),
        "mode_counts_successes": dict(sorted(modes.items())),
        "successful_mode_coverage": len(
            {mode for mode in modes if mode in ("upper-left", "lower-right")}
        ),
        "failure_reason_counts": dict(sorted(failures.items())),
    }


def summarize_per_gamma(
    rows: Sequence[Mapping[str, Any]], gammas: Sequence[float]
) -> dict[str, dict[str, Any]]:
    return {
        f"{float(gamma):g}": summarize_rows(
            row for row in rows if float(row["gamma"]) == float(gamma)
        )
        for gamma in gammas
    }


def run_fixed_rollouts(
    policy: torch.nn.Module,
    *,
    config: MizutaConfig,
    gammas: Sequence[float],
    repetitions: int,
    seed0: int,
    temperature: float,
    max_steps: int,
    reach_m: float,
) -> list[dict[str, Any]]:
    """Run exactly ``repetitions`` independent attempted episodes per gamma."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    rows: list[dict[str, Any]] = []
    for gamma_index, gamma in enumerate(gammas):
        for repetition in range(repetitions):
            # A compact contiguous range makes uniqueness independent of the
            # requested repetition count.  Stage-level offsets keep tuning,
            # scientific evaluation, and gallery streams disjoint.
            seed = int(seed0 + gamma_index * repetitions + repetition)
            env = make_ood_scene(radius=1.2)
            row = deploy_mizuta(
                policy,
                env,
                float(gamma),
                config=config,
                seed=seed,
                temperature=temperature,
                max_steps=max_steps,
                reach_m=reach_m,
            )
            row["repetition"] = repetition
            rows.append(row)
            print(
                f"[mizuta {config.tag} T={temperature:g} g={gamma:g} "
                f"{repetition + 1}/{repetitions}] {row['failure_reason'] or 'success'} "
                f"steps={len(row['actions'])} clearance={row['min_clearance_m']:.3f}",
                flush=True,
            )
    expected = len(gammas) * repetitions
    if len(rows) != expected:
        raise RuntimeError("fixed-count Mizuta evaluation dropped attempted rollouts")
    if len({int(row["seed"]) for row in rows}) != expected:
        raise RuntimeError("fixed-count Mizuta evaluation reused an episode seed")
    return rows


def sweep_summary(
    configs: Sequence[MizutaConfig], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in configs:
        if not config.low_guidance_admissible:
            raise ValueError(f"sweep config {config.tag!r} exceeds low-guidance bounds")
        selected = [row for row in rows if row["config_tag"] == config.tag]
        metrics = summarize_rows(selected)
        output.append({"config": asdict(config), "metrics": metrics})
    return output


def select_sweep_config(rows: Sequence[Mapping[str, Any]]) -> MizutaConfig:
    if not rows:
        raise ValueError("cannot select from an empty Mizuta sweep")

    def score(row: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = row["metrics"]
        config = row["config"]
        return (
            float(metrics["success_rate"]),
            -float(metrics["collision_rate"]),
            float(metrics["mean_goal_progress_m"]),
            float(metrics["successful_mode_coverage"]),
            -float(config["safe_coef"]),
        )

    selected = max(rows, key=score)
    config = MizutaConfig(**dict(selected["config"]))
    if not config.low_guidance_admissible:
        raise RuntimeError("selected Mizuta config is outside the preregistered low-guidance bound")
    return config


def build_rollout_artifact(
    *,
    role: str,
    temperature: float,
    config: MizutaConfig,
    rows: Sequence[Mapping[str, Any]],
    gammas: Sequence[float],
    repetitions: int,
    checkpoint_file_sha256: str,
    checkpoint_state_sha256: str,
    checkpoint_config_sha256: str,
    scene_fingerprint_sha256: str,
    reference_source_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    if role not in ("scientific", "gallery_only"):
        raise ValueError("rollout artifact role must be scientific or gallery_only")
    required_temperature = (
        SCIENTIFIC_TEMPERATURE if role == "scientific" else GALLERY_TEMPERATURE
    )
    if float(temperature) != required_temperature:
        raise ValueError(
            f"{role} Mizuta artifact requires temperature {required_temperature:g}"
        )
    if repetitions <= 0 or len(rows) != len(gammas) * repetitions:
        raise ValueError("rollout artifact does not contain the fixed count for every gamma")
    checkpoint_file_sha256 = _require_sha256(
        checkpoint_file_sha256, name="checkpoint_file_sha256"
    )
    checkpoint_state_sha256 = _require_sha256(
        checkpoint_state_sha256, name="checkpoint_state_sha256"
    )
    checkpoint_config_sha256 = _require_sha256(
        checkpoint_config_sha256, name="checkpoint_config_sha256"
    )
    scene_fingerprint_sha256 = _require_sha256(
        scene_fingerprint_sha256, name="scene_fingerprint_sha256"
    )
    reference_source_sha256 = _require_sha256(
        reference_source_sha256, name="reference_source_sha256"
    )
    implementation_sha256 = _require_sha256(
        implementation_sha256, name="implementation_sha256"
    )
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("rollout artifact episode seeds must be unique")
    per_gamma = summarize_per_gamma(rows, gammas)
    for gamma in gammas:
        key = f"{float(gamma):g}"
        selected = [row for row in rows if float(row["gamma"]) == float(gamma)]
        if len(selected) != repetitions:
            raise ValueError(f"gamma={key} does not have exactly {repetitions} attempts")
        if {int(row.get("repetition", -1)) for row in selected} != set(range(repetitions)):
            raise ValueError(f"gamma={key} does not contain each fixed repetition exactly once")
        if any(float(row["temperature"]) != required_temperature for row in selected):
            raise ValueError(f"gamma={key} contains a rollout at the wrong temperature")
        if any(str(row.get("config_tag")) != config.tag for row in selected):
            raise ValueError(f"gamma={key} contains a rollout from the wrong Mizuta config")
        if any(int(row.get("verifier_calls", -1)) != 0 for row in selected):
            raise ValueError("Mizuta rows must remain verifier-free")
        if any(bool(row.get("safety_filter_used", True)) for row in selected):
            raise ValueError("Mizuta rows must not use a safety filter")
    return {
        "schema_version": ROLLOUT_SCHEMA,
        "method": "CFM-MPPI* / Mizuta low guidance",
        "role": role,
        "temperature": required_temperature,
        "temperature_contract": (
            "ordinary scientific source sampling"
            if role == "scientific"
            else "visualization only; excluded from all metrics"
        ),
        "fixed_attempts_per_gamma": int(repetitions),
        "gammas": [float(gamma) for gamma in gammas],
        "config": asdict(config),
        "config_sha256": _canonical_json_sha256(asdict(config)),
        "algorithm_contract": algorithm_contract(),
        "algorithm_contract_sha256": _canonical_json_sha256(algorithm_contract()),
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "scene_fingerprint_sha256": scene_fingerprint_sha256,
        "reference_source_sha256": reference_source_sha256,
        "implementation_sha256": implementation_sha256,
        "verifier_calls": 0,
        "socp_calls": 0,
        "safety_filter_used": False,
        "metrics_included": role == "scientific",
        "rows": [dict(row) for row in rows],
        "rollouts": [dict(row) for row in rows],
        "per_gamma": per_gamma if role == "scientific" else None,
        "overall": summarize_rows(rows) if role == "scientific" else None,
    }


def load_clean_stage3_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, Mapping[str, Any], str, str]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    file_hash = sha256_file(checkpoint_path)
    policy, payload = HP.load_hp(checkpoint_path, device=device)
    state_hash = model_state_hash(policy)
    try:
        require_promoted_fresh_pretrain(policy, payload)
    except RuntimeError as exc:
        raise RuntimeError(
            "Stage 04B requires the fresh endpoint-free hash-locked Stage-03 checkpoint"
        ) from exc
    config = payload.get("config", {})
    source_query_digest = str(payload.get("source_query_hash_digest", ""))
    if (
        payload.get("stage_schema") != "afe_fresh_pretrain_v1"
        or payload.get("fresh_from_scratch") is not True
        or payload.get("endpoint_free") is not True
        or payload.get("expansion_promotion") is not True
        or payload.get("id_mode_diversity_gate_passed") is not True
        or float(payload.get("id_evaluation_temperature", math.nan)) != 1.0
        or payload.get("id_evaluation_uncertainty_tilting") is not False
        or payload.get("frozen_feature_snapshot") is True
        or payload.get("model_state_sha256") != state_hash
        or config.get("arch") != "hp-repr"
        or config.get("schema_version") != "w8sg-hp-v2-low5-only"
        or config.get("raw_start_goal") is not False
        or int(config.get("ctx_dim", -1)) != 37
        or int(config.get("repr_dim", -1)) != 32
        or int(config.get("H_pred", -1)) != 10
        or tuple(config.get("grid_shape", ())) != (1, 32, 32)
        or int(config.get("K_hist", -1)) != 16
        or not math.isclose(float(config.get("u_max", math.nan)), 1.0)
        or config.get("use_gru") is not False
        or config.get("boundary_adapter") is not False
        or len(source_query_digest) != 64
    ):
        raise RuntimeError(
            "Stage 04B requires the fresh endpoint-free hash-locked Stage-03 checkpoint"
        )
    _require_sha256(source_query_digest, name="source_query_hash_digest")
    if not str(payload.get("source_manifest", "")):
        raise RuntimeError("fresh Stage-03 checkpoint does not identify its source manifest")
    policy.eval()
    return policy, payload, file_hash, state_hash


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    device = _configure_device(args.device)
    outdir = args.outdir.resolve()
    _require_clean_output_dir(outdir)
    for directory in (outdir / "data", outdir / "logs", outdir / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
    implementation_path = Path(__file__).resolve()
    implementation_hash = sha256_file(implementation_path)
    reference_hash = sha256_file(REFERENCE_SOURCE)
    policy, checkpoint_payload, checkpoint_file_hash, checkpoint_state_hash = (
        load_clean_stage3_checkpoint(args.checkpoint.resolve(), device)
    )
    checkpoint_config_hash = _canonical_json_sha256(checkpoint_payload["config"])
    scene = describe_ood_scene(make_ood_scene(radius=1.2))
    configs = LOW_GUIDANCE_SWEEP
    if len(configs) > 4 or any(not config.low_guidance_admissible for config in configs):
        raise RuntimeError("the Stage-04B sweep must remain bounded to four low-guidance configs")

    tuning_rows: list[dict[str, Any]] = []
    tuning_seed0 = int(args.seed)
    for config_index, config in enumerate(configs):
        tuning_rows.extend(
            run_fixed_rollouts(
                policy,
                config=config,
                gammas=TUNING_GAMMAS,
                repetitions=args.tune_repetitions,
                seed0=tuning_seed0 + config_index * 100_000,
                temperature=SCIENTIFIC_TEMPERATURE,
                max_steps=args.max_steps,
                reach_m=args.reach,
            )
        )
    tuning_summary = sweep_summary(configs, tuning_rows)
    selected = select_sweep_config(tuning_summary)
    sweep_path = outdir / "data/mizuta_low_guidance_sweep.pt"
    sweep_payload = {
        "schema_version": SWEEP_SCHEMA,
        "method": "CFM-MPPI* / Mizuta",
        "temperature": SCIENTIFIC_TEMPERATURE,
        "temperature_role": "bounded hyperparameter selection only",
        "tuning_gammas": list(TUNING_GAMMAS),
        "fixed_attempts_per_config_gamma": int(args.tune_repetitions),
        "configs": [asdict(config) for config in configs],
        "configs_sha256": _canonical_json_sha256(
            [asdict(config) for config in configs]
        ),
        "summaries": tuning_summary,
        "selected_config": asdict(selected),
        "algorithm_contract": algorithm_contract(),
        "algorithm_contract_sha256": _canonical_json_sha256(algorithm_contract()),
        "selection_rule": (
            "maximize SR, then minimize CR, maximize goal progress and successful mode "
            "coverage, then minimize safe guidance"
        ),
        "rollouts": tuning_rows,
        "verifier_calls": 0,
        "socp_calls": 0,
        "safety_filter_used": False,
        "checkpoint_file_sha256": checkpoint_file_hash,
        "checkpoint_state_sha256": checkpoint_state_hash,
        "checkpoint_config_sha256": checkpoint_config_hash,
        "scene_fingerprint_sha256": scene["fingerprint_sha256"],
        "reference_source_sha256": reference_hash,
        "implementation_sha256": implementation_hash,
    }
    _atomic_torch_save(sweep_path, sweep_payload)

    scientific_seed0 = tuning_seed0 + 1_000_000
    scientific_rows = run_fixed_rollouts(
        policy,
        config=selected,
        gammas=GAMMAS,
        repetitions=args.scientific_repetitions,
        seed0=scientific_seed0,
        temperature=SCIENTIFIC_TEMPERATURE,
        max_steps=args.max_steps,
        reach_m=args.reach,
    )
    scientific_payload = build_rollout_artifact(
        role="scientific",
        temperature=SCIENTIFIC_TEMPERATURE,
        config=selected,
        rows=scientific_rows,
        gammas=GAMMAS,
        repetitions=args.scientific_repetitions,
        checkpoint_file_sha256=checkpoint_file_hash,
        checkpoint_state_sha256=checkpoint_state_hash,
        checkpoint_config_sha256=checkpoint_config_hash,
        scene_fingerprint_sha256=scene["fingerprint_sha256"],
        reference_source_sha256=reference_hash,
        implementation_sha256=implementation_hash,
    )
    scientific_path = outdir / "data/mizuta_temperature1_scientific.pt"
    _atomic_torch_save(scientific_path, scientific_payload)

    gallery_seed0 = tuning_seed0 + 2_000_000
    gallery_rows = run_fixed_rollouts(
        policy,
        config=selected,
        gammas=GAMMAS,
        repetitions=args.gallery_repetitions,
        seed0=gallery_seed0,
        temperature=GALLERY_TEMPERATURE,
        max_steps=args.max_steps,
        reach_m=args.reach,
    )
    gallery_payload = build_rollout_artifact(
        role="gallery_only",
        temperature=GALLERY_TEMPERATURE,
        config=selected,
        rows=gallery_rows,
        gammas=GAMMAS,
        repetitions=args.gallery_repetitions,
        checkpoint_file_sha256=checkpoint_file_hash,
        checkpoint_state_sha256=checkpoint_state_hash,
        checkpoint_config_sha256=checkpoint_config_hash,
        scene_fingerprint_sha256=scene["fingerprint_sha256"],
        reference_source_sha256=reference_hash,
        implementation_sha256=implementation_hash,
    )
    gallery_path = outdir / "data/mizuta_temperature0.5_gallery.pt"
    _atomic_torch_save(gallery_path, gallery_payload)

    tuning_seeds = {int(row["seed"]) for row in tuning_rows}
    scientific_seeds = {int(row["seed"]) for row in scientific_rows}
    gallery_seeds = {int(row["seed"]) for row in gallery_rows}
    if tuning_seeds & scientific_seeds or tuning_seeds & gallery_seeds or scientific_seeds & gallery_seeds:
        raise RuntimeError("Mizuta tuning, scientific, and gallery episode seeds must be disjoint")
    post_evaluation_state_hash = model_state_hash(policy)
    if post_evaluation_state_hash != checkpoint_state_hash:
        raise RuntimeError("frozen Stage-03 checkpoint changed during Mizuta evaluation")
    post_evaluation_file_hash = sha256_file(args.checkpoint.resolve())
    if post_evaluation_file_hash != checkpoint_file_hash:
        raise RuntimeError("Stage-03 checkpoint file changed during Mizuta evaluation")
    if sha256_file(REFERENCE_SOURCE) != reference_hash:
        raise RuntimeError("Mizuta reference source changed during evaluation")
    if sha256_file(implementation_path) != implementation_hash:
        raise RuntimeError("Stage-04B implementation changed during evaluation")

    table_rows = []
    for gamma in GAMMAS:
        metrics = scientific_payload["per_gamma"][f"{float(gamma):g}"]
        table_rows.append({"gamma": float(gamma), **metrics})
    table_path = outdir / "tables/scientific_metrics_by_gamma.json"
    _atomic_json(table_path, {"rows": table_rows})

    artifacts = {
        "bounded_sweep": {"path": str(sweep_path), "sha256": sha256_file(sweep_path)},
        "temperature1_scientific": {
            "path": str(scientific_path),
            "sha256": sha256_file(scientific_path),
        },
        "temperature0.5_gallery_only": {
            "path": str(gallery_path),
            "sha256": sha256_file(gallery_path),
        },
        "scientific_metrics_table": {
            "path": str(table_path),
            "sha256": sha256_file(table_path),
        },
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "MIZUTA_BASELINE_COMPLETE",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "method": {
            "name": "CFM-MPPI* / Mizuta low guidance",
            "learning_performed": False,
            "verifier_calls": 0,
            "socp_calls": 0,
            "safety_filter_used": False,
            "reference_adapter": str(REFERENCE_SOURCE.resolve()),
            "reference_sha256": reference_hash,
            "implementation_sha256": implementation_hash,
            "algorithm_contract": algorithm_contract(),
            "algorithm_contract_sha256": _canonical_json_sha256(
                algorithm_contract()
            ),
            "legacy_artifact_reuse": False,
            "resume_supported": False,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "file_sha256": checkpoint_file_hash,
            "post_evaluation_file_sha256": post_evaluation_file_hash,
            "model_state_sha256": checkpoint_state_hash,
            "config_sha256": checkpoint_config_hash,
            "post_evaluation_model_state_sha256": post_evaluation_state_hash,
            "stage_schema": checkpoint_payload.get("stage_schema"),
            "fresh_from_scratch": checkpoint_payload.get("fresh_from_scratch"),
            "endpoint_free": checkpoint_payload.get("endpoint_free"),
            "source_manifest": checkpoint_payload.get("source_manifest"),
            "source_query_hash_digest": checkpoint_payload.get(
                "source_query_hash_digest"
            ),
        },
        "scene": scene,
        "sweep": {
            "config_count": len(configs),
            "tuning_gammas": list(TUNING_GAMMAS),
            "fixed_attempts_per_config_gamma": args.tune_repetitions,
            "selected_config": asdict(selected),
            "selected_config_sha256": _canonical_json_sha256(asdict(selected)),
            "seed0": tuning_seed0,
        },
        "scientific_evaluation": {
            "temperature": SCIENTIFIC_TEMPERATURE,
            "fixed_attempts_per_gamma": args.scientific_repetitions,
            "seed0": scientific_seed0,
            "per_gamma": scientific_payload["per_gamma"],
            "overall": scientific_payload["overall"],
        },
        "gallery": {
            "temperature": GALLERY_TEMPERATURE,
            "role": "visualization only; excluded from metrics",
            "fixed_attempts_per_gamma": args.gallery_repetitions,
            "seed0": gallery_seed0,
        },
        "artifacts": artifacts,
        "args": vars(args),
    }
    manifest_path = outdir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    _atomic_json(
        outdir / "logs/manifest_hash.json",
        {
            "schema_version": "afe_manifest_hash_v1",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
        },
    )
    _atomic_json(
        outdir / "logs/stage_summary.json",
        {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash},
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_config": selected.tag,
                "temperature1_SR": scientific_payload["overall"]["success_rate"],
                "temperature1_CR": scientific_payload["overall"]["collision_rate"],
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_hash,
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="fresh endpoint-free Stage-03 checkpoint_best.pt only",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="new or empty directory; Stage 04B never resumes old artifacts",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="use cuda:0 behind CUDA_VISIBLE_DEVICES=1, or cpu for inspection",
    )
    parser.add_argument("--seed", type=int, default=140_000)
    parser.add_argument("--tune-repetitions", type=int, default=1)
    parser.add_argument("--scientific-repetitions", type=int, default=8)
    parser.add_argument("--gallery-repetitions", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--reach", type=float, default=0.20)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if min(
        args.tune_repetitions,
        args.scientific_repetitions,
        args.gallery_repetitions,
        args.max_steps,
    ) <= 0:
        raise ValueError("all Stage-04B counts must be positive")
    if args.reach <= 0.0:
        raise ValueError("invalid Stage-04B reach configuration")
    run_stage(args)


if __name__ == "__main__":
    main()
