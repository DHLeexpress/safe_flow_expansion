"""Full deterministic safety verification of the exact planned H=10 object.

Unlike the legacy caller, this module passes current plus ten predicted
positions to the fitted-polytope SOCP verifier.  The binary safety label is
strict task-space bounds AND the SOCP certificate.  Physical clearance and
goal progress are diagnostics only and cannot change that label.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DynamicsConfig, VerifierConfig
from .dynamics import as_plan, as_state, rollout_plan


# Resolve the one verifier implementation used by the existing giant-obstacle
# experiment, then assert that a same-named module elsewhere did not win.
_THIS_FILE = Path(__file__).resolve()
_RUN_0706 = _THIS_FILE.parents[3]
_EXPECTED_VERIFIER = _RUN_0706.parent / "overnight_run_2026-07-01" / "verifier_polytope.py"
if str(_RUN_0706) not in sys.path:
    sys.path.insert(0, str(_RUN_0706))
import _paths as _legacy_paths  # noqa: E402,F401
import verifier_polytope as VP  # noqa: E402

if Path(VP.__file__).resolve() != _EXPECTED_VERIFIER.resolve():
    raise ImportError(
        "ambiguous verifier_polytope import: "
        f"expected {_EXPECTED_VERIFIER}, loaded {Path(VP.__file__).resolve()}"
    )


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlanVerification:
    """Structured result for one actual full-verifier query."""

    safe: bool
    in_bounds: bool
    socp_ok: bool
    bounds_margin_m: float
    physical_clearance_m: float
    face_margin_m: float
    certificate_residual: float
    certificate_worst_step: int
    progress_m: float
    start_goal_distance_m: float
    terminal_goal_distance_m: float
    gamma: float
    states: FloatArray
    positions: FloatArray

    def __post_init__(self) -> None:
        if self.states.shape != (11, 4):
            raise ValueError(f"verification states must have shape (11,4), got {self.states.shape}")
        if self.positions.shape != (11, 2):
            raise ValueError(f"verification positions must have shape (11,2), got {self.positions.shape}")
        if self.safe != (self.in_bounds and self.socp_ok):
            raise ValueError("safe must be exactly in_bounds AND socp_ok")


def _to_numpy(value: Any, *, name: str) -> FloatArray:
    """Convert a tensor-like environment field without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _environment_geometry(env: Any) -> tuple[FloatArray, float]:
    if not hasattr(env, "obstacles") or not hasattr(env, "r_robot"):
        raise TypeError("env must expose obstacles and r_robot")
    obstacles = _to_numpy(env.obstacles, name="env.obstacles")
    if obstacles.size == 0:
        obstacles = np.empty((0, 3), dtype=np.float64)
    if obstacles.ndim != 2 or obstacles.shape[1] != 3:
        raise ValueError(f"env.obstacles must have shape (N,3), got {obstacles.shape}")
    robot_radius = float(env.r_robot)
    if not np.isfinite(robot_radius) or robot_radius < 0.0:
        raise ValueError("env.r_robot must be a finite nonnegative scalar")
    return obstacles, robot_radius


def _goal_xy(env: Any, goal: ArrayLike | None) -> FloatArray:
    source = getattr(env, "goal", None) if goal is None else goal
    if source is None:
        raise TypeError("goal must be supplied or available as env.goal")
    value = _to_numpy(source, name="goal").reshape(-1)
    if value.size < 2:
        raise ValueError("goal must contain at least x and y")
    return value[:2].copy()


def _bounds_margin(positions: FloatArray, config: DynamicsConfig) -> float:
    lower = positions - float(config.workspace_low)
    upper = float(config.workspace_high) - positions
    return float(np.minimum(lower, upper).min())


def _physical_clearance(
    positions: FloatArray,
    obstacles: FloatArray,
    robot_radius: float,
) -> float:
    if len(obstacles) == 0:
        return float("inf")
    clearance = (
        np.linalg.norm(positions[:, None, :] - obstacles[None, :, :2], axis=2)
        - obstacles[None, :, 2]
        - robot_radius
    )
    return float(clearance.min())


def _face_margin(faces: list[Any], effective_radius: float) -> float:
    """Minimum optimized real-face margin, or local radius if none was sensed.

    An infeasible real face is reported as ``-inf`` rather than disguising it
    as a positive continuous margin; ``socp_ok`` and certificate residual
    retain the actual rejection signal.  ``-inf`` is also serializable by the
    immutable ledger schema, whereas NaN is deliberately forbidden.
    """

    real = [face for face in faces if getattr(face, "kind", None) == "real"]
    if not real:
        return float(effective_radius)
    if any((not bool(getattr(face, "feasible", False)))
           or not np.isfinite(float(getattr(face, "m", np.nan)))
           or float(face.m) <= 0.0 for face in real):
        return float("-inf")
    return float(min(float(face.m) for face in real))


def verify_plan(
    state: ArrayLike,
    controls: ArrayLike,
    env: Any,
    gamma: float,
    *,
    goal: ArrayLike | None = None,
    dynamics: DynamicsConfig = DynamicsConfig(),
    verifier: VerifierConfig = VerifierConfig(),
) -> PlanVerification:
    """Fully verify exactly one planned window and return separate diagnostics.

    The SOCP call is not skipped for an out-of-bounds plan: this function is
    the full query, and it records both components before forming their AND.
    """

    gamma_value = float(gamma)
    if not np.isfinite(gamma_value) or not 0.0 < gamma_value <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    current = as_state(state)
    plan = as_plan(controls, dynamics.horizon)
    states = rollout_plan(current, plan, config=dynamics)
    positions = states[:, :2].copy()
    if positions.shape != (dynamics.horizon + 1, 2):
        raise AssertionError("verifier must receive current plus all ten predicted positions")

    obstacles, robot_radius = _environment_geometry(env)
    goal_xy = _goal_xy(env, goal)
    bounds_margin = _bounds_margin(positions, dynamics)
    in_bounds = bool(bounds_margin >= 0.0)
    physical_clearance = _physical_clearance(positions, obstacles, robot_radius)

    socp_raw, faces, _raw_obstacles, effective_radius = VP.certify_window(
        positions,
        obstacles,
        robot_radius,
        gamma_value,
        R=float(verifier.sensing_radius),
        K=int(verifier.artificial_faces),
        rho_art=float(verifier.artificial_radius),
        m_min=float(verifier.minimum_face_margin),
        m_max=verifier.maximum_face_margin,
        n_theta=int(verifier.angle_samples),
        r_pad=float(verifier.rollout_padding_factor),
    )
    alpha = (1.0 - gamma_value) ** np.arange(len(positions), dtype=np.float64)
    socp_checked, certificate_residual, worst_step = VP.check_certificate(
        faces,
        positions - positions[0],
        alpha,
        include_start=False,
    )
    if bool(socp_raw) != bool(socp_checked):
        raise RuntimeError(
            "full verifier disagreed with its certificate replay: "
            f"certify_window={bool(socp_raw)}, check_certificate={bool(socp_checked)}"
        )
    socp_ok = bool(socp_checked)

    start_distance = float(np.linalg.norm(positions[0] - goal_xy))
    terminal_distance = float(np.linalg.norm(positions[-1] - goal_xy))
    progress = start_distance - terminal_distance

    # Keep the query evidence immutable enough for accidental downstream writes
    # to fail immediately rather than changing a stored verifier result.
    states.setflags(write=False)
    positions.setflags(write=False)
    return PlanVerification(
        safe=bool(in_bounds and socp_ok),
        in_bounds=in_bounds,
        socp_ok=socp_ok,
        bounds_margin_m=bounds_margin,
        physical_clearance_m=physical_clearance,
        face_margin_m=_face_margin(list(faces), float(effective_radius)),
        certificate_residual=float(certificate_residual),
        certificate_worst_step=int(worst_step),
        progress_m=progress,
        start_goal_distance_m=start_distance,
        terminal_goal_distance_m=terminal_distance,
        gamma=gamma_value,
        states=states,
        positions=positions,
    )
