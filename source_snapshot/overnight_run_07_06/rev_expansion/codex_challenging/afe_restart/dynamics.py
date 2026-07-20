"""Exact double-integrator rollout for one immutable H=10 planned query."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DynamicsConfig


FloatArray = NDArray[np.float64]


def as_state(state: ArrayLike) -> FloatArray:
    """Return a finite copied state with canonical shape ``[4]``."""

    value = np.asarray(state, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("state must contain only finite values")
    return value.copy()


def as_plan(controls: ArrayLike, horizon: int = 10) -> FloatArray:
    """Return a finite copied planned action window with shape ``[H,2]``."""

    value = np.asarray(controls, dtype=np.float64)
    expected = (int(horizon), 2)
    if value.shape != expected:
        raise ValueError(f"planned controls must have shape {expected}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("planned controls must contain only finite values")
    return value.copy()


def step_state(state: ArrayLike, action: ArrayLike, dt: float = 0.1) -> FloatArray:
    """Apply the same constant-acceleration DI transition used by SafeMPPI."""

    current = as_state(state)
    control = np.asarray(action, dtype=np.float64)
    if control.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {control.shape}")
    if not np.isfinite(control).all():
        raise ValueError("action must contain only finite values")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    next_state = np.empty(4, dtype=np.float64)
    next_state[:2] = current[:2] + dt * current[2:] + 0.5 * dt * dt * control
    next_state[2:] = current[2:] + dt * control
    return next_state


def rollout_plan(
    state: ArrayLike,
    controls: ArrayLike,
    *,
    config: DynamicsConfig = DynamicsConfig(),
) -> FloatArray:
    """Roll one plan and return current plus predicted states, shape ``[H+1,4]``.

    No control is clipped or replanned here: the returned rows are exactly the
    states induced by the supplied immutable planned window.
    """

    plan = as_plan(controls, config.horizon)
    states = np.empty((config.horizon + 1, config.state_dim), dtype=np.float64)
    states[0] = as_state(state)
    for step, action in enumerate(plan):
        states[step + 1] = step_state(states[step], action, config.dt)
    return states


def planned_positions(
    state: ArrayLike,
    controls: ArrayLike,
    *,
    config: DynamicsConfig = DynamicsConfig(),
) -> FloatArray:
    """Return all verifier positions, including the current point, ``[H+1,2]``."""

    return rollout_plan(state, controls, config=config)[:, :2]


def execute_first_action(
    state: ArrayLike,
    verified_plan: ArrayLike,
    *,
    config: DynamicsConfig = DynamicsConfig(),
) -> FloatArray:
    """Execute only ``U[0]`` after the caller has verified the complete plan."""

    plan = as_plan(verified_plan, config.horizon)
    return step_state(state, plan[0], config.dt)
