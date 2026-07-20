"""Observational route-mode metrics for diagonal goal-reaching tasks.

These helpers only measure which side of the start--goal line a plan or
trajectory uses.  They do not alter acquisition, verification, execution, or
training.  For the canonical lower-left to upper-right task, ``U`` is the
positive-cross-track side (``y > x``) and ``R`` is the negative side
(``y < x``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


MODE_U = np.int8(1)
MODE_AMBIGUOUS = np.int8(0)
MODE_R = np.int8(-1)
DEFAULT_AMBIGUITY_BAND = 0.05


def _finite_xy(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1:] != (2,):
        raise ValueError(f"{name} must have final dimension 2, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def signed_cross_track(
    points: Any,
    *,
    start: Any,
    goal: Any,
) -> np.ndarray:
    """Return signed perpendicular distance from the oriented start--goal line.

    Positive values lie to the left of the oriented line from ``start`` to
    ``goal``.  Thus the canonical diagonal labels points with ``y > x`` as
    positive.  The result is in the same distance units as the inputs.
    """

    xy = _finite_xy("points", points)
    start_xy = _finite_xy("start", start)
    goal_xy = _finite_xy("goal", goal)
    if start_xy.shape != (2,) or goal_xy.shape != (2,):
        raise ValueError("start and goal must each have shape (2,)")
    direction = goal_xy - start_xy
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("start and goal must be distinct")
    relative = xy - start_xy
    return (direction[0] * relative[..., 1] - direction[1] * relative[..., 0]) / length


def classify_cross_track(
    cross_track: Any,
    *,
    ambiguity_band: float = DEFAULT_AMBIGUITY_BAND,
) -> np.ndarray:
    """Classify signed distances as ``U=+1``, ``R=-1``, or ambiguous ``0``."""

    values = np.asarray(cross_track, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("cross_track must contain only finite values")
    band = float(ambiguity_band)
    if not math.isfinite(band) or band < 0.0:
        raise ValueError("ambiguity_band must be finite and nonnegative")
    labels = np.full(values.shape, MODE_AMBIGUOUS, dtype=np.int8)
    labels[values > band] = MODE_U
    labels[values < -band] = MODE_R
    return labels


def classify_plan_endpoints(
    endpoints: Any,
    *,
    start: Any,
    goal: Any,
    ambiguity_band: float = DEFAULT_AMBIGUITY_BAND,
) -> np.ndarray:
    """Classify plan endpoint positions without inspecting the rest of a plan."""

    return classify_cross_track(
        signed_cross_track(endpoints, start=start, goal=goal),
        ambiguity_band=ambiguity_band,
    )


def closest_approach_points(
    trajectories: Any,
    *,
    obstacle_centers: Any,
    obstacle_radii: Any = 0.0,
) -> dict[str, np.ndarray]:
    """Select each trajectory's point of minimum obstacle-boundary clearance.

    ``trajectories`` has shape ``(..., T, 2)`` and ``obstacle_centers`` has
    shape ``(M, 2)`` (or ``(2,)`` for one obstacle).  A scalar radius is
    broadcast to all obstacles.  Returned arrays preserve the trajectory's
    leading dimensions.
    """

    states = _finite_xy("trajectories", trajectories)
    if states.ndim < 2 or states.shape[-2] < 1:
        raise ValueError("trajectories must have shape (..., T, 2) with T >= 1")
    centers = _finite_xy("obstacle_centers", obstacle_centers)
    if centers.ndim == 1:
        centers = centers[None, :]
    if centers.ndim != 2 or centers.shape[0] < 1:
        raise ValueError("obstacle_centers must have shape (M, 2) with M >= 1")
    radii = np.asarray(obstacle_radii, dtype=np.float64)
    if radii.ndim == 0:
        radii = np.full(centers.shape[0], float(radii), dtype=np.float64)
    if radii.shape != (centers.shape[0],):
        raise ValueError("obstacle_radii must be scalar or have one value per obstacle")
    if not np.isfinite(radii).all() or np.any(radii < 0.0):
        raise ValueError("obstacle_radii must be finite and nonnegative")

    leading_shape = states.shape[:-2]
    flat = states.reshape((-1, states.shape[-2], 2))
    distances = np.linalg.norm(
        flat[:, :, None, :] - centers[None, None, :, :], axis=-1
    )
    clearance = distances - radii[None, None, :]
    flat_index = np.argmin(clearance.reshape((flat.shape[0], -1)), axis=1)
    time_index, obstacle_index = np.divmod(flat_index, centers.shape[0])
    row_index = np.arange(flat.shape[0])
    points = flat[row_index, time_index]
    minimum_clearance = clearance[row_index, time_index, obstacle_index]

    return {
        "points": points.reshape((*leading_shape, 2)),
        "time_index": time_index.reshape(leading_shape),
        "obstacle_index": obstacle_index.reshape(leading_shape),
        "clearance": minimum_clearance.reshape(leading_shape),
    }


def classify_trajectories_at_closest_approach(
    trajectories: Any,
    *,
    start: Any,
    goal: Any,
    obstacle_centers: Any,
    obstacle_radii: Any = 0.0,
    ambiguity_band: float = DEFAULT_AMBIGUITY_BAND,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Classify trajectories using their closest point to any obstacle boundary."""

    closest = closest_approach_points(
        trajectories,
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
    )
    labels = classify_plan_endpoints(
        closest["points"],
        start=start,
        goal=goal,
        ambiguity_band=ambiguity_band,
    )
    return labels, closest


def summarize_modes(labels: Any) -> dict[str, float | int]:
    """Summarize U/R diversity while keeping ambiguous samples explicit.

    ``u_fraction`` and ``r_fraction`` are conditional on a resolved label.
    ``balance`` is ``2 min(n_U,n_R)/(n_U+n_R)``.  ``binary_entropy`` is
    normalized by ``log(2)`` and also excludes ambiguous samples.  When there
    are no resolved samples, all three resolved-mode statistics are defined as
    zero and ``ambiguous_fraction`` reveals that degeneracy.
    """

    values = np.asarray(labels)
    valid = np.isin(values, (MODE_R, MODE_AMBIGUOUS, MODE_U))
    if not bool(np.all(valid)):
        raise ValueError("labels may only contain MODE_U (+1), MODE_R (-1), or 0")
    total = int(values.size)
    u_count = int(np.count_nonzero(values == MODE_U))
    r_count = int(np.count_nonzero(values == MODE_R))
    ambiguous_count = int(np.count_nonzero(values == MODE_AMBIGUOUS))
    resolved = u_count + r_count

    if resolved:
        u_fraction = u_count / resolved
        r_fraction = r_count / resolved
        balance = 2.0 * min(u_count, r_count) / resolved
        entropy = 0.0
        for probability in (u_fraction, r_fraction):
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        entropy /= math.log(2.0)
    else:
        u_fraction = 0.0
        r_fraction = 0.0
        balance = 0.0
        entropy = 0.0

    return {
        "total_count": total,
        "resolved_count": resolved,
        "u_count": u_count,
        "r_count": r_count,
        "ambiguous_count": ambiguous_count,
        "resolved_fraction": float(resolved / total) if total else 0.0,
        "u_fraction": float(u_fraction),
        "r_fraction": float(r_fraction),
        "balance": float(balance),
        "coverage_weighted_balance": (
            float(balance * resolved / total) if total else 0.0
        ),
        "binary_entropy": float(entropy),
        "ambiguous_fraction": float(ambiguous_count / total) if total else 0.0,
    }
