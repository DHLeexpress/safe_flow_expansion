"""Fail-closed execution selection using SafeMPPI's nominal first-step level set."""
from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
for _path in (_WORK, _REV, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import grid_feats as GF
import grid_rollout as GR
import grid_scene as GS


NOMINAL_HP_TOLERANCE = 1.0e-8
MAX_STEP_PROGRESS = "nominal_hp_max_step_progress"
MAX_STEP_MARGIN = "nominal_hp_max_step_margin"
MAX_STEP_MARGIN_ONLY = "nominal_hp_max_step_margin_only"


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _selection_key(row: dict, selector: str) -> tuple[float | int, ...]:
    progress = float(row["step_progress"])
    margin = float(row["nominal_hp_step_margin"])
    candidate_id = int(row["candidate_id"])
    if selector == MAX_STEP_PROGRESS:
        return progress, margin, -candidate_id
    if selector == MAX_STEP_MARGIN:
        return margin, progress, -candidate_id
    if selector == MAX_STEP_MARGIN_ONLY:
        return margin, -candidate_id
    raise ValueError(f"unknown execution selector: {selector}")


def select_nominal_hp_execution(
    current_state,
    candidates,
    verifier_results: Sequence[dict],
    gamma: float,
    env,
    *,
    segments=None,
    candidate_ids: Sequence[int] | None = None,
    selector: str = MAX_STEP_PROGRESS,
) -> dict:
    """Select one execution-verified candidate whose first step passes nominal H_P.

    The nominal polytope is rebuilt once at the current state using the exact
    SafeMPPI construction.  This helper only chooses an action; it does not
    provide a fallback, relax gamma, or alter verifier/training labels.  A
    certified goal-reaching prefix is execution-verified via ``exec_y`` but
    remains outside D+ when its full-window label ``y`` is zero.
    """

    state = _numpy(current_state).astype(np.float64, copy=False).reshape(-1)
    if state.size < 4 or not np.isfinite(state[:4]).all():
        raise ValueError("current_state must contain four finite values")
    gamma_value = float(gamma)
    if not np.isfinite(gamma_value) or not 0.0 <= gamma_value <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if selector not in (MAX_STEP_PROGRESS, MAX_STEP_MARGIN, MAX_STEP_MARGIN_ONLY):
        raise ValueError(f"unknown execution selector: {selector}")

    controls = _numpy(candidates)
    if controls.ndim != 3 or controls.shape[1] < 1 or controls.shape[2] != 2:
        raise ValueError("candidates must have shape [B, H>=1, 2]")
    count = int(controls.shape[0])
    if len(verifier_results) != count:
        raise ValueError("verifier_results must align one-to-one with candidates")

    if candidate_ids is None:
        ids = list(range(count))
    else:
        ids = [int(candidate_id) for candidate_id in candidate_ids]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("candidate_ids must be unique and align with candidates")

    if segments is None:
        planned_positions = GR.di_rollout_batch(
            state.astype(np.float32), controls.astype(np.float32), float(env.dt)
        ).astype(np.float64)
    else:
        planned_positions = _numpy(segments).astype(np.float64, copy=False)
        if (
            planned_positions.ndim != 3
            or planned_positions.shape[0] != count
            or planned_positions.shape[1] < 1
            or planned_positions.shape[2] != 2
        ):
            raise ValueError("segments must have shape [B, H>=1, 2]")

    goal = _numpy(env.goal).astype(np.float64, copy=False).reshape(-1)[:2]
    if goal.size != 2 or not np.isfinite(goal).all():
        raise ValueError("env.goal must contain two finite values")

    planner_config = GS.mode1_config()
    nominal_hp, _ = GF.polytope_HP(
        state[:2],
        GS.planner_obstacles(env),
        sensing=float(planner_config["barrier_activation_radius"]),
        n_base=int(planner_config.get("polytope_nbase", 16)),
        predict_gain=float(planner_config.get("predict_gain", 0.0)),
    )
    first_positions = planned_positions[:, 0, :]
    hp_values = np.asarray(
        nominal_hp(np.vstack((state[:2], first_positions))), dtype=np.float64
    ).reshape(-1)
    if hp_values.size != count + 1:
        raise ValueError("nominal H_P callable returned an unexpected shape")
    hp0 = float(hp_values[0])
    start_distance = float(np.linalg.norm(state[:2] - goal))

    rows = []
    for local_index, (candidate_id, result, first_position, hp1) in enumerate(
        zip(ids, verifier_results, first_positions, hp_values[1:])
    ):
        progress = float(start_distance - np.linalg.norm(first_position - goal))
        margin = float(hp1 - (1.0 - gamma_value) * hp0)
        full_positive = bool(result["y"] == 1)
        execution_positive = bool(result.get("exec_y", result["y"]) == 1)
        finite_score = bool(np.isfinite(progress) and np.isfinite(margin))
        rows.append({
            "local_index": int(local_index),
            "candidate_id": int(candidate_id),
            "full_socp_positive": full_positive,
            "execution_verifier_positive": execution_positive,
            "step_progress": progress,
            "nominal_hp_step_margin": margin,
            "eligible": bool(
                execution_positive
                and finite_score
                and margin >= -NOMINAL_HP_TOLERANCE
            ),
        })

    eligible = [row for row in rows if row["eligible"]]
    chosen = max(eligible, key=lambda row: _selection_key(row, selector)) if eligible else None
    return {
        "selector": selector,
        "chosen": (None if chosen is None else dict(chosen)),
        "failure": (None if chosen is not None else "no_exec_verified_nominal_hp_step"),
        "counts": {
            "candidates": count,
            "full_socp_positive": sum(row["full_socp_positive"] for row in rows),
            "execution_verifier_positive": sum(
                row["execution_verifier_positive"] for row in rows
            ),
            "nominal_hp_eligible": len(eligible),
        },
        "nominal_hp_at_state": hp0,
        "per_candidate": rows,
    }


def nominal_hp_max_step_progress(
    current_state,
    candidates,
    verifier_results: Sequence[dict],
    gamma: float,
    env,
    *,
    segments=None,
    candidate_ids: Sequence[int] | None = None,
) -> dict:
    """Choose by one-step goal progress, then H_P margin, then candidate id."""

    return select_nominal_hp_execution(
        current_state,
        candidates,
        verifier_results,
        gamma,
        env,
        segments=segments,
        candidate_ids=candidate_ids,
        selector=MAX_STEP_PROGRESS,
    )


def nominal_hp_max_step_margin(
    current_state,
    candidates,
    verifier_results: Sequence[dict],
    gamma: float,
    env,
    *,
    segments=None,
    candidate_ids: Sequence[int] | None = None,
) -> dict:
    """Choose by H_P margin, then one-step goal progress, then candidate id."""

    return select_nominal_hp_execution(
        current_state,
        candidates,
        verifier_results,
        gamma,
        env,
        segments=segments,
        candidate_ids=candidate_ids,
        selector=MAX_STEP_MARGIN,
    )


def nominal_hp_max_step_margin_only(
    current_state,
    candidates,
    verifier_results: Sequence[dict],
    gamma: float,
    env,
    *,
    segments=None,
    candidate_ids: Sequence[int] | None = None,
) -> dict:
    """Choose only by H_P margin; candidate id resolves an exact tie."""

    return select_nominal_hp_execution(
        current_state,
        candidates,
        verifier_results,
        gamma,
        env,
        segments=segments,
        candidate_ids=candidate_ids,
        selector=MAX_STEP_MARGIN_ONLY,
    )
