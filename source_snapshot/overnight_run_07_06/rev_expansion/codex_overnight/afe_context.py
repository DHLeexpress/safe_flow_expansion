"""Fail-closed conditioning adapter shared by AFE gathering and evaluation.

The legacy wire name ``low5`` is retained in query records, but the declared
low7 schema always contains seven values.  This module is the only expansion
adapter allowed to choose between the two schemas.  In particular, low7 calls
``afe_restart.scene.context_from_state_low7`` directly so its scaling and
closest-boundary definition are exactly the pretraining implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_REV = _HERE.parent
_WORK = _REV.parent
for _path in (_WORK, _REV, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _paths  # noqa: F401,E402
import grid_feats as GF
from codex_challenging.afe_restart.scene import (  # noqa: E402
    context_from_state,
    context_from_state_low7,
)


LOW5_SCHEMA = "low5"
LOW7_SCHEMA = "low7_closest_boundary"
LOW7_TIE_SCHEMA = "low7_closest_boundary_tie_mean"
LOW7_SCHEMAS = (LOW7_SCHEMA, LOW7_TIE_SCHEMA)
SCHEMA_DIMS = {LOW5_SCHEMA: 5, LOW7_SCHEMA: 7, LOW7_TIE_SCHEMA: 7}


def declared_gamma_storage_map(gammas) -> dict[float, float]:
    """Map float32 conditioning-wire values to original declared gamma keys."""

    mapping: dict[float, float] = {}
    declared_keys: set[float] = set()
    for gamma in gammas:
        declared = round(float(gamma), 8)
        storage = float(np.float32(gamma))
        if not np.isfinite(declared) or not np.isfinite(storage):
            raise ValueError("declared gammas must be finite")
        if storage in mapping or declared in declared_keys:
            raise ValueError("declared gammas are not unique at conditioning precision")
        mapping[storage] = declared
        declared_keys.add(declared)
    if not mapping:
        raise ValueError("at least one gamma must be declared")
    return mapping


def canonical_declared_gamma(value, storage_map: dict[float, float]) -> float:
    """Recover a declared gamma after the conditioning vector's float32 cast."""

    storage = float(np.float32(value))
    if not np.isfinite(storage) or storage not in storage_map:
        raise ValueError(f"gamma {value!r} is not declared")
    return storage_map[storage]


@dataclass(frozen=True)
class ConditioningContract:
    schema: str
    raw_condition_dim: int
    ctx_dim: int
    trunk_input_dim: int


def policy_contract(policy: Any) -> ConditioningContract:
    schema = str(getattr(policy, "conditioning_schema", ""))
    raw_dim = int(getattr(policy, "raw_condition_dim", -1))
    ctx_dim = int(getattr(policy, "ctx_dim", -1))
    trunk = getattr(policy, "trunk", None)
    try:
        trunk_input = int(trunk[0].in_features)
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("policy has no declared first trunk linear layer") from exc
    expected = {
        LOW5_SCHEMA: (5, 37, 89),
        LOW7_SCHEMA: (7, 39, 91),
        LOW7_TIE_SCHEMA: (7, 39, 91),
    }.get(schema)
    if expected is None:
        raise RuntimeError(f"unsupported policy conditioning schema {schema!r}")
    if (raw_dim, ctx_dim, trunk_input) != expected:
        raise RuntimeError(
            f"{schema} policy dimensions {(raw_dim, ctx_dim, trunk_input)} != {expected}"
        )
    return ConditioningContract(schema, raw_dim, ctx_dim, trunk_input)


def require_declared_contract(
    policy: Any,
    schema: str,
    raw_condition_dim: int,
) -> ConditioningContract:
    contract = policy_contract(policy)
    if contract.schema != str(schema) or contract.raw_condition_dim != int(raw_condition_dim):
        raise RuntimeError(
            "policy and expansion conditioning contracts disagree: "
            f"policy={contract}, expansion={(schema, raw_condition_dim)}"
        )
    return contract


def build_context(
    state: np.ndarray,
    goal: np.ndarray,
    gamma: float,
    executed_controls: Sequence[np.ndarray] | np.ndarray,
    env: Any,
    schema: str,
):
    """Build and validate one exact policy/verifier query context."""

    schema = str(schema)
    if schema in LOW7_SCHEMAS:
        record = context_from_state_low7(
            state,
            goal,
            gamma,
            executed_controls,
            env,
            tie_average_boundary=(schema == LOW7_TIE_SCHEMA),
        )
    elif schema == LOW5_SCHEMA:
        record = context_from_state(state, goal, gamma, executed_controls, env)
    else:
        raise RuntimeError(f"unsupported conditioning schema {schema!r}")

    condition = np.asarray(record.low5, dtype=np.float32)
    expected_dim = SCHEMA_DIMS[schema]
    if condition.shape != (expected_dim,) or not np.isfinite(condition).all():
        raise RuntimeError(
            f"{schema} context has invalid condition shape/values: {condition.shape}"
        )
    if not np.isclose(condition[-1], float(gamma), atol=5.0e-7, rtol=0.0):
        raise RuntimeError("context gamma is not the final raw condition")
    if schema in LOW7_SCHEMAS:
        obstacles = env.obstacles.detach().cpu().numpy()
        exact = GF.low7(
            state,
            goal,
            gamma,
            obstacles,
            float(env.r_robot),
            tie_average=(schema == LOW7_TIE_SCHEMA),
        )
        if not np.array_equal(condition, exact):
            raise RuntimeError(
                "context_from_state_low7 disagrees with shared low7 featurization"
            )
    grid = np.asarray(record.grid, dtype=np.float32)
    history = np.asarray(record.hist, dtype=np.float32)
    if grid.shape != (3, 32, 32):
        raise RuntimeError(f"context grid shape {grid.shape} != (3,32,32)")
    if history.shape != (GF.K_HIST, 2):
        raise RuntimeError(
            f"context history shape {history.shape} != {(GF.K_HIST, 2)}"
        )
    return record


def arrays_for_episodes(episodes: Sequence[dict[str, Any]], env: Any, schema: str):
    goal = env.goal.detach().cpu().numpy()
    grids, conditions, histories = [], [], []
    for episode in episodes:
        record = build_context(
            episode["state"],
            goal,
            episode["gamma"],
            episode.get("hist", episode.get("history", [])),
            env,
            schema,
        )
        grids.append(record.grid)
        conditions.append(record.low5)
        histories.append(record.hist)
    return (
        np.asarray(grids, dtype=np.float32),
        np.asarray(conditions, dtype=np.float32),
        np.asarray(histories, dtype=np.float32),
    )
