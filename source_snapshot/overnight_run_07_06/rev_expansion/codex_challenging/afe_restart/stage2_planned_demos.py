#!/usr/bin/env python3
"""Stage 02: real, balanced SafeMPPI *planned-window* demonstrations.

This stage deliberately does not reuse any legacy training target.  A legacy
seed/signature census may only influence the order in which seeds are tried.
For every receding-horizon step the data path is exactly::

    SafeMPPI H=10 cost-selected outputs -> full verifier -> select safe by progress
        -> save that exact H=10 plan -> execute only plan[0]

Raw MPPI debug rollouts are also fully verified for diagnostics, but are never
executed or emitted as training targets.  If neither cost-selected output
passes, that teacher attempt fails closed and another seed is tried.

The candidate episodes are resumable.  The final dataset contains 12 real
R-first and 12 real U-first successful trajectories for every gamma (unless a
smoke run explicitly disables the quota requirement).  There is no reflection,
trajectory padding, executed-window reconstruction, or legacy checkpoint use.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from .config import clean_method_absence_manifest
from .deps import assert_no_legacy_expansion_imports, sha256_file, write_dependency_manifest
from .dynamics import execute_first_action
from .fallback import BackupProposal, SafeMPPIBackup
from .scene import (
    GAMMAS,
    GOAL,
    START,
    context_from_state,
    make_id_scene,
    verifier_spec_fingerprint,
)
from .schemas import QueryContext, query_content_hash
from .verifier import PlanVerification, verify_plan


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = PACKAGE_ROOT / "stage_results/02_planned_demos"
LEGACY_HINT_PATH = (
    PACKAGE_ROOT.parent
    / "giant_obstacle_ood/stage_results/02b_balanced_id/data/balanced_id_paths_all_gamma.npz"
)
SCHEMA_VERSION = "afe_planned_demo_v2_exact_verifier_identity"
SWEEP_SCHEMA_VERSION = "afe_stage2_expert_target_smoothness_sweep_v1"


def _canonical_supported_gamma(value: float) -> float:
    """Map a serialized scalar to the declared conditioning grid.

    ``gamma`` is stored in the tensor payload as float32, whereas ``GAMMAS``
    contains Python float literals.  Exact set equality would therefore reject
    valid values such as float32(0.1).  Canonicalization is intentionally tight:
    it accepts only the unique declared grid point within float32 roundoff.
    """

    scalar = float(value)
    matches = [float(gamma) for gamma in GAMMAS if abs(scalar - float(gamma)) <= 5e-7]
    if len(matches) != 1:
        raise RuntimeError(f"gamma={scalar!r} is not a unique member of {tuple(GAMMAS)}")
    return matches[0]


@dataclass(frozen=True)
class DemoRunConfig:
    """Rollout and exact-balancing settings recorded in the manifest."""

    max_steps: int = 240
    reach_m: float = 0.20
    smooth_weight: float = 8.0
    retreat_weight: float = 1.0
    noise_var_mult: float = 3.0
    max_debug_candidates: int = 6
    max_proposals_per_step: int = 8
    quota_per_direction: int = 12
    max_candidate_seeds_per_gamma: int = 256
    seed0: int = 72_000

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.reach_m <= 0.0:
            raise ValueError("reach_m must be positive")
        if self.smooth_weight < 0.0 or self.retreat_weight < 0.0:
            raise ValueError("SafeMPPI cost weights must be nonnegative")
        if not math.isfinite(self.noise_var_mult) or self.noise_var_mult <= 0.0:
            raise ValueError("noise_var_mult must be finite and positive")
        if self.max_debug_candidates < 0 or self.max_proposals_per_step <= 0:
            raise ValueError("proposal counts must be positive")
        if self.quota_per_direction <= 0:
            raise ValueError("quota_per_direction must be positive")
        if self.max_candidate_seeds_per_gamma <= 0:
            raise ValueError("max_candidate_seeds_per_gamma must be positive")


def gamma_tag(gamma: float) -> str:
    return f"{float(gamma):g}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    temporary.replace(path)


def _np_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def _environment_clearance(path: np.ndarray, env: Any) -> np.ndarray:
    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64, copy=False)
    if len(obstacles) == 0:
        return np.full(len(path), np.inf, dtype=np.float64)
    return (
        np.linalg.norm(path[:, None, :] - obstacles[None, :, :2], axis=2)
        - obstacles[None, :, 2]
        - float(env.r_robot)
    ).min(axis=1)


def _first_crossing_time(values: np.ndarray, threshold: float = 1.0) -> float:
    for index in range(1, len(values)):
        left, right = float(values[index - 1]), float(values[index])
        if left < threshold <= right:
            denominator = right - left
            fraction = 1.0 if abs(denominator) <= 1.0e-12 else (threshold - left) / denominator
            return float(index - 1 + fraction)
    return math.inf


def direction_class(path: np.ndarray, *, tie_tolerance_steps: float = 1.0e-5) -> str:
    """Classify a real successful path by its first x=1/y=1 crossing.

    Linear interpolation avoids an x-loop ordering bias.  Ties and missing
    crossings are explicitly unclassified and can never fill either quota.
    """

    xy = np.asarray(path, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"path must have shape [N,2], got {xy.shape}")
    right_time = _first_crossing_time(xy[:, 0])
    up_time = _first_crossing_time(xy[:, 1])
    if not np.isfinite(right_time) or not np.isfinite(up_time):
        return "unclassified"
    if abs(right_time - up_time) <= float(tie_tolerance_steps):
        return "unclassified"
    return "R-first" if right_time < up_time else "U-first"


def _proposal_metrics(result: Any) -> dict[str, Any]:
    """Serialize the structured verifier output without changing its label."""

    return {
        "safe": bool(result.safe),
        "in_bounds": bool(getattr(result, "in_bounds", result.safe)),
        "socp_ok": bool(getattr(result, "socp_ok", result.safe)),
        "bounds_margin_m": _np_float(getattr(result, "bounds_margin_m", math.nan)),
        "physical_clearance_m": _np_float(
            getattr(result, "physical_clearance_m", math.nan)
        ),
        "face_margin_m": _np_float(getattr(result, "face_margin_m", math.nan)),
        "certificate_residual": _np_float(
            getattr(result, "certificate_residual", math.nan)
        ),
        "certificate_worst_step": int(
            getattr(result, "certificate_worst_step", -1)
        ),
        "progress_m": _np_float(getattr(result, "progress_m", -math.inf)),
        "start_goal_distance_m": _np_float(
            getattr(result, "start_goal_distance_m", math.nan)
        ),
        "terminal_goal_distance_m": _np_float(
            getattr(result, "terminal_goal_distance_m", math.nan)
        ),
    }


def _context_arrays(contexts: Sequence[QueryContext]) -> dict[str, np.ndarray]:
    if not contexts:
        # Empty smoke/fail-closed episodes still receive explicit arrays.  The
        # final balanced dataset never contains an empty successful episode.
        return {
            "context_grid": np.empty((0, 0), dtype=np.float32),
            "context_low5": np.empty((0, 0), dtype=np.float32),
            "context_hist": np.empty((0, 0), dtype=np.float32),
            "context_verifier_state": np.empty((0, 4), dtype=np.float64),
            "context_verifier_spec_fingerprint": np.empty((0,), dtype="U64"),
        }
    return {
        "context_grid": np.asarray([item.grid for item in contexts], dtype=np.float32),
        "context_low5": np.asarray([item.low5 for item in contexts], dtype=np.float32),
        "context_hist": np.asarray([item.hist for item in contexts], dtype=np.float32),
        "context_verifier_state": np.asarray(
            [item.verifier_state for item in contexts], dtype=np.float64
        ),
        "context_verifier_spec_fingerprint": np.asarray(
            [item.verifier_spec_fingerprint for item in contexts], dtype="U64"
        ),
    }


def _default_backup(config: DemoRunConfig) -> SafeMPPIBackup:
    return SafeMPPIBackup(
        smooth_weight=config.smooth_weight,
        retreat_weight=config.retreat_weight,
        max_debug_candidates=config.max_debug_candidates,
        noise_var_mult=config.noise_var_mult,
    )


def _rollout_config(config: DemoRunConfig) -> dict[str, Any]:
    """Only fields that can change one candidate trajectory."""

    return {
        key: value
        for key, value in asdict(config).items()
        if key
        in {
            "max_steps",
            "reach_m",
            "smooth_weight",
            "retreat_weight",
            "noise_var_mult",
            "max_debug_candidates",
            "max_proposals_per_step",
        }
    }


@torch.inference_mode()
def run_expert_rollout(
    *,
    env: Any,
    gamma: float,
    seed: int,
    device: torch.device,
    config: DemoRunConfig,
    backup: Any | None = None,
    verify_fn: Callable[..., Any] = verify_plan,
    context_fn: Callable[..., QueryContext] = context_from_state,
) -> dict[str, Any]:
    """Generate one real receding-horizon trajectory under the clean contract.

    The return object keeps all queried H=10 plans and their step indices, so
    every verifier call can be reconstructed as ``(context[step], plan)``.
    ``training_plans`` is an exact view of selected verified-safe query plans.
    """

    if backup is None:
        backup = _default_backup(config)
    if not np.isclose(float(getattr(env, "dt", 0.1)), 0.1, atol=0.0, rtol=0.0):
        raise ValueError("demo dynamics, full verifier, and execution require env.dt == 0.1")
    state = np.asarray(env.x0.detach().cpu().numpy(), dtype=np.float64).copy()
    goal = np.asarray(env.goal.detach().cpu().numpy(), dtype=np.float64).reshape(-1)[:2]
    initial_state = state.copy()
    states: list[np.ndarray] = [state.copy()]
    executed_actions: list[np.ndarray] = []
    contexts: list[QueryContext] = []
    query_plans: list[np.ndarray] = []
    query_steps: list[int] = []
    query_hashes: list[str] = []
    query_kinds: list[str] = []
    query_internal_feasible: list[int] = []
    query_metrics: list[dict[str, Any]] = []
    selected_query_indices: list[int] = []
    telemetry_rows: list[dict[str, Any]] = []
    dead_reason: str | None = None
    started = time.perf_counter()

    # Never query/execute after the episode is already at its goal.
    if float(np.linalg.norm(state[:2] - goal)) < config.reach_m:
        dead_reason = None
    else:
        for step in range(config.max_steps):
            context = context_fn(state, goal, gamma, executed_actions, env)
            if not isinstance(context, QueryContext):
                raise TypeError("context_fn must return QueryContext")
            if not np.array_equal(context.verifier_state, state):
                raise RuntimeError(
                    "demo query context does not retain the exact verifier state"
                )
            if context.verifier_spec_fingerprint != verifier_spec_fingerprint(
                env, goal
            ):
                raise RuntimeError(
                    "demo query context verifier specification does not match "
                    "the full verifier invocation"
                )
            contexts.append(context)
            proposals, telemetry = backup.propose(
                state,
                goal,
                env,
                float(gamma),
                seed=int(seed) * 10_000 + step,
                device=device,
            )
            proposals = list(proposals)[: config.max_proposals_per_step]
            telemetry_rows.append({"step": step, **dict(telemetry)})
            safe_at_step: list[tuple[float, int]] = []
            safe_expert_at_step: list[tuple[float, int]] = []
            for proposal in proposals:
                if not isinstance(proposal, BackupProposal):
                    # Tests and alternative proposal sources may use any object
                    # exposing the same immutable fields.
                    plan = np.asarray(proposal.plan, dtype=np.float32).copy()
                    kind = str(getattr(proposal, "kind", "proposal"))
                    internal = getattr(proposal, "internal_feasible", None)
                else:
                    plan = np.asarray(proposal.plan, dtype=np.float32).copy()
                    kind = proposal.kind
                    internal = proposal.internal_feasible
                if plan.shape != (10, 2) or not np.isfinite(plan).all():
                    raise ValueError(f"proposal must be a finite H=10 plan, got {plan.shape}")
                if float(np.max(np.abs(plan), initial=0.0)) > 1.0 + 1.0e-6:
                    raise ValueError("SafeMPPI proposal violates the fixed |u| <= 1 action bound")

                generated_hash = query_content_hash(context, gamma, plan)
                verifier_plan = plan.copy()
                result = verify_fn(
                    context.verifier_state,
                    verifier_plan,
                    env,
                    gamma,
                    goal=goal,
                )
                # A verifier must never mutate the object it was asked about.
                if not np.array_equal(verifier_plan, plan):
                    raise RuntimeError("full verifier mutated its planned-window input")
                verifier_hash = query_content_hash(context, gamma, verifier_plan)
                if generated_hash != verifier_hash:
                    raise RuntimeError("generated and fully verified plan identities differ")

                query_index = len(query_plans)
                query_plans.append(plan)
                query_steps.append(step)
                query_hashes.append(generated_hash)
                query_kinds.append(kind)
                query_internal_feasible.append(-1 if internal is None else int(bool(internal)))
                metrics = _proposal_metrics(result)
                query_metrics.append(metrics)
                if metrics["safe"]:
                    safe_at_step.append((metrics["progress_m"], query_index))
                    # ``debug_candidate`` rows are raw Monte-Carlo samples
                    # exposed by SafeMPPI for diagnostics.  Their individual
                    # controls have not been selected by SafeMPPI's cost, so
                    # ranking them together with ``weighted_mean`` and
                    # ``internal_best`` silently bypasses (among other terms)
                    # the expert's smoothness cost.  They remain fully queried
                    # for audit telemetry, but can never be executed or trained.
                    if kind != "debug_candidate":
                        safe_expert_at_step.append(
                            (metrics["progress_m"], query_index)
                        )

            if not safe_at_step:
                # Fail closed: no state transition and no target is emitted.
                dead_reason = "no_certified_plan"
                break
            if not safe_expert_at_step:
                # A safe raw debug draw is not a cost-selected expert output.
                # Executing it would bypass the configured smoothness cost.
                dead_reason = "no_certified_cost_selected_plan"
                break

            # Progress ranks verified-safe cost-selected SafeMPPI outputs only.
            # Stable query-index tie-breaking makes selection reproducible.
            _progress, selected_index = max(
                safe_expert_at_step, key=lambda pair: (pair[0], -pair[1])
            )
            selected_plan = query_plans[selected_index]
            selected_hash = query_hashes[selected_index]
            if selected_hash != query_content_hash(context, gamma, selected_plan):
                raise RuntimeError("selected training target identity changed after verification")
            selected_query_indices.append(selected_index)

            executed_action = np.asarray(selected_plan[0], dtype=np.float64).copy()
            next_state = execute_first_action(state, selected_plan)
            # The action log is the literal first row of the saved target.
            if not np.array_equal(
                executed_action.astype(selected_plan.dtype, copy=False), selected_plan[0]
            ):
                raise RuntimeError("executed action is not selected_plan[0]")
            executed_actions.append(executed_action.astype(np.float32))
            state = next_state
            states.append(state.copy())

            position = state[:2]
            clearance = float(_environment_clearance(position[None], env)[0])
            if clearance < 0.0:
                dead_reason = "collision_after_verified_action"
                break
            if bool(np.any(position < 0.0) or np.any(position > 5.0)):
                dead_reason = "out_of_bounds_after_verified_action"
                break
            if float(np.linalg.norm(position - goal)) < config.reach_m:
                break
        else:
            dead_reason = "timeout"

    states_array = np.asarray(states, dtype=np.float32)
    path = states_array[:, :2]
    actions_array = np.asarray(executed_actions, dtype=np.float32).reshape(-1, 2)
    query_plan_array = np.asarray(query_plans, dtype=np.float32).reshape(-1, 10, 2)
    selected_indices_array = np.asarray(selected_query_indices, dtype=np.int64)
    training_plans = (
        query_plan_array[selected_indices_array]
        if len(selected_indices_array)
        else np.empty((0, 10, 2), dtype=np.float32)
    )
    training_hashes = [query_hashes[index] for index in selected_query_indices]
    selected_context_steps = [query_steps[index] for index in selected_query_indices]
    if selected_context_steps != list(range(len(training_plans))):
        raise RuntimeError("each executed action must select exactly one query at its current step")
    if len(training_plans) != len(actions_array):
        raise RuntimeError("one exact planned target is required for every executed action")
    if len(training_plans) and not np.array_equal(training_plans[:, 0], actions_array):
        raise RuntimeError("saved target first actions differ from executed action log")

    endpoint_distance = float(np.linalg.norm(path[-1] - goal))
    reached = endpoint_distance < config.reach_m
    clearance = _environment_clearance(path.astype(np.float64), env)
    collision = bool(np.min(clearance, initial=np.inf) < 0.0)
    in_bounds = bool(np.all((path >= 0.0) & (path <= 5.0)))
    success = bool(reached and not collision and in_bounds and dead_reason is None)
    if dead_reason is None and not reached:
        dead_reason = "timeout"
    route_class = direction_class(path) if success else "unclassified"
    status = "success" if success else str(dead_reason)

    metric_arrays: dict[str, np.ndarray] = {}
    metric_fields = (
        "safe",
        "in_bounds",
        "socp_ok",
        "bounds_margin_m",
        "physical_clearance_m",
        "face_margin_m",
        "certificate_residual",
        "certificate_worst_step",
        "progress_m",
        "start_goal_distance_m",
        "terminal_goal_distance_m",
    )
    for field in metric_fields:
        values = [row[field] for row in query_metrics]
        if field in {"safe", "in_bounds", "socp_ok"}:
            metric_arrays[f"query_{field}"] = np.asarray(values, dtype=bool)
        elif field == "certificate_worst_step":
            metric_arrays[f"query_{field}"] = np.asarray(values, dtype=np.int16)
        else:
            metric_arrays[f"query_{field}"] = np.asarray(values, dtype=np.float64)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rollout_config": _rollout_config(config),
        "gamma": float(gamma),
        "seed": int(seed),
        "status": status,
        "success": success,
        "reached": bool(reached),
        "collision": collision,
        "in_bounds": in_bounds,
        "dead_reason": dead_reason,
        "direction_class": route_class,
        "steps": len(actions_array),
        "queries": len(query_plans),
        "safe_queries": int(sum(row["safe"] for row in query_metrics)),
        "query_acceptance": float(np.mean([row["safe"] for row in query_metrics]))
        if query_metrics
        else 0.0,
        "endpoint_distance_m": endpoint_distance,
        "min_clearance_m": float(np.min(clearance, initial=np.inf)),
        "path_length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "wall_seconds": time.perf_counter() - started,
        "initial_state": initial_state.astype(np.float32),
        "goal": goal.astype(np.float32),
        "states": states_array,
        "path": path,
        "executed_actions": actions_array,
        "contexts": contexts,
        "query_plans": query_plan_array,
        "query_steps": np.asarray(query_steps, dtype=np.int32),
        "query_hashes": query_hashes,
        "query_kinds": query_kinds,
        "query_internal_feasible": np.asarray(query_internal_feasible, dtype=np.int8),
        "selected_query_indices": selected_indices_array,
        "training_plans": training_plans,
        "training_hashes": training_hashes,
        "telemetry": telemetry_rows,
        **metric_arrays,
    }
    return result


def _episode_stem(gamma: float, seed: int) -> str:
    return f"g{gamma_tag(gamma)}_seed{int(seed)}"


def save_episode(episode: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    """Persist one candidate with enough data to revalidate every query hash."""

    directory.mkdir(parents=True, exist_ok=True)
    stem = _episode_stem(episode["gamma"], episode["seed"])
    array_path = directory / f"{stem}.npz"
    meta_path = directory / f"{stem}.json"
    contexts = _context_arrays(episode["contexts"])
    temporary = array_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            states=episode["states"],
            executed_actions=episode["executed_actions"],
            query_plans=episode["query_plans"],
            query_steps=episode["query_steps"],
            query_hashes=np.asarray(episode["query_hashes"], dtype="U64"),
            query_kinds=np.asarray(episode["query_kinds"], dtype="U32"),
            query_internal_feasible=episode["query_internal_feasible"],
            selected_query_indices=episode["selected_query_indices"],
            **contexts,
            **{
                key: value
                for key, value in episode.items()
                if key.startswith("query_")
                and isinstance(value, np.ndarray)
                and key
                not in {
                    "query_plans",
                    "query_steps",
                    "query_internal_feasible",
                }
            },
        )
    temporary.replace(array_path)
    omitted = {
        "initial_state",
        "states",
        "path",
        "executed_actions",
        "contexts",
        "query_plans",
        "query_steps",
        "query_hashes",
        "query_kinds",
        "query_internal_feasible",
        "selected_query_indices",
        "training_plans",
        "training_hashes",
        "telemetry",
    }
    omitted.update(key for key in episode if key.startswith("query_") and isinstance(episode[key], np.ndarray))
    metadata = {key: value for key, value in episode.items() if key not in omitted}
    metadata.update(
        {
            "array_file": array_path.name,
            "array_sha256": sha256_file(array_path),
            "generator_sha256": sha256_file(Path(__file__)),
            "training_hashes": episode["training_hashes"],
            "telemetry": episode["telemetry"],
            "identity_contract": {
                "generated_equals_verifier_input_equals_training_target": True,
                "executed_action_equals_training_plan_first_action": True,
            },
        }
    )
    _atomic_json(meta_path, metadata)
    return array_path, meta_path


def load_episode(meta_path: Path, *, validate: bool = True) -> dict[str, Any]:
    metadata = json.loads(meta_path.read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "legacy planned-demo artifact lacks exact verifier query identity; "
            "regenerate Stage 02"
        )
    array_path = meta_path.parent / metadata["array_file"]
    if sha256_file(array_path) != metadata["array_sha256"]:
        raise RuntimeError(f"candidate array checksum mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    contexts = [
        QueryContext(
            arrays["context_grid"][i],
            arrays["context_low5"][i],
            arrays["context_hist"][i],
            arrays["context_verifier_state"][i],
            str(arrays["context_verifier_spec_fingerprint"][i]),
        )
        for i in range(len(arrays["context_grid"]))
    ]
    query_hashes = arrays["query_hashes"].astype(str).tolist()
    selected = arrays["selected_query_indices"].astype(np.int64)
    training_plans = arrays["query_plans"][selected]
    training_hashes = [query_hashes[index] for index in selected]
    episode: dict[str, Any] = {
        **metadata,
        **arrays,
        "contexts": contexts,
        "path": arrays["states"][:, :2],
        "query_hashes": query_hashes,
        "query_kinds": arrays["query_kinds"].astype(str).tolist(),
        "training_plans": training_plans,
        "training_hashes": training_hashes,
    }
    if validate:
        gamma = float(metadata["gamma"])
        steps = arrays["query_steps"].astype(int)
        plans = arrays["query_plans"]
        if len(steps) != len(plans) or len(query_hashes) != len(plans):
            raise RuntimeError(f"query array length mismatch: {array_path}")
        for index, (step, plan, stored_hash) in enumerate(zip(steps, plans, query_hashes)):
            if not 0 <= step < len(contexts):
                raise RuntimeError(f"query {index} has invalid context step {step}")
            actual = query_content_hash(contexts[step], gamma, plan)
            if actual != stored_hash:
                raise RuntimeError(f"query identity mismatch at {array_path}:{index}")
        if len(selected) != len(arrays["executed_actions"]):
            raise RuntimeError("selected queries/actions length mismatch")
        if len(selected) and not np.array_equal(
            plans[selected, 0], arrays["executed_actions"]
        ):
            raise RuntimeError("loaded executed actions differ from selected plan[0]")
        if training_hashes != metadata["training_hashes"]:
            raise RuntimeError("loaded training-target identity list differs from manifest")
    return episode


def legacy_seed_hints(gamma: float, path: Path = LEGACY_HINT_PATH) -> tuple[list[int], dict[str, Any]]:
    """Read only legacy ``gammas/seeds/signatures`` as candidate-order hints."""

    if not path.exists():
        return [], {"available": False, "path": str(path)}
    with np.load(path, allow_pickle=False) as payload:
        # Do not access legacy paths, states, controls, windows, or checkpoints.
        old_gammas = payload["gammas"].astype(float)
        old_seeds = payload["seeds"].astype(np.int64)
        old_signatures = payload["signatures"].astype(str)
    mask = np.isclose(old_gammas, float(gamma), atol=1.0e-7)
    right = [int(seed) for seed, word in zip(old_seeds[mask], old_signatures[mask]) if word.startswith("R")]
    up = [int(seed) for seed, word in zip(old_seeds[mask], old_signatures[mask]) if word.startswith("U")]
    ordered: list[int] = []
    ordered_modes: list[str] = []
    for index in range(max(len(right), len(up))):
        if index < len(right):
            ordered.append(right[index])
            ordered_modes.append("R-first")
        if index < len(up):
            ordered.append(up[index])
            ordered_modes.append("U-first")
    # Stable de-duplication; a seed is evaluated once under the new contract.
    unique: dict[int, str] = {}
    for seed, mode in zip(ordered, ordered_modes):
        unique.setdefault(seed, mode)
    ordered = list(unique)
    ordered_modes = [unique[seed] for seed in ordered]
    return ordered, {
        "available": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "fields_read": ["gammas", "seeds", "signatures"],
        "fields_explicitly_not_reused": [
            "paths",
            "states",
            "controls",
            "legacy training windows",
            "legacy checkpoints",
        ],
        "matched_hint_count": len(ordered),
        "ordered_requested_modes": ordered_modes,
    }


def candidate_seed_order(gamma: float, config: DemoRunConfig) -> tuple[list[int], dict[str, Any]]:
    hints, provenance = legacy_seed_hints(gamma)
    fallback = range(config.seed0, config.seed0 + config.max_candidate_seeds_per_gamma * 4)
    ordered = list(dict.fromkeys([*hints, *fallback]))
    return ordered[: config.max_candidate_seeds_per_gamma], provenance


def mode_paired_sweep_seed_schedule(
    gammas: Sequence[float], config: DemoRunConfig, seeds_per_gamma: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Freeze matched legacy-hinted R/U seeds for every sweep cell."""

    if seeds_per_gamma <= 0 or seeds_per_gamma % 2:
        raise ValueError("a mode-paired sweep requires a positive even seeds-per-gamma")
    schedule: dict[str, list[dict[str, Any]]] = {}
    provenance_by_gamma: dict[str, Any] = {}
    for gamma in gammas:
        ordered, provenance = candidate_seed_order(float(gamma), config)
        modes = list(provenance.get("ordered_requested_modes", ()))
        if len(modes) < seeds_per_gamma or int(provenance.get("matched_hint_count", 0)) < seeds_per_gamma:
            raise RuntimeError(
                f"gamma={gamma:g} lacks {seeds_per_gamma} legacy-hinted mode-paired seeds"
            )
        rows = [
            {"seed": int(seed), "requested_mode": str(mode)}
            for seed, mode in zip(ordered[:seeds_per_gamma], modes[:seeds_per_gamma])
        ]
        requested_r = sum(row["requested_mode"] == "R-first" for row in rows)
        requested_u = sum(row["requested_mode"] == "U-first" for row in rows)
        if requested_r != requested_u:
            raise RuntimeError(
                f"gamma={gamma:g} fixed sweep seeds are not R/U paired: "
                f"R={requested_r}, U={requested_u}"
            )
        tag = gamma_tag(gamma)
        schedule[tag] = rows
        provenance_by_gamma[tag] = provenance
    return schedule, provenance_by_gamma


def _candidate_meta_path(directory: Path, gamma: float, seed: int) -> Path:
    return directory / f"{_episode_stem(gamma, seed)}.json"


def _quality_score(episode: dict[str, Any]) -> tuple[float, float, int]:
    path = np.asarray(episode["path"], dtype=np.float64)
    goal_distance = np.linalg.norm(path - GOAL.astype(np.float64)[None], axis=1)
    retreat = float(np.maximum(np.diff(goal_distance), 0.0).sum())
    # Prefer less backtracking, then higher clearance, then shorter execution.
    return (
        retreat,
        -float(episode["min_clearance_m"]),
        int(episode["steps"]),
    )


def select_exact_balance(
    episodes: Iterable[dict[str, Any]], quota: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = {
        label: [
            episode
            for episode in episodes
            if episode["success"] and episode["direction_class"] == label
        ]
        for label in ("R-first", "U-first")
    }
    for label in groups:
        groups[label].sort(key=lambda item: (_quality_score(item), int(item["seed"])))
    if any(len(groups[label]) < quota for label in groups):
        raise RuntimeError(
            "insufficient real balanced trajectories: "
            + ", ".join(f"{label}={len(groups[label])}/{quota}" for label in groups)
        )
    selected = groups["R-first"][:quota] + groups["U-first"][:quota]
    selected.sort(key=lambda item: (item["direction_class"], int(item["seed"])))
    audit = {
        "R-first": len([item for item in selected if item["direction_class"] == "R-first"]),
        "U-first": len([item for item in selected if item["direction_class"] == "U-first"]),
        "real_successful_trajectories_only": bool(all(item["success"] for item in selected)),
        "synthetic_reflections": 0,
        "trajectory_padding": 0,
        "target_padding": 0,
    }
    return selected, audit


def build_dataset(
    selected_by_gamma: dict[float, list[dict[str, Any]]], output: Path
) -> dict[str, Any]:
    grids: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    verifier_states: list[np.ndarray] = []
    verifier_spec_fingerprints: list[str] = []
    plans: list[np.ndarray] = []
    plan_kinds: list[str] = []
    hashes: list[str] = []
    gammas: list[float] = []
    seeds: list[int] = []
    trajectory_ids: list[int] = []
    trajectory_steps: list[int] = []
    directions: list[int] = []
    query_progress: list[float] = []
    query_clearance: list[float] = []
    target_safe: list[bool] = []
    target_in_bounds: list[bool] = []
    target_socp_ok: list[bool] = []
    trajectory_rows: list[dict[str, Any]] = []

    trajectory_id = 0
    for gamma in sorted(selected_by_gamma):
        for episode in selected_by_gamma[gamma]:
            if not bool(episode["success"]):
                raise RuntimeError("only real successful trajectories may enter pretraining")
            if episode["direction_class"] not in {"R-first", "U-first"}:
                raise RuntimeError("an unclassified trajectory cannot fill a geometric quota")
            selected = np.asarray(episode["selected_query_indices"], dtype=np.int64)
            query_steps = np.asarray(episode["query_steps"], dtype=np.int64)
            query_plans = np.asarray(episode["query_plans"], dtype=np.float32)
            query_hashes = list(episode["query_hashes"])
            query_kinds = list(episode["query_kinds"])
            contexts: list[QueryContext] = episode["contexts"]
            if len(selected) != int(episode["steps"]):
                raise RuntimeError("successful trajectory is missing selected planned targets")
            for local_step, query_index in enumerate(selected):
                context_step = int(query_steps[query_index])
                if context_step != local_step:
                    raise RuntimeError("selected target/context step mismatch")
                context = contexts[context_step]
                plan = query_plans[query_index]
                plan_kind = str(query_kinds[query_index])
                if plan_kind not in {"weighted_mean", "internal_best"}:
                    raise RuntimeError(
                        "raw/debug or unknown proposal cannot become an expert training target: "
                        f"kind={plan_kind!r}"
                    )
                identity = query_content_hash(context, gamma, plan)
                if identity != query_hashes[query_index]:
                    raise RuntimeError("training target differs from fully verified query")
                grids.append(np.asarray(context.grid, dtype=np.float32))
                lows.append(np.asarray(context.low5, dtype=np.float32))
                histories.append(np.asarray(context.hist, dtype=np.float32))
                verifier_states.append(
                    np.asarray(context.verifier_state, dtype=np.float64)
                )
                verifier_spec_fingerprints.append(
                    context.verifier_spec_fingerprint
                )
                plans.append(plan.copy())
                plan_kinds.append(plan_kind)
                hashes.append(identity)
                gammas.append(float(gamma))
                seeds.append(int(episode["seed"]))
                trajectory_ids.append(trajectory_id)
                trajectory_steps.append(local_step)
                directions.append(0 if episode["direction_class"] == "R-first" else 1)
                query_progress.append(float(episode["query_progress_m"][query_index]))
                query_clearance.append(
                    float(episode["query_physical_clearance_m"][query_index])
                )
                target_safe.append(bool(episode["query_safe"][query_index]))
                target_in_bounds.append(bool(episode["query_in_bounds"][query_index]))
                target_socp_ok.append(bool(episode["query_socp_ok"][query_index]))
            trajectory_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "gamma": float(gamma),
                    "seed": int(episode["seed"]),
                    "direction_class": episode["direction_class"],
                    "steps": int(episode["steps"]),
                    "min_clearance_m": float(episode["min_clearance_m"]),
                    "path_length_m": float(episode["path_length_m"]),
                    "query_acceptance": float(episode["query_acceptance"]),
                }
            )
            trajectory_id += 1

    if not plans:
        raise RuntimeError("cannot build a training dataset with no verified planned targets")
    if not all(target_safe) or not all(target_in_bounds) or not all(target_socp_ok):
        raise RuntimeError("a selected training target is not bounds+SOCP verified-safe")
    trajectory_id_array = np.asarray(trajectory_ids, dtype=np.int64)
    per_trajectory_windows = np.bincount(trajectory_id_array, minlength=trajectory_id)
    trajectory_balanced_weight = 1.0 / per_trajectory_windows[trajectory_id_array]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "grid": torch.from_numpy(np.asarray(grids, dtype=np.float32)),
        "low5": torch.from_numpy(np.asarray(lows, dtype=np.float32)),
        "hist": torch.from_numpy(np.asarray(histories, dtype=np.float32)),
        "verifier_state": torch.from_numpy(
            np.asarray(verifier_states, dtype=np.float64)
        ),
        "verifier_spec_fingerprint": list(verifier_spec_fingerprints),
        "U": torch.from_numpy(np.asarray(plans, dtype=np.float32)),
        "window_plan_kind": list(plan_kinds),
        "gamma": torch.tensor(gammas, dtype=torch.float32),
        "window_seeds": torch.tensor(seeds, dtype=torch.long),
        "window_trajectory_ids": torch.from_numpy(trajectory_id_array),
        "source_trajectory_ids": torch.from_numpy(trajectory_id_array.copy()),
        # Sampling/loss with this weight assigns total mass one to every real
        # trajectory; exact 12/12 trajectory balance therefore remains exact
        # even when R-first and U-first paths have different durations.
        "trajectory_balanced_weight": torch.from_numpy(
            trajectory_balanced_weight.astype(np.float32)
        ),
        "window_steps": torch.tensor(trajectory_steps, dtype=torch.int32),
        "window_direction": torch.tensor(directions, dtype=torch.int8),
        "query_progress_m": torch.tensor(query_progress, dtype=torch.float32),
        "query_physical_clearance_m": torch.tensor(query_clearance, dtype=torch.float32),
        "target_safe": torch.tensor(target_safe, dtype=torch.bool),
        "target_in_bounds": torch.tensor(target_in_bounds, dtype=torch.bool),
        "target_socp_ok": torch.tensor(target_socp_ok, dtype=torch.bool),
        "query_hashes": hashes,
        "target_query_hash": list(hashes),
        "generated_hash": list(hashes),
        "verifier_input_hash": list(hashes),
        "training_target_hash": list(hashes),
        "generated_hashes": list(hashes),
        "verifier_input_hashes": list(hashes),
        "training_target_hashes": list(hashes),
        "trajectory_rows": trajectory_rows,
        "start": torch.from_numpy(START.copy()),
        "goal": torch.from_numpy(GOAL.copy()),
        "contract": {
            "generated_equals_verified_equals_training": True,
            "planned_horizon": 10,
            "only_first_action_executed": True,
            "all_targets_pre_execution_fully_verified": True,
            "progress_not_in_safety_label": True,
            "synthetic_reflections": 0,
            "padding": 0,
            "debug_training_targets": 0,
            "debug_target_share": 0.0,
            "allowed_training_plan_kinds": ["weighted_mean", "internal_best"],
            "trajectory_balanced_total_mass_per_path": 1.0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return payload


def _draw_scene(axis: Any, env: Any) -> None:
    obstacles = env.obstacles.detach().cpu().numpy()
    for x, y, radius in obstacles:
        axis.add_patch(Circle((float(x), float(y)), float(radius), color="0.72", zorder=1))
    axis.scatter(*START, marker="s", s=34, color="black", zorder=6)
    axis.scatter(*GOAL, marker="*", s=120, color="#ffd21f", edgecolor="black", zorder=6)
    axis.set(xlim=(-0.15, 5.15), ylim=(-0.15, 5.15), aspect="equal")
    axis.set_xticks([])
    axis.set_yticks([])


def render_selected(
    env: Any, selected_by_gamma: dict[float, list[dict[str, Any]]], output: Path
) -> None:
    gamma_values = sorted(selected_by_gamma)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(gamma_values)))
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
    for axis, gamma, color in zip(axes.ravel(), gamma_values, colors):
        _draw_scene(axis, env)
        for episode in selected_by_gamma[gamma]:
            linestyle = "-" if episode["direction_class"] == "R-first" else "--"
            path = np.asarray(episode["path"])
            axis.plot(path[:, 0], path[:, 1], color=color, lw=1.0, alpha=0.66, ls=linestyle)
        r_count = sum(item["direction_class"] == "R-first" for item in selected_by_gamma[gamma])
        u_count = sum(item["direction_class"] == "U-first" for item in selected_by_gamma[gamma])
        axis.set_title(rf"$\gamma={gamma:g}$: R={r_count}, U={u_count}")
    for axis in axes.ravel()[len(gamma_values):]:
        axis.axis("off")
    fig.suptitle(
        "Real fully verified planned-window SafeMPPI demonstrations\n"
        "solid: R-first, dashed: U-first (no reflection or padding)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_target_audit(payload: dict[str, Any], output: Path) -> None:
    gamma = payload["gamma"].cpu().numpy()
    progress = payload["query_progress_m"].cpu().numpy()
    clearance = payload["query_physical_clearance_m"].cpu().numpy()
    direction = payload["window_direction"].cpu().numpy()
    gammas = sorted(set(float(value) for value in gamma))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for code, label, marker in ((0, "R-first", "o"), (1, "U-first", "^")):
        mask = direction == code
        axes[0].scatter(gamma[mask], progress[mask], s=7, alpha=0.24, marker=marker, label=label)
        axes[1].scatter(gamma[mask], clearance[mask], s=7, alpha=0.24, marker=marker, label=label)
    axes[0].axhline(0.0, color="black", lw=0.7, ls=":")
    axes[0].set(title="Verified training plans: progress (ranking only)", ylabel="H=10 progress [m]")
    axes[1].axhline(0.0, color="black", lw=0.7, ls=":")
    axes[1].set(title="Verified training plans: physical clearance", ylabel="clearance [m]")
    for axis in axes:
        axis.set_xlabel(r"safety level $\gamma$")
        axis.set_xticks(gammas)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("The exact planned samples passed to pretraining")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _summarize_candidates(episodes: Sequence[dict[str, Any]], gamma: float) -> dict[str, Any]:
    return {
        "gamma": float(gamma),
        "candidates": len(episodes),
        "successes": sum(bool(item["success"]) for item in episodes),
        "R-first_successes": sum(
            bool(item["success"]) and item["direction_class"] == "R-first" for item in episodes
        ),
        "U-first_successes": sum(
            bool(item["success"]) and item["direction_class"] == "U-first" for item in episodes
        ),
        "unclassified_successes": sum(
            bool(item["success"]) and item["direction_class"] == "unclassified" for item in episodes
        ),
        "fail_closed": sum(
            str(item.get("dead_reason", "")).startswith("no_certified")
            for item in episodes
        ),
        "queries": sum(int(item["queries"]) for item in episodes),
        "safe_queries": sum(int(item["safe_queries"]) for item in episodes),
    }


def _target_metrics(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize selected exact plans eligible for the Stage-3 dataset."""

    plans: list[np.ndarray] = []
    kinds: list[str] = []
    eligible = [
        episode
        for episode in episodes
        if bool(episode["success"])
        and str(episode["direction_class"]) in {"R-first", "U-first"}
    ]
    for episode in eligible:
        indices = np.asarray(episode["selected_query_indices"], dtype=np.int64)
        query_plans = np.asarray(episode["query_plans"], dtype=np.float32)
        query_kinds = list(episode["query_kinds"])
        plans.extend(query_plans[index].copy() for index in indices)
        kinds.extend(str(query_kinds[index]) for index in indices)

    count = len(plans)
    declared_kinds = ("weighted_mean", "internal_best", "debug_candidate")
    kind_counts = {kind: kinds.count(kind) for kind in declared_kinds}
    for kind in sorted(set(kinds) - set(declared_kinds)):
        kind_counts[kind] = kinds.count(kind)
    debug_count = kind_counts["debug_candidate"]
    if count:
        array = np.asarray(plans, dtype=np.float32)
        adjacent_jump = float(np.abs(np.diff(array, axis=1)).mean())
        saturation = float((np.abs(array) > 0.95).mean())
        absolute_action = float(np.abs(array).mean())
    else:
        adjacent_jump = None
        saturation = None
        absolute_action = None
    return {
        "training_eligible_trajectories": len(eligible),
        "training_targets": count,
        "target_kind_counts": kind_counts,
        "cost_selected_targets": count - debug_count,
        "cost_selected_target_share": None if not count else (count - debug_count) / count,
        "debug_targets": debug_count,
        "debug_target_share": 0.0 if not count else debug_count / count,
        "mean_adjacent_action_jump": adjacent_jump,
        "saturated_action_coordinate_share": saturation,
        "mean_absolute_action": absolute_action,
    }


def summarize_smoothness_sweep_cell(
    episodes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return matched metrics for one smoothness/noise sweep cell."""

    attempts = len(episodes)
    if not attempts:
        raise ValueError("a smoothness-sweep cell cannot be empty")
    successful = [episode for episode in episodes if bool(episode["success"])]
    clearance = [float(episode["min_clearance_m"]) for episode in successful]
    successful_steps = [int(episode["steps"]) for episode in successful]
    queries = sum(int(episode["queries"]) for episode in episodes)
    safe_queries = sum(int(episode["safe_queries"]) for episode in episodes)
    observed_r = sum(
        bool(episode["success"]) and episode["direction_class"] == "R-first"
        for episode in episodes
    )
    observed_u = sum(
        bool(episode["success"]) and episode["direction_class"] == "U-first"
        for episode in episodes
    )
    return {
        "attempts": attempts,
        "requested_R-first_attempts": sum(
            episode.get("requested_mode") == "R-first" for episode in episodes
        ),
        "requested_U-first_attempts": sum(
            episode.get("requested_mode") == "U-first" for episode in episodes
        ),
        "successes": len(successful),
        "success_rate": len(successful) / attempts,
        "R-first_successes": observed_r,
        "U-first_successes": observed_u,
        "observed_R-first_successes": observed_r,
        "observed_U-first_successes": observed_u,
        "unclassified_successes": sum(
            bool(episode["success"]) and episode["direction_class"] == "unclassified"
            for episode in episodes
        ),
        "fail_closed": sum(
            str(episode.get("dead_reason", "")).startswith("no_certified")
            for episode in episodes
        ),
        "query_acceptance": safe_queries / queries if queries else 0.0,
        "mean_min_clearance_m_success": float(np.mean(clearance)) if clearance else None,
        "minimum_clearance_m_success": min(clearance) if clearance else None,
        "mean_successful_steps": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "mean_time_to_goal_s_success": (
            0.1 * float(np.mean(successful_steps)) if successful_steps else None
        ),
        "wall_seconds": sum(float(episode["wall_seconds"]) for episode in episodes),
        "mean_wall_seconds_per_attempt": float(
            np.mean([float(episode["wall_seconds"]) for episode in episodes])
        ),
        **_target_metrics(episodes),
    }


def _sweep_episode_row(episode: dict[str, Any]) -> dict[str, Any]:
    target = _target_metrics((episode,))
    return {
        "gamma": float(episode["gamma"]),
        "seed": int(episode["seed"]),
        "requested_mode": episode.get("requested_mode"),
        "success": bool(episode["success"]),
        "direction_class": str(episode["direction_class"]),
        "dead_reason": episode.get("dead_reason"),
        "steps": int(episode["steps"]),
        "time_to_goal_s": (
            0.1 * int(episode["steps"]) if bool(episode["success"]) else None
        ),
        "queries": int(episode["queries"]),
        "query_acceptance": float(episode["query_acceptance"]),
        "min_clearance_m": float(episode["min_clearance_m"]),
        "endpoint_distance_m": float(episode["endpoint_distance_m"]),
        "path_length_m": float(episode["path_length_m"]),
        "wall_seconds": float(episode["wall_seconds"]),
        **target,
    }


def run_smoothness_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed-seed 3x3 SafeMPPI expert-target quality sweep."""

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    gammas = tuple(float(value) for value in args.gammas)
    unknown = sorted(set(gammas) - set(float(value) for value in GAMMAS))
    if unknown:
        raise ValueError(f"unsupported gammas: {unknown}")
    smooth_weights = tuple(float(value) for value in args.sweep_smooth_weights)
    noise_values = tuple(float(value) for value in args.sweep_noise_var_mults)
    if any(not math.isfinite(value) or value < 0.0 for value in smooth_weights):
        raise ValueError("sweep smoothness weights must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0.0 for value in noise_values):
        raise ValueError("sweep noise multipliers must be finite and positive")
    if args.sweep_seeds_per_gamma <= 0 or args.sweep_seeds_per_gamma % 2:
        raise ValueError("sweep-seeds-per-gamma must be positive and even for R/U pairing")
    if args.max_proposals < args.max_debug_candidates + 2:
        raise ValueError(
            "the sweep must verify every returned SafeMPPI proposal: require "
            "max_proposals >= max_debug_candidates + 2"
        )

    outdir = (
        args.sweep_outdir
        if args.sweep_outdir is not None
        else args.outdir / "smoothness_sweep"
    ).resolve()
    for directory in (outdir / "logs", outdir / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
    dependency_manifest = write_dependency_manifest(outdir / "logs/dependencies.json")
    assert_no_legacy_expansion_imports()
    generator_sha256 = sha256_file(Path(__file__))
    schedule_config = DemoRunConfig(
        max_steps=args.max_steps,
        reach_m=args.reach,
        smooth_weight=smooth_weights[0],
        retreat_weight=args.retreat_weight,
        noise_var_mult=noise_values[0],
        max_debug_candidates=args.max_debug_candidates,
        max_proposals_per_step=args.max_proposals,
        quota_per_direction=1,
        max_candidate_seeds_per_gamma=args.sweep_seeds_per_gamma,
        seed0=args.seed0,
    )
    fixed_seed_schedule, seed_hint_provenance = mode_paired_sweep_seed_schedule(
        gammas, schedule_config, args.sweep_seeds_per_gamma
    )
    env = make_id_scene()
    env.T = int(args.max_steps)
    sweep_verifier_spec = verifier_spec_fingerprint(env, env.goal)
    started = time.perf_counter()
    cells: list[dict[str, Any]] = []

    for smooth_weight in smooth_weights:
        for noise_var_mult in noise_values:
            config = DemoRunConfig(
                max_steps=args.max_steps,
                reach_m=args.reach,
                smooth_weight=smooth_weight,
                retreat_weight=args.retreat_weight,
                noise_var_mult=noise_var_mult,
                max_debug_candidates=args.max_debug_candidates,
                max_proposals_per_step=args.max_proposals,
                quota_per_direction=1,
                max_candidate_seeds_per_gamma=args.sweep_seeds_per_gamma,
                seed0=args.seed0,
            )
            tag = f"smooth{smooth_weight:g}_noise{noise_var_mult:g}"
            cell_path = outdir / "logs" / f"{tag}.json"
            expected = {
                "schema_version": SWEEP_SCHEMA_VERSION,
                "generator_sha256": generator_sha256,
                "config": asdict(config),
                "gammas": list(gammas),
                "fixed_seed_schedule": fixed_seed_schedule,
                "verifier_spec_fingerprint": sweep_verifier_spec,
            }
            if cell_path.exists() and not args.sweep_overwrite:
                cached = json.loads(cell_path.read_text())
                if all(cached.get(key) == value for key, value in expected.items()):
                    cells.append(cached)
                    print(f"[smoothness-sweep] reuse complete cell {tag}", flush=True)
                    continue

            episodes: list[dict[str, Any]] = []
            for gamma in gammas:
                for seed_row in fixed_seed_schedule[gamma_tag(gamma)]:
                    seed = int(seed_row["seed"])
                    # A fresh planner per episode prevents warm-start state from
                    # leaking between gamma, seed, or hyperparameter cells.
                    episode = run_expert_rollout(
                        env=env,
                        gamma=gamma,
                        seed=seed,
                        device=device,
                        config=config,
                        backup=_default_backup(config),
                    )
                    episode["requested_mode"] = seed_row["requested_mode"]
                    episodes.append(episode)
                    print(
                        f"[smoothness-sweep {tag}] gamma={gamma:g} seed={seed} "
                        f"status={episode['status']} mode={episode['direction_class']} "
                        f"steps={episode['steps']}",
                        flush=True,
                    )
            overall = summarize_smoothness_sweep_cell(episodes)
            if overall["debug_targets"] != 0 or overall["debug_target_share"] != 0.0:
                raise RuntimeError(
                    "Stage2 teacher sweep emitted a raw debug rollout as a target"
                )
            per_gamma = {
                gamma_tag(gamma): summarize_smoothness_sweep_cell(
                    [episode for episode in episodes if episode["gamma"] == gamma]
                )
                for gamma in gammas
            }
            cell = {
                **expected,
                "status": "COMPLETE",
                "tag": tag,
                "smooth_weight": smooth_weight,
                "noise_var_mult": noise_var_mult,
                "overall": overall,
                "per_gamma": per_gamma,
                "episodes": [_sweep_episode_row(episode) for episode in episodes],
            }
            _atomic_json(cell_path, cell)
            cells.append(cell)
            print(
                f"[smoothness-sweep {tag}] SR={overall['success_rate']:.3f} "
                f"cost-share={overall['cost_selected_target_share']} "
                f"jump={overall['mean_adjacent_action_jump']} "
                f"sat={overall['saturated_action_coordinate_share']}",
                flush=True,
            )

    csv_rows = []
    for cell in cells:
        overall = cell["overall"]
        csv_rows.append({
            "smooth_weight": cell["smooth_weight"],
            "noise_var_mult": cell["noise_var_mult"],
            "attempts": overall["attempts"],
            "requested_R-first_attempts": overall["requested_R-first_attempts"],
            "requested_U-first_attempts": overall["requested_U-first_attempts"],
            "successes": overall["successes"],
            "success_rate": overall["success_rate"],
            "observed_R-first_successes": overall["observed_R-first_successes"],
            "observed_U-first_successes": overall["observed_U-first_successes"],
            "fail_closed": overall["fail_closed"],
            "mean_min_clearance_m_success": overall["mean_min_clearance_m_success"],
            "minimum_clearance_m_success": overall["minimum_clearance_m_success"],
            "mean_successful_steps": overall["mean_successful_steps"],
            "mean_time_to_goal_s_success": overall["mean_time_to_goal_s_success"],
            "training_targets": overall["training_targets"],
            "cost_selected_target_share": overall["cost_selected_target_share"],
            "debug_target_share": overall["debug_target_share"],
            "mean_adjacent_action_jump": overall["mean_adjacent_action_jump"],
            "saturated_action_coordinate_share": overall[
                "saturated_action_coordinate_share"
            ],
            "wall_seconds": overall["wall_seconds"],
            "mean_wall_seconds_per_attempt": overall["mean_wall_seconds_per_attempt"],
        })
    table_path = outdir / "tables/expert_target_smoothness_sweep.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "status": "COMPLETE",
        "created_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "generator_sha256": generator_sha256,
        "scene": "ordinary_symmetric_4x4_ID_stadium",
        "gammas": list(gammas),
        "fixed_seed_schedule": fixed_seed_schedule,
        "seed_hint_provenance": seed_hint_provenance,
        "verifier_spec_fingerprint": sweep_verifier_spec,
        "legacy_mechanisms": clean_method_absence_manifest(),
        "matched_axes": {
            "smooth_weights": list(smooth_weights),
            "noise_var_mults": list(noise_values),
            "all_other_DemoRunConfig_fields_fixed": True,
        },
        "target_population": (
            "selected exact plans from successful classified trajectories only"
        ),
        "target_contract": {
            "allowed_plan_kinds": ["weighted_mean", "internal_best"],
            "debug_training_targets": 0,
            "debug_target_share": 0.0,
        },
        "metric_definitions": {
            "mean_adjacent_action_jump": "mean |u[k+1]-u[k]| over target steps and coordinates",
            "saturated_action_coordinate_share": "share of target coordinates with |u| > 0.95",
            "mean_time_to_goal_s_success": "0.1 s times successful executed-step count",
        },
        "cells": cells,
        "table": str(table_path),
        "dependency_manifest": dependency_manifest,
    }
    _atomic_json(outdir / "manifest.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "cells": len(cells),
        "manifest": str(outdir / "manifest.json"),
        "table": str(table_path),
    }, indent=2), flush=True)
    return summary


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    gammas = tuple(float(value) for value in args.gammas)
    unknown = sorted(set(gammas) - set(float(value) for value in GAMMAS))
    if unknown:
        raise ValueError(f"unsupported gammas: {unknown}")
    config = DemoRunConfig(
        max_steps=args.max_steps,
        reach_m=args.reach,
        smooth_weight=args.smooth_weight,
        retreat_weight=args.retreat_weight,
        noise_var_mult=args.noise_var_mult,
        max_debug_candidates=args.max_debug_candidates,
        max_proposals_per_step=args.max_proposals,
        quota_per_direction=args.quota,
        max_candidate_seeds_per_gamma=args.max_candidate_seeds,
        seed0=args.seed0,
    )
    if not args.smoke and config.max_proposals_per_step < config.max_debug_candidates + 2:
        raise ValueError(
            "full runs must verify every returned SafeMPPI proposal: require "
            "max_proposals >= max_debug_candidates + 2"
        )
    outdir = args.outdir.resolve()
    candidate_dir = outdir / "data/candidates"
    for directory in (candidate_dir, outdir / "data", outdir / "logs", outdir / "tables", outdir / "viz"):
        directory.mkdir(parents=True, exist_ok=True)
    dependency_manifest = write_dependency_manifest(outdir / "logs/dependencies.json")
    assert_no_legacy_expansion_imports()
    env = make_id_scene()
    env.T = int(config.max_steps)
    started = time.perf_counter()
    selected_by_gamma: dict[float, list[dict[str, Any]]] = {}
    all_by_gamma: dict[float, list[dict[str, Any]]] = {}
    hint_rows: dict[str, Any] = {}

    for gamma in gammas:
        seed_order, hint_info = candidate_seed_order(gamma, config)
        hint_rows[gamma_tag(gamma)] = hint_info
        episodes: list[dict[str, Any]] = []
        for ordinal, seed in enumerate(seed_order, start=1):
            meta_path = _candidate_meta_path(candidate_dir, gamma, seed)
            if meta_path.exists():
                episode = load_episode(meta_path)
                reusable = (
                    episode.get("schema_version") == SCHEMA_VERSION
                    and episode.get("rollout_config") == _rollout_config(config)
                    and episode.get("generator_sha256") == sha256_file(Path(__file__))
                )
                if not reusable:
                    print(
                        f"[planned-demo] stale candidate regenerated: gamma={gamma:g} seed={seed}",
                        flush=True,
                    )
                    backup = _default_backup(config)
                    episode = run_expert_rollout(
                        env=env,
                        gamma=gamma,
                        seed=seed,
                        device=device,
                        config=config,
                        backup=backup,
                    )
                    save_episode(episode, candidate_dir)
            else:
                backup = _default_backup(config)
                episode = run_expert_rollout(
                    env=env,
                    gamma=gamma,
                    seed=seed,
                    device=device,
                    config=config,
                    backup=backup,
                )
                save_episode(episode, candidate_dir)
            episodes.append(episode)
            counts = _summarize_candidates(episodes, gamma)
            print(
                f"[planned-demo] gamma={gamma:g} seed={seed} ({ordinal}/{len(seed_order)}) "
                f"status={episode['status']} class={episode['direction_class']} "
                f"R={counts['R-first_successes']} U={counts['U-first_successes']} "
                f"queries={episode['queries']}",
                flush=True,
            )
            if args.smoke:
                if ordinal >= args.smoke_seeds:
                    break
            elif (
                counts["R-first_successes"] >= config.quota_per_direction
                and counts["U-first_successes"] >= config.quota_per_direction
            ):
                break
        all_by_gamma[gamma] = episodes
        if not args.smoke:
            selected, balance = select_exact_balance(episodes, config.quota_per_direction)
            selected_by_gamma[gamma] = selected
            print(f"[balance] gamma={gamma:g} {balance}", flush=True)

    candidate_rows = [_summarize_candidates(all_by_gamma[gamma], gamma) for gamma in gammas]
    with (outdir / "tables/candidate_census.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    dataset_path: Path | None = None
    dataset_hash: str | None = None
    window_count = 0
    if not args.smoke:
        dataset_path = outdir / "data/planned_id_balanced.pt"
        payload = build_dataset(selected_by_gamma, dataset_path)
        dataset_hash = sha256_file(dataset_path)
        window_count = len(payload["U"])
        render_selected(env, selected_by_gamma, outdir / "viz/selected_real_paths.png")
        render_target_audit(payload, outdir / "viz/training_target_audit.png")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "SMOKE_COMPLETE" if args.smoke else "PLANNED_DEMOS_COMPLETE",
        "created_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_visible_devices": str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scene": {
            "name": "ordinary_symmetric_4x4_ID_stadium",
            "start": START.tolist(),
            "goal": GOAL.tolist(),
            "gammas": list(gammas),
        },
        "config": asdict(config),
        "legacy_mechanisms": clean_method_absence_manifest(),
        "contract": {
            "generated_object": "SafeMPPI planned H=10 control window",
            "queried_object": "same planned H=10 control window",
            "verified_object": "same planned H=10 control window",
            "training_object": "same planned H=10 control window",
            "executed_object": "only first action of selected verified-safe plan",
            "safe_selection": (
                "maximum progress among full-verifier-safe cost-selected SafeMPPI "
                "outputs; raw debug rollouts are audit-only"
            ),
            "no_safe_behavior": (
                "fail closed with no action/target when no cost-selected output is "
                "safe; try another source seed"
            ),
            "progress_is_safety_label": False,
            "debug_training_targets": 0,
            "debug_target_share": 0.0,
            "synthetic_reflections": 0,
            "target_padding": 0,
            "legacy_training_targets_reused": False,
            "legacy_checkpoints_reused": False,
        },
        "legacy_seed_hints": hint_rows,
        "candidate_census": candidate_rows,
        "balance": None
        if args.smoke
        else {
            gamma_tag(gamma): {
                "R-first": sum(item["direction_class"] == "R-first" for item in selected_by_gamma[gamma]),
                "U-first": sum(item["direction_class"] == "U-first" for item in selected_by_gamma[gamma]),
            }
            for gamma in gammas
        },
        "training_windows": window_count,
        "dataset": str(dataset_path) if dataset_path is not None else None,
        "dataset_sha256": dataset_hash,
        "dependency_manifest": dependency_manifest,
    }
    _atomic_json(outdir / "manifest.json", manifest)
    _atomic_json(outdir / "logs/stage_summary.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "candidate_census": candidate_rows,
        "training_windows": window_count,
        "manifest": str(outdir / "manifest.json"),
    }, indent=2), flush=True)
    return manifest


def _render_merged_dataset_paths(env: Any, payload: dict[str, Any], output: Path) -> None:
    """Render real executed paths reconstructed from exact per-step states."""

    trajectory_ids = payload["window_trajectory_ids"].cpu().numpy()
    steps = payload["window_steps"].cpu().numpy()
    states = payload["verifier_state"].cpu().numpy()
    plans = payload["U"].cpu().numpy()
    rows = {int(row["trajectory_id"]): row for row in payload["trajectory_rows"]}
    gamma_values = sorted({float(row["gamma"]) for row in rows.values()})
    colors = dict(zip(gamma_values, plt.cm.plasma(np.linspace(0.08, 0.92, len(gamma_values)))))
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
    for axis, gamma in zip(axes.ravel(), gamma_values):
        _draw_scene(axis, env)
        selected_ids = [tid for tid, row in rows.items() if float(row["gamma"]) == gamma]
        for tid in selected_ids:
            indices = np.flatnonzero(trajectory_ids == tid)
            indices = indices[np.argsort(steps[indices])]
            positions = [states[index, :2] for index in indices]
            positions.append(
                execute_first_action(states[indices[-1]], plans[indices[-1]])[:2]
            )
            path = np.asarray(positions)
            linestyle = "-" if rows[tid]["direction_class"] == "R-first" else "--"
            axis.plot(
                path[:, 0], path[:, 1], color=colors[gamma], lw=0.9,
                alpha=0.62, ls=linestyle,
            )
        r_count = sum(rows[tid]["direction_class"] == "R-first" for tid in selected_ids)
        u_count = sum(rows[tid]["direction_class"] == "U-first" for tid in selected_ids)
        axis.set_title(rf"$\gamma={gamma:g}$: R={r_count}, U={u_count}")
    for axis in axes.ravel()[len(gamma_values):]:
        axis.axis("off")
    fig.suptitle(
        "Merged real SafeMPPI planned-window demonstrations\n"
        "solid: R-first, dashed: U-first; debug targets = 0",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def combine_planned_demo_shards(args: argparse.Namespace) -> dict[str, Any]:
    """Validate and merge independently generated per-gamma Stage-02 shards."""

    if not args.shard_manifests:
        raise ValueError("combine requires --shard-manifests")
    manifests: list[tuple[Path, dict[str, Any], Path, dict[str, Any]]] = []
    for raw_path in args.shard_manifests:
        manifest_path = Path(raw_path).resolve()
        metadata = json.loads(manifest_path.read_text())
        if (
            metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("status") != "PLANNED_DEMOS_COMPLETE"
        ):
            raise RuntimeError(f"incompatible or incomplete Stage-02 shard: {manifest_path}")
        dataset_path = Path(metadata["dataset"])
        if not dataset_path.is_absolute():
            dataset_path = (manifest_path.parent / dataset_path).resolve()
        if sha256_file(dataset_path) != metadata.get("dataset_sha256"):
            raise RuntimeError(f"Stage-02 shard checksum mismatch: {dataset_path}")
        payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Stage-02 shard payload schema mismatch: {dataset_path}")
        manifests.append((manifest_path, metadata, dataset_path, payload))

    reference_config = dict(manifests[0][1].get("config", {}))
    quota = int(reference_config.get("quota_per_direction", 0))
    if quota <= 0:
        raise RuntimeError("Stage-02 shard config has no positive direction quota")
    for manifest_path, metadata, _dataset_path, _payload in manifests[1:]:
        if dict(metadata.get("config", {})) != reference_config:
            raise RuntimeError(
                f"Stage-02 shards disagree on rollout config: {manifest_path}"
            )

    seen_gammas: set[float] = set()
    tensor_keys = (
        "grid", "low5", "hist", "verifier_state", "U", "gamma",
        "window_seeds", "window_trajectory_ids", "source_trajectory_ids",
        "trajectory_balanced_weight", "window_steps", "window_direction",
        "query_progress_m", "query_physical_clearance_m", "target_safe",
        "target_in_bounds", "target_socp_ok",
    )
    list_keys = (
        "verifier_spec_fingerprint", "window_plan_kind", "query_hashes",
        "target_query_hash", "generated_hash", "verifier_input_hash",
        "training_target_hash", "generated_hashes", "verifier_input_hashes",
        "training_target_hashes",
    )
    tensor_parts: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys}
    list_parts: dict[str, list[Any]] = {key: [] for key in list_keys}
    trajectory_rows: list[dict[str, Any]] = []
    trajectory_offset = 0
    reference_contract: dict[str, Any] | None = None
    reference_start: torch.Tensor | None = None
    reference_goal: torch.Tensor | None = None
    shard_rows: list[dict[str, Any]] = []

    for manifest_path, metadata, dataset_path, payload in sorted(
        manifests, key=lambda item: float(item[1]["scene"]["gammas"][0])
    ):
        shard_gammas = sorted(
            {_canonical_supported_gamma(float(value)) for value in payload["gamma"].tolist()}
        )
        if len(shard_gammas) != 1 or shard_gammas[0] in seen_gammas:
            raise RuntimeError(f"each shard must contain one unique gamma: {dataset_path}")
        gamma = shard_gammas[0]
        seen_gammas.add(gamma)
        contract = dict(payload["contract"])
        if reference_contract is None:
            reference_contract = contract
            reference_start = payload["start"].clone()
            reference_goal = payload["goal"].clone()
        elif (
            contract != reference_contract
            or not torch.equal(payload["start"], reference_start)
            or not torch.equal(payload["goal"], reference_goal)
        ):
            raise RuntimeError("Stage-02 shards disagree on contract or endpoints")
        if contract.get("debug_training_targets") != 0 or contract.get("debug_target_share") != 0.0:
            raise RuntimeError("a Stage-02 shard contains forbidden debug targets")

        local_ids = payload["window_trajectory_ids"].long()
        unique_ids = sorted(int(value) for value in torch.unique(local_ids))
        if unique_ids != list(range(len(unique_ids))):
            raise RuntimeError("shard trajectory IDs must be contiguous from zero")
        remapped_ids = local_ids + trajectory_offset
        for key in tensor_keys:
            value = payload[key]
            if key in {"window_trajectory_ids", "source_trajectory_ids"}:
                value = remapped_ids.clone()
            tensor_parts[key].append(value)
        for key in list_keys:
            list_parts[key].extend(list(payload[key]))
        local_rows = list(payload["trajectory_rows"])
        r_count = u_count = 0
        for raw_row in local_rows:
            row = dict(raw_row)
            row["trajectory_id"] = int(row["trajectory_id"]) + trajectory_offset
            trajectory_rows.append(row)
            r_count += row["direction_class"] == "R-first"
            u_count += row["direction_class"] == "U-first"
        if r_count != quota or u_count != quota:
            raise RuntimeError(
                f"gamma={gamma:g} shard is not exact {quota}/{quota} "
                f"R/U balanced: {r_count}/{u_count}"
            )
        trajectory_offset += len(unique_ids)
        shard_rows.append(
            {
                "gamma": gamma,
                "manifest": str(manifest_path),
                "dataset": str(dataset_path),
                "dataset_sha256": metadata["dataset_sha256"],
                "windows": len(payload["U"]),
                "R-first": r_count,
                "U-first": u_count,
            }
        )

    if seen_gammas != set(float(value) for value in GAMMAS):
        raise RuntimeError(f"merged Stage-02 gammas are incomplete: {sorted(seen_gammas)}")
    merged: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        **{key: torch.cat(parts, dim=0) for key, parts in tensor_parts.items()},
        **list_parts,
        "trajectory_rows": trajectory_rows,
        "start": reference_start,
        "goal": reference_goal,
        "contract": reference_contract,
    }
    hashes = list(merged["query_hashes"])
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("merged Stage-02 dataset contains duplicate exact query hashes")
    for alias in (
        "target_query_hash", "generated_hash", "verifier_input_hash",
        "training_target_hash", "generated_hashes", "verifier_input_hashes",
        "training_target_hashes",
    ):
        if list(merged[alias]) != hashes:
            raise RuntimeError(f"merged identity alias differs: {alias}")
    if not bool(
        merged["target_safe"].all()
        and merged["target_in_bounds"].all()
        and merged["target_socp_ok"].all()
    ):
        raise RuntimeError("merged dataset contains a failed full-verifier target")

    outdir = args.outdir.resolve()
    dataset_path = outdir / "data/planned_id_balanced.pt"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    torch.save(merged, temporary)
    temporary.replace(dataset_path)
    dataset_sha = sha256_file(dataset_path)
    env = make_id_scene()
    _render_merged_dataset_paths(env, merged, outdir / "viz/selected_real_paths.png")
    render_target_audit(merged, outdir / "viz/training_target_audit.png")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PLANNED_DEMOS_COMPLETE",
        "created_at_utc": _utc_now(),
        "scene": {
            "name": "ordinary_symmetric_4x4_ID_stadium",
            "start": START.tolist(),
            "goal": GOAL.tolist(),
            "gammas": list(GAMMAS),
        },
        "config": reference_config,
        "legacy_mechanisms": clean_method_absence_manifest(),
        "contract": {
            "generated_equals_verified_equals_training": True,
            "debug_training_targets": 0,
            "debug_target_share": 0.0,
            "parallelization_only": "independent per-gamma processes with identical config",
        },
        "balance": {
            gamma_tag(gamma): {"R-first": quota, "U-first": quota}
            for gamma in GAMMAS
        },
        "trajectories": len(trajectory_rows),
        "training_windows": len(merged["U"]),
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "shards": shard_rows,
        "generator_sha256": sha256_file(Path(__file__)),
    }
    _atomic_json(outdir / "manifest.json", summary)
    _atomic_json(outdir / "logs/stage_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "sweep", "combine"))
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gammas", nargs="+", type=float, default=list(GAMMAS))
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--reach", type=float, default=0.20)
    parser.add_argument("--smooth-weight", type=float, default=8.0)
    parser.add_argument("--retreat-weight", type=float, default=1.0)
    parser.add_argument(
        "--noise-var-mult",
        type=float,
        default=3.0,
        help="SafeMPPI sampling-variance multiplier (legacy-compatible default: 3)",
    )
    parser.add_argument("--max-debug-candidates", type=int, default=6)
    parser.add_argument("--max-proposals", type=int, default=8)
    parser.add_argument("--quota", type=int, default=12)
    parser.add_argument("--max-candidate-seeds", type=int, default=256)
    parser.add_argument("--seed0", type=int, default=72_000)
    parser.add_argument("--smoke-seeds", type=int, default=1)
    parser.add_argument(
        "--sweep-smooth-weights", nargs="+", type=float, default=(32.0, 64.0, 128.0)
    )
    parser.add_argument(
        "--sweep-noise-var-mults", nargs="+", type=float, default=(1.0, 2.0, 3.0)
    )
    parser.add_argument("--sweep-seeds-per-gamma", type=int, default=2)
    parser.add_argument("--sweep-outdir", type=Path)
    parser.add_argument("--shard-manifests", nargs="+", type=Path)
    parser.add_argument(
        "--sweep-overwrite", action="store_true", help="rerun complete matching sweep cells"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    args.smoke = args.command == "smoke"
    if args.smoke_seeds <= 0:
        raise ValueError("smoke-seeds must be positive")
    if args.command == "sweep":
        run_smoothness_sweep(args)
    elif args.command == "combine":
        combine_planned_demo_shards(args)
    else:
        run_stage(args)


if __name__ == "__main__":
    main()
