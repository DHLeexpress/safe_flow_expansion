#!/usr/bin/env python3
"""Paired full-space SafeMPPI demonstrations for the additive low7 model.

This stage is deliberately separate from :mod:`stage2_planned_demos`.  It
changes endpoint sampling and conditioning only.  Every training target still
comes from the exact selected, fully verified H=10 SafeMPPI plan saved by
``run_expert_rollout``; only that plan's first action is executed.

The data commands are intended to be run in order::

    endpoints  -> freeze one IID free-space start/goal bank
    collect    -> collect one gamma shard (run seven processes independently)
    combine    -> authenticate and merge all seven shards
    render     -> draw all real attempts over the full workspace
    video      -> deterministic verifier replay for three diagnostic gammas

Retries never replace an endpoint pair.  The planner seed for a given
``(pair_id, retry)`` is also independent of gamma, so the seven gamma shards
remain paired experiments even when their success/missingness differs.
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
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import torch

import grid_feats as grid_features
import grid_scene as planner_scene

from .config import VerifierConfig, clean_method_absence_manifest
from .deps import (
    assert_no_legacy_expansion_imports,
    sha256_file,
    write_dependency_manifest,
)
from .scene import (
    GAMMAS,
    context_from_state_low7,
    make_id_scene,
    verifier_implementation_fingerprint,
    verifier_spec_fingerprint,
)
from .schemas import QueryContext, query_content_hash
from .verifier import VP, verify_plan
from . import stage2_planned_demos as expert_stage2


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PACKAGE_ROOT / "stage_results/02_low7_randomized"
DEFAULT_ENDPOINT_MANIFEST = DEFAULT_ROOT / "endpoints.json"
DATA_SCHEMA = "afe_planned_demo_v3_low7_uniform_pairs"
ENDPOINT_SCHEMA = "afe_low7_iid_free_endpoint_manifest_v1"
FIXED_GRID_ENDPOINT_SCHEMA = "afe_low7_fixed_goal_full_grid_endpoint_manifest_v1"
SHARD_STATUS = "LOW7_RANDOMIZED_GAMMA_SHARD_COMPLETE"
COMBINED_STATUS = "LOW7_RANDOMIZED_ALL_GAMMA_COMPLETE"
WORKSPACE_LOW = 0.0
WORKSPACE_HIGH = 5.0
FREE_CLEARANCE_M = 1.0e-4
GRID_LOW = 0.1
GRID_HIGH = 4.9
GRID_SIZE = 32
GRID_JITTER_M = 0.02
GRID_FREE_CLEARANCE_M = 0.05
FIXED_GOAL = np.asarray((4.7, 4.7), dtype=np.float64)
DEFAULT_PAIR_COUNT = 100
DEFAULT_ENDPOINT_SEED = 20_260_717
DEFAULT_PLANNER_SEED0 = 810_000
ALLOWED_PLAN_KINDS = ("weighted_mean", "internal_best")
CANONICAL_EXPERT_RECIPE = {
    "max_steps": 800,
    "reach_m": 0.15,
    "smooth_weight": 0.12,
    "retreat_weight": 0.0,
    "noise_var_mult": 3.0,
    "max_debug_candidates": 0,
    "max_proposals_per_step": 2,
}


@dataclass(frozen=True)
class EndpointBank:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    starts: np.ndarray
    goals: np.ndarray

    @property
    def count(self) -> int:
        return len(self.starts)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n"
    )
    temporary.replace(path)


def _scene_geometry_sha256(env: Any) -> str:
    obstacles = np.ascontiguousarray(
        env.obstacles.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    digest = hashlib.sha256(b"afe-low7-id-scene-geometry-v1\x00")
    digest.update(obstacles.dtype.str.encode("ascii") + b"\x00")
    digest.update(repr(obstacles.shape).encode("ascii") + b"\x00")
    digest.update(obstacles.tobytes(order="C"))
    digest.update(
        json.dumps(
            {
                "robot_radius": float(env.r_robot),
                "environment_dt": float(env.dt),
                "workspace": [WORKSPACE_LOW, WORKSPACE_HIGH],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _clearance(point: np.ndarray, env: Any) -> float:
    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64, copy=False)
    return float(
        (
            np.linalg.norm(obstacles[:, :2] - point[None], axis=1)
            - obstacles[:, 2]
            - float(env.r_robot)
        ).min()
    )


def _sample_free_point(
    rng: np.random.Generator, env: Any
) -> tuple[np.ndarray, int]:
    """Rejection-sample exactly uniform workspace area conditioned on free space."""

    for attempts in range(1, 1_000_001):
        # The float32 value is the value stored and used by the environment;
        # clearance is deliberately recomputed after this conversion.
        point = rng.uniform(WORKSPACE_LOW, WORKSPACE_HIGH, size=2).astype(np.float32)
        if _clearance(point.astype(np.float64), env) > FREE_CLEARANCE_M:
            return point, attempts
    raise RuntimeError("failed to sample a free endpoint in one million proposals")


def generate_endpoint_payload(*, pair_count: int, seed: int) -> dict[str, Any]:
    """Return a deterministic IID pair bank; no pairwise geometric filter exists."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    env = make_id_scene()
    rng = np.random.default_rng(int(seed))
    pairs: list[dict[str, Any]] = []
    start_proposals = 0
    goal_proposals = 0
    for pair_id in range(pair_count):
        start, start_attempts = _sample_free_point(rng, env)
        goal, goal_attempts = _sample_free_point(rng, env)
        start_proposals += start_attempts
        goal_proposals += goal_attempts
        pairs.append(
            {
                "pair_id": pair_id,
                "start": start.tolist(),
                "goal": goal.tolist(),
                "start_sampling_proposals": start_attempts,
                "goal_sampling_proposals": goal_attempts,
            }
        )
    return {
        "schema_version": ENDPOINT_SCHEMA,
        "status": "IMMUTABLE_ENDPOINT_BANK_COMPLETE",
        "created_at_utc": _utc_now(),
        "seed": int(seed),
        "pair_count": pair_count,
        "scene": {
            "name": "ordinary_symmetric_4x4_ID_stadium",
            "geometry_sha256": _scene_geometry_sha256(env),
            "obstacle_count": int(len(env.obstacles)),
            "robot_radius_m": float(env.r_robot),
            "workspace_m": [WORKSPACE_LOW, WORKSPACE_HIGH],
        },
        "sampling": {
            "start_distribution": "iid_uniform_workspace_conditioned_only_on_free_space",
            "goal_distribution": "iid_uniform_workspace_conditioned_only_on_free_space",
            "start_goal_independent": True,
            "float_storage": "float32_before_clearance_test",
            "strict_free_clearance_m": FREE_CLEARANCE_M,
            "endpoint_relation_constraints": [],
            "diagonal_constraint": False,
            "minimum_start_goal_distance_m": None,
            "success_conditioned_resampling": False,
            "start_sampling_proposals": start_proposals,
            "goal_sampling_proposals": goal_proposals,
        },
        "pairs": pairs,
    }


def generate_fixed_goal_grid_payload(*, seed: int) -> dict[str, Any]:
    """Return the old fixed-jitter grid without its off-diagonal exclusion.

    This changes only the start support relative to the 566-start corpus: the
    5 cm obstacle-clearance rule, zero initial velocity, and fixed goal remain
    explicit.  Grid points are identities, so failed expert rollouts are not
    success-conditioned replacements.
    """

    env = make_id_scene(goal=FIXED_GOAL)
    cell = (GRID_HIGH - GRID_LOW) / GRID_SIZE
    centers = GRID_LOW + cell * (np.arange(GRID_SIZE) + 0.5)
    x_grid, y_grid = np.meshgrid(centers, centers)
    points = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
    points += np.random.default_rng(int(seed)).uniform(
        -GRID_JITTER_M, GRID_JITTER_M, points.shape
    )
    clearance = np.asarray(
        [_clearance(point.astype(np.float64), env) for point in points]
    )
    starts = points[clearance > GRID_FREE_CLEARANCE_M].astype(np.float32)
    start_clearance = clearance[clearance > GRID_FREE_CLEARANCE_M]
    pairs = [
        {
            "pair_id": pair_id,
            "start": start.tolist(),
            "goal": FIXED_GOAL.tolist(),
            "start_clearance_m": float(start_clearance[pair_id]),
            "initial_velocity": [0.0, 0.0],
        }
        for pair_id, start in enumerate(starts)
    ]
    diagonal = np.abs(starts[:, 1] - starts[:, 0]) < 1.0
    wall_clearance = np.minimum.reduce(
        (starts[:, 0], starts[:, 1], 5.0 - starts[:, 0], 5.0 - starts[:, 1])
    )
    return {
        "schema_version": FIXED_GRID_ENDPOINT_SCHEMA,
        "status": "IMMUTABLE_FIXED_GOAL_FULL_GRID_BANK_COMPLETE",
        "created_at_utc": _utc_now(),
        "seed": int(seed),
        "pair_count": len(pairs),
        "scene": {
            "name": "ordinary_symmetric_4x4_ID_stadium",
            "geometry_sha256": _scene_geometry_sha256(env),
            "obstacle_count": int(len(env.obstacles)),
            "robot_radius_m": float(env.r_robot),
            "workspace_m": [WORKSPACE_LOW, WORKSPACE_HIGH],
        },
        "sampling": {
            "mode": "fixed_jitter_full_grid_starts_fixed_goal",
            "start_distribution": "32x32_uniform_grid_centers_with_fixed_uniform_jitter",
            "goal_distribution": "fixed",
            "fixed_goal": FIXED_GOAL.tolist(),
            "initial_velocity": [0.0, 0.0],
            "grid_low_m": GRID_LOW,
            "grid_high_m": GRID_HIGH,
            "grid_size": GRID_SIZE,
            "jitter_m": GRID_JITTER_M,
            "strict_free_clearance_m": GRID_FREE_CLEARANCE_M,
            "endpoint_relation_constraints": ["fixed_goal_only"],
            "diagonal_constraint": False,
            "minimum_start_goal_distance_m": None,
            "success_conditioned_resampling": False,
            "raw_grid_points": int(GRID_SIZE**2),
            "retained_starts": len(pairs),
            "legacy_off_diagonal_count": int((~diagonal).sum()),
            "new_diagonal_region_count": int(diagonal.sum()),
            "minimum_obstacle_clearance_m": float(start_clearance.min()),
            "minimum_workspace_wall_distance_m": float(wall_clearance.min()),
        },
        "pairs": pairs,
    }


def load_endpoint_bank(path: str | Path) -> EndpointBank:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    schema = payload.get("schema_version")
    if schema not in {ENDPOINT_SCHEMA, FIXED_GRID_ENDPOINT_SCHEMA}:
        raise RuntimeError(f"unsupported endpoint manifest schema: {path}")
    count = int(payload.get("pair_count", -1))
    rows = list(payload.get("pairs", ()))
    if count <= 0 or len(rows) != count:
        raise RuntimeError("endpoint manifest pair count is inconsistent")
    if [int(row.get("pair_id", -1)) for row in rows] != list(range(count)):
        raise RuntimeError("endpoint pair IDs must be contiguous from zero")
    sampling = dict(payload.get("sampling", {}))
    if schema == ENDPOINT_SCHEMA:
        if (
            sampling.get("endpoint_relation_constraints") != []
            or sampling.get("diagonal_constraint") is not False
            or sampling.get("minimum_start_goal_distance_m") is not None
            or sampling.get("start_goal_independent") is not True
            or float(sampling.get("strict_free_clearance_m", math.nan))
            != FREE_CLEARANCE_M
        ):
            raise RuntimeError("endpoint manifest does not declare the IID free-space contract")
        env = make_id_scene()
        required_clearance = FREE_CLEARANCE_M
    else:
        if (
            sampling.get("mode") != "fixed_jitter_full_grid_starts_fixed_goal"
            or sampling.get("diagonal_constraint") is not False
            or sampling.get("success_conditioned_resampling") is not False
            or int(sampling.get("grid_size", -1)) != GRID_SIZE
            or float(sampling.get("strict_free_clearance_m", math.nan))
            != GRID_FREE_CLEARANCE_M
            or not np.allclose(
                np.asarray(sampling.get("fixed_goal"), dtype=np.float32),
                FIXED_GOAL,
                rtol=0.0,
                atol=5.0e-7,
            )
        ):
            raise RuntimeError("endpoint manifest does not declare the fixed-goal full-grid contract")
        env = make_id_scene(goal=FIXED_GOAL)
        required_clearance = GRID_FREE_CLEARANCE_M
    if payload.get("scene", {}).get("geometry_sha256") != _scene_geometry_sha256(env):
        raise RuntimeError("endpoint manifest belongs to different scene geometry")
    starts = np.asarray([row["start"] for row in rows], dtype=np.float32)
    goals = np.asarray([row["goal"] for row in rows], dtype=np.float32)
    if starts.shape != (count, 2) or goals.shape != (count, 2):
        raise RuntimeError("endpoint arrays must have shape [pair_count,2]")
    for label, array in (("start", starts), ("goal", goals)):
        if not np.isfinite(array).all():
            raise RuntimeError(f"{label} endpoints contain nonfinite coordinates")
        if bool(np.any(array < WORKSPACE_LOW) or np.any(array > WORKSPACE_HIGH)):
            raise RuntimeError(f"{label} endpoint lies outside the full workspace")
        clearance = np.asarray(
            [_clearance(point.astype(np.float64), env) for point in array]
        )
        if label == "start" and bool(np.any(clearance <= required_clearance)):
            raise RuntimeError(f"{label} endpoint violates strict free-space clearance")
    if schema == FIXED_GRID_ENDPOINT_SCHEMA:
        if not bool(np.allclose(goals, FIXED_GOAL[None], rtol=0.0, atol=5.0e-7)):
            raise RuntimeError("fixed-grid endpoint bank contains a nonfixed goal")
        if count != int(sampling.get("retained_starts", -1)):
            raise RuntimeError("fixed-grid retained-start count is inconsistent")
    return EndpointBank(
        path=path,
        sha256=sha256_file(path),
        payload=payload,
        starts=starts,
        goals=goals,
    )


def generate_endpoints(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable endpoint bank: {output}")
    payload = (
        generate_fixed_goal_grid_payload(seed=args.seed)
        if args.fixed_goal_grid
        else generate_endpoint_payload(pair_count=args.pairs, seed=args.seed)
    )
    _atomic_json(output, payload)
    result = {**payload, "manifest": str(output), "manifest_sha256": sha256_file(output)}
    print(
        json.dumps(
            {
                "status": result["status"],
                "pairs": result["pair_count"],
                "manifest": result["manifest"],
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def _canonical_gamma(value: float) -> float:
    matches = [float(gamma) for gamma in GAMMAS if abs(float(value) - gamma) <= 5.0e-7]
    if len(matches) != 1:
        raise ValueError(f"gamma must be exactly one of {tuple(GAMMAS)}, got {value!r}")
    return matches[0]


def _endpoint_scientific_change(bank: EndpointBank) -> str:
    if bank.payload.get("schema_version") == FIXED_GRID_ENDPOINT_SCHEMA:
        return "fixed_goal_full_grid_starts_zero_velocity_no_diagonal_exclusion"
    return "iid_uniform_free_start_goal_pairs"


def _planner_seed(seed0: int, pair_id: int, retry: int, retries: int) -> int:
    # Gamma is intentionally absent: this is the matched-randomness contract.
    return int(seed0) + int(pair_id) * int(retries) + int(retry)


def _candidate_meta_path(directory: Path, gamma: float, seed: int) -> Path:
    return directory / f"g{expert_stage2.gamma_tag(gamma)}_seed{int(seed)}.json"


def _expert_generator_sha256() -> str:
    return sha256_file(Path(expert_stage2.__file__).resolve())


def _candidate_reusable(
    episode: Mapping[str, Any],
    *,
    config: expert_stage2.DemoRunConfig,
    pair_id: int,
    retry: int,
    start: np.ndarray,
    goal: np.ndarray,
    endpoint_sha256: str,
    randomized_generator_sha256: str,
    expert_generator_sha256: str,
) -> bool:
    try:
        stored_start = np.asarray(episode.get("start"), dtype=np.float32)
        stored_goal = np.asarray(episode.get("goal"), dtype=np.float32)
        contexts = list(episode.get("contexts", ()))
        return bool(
            episode.get("schema_version") == expert_stage2.SCHEMA_VERSION
            and episode.get("randomized_dataset_schema") == DATA_SCHEMA
            and episode.get("rollout_config") == expert_stage2._rollout_config(config)
            and int(episode.get("pair_id", -1)) == pair_id
            and int(episode.get("retry_index", -1)) == retry
            and stored_start.shape == (2,)
            and stored_goal.shape == (2,)
            and np.array_equal(stored_start, start)
            and np.array_equal(stored_goal, goal)
            and episode.get("endpoint_manifest_sha256") == endpoint_sha256
            and episode.get("randomized_generator_sha256")
            == randomized_generator_sha256
            and episode.get("generator_sha256") == expert_generator_sha256
            and episode.get("context_feature_schema") == "low7_closest_boundary"
            and all(np.asarray(context.low5).shape == (7,) for context in contexts)
            and np.array_equal(
                np.asarray(episode["states"], dtype=np.float32)[0, :2], start
            )
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _relative_to(path: Path, directory: Path) -> str:
    try:
        return str(path.resolve().relative_to(directory.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_from(raw: str | Path, directory: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def _attempt_row(
    episode: Mapping[str, Any],
    *,
    meta_path: Path,
    outdir: Path,
    pair_id: int,
    retry: int,
    start: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "retry_index": retry,
        "seed": int(episode["seed"]),
        "start": start.tolist(),
        "goal": goal.tolist(),
        "success": bool(episode["success"]),
        "training_eligible": bool(episode["success"] and int(episode["steps"]) > 0),
        "status": str(episode["status"]),
        "dead_reason": episode.get("dead_reason"),
        "steps": int(episode["steps"]),
        "queries": int(episode["queries"]),
        "min_clearance_m": float(episode["min_clearance_m"]),
        "endpoint_distance_m": float(episode["endpoint_distance_m"]),
        "candidate_meta": _relative_to(meta_path, outdir),
        "candidate_meta_sha256": sha256_file(meta_path),
    }


def _build_dataset(
    episodes: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    endpoint_manifest: EndpointBank,
    gamma: float,
) -> dict[str, Any]:
    """Serialize successful exact selected queries under the v3 payload names."""

    grids: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    verifier_states: list[np.ndarray] = []
    verifier_fingerprints: list[str] = []
    plans: list[np.ndarray] = []
    plan_kinds: list[str] = []
    query_hashes: list[str] = []
    window_gammas: list[float] = []
    window_pair_ids: list[int] = []
    window_starts: list[np.ndarray] = []
    window_goals: list[np.ndarray] = []
    window_seeds: list[int] = []
    window_retries: list[int] = []
    window_trajectory_ids: list[int] = []
    window_steps: list[int] = []
    query_progress: list[float] = []
    query_clearance: list[float] = []
    target_safe: list[bool] = []
    target_in_bounds: list[bool] = []
    target_socp_ok: list[bool] = []
    trajectory_rows: list[dict[str, Any]] = []

    ordered = sorted(episodes, key=lambda episode: int(episode["pair_id"]))
    seen_pairs: set[int] = set()
    for trajectory_id, episode in enumerate(ordered):
        pair_id = int(episode["pair_id"])
        if pair_id in seen_pairs:
            raise RuntimeError(f"pair {pair_id} has more than one selected trajectory")
        seen_pairs.add(pair_id)
        if not bool(episode["success"]) or int(episode["steps"]) <= 0:
            raise RuntimeError("only nonempty successful trajectories may enter training")
        if _canonical_gamma(float(episode["gamma"])) != gamma:
            raise RuntimeError("episode gamma differs from shard gamma")
        start = endpoint_manifest.starts[pair_id]
        goal = endpoint_manifest.goals[pair_id]
        if not np.array_equal(np.asarray(episode["states"])[0, :2], start):
            raise RuntimeError("episode initial state differs from its immutable endpoint pair")
        if not np.array_equal(np.asarray(episode["goal"], dtype=np.float32), goal):
            raise RuntimeError("episode goal differs from its immutable endpoint pair")
        selected = np.asarray(episode["selected_query_indices"], dtype=np.int64)
        query_steps = np.asarray(episode["query_steps"], dtype=np.int64)
        episode_plans = np.asarray(episode["query_plans"], dtype=np.float32)
        episode_hashes = list(episode["query_hashes"])
        episode_kinds = list(episode["query_kinds"])
        contexts: list[QueryContext] = list(episode["contexts"])
        if len(selected) != int(episode["steps"]):
            raise RuntimeError("successful trajectory lacks one selected target per step")
        for local_step, query_index in enumerate(selected):
            context_step = int(query_steps[query_index])
            if context_step != local_step:
                raise RuntimeError("selected query/context step identity changed")
            context = contexts[context_step]
            low7 = np.asarray(context.low5, dtype=np.float32)
            if low7.shape != (7,) or not np.isclose(
                float(low7[-1]), gamma, atol=5.0e-7, rtol=0.0
            ):
                raise RuntimeError("v3 context must contain low7 with gamma last")
            plan = episode_plans[query_index]
            plan_kind = str(episode_kinds[query_index])
            if plan_kind not in ALLOWED_PLAN_KINDS:
                raise RuntimeError(f"forbidden training plan kind: {plan_kind!r}")
            identity = query_content_hash(context, gamma, plan)
            if identity != episode_hashes[query_index]:
                raise RuntimeError("training target differs from the verified query object")
            if not (
                bool(episode["query_safe"][query_index])
                and bool(episode["query_in_bounds"][query_index])
                and bool(episode["query_socp_ok"][query_index])
            ):
                raise RuntimeError("selected target is not fully verifier-positive")
            grids.append(np.asarray(context.grid, dtype=np.float32))
            lows.append(low7)
            histories.append(np.asarray(context.hist, dtype=np.float32))
            verifier_states.append(np.asarray(context.verifier_state, dtype=np.float64))
            verifier_fingerprints.append(context.verifier_spec_fingerprint)
            plans.append(plan.copy())
            plan_kinds.append(plan_kind)
            query_hashes.append(identity)
            window_gammas.append(gamma)
            window_pair_ids.append(pair_id)
            window_starts.append(start.copy())
            window_goals.append(goal.copy())
            window_seeds.append(int(episode["seed"]))
            window_retries.append(int(episode["retry_index"]))
            window_trajectory_ids.append(trajectory_id)
            window_steps.append(local_step)
            query_progress.append(float(episode["query_progress_m"][query_index]))
            query_clearance.append(
                float(episode["query_physical_clearance_m"][query_index])
            )
            target_safe.append(True)
            target_in_bounds.append(True)
            target_socp_ok.append(True)
        trajectory_rows.append(
            {
                "trajectory_id": trajectory_id,
                "pair_id": pair_id,
                "gamma": gamma,
                "seed": int(episode["seed"]),
                "retry_index": int(episode["retry_index"]),
                "start": start.tolist(),
                "goal": goal.tolist(),
                "steps": int(episode["steps"]),
                "min_clearance_m": float(episode["min_clearance_m"]),
                "path_length_m": float(episode["path_length_m"]),
                "query_acceptance": float(episode["query_acceptance"]),
                "candidate_meta": str(episode["candidate_meta"]),
            }
        )

    if not plans:
        raise RuntimeError("gamma shard has no nonempty successful expert trajectory")
    if len(set(query_hashes)) != len(query_hashes):
        raise RuntimeError("gamma shard contains duplicate exact query identities")
    trajectory_ids = np.asarray(window_trajectory_ids, dtype=np.int64)
    counts = np.bincount(trajectory_ids, minlength=len(trajectory_rows))
    if bool(np.any(counts <= 0)):
        raise RuntimeError("every trajectory must contribute at least one target")
    trajectory_weights = 1.0 / counts[trajectory_ids]
    identity_aliases = {
        "target_query_hash": list(query_hashes),
        "generated_hash": list(query_hashes),
        "verifier_input_hash": list(query_hashes),
        "training_target_hash": list(query_hashes),
    }
    payload: dict[str, Any] = {
        "schema_version": DATA_SCHEMA,
        "grid": torch.from_numpy(np.asarray(grids, dtype=np.float32)),
        # QueryContext retains the legacy wire attribute ``low5``; the public
        # v3 dataset key is explicit about the actual seven-value payload.
        "low7": torch.from_numpy(np.asarray(lows, dtype=np.float32)),
        "hist": torch.from_numpy(np.asarray(histories, dtype=np.float32)),
        "verifier_state": torch.from_numpy(np.asarray(verifier_states, dtype=np.float64)),
        "verifier_spec_fingerprint": verifier_fingerprints,
        "U": torch.from_numpy(np.asarray(plans, dtype=np.float32)),
        "window_plan_kind": plan_kinds,
        "gamma": torch.tensor(window_gammas, dtype=torch.float32),
        "window_pair_ids": torch.tensor(window_pair_ids, dtype=torch.long),
        "window_start": torch.from_numpy(np.asarray(window_starts, dtype=np.float32)),
        "window_goal": torch.from_numpy(np.asarray(window_goals, dtype=np.float32)),
        "window_seeds": torch.tensor(window_seeds, dtype=torch.long),
        "window_retries": torch.tensor(window_retries, dtype=torch.int16),
        "window_trajectory_ids": torch.from_numpy(trajectory_ids),
        "trajectory_balanced_weight": torch.from_numpy(
            trajectory_weights.astype(np.float32)
        ),
        "window_steps": torch.tensor(window_steps, dtype=torch.int32),
        "query_progress_m": torch.tensor(query_progress, dtype=torch.float32),
        "query_physical_clearance_m": torch.tensor(
            query_clearance, dtype=torch.float32
        ),
        "target_safe": torch.tensor(target_safe, dtype=torch.bool),
        "target_in_bounds": torch.tensor(target_in_bounds, dtype=torch.bool),
        "target_socp_ok": torch.tensor(target_socp_ok, dtype=torch.bool),
        "query_hashes": query_hashes,
        **identity_aliases,
        "trajectory_rows": trajectory_rows,
        "endpoint_manifest": str(endpoint_manifest.path),
        "endpoint_manifest_sha256": endpoint_manifest.sha256,
        "contract": {
            "generated_equals_verified_equals_training": True,
            "planned_horizon": 10,
            "only_first_action_executed": True,
            "allowed_training_plan_kinds": list(ALLOWED_PLAN_KINDS),
            "debug_training_targets": 0,
            "padding": 0,
            "synthetic_reflections": 0,
            "direction_labels": 0,
            "query_context_wire_field": "low5",
            "dataset_conditioning_key": "low7",
            "low7_shape": 7,
            "gamma_last": True,
            "inverse_trajectory_length_weighting": True,
            "trajectory_balanced_total_mass_per_path": 1.0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return payload


def _require_tensor(
    payload: Mapping[str, Any], key: str, count: int, tail: tuple[int, ...]
) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (count, *tail):
        raise RuntimeError(f"{key} must have shape {(count, *tail)}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{key} contains nonfinite values")
    return value


def _validate_dataset_payload(
    payload: Mapping[str, Any],
    *,
    endpoint_manifest: EndpointBank,
    expected_gamma: float | None,
) -> None:
    if payload.get("schema_version") != DATA_SCHEMA:
        raise RuntimeError("dataset payload is not low7 randomized schema v3")
    if payload.get("endpoint_manifest_sha256") != endpoint_manifest.sha256:
        raise RuntimeError("dataset payload endpoint-bank checksum differs")
    if "low5" in payload or "window_direction" in payload:
        raise RuntimeError("v3 dataset must not expose low5 or fake direction labels")
    plans = payload.get("U")
    if not isinstance(plans, torch.Tensor) or plans.ndim != 3:
        raise RuntimeError("U must be a tensor with shape [N,10,2]")
    count = len(plans)
    if count <= 0:
        raise RuntimeError("dataset must contain at least one exact target")
    _require_tensor(payload, "U", count, (10, 2))
    grid = payload.get("grid")
    if not isinstance(grid, torch.Tensor) or grid.ndim != 4 or len(grid) != count:
        raise RuntimeError("grid must have shape [N,C,H,W]")
    low7 = _require_tensor(payload, "low7", count, (7,))
    hist = payload.get("hist")
    if not isinstance(hist, torch.Tensor) or hist.ndim != 3 or len(hist) != count:
        raise RuntimeError("hist must have shape [N,K,2]")
    if hist.shape[-1] != 2:
        raise RuntimeError("hist final dimension must be two")
    states = _require_tensor(payload, "verifier_state", count, (4,))
    gamma = _require_tensor(payload, "gamma", count, ()).double()
    pair_ids = _require_tensor(payload, "window_pair_ids", count, ()).long()
    starts = _require_tensor(payload, "window_start", count, (2,)).float()
    goals = _require_tensor(payload, "window_goal", count, (2,)).float()
    trajectory_ids = _require_tensor(
        payload, "window_trajectory_ids", count, ()
    ).long()
    steps = _require_tensor(payload, "window_steps", count, ()).long()
    _require_tensor(payload, "window_seeds", count, ())
    _require_tensor(payload, "window_retries", count, ())
    weights = _require_tensor(
        payload, "trajectory_balanced_weight", count, ()
    ).double()
    if not bool((weights > 0.0).all()):
        raise RuntimeError("trajectory-balanced weights must be positive")
    for value in gamma.tolist():
        canonical = _canonical_gamma(value)
        if expected_gamma is not None and canonical != expected_gamma:
            raise RuntimeError("gamma shard payload contains another gamma")
    if not bool(torch.isclose(low7[:, -1].double(), gamma, atol=5.0e-7).all()):
        raise RuntimeError("serialized low7 gamma coordinate differs from gamma tensor")
    for index, pair_id in enumerate(pair_ids.tolist()):
        if not 0 <= pair_id < endpoint_manifest.count:
            raise RuntimeError("window pair id is outside the immutable endpoint bank")
        if not torch.equal(starts[index], torch.from_numpy(endpoint_manifest.starts[pair_id])):
            raise RuntimeError("window start differs from endpoint manifest")
        if not torch.equal(goals[index], torch.from_numpy(endpoint_manifest.goals[pair_id])):
            raise RuntimeError("window goal differs from endpoint manifest")
    unique_trajectories = sorted(int(value) for value in torch.unique(trajectory_ids))
    if unique_trajectories != list(range(len(unique_trajectories))):
        raise RuntimeError("trajectory ids must be contiguous from zero")
    rows = list(payload.get("trajectory_rows", ()))
    if len(rows) != len(unique_trajectories):
        raise RuntimeError("trajectory rows do not match window trajectory ids")
    for trajectory_id, row in enumerate(rows):
        if int(row.get("trajectory_id", -1)) != trajectory_id:
            raise RuntimeError("trajectory row ordering differs from ids")
        indices = torch.where(trajectory_ids == trajectory_id)[0]
        expected_steps = torch.arange(len(indices), dtype=torch.long)
        actual_steps = torch.sort(steps[indices]).values.cpu()
        if not torch.equal(actual_steps, expected_steps):
            raise RuntimeError("trajectory window steps are not contiguous from zero")
        expected_weight = torch.full(
            (len(indices),), 1.0 / len(indices), dtype=torch.float64
        )
        if not torch.allclose(weights[indices].cpu(), expected_weight, atol=1.0e-7, rtol=0.0):
            raise RuntimeError("trajectory-balanced weights are not inverse path length")
        pair_values = torch.unique(pair_ids[indices]).tolist()
        if pair_values != [int(row["pair_id"])]:
            raise RuntimeError("trajectory row pair id differs from its windows")
        pair_id = int(row["pair_id"])
        if not np.array_equal(
            np.asarray(row["start"], dtype=np.float32),
            endpoint_manifest.starts[pair_id],
        ) or not np.array_equal(
            np.asarray(row["goal"], dtype=np.float32),
            endpoint_manifest.goals[pair_id],
        ):
            raise RuntimeError("trajectory row endpoints differ from endpoint manifest")
    for label in ("target_safe", "target_in_bounds", "target_socp_ok"):
        value = _require_tensor(payload, label, count, ()).bool()
        if not bool(value.all()):
            raise RuntimeError(f"every training target must satisfy {label}")
    fingerprints = list(payload.get("verifier_spec_fingerprint", ()))
    kinds = list(payload.get("window_plan_kind", ()))
    hashes = [str(value) for value in payload.get("query_hashes", ())]
    if not (len(fingerprints) == len(kinds) == len(hashes) == count):
        raise RuntimeError("query identity lists have inconsistent lengths")
    if any(kind not in ALLOWED_PLAN_KINDS for kind in kinds):
        raise RuntimeError("dataset contains a raw/debug training target")
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("dataset contains duplicate exact query identities")
    for alias in (
        "target_query_hash",
        "generated_hash",
        "verifier_input_hash",
        "training_target_hash",
    ):
        if list(payload.get(alias, ())) != hashes:
            raise RuntimeError(f"query identity alias differs: {alias}")
    for index in range(count):
        context = QueryContext(
            grid[index].numpy(),
            low7[index].numpy(),
            hist[index].numpy(),
            states[index].double().numpy(),
            fingerprints[index],
        )
        actual = query_content_hash(
            context, _canonical_gamma(float(gamma[index])), plans[index].numpy()
        )
        if actual != hashes[index]:
            raise RuntimeError(f"exact query identity mismatch at dataset row {index}")


def collect_gamma_shard(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    gamma = _canonical_gamma(args.gamma)
    endpoint_manifest = load_endpoint_bank(args.endpoint_manifest)
    if args.retries <= 0:
        raise ValueError("retries must be positive")
    if args.max_proposals < args.max_debug_candidates + 2:
        raise ValueError(
            "all returned expert proposals must be verified: max-proposals must be "
            "at least max-debug-candidates + 2"
        )
    config = expert_stage2.DemoRunConfig(
        max_steps=args.max_steps,
        reach_m=args.reach,
        smooth_weight=args.smooth_weight,
        retreat_weight=args.retreat_weight,
        noise_var_mult=args.noise_var_mult,
        max_debug_candidates=args.max_debug_candidates,
        max_proposals_per_step=args.max_proposals,
        quota_per_direction=1,
        max_candidate_seeds_per_gamma=args.retries,
        seed0=args.planner_seed0,
    )
    actual_recipe = {
        key: asdict(config)[key] for key in CANONICAL_EXPERT_RECIPE
    }
    if actual_recipe != CANONICAL_EXPERT_RECIPE:
        raise ValueError(
            "this additive experiment pins the canonical SafeMPPI teacher recipe; "
            f"actual={actual_recipe}, required={CANONICAL_EXPERT_RECIPE}"
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    outdir = args.outdir.resolve()
    candidate_dir = outdir / "candidates"
    for directory in (candidate_dir, outdir / "data", outdir / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    dependencies = write_dependency_manifest(outdir / "logs/dependencies.json")
    assert_no_legacy_expansion_imports()
    randomized_generator_sha256 = sha256_file(Path(__file__))
    expert_generator_sha256 = _expert_generator_sha256()
    attempts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for pair_id, (start, goal) in enumerate(
        zip(endpoint_manifest.starts, endpoint_manifest.goals)
    ):
        for retry in range(args.retries):
            seed = _planner_seed(args.planner_seed0, pair_id, retry, args.retries)
            meta_path = _candidate_meta_path(candidate_dir, gamma, seed)
            episode: dict[str, Any] | None = None
            if meta_path.exists():
                try:
                    cached = expert_stage2.load_episode(meta_path)
                except Exception as error:  # corrupted/stale caches are regenerated
                    print(
                        f"[low7 gamma={gamma:g}] stale pair={pair_id} retry={retry}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                else:
                    if _candidate_reusable(
                        cached,
                        config=config,
                        pair_id=pair_id,
                        retry=retry,
                        start=start,
                        goal=goal,
                        endpoint_sha256=endpoint_manifest.sha256,
                        randomized_generator_sha256=randomized_generator_sha256,
                        expert_generator_sha256=expert_generator_sha256,
                    ):
                        episode = cached
            if episode is None:
                env = make_id_scene(start=start, goal=goal)
                env.T = config.max_steps
                episode = expert_stage2.run_expert_rollout(
                    env=env,
                    gamma=gamma,
                    seed=seed,
                    device=device,
                    config=config,
                    context_fn=context_from_state_low7,
                )
                # The legacy R/U diagnostic is tied to fixed canonical
                # endpoints and is meaningless for arbitrary pairs.
                episode["direction_class"] = "not_applicable_random_endpoints"
                episode.update(
                    {
                        "randomized_dataset_schema": DATA_SCHEMA,
                        "context_feature_schema": "low7_closest_boundary",
                        "pair_id": pair_id,
                        "retry_index": retry,
                        "start": start.tolist(),
                        "goal": goal.tolist(),
                        "endpoint_manifest": str(endpoint_manifest.path),
                        "endpoint_manifest_sha256": endpoint_manifest.sha256,
                        "randomized_generator_sha256": randomized_generator_sha256,
                    }
                )
                expert_stage2.save_episode(episode, candidate_dir)
                # Reload the bytes that later combine/render stages will see.
                episode = expert_stage2.load_episode(meta_path)
                if not _candidate_reusable(
                    episode,
                    config=config,
                    pair_id=pair_id,
                    retry=retry,
                    start=start,
                    goal=goal,
                    endpoint_sha256=endpoint_manifest.sha256,
                    randomized_generator_sha256=randomized_generator_sha256,
                    expert_generator_sha256=expert_generator_sha256,
                ):
                    raise RuntimeError("new candidate failed its own cache binding")
            episode["candidate_meta"] = _relative_to(meta_path, outdir)
            attempts.append(
                _attempt_row(
                    episode,
                    meta_path=meta_path,
                    outdir=outdir,
                    pair_id=pair_id,
                    retry=retry,
                    start=start,
                    goal=goal,
                )
            )
            print(
                f"[low7 gamma={gamma:g}] pair={pair_id + 1}/{endpoint_manifest.count} "
                f"retry={retry + 1}/{args.retries} status={episode['status']} "
                f"steps={episode['steps']}",
                flush=True,
            )
            if bool(episode["success"]):
                if int(episode["steps"]) > 0:
                    selected.append(episode)
                break

    dataset_path = outdir / "data" / f"low7_randomized_gamma_{gamma:g}.pt"
    payload = _build_dataset(
        selected,
        dataset_path,
        endpoint_manifest=endpoint_manifest,
        gamma=gamma,
    )
    _validate_dataset_payload(
        payload, endpoint_manifest=endpoint_manifest, expected_gamma=gamma
    )
    dataset_sha256 = sha256_file(dataset_path)
    attempted_pairs = sorted({int(row["pair_id"]) for row in attempts})
    if attempted_pairs != list(range(endpoint_manifest.count)):
        raise RuntimeError("collector did not attempt every immutable endpoint pair")
    manifest = {
        "schema_version": DATA_SCHEMA,
        "artifact_kind": "per_gamma_shard",
        "status": SHARD_STATUS,
        "created_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "gamma": gamma,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scene": {
            "name": "ordinary_symmetric_4x4_ID_stadium",
            "geometry_sha256": endpoint_manifest.payload["scene"]["geometry_sha256"],
        },
        "endpoint_manifest": str(endpoint_manifest.path),
        "endpoint_manifest_sha256": endpoint_manifest.sha256,
        "pair_count": endpoint_manifest.count,
        "config": {
            "rollout": asdict(config),
            "retries_per_pair": int(args.retries),
            "planner_seed0": int(args.planner_seed0),
            "planner_seed_paired_across_gamma": True,
        },
        "recipe_contract": {
            "canonical_expert_recipe": CANONICAL_EXPERT_RECIPE,
            "actual_matches_canonical": all(
                asdict(config)[key] == value
                for key, value in CANONICAL_EXPERT_RECIPE.items()
            ),
            "scientific_changes": [
                "low7_closest_boundary_conditioning",
                _endpoint_scientific_change(endpoint_manifest),
            ],
        },
        "generator_sha256": randomized_generator_sha256,
        "expert_generator_sha256": expert_generator_sha256,
        "legacy_mechanisms": clean_method_absence_manifest(),
        "attempts": attempts,
        "counts": {
            "attempted_pairs": len(attempted_pairs),
            "attempts": len(attempts),
            "rollout_successes": sum(bool(row["success"]) for row in attempts),
            "zero_step_successes": sum(
                bool(row["success"]) and not bool(row["training_eligible"])
                for row in attempts
            ),
            "training_trajectories": len(selected),
            "training_windows": len(payload["U"]),
            "failed_pairs": endpoint_manifest.count
            - sum(bool(row["success"]) for row in attempts),
        },
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "contract": payload["contract"],
        "dependencies": dependencies,
    }
    _atomic_json(outdir / "manifest.json", manifest)
    _atomic_json(outdir / "logs/stage_summary.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "gamma": gamma,
                "counts": manifest["counts"],
                "dataset": manifest["dataset"],
                "dataset_sha256": dataset_sha256,
                "manifest": str(outdir / "manifest.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def _validate_attempt_schedule(
    manifest: Mapping[str, Any], endpoint_manifest: EndpointBank
) -> None:
    attempts = list(manifest.get("attempts", ()))
    retries = int(manifest["config"]["retries_per_pair"])
    seed0 = int(manifest["config"]["planner_seed0"])
    by_pair: dict[int, list[Mapping[str, Any]]] = {
        pair_id: [] for pair_id in range(endpoint_manifest.count)
    }
    for row in attempts:
        pair_id = int(row["pair_id"])
        if pair_id not in by_pair:
            raise RuntimeError("attempt row has an unknown pair id")
        by_pair[pair_id].append(row)
    for pair_id, rows in by_pair.items():
        if not rows:
            raise RuntimeError(f"pair {pair_id} was never attempted")
        rows.sort(key=lambda row: int(row["retry_index"]))
        retry_indices = [int(row["retry_index"]) for row in rows]
        if retry_indices != list(range(len(rows))) or len(rows) > retries:
            raise RuntimeError(f"pair {pair_id} has a nonsequential retry schedule")
        if any(bool(row["success"]) for row in rows[:-1]):
            raise RuntimeError(f"pair {pair_id} was retried after success")
        if not bool(rows[-1]["success"]) and len(rows) != retries:
            raise RuntimeError(f"failed pair {pair_id} did not exhaust fixed retries")
        for row in rows:
            retry = int(row["retry_index"])
            if int(row["seed"]) != _planner_seed(seed0, pair_id, retry, retries):
                raise RuntimeError("attempt planner seed violates paired schedule")
            if not np.array_equal(
                np.asarray(row["start"], dtype=np.float32),
                endpoint_manifest.starts[pair_id],
            ) or not np.array_equal(
                np.asarray(row["goal"], dtype=np.float32),
                endpoint_manifest.goals[pair_id],
            ):
                raise RuntimeError("attempt row endpoints differ from immutable pair bank")


def combine_shards(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.shard_manifests) != len(GAMMAS):
        raise ValueError(f"combine requires exactly {len(GAMMAS)} gamma shard manifests")
    records: list[tuple[Path, dict[str, Any], Path, dict[str, Any]]] = []
    endpoint_manifest: EndpointBank | None = None
    reference_config: Mapping[str, Any] | None = None
    reference_generator: str | None = None
    reference_expert_generator: str | None = None
    seen_gammas: set[float] = set()
    for raw_path in args.shard_manifests:
        manifest_path = Path(raw_path).resolve()
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version") != DATA_SCHEMA
            or manifest.get("artifact_kind") != "per_gamma_shard"
            or manifest.get("status") != SHARD_STATUS
        ):
            raise RuntimeError(f"incomplete or incompatible shard: {manifest_path}")
        gamma = _canonical_gamma(float(manifest["gamma"]))
        if gamma in seen_gammas:
            raise RuntimeError(f"duplicate gamma shard: {gamma:g}")
        seen_gammas.add(gamma)
        bank = load_endpoint_bank(manifest["endpoint_manifest"])
        if bank.sha256 != manifest.get("endpoint_manifest_sha256"):
            raise RuntimeError("shard endpoint manifest checksum differs")
        if endpoint_manifest is None:
            endpoint_manifest = bank
            reference_config = dict(manifest["config"])
            reference_generator = str(manifest["generator_sha256"])
            reference_expert_generator = str(manifest["expert_generator_sha256"])
        elif bank.sha256 != endpoint_manifest.sha256:
            raise RuntimeError("gamma shards do not share the exact endpoint bank")
        if dict(manifest["config"]) != reference_config:
            raise RuntimeError("gamma shards disagree on rollout/retry configuration")
        if (
            manifest.get("generator_sha256") != reference_generator
            or manifest.get("expert_generator_sha256") != reference_expert_generator
        ):
            raise RuntimeError("gamma shards were produced by different generators")
        if int(manifest.get("pair_count", -1)) != bank.count:
            raise RuntimeError("shard pair count differs from endpoint bank")
        if manifest.get("recipe_contract", {}).get("actual_matches_canonical") is not True:
            raise RuntimeError("gamma shard did not use the canonical SafeMPPI teacher recipe")
        actual_recipe = {
            key: manifest["config"]["rollout"].get(key)
            for key in CANONICAL_EXPERT_RECIPE
        }
        if actual_recipe != CANONICAL_EXPERT_RECIPE:
            raise RuntimeError(
                f"gamma shard canonical recipe values differ: {actual_recipe}"
            )
        _validate_attempt_schedule(manifest, bank)
        dataset_path = _resolve_from(manifest["dataset"], manifest_path.parent)
        if sha256_file(dataset_path) != manifest.get("dataset_sha256"):
            raise RuntimeError(f"gamma shard dataset checksum mismatch: {dataset_path}")
        payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        _validate_dataset_payload(payload, endpoint_manifest=bank, expected_gamma=gamma)
        records.append((manifest_path, manifest, dataset_path, payload))
    assert endpoint_manifest is not None
    if seen_gammas != {float(gamma) for gamma in GAMMAS}:
        raise RuntimeError(f"gamma shards are incomplete: {sorted(seen_gammas)}")
    if reference_generator != sha256_file(Path(__file__)):
        raise RuntimeError("current randomized generator differs from authenticated shards")
    if reference_expert_generator != _expert_generator_sha256():
        raise RuntimeError("current expert generator differs from authenticated shards")

    tensor_keys = (
        "grid",
        "low7",
        "hist",
        "verifier_state",
        "U",
        "gamma",
        "window_pair_ids",
        "window_start",
        "window_goal",
        "window_seeds",
        "window_retries",
        "window_trajectory_ids",
        "trajectory_balanced_weight",
        "window_steps",
        "query_progress_m",
        "query_physical_clearance_m",
        "target_safe",
        "target_in_bounds",
        "target_socp_ok",
    )
    list_keys = (
        "verifier_spec_fingerprint",
        "window_plan_kind",
        "query_hashes",
        "target_query_hash",
        "generated_hash",
        "verifier_input_hash",
        "training_target_hash",
    )
    tensor_parts: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys}
    list_parts: dict[str, list[Any]] = {key: [] for key in list_keys}
    trajectory_rows: list[dict[str, Any]] = []
    trajectory_offset = 0
    shard_rows: list[dict[str, Any]] = []
    reference_contract: Mapping[str, Any] | None = None
    for manifest_path, manifest, dataset_path, payload in sorted(
        records, key=lambda record: float(record[1]["gamma"])
    ):
        local_ids = payload["window_trajectory_ids"].long()
        local_count = len(payload["trajectory_rows"])
        remapped = local_ids + trajectory_offset
        for key in tensor_keys:
            value = payload[key]
            tensor_parts[key].append(remapped if key == "window_trajectory_ids" else value)
        for key in list_keys:
            list_parts[key].extend(list(payload[key]))
        for raw_row in payload["trajectory_rows"]:
            row = dict(raw_row)
            row["trajectory_id"] = int(row["trajectory_id"]) + trajectory_offset
            row["candidate_meta"] = str(
                _resolve_from(row["candidate_meta"], manifest_path.parent)
            )
            row["source_shard_manifest"] = str(manifest_path)
            trajectory_rows.append(row)
        contract = dict(payload["contract"])
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise RuntimeError("gamma shards disagree on dataset contract")
        trajectory_offset += local_count
        shard_rows.append(
            {
                "gamma": float(manifest["gamma"]),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "dataset": str(dataset_path),
                "dataset_sha256": manifest["dataset_sha256"],
                "attempts": int(manifest["counts"]["attempts"]),
                "training_trajectories": local_count,
                "training_windows": len(payload["U"]),
            }
        )
    merged: dict[str, Any] = {
        "schema_version": DATA_SCHEMA,
        **{key: torch.cat(parts, dim=0) for key, parts in tensor_parts.items()},
        **list_parts,
        "trajectory_rows": trajectory_rows,
        "endpoint_manifest": str(endpoint_manifest.path),
        "endpoint_manifest_sha256": endpoint_manifest.sha256,
        "contract": reference_contract,
    }
    _validate_dataset_payload(
        merged, endpoint_manifest=endpoint_manifest, expected_gamma=None
    )
    outdir = args.outdir.resolve()
    manifest_path = outdir / "manifest.json"
    dataset_path = outdir / "data/low7_randomized_all_gamma.pt"
    if manifest_path.exists() or dataset_path.exists():
        raise FileExistsError(f"refusing to overwrite combined artifact in {outdir}")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    torch.save(merged, temporary)
    temporary.replace(dataset_path)
    dataset_sha256 = sha256_file(dataset_path)
    counts_by_gamma = {
        f"{row['gamma']:g}": {
            "attempts": row["attempts"],
            "training_trajectories": row["training_trajectories"],
            "training_windows": row["training_windows"],
        }
        for row in shard_rows
    }
    manifest = {
        "schema_version": DATA_SCHEMA,
        "artifact_kind": "all_gamma_combined",
        "status": COMBINED_STATUS,
        "created_at_utc": _utc_now(),
        "scene": {
            "name": "ordinary_symmetric_4x4_ID_stadium",
            "geometry_sha256": endpoint_manifest.payload["scene"]["geometry_sha256"],
            "gammas": list(GAMMAS),
        },
        "endpoint_manifest": str(endpoint_manifest.path),
        "endpoint_manifest_sha256": endpoint_manifest.sha256,
        "pair_count": endpoint_manifest.count,
        "config": reference_config,
        "recipe_contract": {
            "canonical_expert_recipe": CANONICAL_EXPERT_RECIPE,
            "all_shards_match_canonical": True,
            "scientific_changes": [
                "low7_closest_boundary_conditioning",
                _endpoint_scientific_change(endpoint_manifest),
            ],
        },
        "generator_sha256": reference_generator,
        "expert_generator_sha256": reference_expert_generator,
        "legacy_mechanisms": clean_method_absence_manifest(),
        "counts": {
            "training_trajectories": len(trajectory_rows),
            "training_windows": len(merged["U"]),
            "unique_training_pairs": len(torch.unique(merged["window_pair_ids"])),
            "per_gamma": counts_by_gamma,
        },
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "shards": shard_rows,
        "contract": reference_contract,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(outdir / "logs/stage_summary.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "dataset": str(dataset_path),
                "dataset_sha256": dataset_sha256,
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def _draw_scene(axis: Any, env: Any) -> None:
    for x, y, radius in env.obstacles.detach().cpu().numpy():
        axis.add_patch(
            Circle((float(x), float(y)), float(radius), color="0.76", zorder=1)
        )
    axis.set(
        xlim=(WORKSPACE_LOW, WORKSPACE_HIGH),
        ylim=(WORKSPACE_LOW, WORKSPACE_HIGH),
        aspect="equal",
    )
    axis.set_xticks((0, 1, 2, 3, 4, 5))
    axis.set_yticks((0, 1, 2, 3, 4, 5))
    axis.grid(color="0.9", lw=0.5, zorder=0)


def render_endpoint_starts(args: argparse.Namespace) -> dict[str, Any]:
    """Show the immutable start bank before any expert rollout is collected."""

    bank = load_endpoint_bank(args.endpoint_manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    goal = bank.goals[0]
    env = make_id_scene(goal=goal)
    clearance = np.asarray(
        [_clearance(point.astype(np.float64), env) for point in bank.starts]
    )
    figure, axis = plt.subplots(figsize=(8.2, 7.6))
    _draw_scene(axis, env)
    scatter = axis.scatter(
        bank.starts[:, 0],
        bank.starts[:, 1],
        c=clearance,
        cmap="viridis",
        vmin=0.05,
        vmax=float(np.quantile(clearance, 0.95)),
        s=14,
        linewidths=0.0,
        zorder=4,
    )
    axis.plot(goal[0], goal[1], "*", color="#ffd60a", mec="black", mew=0.6, ms=14, zorder=6)
    axis.plot((0.0, 5.0), (0.0, 5.0), ":", color="0.25", lw=0.8, zorder=2)
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.047, pad=0.03)
    colorbar.set_label("initial obstacle-boundary clearance [m]")
    sampling = bank.payload["sampling"]
    axis.set_title(
        f"Fixed-goal full-grid start bank: {bank.count} zero-velocity starts\n"
        f"no diagonal exclusion; clearance > {float(sampling['strict_free_clearance_m']):.2f} m"
    )
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    sidecar = {
        "schema_version": "afe_low7_endpoint_start_map_v1",
        "status": "COMPLETE",
        "endpoint_manifest": str(bank.path),
        "endpoint_manifest_sha256": bank.sha256,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "start_count": bank.count,
        "goal": goal.tolist(),
        "initial_velocity": [0.0, 0.0],
        "minimum_obstacle_clearance_m": float(clearance.min()),
        "median_obstacle_clearance_m": float(np.median(clearance)),
        "diagonal_constraint": False,
    }
    _atomic_json(output.with_suffix(".json"), sidecar)
    print(json.dumps(sidecar, indent=2), flush=True)
    return sidecar


def _candidate_path(meta_path: Path, expected_meta_sha256: str) -> np.ndarray:
    if sha256_file(meta_path) != expected_meta_sha256:
        raise RuntimeError(f"candidate metadata checksum mismatch: {meta_path}")
    metadata = json.loads(meta_path.read_text())
    array_path = _resolve_from(metadata["array_file"], meta_path.parent)
    if sha256_file(array_path) != metadata.get("array_sha256"):
        raise RuntimeError(f"candidate array checksum mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        states = arrays["states"].astype(np.float32, copy=True)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise RuntimeError(f"candidate states have invalid shape/content: {array_path}")
    return states[:, :2]


def _add_paths(
    axis: Any,
    paths: Sequence[np.ndarray],
    *,
    color: Any,
    linewidth: float,
    alpha: float,
    linestyle: str = "solid",
    zorder: int = 3,
) -> None:
    segments = [path for path in paths if len(path) >= 2]
    if segments:
        axis.add_collection(
            LineCollection(
                segments,
                colors=[color],
                linewidths=linewidth,
                alpha=alpha,
                linestyles=linestyle,
                zorder=zorder,
            )
        )


def render_overlay(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != DATA_SCHEMA
        or manifest.get("artifact_kind") != "all_gamma_combined"
        or manifest.get("status") != COMBINED_STATUS
    ):
        raise RuntimeError("render requires a complete combined low7 manifest")
    dataset_path = _resolve_from(manifest["dataset"], manifest_path.parent)
    if sha256_file(dataset_path) != manifest.get("dataset_sha256"):
        raise RuntimeError("combined dataset checksum mismatch before rendering")
    endpoint_manifest = load_endpoint_bank(manifest["endpoint_manifest"])
    if endpoint_manifest.sha256 != manifest.get("endpoint_manifest_sha256"):
        raise RuntimeError("combined endpoint bank checksum mismatch")
    paths_by_gamma: dict[float, dict[str, list[np.ndarray]]] = {
        float(gamma): {"success": [], "failure": []} for gamma in GAMMAS
    }
    rows_by_gamma: dict[float, list[Mapping[str, Any]]] = {
        float(gamma): [] for gamma in GAMMAS
    }
    for shard in manifest["shards"]:
        shard_path = Path(shard["manifest"]).resolve()
        if sha256_file(shard_path) != shard["manifest_sha256"]:
            raise RuntimeError(f"shard manifest checksum mismatch: {shard_path}")
        shard_manifest = json.loads(shard_path.read_text())
        gamma = _canonical_gamma(float(shard_manifest["gamma"]))
        _validate_attempt_schedule(shard_manifest, endpoint_manifest)
        for row in shard_manifest["attempts"]:
            meta_path = _resolve_from(row["candidate_meta"], shard_path.parent)
            path = _candidate_path(meta_path, str(row["candidate_meta_sha256"]))
            label = "success" if bool(row["success"]) else "failure"
            paths_by_gamma[gamma][label].append(path)
            rows_by_gamma[gamma].append(row)

    output = (
        args.output.resolve()
        if args.output is not None
        else manifest_path.parent / "viz/full_space_all_gamma_trajectory_overlay.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    env = make_id_scene()
    colors = {
        float(gamma): plt.cm.plasma(value)
        for gamma, value in zip(GAMMAS, np.linspace(0.08, 0.92, len(GAMMAS)))
    }
    figure = plt.figure(figsize=(17.2, 10.2))
    layout = figure.add_gridspec(
        4, 4, width_ratios=(1.35, 1.35, 1.0, 1.0), hspace=0.30, wspace=0.20
    )
    main_axis = figure.add_subplot(layout[:, :2])
    mini_axes = [
        figure.add_subplot(layout[row, column])
        for row in range(4)
        for column in (2, 3)
    ]
    _draw_scene(main_axis, env)
    all_failures = [
        path
        for gamma in GAMMAS
        for path in paths_by_gamma[float(gamma)]["failure"]
    ]
    _add_paths(
        main_axis,
        all_failures,
        color="0.35",
        linewidth=0.45,
        alpha=0.055,
        linestyle="dashed",
        zorder=2,
    )
    for gamma in GAMMAS:
        _add_paths(
            main_axis,
            paths_by_gamma[float(gamma)]["success"],
            color=colors[float(gamma)],
            linewidth=0.75,
            alpha=0.25,
            zorder=3,
        )
    fixed_goal_grid = (
        endpoint_manifest.payload.get("schema_version") == FIXED_GRID_ENDPOINT_SCHEMA
    )
    main_axis.scatter(
        endpoint_manifest.starts[:, 0],
        endpoint_manifest.starts[:, 1],
        s=8,
        marker="o",
        facecolors="none",
        edgecolors="#168aad",
        linewidths=0.35,
        alpha=0.38,
        zorder=5,
        label=("full-grid starts" if fixed_goal_grid else "IID starts"),
    )
    displayed_goals = endpoint_manifest.goals[:1] if fixed_goal_grid else endpoint_manifest.goals
    main_axis.scatter(
        displayed_goals[:, 0],
        displayed_goals[:, 1],
        s=11,
        marker="*",
        color="#d62828",
        linewidths=0.25,
        alpha=0.36,
        zorder=5,
        label=("fixed goal" if fixed_goal_grid else "IID goals"),
    )
    total_attempts = sum(len(rows) for rows in rows_by_gamma.values())
    total_success = sum(
        len(paths_by_gamma[float(gamma)]["success"]) for gamma in GAMMAS
    )
    main_axis.set_title(
        "All real SafeMPPI attempts over the full 5×5 workspace\n"
        f"{endpoint_manifest.count} {'fixed-goal grid starts' if fixed_goal_grid else 'paired IID endpoints'} × 7 γ; "
        f"success={total_success}/{total_attempts}",
        fontsize=13,
    )
    main_axis.set_xlabel("world x [m]")
    main_axis.set_ylabel("world y [m]")
    legend_handles = [
        Line2D([0], [0], color=colors[float(gamma)], lw=2, label=rf"$\gamma={gamma:g}$")
        for gamma in GAMMAS
    ]
    legend_handles.append(
        Line2D([0], [0], color="0.35", lw=1, ls="--", label="failed retry")
    )
    main_axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=4,
        frameon=False,
        fontsize=8,
    )

    for axis, gamma in zip(mini_axes, GAMMAS):
        gamma = float(gamma)
        _draw_scene(axis, env)
        _add_paths(
            axis,
            paths_by_gamma[gamma]["failure"],
            color="#b23a48",
            linewidth=0.45,
            alpha=0.09,
            linestyle="dashed",
            zorder=2,
        )
        _add_paths(
            axis,
            paths_by_gamma[gamma]["success"],
            color=colors[gamma],
            linewidth=0.65,
            alpha=0.30,
            zorder=3,
        )
        rows = rows_by_gamma[gamma]
        successes = sum(bool(row["success"]) for row in rows)
        training = sum(bool(row["training_eligible"]) for row in rows)
        axis.set_title(
            rf"$\gamma={gamma:g}$  success {successes}/{len(rows)}; train {training}",
            fontsize=8.5,
        )
        axis.tick_params(labelsize=6)
    mini_axes[-1].axis("off")
    figure.suptitle(
        (
            "Low7 fixed-goal full-grid expert demonstrations — zero initial velocity"
            if fixed_goal_grid
            else "Low7 randomized expert demonstrations — no diagonal restriction, reflection, or padding"
        ),
        fontsize=15,
        y=0.995,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    sidecar = {
        "schema_version": "afe_low7_randomized_overlay_v1",
        "status": "COMPLETE",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "endpoint_manifest_sha256": endpoint_manifest.sha256,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "pair_count": endpoint_manifest.count,
        "attempts": total_attempts,
        "successes": total_success,
        "includes_failed_retries": True,
        "includes_actual_executed_paths": True,
        "diagonal_band": False,
        "fixed_goal_grid": fixed_goal_grid,
    }
    _atomic_json(output.with_suffix(".json"), sidecar)
    print(json.dumps(sidecar, indent=2), flush=True)
    return sidecar


VIDEO_GAMMAS = (0.1, 0.5, 1.0)
NOMINAL_BLUE = "#0072B2"
VERIFIER_GREEN = "#009E73"
NOMINAL_SCHEDULE_TOLERANCE = 1.0e-8


class NominalScheduleError(RuntimeError):
    def __init__(
        self,
        *,
        gamma: float,
        episode_step: int,
        horizon_step: int,
        residual: float,
    ) -> None:
        self.gamma = float(gamma)
        self.episode_step = int(episode_step)
        self.horizon_step = int(horizon_step)
        self.residual = float(residual)
        super().__init__(
            "exact planner nominal schedule failed: "
            f"gamma={gamma:g}, episode_step={episode_step}, "
            f"horizon_step={horizon_step}, residual={residual:.12g}"
        )


def _load_video_episodes(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    EndpointBank,
    int,
    dict[float, dict[str, Any]],
    dict[float, dict[int, dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Choose the lowest common-success pair passing the declared video gate."""

    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != DATA_SCHEMA
        or manifest.get("artifact_kind") != "all_gamma_combined"
        or manifest.get("status") != COMBINED_STATUS
    ):
        raise RuntimeError("video requires a complete combined low7 manifest")
    dataset_path = _resolve_from(manifest["dataset"], manifest_path.parent)
    if sha256_file(dataset_path) != manifest.get("dataset_sha256"):
        raise RuntimeError("combined dataset checksum mismatch before video replay")
    if manifest.get("generator_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("current randomized generator differs from combined artifact")
    if manifest.get("expert_generator_sha256") != _expert_generator_sha256():
        raise RuntimeError("current expert generator differs from combined artifact")
    bank = load_endpoint_bank(manifest["endpoint_manifest"])
    if bank.sha256 != manifest.get("endpoint_manifest_sha256"):
        raise RuntimeError("combined endpoint bank checksum mismatch")
    candidate_rows: dict[float, dict[int, tuple[Path, Mapping[str, Any]]]] = {}
    for shard in manifest["shards"]:
        gamma = _canonical_gamma(float(shard["gamma"]))
        if gamma not in VIDEO_GAMMAS:
            continue
        shard_path = Path(shard["manifest"]).resolve()
        if sha256_file(shard_path) != shard["manifest_sha256"]:
            raise RuntimeError(f"shard manifest checksum mismatch: {shard_path}")
        shard_manifest = json.loads(shard_path.read_text())
        _validate_attempt_schedule(shard_manifest, bank)
        successful: dict[int, tuple[Path, Mapping[str, Any]]] = {}
        for row in shard_manifest["attempts"]:
            if not bool(row["training_eligible"]):
                continue
            pair_id = int(row["pair_id"])
            if pair_id in successful:
                raise RuntimeError("a gamma shard has two successful episodes for one pair")
            successful[pair_id] = (shard_path, row)
        candidate_rows[gamma] = successful
    if set(candidate_rows) != set(VIDEO_GAMMAS):
        raise RuntimeError("combined manifest lacks one of the three diagnostic gamma shards")
    common = set.intersection(*(set(rows) for rows in candidate_rows.values()))
    if not common:
        raise RuntimeError(
            "no fixed endpoint pair has successful episodes at gamma 0.1, 0.5, and 1.0"
        )
    selection_census: list[dict[str, Any]] = []
    for pair_id in sorted(common):
        selected: dict[float, dict[str, Any]] = {}
        for gamma in VIDEO_GAMMAS:
            shard_path, row = candidate_rows[gamma][pair_id]
            meta_path = _resolve_from(row["candidate_meta"], shard_path.parent)
            if sha256_file(meta_path) != row["candidate_meta_sha256"]:
                raise RuntimeError(
                    f"selected candidate metadata checksum mismatch: {meta_path}"
                )
            episode = expert_stage2.load_episode(meta_path, validate=True)
            if (
                int(episode.get("pair_id", -1)) != pair_id
                or _canonical_gamma(float(episode["gamma"])) != gamma
                or not bool(episode["success"])
                or int(episode["steps"]) <= 0
            ):
                raise RuntimeError(
                    "diagnostic candidate success/provenance is inconsistent"
                )
            if not np.array_equal(
                np.asarray(episode["states"])[0, :2], bank.starts[pair_id]
            ) or not np.array_equal(
                np.asarray(episode["goal"], dtype=np.float32), bank.goals[pair_id]
            ):
                raise RuntimeError("diagnostic candidate endpoints changed")
            episode["candidate_meta"] = str(meta_path)
            episode["candidate_meta_sha256"] = str(row["candidate_meta_sha256"])
            selected[gamma] = episode

        env = make_id_scene(start=bank.starts[pair_id], goal=bank.goals[pair_id])
        replay_cache: dict[float, dict[int, dict[str, Any]]] = {
            gamma: {} for gamma in VIDEO_GAMMAS
        }
        census_row: dict[str, Any] = {
            "pair_id": pair_id,
            "successful_all_three_gamma": True,
            "episode_steps": {
                f"{gamma:g}": int(selected[gamma]["steps"])
                for gamma in VIDEO_GAMMAS
            },
            "checked_steps": {f"{gamma:g}": 0 for gamma in VIDEO_GAMMAS},
            "minimum_nominal_residual": None,
            "minimum_nominal_residual_location": None,
            "diagnostic_predicate_pass": False,
            "rejection": None,
        }
        rejected = False
        for gamma in VIDEO_GAMMAS:
            for step in range(int(selected[gamma]["steps"])):
                try:
                    replay = _replay_selected_polytope(selected[gamma], step, env)
                except NominalScheduleError as error:
                    census_row["rejection"] = {
                        "reason": "exact_planner_nominal_multistep_failure",
                        "gamma": error.gamma,
                        "episode_step": error.episode_step,
                        "horizon_step": error.horizon_step,
                        "residual": error.residual,
                        "tolerance": NOMINAL_SCHEDULE_TOLERANCE,
                    }
                    rejected = True
                    break
                replay_cache[gamma][step] = replay
                census_row["checked_steps"][f"{gamma:g}"] += 1
                residual = float(replay["nominal_worst_residual"])
                current_minimum = census_row["minimum_nominal_residual"]
                if current_minimum is None or residual < float(current_minimum):
                    census_row["minimum_nominal_residual"] = residual
                    census_row["minimum_nominal_residual_location"] = {
                        "gamma": gamma,
                        "episode_step": step,
                        "horizon_step": int(replay["nominal_worst_horizon_step"]),
                    }
            if rejected:
                break
        if rejected:
            selection_census.append(census_row)
            continue
        census_row["diagnostic_predicate_pass"] = True
        selection_census.append(census_row)
        return manifest, bank, pair_id, selected, replay_cache, selection_census
    raise RuntimeError(
        "no common-success pair passes exact nominal multi-step and external fitted "
        f"replay at every executed step; census={selection_census}"
    )


def _face_signature(face: Any) -> dict[str, Any]:
    interval = getattr(face, "interval", None)
    return {
        "a_float64_hex": [float(value).hex() for value in np.asarray(face.a, dtype=np.float64)],
        "m_float64_hex": float(face.m).hex(),
        "kind": str(face.kind),
        "label": str(face.label),
        "coefficient_float64_hex": float(getattr(face, "coefficient", 1.0)).hex(),
        "feasible": bool(face.feasible),
        "interval_float64_hex": None
        if interval is None
        else [float(value).hex() for value in interval],
    }


def _signature_sha256(signatures: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(signatures), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _require_exact_metric(actual: float, stored: float, *, label: str) -> None:
    actual64 = np.float64(actual)
    stored64 = np.float64(stored)
    if actual64.tobytes() != stored64.tobytes():
        raise RuntimeError(
            f"deterministic verifier replay metric differs for {label}: "
            f"actual={actual!r}, stored={stored!r}"
        )


def _replay_selected_polytope(
    episode: Mapping[str, Any], step: int, env: Any
) -> dict[str, Any]:
    """Replay and authenticate one stored selected query before visualization."""

    gamma = _canonical_gamma(float(episode["gamma"]))
    selected = np.asarray(episode["selected_query_indices"], dtype=np.int64)
    if not 0 <= step < len(selected):
        raise RuntimeError("video requested a step without a selected exact H10 plan")
    query_index = int(selected[step])
    query_steps = np.asarray(episode["query_steps"], dtype=np.int64)
    query_kinds = list(episode["query_kinds"])
    query_safe = np.asarray(episode["query_safe"], dtype=bool)
    same_step = np.flatnonzero(query_steps == step)
    certified_cost_selected = [
        int(index)
        for index in same_step
        if query_kinds[int(index)] in ALLOWED_PLAN_KINDS and query_safe[int(index)]
    ]
    if not certified_cost_selected or query_index not in certified_cost_selected:
        raise RuntimeError(
            "shown step lacks a certified cost-selected proposal or selected safe plan"
        )
    if int(query_steps[query_index]) != step:
        raise RuntimeError("selected query does not belong to the shown receding-horizon step")
    context: QueryContext = episode["contexts"][step]
    plan = np.asarray(episode["query_plans"][query_index], dtype=np.float32)
    if plan.shape != (10, 2) or not np.array_equal(
        plan[0], np.asarray(episode["executed_actions"])[step]
    ):
        raise RuntimeError("selected video plan is not the exact executed-first-action target")
    stored_hash = str(episode["query_hashes"][query_index])
    if query_content_hash(context, gamma, plan) != stored_hash:
        raise RuntimeError("selected video query hash differs from stored exact identity")
    goal = np.asarray(episode["goal"], dtype=np.float64)
    current_spec = verifier_spec_fingerprint(env, goal)
    if context.verifier_spec_fingerprint != current_spec:
        raise RuntimeError("current deterministic verifier spec differs from stored query")
    result = verify_plan(
        context.verifier_state, plan.copy(), env, gamma, goal=goal
    )
    finite_metrics = (
        result.bounds_margin_m,
        result.physical_clearance_m,
        result.face_margin_m,
        result.certificate_residual,
        result.progress_m,
        result.start_goal_distance_m,
        result.terminal_goal_distance_m,
    )
    if not (
        np.isfinite(result.states).all()
        and np.isfinite(result.positions).all()
        and np.isfinite(np.asarray(finite_metrics, dtype=np.float64)).all()
    ):
        raise RuntimeError("deterministic selected-plan replay returned nonfinite evidence")
    for field, stored_key in (
        ("safe", "query_safe"),
        ("in_bounds", "query_in_bounds"),
        ("socp_ok", "query_socp_ok"),
    ):
        if bool(getattr(result, field)) != bool(episode[stored_key][query_index]):
            raise RuntimeError(f"deterministic verifier replay differs for {field}")
    if not (result.safe and result.in_bounds and result.socp_ok):
        raise RuntimeError("shown selected H10 plan is not fully verifier-safe on replay")
    for field in (
        "bounds_margin_m",
        "physical_clearance_m",
        "face_margin_m",
        "certificate_residual",
        "progress_m",
        "start_goal_distance_m",
        "terminal_goal_distance_m",
    ):
        _require_exact_metric(
            float(getattr(result, field)),
            float(episode[f"query_{field}"][query_index]),
            label=field,
        )
    if int(result.certificate_worst_step) != int(
        episode["query_certificate_worst_step"][query_index]
    ):
        raise RuntimeError("deterministic certificate worst-step replay differs")

    verifier = VerifierConfig()
    obstacles = env.obstacles.detach().cpu().numpy()
    socp_ok, fitted_faces, _raw_obstacles, effective_radius = VP.certify_window(
        result.positions,
        obstacles,
        float(env.r_robot),
        gamma,
        R=float(verifier.sensing_radius),
        K=int(verifier.artificial_faces),
        rho_art=float(verifier.artificial_radius),
        m_min=float(verifier.minimum_face_margin),
        m_max=verifier.maximum_face_margin,
        n_theta=int(verifier.angle_samples),
        r_pad=float(verifier.rollout_padding_factor),
    )
    if not bool(socp_ok) or not all(
        bool(face.feasible) and float(face.m) > 0.0 for face in fitted_faces
    ):
        raise RuntimeError("replayed fitted face set is not an exact feasible SOCP polytope")
    planner_config = planner_scene.mode1_config(
        noise_var_mult=CANONICAL_EXPERT_RECIPE["noise_var_mult"]
    )
    nominal_radius = float(planner_config["barrier_activation_radius"])
    nominal_nbase = int(planner_config.get("polytope_nbase", 16))
    nominal_predict_gain = float(planner_config.get("predict_gain", 0.0))
    nominal_hp, (nominal_A, _nominal_b, nominal_margins) = (
        grid_features.polytope_HP(
            result.positions[0],
            planner_scene.planner_obstacles(env),
            sensing=nominal_radius,
            n_base=nominal_nbase,
            predict_gain=nominal_predict_gain,
        )
    )
    nominal_values = np.asarray(nominal_hp(result.positions), dtype=np.float64)
    nominal_schedule = (1.0 - gamma) ** np.arange(len(result.positions), dtype=np.float64)
    nominal_residuals = nominal_values - nominal_schedule
    if not np.isfinite(nominal_residuals).all() or abs(float(nominal_values[0]) - 1.0) > NOMINAL_SCHEDULE_TOLERANCE:
        raise RuntimeError("exact planner nominal H_P replay is nonfinite or H_0 differs from one")
    worst_horizon_step = int(np.argmin(nominal_residuals[1:]) + 1)
    worst_nominal_residual = float(nominal_residuals[worst_horizon_step])
    if worst_nominal_residual < -NOMINAL_SCHEDULE_TOLERANCE:
        raise NominalScheduleError(
            gamma=gamma,
            episode_step=step,
            horizon_step=worst_horizon_step,
            residual=worst_nominal_residual,
        )
    # These Face objects are an exact representation of the A/margin arrays
    # returned by the planner's own nominal polytope_HP path.  They are used
    # only so the common H_grid routine can evaluate the same H_P globally.
    nominal_faces = [
        VP.Face(
            np.asarray(normal, dtype=np.float64),
            float(margin),
            "planner-nominal",
            f"planner_face{index}",
        )
        for index, (normal, margin) in enumerate(zip(nominal_A, nominal_margins))
    ]
    fitted_signatures = [_face_signature(face) for face in fitted_faces]
    nominal_signatures = [_face_signature(face) for face in nominal_faces]
    return {
        "gamma": gamma,
        "step": step,
        "query_index": query_index,
        "query_hash": stored_hash,
        "certified_cost_selected_count": len(certified_cost_selected),
        "positions": np.asarray(result.positions, dtype=np.float64),
        "fitted_faces": fitted_faces,
        "nominal_faces": nominal_faces,
        "effective_radius_m": float(effective_radius),
        "nominal_sensing_radius_m": nominal_radius,
        "nominal_polytope_nbase": nominal_nbase,
        "nominal_predict_gain": nominal_predict_gain,
        "nominal_schedule_tolerance": NOMINAL_SCHEDULE_TOLERANCE,
        "nominal_worst_residual": worst_nominal_residual,
        "nominal_worst_horizon_step": worst_horizon_step,
        "verifier_spec_fingerprint": current_spec,
        "fitted_face_signatures": fitted_signatures,
        "nominal_face_signatures": nominal_signatures,
        "fitted_face_sha256": _signature_sha256(fitted_signatures),
        "nominal_face_sha256": _signature_sha256(nominal_signatures),
    }


def _contour_face_level(
    axis: Any,
    faces: Sequence[Any],
    center: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    zorder: float,
) -> None:
    coordinates = np.linspace(WORKSPACE_LOW, WORKSPACE_HIGH, 181)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    values = VP.H_grid(faces, grid_x - center[0], grid_y - center[1])
    if float(np.min(values)) <= 0.0 <= float(np.max(values)):
        axis.contour(
            grid_x,
            grid_y,
            values,
            levels=[0.0],
            colors=[color],
            linestyles=[linestyle],
            linewidths=[linewidth],
            zorder=zorder,
        )


def _render_polytope_panel(
    axis: Any,
    *,
    episode: Mapping[str, Any],
    replay: Mapping[str, Any],
    env: Any,
    pair_id: int,
) -> None:
    _draw_scene(axis, env)
    gamma = float(replay["gamma"])
    step = int(replay["step"])
    path = np.asarray(episode["states"], dtype=np.float64)[:, :2]
    center = path[step]
    accepted = np.asarray(replay["positions"], dtype=np.float64)
    axis.plot(path[:, 0], path[:, 1], color="0.72", lw=0.8, zorder=2)
    axis.plot(
        path[: step + 1, 0],
        path[: step + 1, 1],
        color=plt.cm.plasma(0.15 + 0.7 * VIDEO_GAMMAS.index(gamma) / 2.0),
        lw=2.0,
        zorder=6,
    )
    axis.plot(
        accepted[:, 0],
        accepted[:, 1],
        color="black",
        lw=1.2,
        marker=".",
        ms=3.0,
        alpha=0.85,
        zorder=9,
    )
    # Draw fitted first, then the dashed nominal boundary above it.  Otherwise
    # coincident faces make the green solid line hide the requested blue
    # nominal evidence.
    _contour_face_level(
        axis,
        replay["fitted_faces"],
        center,
        color=VERIFIER_GREEN,
        linestyle="-",
        linewidth=2.4,
        zorder=7,
    )
    _contour_face_level(
        axis,
        replay["nominal_faces"],
        center,
        color=NOMINAL_BLUE,
        linestyle="--",
        linewidth=2.0,
        zorder=8,
    )
    start = np.asarray(episode["states"])[0, :2]
    goal = np.asarray(episode["goal"])
    axis.scatter(*start, marker="o", s=24, facecolors="white", edgecolors="black", zorder=10)
    axis.scatter(*goal, marker="*", s=85, color="#ffd21f", edgecolors="black", zorder=10)
    axis.scatter(*center, marker="o", s=22, color="black", zorder=11)
    axis.set_title(
        rf"$\gamma={gamma:g}$ · pair {pair_id} · step {step}/{episode['steps']}"
        "\n"
        f"safe cost-selected={replay['certified_cost_selected_count']} · "
        f"nominal residual={replay['nominal_worst_residual']:.2e} · "
        f"query {str(replay['query_hash'])[:10]}…",
        fontsize=9,
    )
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")


def render_polytope_video(args: argparse.Namespace) -> dict[str, Any]:
    """Render a diagnostic-only MP4 from authenticated deterministic verifier replay."""

    if args.frame_stride <= 0 or args.fps <= 0 or args.seconds_per_frame <= 0.0:
        raise ValueError("frame stride, fps, and seconds per frame must be positive")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("both ffmpeg and ffprobe are required for the diagnostic MP4")
    manifest_path = args.manifest.resolve()
    (
        source_manifest,
        bank,
        pair_id,
        episodes,
        replay_cache,
        selection_census,
    ) = _load_video_episodes(manifest_path)
    minimum_steps = min(int(episode["steps"]) for episode in episodes.values())
    shown_steps = list(range(0, minimum_steps, args.frame_stride))
    if shown_steps[-1] != minimum_steps - 1:
        shown_steps.append(minimum_steps - 1)
    output = (
        args.output.resolve()
        if args.output is not None
        else manifest_path.parent / "viz/low7_exact_polytope_replay.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_directory = output.with_name(f"{output.stem}_frames")
    frame_directory.mkdir(parents=True, exist_ok=True)
    for old in frame_directory.glob("frame_*.png"):
        old.unlink()
    preview = output.with_name(f"{output.stem}_preview.png")
    signature_log = output.with_name(f"{output.stem}_face_signatures.jsonl")
    frame_records: list[dict[str, Any]] = []
    env = make_id_scene(start=bank.starts[pair_id], goal=bank.goals[pair_id])
    expected_spec = verifier_spec_fingerprint(env, bank.goals[pair_id])
    started = time.perf_counter()
    for frame_index, step in enumerate(shown_steps):
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
        for axis, gamma in zip(axes, VIDEO_GAMMAS):
            replay = replay_cache[gamma][step]
            if replay["verifier_spec_fingerprint"] != expected_spec:
                raise RuntimeError("diagnostic panels do not share the frozen verifier spec")
            _render_polytope_panel(
                axis,
                episode=episodes[gamma],
                replay=replay,
                env=env,
                pair_id=pair_id,
            )
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "step": step,
                    "gamma": gamma,
                    "query_index": replay["query_index"],
                    "query_hash": replay["query_hash"],
                    "certified_cost_selected_count": replay[
                        "certified_cost_selected_count"
                    ],
                    "verifier_spec_fingerprint": replay[
                        "verifier_spec_fingerprint"
                    ],
                    "effective_radius_m": replay["effective_radius_m"],
                    "nominal_sensing_radius_m": replay[
                        "nominal_sensing_radius_m"
                    ],
                    "nominal_polytope_nbase": replay["nominal_polytope_nbase"],
                    "nominal_predict_gain": replay["nominal_predict_gain"],
                    "nominal_schedule_tolerance": replay[
                        "nominal_schedule_tolerance"
                    ],
                    "nominal_worst_residual": replay["nominal_worst_residual"],
                    "nominal_worst_horizon_step": replay[
                        "nominal_worst_horizon_step"
                    ],
                    "fitted_face_sha256": replay["fitted_face_sha256"],
                    "nominal_face_sha256": replay["nominal_face_sha256"],
                    "fitted_face_signatures": replay["fitted_face_signatures"],
                    "nominal_face_signatures": replay["nominal_face_signatures"],
                }
            )
        figure.legend(
            handles=[
                Line2D([0], [0], color=NOMINAL_BLUE, lw=2, ls="--", label="nominal level H=0"),
                Line2D([0], [0], color=VERIFIER_GREEN, lw=2, label="fitted SOCP level H=0"),
                Line2D([0], [0], color="black", lw=1.2, marker=".", label="accepted exact H10"),
                Line2D([0], [0], color="0.35", lw=2, label="executed path to current step"),
            ],
            loc="lower center",
            ncol=4,
            frameon=False,
        )
        figure.suptitle(
            "Deterministic replay of stored exact SafeMPPI queries — fitted SOCP is the execution gate; nominal is separate",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0.08, 1, 0.93))
        frame_path = frame_directory / f"frame_{frame_index:06d}.png"
        figure.savefig(frame_path, dpi=100, facecolor="white")
        plt.close(figure)
        if frame_index % 10 == 0 or frame_index + 1 == len(shown_steps):
            print(
                f"[low7 video] frame={frame_index + 1}/{len(shown_steps)} "
                f"step={step} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    middle_frame = frame_directory / f"frame_{len(shown_steps) // 2:06d}.png"
    shutil.copyfile(middle_frame, preview)
    with signature_log.open("w") as stream:
        for row in frame_records:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        f"{1.0 / args.seconds_per_frame:.12g}",
        "-i",
        str(frame_directory / "frame_%06d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-r",
        str(args.fps),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    probe_result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ffprobe = json.loads(probe_result.stdout)
    face_hash_rows = [
        {
            "gamma": row["gamma"],
            "step": row["step"],
            "query_hash": row["query_hash"],
            "fitted": row["fitted_face_sha256"],
            "nominal": row["nominal_face_sha256"],
        }
        for row in frame_records
    ]
    face_aggregate_sha256 = hashlib.sha256(
        json.dumps(face_hash_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    planner_nominal_config = planner_scene.mode1_config(
        noise_var_mult=CANONICAL_EXPERT_RECIPE["noise_var_mult"]
    )
    candidate_sources = {
        f"{gamma:g}": {
            "candidate_meta": episode["candidate_meta"],
            "candidate_meta_sha256": episode["candidate_meta_sha256"],
            "array_sha256": episode["array_sha256"],
            "seed": int(episode["seed"]),
            "steps": int(episode["steps"]),
        }
        for gamma, episode in episodes.items()
    }
    video_manifest = {
        "schema_version": "afe_low7_exact_polytope_replay_video_v1",
        "status": "PASS",
        "diagnostic_only": True,
        "dataset_selection_changed": False,
        "visualization_evidence": "deterministic verifier replay",
        "video_renderer_sha256": sha256_file(Path(__file__)),
        "face_evidence": "recomputed from authenticated stored query; not originally persisted faces",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_dataset_sha256": source_manifest["dataset_sha256"],
        "selection_rule": (
            "lowest pair_id satisfying the declared diagnostic predicate: nonempty success "
            "at gamma {0.1,0.5,1.0}, exact planner-nominal H_t >= (1-gamma)^t "
            "for t=1..10 at every executed step, and independent fitted-SOCP replay"
        ),
        "diagnostic_pair_selection_success_conditioned": True,
        "diagnostic_pair_selection_nominal_conditioned": True,
        "selection_scope": "visual diagnostic only; no trajectory/window is added to or removed from the dataset",
        "selection_candidate_census": selection_census,
        "selected_pair_id": pair_id,
        "start": bank.starts[pair_id].tolist(),
        "goal": bank.goals[pair_id].tolist(),
        "gammas": list(VIDEO_GAMMAS),
        "candidate_sources": candidate_sources,
        "shown_steps": shown_steps,
        "common_safe_step_count": minimum_steps,
        "synchronization_rule": (
            "three panels use shared receding-horizon indices from step 0 through "
            "the shortest selected episode; longer post-arrival tails are not shown"
        ),
        "omitted_tail_steps_by_gamma": {
            f"{gamma:g}": int(episode["steps"]) - minimum_steps
            for gamma, episode in episodes.items()
        },
        "frame_stride": args.frame_stride,
        "frames": len(shown_steps),
        "fps": args.fps,
        "seconds_per_frame": args.seconds_per_frame,
        "verifier_implementation_fingerprint": verifier_implementation_fingerprint(),
        "verifier_spec_fingerprint": expected_spec,
        "polytope_replay_config": {
            "blue_nominal_source": (
                "grid_feats.polytope_HP over grid_scene.planner_obstacles(env); "
                "therefore preserves build_polytope_v2 12-nearest cap"
            ),
            "blue_nominal_sensing_radius_m": float(
                planner_nominal_config["barrier_activation_radius"]
            ),
            "blue_nominal_polytope_nbase": int(
                planner_nominal_config.get("polytope_nbase", 16)
            ),
            "blue_nominal_predict_gain": float(
                planner_nominal_config.get("predict_gain", 0.0)
            ),
            "blue_nominal_multistep_rule": "H_t >= (1-gamma)^t for t=1..10",
            "blue_nominal_multistep_tolerance": NOMINAL_SCHEDULE_TOLERANCE,
            "green_fitted_source": "verifier.VP.certify_window",
            "green_base_sensing_radius_m": float(VerifierConfig().sensing_radius),
            "green_effective_radius": "recorded per panel after rollout-padding rule",
        },
        "face_signature_schema": {
            "normal_and_margin": "IEEE-754 float64 hex",
            "ordering": "verifier return order",
            "per_panel_signatures": str(signature_log),
            "per_panel_signatures_sha256": sha256_file(signature_log),
            "aggregate_face_hash_sha256": face_aggregate_sha256,
        },
        "fail_closed_checks": {
            "stored_query_hash_recomputed": True,
            "selected_plan_is_cost_selected": True,
            "selected_plan_is_stored_safe_and_socp_positive": True,
            "deterministic_replay_metrics_bit_exact": True,
            "selected_plan_first_action_equals_executed_action": True,
            "every_shown_step_checked": True,
            "every_selected_episode_step_preflight_checked": True,
            "exact_planner_nominal_multistep_pass": True,
        },
        "colors": {"nominal": NOMINAL_BLUE, "fitted_socp": VERIFIER_GREEN},
        "nominal_vs_fitted_disclosure": (
            "nominal interior/internal_feasible does not imply external fitted-SOCP acceptance; "
            "the two contours and predicates are not conflated"
        ),
        "output_mp4": str(output),
        "preview_png": str(preview),
        "sha256": {"mp4": sha256_file(output), "preview": sha256_file(preview)},
        "ffmpeg_command": command,
        "ffprobe": ffprobe,
        "wall_seconds": time.perf_counter() - started,
    }
    video_manifest_path = output.with_name(f"{output.stem}_manifest.json")
    _atomic_json(video_manifest_path, video_manifest)
    if not args.keep_frames:
        shutil.rmtree(frame_directory)
    else:
        video_manifest["rendered_frames"] = str(frame_directory)
        _atomic_json(video_manifest_path, video_manifest)
    print(json.dumps(video_manifest, indent=2), flush=True)
    return video_manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    endpoints = commands.add_parser("endpoints", help="freeze one immutable endpoint bank")
    endpoints.add_argument("--output", type=Path, default=DEFAULT_ENDPOINT_MANIFEST)
    endpoints.add_argument("--pairs", type=int, default=DEFAULT_PAIR_COUNT)
    endpoints.add_argument("--seed", type=int, default=DEFAULT_ENDPOINT_SEED)
    endpoints.add_argument(
        "--fixed-goal-grid",
        action="store_true",
        help="use all free fixed-jitter grid starts and the canonical fixed goal",
    )

    starts = commands.add_parser("starts", help="render the endpoint start bank before rollout")
    starts.add_argument("--endpoint-manifest", type=Path, required=True)
    starts.add_argument("--output", type=Path, required=True)

    collect = commands.add_parser("collect", help="collect one gamma shard")
    collect.add_argument("--endpoint-manifest", type=Path, required=True)
    collect.add_argument("--gamma", type=float, required=True)
    collect.add_argument("--outdir", type=Path, required=True)
    collect.add_argument("--device", default="cuda:0")
    collect.add_argument("--retries", type=int, default=3)
    collect.add_argument("--planner-seed0", type=int, default=DEFAULT_PLANNER_SEED0)
    collect.add_argument(
        "--max-steps", type=int, default=CANONICAL_EXPERT_RECIPE["max_steps"]
    )
    collect.add_argument(
        "--reach", type=float, default=CANONICAL_EXPERT_RECIPE["reach_m"]
    )
    collect.add_argument(
        "--smooth-weight",
        type=float,
        default=CANONICAL_EXPERT_RECIPE["smooth_weight"],
    )
    collect.add_argument(
        "--retreat-weight",
        type=float,
        default=CANONICAL_EXPERT_RECIPE["retreat_weight"],
    )
    collect.add_argument(
        "--noise-var-mult",
        type=float,
        default=CANONICAL_EXPERT_RECIPE["noise_var_mult"],
    )
    collect.add_argument(
        "--max-debug-candidates",
        type=int,
        default=CANONICAL_EXPERT_RECIPE["max_debug_candidates"],
    )
    collect.add_argument(
        "--max-proposals",
        type=int,
        default=CANONICAL_EXPERT_RECIPE["max_proposals_per_step"],
    )

    combine = commands.add_parser("combine", help="authenticate and merge seven gamma shards")
    combine.add_argument("--shard-manifests", nargs="+", type=Path, required=True)
    combine.add_argument("--outdir", type=Path, required=True)

    render = commands.add_parser("render", help="render full-space all-gamma trajectories")
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--output", type=Path)

    video = commands.add_parser(
        "video", help="replay exact selected queries with nominal/fitted polytopes"
    )
    video.add_argument("--manifest", type=Path, required=True)
    video.add_argument("--output", type=Path)
    video.add_argument("--frame-stride", type=int, default=1)
    video.add_argument("--fps", type=int, default=12)
    video.add_argument("--seconds-per-frame", type=float, default=0.15)
    video.add_argument("--keep-frames", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.command == "endpoints":
        generate_endpoints(args)
    elif args.command == "starts":
        render_endpoint_starts(args)
    elif args.command == "collect":
        collect_gamma_shard(args)
    elif args.command == "combine":
        combine_shards(args)
    elif args.command == "render":
        render_overlay(args)
    elif args.command == "video":
        render_polytope_video(args)
    else:  # pragma: no cover - argparse makes this unreachable
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
