#!/usr/bin/env python3
"""Authenticated pretrained-only evaluation for the low7 candidate policy.

The evaluator is deliberately downstream-only: it loads one fresh low7
candidate, generates raw untilted receding-horizon rollouts, and only then
measures sampled-plan validity.  Verifier results never select, replace, or
modify an executed action.

Three endpoint-matched scenes are evaluated with common random numbers:

* the nominal 4x4 radius-0.2 stadium;
* the four-to-one radius-1.0 central obstacle scene; and
* the 4x4 scene with all sixteen interior radii changed to 0.3.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
CHALLENGING = HERE.parent
REV_EXPANSION = CHALLENGING.parent
WORK = REV_EXPANSION.parent
CODEX_OVERNIGHT = REV_EXPANSION / "codex_overnight"
for _path in (WORK, REV_EXPANSION, CODEX_OVERNIGHT, CHALLENGING):
    value = str(_path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401,E402
import grid_hp_expt as HP  # noqa: E402
import grid_feats as GF  # noqa: E402
from afe2_scene_profiles import (  # noqa: E402
    assert_scene_snapshot,
    build_scene,
    get_scene_profile,
    scene_snapshot,
)
from afe_restart.config import DynamicsConfig, VerifierConfig  # noqa: E402
from afe_restart.dynamics import step_state  # noqa: E402
from afe_restart.policy import model_state_hash  # noqa: E402
from afe_restart.scene import (  # noqa: E402
    verifier_implementation_fingerprint,
    verifier_spec_fingerprint,
)
from afe_restart.verifier import verify_plan as verify_full_plan  # noqa: E402


if Path(HP.__file__).resolve().parent != CHALLENGING:
    raise ImportError(f"expected challenging/grid_hp_expt.py, imported {HP.__file__}")
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
SCENE_NAMES = (
    "low7_id_canonical_v1",
    "low7_radius1_canonical_v1",
    "low7_radius03_canonical_v1",
)
SCENE_LABELS = {
    "low7_id_canonical_v1": "Nominal r=0.2",
    "low7_radius1_canonical_v1": "Giant center r=1.0",
    "low7_radius03_canonical_v1": "All interior r=0.3",
}
METRIC_VERSION = "low7_pretrained_raw_v1"
CHECKPOINT_STAGE_SCHEMA = "afe_fresh_pretrain_v2_low7_uniform_pairs"
REFLECTION_CHECKPOINT_STAGE_SCHEMA = "afe_fresh_pretrain_v3_low7_reflection_paired"
EQUIVARIANT_CHECKPOINT_STAGE_SCHEMA = (
    "afe_fresh_pretrain_v4_low7_reflection_equivariant"
)
GROUP_AVERAGED_CHECKPOINT_STAGE_SCHEMA = (
    "afe_fresh_pretrain_v5_low7_reflection_group_average"
)
MODEL_SCHEMA = "w8sg-hp-v3-low7-closest-boundary"
GROUP_AVERAGED_MODEL_SCHEMA = "w8sg-hp-v4-low7-closest-boundary-tie-mean"
LOW7_SCHEMA = "low7_closest_boundary"
LOW7_TIE_SCHEMA = "low7_closest_boundary_tie_mean"
T = 300
REACH = 0.15
NFE = 12
TEMP = 1.0
N_THETA = 180
WORKSPACE_LOW = 0.0
WORKSPACE_HIGH = 5.0
DELTA_PROG = 0.10
DEFAULT_M = 20
EXPECTED_PARAMETER_COUNT = 70_308


class CheckpointContractError(RuntimeError):
    """The requested file is not the unpromoted fresh low7 candidate."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(
                json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n"
            )


def _require_exact(config: Mapping[str, Any], field: str, expected: Any) -> None:
    actual = config.get(field)
    if field in {"grid_shape", "grid_hw", "trunk_hidden"} and actual is not None:
        actual = tuple(actual)
        expected = tuple(expected)
    if actual != expected:
        raise CheckpointContractError(
            f"checkpoint config {field}={actual!r}, expected {expected!r}"
        )


def load_low7_candidate(
    checkpoint_path: Path,
    expected_file_sha256: str,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Fail closed unless the file is exactly the fresh, unpromoted low7 model."""

    checkpoint_path = checkpoint_path.resolve()
    expected = str(expected_file_sha256).strip().lower()
    if not _is_sha256(expected):
        raise CheckpointContractError("expected checkpoint SHA-256 must be 64 lowercase hex digits")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    before_hash = sha256_file(checkpoint_path)
    if before_hash != expected:
        raise CheckpointContractError(
            f"checkpoint SHA-256 {before_hash} != caller-declared {expected}"
        )
    policy, payload = HP.load_hp(checkpoint_path, device="cpu")
    if sha256_file(checkpoint_path) != before_hash:
        raise CheckpointContractError("checkpoint changed while it was being loaded")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise CheckpointContractError("checkpoint has no model config")
    stage_schema = payload.get("stage_schema")
    group_averaged = stage_schema == GROUP_AVERAGED_CHECKPOINT_STAGE_SCHEMA
    exact_fields = {
        "arch": "hp-repr",
        "schema_version": (
            GROUP_AVERAGED_MODEL_SCHEMA if group_averaged else MODEL_SCHEMA
        ),
        "raw_start_goal": False,
        "H_pred": 10,
        "grid_shape": (1, 32, 32),
        "K_hist": 16,
        "width": 256,
        "depth": 2,
        "u_max": 1.0,
        "ctx_dim": 39,
        "raw_condition_dim": 7,
        "conditioning_schema": LOW7_TIE_SCHEMA if group_averaged else LOW7_SCHEMA,
        "use_gru": False,
        "repr_dim": 32,
        "grid_hw": (32, 32),
        "trunk_hidden": (160, 96),
        "enc_depth": 3,
        "boundary_adapter": False,
    }
    for field, expected_value in exact_fields.items():
        _require_exact(config, field, expected_value)
    if stage_schema not in {
        CHECKPOINT_STAGE_SCHEMA,
        REFLECTION_CHECKPOINT_STAGE_SCHEMA,
        EQUIVARIANT_CHECKPOINT_STAGE_SCHEMA,
        GROUP_AVERAGED_CHECKPOINT_STAGE_SCHEMA,
    }:
        raise CheckpointContractError(
            f"checkpoint payload stage_schema={stage_schema!r} is not supported"
        )
    required_payload = {
        "fresh_from_scratch": True,
        "endpoint_free": True,
        "encoder_trainable_during_pretraining": True,
        "expansion_promotion": False,
    }
    for field, expected_value in required_payload.items():
        if payload.get(field) != expected_value:
            raise CheckpointContractError(
                f"checkpoint payload {field}={payload.get(field)!r}, expected {expected_value!r}"
            )
    if stage_schema in {
        REFLECTION_CHECKPOINT_STAGE_SCHEMA,
        EQUIVARIANT_CHECKPOINT_STAGE_SCHEMA,
        GROUP_AVERAGED_CHECKPOINT_STAGE_SCHEMA,
    }:
        if payload.get("reflection_paired_pretraining") is not True:
            raise CheckpointContractError(
                "reflection-paired checkpoint lost its pretraining contract"
            )
    if stage_schema == EQUIVARIANT_CHECKPOINT_STAGE_SCHEMA:
        if not float(payload.get("equivariance_weight", 0.0)) > 0.0:
            raise CheckpointContractError(
                "equivariant checkpoint lacks a positive consistency weight"
            )
    if stage_schema == GROUP_AVERAGED_CHECKPOINT_STAGE_SCHEMA:
        if payload.get("reflection_group_average") is not True:
            raise CheckpointContractError(
                "group-averaged checkpoint lost its exact symmetry contract"
            )
        if config.get("reflection_group_average") is not True:
            raise CheckpointContractError(
                "group-averaged checkpoint does not reconstruct the symmetric model"
            )
        transform = payload.get("conditioning_transform")
        if not isinstance(transform, Mapping) or transform.get("name") != (
            "equal-nearest-boundary-vector-mean-v1"
        ):
            raise CheckpointContractError(
                "group-averaged checkpoint lacks its tie-mean conditioning transform"
            )
    fixed_goal_grid = payload.get("fixed_goal") is not None
    if fixed_goal_grid:
        if (
            payload.get("domain_randomized_start_goal") is not False
            or payload.get("domain_randomized_start") is not True
            or payload.get("zero_initial_velocity") is not True
            or payload.get("diagonal_start_exclusion") is not False
            or payload.get("fixed_goal") != [4.7, 4.7]
        ):
            raise CheckpointContractError("fixed-goal grid checkpoint provenance is inconsistent")
    elif payload.get("domain_randomized_start_goal") is not True:
        raise CheckpointContractError("random-pair checkpoint lost its endpoint provenance")
    if payload.get("frozen_feature_snapshot", False):
        raise CheckpointContractError("evaluation requires the policy candidate, not phi0 snapshot")
    if not _is_sha256(payload.get("source_query_hash_digest")):
        raise CheckpointContractError("checkpoint lacks an authenticated source-query digest")
    if not str(payload.get("source_manifest", "")).strip():
        raise CheckpointContractError("checkpoint lacks source-manifest provenance")
    embedded_state_hash = payload.get("model_state_sha256")
    actual_state_hash = model_state_hash(policy)
    if not _is_sha256(embedded_state_hash) or embedded_state_hash != actual_state_hash:
        raise CheckpointContractError("embedded model-state hash does not match loaded tensors")
    if policy.ctx_dim != 39 or policy.trunk[0].in_features != 91:
        raise CheckpointContractError("loaded low7 model does not have ctx=39 / trunk input=91")
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise CheckpointContractError(
            f"low7 parameter count {parameter_count} != {EXPECTED_PARAMETER_COUNT}"
        )
    contract = {
        "path": str(checkpoint_path),
        "file_sha256": before_hash,
        "caller_hash_verified": True,
        "model_state_sha256": actual_state_hash,
        "config": dict(config),
        "stage_schema": stage_schema,
        "reflection_paired_pretraining": bool(
            payload.get("reflection_paired_pretraining", False)
        ),
        "equivariance_weight": float(payload.get("equivariance_weight", 0.0)),
        "reflection_group_average": bool(
            payload.get("reflection_group_average", False)
        ),
        "conditioning_transform": payload.get("conditioning_transform"),
        "source_manifest": payload["source_manifest"],
        "source_query_hash_digest": payload["source_query_hash_digest"],
        "best_epoch": int(payload["best_epoch"]),
        "best_validation_cfm": float(payload["best_validation_cfm"]),
        "expansion_promotion": False,
        "fixed_goal_grid": fixed_goal_grid,
        "parameter_count": parameter_count,
    }
    return policy.to(device).eval(), contract


def _interior_rows(obstacles: np.ndarray) -> list[int]:
    rows = []
    for x in (1.0, 2.0, 3.0, 4.0):
        for y in (1.0, 2.0, 3.0, 4.0):
            match = np.flatnonzero(
                np.all(
                    np.isclose(obstacles[:, :2], (x, y), rtol=0.0, atol=1.0e-7),
                    axis=1,
                )
            )
            if len(match) == 1:
                rows.append(int(match[0]))
    return rows


def validate_scene_contract(name: str, env: Any) -> dict[str, Any]:
    """Verify endpoints and the exact scientific geometry of one declared scene."""

    profile = get_scene_profile(name)
    snapshot = scene_snapshot(env, profile)
    assert_scene_snapshot(snapshot)
    start = env.x0.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    np.testing.assert_allclose(start, (0.3, 0.3, 0.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(goal, (4.7, 4.7), atol=1.0e-7)
    obstacles = env.obstacles.detach().cpu().numpy()
    interior = _interior_rows(obstacles)
    if name == "low7_id_canonical_v1":
        if len(interior) != 16 or not np.allclose(obstacles[interior, 2], 0.2):
            raise RuntimeError("nominal low7 scene is not the 4x4 radius-0.2 grid")
    elif name == "low7_radius03_canonical_v1":
        if len(interior) != 16 or not np.allclose(obstacles[interior, 2], 0.3):
            raise RuntimeError("radius-0.3 low7 scene did not change all sixteen disks")
    elif name == "low7_radius1_canonical_v1":
        if len(interior) != 12:
            raise RuntimeError("giant scene did not remove exactly four central disks")
        giant = np.flatnonzero(
            np.all(
                np.isclose(obstacles[:, :2], (2.5, 2.5), rtol=0.0, atol=1.0e-7),
                axis=1,
            )
            & np.isclose(obstacles[:, 2], 1.0, rtol=0.0, atol=1.0e-7)
        )
        if len(giant) != 1:
            raise RuntimeError("giant scene lacks the unique radius-1.0 central disk")
    else:
        raise ValueError(f"undeclared low7 evaluation scene {name!r}")
    return snapshot


def raw_noise_seed(
    gamma: float,
    rollout_index: int,
    control_t: int,
    *,
    seed_bank: str = METRIC_VERSION,
) -> int:
    """Common-random-number seed; deliberately independent of scene geometry."""

    key = (
        f"{seed_bank}|raw|gamma={float(gamma):.1f}|"
        f"index={int(rollout_index)}|t={int(control_t)}"
    ).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


@torch.no_grad()
def run_raw_rollouts(
    policy: torch.nn.Module,
    env: Any,
    scene_name: str,
    *,
    m: int,
    gammas: Sequence[float] = GAMMAS,
    horizon: int = T,
    reach: float = REACH,
    nfe: int = NFE,
    device: str | torch.device = "cpu",
    seed_bank: str = METRIC_VERSION,
    reflection_antithetic: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate raw receding-horizon trajectories without any verifier call."""

    device = torch.device(device)
    if reflection_antithetic:
        if m % 2:
            raise ValueError("reflection-antithetic raw evaluation requires even M")
        if not bool(getattr(policy, "reflection_group_average", False)):
            raise ValueError(
                "reflection-antithetic raw evaluation requires an exactly "
                "reflection-group-averaged policy"
            )
    start = env.x0.detach().cpu().numpy().astype(np.float64)
    goal = env.goal.detach().cpu().numpy().astype(np.float64)
    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64)
    robot_radius = float(env.r_robot)
    episodes: list[dict[str, Any]] = []
    for gamma in gammas:
        for rollout_index in range(m):
            episodes.append(
                {
                    "scene": scene_name,
                    "gamma": float(gamma),
                    "rollout_index": int(rollout_index),
                    "state": start.copy(),
                    "history": [],
                    "path": [start[:2].copy()],
                    "controls": [],
                    "status": None,
                }
            )
    sampled_plans: list[dict[str, Any]] = []
    for control_t in range(horizon):
        active = [episode for episode in episodes if episode["status"] is None]
        if not active:
            break
        grids, conditions, histories, noises = [], [], [], []
        for episode in active:
            state = episode["state"]
            grids.append(GF.axis_grid(state[:2], obstacles, robot_radius))
            conditions.append(
                GF.low7(
                    state,
                    goal,
                    episode["gamma"],
                    obstacles,
                    robot_radius,
                    tie_average=(
                        getattr(policy, "conditioning_schema", LOW7_SCHEMA)
                        == LOW7_TIE_SCHEMA
                    ),
                )
            )
            recent = np.asarray(episode["history"][-GF.K_HIST :], dtype=np.float32)
            histories.append(
                GF.hist_pad(recent if recent.size else np.zeros((0, 2)), GF.K_HIST)
            )
            pair_size = m // 2 if reflection_antithetic else m
            base_rollout_index = (
                episode["rollout_index"] % pair_size
                if reflection_antithetic
                else episode["rollout_index"]
            )
            seed = raw_noise_seed(
                episode["gamma"],
                base_rollout_index,
                control_t,
                seed_bank=seed_bank,
            )
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(policy.d, generator=generator, device=device)
            if reflection_antithetic and episode["rollout_index"] >= pair_size:
                noise = noise.reshape(-1, 2).flip(-1).reshape_as(noise)
            noises.append(noise)
        grid_tensor = torch.as_tensor(np.asarray(grids), device=device)
        condition_tensor = torch.as_tensor(np.asarray(conditions), device=device)
        history_tensor = torch.as_tensor(np.asarray(histories), device=device)
        context = policy.ctx_from(grid_tensor, condition_tensor, history_tensor)
        windows = policy.sample(
            len(active),
            context,
            nfe=nfe,
            temp=TEMP,
            initial_noise=torch.stack(noises),
        ).detach().cpu().numpy()
        for episode, window in zip(active, windows):
            state_before = episode["state"].copy()
            pair_size = m // 2 if reflection_antithetic else m
            base_rollout_index = (
                episode["rollout_index"] % pair_size
                if reflection_antithetic
                else episode["rollout_index"]
            )
            seed = raw_noise_seed(
                episode["gamma"],
                base_rollout_index,
                control_t,
                seed_bank=seed_bank,
            )
            sampled_plans.append(
                {
                    "scene": scene_name,
                    "gamma": float(episode["gamma"]),
                    "rollout_index": int(episode["rollout_index"]),
                    "control_t": int(control_t),
                    "seed": int(seed),
                    "reflection_antithetic": bool(reflection_antithetic),
                    "reflection_pair_index": int(base_rollout_index),
                    "state": state_before,
                    "plan": np.asarray(window, dtype=np.float32),
                }
            )
            action = np.asarray(window[0], dtype=np.float32)
            episode["state"] = step_state(state_before, action, dt=float(env.dt))
            episode["history"].append(action)
            episode["controls"].append(action)
            episode["path"].append(episode["state"][:2].copy())
            point = episode["state"][:2]
            if (point < WORKSPACE_LOW).any() or (point > WORKSPACE_HIGH).any():
                episode["status"] = "oob"
            elif obstacles.size and (
                np.linalg.norm(point[None] - obstacles[:, :2], axis=1)
                - obstacles[:, 2]
                - robot_radius
            ).min() < 0.0:
                episode["status"] = "collision"
            elif np.linalg.norm(point - goal) < reach:
                episode["status"] = "reached"
    output = []
    for episode in episodes:
        status = episode["status"] or "timeout"
        output.append(
            {
                "scene": scene_name,
                "gamma": float(episode["gamma"]),
                "rollout_index": int(episode["rollout_index"]),
                "path": np.asarray(episode["path"], dtype=np.float32),
                "controls": np.asarray(episode["controls"], dtype=np.float32).reshape(-1, 2),
                "status": status,
            }
        )
    return output, sampled_plans


def trajectory_metrics(episode: Mapping[str, Any], env: Any) -> dict[str, Any]:
    points = np.asarray(episode["path"], dtype=np.float64)
    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64)
    clearance = float(
        (
            np.linalg.norm(points[:, None] - obstacles[None, :, :2], axis=2)
            - obstacles[None, :, 2]
            - float(env.r_robot)
        ).min()
    )
    collision = clearance < 0.0
    oob = bool((points < WORKSPACE_LOW).any() or (points > WORKSPACE_HIGH).any())
    reached = episode["status"] == "reached"
    success = bool(reached and not collision and not oob)
    steps = len(points) - 1
    goal = env.goal.detach().cpu().numpy()
    return {
        "success": success,
        "collision": bool(collision),
        "oob": oob,
        "timeout": bool(episode["status"] == "timeout"),
        "minimum_clearance": clearance,
        "time_to_goal": float(steps * env.dt) if success else None,
        "steps": int(steps),
        "endpoint_distance": float(np.linalg.norm(points[-1] - goal)),
    }


def plan_validity_from_verification(verification: Any) -> dict[str, Any]:
    safe = bool(verification.safe)
    full = bool(
        safe
        and float(verification.progress_m)
        >= min(DELTA_PROG, 0.5 * float(verification.start_goal_distance_m))
    )
    rejected = []
    if not bool(verification.in_bounds):
        rejected.append("out_of_bounds")
    if not bool(verification.socp_ok):
        rejected.append("socp_rejected")
    return {
        "v_safe": safe,
        "v_full": full,
        "reason": "safe" if safe else "+".join(rejected),
        "bounds_margin": float(verification.bounds_margin_m),
        "physical_clearance": float(verification.physical_clearance_m),
        "margin": float(verification.face_margin_m),
        "residual": float(verification.certificate_residual),
        "progress": float(verification.progress_m),
        "initial_goal_distance": float(verification.start_goal_distance_m),
    }


_WORKER_SCENES: dict[str, Any] = {}


def _verify_plan_chunk(
    request: tuple[int, list[tuple[str, np.ndarray, np.ndarray, float]]]
) -> list[dict[str, Any]]:
    n_theta, items = request
    output = []
    for scene_name, state, plan, gamma in items:
        if scene_name not in _WORKER_SCENES:
            _WORKER_SCENES[scene_name] = build_scene(get_scene_profile(scene_name))
        env = _WORKER_SCENES[scene_name]
        verifier = VerifierConfig(angle_samples=int(n_theta))
        verification = verify_full_plan(
            state,
            plan,
            env,
            gamma,
            goal=env.goal.detach().cpu().numpy(),
            dynamics=DynamicsConfig(),
            verifier=verifier,
        )
        output.append(plan_validity_from_verification(verification))
    return output


def verify_sampled_plans(
    records: list[dict[str, Any]],
    *,
    workers: int,
    chunk_size: int,
    n_theta: int = N_THETA,
) -> list[dict[str, Any]]:
    """Measure plan validity after rollout generation, preserving record order."""

    tasks = [
        (
            str(record["scene"]),
            np.asarray(record["state"], dtype=np.float64),
            np.asarray(record["plan"], dtype=np.float32),
            float(record["gamma"]),
        )
        for record in records
    ]
    chunks = [tasks[offset : offset + chunk_size] for offset in range(0, len(tasks), chunk_size)]
    requests = [(int(n_theta), chunk) for chunk in chunks]
    output: list[dict[str, Any]] = []
    if workers == 1:
        for request in requests:
            output.extend(_verify_plan_chunk(request))
        return output
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for index, result in enumerate(
            executor.map(_verify_plan_chunk, requests, chunksize=1), start=1
        ):
            output.extend(result)
            if index % max(1, len(requests) // 20) == 0 or index == len(requests):
                print(
                    f"[plan validity] {min(index * chunk_size, len(records))}/{len(records)}",
                    flush=True,
                )
    if len(output) != len(records):
        raise RuntimeError("sampled-plan verifier lost records")
    return output


def _rate_entry(values: Sequence[bool]) -> dict[str, Any]:
    n = len(values)
    count = sum(bool(value) for value in values)
    if n == 0:
        return {"count": 0, "n": 0, "estimate": None, "wilson95": [None, None]}
    estimate = float(count / n)
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    center = (estimate + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(
        estimate * (1.0 - estimate) / n + z * z / (4.0 * n * n)
    ) / denominator
    return {
        "count": int(count),
        "n": n,
        "estimate": estimate,
        "wilson95": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def _continuous_entry(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": float(np.mean(finite)) if finite else None,
        "median": float(np.median(finite)) if finite else None,
        "minimum": float(np.min(finite)) if finite else None,
        "values": finite,
    }


def aggregate_metrics(
    scene_name: str,
    gamma: float | None,
    episodes: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    *,
    m: int,
) -> dict[str, Any]:
    selected_episodes = [
        episode
        for episode in episodes
        if episode["scene"] == scene_name
        and (gamma is None or episode["gamma"] == float(gamma))
    ]
    selected_plans = [
        plan
        for plan in plans
        if plan["scene"] == scene_name and (gamma is None or plan["gamma"] == float(gamma))
    ]
    if len(selected_episodes) != (m * len(GAMMAS) if gamma is None else m):
        raise RuntimeError(f"incomplete rollout metric cell for {scene_name}, gamma={gamma}")
    metrics = [episode["metrics"] for episode in selected_episodes]
    successful_times = [
        value for value in (metric["time_to_goal"] for metric in metrics) if value is not None
    ]
    episode_plan_values = []
    for episode in selected_episodes:
        matches = [
            plan["validity"]
            for plan in selected_plans
            if plan["gamma"] == episode["gamma"]
            and plan["rollout_index"] == episode["rollout_index"]
        ]
        if matches:
            episode_plan_values.append(
                {
                    "V_safe": float(np.mean([value["v_safe"] for value in matches])),
                    "V_full": float(np.mean([value["v_full"] for value in matches])),
                }
            )
    return {
        "metric_version": METRIC_VERSION,
        "scene": scene_name,
        "scene_label": SCENE_LABELS[scene_name],
        "scope": "pooled" if gamma is None else "gamma",
        "gamma": gamma,
        "M_per_gamma": m,
        "rollout": {
            "n": len(metrics),
            "SR": _rate_entry([metric["success"] for metric in metrics]),
            "collision": _rate_entry([metric["collision"] for metric in metrics]),
            "OOB": _rate_entry([metric["oob"] for metric in metrics]),
            "timeout": _rate_entry([metric["timeout"] for metric in metrics]),
            "minimum_clearance": _continuous_entry(
                [metric["minimum_clearance"] for metric in metrics]
            ),
            "successful_time_to_goal": _continuous_entry(successful_times),
            "endpoint_distance": _continuous_entry(
                [metric["endpoint_distance"] for metric in metrics]
            ),
            "terminal_status_counts": dict(
                sorted(Counter(str(episode["status"]) for episode in selected_episodes).items())
            ),
        },
        "sampled_plan": {
            "n": len(selected_plans),
            "V_safe": _rate_entry([plan["validity"]["v_safe"] for plan in selected_plans]),
            "V_full": _rate_entry([plan["validity"]["v_full"] for plan in selected_plans]),
            "reason_counts": dict(
                sorted(Counter(plan["validity"]["reason"] for plan in selected_plans).items())
            ),
            "denominator_note": (
                "on-policy decision-weighted: all raw H=10 plans sampled at executed "
                "receding-horizon control steps; verification is post-hoc and cannot affect rollout"
            ),
            "episode_macro_V_safe": _continuous_entry(
                [value["V_safe"] for value in episode_plan_values]
            ),
            "episode_macro_V_full": _continuous_entry(
                [value["V_full"] for value in episode_plan_values]
            ),
            "episode_macro_note": (
                "mean of each episode's decision-level validity, then equal weight per episode"
            ),
        },
    }


def _object_array(values: Sequence[Any]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def save_archives(
    outdir: Path,
    episodes: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    paths_file = outdir / "raw_rollouts.npz"
    np.savez_compressed(
        paths_file,
        paths=_object_array([episode["path"] for episode in episodes]),
        controls=_object_array([episode["controls"] for episode in episodes]),
        scene=np.asarray([episode["scene"] for episode in episodes]),
        gamma=np.asarray([episode["gamma"] for episode in episodes], dtype=np.float32),
        rollout_index=np.asarray(
            [episode["rollout_index"] for episode in episodes], dtype=np.int32
        ),
        status=np.asarray([episode["status"] for episode in episodes]),
        minimum_clearance=np.asarray(
            [episode["metrics"]["minimum_clearance"] for episode in episodes],
            dtype=np.float32,
        ),
        success=np.asarray([episode["metrics"]["success"] for episode in episodes]),
    )
    plans_file = outdir / "raw_sampled_plans.npz"
    np.savez_compressed(
        plans_file,
        state=np.stack([plan["state"] for plan in plans]).astype(np.float32),
        plan=np.stack([plan["plan"] for plan in plans]).astype(np.float32),
        scene=np.asarray([plan["scene"] for plan in plans]),
        gamma=np.asarray([plan["gamma"] for plan in plans], dtype=np.float32),
        rollout_index=np.asarray([plan["rollout_index"] for plan in plans], dtype=np.int32),
        control_t=np.asarray([plan["control_t"] for plan in plans], dtype=np.int16),
        seed=np.asarray([plan["seed"] for plan in plans], dtype=np.int64),
        v_safe=np.asarray([plan["validity"]["v_safe"] for plan in plans]),
        v_full=np.asarray([plan["validity"]["v_full"] for plan in plans]),
        reason=np.asarray([plan["validity"]["reason"] for plan in plans]),
        margin=np.asarray([plan["validity"]["margin"] for plan in plans], dtype=np.float32),
        progress=np.asarray([plan["validity"]["progress"] for plan in plans], dtype=np.float32),
    )
    return paths_file, plans_file


def _draw_scene(axis: Any, env: Any) -> None:
    for x, y, radius in env.obstacles.detach().cpu().numpy():
        axis.add_patch(plt.Circle((x, y), radius, color="#b8b8b8", ec="none", zorder=1))
    axis.set_xlim(-0.35, 5.35)
    axis.set_ylim(-0.35, 5.35)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def render_gallery(
    outdir: Path,
    episodes: Sequence[Mapping[str, Any]],
    scenes: Mapping[str, Any],
    *,
    m: int,
) -> tuple[list[Path], Path]:
    indices = tuple(range(min(10, m)))
    cmap = plt.get_cmap("plasma")
    colors = {
        gamma: cmap(0.08 + 0.84 * index / (len(GAMMAS) - 1))
        for index, gamma in enumerate(GAMMAS)
    }
    figure, axes = plt.subplots(3, 7, figsize=(23.5, 10.5), squeeze=False)
    for row, scene_name in enumerate(SCENE_NAMES):
        env = scenes[scene_name]
        for column, gamma in enumerate(GAMMAS):
            axis = axes[row, column]
            _draw_scene(axis, env)
            cell = [
                episode
                for episode in episodes
                if episode["scene"] == scene_name
                and episode["gamma"] == gamma
                and episode["rollout_index"] in indices
            ]
            cell.sort(key=lambda episode: episode["rollout_index"])
            if [episode["rollout_index"] for episode in cell] != list(indices):
                raise RuntimeError("fixed-index gallery cell is incomplete")
            for episode in cell:
                path = np.asarray(episode["path"])
                axis.plot(
                    path[:, 0], path[:, 1], color=colors[gamma], lw=1.05, alpha=0.72, zorder=3
                )
                if not episode["metrics"]["success"]:
                    axis.plot(path[-1, 0], path[-1, 1], "x", color="#cc3311", ms=4, zorder=5)
            axis.plot(*env.x0[:2].detach().cpu().numpy(), "ks", ms=3.5, zorder=6)
            axis.plot(
                *env.goal.detach().cpu().numpy(), marker="*", color="gold", mec="k", ms=9, zorder=6
            )
            if row == 0:
                axis.set_title(rf"$\gamma={gamma:g}$", fontsize=10)
            if column == 0:
                axis.set_ylabel(SCENE_LABELS[scene_name], fontsize=11)
    figure.suptitle(
        f"Pretrained low7 raw rollouts — fixed non-curated indices {list(indices)}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"low7_pretrained_fixed_index_gallery.{suffix}"
        figure.savefig(path, dpi=170)
        outputs.append(path)
    plt.close(figure)
    manifest = outdir / "gallery_indices.json"
    write_json(
        manifest,
        {
            "selection_rule": "first fixed archive indices; no outcome inspection",
            "indices": indices,
            "scenes": SCENE_NAMES,
            "gammas": GAMMAS,
            "M": m,
        },
    )
    return outputs, manifest


def render_metric_figure(outdir: Path, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    gamma_rows = {(row["scene"], row["gamma"]): row for row in rows if row["scope"] == "gamma"}
    specs = (
        ("SR", "Raw success rate", lambda row: row["rollout"]["SR"]["estimate"]),
        ("collision", "Collision rate", lambda row: row["rollout"]["collision"]["estimate"]),
        ("OOB", "Out-of-bounds rate", lambda row: row["rollout"]["OOB"]["estimate"]),
        ("timeout", "Timeout rate", lambda row: row["rollout"]["timeout"]["estimate"]),
        (
            "clearance",
            "Mean minimum clearance [m]",
            lambda row: row["rollout"]["minimum_clearance"]["mean"],
        ),
        (
            "time",
            "Successful time-to-goal [s]",
            lambda row: row["rollout"]["successful_time_to_goal"]["mean"],
        ),
        ("V_safe", "Raw sampled-plan V-safe", lambda row: row["sampled_plan"]["V_safe"]["estimate"]),
        ("V_full", "Raw sampled-plan V-full", lambda row: row["sampled_plan"]["V_full"]["estimate"]),
    )
    colors = ("#0072b2", "#d55e00", "#009e73")
    figure, axes = plt.subplots(2, 4, figsize=(18.5, 8.3), squeeze=False)
    for axis, (key, title, extractor) in zip(axes.flat, specs):
        for color, scene_name in zip(colors, SCENE_NAMES):
            values = []
            for gamma in GAMMAS:
                value = extractor(gamma_rows[(scene_name, gamma)])
                values.append(np.nan if value is None else float(value))
            axis.plot(GAMMAS, values, marker="o", ms=4, lw=1.8, color=color, label=SCENE_LABELS[scene_name])
        axis.set_title(title)
        axis.set_xlabel(r"$\gamma$")
        axis.grid(alpha=0.25)
        if key in {"SR", "collision", "OOB", "timeout", "V_safe", "V_full"}:
            axis.set_ylim(-0.03, 1.03)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Pretrained-only low7 evaluation (raw untilted controller; verifier used only post hoc)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"low7_pretrained_metrics.{suffix}"
        figure.savefig(path, dpi=170)
        outputs.append(path)
    plt.close(figure)
    return outputs


def _git_provenance() -> dict[str, Any]:
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode != 0
        return {"root": root, "commit": commit, "tracked_dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def _device_provenance(device: torch.device) -> dict[str, Any]:
    output = {
        "requested": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
        torch.cuda.set_device(device)
        output.update(
            process_device_index=torch.cuda.current_device(),
            name=torch.cuda.get_device_name(device),
        )
    return output


def _artifact_inventory(outdir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(outdir)): sha256_file(path)
        for path in sorted(outdir.rglob("*"))
        if path.is_file() and path.name not in {"EVALUATION_COMPLETE.json", "SHA256SUMS"}
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"evaluation output root must be absent: {outdir}")
    if args.m <= 0 or args.verifier_workers <= 0 or args.verifier_chunk_size <= 0:
        raise ValueError("M, verifier workers, and verifier chunk size must be positive")
    if args.nfe <= 0:
        raise ValueError("NFE must be positive")
    device = torch.device(args.device)
    device_provenance = _device_provenance(device)
    policy, checkpoint = load_low7_candidate(
        args.checkpoint, args.expected_checkpoint_sha256, device
    )
    scenes: dict[str, Any] = {}
    snapshots = {}
    for name in SCENE_NAMES:
        env = build_scene(get_scene_profile(name))
        snapshots[name] = validate_scene_contract(name, env)
        scenes[name] = env

    outdir.mkdir(parents=True)
    episodes: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    for scene_name in SCENE_NAMES:
        print(f"[raw rollout] {SCENE_LABELS[scene_name]}", flush=True)
        scene_episodes, scene_plans = run_raw_rollouts(
            policy,
            scenes[scene_name],
            scene_name,
            m=args.m,
            nfe=args.nfe,
            device=device,
        )
        for episode in scene_episodes:
            episode["metrics"] = trajectory_metrics(episode, scenes[scene_name])
        episodes.extend(scene_episodes)
        plans.extend(scene_plans)

    # This is intentionally after every action and path have already been fixed.
    validity = verify_sampled_plans(
        plans,
        workers=args.verifier_workers,
        chunk_size=args.verifier_chunk_size,
    )
    for plan, result in zip(plans, validity):
        plan["validity"] = result

    metric_rows = []
    for scene_name in SCENE_NAMES:
        for gamma in GAMMAS:
            metric_rows.append(
                aggregate_metrics(scene_name, gamma, episodes, plans, m=args.m)
            )
        metric_rows.append(
            aggregate_metrics(scene_name, None, episodes, plans, m=args.m)
        )
    write_jsonl(outdir / "metrics.jsonl", metric_rows)
    episode_rows = []
    for episode in episodes:
        episode_plans = [
            plan
            for plan in plans
            if plan["scene"] == episode["scene"]
            and plan["gamma"] == episode["gamma"]
            and plan["rollout_index"] == episode["rollout_index"]
        ]
        episode_rows.append(
            {
                "metric_version": METRIC_VERSION,
                "scene": episode["scene"],
                "gamma": episode["gamma"],
                "rollout_index": episode["rollout_index"],
                "status": episode["status"],
                **episode["metrics"],
                "sampled_plan_count": len(episode_plans),
                "sampled_plan_V_safe": _rate_entry(
                    [plan["validity"]["v_safe"] for plan in episode_plans]
                ),
                "sampled_plan_V_full": _rate_entry(
                    [plan["validity"]["v_full"] for plan in episode_plans]
                ),
            }
        )
    write_jsonl(outdir / "episodes.jsonl", episode_rows)
    save_archives(outdir, episodes, plans)
    gallery_outputs, gallery_manifest = render_gallery(
        outdir, episodes, scenes, m=args.m
    )
    metric_outputs = render_metric_figure(outdir, metric_rows)

    provenance = {
        "schema_version": METRIC_VERSION,
        "started_at_utc": started_at,
        "command": sys.argv,
        "source": {
            "evaluator": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(__file__),
            "git": _git_provenance(),
        },
        "device": device_provenance,
        "checkpoint": checkpoint,
        "scenes": snapshots,
        "protocol": {
            "mode": "raw untilted receding-horizon generator",
            "M_per_gamma": args.m,
            "gammas": GAMMAS,
            "T": T,
            "reach": REACH,
            "nfe": args.nfe,
            "temperature": TEMP,
            "verifier_n_theta": N_THETA,
            "verifier_config": asdict(VerifierConfig(angle_samples=N_THETA)),
            "dynamics_config": asdict(DynamicsConfig()),
            "verifier_implementation_fingerprint": verifier_implementation_fingerprint(),
            "verifier_spec_fingerprint_by_scene": {
                name: verifier_spec_fingerprint(scenes[name], scenes[name].goal)
                for name in SCENE_NAMES
            },
            "strict_task_bounds_m": [WORKSPACE_LOW, WORKSPACE_HIGH],
            "verifier_workers": args.verifier_workers,
            "low7_gamma_last": True,
            "common_random_numbers_across_scenes": True,
            "seed_rule": (
                "SHA256(metric_version, raw, gamma, fixed rollout index, control step); "
                "scene deliberately excluded"
            ),
            "verifier_acquisition": False,
            "verifier_fallback": False,
            "verifier_affects_execution": False,
            "sampled_plan_V_safe": "full-H in-bounds and SOCP verifier positive",
            "sampled_plan_V_full": (
                f"V_safe and progress >= min({DELTA_PROG}, 0.5*initial_goal_distance)"
            ),
        },
        "figures": [str(path) for path in (*gallery_outputs, *metric_outputs)],
        "gallery_manifest": str(gallery_manifest),
    }
    write_json(outdir / "provenance.json", provenance)

    inventory = _artifact_inventory(outdir)
    checksum_path = outdir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in inventory.items())
    )
    complete_inventory = {**inventory, "SHA256SUMS": sha256_file(checksum_path)}
    complete = {
        "schema_version": METRIC_VERSION,
        "status": "LOW7_PRETRAINED_EVALUATION_COMPLETE",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
        "M_per_gamma": args.m,
        "episode_count": len(episodes),
        "sampled_plan_count": len(plans),
        "metric_row_count": len(metric_rows),
        "artifact_sha256": complete_inventory,
    }
    write_json(outdir / "EVALUATION_COMPLETE.json", complete)
    return complete


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--M", dest="m", type=int, default=DEFAULT_M)
    parser.add_argument(
        "--nfe",
        type=int,
        default=NFE,
        help="flow ODE function evaluations; 12 matches the canonical pretraining audit",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--verifier-workers", type=int, default=min(16, os.cpu_count() or 1)
    )
    parser.add_argument("--verifier-chunk-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    complete = run(args)
    print(json.dumps(complete, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
