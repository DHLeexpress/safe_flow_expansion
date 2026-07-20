"""Portable evaluation-only M=20 screen for one completed AFE run.

This additive entry point deliberately does not import expansion state into a
trainer or write inside either completed run.  It authenticates the complete
trainer inventory, loads only declared round checkpoints, and evaluates:

* the raw, untilted receding-horizon generator; and
* the run's expert-free verified controller (full verifier,
  absorbing goal-prefix execution semantics, no fallback, NVP termination).

Proposal-noise streams are keyed only by mode, gamma, rollout index, and
control time.  They therefore provide common random numbers across checkpoint
rounds.  M=20 is explicitly a screening evaluation, not a final estimate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

_HERE = Path(__file__).resolve().parent.parent
_REV = _HERE.parent
_WORK = _REV.parent
for _path in (_WORK, _REV, _HERE):
    sys.path.insert(0, str(_path))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import afe_core as AC
import afe_context as CX
import afe_ensemble_core as EC
import afe_rbf_core as RC
from afe2_scene_profiles import (
    assert_scene_snapshot,
    build_scene,
    get_scene_profile,
    scene_snapshot,
)
import grid_expand_afe2 as AFE2
import grid_expand_afe_ensemble as ENS
import grid_expand_afe_rbf as RBF
import grid_feats as GF
import grid_hp_expt as HP
import grid_metrics as GM
import grid_metrics2 as GM2
from di_grid_viz import di_step


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
M = 20
T = 300
REACH = 0.15
VERIFIER_WORKERS = 16
METRIC_VERSION = "afe_m20_portable_v1"
SCREENING_NOTE = (
    "M=20 per gamma is a screening evaluation, not a final estimate or a "
    "probabilistic safety guarantee."
)
GALLERY_INDICES = tuple(range(10))
SUPPORTED_ALGORITHM = "afe_deep_ensemble_adaptive_ess_parallel_v2"
LOW7_ALGORITHM = "afe_low7_deep_ensemble_adaptive_ess_parallel_v1"
SUPPORTED_ALGORITHMS = {SUPPORTED_ALGORITHM, LOW7_ALGORITHM}
Z95 = 1.959963984540054

_WORKER_ENV = None


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path) as stream:
        return json.load(stream)


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    with open(path, "w") as stream:
        json.dump(AFE2._json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def git_state() -> dict[str, Any]:
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=_HERE, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"], cwd=root, text=True
    ).strip()
    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode != 0
    )
    untracked_runtime = [
        item
        for item in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
        ).splitlines()
        if item.endswith((".py", ".sh"))
    ]
    return {
        "root": root,
        "commit": commit,
        "parent": parent,
        "tracked_dirty": tracked_dirty,
        "untracked_runtime_sources": untracked_runtime,
    }


def require_clean_additive_source(expected_base: str) -> dict[str, Any]:
    state = git_state()
    if state["commit"] != expected_base and state["parent"] != expected_base:
        raise RuntimeError(
            "evaluation source is neither the trainer commit nor its direct additive child: "
            f"commit={state['commit']} parent={state['parent']} trainer={expected_base}"
        )
    if state["tracked_dirty"] or state["untracked_runtime_sources"]:
        raise RuntimeError(
            "evaluation requires a committed clean additive source tree; "
            f"untracked runtime sources={state['untracked_runtime_sources']}"
        )
    return state


def expected_inventory(algorithm: str, rounds: int) -> set[str]:
    common = {
        "recipe.json",
        "probe.jsonl",
        "final.pt",
        "dstore.pt",
        *{f"ckpt_{round_i}.pt" for round_i in range(rounds + 1)},
        *{f"viz_db/round{round_i}.pt" for round_i in range(1, rounds + 1)},
    }
    if algorithm in SUPPORTED_ALGORITHMS:
        required = common | {
            "ensemble_calibration.json",
            *{f"ensemble_round{round_i}.pt" for round_i in range(rounds + 1)},
        }
        if algorithm == LOW7_ALGORITHM:
            required.add("viz_db/round0.pt")
        return required
    raise ValueError(f"unsupported completed algorithm: {algorithm}")


def validate_completed_run(
    run_root: str | os.PathLike[str],
    scene_profile: str,
) -> dict[str, Any]:
    """Fail-closed authentication against the trainer-written COMPLETE inventory."""

    root = Path(run_root).resolve()
    recipe_path = root / "recipe.json"
    complete_path = root / "COMPLETE.json"
    probe_path = root / "probe.jsonl"
    for path in (recipe_path, complete_path, probe_path):
        if not path.is_file():
            raise FileNotFoundError(f"completed run artifact is missing: {path}")
    recipe = load_json(recipe_path)
    complete = load_json(complete_path)
    expected_algorithm = str(recipe.get("algorithm"))
    if expected_algorithm not in SUPPORTED_ALGORITHMS:
        raise RuntimeError(f"unsupported recipe algorithm: {expected_algorithm}")
    method = "afe"
    if recipe.get("algorithm") != expected_algorithm:
        raise RuntimeError(
            f"{method} recipe algorithm {recipe.get('algorithm')} != {expected_algorithm}"
        )
    if complete.get("algorithm") != expected_algorithm:
        raise RuntimeError(f"{method} COMPLETE algorithm disagrees with its recipe")
    if recipe.get("arm") != "afe" or recipe.get("single_arm") is not True:
        raise RuntimeError(f"{method} is not a declared single AFE arm")
    completed_round = int(complete.get("completed_round", -1))
    required_round = 100 if expected_algorithm == LOW7_ALGORITHM else 50
    if complete.get("status") != "COMPLETE" or completed_round != required_round:
        raise RuntimeError(
            f"{method} is not a completed {required_round}-round trainer run"
        )
    if recipe.get("source_git_commit") != complete.get("source_git_commit"):
        raise RuntimeError(f"{method} recipe and COMPLETE disagree on source commit")
    if recipe.get("scene", {}).get("profile", {}).get("name") != scene_profile:
        raise RuntimeError(f"{method} recipe has the wrong scene profile")
    if complete.get("scene_sha256") != recipe.get("scene", {}).get("sha256"):
        raise RuntimeError(f"{method} scene hash disagrees between recipe and COMPLETE")
    checks = {
        "checkpoint_sha256": "source_checkpoint_sha256",
        "checkpoint_model_sha256": "source_checkpoint_model_sha256",
        "checkpoint_contract_sha256": "source_checkpoint_contract_sha256",
    }
    for complete_key, recipe_key in checks.items():
        if complete.get(complete_key) != recipe.get(recipe_key):
            raise RuntimeError(f"{method} COMPLETE disagrees on {complete_key}")
    for flag in ("no_curriculum", "no_anchor", "no_prox", "no_fallback"):
        if recipe.get(flag) is not True:
            raise RuntimeError(f"{method} recipe no longer declares {flag}=true")

    inventory = complete.get("artifact_sha256", {})
    required = expected_inventory(expected_algorithm, completed_round)
    if set(inventory) != required:
        missing = sorted(required - set(inventory))
        extra = sorted(set(inventory) - required)
        raise RuntimeError(
            f"{method} trainer inventory mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected_hash in inventory.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"{method} inventoried artifact is missing: {relative}")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"{method} inventoried artifact hash mismatch: {relative}")

    selected = {}
    selected_rounds = (0, completed_round)
    for round_i in selected_rounds:
        relative = f"ckpt_{round_i}.pt"
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"{method} required {relative} is absent; final.pt substitution is forbidden"
            )
        actual = sha256_file(path)
        if actual != inventory.get(relative):
            raise RuntimeError(f"{method} checkpoint {round_i} is not the inventoried artifact")
        selected[int(round_i)] = {"path": str(path), "sha256": actual}

    delivery_candidates = (
        root.parent / "DELIVERY_COMPLETE.json",
        root.parent / "TRAINER_DELIVERY_COMPLETE.json",
    )
    delivery_path = next((path for path in delivery_candidates if path.is_file()), None)
    if delivery_path is None:
        raise FileNotFoundError(
            f"completed run lacks a trainer delivery manifest under {root.parent}"
        )
    delivery = load_json(delivery_path)
    if delivery.get("status") != "AFE_ENSEMBLE_DELIVERY_COMPLETE":
        raise RuntimeError("run parent is not a validated AFE ensemble delivery")
    if delivery.get("run", {}).get("sha256") != sha256_file(complete_path):
        raise RuntimeError("delivery manifest does not authenticate trainer COMPLETE.json")
    if delivery.get("recipe", {}).get("sha256") != sha256_file(recipe_path):
        raise RuntimeError("delivery manifest does not authenticate trainer recipe")
    return {
        "method": method,
        "algorithm": expected_algorithm,
        "run_root": str(root),
        "recipe": recipe,
        "recipe_sha256": sha256_file(recipe_path),
        "complete_sha256": sha256_file(complete_path),
        "delivery_manifest": str(delivery_path),
        "delivery_sha256": sha256_file(delivery_path),
        "probe_sha256": sha256_file(probe_path),
        "scene_sha256": complete["scene_sha256"],
        "source_git_commit": complete["source_git_commit"],
        "source_checkpoint_sha256": complete["checkpoint_sha256"],
        "source_checkpoint_model_sha256": complete["checkpoint_model_sha256"],
        "source_checkpoint_contract_sha256": complete["checkpoint_contract_sha256"],
        "selected_checkpoints": selected,
        "authenticated_artifact_count": len(inventory),
        "completed_round": completed_round,
    }


def paired_seed(
    scene_profile: str,
    mode: str,
    gamma: float,
    rollout_index: int,
    control_t: int | None = None,
) -> int:
    parts = [METRIC_VERSION, scene_profile, mode, f"{float(gamma):.1f}", int(rollout_index)]
    if control_t is not None:
        parts.append(int(control_t))
    raw = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def wilson95(count: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = count / n
    den = 1.0 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2.0 * n)) / den
    half = Z95 * math.sqrt(p * (1.0 - p) / n + Z95 * Z95 / (4.0 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap95(values: list[float], key: Any, n_boot: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(sha256_json(key)[:16], 16) % (2**63 - 1))
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    means = array[indices].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _gpu_record() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU index; "
            f"got {visible!r}"
        )
    physical_index = int(visible)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible CUDA device")
    line = subprocess.check_output(
        [
            "nvidia-smi", "-i", str(physical_index),
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    index, uuid, name = [part.strip() for part in line.split(",", 2)]
    if index != str(physical_index):
        raise RuntimeError(f"nvidia-smi resolved the wrong physical GPU: {line}")
    return {
        "physical_index": physical_index,
        "process_device": "cuda:0",
        "uuid": uuid,
        "name": name,
        "cuda_visible_devices": visible,
    }


def _cfg_from_contract(contract: dict[str, Any], scene_profile: str):
    recipe = contract["recipe"]
    conditioning = recipe.get("conditioning", {})
    schema = str(conditioning.get("schema", CX.LOW5_SCHEMA))
    raw_condition_dim = int(conditioning.get("raw_condition_dim", 5))
    common = dict(
        rounds=int(contract["completed_round"]),
        T=int(recipe["T"]),
        reach=float(recipe["reach"]),
        K=int(recipe["K"]),
        B=int(recipe["B"]),
        beta=float(recipe["beta"]),
        s=float(recipe["s"]),
        nfe=int(recipe["nfe"]),
        temp=1.0,
        gammas=tuple(float(value) for value in recipe["gammas"]),
        arm="afe",
        M_eval=M,
        wall_plugs=int(recipe["scene"]["profile"]["wall_plugs"]),
        start_eps=float(recipe["scene"]["profile"]["start"][0]),
        goal_xy=tuple(float(value) for value in recipe["scene"]["profile"]["goal"]),
        scene_profile=scene_profile,
        seed=int(recipe["seed"]),
        replicas=int(recipe["rollout_replicas"]),
        verifier_workers=VERIFIER_WORKERS,
        replay_window=int(recipe["replay_window"]),
        adaptive_ess_target=float(recipe["adaptive_ess_target"]),
        conditioning_schema=schema,
        raw_condition_dim=raw_condition_dim,
        freeze_visual_encoder=bool(recipe.get("freeze_visual_encoder", False)),
    )
    if tuple(common["gammas"]) != GAMMAS or common["T"] != T or common["reach"] != REACH:
        raise RuntimeError("run recipe disagrees with the screening protocol")
    if common["K"] != 64 or common["B"] != 8 or common["nfe"] != 8:
        raise RuntimeError("run recipe proposal semantics are not canonical")
    if common["replay_window"] != 5 or common["adaptive_ess_target"] != 0.5:
        raise RuntimeError("run recipe is not the canonical W=5 / ESS=0.5 confirmation")
    expected_schema = (
        CX.LOW7_SCHEMA if contract["algorithm"] == LOW7_ALGORITHM else CX.LOW5_SCHEMA
    )
    expected_dim = CX.SCHEMA_DIMS[expected_schema]
    if schema != expected_schema or raw_condition_dim != expected_dim:
        raise RuntimeError("run recipe conditioning schema is inconsistent")
    return ENS.AFEEnsembleConfig(**common)


def _load_policy(contract: dict[str, Any], round_i: int, device: str):
    entry = contract["selected_checkpoints"][int(round_i)]
    if sha256_file(entry["path"]) != entry["sha256"]:
        raise RuntimeError(
            f"{contract['method']} checkpoint {round_i} changed after inventory authentication"
        )
    policy, payload = HP.load_hp(entry["path"], device="cpu")
    if int(payload.get("iter", -1)) != int(round_i):
        raise RuntimeError(f"checkpoint payload iter does not equal round {round_i}")
    embedded = payload.get("recipe", {})
    if embedded.get("algorithm") != contract["algorithm"]:
        raise RuntimeError(f"checkpoint {round_i} embeds the wrong algorithm recipe")
    if payload.get("resumable") is not False:
        raise RuntimeError(f"checkpoint {round_i} violates the evaluation-only contract")
    return policy.to(device).eval(), payload


def _model_state_sha256(policy) -> str:
    from codex_challenging.afe_restart.policy import model_state_hash

    return model_state_hash(policy)


def _restore_store(path: str | os.PathLike[str]) -> AC.DStore:
    state = torch.load(path, map_location="cpu", weights_only=False)
    store = AC.DStore(
        conditioning_schema=state.get("conditioning_schema", CX.LOW5_SCHEMA),
        condition_dim=int(state.get("condition_dim", 5)),
    )
    for key, value in state.items():
        setattr(store, key, value)
    store.pos_ids = [int(index) for index, label in enumerate(store.q_y) if int(label) == 1]
    return store


def _worker_init(scene_profile: str, reach: float, n_theta: int) -> None:
    global _WORKER_ENV
    RC.initialize_verifier_worker(scene_profile, reach, n_theta)
    profile = get_scene_profile(scene_profile)
    _WORKER_ENV = build_scene(profile)
    GM2.GOAL_XY = np.asarray(profile.goal, dtype=float)


def _trajectory_metrics_worker(task: tuple[np.ndarray, float, str, float, float]) -> dict[str, Any]:
    path, gamma, status, dt, reach = task
    if _WORKER_ENV is None:
        raise RuntimeError("metric worker was not initialized")
    env = _WORKER_ENV
    points = np.asarray(path, dtype=np.float64)
    obstacles = env.obstacles.detach().cpu().numpy()
    if obstacles.size:
        clearance = float(
            (
                np.linalg.norm(points[:, None, :] - obstacles[None, :, :2], axis=2)
                - obstacles[None, :, 2]
                - float(env.r_robot)
            ).min()
        )
    else:
        clearance = float("inf")
    collision = bool(clearance < 0.0)
    oob = bool((points < -GM.EPS_TASK).any() or (points > GM.GRID_M + GM.EPS_TASK).any())
    reached = bool(status == "reached")
    valid, breakdown = GM2.traj_breakdown(points, env, float(gamma))
    return {
        "status": str(status),
        "success": reached,
        "collision": collision,
        "oob": oob,
        "cr": bool(collision or oob),
        "nvp": bool(status == "nvp"),
        "timeout": bool(status == "timeout"),
        "v_safe": bool(breakdown["taskspace"] and breakdown["socp"]),
        "v_full": bool(valid),
        "minimum_clearance": clearance,
        "steps": int(len(points) - 1),
        "time_to_goal": float((len(points) - 1) * dt) if reached else None,
    }


@torch.no_grad()
def run_raw_batch(policy, env, cfg, device: str) -> list[dict[str, Any]]:
    """Batched untilted H=10 receding-horizon generator with paired noise."""

    start = env.x0.detach().cpu().numpy().astype(np.float32)
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    robot_radius = float(env.r_robot)
    episodes = []
    for gamma_index, gamma in enumerate(cfg.gammas):
        for rollout_index in range(M):
            episodes.append({
                "episode_id": gamma_index * M + rollout_index,
                "rollout_index": rollout_index,
                "gamma": float(gamma),
                "state": start.copy(),
                "history": [],
                "path": [start[:2].copy()],
                "status": None,
            })
    for control_t in range(cfg.T):
        active = [episode for episode in episodes if episode["status"] is None]
        if not active:
            break
        grids, lows, histories = [], [], []
        noises = []
        for episode in active:
            state = episode["state"]
            record = CX.build_context(
                state,
                goal,
                episode["gamma"],
                episode["history"],
                env,
                cfg.conditioning_schema,
            )
            grids.append(record.grid)
            lows.append(record.low5)
            histories.append(record.hist)
            generator = torch.Generator(device=device)
            generator.manual_seed(paired_seed(
                cfg.scene_profile, "raw", episode["gamma"], episode["rollout_index"], control_t
            ))
            noises.append(torch.randn(policy.d, generator=generator, device=device))
        grid = torch.as_tensor(np.asarray(grids, np.float32), device=device)
        low = torch.as_tensor(np.asarray(lows, np.float32), device=device)
        hist = torch.as_tensor(np.asarray(histories, np.float32), device=device)
        context = policy.ctx_from(grid, low, hist)
        controls = policy.sample(
            len(active),
            context,
            nfe=cfg.nfe,
            temp=cfg.temp,
            initial_noise=torch.stack(noises),
        ).detach().cpu().numpy()
        for episode, window in zip(active, controls):
            action = np.asarray(window[0], dtype=np.float32)
            episode["state"] = di_step(episode["state"], action, dt=env.dt)
            episode["history"].append(action)
            episode["path"].append(episode["state"][:2].copy())
            point = episode["state"][:2]
            if np.linalg.norm(point - goal) < cfg.reach:
                episode["status"] = "reached"
            elif (point < -cfg.taskspace_epsilon).any() or (
                point > GM.GRID_M + cfg.taskspace_epsilon
            ).any():
                episode["status"] = "oob"
            elif obstacles.size and (
                np.linalg.norm(point[None] - obstacles[:, :2], axis=1)
                - obstacles[:, 2]
                - robot_radius
            ).min() < 0.0:
                episode["status"] = "collision"
    output = []
    for episode in episodes:
        if episode["status"] is None:
            episode["status"] = "timeout"
        output.append({
            "episode_id": int(episode["episode_id"]),
            "rollout_index": int(episode["rollout_index"]),
            "gamma": float(episode["gamma"]),
            "path": np.asarray(episode["path"], dtype=np.float32),
            "status": str(episode["status"]),
        })
    return output


def _save_cell(
    outdir: Path,
    mode: str,
    contract: dict[str, Any],
    round_i: int,
    gamma: float,
    episodes: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    records = [episode for episode in episodes if episode["gamma"] == float(gamma)]
    metric_rows = [
        metric for episode, metric in zip(episodes, metrics)
        if episode["gamma"] == float(gamma)
    ]
    if len(records) != M or len(metric_rows) != M:
        raise RuntimeError(
            f"{mode}/{contract['method']}/r{round_i}/g{gamma}: expected M={M}"
        )
    if [record["rollout_index"] for record in records] != list(range(M)):
        raise RuntimeError("rollout records are not in fixed index order")
    cell_dir = outdir / "cells" / mode / contract["method"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    stem = f"r{round_i:03d}_g{gamma:.1f}"
    archive_path = cell_dir / f"{stem}.npz"
    provenance_path = cell_dir / f"{stem}.provenance.json"
    if archive_path.exists() or provenance_path.exists():
        raise FileExistsError(f"stale evaluation cell exists: {stem}")
    paths = np.empty(M, dtype=object)
    for index, record in enumerate(records):
        paths[index] = record["path"]
    if mode == "raw":
        pairing_keys = [
            paired_seed(contract["recipe"]["scene"]["profile"]["name"], mode, gamma, index)
            for index in range(M)
        ]
        pairing_rule = (
            "initial-noise seeds keyed by (metric version, scene, mode, gamma, "
            "rollout index, control time), independent of algorithm and checkpoint round"
        )
    else:
        pairing_keys = [
            AFE2.named_seed(
                int(contract["recipe"]["seed"]),
                "controller_eval",
                str(float(gamma)),
                index,
            )
            for index in range(M)
        ]
        pairing_rule = (
            "frozen controller proposal/acquisition streams keyed by the authenticated "
            "trainer seed, purpose, "
            "gamma-major episode id, and control time; independent of algorithm and "
            "checkpoint round"
        )
    np.savez_compressed(
        archive_path,
        paths=paths,
        status=np.asarray([record["status"] for record in records]),
        rollout_index=np.arange(M, dtype=np.int32),
        pairing_keys=np.asarray(pairing_keys, dtype=np.int64),
    )
    checkpoint = contract["selected_checkpoints"][round_i]
    provenance = {
        "metric_version": METRIC_VERSION,
        "screening_evaluation": True,
        "screening_note": SCREENING_NOTE,
        "mode": mode,
        "method": contract["method"],
        "algorithm": contract["algorithm"],
        "round": int(round_i),
        "gamma": float(gamma),
        "M": M,
        "T": T,
        "reach": REACH,
        "paired_seed_rule": pairing_rule,
        "rollout_pairing_keys": pairing_keys,
        "checkpoint": checkpoint,
        "trainer_complete_sha256": contract["complete_sha256"],
        "trainer_recipe_sha256": contract["recipe_sha256"],
        "trainer_source_git_commit": contract["source_git_commit"],
        "scene_sha256": contract["scene_sha256"],
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "statuses": [record["status"] for record in records],
    }
    write_json(provenance_path, provenance)
    relative_archive = str(archive_path.relative_to(outdir))
    relative_provenance = str(provenance_path.relative_to(outdir))
    artifacts = {
        relative_archive: sha256_file(archive_path),
        relative_provenance: sha256_file(provenance_path),
    }
    return _aggregate_metrics(
        metric_rows,
        mode=mode,
        method=contract["method"],
        algorithm=contract["algorithm"],
        round_i=round_i,
        gamma=gamma,
        scope="gamma",
    ), artifacts


def _rate_entry(count: int, n: int) -> dict[str, Any]:
    return {
        "count": int(count),
        "n": int(n),
        "estimate": float(count / n),
        "wilson95": list(wilson95(count, n)),
    }


def _aggregate_metrics(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    method: str,
    algorithm: str,
    round_i: int,
    gamma: float | None,
    scope: str,
) -> dict[str, Any]:
    n = len(rows)
    if n <= 0:
        raise ValueError("cannot aggregate an empty metric cell")
    mapping = {
        "SR": "success",
        "CR": "cr",
        "NVP": "nvp",
        "timeout": "timeout",
        "V_safe": "v_safe",
        "V_full": "v_full",
        "collision": "collision",
        "OOB": "oob",
    }
    binary = {
        label: _rate_entry(sum(bool(row[key]) for row in rows), n)
        for label, key in mapping.items()
    }
    clearance_values = [float(row["minimum_clearance"]) for row in rows]
    success_times = [
        float(row["time_to_goal"])
        for row in rows if row["time_to_goal"] is not None
    ]
    clearance_ci = bootstrap95(
        clearance_values, [mode, method, round_i, gamma, "minimum_clearance"]
    )
    time_ci = bootstrap95(
        success_times, [mode, method, round_i, gamma, "successful_time_to_goal"]
    )
    return {
        "metric_version": METRIC_VERSION,
        "screening_evaluation": True,
        "screening_note": SCREENING_NOTE,
        "mode": mode,
        "method": method,
        "algorithm": algorithm,
        "round": int(round_i),
        "scope": scope,
        "gamma": None if gamma is None else float(gamma),
        "M_per_gamma": M,
        "n": n,
        "binary": binary,
        "minimum_clearance": {
            "n": n,
            "mean": float(np.mean(clearance_values)),
            "bootstrap95": list(clearance_ci),
            "values": clearance_values,
        },
        "successful_time_to_goal": {
            "n": len(success_times),
            "mean": float(np.mean(success_times)) if success_times else None,
            "bootstrap95": list(time_ci),
            "values": success_times,
        },
        "ci_note": (
            "Wilson 95% intervals for binary counts; deterministic episode bootstrap "
            "95% intervals for continuous means. " + SCREENING_NOTE
        ),
    }


def _load_cell(outdir: Path, mode: str, method: str, round_i: int, gamma: float):
    path = outdir / "cells" / mode / method / f"r{round_i:03d}_g{gamma:.1f}.npz"
    with np.load(path, allow_pickle=True) as archive:
        return list(archive["paths"]), list(archive["status"]), list(archive["rollout_index"])


def _render_curves(
    outdir: Path,
    probe_rows: list[dict[str, Any]],
    headline_round: int,
) -> list[Path]:
    """Render all-round diagnostics only from the authenticated trainer probe."""

    if [int(row["round"]) for row in probe_rows] != list(range(headline_round + 1)):
        raise RuntimeError("probe.jsonl does not contain exactly rounds 0 through headline")
    specs = [
        ("SR", "controller success rate"),
        ("CR", "controller collision/OOB rate"),
        ("NVP", "controller NVP rate"),
        ("TO", "controller timeout rate"),
        ("V_safe", "untilted V_safe"),
        ("V_full", "untilted V_full"),
        ("clear", "controller minimum clearance [m]"),
        ("time", "successful controller time-to-goal [s]"),
    ]
    cmap = plt.get_cmap("plasma")
    gamma_colors = {gamma: cmap(0.08 + 0.84 * index / (len(GAMMAS) - 1)) for index, gamma in enumerate(GAMMAS)}
    rounds = [int(row["round"]) for row in probe_rows]
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), squeeze=False)
    for ax, (key, title) in zip(axes.flat, specs):
        for gamma in GAMMAS:
            values = []
            for row in probe_rows:
                gamma_key = str(gamma)
                if key == "V_safe":
                    value = row["V_safe_gamma"][gamma_key]
                elif key == "V_full":
                    value = row["V_full_gamma"][gamma_key]
                else:
                    value = row["ctrl"][gamma_key][key]
                values.append(value)
            ax.plot(rounds, values, color=gamma_colors[gamma], lw=1.1, alpha=0.68)
        pooled = []
        for row in probe_rows:
            if key in ("V_safe", "V_full"):
                value = row[key]
            elif key in ("SR", "CR", "NVP"):
                value = row["ctrl_pooled"][key]
            else:
                values = [row["ctrl"][str(gamma)][key] for gamma in GAMMAS]
                finite = [float(value) for value in values if value is not None and np.isfinite(value)]
                value = float(np.mean(finite)) if finite else np.nan
            pooled.append(value)
        ax.plot(rounds, pooled, color="black", lw=2.8, label="pooled")
        ax.axvline(headline_round, color="0.35", ls=":", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("trainer round")
        ax.grid(alpha=0.25)
        if key in ("SR", "CR", "NVP", "TO", "V_safe", "V_full"):
            ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=7, loc="best")
    gamma_handles = [
        plt.Line2D([0], [0], color=gamma_colors[gamma], lw=2, label=f"γ={gamma}")
        for gamma in GAMMAS
    ]
    fig.legend(handles=gamma_handles, loc="upper center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        "Authenticated trainer probe curves — per-γ thin lines, pooled black; "
        f"headline checkpoint r{headline_round}\n" + SCREENING_NOTE,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"afe_m20_probe_curves.{suffix}"
        fig.savefig(path, dpi=150)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _draw_scene(ax, profile, env, paths, gamma: float, title: str, statuses: list[str]):
    obstacles = env.obstacles.detach().cpu().numpy()
    for obstacle in obstacles:
        ax.add_patch(plt.Circle(obstacle[:2], obstacle[2], color="#bdbdbd", zorder=1))
    color = plt.get_cmap("plasma")(0.08 + 0.84 * GAMMAS.index(gamma) / (len(GAMMAS) - 1))
    for index in GALLERY_INDICES:
        path = np.asarray(paths[index], dtype=float)
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.25, alpha=0.82, zorder=3)
        if statuses[index] != "reached":
            ax.plot(path[-1, 0], path[-1, 1], "x", color="#cc3311", ms=6, mew=1.5, zorder=5)
    ax.plot(*profile.start, "ks", ms=4, zorder=6)
    ax.plot(*profile.goal, marker="*", color="gold", mec="k", ms=10, zorder=6)
    ax.set_xlim(-0.35, 5.35)
    ax.set_ylim(-0.35, 5.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def _render_gallery(
    outdir: Path,
    profile,
    env,
    headline_round: int,
) -> tuple[list[Path], Path]:
    rows = [
        ("Raw pretrained r0", "raw", 0),
        (f"Raw AFE r{headline_round}", "raw", headline_round),
        ("Verified pretrained r0", "verified", 0),
        (f"Verified AFE r{headline_round}", "verified", headline_round),
    ]
    fig, axes = plt.subplots(len(rows), len(GAMMAS), figsize=(24, 13.5), squeeze=False)
    for row_index, (label, mode, round_i) in enumerate(rows):
        for gamma_index, gamma in enumerate(GAMMAS):
            paths, statuses, indices = _load_cell(outdir, mode, "afe", round_i, gamma)
            if [int(value) for value in indices] != list(range(M)):
                raise RuntimeError("gallery source archive lost fixed rollout indices")
            title = f"γ={gamma}" if row_index == 0 else ""
            _draw_scene(axes[row_index, gamma_index], profile, env, paths, gamma, title, statuses)
            if gamma_index == 0:
                axes[row_index, gamma_index].set_ylabel(label, fontsize=12)
    fig.suptitle(
        f"Fixed non-curated indices {list(GALLERY_INDICES)} in every comparable cell\n"
        + SCREENING_NOTE,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"afe_m20_fixed_index_gallery.{suffix}"
        fig.savefig(path, dpi=150)
        outputs.append(path)
    plt.close(fig)
    manifest_path = outdir / "gallery_indices.json"
    write_json(manifest_path, {
        "rule": "fixed non-curated archive indices; no outcome inspection or selection",
        "indices": list(GALLERY_INDICES),
        "modes": ["raw", "verified"],
        "rows": [
            {"label": label, "mode": mode, "method": "afe", "round": round_i}
            for label, mode, round_i in rows
        ],
        "gammas": list(GAMMAS),
        "M": M,
        "screening_note": SCREENING_NOTE,
    })
    return outputs, manifest_path


def _write_metrics(outdir: Path, rows: list[dict[str, Any]]) -> Path:
    path = outdir / "metrics.jsonl"
    with open(path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(AFE2._json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    return path


def _authenticate_output_cells(outdir: Path, metric_rows: list[dict[str, Any]]) -> None:
    gamma_rows = [row for row in metric_rows if row["scope"] == "gamma"]
    pooled_rows = [row for row in metric_rows if row["scope"] == "pooled"]
    rounds = sorted({int(row["round"]) for row in gamma_rows})
    if len(rounds) != 2 or rounds[0] != 0:
        raise RuntimeError("evaluation must contain only authenticated r0 and headline checkpoints")
    expected_gamma_rows = 2 * len(rounds) * len(GAMMAS)
    expected_pooled_rows = 2 * len(rounds)
    if len(gamma_rows) != expected_gamma_rows or len(pooled_rows) != expected_pooled_rows:
        raise RuntimeError(
            f"metric grid incomplete: gamma={len(gamma_rows)}/{expected_gamma_rows}, "
            f"pooled={len(pooled_rows)}/{expected_pooled_rows}"
        )
    keys = {
        (row["mode"], row["method"], row["round"], row["gamma"])
        for row in gamma_rows
    }
    expected = {
        (mode, "afe", round_i, gamma)
        for mode in ("raw", "verified")
        for round_i in rounds
        for gamma in GAMMAS
    }
    if keys != expected:
        raise RuntimeError("evaluation metric cell key set is incomplete")
    for row in gamma_rows:
        if row["n"] != M:
            raise RuntimeError("per-gamma metric row does not contain M=20 episodes")
        count_total = (
            row["binary"]["SR"]["count"]
            + row["binary"]["NVP"]["count"]
            + row["binary"]["timeout"]["count"]
            + row["binary"]["CR"]["count"]
        )
        if count_total != M:
            raise RuntimeError(
                f"terminal outcome counts do not partition M in {row['mode']}/{row['method']}"
            )
    expected_archives = expected_gamma_rows
    if len(list((outdir / "cells").rglob("*.npz"))) != expected_archives:
        raise RuntimeError("raw cell archive count is incomplete")
    if len(list((outdir / "cells").rglob("*.provenance.json"))) != expected_archives:
        raise RuntimeError("cell provenance count is incomplete")


def _artifact_inventory(outdir: Path) -> dict[str, str]:
    inventory = {}
    for path in sorted(outdir.rglob("*")):
        if not path.is_file() or path.name in {"EVALUATION_COMPLETE.json"}:
            continue
        relative = str(path.relative_to(outdir))
        inventory[relative] = sha256_file(path)
    return inventory


def run_evaluation(args) -> None:
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"evaluation output root must be absent/new: {outdir}")
    contract = validate_completed_run(args.run_root, args.scene_profile)
    source_state = require_clean_additive_source(contract["source_git_commit"])
    gpu = _gpu_record()
    headline_round = int(contract["completed_round"])
    eval_rounds = (0, headline_round)
    policy0, _ = _load_policy(contract, 0, "cpu")
    r0_model_sha256 = _model_state_sha256(policy0)
    del policy0

    profile = get_scene_profile(args.scene_profile)
    env = build_scene(profile)
    snapshot = scene_snapshot(env, profile)
    assert_scene_snapshot(snapshot)
    if snapshot["sha256"] != contract["scene_sha256"]:
        raise RuntimeError("rebuilt evaluation scene does not match completed run")
    GM2.GOAL_XY = np.asarray(profile.goal, dtype=float)
    device = "cuda:0"
    probe_path = Path(args.run_root) / "probe.jsonl"
    probe_rows = [json.loads(line) for line in probe_path.read_text().splitlines() if line]
    outdir.mkdir(parents=True)
    write_json(outdir / "evaluation_contract.json", {
        "metric_version": METRIC_VERSION,
        "screening_evaluation": True,
        "screening_note": SCREENING_NOTE,
        "trainer_source_commit": contract["source_git_commit"],
        "evaluation_source": source_state,
        "gpu": gpu,
        "scene": snapshot,
        "rounds": list(eval_rounds),
        "headline_round": headline_round,
        "gammas": list(GAMMAS),
        "M": M,
        "T": T,
        "reach": REACH,
        "common_random_numbers": (
            "paired by (gamma, rollout_index), with control-time substreams; "
            "never keyed by checkpoint round"
        ),
        "checkpoint_selection_rule": (
            "authenticate and evaluate ckpt_0.pt plus ckpt_<completed_round>.pt only; "
            "no evaluation seed, metric, or outcome selects a checkpoint or recipe"
        ),
        "pretrained_r0_model_state_sha256": r0_model_sha256,
        "modes": {
            "raw": "untilted generator, temp=1, nfe=8, no verifier or fallback",
            "verified": (
                "frozen estimator acquisition, full verifier before execution, "
                "maximum progress among verified plans, no fallback, NVP termination"
            ),
        },
        "completed_run": contract,
    })

    cfg = _cfg_from_contract(contract, args.scene_profile)
    metric_rows: list[dict[str, Any]] = []

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=VERIFIER_WORKERS,
        mp_context=context,
        initializer=_worker_init,
        initargs=(args.scene_profile, REACH, 180),
    ) as executor:
        for round_i in eval_rounds:
            cell_started = time.perf_counter()
            policy, _ = _load_policy(contract, round_i, device)
            raw_episodes = run_raw_batch(policy, env, cfg, device)
            raw_tasks = [
                (episode["path"], episode["gamma"], episode["status"], float(env.dt), REACH)
                for episode in raw_episodes
            ]
            raw_metrics = list(executor.map(_trajectory_metrics_worker, raw_tasks, chunksize=2))

            estimator_path = Path(args.run_root) / f"ensemble_round{round_i}.pt"
            estimator_payload = torch.load(estimator_path, map_location="cpu", weights_only=False)
            if int(estimator_payload.get("round", -1)) != round_i:
                raise RuntimeError(f"ensemble estimator round mismatch at {round_i}")
            if estimator_payload.get("source_git_commit") != contract["source_git_commit"]:
                raise RuntimeError("ensemble estimator has the wrong trainer source commit")
            estimator = EC.DeepEnsembleSigma.from_state_dict(
                estimator_payload["estimator"], device=device
            )
            acquisition_mode = "uniform" if round_i == 0 else "sequential"

            scratch = AC.DStore(
                conditioning_schema=cfg.conditioning_schema,
                condition_dim=cfg.raw_condition_dim,
            )
            verified_episodes, _ = RBF.run_parallel_episodes(
                policy, estimator, env, cfg, scratch, round_i, M, device, executor,
                collect=False, viz=None, purpose="controller_eval",
                acquisition_mode=acquisition_mode,
            )
            for episode in verified_episodes:
                episode["rollout_index"] = int(episode["replica"])
            verified_tasks = [
                (episode["path"], episode["gamma"], episode["status"], float(env.dt), REACH)
                for episode in verified_episodes
            ]
            verified_metrics = list(
                executor.map(_trajectory_metrics_worker, verified_tasks, chunksize=2)
            )

            for mode, episodes, metrics in (
                ("raw", raw_episodes, raw_metrics),
                ("verified", verified_episodes, verified_metrics),
            ):
                pooled_metrics = []
                for gamma in GAMMAS:
                    row, _ = _save_cell(
                        outdir, mode, contract, round_i, gamma, episodes, metrics
                    )
                    metric_rows.append(row)
                    pooled_metrics.extend([
                        metric for episode, metric in zip(episodes, metrics)
                        if episode["gamma"] == gamma
                    ])
                metric_rows.append(_aggregate_metrics(
                    pooled_metrics,
                    mode=mode,
                    method="afe",
                    algorithm=contract["algorithm"],
                    round_i=round_i,
                    gamma=None,
                    scope="pooled",
                ))
            del policy, estimator
            torch.cuda.empty_cache()
            print(
                f"[eval afe r{round_i:03d}] raw+verified complete "
                f"in {time.perf_counter() - cell_started:.1f}s",
                flush=True,
            )

    _authenticate_output_cells(outdir, metric_rows)
    metrics_path = _write_metrics(outdir, metric_rows)
    curve_paths = _render_curves(outdir, probe_rows, headline_round)
    gallery_paths, gallery_manifest = _render_gallery(
        outdir, profile, env, headline_round
    )
    elapsed = time.perf_counter() - started
    finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary_path = outdir / "evaluation_summary.json"
    write_json(summary_path, {
        "status": "AFE_M20_SCREEN_COMPLETE",
        "metric_version": METRIC_VERSION,
        "screening_evaluation": True,
        "screening_note": SCREENING_NOTE,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": elapsed,
        "rounds": list(eval_rounds),
        "headline_round": headline_round,
        "gammas": list(GAMMAS),
        "M": M,
        "cell_count": 2 * len(eval_rounds) * len(GAMMAS),
        "metric_row_count": len(metric_rows),
        "gpu": gpu,
        "source": source_state,
        "run": contract,
        "outputs": {
            "metrics": str(metrics_path),
            "curves": [str(path) for path in curve_paths],
            "gallery": [str(path) for path in gallery_paths],
            "gallery_indices": str(gallery_manifest),
        },
    })
    inventory = _artifact_inventory(outdir)
    complete_path = outdir / "EVALUATION_COMPLETE.json"
    write_json(complete_path, {
        "status": "AFE_M20_EVALUATION_DELIVERY_COMPLETE",
        "metric_version": METRIC_VERSION,
        "screening_evaluation": True,
        "screening_note": SCREENING_NOTE,
        "trainer_source_commit": contract["source_git_commit"],
        "evaluation_source_commit": source_state["commit"],
        "scene_sha256": snapshot["sha256"],
        "elapsed_seconds": elapsed,
        "artifact_sha256": inventory,
    })
    print(f"AFE M20 SCREEN COMPLETE: {outdir}", flush=True)


def validate_output(outdir: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(outdir).resolve()
    complete_path = root / "EVALUATION_COMPLETE.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"evaluation completion manifest is missing: {complete_path}")
    complete = load_json(complete_path)
    if complete.get("status") != "AFE_M20_EVALUATION_DELIVERY_COMPLETE":
        raise RuntimeError("evaluation completion status is invalid")
    inventory = complete.get("artifact_sha256", {})
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "EVALUATION_COMPLETE.json"
    }
    if set(inventory) != actual_files:
        raise RuntimeError("evaluation delivery inventory does not match output files")
    for relative, expected in inventory.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"evaluation output hash mismatch: {relative}")
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    _authenticate_output_cells(root, rows)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root")
    parser.add_argument("--scene-profile")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_output(args.outdir)
        print(f"AFE M20 OUTPUT VALID: {Path(args.outdir).resolve()}")
        return
    if not args.run_root or not args.scene_profile:
        parser.error("--run-root and --scene-profile are required unless --validate-only is used")
    run_evaluation(args)


if __name__ == "__main__":
    main()
