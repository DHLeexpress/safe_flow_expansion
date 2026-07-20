"""Raw-only checkpoint sweep for one completed AFE-RBF arm.

This evaluator is deliberately additive and evaluation-only.  It authenticates
the completed trainer inventory and uses one common raw proposal-noise bank.  It
never loads or applies the RBF/GP acquisition rule or verifier controller.  The
default profile preserves the canonical M=50/every-tenth-checkpoint protocol;
the explicit V2 smoke profile evaluates M=10 at every stored checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

_HERE = Path(__file__).resolve().parent.parent
_REV = _HERE.parent
_WORK = _REV.parent
for _path in (_WORK, _REV, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import afe_context as CX
import afe_m20_eval as M20
import afe_route_metrics as RM
from afe2_scene_profiles import (
    SCENE_PROFILES,
    assert_scene_snapshot,
    build_scene,
    get_scene_profile,
    scene_snapshot,
)
from codex_challenging.afe_restart.policy import model_state_hash
from di_grid_viz import di_step
import grid_expand_afe2 as AFE2
import grid_hp_expt as HP
import grid_metrics as GM


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
M = 50
T = 300
REACH = 0.15
NFE = 8
TEMP = 1.0
GALLERY_INDICES = tuple(range(10))
STATE_DOT_STRIDE = 4
METRIC_VERSION = "afe_rbf_raw_sweep_m50_v1"
REPORT_CAPTION = (
    "stored checkpoints re-evaluated on the same raw M=50/gamma seed bank"
)
BEST_SELECTION_RULE = (
    "post-hoc lexicographic ranking on pooled true evaluation: maximize SR, "
    "minimize CR, minimize timeout, maximize mean minimum clearance, then prefer "
    "the earlier round"
)
Z95 = 1.959963984540054
SUPPORTED_ALGORITHMS = {
    "afe_rbf_previous_round_parallel_v1",
    "afe_rbf_batch_conditional_parallel_v2",
    "afe_rbf_sequential_operational_parallel_v3",
    "afe_rbf_adaptive_ess_parallel_v4",
    "afe_uniform_parallel_v1",
    "afe_rbf_low7_signed_execution_sweep_v1",
    "afe_rbf_low7_v2_smoke_v1",
    "afe_rbf_low7_v2_sample_complete_smoke_v2",
    "afe_rbf_low7_v2_lineage_mass_smoke_v1",
    "afe_rbf_low7_v3_optimizer_demo_support_v1",
}


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    m: int
    checkpoint_stride: int
    metric_version: str
    caption: str
    filename_tag: str
    summary_status: str
    delivery_status: str

    @property
    def gallery_indices(self) -> tuple[int, ...]:
        return tuple(range(min(10, self.m)))


DEFAULT_EVAL_PROFILE = "canonical_m50_every10"
V2_SMOKE_EVAL_PROFILE = "v2_smoke_m10_every_round"
EVALUATION_PROFILES = {
    DEFAULT_EVAL_PROFILE: EvaluationProfile(
        name=DEFAULT_EVAL_PROFILE,
        m=M,
        checkpoint_stride=10,
        metric_version=METRIC_VERSION,
        caption=REPORT_CAPTION,
        filename_tag="raw_m50",
        summary_status="AFE_RBF_RAW_M50_SWEEP_COMPLETE",
        delivery_status="AFE_RBF_RAW_M50_EVALUATION_DELIVERY_COMPLETE",
    ),
    V2_SMOKE_EVAL_PROFILE: EvaluationProfile(
        name=V2_SMOKE_EVAL_PROFILE,
        m=10,
        checkpoint_stride=1,
        metric_version="afe_rbf_raw_sweep_m10_every_round_v1",
        caption=(
            "stored checkpoints re-evaluated on the same raw M=10/gamma seed bank"
        ),
        filename_tag="raw_m10",
        summary_status="AFE_RBF_RAW_M10_EVERY_ROUND_SWEEP_COMPLETE",
        delivery_status="AFE_RBF_RAW_M10_EVERY_ROUND_EVALUATION_DELIVERY_COMPLETE",
    ),
}


def resolve_evaluation_profile(
    profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> EvaluationProfile:
    if isinstance(profile, EvaluationProfile):
        return profile
    try:
        return EVALUATION_PROFILES[str(profile)]
    except KeyError as exc:
        raise ValueError(f"unknown raw evaluation profile: {profile}") from exc


@dataclass(frozen=True)
class RawEvalConfig:
    scene_profile: str
    conditioning_schema: str
    gammas: tuple[float, ...] = GAMMAS
    T: int = T
    reach: float = REACH
    nfe: int = NFE
    temp: float = TEMP
    taskspace_epsilon: float = float(GM.EPS_TASK)


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
        json.dump(
            AFE2._json_safe(value),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
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
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode
        != 0
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
            "evaluation source is neither the trainer commit nor its direct additive "
            f"child: commit={state['commit']} parent={state['parent']} trainer={expected_base}"
        )
    if state["tracked_dirty"] or state["untracked_runtime_sources"]:
        raise RuntimeError(
            "evaluation requires a committed clean source tree; "
            f"untracked runtime sources={state['untracked_runtime_sources']}"
        )
    return state


def evaluation_rounds(
    final_round: int,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> tuple[int, ...]:
    final_round = int(final_round)
    if final_round < 1:
        raise ValueError("a completed RBF sweep must contain at least one expansion round")
    profile = resolve_evaluation_profile(eval_profile)
    selected = {
        0,
        final_round,
        *range(profile.checkpoint_stride, final_round + 1, profile.checkpoint_stride),
    }
    return tuple(sorted(selected))


def expected_inventory(
    rounds: int,
    artifact_profile: str = "full",
    compact_checkpoint_every: int = 10,
) -> set[str]:
    if artifact_profile not in {"full", "sweep_compact"}:
        raise ValueError(f"unknown trainer artifact profile: {artifact_profile}")
    compact_checkpoint_every = int(compact_checkpoint_every)
    if compact_checkpoint_every < 1:
        raise ValueError("compact checkpoint interval must be positive")
    viz_rounds = (
        range(1, int(rounds) + 1)
        if artifact_profile == "full"
        else (
            round_i for round_i in range(1, int(rounds) + 1)
            if round_i <= 10 or round_i % 10 == 0
        )
    )
    return {
        "recipe.json",
        "rbf_calibration.json",
        "probe.jsonl",
        "final.pt",
        *({"dstore.pt"} if artifact_profile == "full" else set()),
        *({"nvp_negative_archive.npz"} if artifact_profile == "sweep_compact" else set()),
        *{
            f"ckpt_{round_i}.pt"
            for round_i in (
                range(int(rounds) + 1)
                if artifact_profile == "full"
                else sorted(
                    {
                        0,
                        int(rounds),
                        *range(
                            compact_checkpoint_every,
                            int(rounds) + 1,
                            compact_checkpoint_every,
                        ),
                    }
                )
            )
        },
        *{f"viz_db/round{round_i}.pt" for round_i in viz_rounds},
    }


def _validate_recipe_protocol(recipe: dict[str, Any]) -> None:
    if int(recipe.get("T", -1)) != T:
        raise RuntimeError(f"trainer T={recipe.get('T')} disagrees with canonical T={T}")
    if int(recipe.get("nfe", -1)) != NFE:
        raise RuntimeError(
            f"trainer nfe={recipe.get('nfe')} disagrees with canonical nfe={NFE}"
        )
    if not math.isclose(float(recipe.get("reach", float("nan"))), REACH):
        raise RuntimeError("trainer reach disagrees with the raw evaluation contract")
    gammas = tuple(float(value) for value in recipe.get("gammas", ()))
    if gammas != GAMMAS:
        raise RuntimeError("trainer gamma grid disagrees with the raw evaluation contract")


def validate_completed_run(
    run_root: str | os.PathLike[str],
    scene_profile: str,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> dict[str, Any]:
    """Authenticate one completed variable-length AFE-RBF trainer run."""

    root = Path(run_root).resolve()
    recipe_path = root / "recipe.json"
    complete_path = root / "COMPLETE.json"
    probe_path = root / "probe.jsonl"
    for path in (recipe_path, complete_path, probe_path):
        if not path.is_file():
            raise FileNotFoundError(f"completed RBF artifact is missing: {path}")

    recipe = load_json(recipe_path)
    complete = load_json(complete_path)
    algorithm = str(recipe.get("algorithm"))
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise RuntimeError(f"unsupported completed RBF algorithm: {algorithm}")
    if complete.get("algorithm") != algorithm:
        raise RuntimeError("RBF recipe and COMPLETE.json disagree on algorithm")
    if recipe.get("arm") != "afe" or recipe.get("single_arm") is not True:
        raise RuntimeError("run is not a declared single AFE-RBF arm")
    completed_round = int(complete.get("completed_round", -1))
    if complete.get("status") != "COMPLETE" or completed_round < 1:
        raise RuntimeError("RBF trainer run is not complete")
    if int(recipe.get("rounds", -1)) != completed_round:
        raise RuntimeError("RBF recipe and completion disagree on final round")
    if recipe.get("scene", {}).get("profile", {}).get("name") != scene_profile:
        raise RuntimeError("RBF recipe scene profile does not match evaluation scene")
    if complete.get("scene_sha256") != recipe.get("scene", {}).get("sha256"):
        raise RuntimeError("RBF recipe and completion disagree on scene hash")
    checks = {
        "checkpoint_sha256": "source_checkpoint_sha256",
        "checkpoint_model_sha256": "source_checkpoint_model_sha256",
        "checkpoint_contract_sha256": "source_checkpoint_contract_sha256",
        "source_git_commit": "source_git_commit",
    }
    for complete_key, recipe_key in checks.items():
        if complete.get(complete_key) != recipe.get(recipe_key):
            raise RuntimeError(
                f"RBF recipe and completion disagree on {complete_key}"
            )
    for flag in ("no_curriculum", "no_anchor", "no_prox", "no_fallback"):
        if recipe.get(flag) is not True:
            raise RuntimeError(f"RBF recipe no longer declares {flag}=true")
    _validate_recipe_protocol(recipe)

    inventory = complete.get("artifact_sha256", {})
    artifact_profile = str(recipe.get("artifact_profile", "full"))
    compact_checkpoint_every = int(recipe.get("compact_checkpoint_every", 10))
    required = expected_inventory(
        completed_round, artifact_profile, compact_checkpoint_every
    )
    if set(inventory) != required:
        missing = sorted(required - set(inventory))
        extra = sorted(set(inventory) - required)
        raise RuntimeError(
            f"RBF completion inventory mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected_hash in inventory.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"inventoried RBF artifact is missing: {relative}")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"inventoried RBF artifact hash mismatch: {relative}")

    profile = resolve_evaluation_profile(eval_profile)
    selected_checkpoints = {}
    for round_i in evaluation_rounds(completed_round, profile):
        relative = f"ckpt_{round_i}.pt"
        selected_checkpoints[round_i] = {
            "path": str(root / relative),
            "sha256": inventory[relative],
        }
    return {
        "kind": "single_afe_rbf_raw_sweep",
        "method": "afe_rbf",
        "algorithm": algorithm,
        "run_root": str(root),
        "recipe": recipe,
        "recipe_sha256": sha256_file(recipe_path),
        "complete_sha256": sha256_file(complete_path),
        "probe_sha256": sha256_file(probe_path),
        "scene_sha256": complete["scene_sha256"],
        "source_git_commit": complete["source_git_commit"],
        "source_checkpoint_sha256": complete["checkpoint_sha256"],
        "source_checkpoint_model_sha256": complete["checkpoint_model_sha256"],
        "source_checkpoint_contract_sha256": complete[
            "checkpoint_contract_sha256"
        ],
        "selected_checkpoints": selected_checkpoints,
        "final_checkpoint_alias": {
            "path": str(root / "final.pt"),
            "sha256": inventory["final.pt"],
        },
        "authenticated_artifact_count": len(inventory),
        "completed_round": completed_round,
        "evaluation_profile": profile.name,
        "evaluation_rounds": list(evaluation_rounds(completed_round, profile)),
    }


def paired_seed(
    scene_profile: str,
    gamma: float,
    rollout_index: int,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> int:
    """Stable pairing identifier; intentionally has no arm or round argument."""

    resolve_evaluation_profile(eval_profile)
    raw = "|".join(
        (
            METRIC_VERSION,
            str(scene_profile),
            f"{float(gamma):.1f}",
            str(int(rollout_index)),
        )
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def noise_bank_seed(
    scene_profile: str,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> int:
    resolve_evaluation_profile(eval_profile)
    raw = f"{METRIC_VERSION}|{scene_profile}|raw-temp1-noise-bank".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def build_noise_bank(
    scene_profile: str,
    policy_dim: int,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> tuple[np.ndarray, dict[str, Any]]:
    profile = resolve_evaluation_profile(eval_profile)
    seed = noise_bank_seed(scene_profile, profile)
    generator = np.random.default_rng(seed)
    canonical_bank = generator.standard_normal(
        (len(GAMMAS), M, T, int(policy_dim)), dtype=np.float32
    )
    bank = canonical_bank[:, :profile.m].copy()
    metadata = {
        "seed": seed,
        "shape": list(bank.shape),
        "dtype": str(bank.dtype),
        "sha256": hashlib.sha256(bank.tobytes(order="C")).hexdigest(),
        "indexing": "[gamma_index, rollout_index, control_time, latent_dimension]",
        "independence": "scene-keyed; independent of RBF arm and checkpoint round",
        "evaluation_profile": profile.name,
        "cross_profile_pairing": (
            f"first {profile.m} rollouts per gamma from the canonical M={M} bank"
        ),
    }
    return bank, metadata


def wilson95(count: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = count / n
    den = 1.0 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2.0 * n)) / den
    half = Z95 * math.sqrt(p * (1.0 - p) / n + Z95 * Z95 / (4.0 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap95(
    values: list[float], key: Any, n_boot: int = 2000
) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    array = np.asarray(values, dtype=np.float64)
    seed = int(sha256_json(key)[:16], 16) % (2**63 - 1)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(n_boot, len(array)))
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
            "nvidia-smi",
            "-i",
            str(physical_index),
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


def _load_policy(
    contract: dict[str, Any], round_i: int, device: str
) -> tuple[Any, dict[str, Any], str, CX.ConditioningContract]:
    entry = contract["selected_checkpoints"][int(round_i)]
    if sha256_file(entry["path"]) != entry["sha256"]:
        raise RuntimeError(f"checkpoint r{round_i} changed after authentication")
    policy, payload = HP.load_hp(entry["path"], device="cpu")
    if int(payload.get("iter", -1)) != int(round_i):
        raise RuntimeError(f"checkpoint payload iter does not equal round {round_i}")
    embedded = payload.get("recipe", {})
    if embedded.get("algorithm") != contract["algorithm"]:
        raise RuntimeError(f"checkpoint r{round_i} embeds the wrong algorithm")
    if payload.get("resumable") is not False:
        raise RuntimeError(f"checkpoint r{round_i} violates the evaluation-only contract")
    conditioning = CX.policy_contract(policy)
    model_sha = model_state_hash(policy)
    return policy.to(device).eval(), payload, model_sha, conditioning


def _authenticate_final_alias(contract: dict[str, Any]) -> str:
    final_entry = contract["final_checkpoint_alias"]
    if sha256_file(final_entry["path"]) != final_entry["sha256"]:
        raise RuntimeError("final.pt changed after trainer inventory authentication")
    final_policy, final_payload = HP.load_hp(final_entry["path"], device="cpu")
    final_round = int(contract["completed_round"])
    if int(final_payload.get("iter", -1)) != final_round:
        raise RuntimeError("final.pt embeds the wrong final iteration")
    final_sha = model_state_hash(final_policy)
    ckpt_policy, _, ckpt_sha, _ = _load_policy(contract, final_round, "cpu")
    del final_policy, ckpt_policy
    if final_sha != ckpt_sha:
        raise RuntimeError("final.pt and ckpt_<final>.pt have different model states")
    return final_sha


@torch.no_grad()
def run_raw_batch(
    policy: Any,
    env: Any,
    cfg: RawEvalConfig,
    device: str,
    noise_bank: np.ndarray,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> list[dict[str, Any]]:
    """Batched raw H=10 receding-horizon policy; no GP, tilt, or verifier."""

    profile = resolve_evaluation_profile(eval_profile)
    expected_shape = (len(GAMMAS), profile.m, T, int(policy.d))
    if noise_bank.shape != expected_shape or noise_bank.dtype != np.float32:
        raise RuntimeError(
            f"raw noise bank {noise_bank.shape}/{noise_bank.dtype} != {expected_shape}/float32"
        )
    start = env.x0.detach().cpu().numpy().astype(np.float32)
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    robot_radius = float(env.r_robot)
    episodes: list[dict[str, Any]] = []
    for gamma_index, gamma in enumerate(cfg.gammas):
        for rollout_index in range(profile.m):
            episodes.append(
                {
                    "episode_id": gamma_index * profile.m + rollout_index,
                    "gamma_index": gamma_index,
                    "rollout_index": rollout_index,
                    "gamma": float(gamma),
                    "state": start.copy(),
                    "history": [],
                    "path": [start[:2].copy()],
                    "status": None,
                }
            )

    for control_t in range(cfg.T):
        active = [episode for episode in episodes if episode["status"] is None]
        if not active:
            break
        grids, conditions, histories, noises = [], [], [], []
        for episode in active:
            record = CX.build_context(
                episode["state"],
                goal,
                episode["gamma"],
                episode["history"],
                env,
                cfg.conditioning_schema,
            )
            grids.append(record.grid)
            conditions.append(record.low5)
            histories.append(record.hist)
            noises.append(
                noise_bank[
                    episode["gamma_index"], episode["rollout_index"], control_t
                ]
            )
        grid = torch.as_tensor(np.asarray(grids, np.float32), device=device)
        condition = torch.as_tensor(
            np.asarray(conditions, np.float32), device=device
        )
        history = torch.as_tensor(np.asarray(histories, np.float32), device=device)
        context = policy.ctx_from(grid, condition, history)
        controls = policy.sample(
            len(active),
            context,
            nfe=cfg.nfe,
            temp=cfg.temp,
            initial_noise=torch.as_tensor(np.asarray(noises), device=device),
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
        status = "timeout" if episode["status"] is None else str(episode["status"])
        output.append(
            {
                "episode_id": int(episode["episode_id"]),
                "rollout_index": int(episode["rollout_index"]),
                "gamma": float(episode["gamma"]),
                "path": np.asarray(episode["path"], dtype=np.float32),
                "status": status,
            }
        )
    return output


def normalize_trajectory_metrics(
    episode: dict[str, Any], worker_row: dict[str, Any], dt: float
) -> dict[str, Any]:
    """Add disjoint raw outcomes to the full trajectory-validity worker result."""

    row = dict(worker_row)
    collision = bool(row["collision"] or episode["status"] == "collision")
    oob = bool(row["oob"] or episode["status"] == "oob")
    cr = bool(collision or oob)
    success = bool(episode["status"] == "reached" and not cr)
    timeout = bool(episode["status"] == "timeout" and not cr)
    if sum((success, cr, timeout)) != 1:
        raise RuntimeError(
            "raw terminal outcomes do not partition into success/CR/timeout: "
            f"status={episode['status']} collision={collision} oob={oob}"
        )
    row.pop("nvp", None)
    row.update(
        {
            "status": str(episode["status"]),
            "outcome": "SR" if success else ("CR" if cr else "timeout"),
            "success": success,
            "cr": cr,
            "timeout": timeout,
            "collision": collision,
            "oob": oob,
            "time_to_goal": float(row["steps"] * dt) if success else None,
            "route_mode_early": int(episode["route_mode_early"]),
            "route_mode_closest": int(episode["route_mode_closest"]),
        }
    )
    return row


def _rate_entry(count: int, n: int) -> dict[str, Any]:
    return {
        "count": int(count),
        "n": int(n),
        "estimate": float(count / n),
        "wilson95": list(wilson95(count, n)),
    }


def aggregate_metrics(
    rows: list[dict[str, Any]],
    *,
    round_i: int,
    gamma: float | None,
    scope: str,
    scene_profile: str,
    algorithm: str,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> dict[str, Any]:
    profile = resolve_evaluation_profile(eval_profile)
    n = len(rows)
    if n <= 0:
        raise ValueError("cannot aggregate an empty raw evaluation cell")
    binary = {
        "SR": _rate_entry(sum(bool(row["success"]) for row in rows), n),
        "CR": _rate_entry(sum(bool(row["cr"]) for row in rows), n),
        "timeout": _rate_entry(sum(bool(row["timeout"]) for row in rows), n),
        "V_safe": _rate_entry(sum(bool(row["v_safe"]) for row in rows), n),
        "V_full": _rate_entry(sum(bool(row["v_full"]) for row in rows), n),
    }
    if sum(binary[key]["count"] for key in ("SR", "CR", "timeout")) != n:
        raise RuntimeError("SR/CR/timeout counts do not partition the raw cell")
    clearances = [float(row["minimum_clearance"]) for row in rows]
    success_times = [
        float(row["time_to_goal"])
        for row in rows
        if row["time_to_goal"] is not None
    ]
    bootstrap_key = [profile.metric_version, scene_profile, scope, gamma, n]
    clearance_ci = bootstrap95(clearances, [*bootstrap_key, "clearance"])
    time_ci = bootstrap95(success_times, [*bootstrap_key, "success_time"])
    route_modes = {
        "early_first_10_steps": RM.summarize_modes([
            row["route_mode_early"] for row in rows
        ]),
        "closest_obstacle_approach": RM.summarize_modes([
            row["route_mode_closest"] for row in rows
        ]),
        "closest_obstacle_approach_success_only": RM.summarize_modes([
            row["route_mode_closest"] for row in rows if row["success"]
        ]),
    }
    return {
        "metric_version": profile.metric_version,
        "caption": profile.caption,
        "evaluation_profile": profile.name,
        "mode": "raw",
        "method": "afe_rbf",
        "algorithm": algorithm,
        "round": int(round_i),
        "scope": scope,
        "gamma": None if gamma is None else float(gamma),
        "M_per_gamma": profile.m,
        "n": n,
        "binary": binary,
        "minimum_clearance": {
            "n": n,
            "mean": float(np.mean(clearances)),
            "bootstrap95": list(clearance_ci),
            "values": clearances,
        },
        "successful_time_to_goal": {
            "n": len(success_times),
            "mean": float(np.mean(success_times)) if success_times else None,
            "bootstrap95": list(time_ci),
            "values": success_times,
        },
        "route_modes": route_modes,
        "route_mode_intervention": False,
        "ci_note": (
            "Wilson 95% intervals for SR/CR/timeout/V_safe/V_full; deterministic episode "
            "bootstrap 95% intervals for continuous means"
        ),
    }


def _save_cell(
    outdir: Path,
    contract: dict[str, Any],
    round_i: int,
    gamma: float,
    episodes: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    model_state_sha256: str,
    noise_metadata: dict[str, Any],
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> tuple[dict[str, Any], dict[str, str]]:
    profile = resolve_evaluation_profile(eval_profile)
    pairs = [
        (episode, metric)
        for episode, metric in zip(episodes, metrics)
        if episode["gamma"] == float(gamma)
    ]
    if len(pairs) != profile.m:
        raise RuntimeError(f"raw/r{round_i}/g{gamma}: expected M={profile.m}")
    records = [pair[0] for pair in pairs]
    metric_rows = [pair[1] for pair in pairs]
    if [record["rollout_index"] for record in records] != list(range(profile.m)):
        raise RuntimeError("raw rollout records are not in fixed index order")

    cell_dir = outdir / "cells" / "raw" / "afe_rbf"
    cell_dir.mkdir(parents=True, exist_ok=True)
    stem = f"r{round_i:03d}_g{gamma:.1f}"
    archive_path = cell_dir / f"{stem}.npz"
    provenance_path = cell_dir / f"{stem}.provenance.json"
    if archive_path.exists() or provenance_path.exists():
        raise FileExistsError(f"stale raw evaluation cell exists: {stem}")
    paths = np.empty(profile.m, dtype=object)
    for index, record in enumerate(records):
        paths[index] = record["path"]
    pairing_keys = [
        paired_seed(
            contract["recipe"]["scene"]["profile"]["name"], gamma, index, profile
        )
        for index in range(profile.m)
    ]
    np.savez_compressed(
        archive_path,
        paths=paths,
        status=np.asarray([record["status"] for record in records]),
        outcome=np.asarray([metric["outcome"] for metric in metric_rows]),
        success=np.asarray([metric["success"] for metric in metric_rows], dtype=np.bool_),
        cr=np.asarray([metric["cr"] for metric in metric_rows], dtype=np.bool_),
        timeout=np.asarray([metric["timeout"] for metric in metric_rows], dtype=np.bool_),
        v_safe=np.asarray([metric["v_safe"] for metric in metric_rows], dtype=np.bool_),
        v_full=np.asarray([metric["v_full"] for metric in metric_rows], dtype=np.bool_),
        minimum_clearance=np.asarray(
            [metric["minimum_clearance"] for metric in metric_rows], dtype=np.float64
        ),
        time_to_goal=np.asarray(
            [
                np.nan if metric["time_to_goal"] is None else metric["time_to_goal"]
                for metric in metric_rows
            ],
            dtype=np.float64,
        ),
        rollout_index=np.arange(profile.m, dtype=np.int32),
        pairing_keys=np.asarray(pairing_keys, dtype=np.int64),
    )
    checkpoint = contract["selected_checkpoints"][round_i]
    provenance = {
        "metric_version": profile.metric_version,
        "caption": profile.caption,
        "evaluation_profile": profile.name,
        "mode": "raw",
        "algorithm": contract["algorithm"],
        "round": int(round_i),
        "gamma": float(gamma),
        "M": profile.m,
        "T": T,
        "reach": REACH,
        "nfe": NFE,
        "temp": TEMP,
        "controller": "raw untilted policy; no GP/RBF acquisition or verifier selection",
        "paired_seed_rule": (
            "one scene-keyed raw noise bank indexed by gamma, rollout, control time, "
            "and latent dimension; independent of arm and checkpoint round"
        ),
        "noise_bank": noise_metadata,
        "rollout_pairing_keys": pairing_keys,
        "checkpoint": checkpoint,
        "checkpoint_model_state_sha256": model_state_sha256,
        "trainer_complete_sha256": contract["complete_sha256"],
        "trainer_recipe_sha256": contract["recipe_sha256"],
        "trainer_source_git_commit": contract["source_git_commit"],
        "scene_sha256": contract["scene_sha256"],
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
    }
    write_json(provenance_path, provenance)
    artifacts = {
        str(archive_path.relative_to(outdir)): sha256_file(archive_path),
        str(provenance_path.relative_to(outdir)): sha256_file(provenance_path),
    }
    return (
        aggregate_metrics(
            metric_rows,
            round_i=round_i,
            gamma=gamma,
            scope="gamma",
            scene_profile=contract["recipe"]["scene"]["profile"]["name"],
            algorithm=contract["algorithm"],
            eval_profile=profile,
        ),
        artifacts,
    )


def _write_metrics(outdir: Path, rows: list[dict[str, Any]]) -> Path:
    path = outdir / "metrics.jsonl"
    with open(path, "w") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    AFE2._json_safe(row), sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    return path


def _authenticate_metric_grid(
    metric_rows: list[dict[str, Any]],
    rounds: tuple[int, ...] | list[int],
    outdir: Path | None = None,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> None:
    profile = resolve_evaluation_profile(eval_profile)
    rounds = tuple(int(value) for value in rounds)
    gamma_rows = [row for row in metric_rows if row["scope"] == "gamma"]
    pooled_rows = [row for row in metric_rows if row["scope"] == "pooled"]
    if any(row.get("mode") != "raw" for row in metric_rows):
        raise RuntimeError("raw sweep contains a non-raw metric row")
    if any(row.get("caption") != profile.caption for row in metric_rows):
        raise RuntimeError("raw sweep metric caption changed")
    if any(
        row.get("evaluation_profile", DEFAULT_EVAL_PROFILE) != profile.name
        for row in metric_rows
    ):
        raise RuntimeError("raw sweep evaluation profile changed")
    if len(gamma_rows) != len(rounds) * len(GAMMAS):
        raise RuntimeError("per-gamma raw metric grid is incomplete")
    if len(pooled_rows) != len(rounds):
        raise RuntimeError("pooled raw metric grid is incomplete")
    expected_gamma = {
        (round_i, gamma) for round_i in rounds for gamma in GAMMAS
    }
    actual_gamma = {(int(row["round"]), float(row["gamma"])) for row in gamma_rows}
    if actual_gamma != expected_gamma:
        raise RuntimeError("raw per-gamma metric keys are incomplete")
    if {int(row["round"]) for row in pooled_rows} != set(rounds):
        raise RuntimeError("raw pooled metric keys are incomplete")
    for row in gamma_rows:
        if int(row["n"]) != profile.m:
            raise RuntimeError(
                f"per-gamma raw metric row does not contain M={profile.m}"
            )
        if set(row["binary"]) != {"SR", "CR", "timeout", "V_safe", "V_full"}:
            raise RuntimeError("per-gamma raw binary metric set is incomplete")
        if (
            sum(
                row["binary"][key]["count"]
                for key in ("SR", "CR", "timeout")
            )
            != profile.m
        ):
            raise RuntimeError(
                f"per-gamma terminal counts do not partition M={profile.m}"
            )
    for row in pooled_rows:
        if int(row["n"]) != profile.m * len(GAMMAS):
            raise RuntimeError("pooled raw metric row has the wrong sample count")
        if set(row["binary"]) != {"SR", "CR", "timeout", "V_safe", "V_full"}:
            raise RuntimeError("pooled raw binary metric set is incomplete")
        if sum(
            row["binary"][key]["count"] for key in ("SR", "CR", "timeout")
        ) != profile.m * len(GAMMAS):
            raise RuntimeError("pooled terminal counts do not partition the raw sweep")
    if outdir is not None:
        expected_cells = len(rounds) * len(GAMMAS)
        cell_root = outdir / "cells" / "raw" / "afe_rbf"
        if len(list(cell_root.glob("*.npz"))) != expected_cells:
            raise RuntimeError("raw cell archive count is incomplete")
        if len(list(cell_root.glob("*.provenance.json"))) != expected_cells:
            raise RuntimeError("raw cell provenance count is incomplete")


def _pooled_rows(metric_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["round"]): row
        for row in metric_rows
        if row["scope"] == "pooled" and row["mode"] == "raw"
    }
    if len(rows) != sum(
        row["scope"] == "pooled" and row["mode"] == "raw" for row in metric_rows
    ):
        raise RuntimeError("duplicate pooled metric rows")
    return rows


def best_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    clearance = row["minimum_clearance"]["mean"]
    clearance = float(clearance) if clearance is not None else float("-inf")
    return (
        -float(row["binary"]["SR"]["estimate"]),
        float(row["binary"]["CR"]["estimate"]),
        float(row["binary"]["timeout"]["estimate"]),
        -clearance,
        int(row["round"]),
    )


def select_best_round(metric_rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    pooled = _pooled_rows(metric_rows)
    if not pooled:
        raise ValueError("cannot select a best checkpoint without pooled true evaluation")
    ordered = sorted(pooled.values(), key=best_rank_key)
    ranking = [
        {
            "rank": rank,
            "round": int(row["round"]),
            "SR": float(row["binary"]["SR"]["estimate"]),
            "CR": float(row["binary"]["CR"]["estimate"]),
            "timeout": float(row["binary"]["timeout"]["estimate"]),
            "mean_minimum_clearance": float(row["minimum_clearance"]["mean"]),
        }
        for rank, row in enumerate(ordered, start=1)
    ]
    return int(ordered[0]["round"]), ranking


def _row_lookup(metric_rows: list[dict[str, Any]]) -> dict[tuple[int, float | None], dict[str, Any]]:
    return {
        (int(row["round"]), None if row["gamma"] is None else float(row["gamma"])): row
        for row in metric_rows
    }


def _metric_series(
    rows: list[dict[str, Any]], key: str
) -> tuple[list[float], list[float], list[float]]:
    values, lower, upper = [], [], []
    for row in rows:
        if key in ("SR", "CR", "timeout", "V_safe", "V_full"):
            entry = row["binary"][key]
            value = float(entry["estimate"])
            lo, hi = (float(item) for item in entry["wilson95"])
        elif key == "clearance":
            entry = row["minimum_clearance"]
            value = float(entry["mean"])
            lo, hi = (float(item) for item in entry["bootstrap95"])
        elif key == "time":
            entry = row["successful_time_to_goal"]
            if entry["mean"] is None:
                value = lo = hi = float("nan")
            else:
                value = float(entry["mean"])
                lo, hi = (float(item) for item in entry["bootstrap95"])
        elif key == "route_balance":
            value = float(
                row["route_modes"]["closest_obstacle_approach_success_only"]
                ["coverage_weighted_balance"]
            )
            lo = hi = value
        else:
            raise KeyError(key)
        values.append(value)
        lower.append(lo)
        upper.append(hi)
    return values, lower, upper


def _render_curves(
    outdir: Path,
    metric_rows: list[dict[str, Any]],
    rounds: tuple[int, ...],
    best_round: int,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> list[Path]:
    profile = resolve_evaluation_profile(eval_profile)
    lookup = _row_lookup(metric_rows)
    specs = [
        ("SR", "SR"),
        ("CR", "CR / OOB"),
        ("timeout", "Timeout"),
        ("V_safe", r"$V_{safe}$"),
        ("V_full", r"$V_{full}$"),
        ("clearance", "Min. clearance [m]"),
        ("time", "Time-to-goal [s]"),
        ("route_balance", "U/R balance"),
    ]
    cmap = plt.get_cmap("plasma")
    colors = {
        gamma: cmap(0.08 + 0.84 * index / (len(GAMMAS) - 1))
        for index, gamma in enumerate(GAMMAS)
    }
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
    for plot_index, (ax, (key, title)) in enumerate(zip(axes.flat, specs)):
        for gamma in GAMMAS:
            series = [lookup[(round_i, gamma)] for round_i in rounds]
            values, _, _ = _metric_series(series, key)
            ax.plot(rounds, values, color=colors[gamma], lw=1.1, alpha=0.68)
        pooled = [lookup[(round_i, None)] for round_i in rounds]
        values, lower, upper = _metric_series(pooled, key)
        ax.plot(rounds, values, color="black", lw=2.6)
        ax.fill_between(rounds, lower, upper, color="black", alpha=0.12, lw=0)
        ax.axvline(best_round, color="#0072b2", ls="--", lw=1.2)
        ax.axvline(rounds[-1], color="0.4", ls=":", lw=1.1)
        ax.set_title(title)
        if plot_index >= 4:
            ax.set_xlabel("round")
        ax.grid(alpha=0.25)
        if key in ("SR", "CR", "timeout", "V_safe", "V_full", "route_balance"):
            ax.set_ylim(-0.03, 1.03)
    handles = [
        plt.Line2D([0], [0], color=colors[gamma], lw=2, label=rf"$\gamma={gamma}$")
        for gamma in GAMMAS
    ]
    handles.append(plt.Line2D([0], [0], color="black", lw=2.6, label="pooled"))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=8,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(
        f"Raw M={profile.m}/$\\gamma$ checkpoint evaluation | "
        f"best r{best_round} | final r{rounds[-1]}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"{profile.filename_tag}_checkpoint_curves.{suffix}"
        fig.savefig(path, dpi=160)
        outputs.append(path)
    plt.close(fig)
    return outputs


def render_existing_evaluation(
    evaluation_dir: str | os.PathLike[str],
    presentation_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Render a compact report without changing an authenticated evaluation tree."""
    evaluation_root = Path(evaluation_dir).resolve()
    presentation_root = Path(presentation_dir).resolve()
    if presentation_root.exists():
        raise FileExistsError(
            f"presentation output root must be absent/new: {presentation_root}"
        )
    complete = validate_output(evaluation_root)
    contract = load_json(evaluation_root / "evaluation_contract.json")
    profile = resolve_evaluation_profile(
        contract.get("evaluation_profile", DEFAULT_EVAL_PROFILE)
    )
    rounds = tuple(int(value) for value in contract["rounds"])
    metric_rows = [
        json.loads(line)
        for line in (evaluation_root / "metrics.jsonl").read_text().splitlines()
        if line
    ]
    best_round, _ = select_best_round(metric_rows)
    renderer_source = git_state()
    if (
        renderer_source["tracked_dirty"]
        or renderer_source["untracked_runtime_sources"]
    ):
        raise RuntimeError("presentation rendering requires committed clean source")

    presentation_root.mkdir(parents=True)
    curve_paths = _render_curves(
        presentation_root, metric_rows, rounds, best_round, profile
    )
    report_paths = []
    for source in curve_paths:
        destination = presentation_root / f"report{source.suffix}"
        shutil.copy2(source, destination)
        report_paths.append(destination)
    source_manifest = evaluation_root / "EVALUATION_COMPLETE.json"
    payload = {
        "status": "RAW_EVALUATION_PRESENTATION_COMPLETE",
        "evaluation_profile": profile.name,
        "source_evaluation": str(evaluation_root),
        "source_evaluation_manifest_sha256": sha256_file(source_manifest),
        "source_metrics_sha256": sha256_file(evaluation_root / "metrics.jsonl"),
        "post_hoc_best_round": best_round,
        "rounds": list(rounds),
        "renderer_source": renderer_source,
        "reports": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in report_paths
        ],
        "source_delivery_status": complete["status"],
    }
    write_json(presentation_root / "PRESENTATION_COMPLETE.json", payload)
    return payload


def _load_cell(outdir: Path, round_i: int, gamma: float):
    path = outdir / "cells" / "raw" / "afe_rbf" / f"r{round_i:03d}_g{gamma:.1f}.npz"
    with np.load(path, allow_pickle=True) as archive:
        return (
            list(archive["paths"]),
            list(archive["status"]),
            list(archive["outcome"]),
            list(archive["rollout_index"]),
        )


def _draw_scene(
    ax: Any,
    profile: Any,
    env: Any,
    paths: list[np.ndarray],
    gamma: float,
    title: str,
    outcomes: list[str],
    gallery_indices: tuple[int, ...] = GALLERY_INDICES,
) -> None:
    obstacles = env.obstacles.detach().cpu().numpy()
    for obstacle in obstacles:
        ax.add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#bdbdbd", zorder=1)
        )
    color = plt.get_cmap("plasma")(
        0.08 + 0.84 * GAMMAS.index(gamma) / (len(GAMMAS) - 1)
    )
    for index in gallery_indices:
        path = np.asarray(paths[index], dtype=float)
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.0, alpha=0.72, zorder=3)
        dots = path[::STATE_DOT_STRIDE]
        ax.plot(
            dots[:, 0], dots[:, 1], linestyle="none", marker=".", color=color,
            ms=1.3, alpha=0.52, zorder=4,
        )
        if outcomes[index] != "SR":
            ax.plot(
                path[-1, 0], path[-1, 1], "x", color="#cc3311",
                ms=5, mew=1.2, zorder=5,
            )
    ax.plot(*profile.start, "ks", ms=4, zorder=6)
    ax.plot(*profile.goal, marker="*", color="gold", mec="k", ms=10, zorder=6)
    ax.set_xlim(-0.35, 5.35)
    ax.set_ylim(-0.35, 5.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def _render_galleries(
    outdir: Path,
    profile: Any,
    env: Any,
    metric_rows: list[dict[str, Any]],
    rounds: tuple[int, ...],
    best_round: int,
    eval_profile: str | EvaluationProfile = DEFAULT_EVAL_PROFILE,
) -> tuple[list[Path], Path]:
    eval_spec = resolve_evaluation_profile(eval_profile)
    gallery_indices = eval_spec.gallery_indices
    lookup = _row_lookup(metric_rows)
    final_round = rounds[-1]
    roles = [
        ("pretrained", 0),
        ("post-hoc best", best_round),
        ("final", final_round),
    ]
    fig, axes = plt.subplots(len(roles), len(GAMMAS), figsize=(23, 10.5), squeeze=False)
    for row_index, (role, round_i) in enumerate(roles):
        pooled = lookup[(round_i, None)]
        for gamma_index, gamma in enumerate(GAMMAS):
            paths, _, outcomes, indices = _load_cell(outdir, round_i, gamma)
            if [int(value) for value in indices] != list(range(eval_spec.m)):
                raise RuntimeError("gallery source lost fixed rollout indices")
            gamma_row = lookup[(round_i, gamma)]
            title = (
                f"gamma={gamma} | SR={gamma_row['binary']['SR']['estimate']:.2f} "
                f"CR={gamma_row['binary']['CR']['estimate']:.2f}"
            )
            _draw_scene(
                axes[row_index, gamma_index],
                profile,
                env,
                paths,
                gamma,
                title,
                outcomes,
                gallery_indices,
            )
            if gamma_index == 0:
                axes[row_index, gamma_index].set_ylabel(
                    f"{role}: r{round_i}\npooled SR={pooled['binary']['SR']['estimate']:.3f}, "
                    f"CR={pooled['binary']['CR']['estimate']:.3f}",
                    fontsize=10,
                )
    fig.suptitle(
        f"Fixed raw rollout indices {list(gallery_indices)}; state dots every "
        f"{STATE_DOT_STRIDE} steps\n{eval_spec.caption}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"{eval_spec.filename_tag}_r0_best_final_gallery.{suffix}"
        fig.savefig(path, dpi=160)
        outputs.append(path)
    plt.close(fig)

    round_dir = outdir / "round_galleries"
    round_dir.mkdir()
    for round_i in rounds:
        fig, axes = plt.subplots(1, len(GAMMAS), figsize=(23, 3.7), squeeze=False)
        pooled = lookup[(round_i, None)]
        for gamma_index, gamma in enumerate(GAMMAS):
            paths, _, outcomes, indices = _load_cell(outdir, round_i, gamma)
            if [int(value) for value in indices] != list(range(eval_spec.m)):
                raise RuntimeError("round gallery source lost fixed rollout indices")
            gamma_row = lookup[(round_i, gamma)]
            _draw_scene(
                axes[0, gamma_index],
                profile,
                env,
                paths,
                gamma,
                (
                    f"gamma={gamma}\nSR={gamma_row['binary']['SR']['estimate']:.2f}, "
                    f"CR={gamma_row['binary']['CR']['estimate']:.2f}"
                ),
                outcomes,
                gallery_indices,
            )
        fig.suptitle(
            f"Raw checkpoint r{round_i}: pooled SR={pooled['binary']['SR']['estimate']:.3f}, "
            f"CR={pooled['binary']['CR']['estimate']:.3f}\n{eval_spec.caption}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.85))
        path = round_dir / f"{eval_spec.filename_tag}_round_{round_i:03d}.png"
        fig.savefig(path, dpi=150)
        outputs.append(path)
        plt.close(fig)

    manifest_path = outdir / "gallery_indices.json"
    write_json(
        manifest_path,
        {
            "caption": eval_spec.caption,
            "evaluation_profile": eval_spec.name,
            "rule": "fixed archive indices; no outcome-based trajectory selection",
            "indices": list(gallery_indices),
            "state_dot_stride": STATE_DOT_STRIDE,
            "roles": [
                {"role": role, "round": round_i} for role, round_i in roles
            ],
            "per_round": list(rounds),
            "gammas": list(GAMMAS),
            "M": eval_spec.m,
        },
    )
    return outputs, manifest_path


def _artifact_inventory(outdir: Path) -> dict[str, str]:
    inventory = {}
    for path in sorted(outdir.rglob("*")):
        if not path.is_file() or path.name == "EVALUATION_COMPLETE.json":
            continue
        inventory[str(path.relative_to(outdir))] = sha256_file(path)
    return inventory


def run_evaluation(args: argparse.Namespace) -> None:
    eval_profile = resolve_evaluation_profile(
        getattr(args, "eval_profile", DEFAULT_EVAL_PROFILE)
    )
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"evaluation output root must be absent/new: {outdir}")
    contract = validate_completed_run(
        args.run_root, args.scene_profile, eval_profile
    )
    source_state = require_clean_additive_source(contract["source_git_commit"])
    gpu = _gpu_record()
    rounds = tuple(contract["evaluation_rounds"])

    policy0, _, r0_model_sha, conditioning0 = _load_policy(contract, 0, "cpu")
    if r0_model_sha != contract["source_checkpoint_model_sha256"]:
        raise RuntimeError("ckpt_0 model state does not equal the authenticated pretrained model")
    final_model_sha = _authenticate_final_alias(contract)
    policy_dim = int(policy0.d)
    noise_bank, noise_metadata = build_noise_bank(
        args.scene_profile, policy_dim, eval_profile
    )
    del policy0

    profile = get_scene_profile(args.scene_profile)
    env = build_scene(profile)
    snapshot = scene_snapshot(env, profile)
    assert_scene_snapshot(snapshot)
    if snapshot["sha256"] != contract["scene_sha256"]:
        raise RuntimeError("rebuilt evaluation scene does not match the completed trainer run")
    cfg = RawEvalConfig(
        scene_profile=args.scene_profile,
        conditioning_schema=conditioning0.schema,
    )
    outdir.mkdir(parents=True)
    write_json(
        outdir / "evaluation_contract.json",
        {
            "metric_version": eval_profile.metric_version,
            "caption": eval_profile.caption,
            "evaluation_profile": eval_profile.name,
            "trainer_source_commit": contract["source_git_commit"],
            "evaluation_source": source_state,
            "gpu": gpu,
            "scene": snapshot,
            "rounds": list(rounds),
            "gammas": list(GAMMAS),
            "M": eval_profile.m,
            "T": T,
            "reach": REACH,
            "nfe": NFE,
            "temp": TEMP,
            "mode": "raw only; untilted; no GP/RBF or verifier selection",
            "verifier_workers": int(args.verifier_workers),
            "validity": {
                "V_safe": (
                    "whole executed trajectory is in task space and every stride-2 H=10 "
                    "window passes the unchanged SOCP verifier"
                ),
                "V_full": "V_safe plus the unchanged window-level goal-approach criterion",
                "role": (
                    "post-hoc true-evaluation metrics only; the verifier never selects or "
                    "changes a raw policy action"
                ),
            },
            "noise_bank": noise_metadata,
            "common_random_numbers": (
                "the exact same scene-keyed noise tensor is indexed by gamma, rollout, "
                "and control time for every arm and checkpoint round"
            ),
            "checkpoint_selection_rule": (
                "authenticate all trainer checkpoints; evaluate ckpt_0.pt, each available "
                f"multiple of {eval_profile.checkpoint_stride}, and "
                "ckpt_<completed_round>.pt; never substitute final.pt"
            ),
            "final_alias_model_state_sha256": final_model_sha,
            "post_hoc_best_selection": BEST_SELECTION_RULE,
            "report_metric_source": (
                "SR, CR, timeout, V_safe, V_full, clearance, and time are computed only "
                f"from the stored raw M={eval_profile.m}/gamma evaluation trajectories; "
                "trainer probe "
                "metrics are not read"
            ),
            "completed_run": contract,
        },
    )

    device = "cuda:0"
    metric_rows: list[dict[str, Any]] = []
    round_timing = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.verifier_workers),
        mp_context=context,
        initializer=M20._worker_init,
        initargs=(args.scene_profile, REACH, 180),
    ) as executor:
        for round_i in rounds:
            round_started = time.perf_counter()
            policy, _, model_sha, conditioning = _load_policy(contract, round_i, device)
            if conditioning != conditioning0:
                raise RuntimeError(f"checkpoint r{round_i} changed conditioning architecture")
            episodes = run_raw_batch(
                policy, env, cfg, device, noise_bank, eval_profile
            )
            obstacle_array = env.obstacles.detach().cpu().numpy()
            if profile.center_replacement_radius is not None:
                route_mask = np.linalg.norm(
                    obstacle_array[:, :2] - np.asarray((2.5, 2.5)), axis=1
                ) < 1.0e-6
            else:
                central_centers = np.asarray(
                    ((2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0))
                )
                route_mask = np.any(
                    np.linalg.norm(
                        obstacle_array[:, None, :2] - central_centers[None], axis=2
                    ) < 1.0e-6,
                    axis=1,
                )
            if not route_mask.any():
                raise RuntimeError("raw U/R metric could not identify central obstacles")
            obstacle_centers = obstacle_array[route_mask, :2]
            obstacle_radii = (
                obstacle_array[route_mask, 2] + float(env.r_robot)
            )
            for episode in episodes:
                path = np.asarray(episode["path"], dtype=np.float64)
                early_index = min(10, len(path) - 1)
                episode["route_mode_early"] = int(
                    RM.classify_plan_endpoints(
                        path[early_index:early_index + 1],
                        start=profile.start,
                        goal=profile.goal,
                    )[0]
                )
                closest_labels, _ = RM.classify_trajectories_at_closest_approach(
                    path[None, :, :],
                    start=profile.start,
                    goal=profile.goal,
                    obstacle_centers=obstacle_centers,
                    obstacle_radii=obstacle_radii,
                )
                episode["route_mode_closest"] = int(closest_labels[0])
            tasks = [
                (
                    episode["path"],
                    episode["gamma"],
                    episode["status"],
                    float(env.dt),
                    REACH,
                )
                for episode in episodes
            ]
            worker_rows = list(
                executor.map(M20._trajectory_metrics_worker, tasks, chunksize=2)
            )
            metrics = [
                normalize_trajectory_metrics(episode, worker_row, float(env.dt))
                for episode, worker_row in zip(episodes, worker_rows)
            ]
            pooled_metrics: list[dict[str, Any]] = []
            for gamma in GAMMAS:
                row, _ = _save_cell(
                    outdir,
                    contract,
                    round_i,
                    gamma,
                    episodes,
                    metrics,
                    model_sha,
                    noise_metadata,
                    eval_profile,
                )
                metric_rows.append(row)
                pooled_metrics.extend(
                    metric
                    for episode, metric in zip(episodes, metrics)
                    if episode["gamma"] == gamma
                )
            metric_rows.append(
                aggregate_metrics(
                    pooled_metrics,
                    round_i=round_i,
                    gamma=None,
                    scope="pooled",
                    scene_profile=args.scene_profile,
                    algorithm=contract["algorithm"],
                    eval_profile=eval_profile,
                )
            )
            elapsed = time.perf_counter() - round_started
            round_timing.append({"round": round_i, "elapsed_seconds": elapsed})
            del policy
            torch.cuda.empty_cache()
            print(
                f"[raw M{eval_profile.m} r{round_i:03d}] rollout+validity complete "
                f"in {elapsed:.1f}s",
                flush=True,
            )

    _authenticate_metric_grid(metric_rows, rounds, outdir, eval_profile)
    metrics_path = _write_metrics(outdir, metric_rows)
    best_round, ranking = select_best_round(metric_rows)
    curve_paths = _render_curves(
        outdir, metric_rows, rounds, best_round, eval_profile
    )
    gallery_paths, gallery_manifest = _render_galleries(
        outdir, profile, env, metric_rows, rounds, best_round, eval_profile
    )
    elapsed = time.perf_counter() - started
    finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary_path = outdir / "evaluation_summary.json"
    write_json(
        summary_path,
        {
            "status": eval_profile.summary_status,
            "metric_version": eval_profile.metric_version,
            "caption": eval_profile.caption,
            "evaluation_profile": eval_profile.name,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "elapsed_seconds": elapsed,
            "round_timing": round_timing,
            "rounds": list(rounds),
            "final_round": rounds[-1],
            "post_hoc_best_round": best_round,
            "post_hoc_selection_rule": BEST_SELECTION_RULE,
            "post_hoc_ranking": ranking,
            "gammas": list(GAMMAS),
            "M": eval_profile.m,
            "verifier_workers": int(args.verifier_workers),
            "cell_count": len(rounds) * len(GAMMAS),
            "metric_row_count": len(metric_rows),
            "gpu": gpu,
            "source": source_state,
            "run": contract,
            "outputs": {
                "metrics": str(metrics_path),
                "curves": [str(path) for path in curve_paths],
                "galleries": [str(path) for path in gallery_paths],
                "gallery_indices": str(gallery_manifest),
            },
        },
    )
    inventory = _artifact_inventory(outdir)
    write_json(
        outdir / "EVALUATION_COMPLETE.json",
        {
            "status": eval_profile.delivery_status,
            "metric_version": eval_profile.metric_version,
            "caption": eval_profile.caption,
            "evaluation_profile": eval_profile.name,
            "trainer_source_commit": contract["source_git_commit"],
            "evaluation_source_commit": source_state["commit"],
            "scene_sha256": snapshot["sha256"],
            "elapsed_seconds": elapsed,
            "artifact_sha256": inventory,
        },
    )
    print(
        f"AFE RBF RAW M{eval_profile.m} SWEEP COMPLETE: {outdir}", flush=True
    )


def validate_output(outdir: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(outdir).resolve()
    complete_path = root / "EVALUATION_COMPLETE.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"evaluation completion manifest is missing: {complete_path}")
    complete = load_json(complete_path)
    profile = resolve_evaluation_profile(
        complete.get("evaluation_profile", DEFAULT_EVAL_PROFILE)
    )
    if complete.get("status") != profile.delivery_status:
        raise RuntimeError(f"raw M={profile.m} evaluation completion status is invalid")
    if complete.get("caption") != profile.caption:
        raise RuntimeError(f"raw M={profile.m} completion caption changed")
    inventory = complete.get("artifact_sha256", {})
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "EVALUATION_COMPLETE.json"
    }
    if set(inventory) != actual_files:
        raise RuntimeError("raw M50 delivery inventory does not match output files")
    for relative, expected in inventory.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"raw M={profile.m} output hash mismatch: {relative}")
    contract = load_json(root / "evaluation_contract.json")
    if contract.get("evaluation_profile", DEFAULT_EVAL_PROFILE) != profile.name:
        raise RuntimeError("raw evaluation profile disagrees across manifests")
    if contract.get("caption") != profile.caption:
        raise RuntimeError(f"raw M={profile.m} evaluation contract caption changed")
    rounds = tuple(int(value) for value in contract["rounds"])
    rows = [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text().splitlines()
        if line
    ]
    _authenticate_metric_grid(rows, rounds, root, profile)
    summary = load_json(root / "evaluation_summary.json")
    best_round, ranking = select_best_round(rows)
    if int(summary.get("post_hoc_best_round", -1)) != best_round:
        raise RuntimeError("stored best checkpoint is not reproducible from true metrics")
    if summary.get("post_hoc_ranking") != ranking:
        raise RuntimeError("stored post-hoc ranking is not reproducible from true metrics")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root")
    parser.add_argument("--scene-profile", choices=sorted(SCENE_PROFILES))
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--eval-profile",
        choices=sorted(EVALUATION_PROFILES),
        default=DEFAULT_EVAL_PROFILE,
    )
    parser.add_argument("--verifier-workers", type=int, default=16)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--presentation-outdir")
    args = parser.parse_args()
    if args.validate_only and args.render_only:
        parser.error("--validate-only and --render-only are mutually exclusive")
    if args.validate_only:
        validate_output(args.outdir)
        print(f"AFE RBF RAW OUTPUT VALID: {Path(args.outdir).resolve()}")
        return
    if args.render_only:
        if not args.presentation_outdir:
            parser.error("--presentation-outdir is required with --render-only")
        payload = render_existing_evaluation(args.outdir, args.presentation_outdir)
        print(
            "AFE RBF RAW PRESENTATION COMPLETE: "
            f"{Path(args.presentation_outdir).resolve()} "
            f"(best r{payload['post_hoc_best_round']})"
        )
        return
    if args.presentation_outdir:
        parser.error("--presentation-outdir is only valid with --render-only")
    if not args.run_root or not args.scene_profile:
        parser.error(
            "--run-root and --scene-profile are required unless a read-only mode is used"
        )
    if args.verifier_workers < 1:
        parser.error("--verifier-workers must be positive")
    run_evaluation(args)


if __name__ == "__main__":
    main()
