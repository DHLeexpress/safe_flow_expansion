#!/usr/bin/env python3
"""Build the fixed-index B1_current_best 5x3 comparison gallery.

Rows are SafeMPPI expert, raw pretrained, raw B1 r19, and two diagnostic
Kazuki/CFM-MPPI safety-guidance settings.  The raw rows are read from the
authenticated disjoint M50 archive; they are never regenerated or curated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_REV = _ROOT.parent
_WORK = _REV.parent
for _path in (_WORK, _REV, _ROOT):
    sys.path.insert(0, str(_path))

import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_scene as GS
from afe2_scene_profiles import build_scene, get_scene_profile, scene_snapshot


GAMMAS = (0.1, 0.5, 1.0)
FIXED_INDICES = tuple(range(10))
METRIC_VERSION = "b1_current_best_gallery_v1"
STATE_DOT_STRIDE = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classify_path(path: np.ndarray, env: Any, reach: float) -> str:
    points = np.asarray(path, dtype=float)
    goal = env.goal.detach().cpu().numpy()
    if np.linalg.norm(points[-1] - goal) < reach:
        return "SR"
    if not np.all((points >= 0.0) & (points <= 5.0)):
        return "CR"
    obstacles = env.obstacles.detach().cpu().numpy()
    if obstacles.size:
        clearance = (
            np.linalg.norm(points[:, None, :] - obstacles[None, :, :2], axis=2)
            - obstacles[None, :, 2]
            - float(env.r_robot)
        )
        if float(clearance.min()) < 0.0:
            return "CR"
    return "TO"


def closest_route_value(path: np.ndarray, center: tuple[float, float] = (2.5, 2.5)) -> float:
    points = np.asarray(path, dtype=float)
    index = int(np.argmin(np.linalg.norm(points - np.asarray(center), axis=1)))
    return float(points[index, 1] - points[index, 0])


def route_summary(paths: list[np.ndarray], ambiguity: float = 0.05) -> dict[str, float | int]:
    values = np.asarray([closest_route_value(path) for path in paths], dtype=float)
    u_count = int(np.sum(values > ambiguity))
    r_count = int(np.sum(values < -ambiguity))
    resolved = u_count + r_count
    balance = 2.0 * min(u_count, r_count) / resolved if resolved else 0.0
    return {
        "u_count": u_count,
        "r_count": r_count,
        "ambiguous_count": int(len(values) - resolved),
        "balance": float(balance),
        "route_value_std": float(values.std()),
    }


def choose_low_high(summaries: dict[float, dict[str, float | int]]) -> tuple[float, float]:
    positive = sorted(coef for coef in summaries if coef > 0.0)
    if len(positive) < 2:
        raise ValueError("at least two positive safety coefficients are required")
    return positive[0], positive[-1]


def raw_cell(confirmation: Path, round_i: int, gamma: float) -> tuple[list[np.ndarray], list[str]]:
    path = confirmation / "cells" / f"r{round_i:03d}_g{gamma:.1f}.npz"
    with np.load(path, allow_pickle=True) as archive:
        indices = np.asarray(archive["rollout_index"], dtype=int)
        if not np.array_equal(indices, np.arange(len(indices))):
            raise RuntimeError(f"raw cell lost fixed rollout indices: {path}")
        paths = [np.asarray(value, dtype=np.float32) for value in archive["paths"]]
        outcomes = [str(value) for value in archive["outcome"]]
    return [paths[index] for index in FIXED_INDICES], [outcomes[index] for index in FIXED_INDICES]


def load_reused_expert(root: Path) -> dict[float, tuple[list[np.ndarray], list[str]]]:
    path = root / "expert.npz"
    cells: dict[float, tuple[list[np.ndarray], list[str]]] = {}
    with np.load(path, allow_pickle=True) as archive:
        for gamma in GAMMAS:
            suffix = f"g{gamma:g}"
            cells[gamma] = (
                [np.asarray(value, dtype=np.float32) for value in archive[f"paths_{suffix}"]],
                [str(value) for value in archive[f"outcomes_{suffix}"]],
            )
    return cells


def load_reused_kazuki(root: Path, coefficient: float) -> tuple[list[np.ndarray], list[str]]:
    path = root / f"kazuki_ws_{coefficient:g}.npz"
    with np.load(path, allow_pickle=True) as archive:
        return (
            [np.asarray(value, dtype=np.float32) for value in archive["paths"]],
            [str(value) for value in archive["outcomes"]],
        )


def run_expert(env: Any, gamma: float, m: int, t_cap: int, reach: float) -> tuple[list[np.ndarray], list[str]]:
    from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
    from di_grid_viz import di_step

    goal_t = env.goal.detach().cpu().float()
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    planner_obstacles = GS.planner_obstacles(env)
    config = GS.mode1_config()
    paths: list[np.ndarray] = []
    outcomes: list[str] = []
    for rollout_index in range(m):
        seed = named_seed(METRIC_VERSION, "expert", gamma, rollout_index)
        seed_all(seed)
        adapter = SafeMPPIAdapter(**config)
        state = env.x0.detach().cpu().numpy().astype(np.float32).copy()
        path = [state[:2].copy()]
        for step in range(t_cap):
            action, _ = adapter.plan(
                torch.as_tensor(state, dtype=torch.float32), goal_t,
                planner_obstacles, gamma=gamma, seed=seed + step,
            )
            state = di_step(state, action.detach().cpu().numpy().astype(np.float32), dt=env.dt)
            path.append(state[:2].copy())
            if np.linalg.norm(state[:2] - goal) < reach:
                break
            if not GM.in_taskspace(state[:2][None]):
                break
            if obstacles.size and float((
                np.linalg.norm(state[:2][None] - obstacles[:, :2], axis=1)
                - obstacles[:, 2] - float(env.r_robot)
            ).min()) < 0.0:
                break
        array = np.asarray(path, dtype=np.float32)
        paths.append(array)
        outcomes.append(classify_path(array, env, reach))
    return paths, outcomes


def configure_kazuki() -> Any:
    import kazuki_baseline as baseline

    baseline.GOAL_COEF = 0.0
    baseline.COLL_W = 100.0
    baseline.GOAL_W = 0.1
    baseline.BETA_MPPI = 20.0
    baseline.MPPI_LAMBDA = 0.1
    baseline.MPPI_SIGMA = 0.2
    baseline.R_MARGIN = 0.05
    baseline.N_SAMPLE = 200
    baseline.N_ELITE = 10
    baseline.N_COPY = 200
    return baseline


def run_kazuki(policy: Any, env: Any, safe_coef: float, m: int, t_cap: int,
               reach: float, device: str) -> tuple[list[np.ndarray], list[str]]:
    baseline = configure_kazuki()
    paths: list[np.ndarray] = []
    outcomes: list[str] = []
    for rollout_index in range(m):
        seed = named_seed(METRIC_VERSION, "kazuki", rollout_index)
        seed_all(seed)
        output = baseline.kazuki_deploy(
            policy, env, [safe_coef], gamma_ctx=0.5, T=t_cap, reach=reach,
            device=device, seed=seed,
            conditioning_schema="low7_closest_boundary_tie_mean",
        )
        path = np.asarray(output["path"], dtype=np.float32)
        paths.append(path)
        outcomes.append(classify_path(path, env, reach))
    return paths, outcomes


def draw_scene(ax: Any, env: Any, paths: list[np.ndarray], outcomes: list[str],
               gamma: float, title: str, ylabel: str) -> None:
    obstacles = env.obstacles.detach().cpu().numpy()
    for obstacle in obstacles:
        ax.add_patch(plt.Circle(obstacle[:2], obstacle[2], color="#c8c8c8", zorder=1))
    gamma_index = GAMMAS.index(gamma)
    color = plt.get_cmap("plasma")(0.1 + 0.8 * gamma_index / (len(GAMMAS) - 1))
    for path, outcome in zip(paths, outcomes):
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.35, alpha=0.78, zorder=3)
        dots = path[::STATE_DOT_STRIDE]
        ax.plot(dots[:, 0], dots[:, 1], linestyle="none", marker=".", color=color,
                ms=1.8, alpha=0.62, zorder=4)
        if outcome != "SR":
            ax.plot(path[-1, 0], path[-1, 1], marker="x", linestyle="none",
                    color="#cc3311", ms=8.0, mew=2.0, zorder=6)
    profile = get_scene_profile("low7_radius1_canonical_v1")
    ax.plot(*profile.start, "ks", ms=5.5, zorder=7)
    ax.plot(*profile.goal, marker="*", color="gold", mec="black", ms=13, zorder=7)
    ax.set_xlim(-0.3, 5.3)
    ax.set_ylim(-0.3, 5.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=20, pad=8)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--pretrained-ckpt", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--reach", type=float, default=0.15)
    parser.add_argument(
        "--reuse-rollouts", type=Path, default=None,
        help="reuse an authenticated diagnostic directory; render only, no new expert/Kazuki rollout",
    )
    parser.add_argument(
        "--safe-coef-candidates", type=float, nargs="+",
        default=[0.0, 0.1, 0.9],
    )
    args = parser.parse_args()
    if args.m != len(FIXED_INDICES):
        raise ValueError("this paper gallery is locked to ten fixed trajectories per cell")
    if args.outdir.exists():
        raise FileExistsError(f"fresh output directory required: {args.outdir}")
    args.outdir.mkdir(parents=True)

    profile = get_scene_profile("low7_radius1_canonical_v1")
    env = build_scene(profile)
    from grid_hp_expt import load_hp
    policy, _ = load_hp(str(args.pretrained_ckpt), device="cpu")
    policy = policy.to(args.device).eval()

    if args.reuse_rollouts is None:
        expert: dict[float, tuple[list[np.ndarray], list[str]]] = {}
        for gamma in GAMMAS:
            expert[gamma] = run_expert(env, gamma, args.m, args.T, args.reach)
    else:
        expert = load_reused_expert(args.reuse_rollouts)

    kazuki: dict[float, tuple[list[np.ndarray], list[str]]] = {}
    summaries: dict[float, dict[str, float | int]] = {}
    for coefficient in args.safe_coef_candidates:
        if args.reuse_rollouts is None:
            paths, outcomes = run_kazuki(
                policy, env, coefficient, args.m, args.T, args.reach, args.device,
            )
        else:
            paths, outcomes = load_reused_kazuki(args.reuse_rollouts, coefficient)
        kazuki[coefficient] = (paths, outcomes)
        summary = route_summary(paths)
        summary.update({
            "sr": int(sum(outcome == "SR" for outcome in outcomes)) / args.m,
            "cr": int(sum(outcome == "CR" for outcome in outcomes)) / args.m,
            "timeout": int(sum(outcome == "TO" for outcome in outcomes)) / args.m,
        })
        summaries[coefficient] = summary
    low_coef, high_coef = choose_low_high(summaries)

    raw0 = {gamma: raw_cell(args.confirmation, 0, gamma) for gamma in GAMMAS}
    raw19 = {gamma: raw_cell(args.confirmation, 19, gamma) for gamma in GAMMAS}
    rows = [
        (r"SafeMPPI expert", expert),
        (r"Pretrained", raw0),
        (r"B1 current best", raw19),
        (rf"CFM--MPPI$^*$, $w_s={low_coef:g}$", {g: kazuki[low_coef] for g in GAMMAS}),
        (rf"CFM--MPPI$^*$, $w_s={high_coef:g}$", {g: kazuki[high_coef] for g in GAMMAS}),
    ]
    fig, axes = plt.subplots(5, 3, figsize=(15.0, 23.0), squeeze=False)
    for row_index, (label, cells) in enumerate(rows):
        for column_index, gamma in enumerate(GAMMAS):
            paths, outcomes = cells[gamma]
            draw_scene(
                axes[row_index, column_index], env, paths, outcomes, gamma,
                rf"$\gamma={gamma:g}$" if row_index == 0 else "",
                label if column_index == 0 else "",
            )
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.015, top=0.985, wspace=0.035, hspace=0.055)
    outputs: dict[str, str] = {}
    for suffix in ("png", "pdf"):
        path = args.outdir / f"b1_current_best_5x3_gallery.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        outputs[path.name] = sha256_file(path)
    plt.close(fig)

    for label, cells in (("expert", expert), ("raw_r0", raw0), ("raw_r19", raw19)):
        payload: dict[str, np.ndarray] = {}
        for gamma, (paths, outcomes) in cells.items():
            object_paths = np.empty(len(paths), dtype=object)
            object_paths[:] = paths
            payload[f"paths_g{gamma:g}"] = object_paths
            payload[f"outcomes_g{gamma:g}"] = np.asarray(outcomes)
        np.savez_compressed(args.outdir / f"{label}.npz", **payload)
    for coefficient in args.safe_coef_candidates:
        paths, outcomes = kazuki[coefficient]
        object_paths = np.empty(len(paths), dtype=object)
        object_paths[:] = paths
        np.savez_compressed(
            args.outdir / f"kazuki_ws_{coefficient:g}.npz",
            paths=object_paths, outcomes=np.asarray(outcomes),
        )

    manifest = {
        "status": "B1_CURRENT_BEST_GALLERY_COMPLETE",
        "metric_version": METRIC_VERSION,
        "scene_profile": profile.name,
        "scene_sha256": scene_snapshot(env, profile)["sha256"],
        "gammas": list(GAMMAS),
        "fixed_indices": list(FIXED_INDICES),
        "raw_rule": "first ten fixed indices from each authenticated disjoint M50 cell; no outcome curation",
        "failure_marker": "red X at terminal position for CR or timeout; no text label",
        "kazuki_definition": {
            "gamma_ctx": 0.5,
            "conditioning_schema": "low7_closest_boundary_tie_mean",
            "goal_coef": 0.0,
            "safe_coef_candidates": args.safe_coef_candidates,
            "selected_low": low_coef,
            "selected_high": high_coef,
            "selection_rule": (
                "predeclared original Kazuki safety-grid endpoints: minimum and maximum "
                "positive candidate; summaries are diagnostic and never select the rows"
            ),
            "important_control_note": (
                "goal_coef=safe_coef=0 is not raw pretrained: FlowMPPI elite selection, "
                "perturbation, collision/goal costs, and warm start remain active"
            ),
            "summaries": {str(key): value for key, value in summaries.items()},
        },
        "pretrained_checkpoint": str(args.pretrained_ckpt),
        "pretrained_checkpoint_sha256": sha256_file(args.pretrained_ckpt),
        "confirmation": str(args.confirmation),
        "rollout_source": (
            "new named fixed seeds" if args.reuse_rollouts is None
            else f"reused without regeneration from {args.reuse_rollouts}"
        ),
        "reused_rollout_sha256": (
            None if args.reuse_rollouts is None else {
                "expert.npz": sha256_file(args.reuse_rollouts / "expert.npz"),
                **{
                    f"kazuki_ws_{coefficient:g}.npz": sha256_file(
                        args.reuse_rollouts / f"kazuki_ws_{coefficient:g}.npz"
                    )
                    for coefficient in args.safe_coef_candidates
                },
            }
        ),
        "outputs": outputs,
    }
    manifest_path = args.outdir / "gallery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
