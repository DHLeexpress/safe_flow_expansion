#!/usr/bin/env python3
"""Trace and render four indexed B1 controller videos.

The visual semantics follow the authenticated SFM method video:

* SafeMPPI rows show the nominal polytope returned by the planner (blue).
* The raw B1 policy shows a candidate-specific full-H verifier polytope only
  when the generated window passes the offline verifier (green).  The verifier
  never selects the raw action.
* The Kazuki diagnostic has no nominal or verifier overlay.

Frame PNGs are deliberately retained for later paper snapshots.  This script
does not train or alter a model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_REV = _ROOT.parent
_WORK = _REV.parent
for _path in (_WORK, _REV, _ROOT, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import _paths  # noqa: F401
import afe_context as CX
import afe_core as AC
from afe2_scene_profiles import build_scene, get_scene_profile, scene_snapshot
from di_grid_viz import di_step
import grid_feats as GF
import grid_hp_expt as HP
import grid_metrics as GM
import grid_rollout as GR
import grid_scene as GS
from paper_results import low7_raw_m50_eval as RAW
from paper_results import low7_support_sweep_eval as SUPPORT
import verifier_polytope as VP


VIDEO_VERSION = "b1_indexed_controller_video_suite_v1"
SCHEMA = "low7_closest_boundary_tie_mean"
ID_PROFILE = "low7_id_canonical_v1"
OOD_PROFILE = "low7_radius1_canonical_v1"
REACH = 0.15
T = 300
NFE = 8
RAW_GAMMA = 0.5
RAW_FAILURE_INDEX = 22
KAZUKI_GAMMA = 0.5
KAZUKI_SAFE_COEF = 0.9
MARKUP_CANDIDATES = (1.01, 1.05, 1.09)
BLUE = "#1764ab"
GREEN = "#148f48"
RED = "#cc3311"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
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


def classify_path(path: np.ndarray, env: Any, reach: float = REACH) -> str:
    points = np.asarray(path, dtype=np.float64)
    goal = env.goal.detach().cpu().numpy()
    if np.linalg.norm(points[-1] - goal) < reach:
        return "SR"
    if not GM.in_taskspace(points):
        return "CR"
    obstacles = env.obstacles.detach().cpu().numpy()
    if obstacles.size:
        clearance = (
            np.linalg.norm(points[:, None] - obstacles[None, :, :2], axis=2)
            - obstacles[None, :, 2]
            - float(env.r_robot)
        )
        if float(clearance.min()) < 0.0:
            return "CR"
    return "TO"


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _plan_states(state: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
    positions = GR.window_positions(
        np.asarray(state, dtype=np.float32),
        np.asarray(controls, dtype=np.float32),
        dt,
    )
    return np.vstack((np.asarray(state[:2], dtype=np.float64), positions))


def _nominal_record(polytope: Any) -> dict[str, np.ndarray]:
    if polytope is None or len(polytope) < 4:
        raise RuntimeError("SafeMPPI did not return its nominal polytope")
    return {
        "A": _array(polytope[0]),
        "b": _array(polytope[1]),
        "margins": _array(polytope[3]),
    }


def run_safemppi_trace(
    env: Any,
    gamma: float,
    rollout_index: int,
    *,
    collect_trace: bool,
) -> dict[str, Any]:
    """Run the unmodified gallery SafeMPPI recipe and retain returned faces."""

    from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter

    seed = named_seed(
        "b1_current_best_gallery_native_cost_v2", "expert", gamma, rollout_index
    )
    seed_all(seed)
    adapter = SafeMPPIAdapter(**GS.mode1_config())
    state = env.x0.detach().cpu().numpy().astype(np.float32).copy()
    goal_t = env.goal.detach().cpu().float()
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    planner_obstacles = GS.planner_obstacles(env)
    path = [state[:2].copy()]
    trace: list[dict[str, Any]] = []
    for step in range(T):
        before = state.copy()
        action, info = adapter.plan(
            torch.as_tensor(before, dtype=torch.float32),
            goal_t,
            planner_obstacles,
            gamma=float(gamma),
            seed=seed + step,
        )
        action_np = action.detach().cpu().numpy().astype(np.float32)
        controls = np.asarray(info["mean_sequence"], dtype=np.float32)
        if collect_trace:
            trace.append(
                {
                    "step": step,
                    "state": before,
                    "action": action_np,
                    "plan": _plan_states(before, controls, float(env.dt)),
                    "nominal": _nominal_record(info.get("polytope")),
                }
            )
        state = di_step(before, action_np, dt=env.dt)
        path.append(state[:2].copy())
        if np.linalg.norm(state[:2] - goal) < REACH:
            break
        if not GM.in_taskspace(state[:2][None]):
            break
        if obstacles.size and float(
            (
                np.linalg.norm(state[:2][None] - obstacles[:, :2], axis=1)
                - obstacles[:, 2]
                - float(env.r_robot)
            ).min()
        ) < 0.0:
            break
    path_array = np.asarray(path, dtype=np.float32)
    return {
        "controller": "SafeMPPI",
        "gamma": float(gamma),
        "rollout_index": int(rollout_index),
        "seed": int(seed),
        "path": path_array,
        "outcome": classify_path(path_array, env),
        "trace": trace,
    }


def find_expert_failure(env: Any, gamma: float, bank_size: int) -> int:
    """Return the lowest failing index in a declared, outcome-independent bank."""

    for rollout_index in range(bank_size):
        result = run_safemppi_trace(
            env, gamma, rollout_index, collect_trace=False
        )
        print(
            f"[expert OOD search] gamma={gamma:g} index={rollout_index} "
            f"outcome={result['outcome']}",
            flush=True,
        )
        if result["outcome"] != "SR":
            return rollout_index
    raise RuntimeError(
        f"no unmodified SafeMPPI failure in declared {bank_size}-episode bank"
    )


def _serialize_faces(faces: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "a": _array(face.a),
            "m": float(face.m),
            "kind": str(face.kind),
            "feasible": bool(face.feasible),
        }
        for face in faces
    ]


@torch.no_grad()
def run_raw_failure_trace(
    policy: Any,
    env: Any,
    confirmation_cells: Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Replay the exact disjoint M50 r19 failure and retain its generated H10 plans."""

    noise, noise_meta = SUPPORT.holdout_noise_bank(
        OOD_PROFILE,
        int(policy.d),
        profile=SUPPORT.B1_HOLDOUT_PROFILE,
        study="b1",
    )
    target_gamma_index = RAW.GAMMAS.index(RAW_GAMMA)
    start = env.x0.detach().cpu().numpy().astype(np.float32).copy()
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    episodes = []
    for gamma_index, gamma in enumerate(RAW.GAMMAS):
        for rollout_index in range(SUPPORT.B1_HOLDOUT_PROFILE.m):
            episodes.append(
                {
                    "gamma_index": gamma_index,
                    "rollout_index": rollout_index,
                    "gamma": float(gamma),
                    "state": start.copy(),
                    "history": [],
                    "status": None,
                }
            )
    target = next(
        episode
        for episode in episodes
        if episode["gamma_index"] == target_gamma_index
        and episode["rollout_index"] == RAW_FAILURE_INDEX
    )
    path = [start[:2].copy()]
    trace: list[dict[str, Any]] = []
    for control_t in range(T):
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
                SCHEMA,
            )
            grids.append(record.grid)
            conditions.append(record.low5)
            histories.append(record.hist)
            noises.append(
                noise[
                    episode["gamma_index"],
                    episode["rollout_index"],
                    control_t,
                ]
            )
        grid = torch.as_tensor(np.asarray(grids, np.float32), device=device)
        condition = torch.as_tensor(
            np.asarray(conditions, np.float32), device=device
        )
        hist = torch.as_tensor(np.asarray(histories, np.float32), device=device)
        context = policy.ctx_from(grid, condition, hist)
        windows = policy.sample(
            len(active),
            context,
            nfe=NFE,
            temp=1.0,
            initial_noise=torch.as_tensor(np.asarray(noises), device=device),
        ).detach().cpu().numpy()
        for episode, controls in zip(active, windows):
            before = episode["state"].copy()
            controls = np.asarray(controls, dtype=np.float32)
            if episode is target:
                plan = _plan_states(before, controls, float(env.dt))
                verifier = AC.verify_plan(
                    before, controls, env, RAW_GAMMA, goal, n_theta=180
                )
                ok, faces, _, effective_radius = VP.certify_window(
                    plan,
                    obstacles,
                    float(env.r_robot),
                    RAW_GAMMA,
                    R=2.5,
                    n_theta=180,
                )
                if bool(ok) != bool(verifier["y"]):
                    raise RuntimeError(
                        "offline verifier face replay changed the full-H label"
                    )
                trace.append(
                    {
                        "step": control_t,
                        "state": before,
                        "action": controls[0].copy(),
                        "plan": plan,
                        "verifier_positive": bool(verifier["y"]),
                        "verifier_reason": str(verifier["reason"]),
                        "verifier_faces": _serialize_faces(faces),
                        "effective_radius": float(effective_radius),
                    }
                )
            action = controls[0]
            episode["state"] = di_step(before, action, dt=env.dt)
            episode["history"].append(action.copy())
            point = episode["state"][:2]
            if episode is target:
                path.append(point.copy())
            if np.linalg.norm(point - goal) < REACH:
                episode["status"] = "reached"
            elif (point < -GM.EPS_TASK).any() or (
                point > GM.GRID_M + GM.EPS_TASK
            ).any():
                episode["status"] = "oob"
            elif obstacles.size and float(
                (
                    np.linalg.norm(point[None] - obstacles[:, :2], axis=1)
                    - obstacles[:, 2]
                    - float(env.r_robot)
                ).min()
            ) < 0.0:
                episode["status"] = "collision"
    path_array = np.asarray(path, dtype=np.float32)
    archive = confirmation_cells / f"r019_g{RAW_GAMMA:.1f}.npz"
    with np.load(archive, allow_pickle=True) as stored:
        expected = np.asarray(stored["paths"][RAW_FAILURE_INDEX], dtype=np.float32)
        expected_status = str(stored["status"][RAW_FAILURE_INDEX])
    if path_array.shape != expected.shape or not np.allclose(
        path_array, expected, rtol=0.0, atol=2.0e-6
    ):
        raise RuntimeError("raw r19 trace does not reproduce its authenticated M50 path")
    outcome = classify_path(path_array, env)
    if outcome == "SR" or expected_status == "reached":
        raise RuntimeError("declared raw failure index no longer fails")
    return {
        "controller": "B1 current best r19 (raw temperature 1)",
        "gamma": RAW_GAMMA,
        "rollout_index": RAW_FAILURE_INDEX,
        "path": path_array,
        "outcome": outcome,
        "trace": trace,
        "noise_bank": noise_meta,
        "offline_verifier_only": True,
        "authenticated_path": str(archive),
    }


def configure_kazuki(markup: float) -> Any:
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
    baseline.MARKUP = float(markup)
    if baseline.REFINEMENT_COST != "b1_safemppi":
        raise RuntimeError("Kazuki refinement no longer uses the native B1 cost")
    return baseline


def run_kazuki_bank(
    policy: Any,
    env: Any,
    markup: float,
    m: int,
    *,
    device: str,
) -> dict[str, Any]:
    baseline = configure_kazuki(markup)
    rows = []
    for rollout_index in range(m):
        seed = named_seed(VIDEO_VERSION, "kazuki", KAZUKI_GAMMA, rollout_index)
        seed_all(seed)
        result = baseline.kazuki_deploy(
            policy,
            env,
            [KAZUKI_SAFE_COEF],
            gamma_ctx=KAZUKI_GAMMA,
            T=T,
            reach=REACH,
            device=device,
            seed=seed,
            conditioning_schema=SCHEMA,
        )
        path = np.asarray(result["path"], dtype=np.float32)
        rows.append(
            {
                "rollout_index": rollout_index,
                "seed": seed,
                "outcome": classify_path(path, env),
                "path": path,
            }
        )
        print(
            f"[Kazuki markup] m={markup:g} episode={rollout_index + 1}/{m} "
            f"outcome={rows[-1]['outcome']}",
            flush=True,
        )
    return {
        "markup": float(markup),
        "front_to_last_multiplier": float(markup ** 9),
        "rows": rows,
        "sr": sum(row["outcome"] == "SR" for row in rows) / m,
        "cr": sum(row["outcome"] == "CR" for row in rows) / m,
        "timeout": sum(row["outcome"] == "TO" for row in rows) / m,
    }


def select_kazuki_markup(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the best non-perfect fixed-bank SR, preferring lower markup on ties."""

    mixed = [row for row in results if 0.0 < row["sr"] < 1.0]
    if mixed:
        return max(mixed, key=lambda row: (row["sr"], -row["markup"]))
    nonperfect = [row for row in results if row["sr"] < 1.0]
    if not nonperfect:
        raise RuntimeError("Kazuki markup bank produced no failure for the requested video")
    return max(nonperfect, key=lambda row: (row["sr"], -row["markup"]))


def rerun_kazuki_failure(
    policy: Any,
    env: Any,
    selection: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    failure = next(row for row in selection["rows"] if row["outcome"] != "SR")
    baseline = configure_kazuki(selection["markup"])
    record: list[dict[str, Any]] = []
    seed_all(failure["seed"])
    result = baseline.kazuki_deploy(
        policy,
        env,
        [KAZUKI_SAFE_COEF],
        gamma_ctx=KAZUKI_GAMMA,
        T=T,
        reach=REACH,
        device=device,
        seed=failure["seed"],
        rec=record,
        conditioning_schema=SCHEMA,
    )
    path = np.asarray(result["path"], dtype=np.float32)
    if path.shape != failure["path"].shape or not np.allclose(
        path, failure["path"], rtol=0.0, atol=2.0e-6
    ):
        raise RuntimeError("Kazuki recorded rerun changed the selected fixed-bank path")
    trace = [
        {
            "step": step,
            "state": np.asarray(row["state"], dtype=np.float32),
            "plan": np.vstack((row["state"][:2], row["best"])),
            "candidates": np.asarray(row["cand"], dtype=np.float32),
            "refined": np.asarray(row["refined"], dtype=np.float32),
        }
        for step, row in enumerate(record)
    ]
    return {
        "controller": "CFM--MPPI native-cost diagnostic",
        "gamma": KAZUKI_GAMMA,
        "safe_coef": KAZUKI_SAFE_COEF,
        "goal_coef": 0.0,
        "markup": float(selection["markup"]),
        "rollout_index": int(failure["rollout_index"]),
        "seed": int(failure["seed"]),
        "path": path,
        "outcome": failure["outcome"],
        "trace": trace,
        "polytope": None,
    }


def clip_halfspaces(
    normals: np.ndarray,
    bounds: np.ndarray,
    *,
    box: tuple[float, float] = (-0.35, 5.35),
) -> np.ndarray:
    """Sutherland--Hodgman clipping for a convex 2-D halfspace intersection."""

    low, high = map(float, box)
    polygon = np.asarray(
        ((low, low), (high, low), (high, high), (low, high)), dtype=np.float64
    )
    for normal, bound in zip(np.asarray(normals), np.asarray(bounds)):
        if len(polygon) == 0:
            break
        output = []
        previous = polygon[-1]
        previous_value = float(normal @ previous - bound)
        for current in polygon:
            current_value = float(normal @ current - bound)
            previous_inside = previous_value <= 1.0e-9
            current_inside = current_value <= 1.0e-9
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                if abs(denominator) > 1.0e-14:
                    fraction = previous_value / denominator
                    output.append(previous + fraction * (current - previous))
            if current_inside:
                output.append(current)
            previous, previous_value = current, current_value
        polygon = np.asarray(output, dtype=np.float64).reshape(-1, 2)
    return polygon


def nominal_level_polygons(record: dict[str, Any], gamma: float, horizon: int = 10) -> list[np.ndarray]:
    A = np.asarray(record["A"], dtype=np.float64)
    b = np.asarray(record["b"], dtype=np.float64)
    margins = np.asarray(record["margins"], dtype=np.float64)
    return [
        clip_halfspaces(A, b - (1.0 - gamma) ** step * margins)
        for step in range(1, horizon + 1)
    ]


def verifier_level_polygons(
    faces: list[dict[str, Any]],
    center: np.ndarray,
    gamma: float,
    horizon: int = 10,
) -> list[np.ndarray]:
    feasible = [face for face in faces if face["feasible"]]
    A = np.asarray([face["a"] for face in feasible], dtype=np.float64)
    m = np.asarray([face["m"] for face in feasible], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)[:2]
    return [
        clip_halfspaces(
            A,
            A @ center + (1.0 - (1.0 - gamma) ** step) * m,
        )
        for step in range(1, horizon + 1)
    ]


def _draw_polygons(axis: Any, polygons: list[np.ndarray], color: str) -> None:
    for index, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        closed = np.vstack((polygon, polygon[0]))
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color=color,
            lw=0.7 + 1.5 * (index + 1) / len(polygons),
            alpha=0.28 + 0.62 * (index + 1) / len(polygons),
            zorder=4,
        )


def _draw_scene(axis: Any, env: Any) -> None:
    for obstacle in env.obstacles.detach().cpu().numpy():
        axis.add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#c8c8c8", zorder=1)
        )
    start = env.x0.detach().cpu().numpy()[:2]
    goal = env.goal.detach().cpu().numpy()
    axis.plot(*start, "ks", ms=7, zorder=10)
    axis.plot(*goal, marker="*", color="gold", mec="black", ms=17, zorder=10)
    axis.set_xlim(-0.35, 5.35)
    axis.set_ylim(-0.35, 5.35)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


@dataclass(frozen=True)
class VideoSpec:
    key: str
    method_title: str
    overlay: str
    env: Any
    episode: dict[str, Any]


def render_video(spec: VideoSpec, output_root: Path, *, stride: int, fps: int) -> dict[str, Any]:
    if stride <= 0 or fps <= 0:
        raise ValueError("frame stride and fps must be positive")
    frame_root = output_root / "frames" / spec.key
    frame_root.mkdir(parents=True, exist_ok=False)
    shown_steps = list(range(0, len(spec.episode["trace"]), stride))
    if shown_steps[-1] != len(spec.episode["trace"]) - 1:
        shown_steps.append(len(spec.episode["trace"]) - 1)
    path = np.asarray(spec.episode["path"], dtype=np.float64)
    outcome = str(spec.episode["outcome"])
    for frame_index, step in enumerate(shown_steps):
        row = spec.episode["trace"][step]
        figure, axis = plt.subplots(figsize=(8.6, 8.6))
        _draw_scene(axis, spec.env)
        axis.plot(path[:, 0], path[:, 1], color="0.78", lw=1.2, zorder=2)
        axis.plot(
            path[: step + 2, 0],
            path[: step + 2, 1],
            color="#54278f",
            lw=3.2,
            zorder=7,
        )
        prefix_dots = path[: step + 2 : 4]
        axis.plot(
            prefix_dots[:, 0], prefix_dots[:, 1], linestyle="none", marker=".",
            color="#54278f", ms=4.0, zorder=8,
        )
        plan = np.asarray(row["plan"], dtype=np.float64)
        plan_color = "black"
        if spec.overlay == "verifier" and not row["verifier_positive"]:
            plan_color = RED
        axis.plot(plan[:, 0], plan[:, 1], ls="--", lw=1.8, color=plan_color, zorder=6)
        axis.plot(*np.asarray(row["state"])[:2], marker="o", color="black", ms=7, zorder=11)
        if spec.overlay == "nominal":
            _draw_polygons(
                axis,
                nominal_level_polygons(row["nominal"], spec.episode["gamma"]),
                BLUE,
            )
        elif spec.overlay == "verifier" and row["verifier_positive"]:
            _draw_polygons(
                axis,
                verifier_level_polygons(
                    row["verifier_faces"], row["state"][:2], spec.episode["gamma"]
                ),
                GREEN,
            )
        if frame_index == len(shown_steps) - 1 and outcome != "SR":
            axis.plot(*path[-1], marker="x", color=RED, ms=13, mew=3, zorder=12)
        total = len(shown_steps) - 1
        figure.suptitle(
            rf"$\mathrm{{Frame}}\ {frame_index:03d}/{total:03d}\quad"
            rf"\mathrm{{control\ step}}\ {step:03d}$",
            fontsize=26,
            y=0.988,
        )
        axis.set_title(
            spec.method_title
            + "\n"
            + rf"$\gamma={spec.episode['gamma']:g}\qquad"
            + rf"\mathrm{{outcome}}={outcome}$",
            fontsize=21,
            pad=12,
        )
        handles = [
            Line2D([], [], color="#54278f", lw=3, label="executed prefix"),
            Line2D([], [], color="black", lw=1.8, ls="--", label=r"current $H=10$ plan"),
        ]
        if spec.overlay == "nominal":
            handles.append(Line2D([], [], color=BLUE, lw=2, label=r"nominal $H_P$ levels"))
        elif spec.overlay == "verifier":
            handles.extend(
                [
                    Line2D([], [], color=GREEN, lw=2, label="offline full-H verifier levels"),
                    Line2D([], [], color=RED, lw=1.8, ls="--", label="verifier-rejected plan"),
                ]
            )
        axis.legend(handles=handles, loc="upper left", fontsize=14, frameon=False)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        figure.savefig(frame_root / f"frame_{frame_index:06d}.png", dpi=160)
        plt.close(figure)
    video = output_root / f"{spec.key}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frame_root / "frame_%06d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", str(fps),
            "-pix_fmt", "yuv420p", "-c:v", "libx264", str(video),
        ],
        check=True,
    )
    preview = output_root / f"{spec.key}_preview.png"
    shutil.copyfile(
        frame_root / f"frame_{len(shown_steps) // 2:06d}.png", preview
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames",
                "-of", "json", str(video),
            ],
            text=True,
        )
    )
    return {
        "video": str(video),
        "video_sha256": sha256_file(video),
        "preview": str(preview),
        "preview_sha256": sha256_file(preview),
        "frame_directory": str(frame_root),
        "frames": len(shown_steps),
        "shown_control_steps": shown_steps,
        "ffprobe": probe,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-ckpt", type=Path, required=True)
    parser.add_argument("--latest-ckpt", type=Path, required=True)
    parser.add_argument("--confirmation-cells", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expert-ood-gamma", type=float, choices=(0.5, 1.0), default=0.5)
    parser.add_argument("--expert-ood-failure-index", type=int, default=None)
    parser.add_argument("--expert-search-size", type=int, default=100)
    parser.add_argument("--kazuki-m", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"fresh output directory required: {args.outdir}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    if args.torch_threads < 1:
        raise ValueError("torch thread count must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    args.outdir.mkdir(parents=True)
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": shutil.which("latex") is not None,
        }
    )
    id_profile = get_scene_profile(ID_PROFILE)
    ood_profile = get_scene_profile(OOD_PROFILE)
    id_env = build_scene(id_profile)
    ood_env = build_scene(ood_profile)

    id_expert = run_safemppi_trace(id_env, 0.5, 0, collect_trace=True)
    if id_expert["outcome"] != "SR":
        raise RuntimeError("fixed ID SafeMPPI episode no longer succeeds")
    failure_index = args.expert_ood_failure_index
    if failure_index is None:
        failure_index = find_expert_failure(
            ood_env, args.expert_ood_gamma, args.expert_search_size
        )
    ood_expert = run_safemppi_trace(
        ood_env, args.expert_ood_gamma, failure_index, collect_trace=True
    )
    if ood_expert["outcome"] == "SR":
        raise RuntimeError("declared OOD SafeMPPI failure index succeeds")

    latest, latest_payload = HP.load_hp(str(args.latest_ckpt), device="cpu")
    latest = latest.to(args.device).eval()
    raw_failure = run_raw_failure_trace(
        latest,
        ood_env,
        args.confirmation_cells,
        device=args.device,
    )

    pretrained, _ = HP.load_hp(str(args.pretrained_ckpt), device="cpu")
    pretrained = pretrained.to(args.device).eval()
    markup_results = [
        run_kazuki_bank(
            pretrained, ood_env, markup, args.kazuki_m, device=args.device
        )
        for markup in MARKUP_CANDIDATES
    ]
    selection = select_kazuki_markup(markup_results)
    kazuki_failure = rerun_kazuki_failure(
        pretrained, ood_env, selection, device=args.device
    )

    trace_root = args.outdir / "traces"
    trace_root.mkdir()
    episodes = {
        "01_safemppi_id_nominal": id_expert,
        "02_safemppi_ood_nominal_failure": ood_expert,
        "03_b1_r19_ood_verifier_audit_failure": raw_failure,
        "04_kazuki_ood_markup_failure": kazuki_failure,
    }
    for key, episode in episodes.items():
        torch.save(episode, trace_root / f"{key}.pt")

    specs = (
        VideoSpec(
            "01_safemppi_id_nominal", "SafeMPPI, in distribution", "nominal",
            id_env, id_expert,
        ),
        VideoSpec(
            "02_safemppi_ood_nominal_failure", "SafeMPPI, giant-obstacle OOD",
            "nominal", ood_env, ood_expert,
        ),
        VideoSpec(
            "03_b1_r19_ood_verifier_audit_failure", "B1 current best, giant-obstacle OOD",
            "verifier", ood_env, raw_failure,
        ),
        VideoSpec(
            "04_kazuki_ood_markup_failure", "CFM--MPPI native-cost diagnostic, OOD",
            "none", ood_env, kazuki_failure,
        ),
    )
    rendered = {
        spec.key: render_video(
            spec, args.outdir, stride=args.frame_stride, fps=args.fps
        )
        for spec in specs
    }
    manifest = {
        "status": "B1_INDEXED_CONTROLLER_VIDEO_SUITE_COMPLETE",
        "version": VIDEO_VERSION,
        "source_sha256": sha256_file(Path(__file__)),
        "reference_sfm_video_sha256": "a13a29c3c45a6a9f2c8a9eeb4ddf461b7666927ff73d2e4470e58e65ce7eb801",
        "semantics": {
            "nominal_blue": "actual SafeMPPI-returned nominal polytope; H=1..10 level sets",
            "verifier_green": "offline candidate-specific full-H verifier audit; never selected the raw action",
            "kazuki": "no polytope overlay",
            "frame_index": "large LaTeX/Computer-Modern frame and control-step index; numbered PNGs retained",
        },
        "runtime": {
            "torch_threads": args.torch_threads,
            "gpu_exclusivity_required": False,
            "timing_used_as_scientific_evidence": False,
        },
        "scene": {
            "id": scene_snapshot(id_env, id_profile),
            "ood": scene_snapshot(ood_env, ood_profile),
        },
        "checkpoints": {
            "pretrained": {
                "path": str(args.pretrained_ckpt),
                "sha256": sha256_file(args.pretrained_ckpt),
            },
            "latest_r19": {
                "path": str(args.latest_ckpt),
                "sha256": sha256_file(args.latest_ckpt),
                "iter": int(latest_payload.get("iter", -1)),
            },
        },
        "episodes": {
            key: {
                "gamma": episode["gamma"],
                "outcome": episode["outcome"],
                "rollout_index": episode.get("rollout_index"),
                "seed": episode.get("seed"),
                "steps": len(episode["path"]) - 1,
                "trace_sha256": sha256_file(trace_root / f"{key}.pt"),
            }
            for key, episode in episodes.items()
        },
        "expert_ood_selection": {
            "gamma": args.expert_ood_gamma,
            "declared_bank_size": args.expert_search_size,
            "selected_lowest_failure_index": failure_index,
            "controller_parameters_unchanged": True,
        },
        "kazuki_markup_sweep": {
            "gamma": KAZUKI_GAMMA,
            "safe_coef": KAZUKI_SAFE_COEF,
            "goal_coef": 0.0,
            "candidates": list(MARKUP_CANDIDATES),
            "paper_H80_equivalent_for_H10": float(1.01 ** (79.0 / 9.0)),
            "results": [
                {key: row[key] for key in ("markup", "front_to_last_multiplier", "sr", "cr", "timeout")}
                for row in markup_results
            ],
            "selected": float(selection["markup"]),
            "selection_rule": "highest non-perfect fixed-bank SR; prefer lower markup on ties",
            "refinement_cost": "exact native B1 SafeMPPI cost at all three refinement stages",
        },
        "rendered": rendered,
    }
    manifest_path = args.outdir / "video_suite_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"manifest": str(manifest_path), "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
