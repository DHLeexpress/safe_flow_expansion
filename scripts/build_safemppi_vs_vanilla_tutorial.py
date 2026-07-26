#!/usr/bin/env python3
"""Render the paper/tutorial SafeMPPI versus cost-only MPPI comparison.

The left panel runs the frozen SafeMPPI teacher unchanged.  All 512 proposal
rollouts are evaluated; only a deterministic diagnostic subset of 16 is drawn.
The right panel is an explicit soft-cost-only MPPI ablation: it uses the same
double-integrator, horizon, action bounds, native task/control costs, and MPPI
weighting, but never rejects a proposal.  A smooth obstacle-proximity cost is
the only safety mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_PAPER = (
    _REPO
    / "source_snapshot"
    / "overnight_run_07_06"
    / "rev_expansion"
    / "codex_overnight"
    / "paper_results"
)
_CORE = _PAPER.parent
_REV = _CORE.parent
_WORK = _REV.parent
for _path in (_WORK, _REV, _CORE, _PAPER):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from afe2_scene_profiles import build_scene, get_scene_profile, scene_snapshot
from b1_indexed_video_suite import (
    _draw_scene,
    classify_path,
    clip_halfspaces,
    nominal_level_polygons,
    seed_all,
)
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter, SafeMPPIConfig
from di_grid_viz import di_step
import grid_scene as GS


VIDEO_VERSION = "safemppi_vs_vanilla_mppi_tutorial_v1"
GAMMAS = (0.1, 0.5, 1.0)
GAMMA_COLORS = {
    gamma: plt.get_cmap("plasma")(
        {0.1: 0.08, 0.5: 0.52, 1.0: 0.92}[gamma]
    )
    for gamma in GAMMAS
}
POLYTOPE_BLUE = "#1764ab"
REJECTED_RED = "#cc3311"
ACCEPTED_GLOW = "#8ecae6"
VANILLA_VARIANTS = (
    {"name": "small", "sigma": 0.01, "color": "#009E73"},
    {"name": "medium", "sigma": 0.35, "color": "#E69F00"},
    {"name": "large", "sigma": 2.00, "color": "#7A5195"},
)
EXPECTED_VANILLA_OUTCOMES = {
    "small": "CR",
    "medium": "SR",
    "large": "CR",
}
T = 300
REACH = 0.15
DEBUG_ROLLOUTS = 16
PROXIMITY_WEIGHT = 100.0
PROXIMITY_BETA = 20.0
PROXIMITY_MARGIN = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def named_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**31 - 1
    )


def _step_batch(
    state: torch.Tensor,
    control: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    result = state.clone()
    result[:, 0] = (
        state[:, 0]
        + dt * state[:, 2]
        + 0.5 * dt * dt * control[:, 0]
    )
    result[:, 1] = (
        state[:, 1]
        + dt * state[:, 3]
        + 0.5 * dt * dt * control[:, 1]
    )
    result[:, 2] = state[:, 2] + dt * control[:, 0]
    result[:, 3] = state[:, 3] + dt * control[:, 1]
    return result


def _terminal(state: np.ndarray, path: np.ndarray, env: Any) -> bool:
    goal = env.goal.detach().cpu().numpy()
    if float(np.linalg.norm(state[:2] - goal)) < REACH:
        return True
    point = state[:2]
    if bool((point < 0.0).any() or (point > 5.0).any()):
        return True
    obstacles = env.obstacles.detach().cpu().numpy()
    if obstacles.size:
        clearance = (
            np.linalg.norm(point[None] - obstacles[:, :2], axis=1)
            - obstacles[:, 2]
            - float(env.r_robot)
        ).min()
        if float(clearance) < 0.0:
            return True
    return len(path) >= T + 1


def _nominal_record(polytope: Any) -> dict[str, np.ndarray]:
    if polytope is None or len(polytope) < 4:
        raise RuntimeError("SafeMPPI did not return its nominal polytope")
    return {
        "A": np.asarray(polytope[0], dtype=np.float64),
        "b": np.asarray(polytope[1], dtype=np.float64),
        "margins": np.asarray(polytope[3], dtype=np.float64),
    }


def _plan_states(
    state: np.ndarray,
    controls: np.ndarray,
    dt: float,
) -> np.ndarray:
    current = torch.as_tensor(state, dtype=torch.float32).view(1, 4)
    positions = [current[0, :2].numpy().copy()]
    for control in np.asarray(controls, dtype=np.float32):
        current = _step_batch(
            current,
            torch.as_tensor(control, dtype=torch.float32).view(1, 2),
            dt,
        )
        positions.append(current[0, :2].numpy().copy())
    return np.asarray(positions, dtype=np.float32)


def run_safemppi_episode(
    env: Any,
    gamma: float,
    rollout_index: int,
) -> dict[str, Any]:
    """Run the unchanged teacher while retaining a 16-plan draw subset."""

    seed = named_seed(
        "b1_current_best_gallery_native_cost_v2",
        "expert",
        gamma,
        rollout_index,
    )
    seed_all(seed)
    config = GS.mode1_config()
    config["debug_max_rollouts"] = DEBUG_ROLLOUTS
    adapter = SafeMPPIAdapter(**config)
    state = env.x0.detach().cpu().numpy().astype(np.float32).copy()
    goal_t = env.goal.detach().cpu().float()
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
            return_rollouts=True,
        )
        debug = info["debug_rollouts"]
        debug_states = np.asarray(debug["states"], dtype=np.float32)[..., :2]
        debug_feasible = np.asarray(debug["feasible"], dtype=bool)
        if len(debug_states) > DEBUG_ROLLOUTS:
            raise RuntimeError("SafeMPPI diagnostic subset exceeded 16 plans")
        controls = np.asarray(info["mean_sequence"], dtype=np.float32)
        trace.append(
            {
                "step": step,
                "state": before,
                "plan": _plan_states(before, controls, float(env.dt)),
                "nominal": _nominal_record(info["polytope"]),
                "rollouts": debug_states,
                "feasible": debug_feasible,
                "evaluated_samples": int(config["num_samples"]),
            }
        )
        action_np = action.detach().cpu().numpy().astype(np.float32)
        state = di_step(before, action_np, dt=env.dt)
        path.append(state[:2].copy())
        path_array = np.asarray(path, dtype=np.float32)
        if _terminal(state, path_array, env):
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


def cost_only_mppi_step(
    state: torch.Tensor,
    goal: torch.Tensor,
    obstacles: torch.Tensor,
    config: SafeMPPIConfig,
    sigma: float,
    warm_sequence: torch.Tensor | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """One MPPI update with soft obstacle cost and no feasibility mask."""

    horizon = int(config.horizon)
    count = int(config.num_samples)
    generator = torch.Generator(device=state.device)
    generator.manual_seed(int(seed))
    if warm_sequence is None:
        nominal = torch.zeros(
            horizon,
            2,
            dtype=state.dtype,
            device=state.device,
        )
    else:
        nominal = torch.cat(
            (warm_sequence[1:], warm_sequence[-1:]),
            dim=0,
        )
    controls = nominal.unsqueeze(0) + float(sigma) * torch.randn(
        count,
        horizon,
        2,
        generator=generator,
        dtype=state.dtype,
        device=state.device,
    )
    lower = torch.as_tensor(config.u_min, dtype=state.dtype, device=state.device)
    upper = torch.as_tensor(config.u_max, dtype=state.dtype, device=state.device)
    controls = torch.maximum(torch.minimum(controls, upper), lower)

    states = state.view(1, 4).expand(count, -1).clone()
    rollout_states = [states.clone()]
    costs = torch.zeros(count, dtype=state.dtype, device=state.device)
    initial_distance = torch.linalg.norm(states[:, :2] - goal[:2], dim=1)
    previous_action = torch.zeros(count, 2, dtype=state.dtype, device=state.device)
    radii = obstacles[:, 2] + float(PROXIMITY_MARGIN)
    for horizon_step in range(horizon):
        next_states = _step_batch(states, controls[:, horizon_step], float(config.dt))
        distance = torch.linalg.norm(next_states[:, :2] - goal[:2], dim=1)
        center_distance = torch.linalg.norm(
            next_states[:, None, :2] - obstacles[None, :, :2],
            dim=2,
        )
        nearest_clearance = (
            center_distance - radii.view(1, -1)
        ).min(dim=1).values
        smooth_proximity = PROXIMITY_WEIGHT * torch.clamp(
            torch.exp(-PROXIMITY_BETA * nearest_clearance),
            max=1.0,
        )
        costs += (
            float(config.running_goal_weight) * distance.square()
            + float(config.control_weight)
            * controls[:, horizon_step].square().sum(dim=1)
            + float(config.smooth_weight)
            * (controls[:, horizon_step] - previous_action).square().sum(dim=1)
            - float(config.progress_weight) * (initial_distance - distance)
            + smooth_proximity
        )
        states = next_states
        previous_action = controls[:, horizon_step]
        rollout_states.append(states.clone())
    costs += float(config.terminal_goal_weight) * torch.linalg.norm(
        states[:, :2] - goal[:2],
        dim=1,
    ).square()

    temperature = max(float(config.temperature), 1.0e-6)
    weights = torch.softmax(-(costs - costs.min()) / temperature, dim=0)
    mean_sequence = (weights.view(-1, 1, 1) * controls).sum(dim=0)
    diagnostic = (
        torch.stack(rollout_states, dim=1)[:DEBUG_ROLLOUTS, :, :2]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return mean_sequence[0], mean_sequence.detach(), diagnostic


def run_cost_only_episode(
    env: Any,
    variant: dict[str, Any],
    rollout_index: int,
) -> dict[str, Any]:
    config = SafeMPPIConfig(**GS.mode1_config())
    state = env.x0.detach().cpu().float().clone()
    goal = env.goal.detach().cpu().float()
    obstacles = GS.planner_obstacles(env).detach().cpu().float()
    path = [state[:2].numpy().copy()]
    trace: list[dict[str, Any]] = []
    warm_sequence: torch.Tensor | None = None
    for step in range(T):
        seed = named_seed(
            VIDEO_VERSION,
            "cost_only_common_noise",
            rollout_index,
            step,
        )
        action, warm_sequence, rollouts = cost_only_mppi_step(
            state,
            goal,
            obstacles,
            config,
            float(variant["sigma"]),
            warm_sequence,
            seed,
        )
        trace.append(
            {
                "step": step,
                "state": state.numpy().copy(),
                "rollouts": rollouts,
                "action": action.numpy().copy(),
            }
        )
        next_state = _step_batch(
            state.view(1, 4),
            action.view(1, 2),
            float(config.dt),
        )[0]
        state = next_state
        path.append(state[:2].numpy().copy())
        path_array = np.asarray(path, dtype=np.float32)
        if _terminal(state.numpy(), path_array, env):
            break
    path_array = np.asarray(path, dtype=np.float32)
    return {
        "controller": "cost-only vanilla MPPI",
        "variant": str(variant["name"]),
        "sigma": float(variant["sigma"]),
        "rollout_index": int(rollout_index),
        "path": path_array,
        "outcome": classify_path(path_array, env),
        "trace": trace,
    }


def _draw_nominal_polytope(
    axis: Any,
    record: dict[str, np.ndarray],
    gamma: float,
) -> None:
    """Draw the ten DTCBF levels and the explicit tangent-bounded H_P=0 face."""

    for index, polygon in enumerate(nominal_level_polygons(record, gamma)):
        if len(polygon) < 3:
            continue
        closed = np.vstack((polygon, polygon[0]))
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color=POLYTOPE_BLUE,
            lw=0.65 + 1.15 * (index + 1) / 10.0,
            alpha=0.20 + 0.48 * (index + 1) / 10.0,
            zorder=4,
        )
    outer = clip_halfspaces(record["A"], record["b"])
    if len(outer) >= 3:
        closed = np.vstack((outer, outer[0]))
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color=POLYTOPE_BLUE,
            lw=2.35,
            alpha=0.95,
            zorder=5,
        )


def _draw_safe_rollouts(axis: Any, row: dict[str, Any]) -> None:
    for rollout, feasible in zip(row["rollouts"], row["feasible"]):
        if bool(feasible):
            axis.plot(
                rollout[:, 0],
                rollout[:, 1],
                color=ACCEPTED_GLOW,
                lw=3.1,
                alpha=0.38,
                zorder=5,
            )
            axis.plot(
                rollout[:, 0],
                rollout[:, 1],
                color="black",
                ls="--",
                lw=1.15,
                alpha=0.82,
                zorder=6,
            )
        else:
            axis.plot(
                rollout[:, 0],
                rollout[:, 1],
                color=REJECTED_RED,
                ls="--",
                lw=1.05,
                alpha=0.68,
                zorder=5,
            )


def _draw_cost_only_rollouts(axis: Any, row: dict[str, Any]) -> None:
    for rollout in row["rollouts"]:
        axis.plot(
            rollout[:, 0],
            rollout[:, 1],
            color="#707070",
            ls="--",
            lw=0.95,
            alpha=0.38,
            zorder=4,
        )
        axis.plot(
            rollout[1:, 0],
            rollout[1:, 1],
            ls="none",
            marker="o",
            markerfacecolor="#f5f5f5",
            markeredgecolor="#707070",
            markeredgewidth=0.35,
            markersize=2.4,
            alpha=0.62,
            zorder=5,
        )


def _encode_video(frame_root: Path, output: Path, fps: int) -> dict[str, Any]:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_root / "frame_%06d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(output),
        ],
        check=True,
    )
    probe = json.loads(
        subprocess.check_output(
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
            text=True,
        )
    )
    return {"sha256": sha256_file(output), "ffprobe": probe}


def render_tutorial(
    env: Any,
    safe_episodes: dict[float, dict[str, Any]],
    vanilla_episodes: dict[str, dict[str, Any]],
    outdir: Path,
    fps: int,
    stride: int,
) -> dict[str, Any]:
    frame_root = outdir / "frames"
    frame_root.mkdir(parents=True)
    max_step = max(
        max(len(episode["trace"]) for episode in safe_episodes.values()),
        max(len(episode["trace"]) for episode in vanilla_episodes.values()),
    ) - 1
    shown_steps = list(range(0, max_step + 1, stride))
    if shown_steps[-1] != max_step:
        shown_steps.append(max_step)
    preview_index = min(
        range(len(shown_steps)),
        key=lambda index: abs(shown_steps[index] - 30),
    )

    for frame_index, step in enumerate(shown_steps):
        figure, axes = plt.subplots(1, 2, figsize=(16.8, 8.6))
        for axis in axes:
            _draw_scene(axis, env)

        for gamma, episode in safe_episodes.items():
            path = np.asarray(episode["path"], dtype=float)
            trace_index = min(step, len(episode["trace"]) - 1)
            prefix_end = min(step + 2, len(path))
            row = episode["trace"][trace_index]
            _draw_nominal_polytope(axes[0], row["nominal"], gamma)
            _draw_safe_rollouts(axes[0], row)
            color = GAMMA_COLORS[gamma]
            axes[0].plot(
                path[:prefix_end, 0],
                path[:prefix_end, 1],
                color=color,
                lw=3.0,
                zorder=9,
            )
            axes[0].plot(
                path[:prefix_end:4, 0],
                path[:prefix_end:4, 1],
                ls="none",
                marker="o",
                markerfacecolor=color,
                markeredgecolor="#666666",
                markeredgewidth=0.35,
                markersize=3.6,
                zorder=10,
            )

        for variant in VANILLA_VARIANTS:
            episode = vanilla_episodes[str(variant["name"])]
            path = np.asarray(episode["path"], dtype=float)
            trace_index = min(step, len(episode["trace"]) - 1)
            prefix_end = min(step + 2, len(path))
            _draw_cost_only_rollouts(axes[1], episode["trace"][trace_index])
            axes[1].plot(
                path[:prefix_end, 0],
                path[:prefix_end, 1],
                color=variant["color"],
                lw=3.0,
                zorder=9,
            )
            axes[1].plot(
                path[:prefix_end:4, 0],
                path[:prefix_end:4, 1],
                ls="none",
                marker="o",
                markerfacecolor=variant["color"],
                markeredgecolor="#555555",
                markeredgewidth=0.35,
                markersize=3.6,
                zorder=10,
            )
            if (
                step >= len(episode["trace"]) - 1
                and episode["outcome"] != "SR"
            ):
                axes[1].plot(
                    path[-1, 0],
                    path[-1, 1],
                    marker="x",
                    ls="none",
                    color=REJECTED_RED,
                    markersize=11,
                    markeredgewidth=2.6,
                    zorder=12,
                )

        axes[0].set_title(
            r"$\mathbf{SafeMPPI}$: nominal $H_P$ and hard rejection",
            fontsize=22,
            pad=10,
        )
        axes[1].set_title(
            r"$\mathbf{Vanilla\ MPPI}$: soft cost only, no rejection",
            fontsize=22,
            pad=10,
        )
        axes[0].legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=POLYTOPE_BLUE,
                    lw=2.4,
                    label=r"nominal $H_P$ ($H_P=0$ outer face included)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    ls="--",
                    lw=1.4,
                    label="accepted proposal",
                ),
                Line2D(
                    [0],
                    [0],
                    color=REJECTED_RED,
                    ls="--",
                    lw=1.4,
                    label="rejected proposal",
                ),
            ],
            loc="upper left",
            frameon=False,
            fontsize=12,
        )
        axes[1].legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=variant["color"],
                    lw=2.8,
                    label=(
                        rf"$\sigma={float(variant['sigma']):g}$ "
                        rf"({vanilla_episodes[str(variant['name'])]['outcome']})"
                    ),
                )
                for variant in VANILLA_VARIANTS
            ],
            loc="upper left",
            frameon=False,
            fontsize=13,
        )
        figure.suptitle(
            rf"$\mathrm{{Frame}}\ {frame_index:04d}$",
            fontsize=26,
            y=0.992,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.965))
        figure.savefig(
            frame_root / f"frame_{frame_index:06d}.png",
            dpi=150,
        )
        plt.close(figure)

    video = outdir / "safemppi_vs_vanilla_mppi.mp4"
    record = _encode_video(frame_root, video, fps)
    preview = outdir / "safemppi_vs_vanilla_mppi_preview.png"
    shutil.copyfile(
        frame_root / f"frame_{preview_index:06d}.png",
        preview,
    )
    return {
        "video": video.name,
        "preview": preview.name,
        "frames": len(shown_steps),
        "shown_steps": shown_steps,
        "preview_step": shown_steps[preview_index],
        "video_sha256": sha256_file(video),
        "preview_sha256": sha256_file(preview),
        **record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--rollout-index", type=int, default=12)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"fresh output directory required: {args.outdir}")
    if args.fps <= 0 or args.stride <= 0:
        raise ValueError("fps and stride must be positive")
    args.outdir.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": shutil.which("latex") is not None,
        }
    )

    profile_name = "low7_id_canonical_v1"
    env = build_scene(get_scene_profile(profile_name))
    safe_episodes = {
        gamma: run_safemppi_episode(env, gamma, args.rollout_index)
        for gamma in GAMMAS
    }
    if any(episode["outcome"] != "SR" for episode in safe_episodes.values()):
        raise RuntimeError("fixed SafeMPPI tutorial episode no longer succeeds")

    vanilla_episodes = {
        str(variant["name"]): run_cost_only_episode(
            env,
            variant,
            args.rollout_index,
        )
        for variant in VANILLA_VARIANTS
    }
    observed = {
        name: episode["outcome"]
        for name, episode in vanilla_episodes.items()
    }
    if observed != EXPECTED_VANILLA_OUTCOMES:
        raise RuntimeError(
            "cost-only diagnostic outcomes changed: "
            f"expected {EXPECTED_VANILLA_OUTCOMES}, observed {observed}"
        )

    rendered = render_tutorial(
        env,
        safe_episodes,
        vanilla_episodes,
        args.outdir,
        args.fps,
        args.stride,
    )
    safe_config = SafeMPPIConfig(**GS.mode1_config())
    manifest = {
        "status": "SAFEMPPI_VS_VANILLA_TUTORIAL_COMPLETE",
        "version": VIDEO_VERSION,
        "scene": scene_snapshot(env, get_scene_profile(profile_name)),
        "semantics": {
            "safe_mppi": (
                "frozen teacher; 512 proposals evaluated per state; "
                "deterministic 16-proposal accepted/rejected subset drawn"
            ),
            "nominal_polytope": (
                "blue for every gamma; ten DTCBF levels plus explicit "
                "tangent-bounded H_P=0 outer face"
            ),
            "vanilla_mppi": (
                "same DI/H/action bounds/native task-control costs and MPPI "
                "weighting; no feasibility mask or sample rejection"
            ),
            "vanilla_soft_safety": (
                "smooth proximity cost only; this is a diagnostic ablation, "
                "not a certified controller"
            ),
            "selection": (
                "fixed pedagogical rollout index 12 and predeclared sigmas; "
                "index 12 is the first cell in the declared [0,20) diagnostic "
                "bank with small/interior-collision, medium/success, and "
                "large/interior-collision outcomes"
            ),
        },
        "safe_mppi_recipe": {
            **vars(safe_config),
            "displayed_debug_rollouts": DEBUG_ROLLOUTS,
        },
        "vanilla_soft_cost": {
            "proximity_weight": PROXIMITY_WEIGHT,
            "proximity_beta": PROXIMITY_BETA,
            "proximity_margin": PROXIMITY_MARGIN,
        },
        "episodes": {
            "safe_mppi": {
                f"{gamma:g}": {
                    "rollout_index": episode["rollout_index"],
                    "seed": episode["seed"],
                    "outcome": episode["outcome"],
                    "steps": len(episode["path"]) - 1,
                }
                for gamma, episode in safe_episodes.items()
            },
            "vanilla_mppi": {
                name: {
                    "rollout_index": episode["rollout_index"],
                    "sigma": episode["sigma"],
                    "outcome": episode["outcome"],
                    "steps": len(episode["path"]) - 1,
                }
                for name, episode in vanilla_episodes.items()
            },
        },
        "rendered": rendered,
    }
    manifest_path = args.outdir / "tutorial_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
