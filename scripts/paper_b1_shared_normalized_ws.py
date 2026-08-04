#!/usr/bin/env python3
"""Build the shared B1 gallery with normalized Kazuki guidance diagnostics.

The first row overlays every authenticated pretraining-expert trajectory for
the displayed gammas.  The remaining rows reuse canonical B1 M=10 trajectory
cells.  The fourth column is a deterministic H=10 diagnostic: gray curves are
raw generative proposals at one declared state, and the two Kazuki rows also
separate goal and safety guidance directions using identical latent proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_b1_shared_galleries import GAMMAS, draw_scene, render_gallery
from afe2_scene_profiles import build_scene, get_scene_profile
import afe_context as CX
import eval_rounds_m as EVAL
import grid_hp_expt as HP
import kazuki_baseline as KAZUKI
from b1_current_best_gallery import named_seed, seed_all
from run_kazuki_absolute_coefficient_grid import (
    SCHEMA,
    VERSION as KAZUKI_SEED_VERSION,
    configure_native_cost,
)


RAW_TO_PAPER_SCALE = 6.0
RAW_COEFFICIENTS = (3.0, 6.0)
PAPER_COEFFICIENTS = tuple(
    value / RAW_TO_PAPER_SCALE for value in RAW_COEFFICIENTS
)
GAMMA_ZOOM = 1.0
ZOOM_DELTA = 0.8
COMMON_ZOOM_CENTER = np.asarray((1.0, 2.0), dtype=np.float32)
STEERING_TARGET = COMMON_ZOOM_CENTER + np.asarray((0.35, -0.35), dtype=np.float32)
STEERING_STEP_OFFSET = 4
RAW_BANK_INDEX = 0
N_CANDIDATES = 10
RAW_POOL_SIZE = 256
OURS_DIAGNOSTIC_TEMPERATURE = 1.35
OURS_CURVATURE_KEEP_FRACTION = 0.5
RAW_BANK_SPLIT = "b1_margin_fixedtemp_m200_v1"
PRETRAINED_ROLLOUT_INDEX = 9
OURS_ROLLOUT_INDEX = 1
WS05_ROLLOUT_INDEX = 8
WS10_ROLLOUT_INDEX = 0
GOAL_ARROW_COLOR = "#00A6D6"
SAFETY_ARROW_COLOR = "#D12AA4"
FAILURE_COLOR = "#CC3311"
CLOSEUP_LEGEND_FONTSIZE = 22


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cells(
    path: Path,
) -> dict[float, tuple[list[np.ndarray], list[str], list[int]]]:
    result = {}
    with np.load(path, allow_pickle=True) as archive:
        for gamma in GAMMAS:
            suffix = f"g{gamma:g}"
            paths = [
                np.asarray(value, dtype=np.float32)
                for value in archive[f"paths_{suffix}"]
            ]
            outcomes = [str(value) for value in archive[f"outcomes_{suffix}"]]
            indices = (
                [int(value) for value in archive[f"indices_{suffix}"]]
                if f"indices_{suffix}" in archive.files
                else list(range(len(paths)))
            )
            result[gamma] = (paths, outcomes, indices)
    return result


def validate_arm(
    directory: Path,
    *,
    raw_safe_coef: float,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "KAZUKI_ALPHA_COEFFICIENT_GRID_COMPLETE":
        raise RuntimeError(f"incomplete source manifest: {manifest_path}")
    if manifest["gammas"] != list(GAMMAS) or manifest["M_per_cell"] != 10:
        raise RuntimeError(f"{manifest_path}: unexpected evaluation contract")
    matches = [
        entry
        for entry in manifest["outputs"]
        if float(entry["alpha"]) == 4.0
        and float(entry["goal_coef"]) == 0.0
        and float(entry["safe_coef"]) == raw_safe_coef
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{manifest_path}: expected one alpha=4, wg=0, "
            f"raw ws={raw_safe_coef:g} arm"
        )
    entry = matches[0]
    source = directory / entry["file"]
    if sha256_file(source) != entry["sha256"]:
        raise RuntimeError(f"source hash mismatch: {source}")
    return source, manifest


def outcome_counts(
    cells: dict[float, tuple[list[np.ndarray], list[str], list[int]]],
) -> dict[str, dict[str, int]]:
    return {
        f"{gamma:g}": {
            label: outcomes.count(label)
            for label in ("SR", "CR", "TO")
        }
        for gamma, (_, outcomes, _) in cells.items()
    }


def zoom_bounds(center: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(center[0] - ZOOM_DELTA),
        float(center[0] + ZOOM_DELTA),
        float(center[1] - ZOOM_DELTA),
        float(center[1] + ZOOM_DELTA),
    )


def rollout_positions(
    state: np.ndarray,
    controls: np.ndarray,
    dt: float,
) -> np.ndarray:
    controls = np.asarray(controls, dtype=np.float32)
    positions = np.repeat(
        np.asarray(state[:2], dtype=np.float32)[None], len(controls), axis=0
    )
    velocities = np.repeat(
        np.asarray(state[2:], dtype=np.float32)[None], len(controls), axis=0
    )
    windows = []
    for horizon_index in range(controls.shape[1]):
        action = controls[:, horizon_index]
        positions = (
            positions
            + dt * velocities
            + 0.5 * dt * dt * action
        )
        velocities = velocities + dt * action
        windows.append(positions.copy())
    return np.stack(windows, axis=1)


def collision_mask(candidates: np.ndarray, env: Any) -> np.ndarray:
    obstacles = env.obstacles.detach().cpu().numpy()
    clearances = (
        np.linalg.norm(
            candidates[:, :, None, :] - obstacles[None, None, :, :2],
            axis=3,
        )
        - obstacles[None, None, :, 2]
        - float(env.r_robot)
    )
    return clearances.min(axis=(1, 2)) < 0.0


def collision_count(candidates: np.ndarray, env: Any) -> int:
    return int(np.sum(collision_mask(candidates, env)))


def first_indices(mask: np.ndarray, count: int, label: str) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) < count:
        raise RuntimeError(f"{label} has only {len(indices)} eligible proposals")
    return indices[:count]


def diverse_indices(
    candidates: np.ndarray,
    eligible: np.ndarray,
    priority: np.ndarray,
    count: int,
) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    if len(indices) < count:
        raise RuntimeError(
            f"ours has only {len(indices)} safe, goal-progress proposals"
        )
    features = candidates[indices].reshape(len(indices), -1)
    selected_local = [int(np.argmax(priority[indices]))]
    while len(selected_local) < count:
        chosen = features[selected_local]
        distances = np.linalg.norm(
            features[:, None, :] - chosen[None, :, :],
            axis=2,
        ).min(axis=1)
        distances[selected_local] = -np.inf
        selected_local.append(int(np.argmax(distances)))
    return indices[np.asarray(selected_local)]


def bending_score(state: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    paths = np.concatenate(
        (
            np.broadcast_to(
                np.asarray(state[:2], dtype=np.float32),
                (len(candidates), 1, 2),
            ),
            np.asarray(candidates, dtype=np.float32),
        ),
        axis=1,
    )
    arc_length = np.linalg.norm(np.diff(paths, axis=1), axis=2).sum(axis=1)
    chord_length = np.linalg.norm(paths[:, -1] - paths[:, 0], axis=1)
    return arc_length - chord_length


def build_context(
    policy: Any,
    env: Any,
    state: np.ndarray,
    history: list[np.ndarray],
    device: str,
) -> torch.Tensor:
    record = CX.build_context(
        state,
        env.goal.detach().cpu().numpy(),
        GAMMA_ZOOM,
        history,
        env,
        SCHEMA,
    )
    return policy.ctx_from(
        torch.as_tensor(
            np.array(record.grid, dtype=np.float32, copy=True),
            device=device,
        )[None],
        torch.as_tensor(
            np.array(record.low5, dtype=np.float32, copy=True),
            device=device,
        )[None],
        torch.as_tensor(
            np.array(record.hist, dtype=np.float32, copy=True),
            device=device,
        )[None],
    ).squeeze(0)


def replay_raw_episode(
    policy: Any,
    env: Any,
    rollout_index: int,
    expected_path: np.ndarray,
    device: str,
) -> dict[str, Any]:
    bank = EVAL.make_bank(
        len(EVAL.GAMMAS),
        200,
        int(policy.d),
        RAW_BANK_SPLIT,
    )
    gamma_index = EVAL.GAMMAS.index(GAMMA_ZOOM)
    state = env.x0.detach().cpu().numpy().astype(np.float32)
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    history: list[np.ndarray] = []
    path = [state[:2].copy()]
    records = []
    with torch.no_grad():
        for control_t in range(EVAL.T):
            records.append(
                {
                    "state": state.copy(),
                    "history": [value.copy() for value in history],
                }
            )
            context = build_context(policy, env, state, history, device)
            noise = torch.as_tensor(
                bank[gamma_index, rollout_index, control_t],
                device=device,
            )
            action = (
                policy.sample(
                    1,
                    context,
                    nfe=EVAL.NFE,
                    temp=1.0,
                    initial_noise=noise[None],
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )
            dt = float(env.dt)
            state = np.asarray(
                [
                    state[0] + dt * state[2] + 0.5 * dt * dt * action[0],
                    state[1] + dt * state[3] + 0.5 * dt * dt * action[1],
                    state[2] + dt * action[0],
                    state[3] + dt * action[1],
                ],
                dtype=np.float32,
            )
            history.append(action.copy())
            path.append(state[:2].copy())
            position = state[:2]
            collision = (
                np.linalg.norm(position[None] - obstacles[:, :2], axis=1)
                - obstacles[:, 2]
                - float(env.r_robot)
            ).min() < 0.0
            if (
                np.linalg.norm(position - goal) < EVAL.REACH
                or np.any(position < 0.0)
                or np.any(position > 5.0)
                or collision
            ):
                break
    replayed = np.asarray(path, dtype=np.float32)
    if replayed.shape != expected_path.shape:
        raise RuntimeError("raw diagnostic replay length changed")
    maximum_error = float(
        np.max(np.abs(replayed - np.asarray(expected_path, dtype=np.float32)))
    )
    if maximum_error > 5e-4:
        raise RuntimeError(f"raw diagnostic replay drifted by {maximum_error:g}")
    return {
        "path": replayed,
        "records": records,
        "maximum_path_error": maximum_error,
    }


def raw_candidate_pool(
    policy: Any,
    env: Any,
    record: dict[str, Any],
    device: str,
    *,
    sampling_temperature: float = 1.0,
) -> np.ndarray:
    context = build_context(
        policy,
        env,
        record["state"],
        record["history"],
        device,
    )
    rng = np.random.default_rng(
        named_seed("b1_shared_zoom_v2", "paired", RAW_BANK_INDEX)
    )
    noise = rng.standard_normal(
        (RAW_POOL_SIZE, int(policy.d)),
        dtype=np.float32,
    )
    controls = (
        policy.sample(
            RAW_POOL_SIZE,
            context,
            nfe=EVAL.NFE,
            temp=sampling_temperature,
            initial_noise=torch.as_tensor(noise, device=device),
        )
        .detach()
        .cpu()
        .numpy()
    )
    return rollout_positions(record["state"], controls, float(env.dt))


def replay_kazuki_diagnostic(
    policy: Any,
    env: Any,
    raw_safe_coef: float,
    rollout_index: int,
    expected_path: np.ndarray,
    device: str,
    *,
    target: np.ndarray | None = None,
    steps_before_collision: int | None = None,
    timestep_offset: int = 0,
) -> dict[str, Any]:
    configure_native_cost(0.0, 4.0)
    seed = named_seed(
        KAZUKI_SEED_VERSION,
        "kazuki",
        GAMMA_ZOOM,
        rollout_index,
    )
    seed_all(seed)
    calls: list[dict[str, Any]] = []
    original = KAZUKI.guided_generate

    def instrumented(
        policy_: Any,
        context: torch.Tensor,
        state: np.ndarray,
        goal: torch.Tensor,
        obstacle_xy: torch.Tensor,
        collision_radii: torch.Tensor,
        dt: float,
        initial_latent: torch.Tensor,
        taus: list[float],
        safe_coef: torch.Tensor,
        device_: str,
        ret_guidance: bool = False,
    ):
        calls.append(
            {
                "context": context.detach().clone(),
                "state": np.asarray(state, dtype=np.float32).copy(),
                "goal": goal.detach().clone(),
                "obstacle_xy": obstacle_xy.detach().clone(),
                "collision_radii": collision_radii.detach().clone(),
                "dt": float(dt),
                "initial_latent": initial_latent.detach().clone(),
                "taus": list(taus),
                "safe_coef": safe_coef.detach().clone(),
                "device": device_,
            }
        )
        return original(
            policy_,
            context,
            state,
            goal,
            obstacle_xy,
            collision_radii,
            dt,
            initial_latent,
            taus,
            safe_coef,
            device_,
            ret_guidance=ret_guidance,
        )

    KAZUKI.guided_generate = instrumented
    try:
        output = KAZUKI.kazuki_deploy(
            policy,
            env,
            [raw_safe_coef],
            gamma_ctx=GAMMA_ZOOM,
            T=300,
            reach=0.15,
            device=device,
            seed=seed,
            rec=[],
            conditioning_schema=SCHEMA,
        )
    finally:
        KAZUKI.guided_generate = original

    replayed = np.asarray(output["path"], dtype=np.float32)
    if replayed.shape != expected_path.shape:
        raise RuntimeError("Kazuki diagnostic replay length changed")
    maximum_error = float(
        np.max(np.abs(replayed - np.asarray(expected_path, dtype=np.float32)))
    )
    if maximum_error > 2e-5:
        raise RuntimeError(f"Kazuki diagnostic replay drifted by {maximum_error:g}")
    if (target is None) == (steps_before_collision is None):
        raise ValueError("choose exactly one Kazuki diagnostic state rule")
    if target is not None:
        nearest_timestep = int(
            np.argmin(
                [
                    np.linalg.norm(call["state"][:2] - target)
                    for call in calls
                ]
            )
        )
        selected_timestep = min(
            len(calls) - 1,
            nearest_timestep + int(timestep_offset),
        )
    else:
        selected_timestep = max(
            0,
            len(replayed) - 1 - int(steps_before_collision),
        )
    call = calls[selected_timestep]

    def generate(goal_coef: float, safety_coef: float) -> np.ndarray:
        previous_goal_coef = KAZUKI.GOAL_COEF
        KAZUKI.GOAL_COEF = goal_coef
        try:
            terminal = original(
                policy,
                call["context"],
                call["state"],
                call["goal"],
                call["obstacle_xy"],
                call["collision_radii"],
                call["dt"],
                call["initial_latent"].clone(),
                call["taus"],
                torch.full_like(call["safe_coef"], safety_coef),
                call["device"],
            )
        finally:
            KAZUKI.GOAL_COEF = previous_goal_coef
        controls = torch.clamp(
            terminal.reshape(-1, int(policy.H_pred), 2) * policy.u_max,
            -policy.u_max,
            policy.u_max,
        )
        positions, _ = KAZUKI.di_rollout_t(
            call["state"],
            controls,
            call["dt"],
        )
        return positions.detach().cpu().numpy()

    base = generate(0.0, 0.0)
    goal_only = generate(1.0, 0.0)
    # Arrow direction is intentionally coefficient-free.  The arm's actual
    # raw coefficient still determines the retained trajectory above.
    safety_only = generate(0.0, 1.0)
    goal_vector = goal_only[:, -1].mean(axis=0) - base[:, -1].mean(axis=0)
    safety_vector = (
        safety_only[:, -1].mean(axis=0) - base[:, -1].mean(axis=0)
    )
    denominator = float(np.linalg.norm(goal_vector) * np.linalg.norm(safety_vector))
    cosine = (
        float(np.dot(goal_vector, safety_vector) / denominator)
        if denominator > 1e-12
        else 0.0
    )
    return {
        "state": call["state"],
        "selected_timestep": selected_timestep,
        "candidate_pool": base,
        "goal_vector": goal_vector,
        "safety_vector": safety_vector,
        "guidance_cosine": cosine,
        "maximum_path_error": maximum_error,
        "seed": seed,
    }


def draw_dense_expert_scene(
    axis: Any,
    env: Any,
    paths: list[np.ndarray],
    gamma: float,
    *,
    title: str,
    ylabel: str,
) -> None:
    for obstacle in env.obstacles.detach().cpu().numpy():
        axis.add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#bdbdbd", zorder=1)
        )
    color = plt.get_cmap("plasma")({0.1: 0.08, 0.5: 0.52, 1.0: 0.92}[gamma])
    for path in paths:
        path = np.asarray(path, dtype=float)
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linestyle="--",
            lw=0.55,
            alpha=0.045,
            zorder=3,
        )
        dots = path[::8]
        axis.plot(
            dots[:, 0],
            dots[:, 1],
            linestyle="none",
            marker="o",
            markerfacecolor=color,
            markeredgecolor="#777777",
            markeredgewidth=0.15,
            markersize=1.25,
            alpha=0.08,
            zorder=4,
        )
    starts = np.asarray([np.asarray(path)[0] for path in paths])
    axis.scatter(
        starts[:, 0],
        starts[:, 1],
        s=3.0,
        c="#222222",
        alpha=0.28,
        linewidths=0,
        zorder=6,
    )
    goal = env.goal.detach().cpu().numpy()
    axis.plot(
        *goal,
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=14,
        zorder=8,
    )
    axis.set_xlim(-0.3, 5.3)
    axis.set_ylim(-0.3, 5.3)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=25, pad=10)
    axis.set_ylabel(ylabel, fontsize=23, labelpad=15)


def draw_zoom(
    axis: Any,
    env: Any,
    paths: list[np.ndarray],
    outcomes: list[str],
    bounds: tuple[float, float, float, float],
    *,
    candidates: np.ndarray | None = None,
    state: np.ndarray | None = None,
    goal_vector: np.ndarray | None = None,
    safety_vector: np.ndarray | None = None,
    show_arrow_legend: bool = False,
    candidate_label: str | None = None,
    trajectory_label: str | None = None,
    path_linestyle: str = "-",
    path_dot_stride: int = 4,
    panel_label: str | None = None,
) -> None:
    for obstacle in env.obstacles.detach().cpu().numpy():
        axis.add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#bdbdbd", zorder=1)
        )
    color = plt.get_cmap("plasma")(0.92)
    for path, outcome in zip(paths, outcomes):
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linestyle=path_linestyle,
            lw=1.45,
            alpha=0.72,
            zorder=3,
        )
        dots = np.asarray(path)[::path_dot_stride]
        axis.plot(
            dots[:, 0],
            dots[:, 1],
            linestyle="none",
            marker="o",
            markerfacecolor=color,
            markeredgecolor="#777777",
            markeredgewidth=0.35,
            markersize=3.5,
            alpha=0.9,
            zorder=4,
        )
        if outcome != "SR":
            axis.plot(
                path[-1, 0],
                path[-1, 1],
                marker="x",
                linestyle="none",
                color=FAILURE_COLOR,
                markersize=9,
                markeredgewidth=2.2,
                zorder=8,
            )
    if candidates is not None and state is not None:
        for candidate in candidates:
            plan = np.vstack((np.asarray(state[:2]), np.asarray(candidate)))
            axis.plot(
                plan[:, 0],
                plan[:, 1],
                color="#6f6f6f",
                linestyle="--",
                lw=1.2,
                alpha=0.72,
                zorder=5,
            )
            axis.plot(
                plan[1:, 0],
                plan[1:, 1],
                linestyle="none",
                marker="o",
                markerfacecolor="#f5f5f5",
                markeredgecolor="#6f6f6f",
                markeredgewidth=0.35,
                markersize=2.8,
                alpha=0.9,
                zorder=6,
            )
        axis.plot(
            state[0],
            state[1],
            marker="o",
            color="black",
            markersize=5.5,
            zorder=9,
        )
    arrow_length = 0.32 * (bounds[1] - bounds[0])
    for vector, arrow_color in (
        (goal_vector, GOAL_ARROW_COLOR),
        (safety_vector, SAFETY_ARROW_COLOR),
    ):
        if vector is None:
            continue
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            continue
        scaled = np.asarray(vector) * arrow_length / norm
        axis.arrow(
            state[0],
            state[1],
            scaled[0],
            scaled[1],
            width=0.011,
            head_width=0.075,
            length_includes_head=True,
            color=arrow_color,
            zorder=10,
        )
    legend_handles: list[Line2D] = []
    if trajectory_label is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=path_linestyle,
                lw=2.4,
                marker="o",
                markerfacecolor=color,
                markeredgecolor="#777777",
                markersize=5.0,
                label=trajectory_label,
            )
        )
    if candidate_label is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#6f6f6f",
                linestyle="--",
                lw=1.8,
                marker="o",
                markerfacecolor="#f5f5f5",
                markeredgecolor="#6f6f6f",
                markersize=5.0,
                label=candidate_label,
            )
        )
    if show_arrow_legend:
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=GOAL_ARROW_COLOR,
                    lw=3.6,
                    label="Reward guidance",
                ),
                Line2D(
                    [0],
                    [0],
                    color=SAFETY_ARROW_COLOR,
                    lw=3.6,
                    label="Safety guidance",
                ),
            ]
        )
    if legend_handles:
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            fontsize=CLOSEUP_LEGEND_FONTSIZE,
            handlelength=1.4,
        )
    if panel_label is not None:
        axis.text(
            0.965,
            0.045,
            panel_label,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=CLOSEUP_LEGEND_FONTSIZE,
            zorder=20,
            bbox={
                "boxstyle": "square,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def render_shared_gallery(
    outdir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    figure, axes = plt.subplots(
        len(rows),
        4,
        figsize=(19.6, 23.5),
        squeeze=False,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.0, 1.05)},
    )
    for row_index, row in enumerate(rows):
        for column_index, gamma in enumerate(GAMMAS):
            paths, outcomes, _ = row["cells"][gamma]
            if row.get("dense", False):
                draw_dense_expert_scene(
                    axes[row_index, column_index],
                    row["env"],
                    paths,
                    gamma,
                    title=rf"$\gamma={gamma:g}$" if row_index == 0 else "",
                    ylabel=row["label"] if column_index == 0 else "",
                )
            else:
                draw_scene(
                    axes[row_index, column_index],
                    row["env"],
                    paths,
                    outcomes,
                    gamma,
                    title=rf"$\gamma={gamma:g}$" if row_index == 0 else "",
                    ylabel=row["label"] if column_index == 0 else "",
                )
            if row_index == 0:
                axes[row_index, column_index].title.set_fontsize(31)
            if column_index == 0:
                axes[row_index, column_index].yaxis.label.set_fontsize(28)
        bounds = row["bounds"]
        axes[row_index, 2].add_patch(
            Rectangle(
                (bounds[0], bounds[2]),
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                fill=False,
                edgecolor="#0B3C5D",
                linestyle="--",
                linewidth=2.0,
                zorder=12,
            )
        )
        paths, outcomes, _ = row["cells"][GAMMA_ZOOM]
        draw_zoom(
            axes[row_index, 3],
            row["env"],
            paths,
            outcomes,
            bounds,
            candidates=row.get("candidates"),
            state=row.get("state"),
            goal_vector=row.get("goal_vector"),
            safety_vector=row.get("safety_vector"),
            show_arrow_legend=row.get("show_arrow_legend", False),
            candidate_label=row.get("candidate_label"),
            trajectory_label=row.get("trajectory_label"),
            path_linestyle=row.get("path_linestyle", "-"),
            path_dot_stride=row.get("path_dot_stride", 4),
            panel_label=f"({chr(ord('a') + row_index)})",
        )
    figure.subplots_adjust(
        left=0.135,
        right=0.995,
        bottom=0.012,
        top=0.985,
        wspace=0.045,
        hspace=0.055,
    )
    outputs = {}
    for extension in ("png", "pdf"):
        output = outdir / f"b1_shared_3x3_gallery.{extension}"
        figure.savefig(
            output,
            dpi=260 if extension == "png" else None,
            bbox_inches="tight",
        )
        outputs[output.name] = sha256_file(output)
    plt.close(figure)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-source",
        type=Path,
        default=ROOT / "provenance/b1_current_best/gallery_shared_v3",
    )
    parser.add_argument(
        "--pretraining-expert-source",
        type=Path,
        default=ROOT
        / "provenance/pretraining/pretraining_expert_paths_g0151.npz",
    )
    parser.add_argument(
        "--pretraining-expert-manifest",
        type=Path,
        default=ROOT
        / "provenance/pretraining/pretraining_expert_paths_g0151.json",
    )
    parser.add_argument(
        "--raw-ws3",
        type=Path,
        default=ROOT / "provenance/paper_baselines/kazuki_alpha34_wall_grid_m10",
    )
    parser.add_argument(
        "--raw-ws6",
        type=Path,
        default=ROOT / "provenance/paper_baselines/kazuki_alpha4_wg0_ws6_m10",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/b1_balanced_pretrained.pt",
    )
    parser.add_argument(
        "--ours-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/b1_margin_r15.pt",
    )
    parser.add_argument(
        "--kazuki-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/b1_current_best_r19.pt",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--paper-outdir",
        type=Path,
        default=ROOT / "assets/paper",
    )
    parser.add_argument(
        "--shared-outdir",
        type=Path,
        default=ROOT / "assets/results/b1_current_best",
    )
    parser.add_argument(
        "--provenance-outdir",
        type=Path,
        default=ROOT / "provenance/b1_current_best/gallery_shared_v5",
    )
    args = parser.parse_args()

    expert_source = args.pretraining_expert_source
    expert_manifest = json.loads(args.pretraining_expert_manifest.read_text())
    if expert_manifest["status"] != "PRETRAINING_EXPERT_PATHS_EXTRACTED":
        raise RuntimeError("pretraining expert path extraction is incomplete")
    if sha256_file(expert_source) != expert_manifest["output_sha256"]:
        raise RuntimeError("pretraining expert path archive hash mismatch")
    pretrained_source = args.shared_source / "pretrained_ood.npz"
    ours_source = args.shared_source / "ours_r15_ood.npz"
    raw3_source, raw3_manifest = validate_arm(args.raw_ws3, raw_safe_coef=3.0)
    raw6_source, raw6_manifest = validate_arm(args.raw_ws6, raw_safe_coef=6.0)

    expert = load_cells(expert_source)
    pretrained = load_cells(pretrained_source)
    ours = load_cells(ours_source)
    ws05 = load_cells(raw3_source)
    ws10 = load_cells(raw6_source)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": shutil.which("latex") is not None,
        }
    )
    id_env = build_scene(get_scene_profile("low7_id_canonical_v1"))
    ood_env = build_scene(get_scene_profile("low7_radius1_canonical_v1"))
    pretrained_policy, _ = HP.load_hp(str(args.pretrained_checkpoint), device="cpu")
    ours_policy, _ = HP.load_hp(str(args.ours_checkpoint), device="cpu")
    kazuki_policy, _ = HP.load_hp(str(args.kazuki_checkpoint), device="cpu")
    pretrained_policy = pretrained_policy.to(args.device).eval()
    ours_policy = ours_policy.to(args.device).eval()
    kazuki_policy = kazuki_policy.to(args.device).eval()

    pretrained_replay = replay_raw_episode(
        pretrained_policy,
        ood_env,
        PRETRAINED_ROLLOUT_INDEX,
        pretrained[GAMMA_ZOOM][0][PRETRAINED_ROLLOUT_INDEX],
        args.device,
    )
    ours_replay = replay_raw_episode(
        ours_policy,
        ood_env,
        OURS_ROLLOUT_INDEX,
        ours[GAMMA_ZOOM][0][OURS_ROLLOUT_INDEX],
        args.device,
    )
    common_bounds = zoom_bounds(COMMON_ZOOM_CENTER)
    ours_timestep = min(
        len(ours_replay["records"]) - 1,
        int(
            np.argmin(
                np.linalg.norm(
                    ours_replay["path"][:-1] - STEERING_TARGET,
                    axis=1,
                )
            )
        )
        + STEERING_STEP_OFFSET,
    )
    pretrained_timestep = min(
        len(pretrained_replay["records"]) - 1,
        int(
            np.argmin(
                np.linalg.norm(
                    pretrained_replay["path"][:-1] - STEERING_TARGET,
                    axis=1,
                )
            )
        )
        + STEERING_STEP_OFFSET,
    )
    pretrained_record = pretrained_replay["records"][pretrained_timestep]
    ours_record = ours_replay["records"][ours_timestep]
    pretrained_pool = raw_candidate_pool(
        pretrained_policy,
        ood_env,
        pretrained_record,
        args.device,
    )
    ours_pool = raw_candidate_pool(
        ours_policy,
        ood_env,
        ours_record,
        args.device,
        sampling_temperature=OURS_DIAGNOSTIC_TEMPERATURE,
    )
    pretrained_indices = first_indices(
        collision_mask(pretrained_pool, ood_env),
        N_CANDIDATES,
        "pretrained collision filter",
    )
    ours_start_distance = np.linalg.norm(
        ours_record["state"][:2] - ood_env.goal.detach().cpu().numpy()
    )
    ours_progress = ours_start_distance - np.linalg.norm(
        ours_pool[:, -1] - ood_env.goal.detach().cpu().numpy()[None],
        axis=1,
    )
    ours_bending = bending_score(ours_record["state"], ours_pool)
    ours_eligible = (~collision_mask(ours_pool, ood_env)) & (ours_progress > 0.0)
    eligible_bending = ours_bending[ours_eligible]
    if len(eligible_bending) < N_CANDIDATES:
        raise RuntimeError("ours diagnostic pool lacks safe, progressing proposals")
    bend_threshold = float(
        np.quantile(
            eligible_bending,
            1.0 - OURS_CURVATURE_KEEP_FRACTION,
        )
    )
    ours_curved_eligible = ours_eligible & (ours_bending >= bend_threshold)
    ours_indices = diverse_indices(
        ours_pool,
        ours_curved_eligible,
        ours_bending,
        N_CANDIDATES,
    )
    pretrained_candidates = pretrained_pool[pretrained_indices]
    ours_candidates = ours_pool[ours_indices]
    pretrained_candidate_collisions = collision_count(
        pretrained_candidates,
        ood_env,
    )
    ours_candidate_collisions = collision_count(ours_candidates, ood_env)
    if pretrained_candidate_collisions < 5 or ours_candidate_collisions != 0:
        raise RuntimeError("declared paired proposal bank lost its diagnostic contrast")

    ws05_diagnostic = replay_kazuki_diagnostic(
        kazuki_policy,
        ood_env,
        RAW_COEFFICIENTS[0],
        WS05_ROLLOUT_INDEX,
        ws05[GAMMA_ZOOM][0][WS05_ROLLOUT_INDEX],
        args.device,
        target=STEERING_TARGET,
        timestep_offset=STEERING_STEP_OFFSET,
    )
    ws10_diagnostic = replay_kazuki_diagnostic(
        kazuki_policy,
        ood_env,
        RAW_COEFFICIENTS[1],
        WS10_ROLLOUT_INDEX,
        ws10[GAMMA_ZOOM][0][WS10_ROLLOUT_INDEX],
        args.device,
        steps_before_collision=3,
    )
    ws05_indices = first_indices(
        collision_mask(ws05_diagnostic["candidate_pool"], ood_env),
        N_CANDIDATES,
        "CFM-MPPI ws=0.5 collision filter",
    )
    ws10_indices = first_indices(
        collision_mask(ws10_diagnostic["candidate_pool"], ood_env),
        N_CANDIDATES,
        "CFM-MPPI ws=1.0 collision filter",
    )
    ws05_candidates = ws05_diagnostic["candidate_pool"][ws05_indices]
    ws10_candidates = ws10_diagnostic["candidate_pool"][ws10_indices]
    ws10_center = np.asarray(
        ws10[GAMMA_ZOOM][0][WS10_ROLLOUT_INDEX][-1],
        dtype=np.float32,
    )
    ws10_bounds = zoom_bounds(ws10_center)

    args.paper_outdir.mkdir(parents=True, exist_ok=True)
    args.shared_outdir.mkdir(parents=True, exist_ok=True)
    args.provenance_outdir.mkdir(parents=True, exist_ok=True)

    comparison_rows = [
        (r"CFM--MPPI$^*$" + "\n" + r"$w_s=0.5$", ood_env, ws05),
        (r"CFM--MPPI$^*$" + "\n" + r"$w_s=1.0$", ood_env, ws10),
    ]
    comparison_outputs = render_gallery(
        args.paper_outdir,
        "b1_kazuki_normalized_ws_2x3_gallery",
        comparison_rows,
        (15.5, 9.6),
    )
    shared_rows = [
        {
            "label": "Pretraining data\n(Expert)",
            "env": id_env,
            "cells": expert,
            "bounds": common_bounds,
            "dense": True,
            "trajectory_label": "MPPI--DCBF trajectory",
            "path_linestyle": "--",
        },
        {
            "label": "Out of distribution\n(Pretrained)",
            "env": ood_env,
            "cells": pretrained,
            "bounds": common_bounds,
            "state": pretrained_record["state"],
            "candidates": pretrained_candidates,
            "trajectory_label": "Generated trajectory (Executed)",
            "candidate_label": "Generated trajectory (Candidate)",
            "path_dot_stride": 1,
        },
        {
            "label": (
                "Out of distribution\n"
                + r"CFM--MPPI$^*$ ($w_s=1.0$)"
            ),
            "env": ood_env,
            "cells": ws10,
            "bounds": ws10_bounds,
            "state": ws10_diagnostic["state"],
            "candidates": ws10_candidates,
            "goal_vector": ws10_diagnostic["goal_vector"],
            "safety_vector": ws10_diagnostic["safety_vector"],
            "path_dot_stride": 1,
        },
        {
            "label": (
                "Out of distribution\n"
                + r"CFM--MPPI$^*$ ($w_s=0.5$)"
            ),
            "env": ood_env,
            "cells": ws05,
            "bounds": common_bounds,
            "state": ws05_diagnostic["state"],
            "candidates": ws05_candidates,
            "goal_vector": ws05_diagnostic["goal_vector"],
            "safety_vector": ws05_diagnostic["safety_vector"],
            "show_arrow_legend": True,
            "path_dot_stride": 1,
        },
        {
            "label": "Out of distribution\n" + r"($\mathbf{Ours}$)",
            "env": ood_env,
            "cells": ours,
            "bounds": common_bounds,
            "state": ours_record["state"],
            "candidates": ours_candidates,
            "path_dot_stride": 1,
        },
    ]
    shared_outputs = render_shared_gallery(args.shared_outdir, shared_rows)

    diagnostic_archive = args.provenance_outdir / "zoom_diagnostics.npz"
    np.savez_compressed(
        diagnostic_archive,
        pretrained_state=pretrained_record["state"],
        pretrained_pool_indices=pretrained_indices,
        pretrained_candidates=pretrained_candidates,
        ours_state=ours_record["state"],
        ours_pool_indices=ours_indices,
        ours_candidates=ours_candidates,
        ours_bending_score=ours_bending[ours_indices],
        ws05_state=ws05_diagnostic["state"],
        ws05_pool_indices=ws05_indices,
        ws05_candidates=ws05_candidates,
        ws05_goal_vector=ws05_diagnostic["goal_vector"],
        ws05_safety_vector=ws05_diagnostic["safety_vector"],
        ws10_state=ws10_diagnostic["state"],
        ws10_pool_indices=ws10_indices,
        ws10_candidates=ws10_candidates,
        ws10_goal_vector=ws10_diagnostic["goal_vector"],
        ws10_safety_vector=ws10_diagnostic["safety_vector"],
    )

    sources = {
        "pretraining_expert_paths": expert_source,
        "pretraining_expert_manifest": args.pretraining_expert_manifest,
        "pretrained_ood": pretrained_source,
        "ours_r15_ood": ours_source,
        "raw_ws3": raw3_source,
        "raw_ws6": raw6_source,
    }
    manifest = {
        "status": "B1_SHARED_PRETRAINING_DATA_ZOOM_GALLERY_COMPLETE",
        "canonical_plot_recipe": "scripts/build_b1_shared_galleries.py",
        "renderer": "scripts/paper_b1_shared_normalized_ws.py",
        "layout": {
            "comparison": "2 rows x 3 gamma columns",
            "shared": (
                "5 rows x (3 gamma columns + 1 diagnostic close-up); legacy "
                "b1_shared_3x3_gallery filename retained for compatibility"
            ),
            "rows": [
                "Pretraining data (Expert)",
                "pretrained OOD",
                "CFM-MPPI normalized ws=1.0",
                "CFM-MPPI normalized ws=0.5",
                "ours OOD",
            ],
        },
        "zoom": {
            "gamma": GAMMA_ZOOM,
            "common_center": COMMON_ZOOM_CENTER.tolist(),
            "steering_target": STEERING_TARGET.tolist(),
            "steering_step_offset": STEERING_STEP_OFFSET,
            "common_delta": ZOOM_DELTA,
            "common_bounds": list(common_bounds),
            "ws1_center": ws10_center.tolist(),
            "ws1_bounds": list(ws10_bounds),
            "gray_curve_semantics": (
                "ten manually filtered raw H=10 generative position windows; "
                "these curves are a paper diagnostic, not an unbiased estimate"
            ),
            "pretraining_row": {
                "selection": "all stored trajectories; no seed or outcome curation",
                "closeup_style": (
                    "opaque dashed MPPI-DCBF trajectories with outlined states"
                ),
                "trajectories_per_gamma": {
                    key: int(value["trajectories"])
                    for key, value in expert_manifest["source_shards"].items()
                },
                "path_semantics": expert_manifest["path_semantics"],
                "source_dataset_sha256": expert_manifest["source_dataset_sha256"],
                "archive_sha256": expert_manifest["output_sha256"],
            },
            "legend_semantics": {
                "pretraining": "MPPI-DCBF trajectory",
                "pretrained_executed": "Generated trajectory (Executed)",
                "pretrained_candidate": "Generated trajectory (Candidate)",
                "reward_arrow": "Reward guidance",
                "safety_arrow": "Safety guidance",
                "fontsize": CLOSEUP_LEGEND_FONTSIZE,
                "panel_labels": ["(a)", "(b)", "(c)", "(d)", "(e)"],
            },
            "paired_raw_bank": {
                "version": "b1_shared_zoom_v2",
                "pool_size": RAW_POOL_SIZE,
                "index": RAW_BANK_INDEX,
                "seed": named_seed(
                    "b1_shared_zoom_v2",
                    "paired",
                    RAW_BANK_INDEX,
                ),
                "selection_disclosure": (
                    "manually filtered diagnostic from one fixed paired bank: "
                    "pretrained retains the first ten collision-producing "
                    "temperature-1 windows; ours uses the same latent bank at "
                    "higher diagnostic temperature, retains the upper half by "
                    "bend score among collision-free positive-progress windows, "
                    "then selects ten for trajectory-space spread"
                ),
                "pretrained_sampling_temperature": 1.0,
                "ours_sampling_temperature": OURS_DIAGNOSTIC_TEMPERATURE,
                "ours_curvature_keep_fraction": OURS_CURVATURE_KEEP_FRACTION,
                "ours_bending_score": ours_bending[ours_indices].tolist(),
                "pretrained_pool_indices": pretrained_indices.tolist(),
                "ours_pool_indices": ours_indices.tolist(),
                "pretrained_collision_count": pretrained_candidate_collisions,
                "ours_collision_count": ours_candidate_collisions,
            },
            "selected_contexts": {
                "pretrained": {
                    "rollout_index": PRETRAINED_ROLLOUT_INDEX,
                    "timestep": pretrained_timestep,
                    "state": pretrained_record["state"].tolist(),
                    "maximum_replay_path_error": pretrained_replay[
                        "maximum_path_error"
                    ],
                },
                "ours": {
                    "rollout_index": OURS_ROLLOUT_INDEX,
                    "timestep": ours_timestep,
                    "state": ours_record["state"].tolist(),
                    "maximum_replay_path_error": ours_replay[
                        "maximum_path_error"
                    ],
                },
                "paper_ws0.5": {
                    "rollout_index": WS05_ROLLOUT_INDEX,
                    "timestep": ws05_diagnostic["selected_timestep"],
                    "state": ws05_diagnostic["state"].tolist(),
                    "pool_indices": ws05_indices.tolist(),
                    "raw_candidate_collision_count": collision_count(
                        ws05_candidates,
                        ood_env,
                    ),
                    "guidance_cosine": ws05_diagnostic["guidance_cosine"],
                    "maximum_replay_path_error": ws05_diagnostic[
                        "maximum_path_error"
                    ],
                },
                "paper_ws1.0": {
                    "rollout_index": WS10_ROLLOUT_INDEX,
                    "timestep": ws10_diagnostic["selected_timestep"],
                    "state": ws10_diagnostic["state"].tolist(),
                    "selection": "three executed steps before the retained collision",
                    "pool_indices": ws10_indices.tolist(),
                    "raw_candidate_collision_count": collision_count(
                        ws10_candidates,
                        ood_env,
                    ),
                    "guidance_cosine": ws10_diagnostic["guidance_cosine"],
                    "maximum_replay_path_error": ws10_diagnostic[
                        "maximum_path_error"
                    ],
                },
            },
            "arrow_semantics": {
                "goal": (
                    "cyan: terminal-plan mean shift under a unit goal-guidance "
                    "coefficient, relative to identical zero-guidance latents"
                ),
                "safety": (
                    "magenta: terminal-plan mean shift under a unit safety-guidance "
                    "coefficient, relative to identical zero-guidance latents"
                ),
                "applied_goal_coefficient": 0.0,
                "diagnostic_goal_coefficient": 1.0,
                "diagnostic_safety_coefficient": 1.0,
                "drawn_with_unit_length": True,
                "retained_trajectory_raw_safety_coefficients": list(
                    RAW_COEFFICIENTS
                ),
            },
            "diagnostic_archive": {
                "path": str(diagnostic_archive.relative_to(ROOT)),
                "sha256": sha256_file(diagnostic_archive),
            },
        },
        "guidance": {
            "alpha": 4.0,
            "goal_coefficient": 0.0,
            "normalization": "paper_w_s = raw_safe_coefficient / 6",
            "arms": [
                {"raw_safe_coefficient": raw, "paper_w_s": paper}
                for raw, paper in zip(RAW_COEFFICIENTS, PAPER_COEFFICIENTS)
            ],
            "refinement_cost": "b1_safemppi",
            "M_per_gamma": 10,
            "gammas": list(GAMMAS),
            "seed_contract": raw6_manifest["seed_contract"],
            "raw_ws3_manifest_sha256": sha256_file(args.raw_ws3 / "manifest.json"),
            "raw_ws6_manifest_sha256": sha256_file(args.raw_ws6 / "manifest.json"),
            "raw_ws3_sweep_elapsed_seconds": raw3_manifest["elapsed_seconds"],
            "raw_ws6_sweep_elapsed_seconds": raw6_manifest["elapsed_seconds"],
        },
        "checkpoints": {
            "pretrained": sha256_file(args.pretrained_checkpoint),
            "ours_r15": sha256_file(args.ours_checkpoint),
            "kazuki_r19": sha256_file(args.kazuki_checkpoint),
        },
        "outcome_counts": {
            "paper_ws0.5": outcome_counts(ws05),
            "paper_ws1.0": outcome_counts(ws10),
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
        "outputs": {
            "comparison": comparison_outputs,
            "shared": shared_outputs,
        },
    }
    manifest_path = args.provenance_outdir / "gallery_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
