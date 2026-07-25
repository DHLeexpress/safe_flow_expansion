#!/usr/bin/env python3
"""Render the eight-arm alpha 3/4 Kazuki wall-avoidance screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_metrics_common import json_ready, scene_geometry, summarize_path_cell


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("CR", "Collision rate", ".2f"),
    ("v_safe", "Validity", ".2f"),
    ("clearance", "Min. clearance [m]", ".3f"),
    ("time", "Time-to-goal [s]", ".1f"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cells(path: Path, gammas: list[float]) -> dict:
    cells = {}
    with np.load(path, allow_pickle=True) as archive:
        for gamma in gammas:
            suffix = f"g{gamma:g}"
            cells[gamma] = (
                [np.asarray(value, dtype=np.float64) for value in archive[f"paths_{suffix}"]],
                [str(value) for value in archive[f"outcomes_{suffix}"]],
            )
    return cells


def wall_statistics(path: np.ndarray) -> dict[str, float]:
    """Score wall following away from the start and goal corner neighborhoods."""

    points = np.asarray(path, dtype=float)
    start = np.array([0.3, 0.3])
    goal = np.array([4.7, 4.7])
    transit = (
        (np.linalg.norm(points - start, axis=1) > 1.0)
        & (np.linalg.norm(points - goal, axis=1) > 1.0)
    )
    selected = points[transit] if np.any(transit) else points
    boundary = np.minimum.reduce(
        [selected[:, 0], 5.0 - selected[:, 0], selected[:, 1], 5.0 - selected[:, 1]]
    )
    return {
        "wall_fraction_0p6": float(np.mean(boundary <= 0.6)),
        "mean_boundary_distance": float(np.mean(boundary)),
        "minimum_boundary_distance": float(np.min(boundary)),
        "transit_points": int(len(selected)),
    }


def draw_scene(axis, paths, outcomes, gamma, highlight: int | None = None) -> None:
    recipe = json.loads((ROOT / "configs/b1_current_best_recipe.json").read_text())
    scene = recipe["scene"]
    for obstacle in scene["obstacles"]:
        axis.add_patch(plt.Circle(obstacle[:2], obstacle[2], color="#c6c6c6", zorder=1))
    color = plt.get_cmap("plasma")(0.08 + 0.84 * gamma)
    for index, (path, outcome) in enumerate(zip(paths, outcomes)):
        chosen = index == highlight
        axis.plot(
            path[:, 0],
            path[:, 1],
            color="#0072B2" if chosen else color,
            lw=3.0 if chosen else 0.9,
            alpha=1.0 if chosen else 0.42,
            zorder=4 if chosen else 2,
        )
        if outcome != "SR":
            axis.plot(
                path[-1, 0],
                path[-1, 1],
                "x",
                color="#CC3311",
                markersize=6,
                markeredgewidth=1.4,
                zorder=5,
            )
    axis.plot(*scene["start_state"][:2], "ks", markersize=4)
    axis.plot(
        *scene["goal"],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=9,
    )
    axis.set(xlim=(-0.3, 5.3), ylim=(-0.3, 5.3), aspect="equal")
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "provenance/paper_baselines/kazuki_alpha34_wall_grid_m10",
    )
    parser.add_argument(
        "--wall-search",
        type=Path,
        default=(
            ROOT
            / "provenance/paper_baselines/kazuki_alpha4_wg0_ws3_wall_search_m50"
        ),
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "assets/paper")
    parser.add_argument("--stem", default="kazuki_alpha34_wall_grid")
    args = parser.parse_args()

    manifest_path = args.input / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "KAZUKI_ALPHA_COEFFICIENT_GRID_COMPLETE":
        raise RuntimeError("alpha 3/4 sweep is incomplete")
    if manifest["alphas"] != [3.0, 4.0]:
        raise RuntimeError("unexpected alpha grid")
    if manifest["goal_coefficients"] != [0.0, 1.0]:
        raise RuntimeError("unexpected goal-coefficient grid")
    if manifest["safe_coefficients"] != [3.0, 4.0]:
        raise RuntimeError("unexpected safety-coefficient grid")
    gammas = [float(value) for value in manifest["gammas"]]
    if gammas != [0.1, 0.5, 1.0]:
        raise RuntimeError("unexpected gamma grid")

    obstacles, robot_radius = scene_geometry()
    arms = []
    wall_candidates = []
    for entry in manifest["outputs"]:
        source = args.input / entry["file"]
        if sha256_file(source) != entry["sha256"]:
            raise RuntimeError(f"source hash mismatch: {source}")
        cells = load_cells(source, gammas)
        metrics = {
            gamma: summarize_path_cell(*cells[gamma], gamma, obstacles, robot_radius)
            for gamma in gammas
        }
        arm = {
            "alpha": float(entry["alpha"]),
            "goal_coef": float(entry["goal_coef"]),
            "safe_coef": float(entry["safe_coef"]),
            "source_file": entry["file"],
            "source_sha256": entry["sha256"],
            "cells": cells,
            "metrics": metrics,
        }
        arms.append(arm)
        for gamma in gammas:
            paths, outcomes = cells[gamma]
            for rollout_index, (path, outcome) in enumerate(zip(paths, outcomes)):
                if outcome != "SR":
                    continue
                wall = wall_statistics(path)
                wall_candidates.append(
                    {
                        "arm": arm,
                        "gamma": gamma,
                        "rollout_index": rollout_index,
                        "path": path,
                        **wall,
                    }
                )
    arms.sort(key=lambda arm: (arm["alpha"], arm["goal_coef"], arm["safe_coef"]))

    wall_search_manifest = None
    if args.wall_search.exists():
        wall_search_manifest_path = args.wall_search / "manifest.json"
        wall_search_manifest = json.loads(wall_search_manifest_path.read_text())
        if wall_search_manifest["status"] != "KAZUKI_ALPHA_COEFFICIENT_GRID_COMPLETE":
            raise RuntimeError("wall-search manifest is incomplete")
        if len(wall_search_manifest["outputs"]) != 1:
            raise RuntimeError("wall search must contain exactly one arm")
        entry = wall_search_manifest["outputs"][0]
        source = args.wall_search / entry["file"]
        if sha256_file(source) != entry["sha256"]:
            raise RuntimeError("wall-search source hash mismatch")
        cells = load_cells(
            source, [float(value) for value in wall_search_manifest["gammas"]]
        )
        search_arm = {
            "alpha": float(entry["alpha"]),
            "goal_coef": float(entry["goal_coef"]),
            "safe_coef": float(entry["safe_coef"]),
            "source_file": entry["file"],
            "source_sha256": entry["sha256"],
            "cells": cells,
            "metrics": {},
        }
        for gamma, (paths, outcomes) in cells.items():
            for rollout_index, (path, outcome) in enumerate(zip(paths, outcomes)):
                if outcome != "SR":
                    continue
                wall_candidates.append(
                    {
                        "arm": search_arm,
                        "gamma": gamma,
                        "rollout_index": rollout_index,
                        "path": path,
                        "search_M": int(wall_search_manifest["M_per_cell"]),
                        **wall_statistics(path),
                    }
                )
    best_wall = (
        max(
            wall_candidates,
            key=lambda item: (
                item["wall_fraction_0p6"],
                -item["mean_boundary_distance"],
                item["transit_points"],
                item.get("search_M", 0),
            ),
        )
        if wall_candidates
        else None
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(arms),
        len(gammas),
        figsize=(10.8, 3.35 * len(arms)),
        squeeze=False,
    )
    for row, arm in enumerate(arms):
        for column, gamma in enumerate(gammas):
            paths, outcomes = arm["cells"][gamma]
            highlight = None
            if (
                best_wall is not None
                and best_wall["arm"] is arm
                and best_wall["gamma"] == gamma
            ):
                highlight = int(best_wall["rollout_index"])
            draw_scene(axes[row, column], paths, outcomes, gamma, highlight)
            if row == 0:
                axes[row, column].set_title(rf"$\gamma={gamma:g}$")
            if column == 0:
                axes[row, column].set_ylabel(
                    rf"$\alpha={arm['alpha']:g},\,w_g={arm['goal_coef']:g},"
                    rf"\,w_s={arm['safe_coef']:g}$",
                    fontsize=13,
                )
    fig.suptitle(
        "CFM-MPPI$^*$ high-alpha guidance trajectories",
        fontsize=20,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    trajectory_outputs = []
    for extension in ("png", "pdf"):
        output = args.outdir / f"{args.stem}_trajectories.{extension}"
        fig.savefig(output, dpi=300 if extension == "png" else None, bbox_inches="tight")
        trajectory_outputs.append(output)
    plt.close(fig)

    metric_ranges = {}
    for metric, _, _ in METRICS:
        values = np.asarray(
            [
                arm["metrics"][gamma][metric]["mean"]
                for arm in arms
                for gamma in gammas
            ],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        if metric in {"CR", "v_safe"}:
            metric_ranges[metric] = (0.0, 1.0)
        elif finite.size:
            low, high = float(finite.min()), float(finite.max())
            if abs(high - low) < 1e-12:
                high = low + 1.0
            metric_ranges[metric] = (low, high)
        else:
            metric_ranges[metric] = (0.0, 1.0)

    labels = [
        rf"$\alpha={arm['alpha']:g},\,w_g={arm['goal_coef']:g},"
        rf"\,w_s={arm['safe_coef']:g}$"
        for arm in arms
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 7.8), squeeze=False)
    for column, (metric, title, fmt) in enumerate(METRICS):
        axis = axes[0, column]
        values = np.asarray(
            [
                [arm["metrics"][gamma][metric]["mean"] for gamma in gammas]
                for arm in arms
            ],
            dtype=float,
        )
        low, high = metric_ranges[metric]
        image = axis.imshow(values, cmap="viridis", vmin=low, vmax=high, aspect="auto")
        midpoint = (low + high) / 2.0
        for row in range(len(arms)):
            for gamma_index in range(len(gammas)):
                value = values[row, gamma_index]
                axis.text(
                    gamma_index,
                    row,
                    format(value, fmt) if np.isfinite(value) else "--",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white" if np.isfinite(value) and value < midpoint else "black",
                )
        axis.set_title(title)
        axis.set_xticks(
            range(len(gammas)), [rf"$\gamma={gamma:g}$" for gamma in gammas]
        )
        axis.set_yticks(range(len(arms)), labels if column == 0 else [])
        fig.colorbar(image, ax=axis, fraction=0.05, pad=0.035)
    fig.suptitle("High-alpha guidance: matched $M=10$ screen", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    metric_outputs = []
    for extension in ("png", "pdf"):
        output = args.outdir / f"{args.stem}_metrics.{extension}"
        fig.savefig(output, dpi=300 if extension == "png" else None, bbox_inches="tight")
        metric_outputs.append(output)
    plt.close(fig)

    wall_outputs = []
    wall_payload = None
    if best_wall is not None:
        arm = best_wall["arm"]
        fig, axis = plt.subplots(figsize=(6.4, 6.4))
        paths, outcomes = arm["cells"][best_wall["gamma"]]
        draw_scene(axis, paths, outcomes, best_wall["gamma"], best_wall["rollout_index"])
        axis.add_patch(
            plt.Rectangle(
                (0.6, 0.6),
                3.8,
                3.8,
                fill=False,
                edgecolor="#0072B2",
                linestyle="--",
                linewidth=1.3,
                alpha=0.75,
                zorder=6,
            )
        )
        axis.set_title(
            "Most wall-following successful rollout\n"
            + rf"$\alpha={arm['alpha']:g},\,w_g={arm['goal_coef']:g},"
            + rf"\,w_s={arm['safe_coef']:g},\,\gamma={best_wall['gamma']:g}$"
            + "\n"
            + f"wall-zone fraction = {best_wall['wall_fraction_0p6']:.1%}"
        )
        fig.tight_layout()
        for extension in ("png", "pdf"):
            output = args.outdir / f"{args.stem}_wall_candidate.{extension}"
            fig.savefig(
                output,
                dpi=300 if extension == "png" else None,
                bbox_inches="tight",
            )
            wall_outputs.append(output)
        plt.close(fig)
        wall_payload = {
            key: value
            for key, value in best_wall.items()
            if key not in {"arm", "path"}
        } | {
            "alpha": arm["alpha"],
            "goal_coef": arm["goal_coef"],
            "safe_coef": arm["safe_coef"],
            "outcome": "SR",
        }

    sidecar = {
        "status": "KAZUKI_ALPHA34_WALL_GRID_FIGURES_COMPLETE",
        "input_manifest": str(manifest_path.resolve()),
        "wall_search_manifest": (
            None
            if wall_search_manifest is None
            else str((args.wall_search / "manifest.json").resolve())
        ),
        "wall_metric_definition": (
            "Among successful rollouts, maximize the fraction of transit states "
            "(more than 1 m from start and goal) within 0.6 m of the task-space "
            "boundary; ties favor smaller mean boundary distance."
        ),
        "selected_wall_candidate": wall_payload,
        "arms": [
            {
                key: arm[key]
                for key in (
                    "alpha",
                    "goal_coef",
                    "safe_coef",
                    "source_file",
                    "source_sha256",
                    "metrics",
                )
            }
            for arm in arms
        ],
        "outputs": [
            str(path.resolve())
            for path in (*trajectory_outputs, *metric_outputs, *wall_outputs)
        ],
    }
    sidecar_path = args.outdir / f"{args.stem}.json"
    sidecar_path.write_text(
        json.dumps(json_ready(sidecar), indent=2, sort_keys=True) + "\n"
    )
    for path in (*trajectory_outputs, *metric_outputs, *wall_outputs, sidecar_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
