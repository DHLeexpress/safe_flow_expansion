#!/usr/bin/env python3
"""Render metrics and trajectory overlays for the 4x4 Kazuki coefficient grid."""

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
GAMMAS = (0.1, 1.0)
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


def load_pair(path: Path) -> dict[float, tuple[list[np.ndarray], list[str]]]:
    cells = {}
    with np.load(path, allow_pickle=True) as archive:
        for gamma in GAMMAS:
            suffix = f"g{gamma:g}"
            cells[gamma] = (
                [
                    np.asarray(value, dtype=np.float64)
                    for value in archive[f"paths_{suffix}"]
                ],
                [str(value) for value in archive[f"outcomes_{suffix}"]],
            )
    return cells


def draw_scene(axis, paths, outcomes, gamma):
    recipe = json.loads((ROOT / "configs/b1_current_best_recipe.json").read_text())
    scene = recipe["scene"]
    for obstacle in scene["obstacles"]:
        axis.add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#c6c6c6", zorder=1)
        )
    color = plt.get_cmap("plasma")(0.08 if gamma == 0.1 else 0.92)
    for path, outcome in zip(paths, outcomes):
        axis.plot(path[:, 0], path[:, 1], color=color, lw=1.0, alpha=0.62)
        if outcome != "SR":
            axis.plot(
                path[-1, 0],
                path[-1, 1],
                "x",
                color="#cc3311",
                markersize=5.5,
                markeredgewidth=1.4,
            )
    axis.plot(*scene["start_state"][:2], "ks", markersize=4)
    axis.plot(
        *scene["goal"],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=9,
    )
    axis.set_xlim(-0.3, 5.3)
    axis.set_ylim(-0.3, 5.3)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "provenance/paper_baselines/kazuki_absolute_grid_m10",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "assets/paper")
    parser.add_argument("--stem", default="kazuki_absolute_coefficient_grid")
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["status"] != "KAZUKI_ABSOLUTE_COEFFICIENT_GRID_COMPLETE":
        raise RuntimeError("input sweep is incomplete")

    goals = [float(value) for value in manifest["goal_coefficients"]]
    safeties = [float(value) for value in manifest["safe_coefficients"]]
    obstacles, robot_radius = scene_geometry()
    pairs = {}
    for entry in manifest["outputs"]:
        key = (float(entry["goal_coef"]), float(entry["safe_coef"]))
        source = args.input / entry["file"]
        observed_sha = sha256_file(source)
        if observed_sha != entry["sha256"]:
            raise RuntimeError(f"source hash mismatch: {source}")
        cells = load_pair(source)
        metrics = {
            gamma: summarize_path_cell(
                *cells[gamma], gamma, obstacles, robot_radius
            )
            for gamma in GAMMAS
        }
        pairs[key] = {"cells": cells, "metrics": metrics, **entry}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(18.2, 8.4), squeeze=False)
    for row, gamma in enumerate(GAMMAS):
        for column, (metric, title, fmt) in enumerate(METRICS):
            axis = axes[row, column]
            values = np.asarray(
                [
                    [
                        pairs[(goal, safe)]["metrics"][gamma][metric]["mean"]
                        for safe in safeties
                    ]
                    for goal in goals
                ],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                low, high = float(finite.min()), float(finite.max())
                if abs(high - low) < 1e-12:
                    high = low + 1.0
            else:
                low, high = 0.0, 1.0
            image = axis.imshow(
                values,
                origin="lower",
                cmap="viridis",
                vmin=low,
                vmax=high,
                aspect="equal",
            )
            for goal_index, goal in enumerate(goals):
                for safe_index, safe in enumerate(safeties):
                    value = values[goal_index, safe_index]
                    text = format(value, fmt) if np.isfinite(value) else "--"
                    axis.text(
                        safe_index,
                        goal_index,
                        text,
                        ha="center",
                        va="center",
                        color="white" if np.isfinite(value) and value < (low + high) / 2 else "black",
                        fontsize=11,
                        fontweight="bold",
                    )
            axis.set_xticks(range(len(safeties)), [f"{value:g}" for value in safeties])
            axis.set_yticks(range(len(goals)), [f"{value:g}" for value in goals])
            axis.set_xlabel(r"$w_{\rm safe}$")
            if column == 0:
                axis.set_ylabel(rf"$\gamma={gamma:g}$" + "\n" + r"$w_{\rm goal}$")
            axis.set_title(title)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    fig.suptitle(
        r"CFM-MPPI$^*$ absolute guidance-coefficient sweep "
        r"(native B1 SafeMPPI refinement cost)",
        fontsize=21,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    args.outdir.mkdir(parents=True, exist_ok=True)
    metric_outputs = []
    for suffix in ("png", "pdf"):
        output = args.outdir / f"{args.stem}_metrics.{suffix}"
        fig.savefig(output, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        metric_outputs.append(output)
    plt.close(fig)

    # Four goal-coefficient rows and two four-column gamma blocks.
    fig, axes = plt.subplots(4, 8, figsize=(27.0, 13.4), squeeze=False)
    for goal_index, goal in enumerate(goals):
        for gamma_block, gamma in enumerate(GAMMAS):
            for safe_index, safe in enumerate(safeties):
                column = gamma_block * len(safeties) + safe_index
                axis = axes[goal_index, column]
                paths, outcomes = pairs[(goal, safe)]["cells"][gamma]
                draw_scene(axis, paths, outcomes, gamma)
                if goal_index == 0:
                    axis.set_title(
                        rf"$\gamma={gamma:g}$" + "\n" + rf"$w_s={safe:g}$",
                        fontsize=15,
                    )
                if column == 0:
                    axis.set_ylabel(rf"$w_g={goal:g}$", fontsize=16)
    fig.subplots_adjust(
        left=0.035,
        right=0.997,
        bottom=0.015,
        top=0.94,
        wspace=0.018,
        hspace=0.04,
    )
    overlay_outputs = []
    for suffix in ("png", "pdf"):
        output = args.outdir / f"{args.stem}_overlays.{suffix}"
        fig.savefig(output, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        overlay_outputs.append(output)
    plt.close(fig)

    sidecar = {
        "status": "KAZUKI_ABSOLUTE_COEFFICIENT_GRID_FIGURES_COMPLETE",
        "input_manifest": str((args.input / "manifest.json").resolve()),
        "scientific_contract": (
            "Only guided-flow goal and safety coefficients vary. The r19 "
            "checkpoint, common-random-number bank, scene, and exact native "
            "B1 SafeMPPI refinement cost are fixed."
        ),
        "pairs": {
            f"wg{goal:g}_ws{safe:g}": {
                "goal_coef": goal,
                "safe_coef": safe,
                "metrics": pairs[(goal, safe)]["metrics"],
                "source_file": pairs[(goal, safe)]["file"],
                "source_sha256": pairs[(goal, safe)]["sha256"],
            }
            for goal in goals
            for safe in safeties
        },
        "outputs": [
            str(path.resolve()) for path in (*metric_outputs, *overlay_outputs)
        ],
    }
    sidecar_path = args.outdir / f"{args.stem}.json"
    sidecar_path.write_text(
        json.dumps(json_ready(sidecar), indent=2, sort_keys=True) + "\n"
    )
    for path in (*metric_outputs, *overlay_outputs, sidecar_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
