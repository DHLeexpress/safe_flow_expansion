#!/usr/bin/env python3
"""Render alpha-dependent Kazuki metrics and selected trajectory modes."""

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


def load_pair(path: Path, gammas: list[float]) -> dict:
    cells = {}
    with np.load(path, allow_pickle=True) as archive:
        for gamma in gammas:
            suffix = f"g{gamma:g}"
            cells[gamma] = (
                [np.asarray(value, dtype=np.float64) for value in archive[f"paths_{suffix}"]],
                [str(value) for value in archive[f"outcomes_{suffix}"]],
            )
    return cells


def rolling_dwell(path: np.ndarray, window: int = 20) -> tuple[float, int]:
    """Return the smallest window displacement and its starting index."""

    points = np.asarray(path, dtype=float)
    if len(points) <= window:
        return float(np.linalg.norm(points[-1] - points[0])), 0
    distances = np.linalg.norm(points[window:] - points[:-window], axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), index


def draw_scene(axis, paths, outcomes, gamma, highlight_index: int | None = None):
    recipe = json.loads((ROOT / "configs/b1_current_best_recipe.json").read_text())
    scene = recipe["scene"]
    for obstacle in scene["obstacles"]:
        axis.add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#c6c6c6", zorder=1)
        )
    color = plt.get_cmap("plasma")(0.08 if gamma == 0.1 else 0.92)
    for index, (path, outcome) in enumerate(zip(paths, outcomes)):
        selected = index == highlight_index
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            lw=2.2 if selected else 0.9,
            alpha=0.95 if selected else 0.3,
            zorder=4 if selected else 2,
        )
        if outcome != "SR":
            axis.plot(
                path[-1, 0],
                path[-1, 1],
                "x",
                color="#cc3311",
                markersize=7 if selected else 4.5,
                markeredgewidth=1.7 if selected else 1.1,
                zorder=5,
            )
    axis.plot(*scene["start_state"][:2], "ks", markersize=4)
    axis.plot(
        *scene["goal"],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=10,
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
        default=ROOT / "provenance/paper_baselines/kazuki_alpha_fine_grid_m10",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "assets/paper")
    parser.add_argument("--stem", default="kazuki_alpha_fine_grid")
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["status"] != "KAZUKI_ALPHA_COEFFICIENT_GRID_COMPLETE":
        raise RuntimeError("alpha sweep is incomplete")

    alphas = [float(value) for value in manifest["alphas"]]
    goals = [float(value) for value in manifest["goal_coefficients"]]
    safeties = [float(value) for value in manifest["safe_coefficients"]]
    gammas = [float(value) for value in manifest["gammas"]]
    obstacles, robot_radius = scene_geometry()
    pairs = {}
    for entry in manifest["outputs"]:
        source = args.input / entry["file"]
        if sha256_file(source) != entry["sha256"]:
            raise RuntimeError(f"source hash mismatch: {source}")
        key = (
            float(entry["alpha"]),
            float(entry["goal_coef"]),
            float(entry["safe_coef"]),
        )
        cells = load_pair(source, gammas)
        metrics = {
            gamma: summarize_path_cell(
                *cells[gamma], gamma, obstacles, robot_radius
            )
            for gamma in gammas
        }
        pairs[key] = {"cells": cells, "metrics": metrics, **entry}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )
    metric_outputs = []
    for gamma in gammas:
        metric_ranges = {}
        for metric, _, _ in METRICS:
            all_values = np.asarray(
                [
                    pairs[(alpha, goal, safe)]["metrics"][gamma][metric]["mean"]
                    for alpha in alphas
                    for goal in goals
                    for safe in safeties
                ],
                dtype=float,
            )
            finite = all_values[np.isfinite(all_values)]
            if metric in {"CR", "v_safe"}:
                metric_ranges[metric] = (0.0, 1.0)
            elif finite.size:
                low, high = float(finite.min()), float(finite.max())
                if abs(high - low) < 1e-12:
                    high = low + 1.0
                metric_ranges[metric] = (low, high)
            else:
                metric_ranges[metric] = (0.0, 1.0)
        fig, axes = plt.subplots(
            len(alphas), len(METRICS), figsize=(15.8, 3.45 * len(alphas)),
            squeeze=False,
        )
        for row, alpha in enumerate(alphas):
            for column, (metric, title, fmt) in enumerate(METRICS):
                axis = axes[row, column]
                values = np.asarray(
                    [
                        [
                            pairs[(alpha, goal, safe)]["metrics"][gamma][metric]["mean"]
                            for safe in safeties
                        ]
                        for goal in goals
                    ],
                    dtype=float,
                )
                low, high = metric_ranges[metric]
                image = axis.imshow(
                    values,
                    origin="lower",
                    cmap="viridis",
                    vmin=low,
                    vmax=high,
                    aspect="equal",
                )
                midpoint = (low + high) / 2.0
                for goal_index in range(len(goals)):
                    for safe_index in range(len(safeties)):
                        value = values[goal_index, safe_index]
                        axis.text(
                            safe_index,
                            goal_index,
                            format(value, fmt) if np.isfinite(value) else "--",
                            ha="center",
                            va="center",
                            fontsize=9.5,
                            fontweight="bold",
                            color=(
                                "white"
                                if np.isfinite(value) and value < midpoint
                                else "black"
                            ),
                        )
                axis.set_xticks(
                    range(len(safeties)), [f"{value:g}" for value in safeties]
                )
                axis.set_yticks(
                    range(len(goals)), [f"{value:g}" for value in goals]
                )
                axis.set_xlabel(r"$w_{\rm safe}$")
                if column == 0:
                    axis.set_ylabel(
                        rf"$\alpha={alpha:g}$" + "\n" + r"$w_{\rm goal}$"
                    )
                if row == 0:
                    axis.set_title(title)
                fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
        fig.suptitle(
            rf"CFM-MPPI$^*$ alpha and guidance sweep, $\gamma={gamma:g}$",
            fontsize=20,
            y=0.997,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        for suffix in ("png", "pdf"):
            output = args.outdir / f"{args.stem}_gamma{gamma:g}_metrics.{suffix}"
            fig.savefig(
                output, dpi=300 if suffix == "png" else None, bbox_inches="tight"
            )
            metric_outputs.append(output)
        plt.close(fig)

    # Select one successful, one longest-lived failure, and one shortest collision.
    selected = {}
    for gamma in gammas:
        candidates = []
        for key, pair in pairs.items():
            paths, outcomes = pair["cells"][gamma]
            for rollout_index, (path, outcome) in enumerate(zip(paths, outcomes)):
                dwell, dwell_index = rolling_dwell(path)
                candidates.append(
                    {
                        "key": key,
                        "rollout_index": rollout_index,
                        "outcome": outcome,
                        "steps": len(path) - 1,
                        "dwell_displacement_20": dwell,
                        "dwell_start": dwell_index,
                    }
                )
        successes = [item for item in candidates if item["outcome"] == "SR"]
        failures = [item for item in candidates if item["outcome"] != "SR"]
        timeouts = [item for item in failures if item["outcome"] == "TO"]
        dwell_pool = timeouts or [
            item for item in failures if item["steps"] >= 40
        ] or failures
        selected[gamma] = {
            "success": (
                max(successes, key=lambda item: item["steps"]) if successes else None
            ),
            "long_failure": max(failures, key=lambda item: item["steps"]),
            "max_dwell": min(
                dwell_pool, key=lambda item: item["dwell_displacement_20"]
            ),
            "has_timeout_local_minimum": bool(timeouts),
        }

    fig, axes = plt.subplots(
        len(gammas), 3, figsize=(12.6, 4.2 * len(gammas)), squeeze=False
    )
    labels = (
        ("success", "Successful"),
        ("long_failure", "Longest failure"),
        ("max_dwell", "Maximum dwell"),
    )
    for row, gamma in enumerate(gammas):
        for column, (field, title) in enumerate(labels):
            item = selected[gamma][field]
            axis = axes[row, column]
            if item is None:
                axis.text(0.5, 0.5, "No case", ha="center", va="center")
                axis.axis("off")
                continue
            alpha, goal, safe = item["key"]
            paths, outcomes = pairs[item["key"]]["cells"][gamma]
            draw_scene(axis, paths, outcomes, gamma, item["rollout_index"])
            display_title = (
                title
                if field != "max_dwell"
                else (
                    "Timeout local minimum"
                    if selected[gamma]["has_timeout_local_minimum"]
                    else "Maximum dwell failure"
                )
            )
            axis.set_title(
                display_title
                + "\n"
                + rf"$\alpha={alpha:g},\,w_g={goal:g},\,w_s={safe:g}$"
            )
            if column == 0:
                axis.set_ylabel(rf"$\gamma={gamma:g}$", fontsize=15)
    fig.tight_layout()
    selected_outputs = []
    for suffix in ("png", "pdf"):
        output = args.outdir / f"{args.stem}_selected_modes.{suffix}"
        fig.savefig(output, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        selected_outputs.append(output)
    plt.close(fig)

    sidecar = {
        "status": "KAZUKI_ALPHA_FINE_GRID_FIGURES_COMPLETE",
        "input_manifest": str((args.input / "manifest.json").resolve()),
        "alpha_interpretation": (
            "For h>0, increasing alpha makes hdot + alpha*h >= 0 easier to "
            "satisfy; smaller alpha is the more conservative standard-CBF setting."
        ),
        "pairs": {
            f"a{alpha:g}_wg{goal:g}_ws{safe:g}": {
                "alpha": alpha,
                "goal_coef": goal,
                "safe_coef": safe,
                "metrics": pairs[(alpha, goal, safe)]["metrics"],
                "source_file": pairs[(alpha, goal, safe)]["file"],
                "source_sha256": pairs[(alpha, goal, safe)]["sha256"],
            }
            for alpha in alphas
            for goal in goals
            for safe in safeties
        },
        "selected_modes": {
            str(gamma): {
                field: (
                    None
                    if item is None
                    else {
                        **item,
                        "alpha": item["key"][0],
                        "goal_coef": item["key"][1],
                        "safe_coef": item["key"][2],
                    }
                )
                for field, item in modes.items()
                if field != "has_timeout_local_minimum"
            }
            | {"has_timeout_local_minimum": modes["has_timeout_local_minimum"]}
            for gamma, modes in selected.items()
        },
        "outputs": [
            str(path.resolve()) for path in (*metric_outputs, *selected_outputs)
        ],
    }
    sidecar_path = args.outdir / f"{args.stem}.json"
    sidecar_path.write_text(
        json.dumps(json_ready(sidecar), indent=2, sort_keys=True) + "\n"
    )
    for path in (*metric_outputs, *selected_outputs, sidecar_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
