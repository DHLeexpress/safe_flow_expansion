#!/usr/bin/env python3
"""Compare native-cost CFM--MPPI coefficient pairs and trajectory overlays.

The input manifest is explicit: each entry binds ``(goal_coef, safe_coef)`` to
one packed trajectory archive.  This prevents low/high visualization endpoints
from being mistaken for empirically selected coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_metrics_common import GAMMAS, json_ready, summarize_path_archive


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("SR", "Success rate", (-0.04, 1.04)),
    ("CR", "Collision rate", (-0.04, 1.04)),
    ("clearance", "Min. clearance [m]", None),
    ("time", "Time-to-goal [s]", None),
)


def load_spec(path: Path) -> tuple[list[dict], dict]:
    spec = json.loads(path.read_text())
    pairs = []
    for entry in spec["pairs"]:
        archive = (path.parent / entry["archive"]).resolve()
        pairs.append(
            {
                **entry,
                "archive": str(archive),
                "cells": summarize_path_archive(archive),
            }
        )
    return pairs, spec


def load_paths(path: Path) -> dict[float, tuple[list[np.ndarray], list[str]]]:
    result = {}
    with np.load(path, allow_pickle=True) as archive:
        for gamma in GAMMAS:
            suffix = f"g{gamma:g}"
            result[gamma] = (
                [np.asarray(item, dtype=float) for item in archive[f"paths_{suffix}"]],
                [str(item) for item in archive[f"outcomes_{suffix}"]],
            )
    return result


def draw_scene(axis, paths, outcomes, gamma, row_label, title):
    recipe = json.loads((ROOT / "configs/b1_current_best_recipe.json").read_text())
    scene = recipe["scene"]
    for obstacle in scene["obstacles"]:
        axis.add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#c6c6c6", zorder=1)
        )
    color = plt.get_cmap("plasma")({0.1: 0.08, 0.3: 0.32, 0.5: 0.56, 1.0: 0.92}[gamma])
    for path, outcome in zip(paths, outcomes):
        axis.plot(path[:, 0], path[:, 1], color=color, lw=1.15, alpha=0.68)
        if outcome != "SR":
            axis.plot(
                path[-1, 0],
                path[-1, 1],
                "x",
                color="#cc3311",
                markersize=7,
                markeredgewidth=1.8,
            )
    axis.plot(*scene["start_state"][:2], "ks", markersize=5)
    axis.plot(
        *scene["goal"],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=12,
    )
    axis.set_xlim(-0.3, 5.3)
    axis.set_ylim(-0.3, 5.3)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=17)
    axis.set_ylabel(row_label, fontsize=16, labelpad=8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "configs/kazuki_native_cost_sweep.json",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "assets/paper")
    parser.add_argument("--stem", default="kazuki_native_cost_sweep")
    args = parser.parse_args()
    pairs, spec = load_spec(args.spec)
    if not pairs:
        raise ValueError("coefficient spec contains no pairs")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12.5,
        }
    )
    gamma_colors = {
        gamma: plt.get_cmap("plasma")(0.08 + 0.84 * index / (len(GAMMAS) - 1))
        for index, gamma in enumerate(GAMMAS)
    }
    goal_values = sorted({float(pair["goal_coef"]) for pair in pairs})

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.4), squeeze=False)
    for axis, (metric, title, ylim) in zip(axes.reshape(-1), METRICS):
        for goal_index, goal_coef in enumerate(goal_values):
            subset = sorted(
                (pair for pair in pairs if float(pair["goal_coef"]) == goal_coef),
                key=lambda pair: float(pair["safe_coef"]),
            )
            safe_values = np.asarray([float(pair["safe_coef"]) for pair in subset])
            for gamma in GAMMAS:
                values = np.asarray(
                    [pair["cells"][gamma][metric]["mean"] for pair in subset],
                    dtype=float,
                )
                linestyle = "-" if goal_index == 0 else (0, (4, 2))
                axis.plot(
                    safe_values,
                    values,
                    color=gamma_colors[gamma],
                    linestyle=linestyle,
                    marker="o" if goal_index == 0 else "s",
                    linewidth=2.0,
                    markersize=6,
                    label=(
                        rf"$\gamma={gamma:g},\,w_g={goal_coef:g}$"
                        if metric == "SR"
                        else None
                    ),
                )
        axis.set_title(title)
        axis.set_xlabel(r"safety coefficient $w_s$")
        axis.set_xlim(-0.03, 0.93)
        axis.grid(alpha=0.24)
        if ylim is not None:
            axis.set_ylim(*ylim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    args.outdir.mkdir(parents=True, exist_ok=True)
    metric_outputs = []
    for suffix in ("png", "pdf"):
        output = args.outdir / f"{args.stem}_metrics.{suffix}"
        fig.savefig(output, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        metric_outputs.append(output)
    plt.close(fig)

    # One row per exact coefficient pair; four gamma columns.
    fig, axes = plt.subplots(
        len(pairs), len(GAMMAS), figsize=(4.25 * len(GAMMAS), 4.05 * len(pairs)),
        squeeze=False,
    )
    for row, pair in enumerate(pairs):
        cells = load_paths(Path(pair["archive"]))
        label = rf"$w_g={float(pair['goal_coef']):g},\ w_s={float(pair['safe_coef']):g}$"
        for column, gamma in enumerate(GAMMAS):
            paths, outcomes = cells[gamma]
            draw_scene(
                axes[row, column],
                paths,
                outcomes,
                gamma,
                label if column == 0 else "",
                rf"$\gamma={gamma:g}$" if row == 0 else "",
            )
    fig.subplots_adjust(
        left=0.11, right=0.995, bottom=0.012, top=0.985, wspace=0.025, hspace=0.04
    )
    overlay_outputs = []
    for suffix in ("png", "pdf"):
        output = args.outdir / f"{args.stem}_overlays.{suffix}"
        fig.savefig(output, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        overlay_outputs.append(output)
    plt.close(fig)

    sidecar = {
        "status": "KAZUKI_NATIVE_COST_SWEEP_FIGURES_COMPLETE",
        "spec": str(args.spec.resolve()),
        "pairs": pairs,
        "selection_statement": (
            "No coefficient is promoted by this renderer. The historical ws=0.1 "
            "and ws=0.9 gallery rows were low/high visualization endpoints, not "
            "an optimization result."
        ),
        "outputs": [
            str(path.resolve()) for path in (*metric_outputs, *overlay_outputs)
        ],
        "source_spec": spec,
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
