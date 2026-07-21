#!/usr/bin/env python3
"""Paper figure: what Kazuki's CFM-MPPI actually does on the canonical
giant-obstacle task, across the collapse test and the faithful transplants.

Panels (top row, one scene per arm, trajectories colored by outcome):
  raw          pretrained FM, temperature 1, gamma 0.5 (M50 confirmation cell
               r000_g0.5 -- the collapse-test reference distribution)
  zero         Kazuki windowed port, safe_coef=0, goal_coef=0 (MPPI stage
               cost active)
  zero_nocost  all four coefficients zero
  faithful     their deployment protocol (mixed 5-coef batch, goal_coef 0.1)
  fullh        full-horizon variant (chained-window proposals, their refine)

Bottom row: the full-horizon mechanism trace -- executed path plus the
refined best plan recorded every 10 steps, showing the planned trajectory
leaving the workspace (their open-space cost admits exit-the-arena minima on
a walled scene).

Outputs: assets/paper/kazuki_faithful_comparison.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

USE_TEX = False

OUTCOME_COLORS = {"SR": "#0072b2", "CR": "#c22417", "TO": "#e69f00"}
OUTCOME_LABELS = {"SR": "success", "CR": "collision", "TO": "timeout"}

TITLES = {
    "raw": r"Raw pretrained ($\gamma=0.5$)",
    "zero": r"$w_s=0$, $w_g=0$",
    "zero_nocost": r"all coefficients $=0$",
    "faithful": r"faithful (mixed $w_s$, $w_g{=}0.1$)",
    "fullh": r"full-horizon variant",
    "bounds": r"full-horizon $+$ containment repair",
    "trace": r"Full-horizon mechanism: refined plan leaves the workspace",
}

FONT = {"title": 17.5, "count": 13.5, "legend": 15, "label": 16}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "text.usetex": USE_TEX,
        "axes.unicode_minus": False,
    })


def draw_scene(ax, scene: dict) -> None:
    for ox, oy, orad in scene["obstacles"]:
        ax.add_patch(plt.Circle((ox, oy), orad, facecolor="0.82",
                                edgecolor="none", zorder=1))
    ax.plot(scene["start_state"][0], scene["start_state"][1], "ks",
            ms=6, zorder=7)
    ax.plot(scene["goal"][0], scene["goal"][1], marker="*", color="gold",
            mec="black", ms=15, linestyle="none", zorder=7)
    ax.set_aspect("equal")
    ax.set_xticks(())
    ax.set_yticks(())
    for spine in ax.spines.values():
        spine.set_color("0.35")


def draw_panel(ax, scene, paths, outcomes, title):
    draw_scene(ax, scene)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    for path, outcome in zip(paths, outcomes):
        color = OUTCOME_COLORS[outcome]
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.0,
                alpha=0.55 if outcome == "SR" else 0.75, zorder=3)
        if outcome != "SR":
            ax.plot(path[-1, 0], path[-1, 1], marker="x", linestyle="none",
                    color=color, ms=8, mew=2.0, zorder=6)
    counts = {k: sum(o == k for o in outcomes) for k in ("SR", "CR", "TO")}
    n = max(len(outcomes), 1)
    ax.set_title(title, fontsize=FONT["title"], pad=8)
    ax.text(0.5, -0.02,
            rf"SR {counts['SR']/n:.2f} $\cdot$ CR {counts['CR']/n:.2f} "
            rf"$\cdot$ TO {counts['TO']/n:.2f}  ($M={n}$)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT["count"])


def load_raw_cell(cells: Path):
    z = np.load(cells / "r000_g0.5.npz", allow_pickle=True)
    paths = [np.asarray(p, dtype=float) for p in z["paths"]]
    outcomes = []
    for i in range(len(paths)):
        if bool(z["cr"][i]):
            outcomes.append("CR")
        elif bool(z["success"][i]):
            outcomes.append("SR")
        else:
            outcomes.append("TO")
    return paths, outcomes


def load_arm(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    return ([np.asarray(p, dtype=float) for p in z["paths"]],
            [str(o) for o in z["outcomes"]])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windowed-dir", type=Path, required=True,
                        help="outdir of run_kazuki_faithful_m50.py")
    parser.add_argument("--fullh-npz", type=Path, required=True)
    parser.add_argument("--bounds-npz", type=Path, default=None)
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=root / "assets/paper")
    args = parser.parse_args()

    setup_style()
    scene = json.load((root / "configs/b1_current_best_recipe.json").open())["scene"]
    cells = root / "provenance/b1_current_best/cells"

    arms = [("raw", *load_raw_cell(cells))]
    for arm in ("zero", "zero_nocost", "faithful"):
        matches = sorted(args.windowed_dir.glob(f"kazuki_{arm}_m*.npz"))
        arms.append((arm, *load_arm(matches[-1])))
    arms.append(("fullh", *load_arm(args.fullh_npz)))
    if args.bounds_npz is not None:
        arms.append(("bounds", *load_arm(args.bounds_npz)))

    fig = plt.figure(figsize=(5.0 * len(arms), 5.0 + 5.4))
    grid = GridSpec(2, len(arms), height_ratios=[1.0, 1.05],
                    hspace=0.17, wspace=0.05,
                    left=0.02, right=0.985, top=0.90, bottom=0.05)
    for i, (arm, paths, outcomes) in enumerate(arms):
        ax = fig.add_subplot(grid[0, i])
        draw_panel(ax, scene, paths, outcomes, TITLES[arm])

    # Mechanism trace panel (spans all columns).
    ax = fig.add_subplot(grid[1, :])
    trace = np.load(args.trace_npz, allow_pickle=True)
    draw_scene(ax, scene)
    plan_keys = sorted(
        (k for k in trace.files if k.startswith("plan_")),
        key=lambda k: int(k.split("_")[1]),
    )
    cmap = plt.get_cmap("viridis")
    for j, key in enumerate(plan_keys):
        plan = trace[key]
        t = int(key.split("_")[1])
        ax.plot(plan[:, 0], plan[:, 1], color=cmap(j / max(len(plan_keys) - 1, 1)),
                lw=1.6, alpha=0.85, zorder=3,
                label=rf"plan at $t={t}$")
    path = trace["path"]
    ax.plot(path[:, 0], path[:, 1], color="black", lw=3.0, zorder=5,
            label="executed")
    ax.plot(path[-1, 0], path[-1, 1], marker="x", linestyle="none",
            color="#c22417", ms=13, mew=3.0, zorder=6)
    ax.add_patch(plt.Rectangle((0, 0), 5, 5, fill=False, edgecolor="#0072b2",
                               lw=1.6, ls="--", zorder=4))
    pxmax = max(float(trace[k][:, 0].max()) for k in plan_keys)
    pymax = max(float(trace[k][:, 1].max()) for k in plan_keys)
    ax.set_xlim(-0.4, max(5.4, pxmax + 0.4))
    ax.set_ylim(-0.4, max(5.4, pymax + 0.4))
    ax.set_title(TITLES["trace"], fontsize=FONT["title"], pad=8)
    ax.legend(loc="upper right", fontsize=FONT["count"], ncol=2, frameon=False)

    handles = [
        plt.Line2D([0], [0], color=OUTCOME_COLORS[k], lw=3,
                   label=OUTCOME_LABELS[k])
        for k in ("SR", "CR", "TO")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=FONT["legend"], bbox_to_anchor=(0.5, 0.965))

    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        out = args.outdir / f"kazuki_faithful_comparison.{suffix}"
        fig.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(fig)
    for out in outputs:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
