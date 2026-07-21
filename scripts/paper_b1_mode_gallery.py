#!/usr/bin/env python3
"""Paper-ready B1_current_best route-mode gallery at full M=50 per gamma.

Every trajectory is one of the retained disjoint raw temperature-1 M50
confirmation rollouts (provenance/b1_current_best/cells/), i.e. the exact
rollout strategy of the scientific evaluation -- no re-rolling, no curation,
all 50 episodes per (round, gamma) cell are shown.

The point of the figure: the U/R binary undercounts what B1 actually covers.
Successful routes are classified into four lanes by the signed perpendicular
offset to the start-goal diagonal at the closest pass to the giant obstacle
(the same signed quantity behind the official U/R route metric); |offset|
splits inner (hugging the giant obstacle) from outer (threading outside the
small-obstacle ring). A histogram strip under each panel shows the four
separated lane clusters at r19 versus the smeared spread at r0.

Outputs (PNG dpi 300 + vector PDF):
  assets/paper/b1_mode_gallery_m50.{png,pdf}

Fonts: serif + Computer Modern mathtext (USE_TEX=True needs a LaTeX install).
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

# --------------------------------------------------------------------------
# Editable configuration: titles, labels, colors, thresholds.
# --------------------------------------------------------------------------
USE_TEX = False

GAMMAS = (0.1, 0.5, 1.0)  # columns; all 7 cells exist if you want more
ROWS = (("Pretrained ($r_0$)", 0), ("B1 current best ($r_{19}$)", 19))

CENTER = (2.5, 2.5)   # giant obstacle center
INNER_OUTER_SPLIT = 0.7  # |offset| below -> inner lane, above -> outer lane

# Lane order fixes legend/annotation order.
MODES = ("R-outer", "R-inner", "U-inner", "U-outer")
MODE_COLORS = {  # Okabe-Ito, colorblind-safe
    "R-outer": "#0072b2",
    "R-inner": "#56b4e9",
    "U-inner": "#e69f00",
    "U-outer": "#d55e00",
}
MODE_LABELS = {  # legend text; edit freely
    "R-outer": r"R outer",
    "R-inner": r"R inner",
    "U-inner": r"U inner",
    "U-outer": r"U outer",
}
MODE_SHORT = {  # compact per-panel count tags
    "R-outer": "RO",
    "R-inner": "RI",
    "U-inner": "UI",
    "U-outer": "UO",
}
FAIL_COLOR = "0.62"

TITLES = {
    "suptitle": "",  # e.g. r"Route modes recovered at temperature 1 (all $M{=}50$)"
    "column": r"$\gamma={g:g}$",
    "hist_xlabel": r"signed lane offset [m]",
    "fail_label": r"failure",
}

FONT = {
    "row_label": 21,
    "column_title": 21,
    "legend": 16,
    "count": 13,
    "hist_tick": 12.5,
    "hist_label": 15,
}

TRAJ_LW = 1.25
TRAJ_ALPHA = 0.75
HIST_BINS = np.linspace(-1.3, 1.3, 53)
# --------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "text.usetex": USE_TEX,
        "axes.unicode_minus": False,
    })


def signed_offset(path: np.ndarray) -> float:
    """Signed perpendicular offset to the start-goal diagonal at the closest
    pass to the giant-obstacle center: positive = U side, negative = R side.
    Same signed construction as the official closest-approach route metric."""
    d = path[:, :2] - np.asarray(CENTER)
    k = int(np.argmin(np.linalg.norm(d, axis=1)))
    normal = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    return float(d[k] @ normal)


def classify(offset: float) -> str:
    side = "U" if offset > 0 else "R"
    lane = "outer" if abs(offset) >= INNER_OUTER_SPLIT else "inner"
    return f"{side}-{lane}"


def load_cell(cells: Path, round_i: int, gamma: float):
    z = np.load(cells / f"r{round_i:03d}_g{gamma:.1f}.npz", allow_pickle=True)
    paths = [np.asarray(p, dtype=float) for p in z["paths"]]
    return paths, np.asarray(z["success"], dtype=bool)


def draw_scene(ax, scene: dict) -> None:
    x0, x1, y0, y1 = scene["workspace_bounds"]
    for ox, oy, orad in scene["obstacles"]:
        ax.add_patch(
            plt.Circle((ox, oy), orad, facecolor="0.82", edgecolor="none", zorder=1)
        )
    ax.plot(
        scene["start_state"][0], scene["start_state"][1],
        "ks", ms=7, zorder=7,
    )
    ax.plot(
        scene["goal"][0], scene["goal"][1], marker="*", color="gold",
        mec="black", ms=17, linestyle="none", zorder=7,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks(())
    ax.set_yticks(())
    for spine in ax.spines.values():
        spine.set_color("0.35")


def render(scene: dict, cells: Path, outdir: Path) -> list[Path]:
    ncols = len(GAMMAS)
    nrows = len(ROWS)
    fig = plt.figure(figsize=(5.5 * ncols, 6.6 * nrows))
    grid = GridSpec(
        2 * nrows, ncols,
        height_ratios=[1.0, 0.24] * nrows,
        hspace=0.16, wspace=0.06,
        left=0.045, right=0.995, top=0.90, bottom=0.055,
    )

    for row_i, (row_label, round_i) in enumerate(ROWS):
        for col_i, gamma in enumerate(GAMMAS):
            ax = fig.add_subplot(grid[2 * row_i, col_i])
            ax_h = fig.add_subplot(grid[2 * row_i + 1, col_i])
            paths, success = load_cell(cells, round_i, gamma)

            counts = {m: 0 for m in MODES}
            offsets: dict[str, list[float]] = {m: [] for m in MODES}
            n_fail = 0
            draw_scene(ax, scene)
            for path, ok in zip(paths, success):
                if ok:
                    mode = classify(signed_offset(path))
                    counts[mode] += 1
                    offsets[mode].append(signed_offset(path))
                    ax.plot(
                        path[:, 0], path[:, 1], color=MODE_COLORS[mode],
                        lw=TRAJ_LW, alpha=TRAJ_ALPHA, zorder=3,
                    )
                else:
                    n_fail += 1
                    ax.plot(
                        path[:, 0], path[:, 1], color=FAIL_COLOR,
                        lw=1.0, alpha=0.55, zorder=2,
                    )
                    ax.plot(
                        path[-1, 0], path[-1, 1], marker="x", linestyle="none",
                        color="#c22417", ms=9, mew=2.2, zorder=6,
                    )

            n_lanes = sum(1 for m in MODES if counts[m] > 0)
            count_text = r" $\cdot$ ".join(
                rf"{MODE_SHORT[m]} {counts[m]}" for m in MODES
            )
            annotation = rf"{n_lanes} lanes: {count_text}"
            if n_fail:
                annotation += rf" $\cdot$ fail {n_fail}"
            ax.text(
                0.5, -0.018, annotation,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=FONT["count"], clip_on=False,
            )
            if row_i == 0:
                ax.set_title(
                    TITLES["column"].format(g=gamma),
                    fontsize=FONT["column_title"], pad=10,
                )
            if col_i == 0:
                ax.set_ylabel(
                    row_label, fontsize=FONT["row_label"], labelpad=12
                )

            # Histogram strip: lane-offset clusters of the successes.
            for m in MODES:
                if offsets[m]:
                    ax_h.hist(
                        offsets[m], bins=HIST_BINS, color=MODE_COLORS[m],
                        alpha=0.95, zorder=3,
                    )
            ax_h.axvline(0.0, color="0.25", lw=0.9, zorder=2)
            for split in (-INNER_OUTER_SPLIT, INNER_OUTER_SPLIT):
                ax_h.axvline(split, color="0.55", lw=0.9, ls=":", zorder=2)
            ax_h.set_xlim(HIST_BINS[0], HIST_BINS[-1])
            ax_h.set_yticks(())
            ax_h.tick_params(labelsize=FONT["hist_tick"])
            ax_h.grid(alpha=0.2, axis="x")
            if row_i == nrows - 1:
                ax_h.set_xlabel(
                    TITLES["hist_xlabel"], fontsize=FONT["hist_label"]
                )
            else:
                ax_h.set_xticklabels(())

    handles = [
        plt.Line2D([0], [0], color=MODE_COLORS[m], lw=3.2, label=MODE_LABELS[m])
        for m in MODES
    ]
    handles.append(
        plt.Line2D(
            [0], [0], color=FAIL_COLOR, lw=2.0, marker="x", mec="#c22417",
            ms=9, mew=2.0, label=TITLES["fail_label"],
        )
    )
    fig.legend(
        handles=handles, loc="upper center", ncol=len(handles), frameon=False,
        fontsize=FONT["legend"], bbox_to_anchor=(0.5, 0.965),
        columnspacing=1.6, handletextpad=0.55,
    )
    if TITLES["suptitle"]:
        fig.suptitle(TITLES["suptitle"], fontsize=FONT["column_title"] + 2, y=0.995)

    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"b1_mode_gallery_m50.{suffix}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", type=Path, default=root / "configs/b1_current_best_recipe.json"
    )
    parser.add_argument(
        "--cells", type=Path, default=root / "provenance/b1_current_best/cells"
    )
    parser.add_argument("--outdir", type=Path, default=root / "assets/paper")
    args = parser.parse_args()

    setup_style()
    scene = json.load(args.recipe.open())["scene"]
    for path in render(scene, args.cells, args.outdir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
