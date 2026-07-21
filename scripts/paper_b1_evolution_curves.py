#!/usr/bin/env python3
"""Paper-ready B1_current_best evolution curves over expansion rounds 0-20.

Data sources (both inside this workbook):
  provenance/b1_current_best/screening_m10_metrics.jsonl
      raw temperature-1 M10/gamma screening of every round checkpoint
      (rounds 0..20, 7 gammas + pooled; the per-round curve data).
  provenance/b1_current_best/metrics.jsonl
      disjoint raw temperature-1 M50/gamma confirmation at r0/r19/r20
      (overlaid as bold markers; this is the scientific result).

Outputs (PNG dpi 300 + vector PDF):
  assets/paper/b1_evolution_grid.{png,pdf}     multi-panel metric evolution
  assets/paper/b1_evolution_compact.{png,pdf}  J-vs-round + SR-J trade-off

Fonts: serif + Computer Modern mathtext (LaTeX look without a TeX install;
switch USE_TEX to True if a LaTeX distribution is available).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Editable configuration: titles, labels, fonts. Adjust freely.
# --------------------------------------------------------------------------
USE_TEX = False  # True requires a LaTeX install (latex not present on Helios)

GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
BEST_ROUND = 19

# (metric key, subplot title) for the grid figure; edit titles here.
GRID_SPECS = [
    ("SR", r"Success rate"),
    ("CR", r"Collision rate"),
    ("V_safe", r"$V_{\mathrm{safe}}$"),
    ("V_full", r"$V_{\mathrm{full}}$"),
    ("clearance", r"Min. clearance [m]"),
    ("time", r"Time-to-goal [s]"),
    ("route_balance", r"U/R balance"),
    ("J", r"Route coverage $J$"),
]

TITLES = {
    "grid_suptitle": "",  # e.g. r"Raw temperature-1 evaluation across expansion rounds"
    "compact_left": r"Successful balanced-route coverage",
    "compact_right": r"SR vs. $J$ trade-off",  # NB: cmr10 has no en-dash glyph
    "compact_suptitle": "",
    "xlabel_round": r"expansion round",
    "ylabel_J": r"$J$",
    "xlabel_SR": r"raw SR",
    "pooled_label": r"pooled ($7\gamma$)",
    "m50_label": r"M50 confirmation",
    "selected_label": rf"selected $r_{{{BEST_ROUND}}}$",
}

FONT = {
    "title": 20,
    "label": 19,
    "tick": 16,
    "legend": 14.5,
    "suptitle": 22,
}

M50_ROUNDS = (0, 19, 20)  # confirmation checkpoints to overlay
# per-round label offsets in the SR-J panel (points); r19/r20 nearly coincide
M50_LABEL_OFFSETS = {0: (8, -18), 19: (10, -22), 20: (-30, 10)}
GRID_NCOLS = 4
LW_GAMMA = 1.3
LW_POOLED = 3.0
# --------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "text.usetex": USE_TEX,
        "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "figure.constrained_layout.use": False,
    })


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def row_lookup(rows: list[dict]) -> dict[tuple[int, float | None], dict]:
    return {
        (int(r["round"]), None if r["gamma"] is None else float(r["gamma"])): r
        for r in rows
    }


def metric_series(rows: list[dict], key: str):
    """Mirror low7_raw_m50_eval._metric_series, plus key 'J'."""
    values, lower, upper = [], [], []
    for row in rows:
        if key in ("SR", "CR", "timeout", "V_safe", "V_full"):
            entry = row["binary"][key]
            value = float(entry["estimate"])
            lo, hi = (float(x) for x in entry["wilson95"])
        elif key == "clearance":
            entry = row["minimum_clearance"]
            value = float(entry["mean"])
            lo, hi = (float(x) for x in entry["bootstrap95"])
        elif key == "time":
            entry = row["successful_time_to_goal"]
            if entry["mean"] is None:
                value = lo = hi = float("nan")
            else:
                value = float(entry["mean"])
                lo, hi = (float(x) for x in entry["bootstrap95"])
        elif key == "route_balance":
            value = float(
                row["route_modes"]["closest_obstacle_approach_success_only"]
                ["coverage_weighted_balance"]
            )
            lo = hi = value
        elif key == "J":
            value = float(row["successful_route_coverage"]["C"])
            lo = hi = value
        else:
            raise KeyError(key)
        values.append(value)
        lower.append(lo)
        upper.append(hi)
    return np.array(values), np.array(lower), np.array(upper)


def gamma_colors():
    cmap = plt.get_cmap("plasma")
    return {
        g: cmap(0.08 + 0.84 * i / (len(GAMMAS) - 1)) for i, g in enumerate(GAMMAS)
    }


def draw_metric_axis(ax, lookup, m50_lookup, rounds, key, title):
    colors = gamma_colors()
    for gamma in GAMMAS:
        series = [lookup[(r, gamma)] for r in rounds]
        vals, _, _ = metric_series(series, key)
        ax.plot(rounds, vals, color=colors[gamma], lw=LW_GAMMA, alpha=0.62, zorder=2)
    pooled = [lookup[(r, None)] for r in rounds]
    vals, lo, hi = metric_series(pooled, key)
    ax.plot(rounds, vals, color="black", lw=LW_POOLED, zorder=4)
    ax.fill_between(rounds, lo, hi, color="black", alpha=0.11, lw=0, zorder=1)
    if m50_lookup is not None:
        try:
            m50_rows = [m50_lookup[(r, None)] for r in M50_ROUNDS]
            m50_vals, _, _ = metric_series(m50_rows, key)
            ax.plot(
                M50_ROUNDS, m50_vals, linestyle="none", marker="*", ms=17,
                mfc="#d55e00", mec="black", mew=1.0, zorder=6,
            )
        except KeyError:
            pass
    ax.axvline(BEST_ROUND, color="#0072b2", ls="--", lw=1.6, zorder=3)
    ax.set_title(title, pad=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
    if key in ("SR", "CR", "timeout", "V_safe", "V_full", "route_balance", "J"):
        ax.set_ylim(-0.03, 1.03)


def shared_legend_handles():
    colors = gamma_colors()
    handles = [
        plt.Line2D([0], [0], color=colors[g], lw=2.4, label=rf"$\gamma={g:g}$")
        for g in GAMMAS
    ]
    handles.append(
        plt.Line2D([0], [0], color="black", lw=LW_POOLED, label=TITLES["pooled_label"])
    )
    handles.append(
        plt.Line2D(
            [0], [0], linestyle="none", marker="*", ms=16, mfc="#d55e00",
            mec="black", label=TITLES["m50_label"],
        )
    )
    handles.append(
        plt.Line2D(
            [0], [0], color="#0072b2", ls="--", lw=1.8,
            label=TITLES["selected_label"],
        )
    )
    return handles


def render_grid(lookup, m50_lookup, rounds, outdir: Path) -> list[Path]:
    nrows = int(np.ceil(len(GRID_SPECS) / GRID_NCOLS))
    fig, axes = plt.subplots(
        nrows, GRID_NCOLS, figsize=(5.1 * GRID_NCOLS, 4.35 * nrows), squeeze=False
    )
    for i, (ax, (key, title)) in enumerate(zip(axes.flat, GRID_SPECS)):
        draw_metric_axis(ax, lookup, m50_lookup, rounds, key, title)
        if i // GRID_NCOLS == nrows - 1:
            ax.set_xlabel(TITLES["xlabel_round"])
    for ax in list(axes.flat)[len(GRID_SPECS):]:
        ax.axis("off")
    fig.legend(
        handles=shared_legend_handles(), loc="upper center",
        ncol=len(GAMMAS) + 3, frameon=False,
        bbox_to_anchor=(0.5, 1.005), columnspacing=1.25, handletextpad=0.5,
    )
    if TITLES["grid_suptitle"]:
        fig.suptitle(TITLES["grid_suptitle"], fontsize=FONT["suptitle"], y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return save_both(fig, outdir, "b1_evolution_grid")


def render_compact(lookup, m50_lookup, rounds, outdir: Path) -> list[Path]:
    fig, (ax_j, ax_sj) = plt.subplots(1, 2, figsize=(15.2, 6.2))
    colors = gamma_colors()

    # Left: J vs round (per gamma + pooled + M50 stars).
    draw_metric_axis(ax_j, lookup, m50_lookup, rounds, "J", TITLES["compact_left"])
    ax_j.set_xlabel(TITLES["xlabel_round"])
    ax_j.set_ylabel(TITLES["ylabel_J"])

    # Right: pooled SR-J trajectory over rounds, colored by round.
    pooled = [lookup[(r, None)] for r in rounds]
    sr, _, _ = metric_series(pooled, "SR")
    jj, _, _ = metric_series(pooled, "J")
    ax_sj.plot(sr, jj, color="0.55", lw=1.2, zorder=2)
    sc = ax_sj.scatter(
        sr, jj, c=rounds, cmap="viridis", s=95, zorder=3,
        edgecolors="black", linewidths=0.7,
    )
    if m50_lookup is not None:
        m50_rows = [m50_lookup[(r, None)] for r in M50_ROUNDS]
        m50_sr, _, _ = metric_series(m50_rows, "SR")
        m50_j, _, _ = metric_series(m50_rows, "J")
        ax_sj.plot(
            m50_sr, m50_j, linestyle="none", marker="*", ms=20,
            mfc="#d55e00", mec="black", mew=1.0, zorder=5,
        )
        for r, x, y in zip(M50_ROUNDS, m50_sr, m50_j):
            ax_sj.annotate(
                rf"$r_{{{r}}}$", (x, y), textcoords="offset points",
                xytext=M50_LABEL_OFFSETS.get(r, (8, -16)),
                fontsize=FONT["legend"],
            )
    cbar = fig.colorbar(sc, ax=ax_sj, pad=0.015)
    cbar.set_label(TITLES["xlabel_round"], fontsize=FONT["label"])
    cbar.ax.tick_params(labelsize=FONT["tick"])
    ax_sj.set_xlabel(TITLES["xlabel_SR"])
    ax_sj.set_ylabel(TITLES["ylabel_J"])
    ax_sj.set_title(TITLES["compact_right"], pad=8)
    ax_sj.set_xlim(-0.03, 1.03)
    ax_sj.set_ylim(-0.03, 1.03)
    ax_sj.grid(alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color=colors[g], lw=2.4, label=rf"$\gamma={g:g}$")
        for g in GAMMAS
    ]
    handles += shared_legend_handles()[len(GAMMAS):]
    fig.legend(
        handles=handles, loc="upper center", ncol=len(GAMMAS) + 3,
        frameon=False, bbox_to_anchor=(0.5, 1.005),
        columnspacing=1.1, handletextpad=0.45,
    )
    if TITLES["compact_suptitle"]:
        fig.suptitle(TITLES["compact_suptitle"], fontsize=FONT["suptitle"], y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_both(fig, outdir, "b1_evolution_compact")


def save_both(fig, outdir: Path, stem: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screening",
        type=Path,
        default=root / "provenance/b1_current_best/screening_m10_metrics.jsonl",
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=root / "provenance/b1_current_best/metrics.jsonl",
    )
    parser.add_argument("--outdir", type=Path, default=root / "assets/paper")
    args = parser.parse_args()

    setup_style()
    screening = load_rows(args.screening)
    lookup = row_lookup(screening)
    rounds = sorted({int(r["round"]) for r in screening})
    m50_lookup = row_lookup(load_rows(args.confirmation))

    outputs = render_grid(lookup, m50_lookup, rounds, args.outdir)
    outputs += render_compact(lookup, m50_lookup, rounds, args.outdir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
