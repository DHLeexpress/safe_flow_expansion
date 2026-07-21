#!/usr/bin/env python3
"""Paper figure: execution-rule rows at M=200 with adaptive-gamma overlay.

Layout A (b1_exec_rows.{png,pdf}): 2x4 —
  row 1: B1 current best (SafeMPPI-cost execution arm), rounds 0-20
  row 2: SOCP-gated max-step-progress execution arm
  columns: Collision rate | V_safe | Min. clearance [m] | Time-to-goal [s]
  per-gamma plasma lines with 1-sigma-SE bands, pooled black with band,
  adaptive-gamma scheduler green (row 1), ungated-progress baseline pooled
  black dashed (both rows). No confirmation stars, no selected-round line.

Layout B (b1_exec_baseline.{png,pdf}): 1x4 — the ungated
legacy_max_horizon_progress baseline arm alone (M=100), same styling.

Inputs are the JSONL files written by eval_rounds_m.py. The per-gamma
evaluation-temperature schedule (if any) is read from the rows themselves and
recorded in the sidecar JSON next to the figure.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

USE_TEX = False
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)

METRICS = [
    ("CR", r"Collision rate", (-0.03, 1.03)),
    ("v_safe", r"$V_{\mathrm{safe}}$", (-0.03, 1.03)),
    ("clearance", r"Min. clearance [m]", None),
    ("time", r"Time-to-goal [s]", None),
]

TITLES = {
    "row1": "B1 current best",
    "row2": "Max-progress execution",
    "baseline": "Ungated baseline",
    "xlabel": r"expansion round",
    "pooled": r"pooled ($7\gamma$)",
    "adaptive": r"adaptive $\gamma$",
    "baseline_line": r"ungated-progress baseline (pooled)",
}

FONT = {"title": 19, "label": 18, "tick": 15, "legend": 14.5}
LW_GAMMA = 1.4
LW_POOLED = 3.0
BAND_ALPHA = 0.13


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
    })


def gamma_colors():
    cmap = plt.get_cmap("plasma")
    return {g: cmap(0.08 + 0.84 * i / (len(GAMMAS) - 1))
            for i, g in enumerate(GAMMAS)}


def load_rows(paths):
    """Later files override earlier ones per (round, gamma) — this is how the
    per-gamma temperature-override re-evaluations are spliced in."""
    table = {}
    for path in paths:
        for line in Path(path).open():
            row = json.loads(line)
            table[(row["round"], row.get("gamma"))] = row
    return table


def series(table, gamma, metric):
    rounds = sorted({r for (r, g) in table if g == gamma})
    vals, ses = [], []
    for r in rounds:
        entry = table[(r, gamma)][metric]
        vals.append(entry["mean"] if entry["mean"] is not None else np.nan)
        ses.append(entry["se"])
    return np.array(rounds), np.array(vals, float), np.array(ses, float)


def pooled_series(table, metric):
    rounds = sorted({r for (r, g) in table if g is not None})
    vals, ses = [], []
    for r in rounds:
        cell_means, cell_ses = [], []
        for g in GAMMAS:
            entry = table.get((r, g), {}).get(metric)
            if entry and entry["mean"] is not None:
                cell_means.append(entry["mean"])
                cell_ses.append(entry["se"])
        vals.append(np.mean(cell_means) if cell_means else np.nan)
        ses.append(np.sqrt(np.mean(np.square(cell_ses))) / np.sqrt(max(len(cell_ses), 1))
                   if cell_ses else np.nan)
    return np.array(rounds), np.array(vals, float), np.array(ses, float)


def draw_row(axes, table, adaptive_table, baseline_pooled, row_label, show_xlabel):
    colors = gamma_colors()
    for ax, (metric, title, ylim) in zip(axes, METRICS):
        for g in GAMMAS:
            r, v, s = series(table, g, metric)
            ax.plot(r, v, color=colors[g], lw=LW_GAMMA, alpha=0.75, zorder=3)
            ax.fill_between(r, v - s, v + s, color=colors[g],
                            alpha=BAND_ALPHA, lw=0, zorder=1)
        r, v, s = pooled_series(table, metric)
        ax.plot(r, v, color="black", lw=LW_POOLED, zorder=5)
        ax.fill_between(r, v - s, v + s, color="black", alpha=0.11, lw=0, zorder=2)
        if adaptive_table:
            ra = sorted({rr for (rr, g) in adaptive_table if g is None})
            va = [adaptive_table[(rr, None)][metric]["mean"] for rr in ra]
            sa = [adaptive_table[(rr, None)][metric]["se"] for rr in ra]
            va = np.array([np.nan if x is None else x for x in va], float)
            sa = np.array(sa, float)
            ax.plot(ra, va, color="#009e73", lw=2.6, zorder=6)
            ax.fill_between(ra, va - sa, va + sa, color="#009e73",
                            alpha=0.16, lw=0, zorder=2)
        if baseline_pooled is not None:
            rb, vb, _ = baseline_pooled[metric]
            ax.plot(rb, vb, color="black", lw=2.0, ls="--", zorder=4)
        ax.set_title(title, pad=8)
        ax.grid(alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
        rall = sorted({r for (r, g) in table})
        ax.set_xlim(min(rall) - 0.4, max(rall) + 0.4)
        if show_xlabel:
            ax.set_xlabel(TITLES["xlabel"])
    axes[0].set_ylabel(row_label, fontsize=FONT["title"], labelpad=10)


def legend_handles(with_adaptive, with_baseline):
    colors = gamma_colors()
    handles = [plt.Line2D([0], [0], color=colors[g], lw=2.4,
                          label=rf"$\gamma={g:g}$") for g in GAMMAS]
    handles.append(plt.Line2D([0], [0], color="black", lw=LW_POOLED,
                              label=TITLES["pooled"]))
    if with_adaptive:
        handles.append(plt.Line2D([0], [0], color="#009e73", lw=2.6,
                                  label=TITLES["adaptive"]))
    if with_baseline:
        handles.append(plt.Line2D([0], [0], color="black", lw=2.0, ls="--",
                                  label=TITLES["baseline_line"]))
    return handles


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", nargs="+", type=Path, required=True,
                        help="JSONL(s) for the cost arm; later files override")
    parser.add_argument("--prog", nargs="+", type=Path, required=True)
    parser.add_argument("--legacy", nargs="+", type=Path, required=True)
    parser.add_argument("--adaptive", nargs="+", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=root / "assets/paper")
    args = parser.parse_args()

    setup_style()
    cost = load_rows(args.cost)
    prog = load_rows(args.prog)
    legacy = load_rows(args.legacy)
    adaptive = load_rows(args.adaptive) if args.adaptive else None

    baseline_pooled = {m: pooled_series(legacy, m) for m, _, _ in METRICS}

    # Layout A: 2x4.
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.6), squeeze=False)
    draw_row(axes[0], cost, adaptive, baseline_pooled, TITLES["row1"], False)
    draw_row(axes[1], prog, None, baseline_pooled, TITLES["row2"], True)
    fig.legend(handles=legend_handles(adaptive is not None, True),
               loc="upper center", ncol=10, frameon=False,
               bbox_to_anchor=(0.5, 1.005), columnspacing=1.1,
               handletextpad=0.45)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    outputs = []
    for suffix in ("png", "pdf"):
        path = args.outdir / f"b1_exec_rows.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    # Layout B: baseline alone.
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.1), squeeze=False)
    draw_row(axes[0], legacy, None, None, TITLES["baseline"], True)
    fig.legend(handles=legend_handles(False, False), loc="upper center",
               ncol=9, frameon=False, bbox_to_anchor=(0.5, 1.01),
               columnspacing=1.1, handletextpad=0.45)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in ("png", "pdf"):
        path = args.outdir / f"b1_exec_baseline.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    # Sidecar: record the temperature schedule actually used per (gamma, round)
    schedule = defaultdict(dict)
    for (r, g), row in cost.items():
        if g is not None and row.get("temp", 1.0) != 1.0:
            schedule[f"{g:g}"][str(r)] = row["temp"]
    sidecar = {
        "temperature_schedule_cost_arm": dict(schedule),
        "adaptive": (
            {k: v for k, v in next(iter(adaptive.values())).items()
             if k in ("alpha", "beta")} if adaptive else None
        ),
        "inputs": {
            "cost": [str(p) for p in args.cost],
            "prog": [str(p) for p in args.prog],
            "legacy": [str(p) for p in args.legacy],
            "adaptive": [str(p) for p in (args.adaptive or [])],
        },
    }
    sidecar_path = args.outdir / "b1_exec_rows.provenance.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    outputs.append(sidecar_path)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
