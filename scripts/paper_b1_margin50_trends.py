#!/usr/bin/env python3
"""Render the four requested B1 margin-law M=50 trends for one or more arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
SPECS = (
    ("CR", "Collision rate", (-0.03, 1.03)),
    ("v_safe", r"$V_{\mathrm{safe}}$", (-0.03, 1.03)),
    ("clearance", "Min. clearance [m]", None),
    ("time", "Time-to-goal [s]", None),
)


def load_arm(spec: str):
    if "=" not in spec:
        raise ValueError("--arm must be LABEL=JSONL")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    table = {(int(row["round"]), float(row["gamma"])): row for row in rows}
    rounds = sorted({key[0] for key in table})
    if set(key[1] for key in table) != set(GAMMAS):
        raise RuntimeError(f"{label} does not contain all seven gamma cells")
    return label, path, table, rounds


def series(table, rounds, gamma, metric):
    entries = [table[(round_i, gamma)][metric] for round_i in rounds]
    mean = np.asarray([
        np.nan if entry["mean"] is None else entry["mean"] for entry in entries
    ], dtype=float)
    se = np.asarray([entry["se"] for entry in entries], dtype=float)
    return mean, se


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="LABEL=JSONL")
    parser.add_argument(
        "--highlight", action="append", default=[],
        help="TARGET_ARM_LABEL=JSONL; overlay separately confirmed cells as stars",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--stem", default="b1_margin50_metric_trends")
    args = parser.parse_args()

    arms = [load_arm(spec) for spec in args.arm]
    highlights = {label: (path, table, rounds) for label, path, table, rounds in (
        load_arm(spec) for spec in args.highlight
    )}
    unknown = set(highlights) - {label for label, _, _, _ in arms}
    if unknown:
        raise ValueError(f"highlight targets are not declared arms: {sorted(unknown)}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    colors = {
        gamma: plt.get_cmap("plasma")(0.08 + 0.84 * i / (len(GAMMAS) - 1))
        for i, gamma in enumerate(GAMMAS)
    }
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.titlesize": 18, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
    })
    fig, axes = plt.subplots(
        len(arms), 4, figsize=(20.5, 4.4 * len(arms)), squeeze=False
    )
    for row_i, (label, _, table, rounds) in enumerate(arms):
        for axis, (metric, title, ylim) in zip(axes[row_i], SPECS):
            gamma_means = []
            gamma_ses = []
            for gamma in GAMMAS:
                mean, se = series(table, rounds, gamma, metric)
                gamma_means.append(mean)
                gamma_ses.append(se)
                axis.plot(rounds, mean, color=colors[gamma], lw=1.35, alpha=0.75)
                axis.fill_between(
                    rounds, mean - se, mean + se, color=colors[gamma], alpha=0.10,
                    linewidth=0,
                )
            pooled = np.nanmean(np.stack(gamma_means), axis=0)
            pooled_se = np.sqrt(np.nanmean(np.square(np.stack(gamma_ses)), axis=0))
            axis.plot(rounds, pooled, color="black", lw=3.0)
            axis.fill_between(
                rounds, pooled - pooled_se, pooled + pooled_se,
                color="black", alpha=0.10, linewidth=0,
            )
            if label in highlights:
                _, highlight_table, highlight_rounds = highlights[label]
                for gamma in GAMMAS:
                    values, _ = series(
                        highlight_table, highlight_rounds, gamma, metric
                    )
                    axis.scatter(
                        highlight_rounds, values, marker="*", s=145,
                        facecolor=colors[gamma], edgecolor="black", linewidth=0.7,
                        zorder=8,
                    )
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
            if ylim is not None:
                axis.set_ylim(*ylim)
            if row_i == len(arms) - 1:
                axis.set_xlabel("expansion round")
        axes[row_i, 0].set_ylabel(label, fontsize=17, labelpad=12)

    handles = [
        plt.Line2D([0], [0], color=colors[gamma], lw=2.2, label=rf"$\gamma={gamma:g}$")
        for gamma in GAMMAS
    ]
    handles.append(plt.Line2D([0], [0], color="black", lw=3.0, label="pooled"))
    if highlights:
        handles.append(plt.Line2D(
            [0], [0], marker="*", linestyle="none", markersize=12,
            markerfacecolor="white", markeredgecolor="black",
            label="calibrated holdout",
        ))
    fig.legend(handles=handles, ncol=8, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outputs = []
    for suffix in ("png", "pdf"):
        path = args.outdir / f"{args.stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    provenance = {
        "status": "B1_MARGIN50_METRIC_TRENDS_COMPLETE",
        "inputs": {label: str(path) for label, path, _, _ in arms},
        "highlight_inputs": {
            label: str(path) for label, (path, _, _) in highlights.items()
        },
        "M_per_gamma": sorted({
            table[(rounds[0], GAMMAS[0])]["m"]
            for _, _, table, rounds in arms
        }),
        "claim": (
            "curves are raw temperature-1 metrics-only evaluation; star highlights "
            "are separately calibrated and confirmed metrics; no trajectory archives"
        ),
        "outputs": [str(path) for path in outputs],
    }
    sidecar = args.outdir / f"{args.stem}.json"
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    for path in (*outputs, sidecar):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
