#!/usr/bin/env python3
"""Paper-ready method comparison across four safety-conditioning levels.

The four panels are collision rate, sliding-window verifier validity, minimum
clearance, and successful time-to-goal.  A pending method is shown only in the
legend and provenance; no value is imputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_metrics_common import (
    GAMMAS,
    json_ready,
    load_jsonl_round,
    summarize_path_archive,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("CR", "Collision rate", (-0.04, 1.04)),
    ("v_safe", "Validity", (-0.04, 1.04)),
    ("clearance", "Min. clearance [m]", None),
    ("time", "Time-to-goal [s]", None),
)
STYLE = {
    "SafeMPPI": ("#0072B2", "o", "-"),
    "Ours (max safety)": ("#009E73", "s", "-"),
    "Ours (SafeMPPI cost)": ("#6E6E6E", "D", "--"),
    r"CFM–MPPI$^*$ ($w_g=0,w_s=0.1$)": ("#CC79A7", "^", "-"),
    r"CFM–MPPI$^*$ ($w_g=0,w_s=0.9$)": ("#D55E00", "v", "-"),
}


def method_from_jsonl(label: str, path: Path, round_i: int | None) -> dict:
    selected, table = load_jsonl_round(path, round_i)
    return {
        "label": label,
        "status": "complete",
        "round": selected,
        "source": str(path.resolve()),
        "cells": table,
    }


def method_from_paths(label: str, path: Path) -> dict:
    return {
        "label": label,
        "status": "complete",
        "round": None,
        "source": str(path.resolve()),
        "cells": summarize_path_archive(path),
    }


def cell_value(cell: dict, metric: str) -> tuple[float, float]:
    entry = cell[metric]
    return float(entry["mean"]), float(entry["se"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    local = ROOT / "provenance/paper_baselines/local_native_cost_m10"
    parser.add_argument(
        "--safemppi",
        type=Path,
        default=local / "safemppi_ood_m10.npz",
    )
    parser.add_argument(
        "--max-safety-jsonl",
        type=Path,
        default=ROOT
        / "provenance/b1_margin_goal/fixedtemp_m200_revised_r0_r15.jsonl",
    )
    parser.add_argument("--max-safety-round", type=int, default=15)
    parser.add_argument("--cost-jsonl", type=Path, default=None)
    parser.add_argument("--cost-round", type=int, default=None)
    parser.add_argument(
        "--kazuki-low",
        type=Path,
        default=local / "kazuki_wg0_ws0.1_ood_m10.npz",
    )
    parser.add_argument(
        "--kazuki-high",
        type=Path,
        default=local / "kazuki_wg0_ws0.9_ood_m10.npz",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "assets/paper")
    parser.add_argument("--stem", default="method_gamma_comparison")
    args = parser.parse_args()

    methods = [
        method_from_paths("SafeMPPI", args.safemppi),
        method_from_jsonl(
            "Ours (max safety)", args.max_safety_jsonl, args.max_safety_round
        ),
    ]
    if args.cost_jsonl is None:
        methods.append(
            {
                "label": "Ours (SafeMPPI cost)",
                "status": "pending",
                "round": None,
                "source": None,
                "cells": {},
            }
        )
    else:
        methods.append(
            method_from_jsonl(
                "Ours (SafeMPPI cost)", args.cost_jsonl, args.cost_round
            )
        )
    methods.extend(
        [
            method_from_paths(
                r"CFM–MPPI$^*$ ($w_g=0,w_s=0.1$)", args.kazuki_low
            ),
            method_from_paths(
                r"CFM–MPPI$^*$ ($w_g=0,w_s=0.9$)", args.kazuki_high
            ),
        ]
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.4), squeeze=False)
    x = np.arange(len(GAMMAS), dtype=float)
    for axis, (metric, title, ylim) in zip(axes.reshape(-1), METRICS):
        for method in methods:
            color, marker, linestyle = STYLE[method["label"]]
            if method["status"] == "pending":
                continue
            values, errors = [], []
            for gamma in GAMMAS:
                value, error = cell_value(method["cells"][gamma], metric)
                values.append(value)
                errors.append(error)
            values_array = np.asarray(values)
            errors_array = np.asarray(errors)
            axis.errorbar(
                x,
                values_array,
                yerr=1.96 * errors_array,
                label=method["label"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.1,
                markersize=7,
                capsize=3,
            )
        axis.set_title(title)
        axis.set_xticks(x, [rf"$\gamma={gamma:g}$" for gamma in GAMMAS])
        axis.grid(alpha=0.24)
        if ylim is not None:
            axis.set_ylim(*ylim)
    fig.text(
        0.5,
        0.012,
        "Assembly preview: SafeMPPI/CFM–MPPI* use native deployment at M=10; "
        "max-safety uses its declared per-gamma temperatures at M=200. "
        "Do not interpret this as a matched final baseline test.",
        ha="center",
        fontsize=12.5,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    pending = next(method for method in methods if method["status"] == "pending")
    if pending:
        color, marker, linestyle = STYLE[pending["label"]]
        handles.append(
            plt.Line2D(
                [0], [0], color=color, marker=marker, linestyle=linestyle, alpha=0.6
            )
        )
        labels.append(pending["label"] + " (pending)")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))

    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        output = args.outdir / f"{args.stem}.{suffix}"
        fig.savefig(output, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(output)
    plt.close(fig)

    sidecar = {
        "status": "PAPER_METHOD_GAMMA_COMPARISON_COMPLETE",
        "gammas": list(GAMMAS),
        "metrics": [metric for metric, _, _ in METRICS],
        "methods": methods,
        "comparison_scope": (
            "giant-obstacle OOD; path archives are re-scored with the canonical "
            "sliding-window SOCP predicate; the max-safety JSONL retains its "
            "declared per-gamma sampling temperatures"
        ),
        "paper_claim_status": (
            "assembly preview only; rerun every method on one common M and "
            "temperature contract before using this as a comparative claim"
        ),
        "pending_contract": {
            "method": "Ours (SafeMPPI cost)",
            "command": (
                "python scripts/paper_method_gamma_comparison.py "
                "--cost-jsonl /path/to/rounds.jsonl --cost-round ROUND"
            ),
        },
        "outputs": [str(path.resolve()) for path in outputs],
    }
    sidecar_path = args.outdir / f"{args.stem}.json"
    sidecar_path.write_text(
        json.dumps(json_ready(sidecar), indent=2, sort_keys=True) + "\n"
    )
    for path in (*outputs, sidecar_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
