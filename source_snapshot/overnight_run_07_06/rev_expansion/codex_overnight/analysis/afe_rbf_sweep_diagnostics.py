#!/usr/bin/env python3
"""Render acquisition/update diagnostics without treating gather rollouts as evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def _load_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _series(rows, key):
    return np.asarray([
        np.nan if row.get(key) is None else float(row[key]) for row in rows
    ], dtype=float)


def _route_stat(rows, population, statistic):
    return np.asarray([
        float(
            row.get("route_modes_early", {})
            .get(population, {})
            .get(statistic, np.nan)
        )
        for row in rows
    ], dtype=float)


def render(run: Path, output: Path) -> None:
    recipe = _load_json(run / "recipe.json")
    rows = _load_jsonl(run / "probe.jsonl")
    if not rows or [int(row["round"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("probe rows must be a contiguous round-0 sequence")
    rounds = _series(rows, "round")

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 7.3), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(rounds, _series(rows, "beta_used"), marker=".")
    ax.set(title="Adaptive acquisition temperature", xlabel="round", ylabel=r"$\beta_n$")

    ax = axes[0, 1]
    ax.plot(rounds, _series(rows, "ess_med"), marker=".", label="realized")
    target = recipe.get("adaptive_ess_target")
    if target is not None:
        ax.axhline(float(target), color="k", linestyle="--", linewidth=1, label="target")
    ax.set(title="RBF acquisition selectivity", xlabel="round", ylabel="normalized ESS")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False)

    ax = axes[0, 2]
    ax.plot(rounds, _series(rows, "uplift_med"), marker=".")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set(title="Selected minus pool uncertainty", xlabel="round", ylabel="uplift")

    ax = axes[0, 3]
    for population, label in (
        ("all_K", "all K"),
        ("selected_B", "selected B"),
        ("full_H_positive", r"full-H $D^+$"),
        ("executed", "executed"),
    ):
        values = _route_stat(rows, population, "coverage_weighted_balance")
        if np.isfinite(values).any():
            ax.plot(rounds, values, marker=".", label=label)
    ax.set(
        title="Early U/R diversification (diagnostic)",
        xlabel="round",
        ylabel="resolved fraction x U/R balance",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    ax.plot(rounds, _series(rows, "cfm"), label=r"$L^+$")
    negative = _series(rows, "negative_cfm")
    if np.isfinite(negative).any():
        ax.plot(rounds, negative, label=r"$L^-$")
    ax.set(title="Recent-W replay fit", xlabel="round", ylabel="CFM loss")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(rounds, _series(rows, "rep_cos"), marker=".")
    ax.set(title=r"Fixed-probe $\phi_s$ cosine", xlabel="round", ylabel="cosine to round 0")
    ax.set_ylim(0.0, 1.02)

    ax = axes[1, 2]
    n_pos = _series(rows, "n_Dpos")
    n_neg = _series(rows, "n_Dneg")
    n_neutral = _series(rows, "n_Dneutral")
    n_overlap = _series(rows, "n_Doverlap")
    ax.plot(rounds, n_pos, label=r"$|D^+|$")
    ax.plot(rounds, n_neg, label=r"$|D^-|$")
    if np.isfinite(n_neutral).any() and np.nanmax(n_neutral) > 0:
        ax.plot(rounds, n_neutral, label="neither label")
    if np.isfinite(n_overlap).any() and np.nanmax(n_overlap) > 0:
        ax.plot(rounds, n_overlap, linestyle="--", label=r"$D^+\cap D^-$")
    ax.set(title="Stored verifier labels", xlabel="round", ylabel="count")
    ax.legend(frameon=False)

    ax = axes[1, 3]
    ax.plot(
        rounds, _series(rows, "replay_fresh_fraction"), marker=".", label="current round"
    )
    coverage = _series(rows, "replay_epoch_coverage")
    if np.isfinite(coverage).any():
        ax.plot(rounds, coverage, marker=".", label="eligible unique coverage")
    weight_ess = _series(rows, "replay_weight_ess_fraction")
    if np.isfinite(weight_ess).any():
        ax.plot(rounds, weight_ess, marker=".", label="positive-weight ESS")
    clipped = _series(rows, "grad_clipped_fraction")
    if np.isfinite(clipped).any():
        ax.plot(rounds, clipped, marker=".", label="clipped optimizer steps")
    ax.set(
        title="Replay measure diagnostics",
        xlabel="round",
        ylabel="fraction",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=7)

    update_label = (
        "one exact D+ epoch (dynamic steps)"
        if recipe.get("replay_update_mode") == "one_epoch_without_replacement"
        else f"steps={recipe['afe_steps']}"
    )
    fig.suptitle(
        "Gather/update diagnostics only — SR and CR are intentionally absent\n"
        f"ell={recipe['lengthscale_multiplier']:g}×ell0, alpha={recipe['negative_alpha']:g}, "
        f"{update_label}, execution={recipe['execution']}",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    if not (run / "COMPLETE.json").is_file():
        raise RuntimeError("trainer run is incomplete")
    render(run, Path(args.out).resolve())


if __name__ == "__main__":
    main()
