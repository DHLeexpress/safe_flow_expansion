#!/usr/bin/env python3
"""Qualify raw pretrained U/R balance on the canonical giant-obstacle scene.

This evaluator never uses RBF acquisition, a verifier, SafeMPPI, or route labels
to choose an action.  It first generates one fixed raw temperature-1 rollout
bank and only then labels each trajectory by its closest approach to the giant
obstacle.  The default gate requires at least 80% U/R balance for both all
trajectories and successful trajectories, plus 95% resolved route labels,
independently at every declared gamma.  Gating successful routes is essential:
the expansion studies measure route coverage among successful raw rollouts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
CODEX_OVERNIGHT = HERE.parent
REV_EXPANSION = CODEX_OVERNIGHT.parent
WORK = REV_EXPANSION.parent
CHALLENGING = REV_EXPANSION / "codex_challenging"
for path in (WORK, REV_EXPANSION, CODEX_OVERNIGHT, CHALLENGING):
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import afe_route_metrics as RM  # noqa: E402
from afe_restart import evaluate_low7_pretrained as EVAL  # noqa: E402


SCENE_NAME = "low7_radius1_canonical_v1"
STATUS = "LOW7_REFLECTION_R0_QUALIFICATION_COMPLETE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mode_label(path: np.ndarray, env: Any) -> int:
    labels, _closest = RM.classify_trajectories_at_closest_approach(
        np.asarray(path, dtype=np.float64)[None],
        start=env.x0[:2].detach().cpu().numpy(),
        goal=env.goal.detach().cpu().numpy(),
        obstacle_centers=np.asarray((2.5, 2.5), dtype=np.float64),
        obstacle_radii=1.0,
        ambiguity_band=RM.DEFAULT_AMBIGUITY_BAND,
    )
    return int(labels[0])


def _draw_scene(axis: Any, env: Any) -> None:
    for x, y, radius in env.obstacles.detach().cpu().numpy():
        axis.add_patch(
            plt.Circle((x, y), radius, color="#c5c5c5", ec="none", zorder=1)
        )
    axis.set(xlim=(-0.35, 5.35), ylim=(-0.35, 5.35))
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def _gallery_rows(
    rows: list[dict[str, Any]], count: int, *, reflection_antithetic: bool
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["rollout_index"])
    if not reflection_antithetic:
        return ordered[:count]
    if len(ordered) % 2:
        raise ValueError("reflection-antithetic gallery requires an even cell size")
    half = len(ordered) // 2
    interleaved = [
        row
        for pair_index in range(half)
        for row in (ordered[pair_index], ordered[pair_index + half])
    ]
    return interleaved[:count]


def _render_gallery(
    output: Path,
    episodes: list[dict[str, Any]],
    env: Any,
    summaries: dict[str, dict[str, Any]],
    count: int,
    *,
    reflection_antithetic: bool,
) -> None:
    colors = {int(RM.MODE_U): "#3366cc", int(RM.MODE_R): "#ee7733", 0: "#777777"}
    figure, axes = plt.subplots(1, len(EVAL.GAMMAS), figsize=(23.5, 3.8), squeeze=False)
    for column, gamma in enumerate(EVAL.GAMMAS):
        axis = axes[0, column]
        _draw_scene(axis, env)
        cell = _gallery_rows(
            [row for row in episodes if row["gamma"] == gamma],
            count,
            reflection_antithetic=reflection_antithetic,
        )
        for row in cell:
            path = np.asarray(row["path"])
            color = colors[row["route_mode"]]
            axis.plot(path[:, 0], path[:, 1], color=color, lw=1.0, alpha=0.68, zorder=3)
            axis.scatter(
                path[::5, 0], path[::5, 1], s=2.0, color=color, alpha=0.48, zorder=4
            )
            if not row["metrics"]["success"]:
                axis.plot(path[-1, 0], path[-1, 1], "x", color="#cc3311", ms=4, zorder=5)
        entry = summaries[f"{gamma:g}"]
        route = entry["all_routes"]
        success_route = entry["successful_routes"]
        axis.set_title(
            rf"$\gamma={gamma:g}$  U/R={route['u_count']}/{route['r_count']}"
            + f"\nsuccess U/R={success_route['u_count']}/{success_route['r_count']}"
            + f"\nSR={entry['SR']:.2f}, bal={success_route['balance']:.2f}",
            fontsize=9,
        )
        axis.plot(*env.x0[:2].detach().cpu().numpy(), "ks", ms=3.5, zorder=6)
        axis.plot(
            *env.goal.detach().cpu().numpy(), marker="*", color="gold", mec="k", ms=8, zorder=6
        )
    figure.legend(
        handles=(
            Line2D([], [], color=colors[int(RM.MODE_U)], lw=2, label="U route"),
            Line2D([], [], color=colors[int(RM.MODE_R)], lw=2, label="R route"),
            Line2D([], [], color="#cc3311", marker="x", lw=0, label="failure endpoint"),
        ),
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        (
            f"Raw temperature-1 pretrained rollouts; {count // 2} reflection pairs"
            if reflection_antithetic
            else f"Raw temperature-1 pretrained rollouts; fixed indices 0..{count - 1}"
        ),
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.90))
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.outdir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    policy, checkpoint = EVAL.load_low7_candidate(
        args.checkpoint, args.expected_checkpoint_sha256, device
    )
    env = EVAL.build_scene(EVAL.get_scene_profile(SCENE_NAME))
    scene = EVAL.validate_scene_contract(SCENE_NAME, env)
    episodes, _plans = EVAL.run_raw_rollouts(
        policy,
        env,
        SCENE_NAME,
        m=args.m,
        gammas=EVAL.GAMMAS,
        horizon=args.horizon,
        nfe=args.nfe,
        device=device,
        seed_bank=args.seed_bank,
        reflection_antithetic=args.reflection_antithetic,
    )
    summaries: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        episode["metrics"] = EVAL.trajectory_metrics(episode, env)
        episode["route_mode"] = _mode_label(episode["path"], env)
    failures: list[str] = []
    for gamma in EVAL.GAMMAS:
        cell = [row for row in episodes if row["gamma"] == gamma]
        labels = np.asarray([row["route_mode"] for row in cell], dtype=np.int8)
        all_routes = RM.summarize_modes(labels)
        success_labels = labels[[row["metrics"]["success"] for row in cell]]
        successful_routes = RM.summarize_modes(success_labels)
        resolved = int(all_routes["resolved_count"])
        all_routes["u_fraction_wilson95"] = _wilson(
            int(all_routes["u_count"]), resolved
        )
        successful_routes["u_fraction_wilson95"] = _wilson(
            int(successful_routes["u_count"]),
            int(successful_routes["resolved_count"]),
        )
        success_count = sum(bool(row["metrics"]["success"]) for row in cell)
        collision_count = sum(bool(row["metrics"]["collision"]) for row in cell)
        entry = {
            "gamma": gamma,
            "M": len(cell),
            "SR": success_count / len(cell),
            "CR": collision_count / len(cell),
            "success_count": success_count,
            "collision_count": collision_count,
            "all_routes": all_routes,
            "successful_routes": successful_routes,
        }
        summaries[f"{gamma:g}"] = entry
        if float(all_routes["balance"]) < args.minimum_balance:
            failures.append(
                f"gamma={gamma:g} balance={all_routes['balance']:.4f} < {args.minimum_balance:.4f}"
            )
        if not (
            all_routes["u_fraction_wilson95"][0]
            <= 0.5
            <= all_routes["u_fraction_wilson95"][1]
        ):
            failures.append(
                f"gamma={gamma:g} all-route U fraction rejects 0.5 at Wilson95"
            )
        if success_count < args.minimum_successes:
            failures.append(
                f"gamma={gamma:g} successes={success_count} < {args.minimum_successes}"
            )
        elif float(successful_routes["balance"]) < args.minimum_success_balance:
            failures.append(
                f"gamma={gamma:g} successful_balance={successful_routes['balance']:.4f} "
                f"< {args.minimum_success_balance:.4f}"
            )
        if (
            success_count >= args.minimum_successes
            and not (
                successful_routes["u_fraction_wilson95"][0]
                <= 0.5
                <= successful_routes["u_fraction_wilson95"][1]
            )
        ):
            failures.append(
                f"gamma={gamma:g} successful-route U fraction rejects 0.5 at Wilson95"
            )
        if (
            success_count >= args.minimum_successes
            and float(successful_routes["resolved_fraction"])
            < args.minimum_resolved_fraction
        ):
            failures.append(
                f"gamma={gamma:g} successful_resolved="
                f"{successful_routes['resolved_fraction']:.4f} "
                f"< {args.minimum_resolved_fraction:.4f}"
            )
        if float(all_routes["resolved_fraction"]) < args.minimum_resolved_fraction:
            failures.append(
                f"gamma={gamma:g} resolved={all_routes['resolved_fraction']:.4f} "
                f"< {args.minimum_resolved_fraction:.4f}"
            )
    passed = not failures
    payload = {
        "status": STATUS,
        "created_at_utc": _utc_now(),
        "passed": passed,
        "failures": failures,
        "scientific_mode": "raw untilted temperature-1 generator; no verifier/acquisition/expert",
        "raw_noise_design": (
            "reflection-antithetic common-random-number pairs"
            if args.reflection_antithetic
            else "iid common-random-number bank"
        ),
        "selection_warning": (
            "this seed bank may be used for qualification but a selected model requires a disjoint confirmation bank"
        ),
        "checkpoint": checkpoint,
        "scene": scene,
        "M_per_gamma": args.m,
        "seed_bank": args.seed_bank,
        "nfe": args.nfe,
        "horizon": args.horizon,
        "gate": {
            "minimum_balance_every_gamma": args.minimum_balance,
            "minimum_success_balance_every_gamma": args.minimum_success_balance,
            "minimum_successes_every_gamma": args.minimum_successes,
            "minimum_resolved_fraction_every_gamma": args.minimum_resolved_fraction,
            "u_fraction_wilson95_must_contain_half": True,
        },
        "per_gamma": summaries,
    }
    (output / "qualification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    max_steps = max(len(row["path"]) for row in episodes)
    paths = np.full((len(episodes), max_steps, 2), np.nan, dtype=np.float32)
    lengths = np.zeros(len(episodes), dtype=np.int16)
    for index, row in enumerate(episodes):
        length = len(row["path"])
        paths[index, :length] = row["path"]
        lengths[index] = length
    np.savez_compressed(
        output / "raw_rollouts.npz",
        paths=paths,
        lengths=lengths,
        gamma=np.asarray([row["gamma"] for row in episodes], dtype=np.float32),
        rollout_index=np.asarray([row["rollout_index"] for row in episodes], dtype=np.int16),
        status=np.asarray([row["status"] for row in episodes]),
        route_mode=np.asarray([row["route_mode"] for row in episodes], dtype=np.int8),
    )
    _render_gallery(
        output / "raw_r0_balanced_gallery.png",
        episodes,
        env,
        summaries,
        min(args.gallery_count, args.m),
        reflection_antithetic=args.reflection_antithetic,
    )
    if not passed and not args.report_only:
        raise RuntimeError("raw r0 U/R qualification failed: " + "; ".join(failures))
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--M", dest="m", type=int, default=100)
    parser.add_argument("--nfe", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--gallery-count", type=int, default=20)
    parser.add_argument("--seed-bank", default="low7-balanced-r0-qualification-v1")
    parser.add_argument(
        "--reflection-antithetic",
        action="store_true",
        help=(
            "evaluate each raw temperature-1 noise path together with its x/y "
            "reflection; requires an exactly group-averaged policy and even M"
        ),
    )
    parser.add_argument("--minimum-balance", type=float, default=0.8)
    parser.add_argument("--minimum-success-balance", type=float, default=0.8)
    parser.add_argument("--minimum-successes", type=int, default=10)
    parser.add_argument("--minimum-resolved-fraction", type=float, default=0.95)
    parser.add_argument("--report-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if min(args.m, args.nfe, args.horizon, args.gallery_count) <= 0:
        raise ValueError("M, nfe, horizon, and gallery-count must be positive")
    if args.reflection_antithetic and args.m % 2:
        raise ValueError("reflection-antithetic qualification requires even M")
    if not 0.0 <= args.minimum_balance <= 1.0:
        raise ValueError("minimum-balance must lie in [0,1]")
    if not 0.0 <= args.minimum_success_balance <= 1.0:
        raise ValueError("minimum-success-balance must lie in [0,1]")
    if args.minimum_successes < 1:
        raise ValueError("minimum-successes must be positive")
    if not 0.0 <= args.minimum_resolved_fraction <= 1.0:
        raise ValueError("minimum-resolved-fraction must lie in [0,1]")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
