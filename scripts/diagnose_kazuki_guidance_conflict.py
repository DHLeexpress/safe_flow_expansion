#!/usr/bin/env python3
"""Replay one alpha-sweep episode and measure base/guidance direction conflict.

The diagnostic does not alter the executed guided trajectory.  At every
receding-horizon step it additionally integrates the same latent proposals
with both guidance coefficients set to zero, then compares the base terminal
plan displacement with the guidance-induced terminal displacement shift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "source_snapshot/overnight_run_07_06/rev_expansion/codex_overnight"
PAPER = SNAP / "paper_results"
for entry in (SNAP.parents[1], SNAP.parent, SNAP, PAPER):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import kazuki_baseline as baseline
from afe2_scene_profiles import build_scene, get_scene_profile
from b1_current_best_gallery import classify_path, named_seed, seed_all
from grid_hp_expt import load_hp
from run_kazuki_absolute_coefficient_grid import (
    EXPECTED_CHECKPOINT,
    SCHEMA,
    VERSION,
    configure_native_cost,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-10 else 0.0


def rolling_dwell(path: np.ndarray, window: int = 20) -> tuple[float, int]:
    points = np.asarray(path, dtype=float)
    if len(points) <= window:
        return float(np.linalg.norm(points[-1] - points[0])), 0
    distances = np.linalg.norm(points[window:] - points[:-window], axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), index


def matching_archive(manifest: dict, alpha: float, goal: float, safe: float) -> dict:
    for entry in manifest["outputs"]:
        if (
            float(entry["alpha"]) == alpha
            and float(entry["goal_coef"]) == goal
            and float(entry["safe_coef"]) == safe
        ):
            return entry
    raise KeyError((alpha, goal, safe))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=ROOT / "provenance/paper_baselines/kazuki_alpha_fine_grid_m10",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/b1_current_best_r19.pt",
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--goal-coef", type=float, required=True)
    parser.add_argument("--safe-coef", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--rollout-index", type=int, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--outdir", type=Path, default=ROOT / "assets/paper")
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT:
        raise RuntimeError("unexpected r19 checkpoint")
    manifest = json.loads((args.sweep / "manifest.json").read_text())
    source_entry = matching_archive(
        manifest, args.alpha, args.goal_coef, args.safe_coef
    )
    source = args.sweep / source_entry["file"]
    if sha256_file(source) != source_entry["sha256"]:
        raise RuntimeError("sweep archive hash mismatch")

    env = build_scene(get_scene_profile("low7_radius1_canonical_v1"))
    policy, _ = load_hp(str(args.checkpoint), device="cpu")
    policy = policy.to(args.device).eval()
    configure_native_cost(args.goal_coef, args.alpha)
    seed = named_seed(VERSION, "kazuki", args.gamma, args.rollout_index)
    seed_all(seed)

    records: list[dict] = []
    original = baseline.guided_generate

    def instrumented(
        policy_,
        ctx,
        state,
        goal_t,
        obs_xy,
        r_col,
        dt,
        z_init,
        taus,
        safe_coef,
        device,
        ret_guidance=False,
    ):
        guided_z, last_guide = original(
            policy_,
            ctx,
            state,
            goal_t,
            obs_xy,
            r_col,
            dt,
            z_init,
            taus,
            safe_coef,
            device,
            ret_guidance=True,
        )
        current_goal = baseline.GOAL_COEF
        baseline.GOAL_COEF = 0.0
        try:
            base_z = original(
                policy_,
                ctx,
                state,
                goal_t,
                obs_xy,
                r_col,
                dt,
                z_init,
                taus,
                torch.zeros_like(safe_coef),
                device,
                ret_guidance=False,
            )
        finally:
            baseline.GOAL_COEF = current_goal

        horizon = policy_.d // 2
        with torch.no_grad():
            base_u = torch.clamp(
                base_z.reshape(-1, horizon, 2) * policy_.u_max,
                -policy_.u_max,
                policy_.u_max,
            )
            guided_u = torch.clamp(
                guided_z.reshape(-1, horizon, 2) * policy_.u_max,
                -policy_.u_max,
                policy_.u_max,
            )
            base_pos, _ = baseline.di_rollout_t(state, base_u, dt)
            guided_pos, _ = baseline.di_rollout_t(state, guided_u, dt)
            state_xy = np.asarray(state[:2], dtype=float)
            base_vector = (
                base_pos[:, -1].mean(0).detach().cpu().numpy() - state_xy
            )
            shift_vector = (
                guided_pos[:, -1].mean(0) - base_pos[:, -1].mean(0)
            ).detach().cpu().numpy()
            records.append(
                {
                    "state": state_xy,
                    "base_terminal_vector": base_vector,
                    "guidance_terminal_shift": shift_vector,
                    "cosine": cosine(base_vector, shift_vector),
                    "base_norm": float(np.linalg.norm(base_vector)),
                    "shift_norm": float(np.linalg.norm(shift_vector)),
                }
            )
        return (guided_z, last_guide) if ret_guidance else guided_z

    baseline.guided_generate = instrumented
    try:
        result = baseline.kazuki_deploy(
            policy,
            env,
            [args.safe_coef],
            gamma_ctx=args.gamma,
            T=int(manifest["T"]),
            reach=float(manifest["reach"]),
            device=args.device,
            seed=seed,
            conditioning_schema=SCHEMA,
        )
    finally:
        baseline.guided_generate = original

    path = np.asarray(result["path"], dtype=np.float32)
    outcome = classify_path(path, env, float(manifest["reach"]))
    suffix = f"g{args.gamma:g}"
    with np.load(source, allow_pickle=True) as archive:
        expected_path = np.asarray(
            archive[f"paths_{suffix}"][args.rollout_index], dtype=np.float32
        )
        expected_outcome = str(
            archive[f"outcomes_{suffix}"][args.rollout_index]
        )
    if outcome != expected_outcome or not np.allclose(path, expected_path, atol=2e-5):
        raise RuntimeError("instrumented replay changed the retained trajectory")

    states = np.asarray([record["state"] for record in records])
    base_vectors = np.asarray(
        [record["base_terminal_vector"] for record in records]
    )
    shift_vectors = np.asarray(
        [record["guidance_terminal_shift"] for record in records]
    )
    cosines = np.asarray([record["cosine"] for record in records])
    goal = env.goal.detach().cpu().numpy()
    goal_distance = np.linalg.norm(states - goal[None], axis=1)
    dwell_displacement, dwell_start = rolling_dwell(path)
    dwell_end = min(dwell_start + 20, len(path) - 1)

    recipe = json.loads((ROOT / "configs/b1_current_best_recipe.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for obstacle in recipe["scene"]["obstacles"]:
        axes[0].add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#c6c6c6", zorder=1)
        )
    axes[0].plot(path[:, 0], path[:, 1], color="#333333", lw=2.0)
    axes[0].plot(
        path[dwell_start : dwell_end + 1, 0],
        path[dwell_start : dwell_end + 1, 1],
        color="#CC3311",
        lw=4.0,
        alpha=0.75,
        label="minimum-displacement 20-step segment",
    )
    stride = max(1, len(records) // 18)
    indices = np.arange(0, len(records), stride)
    base_unit = base_vectors[indices] / (
        np.linalg.norm(base_vectors[indices], axis=1, keepdims=True) + 1e-9
    )
    shift_unit = shift_vectors[indices] / (
        np.linalg.norm(shift_vectors[indices], axis=1, keepdims=True) + 1e-9
    )
    axes[0].quiver(
        states[indices, 0],
        states[indices, 1],
        base_unit[:, 0],
        base_unit[:, 1],
        color="#0072B2",
        scale=16,
        width=0.006,
        label="base terminal direction",
    )
    axes[0].quiver(
        states[indices, 0],
        states[indices, 1],
        shift_unit[:, 0],
        shift_unit[:, 1],
        color="#D55E00",
        scale=16,
        width=0.006,
        label="guidance-induced shift",
    )
    axes[0].plot(*recipe["scene"]["start_state"][:2], "ks", markersize=5)
    axes[0].plot(
        *recipe["scene"]["goal"],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=12,
    )
    axes[0].set(xlim=(-0.3, 5.3), ylim=(-0.3, 5.3), aspect="equal")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_title(f"{outcome}: task-space direction conflict")

    axes[1].axhline(0.0, color="black", lw=1.0)
    axes[1].fill_between(
        np.arange(len(cosines)),
        -1.0,
        0.0,
        color="#D55E00",
        alpha=0.08,
        label="opposing directions",
    )
    axes[1].plot(cosines, color="#7B3294", lw=1.8, label="cosine")
    axes[1].axvspan(
        dwell_start,
        dwell_end,
        color="#CC3311",
        alpha=0.10,
        label="minimum-displacement segment",
    )
    axes[1].set(xlabel="receding-horizon step", ylabel="direction cosine", ylim=(-1.05, 1.05))
    second = axes[1].twinx()
    second.plot(goal_distance, color="#009E73", alpha=0.75, label="goal distance")
    second.set_ylabel("distance-to-goal [m]", color="#009E73")
    axes[1].set_title(
        rf"$\alpha={args.alpha:g},\,w_g={args.goal_coef:g},\,"
        rf"w_s={args.safe_coef:g},\,\gamma={args.gamma:g}$"
    )
    fig.tight_layout()

    stem = args.stem or (
        f"kazuki_conflict_a{args.alpha:g}_wg{args.goal_coef:g}_"
        f"ws{args.safe_coef:g}_g{args.gamma:g}_i{args.rollout_index}"
    ).replace(".", "p")
    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf"):
        output = args.outdir / f"{stem}.{extension}"
        fig.savefig(
            output, dpi=300 if extension == "png" else None, bbox_inches="tight"
        )
        outputs.append(output)
    plt.close(fig)

    payload = {
        "status": "KAZUKI_GUIDANCE_CONFLICT_DIAGNOSTIC_COMPLETE",
        "alpha": args.alpha,
        "goal_coef": args.goal_coef,
        "safe_coef": args.safe_coef,
        "gamma": args.gamma,
        "rollout_index": args.rollout_index,
        "seed": seed,
        "outcome": outcome,
        "trajectory_reproduced": True,
        "negative_cosine_fraction": float(np.mean(cosines < 0.0)),
        "mean_cosine": float(np.mean(cosines)),
        "minimum_cosine": float(np.min(cosines)),
        "dwell_displacement_20": dwell_displacement,
        "dwell_start": dwell_start,
        "mean_base_norm": float(
            np.mean([record["base_norm"] for record in records])
        ),
        "mean_guidance_shift_norm": float(
            np.mean([record["shift_norm"] for record in records])
        ),
        "diagnostic_definition": (
            "cosine between the zero-guidance mean terminal-plan displacement "
            "and the guidance-induced shift of that displacement, using the "
            "same context and latent proposals"
        ),
        "outputs": [str(path.resolve()) for path in outputs],
    }
    json_path = args.outdir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for output in (*outputs, json_path):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
