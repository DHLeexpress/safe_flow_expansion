#!/usr/bin/env python3
"""Run the native-cost CFM-MPPI absolute coefficient grid locally.

Only guided-flow ``(alpha, w_goal, w_safe)`` changes.  The r19 checkpoint,
giant-obstacle scene, exact B1 SafeMPPI refinement cost, and common
random-number seed bank remain fixed.  Its historical default still reproduces
the retained alpha=1 absolute-coefficient screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

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


VERSION = "kazuki_absolute_coefficient_grid_v1"
EXPECTED_CHECKPOINT = "60c155472f5ed0e4a1d53581857f09aead7924f8ce11e8e3adf890d5a57fc079"
COEFFICIENTS = (0.0, 1.0, 2.0, 3.0)
GAMMAS = (0.1, 1.0)
SCHEMA = "low7_closest_boundary_tie_mean"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coefficient_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def configure_native_cost(goal_coef: float, alpha: float) -> None:
    baseline.GOAL_COEF = float(goal_coef)
    baseline.A_CBF = float(alpha)
    baseline.COLL_W = 100.0
    baseline.GOAL_W = 0.1
    baseline.BETA_MPPI = 20.0
    baseline.MPPI_LAMBDA = 0.1
    baseline.MPPI_SIGMA = 0.2
    baseline.R_MARGIN = 0.05
    baseline.N_SAMPLE = 200
    baseline.N_ELITE = 10
    baseline.N_COPY = 200
    baseline.REFINEMENT_COST = "b1_safemppi"


def run_cell(
    policy,
    env,
    goal_coef: float,
    safe_coef: float,
    alpha: float,
    gamma: float,
    m: int,
    t_cap: int,
    reach: float,
    device: str,
) -> tuple[list[np.ndarray], list[str]]:
    configure_native_cost(goal_coef, alpha)
    paths: list[np.ndarray] = []
    outcomes: list[str] = []
    started = time.perf_counter()
    for rollout_index in range(m):
        # Excluding coefficients gives every arm the same stochastic stream.
        seed = named_seed(VERSION, "kazuki", gamma, rollout_index)
        seed_all(seed)
        result = baseline.kazuki_deploy(
            policy,
            env,
            [safe_coef],
            gamma_ctx=gamma,
            T=t_cap,
            reach=reach,
            device=device,
            seed=seed,
            conditioning_schema=SCHEMA,
        )
        path = np.asarray(result["path"], dtype=np.float32)
        outcome = classify_path(path, env, reach)
        paths.append(path)
        outcomes.append(outcome)
        counts = {label: outcomes.count(label) for label in ("SR", "CR", "TO")}
        print(
            f"[alpha={alpha:g} wg={goal_coef:g} ws={safe_coef:g} gamma={gamma:g}] "
            f"{rollout_index + 1}/{m} {outcome} steps={len(path) - 1} "
            f"SR={counts['SR'] / len(outcomes):.2f} "
            f"CR={counts['CR'] / len(outcomes):.2f} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return paths, outcomes


def pack_pair(
    path: Path,
    cells: dict[float, tuple[list[np.ndarray], list[str]]],
    goal_coef: float,
    safe_coef: float,
    alpha: float,
) -> None:
    payload: dict[str, np.ndarray] = {}
    for gamma, (paths, outcomes) in cells.items():
        suffix = f"g{gamma:g}"
        packed = np.empty(len(paths), dtype=object)
        packed[:] = paths
        payload[f"paths_{suffix}"] = packed
        payload[f"outcomes_{suffix}"] = np.asarray(outcomes)
    payload["config_json"] = np.asarray(
        json.dumps(
            {
                "goal_coef": goal_coef,
                "safe_coef": safe_coef,
                "alpha": alpha,
                "refinement_cost": "b1_safemppi",
            },
            sort_keys=True,
        )
    )
    np.savez_compressed(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/b1_current_best_r19.pt",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "provenance/paper_baselines/kazuki_absolute_grid_m10",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--reach", type=float, default=0.15)
    parser.add_argument(
        "--goal-coefficients",
        nargs="+",
        type=float,
        default=list(COEFFICIENTS),
    )
    parser.add_argument(
        "--safe-coefficients",
        nargs="+",
        type=float,
        default=list(COEFFICIENTS),
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--gammas", nargs="+", type=float, default=list(GAMMAS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"unexpected r19 checkpoint: {checkpoint_sha}")
    if args.outdir.exists() and not args.resume:
        raise FileExistsError(f"fresh output directory required: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    env = build_scene(get_scene_profile("low7_radius1_canonical_v1"))
    policy, _ = load_hp(str(args.checkpoint), device="cpu")
    policy = policy.to(args.device).eval()

    goal_coefficients = tuple(float(value) for value in args.goal_coefficients)
    safe_coefficients = tuple(float(value) for value in args.safe_coefficients)
    alphas = tuple(float(value) for value in args.alphas)
    gammas = tuple(float(value) for value in args.gammas)
    include_alpha_tag = len(alphas) != 1 or alphas[0] != 1.0
    outputs: list[dict[str, object]] = []
    sweep_started = time.perf_counter()
    for alpha in alphas:
        for goal_coef in goal_coefficients:
            for safe_coef in safe_coefficients:
                alpha_tag = (
                    f"a{coefficient_tag(alpha)}_" if include_alpha_tag else ""
                )
                output = args.outdir / (
                    f"kazuki_{alpha_tag}wg{coefficient_tag(goal_coef)}_"
                    f"ws{coefficient_tag(safe_coef)}_m{args.M}.npz"
                )
                if output.exists() and args.resume:
                    print(f"[resume] {output.name}", flush=True)
                else:
                    cells = {
                        gamma: run_cell(
                            policy,
                            env,
                            goal_coef,
                            safe_coef,
                            alpha,
                            gamma,
                            args.M,
                            args.T,
                            args.reach,
                            args.device,
                        )
                        for gamma in gammas
                    }
                    pack_pair(output, cells, goal_coef, safe_coef, alpha)
                outputs.append(
                    {
                        "alpha": alpha,
                        "goal_coef": goal_coef,
                        "safe_coef": safe_coef,
                        "file": output.name,
                        "sha256": sha256_file(output),
                    }
                )

    manifest = {
        "status": (
            "KAZUKI_ALPHA_COEFFICIENT_GRID_COMPLETE"
            if include_alpha_tag
            else "KAZUKI_ABSOLUTE_COEFFICIENT_GRID_COMPLETE"
        ),
        "version": VERSION,
        "scene": "low7_radius1_canonical_v1",
        "checkpoint_sha256": checkpoint_sha,
        "conditioning_schema": SCHEMA,
        "gammas": list(gammas),
        "goal_coefficients": list(goal_coefficients),
        "safe_coefficients": list(safe_coefficients),
        "alphas": list(alphas),
        "M_per_cell": args.M,
        "T": args.T,
        "reach": args.reach,
        "device": args.device,
        "seed_contract": (
            "named_seed(version, 'kazuki', gamma, rollout_index); alpha and "
            "coefficients excluded for common random numbers"
        ),
        "fixed_native_cost": {
            "refinement_cost": "b1_safemppi",
            "coll_w": 100.0,
            "goal_w": 0.1,
            "beta_mppi": 20.0,
            "mppi_lambda": 0.1,
            "mppi_sigma": 0.2,
            "n_sample": 200,
            "n_elite": 10,
            "n_copy": 200,
        },
        "elapsed_seconds": time.perf_counter() - sweep_started,
        "outputs": outputs,
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(args.outdir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
