#!/usr/bin/env python3
"""Raw temperature-1 screening/confirmation evaluator for the V3 support sweep."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

_HERE = Path(__file__).resolve().parent.parent
_REV = _HERE.parent
_WORK = _REV.parent
for _path in (_WORK, _REV, _HERE, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import afe_m20_eval as M20
import afe_route_metrics as RM
import low7_raw_m50_eval as EV
from afe2_scene_profiles import assert_scene_snapshot, build_scene, get_scene_profile, scene_snapshot


SCREEN_PROFILE = EV.EvaluationProfile(
    name="v3_support_screen_m10",
    m=10,
    checkpoint_stride=1,
    metric_version=EV.EVALUATION_PROFILES[
        EV.V2_SMOKE_EVAL_PROFILE
    ].metric_version,
    caption=EV.EVALUATION_PROFILES[EV.V2_SMOKE_EVAL_PROFILE].caption,
    filename_tag="support_screen_m10",
    summary_status="AFE_RBF_V3_SUPPORT_SCREEN_COMPLETE",
    delivery_status="AFE_RBF_V3_SUPPORT_SCREEN_DELIVERY_COMPLETE",
)
HOLDOUT_PROFILE = EV.EvaluationProfile(
    name="v3_support_holdout_m50",
    m=50,
    checkpoint_stride=1,
    metric_version="afe_rbf_v3_support_raw_holdout_m50_v1",
    caption="disjoint raw temperature-1 M=50/gamma confirmation holdout",
    filename_tag="support_holdout_m50",
    summary_status="AFE_RBF_V3_SUPPORT_HOLDOUT_COMPLETE",
    delivery_status="AFE_RBF_V3_SUPPORT_HOLDOUT_DELIVERY_COMPLETE",
)
SCENE = "low7_radius1_canonical_v1"
COARSE_ROUNDS = (0, *range(5, 101, 5))


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(EV.AFE2._json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(EV.AFE2._json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def holdout_noise_bank(scene_profile: str, policy_dim: int):
    raw = (
        "afe-rbf-v3-support|genuinely-disjoint-M50-holdout|"
        f"{scene_profile}|temperature-1"
    ).encode()
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)
    generator = np.random.default_rng(seed)
    bank = generator.standard_normal(
        (len(EV.GAMMAS), HOLDOUT_PROFILE.m, EV.T, int(policy_dim)), dtype=np.float32
    )
    screen, screen_meta = EV.build_noise_bank(scene_profile, policy_dim, SCREEN_PROFILE)
    if seed == int(screen_meta["seed"]):
        raise RuntimeError("M50 holdout and M10 screening resolved to the same master seed")
    if np.array_equal(bank[:, :SCREEN_PROFILE.m], screen):
        raise RuntimeError("M50 holdout bank contains the M10 screen bank as a prefix")
    return bank, {
        "seed": seed,
        "shape": list(bank.shape),
        "dtype": str(bank.dtype),
        "sha256": hashlib.sha256(bank.tobytes(order="C")).hexdigest(),
        "screen_sha256": screen_meta["sha256"],
        "screen_seed": int(screen_meta["seed"]),
        "master_seed_disjoint_from_screen": True,
        "disjoint_from_screen_prefix": True,
        "indexing": "[gamma, rollout_index, control_time, latent_dimension]",
        "independence": "fixed across the three selected-arm endpoint comparison checkpoints",
    }


def successful_route_coverage(metrics: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
    successful = [row for row in metrics if row["success"]]
    n_u = sum(int(row["route_mode_closest"]) == int(RM.MODE_U) for row in successful)
    n_r = sum(int(row["route_mode_closest"]) == int(RM.MODE_R) for row in successful)
    n_ambiguous = sum(
        int(row["route_mode_closest"]) == int(RM.MODE_AMBIGUOUS)
        for row in successful
    )
    return {
        "n_success_U": int(n_u),
        "n_success_R": int(n_r),
        "n_success_ambiguous": int(n_ambiguous),
        "n_success_total": len(successful),
        "denominator": int(denominator),
        "C": float(2 * min(n_u, n_r) / denominator),
        "definition": "2*min(n_success_U,n_success_R)/M_eval",
    }


def round_score(metric_rows: list[dict[str, Any]], round_i: int) -> dict[str, Any]:
    gamma_rows = [
        row for row in metric_rows
        if row["scope"] == "gamma" and int(row["round"]) == int(round_i)
    ]
    pooled = next(
        row for row in metric_rows
        if row["scope"] == "pooled" and int(row["round"]) == int(round_i)
    )
    if len(gamma_rows) != len(EV.GAMMAS):
        raise RuntimeError(f"round {round_i} lacks seven gamma screening cells")
    return {
        "round": int(round_i),
        "J": float(np.mean([
            row["successful_route_coverage"]["C"] for row in gamma_rows
        ])),
        "SR": float(pooled["binary"]["SR"]["estimate"]),
        "CR": float(pooled["binary"]["CR"]["estimate"]),
        "timeout": float(pooled["binary"]["timeout"]["estimate"]),
        "minimum_clearance": float(pooled["minimum_clearance"]["mean"]),
    }


def selection_key(score: dict[str, Any]):
    return (
        -score["J"], -score["SR"], score["CR"], score["timeout"],
        -score["minimum_clearance"], score["round"],
    )


def select_round(metric_rows: list[dict[str, Any]], rounds) -> tuple[int, list[dict[str, Any]]]:
    ranking = sorted(
        (round_score(metric_rows, value) for value in rounds), key=selection_key
    )
    return int(ranking[0]["round"]), ranking


def pareto_frontier(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for candidate in scores:
        dominated = any(
            other is not candidate
            and other["SR"] >= candidate["SR"]
            and other["J"] >= candidate["J"]
            and (other["SR"] > candidate["SR"] or other["J"] > candidate["J"])
            for other in scores
        )
        if not dominated:
            output.append(candidate)
    return sorted(output, key=lambda row: (row["SR"], row["J"], -row["round"]))


def _route_obstacles(env, profile):
    obstacles = env.obstacles.detach().cpu().numpy()
    mask = np.linalg.norm(obstacles[:, :2] - np.asarray((2.5, 2.5)), axis=1) < 1.0e-6
    if profile.center_replacement_radius is None or int(mask.sum()) != 1:
        raise RuntimeError("support evaluator requires the canonical single giant obstacle")
    return obstacles[mask, :2], obstacles[mask, 2] + float(env.r_robot)


def _annotate_routes(episodes, env, profile):
    centers, radii = _route_obstacles(env, profile)
    for episode in episodes:
        path = np.asarray(episode["path"], dtype=np.float64)
        early_index = min(10, len(path) - 1)
        episode["route_mode_early"] = int(RM.classify_plan_endpoints(
            path[early_index:early_index + 1], start=profile.start, goal=profile.goal
        )[0])
        labels, _ = RM.classify_trajectories_at_closest_approach(
            path[None], start=profile.start, goal=profile.goal,
            obstacle_centers=centers, obstacle_radii=radii,
        )
        episode["route_mode_closest"] = int(labels[0])


def _save_cell(
    outdir: Path,
    contract,
    round_i: int,
    gamma: float,
    episodes,
    metrics,
    model_sha: str,
    bank_meta,
    profile,
):
    pairs = [
        (episode, metric) for episode, metric in zip(episodes, metrics)
        if float(episode["gamma"]) == float(gamma)
    ]
    if len(pairs) != profile.m:
        raise RuntimeError(f"r{round_i}/g{gamma} has {len(pairs)} != M={profile.m}")
    cell = outdir / "cells"
    cell.mkdir(exist_ok=True)
    stem = f"r{round_i:03d}_g{gamma:.1f}"
    archive = cell / f"{stem}.npz"
    provenance = cell / f"{stem}.provenance.json"
    if archive.exists() or provenance.exists():
        raise FileExistsError(f"support raw cell already exists: {stem}")
    path_array = np.empty(profile.m, dtype=object)
    for index, (episode, _) in enumerate(pairs):
        if int(episode["rollout_index"]) != index:
            raise RuntimeError("support raw cell lost fixed rollout index ordering")
        path_array[index] = episode["path"]
    np.savez_compressed(
        archive,
        paths=path_array,
        rollout_index=np.arange(profile.m, dtype=np.int32),
        status=np.asarray([metric["status"] for _, metric in pairs]),
        outcome=np.asarray([metric["outcome"] for _, metric in pairs]),
        success=np.asarray([metric["success"] for _, metric in pairs], np.bool_),
        cr=np.asarray([metric["cr"] for _, metric in pairs], np.bool_),
        timeout=np.asarray([metric["timeout"] for _, metric in pairs], np.bool_),
        v_safe=np.asarray([metric["v_safe"] for _, metric in pairs], np.bool_),
        v_full=np.asarray([metric["v_full"] for _, metric in pairs], np.bool_),
        minimum_clearance=np.asarray([
            metric["minimum_clearance"] for _, metric in pairs
        ], np.float64),
        route_mode_closest=np.asarray([
            metric["route_mode_closest"] for _, metric in pairs
        ], np.int8),
    )
    write_json(provenance, {
        "mode": "raw temperature-1; no tilt, verifier filtering, controller, or fallback",
        "round": round_i,
        "gamma": gamma,
        "M": profile.m,
        "checkpoint": contract["selected_checkpoints"][round_i],
        "checkpoint_model_state_sha256": model_sha,
        "noise_bank": bank_meta,
        "trainer_complete_sha256": contract["complete_sha256"],
        "scene_sha256": contract["scene_sha256"],
        "archive": str(archive),
        "archive_sha256": EV.sha256_file(archive),
    })
    rows = [metric for _, metric in pairs]
    aggregate = EV.aggregate_metrics(
        rows, round_i=round_i, gamma=gamma, scope="gamma",
        scene_profile=SCENE, algorithm=contract["algorithm"], eval_profile=profile,
    )
    aggregate["successful_route_coverage"] = successful_route_coverage(rows, profile.m)
    return aggregate


def evaluate_round(
    contract, round_i, env, profile_scene, cfg, device, bank, bank_meta,
    profile, executor, outdir,
):
    started = time.perf_counter()
    policy, _, model_sha, conditioning = EV._load_policy(contract, round_i, device)
    if conditioning.schema != cfg.conditioning_schema:
        raise RuntimeError("checkpoint conditioning changed during support evaluation")
    episodes = EV.run_raw_batch(policy, env, cfg, device, bank, profile)
    _annotate_routes(episodes, env, profile_scene)
    tasks = [
        (episode["path"], episode["gamma"], episode["status"], float(env.dt), EV.REACH)
        for episode in episodes
    ]
    worker_rows = list(executor.map(M20._trajectory_metrics_worker, tasks, chunksize=2))
    metrics = [
        EV.normalize_trajectory_metrics(episode, worker, float(env.dt))
        for episode, worker in zip(episodes, worker_rows)
    ]
    rows = []
    for gamma in EV.GAMMAS:
        rows.append(_save_cell(
            outdir, contract, round_i, gamma, episodes, metrics, model_sha,
            bank_meta, profile,
        ))
    pooled = EV.aggregate_metrics(
        metrics, round_i=round_i, gamma=None, scope="pooled",
        scene_profile=SCENE, algorithm=contract["algorithm"], eval_profile=profile,
    )
    pooled["successful_route_coverage"] = successful_route_coverage(
        metrics, profile.m * len(EV.GAMMAS)
    )
    rows.append(pooled)
    del policy
    torch.cuda.empty_cache()
    return rows, time.perf_counter() - started


def _reference_r0_check(outdir: Path, metric_rows, reference: Path) -> dict[str, Any]:
    reference_rows = [
        json.loads(line) for line in (reference / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    actual = {
        (row["scope"], row["gamma"]): row for row in metric_rows if row["round"] == 0
    }
    expected = {
        (row["scope"], row["gamma"]): row for row in reference_rows if row["round"] == 0
    }
    if set(actual) != set(expected):
        raise RuntimeError("r0 reference metric key mismatch")
    fields = ("binary", "minimum_clearance", "successful_time_to_goal", "route_modes")
    for key in actual:
        for field in fields:
            if actual[key][field] != expected[key][field]:
                raise RuntimeError(f"r0 metric mismatch at {key}/{field}")
    for gamma in EV.GAMMAS:
        ours = np.load(outdir / "cells" / f"r000_g{gamma:.1f}.npz", allow_pickle=True)
        theirs = np.load(
            reference / "cells" / "raw" / "afe_rbf" / f"r000_g{gamma:.1f}.npz",
            allow_pickle=True,
        )
        for field in ("status", "outcome", "success", "cr", "timeout", "v_safe", "v_full"):
            if not np.array_equal(ours[field], theirs[field]):
                raise RuntimeError(f"r0 status/count archive mismatch g={gamma}/{field}")
        for left, right in zip(ours["paths"], theirs["paths"]):
            if not np.array_equal(left, right):
                raise RuntimeError(f"r0 path mismatch g={gamma}")
    return {
        "status": "EXACT_R0_MATCH",
        "reference": str(reference),
        "reference_complete_sha256": EV.sha256_file(reference / "EVALUATION_COMPLETE.json"),
        "metric_fields": list(fields),
        "paths_and_statuses_exact": True,
    }


def _render_screen_curves(outdir: Path, rows, scores, best_round):
    rounds = [score["round"] for score in sorted(scores, key=lambda row: row["round"])]
    score_map = {score["round"]: score for score in scores}
    pooled = {
        int(row["round"]): row for row in rows if row["scope"] == "pooled"
    }
    specs = (("SR", "SR"), ("CR", "CR"), ("timeout", "timeout"),
             ("V_safe", "V_safe"), ("V_full", "V_full"))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    for axis, (key, title) in zip(axes.flat, specs):
        axis.plot(rounds, [pooled[r]["binary"][key]["estimate"] for r in rounds], "o-")
        axis.axvline(best_round, color="#0072b2", ls="--")
        axis.set(title=title, xlabel="round", ylim=(-0.03, 1.03))
        axis.grid(alpha=0.25)
    axis = axes.flat[-1]
    axis.plot(rounds, [score_map[r]["J"] for r in rounds], "o-", color="#cc79a7")
    axis.axvline(best_round, color="#0072b2", ls="--")
    axis.set(title="successful balanced-route coverage J", xlabel="round", ylim=(-0.03, 1.03))
    axis.grid(alpha=0.25)
    fig.suptitle("Raw temperature-1 M=10/gamma screening (tilted gathering is not evaluation)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = outdir / "screening_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _draw_scene(axis, env):
    obstacles = env.obstacles.detach().cpu().numpy()
    for x, y, radius in obstacles:
        axis.add_patch(plt.Circle((x, y), radius, color="0.78", zorder=0))
    axis.plot(float(env.x0[0]), float(env.x0[1]), "ks", ms=4)
    axis.plot(float(env.goal[0]), float(env.goal[1]), marker="*", ms=10,
              color="#ffd400", mec="black")
    axis.set(xlim=(-0.05, 5.05), ylim=(-0.05, 5.05), aspect="equal")
    axis.set_xticks([]); axis.set_yticks([])


def _render_holdout_gallery(outdir: Path, env, rounds, selected_round):
    display_rounds = [0, int(selected_round), 100]
    fig, axes = plt.subplots(len(EV.GAMMAS), len(display_rounds), figsize=(4 * len(display_rounds), 3.5 * len(EV.GAMMAS)))
    cmap = plt.get_cmap("plasma")
    colors = {gamma: cmap(0.08 + 0.84 * i / 6) for i, gamma in enumerate(EV.GAMMAS)}
    for row_index, gamma in enumerate(EV.GAMMAS):
        for col_index, round_i in enumerate(display_rounds):
            axis = axes[row_index, col_index]
            _draw_scene(axis, env)
            archive = np.load(
                outdir / "cells" / f"r{round_i:03d}_g{gamma:.1f}.npz", allow_pickle=True
            )
            for index in range(10):
                path = np.asarray(archive["paths"][index])
                axis.plot(path[:, 0], path[:, 1], color=colors[gamma], lw=0.9, alpha=0.66)
                dots = path[::4]
                axis.scatter(dots[:, 0], dots[:, 1], s=3.5, color=colors[gamma], alpha=0.8)
            if row_index == 0:
                label = ("pretrained" if col_index == 0 else
                         "selected best" if col_index == 1 else "final")
                axis.set_title(f"{label} r{round_i}")
            if col_index == 0:
                axis.set_ylabel(rf"$\gamma={gamma}$")
    fig.suptitle("Fixed indices 0–9 | raw temperature-1 | small dots are executed states")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"selected_raw_m50_gallery.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _render_holdout_report(outdir: Path, rows, rounds, selected_round):
    lookup = {(row["round"], row["gamma"]): row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(rounds)); width = 0.22
    for offset, key, color in ((-width, "SR", "#009e73"), (0, "CR", "#d55e00"), (width, "timeout", "#777777")):
        axes[0, 0].bar(x + offset, [lookup[(r, None)]["binary"][key]["estimate"] for r in rounds], width, label=key, color=color)
    axes[0, 0].set_xticks(x, [f"r{r}" for r in rounds]); axes[0, 0].set_ylim(0, 1); axes[0, 0].legend(); axes[0, 0].grid(axis="y", alpha=.25)
    gammas = list(EV.GAMMAS); gx = np.arange(len(gammas)); selected = [lookup[(selected_round, g)] for g in gammas]
    axes[0, 1].plot(gx, [r["binary"]["SR"]["estimate"] for r in selected], "o-", label="SR")
    axes[0, 1].plot(gx, [r["successful_route_coverage"]["C"] for r in selected], "s-", label="C_gamma")
    axes[0, 1].set_xticks(gx, [str(g) for g in gammas]); axes[0, 1].set_ylim(0, 1); axes[0, 1].set_xlabel("gamma"); axes[0, 1].legend(); axes[0, 1].grid(alpha=.25)
    u = [r["successful_route_coverage"]["n_success_U"] for r in selected]
    rr = [r["successful_route_coverage"]["n_success_R"] for r in selected]
    axes[1, 0].bar(gx, u, label="successful U"); axes[1, 0].bar(gx, rr, bottom=u, label="successful R")
    axes[1, 0].set_xticks(gx, [str(g) for g in gammas]); axes[1, 0].set_xlabel("gamma"); axes[1, 0].set_ylabel("count / 50"); axes[1, 0].legend(); axes[1, 0].grid(axis="y", alpha=.25)
    pooled = lookup[(selected_round, None)]
    text = [
        "Headline source: disjoint raw temperature-1 M=50/gamma holdout",
        "No uncertainty tilt, verifier-selected execution, or expert fallback",
        f"Selected checkpoint fixed before M50: r{selected_round}",
        f"Pooled SR={pooled['binary']['SR']['estimate']:.3f}",
        f"CR={pooled['binary']['CR']['estimate']:.3f}, timeout={pooled['binary']['timeout']['estimate']:.3f}",
        f"V_safe={pooled['binary']['V_safe']['estimate']:.3f}, V_full={pooled['binary']['V_full']['estimate']:.3f}",
        f"mean minimum clearance={pooled['minimum_clearance']['mean']:.3f} m",
    ]
    axes[1, 1].axis("off"); axes[1, 1].text(0, 1, "\n".join(text), va="top", fontsize=11)
    fig.suptitle("Low7 giant-obstacle AFE support sweep — M50 confirmation")
    fig.tight_layout(rect=(0, 0, 1, .96))
    outputs = []
    for suffix in ("png", "pdf"):
        path = outdir / f"report.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _inventory(outdir: Path):
    return {
        str(path.relative_to(outdir)): EV.sha256_file(path)
        for path in sorted(outdir.rglob("*"))
        if path.is_file() and path.name != "EVALUATION_COMPLETE.json"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mode", choices=("screen", "holdout"), required=True)
    parser.add_argument("--selected-round", type=int)
    parser.add_argument("--reference-r0-evaluation", type=Path)
    parser.add_argument("--verifier-workers", type=int, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"support evaluation output must be absent: {args.outdir}")
    profile = SCREEN_PROFILE if args.mode == "screen" else HOLDOUT_PROFILE
    contract = EV.validate_completed_run(
        args.run_root, SCENE, EV.V2_SMOKE_EVAL_PROFILE
    )
    if contract["algorithm"] != "afe_rbf_low7_v3_optimizer_demo_support_v1":
        raise RuntimeError("support evaluator received the wrong trainer algorithm")
    source = EV.require_clean_additive_source(contract["source_git_commit"])
    gpu = EV._gpu_record()
    policy0, _, r0_sha, conditioning = EV._load_policy(contract, 0, "cpu")
    if r0_sha != contract["source_checkpoint_model_sha256"]:
        raise RuntimeError("support r0 checkpoint differs from authenticated pretraining")
    if args.mode == "screen":
        bank, bank_meta = EV.build_noise_bank(SCENE, int(policy0.d), SCREEN_PROFILE)
        rounds = list(COARSE_ROUNDS)
    else:
        if args.selected_round is None or not 0 <= args.selected_round <= 100:
            raise ValueError("holdout requires a fixed selected round in [0,100]")
        bank, bank_meta = holdout_noise_bank(SCENE, int(policy0.d))
        rounds = sorted({0, int(args.selected_round), 100})
    del policy0
    scene_profile = get_scene_profile(SCENE)
    env = build_scene(scene_profile)
    snapshot = scene_snapshot(env, scene_profile)
    assert_scene_snapshot(snapshot)
    if snapshot["sha256"] != contract["scene_sha256"]:
        raise RuntimeError("support evaluation scene hash mismatch")
    cfg = EV.RawEvalConfig(scene_profile=SCENE, conditioning_schema=conditioning.schema)
    args.outdir.mkdir(parents=True)
    write_json(args.outdir / "evaluation_contract.json", {
        "mode": args.mode,
        "raw_policy": "temperature=1, no tilt/filter/controller/fallback",
        "profile": profile.__dict__,
        "run": contract,
        "source": source,
        "gpu": gpu,
        "scene": snapshot,
        "noise_bank": bank_meta,
        "verifier_workers": args.verifier_workers,
        "selected_round_fixed_before_holdout": args.selected_round if args.mode == "holdout" else None,
    })
    rows: list[dict[str, Any]] = []
    timings = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.verifier_workers,
        mp_context=context,
        initializer=M20._worker_init,
        initargs=(SCENE, EV.REACH, 180),
    ) as executor:
        for round_i in rounds:
            new_rows, elapsed = evaluate_round(
                contract, round_i, env, scene_profile, cfg, "cuda:0", bank,
                bank_meta, profile, executor, args.outdir,
            )
            rows.extend(new_rows)
            timings.append({"round": round_i, "elapsed_seconds": elapsed})
            print(f"[support {args.mode}] r{round_i:03d} complete in {elapsed:.1f}s", flush=True)
        if args.mode == "screen":
            coarse_best, coarse_ranking = select_round(rows, rounds)
            local = [
                value for value in range(max(0, coarse_best - 2), min(100, coarse_best + 2) + 1)
                if value not in rounds
            ]
            for round_i in local:
                new_rows, elapsed = evaluate_round(
                    contract, round_i, env, scene_profile, cfg, "cuda:0", bank,
                    bank_meta, profile, executor, args.outdir,
                )
                rows.extend(new_rows)
                timings.append({"round": round_i, "elapsed_seconds": elapsed})
                print(f"[support screen local] r{round_i:03d} complete in {elapsed:.1f}s", flush=True)
            rounds = sorted((*rounds, *local))

    if args.mode == "screen":
        best_round, ranking = select_round(rows, rounds)
        scores = [round_score(rows, value) for value in rounds]
        reference = _reference_r0_check(
            args.outdir, rows, args.reference_r0_evaluation.resolve()
        ) if args.reference_r0_evaluation else None
        write_json(args.outdir / "r0_reference_check.json", reference)
        curves = [_render_screen_curves(args.outdir, rows, scores, best_round)]
        selection = {
            "rule": "maximize mean_gamma C_gamma; then SR, CR, timeout, clearance, earlier round",
            "coarse_rounds": list(COARSE_ROUNDS),
            "coarse_best_round": coarse_best,
            "coarse_ranking": coarse_ranking,
            "local_rounds": local,
            "best_round": best_round,
            "ranking": ranking,
            "pareto_SR_J": pareto_frontier(scores),
        }
        write_json(args.outdir / "selection.json", selection)
        presentation = curves
    else:
        selection = {
            "selected_round": int(args.selected_round),
            "selection_source": "fixed Stage A result; M50 did not select or change checkpoint",
            "rounds": rounds,
        }
        write_json(args.outdir / "selection.json", selection)
        presentation = [
            *_render_holdout_report(args.outdir, rows, rounds, int(args.selected_round)),
            *_render_holdout_gallery(args.outdir, env, rounds, int(args.selected_round)),
        ]
    metrics_path = args.outdir / "metrics.jsonl"
    write_metrics(metrics_path, sorted(rows, key=lambda row: (row["round"], row["scope"] != "gamma", -1 if row["gamma"] is None else row["gamma"])))
    with (args.outdir / "round_summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("round", "J", "SR", "CR", "timeout", "minimum_clearance"))
        writer.writeheader()
        for round_i in rounds:
            writer.writerow(round_score(rows, round_i))
    summary = {
        "status": profile.summary_status,
        "mode": args.mode,
        "rounds": rounds,
        "M_per_gamma": profile.m,
        "timings": timings,
        "selection": selection,
        "metrics": str(metrics_path),
        "presentation": [str(path) for path in presentation],
    }
    write_json(args.outdir / "evaluation_summary.json", summary)
    inventory = _inventory(args.outdir)
    write_json(args.outdir / "EVALUATION_COMPLETE.json", {
        "status": profile.delivery_status,
        "mode": args.mode,
        "trainer_source_commit": contract["source_git_commit"],
        "evaluation_source_commit": source["commit"],
        "scene_sha256": snapshot["sha256"],
        "artifact_sha256": inventory,
    })


if __name__ == "__main__":
    main()
