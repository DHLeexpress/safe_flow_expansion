#!/usr/bin/env python3
"""Build the five indexed B1/SFM method videos requested for the paper.

Only the top title strip contains text.  The expansion video is a diagnostic
replay: it loads the actual r0--r15 checkpoints and logged betas, advances one
synchronous episode per displayed gamma with the unchanged K=16/B=4 verifier
loop, and reconstructs the W=2 RBF memory from the replay's own positives.
It is not presented as the compact run's missing historical r15 trace.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_PAPER = (
    _REPO / "source_snapshot" / "overnight_run_07_06" / "rev_expansion"
    / "codex_overnight" / "paper_results"
)
_CORE = _PAPER.parent
_REV = _CORE.parent
_WORK = _REV.parent
for _path in (_WORK, _REV, _CORE, _PAPER):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch

import afe_core as AC
import afe_rbf_core as RC
import grid_expand_afe2 as AFE2
import grid_expand_afe_rbf as RBF
import grid_hp_expt as HP
import grid_rollout as GR
from afe2_scene_profiles import build_scene, get_scene_profile, scene_snapshot
from b1_indexed_video_suite import (
    _draw_scene,
    _draw_polygons,
    _serialize_faces,
    nominal_level_polygons,
    run_safemppi_trace,
    verifier_level_polygons,
)
import verifier_polytope as VP


GAMMAS = (0.1, 0.5, 1.0)
DISPLAY_ROUNDS = (0, 5, 10, 15)
COLORS = {
    gamma: plt.get_cmap("plasma")({0.1: 0.08, 0.5: 0.52, 1.0: 0.92}[gamma])
    for gamma in GAMMAS
}
GREEN = "#148f48"
BLUE = "#1764ab"
RED = "#cc3311"
VIDEO_VERSION = "b1_shared_indexed_videos_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def named_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def encode_video(frame_root: Path, output: Path, fps: int) -> dict[str, Any]:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frame_root / "frame_%06d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", str(fps),
            "-pix_fmt", "yuv420p", "-c:v", "libx264", str(output),
        ],
        check=True,
    )
    probe = json.loads(subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames",
            "-of", "json", str(output),
        ],
        text=True,
    ))
    return {"sha256": sha256_file(output), "ffprobe": probe}


def set_title(figure: Any, frame_index: int, detail: str = "") -> None:
    suffix = "" if not detail else rf"\quad {detail}"
    figure.suptitle(
        rf"$\mathrm{{Frame}}\ {frame_index:04d}{suffix}$",
        fontsize=27,
        y=0.992,
    )


def safe_mppi_episode(profile: str, gamma: float, index: int) -> dict[str, Any]:
    env = build_scene(get_scene_profile(profile))
    return run_safemppi_trace(env, gamma, index, collect_trace=True)


def _safe_mppi_outcome(task: tuple[str, float, int]) -> tuple[int, str]:
    profile, gamma, index = task
    torch.set_num_threads(1)
    env = build_scene(get_scene_profile(profile))
    episode = run_safemppi_trace(env, gamma, index, collect_trace=False)
    return index, str(episode["outcome"])


def first_safemppi_failure(
    profile: str,
    gamma: float,
    bank_size: int,
    workers: int,
) -> int:
    context = mp.get_context("spawn")
    tasks = ((profile, gamma, index) for index in range(bank_size))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for index, outcome in executor.map(_safe_mppi_outcome, tasks, chunksize=1):
            if outcome != "SR":
                executor.shutdown(wait=True, cancel_futures=True)
                return int(index)
    raise RuntimeError(f"no SafeMPPI failure for gamma={gamma:g} in [0,{bank_size})")


def render_safemppi_video(
    outdir: Path,
    key: str,
    profile_name: str,
    episodes: dict[float, dict[str, Any]],
    fps: int,
) -> dict[str, Any]:
    env = build_scene(get_scene_profile(profile_name))
    frame_root = outdir / "frames" / key
    frame_root.mkdir(parents=True)
    max_step = max(len(episode["trace"]) for episode in episodes.values()) - 1
    shown = list(range(0, max_step + 1, 2))
    if shown[-1] != max_step:
        shown.append(max_step)
    for frame_index, step in enumerate(shown):
        figure, axis = plt.subplots(figsize=(9.2, 9.2))
        _draw_scene(axis, env)
        for gamma, episode in episodes.items():
            color = COLORS[gamma]
            path = np.asarray(episode["path"], dtype=float)
            prefix_end = min(step + 2, len(path))
            axis.plot(path[:prefix_end, 0], path[:prefix_end, 1], color=color, lw=3.0, zorder=7)
            axis.plot(
                path[:prefix_end:4, 0], path[:prefix_end:4, 1], linestyle="none",
                marker=".", color=color, markersize=4, zorder=8,
            )
            trace_step = min(step, len(episode["trace"]) - 1)
            row = episode["trace"][trace_step]
            plan = np.asarray(row["plan"], dtype=float)
            axis.plot(plan[:, 0], plan[:, 1], ls="--", lw=1.5, color=color, alpha=0.9, zorder=6)
            _draw_polygons(
                axis,
                nominal_level_polygons(row["nominal"], gamma),
                color,
            )
            if step >= len(episode["trace"]) - 1 and episode["outcome"] != "SR":
                axis.plot(*path[-1], marker="x", color=RED, markersize=13, markeredgewidth=3, zorder=12)
        set_title(figure, frame_index)
        figure.tight_layout(rect=(0, 0, 1, 0.955))
        figure.savefig(frame_root / f"frame_{frame_index:06d}.png", dpi=165)
        plt.close(figure)
    video = outdir / f"{key}.mp4"
    record = encode_video(frame_root, video, fps)
    preview = outdir / f"{key}_preview.png"
    shutil.copyfile(frame_root / f"frame_{len(shown)//2:06d}.png", preview)
    return {
        "video": str(video),
        "preview": str(preview),
        "frames": len(shown),
        "shown_steps": shown,
        **record,
    }


def probe_betas(path: Path) -> dict[int, float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    betas = {int(row["round"]): float(row["beta_used"]) for row in rows}
    missing = [round_i for round_i in range(16) if round_i not in betas]
    if missing:
        raise RuntimeError(f"probe lacks beta_used at rounds {missing}")
    return betas


def replay_config(recipe: dict[str, Any], beta: float) -> RBF.AFERBFConfig:
    return RBF.AFERBFConfig(
        protocol_profile=str(recipe.get("protocol_profile", "b1_balanced_v2")),
        rounds=15,
        T=300,
        K=16,
        B=4,
        beta=float(beta),
        s=float(recipe.get("s", 0.9)),
        nfe=int(recipe.get("nfe", 8)),
        temp=float(recipe.get("temp", 1.0)),
        n_theta=int(recipe.get("n_theta", 180)),
        gammas=GAMMAS,
        arm="afe",
        batch=128,
        afe_lr=1.0e-5,
        afe_steps=0,
        M_eval=1,
        wall_plugs=8,
        start_eps=0.3,
        goal_xy=(4.7, 4.7),
        scene_profile="low7_radius1_canonical_v1",
        conditioning_schema="low7_closest_boundary_tie_mean",
        raw_condition_dim=7,
        freeze_visual_encoder=True,
        seed=int(recipe.get("seed", 910)),
        replicas=1,
        gp_cap=768,
        gp_lam=1.0e-2,
        verifier_workers=16,
        acquisition_mode="sequential",
        adaptive_ess_target=0.5,
        gp_replay_window=2,
        gp_replay_sampling="round_gamma",
        negative_alpha=0.01,
        execution_rule="nominal_hp_max_step_margin",
        training_probes=False,
        nvp_audit_all_k=False,
    )


def round_gp(
    policy: Any,
    store: AC.DStore,
    cfg: RBF.AFERBFConfig,
    round_i: int,
    lengthscale: float,
    device: str,
) -> tuple[RC.RBFGPSigma, list[int]]:
    gp = RC.RBFGPSigma(lengthscale, cfg.gp_lam)
    if round_i == 0 or not store.q_y:
        return gp, []
    query_ids = RC.recent_round_positive_ids_hierarchical(
        store,
        round_i - 1,
        cfg.gp_replay_window,
        cfg.gp_cap,
        AFE2.named_seed(cfg.seed, "paper_diagnostic_gp", round_i),
    )
    if query_ids:
        features = AFE2.embed_queries(policy, store, cfg, device, ids=query_ids)
        gp.set_buffer(features.to(device))
    return gp, [int(value) for value in query_ids]


def status_map(episodes: list[dict[str, Any]]) -> dict[float, str]:
    return {float(episode["gamma"]): str(episode["status"]) for episode in episodes}


def replay_round(
    policy: Any,
    gp: RC.RBFGPSigma,
    env: Any,
    cfg: RBF.AFERBFConfig,
    store: AC.DStore,
    round_i: int,
    device: str,
    executor: ProcessPoolExecutor,
    purpose: str,
    collect: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    viz: list[dict[str, Any]] = []
    episodes, _ = RBF.run_parallel_episodes(
        policy, gp, env, cfg, store, round_i, 1, device, executor,
        collect=collect, viz=viz, purpose=purpose,
        acquisition_mode="sequential",
    )
    return episodes, viz


def choose_replay_purpose(
    ckpt0: Path,
    beta0: float,
    env: Any,
    recipe: dict[str, Any],
    lengthscale: float,
    device: str,
    executor: ProcessPoolExecutor,
) -> tuple[str, dict[float, str]]:
    policy, _ = HP.load_hp(str(ckpt0), device="cpu")
    policy = policy.to(device).eval()
    cfg = replay_config(recipe, beta0)
    gp = RC.RBFGPSigma(lengthscale, cfg.gp_lam)
    fallback = None
    for index in range(16):
        purpose = f"paper_expansion_replay_{index:02d}"
        empty = AC.DStore(
            conditioning_schema=cfg.conditioning_schema,
            condition_dim=cfg.raw_condition_dim,
        )
        episodes, _ = replay_round(
            policy, gp, env, cfg, empty, 0, device, executor, purpose, False
        )
        statuses = status_map(episodes)
        fallback = (purpose, statuses)
        if statuses[0.1] == "reached" and (
            statuses[0.5] == "nvp" or statuses[1.0] == "nvp"
        ):
            return purpose, statuses
    assert fallback is not None
    return fallback


def verifier_faces_for_viz(row: dict[str, Any], env: Any) -> list[dict[str, Any]]:
    selected = int(row["sel"])
    if selected < 0:
        return []
    plan = np.vstack((np.asarray(row["state"], dtype=float)[:2], np.asarray(row["segsK"])[selected]))
    ok, faces, _, _ = VP.certify_window(
        plan,
        env.obstacles.detach().cpu().numpy(),
        float(env.r_robot),
        float(row["gamma"]),
        R=2.5,
        n_theta=180,
    )
    return _serialize_faces(faces) if ok else []


def generate_expansion_replay(
    run_dir: Path,
    probe: Path,
    device: str,
    verifier_workers: int,
) -> dict[str, Any]:
    betas = probe_betas(probe)
    payload0 = torch.load(run_dir / "ckpt_0.pt", map_location="cpu", weights_only=False)
    recipe = dict(payload0["recipe"])
    lengthscale = float(recipe.get("lengthscale", recipe.get("rbf_lengthscale", 0.20032394292220754)))
    env = build_scene(get_scene_profile("low7_radius1_canonical_v1"))
    context = mp.get_context("spawn")
    store = AC.DStore(
        conditioning_schema="low7_closest_boundary_tie_mean",
        condition_dim=7,
    )
    retained = {}
    with ProcessPoolExecutor(
        max_workers=verifier_workers,
        mp_context=context,
        initializer=RC.initialize_verifier_worker,
        initargs=("low7_radius1_canonical_v1", 0.15, 180),
    ) as executor:
        purpose, r0_search_status = choose_replay_purpose(
            run_dir / "ckpt_0.pt", betas[0], env, recipe, lengthscale,
            device, executor,
        )
        for round_i in range(16):
            policy, _ = HP.load_hp(str(run_dir / f"ckpt_{round_i}.pt"), device="cpu")
            policy = policy.to(device).eval()
            cfg = replay_config(recipe, betas[round_i])
            gp, query_ids = round_gp(policy, store, cfg, round_i, lengthscale, device)
            episodes, viz = replay_round(
                policy, gp, env, cfg, store, round_i, device, executor,
                purpose, True,
            )
            if round_i in DISPLAY_ROUNDS:
                for row in viz:
                    row["verifier_faces"] = verifier_faces_for_viz(row, env)
                retained[round_i] = {
                    "episodes": episodes,
                    "viz": viz,
                    "beta": betas[round_i],
                    "gp_size": gp.n,
                    "gp_query_ids": query_ids,
                    "status": status_map(episodes),
                }

        # Search a separate, declared r15 diagnostic seed for three successes.
        policy15, _ = HP.load_hp(str(run_dir / "ckpt_15.pt"), device="cpu")
        policy15 = policy15.to(device).eval()
        cfg15 = replay_config(recipe, betas[15])
        gp15, query_ids15 = round_gp(policy15, store, cfg15, 15, lengthscale, device)
        success = None
        for index in range(64):
            success_purpose = f"paper_r15_success_{index:03d}"
            episodes, viz = replay_round(
                policy15, gp15, env, cfg15, store, 15, device, executor,
                success_purpose, False,
            )
            if all(episode["status"] == "reached" for episode in episodes):
                for row in viz:
                    row["verifier_faces"] = verifier_faces_for_viz(row, env)
                success = {
                    "purpose": success_purpose,
                    "episodes": episodes,
                    "viz": viz,
                    "beta": betas[15],
                    "gp_size": gp15.n,
                    "gp_query_ids": query_ids15,
                    "status": status_map(episodes),
                }
                break
        if success is None:
            raise RuntimeError("no all-success r15 triple in declared 64-seed bank")
    return {
        "retained": retained,
        "success_r15": success,
        "purpose": purpose,
        "base_seed": int(recipe.get("seed", 910)),
        "r0_search_status": r0_search_status,
        "lengthscale": lengthscale,
        "betas": {str(key): value for key, value in betas.items()},
        "diagnostic_replay_only": True,
    }


def viz_by_episode(viz: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for row in viz:
        output[int(row["episode"])].append(row)
    for rows in output.values():
        rows.sort(key=lambda row: int(row["t"]))
    return output


def render_verified_panels(
    outdir: Path,
    key: str,
    env: Any,
    episodes: list[dict[str, Any]],
    viz: list[dict[str, Any]],
    fps: int,
    *,
    uncertainty: bool,
    round_i: int | None,
    sigma_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    grouped = viz_by_episode(viz)
    max_step = max(len(rows) for rows in grouped.values()) - 1
    shown = list(range(0, max_step + 1, 2))
    if shown[-1] != max_step:
        shown.append(max_step)
    all_sigma = [float(value) for row in viz for value in row.get("sigma_B", [])]
    if sigma_limits is None:
        sigma_limits = (
            min(all_sigma) if all_sigma else 0.0,
            max(all_sigma) if all_sigma and max(all_sigma) > min(all_sigma) else 1.0,
        )
    norm = Normalize(vmin=sigma_limits[0], vmax=sigma_limits[1])
    cmap = plt.get_cmap("viridis")
    frame_root = outdir / "frames" / key
    frame_root.mkdir(parents=True)
    accumulated_positive = {episode: [] for episode in grouped}
    accumulated_nvp = {episode: [] for episode in grouped}
    for frame_index, step in enumerate(shown):
        figure, axes = plt.subplots(1, 3, figsize=(18.2, 6.5), squeeze=False)
        for episode_id, gamma in enumerate(GAMMAS):
            axis = axes[0, episode_id]
            _draw_scene(axis, env)
            rows = grouped[episode_id]
            row_index = min(step, len(rows) - 1)
            for source in rows[: row_index + 1]:
                state = np.asarray(source["state"], dtype=float)[:2]
                if any(int(value) == 1 for value in source["y"]):
                    accumulated_positive[episode_id].append(state)
                if int(source["sel"]) < 0:
                    accumulated_nvp[episode_id].append(state)
            if accumulated_positive[episode_id]:
                points = np.unique(np.asarray(accumulated_positive[episode_id]), axis=0)
                axis.plot(points[:, 0], points[:, 1], "o", color=BLUE, markersize=3.5, zorder=8)
            if accumulated_nvp[episode_id]:
                points = np.unique(np.asarray(accumulated_nvp[episode_id]), axis=0)
                axis.plot(points[:, 0], points[:, 1], "o", color=RED, markersize=6, zorder=9)
            path = np.asarray(episodes[episode_id]["path"], dtype=float)
            prefix_end = min(row_index + 2, len(path))
            axis.plot(path[:prefix_end, 0], path[:prefix_end, 1], color=COLORS[gamma], lw=3.0, zorder=7)
            row = rows[row_index]
            drawn = [int(value) for value in row["drawn"]]
            sigma_b = row.get("sigma_B", [0.0] * len(drawn))
            for candidate_id, sigma in zip(drawn, sigma_b):
                plan = np.vstack((np.asarray(row["state"])[:2], np.asarray(row["segsK"])[candidate_id]))
                color = cmap(norm(float(sigma))) if uncertainty else "0.2"
                axis.plot(plan[:, 0], plan[:, 1], ls="--", lw=1.55, color=color, zorder=6)
            if row.get("verifier_faces"):
                _draw_polygons(
                    axis,
                    verifier_level_polygons(row["verifier_faces"], row["state"][:2], gamma),
                    GREEN,
                )
            axis.set_title(rf"$\gamma={gamma:g}$", fontsize=23, pad=6)
        detail = "" if round_i is None else rf"\mathrm{{round}}\ {round_i}"
        set_title(figure, frame_index, detail)
        if uncertainty:
            colorbar = figure.colorbar(
                ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[0]),
                fraction=0.025, pad=0.018,
            )
            colorbar.set_label(r"$\sigma(\phi_s)$", fontsize=23)
            colorbar.ax.tick_params(labelsize=15)
        figure.tight_layout(rect=(0, 0, 0.97 if uncertainty else 1, 0.94))
        figure.savefig(frame_root / f"frame_{frame_index:06d}.png", dpi=155)
        plt.close(figure)
    # Pause on the terminal frame without adding scientific content.
    terminal = frame_root / f"frame_{len(shown)-1:06d}.png"
    for extra in range(1, fps * 2 + 1):
        shutil.copyfile(terminal, frame_root / f"frame_{len(shown)-1+extra:06d}.png")
    video = outdir / f"{key}.mp4"
    record = encode_video(frame_root, video, fps)
    preview = outdir / f"{key}_preview.png"
    shutil.copyfile(terminal, preview)
    return {
        "video": str(video), "preview": str(preview),
        "scientific_frames": len(shown), "terminal_pause_frames": fps * 2,
        "shown_steps": shown, **record,
    }


def render_expansion_video(
    outdir: Path,
    key: str,
    replay: dict[str, Any],
    env: Any,
    fps: int,
) -> dict[str, Any]:
    segment_dirs = []
    segments = []
    all_sigma = [
        float(value)
        for round_i in DISPLAY_ROUNDS
        for row in replay["retained"][round_i]["viz"]
        for value in row.get("sigma_B", [])
    ]
    sigma_limits = (
        min(all_sigma) if all_sigma else 0.0,
        max(all_sigma) if all_sigma and max(all_sigma) > min(all_sigma) else 1.0,
    )
    for round_i in DISPLAY_ROUNDS:
        record = replay["retained"][round_i]
        segment_key = f"_{key}_r{round_i:02d}"
        result = render_verified_panels(
            outdir, segment_key, env, record["episodes"], record["viz"], fps,
            uncertainty=True, round_i=round_i, sigma_limits=sigma_limits,
        )
        segment_dirs.append(outdir / "frames" / segment_key)
        segments.append(result)
    target_root = outdir / "frames" / key
    target_root.mkdir(parents=True)
    counter = 0
    for source_root in segment_dirs:
        for source in sorted(source_root.glob("frame_*.png")):
            shutil.copyfile(source, target_root / f"frame_{counter:06d}.png")
            counter += 1
    video = outdir / f"{key}.mp4"
    record = encode_video(target_root, video, fps)
    preview = outdir / f"{key}_preview.png"
    shutil.copyfile(target_root / f"frame_{counter//2:06d}.png", preview)
    return {
        "video": str(video), "preview": str(preview), "frames": counter,
        "round_segments": segments, "sigma_limits": sigma_limits, **record,
    }


def kazuki_episode(
    policy: Any,
    env: Any,
    gamma: float,
    seed_index: int,
    device: str,
) -> dict[str, Any]:
    import kazuki_baseline as baseline

    baseline.GOAL_COEF = 0.0
    baseline.MARKUP = 1.09
    baseline.REFINEMENT_COST = "b1_safemppi"
    rec: list[dict[str, Any]] = []
    seed = named_seed(VIDEO_VERSION, "kazuki_r19", gamma, seed_index)
    output = baseline.kazuki_deploy(
        policy, env, [0.1], gamma_ctx=gamma, T=300, reach=0.15,
        device=device, seed=seed, rec=rec,
        conditioning_schema="low7_closest_boundary_tie_mean",
    )
    outcome = "SR" if output["reached"] else ("CR" if output["collided"] or output["oob"] else "TO")
    return {
        "gamma": gamma, "seed_index": seed_index, "seed": seed,
        "path": np.asarray(output["path"], dtype=np.float32),
        "outcome": outcome, "rec": rec,
    }


def render_kazuki_video(
    outdir: Path,
    key: str,
    env: Any,
    episodes: dict[float, dict[str, Any]],
    fps: int,
) -> dict[str, Any]:
    frame_root = outdir / "frames" / key
    frame_root.mkdir(parents=True)
    max_step = max(len(episode["rec"]) for episode in episodes.values()) - 1
    shown = list(range(0, max_step + 1, 2))
    if shown[-1] != max_step:
        shown.append(max_step)
    for frame_index, step in enumerate(shown):
        figure, axis = plt.subplots(figsize=(9.2, 9.2))
        _draw_scene(axis, env)
        for gamma, episode in episodes.items():
            color = COLORS[gamma]
            path = episode["path"]
            end = min(step + 2, len(path))
            axis.plot(path[:end, 0], path[:end, 1], color=color, lw=3.0, zorder=7)
            rec_index = min(step, len(episode["rec"]) - 1)
            row = episode["rec"][rec_index]
            state = np.asarray(row["state"], dtype=float)[:2]
            guidance = np.asarray(row["guidance"], dtype=float).mean(axis=0)
            axis.arrow(
                state[0], state[1], 0.18 * guidance[0], 0.18 * guidance[1],
                width=0.012, head_width=0.085, length_includes_head=True,
                color=color, alpha=0.9, zorder=10,
            )
            best = np.asarray(row["best"], dtype=float)
            axis.plot(best[:, 0], best[:, 1], ls="--", lw=1.5, color=color, zorder=6)
            if step >= len(episode["rec"]) - 1 and episode["outcome"] != "SR":
                axis.plot(*path[-1], marker="x", color=RED, markersize=13, markeredgewidth=3, zorder=12)
        set_title(figure, frame_index)
        figure.tight_layout(rect=(0, 0, 1, 0.955))
        figure.savefig(frame_root / f"frame_{frame_index:06d}.png", dpi=165)
        plt.close(figure)
    video = outdir / f"{key}.mp4"
    record = encode_video(frame_root, video, fps)
    preview = outdir / f"{key}_preview.png"
    shutil.copyfile(frame_root / f"frame_{len(shown)//2:06d}.png", preview)
    return {"video": str(video), "preview": str(preview), "frames": len(shown), **record}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--latest-r19-ckpt", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--verifier-workers", type=int, default=16)
    parser.add_argument("--expert-search-workers", type=int, default=16)
    parser.add_argument("--expert-search-size", type=int, default=2000)
    parser.add_argument(
        "--reuse-expansion-replay", type=Path, default=None,
        help="reuse an already completed diagnostic replay after a render-only restart",
    )
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"fresh output directory required: {args.outdir}")
    args.outdir.mkdir(parents=True)
    (args.outdir / "frames").mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": shutil.which("latex") is not None,
    })

    id_profile = "low7_id_canonical_v1"
    ood_profile = "low7_radius1_canonical_v1"
    id_env = build_scene(get_scene_profile(id_profile))
    ood_env = build_scene(get_scene_profile(ood_profile))

    id_episodes = {}
    for gamma in GAMMAS:
        chosen = None
        for index in range(50):
            episode = safe_mppi_episode(id_profile, gamma, index)
            if episode["outcome"] == "SR":
                chosen = episode
                break
        if chosen is None:
            raise RuntimeError(f"no ID SafeMPPI success for gamma={gamma:g}")
        id_episodes[gamma] = chosen

    ood_failure_indices = {
        gamma: first_safemppi_failure(
            ood_profile, gamma, args.expert_search_size, args.expert_search_workers
        )
        for gamma in (0.5, 1.0)
    }
    ood_failures = {
        gamma: safe_mppi_episode(ood_profile, gamma, index)
        for gamma, index in ood_failure_indices.items()
    }

    replay = (
        torch.load(args.reuse_expansion_replay, map_location="cpu", weights_only=False)
        if args.reuse_expansion_replay is not None
        else generate_expansion_replay(
            args.run_dir, args.probe, args.device, args.verifier_workers
        )
    )
    torch.save(replay, args.outdir / "expansion_replay.pt")

    latest, _ = HP.load_hp(str(args.latest_r19_ckpt), device="cpu")
    latest = latest.to(args.device).eval()
    kazuki = {
        gamma: kazuki_episode(latest, ood_env, gamma, 0, args.device)
        for gamma in GAMMAS
    }
    if any(episode["outcome"] == "SR" for episode in kazuki.values()):
        # The requested diagnostic is a failure video.  Choose the first shared
        # fixed seed index whose three cells all fail and record it explicitly.
        for index in range(1, 20):
            candidate = {
                gamma: kazuki_episode(latest, ood_env, gamma, index, args.device)
                for gamma in GAMMAS
            }
            if all(episode["outcome"] != "SR" for episode in candidate.values()):
                kazuki = candidate
                break
        else:
            raise RuntimeError("no all-failure Kazuki triple in declared seed bank")

    rendered = {}
    rendered["01_safemppi_id_nominal"] = render_safemppi_video(
        args.outdir, "01_safemppi_id_nominal", id_profile, id_episodes, args.fps
    )
    rendered["02_safemppi_ood_nominal_failure"] = render_safemppi_video(
        args.outdir, "02_safemppi_ood_nominal_failure", ood_profile,
        ood_failures, args.fps,
    )
    rendered["03_b1_r0_r15_expansion_mechanism"] = render_expansion_video(
        args.outdir, "03_b1_r0_r15_expansion_mechanism", replay, ood_env,
        args.fps,
    )
    success = replay["success_r15"]
    rendered["04_b1_r15_ood_verifier_success"] = render_verified_panels(
        args.outdir, "04_b1_r15_ood_verifier_success", ood_env,
        success["episodes"], success["viz"], args.fps,
        uncertainty=False, round_i=15,
    )
    rendered["05_kazuki_r19_ood_guidance_failure"] = render_kazuki_video(
        args.outdir, "05_kazuki_r19_ood_guidance_failure", ood_env, kazuki,
        args.fps,
    )

    manifest = {
        "status": "B1_SHARED_INDEXED_VIDEOS_COMPLETE",
        "version": VIDEO_VERSION,
        "semantics": {
            "top_title_only": True,
            "frame_index": "zero-based rendered frame index",
            "safe_mppi": "actual nominal H_P level sets returned by unmodified SafeMPPI",
            "expansion": "diagnostic replay with actual checkpoints and logged beta; one episode/gamma; K=16, B=4; not a claim of historical compact-r15 trace",
            "positive_dot": "blue robot-state dot when at least one selected-B full-H query is positive",
            "negative_dot": "red robot-state dot at NVP",
            "verifier_polytope": "green full-H candidate-specific verifier level set for the executed query when certified",
            "kazuki_guidance": "arrow is the mean first-step reward-guidance vector over the 200 generated samples; native B1 SafeMPPI cost is used at all three refinement stages",
        },
        "scenes": {
            "id": scene_snapshot(id_env, get_scene_profile(id_profile)),
            "ood": scene_snapshot(ood_env, get_scene_profile(ood_profile)),
        },
        "episodes": {
            "safe_mppi_id": {
                f"{gamma:g}": {
                    "rollout_index": episode["rollout_index"],
                    "seed": episode["seed"],
                    "outcome": episode["outcome"],
                } for gamma, episode in id_episodes.items()
            },
            "safe_mppi_ood_failure": {
                f"{gamma:g}": {
                    "rollout_index": episode["rollout_index"],
                    "seed": episode["seed"],
                    "outcome": episode["outcome"],
                } for gamma, episode in ood_failures.items()
            },
            "expansion_replay_purpose": replay["purpose"],
            "expansion_seed_contract": {
                "base_seed": replay["base_seed"],
                "proposal": "named_seed(base_seed, 'proposal', purpose, round, gamma_episode_id, control_t)",
                "acquisition": "named_seed(base_seed, 'acquisition', purpose, round, gamma_episode_id, control_t)",
                "gamma_episode_id": "0 for gamma=0.1, 1 for gamma=0.5, 2 for gamma=1.0",
            },
            "expansion_round_status": {
                str(round_i): record["status"]
                for round_i, record in replay["retained"].items()
            },
            "r15_success_purpose": success["purpose"],
            "kazuki": {
                f"{gamma:g}": {
                    "seed_index": episode["seed_index"],
                    "seed": episode["seed"],
                    "outcome": episode["outcome"],
                } for gamma, episode in kazuki.items()
            },
        },
        "kazuki_safe_coef": 0.1,
        "beta_used": replay["betas"],
        "rbf_lengthscale": replay["lengthscale"],
        "checkpoints": {
            f"r{round_i}": sha256_file(args.run_dir / f"ckpt_{round_i}.pt")
            for round_i in range(16)
        },
        "latest_r19_checkpoint_sha256": sha256_file(args.latest_r19_ckpt),
        "rendered": rendered,
    }
    (args.outdir / "video_suite_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
