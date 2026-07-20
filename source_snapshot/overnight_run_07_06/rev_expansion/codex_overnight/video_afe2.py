"""AFE2 expansion video with all seven gamma panels in every rendered round.

Colors (fixed by spec):
  gray        all K=64 generated plans at every executed control step
  orange      every B full-verifier query object (halo; SOCP solve count is separate)
  green       full-H SOCP-positive queried plans
  red         full-H rejected queried plans (some may be terminal-prefix admissible)
  blue/thick  cost-selected plan (argmax progress) + the executed first-action path
  X           NO_VERIFIED_POSITIVE termination point
  text        positive count, min SOCP margin, raw untilted validity (audit), termination timestep

Usage: python video_afe2.py --run results/afe2/afe_s910 --out paper_results/afe2_afe.mp4
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

import _paths  # noqa: F401
from afe2_scene_profiles import (
    assert_scene_snapshot,
    build_scene,
    get_scene_profile,
    scene_snapshot,
)

GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def select_video_rounds(rounds, dense_until=None, every_after=None):
    """Select authenticated rounds for rendering without deleting trainer artifacts."""

    values = [int(value) for value in rounds]
    if (dense_until is None) != (every_after is None):
        raise ValueError("--dense-until and --every-after must be supplied together")
    if dense_until is None:
        return values
    dense_until = int(dense_until)
    every_after = int(every_after)
    if dense_until < 0 or every_after <= 0:
        raise ValueError("video schedule requires dense-until >= 0 and every-after > 0")
    selected = [
        value for value in values
        if value <= dense_until or (value > dense_until and value % every_after == 0)
    ]
    if not selected:
        raise ValueError("video schedule selected no rounds")
    return selected


def expected_viz_rounds(recipe):
    """Return the authenticated viz inventory declared by the trainer recipe."""

    first_round = 0 if recipe.get("video_include_round0") else 1
    values = range(first_round, int(recipe["rounds"]) + 1)
    profile = recipe.get("artifact_profile", "full")
    if profile == "full":
        return list(values)
    if profile == "sweep_compact":
        return [round_i for round_i in values if round_i <= 10 or round_i % 10 == 0]
    raise ValueError(f"unknown trainer artifact profile: {profile}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_scene(ax, scene, goal, x0):
    for o in np.asarray(scene["obstacles"], dtype=float):
        ax.add_patch(plt.Circle((o[0], o[1]), o[2], color="0.82", zorder=1))
    ax.plot([x0[0]], [x0[1]], "ks", ms=5, zorder=6)
    ax.plot([goal[0]], [goal[1]], "*", color="gold", mec="k", ms=13, zorder=6)
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 5.2)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])


def render_round(db, out_png, arm, allow_legacy_claude=False):
    scene = db.get("scene")
    if scene is None:
        if not allow_legacy_claude:
            raise ValueError("round artifact has no serialized scene")
        legacy_profile = get_scene_profile("claude_grid_v1")
        scene = scene_snapshot(
            build_scene(legacy_profile),
            legacy_profile,
        )
    assert_scene_snapshot(scene)
    goal = np.asarray(db["goal"], float)
    x0 = np.asarray(db["x0"], float)
    if not np.array_equal(goal, np.asarray(scene["goal"], float)):
        raise ValueError("round goal does not match its serialized scene")
    if not np.array_equal(x0[:2], np.asarray(scene["start_state"], float)[:2]):
        raise ValueError("round start does not match its serialized scene")
    audit = db.get("audit") or {}
    vg = audit.get("V_gamma", {})
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.6))
    by_g = {g: [] for g in GAMMAS}
    for v in db["viz"]:
        by_g[round(float(v["gamma"]), 2)].append(v)
    ep_by_g = {g: [] for g in GAMMAS}
    for episode in db["eps"]:
        ep_by_g[round(float(episode["gamma"]), 2)].append(episode)
    for ax, g in zip(axes.flat[:7], GAMMAS):
        draw_scene(ax, scene, goal, x0)
        steps = by_g[g]
        gamma_episodes = ep_by_g.get(g, [])
        nquery_tot = npos_tot = nexec_tot = nhp_tot = nrescue_tot = nsolve_tot = 0
        min_marg = np.inf
        for v in steps:
            segs = np.asarray(v["segsK"], np.float32)
            ax.add_collection(LineCollection(
                segs,
                colors=[(0.45, 0.45, 0.45, 0.045)],
                linewidths=0.22,
                zorder=2,
                rasterized=True,
            ))
            for j, y in zip(v["drawn"], v["y"]):
                ax.plot(segs[j, :, 0], segs[j, :, 1], "-",
                        color=(1.00, 0.55, 0.00, 0.45), lw=2.0, zorder=3)
                if int(y) != -1:
                    c = ((0.10, 0.55, 0.15, 0.90) if int(y) == 1
                         else (0.85, 0.12, 0.10, 0.85))
                    ax.plot(segs[j, :, 0], segs[j, :, 1], "-", color=c, lw=0.8, zorder=3.1)
            if v["sel"] >= 0:
                selected = int(v["sel"])
                query_i = list(v["drawn"]).index(selected)
                rescued = bool(v.get("terminal_rescue", [False] * len(v["drawn"]))[query_i])
                tau = v.get("terminal_tau", [None] * len(v["drawn"]))[query_i]
                if rescued:
                    tau = int(tau)
                    ax.plot(segs[selected, :tau, 0], segs[selected, :tau, 1], "-",
                            color="#1155cc", lw=1.5, alpha=0.9, zorder=4)
                    ax.plot(segs[selected, max(tau - 1, 0):, 0],
                            segs[selected, max(tau - 1, 0):, 1], "--",
                            color="#cc3311", lw=0.9, alpha=0.8, zorder=3.5)
                    ax.plot(segs[selected, tau - 1, 0], segs[selected, tau - 1, 1], "o",
                            color="#1155cc", ms=3.5, zorder=4.5)
                else:
                    ax.plot(segs[selected, :, 0], segs[selected, :, 1], "-", color="#1155cc",
                            lw=1.1, alpha=0.8, zorder=4)
            npos_tot += int(np.sum(np.asarray(v["y"]) == 1))
            nquery_tot += len(v["drawn"])
            nexec_tot += int(np.sum(np.asarray(v.get("exec_y", v["y"])) == 1))
            hp_labels = v.get("exec_verified_hp_y")
            if hp_labels is not None:
                nhp_tot += int(np.sum(np.asarray(hp_labels) == 1))
            nrescue_tot += int(np.sum(np.asarray(v.get("terminal_rescue", []), dtype=bool)))
            nsolve_tot += int(v.get("n_socp_solve", 0))
            if np.isfinite(v.get("min_margin", np.nan)):
                min_marg = min(min_marg, float(v["min_margin"]))
        for ep in gamma_episodes:                       # blue/thick: every executed replica
            p = np.asarray(ep["path"], float)
            ax.plot(p[:, 0], p[:, 1], "-", color="#0b3d91", lw=2.4,
                    alpha=0.85, zorder=5)
            if ep["status"] == "nvp":
                ax.plot([p[-1, 0]], [p[-1, 1]], "x", color="k", ms=13, mew=3.2, zorder=7)
            elif ep["status"] == "reached":
                ax.plot([p[-1, 0]], [p[-1, 1]], "*", color="#0b3d91", mec="k", ms=12, zorder=7)
            elif ep["status"] in ("collision", "oob"):
                ax.plot([p[-1, 0]], [p[-1, 1]], "x", color="#cc3311", ms=13, mew=3.2, zorder=7)
        status_counts = {
            name: sum(ep["status"] == name for ep in gamma_episodes)
            for name in ("reached", "nvp", "timeout", "collision", "oob")
        }
        stat = "/".join(
            f"{name}:{count}" for name, count in status_counts.items() if count
        ) or "-"
        nvp_times = [ep.get("term_t") for ep in gamma_episodes if ep["status"] == "nvp"]
        nvp_reasons = sorted({
            str(ep.get("nvp_reason", "unspecified"))
            for ep in gamma_episodes if ep["status"] == "nvp"
        })
        ax.set_title(f"γ={g}", fontsize=13)
        margin_text = f"{min_marg:.3f}" if np.isfinite(min_marg) else "—"
        ax.text(0.02, 0.98,
                f"query objects {nquery_tot} / SOCP solves {nsolve_tot}\n"
                f"full+ {npos_tot} / prefix+ {nexec_tot} / nominal-step+ {nhp_tot}\n"
                f"terminal rescue {nrescue_tot}\n"
                f"min execution-certificate m {margin_text}\n"
                f"V̂_H full {float(vg.get(str(g), np.nan)):.2f}\n{stat}"
                + (f" NVP t={nvp_times} reason={nvp_reasons}" if nvp_times else ""),
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="0.6", alpha=0.85))
    axL = axes.flat[7]
    axL.axis("off")
    axL.legend(handles=[
        Line2D([], [], color="0.55", lw=1.5, alpha=0.6, label="all K=64 generated plans / step"),
        Line2D([], [], color=(1.0, 0.55, 0.0), lw=3,
               label="B full-verifier query objects (orange halo)"),
        Line2D([], [], color=(0.10, 0.55, 0.15), lw=2, label="full-H positive (D+ eligible)"),
        Line2D([], [], color=(0.85, 0.12, 0.10), lw=2, label="full-H rejected"),
        Line2D([], [], color="#1155cc", lw=1.6,
               label="selected certified plan/prefix (red dashed = unverified suffix)"),
        Line2D([], [], color="#0b3d91", lw=2.6, label="executed path"),
        Line2D([], [], color="k", marker="x", ls="", mew=3, ms=10,
               label="NO_EXECUTION_ELIGIBLE / NVP (terminate; no fallback)"),
    ], loc="center", fontsize=10.5, frameon=False)
    scene_name = scene.get("profile", {}).get("name", "unknown_scene")
    if "ensemble_diagnostics" in db:
        algorithm = "AFE-Deep-Ensemble"
    elif "gp_diagnostics" in db:
        algorithm = "AFE-RBF"
    else:
        algorithm = "AFE2"
    axL.text(0.5, 0.05, f"{algorithm}: {arm} — round {int(db['round'])} — {scene_name}\n"
             f"absorbing goal; terminal-prefix rescue is execution-only, never D+",
             ha="center", fontsize=10, color="#333333", transform=axL.transAxes)
    fig.suptitle(f"{algorithm} expert-free verified expansion — {arm}, round {int(db['round'])}",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=105)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument(
        "--dense-until",
        type=int,
        default=None,
        help="render every round through this index (requires --every-after)",
    )
    ap.add_argument(
        "--every-after",
        type=int,
        default=None,
        help="after --dense-until, render rounds divisible by this value",
    )
    ap.add_argument(
        "--allow-legacy-claude",
        action="store_true",
        help="opt in to reconstructing claude_grid_v1 for old artifacts without scene snapshots",
    )
    args = ap.parse_args()
    arm = os.path.basename(args.run.rstrip("/"))
    dbs = sorted(glob.glob(os.path.join(args.run, "viz_db", "round*.pt")),
                 key=lambda p: int(re.findall(r"round(\d+)\.pt", p)[0]))
    recipe_path = os.path.join(args.run, "recipe.json")
    if os.path.isfile(recipe_path):
        with open(recipe_path) as stream:
            recipe = json.load(stream)
        expected = expected_viz_rounds(recipe)
        observed = [int(re.findall(r"round(\d+)\.pt", path)[0]) for path in dbs]
        if observed != expected:
            raise RuntimeError(f"viz rounds are {observed}; expected {expected}")
        expected_scene_sha = (recipe.get("scene") or {}).get("sha256")
        complete_path = os.path.join(args.run, "COMPLETE.json")
        if not os.path.isfile(complete_path):
            if not args.allow_legacy_claude:
                raise RuntimeError("run has no trainer-written COMPLETE.json")
        else:
            with open(complete_path) as stream:
                complete = json.load(stream)
            if complete.get("status") != "COMPLETE":
                raise RuntimeError("run completion marker is invalid")
            for path, round_i in zip(dbs, observed):
                relative = f"viz_db/round{round_i}.pt"
                if sha256_file(path) != complete.get("artifact_sha256", {}).get(relative):
                    raise RuntimeError(f"viz artifact hash mismatch: {relative}")
    else:
        if not args.allow_legacy_claude:
            raise RuntimeError("run has no recipe.json; legacy mode must be explicit")
        expected_scene_sha = None
    observed = [int(re.findall(r"round(\d+)\.pt", path)[0]) for path in dbs]
    selected_rounds = select_video_rounds(
        observed,
        dense_until=args.dense_until,
        every_after=args.every_after,
    )
    selected = set(selected_rounds)
    render_dbs = [
        path for path, round_i in zip(dbs, observed) if round_i in selected
    ]
    tmp = tempfile.mkdtemp(prefix="afe2_vid_")
    try:
        for k, p in enumerate(render_dbs):
            db = torch.load(p, map_location="cpu", weights_only=False)
            if expected_scene_sha is not None and (db.get("scene") or {}).get("sha256") != expected_scene_sha:
                raise RuntimeError(f"scene mismatch in {p}")
            render_round(
                db,
                os.path.join(tmp, f"frame_{k:03d}.png"),
                arm,
                allow_legacy_claude=args.allow_legacy_claude,
            )
            print(f"rendered round {db['round']}", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", os.path.join(tmp, "frame_%03d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", args.out],
                       check=True)
        print("saved", args.out, f"({len(render_dbs)} frames: {selected_rounds})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
