#!/usr/bin/env python3
"""Generate the paper/SFM B1 comparison galleries from authenticated rollouts.

The learned row is regenerated from the exact r15 checkpoint using the same
M=200 CRN bank and per-gamma temperatures as the revised r0--r15 result plot.
Only successful trajectories are displayed in that row, as requested; their
original M=200 rollout indices are written to the manifest.  SafeMPPI and
CFM--MPPI rows use fixed named seeds and are never selected by appearance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
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
for _path in (_REPO / "scripts", _WORK, _REV, _CORE, _PAPER):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afe2_scene_profiles import build_scene, get_scene_profile, scene_snapshot
import grid_hp_expt as HP
import eval_rounds_m as EVAL
from b1_current_best_gallery import (
    classify_path,
    run_expert,
    run_kazuki,
)


GAMMAS = (0.1, 0.5, 1.0)
REVISED_TEMPERATURES = (0.5, 0.75, 1.0)
M200_SPLIT = "b1_margin_fixedtemp_m200_v1"
EXPECTED_R15_SHA256 = "604f8fe657254c1ec644dce16dbea28a36d6e3308aa9ee712adbac3ec68ca672"
STATE_DOT_STRIDE = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_cells(path: Path, cells: dict[float, tuple[list[np.ndarray], list[str], list[int]]]) -> None:
    payload: dict[str, np.ndarray] = {}
    for gamma, (paths, outcomes, indices) in cells.items():
        suffix = f"g{gamma:g}"
        object_paths = np.empty(len(paths), dtype=object)
        object_paths[:] = paths
        payload[f"paths_{suffix}"] = object_paths
        payload[f"outcomes_{suffix}"] = np.asarray(outcomes)
        payload[f"indices_{suffix}"] = np.asarray(indices, dtype=np.int32)
    np.savez_compressed(path, **payload)


def raw_cells(
    policy: Any,
    env: Any,
    device: str,
    bank: np.ndarray,
    m: int,
    temperatures: tuple[float, ...],
    successful_only: bool,
) -> dict[float, tuple[list[np.ndarray], list[str], list[int]]]:
    episodes = EVAL.run_fixed(
        policy, env, device, bank, m, GAMMAS, temperatures, seed_round=15
    )
    result = {}
    for gamma_index, gamma in enumerate(GAMMAS):
        records = [episode for episode in episodes if episode["gamma_index"] == gamma_index]
        paths = [np.asarray(record["path"], dtype=np.float32) for record in records]
        outcomes = [classify_path(path, env, EVAL.REACH) for path in paths]
        eligible = [index for index, outcome in enumerate(outcomes) if outcome == "SR"]
        if not successful_only:
            eligible = list(range(len(paths)))
        if len(eligible) < 10:
            raise RuntimeError(
                f"gamma={gamma:g} has only {len(eligible)} eligible trajectories"
            )
        chosen = eligible[:10]
        result[gamma] = (
            [paths[index] for index in chosen],
            [outcomes[index] for index in chosen],
            chosen,
        )
    return result


def expert_cells(env: Any, m: int) -> dict[float, tuple[list[np.ndarray], list[str], list[int]]]:
    return {
        gamma: (*run_expert(env, gamma, m, EVAL.T, EVAL.REACH), list(range(m)))
        for gamma in GAMMAS
    }


def kazuki_cells(
    policy: Any,
    env: Any,
    safe_coef: float,
    m: int,
    device: str,
) -> dict[float, tuple[list[np.ndarray], list[str], list[int]]]:
    return {
        gamma: (
            *run_kazuki(policy, env, safe_coef, gamma, m, EVAL.T, EVAL.REACH, device),
            list(range(m)),
        )
        for gamma in GAMMAS
    }


def failure_only_cells(
    cells: dict[float, tuple[list[np.ndarray], list[str], list[int]]],
) -> dict[float, tuple[list[np.ndarray], list[str], list[int]]]:
    """Retain every fixed-bank failure; never replace it with a curated success."""

    output = {}
    for gamma, (paths, outcomes, indices) in cells.items():
        chosen = [index for index, outcome in enumerate(outcomes) if outcome != "SR"]
        if not chosen:
            raise RuntimeError(f"Kazuki has no failure to display at gamma={gamma:g}")
        output[gamma] = (
            [paths[index] for index in chosen],
            [outcomes[index] for index in chosen],
            [indices[index] for index in chosen],
        )
    return output


def draw_scene(
    axis: Any,
    env: Any,
    paths: list[np.ndarray],
    outcomes: list[str],
    gamma: float,
    *,
    title: str,
    ylabel: str,
) -> None:
    for obstacle in env.obstacles.detach().cpu().numpy():
        axis.add_patch(plt.Circle(obstacle[:2], obstacle[2], color="#bdbdbd", zorder=1))
    color = plt.get_cmap("plasma")({0.1: 0.08, 0.5: 0.52, 1.0: 0.92}[gamma])
    for path, outcome in zip(paths, outcomes):
        path = np.asarray(path, dtype=float)
        axis.plot(path[:, 0], path[:, 1], color=color, lw=1.5, alpha=0.72, zorder=3)
        dots = path[::STATE_DOT_STRIDE]
        axis.plot(
            dots[:, 0], dots[:, 1], linestyle="none", marker=".", color=color,
            markersize=2.4, alpha=0.72, zorder=4,
        )
        if outcome != "SR":
            axis.plot(
                path[-1, 0], path[-1, 1], marker="x", linestyle="none",
                color="#cc3311", markersize=10, markeredgewidth=2.4, zorder=7,
            )
    start = env.x0.detach().cpu().numpy()[:2]
    goal = env.goal.detach().cpu().numpy()
    axis.plot(*start, "ks", markersize=6, zorder=8)
    axis.plot(*goal, marker="*", color="gold", markeredgecolor="black", markersize=14, zorder=8)
    axis.set_xlim(-0.3, 5.3)
    axis.set_ylim(-0.3, 5.3)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=25, pad=10)
    axis.set_ylabel(ylabel, fontsize=23, labelpad=15)


def render_gallery(
    outdir: Path,
    stem: str,
    rows: list[tuple[str, Any, dict[float, tuple[list[np.ndarray], list[str], list[int]]]]],
    size: tuple[float, float],
) -> dict[str, str]:
    figure, axes = plt.subplots(len(rows), 3, figsize=size, squeeze=False)
    for row_index, (label, env, cells) in enumerate(rows):
        for column_index, gamma in enumerate(GAMMAS):
            paths, outcomes, _ = cells[gamma]
            draw_scene(
                axes[row_index, column_index], env, paths, outcomes, gamma,
                title=rf"$\gamma={gamma:g}$" if row_index == 0 else "",
                ylabel=label if column_index == 0 else "",
            )
    figure.subplots_adjust(
        left=0.17, right=0.99, bottom=0.015, top=0.985,
        wspace=0.035, hspace=0.055,
    )
    outputs = {}
    for suffix in ("png", "pdf"):
        path = outdir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=260 if suffix == "png" else None, bbox_inches="tight")
        outputs[path.name] = sha256_file(path)
    plt.close(figure)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-ckpt", type=Path, required=True)
    parser.add_argument("--r15-ckpt", type=Path, required=True)
    parser.add_argument("--latest-r19-ckpt", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m-display", type=int, default=10)
    args = parser.parse_args()
    if args.m_display != 10:
        raise ValueError("paper galleries are locked to ten trajectories per cell")
    if args.outdir.exists():
        raise FileExistsError(f"fresh output directory required: {args.outdir}")
    if sha256_file(args.r15_ckpt) != EXPECTED_R15_SHA256:
        raise RuntimeError("r15 checkpoint does not match the revised M200 plot")
    args.outdir.mkdir(parents=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": shutil.which("latex") is not None,
    })
    id_profile = get_scene_profile("low7_id_canonical_v1")
    ood_profile = get_scene_profile("low7_radius1_canonical_v1")
    id_env = build_scene(id_profile)
    ood_env = build_scene(ood_profile)

    pretrained, _ = HP.load_hp(str(args.pretrained_ckpt), device="cpu")
    r15, _ = HP.load_hp(str(args.r15_ckpt), device="cpu")
    latest, _ = HP.load_hp(str(args.latest_r19_ckpt), device="cpu")
    pretrained = pretrained.to(args.device).eval()
    r15 = r15.to(args.device).eval()
    latest = latest.to(args.device).eval()

    full_bank = EVAL.make_bank(len(EVAL.GAMMAS), 200, int(r15.d), M200_SPLIT)
    selected_bank = full_bank[[EVAL.GAMMAS.index(gamma) for gamma in GAMMAS]]
    expert = expert_cells(id_env, args.m_display)
    raw0 = raw_cells(
        pretrained, ood_env, args.device, selected_bank[:, :10], 10,
        (1.0, 1.0, 1.0), successful_only=False,
    )
    ours = raw_cells(
        r15, ood_env, args.device, selected_bank, 200,
        REVISED_TEMPERATURES, successful_only=True,
    )
    kazuki_low = kazuki_cells(latest, ood_env, 0.1, args.m_display, args.device)
    kazuki_high = kazuki_cells(latest, ood_env, 0.9, args.m_display, args.device)
    kazuki_failures = failure_only_cells(kazuki_low)

    cells = {
        "expert_id": expert,
        "pretrained_ood": raw0,
        "ours_r15_ood": ours,
        "kazuki_r19_ws01": kazuki_low,
        "kazuki_r19_ws09": kazuki_high,
    }
    for name, values in cells.items():
        pack_cells(args.outdir / f"{name}.npz", values)

    five_rows = [
        ("SafeMPPI\n(in distribution)", id_env, expert),
        ("Pretrained\n(out of distribution)", ood_env, raw0),
        ("Ours, r15\n(out of distribution)", ood_env, ours),
        ("CFM–MPPI$^*$, $w_s=0.1$", ood_env, kazuki_low),
        ("CFM–MPPI$^*$, $w_s=0.9$", ood_env, kazuki_high),
    ]
    three_rows = [
        ("In distribution\n(SafeMPPI)", id_env, expert),
        ("Out of distribution\n(Ours, r15)", ood_env, ours),
        ("Out of distribution\n(CFM–MPPI$^*$)", ood_env, kazuki_failures),
    ]
    outputs = {}
    outputs.update(render_gallery(
        args.outdir, "b1_current_best_5x3_gallery", five_rows, (15.5, 23.5)
    ))
    outputs.update(render_gallery(
        args.outdir, "b1_shared_3x3_gallery", three_rows, (15.5, 14.4)
    ))

    manifest = {
        "status": "B1_SHARED_GALLERIES_COMPLETE",
        "scenes": {
            "id": scene_snapshot(id_env, id_profile),
            "ood": scene_snapshot(ood_env, ood_profile),
        },
        "checkpoints": {
            "pretrained": sha256_file(args.pretrained_ckpt),
            "ours_r15": sha256_file(args.r15_ckpt),
            "kazuki_latest_r19": sha256_file(args.latest_r19_ckpt),
        },
        "ours_r15": {
            "evaluation": "bare receding-horizon policy; no GP, acquisition, verifier, or fallback",
            "bank_version": EVAL.BANK_VERSION,
            "bank_split": M200_SPLIT,
            "M_per_gamma": 200,
            "temperatures": {f"{g:g}": t for g, t in zip(GAMMAS, REVISED_TEMPERATURES)},
            "display_rule": "first ten successful rollout indices in the declared M=200 bank",
            "display_indices": {f"{g:g}": ours[g][2] for g in GAMMAS},
        },
        "kazuki": {
            "model": "latest B1 r19",
            "goal_coef": 0.0,
            "safe_coefs": [0.1, 0.9],
            "refinement_cost": "exact native B1 SafeMPPI cost at generated-mode ranking, perturbation weights, and refined-mode selection",
            "display_rule": "fixed named indices 0--9; no outcome curation",
            "three_by_three_failure_row": {
                "rule": "all failures among the fixed low-guidance indices 0--9",
                "safe_coef": 0.1,
                "indices": {
                    f"{gamma:g}": kazuki_failures[gamma][2]
                    for gamma in GAMMAS
                },
            },
        },
        "failure_marker": "red X at terminal point; no outcome text",
        "outputs": outputs,
    }
    (args.outdir / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
