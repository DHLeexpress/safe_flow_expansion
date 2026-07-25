#!/usr/bin/env python3
"""Complete the local M10 gamma grid used by the paper baseline plotters.

Existing authenticated gamma 0.1/0.5/1.0 trajectories are copied unchanged.
Only the absent gamma 0.3 cells are regenerated, using the same named-seed
contracts and current r19 checkpoint as the retained native-cost gallery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = (
    ROOT
    / "source_snapshot"
    / "overnight_run_07_06"
    / "rev_expansion"
    / "codex_overnight"
    / "paper_results"
)
for entry in (PAPER, PAPER.parent, PAPER.parent.parent, PAPER.parent.parent.parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from afe2_scene_profiles import build_scene, get_scene_profile
from b1_current_best_gallery import METRIC_VERSION, run_expert, run_kazuki
from grid_hp_expt import load_hp


COEFFICIENTS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
GAMMAS = (0.1, 0.3, 0.5, 1.0)
EXPECTED_CHECKPOINT = "60c155472f5ed0e4a1d53581857f09aead7924f8ce11e8e3adf890d5a57fc079"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cell(path: Path, gamma: float) -> tuple[list[np.ndarray], list[str]]:
    suffix = f"g{gamma:g}"
    with np.load(path, allow_pickle=True) as archive:
        return (
            [np.asarray(value, dtype=np.float32) for value in archive[f"paths_{suffix}"]],
            [str(value) for value in archive[f"outcomes_{suffix}"]],
        )


def pack(path: Path, cells: dict[float, tuple[list[np.ndarray], list[str]]]) -> None:
    payload = {}
    for gamma, (paths, outcomes) in cells.items():
        suffix = f"g{gamma:g}"
        object_paths = np.empty(len(paths), dtype=object)
        object_paths[:] = paths
        payload[f"paths_{suffix}"] = object_paths
        payload[f"outcomes_{suffix}"] = np.asarray(outcomes)
    np.savez_compressed(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "provenance/b1_current_best/gallery_native_cost_v2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/b1_current_best_r19.pt",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "provenance/paper_baselines/local_native_cost_m10",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"fresh output directory required: {args.outdir}")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"unexpected r19 checkpoint: {checkpoint_sha}")
    args.outdir.mkdir(parents=True)

    env = build_scene(get_scene_profile("low7_radius1_canonical_v1"))
    policy, _ = load_hp(str(args.checkpoint), device="cpu")
    policy = policy.to(args.device).eval()

    expert = {
        gamma: load_cell(args.source_dir / "expert.npz", gamma)
        for gamma in (0.1, 0.5, 1.0)
    }
    expert[0.3] = run_expert(env, 0.3, 10, 300, 0.15)
    expert_path = args.outdir / "safemppi_ood_m10.npz"
    pack(expert_path, expert)

    outputs = {"safemppi": expert_path.name}
    for coefficient in COEFFICIENTS:
        source = args.source_dir / f"kazuki_ws_{coefficient:g}.npz"
        cells = {
            gamma: load_cell(source, gamma)
            for gamma in (0.1, 0.5, 1.0)
        }
        cells[0.3] = run_kazuki(
            policy, env, coefficient, 0.3, 10, 300, 0.15, args.device
        )
        output = args.outdir / f"kazuki_wg0_ws{coefficient:g}_ood_m10.npz"
        pack(output, cells)
        outputs[f"wg0_ws{coefficient:g}"] = output.name

    manifest = {
        "status": "PAPER_BASELINE_LOCAL_M10_COMPLETE",
        "scientific_scope": (
            "fixed-index local baseline screening; existing gamma 0.1/0.5/1.0 "
            "cells copied byte-for-value from gallery_native_cost_v2, gamma 0.3 "
            "regenerated from the same named-seed implementation"
        ),
        "metric_version": METRIC_VERSION,
        "scene": "low7_radius1_canonical_v1",
        "checkpoint_sha256": checkpoint_sha,
        "goal_coef": 0.0,
        "safe_coefficients": list(COEFFICIENTS),
        "gammas": list(GAMMAS),
        "M_per_cell": 10,
        "T": 300,
        "reach": 0.15,
        "device": args.device,
        "source_dir": str(args.source_dir.resolve()),
        "source_sha256": {
            "expert.npz": sha256_file(args.source_dir / "expert.npz"),
            **{
                f"kazuki_ws_{coefficient:g}.npz": sha256_file(
                    args.source_dir / f"kazuki_ws_{coefficient:g}.npz"
                )
                for coefficient in COEFFICIENTS
            },
        },
        "outputs": {
            key: {"file": value, "sha256": sha256_file(args.outdir / value)}
            for key, value in outputs.items()
        },
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
