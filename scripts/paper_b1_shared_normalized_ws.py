#!/usr/bin/env python3
"""Replace the shared B1 baseline row with the normalized high-alpha screen.

The first two rows are loaded from the authenticated B1 shared-gallery
archives.  Baseline trajectories use the retained common-random-number M=10
screen at raw safety coefficients 3 and 6, displayed as paper coefficients
0.5 and 1.0 under the declared normalization ``w_s = raw_w_s / 6``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_b1_shared_galleries import GAMMAS, render_gallery
from afe2_scene_profiles import build_scene, get_scene_profile


RAW_TO_PAPER_SCALE = 6.0
RAW_COEFFICIENTS = (3.0, 6.0)
PAPER_COEFFICIENTS = tuple(
    value / RAW_TO_PAPER_SCALE for value in RAW_COEFFICIENTS
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cells(
    path: Path,
) -> dict[float, tuple[list[np.ndarray], list[str], list[int]]]:
    result = {}
    with np.load(path, allow_pickle=True) as archive:
        for gamma in GAMMAS:
            suffix = f"g{gamma:g}"
            paths = [
                np.asarray(value, dtype=np.float32)
                for value in archive[f"paths_{suffix}"]
            ]
            outcomes = [str(value) for value in archive[f"outcomes_{suffix}"]]
            indices = (
                [int(value) for value in archive[f"indices_{suffix}"]]
                if f"indices_{suffix}" in archive.files
                else list(range(len(paths)))
            )
            result[gamma] = (paths, outcomes, indices)
    return result


def validate_arm(
    directory: Path,
    *,
    raw_safe_coef: float,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "KAZUKI_ALPHA_COEFFICIENT_GRID_COMPLETE":
        raise RuntimeError(f"incomplete source manifest: {manifest_path}")
    if manifest["gammas"] != list(GAMMAS) or manifest["M_per_cell"] != 10:
        raise RuntimeError(f"{manifest_path}: unexpected evaluation contract")
    matches = [
        entry
        for entry in manifest["outputs"]
        if float(entry["alpha"]) == 4.0
        and float(entry["goal_coef"]) == 0.0
        and float(entry["safe_coef"]) == raw_safe_coef
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{manifest_path}: expected one alpha=4, wg=0, "
            f"raw ws={raw_safe_coef:g} arm"
        )
    entry = matches[0]
    source = directory / entry["file"]
    if sha256_file(source) != entry["sha256"]:
        raise RuntimeError(f"source hash mismatch: {source}")
    return source, manifest


def outcome_counts(
    cells: dict[float, tuple[list[np.ndarray], list[str], list[int]]],
) -> dict[str, dict[str, int]]:
    return {
        f"{gamma:g}": {
            label: outcomes.count(label)
            for label in ("SR", "CR", "TO")
        }
        for gamma, (_, outcomes, _) in cells.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-source",
        type=Path,
        default=ROOT / "provenance/b1_current_best/gallery_shared_v3",
    )
    parser.add_argument(
        "--raw-ws3",
        type=Path,
        default=ROOT / "provenance/paper_baselines/kazuki_alpha34_wall_grid_m10",
    )
    parser.add_argument(
        "--raw-ws6",
        type=Path,
        default=ROOT / "provenance/paper_baselines/kazuki_alpha4_wg0_ws6_m10",
    )
    parser.add_argument(
        "--paper-outdir",
        type=Path,
        default=ROOT / "assets/paper",
    )
    parser.add_argument(
        "--shared-outdir",
        type=Path,
        default=ROOT / "assets/results/b1_current_best",
    )
    parser.add_argument(
        "--provenance-outdir",
        type=Path,
        default=ROOT / "provenance/b1_current_best/gallery_shared_v4",
    )
    args = parser.parse_args()

    expert_source = args.shared_source / "expert_id.npz"
    ours_source = args.shared_source / "ours_r15_ood.npz"
    raw3_source, raw3_manifest = validate_arm(args.raw_ws3, raw_safe_coef=3.0)
    raw6_source, raw6_manifest = validate_arm(args.raw_ws6, raw_safe_coef=6.0)

    expert = load_cells(expert_source)
    ours = load_cells(ours_source)
    ws05 = load_cells(raw3_source)
    ws10 = load_cells(raw6_source)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": shutil.which("latex") is not None,
        }
    )
    id_env = build_scene(get_scene_profile("low7_id_canonical_v1"))
    ood_env = build_scene(get_scene_profile("low7_radius1_canonical_v1"))

    args.paper_outdir.mkdir(parents=True, exist_ok=True)
    args.shared_outdir.mkdir(parents=True, exist_ok=True)
    args.provenance_outdir.mkdir(parents=True, exist_ok=True)

    comparison_rows = [
        (r"CFM--MPPI$^*$" + "\n" + r"$w_s=0.5$", ood_env, ws05),
        (r"CFM--MPPI$^*$" + "\n" + r"$w_s=1.0$", ood_env, ws10),
    ]
    comparison_outputs = render_gallery(
        args.paper_outdir,
        "b1_kazuki_normalized_ws_2x3_gallery",
        comparison_rows,
        (15.5, 9.6),
    )
    shared_rows = [
        ("In distribution\n(SafeMPPI)", id_env, expert),
        ("Out of distribution\n(Ours, r15)", ood_env, ours),
        *comparison_rows,
    ]
    shared_outputs = render_gallery(
        args.shared_outdir,
        "b1_shared_3x3_gallery",
        shared_rows,
        (15.5, 19.2),
    )

    sources = {
        "expert_id": expert_source,
        "ours_r15_ood": ours_source,
        "raw_ws3": raw3_source,
        "raw_ws6": raw6_source,
    }
    manifest = {
        "status": "B1_SHARED_NORMALIZED_WS_GALLERIES_COMPLETE",
        "canonical_plot_recipe": "scripts/build_b1_shared_galleries.py",
        "renderer": "scripts/paper_b1_shared_normalized_ws.py",
        "layout": {
            "comparison": "2 rows x 3 gamma columns",
            "shared": (
                "4 rows x 3 gamma columns; legacy b1_shared_3x3_gallery filename "
                "retained for downstream compatibility"
            ),
            "shared_first_two_rows": (
                "unchanged authenticated SafeMPPI-ID and Ours-r15 cells"
            ),
        },
        "guidance": {
            "alpha": 4.0,
            "goal_coefficient": 0.0,
            "normalization": "paper_w_s = raw_safe_coefficient / 6",
            "arms": [
                {"raw_safe_coefficient": raw, "paper_w_s": paper}
                for raw, paper in zip(RAW_COEFFICIENTS, PAPER_COEFFICIENTS)
            ],
            "refinement_cost": "b1_safemppi",
            "M_per_gamma": 10,
            "gammas": list(GAMMAS),
            "seed_contract": raw6_manifest["seed_contract"],
            "raw_ws3_manifest_sha256": sha256_file(args.raw_ws3 / "manifest.json"),
            "raw_ws6_manifest_sha256": sha256_file(args.raw_ws6 / "manifest.json"),
            "raw_ws3_elapsed_seconds": raw3_manifest["elapsed_seconds"],
            "raw_ws6_elapsed_seconds": raw6_manifest["elapsed_seconds"],
        },
        "outcome_counts": {
            "paper_ws0.5": outcome_counts(ws05),
            "paper_ws1.0": outcome_counts(ws10),
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
        "outputs": {
            "comparison": comparison_outputs,
            "shared": shared_outputs,
        },
    }
    manifest_path = args.provenance_outdir / "gallery_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
