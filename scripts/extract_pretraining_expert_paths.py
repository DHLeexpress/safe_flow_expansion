#!/usr/bin/env python3
"""Extract executed SafeMPPI paths from the authenticated pretraining shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_GAMMAS = (0.1, 0.5, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gamma_key(value: float) -> str:
    return f"{value:g}"


def extract_shard(path: Path) -> tuple[list[np.ndarray], list[int], list[int]]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    states = payload["verifier_state"].numpy()
    trajectory_ids = payload["window_trajectory_ids"].numpy()
    window_steps = payload["window_steps"].numpy()
    rows = payload["trajectory_rows"]

    paths = []
    seeds = []
    pair_ids = []
    for row in rows:
        trajectory_id = int(row["trajectory_id"])
        selected = np.flatnonzero(trajectory_ids == trajectory_id)
        order = np.argsort(window_steps[selected], kind="stable")
        selected = selected[order]
        expected_steps = int(row["steps"])
        if len(selected) != expected_steps:
            raise RuntimeError(
                f"{path}: trajectory {trajectory_id} has {len(selected)} "
                f"stored states, expected {expected_steps}"
            )
        if not np.array_equal(window_steps[selected], np.arange(expected_steps)):
            raise RuntimeError(
                f"{path}: trajectory {trajectory_id} steps are not contiguous"
            )
        paths.append(np.asarray(states[selected, :2], dtype=np.float32))
        seeds.append(int(row["seed"]))
        pair_ids.append(int(row["pair_id"]))
    return paths, seeds, pair_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAMMAS),
    )
    args = parser.parse_args()

    combined = json.loads(args.combined_manifest.read_text())
    shards = {float(entry["gamma"]): entry for entry in combined["shards"]}
    archive: dict[str, Any] = {}
    source_shards = {}
    for gamma in args.gammas:
        entry = shards[float(gamma)]
        shard_path = Path(entry["dataset"])
        if sha256_file(shard_path) != entry["dataset_sha256"]:
            raise RuntimeError(f"shard hash mismatch: {shard_path}")
        paths, seeds, pair_ids = extract_shard(shard_path)
        key = gamma_key(gamma)
        archive[f"paths_g{key}"] = np.asarray(paths, dtype=object)
        archive[f"outcomes_g{key}"] = np.asarray(["SR"] * len(paths))
        archive[f"indices_g{key}"] = np.asarray(seeds, dtype=np.int64)
        archive[f"pair_ids_g{key}"] = np.asarray(pair_ids, dtype=np.int32)
        source_shards[key] = {
            "path": str(shard_path),
            "sha256": entry["dataset_sha256"],
            "trajectories": len(paths),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **archive)
    result = {
        "status": "PRETRAINING_EXPERT_PATHS_EXTRACTED",
        "source_combined_manifest": str(args.combined_manifest),
        "source_combined_manifest_sha256": sha256_file(args.combined_manifest),
        "source_dataset_sha256": combined["dataset_sha256"],
        "gammas": [float(value) for value in args.gammas],
        "path_semantics": (
            "actual executed SafeMPPI states reconstructed as verifier_state "
            "ordered by (trajectory_id, window_step); no rollout is resampled"
        ),
        "source_shards": source_shards,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.manifest_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
