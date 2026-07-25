#!/usr/bin/env python3
"""Authenticated raw temperature-1 evaluation for the B1 execution-rule study.

This tool deliberately reuses ``eval_rounds_m`` for bank generation, bare
receding-horizon rollouts, trajectory verification metrics, and aggregation.
Its only addition is a deterministic per-cell NPZ archive plus explicit route
mode counts.  No gathering-time execution rule is consulted during evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

WORKBOOK = Path(__file__).resolve().parents[1]
SCRIPTS = WORKBOOK / "scripts"
SNAP = (
    WORKBOOK
    / "source_snapshot/overnight_run_07_06/rev_expansion/codex_overnight"
)
for entry in (
    SCRIPTS,
    SNAP.parents[1],
    SNAP.parent,
    SNAP,
    SNAP / "paper_results",
):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import eval_rounds_m as EVAL  # noqa: E402


EXPECTED_M = 200
EXPECTED_SPLIT = "b1_exec_rule_temp1_m200_v1"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_rounds(text: str) -> list[int]:
    result: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"descending round range: {part}")
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(part))
    if not result or len(result) != len(set(result)):
        raise ValueError("rounds must be nonempty and unique")
    return result


def gamma_slug(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ whose container metadata is stable across invocations."""

    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def route_labels(episodes, env) -> np.ndarray:
    import afe_route_metrics as routes

    obstacles = env.obstacles.detach().cpu().numpy()
    centers = obstacles[:, :2]
    radii = obstacles[:, 2] + float(env.r_robot)
    start = env.x0.detach().cpu().numpy()[:2]
    goal = env.goal.detach().cpu().numpy()
    labels = []
    for episode in episodes:
        path = np.asarray(episode["path"], dtype=np.float32)
        value, _ = routes.classify_trajectories_at_closest_approach(
            path[None],
            start=start,
            goal=goal,
            obstacle_centers=centers,
            obstacle_radii=radii,
        )
        labels.append(int(value[0]))
    return np.asarray(labels, dtype=np.int8)


def pack_cell(episodes, metrics, env) -> tuple[dict[str, np.ndarray], dict]:
    import afe_route_metrics as routes

    m = len(episodes)
    paths = np.full((m, EVAL.T + 1, 2), np.nan, dtype=np.float32)
    lengths = np.empty(m, dtype=np.int16)
    status = np.empty(m, dtype="S9")
    for index, episode in enumerate(episodes):
        path = np.asarray(episode["path"], dtype=np.float32)
        paths[index, : len(path)] = path
        lengths[index] = len(path)
        status[index] = (
            "timeout" if episode["status"] is None else str(episode["status"])
        ).encode("ascii")

    success = np.asarray([row["success"] for row in metrics], dtype=np.bool_)
    cr = np.asarray([row["cr"] for row in metrics], dtype=np.bool_)
    timeout = np.asarray([row["timeout"] for row in metrics], dtype=np.bool_)
    validity = np.asarray([row["v_safe"] for row in metrics], dtype=np.bool_)
    clearance = np.asarray(
        [row["minimum_clearance"] for row in metrics], dtype=np.float64
    )
    steps = np.asarray([row["steps"] for row in metrics], dtype=np.int16)
    time_to_goal = np.asarray(
        [
            np.nan if row["time_to_goal"] is None else row["time_to_goal"]
            for row in metrics
        ],
        dtype=np.float64,
    )
    route = route_labels(episodes, env)
    route_summary = routes.summarize_modes(route)

    arrays = {
        "rollout_index": np.arange(m, dtype=np.int16),
        "paths": paths,
        "path_length": lengths,
        "status": status,
        "success": success,
        "collision_rate_event": cr,
        "timeout_event": timeout,
        "validity_event": validity,
        "minimum_clearance": clearance,
        "steps": steps,
        "successful_time_to_goal": time_to_goal,
        "route_label": route,
    }

    def mean_se(values: np.ndarray) -> dict:
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "se": (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            ),
            "n": int(len(values)),
        }

    finite_time = time_to_goal[np.isfinite(time_to_goal)]
    aggregate = {
        "SR": mean_se(success),
        "CR": mean_se(cr),
        "timeout": mean_se(timeout),
        "Validity": mean_se(validity),
        "minimum_clearance": mean_se(clearance),
        "successful_time_to_goal": (
            mean_se(finite_time)
            if len(finite_time)
            else {"mean": None, "se": 0.0, "n": 0}
        ),
        "route": route_summary,
    }
    return arrays, aggregate


def configure_determinism(torch) -> dict:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def nvidia_driver_version() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--rounds", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tag", choices=("margin", "cost"), required=True)
    parser.add_argument("--m", type=int, default=EXPECTED_M)
    parser.add_argument("--bank-split", default=EXPECTED_SPLIT)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.m != EXPECTED_M:
        raise ValueError(f"this study is locked to M={EXPECTED_M}")
    if args.bank_split != EXPECTED_SPLIT:
        raise ValueError(f"this study is locked to split {EXPECTED_SPLIT}")
    if args.outdir.exists():
        raise FileExistsError(f"fresh shard output required: {args.outdir}")
    rounds = parse_rounds(args.rounds)
    if min(rounds) < 0 or max(rounds) > 15:
        raise ValueError("paper comparison is prespecified to r0-r15")
    missing = [
        round_i
        for round_i in rounds
        if not (args.arm_dir / f"ckpt_{round_i}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    import torch
    import afe_m20_eval as M20
    import grid_hp_expt as HP
    from afe2_scene_profiles import build_scene, get_scene_profile

    deterministic = configure_determinism(torch)
    args.outdir.mkdir(parents=True)
    archive_dir = args.outdir / "cells"
    archive_dir.mkdir()
    jsonl_path = args.outdir / f"{args.tag}.jsonl"
    contract_path = args.outdir / f"{args.tag}.contract.json"
    bank_path = args.outdir / f"{args.tag}_common_bank.npy"

    env = build_scene(get_scene_profile(EVAL.SCENE))
    probe, _ = HP.load_hp(
        str(args.arm_dir / f"ckpt_{rounds[0]}.pt"), device="cpu"
    )
    bank = EVAL.make_bank(
        len(EVAL.GAMMAS), args.m, int(probe.d), args.bank_split
    )
    np.save(bank_path, bank, allow_pickle=False)
    bank_sha256 = hashlib.sha256(bank.tobytes(order="C")).hexdigest()

    checkpoint_hashes = {
        str(round_i): sha256_file(args.arm_dir / f"ckpt_{round_i}.pt")
        for round_i in rounds
    }
    archive_hashes: dict[str, str] = {}
    t0 = time.time()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=M20._worker_init,
        initargs=(EVAL.SCENE, EVAL.REACH, EVAL.N_THETA),
    ) as executor, jsonl_path.open("x") as stream:
        for round_i in rounds:
            policy, _ = HP.load_hp(
                str(args.arm_dir / f"ckpt_{round_i}.pt"), device="cpu"
            )
            policy = policy.to(args.device).eval()
            episodes = EVAL.run_fixed(
                policy,
                env,
                args.device,
                bank,
                args.m,
                EVAL.GAMMAS,
                (1.0,) * len(EVAL.GAMMAS),
                seed_round=round_i,
            )
            tasks = [
                (
                    np.asarray(episode["path"], dtype=np.float32),
                    episode["gamma"],
                    "timeout"
                    if episode["status"] is None
                    else episode["status"],
                    float(env.dt),
                    EVAL.REACH,
                )
                for episode in episodes
            ]
            metrics = list(
                executor.map(M20._trajectory_metrics_worker, tasks, chunksize=4)
            )
            for gamma_index, gamma in enumerate(EVAL.GAMMAS):
                indices = [
                    index
                    for index, episode in enumerate(episodes)
                    if episode["gamma_index"] == gamma_index
                ]
                cell_episodes = [episodes[index] for index in indices]
                cell_metrics = [metrics[index] for index in indices]
                arrays, aggregate = pack_cell(cell_episodes, cell_metrics, env)
                relative = (
                    f"cells/r{round_i:03d}_g{gamma_slug(gamma)}.npz"
                )
                cell_path = args.outdir / relative
                deterministic_npz(cell_path, arrays)
                archive_hashes[relative] = sha256_file(cell_path)
                record = {
                    "round": round_i,
                    "gamma": gamma,
                    "mode": "fixed",
                    "temperature": 1.0,
                    "M": args.m,
                    "NFE": EVAL.NFE,
                    "bank_split": args.bank_split,
                    "bank_sha256": bank_sha256,
                    "archive": relative,
                    "archive_sha256": archive_hashes[relative],
                    **aggregate,
                }
                stream.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
                print(
                    f"[{args.tag}] r{round_i:02d} g{gamma:g} "
                    f"SR={record['SR']['mean']:.3f} "
                    f"CR={record['CR']['mean']:.3f} "
                    f"V={record['Validity']['mean']:.3f}",
                    flush=True,
                )
            del policy, episodes, metrics
            torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(args.device)
    contract = {
        "status": "B1_EXEC_RULE_TEMP1_M200_SHARD_COMPLETE",
        "arm": args.tag,
        "arm_dir": str(args.arm_dir.resolve()),
        "rounds": rounds,
        "gammas": list(EVAL.GAMMAS),
        "cells": len(rounds) * len(EVAL.GAMMAS),
        "M_per_gamma": args.m,
        "temperature": 1.0,
        "NFE": EVAL.NFE,
        "scene": EVAL.SCENE,
        "raw_policy": (
            "bare receding-horizon flow; no GP, uncertainty tilt, verifier "
            "filter, verified controller, fallback, or execution rule"
        ),
        "bank_version": EVAL.BANK_VERSION,
        "bank_split": args.bank_split,
        "bank_shape": list(bank.shape),
        "bank_sha256": bank_sha256,
        "bank_file": str(bank_path.resolve()),
        "bank_file_sha256": sha256_file(bank_path),
        "checkpoint_sha256": checkpoint_hashes,
        "scientific_archive_sha256": archive_hashes,
        "metrics": str(jsonl_path.resolve()),
        "metrics_sha256": sha256_file(jsonl_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
            "device_uuid": str(properties.uuid),
            "driver": nvidia_driver_version(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(args.device),
            "determinism": deterministic,
        },
        "batch_layout": (
            "one synchronous GPU batch containing every active trajectory "
            "across all seven gamma cells at each receding-horizon step"
        ),
        "workers": args.workers,
        "elapsed_seconds": time.time() - t0,
    }
    with contract_path.open("x") as stream:
        json.dump(contract, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
