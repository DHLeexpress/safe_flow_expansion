#!/usr/bin/env python3
"""Shared, source-grounded metric helpers for the paper baseline figures."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "source_snapshot"
for entry in (
    SNAPSHOT / "overnight_run_2026-07-01",
    SNAPSHOT / "overnight_run_07_06" / "rev_expansion" / "codex_overnight",
    SNAPSHOT / "overnight_run_07_06" / "rev_expansion" / "codex_overnight" / "paper_results",
):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


GAMMAS = (0.1, 0.3, 0.5, 1.0)
DT = 0.1
REACH = 0.15
H_WIN = 10
STRIDE = 2
SOCP_R = 2.5
N_THETA = 180


def gamma_key(gamma: float) -> str:
    return f"g{gamma:g}"


def load_path_archive(path: Path) -> dict[float, dict[str, Any]]:
    """Load packed ``paths_g*`` / ``outcomes_g*`` trajectory cells."""

    cells: dict[float, dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as archive:
        for gamma in GAMMAS:
            suffix = gamma_key(gamma)
            path_key = f"paths_{suffix}"
            outcome_key = f"outcomes_{suffix}"
            if path_key not in archive.files:
                continue
            paths = [np.asarray(value, dtype=np.float64) for value in archive[path_key]]
            outcomes = (
                [str(value) for value in archive[outcome_key]]
                if outcome_key in archive.files
                else ["" for _ in paths]
            )
            cells[gamma] = {"paths": paths, "outcomes": outcomes}
    return cells


def _rate(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    mean = float(values.mean()) if n else math.nan
    se = float(math.sqrt(mean * (1.0 - mean) / n)) if n else math.nan
    return {"mean": mean, "se": se, "n": n}


def _mean(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if not n:
        return {"mean": math.nan, "se": math.nan, "n": 0}
    se = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": float(values.mean()), "se": se, "n": n}


def path_is_valid(path: np.ndarray, gamma: float, obstacles: np.ndarray, robot_radius: float) -> bool:
    """Match the raw evaluator's taskspace + sliding-window SOCP predicate."""

    from verifier_polytope import certify_window

    points = np.asarray(path, dtype=np.float64)
    n_steps = len(points) - 1
    if n_steps < H_WIN or np.any(points < 0.0) or np.any(points > 5.0):
        return False
    for start in range(0, n_steps, STRIDE):
        span = min(H_WIN, n_steps - start)
        ok, *_ = certify_window(
            points[start : start + span + 1],
            obstacles,
            robot_radius,
            gamma,
            R=SOCP_R,
            n_theta=N_THETA,
        )
        if not ok:
            return False
    return True


def summarize_path_cell(
    paths: list[np.ndarray],
    outcomes: list[str],
    gamma: float,
    obstacles: np.ndarray,
    robot_radius: float,
) -> dict[str, Any]:
    """Compute the four paper metrics plus SR/timeout from retained paths."""

    if not paths:
        raise ValueError(f"empty trajectory cell for gamma={gamma:g}")
    cr = np.asarray([outcome == "CR" for outcome in outcomes], dtype=np.float64)
    sr = np.asarray([outcome == "SR" for outcome in outcomes], dtype=np.float64)
    timeout = np.asarray([outcome == "TO" for outcome in outcomes], dtype=np.float64)
    clearances = []
    valid = []
    times = []
    for path, outcome in zip(paths, outcomes):
        points = np.asarray(path, dtype=np.float64)
        if obstacles.size:
            clearance = (
                np.linalg.norm(points[:, None, :] - obstacles[None, :, :2], axis=2)
                - obstacles[None, :, 2]
                - robot_radius
            ).min()
        else:
            clearance = math.inf
        clearances.append(float(clearance))
        valid.append(path_is_valid(points, gamma, obstacles, robot_radius))
        if outcome == "SR":
            times.append((len(points) - 1) * DT)
    return {
        "M": len(paths),
        "SR": _rate(sr),
        "CR": _rate(cr),
        "timeout": _rate(timeout),
        "v_safe": _rate(np.asarray(valid, dtype=np.float64)),
        "clearance": _mean(np.asarray(clearances, dtype=np.float64)),
        "time": _mean(np.asarray(times, dtype=np.float64)),
    }


def scene_geometry() -> tuple[np.ndarray, float]:
    from afe2_scene_profiles import build_scene, get_scene_profile

    env = build_scene(get_scene_profile("low7_radius1_canonical_v1"))
    return env.obstacles.detach().cpu().numpy(), float(env.r_robot)


def summarize_path_archive(path: Path) -> dict[float, dict[str, Any]]:
    obstacles, robot_radius = scene_geometry()
    return {
        gamma: summarize_path_cell(
            cell["paths"], cell["outcomes"], gamma, obstacles, robot_radius
        )
        for gamma, cell in load_path_archive(path).items()
    }


def load_jsonl_round(path: Path, round_i: int | None = None) -> tuple[int, dict[float, dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    rounds = sorted({int(row["round"]) for row in rows})
    selected = rounds[-1] if round_i is None else int(round_i)
    table = {
        float(row["gamma"]): row
        for row in rows
        if int(row["round"]) == selected and float(row["gamma"]) in GAMMAS
    }
    missing = set(GAMMAS) - set(table)
    if missing:
        raise ValueError(f"{path} round {selected} is missing gammas {sorted(missing)}")
    return selected, table


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
