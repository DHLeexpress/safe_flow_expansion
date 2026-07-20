from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np


SAFEGPC_V4_1_GRID = [float(x) for x in np.geomspace(0.05, 1.0, 10)]
SAFEGPC_V4_2_GRID = [float(x) for x in np.geomspace(0.1, 1.0, 10)]


def gamma_distance_velocity(
    d: float,
    v_proj: float,
    g_min: float = 0.1,
    g_max: float = 1.0,
    alpha: float = 0.1541,
    beta: float = 1.5826,
) -> float:
    d = max(0.0, float(d))
    v_eff = max(0.0, float(v_proj))
    gamma = g_min + (g_max - g_min) * (1.0 - math.exp(-beta * d)) * math.exp(-alpha * v_eff)
    return float(np.clip(gamma, g_min, g_max))


def gamma_schedule_values(name: str) -> List[float]:
    if name == "safeGPC_v4_1":
        return SAFEGPC_V4_1_GRID
    if name == "safeGPC_v4_2":
        return SAFEGPC_V4_2_GRID
    if name in {"fallback_grid_not_paper_schedule", "fallback"}:
        return [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9, 0.99]
    raise ValueError(f"Unknown gamma schedule {name!r}")


def resolve_gamma_schedule(gamma_grid: Iterable[float] | None, gamma_schedule: str | None) -> List[float]:
    if gamma_grid:
        return [float(g) for g in gamma_grid]
    if gamma_schedule:
        return gamma_schedule_values(gamma_schedule)
    return gamma_schedule_values("fallback_grid_not_paper_schedule")
