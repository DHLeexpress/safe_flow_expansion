"""polytope_v2: a clean, GENERAL deterministic convex polytope (reset of the polytope.py concept).

Differences from `build_nominal_polytope` (the "6:2:4:4" box):
  - NO head bias: the base is NOT a forward-oriented box. It is a robot-centered sensing DISK of radius R,
    approximated by a fixed, axis-aligned inner K-gon (K>=4). Orientation does not depend on heading.
  - NO forced symmetry: the only asymmetry comes from the actually-detected obstacles (real clutter).
  - General polytope (faces = K + #detected obstacles), tangent to obstacles, and CONTINUOUS as obstacles move.

Construction (support-function view; robot center c, sensing radius R, obstacle inflation `margin`):
  - base face k:        n_k . (p - c) <= R*cos(pi/K)          (inner K-gon of the disk; polygon ⊆ disk)
  - obstacle face j:    m_j . (p - c) <= ||o_j - c|| - rho_j  (tangent to the inflated obstacle, robot inside)
    with m_j = (o_j - c)/||o_j - c||, rho_j = r_j + margin, only for obstacles with clearance <= R.

Continuity: each face offset is a support value, continuous in obstacle positions. As an obstacle leaves the
disk its offset -> ~R*... and becomes redundant with the base K-gon, so the polytope relaxes smoothly (no jump).

Reuses the `Polytope` dataclass (A,b,ref,margins,contains,barrier) so all consumers + `_norm_barrier` rendering
work unchanged.
"""
from __future__ import annotations

from typing import Optional, Tuple

import math
import numpy as np
import torch

from .polytope import Polytope


def build_polytope_v2(
    pos: torch.Tensor | np.ndarray,            # [2] robot position (disk center)
    obstacles: torch.Tensor | np.ndarray,      # [N,3] (cx,cy,radius) RAW
    *,
    sensing_range: float = 4.0,                # disk radius R
    n_base: int = 16,                          # K base directions (>=4); larger K -> rounder disk
    margin: float = 0.0,                       # obstacle inflation (0 = tangent to the actual obstacle)
    max_obstacles: int = 12,
    obstacle_velocities=None,                  # [N,2] pedestrian velocities (for the predictive offset)
    robot_velocity=None,                       # [2] robot velocity
    predict_gain: float = 0.0,                 # kappa: face retreats by kappa*tau*max(0,closing speed)
    predict_tau: float = 1.0,                  # tau = H*dt prediction horizon (seconds)
) -> Tuple[Polytope, dict]:
    c = (pos.detach().cpu().numpy() if torch.is_tensor(pos) else np.asarray(pos, float)).astype(float).reshape(2)
    R = float(sensing_range)
    K = max(4, int(n_base))
    vrob = (np.zeros(2) if robot_velocity is None else
            (robot_velocity.detach().cpu().numpy() if torch.is_tensor(robot_velocity) else np.asarray(robot_velocity, float)).reshape(2))
    vobs = (None if obstacle_velocities is None else
            (obstacle_velocities.detach().cpu().numpy() if torch.is_tensor(obstacle_velocities) else np.asarray(obstacle_velocities, float)).reshape(-1, 2))

    # --- base: inner K-gon of the robot-centered disk (fixed axis-aligned orientation, no head bias) ---
    thetas = np.arange(K) * (2 * math.pi / K)
    A_rows = [np.array([math.cos(t), math.sin(t)]) for t in thetas]
    base_off = R * math.cos(math.pi / K)                      # apothem -> every face strictly inside the disk
    b_rows = [float(n @ c + base_off) for n in A_rows]

    # --- one tangent half-space per detected (nearby) obstacle (+ velocity-predictive retreat) ---
    obs = (obstacles.detach().cpu().numpy() if torch.is_tensor(obstacles) else np.asarray(obstacles, float))
    obs = obs.reshape(-1, 3).astype(float) if obs.size else np.zeros((0, 3))
    n_detected = 0
    if obs.shape[0]:
        d = np.linalg.norm(obs[:, :2] - c, axis=1)
        clr = d - (obs[:, 2] + margin)
        for j in np.argsort(clr):
            if n_detected >= max_obstacles or clr[j] > R:
                break
            if d[j] < 1e-9:
                continue
            m = (obs[j, :2] - c) / d[j]                       # robot -> obstacle (outward)
            off = float(d[j] - (obs[j, 2] + margin))          # tangent to inflated obstacle, robot side
            if predict_gain > 0.0 and vobs is not None and j < vobs.shape[0]:
                v_close = float(m @ (vrob - vobs[j]))         # >0: gap closing -> retreat the face
                off -= predict_gain * predict_tau * max(0.0, v_close)
            A_rows.append(m); b_rows.append(float(m @ c + off)); n_detected += 1

    A = torch.tensor(np.stack(A_rows), dtype=torch.float32)
    b = torch.tensor(np.array(b_rows), dtype=torch.float32)
    poly = Polytope(A=A, b=b, ref=torch.tensor(c, dtype=torch.float32))
    info = {"center": c, "R": R, "K": K, "n_detected": n_detected, "n_faces": A.shape[0],
            "contains_robot": bool(np.all(np.stack(A_rows) @ c <= np.array(b_rows) + 1e-6))}
    return poly, info
