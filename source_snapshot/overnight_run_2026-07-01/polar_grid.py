"""Robot-centered, goal-aligned POLAR polytope-occupancy grid — what the SafeMPPI expert 'sees'.

Grid `[3, N_θ=16, N_r=12]` in the goal-aligned local frame (matches the expert config: R = sensing =
barrier_activation_radius = 3.0, N_θ = polytope_nbase = 16):
  θ_j = −π + (j+.5)·2π/N_θ ,  r_k = (k+.5)·R/N_r ,  p(j,k) = c + r_k(cosθ_j·e_g + sinθ_j·e_lat)
Channels:
  0 occupancy    — inside an inflated obstacle (r_j + r_robot)  → collision geometry
  1 polytope_mask— H_P(p) ≥ 0  (inside the nominal SafeMPPI polytope)
  2 H_P clipped  — clip(H_P(p), −1, 1)  (nominal barrier value; H_P(c)=1, =0 on a face)

Reuses `build_polytope_v2` (the exact polytope SafeMPPI builds) and the H_P formula
`min_k (b_k − a_k·x)/margin_k`. Later: raise N_θ/N_r and use a circular-padded polar CNN.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from cfm_mppi.safegpc_adapter.polytope_v2 import build_polytope_v2
from local_frame import goal_frame

R_SENSE = 3.0        # = barrier_activation_radius
N_THETA = 16         # = polytope_nbase
N_R = 12
R_ROBOT = 0.2


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=float)


def polar_points(c, goal, R=R_SENSE, n_theta=N_THETA, n_r=N_R):
    """Return grid points [n_theta, n_r, 2] and the (e_g, e_lat) frame."""
    c = np.asarray(c, dtype=float)[:2]
    e_g, e_lat, _ = goal_frame(c, goal)
    theta = -np.pi + (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta      # [n_theta]
    r = (np.arange(n_r) + 0.5) * R / n_r                                    # [n_r]
    dirs = (np.cos(theta)[:, None, None] * e_g[None, None, :]
            + np.sin(theta)[:, None, None] * e_lat[None, None, :])          # [n_theta,1,2]
    pts = c[None, None, :] + r[None, :, None] * dirs                        # [n_theta,n_r,2]
    return pts, (e_g, e_lat)


def polytope_HP(c, obstacles, sensing=R_SENSE, n_base=N_THETA, predict_gain=0.0):
    """Nominal SafeMPPI polytope + a callable H_P(points[M,2])->[M]. Matches the expert (margin=0)."""
    c = np.asarray(c, dtype=float)[:2]
    obs = _np(obstacles)
    poly, _ = build_polytope_v2(c, obs, sensing_range=float(sensing), n_base=int(n_base),
                                margin=0.0, predict_gain=float(predict_gain))
    A = _np(poly.A); b = _np(poly.b)
    margins = np.maximum(b - A @ c, 1e-3)

    def HP(pts):
        pts = np.asarray(pts, dtype=float)
        return ((b[None] - pts @ A.T) / margins[None]).min(1)
    return HP, (A, b, margins)


def polar_grid(c, goal, obstacles, r_robot=R_ROBOT, R=R_SENSE, n_theta=N_THETA, n_r=N_R,
               sensing=R_SENSE, predict_gain=0.0):
    """Return grid [3, n_theta, n_r] (float32) and the (e_g,e_lat) frame."""
    pts, frame = polar_points(c, goal, R, n_theta, n_r)
    flat = pts.reshape(-1, 2)                                              # [M,2]
    obs = _np(obstacles)
    if obs.size:
        d = np.linalg.norm(flat[:, None, :] - obs[None, :, :2], axis=2) - (obs[None, :, 2] + r_robot)
        occ = (d.min(1) < 0).astype(np.float32)                           # ch0 occupancy
    else:
        occ = np.zeros(len(flat), np.float32)
    HP, _ = polytope_HP(c, obs, sensing=sensing, n_base=n_theta, predict_gain=predict_gain)
    hp = HP(flat)
    mask = (hp >= 0).astype(np.float32)                                   # ch1 polytope_mask
    hclip = np.clip(hp, -1.0, 1.0).astype(np.float32)                     # ch2 clipped H_P
    grid = np.stack([occ, mask, hclip], 0).reshape(3, n_theta, n_r)
    return grid.astype(np.float32), frame


if __name__ == "__main__":
    import scenes
    env = scenes.make_narrow_gap(gap_offset=0.80, gap_r=0.35)
    c = np.array([2.4, 0.0])                                              # just before the gap
    grid, (e_g, e_lat) = polar_grid(c, env.goal.numpy(), env.obstacles.numpy())
    print("grid shape", grid.shape, "(expect (3,16,12))")
    print("occupancy frac:", round(float(grid[0].mean()), 3),
          "| polytope_mask frac:", round(float(grid[1].mean()), 3),
          "| H_P range:", round(float(grid[2].min()), 2), round(float(grid[2].max()), 2))
    HP, _ = polytope_HP(c, env.obstacles.numpy())
    print("H_P(center) =", round(float(HP(c[None])[0]), 4), "(expect 1.0)")
