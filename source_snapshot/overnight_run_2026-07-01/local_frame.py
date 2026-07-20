"""Goal-aligned robot-centered local frame (agent-centric normalization).

x-axis = current goal direction `e_g = normalize(goal − pos)`, `e_lat = rotate90(e_g)`. Velocity, previous
action and the TARGET control window are all expressed in this frame so forward/lateral are comparable
across scenes/positions (motion-forecasting convention). Inference rotates the generated `U_local` back to
world before dynamics/verifier.
"""
from __future__ import annotations

import numpy as np


def goal_frame(pos, goal):
    """Return (e_g, e_lat, goal_dist). e_lat = rotate90(e_g) (left-perpendicular)."""
    p = np.asarray(pos, dtype=float)[:2]
    g = np.asarray(goal, dtype=float)[:2]
    d = g - p
    n = float(np.linalg.norm(d))
    e_g = d / n if n > 1e-9 else np.array([1.0, 0.0])
    e_lat = np.array([-e_g[1], e_g[0]])
    return e_g, e_lat, n


def to_local(vec, e_g, e_lat):
    """World vec(s) [...,2] -> local [dot(·,e_g), dot(·,e_lat)]."""
    v = np.asarray(vec, dtype=float)
    x = v[..., 0] * e_g[0] + v[..., 1] * e_g[1]
    y = v[..., 0] * e_lat[0] + v[..., 1] * e_lat[1]
    return np.stack([x, y], axis=-1)


def to_world(vec_local, e_g, e_lat):
    """Local vec(s) [...,2] -> world = x·e_g + y·e_lat."""
    v = np.asarray(vec_local, dtype=float)
    return v[..., 0:1] * e_g + v[..., 1:2] * e_lat


# scales (config knobs)
U_MAX = 2.0
V_SCALE = 2.0
R_GOAL_SCALE = 6.0


def low_dim_features(state, goal, gamma, a_prev=None, prev_valid=False,
                     u_max=U_MAX, v_scale=V_SCALE, r_goal=R_GOAL_SCALE):
    """low_dim[7] = [goal_dist/R, v·e_g/vs, v·e_lat/vs, a_prev·e_g/umax, a_prev·e_lat/umax, γ, prev_valid]."""
    s = np.asarray(state, dtype=float)
    p, v = s[:2], s[2:4]
    e_g, e_lat, gd = goal_frame(p, goal)
    v_loc = to_local(v, e_g, e_lat) / v_scale
    if a_prev is not None and prev_valid:
        a_loc = to_local(a_prev, e_g, e_lat) / u_max
    else:
        a_loc = np.zeros(2)
    low = np.array([gd / r_goal, v_loc[0], v_loc[1], a_loc[0], a_loc[1], float(gamma), float(bool(prev_valid))],
                   dtype=np.float32)
    return low, (e_g, e_lat)


if __name__ == "__main__":
    # round-trip world -> local -> world
    e_g, e_lat, gd = goal_frame([1.0, 1.0], [4.0, 5.0])
    u = np.array([[0.3, -1.2], [1.0, 0.5], [-0.7, 0.2]])
    ul = to_local(u, e_g, e_lat)
    uw = to_world(ul, e_g, e_lat)
    print("frame e_g", e_g.round(3), "e_lat", e_lat.round(3), "goal_dist", round(gd, 3))
    print("round-trip max|Δ|:", float(np.abs(uw - u).max()))
    low, _ = low_dim_features([1, 1, 0.5, -0.3], [4, 5], 0.5, a_prev=[0.2, 0.1], prev_valid=True)
    print("low_dim[7]:", low.round(3), "len", len(low))
