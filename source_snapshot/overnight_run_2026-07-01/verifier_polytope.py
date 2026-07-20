"""Compact-SOCP verifier polytope with the m_min/m_max knob (Pillar 3).

Do NOT modify `ieee_compact_polytope_verifier_package/`. We import its clean plotting/certificate API
from `demo_verifier_polytope.py` and REPLICATE only the ~27-line margin-bounded face solver
`solve_face_bounded_margin` (which lives in the unimportable `pillar3_m_bounds_6x3.py`, since that file
mkdirs /mnt/data at import). The cap `ub = min(m_max, a·d − r)` is the entire "sharp wedge → tube/corridor"
knob; `m_min` is the feasibility floor.

The verifier is a FITTED polytope (it chooses face normals/margins per trajectory), so it can certify a
tight thread that SafeMPPI's fixed NOMINAL polytope rejects — the "less-conservative certificate" thesis.
"""
from __future__ import annotations

import math

import numpy as np

import _paths  # noqa: F401
# clean, side-effect-free API from the ieee package (UNMODIFIED):
from demo_verifier_polytope import (
    Face, draw_panel, H_grid, check_certificate, artificial_obstacles, make_nominal_radial_faces,
    make_variable_faces,   # ORIGINAL unbounded max-margin SOCP (the verifier that EXPANDS the polytope)
)


# ---------------------------------------------------------------- margin-bounded face solver (replicated)
def solve_face_bounded_margin(d, radius, traj, beta, m_min, m_max, kind, label, n_theta=720):
    """One variable tangent face with m_min ≤ m ≤ m_max (the m_max corridor knob). Returns a demo Face.
    traj: robot-centered [H+1,2]; beta: 1−(1−γ)^t; d=o−c (centered obstacle)."""
    d = np.asarray(d, dtype=float).reshape(2)
    pts, bts = traj[1:], beta[1:]
    ths = np.linspace(-math.pi, math.pi, n_theta, endpoint=False)
    best = None
    for th in ths:
        a = np.array([math.cos(th), math.sin(th)])
        ub_obst = float(a @ d - radius)                    # disk-tangent upper bound (||a||=1)
        ub = min(m_max, ub_obst)                           # <<< m_max CAP (corridor knob)
        lb_traj = float(np.max((pts @ a) / bts)) if len(pts) else -np.inf
        lb = max(m_min, lb_traj)                            # <<< m_min FLOOR + trajectory containment
        if lb <= ub + 1e-8:
            m = ub
            score = (m, float(a @ d))
            if best is None or score > best[0]:
                best = (score, Face(a=a, m=m, kind=kind, label=label, feasible=True))
    if best is None:
        return Face(np.array([1.0, 0.0]), 0.0, kind, label, feasible=False)
    return best[1]


def build_faces(traj_c, obstacles_c, gamma, *, R=2.0, K=12, rho_art=0.16,
                m_min=1e-4, m_max=None, n_theta=720):
    """Faces for one robot-centered window. obstacles_c: list of (dx,dy,r_effective). Returns (faces, art).

    m_max=None (default) → the ORIGINAL ieee UNBOUNDED max-margin SOCP (make_variable_faces): the verifier
    EXPANDS each face out to the obstacle tangent, certifying tight paths the fixed nominal polytope rejects.
    m_max set → optional margin-bounded 'tube/corridor' variant (replicated solver), just a visual knob.
    """
    H = traj_c.shape[0] - 1
    alpha = (1.0 - gamma) ** np.arange(H + 1, dtype=float)
    beta = 1.0 - alpha
    if m_max is None:
        return make_variable_faces(list(obstacles_c), traj_c, beta, R=R, K_artificial=K,
                                   rho_art=rho_art, m_min=m_min)
    faces = []
    for j, (ox, oy, rr) in enumerate(obstacles_c):
        faces.append(solve_face_bounded_margin(np.array([ox, oy]), rr, traj_c, beta,
                                               m_min, m_max, "real", f"real{j}", n_theta))
    art = artificial_obstacles(R, K, rho_art)
    for l, (ox, oy, rr) in enumerate(art):
        faces.append(solve_face_bounded_margin(np.array([ox, oy]), rr, traj_c, beta,
                                               m_min, m_max, "artificial", f"art{l}", n_theta))
    return faces, art


def _sense_center(seg, obstacles, r_robot, R_eff):
    """Re-center a window at seg[0]; return (traj_c, real_obs_c[inflated], real_obs_raw_c)."""
    c = np.asarray(seg[0], dtype=float)
    traj_c = np.asarray(seg, dtype=float) - c
    infl, raw = [], []
    for (ox, oy, rr) in np.asarray(obstacles, dtype=float):
        dx, dy = ox - c[0], oy - c[1]
        if math.hypot(dx, dy) - rr <= R_eff:
            infl.append((dx, dy, rr + r_robot))            # Minkowski-inflated for separation
            raw.append((dx, dy, rr))                        # true disk for display
    return traj_c, infl, raw


def certify_window(seg, obstacles, r_robot, gamma, *, R=2.0, K=12, rho_art=0.16,
                   m_min=1e-4, m_max=None, n_theta=720, r_pad=1.3):
    """Certify one H-step window (seg[0]=center). Returns (ok, faces, real_obs_raw_c, R_eff)."""
    traj_c0 = np.asarray(seg, dtype=float) - np.asarray(seg[0], dtype=float)
    R_eff = max(R, r_pad * float(np.linalg.norm(traj_c0, axis=1).max()))
    traj_c, infl, raw = _sense_center(seg, obstacles, r_robot, R_eff)
    faces, _ = build_faces(traj_c, infl, gamma, R=R_eff, K=K, rho_art=rho_art,
                           m_min=m_min, m_max=m_max, n_theta=n_theta)
    alpha = (1.0 - gamma) ** np.arange(traj_c.shape[0], dtype=float)
    ok, _slack, _t = check_certificate(faces, traj_c, alpha, include_start=False)
    return ok, faces, raw, R_eff


def certify_trajectory(traj, obstacles, r_robot, gamma, *, H_win=10, stride=2,
                       R=2.0, K=12, rho_art=0.16, m_min=1e-4, m_max=None, n_theta=360):
    """Whole path certified iff every sliding H_win window certifies."""
    T = len(traj) - 1
    for k in range(0, T, stride):
        Hs = min(H_win, T - k)
        if Hs < 1:
            break
        ok, *_ = certify_window(traj[k:k + Hs + 1], obstacles, r_robot, gamma,
                                R=R, K=K, rho_art=rho_art, m_min=m_min, m_max=m_max, n_theta=n_theta)
        if not ok:
            return False
    return True


def plot_green_polytope(ax, seg, obstacles, r_robot, gamma, *, R=2.0, K=12, rho_art=0.16,
                        m_min=1e-4, m_max=None, n_theta=1440, world=True, xlim=None, ylim=None,
                        show_nominal=False):
    """Draw the GREEN verifier polytope for one window via the ieee draw_panel (robot-centered).
    If world=True the axes are shifted so the drawing sits at the true robot position."""
    c = np.asarray(seg[0], dtype=float)
    ok, faces, raw, R_eff = certify_window(seg, obstacles, r_robot, gamma, R=R, K=K, rho_art=rho_art,
                                           m_min=m_min, m_max=m_max, n_theta=n_theta)
    traj_c = np.asarray(seg, dtype=float) - c
    H = traj_c.shape[0] - 1
    alpha = (1.0 - gamma) ** np.arange(H + 1, dtype=float)
    art = artificial_obstacles(R_eff, K, rho_art)
    nominal = make_nominal_radial_faces(raw, R=R_eff, K_base=16) if show_nominal else None
    xl = xlim or (-R_eff - 0.2, R_eff + 0.2)
    yl = ylim or (-R_eff - 0.2, R_eff + 0.2)
    draw_panel(ax, faces, raw, art, traj_c, alpha, f"γ={gamma}  cert={ok}  m∈[{m_min},{m_max}]",
               xlim=xl, ylim=yl, nominal_faces=nominal)
    if world:                                              # relabel ticks to world coords (drawing stays centered)
        ax.set_title(ax.get_title(), fontsize=8)
    return ok, R_eff


if __name__ == "__main__":
    # Tight gap: obstacles (3,±0.6) r=0.35 -> robot-center corridor 0.6-0.35-0.2 = 0.05 (razor-thin thread).
    obs = np.array([[3.0, 0.6, 0.35], [3.0, -0.6, 0.35]])
    T = 20
    s = np.linspace(0, 1, T + 1)
    thread = np.stack([6 * s, 0.0 * s], 1)                  # straight thread through gap center
    hit = np.stack([6 * s, 0.0 * s], 1); hit[:, 1] = 0.6    # straight into the top obstacle
    print("m_min/m_max sanity (tight gap corridor≈0.05):")
    for mmax in (0.6, 0.25, 0.1):
        ok_t = certify_trajectory(thread, obs, 0.2, 0.5, H_win=10, stride=2, m_max=mmax, n_theta=360)
        print(f"  m_max={mmax}: straight-thread certified={ok_t}")
    ok_hit = certify_trajectory(hit, obs, 0.2, 0.5, H_win=10, stride=2, m_max=0.6, n_theta=360)
    print(f"  into-obstacle certified={ok_hit} (want False)")
