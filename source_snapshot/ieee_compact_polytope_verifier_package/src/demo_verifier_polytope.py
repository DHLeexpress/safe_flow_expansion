#!/usr/bin/env python3
"""
Verifier polytope demonstrations for Pillar 3.

The mathematical SOCP is

    maximize    sum_i w_i m_i
    subject to  a_i^T(q_t-c) <= beta_t m_i
                r_i ||a_i||_2 <= a_i^T(o_i-c) - m_i
                ||a_i||_2 <= 1
                m_i >= m_min.

For 2-D circular obstacles with one face per obstacle/anchor and positive
weights, this file uses the exact angular interval solution of each independent
SOCP block.  No dense theta sweep is used.

Run:
    python demo_verifier_polytope.py --out demo_outputs --all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import HalfspaceIntersection


@dataclass
class Face:
    """A local polytope face a^T(x-c) <= m, with c=(0,0) in the demos."""
    a: np.ndarray
    m: float
    kind: str
    label: str
    coefficient: float = 1.0
    feasible: bool = True
    interval: Optional[tuple[float, float]] = None


# -----------------------------------------------------------------------------
# Demo scene settings
# -----------------------------------------------------------------------------


def demo_trajectory(H: int = 10) -> np.ndarray:
    t = np.arange(H + 1, dtype=float)
    return np.stack([1.28 * t / H, 0.035 * np.sin(np.pi * t / H)], axis=1)


def demo_real_obstacles() -> list[tuple[float, float, float]]:
    # Two close circular obstacles forming a narrow gap.
    return [(0.78, 0.55, 0.35), (0.78, -0.55, 0.35)]


def artificial_obstacles(R: float, K: int, rho_art: float) -> list[tuple[float, float, float]]:
    if K <= 0:
        return []
    M = R * math.cos(math.pi / K) if K >= 3 else R
    out = []
    for ell in range(K):
        th = 2.0 * math.pi * ell / K
        n = np.array([math.cos(th), math.sin(th)])
        center = (M + rho_art) * n
        out.append((float(center[0]), float(center[1]), float(rho_art)))
    return out


# -----------------------------------------------------------------------------
# Exact interval solution for 2-D circular obstacle SOCP blocks
# -----------------------------------------------------------------------------


def wrap_interval_intersection(current: tuple[float, float], center: float, halfwidth: float) -> Optional[tuple[float, float]]:
    """Intersect an unwrapped interval with a circular interval.

    current is represented on an unwrapped real line.  We shift the new circular
    interval by multiples of 2*pi to overlap current as much as possible.
    """
    lo, hi = current
    mid = 0.5 * (lo + hi)
    center0 = center + 2.0 * math.pi * round((mid - center) / (2.0 * math.pi))
    best = None
    for cc in (center0 - 2.0 * math.pi, center0, center0 + 2.0 * math.pi):
        a, b = cc - halfwidth, cc + halfwidth
        ilo, ihi = max(lo, a), min(hi, b)
        if ilo <= ihi + 1e-12:
            if best is None or (ihi - ilo) > (best[1] - best[0]):
                best = (ilo, ihi)
    return best


def feasible_theta_interval(
    d: np.ndarray,
    radius: float,
    trajectory: np.ndarray,
    beta: np.ndarray,
    *,
    m_min: float = 1e-6,
) -> Optional[tuple[float, float]]:
    """Return the exact feasible theta interval for one circular face.

    Unit normal a(theta) = [cos theta, sin theta].  The face margin is
        m(theta) = a(theta)^T d - radius,
    where d=o-c.  The constraints are
        m(theta) >= m_min,
        a(theta)^T(q_t-c) <= beta_t m(theta), t=1..H.

    The second constraint rearranges to
        a(theta)^T(beta_t d - (q_t-c)) >= beta_t radius,
    which is an angular interval.  The feasible set is the intersection of such
    intervals.
    """
    d = np.asarray(d, dtype=float).reshape(2)
    D = float(np.linalg.norm(d))
    if D <= radius + m_min + 1e-12:
        return None

    phi = math.atan2(d[1], d[0])
    # Margin lower bound: a^T d >= radius + m_min.
    half = math.acos(max(-1.0, min(1.0, (radius + m_min) / D)))
    current = (phi - half, phi + half)

    for p_t, beta_t in zip(trajectory[1:], beta[1:]):
        w = beta_t * d - p_t
        rho = beta_t * radius
        W = float(np.linalg.norm(w))
        if W <= 1e-12:
            if rho > 1e-12:
                return None
            continue
        ratio = rho / W
        if ratio > 1.0 + 1e-12:
            return None
        ratio = max(-1.0, min(1.0, ratio))
        center = math.atan2(w[1], w[0])
        half = math.acos(ratio)
        current = wrap_interval_intersection(current, center, half)
        if current is None:
            return None
    return current


def solve_face_interval(
    d: np.ndarray,
    radius: float,
    trajectory: np.ndarray,
    beta: np.ndarray,
    *,
    coefficient: float,
    kind: str,
    label: str,
    m_min: float = 1e-6,
    signed_unit_diagnostic: bool = False,
) -> Face:
    """Solve one 2-D circular face block.

    For coefficient > 0 this is the exact positive-weight max-margin SOCP block.
    For coefficient < 0 with signed_unit_diagnostic=True, this chooses the
    smallest unit-normal margin in the feasible interval, bounded by m_min.  That
    optional mode is a diagnostic, not the pure positive max-margin theorem.
    """
    d = np.asarray(d, dtype=float).reshape(2)
    interval = feasible_theta_interval(d, radius, trajectory, beta, m_min=m_min)
    if interval is None:
        return Face(np.array([1.0, 0.0]), 0.0, kind, label, coefficient, False, None)

    lo, hi = interval
    phi = math.atan2(d[1], d[0])
    mid = 0.5 * (lo + hi)
    phi = phi + 2.0 * math.pi * round((mid - phi) / (2.0 * math.pi))

    candidates = [lo, hi]
    if lo <= phi <= hi:
        candidates.append(phi)

    def margin_at(theta: float) -> float:
        return float(np.array([math.cos(theta), math.sin(theta)]) @ d - radius)

    if coefficient >= 0.0 or not signed_unit_diagnostic:
        # Maximize margin: choose feasible angle closest to the obstacle center direction.
        theta = min(max(phi, lo), hi)
    else:
        # Diagnostic: minimize the unit-normal margin while respecting the interval.
        theta = min(candidates, key=margin_at)

    a = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    m = float(a @ d - radius)
    feasible = bool(m >= m_min - 1e-9)
    return Face(a, m, kind, label, coefficient, feasible, interval)


def make_variable_faces(
    real_obs: Sequence[tuple[float, float, float]],
    trajectory: np.ndarray,
    beta: np.ndarray,
    *,
    R: float,
    K_artificial: int,
    rho_art: float,
    coeff_real: float = 1.0,
    coeff_artificial: float = 1.0,
    m_min: float = 1e-6,
    signed_unit_diagnostic: bool = False,
) -> tuple[list[Face], list[tuple[float, float, float]]]:
    faces: list[Face] = []
    for j, (ox, oy, rr) in enumerate(real_obs):
        faces.append(
            solve_face_interval(
                np.array([ox, oy]), rr, trajectory, beta,
                coefficient=coeff_real, kind="real", label=f"real{j}",
                m_min=m_min, signed_unit_diagnostic=signed_unit_diagnostic,
            )
        )

    art = artificial_obstacles(R, K_artificial, rho_art)
    for ell, (ox, oy, rr) in enumerate(art):
        faces.append(
            solve_face_interval(
                np.array([ox, oy]), rr, trajectory, beta,
                coefficient=coeff_artificial, kind="artificial", label=f"art{ell}",
                m_min=m_min, signed_unit_diagnostic=signed_unit_diagnostic,
            )
        )
    return faces, art


def make_nominal_radial_faces(
    real_obs: Sequence[tuple[float, float, float]],
    *,
    R: float = 2.0,
    K_base: int = 16,
) -> list[Face]:
    faces: list[Face] = []
    M = R * math.cos(math.pi / K_base)
    for ell in range(K_base):
        th = 2.0 * math.pi * ell / K_base
        faces.append(Face(np.array([math.cos(th), math.sin(th)]), M, "nominal-base", f"base{ell}"))
    for j, (ox, oy, rr) in enumerate(real_obs):
        d = np.array([ox, oy], dtype=float)
        D = float(np.linalg.norm(d))
        faces.append(Face(d / D, D - rr, "nominal-real", f"nominal_real{j}"))
    return faces


# -----------------------------------------------------------------------------
# Geometry, certification, plotting
# -----------------------------------------------------------------------------


def H_grid(faces: Sequence[Face], GX: np.ndarray, GY: np.ndarray) -> np.ndarray:
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    values = []
    for f in faces:
        if f.feasible and f.m > 1e-12:
            values.append((f.m - pts @ f.a) / f.m)
    if not values:
        return np.full(GX.shape, -1.0)
    return np.min(np.stack(values, axis=1), axis=1).reshape(GX.shape)


def check_certificate(faces: Sequence[Face], trajectory: np.ndarray, alpha: np.ndarray, *, include_start: bool = False) -> tuple[bool, float, int]:
    if any((not f.feasible) or f.m <= 1e-12 for f in faces):
        return False, -float("inf"), -1
    worst = float("inf")
    worst_t = -1
    start = 0 if include_start else 1
    for t in range(start, len(trajectory)):
        h = min((f.m - float(f.a @ trajectory[t])) / f.m for f in faces)
        slack = h - float(alpha[t])
        if slack < worst:
            worst = slack
            worst_t = t
    return bool(worst >= -1e-8), float(worst), int(worst_t)


def polygon_area(faces: Sequence[Face]) -> tuple[Optional[float], bool]:
    rows = []
    for f in faces:
        if f.feasible and f.m > 1e-12:
            rows.append(f.a / f.m)
    if len(rows) < 3:
        return None, False
    U = np.vstack(rows)
    halfspaces = np.hstack([U, -np.ones((U.shape[0], 1))])
    try:
        hs = HalfspaceIntersection(halfspaces, np.array([0.0, 0.0]))
        V = hs.intersections
        if V.shape[0] < 3:
            return None, False
        center = V.mean(axis=0)
        V = V[np.argsort(np.arctan2(V[:, 1] - center[1], V[:, 0] - center[0]))]
        x, y = V[:, 0], V[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        return float(area), True
    except Exception:
        return None, False


def face_signature(faces: Sequence[Face]) -> np.ndarray:
    return np.array([[float(f.a[0]), float(f.a[1]), float(f.m)] for f in faces if f.feasible and f.m > 1e-12])


def max_geometry_delta(faces_a: Sequence[Face], faces_b: Sequence[Face]) -> float:
    A = face_signature(faces_a)
    B = face_signature(faces_b)
    if A.shape != B.shape:
        return float("nan")
    return float(np.max(np.abs(A - B))) if A.size else 0.0


def face_line_segment(a: np.ndarray, m: float, xlim: tuple[float, float], ylim: tuple[float, float]) -> Optional[np.ndarray]:
    ax, ay = float(a[0]), float(a[1])
    pts = []
    for x in xlim:
        if abs(ay) > 1e-12:
            y = (m - ax * x) / ay
            if ylim[0] <= y <= ylim[1]:
                pts.append((x, y))
    for y in ylim:
        if abs(ax) > 1e-12:
            x = (m - ay * y) / ax
            if xlim[0] <= x <= xlim[1]:
                pts.append((x, y))
    unique = []
    for p in pts:
        if not any(np.linalg.norm(np.asarray(p) - np.asarray(q)) < 1e-8 for q in unique):
            unique.append(p)
    return np.asarray(unique[:2]) if len(unique) >= 2 else None


def draw_panel(
    ax,
    faces: Sequence[Face],
    real_obs: Sequence[tuple[float, float, float]],
    art_obs: Sequence[tuple[float, float, float]],
    trajectory: np.ndarray,
    alpha: np.ndarray,
    title: str,
    *,
    xlim: tuple[float, float] = (-2.35, 2.35),
    ylim: tuple[float, float] = (-2.35, 2.35),
    nominal_faces: Optional[Sequence[Face]] = None,
) -> None:
    gx = np.linspace(*xlim, 180)
    gy = np.linspace(*ylim, 180)
    GX, GY = np.meshgrid(gx, gy)
    Hh = H_grid(faces, GX, GY)
    level_indices = [1, 2, 4, 7, 10]
    levels = sorted(set([0.0, 1.0001] + [round(float(alpha[i]), 5) for i in level_indices if i < len(alpha)]))
    try:
        ax.contourf(GX, GY, Hh, levels=levels, cmap="Greens", alpha=0.42, zorder=1)
        ax.contour(GX, GY, Hh, levels=[0.0], colors="#006d2c", linewidths=1.5, zorder=3)
        ax.contour(GX, GY, Hh, levels=[v for v in levels if 0.0 < v < 1.0], colors="#238b45", linewidths=0.35, alpha=0.75, zorder=2)
    except Exception:
        pass

    if nominal_faces is not None:
        Hn = H_grid(nominal_faces, GX, GY)
        try:
            ax.contour(GX, GY, Hn, levels=[0.0], colors="#08519c", linewidths=0.8, linestyles="--", alpha=0.8, zorder=3)
        except Exception:
            pass

    for f in faces:
        if not f.feasible or f.m <= 1e-12:
            continue
        seg = face_line_segment(f.a, f.m, xlim, ylim)
        if seg is None:
            continue
        if f.kind == "real":
            ax.plot(seg[:, 0], seg[:, 1], "-", color="#006d2c", lw=1.0, alpha=0.95, zorder=4)
        elif f.kind == "artificial":
            ax.plot(seg[:, 0], seg[:, 1], "--", color="0.30", lw=0.5, alpha=0.65, zorder=4)

    for ox, oy, rr in art_obs:
        ax.add_patch(Circle((ox, oy), rr, facecolor="none", edgecolor="0.25", lw=0.7, ls="--", alpha=0.8, zorder=5))
        ax.plot([ox], [oy], marker=".", color="0.25", ms=2.6, zorder=6)

    for ox, oy, rr in real_obs:
        ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.9, alpha=0.75, zorder=7))

    ax.plot(trajectory[:, 0], trajectory[:, 1], "k.-", lw=1.35, ms=3.8, zorder=9)
    ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], s=30, c="#00a000", edgecolor="k", zorder=10)
    ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", fontsize=7.0,
            bbox=dict(boxstyle="round", fc="white", alpha=0.88, lw=0.25))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# -----------------------------------------------------------------------------
# Demo generation
# -----------------------------------------------------------------------------


def benchmark_make_faces(n_repeat: int, *args, **kwargs) -> tuple[list[Face], list[tuple[float, float, float]], float]:
    for _ in range(100):
        make_variable_faces(*args, **kwargs)
    t0 = time.perf_counter()
    faces: list[Face] = []
    art: list[tuple[float, float, float]] = []
    for _ in range(n_repeat):
        faces, art = make_variable_faces(*args, **kwargs)
    ms = 1000.0 * (time.perf_counter() - t0) / n_repeat
    return faces, art, ms


def demo_nominal_vs_variable(out_dir: Path) -> dict:
    R = 2.0
    H = 10
    gamma = 0.5
    K = 16
    rho_art = 0.12
    m_min = 1e-6
    trajectory = demo_trajectory(H)
    alpha = (1.0 - gamma) ** np.arange(H + 1, dtype=float)
    beta = 1.0 - alpha
    real_obs = demo_real_obstacles()
    nominal = make_nominal_radial_faces(real_obs, R=R, K_base=16)
    variable, art, ms = benchmark_make_faces(
        200, real_obs, trajectory, beta,
        R=R, K_artificial=K, rho_art=rho_art,
        coeff_real=1.0, coeff_artificial=1.0, m_min=m_min,
    )
    nom_ok, nom_slack, nom_t = check_certificate(nominal, trajectory, alpha)
    var_ok, var_slack, var_t = check_certificate(variable, trajectory, alpha)
    area, bounded = polygon_area(variable)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), squeeze=False)
    draw_panel(
        axes[0, 0], nominal, real_obs, [], trajectory, alpha,
        f"Nominal radial\ncert={nom_ok}\nslack={nom_slack:.3f}",
        xlim=(-0.30, 2.15), ylim=(-1.18, 1.18),
    )
    draw_panel(
        axes[0, 1], variable, real_obs, art, trajectory, alpha,
        f"Variable tangent SOCP\nK={K}, γ={gamma}\ncert={var_ok}, time={ms:.3f} ms",
        xlim=(-0.30, 2.15), ylim=(-1.18, 1.18), nominal_faces=nominal,
    )
    fig.suptitle("Canonical narrow-gap demo: nominal radial fails, variable tangent certifies", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    path = out_dir / "socp_narrow_gap_nominal_vs_variable.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "file": str(path), "gamma": gamma, "K": K, "nominal_certified": nom_ok,
        "variable_certified": var_ok, "variable_time_ms": ms, "variable_area": area,
        "bounded": bounded, "variable_slack": var_slack,
    }


def demo_weight_pair_invariance(out_dir: Path) -> list[dict]:
    R = 2.0
    H = 10
    gamma = 0.5
    K = 16
    rho_art = 0.12
    m_min = 1e-6
    trajectory = demo_trajectory(H)
    alpha = (1.0 - gamma) ** np.arange(H + 1, dtype=float)
    beta = 1.0 - alpha
    real_obs = demo_real_obstacles()
    pairs = [(0.01, 1.0), (0.1, 1.0), (1.0, 1.0), (10.0, 1.0), (100.0, 1.0), (1.0, 0.01), (1.0, 0.1), (1.0, 10.0), (1.0, 100.0)]
    baseline, _, _ = benchmark_make_faces(
        500, real_obs, trajectory, beta,
        R=R, K_artificial=K, rho_art=rho_art,
        coeff_real=1.0, coeff_artificial=1.0, m_min=m_min,
    )
    records = []
    faces_by_pair = {}
    for wr, wa in pairs:
        faces, art, ms = benchmark_make_faces(
            200, real_obs, trajectory, beta,
            R=R, K_artificial=K, rho_art=rho_art,
            coeff_real=wr, coeff_artificial=wa, m_min=m_min,
        )
        ok, slack, worst_t = check_certificate(faces, trajectory, alpha)
        area, bounded = polygon_area(faces)
        real_m = [f.m for f in faces if f.kind == "real"]
        art_m = [f.m for f in faces if f.kind == "artificial"]
        delta = max_geometry_delta(faces, baseline)
        records.append({
            "w_real": wr, "w_artificial": wa, "certified": ok,
            "area": area, "bounded": bounded,
            "real_margin_mean": float(np.mean(real_m)),
            "artificial_margin_mean": float(np.mean(art_m)),
            "geometry_delta_vs_1_1": delta,
            "mean_time_ms": ms,
            "min_slack": slack, "worst_t": worst_t,
        })
        faces_by_pair[(wr, wa)] = (faces, art)

    fig, axes = plt.subplots(3, 3, figsize=(11.8, 11.2), squeeze=False)
    nominal = make_nominal_radial_faces(real_obs, R=R, K_base=16)
    for ax, rec in zip(axes.ravel(), records):
        wr, wa = rec["w_real"], rec["w_artificial"]
        faces, art = faces_by_pair[(wr, wa)]
        title = (f"w_real={wr:g}, w_art={wa:g}\n"
                 f"cert={rec['certified']}, area={rec['area']:.3f}\n"
                 f"Δgeom={rec['geometry_delta_vs_1_1']:.1e}, time={rec['mean_time_ms']:.3f} ms")
        draw_panel(ax, faces, real_obs, art, trajectory, alpha, title,
                   xlim=(-2.35, 2.35), ylim=(-2.35, 2.35), nominal_faces=nominal)
    fig.suptitle("Positive weight-pair invariance: shape is unchanged for independent max-margin faces", fontsize=12)
    fig.tight_layout(rect=[0, 0.012, 1, 0.96])
    png = out_dir / "weight_pair_invariance.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)

    write_table(out_dir / "weight_pair_invariance.csv", records)
    (out_dir / "weight_pair_invariance.json").write_text(json.dumps(records, indent=2))
    return records


def demo_k_gamma_positive_weights(out_dir: Path) -> list[dict]:
    R = 2.0
    H = 10
    trajectory = demo_trajectory(H)
    real_obs = demo_real_obstacles()
    rho_art = 0.12
    m_min = 1e-6
    gammas = [0.3, 0.5, 0.8]
    rows_spec = [
        (4, 0.01, 1.0),
        (4, 100.0, 1.0),
        (8, 0.01, 1.0),
        (8, 100.0, 1.0),
        (16, 0.01, 1.0),
        (16, 100.0, 1.0),
    ]
    records = []
    faces_cache = {}
    nominal = make_nominal_radial_faces(real_obs, R=R, K_base=16)

    for K, wr, wa in rows_spec:
        for gamma in gammas:
            alpha = (1.0 - gamma) ** np.arange(H + 1, dtype=float)
            beta = 1.0 - alpha
            nrep = 200 if K >= 16 else 300
            faces, art, ms = benchmark_make_faces(
                nrep, real_obs, trajectory, beta,
                R=R, K_artificial=K, rho_art=rho_art,
                coeff_real=wr, coeff_artificial=wa, m_min=m_min,
            )
            ok, slack, worst_t = check_certificate(faces, trajectory, alpha)
            area, bounded = polygon_area(faces)
            real_m = [f.m for f in faces if f.kind == "real"]
            art_m = [f.m for f in faces if f.kind == "artificial"]
            record = {
                "K_artificial": K, "w_real": wr, "w_artificial": wa,
                "gamma": gamma, "decision_faces": len(faces), "certified": ok,
                "min_nonstart_levelset_margin": slack, "worst_t": worst_t,
                "mean_solve_time_ms": ms, "area": area, "bounded": bounded,
                "real_margin_mean": float(np.mean(real_m)),
                "artificial_margin_mean": float(np.mean(art_m)),
            }
            records.append(record)
            faces_cache[(K, wr, wa, gamma)] = (faces, art)

    # Compare low/high positive real weight for fixed K,gamma.
    for K in [4, 8, 16]:
        for gamma in gammas:
            low = faces_cache[(K, 0.01, 1.0, gamma)][0]
            high = faces_cache[(K, 100.0, 1.0, gamma)][0]
            delta = max_geometry_delta(low, high)
            for rec in records:
                if rec["K_artificial"] == K and abs(rec["gamma"] - gamma) < 1e-12:
                    rec["geometry_delta_low_vs_high_wreal"] = delta

    fig, axes = plt.subplots(6, 3, figsize=(12.0, 22.2), squeeze=False)
    for ri, (K, wr, wa) in enumerate(rows_spec):
        for ci, gamma in enumerate(gammas):
            alpha = (1.0 - gamma) ** np.arange(H + 1, dtype=float)
            faces, art = faces_cache[(K, wr, wa, gamma)]
            rec = next(r for r in records if r["K_artificial"] == K and r["w_real"] == wr and r["w_artificial"] == wa and abs(r["gamma"] - gamma) < 1e-12)
            title = (f"K={K}, w_real={wr:g}, w_art={wa:g}\n"
                     f"γ={gamma:g}, cert={rec['certified']}, time={rec['mean_solve_time_ms']:.3f} ms\n"
                     f"m_real={rec['real_margin_mean']:.3f}, m_art={rec['artificial_margin_mean']:.3f}\n"
                     f"Δw={rec['geometry_delta_low_vs_high_wreal']:.1e}")
            draw_panel(axes[ri, ci], faces, real_obs, art, trajectory, alpha, title,
                       xlim=(-2.35, 2.35), ylim=(-2.35, 2.35), nominal_faces=nominal)
            if ri == 0:
                axes[ri, ci].set_title(f"gamma = {gamma:g}", fontsize=11)
        axes[ri, 0].set_ylabel(f"K={K}\nw_real={wr:g}", fontsize=10)
    fig.suptitle("6×3 grid: K and gamma vary; positive weight ratio is shown but does not change geometry", fontsize=13)
    fig.tight_layout(rect=[0, 0.012, 1, 0.982])
    png = out_dir / "k_gamma_positive_weights_6x3.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)

    write_table(out_dir / "k_gamma_positive_weights_6x3.csv", records)
    (out_dir / "k_gamma_positive_weights_6x3.json").write_text(json.dumps(records, indent=2))
    return records


def demo_tube_biased_diagnostic(out_dir: Path) -> list[dict]:
    """Optional signed/unit-normal visualization.

    This is not the pure positive max-margin SOCP theorem.  It is included only
    because it demonstrates what it means to pull real-obstacle faces inward while
    keeping artificial faces outward.
    """
    R = 2.0
    H = 10
    trajectory = demo_trajectory(H)
    real_obs = demo_real_obstacles()
    rho_art = 0.12
    m_min = 0.10
    gammas = [0.3, 0.5, 0.8]
    rows_spec = [
        (4, "max-margin", 1.0, 1.0, False, 1e-6),
        (4, "tube-biased diagnostic", -1.0, 1.0, True, m_min),
        (8, "max-margin", 1.0, 1.0, False, 1e-6),
        (8, "tube-biased diagnostic", -1.0, 1.0, True, m_min),
        (16, "max-margin", 1.0, 1.0, False, 1e-6),
        (16, "tube-biased diagnostic", -1.0, 1.0, True, m_min),
    ]
    records = []
    faces_cache = {}
    nominal = make_nominal_radial_faces(real_obs, R=R, K_base=16)
    for K, mode, cr, ca, signed, mm in rows_spec:
        for gamma in gammas:
            alpha = (1.0 - gamma) ** np.arange(H + 1, dtype=float)
            beta = 1.0 - alpha
            nrep = 150 if K >= 16 else 250
            faces, art, ms = benchmark_make_faces(
                nrep, real_obs, trajectory, beta,
                R=R, K_artificial=K, rho_art=rho_art,
                coeff_real=cr, coeff_artificial=ca, m_min=mm,
                signed_unit_diagnostic=signed,
            )
            ok, slack, worst_t = check_certificate(faces, trajectory, alpha)
            area, bounded = polygon_area(faces)
            real_m = [f.m for f in faces if f.kind == "real"]
            art_m = [f.m for f in faces if f.kind == "artificial"]
            rec = {
                "K_artificial": K, "mode": mode, "coefficient_real": cr,
                "coefficient_artificial": ca, "gamma": gamma,
                "signed_unit_diagnostic": signed, "m_min_used": mm,
                "decision_faces": len(faces), "certified": ok,
                "min_nonstart_levelset_margin": slack, "worst_t": worst_t,
                "mean_solve_time_ms": ms, "area": area, "bounded": bounded,
                "real_margin_mean": float(np.mean(real_m)),
                "artificial_margin_mean": float(np.mean(art_m)),
            }
            records.append(rec)
            faces_cache[(K, mode, gamma)] = (faces, art)

    fig, axes = plt.subplots(6, 3, figsize=(12.0, 22.2), squeeze=False)
    for ri, (K, mode, cr, ca, signed, mm) in enumerate(rows_spec):
        for ci, gamma in enumerate(gammas):
            alpha = (1.0 - gamma) ** np.arange(H + 1, dtype=float)
            faces, art = faces_cache[(K, mode, gamma)]
            rec = next(r for r in records if r["K_artificial"] == K and r["mode"] == mode and abs(r["gamma"] - gamma) < 1e-12)
            title = (f"K={K}, {mode}\n"
                     f"c_real={cr:g}, c_art={ca:g}, γ={gamma:g}\n"
                     f"cert={rec['certified']}, time={rec['mean_solve_time_ms']:.3f} ms\n"
                     f"m_real={rec['real_margin_mean']:.3f}, area={rec['area']:.3f}")
            draw_panel(axes[ri, ci], faces, real_obs, art, trajectory, alpha, title,
                       xlim=(-2.35, 2.35), ylim=(-2.35, 2.35), nominal_faces=nominal)
            if ri == 0:
                axes[ri, ci].set_title(f"gamma = {gamma:g}", fontsize=11)
        axes[ri, 0].set_ylabel(f"K={K}\n{mode}", fontsize=9)
    fig.suptitle("Optional diagnostic: signed real-face coefficient pulls faces inward; not pure positive max-margin", fontsize=13)
    fig.tight_layout(rect=[0, 0.012, 1, 0.982])
    png = out_dir / "tube_biased_diagnostic_6x3.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)

    write_table(out_dir / "tube_biased_diagnostic_6x3.csv", records)
    (out_dir / "tube_biased_diagnostic_6x3.json").write_text(json.dumps(records, indent=2))
    return records


def write_table(path: Path, records: Sequence[dict]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def make_bundle(root: Path) -> Path:
    bundle = root / "verifier_polytope_demo_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob("*"):
            if p.is_file() and p.name != bundle.name:
                z.write(p, p.relative_to(root))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="demo_outputs")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--no-diagnostic", action="store_true", help="Skip optional signed diagnostic")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"outputs": {}, "settings": {
        "R": 2.0, "H": 10, "trajectory": demo_trajectory(10).tolist(),
        "real_obstacles": demo_real_obstacles(), "rho_art": 0.12,
        "gammas": [0.3, 0.5, 0.8], "K_values": [4, 8, 16],
    }}
    manifest["outputs"]["nominal_vs_variable"] = demo_nominal_vs_variable(out_dir)
    manifest["outputs"]["weight_pair_invariance"] = demo_weight_pair_invariance(out_dir)
    manifest["outputs"]["k_gamma_positive_weights"] = demo_k_gamma_positive_weights(out_dir)
    if not args.no_diagnostic:
        manifest["outputs"]["tube_biased_diagnostic"] = demo_tube_biased_diagnostic(out_dir)
    write_manifest(out_dir, manifest)
    bundle = make_bundle(root)
    print(json.dumps(manifest, indent=2))
    print(f"Bundle: {bundle}")


if __name__ == "__main__":
    main()
