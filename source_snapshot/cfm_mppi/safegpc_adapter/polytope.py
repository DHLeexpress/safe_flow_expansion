"""Convex-polytope feasible-region module (separable / swappable).

Design (Dohyun, 2026-06-24):
- Finite sensing range => start from a NOMINAL-CONTROL-ORIENTED bounded box (you
  can't see infinitely), then CUT it with one separating hyperplane per nearby
  obstacle. The result is a bounded convex polytope F = {p : A p <= b}.
- Single obstacle => the box is large and one cut dominates => reduces to the
  affine-half-space method (clean fallback).
- This module is intentionally standalone with a stable `Polytope` interface so the
  construction can later be upgraded (IRIS / max-volume inscribed ellipsoid /
  learned) WITHOUT touching the planner that consumes it.

Convention: feasible region F = { p in R^2 : A_i . p <= b_i, all i }.
margin_i(p) = b_i - A_i.p  (>=0 inside);  contains(p) = all margins >= 0.
Smooth barrier H(p) = softmin_i margin_i  (=0 on the boundary, >0 inside).

Theoretical basis / upgrade path (see overnight_run_2026-06-23/POLYTOPE_IDEA.md):
- `build_nominal_polytope` is the Safe-Flight-Corridor tangent-hyperplane family
  (Liu et al., RA-L 2017). Upgrade `build_*` to FIRI (Wang et al., arXiv:2403.02977
  — analytic max-area inscribed ellipse, no SDP, real-time) or IRIS (Deits-Tedrake
  WAFR 2014) for a maximum-volume region; seed via Chebyshev-center LP / MVE.
- `barrier` is the log-sum-exp composed control-barrier function (Molnar & Ames,
  L-CSS 2023, arXiv:2309.06647): {H>=0} is an inner approximation of the polytope,
  reducing to a single affine half-space when there is one face (m=1).
The consumer (planner / proposal) sees only the `Polytope` interface, so the
construction is swappable without touching it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Polytope:
    A: torch.Tensor      # [F, 2] outward face normals
    b: torch.Tensor      # [F]    offsets;  feasible = {p : A p <= b}
    ref: torch.Tensor    # [2]    reference interior point (the robot)

    def margins(self, p: torch.Tensor) -> torch.Tensor:
        """Signed margins to every face. p: [...,2] -> [..., F] (>=0 inside)."""
        return self.b.view(*([1] * (p.ndim - 1)), -1) - p @ self.A.T

    def contains(self, p: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
        return (self.margins(p) >= -tol).all(dim=-1)

    def barrier(self, p: torch.Tensor, beta: float = 8.0) -> torch.Tensor:
        """Smooth soft-min of the face margins (log-sum-exp). =0 on boundary,
        >0 inside; -> min_i margin_i as beta -> inf. Gives the nested convex
        level sets {H >= (1-gamma)^i} for the DCBF schedule."""
        m = self.margins(p)
        return -(1.0 / beta) * torch.logsumexp(-beta * m, dim=-1)

    @property
    def num_faces(self) -> int:
        return int(self.A.shape[0])


def build_nominal_polytope(
    pos: torch.Tensor,                 # [2] robot position
    heading: torch.Tensor,            # [2] nominal-control / velocity direction
    obstacles: torch.Tensor,          # [N,3] (cx,cy,radius) already-safety-inflated
    obstacle_velocities: Optional[torch.Tensor] = None,  # [N,2]
    *,
    sensing_range: float = 6.0,       # finite forward vision
    back_range: float = 2.0,
    half_width: float = 4.0,
    eta: float = 0.0,                  # HOCBF velocity look-ahead (relative closing)
    robot_vel: Optional[torch.Tensor] = None,  # [2] for the eta term
    max_obstacles: int = 8,
) -> Polytope:
    """Nominal-oriented bounded box, cut by per-obstacle separating hyperplanes."""
    device, dtype = pos.device, pos.dtype
    t = heading.to(device=device, dtype=dtype)
    tn = torch.linalg.norm(t)
    t = t / tn if float(tn) > 1e-6 else torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    npp = torch.stack((-t[1], t[0]))  # left-perpendicular

    # --- bounded box faces (finite range), oriented by the nominal direction ---
    A_rows = [t, -t, npp, -npp]
    b_rows = [
        (t @ pos) + sensing_range,
        (-t @ pos) + back_range,
        (npp @ pos) + half_width,
        (-npp @ pos) + half_width,
    ]

    # --- obstacle cuts: one supporting hyperplane per nearby obstacle ---
    if obstacles is not None and obstacles.numel():
        obs = obstacles.to(device=device, dtype=dtype)
        centers, radii = obs[:, :2], obs[:, 2]
        dvec = centers - pos.view(1, 2)
        dist = torch.linalg.norm(dvec, dim=1).clamp_min(1e-9)
        clearance = dist - radii
        # only obstacles within sensing range matter (finite vision)
        order = torch.argsort(clearance)
        for j in order[:max_obstacles].tolist():
            if float(clearance[j]) > sensing_range:
                break
            n = dvec[j] / dist[j]                      # unit, robot -> obstacle (outward)
            p_o = centers[j] - n * radii[j]            # nearest boundary point
            bj = n @ p_o                                # feasible: n.p <= n.p_o
            if eta != 0.0 and robot_vel is not None and obstacle_velocities is not None:
                v_rel = robot_vel.to(device=device, dtype=dtype) - obstacle_velocities[j].to(device=device, dtype=dtype)
                bj = bj - eta * torch.relu(n @ v_rel)  # tighten when closing (HOCBF lift)
            A_rows.append(n)
            b_rows.append(bj)

    A = torch.stack(A_rows, dim=0)
    b = torch.stack([x if torch.is_tensor(x) else torch.tensor(x, device=device, dtype=dtype) for x in b_rows], dim=0)
    return Polytope(A=A, b=b, ref=pos.clone())


def project_into(poly: Polytope, p: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Cheap Dykstra-free cyclic projection of a point onto the polytope (for the
    sampling-mean guidance). Projects onto each violated half-space in turn."""
    q = p.clone()
    for _ in range(iters):
        m = poly.margins(q)                 # [F]
        viol = m < 0
        if not bool(viol.any()):
            break
        j = int(torch.argmin(m))
        a = poly.A[j]
        q = q + (m[j] / (a @ a).clamp_min(1e-9)) * a  # m[j]<0 => step inward to a.q=b_j
    return q
