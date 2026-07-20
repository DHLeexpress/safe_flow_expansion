from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .barrier import (
    affine_barrier_h,
    affine_barrier_h_ho,
    affine_barrier_h_ho_all,
    barrier_clearance,
)
from .polytope_v2 import build_polytope_v2


@dataclass
class SafeMPPIConfig:
    horizon: int = 20
    dt: float = 0.1
    num_samples: int = 128
    gamma: float = 0.5
    temperature: float = 1.0
    noise_sigma: float | Tuple[float, float] = 0.6
    u_min: Tuple[float, float] = (-2.0, -2.0)
    u_max: Tuple[float, float] = (2.0, 2.0)
    # --- BIMODAL polytope proposal: each control drawn from a mixture of two Gaussians over ALL steps ---
    #   Mode A ~ N(warm, Sigma_iso)          (goal-ward, the warm-start)
    #   Mode B ~ N(warm + B+ d_centroid, Sigma_aniso)   (opening-ward, toward the EXACT polytope centroid)
    #   mixture weight p = clip(centroid_gain*trapped, 0, 1); trapped=(R-size)/(size+eps) ("1/volume"-like, 0 if open)
    centroid_gain: float = 0.0             # Mode-B mixture weight gain (p = clip(gain*trapped,0,1)); 0=off
    centroid_smooth: float = 0.5           # temporal low-pass on p across plan steps (smoothness; 0=off)
    centroid_eps: float = 0.15             # numerical-stability floor in trapped=(R-size)/(size+eps)
    sigma_volume_gain: float = 0.0         # widen sampling sigma when the polytope is small (trapped); 0=off
    sigma_aniso: float = 2.0               # Mode-B ellipsoid anisotropy: wide (xPARALLEL opening) / narrow (xPERP)
    sigma_max_mult: float = 3.0            # cap on the sigma blow-up (numerical stability)
    random_backup_frac: float = 0.0        # Mode-C: ALWAYS-ON random-backup fraction p_c (evenly-spread 360deg escape
                                           #   samples every frame; fires even when p_b=0/degenerate). 0.03 ~= 15@512.
    # --- importance-sampling experiment (mode {1,4}) ---
    urgency_size_diff: bool = False        # mode 4: rho_k = max(0, size_{k-1}-size_k) (shrink rate) instead of
                                           #   mode 1: rho_k = (R-size_k)/(size_k+eps). p_b = clip(centroid_gain*rho,0,1).
    polytope_area_sampling: bool = False   # Mode B = random rays INSIDE the (retreated) polytope (span its whole area,
                                           #   importance-sample the safe set) instead of pointing only at the centroid.
    urgency_floor: float = 0.0             # lower clip on p_b: p_b = clip(c_g*rho, urgency_floor, 1). >0 keeps Mode B
                                           #   ALWAYS slightly active so it escapes local minima (where mode-4 rho->0).
    temp_trapped_gain: float = 0.0         # softmax temp softens when the polytope is SMALL: temp_eff =
                                           #   temp*(1 + temp_trapped_gain*(R-size)/(size+eps)) — lets accepted ESCAPE
                                           #   samples (Mode B) drive the executed action OUT of a local-minimum pocket.
    use_polytope_barrier: bool = False     # reject on the nominal polytope level sets: H_P(x_{i+1}) >= (1-g) H_P(x_i)
    polytope_nbase: int = 16               # K base faces of the robot-centered sensing disk
    predict_gain: float = 0.0              # velocity-predictive face retreat (kappa) for the polytope (req 1)
    # --- MPPI nominal: WE DON'T refine a goal-seeking nominal (Mizuta does); ours = polytope mean + cost ---
    use_goal_nominal: bool = True          # False => base nominal = 0 (no goal-seeking); mean comes from the polytope,
                                           #          the goal is handled by the progress/terminal cost (MPPI spirit)
    warm_start: bool = False               # nominal = previous MPPI-averaged solution (shifted); samples explore it
    nominal_speed: float = 0.0             # cap the goal-seeking nominal SPEED (0 = saturate at u_max, the old way)
    safety_margin: float = 0.5
    running_goal_weight: float = 0.25
    terminal_goal_weight: float = 80.0
    control_weight: float = 0.03
    smooth_weight: float = 0.12
    soft_clearance_weight: float = 25.0
    progress_weight: float = 2.0
    # Optional anti-retreat shaping for expert-data generation.  A positive
    # value softly penalizes each predicted increase in goal distance as
    # weight * expm1(delta_distance / scale).  It is deliberately disabled by
    # default so existing planner configurations and results are unchanged.
    goal_retreat_exp_weight: float = 0.0
    goal_retreat_exp_scale: float = 0.05
    goal_retreat_exp_cap: float = 6.0
    heading_weight: float = 0.4
    check_first_control_only: bool = False
    dynamics_type: str = "doubleintegrator"
    debug_max_rollouts: int = 80
    # --- Guided Safe MPPI (overnight contribution) ---
    use_ho_barrier: bool = False          # affine higher-order DCBF in (p, v)
    barrier_topk: int = 0                  # cap enforced obstacles to k nearest (0 = no cap)
    barrier_activation_radius: float = 3.5  # enforce obstacles within this current clearance (0 = all)
    eta: float = 0.6                       # velocity look-ahead (braking horizon, s)
    use_guidance: bool = False            # PSF projection of sampling mean into feasible half-space
    guidance_relax: float = 1.0           # in (0,1]: fraction of the deficit to close (1 = full projection)
    guidance_horizon: int = 12            # only project the first k nominal controls (we apply step 0 & replan)
    use_aniso_cov: bool = False           # anisotropic covariance (tangent-wide, normal-narrow)
    aniso_normal_scale: float = 0.5       # noise scale along obstacle normal
    aniso_tangent_scale: float = 1.7      # noise scale along obstacle tangent (multi-modality)
    barrier_extra_margin: float = 0.0     # buffer added to barrier radius beyond collision margin (CS-MPPI tightening)
    adaptive_gamma: bool = False          # per-step gamma schedule from distance/closing-velocity
    gamma_min: float = 0.1
    gamma_max: float = 1.0
    filter_output: bool = False           # project final applied control through one-step PSF (hard per-step guarantee)
    filter_iters: int = 3
    proposal_gaussian_mix: int = 96       # # Gaussian-around-damped-nominal samples mixed into a learned proposal (velocity regulation + coverage)
    use_sets_backup: bool = False
    sets_num_modes: int = 3
    sets_branch_scale: float = 0.85
    sets_include_cbf_backup: bool = True
    sets_cbf_push: float = 1.25
    sets_reverse_speed: float = 0.75
    sets_turn_rate: float = 1.4


class SafeMPPIAdapter:
    """
    Minimal PyTorch port of local safeGPC MPPI sample rejection.

    Source parity:
    safeGPC `utils/alg_base.py` rejects samples when
    `h_new < (1 - gamma) * h_old`; `tasks/doubleIntegrator.py` supplies
    `huniversal_proj_torch`, which reduces to the affine circle projection
    implemented in `affine_barrier_h`.
    """

    def __init__(self, **kwargs):
        self.config = SafeMPPIConfig(**kwargs)
        self.u_min = torch.tensor(self.config.u_min, dtype=torch.float32)
        self.u_max = torch.tensor(self.config.u_max, dtype=torch.float32)
        self._u_prev = None                # warm-start: previous MPPI-averaged control sequence [H,2]
        self._p_prev = None                # temporal low-pass state for the Mode-B mixture weight p
        self._size_prev = None             # previous polytope size (for the mode-4 shrink-rate urgency)

    def _sigma(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = self.config.noise_sigma
        if isinstance(sigma, (tuple, list)):
            value = torch.tensor(sigma, dtype=dtype, device=device)
        else:
            value = torch.full((2,), float(sigma), dtype=dtype, device=device)
        return value

    def _anisotropic(self, noise: torch.Tensor, normal_axis: torch.Tensor) -> torch.Tensor:
        """Reshape isotropic noise into an anisotropic cloud: narrow along the
        obstacle normal, wide along the tangent (THEORY.md Fix 5 / covariance
        steering) to spread samples into the left/right homotopy classes."""
        H = noise.shape[1]
        n = normal_axis.view(1, H, 2)
        tang = torch.stack((-normal_axis[:, 1], normal_axis[:, 0]), dim=1).view(1, H, 2)
        cn = (noise * n).sum(dim=-1, keepdim=True)
        ct = (noise * tang).sum(dim=-1, keepdim=True)
        shaped = self.config.aniso_normal_scale * cn * n + self.config.aniso_tangent_scale * ct * tang
        # steps with no defined normal axis (e.g. past guidance horizon / open space)
        # keep isotropic noise instead of collapsing to zero.
        valid = (torch.linalg.norm(normal_axis, dim=1) > 1e-6).view(1, H, 1)
        return torch.where(valid, shaped, noise)

    @staticmethod
    def _polygon_centroid(A_np, b_np, interior):
        """EXACT area-centroid of the convex polygon {A x <= b} via halfspace intersection + shoelace centroid.
        Falls back to the interior point (robot) on degeneracy."""
        interior = np.asarray(interior, float).reshape(2)
        try:
            from scipy.spatial import HalfspaceIntersection
            hs = HalfspaceIntersection(np.hstack([A_np, -b_np[:, None]]).astype(float), interior)  # a.x - b <= 0
            V = hs.intersections
            if V.shape[0] < 3:
                return interior
            m = V.mean(0); V = V[np.argsort(np.arctan2(V[:, 1] - m[1], V[:, 0] - m[0]))]            # order CCW
            x, y = V[:, 0], V[:, 1]; xs, ys = np.roll(x, -1), np.roll(y, -1)
            cr = x * ys - xs * y; area = 0.5 * cr.sum()
            if abs(area) < 1e-9:
                return interior
            return np.array([((x + xs) * cr).sum() / (6 * area), ((y + ys) * cr).sum() / (6 * area)])
        except Exception:
            return interior

    def _polytope_proposal(self, state, safe_obstacles, obstacle_velocities):
        """Build the NOMINAL robot-centered polytope at x0. Returns (A,b,c, margins, d_centroid, size, C):
          A,b,c                : faces a_k.x <= b_k, robot center c (level-set rejection H_P).
          margins[F]=b-A@c >0  : face clearances; size = min margin (trapped indicator).
          C[2]                 : the EXACT geometric (area) centroid of the polytope.
          d_centroid[2]        : unit direction robot -> C (free-space / opening direction)."""
        c_np = state[0, :2].detach().cpu().numpy()
        obs_np = safe_obstacles[:, :3].detach().cpu().numpy() if safe_obstacles.numel() else np.zeros((0, 3))
        vrob = state[0, 2:4].detach().cpu().numpy() if state.shape[1] >= 4 else None
        vobs = obstacle_velocities.detach().cpu().numpy() if obstacle_velocities is not None else None
        poly, _ = build_polytope_v2(
            c_np, obs_np, sensing_range=float(self.config.barrier_activation_radius) or 3.0,
            n_base=int(self.config.polytope_nbase), margin=0.0, obstacle_velocities=vobs, robot_velocity=vrob,
            predict_gain=float(self.config.predict_gain), predict_tau=float(self.config.horizon) * float(self.config.dt))
        A = poly.A.to(device=state.device, dtype=state.dtype)
        b = poly.b.to(device=state.device, dtype=state.dtype)
        c = poly.ref.to(device=state.device, dtype=state.dtype)
        margins = (b - A @ c).clamp_min(1e-3)
        C_np = self._polygon_centroid(poly.A.numpy(), poly.b.numpy(), c_np)                 # exact area-centroid
        C = torch.tensor(C_np, device=state.device, dtype=state.dtype)
        d = C - c
        dn = torch.linalg.norm(d)
        d_centroid = d / dn if float(dn) > 1e-6 else torch.zeros(2, device=state.device, dtype=state.dtype)
        return A, b, c, margins, d_centroid, float(margins.min()), C

    @staticmethod
    def _polytope_H(x_pts, A, b, margins):
        """Robot-centered polytope barrier H_P(x) = min_k (b_k - a_k.x)/margin_k. =1 at robot, 0 on a face."""
        return ((b.unsqueeze(0) - x_pts @ A.t()) / margins.unsqueeze(0)).min(dim=1).values

    def _polytope_ray_controls(self, A, b, c, n, Bpos, u_max, gen, device, dtype):
        """IMPORTANCE SAMPLING (Mode-4): n random rays from the robot center c into the polytope {A x <= b}. Each
        returns a CONSTANT control [n,2] aiming at a random interior point (uniform-in-area along the ray), magnitude
        set to ~reach it over the horizon => the Mode-B rollouts span the WHOLE (velocity-retreated) polytope and land
        inside the safe set (more accepted). If the polytope is a half-disk, the rays span its actual radius+theta."""
        two_pi = 2.0 * float(np.pi)
        theta = torch.rand(n, generator=gen, device=device, dtype=dtype) * two_pi
        dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)              # [n,2] random unit directions
        margins = (b - A @ c).clamp_min(1e-3)                                        # [F] face clearances (>0)
        adir = dirs @ A.t()                                                          # [n,F] = a_k . dir
        r_max = torch.where(adir > 1e-6, margins.unsqueeze(0) / adir.clamp_min(1e-6),
                            torch.full_like(adir, 1e6)).min(dim=1).values            # [n] ray->boundary distance
        r = torch.sqrt(torch.rand(n, generator=gen, device=device, dtype=dtype)) * r_max     # uniform-in-area radius
        Hdt = float(self.config.horizon) * float(self.config.dt)
        reach = (0.5 * Hdt * Hdt) if self.config.dynamics_type == "doubleintegrator" else Hdt   # per-unit-u reach over H
        umag = (r / max(reach, 1e-6)).clamp(max=float(u_max.max()))                  # [n] magnitude to reach the target
        cdir = (torch.linalg.pinv(Bpos) @ dirs.t()).t()                             # B+ dir -> control direction
        cdir = cdir / torch.linalg.norm(cdir, dim=1, keepdim=True).clamp_min(1e-9)
        return umag.unsqueeze(1) * cdir                                              # [n,2] per-sample constant control

    def _step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        dt = self.config.dt
        if self.config.dynamics_type == "doubleintegrator":
            new_state = state.clone()
            new_state[:, 0] = state[:, 0] + dt * state[:, 2] + 0.5 * dt * dt * control[:, 0]
            new_state[:, 1] = state[:, 1] + dt * state[:, 3] + 0.5 * dt * dt * control[:, 1]
            new_state[:, 2] = state[:, 2] + dt * control[:, 0]
            new_state[:, 3] = state[:, 3] + dt * control[:, 1]
            return new_state
        if self.config.dynamics_type == "unicycle":
            new_state = state.clone()
            new_state[:, 0] = state[:, 0] + dt * control[:, 0] * torch.cos(state[:, 2])
            new_state[:, 1] = state[:, 1] + dt * control[:, 0] * torch.sin(state[:, 2])
            new_state[:, 2] = torch.atan2(
                torch.sin(state[:, 2] + dt * control[:, 1]),
                torch.cos(state[:, 2] + dt * control[:, 1]),
            )
            return new_state
        new_state = state.clone()
        new_state[:, :2] = state[:, :2] + dt * control
        return new_state

    def _linear_matrices(self, state: torch.Tensor, control: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = float(self.config.dt)
        state_dim = int(state.numel())
        device = state.device
        dtype = state.dtype
        if self.config.dynamics_type == "doubleintegrator" and state_dim >= 4:
            A = torch.eye(state_dim, dtype=dtype, device=device)
            B = torch.zeros(state_dim, 2, dtype=dtype, device=device)
            A[0, 2] = dt
            A[1, 3] = dt
            B[0, 0] = 0.5 * dt * dt
            B[1, 1] = 0.5 * dt * dt
            B[2, 0] = dt
            B[3, 1] = dt
            return A, B
        if self.config.dynamics_type == "unicycle" and state_dim >= 3:
            theta = state[2]
            v = control[0]
            c = torch.cos(theta)
            s = torch.sin(theta)
            A = torch.eye(state_dim, dtype=dtype, device=device)
            B = torch.zeros(state_dim, 2, dtype=dtype, device=device)
            A[0, 2] = -dt * v * s
            A[1, 2] = dt * v * c
            B[0, 0] = dt * c
            B[1, 0] = dt * s
            B[2, 1] = dt
            return A, B
        A = torch.eye(state_dim, dtype=dtype, device=device)
        B = torch.zeros(state_dim, 2, dtype=dtype, device=device)
        B[:2, :2] = dt * torch.eye(2, dtype=dtype, device=device)
        return A, B

    def _nominal_control(self, state: torch.Tensor, goal: torch.Tensor, horizon: int, u_min: torch.Tensor, u_max: torch.Tensor) -> torch.Tensor:
        to_goal = goal[:2].to(device=state.device, dtype=state.dtype) - state[0, :2]
        if self.config.dynamics_type == "unicycle":
            distance = torch.linalg.norm(to_goal).clamp_min(1e-6)
            desired_heading = torch.atan2(to_goal[1], to_goal[0])
            heading_error = torch.atan2(
                torch.sin(desired_heading - state[0, 2]),
                torch.cos(desired_heading - state[0, 2]),
            )
            v = torch.clamp(distance / max(horizon * self.config.dt, 1e-6), min=0.0, max=float(u_max[0]))
            omega = torch.clamp(1.5 * heading_error, min=float(u_min[1]), max=float(u_max[1]))
            return torch.stack([v, omega]).to(device=state.device, dtype=state.dtype)
        if self.config.dynamics_type == "doubleintegrator" and state.shape[1] >= 4:
            vel_err = -state[0, 2:4]
            nominal = 0.45 * to_goal + 0.8 * vel_err
            return torch.clamp(nominal, u_min, u_max)
        nominal = to_goal / max(horizon * self.config.dt, 1e-6)
        if self.config.nominal_speed > 0.0:                  # cap the cruise speed so the nominal does NOT saturate
            sp = torch.linalg.norm(nominal).clamp_min(1e-9)
            nominal = nominal / sp * torch.clamp(sp, max=float(self.config.nominal_speed))
        return torch.clamp(nominal, u_min, u_max)

    def _nominal_sequence(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        horizon: int,
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = state[0:1].clone()
        controls = []
        states = [x[0].clone()]
        for t in range(horizon):
            remaining = max(horizon - t, 1)
            u = self._nominal_control(x, goal, remaining, u_min, u_max)
            controls.append(u)
            x = self._step(x, u.view(1, 2))
            states.append(x[0].clone())
        return torch.stack(controls, dim=0), torch.stack(states, dim=0)

    def _eta_eff(self) -> float:
        return float(self.config.eta) if self.config.use_ho_barrier else 0.0

    def _barrier_h(self, x0, x, obstacles, obstacle_velocities):
        """Dispatch: higher-order/relative-velocity barrier when enabled, else the
        original position-only affine barrier (exact backward compatibility)."""
        if self.config.use_ho_barrier:
            return affine_barrier_h_ho(x0, x, obstacles, obstacle_velocities, self._eta_eff())
        return affine_barrier_h(x0, x, obstacles)

    def _guide_nominal(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        safe_obstacles: torch.Tensor,
        obstacle_velocities: Optional[torch.Tensor],
        gamma: float,
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predictive-safety-filter guidance (THEORY.md Fix 3).

        Greedily projects each nominal control onto the per-step affine half-space
        constraint  h(x_{t+1}) >= (1-gamma) h(x_t)  so the resulting reference
        sequence is feasible and the Gaussian samples centred on it are no longer
        mass-rejected. Exact for the (affine) double integrator; a first-order
        heuristic otherwise. Returns (guided_seq [H,2], normal_axis [H,2]).
        """
        H = int(self.config.horizon)
        dt = float(self.config.dt)
        eta = self._eta_eff()
        relax = float(self.config.guidance_relax)
        nominal_seq, _ = self._nominal_sequence(state, goal, H, u_min, u_max)
        x0 = state[0:1].clone()
        x = x0.clone()
        guided = []
        normal_axis = torch.zeros(H, 2, device=state.device, dtype=state.dtype)
        di = self.config.dynamics_type == "doubleintegrator"
        gh = min(H, int(self.config.guidance_horizon)) if self.config.guidance_horizon else H
        for t in range(gh):
            if obstacle_velocities is not None and safe_obstacles.numel():
                obs_t = safe_obstacles.clone()
                obs_t[..., :2] = obs_t[..., :2] + obstacle_velocities[..., :2] * (dt * t)
                obs_n = safe_obstacles.clone()
                obs_n[..., :2] = obs_n[..., :2] + obstacle_velocities[..., :2] * (dt * (t + 1))
            else:
                obs_t = obs_n = safe_obstacles
            u = nominal_seq[t].clone().view(1, 2)
            k = int(self.config.barrier_topk)
            ar = float(self.config.barrier_activation_radius)
            h_old_a, _, active = affine_barrier_h_ho_all(
                x0, x, obs_t, obstacle_velocities, eta, k, ar
            )
            x_next = self._step(x, u)
            h_new_a, grad_a, _ = affine_barrier_h_ho_all(
                x0, x_next, obs_n, obstacle_velocities, eta, k, ar
            )
            # g_j = d h_new_j / d u  (exact for double integrator; first-order else)
            scale = (0.5 * dt * dt + eta * dt) if di else dt
            g_a = grad_a * scale  # [1,N,2]
            deficit = (1.0 - gamma) * h_old_a - h_new_a  # [1,N] >0 => violates obstacle j
            deficit = torch.where(active, deficit, torch.zeros_like(deficit))
            gg = (g_a * g_a).sum(dim=2).clamp_min(1e-9)  # [1,N]
            corr = torch.clamp(deficit, min=0.0) / gg * relax  # [1,N]
            delta = (corr.unsqueeze(2) * g_a).sum(dim=1)  # [1,2] sum of per-obstacle pushes
            u = torch.clamp(u + delta, u_min, u_max)
            guided.append(u[0])
            # covariance axis = normal of the most-binding active obstacle
            masked_h = torch.where(active, h_new_a, torch.full_like(h_new_a, float("inf")))
            jstar = int(torch.argmin(masked_h[0]).item())
            gvec = grad_a[0, jstar]
            normal_axis[t] = gvec / torch.linalg.norm(gvec).clamp_min(1e-9)
            x = self._step(x, u)
        for t in range(gh, H):
            guided.append(nominal_seq[t])  # past guidance horizon: keep nominal (we replan each step)
        return torch.stack(guided, dim=0), normal_axis

    def safety_filter_action(
        self,
        state: torch.Tensor,
        obstacles: torch.Tensor,
        action: torch.Tensor,
        *,
        gamma: Optional[float] = None,
        obstacle_velocities: Optional[torch.Tensor] = None,
        iters: int = 3,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Runtime predictive safety filter (THEORY.md §7): minimally project a
        single proposed ``action`` (e.g. a one-step drifting/CFM output) onto the
        intersection of the active affine HO-DCBF half-spaces so the next state
        satisfies h_j(x1) >= (1-gamma) h_j(x0) for every active obstacle. A few
        Jacobi sweeps approximate the QP projection; returns (safe_action, info)
        and gives the learned policy a hard per-step certificate at ~us cost."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        device, dtype = state.device, state.dtype
        u_min = self.u_min.to(device=device, dtype=dtype)
        u_max = self.u_max.to(device=device, dtype=dtype)
        gamma_value = float(self.config.gamma if gamma is None else gamma)
        eta = self._eta_eff()
        ar = float(self.config.barrier_activation_radius)
        k = int(self.config.barrier_topk)
        di = self.config.dynamics_type == "doubleintegrator"
        dt = float(self.config.dt)
        obs = obstacles.to(device=device, dtype=dtype)
        if obs.numel() and self.config.safety_margin:
            obs = obs.clone()
            obs[..., 2] = obs[..., 2] + float(self.config.safety_margin) + float(self.config.barrier_extra_margin)
        u = action.detach().clone().view(1, 2).to(device=device, dtype=dtype)
        x0 = state[0:1]
        obs_next = obs
        if obstacle_velocities is not None and obstacle_velocities.numel():
            obstacle_velocities = obstacle_velocities.to(device=device, dtype=dtype)
            if obs.numel():
                obs_next = obs.clone()
                obs_next[..., :2] = obs_next[..., :2] + obstacle_velocities[..., :2] * dt
        n_corr = 0
        max_deficit = 0.0
        n_active = 0
        for _ in range(max(1, iters)):
            x1 = self._step(x0, u)
            h_old, _, active = affine_barrier_h_ho_all(x0, x0, obs, obstacle_velocities, eta, k, ar)
            # check the robot's next state against the obstacle's PREDICTED next position
            h_new, grad, _ = affine_barrier_h_ho_all(x0, x1, obs_next, obstacle_velocities, eta, k, ar)
            scale = (0.5 * dt * dt + eta * dt) if di else dt
            g = grad * scale
            deficit = (1.0 - gamma_value) * h_old - h_new
            deficit = torch.where(active, deficit, torch.zeros_like(deficit))
            n_active = int(active.sum().detach().cpu())
            max_deficit = float(torch.clamp(deficit, min=0.0).max().detach().cpu())
            if max_deficit <= 1e-6:
                break
            gg = (g * g).sum(dim=2).clamp_min(1e-9)
            corr = torch.clamp(deficit, min=0.0) / gg
            delta = (corr.unsqueeze(2) * g).sum(dim=1)
            u = torch.clamp(u + delta, u_min, u_max)
            n_corr += 1
        # recompute residual deficit at the final clamped control (clamping may
        # reintroduce a deficit even if the unclamped projection was feasible)
        x1 = self._step(x0, u)
        h_old, _, active = affine_barrier_h_ho_all(x0, x0, obs, obstacle_velocities, eta, k, ar)
        h_new, _, _ = affine_barrier_h_ho_all(x0, x1, obs_next, obstacle_velocities, eta, k, ar)
        final_def = torch.where(active, (1.0 - gamma_value) * h_old - h_new, torch.zeros_like(h_new))
        max_deficit = float(torch.clamp(final_def, min=0.0).max().detach().cpu())
        info = {"filter_iters": n_corr,
                "filter_feasible": bool(max_deficit <= 1e-4),
                "filter_max_deficit": max_deficit,
                "filter_num_active": n_active,
                "correction_magnitude": float(torch.linalg.norm(u.view(-1) - action.view(-1).to(u)).detach().cpu())}
        return u.view(-1), info

    def _controllability_matrix(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        input_width: torch.Tensor,
    ) -> torch.Tensor:
        horizon = int(controls.shape[0])
        state_dim = int(states.shape[1])
        blocks = []
        suffix = torch.eye(state_dim, dtype=states.dtype, device=states.device)
        for k in reversed(range(horizon)):
            A, B = self._linear_matrices(states[k], controls[k])
            B_norm = B * input_width.view(1, 2)
            blocks.append(suffix @ B_norm)
            suffix = suffix @ A
        blocks.reverse()
        return torch.cat(blocks, dim=1)

    def _sets_backup_controls(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        safe_obstacles: torch.Tensor,
        obstacle_velocities: Optional[torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, list[str], list[str]]:
        horizon = int(self.config.horizon)
        if horizon <= 0:
            empty = torch.empty(0, 0, 2, dtype=state.dtype, device=state.device)
            return empty, [], []

        nominal_seq, nominal_states = self._nominal_sequence(state, goal, horizon, u_min, u_max)
        input_width = (u_max - u_min).clamp_min(1e-6)
        normalized_nominal = ((nominal_seq - u_min.view(1, 2)) / input_width.view(1, 2)).clamp(0.0, 1.0)
        cmat = self._controllability_matrix(nominal_states, nominal_seq, input_width)
        if cmat.numel() == 0:
            empty = torch.empty(0, horizon, 2, dtype=state.dtype, device=state.device)
            return empty, [], []

        gramian = cmat @ cmat.T
        eigvals, eigvecs = torch.linalg.eigh(gramian)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        pinv_c = torch.linalg.pinv(cmat)

        branches = []
        labels: list[str] = []
        kinds: list[str] = []

        def add_linear_branch(target: torch.Tensor, label: str, kind: str) -> None:
            delta_v = pinv_c @ target
            normalized = (normalized_nominal.reshape(-1) + delta_v).view(horizon, 2).clamp(0.0, 1.0)
            branches.append(u_min.view(1, 2) + normalized * input_width.view(1, 2))
            labels.append(label)
            kinds.append(kind)

        max_modes = min(int(self.config.sets_num_modes), eigvecs.shape[1])
        scale = float(self.config.sets_branch_scale)
        for mode in range(max_modes):
            if float(eigvals[mode].detach().cpu()) <= 1e-10:
                continue
            axis = torch.sqrt(eigvals[mode].clamp_min(0.0)) * eigvecs[:, mode] * scale
            add_linear_branch(axis, f"m{mode}+", "sets_mode")
            add_linear_branch(-axis, f"m{mode}-", "sets_mode")

        if self.config.sets_include_cbf_backup and safe_obstacles.numel():
            centers = safe_obstacles[:, :2]
            radii = safe_obstacles[:, 2]
            pos = state[0, :2]
            clearances = torch.linalg.norm(centers - pos.view(1, 2), dim=1) - radii
            obs_idx = int(torch.argmin(clearances).detach().cpu())
            center = centers[obs_idx]
            rel = pos - center
            rel_norm = torch.linalg.norm(rel).clamp_min(1e-6)
            away = rel / rel_norm
            tangent = torch.stack((-away[1], away[0]))
            push = float(self.config.sets_cbf_push)
            target = torch.zeros(nominal_states.shape[1], dtype=state.dtype, device=state.device)
            target[:2] = push * away
            add_linear_branch(target, "away", "cbf_backup")
            target_tan = torch.zeros_like(target)
            target_tan[:2] = 0.75 * push * tangent
            add_linear_branch(target_tan, "tan+", "cbf_backup")
            add_linear_branch(-target_tan, "tan-", "cbf_backup")

            if self.config.dynamics_type == "unicycle":
                reverse = torch.zeros(horizon, 2, dtype=state.dtype, device=state.device)
                reverse[:, 0] = -min(float(self.config.sets_reverse_speed), abs(float(u_min[0])))
                reverse[:, 1] = 0.0
                branches.append(torch.clamp(reverse, u_min.view(1, 2), u_max.view(1, 2)))
                labels.append("back")
                kinds.append("hard_backup")
                for sign, label in [(1.0, "rev+"), (-1.0, "rev-")]:
                    rev_turn = reverse.clone()
                    rev_turn[:, 1] = sign * min(float(self.config.sets_turn_rate), float(u_max[1]))
                    branches.append(torch.clamp(rev_turn, u_min.view(1, 2), u_max.view(1, 2)))
                    labels.append(label)
                    kinds.append("hard_backup")

        if not branches:
            empty = torch.empty(0, horizon, 2, dtype=state.dtype, device=state.device)
            return empty, [], []
        return torch.stack(branches, dim=0), labels, kinds

    def plan(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        *,
        gamma: Optional[float] = None,
        obstacle_velocities: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        return_rollouts: bool = False,
        proposal_controls: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        t0 = time.perf_counter()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if goal.ndim > 1:
            goal = goal[0]
        device = state.device
        dtype = state.dtype
        u_min = self.u_min.to(device=device, dtype=dtype)
        u_max = self.u_max.to(device=device, dtype=dtype)
        obstacles = obstacles.to(device=device, dtype=dtype)
        safe_obstacles = obstacles.clone()
        if safe_obstacles.numel():
            safe_obstacles[..., 2] = safe_obstacles[..., 2] + float(self.config.safety_margin) + float(self.config.barrier_extra_margin)
        if obstacle_velocities is not None:
            obstacle_velocities = obstacle_velocities.to(device=device, dtype=dtype)
            if obstacle_velocities.ndim == 1:
                obstacle_velocities = obstacle_velocities.unsqueeze(0)
        if obstacles.ndim == 2:
            obstacles_batch0 = safe_obstacles.unsqueeze(0).expand(self.config.num_samples, -1, -1)
        else:
            obstacles_batch0 = safe_obstacles
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(int(seed))
        sigma = self._sigma(device, dtype)
        # --- nominal robot-centered polytope at x0: drives the level-set rejection + mean/sigma steering ---
        poly = None; trapped = 0.0; size_trapped = 0.0
        if (self.config.use_polytope_barrier or self.config.centroid_gain > 0.0
                or self.config.sigma_volume_gain > 0.0) and safe_obstacles.numel():
            poly = self._polytope_proposal(state, safe_obstacles, obstacle_velocities)  # (A,b,c,margins,d_centroid,size)
            R = float(self.config.barrier_activation_radius) or 3.0
            if self.config.urgency_size_diff:                          # mode 4: SHRINK RATE (sensitive to the onset)
                sp = self._size_prev if self._size_prev is not None else poly[5]
                trapped = max(0.0, sp - poly[5])                       # rho = max(0, size_{k-1}-size_k); p_b=clip(c_g*rho)
                self._size_prev = poly[5]
            else:                                                      # mode 1: (R-size)/(size+eps), "1/volume"-like, 0 if open
                trapped = max(0.0, (R - poly[5]) / (poly[5] + float(self.config.centroid_eps)))
            size_trapped = max(0.0, (R - poly[5]) / (poly[5] + float(self.config.centroid_eps)))   # size-based (for trapped-temp)
            if self.config.sigma_volume_gain > 0.0:                 # small polytope (trapped) -> wide sigma (capped)
                sigma = sigma * min(1.0 + self.config.sigma_volume_gain * trapped, float(self.config.sigma_max_mult))
        gamma_value = float(self.config.gamma if gamma is None else gamma)
        if self.config.adaptive_gamma and safe_obstacles.numel():
            from .gamma_schedule import gamma_distance_velocity
            obs2 = safe_obstacles if safe_obstacles.ndim == 2 else safe_obstacles[0]
            pos = state[0, :2]
            centers = obs2[:, :2]
            radii = obs2[:, 2]
            clr = torch.linalg.norm(centers - pos.view(1, 2), dim=1) - radii
            j = int(torch.argmin(clr).item())
            d = float(clr[j].clamp_min(0.0).item())
            dirn = (centers[j] - pos)
            dirn = dirn / torch.linalg.norm(dirn).clamp_min(1e-6)
            vel = state[0, 2:4] if state.shape[1] >= 4 else torch.zeros(2, device=device, dtype=dtype)
            if obstacle_velocities is not None and obstacle_velocities.numel():
                vrel = vel - obstacle_velocities[min(j, obstacle_velocities.shape[0] - 1)]
            else:
                vrel = vel
            v_proj = float(torch.sum(vrel * dirn).clamp_min(0.0).item())  # closing rate (>0 approaching)
            gamma_value = gamma_distance_velocity(
                d, v_proj, g_min=float(self.config.gamma_min), g_max=float(self.config.gamma_max)
            )
        mix_mean = None; mix_p = 0.0; sample_mode = None   # mixture diagnostics (set in the polytope-proposal branch)
        if proposal_controls is not None:
            # Learned-proposal mode (THEORY §10): use externally-supplied control
            # sequences (e.g. from a gamma-conditioned flow) as the MPPI proposal;
            # the DCBF rejection + averaging + output filter remain the certificate.
            controls = torch.clamp(
                proposal_controls.to(device=device, dtype=dtype), u_min, u_max
            )
            H = int(self.config.horizon)
            if controls.shape[1] > H:
                controls = controls[:, :H]
            elif controls.shape[1] < H:
                pad = controls[:, -1:].expand(-1, H - controls.shape[1], -1)
                controls = torch.cat([controls, pad], dim=1)
            # Mix in Gaussian samples around the velocity-damped nominal so MPPI
            # can pick braking near the goal (the learned proposal alone has no
            # velocity regulation and overshoots/diverges past the goal).
            nominal_seq, _ = self._nominal_sequence(state, goal, H, u_min, u_max)
            nmix = int(self.config.proposal_gaussian_mix)
            if nmix > 0:
                gnoise = torch.randn(nmix, H, 2, generator=gen, device=device, dtype=dtype) * sigma.view(1, 1, 2)
                gmix = torch.clamp(gnoise + nominal_seq.unsqueeze(0), u_min, u_max)
                controls = torch.cat([controls, gmix], dim=0)
        else:
            if self.config.use_guidance:
                nominal_seq, normal_axis = self._guide_nominal(
                    state, goal, safe_obstacles, obstacle_velocities, gamma_value, u_min, u_max
                )
            else:
                if self.config.warm_start and self._u_prev is not None and self._u_prev.shape[0] == self.config.horizon:
                    # MPPI spirit: nominal = the PREVIOUS reward-weighted sequence, shifted one step (repeat last).
                    # Cold start is 0; it evolves into the goal-directed solution, so the random rollouts explore
                    # around it and the executed 1st action stops being random.
                    nominal_seq = torch.cat([self._u_prev[1:], self._u_prev[-1:]], dim=0).to(device=device, dtype=dtype)
                elif self.config.use_goal_nominal:
                    nominal_seq, _ = self._nominal_sequence(state, goal, self.config.horizon, u_min, u_max)
                else:
                    nominal_seq = torch.zeros(self.config.horizon, 2, device=device, dtype=dtype)  # cold seed = 0
                normal_axis = None
            # --- 3-MODE categorical proposal over ALL H steps (the "clever sampling" where safety comes from) ---
            #   Mode A ~ N(warm, sigma)                       goal-ward (warm-start);  p_a = 1 - p_b - p_c
            #   Mode B ~ N(warm + u_max*d_ctrl, sigma_aniso)  opening-ward toward the EXACT polytope centroid;  p_b = p_t
            #   Mode C = braking clamp(-v/dt) + random-360    ALWAYS-ON backup;  p_c = random_backup_frac
            #   p_t = clip(centroid_gain*trapped, 0, 1), temporally low-passed (smoothness). d_ctrl = B+ d_centroid.
            H = int(self.config.horizon); N = int(self.config.num_samples)
            mix_p = 0.0; u_target = torch.zeros(2, device=device, dtype=dtype); d_ctrl = None; Bpos = None
            if poly is not None and (self.config.centroid_gain > 0.0 or self.config.random_backup_frac > 0.0
                                     or self.config.polytope_area_sampling):
                _, B = self._linear_matrices(state[0], nominal_seq[0]); Bpos = B[:2, :]   # control -> position map
            if self.config.centroid_gain > 0.0 and poly is not None:
                mix_p = min(max(self.config.centroid_gain * trapped, float(self.config.urgency_floor)), 1.0)  # p_b=clip(c_g*rho, floor, 1)
                if self.config.centroid_smooth > 0.0 and self._p_prev is not None:
                    mix_p = (1 - self.config.centroid_smooth) * mix_p + self.config.centroid_smooth * float(self._p_prev)
                self._p_prev = mix_p
                if float(torch.linalg.norm(poly[4])) > 1e-6:           # centroid direction (for the standard Mode B)
                    u_dir = torch.linalg.pinv(Bpos) @ poly[4]          # B+ d_centroid (~ d_centroid for SI/DI)
                    if float(torch.linalg.norm(u_dir)) > 1e-9:
                        d_ctrl = u_dir / torch.linalg.norm(u_dir); u_target = float(u_max.max()) * d_ctrl
            # 3-mode categorical: Mode A (warm iso) / Mode B (centroid aniso, p_b=p_t) / Mode C (ALWAYS-ON random
            #   backup, p_c=random_backup_frac): even 360deg escape samples EVERY frame (incl. degenerate => p_t=0).
            area = bool(self.config.polytope_area_sampling)
            p_c = float(self.config.random_backup_frac) if poly is not None else 0.0   # Mode C (braking+random) may coexist with area
            nC = min(int(round(p_c * N)), N)
            p_b = mix_p if (d_ctrl is not None or area) else 0.0        # area Mode B needs no centroid dir (uses rays)
            nB = min(int(round(p_b * N)), N - nC); nA = N - nB - nC
            noise = torch.randn(N, H, 2, generator=gen, device=device, dtype=dtype) * sigma.view(1, 1, 2)
            ctr = nominal_seq.unsqueeze(0).expand(N, -1, -1).clone()
            sample_mode = torch.zeros(N, dtype=torch.long, device=device)
            if nB > 0 and area and Bpos is not None and poly is not None:  # Mode B = polytope-AREA rays (importance sampling)
                sample_mode[nA:nA + nB] = 1
                u_area = self._polytope_ray_controls(poly[0], poly[1], poly[2], nB, Bpos, u_max, gen, device, dtype)
                ctr[nA:nA + nB] = ctr[nA:nA + nB] + u_area.view(nB, 1, 2)   # constant control toward a random interior point
            elif nB > 0 and d_ctrl is not None:                         # Mode B: centroid/opening, anisotropic (standard)
                sample_mode[nA:nA + nB] = 1
                ctr[nA:nA + nB] = ctr[nA:nA + nB] + u_target.view(1, 1, 2)
                n = d_ctrl.view(1, 1, 2); tg = torch.stack((-d_ctrl[1], d_ctrl[0])).view(1, 1, 2)
                seg = noise[nA:nA + nB]; cn = (seg * n).sum(-1, keepdim=True); ct = (seg * tg).sum(-1, keepdim=True)
                noise[nA:nA + nB] = float(self.config.sigma_aniso) * cn * n + ct * tg   # ellipsoid wide || opening
            if nC > 0:                                                   # Mode C backup = BRAKING + random-360 escape
                sample_mode[nA + nB:] = 2; off = nA + nB; nBrake = nC // 2; nRand = nC - nBrake
                vcur = state[0, 2:4] if state.shape[1] >= 4 else torch.zeros(2, device=device, dtype=dtype)
                if nBrake > 0:                                           # full deceleration u=clamp(-v/dt): robot brakes/
                    u_brake = torch.clamp(-vcur / float(self.config.dt), u_min, u_max)   # backs off => H preserved => accepted
                    ctr[off:off + nBrake] = u_brake.view(1, 1, 2)        # override warm with sustained braking
                if nRand > 0:                                            # even 360deg escape at u_max (exploration + visible spread)
                    two_pi = 2.0 * float(np.pi)
                    theta0 = float(torch.rand(1, generator=gen, device=device)[0]) * two_pi
                    ang = theta0 + torch.arange(nRand, device=device, dtype=dtype) * (two_pi / nRand)
                    rdir = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
                    rc = (torch.linalg.pinv(Bpos) @ rdir.t()).t() if Bpos is not None else rdir   # B+ -> control dirs
                    rc = rc / torch.linalg.norm(rc, dim=1, keepdim=True).clamp_min(1e-9)
                    ctr[off + nBrake:] = ctr[off + nBrake:] - float(u_max.max()) * rc.unsqueeze(1)
            controls = torch.clamp(ctr + noise, u_min, u_max)
            mix_mean = nominal_seq[0] + mix_p * u_target                 # effective Mode-B sampling mean, for viz
        branch_labels: list[str] = []
        branch_kinds: list[str] = []
        branch_indices = torch.empty(0, dtype=torch.long, device=device)
        if self.config.use_sets_backup:
            branch_controls, branch_labels, branch_kinds = self._sets_backup_controls(
                state,
                goal,
                safe_obstacles,
                obstacle_velocities,
                u_min,
                u_max,
            )
            if branch_controls.numel():
                branch_indices = torch.arange(controls.shape[0], controls.shape[0] + branch_controls.shape[0], device=device)
                controls = torch.cat([controls, branch_controls], dim=0)
        sample_count = int(controls.shape[0])
        nominal = nominal_seq[0]
        if obstacles.ndim == 2:
            obstacles_batch0 = safe_obstacles.unsqueeze(0).expand(sample_count, -1, -1)
        else:
            obstacles_batch0 = safe_obstacles
        x0 = state[0].unsqueeze(0).expand(sample_count, -1)
        x = x0.clone()
        state_seq = [x.clone()]
        costs = torch.zeros(sample_count, device=device, dtype=dtype)
        infeasible = torch.zeros(sample_count, device=device, dtype=torch.bool)
        min_h = torch.full((sample_count,), float("inf"), device=device, dtype=dtype)
        initial_goal_distance = torch.linalg.norm(x[:, :2] - goal[:2].to(device=device, dtype=dtype), dim=1)
        previous_goal_distance = initial_goal_distance
        prev_action = torch.zeros_like(controls[:, 0])
        for t in range(self.config.horizon):
            if obstacle_velocities is not None and safe_obstacles.numel():
                obs_t = safe_obstacles.clone()
                obs_t[..., :2] = obs_t[..., :2] + obstacle_velocities[..., :2] * (self.config.dt * t)
                obs_next = safe_obstacles.clone()
                obs_next[..., :2] = obs_next[..., :2] + obstacle_velocities[..., :2] * (self.config.dt * (t + 1))
                obstacles_batch = obs_t.unsqueeze(0).expand(sample_count, -1, -1) if obs_t.ndim == 2 else obs_t
                obstacles_batch_next = obs_next.unsqueeze(0).expand(sample_count, -1, -1) if obs_next.ndim == 2 else obs_next
            else:
                obstacles_batch = obstacles_batch0
                obstacles_batch_next = obstacles_batch0
            x_next = self._step(x, controls[:, t])
            if self.config.use_polytope_barrier and poly is not None:
                # reject on the NOMINAL polytope level sets: H_P(x_{i+1}) >= (1-gamma) H_P(x_i). The polytope is FIXED
                # at x0 and accounts for every nearby obstacle (K-gon ∩ tangents, smooth), so feasibility is not an
                # artifact of a single jumpy nearest obstacle.
                h_old = self._polytope_H(x[:, :2], poly[0], poly[1], poly[3])
                h_new = self._polytope_H(x_next[:, :2], poly[0], poly[1], poly[3])
                min_h = torch.minimum(min_h, h_new)
                violation = h_new < (1.0 - gamma_value) * h_old
            elif self.config.use_ho_barrier:
                eta_eff = self._eta_eff()
                k = int(self.config.barrier_topk)
                ar = float(self.config.barrier_activation_radius)
                h_old_a, _, active = affine_barrier_h_ho_all(
                    x0, x, obstacles_batch, obstacle_velocities, eta_eff, k, ar
                )
                h_new_a, _, _ = affine_barrier_h_ho_all(
                    x0, x_next, obstacles_batch_next, obstacle_velocities, eta_eff, k, ar
                )
                viol_j = (h_new_a < (1.0 - gamma_value) * h_old_a) & active
                violation = viol_j.any(dim=1)
                min_h = torch.minimum(min_h, h_new_a.min(dim=1).values)
            else:
                h_old = self._barrier_h(x0, x, obstacles_batch, obstacle_velocities)
                h_new = self._barrier_h(x0, x_next, obstacles_batch_next, obstacle_velocities)
                min_h = torch.minimum(min_h, h_new)
                violation = h_new < (1.0 - gamma_value) * h_old
            if self.config.check_first_control_only:
                if t == 0:
                    infeasible |= violation
            else:
                infeasible |= violation
            goal_distance = torch.linalg.norm(x_next[:, :2] - goal[:2].to(device=device, dtype=dtype), dim=1)
            goal_cost = self.config.running_goal_weight * goal_distance**2
            effort = self.config.control_weight * torch.sum(controls[:, t] ** 2, dim=1)
            smooth = self.config.smooth_weight * torch.sum((controls[:, t] - prev_action) ** 2, dim=1)
            progress = -self.config.progress_weight * (initial_goal_distance - goal_distance)
            if self.config.goal_retreat_exp_weight > 0.0:
                scale = max(float(self.config.goal_retreat_exp_scale), torch.finfo(dtype).eps)
                retreat = torch.relu(goal_distance - previous_goal_distance)
                normalized_retreat = torch.clamp(
                    retreat / scale,
                    max=max(float(self.config.goal_retreat_exp_cap), 0.0),
                )
                retreat_cost = self.config.goal_retreat_exp_weight * torch.expm1(normalized_retreat)
            else:
                # Keep the default path algebraically identical to the
                # pre-feature planner, including its floating-point operation
                # order.  This makes the option safe for legacy checkpoints.
                retreat_cost = 0.0
            clearance = barrier_clearance(x_next[:, :2], obstacles_batch_next)
            soft_clearance = self.config.soft_clearance_weight * torch.relu(-clearance) ** 2
            if self.config.dynamics_type == "unicycle":
                to_goal_next = goal[:2].to(device=device, dtype=dtype) - x_next[:, :2]
                desired_heading = torch.atan2(to_goal_next[:, 1], to_goal_next[:, 0])
                heading_error = torch.atan2(torch.sin(desired_heading - x_next[:, 2]), torch.cos(desired_heading - x_next[:, 2]))
                heading_cost = self.config.heading_weight * heading_error**2
            else:
                heading_cost = 0.0
            costs += goal_cost + effort + smooth + soft_clearance + progress + heading_cost + retreat_cost
            x = x_next
            previous_goal_distance = goal_distance
            prev_action = controls[:, t]
            state_seq.append(x.clone())
        terminal_goal = torch.linalg.norm(x[:, :2] - goal[:2].to(device=device, dtype=dtype), dim=1)
        costs = costs + self.config.terminal_goal_weight * terminal_goal**2
        raw_costs = costs.clone()
        costs = torch.where(infeasible, torch.full_like(costs, float("inf")), costs)
        if torch.isinf(costs).all():
            # No feasible sample (degenerate dense moment): fall back to the SAFEST rollout (highest barrier min_h),
            # NOT the lowest-cost (goal-seeking) one -- the latter drives straight into a pedestrian. A small goal
            # term only breaks ties between equally-safe rollouts.
            costs = -min_h + 1e-3 * raw_costs
        best = torch.argmin(costs)
        # MPPI temperature-weighted average over the SURVIVING (non-rejected) rollouts. Rejected samples have
        # cost=inf -> weight 0, so gamma (which sets the surviving set) shapes the mean. temperature->0 recovers
        # the argmin (cold) limit; larger temperature averages more broadly (hot).
        temp = max(float(self.config.temperature) * (1.0 + float(self.config.temp_trapped_gain) * size_trapped), 1e-6)  # trapped-temp
        w = torch.softmax(-(costs - costs.min()) / temp, dim=0)
        w = torch.nan_to_num(w, nan=0.0)
        if float(w.sum()) < 1e-8:
            action = controls[best, 0].clamp(u_min, u_max)
            u_avg_seq = controls[best]
        else:
            action = (w.unsqueeze(1) * controls[:, 0]).sum(0).clamp(u_min, u_max)
            u_avg_seq = (w.view(-1, 1, 1) * controls).sum(0)            # full reward-weighted sequence
        if self.config.warm_start:                                     # carry it forward (shifted) next step
            self._u_prev = u_avg_seq.detach().clamp(u_min.view(1, 2), u_max.view(1, 2))
        filt_info = None
        if self.config.filter_output and self.config.use_ho_barrier and obstacles.numel():
            # PSF guarantee on the APPLIED control: even if every sample was
            # rejected (degenerate), project onto the active half-spaces so the
            # executed action provably satisfies the DCBF (THEORY §4/§7).
            action, filt_info = self.safety_filter_action(
                state[0], obstacles, action, gamma=gamma_value,
                obstacle_velocities=obstacle_velocities, iters=int(self.config.filter_iters),
            )
            action = action.clamp(u_min, u_max)
        clearance = barrier_clearance(state[:, :2], safe_obstacles.unsqueeze(0) if safe_obstacles.ndim == 2 else safe_obstacles[:1]).min()
        info = {
            "gamma": gamma_value,
            "min_barrier_h": float(min_h[best].detach().cpu()),
            "min_clearance": float(clearance.detach().cpu()),
            "num_barrier_violations": int(infeasible.sum().detach().cpu()),
            "infeasibility_rate": float(infeasible.float().mean().detach().cpu()),
            "correction_magnitude": float(torch.linalg.norm(action - nominal).detach().cpu()),
            "num_backup_branches": int(branch_indices.numel()),
            "selected_backup_branch": None,
            "solve_time": time.perf_counter() - t0,
            # control-space proposal diagnostics (for the sampling viz): first-step samples + accept/reject + mean + cov
            "first_controls": controls[:, 0].detach().cpu().numpy(),       # [M,2]
            "feasible": (~infeasible).detach().cpu().numpy(),              # [M] bool
            "mean_control": action.detach().cpu().numpy(),                 # [2] executed (reward-weighted / safe-fallback)
            "sample_mean": (mix_mean.detach().cpu().numpy() if mix_mean is not None else nominal.detach().cpu().numpy()),
            "mixture_p": float(mix_p),                                      # Mode-B (opening) mixture fraction
            "sample_mode": (sample_mode.detach().cpu().numpy() if sample_mode is not None else None),  # [N] 0=A/1=B/2=C
            "sigma": sigma.detach().cpu().reshape(-1).numpy(),            # [2] sampling std (control units)
            # Exact full-plan observability for external deterministic
            # verification.  These fields do not alter MPPI selection or the
            # returned action.  A caller making a runtime certificate claim
            # must verify the same sequence whose first action it executes.
            "mean_sequence": u_avg_seq.detach().cpu().numpy(),
            "best_sequence": controls[best].detach().cpu().numpy(),
            "best_feasible_internal": bool((~infeasible[best]).detach().cpu()),
            "all_samples_infeasible_internal": bool(infeasible.all().detach().cpu()),
            # polytope diagnostics: faces (A,b,c,margins) + free-space centroid DIR + size + EXACT centroid POSITION
            "polytope": None if poly is None else tuple(t.detach().cpu().numpy() for t in poly[:4]),
            "centroid_dir": None if poly is None else poly[4].detach().cpu().numpy(),
            "polytope_size": None if poly is None else float(poly[5]),
            "centroid_pos": None if poly is None else poly[6].detach().cpu().numpy(),
        }
        if filt_info is not None:
            info["filter_feasible"] = filt_info["filter_feasible"]
            info["filter_max_deficit"] = filt_info["filter_max_deficit"]
            info["filter_infeasible"] = (not filt_info["filter_feasible"])
        if branch_indices.numel():
            branch_hit = torch.nonzero(branch_indices == best, as_tuple=False).flatten()
            if branch_hit.numel():
                info["selected_backup_branch"] = branch_labels[int(branch_hit[0].detach().cpu())]
        if return_rollouts:
            state_seq_t = torch.stack(state_seq, dim=1).detach().cpu()
            feasible = (~infeasible).detach().cpu()
            max_rollouts = max(1, int(self.config.debug_max_rollouts))
            branch_indices_cpu = branch_indices.detach().cpu()
            sample_mask = torch.ones(state_seq_t.shape[0], dtype=torch.bool)
            if branch_indices_cpu.numel():
                sample_mask[branch_indices_cpu] = False
            if state_seq_t.shape[0] > max_rollouts:
                accept_idx = torch.nonzero(feasible & sample_mask, as_tuple=False).flatten()[: max_rollouts // 2]
                reject_idx = torch.nonzero((~feasible) & sample_mask, as_tuple=False).flatten()[: max_rollouts - accept_idx.numel()]
                draw_idx = torch.cat([accept_idx, reject_idx], dim=0)
                if draw_idx.numel() == 0:
                    draw_idx = torch.arange(min(max_rollouts, state_seq_t.shape[0]))
            else:
                draw_idx = torch.nonzero(sample_mask, as_tuple=False).flatten()
            info["debug_rollouts"] = {
                "states": state_seq_t[draw_idx].numpy(),
                "controls": controls.detach().cpu()[draw_idx].numpy(),
                "feasible": feasible[draw_idx].numpy(),
                "best_state": state_seq_t[best].numpy(),
            }
            if branch_indices_cpu.numel():
                info["debug_rollouts"]["branch_states"] = state_seq_t[branch_indices_cpu].numpy()
                info["debug_rollouts"]["branch_feasible"] = feasible[branch_indices_cpu].numpy()
                info["debug_rollouts"]["branch_labels"] = branch_labels
                info["debug_rollouts"]["branch_kinds"] = branch_kinds
        return action, info
