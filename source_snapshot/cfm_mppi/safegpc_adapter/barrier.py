from __future__ import annotations

import torch


def barrier_clearance(pos: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
    if obstacles.numel() == 0:
        return torch.full(pos.shape[:-1], float("inf"), device=pos.device, dtype=pos.dtype)
    centers = obstacles[..., :2]
    radii = obstacles[..., 2]
    d = torch.linalg.norm(pos.unsqueeze(-2) - centers, dim=-1) - radii
    return torch.min(d, dim=-1).values


def affine_barrier_h(x0: torch.Tensor, x: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
    """
    Port of safeGPC DoubleIntegrator2D.hnew_torch/huniversal_proj_torch.

    It selects the nearest circle by current position and projects the current
    position onto the initial state's nearest-boundary normal.
    """
    if obstacles.numel() == 0:
        return torch.full((x.shape[0],), float("inf"), device=x.device, dtype=x.dtype)
    obs = obstacles.to(device=x.device, dtype=x.dtype)
    if obs.ndim == 2:
        obs = obs.unsqueeze(0).expand(x.shape[0], -1, -1)
    centers = obs[:, :, :2]
    radii = obs[:, :, 2]
    pos0 = x0[:, :2]
    pos = x[:, :2]
    d_current = torch.linalg.norm(pos[:, None, :] - centers, dim=2) - radii
    idx = torch.argmin(d_current, dim=1)
    batch = torch.arange(x.shape[0], device=x.device)
    c_sel = centers[batch, idx]
    r_sel = radii[batch, idx]
    diff0 = pos0 - c_sel
    dist0 = torch.linalg.norm(diff0, dim=1).clamp_min(1e-12)
    nearest0 = c_sel + diff0 / dist0.unsqueeze(1) * r_sel.unsqueeze(1)
    d0b = (dist0 - r_sel).clamp_min(1e-12)
    normal = nearest0 - pos0
    normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-12)
    raw_proj = torch.sum((nearest0 - pos) * normal, dim=1)
    return raw_proj / d0b


def affine_barrier_h_ho(
    x0: torch.Tensor,
    x: torch.Tensor,
    obstacles: torch.Tensor,
    obstacle_velocities: torch.Tensor | None = None,
    eta: float = 0.0,
    return_grad: bool = False,
):
    """Affine higher-order (relative-degree-aware) DCBF, still a half-space.

    Extends :func:`affine_barrier_h` with a velocity look-ahead term so the
    barrier is a valid CBF for relative-degree-2 systems (double integrator) and
    is aware of obstacle motion (pedestrians):

        h_ho(x) = [ (nearest0 - p)·n  -  eta * ((v - v_obs)·n) ] / d0b

    where n is the outward normal toward the nearest obstacle, v the robot
    velocity (state dims 2:4 when present), and v_obs the obstacle velocity.
    h_ho is affine in (p, v) => its >=0 set is a convex half-space, so the
    MPPI convex-averaging safety argument (Props 1-2) is preserved while the
    barrier now encodes braking/closing-rate (cures the relative-degree freeze).

    With ``return_grad`` the per-sample gradient g = d h_ho / d p of the
    position part w.r.t. the robot position is also returned (unit normal
    scaled by 1/d0b), used by the safety-filter guidance projection.
    """
    if obstacles.numel() == 0:
        h = torch.full((x.shape[0],), float("inf"), device=x.device, dtype=x.dtype)
        if return_grad:
            grad = torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)
            ones = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            return h, grad, ones
        return h
    obs = obstacles.to(device=x.device, dtype=x.dtype)
    if obs.ndim == 2:
        obs = obs.unsqueeze(0).expand(x.shape[0], -1, -1)
    centers = obs[:, :, :2]
    radii = obs[:, :, 2]
    pos0 = x0[:, :2]
    pos = x[:, :2]
    d_current = torch.linalg.norm(pos[:, None, :] - centers, dim=2) - radii
    idx = torch.argmin(d_current, dim=1)
    batch = torch.arange(x.shape[0], device=x.device)
    c_sel = centers[batch, idx]
    r_sel = radii[batch, idx]
    diff0 = pos0 - c_sel
    dist0 = torch.linalg.norm(diff0, dim=1).clamp_min(1e-12)
    nearest0 = c_sel + diff0 / dist0.unsqueeze(1) * r_sel.unsqueeze(1)
    d0b = (dist0 - r_sel).clamp_min(1e-12)
    normal = nearest0 - pos0
    normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-12)
    raw_proj = torch.sum((nearest0 - pos) * normal, dim=1)
    h = raw_proj
    if eta != 0.0 and x.shape[1] >= 4:
        vel = x[:, 2:4]
        if obstacle_velocities is not None and obstacle_velocities.numel():
            v_obs = obstacle_velocities.to(device=x.device, dtype=x.dtype)
            if v_obs.ndim == 2:
                # one velocity per obstacle: pick the selected obstacle's velocity
                v_obs = v_obs[idx]
            vel = vel - v_obs
        closing = torch.sum(vel * normal, dim=1)  # >0 => approaching along normal
        h = h - eta * closing
    h = h / d0b
    if return_grad:
        # d(raw_proj)/d p = -normal ; divided by d0b
        grad = (-normal) / d0b.unsqueeze(1)
        return h, grad, d0b
    return h


def affine_barrier_h_ho_all(
    x0: torch.Tensor,
    x: torch.Tensor,
    obstacles: torch.Tensor,
    obstacle_velocities: torch.Tensor | None = None,
    eta: float = 0.0,
    topk: int = 0,
    activation_radius: float = 0.0,
):
    """Per-obstacle affine HO-DCBF for ALL obstacles simultaneously.

    Returns (h [B, N], grad [B, N, 2], active [B, N] bool). Each obstacle gets
    its own supporting half-space (normal from x0 toward that obstacle). The MPPI
    rejection then enforces the *intersection* of these half-spaces (a convex
    polytope), which is the multi-obstacle case of THEORY.md §6 and is required
    for crowds. Activation uses the CURRENT clearance (so pedestrians that
    approach later are enforced once near): an obstacle is active iff its current
    clearance < ``activation_radius`` (0 => no radius gate), further capped to the
    ``topk`` nearest by current clearance (0 => no cap).
    """
    B = x.shape[0]
    if obstacles.numel() == 0:
        return (
            torch.full((B, 1), float("inf"), device=x.device, dtype=x.dtype),
            torch.zeros(B, 1, 2, device=x.device, dtype=x.dtype),
            torch.zeros(B, 1, device=x.device, dtype=torch.bool),
        )
    obs = obstacles.to(device=x.device, dtype=x.dtype)
    if obs.ndim == 2:
        obs = obs.unsqueeze(0).expand(B, -1, -1)
    N = obs.shape[1]
    centers = obs[:, :, :2]            # [B,N,2]
    radii = obs[:, :, 2]               # [B,N]
    pos0 = x0[:, :2]                   # [B,2]
    pos = x[:, :2]                     # [B,2]
    diff0 = pos0[:, None, :] - centers
    dist0 = torch.linalg.norm(diff0, dim=2).clamp_min(1e-12)
    nearest0 = centers + diff0 / dist0.unsqueeze(2) * radii.unsqueeze(2)
    d0b = (dist0 - radii).clamp_min(1e-12)
    normal = nearest0 - pos0[:, None, :]
    normal = normal / torch.linalg.norm(normal, dim=2, keepdim=True).clamp_min(1e-12)
    raw = torch.sum((nearest0 - pos[:, None, :]) * normal, dim=2)
    h = raw
    if eta != 0.0 and x.shape[1] >= 4:
        vel = x[:, 2:4]
        if obstacle_velocities is not None and obstacle_velocities.numel():
            v_obs = obstacle_velocities.to(device=x.device, dtype=x.dtype)
            if v_obs.ndim == 2:
                v_obs = v_obs.unsqueeze(0)  # [1,N,2]
            rel = vel[:, None, :] - v_obs
        else:
            rel = vel[:, None, :].expand(-1, N, -1)
        closing = torch.sum(rel * normal, dim=2)
        h = h - eta * closing
    h = h / d0b
    grad = (-normal) / d0b.unsqueeze(2)
    # activation by CURRENT clearance (pedestrians approaching later get enforced)
    cur_clear = torch.linalg.norm(pos[:, None, :] - centers, dim=2) - radii  # [B,N]
    active = torch.ones(B, N, device=x.device, dtype=torch.bool)
    if activation_radius and activation_radius > 0.0:
        active = cur_clear < float(activation_radius)
    if topk and topk < N:
        order = torch.argsort(cur_clear, dim=1)
        keep = order[:, :topk]
        keep_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        keep_mask.scatter_(1, keep, True)
        active = active & keep_mask
    return h, grad, active
