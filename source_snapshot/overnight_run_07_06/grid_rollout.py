"""Receding-horizon deployment of the windowed FM policy on the grid (measurement + ACTFLOW exploration).

fm_deploy(): from (0,0), at each step featurize (grid, low5, past-control history) -> sample the FM window
-> execute its first control (H_exec=1) -> step DI dynamics -> repeat until reach / die (off-grid or
collision) / T. With `tilt` set it does Eq-9 active exploration (sample N candidate windows + a broad
right/up 'surrounding' proposal, score by GP σ over φ_s, exp((σ−maxσ)/β) tilt, systematic-resample 1).
`record=True` returns per-step (grid, low5, hist, U, verifier-label) for the ACTFLOW buffer.
"""
from __future__ import annotations

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import verifier_polytope as VP
from di_grid_viz import di_step


def systematic_resample(w, n):
    w = (w / w.sum().clamp_min(1e-12)).flatten()
    u0 = float(torch.rand(1, device=w.device))
    pos = (torch.arange(n, device=w.device).float() + u0) / n
    cdf = torch.cumsum(w, 0)
    idx = torch.searchsorted(cdf, pos.clamp(max=1.0))
    return idx.clamp(max=len(w) - 1)


def broad_proposal(state, goal, env, n, device):
    """n right/up 'surrounding' candidate windows with random per-window right-fraction -> diverse staircases."""
    um = float(env.u_max)
    rho = np.random.rand(n, 1, 1)                                   # right-fraction per window
    base = np.concatenate([rho, 1.0 - rho], axis=2)                # [n,1,2] (right, up)
    base = base / (np.linalg.norm(base, axis=2, keepdims=True) + 1e-9)
    U = base * um + np.random.randn(n, GF.H_PRED, 2) * 0.45 * um
    return torch.tensor(np.clip(U, -um, um).astype(np.float32), device=device)


def broad_targeted(state, direction, env, n, device):
    """n surrounding candidates pushed toward `direction` (the target staircase's next R or U move)."""
    um = float(env.u_max)
    d = np.asarray(direction, np.float32)
    d = d / (np.linalg.norm(d) + 1e-9)
    U = d[None, None, :] * um + np.random.randn(n, GF.H_PRED, 2).astype(np.float32) * 0.35 * um
    return torch.tensor(np.clip(U, -um, um).astype(np.float32), device=device)


def window_positions(state, U, dt):
    """Roll DI dynamics over the H controls -> planned positions [H,2]."""
    st = np.asarray(state, np.float32).copy()
    pos = []
    for a in U:
        st = di_step(st, np.asarray(a, np.float32), dt=dt)
        pos.append(st[:2].copy())
    return np.array(pos, np.float32)


def di_rollout_batch(state, U, dt):
    """Vectorized DI rollout of M candidate windows U [M,H,2] from `state` -> positions [M,H,2]."""
    s = np.asarray(state, np.float32)
    px = np.full(U.shape[0], s[0], np.float32); py = np.full(U.shape[0], s[1], np.float32)
    vx = np.full(U.shape[0], s[2], np.float32); vy = np.full(U.shape[0], s[3], np.float32)
    out = np.empty((U.shape[0], U.shape[1], 2), np.float32)
    for h in range(U.shape[1]):
        ax, ay = U[:, h, 0], U[:, h, 1]
        px = px + dt * vx + 0.5 * dt * dt * ax
        py = py + dt * vy + 0.5 * dt * dt * ay
        vx = vx + dt * ax; vy = vy + dt * ay
        out[:, h, 0] = px; out[:, h, 1] = py
    return out


def safe_mask(state, U_np, obs, r_robot, dt):
    """Cheap (no-SOCP) per-candidate safety: window collision-free ∧ in task space. -> bool[M]."""
    pos = di_rollout_batch(state, U_np, dt)                          # [M,H,2]
    intask = ((pos >= -GM.EPS_TASK) & (pos <= GM.GRID_M + GM.EPS_TASK)).all(axis=(1, 2))
    if obs.size:
        d = np.linalg.norm(pos[:, :, None, :] - obs[None, None, :, :2], axis=3) - obs[None, None, :, 2] - r_robot
        clr = d.min(axis=(1, 2))
    else:
        clr = np.ones(U_np.shape[0])
    return intask & (clr >= 0.0)


def verify_window(state, U, env, gamma, back_tol=0.25, socp=True, n_theta=120):
    """Binary window verifier (buffer labels): (2) task-space ∧ (3) right/up (no step back beyond back_tol)
    ∧ (1) local SOCP certificate. Matches the trajectory-level criteria at the window level."""
    seg = window_positions(state, U, env.dt)
    if not GM.in_taskspace(seg):
        return False
    d = np.diff(np.vstack([np.asarray(state, float)[:2], seg]), axis=0)
    if (d[:, 0] < -back_tol).any() or (d[:, 1] < -back_tol).any():
        return False
    if socp:
        obs = env.obstacles.detach().cpu().numpy()
        if not bool(VP.certify_window(seg, obs, float(env.r_robot), float(gamma), R=2.5, n_theta=n_theta)[0]):
            return False
    return True


@torch.no_grad()
def fm_deploy(policy, env, gamma, T=250, temp=1.0, nfe=8, tilt=None, target=None, style_rho=None,
              record=False, verify_fn=verify_window, reach=GM.REACH, device="cpu"):
    """Deploy the FM policy receding-horizon. `style_rho`∈[0,1] softly biases the WHOLE trajectory toward a
    constant right/up ratio [ρ,1−ρ] (coherent diverse staircases); `target` (a 10-char R/U word) instead aims
    per-move at a specific uncovered staircase. Returns dict(path,reached,dead,steps,recs)."""
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    goal = env.goal.detach().cpu().numpy()
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    hist, path, recs = [], [st[:2].copy()], []
    reached = dead = False
    cx = cy = 0                                                     # committed R/U crossings (for target indexing)
    for t in range(T):
        grid_np = GF.axis_grid(st[:2], obs, rr)
        l5_np = GF.low5(st, goal, gamma)
        h_np = GF.hist_pad(np.array(hist[-GF.K_HIST:]) if hist else np.zeros((0, 2)), GF.K_HIST)
        gT = torch.tensor(grid_np, device=device); lT = torch.tensor(l5_np, device=device)
        hT = torch.tensor(h_np, device=device)
        if tilt is None:
            U = policy.sample_window(gT, lT, hT, n=1, temp=temp, nfe=nfe)[0].detach().cpu().numpy()
        else:
            Ucand = policy.sample_window(gT, lT, hT, n=tilt["N"], temp=tilt.get("temp", 1.1),
                                         nfe=nfe, churn=tilt.get("churn", 0.05))
            if tilt.get("broad", 0) > 0:
                Ucand = torch.cat([Ucand, broad_proposal(st, goal, env, tilt["broad"], device)], 0)
            pdir = None
            if style_rho is not None:                              # constant right/up ratio for the whole traj
                pdir = np.array([style_rho, 1.0 - style_rho], np.float32)
                pdir = pdir / (np.linalg.norm(pdir) + 1e-9)
                Ucand = torch.cat([Ucand, broad_targeted(st, pdir, env, tilt.get("n_target", 32), device)], 0)
            elif target is not None:                               # next prescribed move R/U of the target word
                k = cx + cy
                pdir = (np.array([1.0, 0.0], np.float32) if (k < 10 and target[k] == "R")
                        else np.array([0.0, 1.0], np.float32)) if k < 10 else (goal - st[:2])
                pdir = pdir / (np.linalg.norm(pdir) + 1e-9)
                Ucand = torch.cat([Ucand, broad_targeted(st, pdir, env, tilt.get("n_target", 32), device)], 0)
            Uc_np = Ucand.detach().cpu().numpy()
            if tilt.get("safe_filter", True):                      # keep collision-free / in-task candidates
                m = safe_mask(st, Uc_np, obs, rr, env.dt)
                if m.any():
                    keep = np.where(m)[0]
                    Ucand = Ucand[torch.as_tensor(keep, device=Ucand.device)]; Uc_np = Uc_np[keep]
            if tilt.get("feature", "phi_s") == "rawU":              # context-invariant control-content feature
                feat = Ucand.reshape(Ucand.shape[0], -1) / policy.u_max
            else:
                feat = policy.phi_s_at(Ucand, gT, lT, hT, s=tilt["s"])   # Eq-10 entangled features
            sig = tilt["unc"].sigma(feat)                           # GP σ (window-to-window novelty)
            w = torch.exp(((sig - sig.max()) / max(tilt["beta"], 1e-6)).clamp(-30, 30))   # Eq-9 tilt
            if pdir is not None:                                    # bias toward the target's prescribed direction
                net = di_rollout_batch(st, Uc_np, env.dt)[:, -1, :] - st[:2]
                align = torch.as_tensor((net @ pdir).astype(np.float32), device=w.device)
                w = w * torch.exp((align / max(tilt.get("align_temp", 0.2), 1e-3)).clamp(-20, 20))
            U = Ucand[systematic_resample(w, 1)[0]].detach().cpu().numpy()
        a = U[0]
        if record and verify_fn is not None:
            recs.append((grid_np, l5_np, h_np, U.astype(np.float32),
                         bool(verify_fn(st, U, env, gamma))))
        px, py = st[0], st[1]
        st = di_step(st, np.asarray(a, np.float32), dt=env.dt)
        hist.append(np.asarray(a, np.float32))
        path.append(st[:2].copy())
        while cx < 5 and st[0] >= cx + 1 - 1e-6:                    # advance committed crossings
            cx += 1
        while cy < 5 and st[1] >= cy + 1 - 1e-6:
            cy += 1
        if np.linalg.norm(st[:2] - goal) < reach:
            reached = True; break
        if (st[:2] < -GM.EPS_TASK).any() or (st[:2] > GM.GRID_M + GM.EPS_TASK).any():
            dead = True; break
        if (np.linalg.norm(st[:2][None] - obs[:, :2], axis=1) - obs[:, 2] - rr).min() < 0.0:
            dead = True; break
    return dict(path=np.array(path, np.float32), reached=reached, dead=dead, steps=len(path) - 1, recs=recs)


def deploy_many(policy, env, gamma, n, T=250, temp=1.0, nfe=8, device="cpu"):
    """n plain (no-tilt) deploys -> list of paths (for coverage/validity measurement)."""
    return [fm_deploy(policy, env, gamma, T=T, temp=temp, nfe=nfe, device=device)["path"] for _ in range(n)]


if __name__ == "__main__":
    import time
    import grid_scene as GS
    import grid_policy as GP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid(); pol = GP.build_policy(device=dev)
    t0 = time.time(); r = fm_deploy(pol, env, 0.5, T=60, nfe=8, record=True, device=dev)
    print(f"plain deploy: steps={r['steps']} reached={r['reached']} dead={r['dead']} recs={len(r['recs'])} "
          f"{time.time()-t0:.2f}s  (untrained policy)")
    from uncertainty import GPUncertainty
    unc = GPUncertainty(kernel="linear", lengthscale=0.2, lam=1e-2, normalize=True); unc.set_buffer(None)
    t0 = time.time()
    r = fm_deploy(pol, env, 0.5, T=60, nfe=6, tilt=dict(unc=unc, beta=0.077, N=32, s=0.9, broad=16),
                  record=True, device=dev)
    print(f"tilt deploy:  steps={r['steps']} recs={len(r['recs'])} pos-labels={sum(x[4] for x in r['recs'])} "
          f"{time.time()-t0:.2f}s")
