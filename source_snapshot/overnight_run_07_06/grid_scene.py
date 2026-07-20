"""5x5 m obstacle-grid scene + mode-1 (Gaussian) SafeMPPI config + light rollout + success test."""
from __future__ import annotations

import numpy as np
import torch

import _paths  # noqa: F401
from dynamics import Env
from di_grid_viz import load_best_config, di_step
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter

GRID_M = 5.0          # 5 m x 5 m
OBS_R = 0.2           # obstacle radius
R_ROBOT = 0.0         # robot radius (point robot: planner == robot, no body-inflation mismatch)
RANGE_M = 2.0         # sensing / barrier range


WALL_OFF = 0.2        # boundary-wall circles sit this far OUTSIDE the [0,GRID] edge (inner edge ~ on the boundary)
PLAN_MARGIN = 0.0     # planner buffer; 0 => polytope hugs the TRUE obstacles (no constant robot<->obstacle offset)
N_INTERIOR = 16       # first 16 obstacles are the 4x4 block; the rest are boundary-wall circles
N_EDGE = 14           # circles per outer edge (spacing ~0.38 m <= 2r => touching => polytope does not leak out)


def make_grid(obs_r=OBS_R, r_robot=R_ROBOT, u_max=1.0, dt=0.1, T=250, walls=True, n_edge=N_EDGE,
              corner_gap=0.55, device="cpu"):
    """4x4 obstacles at the interior vertices {1,2,3,4}^2 (r=0.2) + `n_edge` touching circles per outer edge
    (dense => the nominal/verifier polytope is tightly bounded, no out-of-grid leak) with a small opening at
    the start/goal corners (the robot must enter/exit there); start (0,0) -> goal (5,5)."""
    xs = [1.0, 2.0, 3.0, 4.0]
    obs = [[x, y, obs_r] for x in xs for y in xs]                       # 16 interior obstacles
    if walls:
        corners = [(0.0, 0.0), (GRID_M, GRID_M)]                        # keep start/goal openings
        for p in np.linspace(0.0, GRID_M, n_edge):
            for (cx, cy) in [(p, -WALL_OFF), (p, GRID_M + WALL_OFF), (-WALL_OFF, p), (GRID_M + WALL_OFF, p)]:
                if all((cx - kx) ** 2 + (cy - ky) ** 2 > corner_gap ** 2 for (kx, ky) in corners):
                    obs.append([float(cx), float(cy), obs_r])
    obs_t = torch.tensor(obs, dtype=torch.float32)
    env = Env(name="grid5", x0=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
              goal=torch.tensor([GRID_M, GRID_M], dtype=torch.float32),
              obstacles=obs_t, obs_vel=torch.zeros(len(obs_t), 2, dtype=torch.float32),
              T=T, dt=dt, u_max=u_max, r_robot=r_robot,
              xlim=(-0.7, GRID_M + 0.7), ylim=(-0.7, GRID_M + 0.7))
    return env.to(device)


def mode1_config(range_m=RANGE_M, u_max=1.0, noise_var_mult=3.0):
    """Frozen best config, switched to MODE 1 = plain Gaussian sampling, sensing range = 2 m,
    u_max halved (2->1), and the sampling VARIANCE tripled (sigma^2 x3 => sigma 0.5 -> 0.5*sqrt3 ~ 0.87)."""
    cfg = dict(load_best_config())
    cfg["polytope_area_sampling"] = False      # mode 1: Gaussian proposal (not polytope-area importance sampling)
    cfg["urgency_size_diff"] = False           # mode-1 urgency (magnitude, not shrink-rate)
    cfg["barrier_activation_radius"] = float(range_m)
    cfg["u_min"] = [-float(u_max), -float(u_max)]     # halve the control authority
    cfg["u_max"] = [float(u_max), float(u_max)]
    cfg["noise_sigma"] = [0.5 * (noise_var_mult ** 0.5)] * 2   # triple the sampling variance
    return cfg


def planner_obstacles(env, margin=PLAN_MARGIN):
    """Obstacles (interior + grey walls) grown by (r_robot + planning margin) so the polytope the planner
    builds keeps the point robot clear and absorbs DI overshoot. The walls thus constrain the polytope."""
    obs = env.obstacles.detach().cpu().clone()
    obs[:, 2] = obs[:, 2] + float(env.r_robot) + margin
    return obs


def inflated_env(env):
    """A copy of `env` whose obstacles are inflated by r_robot (used for planning in the sweep viz)."""
    import dataclasses
    return dataclasses.replace(env, obstacles=planner_obstacles(env).to(env.obstacles.device))


def rollout_path(env, gamma, cfg, seed, reach=0.4, inflate=True):
    """Light receding-horizon rollout (no debug rollouts) -> executed path [T+1,2].
    `inflate`: plan against obstacles grown by r_robot (collision is still checked at the true radius)."""
    ad = SafeMPPIAdapter(**cfg)
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    goal_t = env.goal.detach().cpu().float()
    obs_t = planner_obstacles(env) if inflate else env.obstacles.detach().cpu().float()
    goal = env.goal.detach().cpu().numpy()
    path, reached = [st[:2].copy()], False
    for t in range(env.T):
        if not reached:
            a, _ = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_t, gamma=gamma, seed=seed * 1000 + t)
            st = di_step(st, a.detach().cpu().numpy(), dt=env.dt)
            if np.linalg.norm(st[:2] - goal) < reach:
                reached = True
        path.append(st[:2].copy())
    return np.array(path, np.float32)


def is_success(path, env, reach=0.45, margin=0.12):
    """Success = reach the goal AND collision-free AND stay ON the 5x5 grid (off-grid is a failure)."""
    p = np.asarray(path, float)
    goal = env.goal.detach().cpu().numpy()
    reached = np.linalg.norm(p - goal, axis=1).min() < reach
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    clear = (np.linalg.norm(p[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - rr).min()
    on_grid = bool((p >= -margin).all() and (p <= GRID_M + margin).all())
    return bool(reached and clear >= 0.0 and on_grid), float(clear)
