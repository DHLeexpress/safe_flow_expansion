"""2-D double-integrator dynamics + the two fixed overfit environments (spec 3.3).

State x = [px, py, vx, vy], control u = [ax, ay] (box-limited acceleration).
    p_{t+1} = p_t + dt v_t + 0.5 dt^2 u_t
    v_{t+1} = v_t + dt u_t
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Env:
    """A fixed problem instance c = (x0, goal, obstacles)."""
    name: str
    x0: torch.Tensor            # [4]  (px,py,vx,vy)
    goal: torch.Tensor          # [2]
    obstacles: torch.Tensor     # [N,3] (cx,cy,radius)
    obs_vel: torch.Tensor       # [N,2] obstacle velocities (0 = static)
    T: int = 40                 # horizon (steps)
    dt: float = 0.1
    u_max: float = 3.0
    r_robot: float = 0.2
    # plotting window
    xlim: tuple = (-1.0, 7.0)
    ylim: tuple = (-3.5, 3.5)

    def to(self, device) -> "Env":
        return Env(
            name=self.name,
            x0=self.x0.to(device), goal=self.goal.to(device),
            obstacles=self.obstacles.to(device), obs_vel=self.obs_vel.to(device),
            T=self.T, dt=self.dt, u_max=self.u_max, r_robot=self.r_robot,
            xlim=self.xlim, ylim=self.ylim,
        )

    @property
    def n_obs(self) -> int:
        return self.obstacles.shape[0]

    def obstacle_centers_over_time(self) -> torch.Tensor:
        """Constant-velocity prediction -> [T+1, N, 2]."""
        steps = torch.arange(self.T + 1, device=self.obstacles.device).float() * self.dt
        c0 = self.obstacles[:, :2]                       # [N,2]
        return c0[None] + steps[:, None, None] * self.obs_vel[None]   # [T+1,N,2]


def rollout(U: torch.Tensor, env: Env) -> torch.Tensor:
    """Roll out control sequences.

    U:   [B, T, 2] accelerations (already clipped to box)
    ret: states [B, T+1, 4]
    """
    B = U.shape[0]
    dt = env.dt
    x = env.x0.to(U.device).expand(B, 4).clone()
    out = [x]
    for t in range(env.T):
        u = U[:, t]
        p, v = x[:, :2], x[:, 2:4]
        p_next = p + dt * v + 0.5 * dt * dt * u
        v_next = v + dt * u
        x = torch.cat([p_next, v_next], dim=1)
        out.append(x)
    return torch.stack(out, dim=1)                       # [B,T+1,4]


def clip_controls(U: torch.Tensor, env: Env) -> torch.Tensor:
    return U.clamp(-env.u_max, env.u_max)


# --------------------------------------------------------------------------- envs

def make_env(name: str, device="cpu") -> Env:
    if name == "single":
        # ENV-A: one obstacle on the start->goal line  => left/right dilemma (bimodal Omega*)
        env = Env(
            name="single",
            x0=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            goal=torch.tensor([6.0, 0.0]),
            obstacles=torch.tensor([[3.0, 0.0, 0.8]]),
            obs_vel=torch.zeros(1, 2),
            T=40, dt=0.12, u_max=3.5, r_robot=0.2,
            xlim=(-1.0, 7.0), ylim=(-3.0, 3.0),
        )
    elif name == "gap":
        # ENV-B: two stacked obstacles, narrow passable gap => left/middle/right (trimodal)
        # gap clearance for robot center: |y -/+ 1.0| >= 0.6+0.2=0.8  => |y|<=0.2 through gap
        env = Env(
            name="gap",
            x0=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            goal=torch.tensor([6.0, 0.0]),
            obstacles=torch.tensor([[3.0, 1.0, 0.6],
                                    [3.0, -1.0, 0.6]]),
            obs_vel=torch.zeros(2, 2),
            T=40, dt=0.12, u_max=3.5, r_robot=0.2,
            xlim=(-1.0, 7.0), ylim=(-3.2, 3.2),
        )
    else:
        raise ValueError(f"unknown env {name!r} (use 'single' or 'gap')")
    return env.to(device)
