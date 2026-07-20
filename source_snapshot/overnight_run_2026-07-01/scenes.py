"""Scene factory at the frozen best_area_mode4 scale (double-integrator, u_max=2, dt=0.1, static).

Reuses the `Env` dataclass from overnight_run_today/src/dynamics.py so the same scenes drive the
planner (Stage 1), the windowed dataset (Stage 2) and the FM policy / verifier (Stage 3).

`narrow_gap`: two stacked static obstacles at x=3 forming a tight passable gap; start (0,0)→goal (6,0)
is a straight line through the gap center, so a CONSERVATIVE planner must choose to thread vs detour.
`threads_gap(path, env)` reports which it did.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import _paths  # noqa: F401
from dynamics import Env  # overnight_run_today/src


def make_narrow_gap(gap_offset: float = 0.75, gap_r: float = 0.35, goal_x: float = 6.0,
                    T: int = 80, dt: float = 0.1, u_max: float = 2.0, r_robot: float = 0.2,
                    device: str = "cpu") -> Env:
    """Two obstacles at (3, ±gap_offset) radius gap_r. Robot-center corridor half-width =
    gap_offset - gap_r - r_robot (default 0.75-0.35-0.2 = 0.20 m: passable but tight)."""
    obs = torch.tensor([[3.0, gap_offset, gap_r], [3.0, -gap_offset, gap_r]], dtype=torch.float32)
    env = Env(
        name="narrow_gap",
        x0=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        goal=torch.tensor([goal_x, 0.0], dtype=torch.float32),
        obstacles=obs, obs_vel=torch.zeros(2, 2, dtype=torch.float32),
        T=T, dt=dt, u_max=u_max, r_robot=r_robot,
        xlim=(-1.0, goal_x + 1.0), ylim=(-3.0, 3.0),
    )
    return env.to(device)


def make_single_obstacle(r: float = 0.6, ox: float = 3.0, goal_x: float = 6.0,
                         T: int = 80, dt: float = 0.1, u_max: float = 2.0, r_robot: float = 0.2,
                         device: str = "cpu") -> Env:
    """One obstacle at (ox,0) on the start→goal line → forces an above/below (left/right) choice."""
    obs = torch.tensor([[ox, 0.0, r]], dtype=torch.float32)
    env = Env(name="single", x0=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
              goal=torch.tensor([goal_x, 0.0], dtype=torch.float32),
              obstacles=obs, obs_vel=torch.zeros(1, 2, dtype=torch.float32),
              T=T, dt=dt, u_max=u_max, r_robot=r_robot,
              xlim=(-1.0, goal_x + 1.0), ylim=(-3.0, 3.0))
    return env.to(device)


def make_slalom(ax: float = 2.4, ay: float = 0.45, bx: float = 3.6, by: float = -0.45, r: float = 0.55,
                goal_x: float = 6.0, T: int = 80, dt: float = 0.1, u_max: float = 2.0, r_robot: float = 0.0,
                device: str = "cpu") -> Env:
    """左右 (sequential) obstacles: A upper-left (ax,ay), B lower-right (bx,by). SafeMPPI sweeps
    above/below both (go-around); a weave between them is the mode expansion can discover."""
    obs = torch.tensor([[ax, ay, r], [bx, by, r]], dtype=torch.float32)
    env = Env(name="slalom", x0=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
              goal=torch.tensor([goal_x, 0.0], dtype=torch.float32),
              obstacles=obs, obs_vel=torch.zeros(2, 2, dtype=torch.float32),
              T=T, dt=dt, u_max=u_max, r_robot=r_robot,
              xlim=(-1.0, goal_x + 1.0), ylim=(-3.0, 3.0))
    return env.to(device)


def gap_geometry(env: Env):
    """(gap_x, half_center, half_body, outer) for the narrow_gap env.
    half_center = |off|-r-r_robot (robot CENTER passes safely iff |y|<=half_center);
    half_body   = |off|-r (point-robot corridor edge);
    outer       = |off|+r+r_robot (robot passes safely AROUND an obstacle iff |y|>=outer)."""
    ox = float(env.obstacles[0, 0])
    off = float(env.obstacles[0, 1].abs())
    r = float(env.obstacles[0, 2])
    rr = float(env.r_robot)
    return ox, off - r - rr, off - r, off + r + rr


def threads_gap(path, env: Env) -> dict:
    """Does the executed path pass BETWEEN the two gap obstacles? path: [T+1,2] positions (np/tensor)."""
    p = np.asarray(path if not torch.is_tensor(path) else path.detach().cpu().numpy(), dtype=float)
    gap_x, half_center, half_body, outer = gap_geometry(env)
    cross_y = None
    for i in range(len(p) - 1):
        x0, x1 = p[i, 0], p[i + 1, 0]
        if (x0 - gap_x) * (x1 - gap_x) <= 0 and x1 != x0:      # segment crosses x = gap_x
            t = (gap_x - x0) / (x1 - x0)
            cross_y = float(p[i, 1] + t * (p[i + 1, 1] - p[i, 1]))
            break
    if cross_y is None:                                          # never reached the gap plane
        return dict(reached_gap_plane=False, threaded=False, went_around=False,
                    cross_y=None, half_center=half_center, half_body=half_body, gap_x=gap_x)
    threaded = abs(cross_y) <= half_center + 1e-6                # safely through the gap
    went_around = abs(cross_y) >= outer - 1e-6                   # safely around an obstacle
    return dict(reached_gap_plane=True, threaded=bool(threaded), went_around=bool(went_around),
                cross_y=cross_y, half_center=half_center, half_body=half_body, gap_x=gap_x)


if __name__ == "__main__":
    e = make_narrow_gap()
    gx, hc, hb, outer = gap_geometry(e)
    print(f"narrow_gap: obstacles {e.obstacles.tolist()}")
    print(f"gap_x={gx}  half_center={hc:.2f} (robot-center corridor)  half_body={hb:.2f}  outer={outer:.2f}")
    # straight line y=0 through the gap -> threaded
    straight = np.stack([np.linspace(0, 6, e.T + 1), np.zeros(e.T + 1)], 1)
    print("straight y=0 :", threads_gap(straight, e))
    # detour around the top obstacle
    s = np.linspace(0, 1, e.T + 1)
    detour = np.stack([6 * s, 1.4 * np.sin(np.pi * s)], 1)
    print("detour       :", threads_gap(detour, e))
