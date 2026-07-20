"""Reusable di_grid-style renderer (the viz method from overnight_run_2026-06-28/di_grid.py).

Row A (scene): polytope level sets {H_P≥(1−γ)^i} — BLUE nominal (SafeMPPI) or a GREEN verifier overlay
(Stage 3, via `verifier_fn`) — + obstacles + accept(green)/reject(red) rollout trajectories + orange
centroid arrow + red executed path + a PERSISTENT BLACK-DOT trail at every executed state + robot + goal
+ accept/reject box.
Row B (control accel): the 3-mode proposal samples (Mode A warm=blue, B opening=green, C backup=magenta),
accepted `o` / rejected `x`, orange μ/Σ ellipse, executed navy ✗.
γ = columns. Saves gif (PillowWriter) AND mp4 (FFMpegWriter; ffmpeg present).

`mppi_rollout` also records `uwin = adapter._u_prev` [H,2] (the reward-weighted planned window) per step,
so the same rollout feeds the Stage-2 windowed dataset.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import _paths
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter

MODE_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff00ff"}   # A warm / B opening / C backup
MODE_SIZE = {0: 6, 1: 6, 2: 44}


def load_best_config() -> dict:
    with open(_paths.BEST_CONFIG) as f:
        return json.load(f)["config"]


def di_step(s, u, dt=0.1):
    s = np.asarray(s, np.float32); u = np.asarray(u, np.float32)
    return np.array([s[0] + dt * s[2] + 0.5 * dt * dt * u[0], s[1] + dt * s[3] + 0.5 * dt * dt * u[1],
                     s[2] + dt * u[0], s[3] + dt * u[1]], np.float32)


def H_grid_nominal(poly, GX, GY):
    """Nominal polytope barrier field H_P(x)=min_k (b_k − a_k·x)/margin_k. poly=(A,b,c,margins)."""
    A, b, _c, margins = poly
    A = np.asarray(A); b = np.asarray(b); margins = np.maximum(np.asarray(margins), 1e-6)
    pts = np.stack([GX.ravel(), GY.ravel()], 1)
    return ((b[None] - pts @ A.T) / margins[None]).min(1).reshape(GX.shape)


def mppi_rollout(env, gamma, cfg, steps=None, seed_base=0, reach_thresh=0.4):
    """Receding-horizon SafeMPPI rollout. Returns (records[list per step], path[T+1,2])."""
    ad = SafeMPPIAdapter(**cfg)
    steps = steps or env.T
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    goal_t = env.goal.detach().cpu().float()
    obs_t = env.obstacles.detach().cpu().float()
    rec, path, reached = [], [st[:2].copy()], False
    for t in range(steps):
        if not reached:
            a, info = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_t,
                              gamma=gamma, seed=seed_base + t, return_rollouts=True)  # static: no obs_vel
            dr = info["debug_rollouts"]
            nrej = int(info["num_barrier_violations"]); rate = float(info["infeasibility_rate"])
            ntot = int(round(nrej / rate)) if rate > 1e-9 else cfg["num_samples"]
            fc = np.asarray(info["first_controls"])
            mode = info.get("sample_mode")
            mode = np.asarray(mode) if mode is not None else np.zeros(len(fc), int)
            uwin = ad._u_prev.detach().cpu().numpy() if getattr(ad, "_u_prev", None) is not None else None
            rec.append(dict(
                p=st[:2].copy(), traj=np.asarray(dr["states"]), feas=np.asarray(dr["feasible"], bool),
                n_acc=max(0, ntot - nrej), n_rej=nrej, poly=info["polytope"], fc=fc,
                fcf=np.asarray(info["feasible"], bool), smean=np.asarray(info["sample_mean"]),
                exec=np.asarray(info["mean_control"]), sigma=np.asarray(info["sigma"]),
                cpos=info.get("centroid_pos"), pmix=float(info.get("mixture_p", 0.0) or 0.0),
                mode=mode, size=info.get("polytope_size"), uwin=uwin))
            st = di_step(st, a.detach().cpu().numpy(), dt=env.dt)
            if np.linalg.norm(st[:2] - env.goal.detach().cpu().numpy()) < reach_thresh:
                reached = True
        else:
            rec.append({**rec[-1], "p": st[:2].copy()})
        path.append(st[:2].copy())
    return rec, np.array(path)


def _draw_scene(ax, env, rec, path, f, g, polytope_mode, verifier_fn, n_show=60):
    ax.clear()                                                          # redraw each frame
    st = rec[min(f, len(rec) - 1)]
    p = st["p"]
    xl, yl = env.xlim, env.ylim
    if polytope_mode == "nominal" and st["poly"] is not None:
        gx = np.linspace(*xl, 130); gy = np.linspace(*yl, 90)
        GX, GY = np.meshgrid(gx, gy)
        Hh = H_grid_nominal(st["poly"], GX, GY)
        lv = sorted({round((1 - g) ** i, 4) for i in range(8)} | {0.0})
        ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.45, zorder=1)
        ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.1, zorder=3)
    elif polytope_mode == "verifier" and verifier_fn is not None:
        verifier_fn(ax, env, st, p, g)                                  # draws GREEN verifier polytope
    for (ox, oy, rr) in env.obstacles.detach().cpu().numpy():
        ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", alpha=0.7, edgecolor="#7b3294", lw=0.6, zorder=4))
    traj, feas = st["traj"], st["feas"]
    for k in np.where(~feas)[0][:n_show]:
        ax.plot(traj[k, :, 0], traj[k, :, 1], "-", color="#d62728", lw=0.4, alpha=0.3, zorder=5)
    for k in np.where(feas)[0][:n_show]:
        ax.plot(traj[k, :, 0], traj[k, :, 1], "-", color="#00a000", lw=0.7, alpha=0.85, zorder=7)
    if st.get("cpos") is not None:
        cp = st["cpos"]
        ax.annotate("", xy=(cp[0], cp[1]), xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=1.6), zorder=9)
    hist = path[:min(f, len(path) - 1) + 1]
    ax.plot(hist[:, 0], hist[:, 1], "-", color="#e6191b", lw=1.5, zorder=8)
    ax.scatter(hist[:, 0], hist[:, 1], s=12, c="k", zorder=8.5)          # PERSISTENT BLACK-DOT TRAIL
    ax.scatter([p[0]], [p[1]], s=40, c="#00a000", edgecolor="k", zorder=10)
    ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=130, c="gold", edgecolor="k", zorder=10)
    ax.text(0.02, 0.97, f"acc {st['n_acc']}/rej {st['n_rej']}\np={st['pmix']:.2f}", transform=ax.transAxes,
            va="top", fontsize=7, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def _draw_control(axc, rec, f, u_max=2.0):
    axc.clear()                                                         # redraw each frame
    st = rec[min(f, len(rec) - 1)]
    fc, fcf, mode = st["fc"], st["fcf"], st["mode"]
    sm, sg = st["smean"], st["sigma"]
    axc.axhline(0, color="#ddd", lw=0.5); axc.axvline(0, color="#ddd", lw=0.5)
    for m, col in MODE_COLORS.items():
        sel = mode == m
        acc, rej = sel & fcf, sel & ~fcf
        axc.scatter(fc[acc, 0], fc[acc, 1], s=MODE_SIZE[m], c=col, marker="o", alpha=0.6, zorder=2)
        axc.scatter(fc[rej, 0], fc[rej, 1], s=MODE_SIZE[m], c=col, marker="x", alpha=0.5, zorder=2)
    axc.add_patch(Ellipse((sm[0], sm[1]), 2 * sg[0], 2 * sg[1], facecolor="none", edgecolor="#ff7f00", lw=1.3, zorder=3))
    axc.annotate("", xy=(sm[0], sm[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=1.4), zorder=4)
    axc.scatter([st["exec"][0]], [st["exec"][1]], s=80, c="#08306b", marker="X", zorder=5)
    lim = u_max * 1.15
    axc.set_xlim(-lim, lim); axc.set_ylim(-lim, lim); axc.set_aspect("equal")
    axc.set_xticks([-2, 0, 2]); axc.set_yticks([-2, 0, 2]); axc.tick_params(labelsize=6)


def render_grid(env, data, gammas, out_path, polytope_mode="nominal", verifier_fn=None,
                title="", show_control=True, fps=6, mp4=True, log=print):
    """data: {gamma: (records, path)}. Renders the di_grid grid to gif (+mp4)."""
    nF = max(len(data[gammas[0]][0]), 1)
    C = len(gammas)
    R = 2 if show_control else 1
    fig, axes = plt.subplots(R, C, figsize=(3.7 * C, 3.2 * R), squeeze=False)

    def draw(f):
        for ci, g in enumerate(gammas):
            rec, path = data[g]
            _draw_scene(axes[0][ci], env, rec, path, f, g, polytope_mode, verifier_fn)
            axes[0][ci].set_title(f"γ={g}", fontsize=11)
            if show_control:
                _draw_control(axes[1][ci], rec, f, u_max=float(env.u_max))
                axes[1][ci].set_title("3-mode accel (o acc / x rej)", fontsize=8)
        fig.suptitle(f"{title}   t={f}", fontsize=11)
        return []

    anim = FuncAnimation(fig, draw, frames=nF, interval=200)
    gif_path = out_path if out_path.endswith(".gif") else out_path + ".gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=90)
    log(f"saved {gif_path}")
    if mp4:
        mp4_path = gif_path[:-4] + ".mp4"
        try:
            anim.save(mp4_path, writer=FFMpegWriter(fps=max(fps, 10), bitrate=2400), dpi=110)
            log(f"saved {mp4_path}")
        except Exception as exc:
            log(f"[mp4] skipped ({exc})")
    draw(nF // 2)
    png = gif_path[:-4] + ".png"
    fig.savefig(png, dpi=120)
    log(f"saved {png}")
    plt.close(fig)
    return gif_path
