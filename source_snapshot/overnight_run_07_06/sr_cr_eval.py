"""Success-rate & collision-rate from the ORIGIN, per gamma (2026-07-07, the 07_06 primary metric).

Inference-time receding-horizon rollouts via GR.fm_deploy (tilt=None = faithful, no expansion). Grid is a
point robot (r_robot=0, OBS_R=0.2), start = env.x0 = (0,0) (OOD for the off-diagonal base).
  SR  = reached <= `reach` (0.1 m) of goal within `T_max` steps AND collision-free   (== stage_e_benchmark)
  CR  = the executed path enters an obstacle (min clearance < 0 at any step)
Also reports out-of-bounds and timeout fractions and mean final goal-distance. Importable: `eval_policy`.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
import grid_scene as GS
import grid_rollout as GR
import grid_hp_expt as HP

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def path_collides(path, env):
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    if not len(obs):
        return False
    d = np.linalg.norm(path[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - rr
    return bool((d.min(1) < 0.0).any())


def eval_policy(pol, env, gammas=GAMMAS, M=32, T_max=250, reach=0.1, temp=1.0, device="cpu",
                seed0=0, keep_paths=0, log=print):
    """Returns (rows: {γ: metrics}, agg, paths: {γ: [np arrays]}). keep_paths>0 stores that many paths/γ."""
    goal = env.goal.detach().cpu().numpy()
    rows, paths = {}, {}
    for g in gammas:
        sr = cr = oob = to = 0
        gd = []; kept = []
        for i in range(M):
            torch.manual_seed(seed0 + i)
            out = GR.fm_deploy(pol, env, float(g), T=T_max, temp=temp, tilt=None, reach=reach, device=device)
            p = out["path"]
            coll = path_collides(p, env)
            reached = bool(out["reached"]) and not coll
            gd.append(float(np.linalg.norm(p[-1] - goal)))
            if reached:
                sr += 1
            if coll:
                cr += 1
            elif out["dead"]:
                oob += 1
            elif not out["reached"]:
                to += 1
            if len(kept) < keep_paths:
                kept.append(p)
        rows[g] = dict(SR=sr / M, CR=cr / M, oob=oob / M, timeout=to / M,
                       mean_goal_dist=float(np.mean(gd)), M=M)
        paths[g] = kept
        log(f"γ{g}: SR {sr/M:.2f}  CR {cr/M:.2f}  oob {oob/M:.2f}  timeout {to/M:.2f}  "
            f"gdist {np.mean(gd):.2f}", flush=True)
    agg = dict(SR=float(np.mean([rows[g]["SR"] for g in gammas])),
               CR=float(np.mean([rows[g]["CR"] for g in gammas])),
               mean_goal_dist=float(np.mean([rows[g]["mean_goal_dist"] for g in gammas])))
    log(f"[AGG] SR {agg['SR']:.3f}  CR {agg['CR']:.3f}  gdist {agg['mean_goal_dist']:.2f}", flush=True)
    return rows, agg, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--T-max", type=int, default=250)
    ap.add_argument("--reach", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol, ck = HP.load_hp(args.ckpt, device=dev)
    env = GS.make_grid()
    print(f"[sr_cr] ckpt {os.path.basename(args.ckpt)} | repr {ck['config'].get('repr_dim')} "
          f"grid {ck['config'].get('grid_shape')} | M={args.M} reach={args.reach} T_max={args.T_max}", flush=True)
    rows, agg, _ = eval_policy(pol, env, gammas=args.gammas, M=args.M, T_max=args.T_max,
                               reach=args.reach, temp=args.temp, device=dev)
    out = args.out or os.path.join(HERE, "results", "hp_repr",
                                   f"srcr_{os.path.splitext(os.path.basename(args.ckpt))[0]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"ckpt": args.ckpt, "config": ck["config"], "reach": args.reach, "T_max": args.T_max,
                   "rows": {str(k): v for k, v in rows.items()}, "agg": agg}, f, indent=2)
    print(f"[sr_cr] saved {out}", flush=True)


if __name__ == "__main__":
    main()
