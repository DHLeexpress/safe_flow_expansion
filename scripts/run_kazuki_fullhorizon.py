#!/usr/bin/env python3
"""Full-horizon Kazuki-style unified framework with OUR pretrained FM as the
multi-modal proposer — the closest faithful realization their method admits
on this task.

Their reference (external_data/kazuki_cfm_mppi, eval_cfm_mppi_doubleintegrator.py)
plans the ENTIRE episode at once: HORIZON=80 equals the full task, the flow
proposes complete start->goal control sequences, FlowMPPI refines them with the
TRUE terminal-goal cost, one action is executed, and the single best plan is
diluted (0.8 prev + 0.2 noise) and re-projected through the flow from tau=0.8.
Our B1 pretrained FM proposes only H=10 windows, so the per-step flow
re-projection of a full-length sequence is structurally impossible.  This
variant keeps every other axis of their design:

  propose    N=200 full-length (H_full=250) candidate control sequences by
             batched autoregressive window chaining of the bare policy at
             temperature 1 (each candidate = one raw multi-modal rollout,
             per-candidate low7 context, gamma_ctx=0.5).
  refine     their FlowMPPI scheme at FULL horizon: top-10 elites by summed
             stage cost with the true terminal-goal term, 200 Gaussian copies
             per elite (sigma 0.2), per-mode softmax (lambda 0.1) refit,
             execute the argmin refined mode's first action.
  warm start single-best dilution, faithful to their broadcast form:
             candidates <- 0.8 * shifted best + 0.2 * fresh noise (control
             space; the flow re-projection step is the ONE declared deviation).
  re-propose every REGEN steps, half the candidate population is replaced by
             fresh window-chained proposals from the current state (the
             coarse-grained analog of their per-step tau=0.8 manifold
             re-projection; keeps multi-modality available to the refiner).

Question answered: with full-horizon multi-modal proposals and their
refinement/warm-start loop, does the method still fall into the local minimum
around the giant obstacle — or was the H=10 window adaptation the binding
constraint?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKBOOK = Path(__file__).resolve().parents[1]
SNAP = WORKBOOK / "source_snapshot/overnight_run_07_06/rev_expansion/codex_overnight"
sys.path.insert(0, str(SNAP.parents[1]))
sys.path.insert(0, str(SNAP.parent))
sys.path.insert(0, str(SNAP))

import afe_context as CX                            # noqa: E402
from afe2_scene_profiles import build_scene, get_scene_profile  # noqa: E402
from grid_hp_expt import load_hp                    # noqa: E402

METRIC_VERSION = "b1_current_best_gallery_v1"
SCHEMA = "low7_closest_boundary_tie_mean"
GAMMA_CTX = 0.5
NFE = 8

N_SAMPLE = 200
N_ELITE = 10
N_COPY = 200
MPPI_LAMBDA = 0.1
MPPI_SIGMA = 0.2
COLL_W = 100.0
GOAL_W = 0.1
BETA_MPPI = 20.0
R_MARGIN = 0.05
BOUNDS_W = 0.0        # workspace containment repair (NOT in their code); try 100
DILUTE = 0.8          # their noise_level: 0.8 prev + 0.2 noise
H_FULL = 250
REGEN = 25            # re-propose half the population every REGEN steps


def named_seed(*parts) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def di_rollout(state, U, dt):
    """state (4,), U [B,H,2] -> pos [B,H,2], vel [B,H,2]."""
    B, H, _ = U.shape
    p = torch.as_tensor(state[:2], dtype=U.dtype, device=U.device).expand(B, 2).clone()
    v = torch.as_tensor(state[2:], dtype=U.dtype, device=U.device).expand(B, 2).clone()
    ps, vs = [], []
    for t in range(H):
        u = U[:, t]
        p = p + dt * v + 0.5 * dt * dt * u
        v = v + dt * u
        ps.append(p)
        vs.append(v)
    return torch.stack(ps, 1), torch.stack(vs, 1)


def stage_cost(pos, U, goal_t, obs_xy, r_col):
    """Their stage cost (mppi/utils.py:69-98) with true terminal goal term and
    the within-plan smoothness 0.1*||u_t - u_{t-1}||. pos/U [B,H,2] -> [B]."""
    B, H, _ = pos.shape
    goal_c = torch.norm(pos - goal_t[None, None], dim=2)
    d = torch.norm(pos.unsqueeze(2) - obs_xy[None, None], dim=3)
    coll = torch.clamp(torch.exp(-BETA_MPPI * (d - r_col)), max=1.0).sum(2)
    tw = COLL_W * (1.0 + 0.99 ** torch.arange(H, dtype=pos.dtype, device=pos.device))[None]
    prev = torch.cat([torch.zeros_like(U[:, :1]), U[:, :-1]], dim=1)
    smooth = torch.norm(U - prev, dim=2)
    cost = (GOAL_W * goal_c + tw * coll + 0.1 * smooth).sum(1)
    if BOUNDS_W > 0.0:  # declared repair: their cost is open-space, our scene is walled
        viol = (torch.clamp(-pos, min=0.0) + torch.clamp(pos - 5.0, min=0.0)).sum((1, 2))
        cost = cost + BOUNDS_W * viol
    return cost + GOAL_W * torch.norm(pos[:, -1] - goal_t[None], dim=1)


def batched_context(policy, states, hists, goal, env, device):
    """Per-candidate low7 context for [B] diverged states."""
    gs, ls, hs = [], [], []
    for st, hist in zip(states, hists):
        record = CX.build_context(st, goal, GAMMA_CTX, hist, env, SCHEMA)
        gs.append(np.array(record.grid, copy=True))
        ls.append(np.array(record.low5, copy=True))
        hs.append(np.array(record.hist, copy=True))
    gT = torch.tensor(np.stack(gs), dtype=torch.float32, device=device)
    lT = torch.tensor(np.stack(ls), dtype=torch.float32, device=device)
    hT = torch.tensor(np.stack(hs), dtype=torch.float32, device=device)
    return policy.ctx_from(gT, lT, hT)


def chain_proposals(policy, env, state, goal, n, h_full, device):
    """n full-length candidates by batched window chaining at temperature 1."""
    d = policy.d
    H = d // 2
    dt = float(env.dt)
    states = [np.asarray(state, dtype=np.float32).copy() for _ in range(n)]
    hists = [[] for _ in range(n)]
    chunks = []
    with torch.no_grad():
        for _ in range(int(np.ceil(h_full / H))):
            ctx = batched_context(policy, states, hists, goal, env, device)
            z = torch.randn(n, d, device=device)
            for k in range(NFE):
                tt = torch.full((n,), k / NFE, device=device).clamp(1e-4, 1.0)
                v = policy.forward(z, tt, ctx)
                z = z + (1.0 / NFE) * v
            U = torch.clamp(z.reshape(n, H, 2) * policy.u_max,
                            -policy.u_max, policy.u_max)
            chunks.append(U)
            U_np = U.cpu().numpy()
            for i in range(n):
                st = states[i]
                for t in range(H):
                    a = U_np[i, t]
                    st = np.array(
                        [st[0] + dt * st[2] + 0.5 * dt * dt * a[0],
                         st[1] + dt * st[3] + 0.5 * dt * dt * a[1],
                         st[2] + dt * a[0], st[3] + dt * a[1]], np.float32)
                    hists[i].append(a.copy())
                states[i] = st
    return torch.cat(chunks, dim=1)[:, :h_full]


def refine(policy, state, goal_t, obs_xy, r_col, dt, U_cand):
    """Their elite -> perturb -> per-mode softmax -> best refined."""
    with torch.no_grad():
        pos, _ = di_rollout(state, U_cand, dt)
        costs = stage_cost(pos, U_cand, goal_t, obs_xy, r_col)
        _, top = torch.topk(costs, k=min(N_ELITE, U_cand.shape[0]), largest=False)
        elites = U_cand[top]
        E = elites.shape[0]
        pert = elites.repeat_interleave(N_COPY, 0)
        pert = pert + MPPI_SIGMA * torch.randn_like(pert)
        pert = torch.clamp(pert, -policy.u_max, policy.u_max)
        posP, _ = di_rollout(state, pert, dt)
        cP = stage_cost(posP, pert, goal_t, obs_xy, r_col).reshape(E, N_COPY)
        b, _ = cP.min(dim=1, keepdim=True)
        w = torch.softmax(-(cP - b) / MPPI_LAMBDA, dim=1)
        refined = (w[:, :, None, None]
                   * pert.reshape(E, N_COPY, *pert.shape[1:])).sum(1)
        posR, _ = di_rollout(state, refined, dt)
        cR = stage_cost(posR, refined, goal_t, obs_xy, r_col)
        best = int(torch.argmin(cR))
    return refined[best]


def deploy(policy, env, T, reach, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    obs_xy = torch.tensor(obs[:, :2], dtype=torch.float32, device=device)
    r_col = torch.as_tensor(obs[:, 2] + rr + R_MARGIN,
                            dtype=torch.float32, device=device)
    goal = env.goal.detach().cpu().numpy()
    goal_t = torch.tensor(goal, dtype=torch.float32, device=device)
    dt = float(env.dt)
    st = env.x0.detach().cpu().numpy().astype(np.float32)

    U_cand = chain_proposals(policy, env, st, goal, N_SAMPLE, H_FULL, device)
    path = [st[:2].copy()]
    reached = collided = False
    for t in range(T):
        U_best = refine(policy, st, goal_t, obs_xy, r_col, dt, U_cand)
        a = U_best[0].detach().cpu().numpy()
        st = np.array([st[0] + dt * st[2] + 0.5 * dt * dt * a[0],
                       st[1] + dt * st[3] + 0.5 * dt * dt * a[1],
                       st[2] + dt * a[0], st[3] + dt * a[1]], np.float32)
        path.append(st[:2].copy())
        dmin = (np.linalg.norm(st[None, :2] - obs[:, :2], axis=1)
                - obs[:, 2] - rr).min()
        if dmin < 0:
            collided = True
            break
        if np.linalg.norm(st[:2] - goal) < reach:
            reached = True
            break
        if not (0.0 <= st[0] <= 5.0 and 0.0 <= st[1] <= 5.0):
            break  # out of taskspace -> counted as timeout-class failure
        # single-best dilution warm start (their broadcast form)
        U_shift = torch.cat([U_best[1:], U_best[-1:]], 0)
        noise = torch.randn(N_SAMPLE, *U_shift.shape, device=device)
        U_cand = torch.clamp(
            DILUTE * U_shift[None] + (1.0 - DILUTE) * policy.u_max * noise,
            -policy.u_max, policy.u_max,
        )
        # coarse flow re-projection analog: refresh half the population
        if REGEN and (t + 1) % REGEN == 0:
            fresh = chain_proposals(
                policy, env, st, goal, N_SAMPLE // 2, U_shift.shape[0], device
            )
            U_cand = torch.cat([U_cand[: N_SAMPLE - fresh.shape[0]], fresh], 0)
    return dict(path=np.array(path), reached=reached, collided=collided,
                steps=len(path) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-ckpt", type=Path,
        default=Path("/home/dohyun/projects/afe2_runs/low7_groupavg_tiemean_r0_pair_0f0c128/"
                     "seed_20260718_eq_0_ga_1/pretrain/data/checkpoint_candidate.pt"),
    )
    parser.add_argument("--expected-ckpt-sha256",
                        default="524c9c0a4fd071221ac509b9d8e6fbbfb85fdf1811aa04160317f2a9e2d3ef90")
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--reach", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bounds-w", type=float, default=0.0,
                        help="workspace containment repair weight (0 = faithful)")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    global BOUNDS_W
    BOUNDS_W = args.bounds_w

    digest = hashlib.sha256(args.pretrained_ckpt.read_bytes()).hexdigest()
    if digest != args.expected_ckpt_sha256:
        raise RuntimeError(f"checkpoint hash mismatch: {digest}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    env = build_scene(get_scene_profile("low7_radius1_canonical_v1"))
    policy, _ = load_hp(str(args.pretrained_ckpt), device="cpu")
    policy = policy.to(args.device).eval()
    obs = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    rr = float(env.r_robot)

    paths, outcomes = [], []
    t0 = time.time()
    for index in range(args.m):
        seed = named_seed(METRIC_VERSION, "kazuki-fullh", index)
        out = deploy(policy, env, args.T, args.reach, args.device, seed)
        path = np.asarray(out["path"], dtype=np.float32)
        paths.append(path)
        if out["collided"]:
            outcome = "CR"
        elif out["reached"]:
            outcome = "SR"
        else:
            outcome = "TO"
        outcomes.append(outcome)
        done = index + 1
        counts = {k: outcomes.count(k) for k in ("SR", "CR", "TO")}
        print(
            f"[fullh] ep{index:03d} {outcome} steps={out['steps']} | "
            f"SR {counts['SR']/done:.2f} CR {counts['CR']/done:.2f} "
            f"TO {counts['TO']/done:.2f} ({(time.time()-t0)/done:.1f}s/ep)",
            flush=True,
        )
    pa = np.empty(len(paths), dtype=object)
    for i, p in enumerate(paths):
        pa[i] = p
    np.savez_compressed(args.outdir / f"kazuki_fullh_m{args.m}.npz",
                        paths=pa, outcomes=np.array(outcomes))
    counts = {k: outcomes.count(k) for k in ("SR", "CR", "TO")}
    summary = dict(
        metric_version=METRIC_VERSION, scene="low7_radius1_canonical_v1",
        schema=SCHEMA, gamma_ctx=GAMMA_CTX, checkpoint_sha256=digest,
        M=args.m, T=args.T, reach=args.reach, H_full=H_FULL, regen=REGEN,
        dilute=DILUTE, bounds_w=BOUNDS_W,
        n_sample=N_SAMPLE, n_elite=N_ELITE, n_copy=N_COPY,
        SR=counts["SR"] / args.m, CR=counts["CR"] / args.m,
        timeout=counts["TO"] / args.m,
    )
    (args.outdir / "summary_fullh.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
