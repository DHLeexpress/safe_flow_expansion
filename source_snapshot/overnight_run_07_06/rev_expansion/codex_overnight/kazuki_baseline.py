"""Phase 2.5 — Kazuki/Mizuta UnifiedGenRefine (CFM-MPPI) baseline ported onto OUR pretrained FM (a32uni),
NO flow expansion. Faithful to external_data/kazuki_cfm_mppi (file:line refs below), adapted to our grid task:

  generate (guided flow)  eval_utils.run_CFM:17-81
    - 200 samples, ODE knots [0,.5,.8,.85,.9,.92,.94,.96,.98,1.0] (9 steps; warm-started runs start at tau=.8)
    - per knot: endpoint estimate x1 = z + (1-tau)*v  ->  reward grads on x1  ->  grads renormalized to the
      GLOBAL ||v|| (their torch.norm(..., keepdim=True) is a whole-tensor scalar)  ->
      v <- v + goal_coef*g_goal + w_safe*g_cbf*markup(1.01^reversed-t)  ->  Euler step.
    - w_safe in {0.1,0.3,0.5,0.7,0.9}, 40 samples each (SAFE_COEF, eval_cfm_mppi_*.py:26).
  guidance rewards        reward.py:5-70 (FAITHFUL CBF form, not exponential)
    - r_safe = sum_t min{0, hdot + a_cbf*h},  h = ||p-p_o||^2 - r^2, a_cbf=1.0; K=5 worst obstacles by cbf
      value, weighted [5,4,3,2,1]; r_goal = -||p_T - goal|| (terminal).
  select + refine (FlowMPPI)  mppi/flowmppi.py:144-314
    - stage cost = 0.1*goal_dist + 100*(1+0.99^t)*sum_obs clamp(exp(-20*(d - r)),max=1) (utils.py:69-98)
      + terminal 0.1*||p_T-goal|| + 0.1*||u - prev||^2 warm-start consistency
    - top-10 elite -> 200 Gaussian perturbations each (sigma scaled to our u_max: 0.4/2*1=0.2), clamp [-1,1],
      per-mode softmax (lambda=0.1) -> refit; execute argmin refined mode's first action.
  dilution / warm-start   eval_cfm_mppi_*.py:187-201
    - next step z_tau = 0.8*prev_solution + 0.2*noise, resume ODE from tau=0.8 (7 steps). prev solution is
      shifted by the executed step (their history-inpainting replaced by our policy's exec-history context).

Ours-specific: H=10 window (theirs 80), u_max=1.0, DI dynamics directly (their SI->DI conversion unneeded),
64 static obstacles (constant-velocity prediction exact, vel=0), r_col = obs_r + r_robot + 0.05 margin.
Our FM is gamma-conditioned; their pipeline has no gamma -> fixed neutral gamma_ctx=0.5, w_safe is the knob.
No SOCP/verifier gate anywhere (their method has none). No git/wandb.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
sys.path.insert(0, _WORK); sys.path.insert(0, _REV); sys.path.insert(0, _HERE)

import argparse, json, time
import numpy as np
import torch

import _paths  # noqa
import grid_scene as GS
import grid_feats as GF
import grid_hp_expt as HP
import sr_cr_eval as SR
import grid_metrics as GM

NFE = 8                              # OUR linear ODE schedule (user caveat 2): knots i/8, i=0..8
ODE_TIMES_FULL = [i / NFE for i in range(NFE + 1)]
TAU_WARM = 0.75                      # 0.8 does not exist on our linear grid -> nearest knot 0.75
SAFE_COEFS = [0.1, 0.3, 0.5, 0.7, 0.9]
GOAL_COEF = 0.1                      # their GOAL_COEF
A_CBF = 1.0                          # their a_cbf
K_WORST = 5                          # their k=5 worst obstacles
BETA_MPPI = 20.0                     # their alpha=20 exponential proximity steepness
COLL_W = 100.0                       # their collision weight (tunable via --coll-w for fairness variants)
GOAL_W = 0.1                         # their goal weight (tunable via --goal-w)
N_SAMPLE = 200
N_ELITE = 10
N_COPY = 200                         # perturbations per elite
MPPI_LAMBDA = 0.1
MPPI_SIGMA = 0.2                     # their DI sigma 0.4 with u_max 2 -> scaled to our u_max 1
MARKUP = 1.01
R_MARGIN = 0.05


def obstacle_collision_radii(obs, robot_radius, margin, *, device, dtype=torch.float32):
    """Per-obstacle collision radii; heterogeneous scene radii must not be averaged."""
    obs = np.asarray(obs, dtype=np.float32)
    return torch.as_tensor(obs[:, 2] + float(robot_radius) + float(margin),
                           dtype=dtype, device=device)


def di_rollout_t(state, U, dt):
    """Differentiable batched DI rollout. state (4,) np/torch; U [B,H,2] torch -> pos [B,H,2], vel [B,H,2]."""
    B, H, _ = U.shape
    p = torch.as_tensor(state[:2], dtype=U.dtype, device=U.device).expand(B, 2).clone()
    v = torch.as_tensor(state[2:], dtype=U.dtype, device=U.device).expand(B, 2).clone()
    ps, vs = [], []
    for t in range(H):
        u = U[:, t]
        p = p + dt * v + 0.5 * dt * dt * u
        v = v + dt * u
        ps.append(p); vs.append(v)
    return torch.stack(ps, 1), torch.stack(vs, 1)


def cbf_reward(pos, vel, obs_xy, r_col):
    """Faithful reward.py:5-56 adapted to DI: per-sample sum_t of the K=5 worst weighted CBF violations.
    pos/vel [B,H,2]; obs_xy [No,2]. h = d^2 - r^2 (per obstacle), hdot = 2(p-p_o)^T v."""
    d = pos.unsqueeze(2) - obs_xy[None, None]                 # [B,H,No,2]
    h = (d ** 2).sum(-1) - r_col ** 2                         # [B,H,No]
    hdot = 2.0 * (d * vel.unsqueeze(2)).sum(-1)               # [B,H,No]
    cbf = hdot + A_CBF * h
    cbf = torch.clamp(cbf, max=0.0)                           # keep only violations
    k = min(K_WORST, obs_xy.shape[0])
    worst, _ = torch.topk(cbf, k=k, dim=2, largest=False)     # [B,H,k] most-violated
    w = torch.arange(k, 0, -1, dtype=pos.dtype, device=pos.device)[None, None]   # [k,...,1]
    return (worst * w).sum(dim=(1, 2))                        # [B]


def goal_reward(pos, goal_t):
    return -torch.norm(pos[:, -1] - goal_t[None], dim=1)      # [B] terminal distance


def stage_cost_batch(pos, U, goal_t, obs_xy, r_col, prev_U=None):
    """Their mppi/utils.py stage_cost + terminal, batched over [B,H,2] rollouts -> [B]."""
    B, H, _ = pos.shape
    goal_c = torch.norm(pos - goal_t[None, None], dim=2)                       # [B,H]
    d = torch.norm(pos.unsqueeze(2) - obs_xy[None, None], dim=3)               # [B,H,No]
    coll = torch.clamp(torch.exp(-BETA_MPPI * (d - r_col)), max=1.0).sum(2)    # [B,H]
    tw = COLL_W * (1.0 + 0.99 ** torch.arange(H, dtype=pos.dtype, device=pos.device))[None]
    cost = (GOAL_W * goal_c + tw * coll).sum(1)
    cost = cost + GOAL_W * torch.norm(pos[:, -1] - goal_t[None], dim=1)        # terminal
    if prev_U is not None:
        cost = cost + 0.1 * ((U - prev_U[None]) ** 2).sum(dim=(1, 2))          # warm-start consistency
    return cost


def guided_generate(policy, ctx, state, goal_t, obs_xy, r_col, dt, z_init, taus, safe_coef, device,
                    ret_guidance=False):
    """run_CFM:17-81 on our FM. z [N,d] flow var (=U/u_max flat); taus = knots list starting at current tau."""
    N = z_init.shape[0]; H = policy.H_pred if hasattr(policy, "H_pred") else 10
    z = z_init
    ctxN = policy._expand_ctx(ctx, N)
    markup = (MARKUP ** torch.arange(H - 1, -1, -1, dtype=z.dtype, device=device))[None, :, None]  # [1,H,1]
    last_guid = None
    for j in range(len(taus) - 1):
        tau, tau_n = float(taus[j]), float(taus[j + 1])
        tt = torch.full((N,), tau, device=device, dtype=z.dtype).clamp(1e-4, 1.0)
        with torch.no_grad():
            v = policy.forward(z, tt, ctxN)                                    # base field [N,d]
        x1 = (z + (1.0 - tau) * v).detach().requires_grad_(True)               # endpoint estimate
        U1 = x1.reshape(N, H, 2) * policy.u_max
        pos, vel = di_rollout_t(state, U1, dt)
        R_cbf = cbf_reward(pos, vel, obs_xy, r_col).sum()
        R_goal = goal_reward(pos, goal_t).sum()
        g_cbf, = torch.autograd.grad(R_cbf, x1, retain_graph=True)
        g_goal, = torch.autograd.grad(R_goal, x1)
        vn = torch.norm(v)                                                     # GLOBAL norm (faithful)
        g_cbf = (g_cbf * vn / (torch.norm(g_cbf) + 1e-8)).reshape(N, H, 2)
        g_goal = (g_goal * vn / (torch.norm(g_goal) + 1e-8)).reshape(N, H, 2)
        guid = GOAL_COEF * g_goal + safe_coef * g_cbf * markup
        last_guid = guid.detach()
        v_new = v + guid.reshape(N, -1)
        z = z + (tau_n - tau) * v_new
    if ret_guidance:
        return z, (last_guid if last_guid is not None else torch.zeros(N, H, 2, device=device))
    return z                                                                    # ~x1 in flow space


def flow_mppi_refine(policy, state, goal_t, obs_xy, r_col, dt, U_gen, prev_U, device, ret_viz=False):
    """flowmppi.py:144-314: elite select -> perturb -> per-mode softmax -> best refined."""
    with torch.no_grad():
        pos, _ = di_rollout_t(state, U_gen, dt)
        costs = stage_cost_batch(pos, U_gen, goal_t, obs_xy, r_col, prev_U)
        _, top = torch.topk(costs, k=min(N_ELITE, U_gen.shape[0]), largest=False)
        elites = U_gen[top]                                                    # [E,H,2]
        E = elites.shape[0]
        pert = elites.repeat_interleave(N_COPY, 0)
        pert = pert + MPPI_SIGMA * torch.randn_like(pert)
        pert = torch.clamp(pert, -policy.u_max, policy.u_max)
        posP, _ = di_rollout_t(state, pert, dt)
        cP = stage_cost_batch(posP, pert, goal_t, obs_xy, r_col, prev_U).reshape(E, N_COPY)
        b, _ = cP.min(dim=1, keepdim=True)
        w = torch.softmax(-(cP - b) / MPPI_LAMBDA, dim=1)                      # [E,C]
        refined = (w[:, :, None, None] * pert.reshape(E, N_COPY, *pert.shape[1:])).sum(1)   # [E,H,2]
        posR, _ = di_rollout_t(state, refined, dt)
        cR = stage_cost_batch(posR, refined, goal_t, obs_xy, r_col, prev_U)
        best = int(torch.argmin(cR))
    if ret_viz:                                                               # windows as PLANNED POSITIONS
        gen_pos, _ = di_rollout_t(state, U_gen, dt)
        return refined[best], dict(cand=gen_pos.cpu().numpy(), refined=posR.cpu().numpy(),
                                   best=posR[best].cpu().numpy())
    return refined[best]                                                       # [H,2]


def kazuki_deploy(policy, env, safe_coefs, gamma_ctx=0.5, T=250, reach=0.1,
                  device="cpu", seed=0, rec=None, conditioning_schema=None):
    """One receding-horizon episode from env.x0 (origin). Returns dict(path, reached, collided, steps).
    If rec is a list, appends per-step {state, cand, refined, best} for the their-style viz."""
    torch.manual_seed(seed); np.random.seed(seed)
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    obs_xy = torch.tensor(obs[:, :2], dtype=torch.float32, device=device)
    r_col = obstacle_collision_radii(obs, rr, R_MARGIN, device=device)
    goal = env.goal.detach().cpu().numpy()
    goal_t = torch.tensor(goal, dtype=torch.float32, device=device)
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    d = policy.d; H = d // 2
    # per-sample w_safe groups (40 each with the default 5 coefs)
    sc = torch.zeros(N_SAMPLE, 1, 1, device=device)
    size = N_SAMPLE // len(safe_coefs)
    for i, c in enumerate(safe_coefs):
        sc[size * i: size * (i + 1)] = c
    if conditioning_schema is not None:
        import afe_context as CX
        CX.require_declared_contract(
            policy, conditioning_schema, CX.SCHEMA_DIMS[conditioning_schema]
        )
    hist, path = [], [st[:2].copy()]
    prev_z = None; prev_U = None
    reached = collided = oob = False
    for t in range(T):
        if conditioning_schema is None:
            gT = torch.tensor(GF.axis_grid(st[:2], obs, rr), device=device)
            lT = torch.tensor(GF.low5(st, goal, gamma_ctx), device=device)
            hT = torch.tensor(
                GF.hist_pad(
                    np.array(hist[-GF.K_HIST:]) if hist else np.zeros((0, 2)),
                    GF.K_HIST,
                ),
                device=device,
            )
        else:
            record = CX.build_context(
                st, goal, gamma_ctx, hist, env, conditioning_schema
            )
            gT = torch.tensor(np.array(record.grid, copy=True), device=device)
            lT = torch.tensor(np.array(record.low5, copy=True), device=device)
            hT = torch.tensor(np.array(record.hist, copy=True), device=device)
        ctx = policy.ctx_from(gT[None], lT[None], hT[None]).squeeze(0)   # 1-D so _expand_ctx broadcasts
        if prev_z is None:
            z = torch.randn(N_SAMPLE, d, device=device); taus = ODE_TIMES_FULL
        else:
            z = TAU_WARM * prev_z[None].expand(N_SAMPLE, d) \
                + (1.0 - TAU_WARM) * torch.randn(N_SAMPLE, d, device=device)
            taus = [t_ for t_ in ODE_TIMES_FULL if t_ >= TAU_WARM]
        if rec is not None:
            z1, guide = guided_generate(policy, ctx, st, goal_t, obs_xy, r_col, env.dt, z, taus, sc,
                                        device, ret_guidance=True)
        else:
            z1 = guided_generate(policy, ctx, st, goal_t, obs_xy, r_col, env.dt, z, taus, sc, device)
        U_gen = torch.clamp(z1.reshape(N_SAMPLE, H, 2) * policy.u_max, -policy.u_max, policy.u_max)
        if rec is not None:
            U_best, vz = flow_mppi_refine(policy, st, goal_t, obs_xy, r_col, env.dt, U_gen, prev_U, device, ret_viz=True)
            rec.append(dict(state=st.copy(), guidance=guide[:, 0].cpu().numpy(), **vz))
        else:
            U_best = flow_mppi_refine(policy, st, goal_t, obs_xy, r_col, env.dt, U_gen, prev_U, device)
        a = U_best[0].detach().cpu().numpy()
        # execute first action (di_step)
        st = np.array([st[0] + env.dt * st[2] + 0.5 * env.dt ** 2 * a[0],
                       st[1] + env.dt * st[3] + 0.5 * env.dt ** 2 * a[1],
                       st[2] + env.dt * a[0], st[3] + env.dt * a[1]], np.float32)
        hist.append(a.copy()); path.append(st[:2].copy())
        dmin = np.linalg.norm(st[None, :2] - obs[:, :2], axis=1) - obs[:, 2] - rr
        if dmin.min() < 0:
            collided = True; break
        if np.linalg.norm(st[:2] - goal) < reach:
            reached = True; break
        if not GM.in_taskspace(st[:2][None]):
            oob = True; break
        # dilution warm-start: shift executed step off, repeat last control
        U_shift = torch.cat([U_best[1:], U_best[-1:]], 0)
        prev_z = (U_shift / policy.u_max).reshape(-1).detach()
        prev_U = U_shift.detach()
    return dict(path=np.array(path), reached=reached, collided=collided, oob=oob,
                steps=len(path) - 1)


def main():
    global COLL_W, GOAL_W, GOAL_COEF, BETA_MPPI, MPPI_LAMBDA, MPPI_SIGMA, N_SAMPLE, N_ELITE, N_COPY, R_MARGIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../results/hp_repr/pretrained_a32uni.pt")
    ap.add_argument("--w-safe", type=float, default=None,
                    help="single w_safe (whole batch); default None = their mixed 5-coef batch")
    ap.add_argument("--gamma-ctx", type=float, default=0.5, help="fixed FM conditioning gamma (neutral)")
    ap.add_argument("--M", type=int, default=25)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--tag", default="kaz")
    ap.add_argument("--outdir", default="results/kazuki")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--coll-w", type=float, default=None, help="override MPPI collision weight (their 100)")
    ap.add_argument("--goal-w", type=float, default=None, help="override MPPI goal weight (their 0.1)")
    ap.add_argument("--goal-coef", type=float, default=GOAL_COEF, help="guided-flow terminal-goal coefficient")
    ap.add_argument("--beta-mppi", type=float, default=BETA_MPPI, help="collision proximity steepness")
    ap.add_argument("--mppi-lambda", type=float, default=MPPI_LAMBDA)
    ap.add_argument("--mppi-sigma", type=float, default=MPPI_SIGMA)
    ap.add_argument("--r-margin", type=float, default=R_MARGIN)
    ap.add_argument("--n-sample", type=int, default=N_SAMPLE)
    ap.add_argument("--n-elite", type=int, default=N_ELITE)
    ap.add_argument("--n-copy", type=int, default=N_COPY)
    ap.add_argument("--viz-out", default=None, help="run ONE episode with per-step candidate/refine recording -> .pt")
    ap.add_argument("--goal-xy", type=float, nargs=2, default=None, help="move goal e.g. 4.7 4.7")
    ap.add_argument("--reach", type=float, default=0.1,
                    help="goal-reach radius; on the walled scene use 0.15 (goal-corner plugs block 0.1)")
    ap.add_argument("--wall-plugs", type=int, choices=[0, 2, 4, 8], default=0,
                    help="evaluate on the plugged/walled scene (same 8-plug scene as the expansion)")
    ap.add_argument("--start-eps", type=float, default=0.0,
                    help="start at (eps,eps); required on the plugged scene (origin ON the corner plugs)")
    args = ap.parse_args()
    if args.coll_w is not None:
        COLL_W = args.coll_w
    if args.goal_w is not None:
        GOAL_W = args.goal_w
    GOAL_COEF = args.goal_coef
    BETA_MPPI = args.beta_mppi
    MPPI_LAMBDA = args.mppi_lambda
    MPPI_SIGMA = args.mppi_sigma
    R_MARGIN = args.r_margin
    N_SAMPLE = args.n_sample
    N_ELITE = args.n_elite
    N_COPY = args.n_copy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol, _ = HP.load_hp(args.ckpt, device=dev)
    env = GS.make_grid()
    if getattr(args, "wall_plugs", 0):             # walled re-baseline (user 2026-07-14): same 8-plug scene
        from eval_ae import _apply_wall_plugs_eval
        _apply_wall_plugs_eval(env, args.wall_plugs)
    if getattr(args, "start_eps", 0.0) > 0.0:      # origin sits ON the corner plugs; match the policy eval
        env.x0 = torch.tensor([args.start_eps, args.start_eps, 0.0, 0.0], dtype=env.x0.dtype)
    if getattr(args, "goal_xy", None) is not None:
        env.goal = torch.tensor([float(args.goal_xy[0]), float(args.goal_xy[1])], dtype=env.goal.dtype)
    coefs = SAFE_COEFS if args.w_safe is None else [args.w_safe]
    os.makedirs(args.outdir, exist_ok=True)
    if args.viz_out:                                                          # record ONE episode + dump
        rec = []
        out = kazuki_deploy(pol, env, coefs, gamma_ctx=args.gamma_ctx, T=args.T, reach=args.reach, device=dev, seed=args.seed0, rec=rec)
        torch.save(dict(rec=rec, path=out["path"], reached=out["reached"], collided=out["collided"],
                        gamma_ctx=args.gamma_ctx, w_safe=coefs, coll_w=COLL_W, goal_w=GOAL_W,
                        goal_coef=GOAL_COEF, beta_mppi=BETA_MPPI, mppi_lambda=MPPI_LAMBDA,
                        mppi_sigma=MPPI_SIGMA, r_margin=R_MARGIN), args.viz_out)
        print(f"[viz] {len(rec)} steps recorded -> {args.viz_out} (reached={out['reached']} coll={out['collided']})", flush=True)
        return
    n_reach = n_coll = 0; steps = []; paths = []
    t0 = time.time()
    for m in range(args.M):
        out = kazuki_deploy(pol, env, coefs, gamma_ctx=args.gamma_ctx, T=args.T, reach=args.reach, device=dev,
                            seed=args.seed0 + m)
        n_reach += int(out["reached"]); n_coll += int(out["collided"]); steps.append(out["steps"])
        paths.append(out["path"])
        print(f"[{args.tag}] ep{m:03d} reached={out['reached']} collided={out['collided']} "
              f"steps={out['steps']} | SR {n_reach/(m+1):.2f} CR {n_coll/(m+1):.2f} "
              f"({(time.time()-t0)/(m+1):.1f}s/ep)", flush=True)
    config = dict(w_safe=coefs, gamma_ctx=args.gamma_ctx, coll_w=COLL_W, goal_w=GOAL_W,
                  goal_coef=GOAL_COEF, beta_mppi=BETA_MPPI, mppi_lambda=MPPI_LAMBDA,
                  mppi_sigma=MPPI_SIGMA, r_margin=R_MARGIN, n_sample=N_SAMPLE,
                  n_elite=N_ELITE, n_copy=N_COPY)
    res = dict(tag=args.tag, **config, M=args.M,
               SR=n_reach / args.M, CR=n_coll / args.M, mean_steps=float(np.mean(steps)),
               ckpt=os.path.basename(args.ckpt))
    with open(os.path.join(args.outdir, f"{args.tag}.json"), "w") as f:
        json.dump(res, f)
    pa = np.empty(len(paths), dtype=object)
    for i, p in enumerate(paths):
        pa[i] = np.asarray(p, dtype=np.float32)
    np.savez_compressed(os.path.join(args.outdir, f"paths_g{float(args.gamma_ctx)}.npz"),
                        paths=pa, gamma=float(args.gamma_ctx), config_json=json.dumps(config))
    print(f"[{args.tag}] FINAL SR {res['SR']:.2f} CR {res['CR']:.2f} -> {args.outdir}/{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
