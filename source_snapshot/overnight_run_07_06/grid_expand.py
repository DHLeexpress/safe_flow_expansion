"""Stage E — faithful ACTFLOW safe flow expansion for the windowed grid policy (per γ).

1 iteration = 1 full receding-horizon trajectory:
  (Eq-10) refit σ_t from φ_s over the QUERIED buffer            unc.set_buffer(φ_s(qbuf))
  (Eq-9)  self-generate ONE trajectory, σ-tilted window pick     grid_rollout.fm_deploy(tilt=...) with LOW β
          (candidates = policy samples + broad right/up proposal, cheap safety-filtered, tilted by exp((σ−maxσ)/β))
  verifier ṽ: if the whole trajectory reaches ∧ is a valid staircase -> add its windows to D_pos (accepted)
              and add the staircase id to the cumulative `covered` set (discovery); else -> D_neg
  UpdateFlow: signed g = ∇L̂⁺ − α∇L̂⁻ with DEMO replay (continued pre-training, anchors safety; α=0 default)

Diagonal-concentrated expert data + a LOW β σ-tilt pushes exploration off-diagonal to grow coverage; demo
replay keeps validity from collapsing. Every `measure_every` iters: `n_measure` fresh no-tilt deploys ->
coverage (cumulative distinct staircases / 252) + validity. Stop when both > goal (or `iters`).
"""
from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

import _paths  # noqa: F401
import grid_metrics as GM
import grid_rollout as GR
import wandb_utils as W
from uncertainty import GPUncertainty

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")


@dataclass
class SFGridConfig:
    iters: int = 400
    # Eq-9 exploration
    N: int = 40
    broad: int = 40
    use_style: bool = False  # directed biasing (style/target) all HURT — forcing a mode kills the policy
    use_target: bool = False # per-move target-directed exploration (forcing a specific mode hurts)
    n_target: int = 40       # directed 'surrounding' candidates aimed at the style ratio / target
    align_temp: float = 0.45 # how hard to bias candidate selection toward the style/target direction (soft)
    beta: float = 1.0 / 25    # LOW temperature => strong σ-tilt => explore off the diagonal (SWEEP)
    s: float = 0.9
    temp_explore: float = 1.3
    churn: float = 0.05
    nfe_explore: int = 6
    safe_filter: bool = True  # cheap collision pre-filter on candidates (OFF => faithful FM-only Eq-9 SNIS)
    # Eq-10 GP
    kernel: str = "rbf"
    ell: float = 0.2          # lengthscale (SWEEP priority; lower => σ more sensitive)
    lam: float = 1e-2
    gp_buf: int = 384
    feature: str = "phi_s"    # σ feature: "phi_s" (original/default) or "rawU" (experimental, context-invariant)
    # signed UpdateFlow (+demo replay)
    alpha: float = 0.0        # negative-unlearning weight (default 0 per user)
    inner_steps: int = 12
    batch: int = 128
    lr: float = 2e-4
    aux_w: float = 0.3
    demo_frac: float = 0.5    # fraction of each update batch drawn from the safe expert demos (anchors validity)
    warmup_pos: int = 40
    cap_pos: int = 50000      # hold ALL discovered staircases (~227*150) so the policy does not FORGET modes
    cap_neg: int = 3000
    qbuf_cap: int = 500
    # measurement / stopping
    measure_every: int = 50
    n_measure: int = 50
    baseline_deploys: int = 500
    track_variance: bool = False   # also record the FM output variance at probe states (de-collapse diagnostic)
    nfe_measure: int = 8
    T: int = 250
    goal_cov: float = 0.90
    goal_val: float = 0.90


def output_variance(policy, env, gamma, device, n=96, probes=(0.15, 0.4, 0.65)):
    """FM output spread at diagonal probe states: total variance of window net-displacement, sampled at temp=1
    (the policy's OWN distribution). ~0 for a collapsed/deterministic policy; grows if it de-collapses off-diagonal."""
    import grid_feats as GF
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot); goal = env.goal.detach().cpu().numpy()
    vs = []
    for pf in probes:
        st = np.array([pf * GM.GRID_M, pf * GM.GRID_M, 0.0, 0.0], np.float32)
        g = GF.axis_grid(st[:2], obs, rr); l = GF.low5(st, goal, gamma); h = GF.hist_pad(np.zeros((0, 2)), 16)
        with torch.no_grad():
            U = policy.sample_window(torch.tensor(g, device=device), torch.tensor(l, device=device),
                                     torch.tensor(h, device=device), n=n, temp=1.0, nfe=8).detach().cpu().numpy()
        nets = GR.di_rollout_batch(st, U, env.dt)[:, -1, :] - st[:2]
        vs.append(float(np.var(nets, axis=0).sum()))
    return float(np.mean(vs))


def load_demo(gamma, device="cpu"):
    f = os.path.join(DATA, f"windows_g{gamma}.pt")
    d = torch.load(f)
    return dict(grid=d["grid"], low5=d["low5"], hist=d["hist"], U=d["U"])


def _to_t(recs):
    G = torch.tensor(np.array([r[0] for r in recs]))
    L = torch.tensor(np.array([r[1] for r in recs]))
    H = torch.tensor(np.array([r[2] for r in recs]))
    U = torch.tensor(np.array([r[3] for r in recs]))
    return G, L, H, U


def _cat(buf, G, L, H, U, tags=None, cap=None):
    if buf is None:
        buf = dict(grid=G, low5=L, hist=H, U=U, tag=list(tags) if tags is not None else None)
    else:
        buf["grid"] = torch.cat([buf["grid"], G]); buf["low5"] = torch.cat([buf["low5"], L])
        buf["hist"] = torch.cat([buf["hist"], H]); buf["U"] = torch.cat([buf["U"], U])
        if tags is not None:
            buf["tag"] = (buf["tag"] or []) + list(tags)
    if cap and buf["U"].shape[0] > cap:
        idx = torch.randperm(buf["U"].shape[0])[:cap]
        buf["grid"], buf["low5"] = buf["grid"][idx], buf["low5"][idx]
        buf["hist"], buf["U"] = buf["hist"][idx], buf["U"][idx]
        if buf["tag"] is not None:
            buf["tag"] = [buf["tag"][i] for i in idx.tolist()]
    return buf


def _buffer_feat(policy, qbuf, feature, s, cap, device):
    if qbuf is None or qbuf["U"].shape[0] == 0:
        return None
    n = qbuf["U"].shape[0]
    idx = torch.randperm(n)[:cap]
    if feature == "rawU":                                            # context-invariant control-content
        return qbuf["U"][idx].reshape(idx.shape[0], -1).to(device) / policy.u_max
    ctx = policy.ctx_from(qbuf["grid"][idx].to(device), qbuf["low5"][idx].to(device), qbuf["hist"][idx].to(device))
    return policy.phi_s(qbuf["U"][idx].to(device), ctx, s=s)


def update_flow(policy, opt, demo, pos, neg, cfg, device):
    """Signed FM update with DEMO replay: min L̂⁺(demo∪pos) − α L̂⁻(neg) + aux. pos sampled inverse-freq
    by staircase (diversifies). α=0 by default (positive-only)."""
    nd = demo["U"].shape[0]
    npos = 0 if pos is None else pos["U"].shape[0]
    use_pos = npos >= cfg.warmup_pos
    if use_pos and pos.get("tag"):
        freq = Counter(pos["tag"])
        wp = torch.tensor([1.0 / freq[t] for t in pos["tag"]], dtype=torch.float32)
        wp = wp / wp.sum()
    policy.train()
    last = 0.0
    nd_b = int(cfg.batch * cfg.demo_frac) if use_pos else cfg.batch   # demo_frac=0 => positive-only after warmup
    for _ in range(cfg.inner_steps):
        Gs, Ls, Hs, Us = [], [], [], []
        if nd_b > 0:
            bd = torch.randint(0, nd, (nd_b,))
            Gs += [demo["grid"][bd]]; Ls += [demo["low5"][bd]]; Hs += [demo["hist"][bd]]; Us += [demo["U"][bd]]
        if use_pos:
            bp = torch.multinomial(wp, cfg.batch - nd_b, replacement=True)
            Gs.append(pos["grid"][bp]); Ls.append(pos["low5"][bp]); Hs.append(pos["hist"][bp]); Us.append(pos["U"][bp])
        G = torch.cat(Gs).to(device); L = torch.cat(Ls).to(device); H = torch.cat(Hs).to(device); U = torch.cat(Us).to(device)
        ctx = policy.ctx_from(G, L, H)
        loss = policy.cfm_loss(U, ctx) + cfg.aux_w * policy.aux_safety_loss(G)
        if cfg.alpha > 0 and neg is not None and neg["U"].shape[0] > 0:
            ni = torch.randint(0, neg["U"].shape[0], (nd_b,))
            nctx = policy.ctx_from(neg["grid"][ni].to(device), neg["low5"][ni].to(device), neg["hist"][ni].to(device))
            loss = loss - cfg.alpha * policy.cfm_loss(neg["U"][ni].to(device), nctx)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    policy.eval()
    return last


def run_expand(policy, env, gamma, cfg: SFGridConfig, demo=None, device="cpu", log=print, run=None, step0=0,
               init_covered=None, iter_offset=0, time_budget=None):
    import time as _time
    _t0 = _time.time()
    if demo is None:
        demo = load_demo(gamma)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    unc = GPUncertainty(kernel=cfg.kernel, lengthscale=cfg.ell, lam=cfg.lam, normalize=True)
    pos = neg = qbuf = None
    covered = set(init_covered) if init_covered is not None else set()   # resume: cumulative coverage carries over
    history = []
    snapshots = []

    base_paths = GR.deploy_many(policy, env, gamma, cfg.baseline_deploys if cfg.iters >= 50 else cfg.n_measure,
                                T=cfg.T, nfe=cfg.nfe_measure, device=device)
    val, cov, steps, _ = GM.measure(base_paths, env, gamma, covered)
    base_rec = dict(iter=iter_offset, coverage=cov, validity=val, avg_steps=steps, n_pos=0)
    if cfg.track_variance:
        base_rec["out_var"] = output_variance(policy, env, gamma, device)
    history.append(base_rec)
    log(f"[γ{gamma}] it{iter_offset:03d} baseline: cov={cov*100:.1f}% val={val*100:.1f}% steps={steps:.0f} covered={len(covered)}/252", flush=True)
    W.log(run, {f"expand/coverage_g{gamma}": cov, f"expand/validity_g{gamma}": val,
                f"expand/avg_steps_g{gamma}": steps}, step=step0)

    for t in range(1, cfg.iters + 1):
        if time_budget is not None and _time.time() - _t0 > time_budget:
            log(f"[γ{gamma}] wall-clock budget {time_budget:.0f}s reached at iter {t}", flush=True)
            break
        unc.set_buffer(_buffer_feat(policy, qbuf, cfg.feature, cfg.s, cfg.gp_buf, device))
        target = style_rho = None
        if cfg.use_style:                                            # coherent per-trajectory right/up ratio
            style_rho = random.uniform(0.05, 0.95)
        elif cfg.use_target:
            frontier = set()
            for w in covered:
                frontier |= GM.neighbors(w)
            frontier -= covered
            target = random.choice(list(frontier)) if frontier else (
                random.choice(list(GM.STAIRCASES - covered)) if len(covered) < GM.N_STAIR else None)
        out = GR.fm_deploy(policy, env, gamma, T=cfg.T, target=target, style_rho=style_rho,
                           tilt=dict(unc=unc, beta=cfg.beta, N=cfg.N, s=cfg.s, broad=cfg.broad, feature=cfg.feature,
                                     temp=cfg.temp_explore, churn=cfg.churn, safe_filter=cfg.safe_filter,
                                     n_target=cfg.n_target, align_temp=cfg.align_temp),
                           nfe=cfg.nfe_explore, record=True, device=device)
        upd = 0.0
        if out["recs"]:
            G, L, H, U = _to_t(out["recs"])
            qbuf = _cat(qbuf, G[::3], L[::3], H[::3], U[::3], cap=cfg.qbuf_cap)   # queried memory for σ
            if out["reached"] or out["dead"]:
                sid = GM.staircase_id(out["path"]) if out["reached"] else None
                if sid is not None and GM.is_valid_traj(out["path"], env, gamma):
                    covered.add(sid)
                    pos = _cat(pos, G, L, H, U, tags=[sid] * G.shape[0], cap=cfg.cap_pos)
                else:
                    neg = _cat(neg, G, L, H, U, cap=cfg.cap_neg)
                upd = update_flow(policy, opt, demo, pos, neg, cfg, device)

        if t % cfg.measure_every == 0 or t == cfg.iters:
            paths = GR.deploy_many(policy, env, gamma, cfg.n_measure, T=cfg.T, nfe=cfg.nfe_measure, device=device)
            val, cov, steps, _ = GM.measure(paths, env, gamma, covered)
            np_ = 0 if pos is None else pos["U"].shape[0]
            rec = dict(iter=iter_offset + t, coverage=cov, validity=val, avg_steps=steps, n_pos=np_, upd=upd)
            if cfg.track_variance:
                rec["out_var"] = output_variance(policy, env, gamma, device)
                W.log(run, {f"expand/out_var_g{gamma}": rec["out_var"]}, step=step0 + t)
            history.append(rec)
            snapshots.append(dict(iter=iter_offset + t, covered=sorted(covered),
                                  paths=[np.asarray(p, np.float32) for p in paths[:8]]))
            log(f"[γ{gamma}] it{iter_offset + t:03d}: cov={cov*100:.1f}% val={val*100:.1f}% steps={steps:.0f} "
                f"npos={np_} covered={len(covered)}/252 upd={upd:.3f}", flush=True)
            W.log(run, {f"expand/coverage_g{gamma}": cov, f"expand/validity_g{gamma}": val,
                        f"expand/avg_steps_g{gamma}": steps, f"expand/n_pos_g{gamma}": np_,
                        f"expand/upd_loss_g{gamma}": upd}, step=step0 + t)
            if cov >= cfg.goal_cov and val >= cfg.goal_val:
                log(f"[γ{gamma}] GOAL reached at iter {t}", flush=True)
                break
    return dict(policy=policy, history=history, covered=covered, snapshots=snapshots, final=history[-1],
                reached_goal=(history[-1]["coverage"] >= cfg.goal_cov and history[-1]["validity"] >= cfg.goal_val))


if __name__ == "__main__":
    import time
    import grid_scene as GS
    import grid_policy as GP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid()
    pol, _ = GP.load_policy("pretrained.pt", device=dev)
    cfg = SFGridConfig(iters=40, measure_every=10, n_measure=30, baseline_deploys=30, warmup_pos=30, inner_steps=10)
    t0 = time.time()
    r = run_expand(pol, env, 0.5, cfg, device=dev)
    print(f"--- 40-iter expand γ0.5: {(time.time()-t0)/60:.1f}min covered={len(r['covered'])}/252", flush=True)
    print("coverage%:", [round(h["coverage"] * 100, 1) for h in r["history"]])
    print("validity%:", [round(h["validity"] * 100, 1) for h in r["history"]])
