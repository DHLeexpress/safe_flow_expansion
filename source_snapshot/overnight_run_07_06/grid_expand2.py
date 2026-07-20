"""Stage E' — v2 safe flow expansion (2026-07-03). Differences vs v1 grid_expand.py (which is untouched):

  LOSS (user 3a/3b): L = L_CFM(U_demo ∪ U_pos, c) − α·L_CFM(U_neg, c_neg) — NO aux term, NO demo_frac:
    one union pool with inverse-frequency CLASS weighting where all demo = ONE class and each discovered
    staircase = one class (demo starts as the whole pool = warm-up, then dilutes to 1/(1+#staircases)).
    No update at all until the first verified positive exists.
  NOTATION (3c): temp = FM sampling temperature (initial-noise scale), beta = Eq-9 tilt temperature.
  SINGLE MODEL, ALL γ (4): training round-robins γ∈{0.5,1.0,0.1}; measurement reports per-γ
    validity2 / coverage_cumulative / coverage_final every measure_every iters.
  METRICS (2a/2b): grid_metrics2 (window-level goal-approach validity2, coverage_final).
  ENTANGLEMENT DIAGNOSIS (user msg 2): per-module grad RMS during updates, frozen-probe context drift,
    demo forgetting probe (seeded val-CFM), GP novelty health; `freeze_enc` = causal arm (context map fixed).
  VALIDITY SANITY (user msg 2): `pos_margin` gate — only windows with min clearance ≥ margin enter D_pos.
  var(σ) (3d-B): variance of σ across the N candidates per exploration step, averaged per measure window.

Worker CLI at the bottom runs ONE config end-to-end (used by run_sweep_0703.py).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_rollout as GR
import grid_expand as GE          # reuse _to_t/_cat/_buffer_feat/load_demo/output_variance (no edits there)
import grid_policy2 as GP2
import wandb_utils as W
from uncertainty import GPUncertainty
from uncertainty_nn import NNUncertainty

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class SFG2Config:
    iters: int = 2000
    # Eq-9 exploration (FM-only candidates)
    N: int = 64
    temp: float = 1.3            # FM sampling temperature (initial-noise scale)
    beta: float = 0.1            # Eq-9 tilt temperature in exp((σ−maxσ)/β)
    churn: float = 0.05
    nfe_explore: int = 6
    safe_filter: bool = True
    # uncertainty estimator (S7 arm): "gp" = linear/RBF-kernel posterior std (novelty) OR
    #   "nn" = paper's bootstrapped 5-MLP ensemble (validity classifier; σ = member disagreement, boundary-seeking)
    unc: str = "gp"
    kernel: str = "rbf"
    ell: float = 0.2
    lam: float = 1e-2
    gp_buf: int = 384
    s: float = 0.9               # φ_s noise level (SWEEP: 0.9 / 0.8 / 0.3)
    # NN estimator (BACKUP plan, shrunk per user: 5×MLP-2×nn_hidden-ReLU-0.1drop; warm-start + early-stop)
    nn_refit_every: int = 16     # refit ensemble every k iters (warm-started, cheap)
    nn_max_steps: int = 1000
    nn_hidden: int = 64          # shrunk from the paper's 100 (smaller problem)
    lbuf_cap: int = 8000         # labeled buffer (both classes) for the NN fit
    # fine-tuning schedule (BACKUP plan): "online" = update every trajectory (inner_steps) OR
    #   "round" = paper-style: collect round_traj trajectories -> refit estimator -> finetune_steps block update
    schedule: str = "online"
    round_traj: int = 16
    finetune_steps: int = 200
    warmup_valid: int = 300      # defer the first block update until this many valid windows accumulated
    # v2 update — POSITIVE-ONLY (user 2026-07-03: U_demo REMOVED from the loss; demo pulled exploration back).
    #   demo is still LOADED for the forgetting/drift PROBES (diagnostic only), never for the gradient.
    demo_anchor: bool = False    # keep False: loss = L_CFM(U_pos) − α·L_CFM(U_neg); no demo term
    demo_frac: float = 0.0       # 2.1 (user 2026-07-05, MACE-multihead-inspired): δ of every batch = demo windows
    lwf_eta: float = 0.0         # 2.2 LwF frozen-teacher: + η·E_{ctx~demo}‖v_θ−v_θ₀‖² (anti-forgetting regularizer)
    grad_clip: float = 0.0       # max grad-norm on trainable params (0=off); explosion guard for α>0 (2026-07-06)
    alpha: float = 0.0           # negative-sample loss weight (SWEEP: 0 / 0.005 / 0.01)
    inner_steps: int = 12
    batch: int = 128
    lr: float = 2e-4             # base (field) lr — SWEEP: 2e-4 / 1e-4 / 1e-5
    # "nice optimizer" for the ENTANGLED encoder+field: per-group lr (context encoders learn at lr·enc_lr_mult,
    #   the velocity field at lr) + cosine schedule. enc_lr_mult=0 == frozen encoder (the causal freeze arm).
    enc_lr_mult: float = 1.0     # SWEEP: 1.0 / 0.3 / 0.1 / 0.0
    sched: str = "cosine"        # "cosine" | "none"
    freeze_enc: bool = False     # legacy alias: if True -> enc_lr_mult forced to 0
    pos_margin: float = 0.0      # data-hygiene gate: window min clearance ≥ margin to enter D_pos
    cap_pos: int = 60000
    cap_neg: int = 4000
    qbuf_cap: int = 500
    # measurement / snapshots
    gammas: tuple = (0.5, 1.0, 0.1)
    measure_every: int = 200
    n_measure: int = 25
    nfe_measure: int = 8
    temp_measure: float = 1.0     # no-tilt deploy temp for MEASUREMENT (reduced models reach only near ~0.5)
    T: int = 250
    snapshot_every: int = 100
    ckpt_every: int = 500


class SigmaRecorder:
    """Wraps GPUncertainty so fm_deploy's σ calls also record per-step candidate-σ variance (var(σ))."""

    def __init__(self, unc):
        self._unc = unc
        self.vars = []
        self.means = []

    def sigma(self, feat):
        s = self._unc.sigma(feat)
        if s.numel() > 1:
            self.vars.append(float(s.var()))
            self.means.append(float(s.mean()))
        return s


class Probes:
    """Entanglement probes on FROZEN demo windows: context drift ‖ctx_t−ctx_0‖/‖ctx_0‖ + cos, and the
    demo forgetting probe (seeded val-CFM, identical noise draws each call; RNG state restored)."""

    def __init__(self, policy, demo, device, n_ctx=256, n_val=1024, seed=4242):
        rng = np.random.default_rng(seed)
        n = demo["U"].shape[0]
        i1 = torch.as_tensor(rng.permutation(n)[:n_ctx].copy())
        i2 = torch.as_tensor(rng.permutation(n)[:n_val].copy())
        self.cg, self.cl, self.ch = demo["grid"][i1].to(device), demo["low5"][i1].to(device), demo["hist"][i1].to(device)
        self.vg, self.vl, self.vh, self.vu = (demo["grid"][i2].to(device), demo["low5"][i2].to(device),
                                              demo["hist"][i2].to(device), demo["U"][i2].to(device))
        with torch.no_grad():
            self.ctx0 = policy.ctx_from(self.cg, self.cl, self.ch).detach().clone()

    @torch.no_grad()
    def measure(self, policy):
        ctx = policy.ctx_from(self.cg, self.cl, self.ch)
        drift = float(((ctx - self.ctx0).norm(dim=1) / self.ctx0.norm(dim=1).clamp_min(1e-8)).mean())
        cos = float(torch.nn.functional.cosine_similarity(ctx, self.ctx0, dim=1).mean())
        cpu_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(777)
        dv = float(policy.cfm_loss(self.vu, policy.ctx_from(self.vg, self.vl, self.vh)))
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        return dict(ctx_drift=drift, ctx_cos=cos, demo_val_cfm=dv)


def load_demo_all(gammas=(0.1, 0.5, 1.0)):
    ds = [GE.load_demo(g) for g in gammas]
    return {k: torch.cat([d[k] for d in ds]) for k in ("grid", "low5", "hist", "U")}


def grad_rms(policy):
    out = {}
    for k, m in policy.module_groups().items():
        g2, n = 0.0, 0
        for p in m.parameters():
            if p.grad is not None:
                g2 += float((p.grad ** 2).sum())
            n += p.numel()
        out[k] = (g2 / max(n, 1)) ** 0.5
    return out


def fit_nn(policy, unc, lbuf, cfg, device, seed=0):
    """Refit the NN ensemble on the labeled buffer under the CURRENT policy's φ_s (co-evolves with θ)."""
    if lbuf is None or lbuf["U"].shape[0] < 16:
        return False
    n = lbuf["U"].shape[0]
    idx = torch.randperm(n)[:cfg.lbuf_cap]
    with torch.no_grad():
        ctx = policy.ctx_from(lbuf["grid"][idx].to(device), lbuf["low5"][idx].to(device), lbuf["hist"][idx].to(device))
        phi = policy.phi_s(lbuf["U"][idx].to(device), ctx, s=cfg.s)
    y = torch.tensor([float(lbuf["tag"][i]) for i in idx.tolist()], dtype=torch.float32, device=device)
    return unc.fit(phi, y, seed=seed)


def update_flow2(policy, opt, demo, pos, neg, cfg, device, n_steps=None):
    """POSITIVE-ONLY update (demo removed from the loss, user 2026-07-03): batch ~ inverse-frequency by
    staircase class over D_pos; loss = cfm(pos) − α·cfm(neg, c_neg). If cfg.demo_anchor is set (off by
    default), demo is added back as one extra class. Returns dict(loss, per-module grad RMS). None until pos.
    n_steps overrides inner_steps (round-based block fine-tune)."""
    npos = 0 if pos is None else pos["U"].shape[0]
    if npos == 0:
        return None
    freq = Counter(pos["tag"])
    use_demo = cfg.demo_anchor and demo is not None
    nd = demo["U"].shape[0] if use_demo else 0
    if use_demo:
        ncls = len(freq) + 1
        w = torch.empty(nd + npos, dtype=torch.double)
        w[:nd] = 1.0 / (ncls * nd)
        w[nd:] = torch.tensor([1.0 / (ncls * freq[t]) for t in pos["tag"]], dtype=torch.double)
    else:
        w = torch.tensor([1.0 / freq[t] for t in pos["tag"]], dtype=torch.double)   # inv-freq over pos classes
    nneg = 0 if neg is None else neg["U"].shape[0]
    policy.train()
    last = 0.0
    steps = n_steps or cfg.inner_steps
    gsum = {k: 0.0 for k in policy.module_groups()}
    ndf = int(round(cfg.demo_frac * cfg.batch)) if (cfg.demo_frac > 0 and demo is not None) else 0
    for _ in range(steps):
        if ndf > 0:                                   # 2.1: δ·batch demo windows + (1−δ)·batch positives
            di = torch.randint(0, demo["U"].shape[0], (ndf,))
            pi = torch.multinomial(w, cfg.batch - ndf, replacement=True)
            if use_demo:
                pi = pi[pi >= nd] - nd                # (demo_anchor off in this mode; keep indices in pos space)
            use_d = True
        else:
            idx = torch.multinomial(w, cfg.batch, replacement=True)
            di = idx[idx < nd]
            pi = idx[idx >= nd] - nd if use_demo else idx
            use_d = use_demo
        Gs, Ls, Hs, Us = [], [], [], []
        if (use_d or ndf > 0) and di.numel():
            Gs += [demo["grid"][di]]; Ls += [demo["low5"][di]]; Hs += [demo["hist"][di]]; Us += [demo["U"][di]]
        if pi.numel():
            Gs += [pos["grid"][pi]]; Ls += [pos["low5"][pi]]; Hs += [pos["hist"][pi]]; Us += [pos["U"][pi]]
        G = torch.cat(Gs).to(device); L = torch.cat(Ls).to(device)
        H = torch.cat(Hs).to(device); U = torch.cat(Us).to(device)
        loss = policy.cfm_loss(U, policy.ctx_from(G, L, H))
        if cfg.alpha > 0 and nneg > 0:
            ni = torch.randint(0, nneg, (min(cfg.batch, nneg),))
            nctx = policy.ctx_from(neg["grid"][ni].to(device), neg["low5"][ni].to(device), neg["hist"][ni].to(device))
            loss = loss - cfg.alpha * policy.cfm_loss(neg["U"][ni].to(device), nctx)
        teacher = getattr(cfg, "_teacher", None)
        if cfg.lwf_eta > 0 and teacher is not None and demo is not None:   # 2.2 LwF on demo contexts
            li = torch.randint(0, demo["U"].shape[0], (cfg.batch,))
            Gd, Ld, Hd = demo["grid"][li].to(device), demo["low5"][li].to(device), demo["hist"][li].to(device)
            Ud = demo["U"][li].to(device)
            B_ = Ud.shape[0]
            x1 = (Ud / policy.u_max).reshape(B_, policy.d)
            x0 = torch.randn_like(x1)
            tau = torch.rand(B_, device=x1.device).clamp(1e-4, 1.0)
            x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
            v_s = policy.forward(x_tau, tau, policy._expand_ctx(policy.ctx_from(Gd, Ld, Hd), B_))
            with torch.no_grad():
                v_t = teacher.forward(x_tau, tau, teacher._expand_ctx(teacher.ctx_from(Gd, Ld, Hd), B_))
            loss = loss + cfg.lwf_eta * ((v_s - v_t) ** 2).mean()
        opt.zero_grad(); loss.backward()
        for k, v in grad_rms(policy).items():
            gsum[k] += v
        if getattr(cfg, "grad_clip", 0.0) > 0:          # explosion guard (α-unlearning is unbounded below,
            torch.nn.utils.clip_grad_norm_(              # 2026-07-06); off by default, no effect on healthy grads
                [p for p in policy.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()
        last = float(loss)
    policy.eval()
    cfg._last_loss = last            # exposed for the block log line (loss-movement tracking, 2026-07-06)
    return dict(loss=last, **{f"grad_{k}": v / steps for k, v in gsum.items()})


def state_from_low5(low5_np):
    """Recover [px,py,vx,vy] from a stored low5 record (exact inverse of grid_feats.low5)."""
    p = GM2.GOAL_XY - np.asarray(low5_np[:2], float) * GF.R_GOAL
    v = np.asarray(low5_np[2:4], float) * GF.V_SCALE
    return np.array([p[0], p[1], v[0], v[1]], np.float32)


def run_expand2(policy, env, cfg: SFG2Config, device="cpu", run=None, outdir=None, log=print):
    gammas = list(cfg.gammas)
    demo = load_demo_all()               # loaded for PROBES only (positive-only loss; no demo gradient)
    enc_mult = 0.0 if cfg.freeze_enc else cfg.enc_lr_mult
    # "nice optimizer": separate param groups so the entangled context encoders and the velocity field can
    # learn at different rates (enc_mult<1 slows the context map -> less drift/forgetting; =0 freezes it).
    # The REDUCED model has NO learned encoder -> only the field group (enc_mult is then a no-op).
    field_params = list(policy.trunk.parameters()) + list(policy.head.parameters())
    enc_params = policy.encoder_modules()
    groups = [{"params": field_params, "lr": cfg.lr}]
    if enc_params:
        if enc_mult <= 0:
            for p in enc_params:                     # hard freeze (2026-07-05): requires_grad=False,
                p.requires_grad_(False)              # not Adam lr*0 — no grads flow into the encoder at all
        else:
            groups.append({"params": enc_params, "lr": cfg.lr * enc_mult})
    opt = torch.optim.Adam(groups)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iters)
             if cfg.sched == "cosine" else None)
    is_nn = cfg.unc == "nn"
    is_round = cfg.schedule == "round"
    if is_nn:
        unc = NNUncertainty(warm_start=True, hidden=cfg.nn_hidden, max_steps=cfg.nn_max_steps,
                            normalize=True, device=device)
    else:
        unc = GPUncertainty(kernel=cfg.kernel, lengthscale=cfg.ell, lam=cfg.lam, normalize=True)
    log(f"[expand2] POSITIVE-ONLY (demo_anchor={cfg.demo_anchor})  estimator={cfg.unc}  schedule={cfg.schedule}"
        f"  enc_lr_mult={enc_mult}  sched={cfg.sched}"
        + (f" (round_traj={cfg.round_traj}, finetune_steps={cfg.finetune_steps}, warmup_valid={cfg.warmup_valid})"
           if is_round else ""), flush=True)
    if cfg.lwf_eta > 0:                              # frozen teacher v_θ₀ for the LwF field-distillation term
        import copy as _copy
        cfg._teacher = _copy.deepcopy(policy).eval()
        for p_ in cfg._teacher.parameters():
            p_.requires_grad_(False)
    probes = Probes(policy, demo, device)
    pos = neg = qbuf = lbuf = None
    covered = {g: set() for g in gammas}
    history, snapshots = [], []
    var_sig_acc, sig_mean_acc = [], []
    last_buf_feats = None
    last_upd = None
    n_upd = 0
    trajs_since_round = 0

    def measure_block(it):
        rec = dict(iter=it, n_pos=0 if pos is None else int(pos["U"].shape[0]),
                   n_neg=0 if neg is None else int(neg["U"].shape[0]), n_upd=n_upd)
        for g in gammas:
            paths = GR.deploy_many(policy, env, g, cfg.n_measure, T=cfg.T, temp=cfg.temp_measure,
                                   nfe=cfg.nfe_measure, device=device)
            m = GM2.measure2(paths, env, g, covered[g])
            rec[f"g{g}"] = m
            snapshots.append(dict(kind="measure", iter=it, gamma=g,
                                  paths=[np.asarray(p, np.float32) for p in paths[:3]],
                                  covered=len(covered[g])))
        rec["var_sigma"] = float(np.mean(var_sig_acc)) if var_sig_acc else 0.0
        rec["sigma_mean"] = float(np.mean(sig_mean_acc)) if sig_mean_acc else 0.0
        var_sig_acc.clear(); sig_mean_acc.clear()
        rec["probes"] = probes.measure(policy)
        rec["out_var"] = GE.output_variance(policy, env, 0.5, device)
        if last_upd is not None:
            rec["upd"] = last_upd
        if last_buf_feats is not None:
            st = np.array([2.0, 2.0, 0.0, 0.0], np.float32)
            g_ = GF.axis_grid(st[:2], env.obstacles.detach().cpu().numpy(), float(env.r_robot))
            l_ = GF.low5(st, env.goal.detach().cpu().numpy(), 0.5)
            h_ = GF.hist_pad(np.zeros((0, 2)), GF.K_HIST)
            with torch.no_grad():
                Uc = policy.sample_window(torch.tensor(g_, device=device), torch.tensor(l_, device=device),
                                          torch.tensor(h_, device=device), n=64, temp=cfg.temp, nfe=6)
                fresh = policy.phi_s_at(Uc, torch.tensor(g_, device=device), torch.tensor(l_, device=device),
                                        torch.tensor(h_, device=device), s=cfg.s)
            try:
                rec["gp"] = {k: v for k, v in unc.diagnostics(last_buf_feats, fresh).items()}
            except Exception:
                pass
        history.append(rec)
        mv = np.mean([rec[f"g{g}"]["validity"] for g in gammas])
        mc = np.mean([rec[f"g{g}"]["coverage_cum"] for g in gammas])
        mf = np.mean([rec[f"g{g}"]["coverage_final"] for g in gammas])
        vio = {k: float(np.mean([rec[f"g{g}"]["violations"][k] for g in gammas]))   # γ-mean violation fractions
               for k in ("taskspace", "approach", "socp")}
        log(f"it{it:05d}: val2 {mv*100:.0f}% (γ:" +
            "/".join(f"{rec[f'g{g}']['validity']*100:.0f}" for g in gammas) +
            f") cov_cum {mc*100:.1f}% cov_fin {mf*100:.1f}% varσ {rec['var_sigma']:.4f} "
            f"viol[task {vio['taskspace']*100:.0f} appr {vio['approach']*100:.0f} socp {vio['socp']*100:.0f}] "
            f"drift {rec['probes']['ctx_drift']:.3f} demoCFM {rec['probes']['demo_val_cfm']:.3f} "
            f"npos {rec['n_pos']} loss {getattr(cfg, '_last_loss', float('nan')):.3f}", flush=True)
        wl = {}
        for g in gammas:
            for k in ("validity", "coverage_cum", "coverage_final", "reach_rate"):
                wl[f"expand2/{k}_g{g}"] = rec[f"g{g}"][k]
            for k in ("taskspace", "approach", "socp"):
                wl[f"viol/{k}_g{g}"] = rec[f"g{g}"]["violations"][k]
        for k, v in vio.items():
            wl[f"viol/{k}_mean"] = v
        wl.update({"expand2/var_sigma": rec["var_sigma"], "expand2/sigma_mean": rec["sigma_mean"],
                   "expand2/out_var": rec["out_var"], "expand2/n_pos": rec["n_pos"],
                   "probe/ctx_drift": rec["probes"]["ctx_drift"], "probe/ctx_cos": rec["probes"]["ctx_cos"],
                   "probe/demo_val_cfm": rec["probes"]["demo_val_cfm"]})
        if last_upd is not None:
            wl.update({f"grad/{k[5:]}": v for k, v in last_upd.items() if k.startswith("grad_")})
            wl["expand2/upd_loss"] = last_upd["loss"]
        if "gp" in rec:
            wl.update({f"gp/{k}": v for k, v in rec["gp"].items() if isinstance(v, (int, float))})
        W.log(run, wl, step=it)

    measure_block(0)
    for t in range(1, cfg.iters + 1):
        g = gammas[(t - 1) % len(gammas)]
        # --- refit uncertainty estimator (GP: every iter, cheap; NN: every nn_refit_every, warm-started) ---
        if is_nn:
            if t == 1 or t % cfg.nn_refit_every == 0:
                fit_nn(policy, unc, lbuf, cfg, device, seed=t)
            last_buf_feats = None                       # GP-style novelty diagnostics N/A for NN
        else:
            last_buf_feats = GE._buffer_feat(policy, qbuf, "phi_s", cfg.s, cfg.gp_buf, device)
            unc.set_buffer(last_buf_feats)
        rec_sig = SigmaRecorder(unc)
        out = GR.fm_deploy(policy, env, g, T=cfg.T,
                           tilt=dict(unc=rec_sig, beta=cfg.beta, N=cfg.N, s=cfg.s, broad=0, feature="phi_s",
                                     temp=cfg.temp, churn=cfg.churn, safe_filter=cfg.safe_filter),
                           nfe=cfg.nfe_explore, record=True,
                           verify_fn=GM2.window_label_cheap, device=device)
        if rec_sig.vars:
            var_sig_acc.append(float(np.mean(rec_sig.vars)))
            sig_mean_acc.append(float(np.mean(rec_sig.means)))
        if out["recs"]:
            G, L, H, U = GE._to_t(out["recs"])
            labels = [bool(r[4]) for r in out["recs"]]              # per-window cheap validity labels (for NN)
            qbuf = GE._cat(qbuf, G[::3], L[::3], H[::3], U[::3], cap=cfg.qbuf_cap)   # GP novelty memory
            lbuf = GE._cat(lbuf, G, L, H, U, tags=labels, cap=cfg.lbuf_cap)         # labeled buffer for NN fit
            if out["reached"] or out["dead"]:
                sid = GM.staircase_id(out["path"]) if out["reached"] else None
                ok2 = GM2.traj_valid2(out["path"], env, g)
                if ok2 and sid is not None:
                    covered[g].add(sid)
                    if cfg.pos_margin > 0:
                        keep = [i for i, r in enumerate(out["recs"])
                                if GM2.window_min_clearance(state_from_low5(r[1]), r[3], env) >= cfg.pos_margin]
                        if keep:
                            ki = torch.as_tensor(keep)
                            pos = GE._cat(pos, G[ki], L[ki], H[ki], U[ki], tags=[sid] * len(keep), cap=cfg.cap_pos)
                    else:
                        pos = GE._cat(pos, G, L, H, U, tags=[sid] * G.shape[0], cap=cfg.cap_pos)
                elif not ok2:
                    neg = GE._cat(neg, G, L, H, U, cap=cfg.cap_neg)
                # --- flow update: online (every trajectory) or round-based (paper-style block fine-tune) ---
                if is_round:
                    trajs_since_round += 1
                    npos = 0 if pos is None else pos["U"].shape[0]
                    if trajs_since_round >= cfg.round_traj and npos >= cfg.warmup_valid:
                        if is_nn:
                            fit_nn(policy, unc, lbuf, cfg, device, seed=100000 + t)   # fresh labels before block
                        upd = update_flow2(policy, opt, demo, pos, neg, cfg, device, n_steps=cfg.finetune_steps)
                        trajs_since_round = 0
                        if upd is not None:
                            last_upd = upd
                            n_upd += 1
                else:
                    upd = update_flow2(policy, opt, demo, pos, neg, cfg, device)
                    if upd is not None:
                        last_upd = upd
                        n_upd += 1
        if t % cfg.snapshot_every == 0:
            snapshots.append(dict(kind="explore", iter=t, gamma=g,
                                  path=np.asarray(out["path"], np.float32),
                                  covered={str(gg): len(covered[gg]) for gg in gammas},
                                  covered_sets={str(gg): sorted(covered[gg]) for gg in gammas}))
        if sched is not None:
            sched.step()
        if outdir and t % cfg.ckpt_every == 0:
            GP2.save_policy2(policy, os.path.join(outdir, f"ckpt_{t}.pt"),
                             extra={"iter": t, "covered": {str(gg): sorted(covered[gg]) for gg in gammas}})
        if t % cfg.measure_every == 0 or t == cfg.iters:
            measure_block(t)

    return dict(policy=policy, history=history, snapshots=snapshots,
                covered={str(g): sorted(covered[g]) for g in gammas}, final=history[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="pretrained2_w*.pt")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--s", type=float, default=0.9)
    ap.add_argument("--temp", type=float, default=1.3)
    ap.add_argument("--ell", type=float, default=0.2)
    ap.add_argument("--unc", choices=["gp", "nn"], default="gp")
    ap.add_argument("--nn-hidden", type=int, default=64)
    ap.add_argument("--schedule", choices=["online", "round"], default="online")
    ap.add_argument("--round-traj", type=int, default=16)
    ap.add_argument("--finetune-steps", type=int, default=200)
    ap.add_argument("--warmup-valid", type=int, default=300)
    ap.add_argument("--enc-lr-mult", type=float, default=1.0)
    ap.add_argument("--sched", choices=["cosine", "none"], default="cosine")
    ap.add_argument("--demo-anchor", action="store_true", help="(off by default) add U_demo back into the loss")
    ap.add_argument("--freeze-enc", action="store_true")
    ap.add_argument("--pos-margin", type=float, default=0.0)
    ap.add_argument("--measure-every", type=int, default=200)
    ap.add_argument("--n-measure", type=int, default=25)
    ap.add_argument("--temp-measure", type=float, default=1.0)
    ap.add_argument("--snapshot-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    import grid_scene as GS
    env = GS.make_grid()
    pol, _ = GP2.load_policy2(args.policy, device=dev)
    cfg = SFG2Config(iters=args.iters, lr=args.lr, alpha=args.alpha, beta=args.beta, s=args.s,
                     temp=args.temp, ell=args.ell, unc=args.unc, nn_hidden=args.nn_hidden, schedule=args.schedule,
                     round_traj=args.round_traj, finetune_steps=args.finetune_steps, warmup_valid=args.warmup_valid,
                     enc_lr_mult=args.enc_lr_mult, sched=args.sched, demo_anchor=args.demo_anchor,
                     freeze_enc=args.freeze_enc, pos_margin=args.pos_margin,
                     measure_every=args.measure_every, n_measure=args.n_measure, temp_measure=args.temp_measure,
                     snapshot_every=args.snapshot_every, ckpt_every=args.ckpt_every)
    rid = args.run_id or os.path.basename(os.path.normpath(args.outdir))
    run = W.init_run(args, name=f"sweep0703-{rid}", config={**vars(args), **asdict(cfg)}, group="sweep-0703")
    print(f"===== expand2 [{rid}]: {json.dumps({k: v for k, v in asdict(cfg).items() if k in ('iters','lr','alpha','beta','s','temp','unc','schedule','enc_lr_mult','demo_anchor','pos_margin')})} =====", flush=True)
    r = run_expand2(pol, env, cfg, device=dev, run=run, outdir=args.outdir)
    GP2.save_policy2(pol, os.path.join(args.outdir, "final.pt"), extra={"covered": r["covered"]})
    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(r["history"], f, indent=1)
    with open(os.path.join(args.outdir, "snapshots.pkl"), "wb") as f:
        pickle.dump(r["snapshots"], f)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "policy": args.policy, "run_id": rid}, f, indent=2)
    fin = r["final"]
    mv = np.mean([fin[f"g{g}"]["validity"] for g in cfg.gammas])
    mc = np.mean([fin[f"g{g}"]["coverage_cum"] for g in cfg.gammas])
    print(f"[{rid}] FINAL mean-val2 {mv*100:.1f}% mean-cov_cum {mc*100:.1f}% "
          f"varσ {fin['var_sigma']:.4f} npos {fin['n_pos']}", flush=True)
    W.finish(run, summary={"final_val": mv, "final_cov": mc})


if __name__ == "__main__":
    main()
