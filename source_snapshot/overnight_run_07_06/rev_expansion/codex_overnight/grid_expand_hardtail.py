"""Fixed-schedule AND-quantile safe-flow expansion (user-locked 2026-07-10).

Copy of overnight_run_07_06/grid_expand_cur.py, redesigned:
  - NO persistent positive buffer / no old pile. Every outer iter gathers K *valid* fresh rollouts and the
    update trains on THOSE windows only (batch composition fully known & controllable — paper-clean).
  - Per-window validity (from the FIXED valid2, net-progress only): taskspace ∧ SOCP(traj) ∧ net-progress≥0.10.
  - 2 classes: frontier = high-σ AND low-margin AND high-progress, with all three planes set by the same
    fixed-schedule quantile.  At 50% this selects approximately 12.5% of windows.
  - VALIDITY floor on net-progress (--valid-prog-floor) REJECTS safe-stationary windows before they are gathered:
    a SOCP-safe but barely-moving window (prog ~0.1) is safe-not-performant; training on it teaches "stay put"
    (→ CR≈0 but SR≪1, the it600 origin-collapse death-spiral). Rejecting them at the gate breaks that spiral.
    (valid but gentle). σ = GP novelty vs a rolling query buffer.
  - Gather keeps sampling until both classes exist (up to an explicit attempt cap).  There is no demo
    backfill and an update is skipped if either class remains empty.
  - inner-steps 1 / 2 / 1 (early/mid/cool) to guard gradient blow-up on tiny batches. No warm-up gate.
  - viz_db (labels+scores) saved every viz_db_every (=100) iters.

All schedules use the absolute iteration index, so checkpoint resumes do not stretch or restart them.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # codex_overnight/
_REV = os.path.dirname(_HERE)                               # rev_expansion/
_WORK = os.path.dirname(_REV)                               # overnight_run_07_06/
sys.path.insert(0, _WORK)                                   # shared grid code
sys.path.insert(0, _REV)                                    # rev_expansion helpers
sys.path.insert(0, _HERE)                                   # this local algorithm copy always wins

import argparse
import copy
import json
import random
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

import _paths  # noqa: F401
import grid_rollout as GR
import grid_expand as GE
import grid_expand2 as GX2          # state_from_low5
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2         # local COPY with the net-progress-only approach_ok fix
import grid_hp_expt as HP
from uncertainty import GPUncertainty
import sr_cr_eval as SR


@dataclass
class CurConfig:
    iters: int = 1000
    start_iter: int = 0              # absolute iteration represented by the input checkpoint
    # exploration (σ-tilt)
    N: int = 64
    temp: float = 1.0
    s: float = 0.9
    churn: float = 0.05
    nfe_explore: int = 8              # match faithful evaluation; avoid training a coarser sampler
    safe_filter: bool = True
    targeted_frac: float = 0.5       # rollout-coherent uncovered-staircase proposals; all still pass exact gates
    n_target: int = 40
    align_temp: float = 0.45
    target_perp_brake: bool = False # target proposal brakes the non-requested crossing axis
    min_modes_per_gamma: int = 2   # exact-valid achieved staircase modes required in each training block
    min_modes_schedule: tuple = () # optional absolute (start_iter, modes/gamma) coverage curriculum
    mode_hit_gate: bool = False     # require exact targeted staircase hits before any coverage update
    min_target_hits: int = 1
    active_min_modes: int = 2       # transient absolute-schedule value
    # GP σ estimator
    kernel: str = "rbf"
    ell: float = 0.2
    lam: float = 1e-2
    gp_buf: int = 384
    qbuf_cap: int = 500
    # FRESH-ONLY curriculum
    rollouts_per_iter: int = 10     # maximum valid rollouts while trying to populate both classes
    gather_attempt_cap: int = 30    # sane hard cap; no demo substitution if a class is still absent
    valid_prog_floor: float = 0.15  # REJECT windows below this net-progress (safe-stationary trap; 0 = off, valid2's 0.10 bar)
    min_rollouts: int = 1           # gather AT LEAST this many valid rollouts (LOCKED recipe = 1; 4 was the failed uni_C knob)
    traj_prog_min: float = 0.0      # dither gate (LOCKED recipe = 0/off; 1.0 was the failed uni_C knob)
    # ---- warm-up noise fixes (user 2026-07-09: noisy near-origin initial windows hammered as easy) ----
    strat_rid: bool = False         # batch draw round-robins across source rollouts (prob #1 at the BATCH level)
    easy_sig_abs: float = 0.0       # ABSOLUTE σ cap: σ >= this can NEVER be easy (quantile split lies when ALL are noisy); 0=off
    easy_skip_first: int = 0        # windows with in-traj index < this are NEVER easy (the noisy initial escape part)
    probe_escape: int = 0           # every N iters: M faithful rollouts -> origin-escape stability probe (0=off)
    probe_cov: int = 0              # every N iters: M=50 faithful @γ0.5 -> instantaneous SR/CR/staircase-coverage
    log_comp_every: int = 0         # composition/rid-diversity log line every N iters (micro mode: 1; 0=off)
    # ---- pile revival (user 2026-07-09: fresh_frac<1 + bounded-staleness pile + no-GD warm-up) ----
    fresh_frac: float = 1.0         # fresh share of the fresh-part batch; rest drawn from the pile (1.0 = fresh-only)
    warmup_gather: int = 0          # first N iters: gather->pile only, NO gradient step (GP σ warms up too)
    pile_cap: int = 3000            # FIFO cap -> pile holds only the last ~10-20 iters (staleness BOUNDED)
    pile_replace: bool = False      # False = least-recently-used draw: every sample gets a turn before any repeats
    pile_relabel_every: int = 10    # recompute σ-dependent labels of the whole pile every N iters (labels refresh)
    batch_cap: int = 32             # TOTAL batch (demo + fresh); demo = round(demo_frac*batch_cap)
    lr: float = 1e-4
    quantile_schedule: tuple = ((0, 0.50), (200, 0.60), (400, 0.70))
    active_quantile: float = 0.50   # set from quantile_schedule at each absolute iteration
    mix_start: tuple = (0.7, 0.3)   # easy / frontier (2-class)
    mix_end: tuple = (0.5, 0.5)
    beta: float = 0.3               # constant; compare 0.2 only in a separately logged arm
    viz_db_every: int = 100
    cooldown_from: int = 400        # absolute fixed phase boundary
    cooldown_lr_mult: float = 0.3
    inner_steps: int = 4            # mid-phase inner steps (user: focus on every window via more passes)
    early_inner: int = 2            # early/cooldown = 2
    cooldown_inner: int = 2
    early_until: int = 100          # absolute fixed phase boundary
    enc_grad_clip: float = 5.0
    field_grad_clip: float = 1.0      # bound behaviorally large trunk/head moves near the OOD origin
    max_functional_step: float = 0.025  # batch-context metric: stable lr2e-5=1.5--1.8%; reject large jumps
    max_anchor_drift: float = 0.016     # cumulative fixed-origin field drift from the lineage teacher
    easy_strict: bool = True        # kept for parity (unused in 2-class labeling; sweep passes --easy-strict)
    # measurement (SR/CR primary)
    measure_every: int = 100
    M_measure: int = 25
    reach: float = 0.1
    T: int = 250
    # anchors / misc
    demo_frac: float = 0.0
    lwf_eta: float = 0.0
    demo_cap: int = 1200
    gammas: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    ckpt_every: int = 500
    collapse_frac: float = 0.45
    collapse_patience: int = 3
    collapse_min_iter: int = 600
    legacy_prime_iters: int = 1       # mandatory GP/query-memory prime for checkpoints without train state
    # ---- hard-tail repair (Claude 2026-07-10; evidence: analysis/seed12_tail_trace.md) ----
    # The two faithful-failure strata are BOUNDARY STRIPS with zero certified data: y<~0 near the origin
    # (absorbing once entered; seed 12 trigger) and y>~5 above the goal (overshoot wander). Repair =
    # (1) gather executed+exact-certified recovery rollouts STARTED on the strips (unchanged Valid2/reach),
    # (2) reserve a small batch sub-quota for strip-context windows, (3) pair those targets with base
    # latents whose current faithful map exits the task box (importance sampling on x0 only; CFM target
    # formula unchanged). Never any inference-time change.
    recovery_frac: float = 0.3        # fraction of gather attempts starting from a strip state (0=off)
    recovery_origin_band: tuple = (0.0, 1.0, -0.08, 0.20, 0.0, 0.45, -0.35, 0.05)   # x0,x1,y0,y1,vx0,vx1,vy0,vy1
    recovery_goal_band: tuple = (4.3, 5.0, 4.6, 5.06, -0.30, 0.30, -0.05, 0.35)
    hard_quota: int = 8               # fresh-batch slots reserved for strip-context windows (0=off)
    guard_quota: int = 0              # certified interior boundary-approach replay (0 keeps Claude arm exact)
    guard_x0: str = ""                 # "inbounds" pairs guard targets with currently successful latents
    fixed_origin_x0_seed: int = -1     # diagnostic only: exact clean-origin latent fiber (-1 disables)
    hard_x0_allow_majority: bool = False
    hard_x0_select: str = "worst"
    endpoint_eta: float = 0.0          # differentiable NFE endpoint repair on mapped hard/guard latents
    cfm_eta: float = 1.0               # 0 only for frozen-base residual endpoint probes
    guard_teacher_endpoint: bool = False
    hard_x0: str = "oob"              # "oob" = pair sub-quota targets with harvested OOB-mapping latents
    hard_x0_cand: int = 32            # candidate latents harvested per strip window
    strip_probe_every: int = 1        # per-iter absorber probe (win-OOB frac at 2 fixed failing contexts)
    escape_quota: int = 0             # immutable-replay rows per inner step (endpoint distillation; 0=off)
    escape_eta: float = 1.0
    # ---- ablation suite (user 2026-07-11): isolate curriculum / multi-step SOCP / progress ----
    ablate_curriculum: bool = False   # (1) no easy/frontier split — every valid sample equal
    ablate_socp: bool = False         # (2) verifier = free-space + goal-progress only; margin axis = geometric clearance
    ablate_progress: bool = False     # (3) verifier = taskspace + SOCP only (no goal-progress condition)
    # From-scratch amendment: stay single-class until competence has been sustained,
    # then irreversibly enable the ordinary easy/frontier curriculum.
    phased_curriculum: bool = False
    phase_sr_threshold: float = 0.85
    phase_sr_patience: int = 2
    phase_uniform_active: bool = False  # transient, derived from measurement history every iteration


# ---------------------------------------------------------------- labeling & sampling
def _single_class_active(cfg):
    return bool(getattr(cfg, "ablate_curriculum", False) or
                getattr(cfg, "phase_uniform_active", False))


def _phased_frontier_ready(history, cfg):
    """Irreversible history-derived competence gate (therefore split-resume deterministic)."""
    if not getattr(cfg, "phased_curriculum", False):
        return True
    patience = int(getattr(cfg, "phase_sr_patience", 2))
    threshold = float(getattr(cfg, "phase_sr_threshold", 0.85))
    if patience < 1:
        raise ValueError("phase_sr_patience must be >= 1")
    passed = [float(h.get("SR", -np.inf)) >= threshold for h in history]
    return any(all(passed[i:i + patience]) for i in range(max(0, len(passed) - patience + 1)))


def _window_progress(low5, U, env):
    """Net progress d0-dH of a single window (goal = GM2.GOAL_XY), plus the distance array for approach_ok."""
    st = GX2.state_from_low5(low5)
    seg = GR.window_positions(st, U, env.dt)
    pts = np.vstack([np.asarray(st, float)[:2][None, :], seg])
    d = np.linalg.norm(pts - GM2.GOAL_XY[None], axis=1)
    return float(d[0] - d[-1]), pts, d


def _sigma_of(policy, unc, data, cfg, device):
    """GP novelty σ of every window in ``data`` (chunked); failures are never silently relabeled."""
    n = data["U"].shape[0]
    sig = []
    with torch.no_grad():
        for i in range(0, n, 2048):
            ctx = policy.ctx_from(data["grid"][i:i + 2048].to(device), data["low5"][i:i + 2048].to(device),
                                  data["hist"][i:i + 2048].to(device))
            phi = policy.phi_s(data["U"][i:i + 2048].to(device), ctx, s=cfg.s)
            sig.append(unc.sigma(phi).detach().cpu().numpy())
    out = np.concatenate(sig) if sig else np.zeros(n)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite GP uncertainty")
    return out


def _quantile_at(schedule, absolute_iter):
    """Piecewise-constant quantile from an absolute-iteration schedule."""
    return float(max((q for start, q in schedule if absolute_iter >= start),
                     default=schedule[0][1]))


def _int_schedule_at(schedule, absolute_iter, default):
    """Piecewise-constant nonnegative integer schedule on absolute iteration."""
    if not schedule:
        return int(default)
    value = max((v for start, v in schedule if absolute_iter >= start), default=int(default))
    value = int(value)
    if value < 0:
        raise ValueError(f"scheduled integer must be nonnegative, got {value}")
    return value


def _target_hit_ready(audit, cfg):
    return (not bool(getattr(cfg, "mode_hit_gate", False)) or
            int(audit.get("target_hits", 0)) >= int(getattr(cfg, "min_target_hits", 1)))


_ORIGINAL_BROAD_TARGETED = GR.broad_targeted


def _perp_braking_targeted(state, direction, env, n, device):
    """Advance toward the requested boundary while braking the other crossing axis.

    This only proposes windows. Executed Valid2 and the exact planned-window
    certificate remain the unchanged acceptance authority.
    """
    d = np.asarray(direction, np.float32)
    if np.abs(d).max() < 1e-6 or np.sum(np.abs(d) > 0.25) != 1:
        return _ORIGINAL_BROAD_TARGETED(state, direction, env, n, device)
    axis = int(np.argmax(np.abs(d))); other = 1 - axis
    sign = 1.0 if d[axis] >= 0 else -1.0
    states = np.repeat(np.asarray(state, np.float32)[None], n, axis=0)
    U = np.empty((n, GF.H_PRED, 2), np.float32)
    um, dt = float(env.u_max), float(env.dt)
    hold = np.floor(states[:, other] + 1e-5) + (0.35 if sign > 0 else 0.65)
    for h in range(GF.H_PRED):
        a = np.zeros((n, 2), np.float32)
        goal_axis = (np.floor(states[:, axis] + 1e-5) + 1.10 if sign > 0
                     else np.ceil(states[:, axis] - 1e-5) - 1.10)
        a[:, axis] = 3.0 * (goal_axis - states[:, axis]) - 1.6 * states[:, axis + 2]
        a[:, other] = 3.0 * (hold - states[:, other]) - 2.4 * states[:, other + 2]
        a += np.random.randn(n, 2).astype(np.float32) * 0.12
        a = np.clip(a, -um, um); U[:, h] = a
        states[:, :2] += dt * states[:, 2:4] + 0.5 * dt * dt * a
        states[:, 2:4] += dt * a
    return torch.as_tensor(U, device=device)


@contextmanager
def _target_proposal_override(enabled):
    if not enabled:
        yield
        return
    old = GR.broad_targeted
    GR.broad_targeted = _perp_braking_targeted
    try:
        yield
    finally:
        GR.broad_targeted = old


def _front_mask(sigma, margin, prog, widx, cfg, gamma=None, return_planes=False):
    """User-fixed AND cell, computed within each conditioning gamma.

    Pooling all gammas made conditions with different certificate/progress scales almost absent from the
    frontier (notably gamma=.1).  Quantiles are therefore conditional, matching the conditional policy and
    the user's requirement to sample sufficient high-uncertainty/safe/progress data for *every* gamma.
    """
    q = float(cfg.active_quantile)
    if not 0.0 < q < 1.0:
        raise ValueError(f"frontier quantile must be in (0,1), got {q}")
    sigma, margin, prog = map(np.asarray, (sigma, margin, prog))
    gamma = np.zeros(len(sigma), dtype=float) if gamma is None else np.asarray(gamma, dtype=float)
    front = np.zeros(len(sigma), dtype=bool)
    planes = {}
    for g in np.unique(gamma):
        gm = np.isclose(gamma, g)
        sp = float(np.quantile(sigma[gm], q))
        mp = float(np.quantile(margin[gm], 1.0 - q))
        pp = float(np.quantile(prog[gm], q))
        front[gm] = ((sigma[gm] >= sp) & (margin[gm] <= mp) & (prog[gm] >= pp))
        planes[str(float(g))] = dict(sigma=sp, margin=mp, prog=pp, n=int(gm.sum()))
    if return_planes:
        return front, planes
    return front


def label_fresh(policy, unc, fresh, env, cfg, device):
    """Label valid windows using the fixed-schedule three-axis AND cell."""
    n = fresh["U"].shape[0]
    sigma = _sigma_of(policy, unc, fresh, cfg, device)
    Ln, Un = fresh["low5"].numpy(), fresh["U"].numpy()
    jerk = (np.linalg.norm(np.diff(Un, n=2, axis=1), axis=2).mean(axis=1)
            if Un.shape[1] >= 3 else np.zeros(n))
    net = Un.sum(axis=1); rg = Ln[:, :2]
    mono = (net * rg).sum(1) / (np.linalg.norm(net, axis=1) * np.linalg.norm(rg, axis=1) + 1e-9)
    gam = fresh["gamma"].numpy()
    if "socp_margin" in fresh:
        margin = np.asarray(fresh["socp_margin"], dtype=float)
    else:  # legacy diagnostic data only; current gather computes and gates this once per window
        margin = np.array([GM2.window_socp_margin(GX2.state_from_low5(Ln[j]), Un[j], env, gam[j])
                           for j in range(n)])
    if not np.isfinite(margin).all() or (margin <= 0.0).any():
        raise RuntimeError("a non-finite/non-positive SOCP face margin survived the exact certificate gate")
    prog = fresh["prog"]
    if _single_class_active(cfg):
        front = np.zeros(len(sigma), dtype=bool)             # (1) one class: nothing is "frontier"
        planes = {str(float(g)): dict(sigma=float(np.median(sigma)), margin=float(np.median(margin)),
                                      prog=float(np.median(prog))) for g in np.unique(gam)}
    else:
        front, planes = _front_mask(sigma, margin, prog, fresh.get("widx"), cfg, gamma=gam,
                                    return_planes=True)
    easy = ~front
    q = float(cfg.active_quantile)
    sp = float(np.median([p["sigma"] for p in planes.values()]))
    mp = float(np.median([p["margin"] for p in planes.values()]))
    pp = float(np.median([p["prog"] for p in planes.values()]))
    scores = dict(sigma=sigma, margin=margin, jerk=jerk, mono=mono, prog=prog,
                  quantile=q, sigma_plane=sp, margin_plane=mp, prog_plane=pp,
                  planes_by_gamma=planes)
    return np.where(easy)[0], np.where(front)[0], scores


def _fresh_batch_plan(n_e, n_f, mix, cap):
    """Largest batch (≤cap) honoring the easy:frontier ratio given availability. e.g. n_e=10,n_f=5 @ 7:3 ->
    B=min(10/.7, 5/.3)=14.3 -> 14 -> (10 easy, 4 frontier). If one class empty, use the other alone."""
    e_frac, f_frac = float(mix[0]), float(mix[1])
    if n_e == 0 or n_f == 0:
        return 0, 0                    # never substitute one class for a missing class
    cands = []
    if e_frac > 0:
        cands.append(n_e / e_frac)
    if f_frac > 0:
        cands.append(n_f / f_frac)
    B = min(cands) if cands else (n_e + n_f)
    B = int(min(B, cap))
    if B <= 0:
        return 0, 0
    ne = int(round(e_frac * B)); nf = B - ne
    return ne, nf


def _draw_strat(idx_pool, n, rids):
    """Stratified class draw (prob #1, batch level): round-robin ONE window per source rollout until n drawn,
    so the CFM update sees every gathered trajectory even when one dominates the window count."""
    by = {}
    for i in idx_pool:
        by.setdefault(int(rids[i]), []).append(int(i))
    groups = list(by.values())
    np.random.shuffle(groups)
    out, k = [], 0
    while len(out) < n:
        out.append(int(np.random.choice(groups[k % len(groups)])))
        k += 1
    return np.asarray(out, dtype=int)


def _draw_gamma_rid_balanced(idx_pool, n, gammas, rids, modes=None):
    """Draw gamma -> staircase mode -> rollout balanced, without replacement when possible.

    Balancing only rollout IDs hid a severe homotopy imbalance (one staircase supplied >70% of accepted
    paths).  Equal gamma quotas are allocated first, available staircase modes second, and source rollouts
    third.  A window is not reused until the corresponding finite pool is exhausted.
    """
    idx_pool = np.asarray(idx_pool, dtype=int)
    if n <= 0 or len(idx_pool) == 0:
        return np.array([], dtype=int)
    gammas = np.asarray(gammas, dtype=float)
    rids = np.asarray(rids, dtype=int)
    modes = np.asarray(["unknown"] * len(gammas), dtype=object) if modes is None else np.asarray(modes, dtype=object)
    by_g = {}
    for i in idx_pool:
        key = round(float(gammas[i]), 6)
        mode = str(modes[i]) if modes[i] is not None else "unknown"
        by_g.setdefault(key, {}).setdefault(mode, {}).setdefault(int(rids[i]), []).append(int(i))
    gkeys = list(by_g)
    np.random.shuffle(gkeys)
    quotas = {g: n // len(gkeys) + int(j < (n % len(gkeys))) for j, g in enumerate(gkeys)}
    out = []
    for g in gkeys:
        original = {m: {r: list(v) for r, v in rs.items()} for m, rs in by_g[g].items()}
        pools = None
        while quotas[g] > 0:
            if pools is None or not any(v for rs in pools.values() for v in rs.values()):
                pools = {m: {r: list(v) for r, v in rs.items()} for m, rs in original.items()}
                for rs in pools.values():
                    for v in rs.values():
                        np.random.shuffle(v)
            mkeys = [m for m, rs in pools.items() if any(rs.values())]
            np.random.shuffle(mkeys)
            for m in mkeys:
                rkeys = [r for r, v in pools[m].items() if v]
                np.random.shuffle(rkeys)
                if not rkeys:
                    continue
                out.append(pools[m][rkeys[0]].pop())
                quotas[g] -= 1
                if quotas[g] == 0:
                    break
    np.random.shuffle(out)
    return np.asarray(out, dtype=int)


class Pile:
    """Persistent positive pile, REVIVED with bounded staleness (user 2026-07-09). Differences vs the old 60k
    pile: (a) FIFO cap ~3k -> holds only the last ~10-20 iters' gathers, never trains on ancient behavior;
    (b) WITHOUT-replacement (least-recently-used) draws -> every sample gets a turn before any repeats
    ('refresh sometimes every samples'); (c) σ-dependent labels RECOMPUTED every pile_relabel_every iters with
    the current policy/GP (margin/prog are geometric -> stored once); windows migrate frontier->easy as the
    policy masters them."""

    def __init__(self, cap):
        self.cap = cap
        self.T = None                    # dict(grid, low5, hist, U) torch tensors
        self.margin = self.prog = self.widx = self.rid = self.it = self.use = None
        self.label = None                # 'easy' / 'frontier' per window

    def __len__(self):
        return 0 if self.T is None else self.T["U"].shape[0]

    def count(self, pool):
        return 0 if self.T is None else int((self.label == pool).sum())

    def add(self, fresh, easy_idx, frontier_idx, scores, t):
        n = fresh["U"].shape[0]
        lab = np.array(["easy"] * n, dtype=object); lab[frontier_idx] = "frontier"
        rid_g = t * 1000 + fresh.get("rid", np.zeros(n, int))   # globally-unique rollout id across iters
        new = dict(grid=fresh["grid"], low5=fresh["low5"], hist=fresh["hist"], U=fresh["U"],
                   gamma=fresh["gamma"])
        if self.T is None:
            self.T = {k: v.clone() for k, v in new.items()}
            self.margin = scores["margin"].copy(); self.prog = scores["prog"].copy()
            self.widx = fresh.get("widx", np.zeros(n, int)).copy(); self.rid = np.asarray(rid_g, int)
            self.it = np.full(n, t, int); self.use = np.zeros(n, float); self.label = lab
        else:
            self.T = {k: torch.cat([self.T[k], new[k]]) for k in self.T}
            self.margin = np.concatenate([self.margin, scores["margin"]])
            self.prog = np.concatenate([self.prog, scores["prog"]])
            self.widx = np.concatenate([self.widx, fresh.get("widx", np.zeros(n, int))])
            self.rid = np.concatenate([self.rid, np.asarray(rid_g, int)])
            self.it = np.concatenate([self.it, np.full(n, t, int)])
            self.use = np.concatenate([self.use, np.zeros(n, float)])
            self.label = np.concatenate([self.label, lab])
        if len(self) > self.cap:                                # FIFO: evict the OLDEST windows
            k = len(self) - self.cap
            self.T = {kk: v[k:] for kk, v in self.T.items()}
            for a in ("margin", "prog", "widx", "rid", "it", "use", "label"):
                setattr(self, a, getattr(self, a)[k:])

    def draw(self, pool, n, replace=False):
        idx = np.where(self.label == pool)[0] if self.T is not None else np.array([], int)
        if n <= 0 or len(idx) == 0:
            return np.array([], int)
        if replace:
            return np.random.choice(idx, n, replace=True)
        key = self.use[idx] + np.random.rand(len(idx))          # least-used first, random tie-break
        take = idx[np.argsort(key)[:min(n, len(idx))]]
        self.use[take] += 1.0
        return take

    def relabel(self, policy, unc, cfg, device):
        if self.T is None or len(self) == 0:
            return
        sigma = _sigma_of(policy, unc, self.T, cfg, device)
        gam = self.T["gamma"].numpy() if "gamma" in self.T else None
        front = (np.zeros(len(sigma), dtype=bool) if _single_class_active(cfg)
                 else _front_mask(sigma, self.margin, self.prog, self.widx, cfg, gamma=gam))
        self.label = np.array(["easy"] * len(self), dtype=object)
        self.label[front] = "frontier"

    def state_dict(self):
        return dict(cap=self.cap,
                    T=None if self.T is None else {k: v.detach().cpu().clone() for k, v in self.T.items()},
                    margin=self.margin, prog=self.prog, widx=self.widx, rid=self.rid,
                    it=self.it, use=self.use, label=self.label)

    @classmethod
    def from_state_dict(cls, state):
        if state is None:
            return None
        obj = cls(int(state["cap"]))
        obj.T = state.get("T")
        for name in ("margin", "prog", "widx", "rid", "it", "use", "label"):
            setattr(obj, name, state.get(name))
        return obj


def _cpu_tensor_dict(data):
    if data is None:
        return None
    return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in data.items()}


def _save_hp_atomic(policy, path, extra=None):
    """Crash-safe checkpoint commit in the destination directory."""
    tmp = path + ".tmp"
    HP.save_hp(policy, tmp, extra=extra)
    os.replace(tmp, path)


def _capture_train_state(iteration, opt, qbuf, covered, pile, teacher, history,
                         roll_reached, roll_coll, last, best_sr, sr0, best_safe_sr,
                         collapse_ct, best_probe, best_probe_cov, cooled, resume_signature=None):
    """Complete continuation state. The GP factorization is rebuilt deterministically from ``qbuf``."""
    teacher_state = None
    if teacher is not None:
        teacher_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
    return dict(
        version=2, iter=int(iteration), optimizer=opt.state_dict(), qbuf=_cpu_tensor_dict(qbuf),
        covered={str(float(g)): sorted(v) for g, v in covered.items()},
        pile=None if pile is None else pile.state_dict(), teacher_state=teacher_state,
        history=history, roll_reached=list(roll_reached), roll_coll=list(roll_coll), last=last,
        best_sr=float(best_sr), sr0=float(sr0), best_safe_sr=tuple(best_safe_sr),
        collapse_ct=int(collapse_ct), best_probe=tuple(best_probe), best_probe_cov=tuple(best_probe_cov),
        cooled=bool(cooled), resume_signature=resume_signature,
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        torch_rng=torch.random.get_rng_state().cpu(),
        cuda_rng=([x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None),
        cuda_device_count=(torch.cuda.device_count() if torch.cuda.is_available() else 0))


def _restore_rng_state(state):
    """Restore RNG last, after model/optimizer construction has consumed its own random draws."""
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    torch.random.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])


def _apply_train_state(state, opt, teacher, gammas, restore_rng=True, expected_signature=None,
                       allow_drift_keys=()):
    """Restore every continuation-critical object and return the non-model loop state."""
    if int(state.get("version", 0)) < 2:
        raise RuntimeError("unsupported/incomplete train-state checkpoint")
    if torch.cuda.is_available() and int(state.get("cuda_device_count", -1)) != torch.cuda.device_count():
        raise RuntimeError("visible CUDA topology differs from the saved continuation state")
    if teacher is not None and state.get("teacher_state") is None:
        raise RuntimeError("LwF is enabled but the continuation checkpoint has no fixed teacher state")
    if expected_signature is not None:
        got = dict(state.get("resume_signature") or {})
        exp = dict(expected_signature)
        for k in allow_drift_keys:                          # swept knobs: relaxed by explicit opt-in only
            got.pop(k, None); exp.pop(k, None)
        if got != exp:
            raise RuntimeError("resume recipe/parameter-group signature differs from the saved continuation")
    opt.load_state_dict(state["optimizer"])
    qbuf = _cpu_tensor_dict(state.get("qbuf"))
    covered = {g: set(state.get("covered", {}).get(str(float(g)), [])) for g in gammas}
    pile = Pile.from_state_dict(state.get("pile"))
    if teacher is not None and state.get("teacher_state") is not None:
        teacher.load_state_dict(state["teacher_state"])
    restored = dict(
        qbuf=qbuf, covered=covered, pile=pile, history=list(state.get("history", [])),
        roll_reached=deque(state.get("roll_reached", []), maxlen=100),
        roll_coll=deque(state.get("roll_coll", []), maxlen=100), last=state.get("last"),
        best_sr=float(state["best_sr"]), sr0=float(state["sr0"]),
        best_safe_sr=tuple(state["best_safe_sr"]), collapse_ct=int(state["collapse_ct"]),
        best_probe=tuple(state["best_probe"]), best_probe_cov=tuple(state["best_probe_cov"]),
        cooled=bool(state["cooled"]))
    if restore_rng:
        _restore_rng_state(state)                           # deliberately the final operation
    return restored


def _nested_state_equal(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_nested_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_nested_state_equal(x, y) for x, y in zip(a, b))
    return a == b


def _train_state_regression_roundtrip():
    """Small in-memory state roundtrip used by the independent semantic regression harness."""
    with _preserve_torch_rng():
        torch.manual_seed(913); np.random.seed(914); random.seed(915)
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        loss = model(torch.ones(2, 3)).square().mean(); loss.backward(); opt.step()
        teacher = torch.nn.Linear(3, 2)
        qbuf = dict(grid=torch.arange(12).reshape(2, 2, 3).float(), low5=torch.ones(2, 5),
                    hist=torch.zeros(2, 10, 2), U=torch.ones(2, 10, 2), tag=None)
        covered = {0.1: {"RURURURURU"}, 0.5: {"URURURURUR"}}
        state = _capture_train_state(
            7, opt, qbuf, covered, None, teacher, [{"iter": 7}], deque([1.0], maxlen=100),
            deque([0.0], maxlen=100), {"loss": 1.0}, 0.9, 0.4, (0.8, -0.2), 0,
            (0.9, 4), (4, 0.9), False)
        model2 = torch.nn.Linear(3, 2)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
        teacher2 = torch.nn.Linear(3, 2)
        restored = _apply_train_state(state, opt2, teacher2, [0.1, 0.5], restore_rng=True)
        np_now = np.random.get_state()
        np_ok = (np_now[0] == state["numpy_rng"][0] and
                 np.array_equal(np_now[1], state["numpy_rng"][1]) and
                 np_now[2:] == state["numpy_rng"][2:])
        cuda_ok = True
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            cuda_ok = all(torch.equal(a.cpu(), b.cpu())
                          for a, b in zip(torch.cuda.get_rng_state_all(), state["cuda_rng"]))
        return dict(
            optimizer=_nested_state_equal(opt.state_dict(), opt2.state_dict()),
            qbuf=_nested_state_equal(qbuf, restored["qbuf"]),
            covered=restored["covered"] == covered,
            teacher=_nested_state_equal(teacher.state_dict(), teacher2.state_dict()),
            history=restored["history"] == [{"iter": 7}], numpy_rng=np_ok,
            torch_rng=torch.equal(torch.random.get_rng_state(), state["torch_rng"]), cuda_rng=cuda_ok)


def _grad_rms(params):
    vals = [float(p.grad.pow(2).mean()) for p in params if p.grad is not None]
    return float(np.sqrt(np.mean(vals))) if vals else 0.0


def _make_origin_trust_anchor(teacher, env, gammas, device, n_per_gamma=32):
    """Fixed OOD-origin panel used to bound cumulative field drift from the lineage anchor."""
    if teacher is None or not hasattr(env, "obstacles"):
        return None
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    state = env.x0.detach().cpu().numpy(); goal = env.goal.detach().cpu().numpy()
    grid = torch.tensor(GF.axis_grid(state[:2], obs, rr), dtype=torch.float32)
    hist = torch.zeros(GF.K_HIST, 2, dtype=torch.float32)
    gen = torch.Generator().manual_seed(20260710)
    base_x = torch.randn(n_per_gamma, teacher.d, generator=gen)
    G, L, H, X, T = [], [], [], [], []
    for g in gammas:
        low = torch.tensor(GF.low5(state, goal, float(g)), dtype=torch.float32)
        G.append(grid[None].repeat(n_per_gamma, 1, 1, 1)); L.append(low[None].repeat(n_per_gamma, 1))
        H.append(hist[None].repeat(n_per_gamma, 1, 1)); X.append(base_x.clone())
        T.append(torch.full((n_per_gamma,), 0.5))
    G, L, H, X, T = [torch.cat(v).to(device) for v in (G, L, H, X, T)]
    with torch.no_grad():
        ref = teacher.forward(X, T, teacher._expand_ctx(teacher.ctx_from(G, L, H), len(X))).detach()
    return dict(grid=G, low5=L, hist=H, x=X, tau=T, ref=ref)


def update_flow_fresh(policy, opt, fresh, easy_idx, frontier_idx, mix, n_steps, cfg,
                      field_params, enc_params, device, demo=None, teacher=None, pile=None,
                      trust_anchor=None, env=None, escape_replay=None):
    nd_demo = int(round(cfg.demo_frac * cfg.batch_cap)) if (cfg.demo_frac > 0 and demo is not None) else 0
    fresh_target = cfg.batch_cap - nd_demo                  # fresh part of the TOTAL batch (e.g. 32-8=24)
    # hard-tail sub-quota: strip-context windows already in the certified pool (either class)
    hard_quota = int(getattr(cfg, "hard_quota", 0))
    guard_quota = int(getattr(cfg, "guard_quota", 0))
    strip_pool = np.array([], dtype=int)
    guard_pool = np.array([], dtype=int)
    guard_kind = np.array([], dtype="U8")
    if hard_quota > 0 and fresh is not None and "strip" in fresh:
        in_class = np.zeros(fresh["U"].shape[0], dtype=bool)
        in_class[easy_idx] = True; in_class[frontier_idx] = True
        strip_pool = np.where((fresh["strip"] != "") & in_class)[0]
    k_hard = int(min(hard_quota, max(fresh_target - 2, 0))) if len(strip_pool) else 0
    fresh_target -= k_hard
    if guard_quota > 0 and fresh is not None:
        guard_kind = (np.asarray(fresh["guard"]).astype("U8") if "guard" in fresh
                      else _guard_flags(fresh["low5"].numpy(), cfg))
        in_class = np.zeros(fresh["U"].shape[0], dtype=bool)
        in_class[easy_idx] = True; in_class[frontier_idx] = True
        guard_pool = np.where((guard_kind != "") & in_class)[0]
    k_guard = int(min(guard_quota, max(fresh_target - 2, 0))) if len(guard_pool) else 0
    fresh_target -= k_guard
    # Guard rows have their own quota; exclude them from ordinary draws so the stated
    # repair/preservation ratio is real rather than being changed by duplicate rows.
    easy_regular = np.setdiff1d(easy_idx, guard_pool, assume_unique=False) if k_guard else easy_idx
    frontier_regular = np.setdiff1d(frontier_idx, guard_pool, assume_unique=False) if k_guard else frontier_idx
    n_e, n_f = len(easy_regular), len(frontier_regular)
    pile_on = pile is not None and len(pile) > 0 and (cfg.fresh_frac < 1.0 or fresh is None)
    ne_pl = nf_pl = 0
    if pile_on:                                             # fresh takes its fresh_frac share; the pile fills
        tgt_e = int(round(float(mix[0]) * fresh_target))    # the rest; each backfills the other's shortfall
        tgt_f = fresh_target - tgt_e
        ne_fr = min(n_e, int(round(cfg.fresh_frac * tgt_e)))
        nf_fr = min(n_f, int(round(cfg.fresh_frac * tgt_f)))
        ne_pl = min(pile.count("easy"), tgt_e - ne_fr)
        nf_pl = min(pile.count("frontier"), tgt_f - nf_fr)
        ne_fr = min(n_e, tgt_e - ne_pl); nf_fr = min(n_f, tgt_f - nf_pl)
        ne = ne_fr + ne_pl; nf = nf_fr + nf_pl
    elif _single_class_active(cfg):
        ne, nf = min(n_e, fresh_target), 0                  # (1) single class: fill the batch from it alone
        ne_fr, nf_fr = ne, nf
    else:
        ne, nf = _fresh_batch_plan(n_e, n_f, mix, fresh_target)
        ne_fr, nf_fr = ne, nf
    if ne == 0 or (nf == 0 and not _single_class_active(cfg)):
        return None                                         # both classes mandatory (unless single-class ablation)
    B = ne + nf + k_hard + k_guard
    if B == 0:
        return None
    nd = demo["U"].shape[0] if demo is not None else 0
    rids = fresh.get("rid") if fresh is not None else None
    fresh_gamma = fresh["gamma"].numpy() if fresh is not None else None
    fresh_modes = fresh.get("mode") if fresh is not None else None
    policy.train()
    losses, fgr, egr, rid_ns, rid_doms, batch_gamma_counts, batch_mode_counts = [], [], [], [], [], [], []
    functional_steps, anchor_drifts, rollback_count = [], [], 0
    hard_used, hard_x0_used, hard_oob_fracs, escape_used = 0, 0, [], 0
    used_easy_all, used_frontier_all = set(), set()         # fresh rows actually drawn into the batch (viz)
    for _ in range(n_steps):
        Gs, Ls, Hs, Us, rid_all = [], [], [], [], []
        parts = []
        hard_rows = np.array([], dtype=int)
        guard_rows = np.array([], dtype=int)
        if k_hard > 0:                                      # strip windows first (batch rows 0..k_hard-1)
            hard_rows = (_draw_gamma_rid_balanced(strip_pool, k_hard, fresh_gamma, rids, fresh_modes)
                         if (fresh_gamma is not None and rids is not None)
                         else np.random.choice(strip_pool, k_hard, replace=len(strip_pool) < k_hard))
            parts.append(hard_rows)
            hard_used += len(hard_rows)
        if k_guard > 0:
            # Protect both sides of the task boundary.  These are ordinary, whole-rollout
            # exact-certified targets immediately INSIDE the strips, never demo backfill.
            op = guard_pool[guard_kind[guard_pool] == "origin"]
            gp = guard_pool[guard_kind[guard_pool] == "goal"]
            ko = min(len(op), k_guard // 2); kg = min(len(gp), k_guard - ko)
            if ko + kg < k_guard:
                rem = np.setdiff1d(guard_pool, np.concatenate([op[:0], gp[:0]]))
                extra = np.random.choice(rem, k_guard - ko - kg, replace=len(rem) < k_guard - ko - kg)
            else:
                extra = np.array([], dtype=int)
            chunks = []
            for pool, count in ((op, ko), (gp, kg)):
                if count:
                    chunks.append(_draw_gamma_rid_balanced(pool, count, fresh_gamma, rids, fresh_modes)
                                  if (fresh_gamma is not None and rids is not None)
                                  else np.random.choice(pool, count, replace=len(pool) < count))
            guard_rows = np.concatenate(chunks + ([extra] if len(extra) else [])).astype(int)
            parts.append(guard_rows)
        if ne_fr > 0:
            _drawn_e = (_draw_gamma_rid_balanced(easy_regular, ne_fr, fresh_gamma, rids, fresh_modes)
                        if (fresh_gamma is not None and rids is not None)
                        else np.random.choice(easy_regular, ne_fr, replace=True))
            parts.append(_drawn_e); used_easy_all.update(int(i) for i in _drawn_e)
        if nf_fr > 0:
            _drawn_f = (_draw_gamma_rid_balanced(frontier_regular, nf_fr, fresh_gamma, rids, fresh_modes)
                        if (fresh_gamma is not None and rids is not None)
                        else np.random.choice(frontier_regular, nf_fr, replace=True))
            parts.append(_drawn_f); used_frontier_all.update(int(i) for i in _drawn_f)
        if parts:
            bi_np = np.concatenate(parts)
            bi = torch.as_tensor(bi_np, dtype=torch.long)
            Gs.append(fresh["grid"][bi]); Ls.append(fresh["low5"][bi])
            Hs.append(fresh["hist"][bi]); Us.append(fresh["U"][bi])
            if rids is not None:
                rid_all.append(rids[bi_np])
            if fresh_gamma is not None:
                ug, cg = np.unique(np.round(fresh_gamma[bi_np], 6), return_counts=True)
                batch_gamma_counts.append({str(float(g)): int(c) for g, c in zip(ug, cg)})
            if fresh_modes is not None:
                um, cm = np.unique(np.asarray(fresh_modes, dtype=object)[bi_np].astype(str), return_counts=True)
                batch_mode_counts.append({str(m): int(c) for m, c in zip(um, cm)})
        if ne_pl > 0 or nf_pl > 0:                          # pile part: LRU without-replacement draw
            pi_np = np.concatenate([pile.draw("easy", ne_pl, cfg.pile_replace),
                                    pile.draw("frontier", nf_pl, cfg.pile_replace)]).astype(int)
            if len(pi_np):
                pi = torch.as_tensor(pi_np, dtype=torch.long)
                Gs.append(pile.T["grid"][pi]); Ls.append(pile.T["low5"][pi])
                Hs.append(pile.T["hist"][pi]); Us.append(pile.T["U"][pi])
                rid_all.append(pile.rid[pi_np])
        G = torch.cat(Gs).to(device); L = torch.cat(Ls).to(device)
        H = torch.cat(Hs).to(device); U = torch.cat(Us).to(device)
        if rid_all:                                         # rid-diversity stats of THIS update's non-demo part
            _, cnts = np.unique(np.concatenate(rid_all), return_counts=True)
            rid_ns.append(len(cnts)); rid_doms.append(float(cnts.max()) / cnts.sum())
        if nd_demo > 0:                                        # δ anchor: mix pretraining-demo windows in
            di = torch.randint(0, nd, (nd_demo,))
            G = torch.cat([G, demo["grid"][di].to(device)]); L = torch.cat([L, demo["low5"][di].to(device)])
            H = torch.cat([H, demo["hist"][di].to(device)]); U = torch.cat([U, demo["U"][di].to(device)])
        max_fstep = float(getattr(cfg, "max_functional_step", 0.0))
        before_policy = before_opt = anchor = v_before = None
        if max_fstep > 0:
            na = min(B, 128)
            ai = torch.linspace(0, max(B - 1, 0), na, device=G.device).long()
            Ga, La, Ha, Ua = G[ai], L[ai], H[ai], U[ai]
            xa = 0.5 * (Ua / policy.u_max).reshape(na, policy.d)
            ta = torch.full((na,), 0.5, device=G.device)
            with torch.no_grad():
                v_before = policy.forward(xa, ta, policy._expand_ctx(policy.ctx_from(Ga, La, Ha), na)).detach()
            anchor = (Ga, La, Ha, xa, ta)
            before_policy = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            before_opt = copy.deepcopy(opt.state_dict())
        x0_map = {}
        if len(hard_rows) > 0 and str(getattr(cfg, "hard_x0", "")) == "oob" and env is not None:
            pool_map, oob_frac = _harvest_bad_x0(policy, fresh, hard_rows, env, cfg, device)
            hard_oob_fracs.append(oob_frac)
            x0_map.update({pos: pool_map[int(pr)] for pos, pr in enumerate(hard_rows) if int(pr) in pool_map})
            hard_x0_used += len(x0_map)
        guard_x0_used = 0; guard_positions = set()
        if (len(guard_rows) > 0 and str(getattr(cfg, "guard_x0", "")) in ("inbounds", "near-hard")
                and env is not None):
            good_map, _ = _harvest_good_x0(policy, fresh, guard_rows, env, cfg, device)
            mapped = {k_hard + pos: good_map[int(pr)] for pos, pr in enumerate(guard_rows)
                      if int(pr) in good_map}
            x0_map.update(mapped); guard_x0_used = len(mapped); guard_positions = set(mapped)
        loss = float(getattr(cfg, "cfm_eta", 1.0)) * _cfm_loss_x0(
            policy, U, policy.ctx_from(G, L, H), x0_override=(x0_map or None))
        if x0_map and float(getattr(cfg, "endpoint_eta", 0.0)) > 0:
            mr = sorted(x0_map)
            mi = torch.as_tensor(mr, dtype=torch.long, device=device)
            X = torch.stack([x0_map[r] for r in mr]).to(device)
            C = policy.ctx_from(G[mi], L[mi], H[mi])
            for j in range(int(cfg.nfe_explore)):
                tt = torch.full((len(mr),), j / int(cfg.nfe_explore), device=device)
                X = X + (1.0 / int(cfg.nfe_explore)) * policy.forward(X, tt, C)
            X1 = (U[mi] / policy.u_max).reshape(len(mr), policy.d)
            if guard_positions and bool(getattr(cfg, "guard_teacher_endpoint", False)) and teacher is not None:
                gi = [j for j, r in enumerate(mr) if r in guard_positions]
                if gi:
                    with torch.no_grad():
                        Xt = torch.stack([x0_map[mr[j]] for j in gi]).to(device)
                        Ct = C[torch.as_tensor(gi, dtype=torch.long, device=device)]
                        for q in range(int(cfg.nfe_explore)):
                            tq = torch.full((len(gi),), q / int(cfg.nfe_explore), device=device)
                            Xt = Xt + (1.0 / int(cfg.nfe_explore)) * teacher.forward(Xt, tq, Ct)
                        X1[torch.as_tensor(gi, dtype=torch.long, device=device)] = Xt
            loss = loss + float(cfg.endpoint_eta) * ((X - X1) ** 2).mean()
        esc_q = int(getattr(cfg, "escape_quota", 0))
        if escape_replay is not None and esc_q > 0:
            rg = np.asarray(escape_replay["gamma"])
            groups = [p for p in (np.where(np.isclose(rg, float(g)))[0] for g in cfg.gammas) if len(p)]
            picks = []
            for j, pool in enumerate(groups):
                count = esc_q // len(groups) + (j < esc_q % len(groups))
                if count and len(pool):
                    picks.append(np.random.choice(pool, count, replace=len(pool) < count))
            if picks:
                ei = torch.as_tensor(np.concatenate(picks), dtype=torch.long)
                Ge = escape_replay["grid"][ei].to(device); Le = escape_replay["low5"][ei].to(device)
                He = escape_replay["hist"][ei].to(device); Xe = escape_replay["x0"][ei].to(device)
                Ye = escape_replay["target_x"][ei].to(device); Ce = policy.ctx_from(Ge, Le, He)
                for q in range(int(cfg.nfe_explore)):
                    tq = torch.full((len(ei),), q / int(cfg.nfe_explore), device=device)
                    Xe = Xe + (1.0 / int(cfg.nfe_explore)) * policy.forward(Xe, tq, Ce)
                loss = loss + float(getattr(cfg, "escape_eta", 1.0)) * ((Xe - Ye) ** 2).mean()
                escape_used += len(ei)
        if cfg.lwf_eta > 0 and teacher is not None and demo is not None:   # η anchor: LwF on demo contexts
            nl = min(nd, cfg.batch_cap)
            li = torch.randint(0, nd, (nl,))
            Gd, Ld, Hd = demo["grid"][li].to(device), demo["low5"][li].to(device), demo["hist"][li].to(device)
            Ud = demo["U"][li].to(device); B_ = Ud.shape[0]
            x1 = (Ud / policy.u_max).reshape(B_, policy.d); x0 = torch.randn_like(x1)
            tau = torch.rand(B_, device=x1.device).clamp(1e-4, 1.0)
            x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
            v_s = policy.forward(x_tau, tau, policy._expand_ctx(policy.ctx_from(Gd, Ld, Hd), B_))
            with torch.no_grad():
                v_t = teacher.forward(x_tau, tau, teacher._expand_ctx(teacher.ctx_from(Gd, Ld, Hd), B_))
            loss = loss + cfg.lwf_eta * ((v_s - v_t) ** 2).mean()
        opt.zero_grad(); loss.backward()
        fgr.append(_grad_rms(field_params)); egr.append(_grad_rms(enc_params))
        if cfg.enc_grad_clip > 0 and enc_params:
            torch.nn.utils.clip_grad_norm_(enc_params, cfg.enc_grad_clip)
        if cfg.field_grad_clip > 0 and field_params:
            torch.nn.utils.clip_grad_norm_(field_params, cfg.field_grad_clip)
        opt.step(); losses.append(float(loss.detach()))
        if anchor is not None:
            Ga, La, Ha, xa, ta = anchor
            with torch.no_grad():
                va = policy.forward(xa, ta, policy._expand_ctx(policy.ctx_from(Ga, La, Ha), len(xa)))
                fstep = float((va - v_before).norm(dim=1).mean() /
                              v_before.norm(dim=1).mean().clamp_min(1e-9))
            functional_steps.append(fstep)
            anchor_drift = 0.0
            if trust_anchor is not None:
                with torch.no_grad():
                    av = policy.forward(
                        trust_anchor["x"], trust_anchor["tau"],
                        policy._expand_ctx(policy.ctx_from(
                            trust_anchor["grid"], trust_anchor["low5"], trust_anchor["hist"]),
                                           len(trust_anchor["x"])))
                    anchor_drift = float((av - trust_anchor["ref"]).norm(dim=1).mean() /
                                         trust_anchor["ref"].norm(dim=1).mean().clamp_min(1e-9))
                anchor_drifts.append(anchor_drift)
            if fstep > max_fstep or anchor_drift > float(getattr(cfg, "max_anchor_drift", float("inf"))):
                policy.load_state_dict(before_policy)
                opt.load_state_dict(before_opt)
                rollback_count += 1
                break
    return dict(loss=float(np.mean(losses)) if losses else float("nan"),
                loss_steps=[float(v) for v in losses],                     # per-step curve (telemetry only)
                fstep_steps=[float(v) for v in functional_steps],
                anchor_steps=[float(v) for v in anchor_drifts],
                field_grad_rms=float(np.mean(fgr)) if fgr else 0.0,
                enc_grad_rms=float(np.mean(egr)) if egr else 0.0, batch=(ne, nf, nd_demo),
                batch_hard=int(k_hard), hard_x0=int(hard_x0_used), strip_pool=int(len(strip_pool)),
                batch_guard=int(k_guard), guard_pool=int(len(guard_pool)),
                guard_x0=int(guard_x0_used if 'guard_x0_used' in locals() else 0),
                escape_replay=int(escape_used),
                strip_oob_frac=float(np.mean(hard_oob_fracs)) if hard_oob_fracs else float("nan"),
                n_pile=ne_pl + nf_pl, pile_batch=(ne_pl, nf_pl),
                rid_n=float(np.mean(rid_ns)) if rid_ns else float("nan"),
                rid_dom=float(np.mean(rid_doms)) if rid_doms else float("nan"),
                functional_step=float(np.mean(functional_steps)) if functional_steps else 0.0,
                anchor_drift=float(anchor_drifts[-1]) if anchor_drifts else 0.0,
                rollback=bool(rollback_count),
                used_easy_idx=np.array(sorted(used_easy_all), dtype=int),      # fresh rows fed to the grad step
                used_frontier_idx=np.array(sorted(used_frontier_all), dtype=int),
                batch_gamma_counts=batch_gamma_counts[-1] if batch_gamma_counts else {},
                batch_mode_counts=batch_mode_counts[-1] if batch_mode_counts else {})


def _executed_horizon_tensors(recs):
    """Contexts plus the H controls that were actually executed after each context.

    Receding-horizon deployment executes only ``proposal[0]`` and replans.  Training on the remaining nine
    unexecuted proposal actions caused a measured full-plan/first-action inconsistency and goal overshoot.
    These targets are the coherent closed-loop segments that the accepted trajectory actually witnessed.
    """
    if not recs:
        return None
    horizon = int(np.asarray(recs[0][3]).shape[0])
    n = len(recs) - horizon + 1
    if n <= 0:
        return None
    G, L, H, _proposal = GE._to_t(recs[:n])
    executed = np.stack([
        np.stack([np.asarray(recs[i + j][3], dtype=np.float32)[0] for j in range(horizon)], axis=0)
        for i in range(n)
    ], axis=0)
    return G, L, H, torch.as_tensor(executed, dtype=torch.float32)


def _coverage_target_pool(covered_modes):
    """Uncovered one-swap homotopy frontier; seed from the canonical diagonal mode."""
    covered_modes = set(covered_modes)
    seed = "RURURURURU"
    source = covered_modes if covered_modes else {seed}
    frontier = set()
    for word in source:
        frontier.update(GM.neighbors(word))
    frontier -= covered_modes
    if not covered_modes:
        frontier.add(seed)
    if not frontier:
        frontier = set(GM.STAIRCASES) - covered_modes
    return sorted(frontier)


def _gather_fresh(policy, unc, env, cfg, gammas, beta, K, target_e, target_f, qbuf, covered, device,
                  gamma_offset=0):
    """Gather valid2 rollouts until the rollout budget and BOTH actual classes are populated.

    ``K`` is the nominal valid-rollout budget (never smaller than the gamma count). If a class is still
    missing at K, gathering automatically continues to ``gather_attempt_cap``. No demo or opposite-class
    window hides a missing class. Absolute-iteration round-robin gamma starts and per-gamma diagnostics
    expose condition imbalance without requiring an often-impossible valid2 trajectory at every gamma in
    every individual update.
    """
    if qbuf is not None:
        qfeat = GE._buffer_feat(policy, qbuf, "phi_s", cfg.s, cfg.gp_buf, device)
        if qfeat is not None:
            unc.set_buffer(qfeat)
    goal_np = env.goal.detach().cpu().numpy()
    gG, gL, gH, gU, prog, socp_margin, cert_residual, rid, widx, wgamma, modes, proposal_targets = (
        [], [], [], [], [], [], [], [], [], [], [], [])
    paths = []                                             # executed trajectories of KEPT rollouts (for viz)
    path_status = []                                        # per-KEPT-path {ok|valid2_fail|collide} (viz)
    reached, coll = [], []
    valid, att, gi = 0, 0, 0
    valid_gammas, attempted_gammas = [], []
    # EMERGENT gamma-curriculum (user 2026-07-14): gather every SOCP-certified window across ALL gammas,
    # but do NOT block the update on gammas that have zero certified windows yet. From the pretrained the
    # strict low gammas (0.1/0.2) are 0% valid (their SOCP needs interior clearance the policy lacks); the
    # curriculum EMERGES from the certificate — low gammas join automatically as the frontier lifts
    # clearance. The demo/LwF anchor holds them meanwhile. This treats every gamma uniformly (no dropping).
    emergent = bool(getattr(cfg, "emergent_gamma", False))
    K_eff = max(K, cfg.min_rollouts) if emergent else max(K, cfg.min_rollouts, len(gammas))
    max_att = max(K_eff, cfg.gather_attempt_cap)
    classes_ready = gamma_ready = gamma_class_ready = mode_ready = False
    target_hit_ready = not bool(getattr(cfg, "mode_hit_gate", False))
    need_gammas = list(gammas)
    valid_modes = {float(g): set() for g in gammas}
    audit = dict(queried_rollouts=0, queried_windows=0, strict_reached=0, valid2_pass=0,
                 coherent_windows_checked=0, coherent_windows_certified=0,
                 accepted_reached=0, accepted_steps=[], targeted_attempts=0,
                 targeted_accepted=0, target_hits=0, targeted_modes=Counter(),
                 recovery_attempts=0, recovery_accepted=0, recovery_origin=0, recovery_goal=0)
    rkind = []                                             # per-window rollout kind (normal/recovery_*)
    rec_i = 0                                              # recovery attempt counter (strip alternation)
    x0_default = None                                      # captured lazily; env untouched when the arm is off
    while att < max_att:
        if (valid >= K_eff and classes_ready and (emergent or (gamma_ready and gamma_class_ready)) and
                mode_ready and target_hit_ready):
            break
        att += 1
        # Round-robin initially, then spend the remaining attempt budget on conditions whose accepted
        # easy/frontier quotas are still missing. Equal attempts cannot repair gamma=.1's much lower Valid2
        # acceptance rate; deficit scheduling can, without weakening any gate.
        # emergent: round-robin ALL gammas uniformly (so low gammas keep getting sampled and join the
        # moment they become certifiable); strict: focus remaining attempts on gammas missing their quota.
        active_gammas = list(gammas) if emergent else (need_gammas if need_gammas else gammas)
        g = active_gammas[(gamma_offset + gi) % len(active_gammas)]; gi += 1
        attempted_gammas.append(float(g))
        target_word = None
        # Hard-tail repair: a deterministic share of attempts starts ON a failure strip (executed +
        # exact-certified acceptance below is UNCHANGED; only the rollout's initial condition differs).
        recovery_frac = float(getattr(cfg, "recovery_frac", 0.0))
        use_recovery = recovery_frac > 0 and (((att * 4783) % 10000) < int(10000 * recovery_frac))
        rollout_kind = "normal"
        if use_recovery:
            which = "origin" if rec_i % 2 == 0 else "goal"
            rec_i += 1
            if x0_default is None:
                x0_default = env.x0.detach().clone()
            env.x0 = torch.as_tensor(_sample_recovery_start(cfg, which), dtype=x0_default.dtype)
            rollout_kind = f"recovery_{which}"
            audit["recovery_attempts"] += 1
        # A fixed target for the whole rollout makes exploration temporally coherent. It only proposes;
        # unchanged executed Valid2 and exact coherent-window certificates remain the acceptance authority.
        targeted_frac = float(getattr(cfg, "targeted_frac", 0.0))
        use_target = (not use_recovery) and targeted_frac > 0 and (((att * 7919) % 10000) < int(10000 * targeted_frac))
        if use_target:
            pool = _coverage_target_pool(covered[g])
            target_word = pool[(gamma_offset + att) % len(pool)] if pool else None
            audit["targeted_attempts"] += int(target_word is not None)
        try:
            with _target_proposal_override(bool(target_word is not None and
                                                getattr(cfg, "target_perp_brake", False))):
                out = GR.fm_deploy(policy, env, float(g), T=cfg.T, target=target_word,
                                   tilt=dict(unc=unc, beta=beta, N=cfg.N, s=cfg.s, broad=0, feature="phi_s",
                                             temp=cfg.temp, churn=cfg.churn, safe_filter=cfg.safe_filter,
                                             n_target=int(getattr(cfg, "n_target", 40)),
                                             align_temp=float(getattr(cfg, "align_temp", 0.45))),
                                   nfe=cfg.nfe_explore, record=True, verify_fn=GM2.window_label_cheap,
                                   reach=cfg.reach, device=device)
        finally:
            if use_recovery:                               # NEVER leak a strip start to probes/eval/callers
                env.x0 = x0_default.detach().clone()
        reached.append(1.0 if out["reached"] else 0.0)
        coll.append(1.0 if SR.path_collides(out["path"], env) else 0.0)
        if not out["recs"]:
            continue
        audit["queried_rollouts"] += 1
        audit["queried_windows"] += len(out["recs"])
        # Query memory records selected proposals even when their resulting trajectory is rejected.  Those
        # regions have been queried and must not remain spuriously maximally novel on the next attempt.
        Gq, Lq, Hq, Uq = GE._to_t(out["recs"])
        qbuf = GE._cat(qbuf, Gq[::3], Lq[::3], Hq[::3], Uq[::3], cap=cfg.qbuf_cap)
        qfeat = GE._buffer_feat(policy, qbuf, "phi_s", cfg.s, cfg.gp_buf, device)
        if qfeat is not None:
            unc.set_buffer(qfeat)
        if out["reached"]:
            final_dist = float(np.linalg.norm(np.asarray(out["path"][-1], float) - goal_np))
            if final_dist >= cfg.reach + 1e-5:
                raise RuntimeError(f"gather marked reached at distance {final_dist:.6f} >= {cfg.reach}")
            audit["strict_reached"] += 1
        if getattr(cfg, "ablate_socp", False):                 # (2) free-space + progress, NO multi-step SOCP
            traj_ok = GM2.traj_valid2(out["path"], env, float(g), check_socp=False)
        elif getattr(cfg, "ablate_progress", False):            # (3) taskspace + SOCP, NO progress condition
            _p = np.asarray(out["path"], dtype=float)
            traj_ok = len(_p) >= 11 and GM.in_taskspace(_p) and GM.socp_ok(_p, env, float(g))
        else:
            traj_ok = GM2.traj_valid2(out["path"], env, float(g))  # exact Valid2: taskspace+progress+SOCP
        # WINDOW-LEVEL (user 2026-07-14): validity is per-WINDOW, not per-trajectory. Harvest every locally
        # valid2 window (taskspace ∧ progress ∧ SOCP, checked in the per-window loop below) even from
        # trajectories that later collide. This makes valid SAMPLES plentiful -> no low-gamma starvation, no
        # emergent-gamma / recovery needed. The pretrained is 'confused' early (a locally-valid window can
        # belong to an eventually-colliding rollout) but SR rises as the field is corrected.
        if not traj_ok and not getattr(cfg, "window_level", False):
            continue
        audit["valid2_pass"] += 1
        pth = np.asarray(out["path"], dtype=float)             # AD-HOC dither gate (prob #2), TWO-TIER:
        d0T = np.linalg.norm(pth[0] - goal_np) - np.linalg.norm(pth[-1] - goal_np)
        if not out["reached"] and cfg.traj_prog_min > 0:       # gate active only when traj_prog_min > 0
            if d0T < 0.3:                                      # tier-1 hard floor: true stay-and-dither -> drop
                continue
            if d0T < cfg.traj_prog_min and valid >= cfg.min_rollouts:
                continue                                       # tier-2 soft: sub-par traj only fills the min quota
        coherent = _executed_horizon_tensors(out["recs"])
        if coherent is None:
            continue
        G, L, H, U = coherent
        keep, wp, wm = [], [], []
        wr = []
        for i in range(U.shape[0]):
            p_i, pts, d = _window_progress(L[i].numpy(), U[i].numpy(), env)
            if not GM.in_taskspace(pts):
                continue
            if not getattr(cfg, "ablate_progress", False):
                if not GM2.approach_ok(d):                      # net-progress ≥ 0.10 (valid2)
                    continue
                if p_i < min(cfg.valid_prog_floor, 0.5 * d[0]):  # reject safe-STATIONARY
                    continue
            audit["coherent_windows_checked"] += 1
            if getattr(cfg, "ablate_socp", False):              # (2) margin axis = geometric min clearance
                face_margin = float(GM2.window_min_clearance(GX2.state_from_low5(L[i].numpy()),
                                                             U[i].numpy(), env))
                residual = 0.0
                if not np.isfinite(face_margin) or face_margin <= 1e-6:
                    continue
            else:
                plan_ok, face_margin, residual = GM2.window_socp_stats(
                    GX2.state_from_low5(L[i].numpy()), U[i].numpy(), env, float(g))
                if not plan_ok:                                 # never train on an infeasible planned target
                    continue
            audit["coherent_windows_certified"] += 1
            keep.append(i); wp.append(p_i); wm.append(face_margin); wr.append(residual)
        if not keep:
            continue
        ki = torch.as_tensor(keep)
        gG.append(G[ki]); gL.append(L[ki]); gH.append(H[ki]); gU.append(U[ki]); prog.extend(wp)
        socp_margin.extend(wm); cert_residual.extend(wr)
        rid.extend([valid] * len(keep)); paths.append(pth)  # rollout id per window + executed traj (viz/diversity)
        path_status.append("collide" if SR.path_collides(pth, env)     # window-level harvest flag: a valid2_fail
                           else ("ok" if traj_ok else "valid2_fail"))  # or colliding rollout still yields ≥1 window
        widx.extend(keep)                                   # in-traj window index (0 = the initial escape window)
        wgamma.extend([float(g)] * len(keep))
        rkind.extend([rollout_kind] * len(keep))
        sid = (GM.staircase_id(pth, reach=cfg.reach)
               if (out["reached"] and rollout_kind == "normal") else None)
        mode = sid if sid is not None else ("unreached" if rollout_kind == "normal" else rollout_kind)
        modes.extend([mode] * len(keep))
        proposal_targets.extend([(target_word if target_word is not None else "ordinary")] * len(keep))
        valid_gammas.append(float(g))
        audit["accepted_reached"] += int(bool(out["reached"]))
        audit["accepted_steps"].append(int(out.get("steps", len(out["path"]) - 1)))
        if rollout_kind != "normal":
            audit["recovery_accepted"] += 1
            audit["recovery_origin" if rollout_kind.endswith("origin") else "recovery_goal"] += 1
        if target_word is not None:
            audit["targeted_accepted"] += 1
            audit["target_hits"] += int(sid == target_word)
            audit["targeted_modes"][mode] += 1
        # Strip-start rollouts NEVER count as coverage modes: min_modes_per_gamma readiness and the
        # coverage bookkeeping keep their original from-origin semantics.
        if sid is not None:
            valid_modes[float(g)].add(sid)
        if out["reached"]:                                      # coverage tracking (not a gate)
            if sid is not None:
                covered[g].add(sid)
        valid += 1
        partial = dict(grid=torch.cat(gG), low5=torch.cat(gL), hist=torch.cat(gH), U=torch.cat(gU),
                       gamma=torch.tensor(wgamma, dtype=torch.float32),
                       prog=np.asarray(prog, dtype=float), rid=np.asarray(rid, dtype=int),
                       socp_margin=np.asarray(socp_margin, dtype=float),
                       cert_residual=np.asarray(cert_residual, dtype=float),
                       widx=np.asarray(widx, dtype=int), mode=np.asarray(modes, dtype=object),
                       proposal_target=np.asarray(proposal_targets, dtype=object), paths=paths,
                       rkind=np.asarray(rkind, dtype=object))
        ei, fi, _ = label_fresh(policy, unc, partial, env, cfg, device)
        classes_ready = len(ei) >= target_e and len(fi) >= target_f
        gamma_ready = all(any(np.isclose(vg, gg) for vg in valid_gammas) for gg in gammas)
        ge = partial["gamma"].numpy()[ei]; gf = partial["gamma"].numpy()[fi]
        qe = max(1, int(np.ceil(target_e / max(len(gammas), 1)))) if target_e else 0
        qf = max(1, int(np.ceil(target_f / max(len(gammas), 1)))) if target_f else 0
        class_need = [gg for gg in gammas
                      if (np.isclose(ge, gg).sum() < qe or np.isclose(gf, gg).sum() < qf)]
        gamma_class_ready = not class_need
        base_min_modes = int(getattr(cfg, "min_modes_per_gamma", 0))
        min_modes = int(getattr(cfg, "active_min_modes", base_min_modes)
                        if getattr(cfg, "min_modes_schedule", ()) else base_min_modes)
        mode_need = [gg for gg in gammas if len(valid_modes[float(gg)]) < min_modes]
        mode_ready = not mode_need
        target_hit_ready = _target_hit_ready(audit, cfg)
        need_gammas = list(dict.fromkeys(class_need + mode_need))
    if not gG:
        return None, qbuf, reached, coll, valid, att, audit
    fresh = dict(grid=torch.cat(gG), low5=torch.cat(gL), hist=torch.cat(gH), U=torch.cat(gU),
                 gamma=torch.tensor(wgamma, dtype=torch.float32),
                 prog=np.asarray(prog, dtype=float), rid=np.asarray(rid, dtype=int),
                 socp_margin=np.asarray(socp_margin, dtype=float),
                 cert_residual=np.asarray(cert_residual, dtype=float),
                 widx=np.asarray(widx, dtype=int), mode=np.asarray(modes, dtype=object),
                 proposal_target=np.asarray(proposal_targets, dtype=object), paths=paths,
                 path_status=np.asarray(path_status, dtype=object),
                 rkind=np.asarray(rkind, dtype=object),
                 rollout_gamma=np.asarray(valid_gammas, dtype=float),
                 attempted_gamma=np.asarray(attempted_gammas, dtype=float),
                 gamma_ready=bool(gamma_ready), classes_ready=bool(classes_ready),
                 gamma_class_ready=bool(gamma_class_ready), mode_ready=bool(mode_ready),
                 target_hit_ready=bool(target_hit_ready), active_min_modes=int(min_modes),
                 modes_per_gamma={str(g): sorted(valid_modes[float(g)]) for g in gammas})
    fresh["strip"] = _strip_flags(fresh["low5"].numpy(), cfg)   # per-window context-strip membership
    ready = (valid >= K_eff and classes_ready and (emergent or (gamma_ready and gamma_class_ready)) and
             mode_ready and target_hit_ready)
    accepted_steps = audit.pop("accepted_steps")
    audit["targeted_modes"] = dict(audit["targeted_modes"])
    audit.update(ready=bool(ready), gamma_ready=bool(gamma_ready), classes_ready=bool(classes_ready),
                 gamma_class_ready=bool(gamma_class_ready), valid_rollouts=int(valid),
                 mode_ready=bool(mode_ready), target_hit_ready=bool(target_hit_ready),
                 active_min_modes=int(min_modes),
                 modes_per_gamma={str(g): sorted(valid_modes[float(g)]) for g in gammas},
                 unmet_gammas=[float(g) for g in need_gammas],
                 accepted_step_mean=(float(np.mean(accepted_steps)) if accepted_steps else None))
    # No demo/opposite-class fallback: an incomplete all-gamma class quota means no gradient update.
    return (fresh if ready else None), qbuf, reached, coll, valid, att, audit


def _load_demo(cfg):
    import pretrain_repr as PR
    prefix = str(getattr(cfg, "demo_prefix", "dr05_") or "dr05_")
    G, L, H, U = PR.load_data(prefix, [str(g) for g in cfg.gammas], cfg.demo_cap)
    return dict(grid=G, low5=L, hist=H, U=U)


# ------------------------------------------------------------------ hard-tail repair helpers
def _sample_recovery_start(cfg, which):
    """Sample a strip start state (x, y, vx, vy) from the configured band (training-stream np RNG)."""
    b = cfg.recovery_origin_band if which == "origin" else cfg.recovery_goal_band
    x = np.random.uniform(b[0], b[1]); y = np.random.uniform(b[2], b[3])
    vx = np.random.uniform(b[4], b[5]); vy = np.random.uniform(b[6], b[7])
    return np.array([x, y, vx, vy], dtype=np.float32)


def _strip_flags(low5_batch, cfg):
    """Per-window strip membership from the CONTEXT state (origin strip incl. the clean origin edge;
    goal strip = above the goal line where every stratum-2 overshoot wandered)."""
    ob = getattr(cfg, "recovery_origin_band", CurConfig.recovery_origin_band)
    gb = getattr(cfg, "recovery_goal_band", CurConfig.recovery_goal_band)
    flags = np.zeros(low5_batch.shape[0], dtype="U8")
    for j in range(low5_batch.shape[0]):
        s = np.asarray(GX2.state_from_low5(np.asarray(low5_batch[j])), dtype=float)
        if s[1] <= ob[3] and s[0] <= ob[1] + 0.2:
            flags[j] = "origin"
        elif s[1] >= gb[2] and s[0] >= gb[0] - 0.2:
            flags[j] = "goal"
    return flags


def _guard_flags(low5_batch, cfg):
    """Certified interior approach bands used only to preserve boundary behavior.

    The guard is deliberately separated from recovery: it samples windows from unchanged
    whole-rollout Valid2 acceptance just before either empty strip, so hard-tail repair cannot
    erase already-successful approaches while learning recovery contexts.
    """
    ob = getattr(cfg, "recovery_origin_band", CurConfig.recovery_origin_band)
    gb = getattr(cfg, "recovery_goal_band", CurConfig.recovery_goal_band)
    flags = np.zeros(low5_batch.shape[0], dtype="U8")
    for j in range(low5_batch.shape[0]):
        s = np.asarray(GX2.state_from_low5(np.asarray(low5_batch[j])), dtype=float)
        if ob[3] < s[1] <= max(0.65, ob[3] + 0.35) and s[0] <= ob[1] + 0.25:
            flags[j] = "origin"
        elif gb[2] - 0.55 <= s[1] < gb[2] and s[0] >= gb[0] - 0.35:
            flags[j] = "goal"
    return flags


@torch.no_grad()
def _integrate_x0(policy, ctx, X0, nfe):
    n = X0.shape[0]
    ctx_e = policy._expand_ctx(ctx, n)
    x = X0.clone()
    for i in range(nfe):
        tau = torch.full((n,), i / nfe, device=x.device)
        x = x + (1.0 / nfe) * policy.forward(x, tau, ctx_e)
    return (x.reshape(n, policy.T, 2) * policy.u_max).clamp(-policy.u_max, policy.u_max)


@torch.no_grad()
def _window_oob_mask(st4, U, env):
    """Open-loop H-step positions exit the task box (the absorber criterion of seed12_tail_trace)."""
    pos = GR.di_rollout_batch(np.asarray(st4, np.float32), U.detach().cpu().numpy(), env.dt)
    lo, hi = -float(GM.EPS_TASK), float(GM.GRID_M) + float(GM.EPS_TASK)
    return ((pos < lo) | (pos > hi)).any(axis=(1, 2))


@torch.no_grad()
def _harvest_bad_x0(policy, fresh, rows, env, cfg, device):
    """For each sub-quota row, harvest base latents whose CURRENT faithful map exits the task box at that
    context (codex handoff step 4: oversample OOB-mapping base-noise states; standard CFM target kept).
    Falls back to None (random x0) when the OOB set is empty or is the majority (mean shift: random draws
    already cover it)."""
    out, oob_fracs = {}, []
    for r in rows:
        gT = fresh["grid"][r].to(device); lT = fresh["low5"][r].to(device); hT = fresh["hist"][r].to(device)
        ctx = policy.ctx_from(gT[None] if gT.dim() == 3 else gT, lT[None] if lT.dim() == 1 else lT,
                              hT[None] if hT.dim() == 2 else hT)
        if ctx.shape[0] == 1:
            ctx = ctx[0]
        X0 = torch.randn(int(cfg.hard_x0_cand), policy.d, device=device)
        U = _integrate_x0(policy, ctx, X0, cfg.nfe_explore)
        st4 = GX2.state_from_low5(np.asarray(fresh["low5"][r]))
        fixed_seed = int(getattr(cfg, "fixed_origin_x0_seed", -1))
        if fixed_seed >= 0 and np.linalg.norm(np.asarray(st4)[:2]) < 0.05:
            gen = torch.Generator(device=device).manual_seed(fixed_seed)
            out[int(r)] = torch.randn(policy.d, generator=gen, device=device)
            oob_fracs.append(float("nan"))
            continue
        bad = _window_oob_mask(st4, U, env)
        frac = float(bad.mean()); oob_fracs.append(frac)
        if 0.0 < frac and (frac <= 0.5 or bool(getattr(cfg, "hard_x0_allow_majority", False))):
            # Cover the extreme/saturated fiber, not merely the first rare OOB draw.
            pos = GR.di_rollout_batch(np.asarray(st4, np.float32), U.detach().cpu().numpy(), env.dt)
            lo, hi = -float(GM.EPS_TASK), float(GM.GRID_M) + float(GM.EPS_TASK)
            if str(getattr(cfg, "hard_x0_select", "worst")) == "random-oob":
                pick = int(np.random.choice(np.where(bad)[0]))
            else:
                severity = np.maximum(lo - pos, pos - hi).max(axis=(1, 2))
                severity[~bad] = -np.inf
                pick = int(np.argmax(severity))
            out[int(r)] = X0[pick].detach().clone()
    return out, (float(np.mean(oob_fracs)) if oob_fracs else 0.0)


@torch.no_grad()
def _harvest_good_x0(policy, fresh, rows, env, cfg, device):
    """Pair interior guard targets with latents whose current map remains in bounds.

    This is the preservation complement of hard-tail OOB harvesting: it covers successful
    latent fibers at the same boundary contexts, preventing recovery gradients from replacing
    all modes with one mean direction.
    """
    out, good_fracs = {}, []
    for r in rows:
        gT = fresh["grid"][r].to(device); lT = fresh["low5"][r].to(device); hT = fresh["hist"][r].to(device)
        ctx = policy.ctx_from(gT[None] if gT.dim() == 3 else gT, lT[None] if lT.dim() == 1 else lT,
                              hT[None] if hT.dim() == 2 else hT)
        if ctx.shape[0] == 1:
            ctx = ctx[0]
        n_cand = int(cfg.hard_x0_cand)
        near_hard = str(getattr(cfg, "guard_x0", "")) == "near-hard"
        if near_hard and int(getattr(cfg, "fixed_origin_x0_seed", -1)) >= 0:
            gen = torch.Generator(device=device).manual_seed(int(cfg.fixed_origin_x0_seed))
            ref = torch.randn(policy.d, generator=gen, device=device)
            n_local = n_cand // 2
            X0 = torch.randn(n_cand, policy.d, device=device)
            X0[:n_local] = ref[None] + 0.5 * torch.randn(n_local, policy.d, device=device)
        else:
            ref = None
            X0 = torch.randn(n_cand, policy.d, device=device)
        U = _integrate_x0(policy, ctx, X0, cfg.nfe_explore)
        st4 = GX2.state_from_low5(np.asarray(fresh["low5"][r]))
        good = ~_window_oob_mask(st4, U, env)
        good_fracs.append(float(good.mean()))
        if good.any():
            gi = np.where(good)[0]
            if ref is not None:
                dist = (X0[torch.as_tensor(gi, device=device)] - ref).square().sum(dim=1)
                pick = int(gi[int(dist.argmin())])
            else:
                pick = int(gi[0])
            out[int(r)] = X0[pick].detach().clone()
    return out, (float(np.mean(good_fracs)) if good_fracs else 0.0)


def _cfm_loss_x0(policy, U_controls, ctx, x0_override=None):
    """FlowPolicy.cfm_loss with optional per-row x0 override (importance sampling on the base noise ONLY;
    interpolation, target = x1 - x0, and the MSE objective are byte-identical to flow_policy.py:60-72)."""
    B = U_controls.shape[0]
    x1 = (U_controls / policy.u_max).reshape(B, policy.d)
    x0 = torch.randn_like(x1)
    if x0_override:
        for r, v in x0_override.items():
            if r < B:
                x0[r] = v
    tau = torch.rand(B, device=x1.device).clamp(1e-4, 1.0)
    x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
    target = x1 - x0
    pred = policy.forward(x_tau, tau, policy._expand_ctx(ctx, B))
    per = ((pred - target) ** 2).mean(dim=1)                 # two-stage mean == FlowPolicy.cfm_loss exactly
    return per.mean()


# ---- wall-plug scene variant (user-approved 2026-07-11; reference: analysis/scene_walls.py) ----
_WALL_STEP = 5.0 / 13.0
_WALL_PLUGS4 = [(_WALL_STEP, -0.2, 0.2), (5.0 - _WALL_STEP, 5.2, 0.2),
                (-0.2, _WALL_STEP, 0.2), (5.2, 5.0 - _WALL_STEP, 0.2)]
# n=8 (user 2026-07-13, corrected placement): close BOTH corner openings so the OOB escape (origin
# down/left departure / goal up/right overshoot) becomes a detectable COLLISION. Use AXIS-ALIGNED plugs
# adjacent to the corner (NOT the diagonal (-r,-r)/(5+r,5+r) — that leaves a straight-down/left gap):
# origin gets (0,-r) below + (-r,0) left; goal gets (5+r,5) right + (5,5+r) above. The robot must start
# at (eps,eps) (--start-eps) since the origin itself now sits on these plugs' boundary.
_WALL_PLUGS8 = _WALL_PLUGS4 + [
    (0.0, -0.2, 0.2), (-0.2, 0.0, 0.2),                       # origin: directly below + directly left
    (5.2, 5.0, 0.2), (5.0, 5.2, 0.2),                        # goal: directly right + directly above
]


def _apply_wall_plugs(env, n):
    """Extend the obstacle perimeter by lattice circles at the corner openings (n=2 bottom+top, n=4 one
    per side, n=8 near-full corner closure). Same radius/spacing as the lattice; env is shared so training
    AND every probe/measure/eval run on the walled scene."""
    if not n:
        return env
    plugs = (_WALL_PLUGS4[:2] if n == 2 else _WALL_PLUGS8 if n == 8 else _WALL_PLUGS4)
    extra = torch.tensor(plugs, dtype=env.obstacles.dtype)
    env.obstacles = torch.cat([env.obstacles, extra], dim=0)
    return env


# fixed absorber-probe contexts (from analysis/seed12_tail_trace.json: seed-12 last-in-bounds state at
# gamma=.5 and the near-goal g.5/s3 wander state; velocities set to the adverse quadrant of each stratum).
# 'deep' = hardest observed failure states; 'mild' = strip entry, where repair must show first.
_STRIP_PROBE_STATES = {"origin": (np.array([0.12, -0.10, 0.25, -0.22], np.float32), 0.5),
                       "goal": (np.array([4.94, 5.10, 0.05, 0.15], np.float32), 0.5),
                       "origin_mild": (np.array([0.10, -0.02, 0.20, -0.10], np.float32), 0.5),
                       "goal_mild": (np.array([4.85, 5.02, 0.10, 0.10], np.float32), 0.5)}


@torch.no_grad()
def _strip_probe(policy, env, cfg, device, n_lat=256):
    """win-OOB fraction of fresh latents at the two fixed failing contexts (RNG-isolated)."""
    res = {}
    with _preserve_torch_rng():
        goal = env.goal.detach().cpu().numpy()
        for name, (st4, g) in _STRIP_PROBE_STATES.items():
            grid_np = GF.axis_grid(st4[:2], env.obstacles.detach().cpu().numpy(), float(env.r_robot))
            l5_np = GF.low5(st4, goal, g)
            h_np = GF.hist_pad(np.zeros((0, 2)), GF.K_HIST)
            gT = torch.tensor(grid_np, device=device); lT = torch.tensor(l5_np, device=device)
            hT = torch.tensor(h_np, device=device)
            ctx = policy.ctx_from(gT, lT, hT)
            if ctx.shape[0] == 1:
                ctx = ctx[0]
            torch.manual_seed(31337)
            X0 = torch.randn(n_lat, policy.d, device=device)
            U = _integrate_x0(policy, ctx, X0, cfg.nfe_explore)
            res[name] = float(_window_oob_mask(st4, U, env).mean())
    return res


@contextmanager
def _preserve_torch_rng():
    """Make diagnostics observational: they must not choose any later training randomness."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _escape_probe(policy, env, cfg, device, M=8, T=60, g=0.5):
    """Origin-escape stability (user 2026-07-09): M FAITHFUL rollouts truncated at T steps. Returns
    (frac that escape ||p||>1, circular std of the initial heading [rad], mean net-progress d0-dT).
    Stable escape = esc→1, hstd small-and-steady; the warm-up pathology = esc jumping + hstd large."""
    import math
    esc, heads, prog = [], [], []
    goal = env.goal.detach().cpu().numpy()
    with _preserve_torch_rng():
        for _ in range(M):
            out = GR.fm_deploy(policy, env, float(g), T=T, temp=1.0, nfe=cfg.nfe_explore, device=device)
            p = np.asarray(out["path"], dtype=float)
            esc.append(1.0 if (np.linalg.norm(p, axis=1) > 1.0).any() else 0.0)
            v = p[min(10, len(p) - 1)] - p[0]
            if np.linalg.norm(v) > 1e-6:
                heads.append(math.atan2(v[1], v[0]))
            prog.append(float(np.linalg.norm(p[0] - goal) - np.linalg.norm(p[-1] - goal)))
    if heads:
        R = min(1.0, math.hypot(float(np.mean(np.cos(heads))), float(np.mean(np.sin(heads)))))
        hstd = math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-9))))
    else:
        hstd = float("nan")
    return float(np.mean(esc)), float(hstd), float(np.mean(prog))


def _cov_probe(policy, env, cfg, device, M=50, g=0.5):
    """INSTANTANEOUS per-iter measurement (user 2026-07-09): M=50 faithful rollouts at one γ ->
    (SR, CR, coverage = #distinct staircase ids among the REACHED rollouts, the ids). NOT cumulative —
    this is the diversity of THIS snapshot's policy, so mode-collapse shows as cov -> 1 even while SR is high."""
    with _preserve_torch_rng():
        rows, _, paths = SR.eval_policy(policy, env, gammas=[g], M=M, T_max=cfg.T, reach=cfg.reach,
                                        temp=1.0, device=device, keep_paths=M, log=lambda *a, **k: None)
    goal = env.goal.detach().cpu().numpy()
    ids = set()
    for p in paths[g]:
        p = np.asarray(p, dtype=float)
        if np.linalg.norm(p[-1] - goal) < cfg.reach:
            sid = GM.staircase_id(p)
            if sid is not None:
                ids.add(sid)
    return float(rows[g]["SR"]), float(rows[g]["CR"]), len(ids), sorted(ids)


def _measure(policy, env, cfg, device):
    with _preserve_torch_rng():
        rows, agg, _ = SR.eval_policy(policy, env, gammas=list(cfg.gammas), M=cfg.M_measure, T_max=cfg.T,
                                      reach=cfg.reach, temp=1.0, device=device, log=lambda *a, **k: None)
    return rows, agg


def _save_viz_db(fresh, scores, easy_idx, frontier_idx, mix, path, it, cap=4096,
                 used_easy_idx=None, used_frontier_idx=None):
    """Save this iter's labeled fresh windows (easy/frontier + σ/margin/jerk/mono/prog) for the stack viz.
    used_easy_idx/used_frontier_idx = the fresh rows actually DRAWN into THIS iter's gradient step, so the
    viz can circle the batch-USED windows against the full gathered pool (the draw is far sparser than the
    pool). Non-fresh batch parts (pile/hard/guard) are not fresh rows and are not marked here."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = fresh["U"].shape[0]
    label = np.array(["easy"] * n, dtype=object); label[frontier_idx] = "frontier"
    used_easy = np.zeros(n, dtype=bool); used_frontier = np.zeros(n, dtype=bool)
    if used_easy_idx is not None and len(used_easy_idx):
        used_easy[np.asarray(used_easy_idx, dtype=int)] = True
    if used_frontier_idx is not None and len(used_frontier_idx):
        used_frontier[np.asarray(used_frontier_idx, dtype=int)] = True
    sel = np.arange(n)
    if n > cap:                                                # stratified subsample to cap
        want_f = int(round(float(mix[1]) * cap))
        fi = np.random.choice(frontier_idx, min(want_f, len(frontier_idx)), replace=False) if len(frontier_idx) else np.array([], int)
        rem = cap - len(fi)
        ei = np.random.choice(easy_idx, min(rem, len(easy_idx)), replace=False) if len(easy_idx) else np.array([], int)
        sel = np.concatenate([ei, fi]).astype(int)
    db = dict(iter=it, mix=list(mix), label=list(label[sel]),
              grid=fresh["grid"][sel].cpu(), low5=fresh["low5"][sel].cpu(),
              hist=fresh["hist"][sel].cpu(), U=fresh["U"][sel].cpu(),
              gamma=fresh["gamma"][sel].cpu(),
              sigma=scores["sigma"][sel], margin=scores["margin"][sel], jerk=scores["jerk"][sel],
              mono=scores["mono"][sel], prog=scores["prog"][sel],
              quantile=scores["quantile"], sigma_plane=scores["sigma_plane"],
              margin_plane=scores["margin_plane"], prog_plane=scores["prog_plane"],
              planes_by_gamma=scores.get("planes_by_gamma", {}),
              cert_residual=fresh.get("cert_residual", np.full(n, np.nan))[sel],
              rid=fresh.get("rid", np.zeros(n, int))[sel],           # rollout id per window (diversity check)
              widx=fresh.get("widx", np.zeros(n, int))[sel],         # in-traj window index (0 = initial escape)
              mode=list(np.asarray(fresh.get("mode", np.array(["unknown"] * n, dtype=object)), dtype=object)[sel]),
              proposal_target=list(np.asarray(
                  fresh.get("proposal_target", np.array(["ordinary"] * n, dtype=object)), dtype=object)[sel]),
              rollout_gamma=fresh.get("rollout_gamma", np.array([])),
              attempted_gamma=fresh.get("attempted_gamma", np.array([])),
              gamma_ready=bool(fresh.get("gamma_ready", False)),
              classes_ready=bool(fresh.get("classes_ready", False)),
              gamma_class_ready=bool(fresh.get("gamma_class_ready", False)),
              mode_ready=bool(fresh.get("mode_ready", False)),
              target_hit_ready=bool(fresh.get("target_hit_ready", False)),
              active_min_modes=int(fresh.get("active_min_modes", 0)),
              modes_per_gamma=fresh.get("modes_per_gamma", {}),
              used_easy=used_easy[sel], used_frontier=used_frontier[sel],   # batch-USED masks (circle in viz)
              path_status=list(np.asarray(fresh.get("path_status", np.array([], dtype=object)), dtype=object)),
              paths=[np.asarray(p) for p in fresh.get("paths", [])])  # executed trajs of the gathered rollouts
    torch.save(db, path)


def _resume_signature(cfg, freeze_enc, enc_lr_mult):
    """Fields that must remain identical for an exact optimizer/query-state continuation."""
    sig = dict(version=2, target="executed_closed_loop_horizon", margin="real_face_m",
                freeze_enc=bool(freeze_enc), enc_lr_mult=float(enc_lr_mult), lr=float(cfg.lr),
                kernel=cfg.kernel, ell=float(cfg.ell), lam=float(cfg.lam), gp_buf=int(cfg.gp_buf),
                qbuf_cap=int(cfg.qbuf_cap), beta=float(cfg.beta), batch=int(cfg.batch_cap),
                demo_frac=float(cfg.demo_frac), lwf_eta=float(cfg.lwf_eta),
                valid_prog_floor=float(cfg.valid_prog_floor), nfe_explore=int(cfg.nfe_explore),
                targeted_frac=float(cfg.targeted_frac), n_target=int(cfg.n_target),
                align_temp=float(cfg.align_temp),
                min_modes_per_gamma=int(cfg.min_modes_per_gamma),
                max_functional_step=float(cfg.max_functional_step),
                max_anchor_drift=float(cfg.max_anchor_drift),
                field_grad_clip=float(cfg.field_grad_clip), enc_grad_clip=float(cfg.enc_grad_clip),
                quantile_schedule=[list(x) for x in cfg.quantile_schedule],
                mix_start=list(cfg.mix_start), mix_end=list(cfg.mix_end),
                early_until=int(cfg.early_until), cooldown_from=int(cfg.cooldown_from),
                inner_steps=[int(cfg.early_inner), int(cfg.inner_steps), int(cfg.cooldown_inner)],
                fresh_frac=float(cfg.fresh_frac), gammas=list(cfg.gammas),
                # hard-tail repair arm (any change here forces a deliberate model-only branch)
                recovery_frac=float(getattr(cfg, "recovery_frac", 0.0)),
                recovery_origin_band=list(getattr(cfg, "recovery_origin_band", ())),
                recovery_goal_band=list(getattr(cfg, "recovery_goal_band", ())),
                hard_quota=int(getattr(cfg, "hard_quota", 0)),
                guard_quota=int(getattr(cfg, "guard_quota", 0)),
                guard_x0=str(getattr(cfg, "guard_x0", "")),
                hard_x0=str(getattr(cfg, "hard_x0", "")),
                hard_x0_cand=int(getattr(cfg, "hard_x0_cand", 0)),
                escape_quota=int(getattr(cfg, "escape_quota", 0)),
                escape_eta=float(getattr(cfg, "escape_eta", 1.0)),
                wall_plugs=int(getattr(cfg, "wall_plugs", 0)),
                ablate_curriculum=bool(getattr(cfg, "ablate_curriculum", False)),
                ablate_socp=bool(getattr(cfg, "ablate_socp", False)),
                ablate_progress=bool(getattr(cfg, "ablate_progress", False)))
    # Keep disabled-arm signatures byte-compatible with the already-running pure
    # WALLS-4 controls.  The new fields become continuation-critical only on the
    # explicitly branched phased arm.
    if getattr(cfg, "phased_curriculum", False):
        sig.update(phased_curriculum=True,
                   phase_sr_threshold=float(getattr(cfg, "phase_sr_threshold", 0.85)),
                   phase_sr_patience=int(getattr(cfg, "phase_sr_patience", 2)))
    if (getattr(cfg, "min_modes_schedule", ()) or getattr(cfg, "mode_hit_gate", False) or
            getattr(cfg, "target_perp_brake", False)):
        sig.update(min_modes_schedule=[list(x) for x in getattr(cfg, "min_modes_schedule", ())],
                   mode_hit_gate=bool(getattr(cfg, "mode_hit_gate", False)),
                   min_target_hits=int(getattr(cfg, "min_target_hits", 1)),
                   target_perp_brake=bool(getattr(cfg, "target_perp_brake", False)))
    return sig


# ---------------------------------------------------------------- main loop
def run_expand_cur(policy, env, cfg: CurConfig, device="cpu", outdir=None, log=print,
                   freeze_enc=True, enc_lr_mult=0.0, tag="", resume_state=None,
                   teacher_ckpt=None, train_seed=None, escape_replay=None):
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    gammas = list(cfg.gammas)
    field_params = list(policy.trunk.parameters()) + list(policy.head.parameters())
    enc = policy.encoder_modules()
    if freeze_enc or enc_lr_mult <= 0:
        for p in enc:
            p.requires_grad_(False)
        enc_params, groups = [], [{"params": field_params, "lr": cfg.lr}]
    else:
        for p in enc:
            p.requires_grad_(True)
        enc_params = enc
        groups = [{"params": field_params, "lr": cfg.lr}, {"params": enc, "lr": cfg.lr * enc_lr_mult}]
    opt = torch.optim.Adam(groups)
    unc = GPUncertainty(kernel=cfg.kernel, lengthscale=cfg.ell, lam=cfg.lam, normalize=True)
    demo = _load_demo(cfg) if (cfg.demo_frac > 0 or cfg.lwf_eta > 0) else None
    teacher = None
    if cfg.lwf_eta > 0:
        if teacher_ckpt:
            teacher, _ = HP.load_hp(teacher_ckpt, device="cpu")
            teacher = teacher.to(device).eval()
        else:
            teacher = copy.deepcopy(policy).eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
    resume_sig = _resume_signature(cfg, freeze_enc, enc_lr_mult)
    recipe = dict(algorithm="and_quantile_fixed_absolute_schedule", start_iter=cfg.start_iter,
                  additional_iters=cfg.iters, quantile_schedule=[list(x) for x in cfg.quantile_schedule],
                  beta=cfg.beta, mix_start=list(cfg.mix_start), mix_end=list(cfg.mix_end),
                  early_until=cfg.early_until, cooldown_from=cfg.cooldown_from,
                  rollouts_per_iter=cfg.rollouts_per_iter, gather_attempt_cap=cfg.gather_attempt_cap,
                  batch=cfg.batch_cap, demo_frac=cfg.demo_frac, valid_prog_floor=cfg.valid_prog_floor,
                  valid2_unchanged=True, demo_backfill=False, lr=cfg.lr, lwf_eta=cfg.lwf_eta,
                  inner_steps=[cfg.early_inner, cfg.inner_steps, cfg.cooldown_inner],
                  gammas=list(cfg.gammas), gamma_rotation="absolute_iteration_round_robin",
                  gather_requires_every_gamma=True, min_valid_rollouts=max(cfg.min_rollouts, len(cfg.gammas)),
                  strict_gather_reach=cfg.reach, executed_gate="traj_valid2",
                  planned_window_socp_gate=True, training_target="executed_closed_loop_horizon",
                  coherent_coverage_proposals=dict(fraction=cfg.targeted_frac, n_target=cfg.n_target,
                                                   align_temp=cfg.align_temp,
                                                   min_modes_per_gamma=cfg.min_modes_per_gamma,
                                                   min_modes_schedule=[list(x) for x in getattr(cfg, "min_modes_schedule", ())],
                                                   exact_target_hit_gate=bool(getattr(cfg, "mode_hit_gate", False)),
                                                   min_target_hits=int(getattr(cfg, "min_target_hits", 1)),
                                                   perpendicular_braking_proposal=bool(getattr(cfg, "target_perp_brake", False)),
                                                   acceptance="unchanged_valid2_and_exact_certificate"),
                  socp_margin="minimum_feasible_real_face_m",
                  frontier_planes="per_gamma", batch_sampling="gamma_then_mode_then_rollout_balanced",
                  probe_rng_isolated=True, nfe_explore=cfg.nfe_explore,
                  field_grad_clip=cfg.field_grad_clip, teacher_ckpt=teacher_ckpt,
                  max_functional_step=cfg.max_functional_step,
                  max_anchor_drift=cfg.max_anchor_drift,
                  stateful_resume=resume_state is not None, legacy_prime_iters=cfg.legacy_prime_iters,
                  train_seed=train_seed, resume_signature=resume_sig,
                  wall_plugs=int(getattr(cfg, "wall_plugs", 0)),
                  phased_curriculum=dict(enabled=bool(getattr(cfg, "phased_curriculum", False)),
                                         sr_threshold=float(getattr(cfg, "phase_sr_threshold", 0.85)),
                                         patience=int(getattr(cfg, "phase_sr_patience", 2)),
                                         switch="first sustained measurement window; irreversible"),
                  ablations=dict(curriculum=bool(getattr(cfg, "ablate_curriculum", False)),
                                 socp=bool(getattr(cfg, "ablate_socp", False)),
                                 progress=bool(getattr(cfg, "ablate_progress", False))),
                  hard_tail=dict(recovery_frac=cfg.recovery_frac,
                                 recovery_origin_band=list(cfg.recovery_origin_band),
                                 recovery_goal_band=list(cfg.recovery_goal_band),
                                 hard_quota=cfg.hard_quota, guard_quota=cfg.guard_quota,
                                 guard_x0=cfg.guard_x0,
                                 hard_x0=cfg.hard_x0,
                                 hard_x0_cand=cfg.hard_x0_cand,
                                 acceptance="unchanged_valid2_and_exact_certificate_started_on_strip",
                                 coverage_and_mode_quotas_exclude_recovery=True,
                                 evidence="analysis/seed12_tail_trace.md"),
                  tag=tag)
    if outdir:
        with open(os.path.join(outdir, "recipe.json"), "w") as f:
            json.dump(recipe, f, indent=2)
        with open(os.path.join(outdir, f"recipe_start_{cfg.start_iter}.json"), "w") as f:
            json.dump(recipe, f, indent=2)
    log(f"[fixed_and_expand{('/'+tag) if tag else ''}] abs={cfg.start_iter}+{cfg.iters} "
        f"rollouts/iter={cfg.rollouts_per_iter} attempt_cap={cfg.gather_attempt_cap} "
        f"q_schedule={cfg.quantile_schedule} valid_prog_floor={cfg.valid_prog_floor} "
        f"min_rollouts={cfg.min_rollouts} traj_prog_min={cfg.traj_prog_min} "
        f"mix {cfg.mix_start}->{cfg.mix_end} "
        f"inner {cfg.early_inner}/{cfg.inner_steps}/{cfg.cooldown_inner} freeze_enc={freeze_enc} "
        f"phase_abs={cfg.early_until}/{cfg.cooldown_from} enc_lr_mult={enc_lr_mult} lr={cfg.lr} "
        f"β={cfg.beta} demo_frac={cfg.demo_frac} lwf_eta={cfg.lwf_eta} NO_DEMO_BACKFILL"
        f" targeted={cfg.targeted_frac:.2f}(n={cfg.n_target},alignT={cfg.align_temp})"
        + (f" demo={demo['U'].shape[0]}" if demo is not None else "")
        + (f" | PILE fresh_frac={cfg.fresh_frac} warmup={cfg.warmup_gather} cap={cfg.pile_cap} "
           f"replace={cfg.pile_replace} relabel_every={cfg.pile_relabel_every}" if
           (cfg.fresh_frac < 1.0 or cfg.warmup_gather > 0) else ""), flush=True)

    qbuf = None
    pile = Pile(cfg.pile_cap) if (cfg.fresh_frac < 1.0 or cfg.warmup_gather > 0) else None
    covered = {g: set() for g in gammas}
    roll_reached, roll_coll = deque(maxlen=100), deque(maxlen=100)
    history = []
    hist_path = os.path.join(outdir, "history.json") if outdir else None
    if resume_state is None and hist_path and cfg.start_iter > 0 and os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
    easy_idx, frontier_idx, scores = np.array([], int), np.array([], int), None
    last = dict(loss=float("nan"), field_grad_rms=0.0, enc_grad_rms=0.0, batch=(0, 0, 0))
    mix = tuple(cfg.mix_start)
    cooled = cfg.start_iter >= cfg.cooldown_from
    best_probe = (-1.0, -1)                                # (SR50, coverage), restricted to CR50==0
    best_probe_cov = (-1, -1.0)                            # (coverage, SR50), also restricted to CR50==0
    best_sr = best_safe_sr = sr0 = collapse_ct = None
    if resume_state is not None:
        if int(resume_state.get("version", 0)) < 2:
            raise RuntimeError("unsupported/incomplete train-state checkpoint")
        if int(resume_state["iter"]) != int(cfg.start_iter):
            raise RuntimeError(f"train-state iter {resume_state['iter']} != requested start {cfg.start_iter}")
        _drift_keys = (("beta", "mix_start", "mix_end", "quantile_schedule",
                        "phased_curriculum", "phase_sr_threshold", "phase_sr_patience",
                        "gp_buf", "qbuf_cap")   # greedy it20 schedule knobs (2026-07-14)
                       if getattr(cfg, "resume_allow_recipe_drift", False) else ())
        rs = _apply_train_state(resume_state, opt, teacher, gammas, restore_rng=True,
                                expected_signature=resume_sig, allow_drift_keys=_drift_keys)
        qbuf, covered, pile = rs["qbuf"], rs["covered"], rs["pile"]
        history, roll_reached, roll_coll = rs["history"], rs["roll_reached"], rs["roll_coll"]
        last = rs["last"] if rs["last"] is not None else last
        best_sr, sr0, best_safe_sr = rs["best_sr"], rs["sr0"], rs["best_safe_sr"]
        collapse_ct, best_probe, best_probe_cov = rs["collapse_ct"], rs["best_probe"], rs["best_probe_cov"]
        cooled = rs["cooled"]
        log(f"[resume] restored full state v{resume_state['version']} at it{cfg.start_iter}: "
            f"qbuf={0 if qbuf is None else qbuf['U'].shape[0]} history={len(history)}", flush=True)
    elif cooled:
        for grp in opt.param_groups:
            grp["lr"] *= cfg.cooldown_lr_mult

    trust_anchor = _make_origin_trust_anchor(teacher, env, gammas, device)

    rows0, agg0 = _measure(policy, env, cfg, device)
    log(f"it{cfg.start_iter:05d} SR {agg0['SR']:.2f} CR {agg0['CR']:.2f} | resume baseline "
        f"(pretrained repr{getattr(policy, 'repr_dim', '?')}, faithful temp=1)", flush=True)
    if not history or history[-1].get("iter") != cfg.start_iter:
        history.append(dict(iter=cfg.start_iter, SR=agg0["SR"], CR=agg0["CR"], gdist=agg0["mean_goal_dist"],
                        rows={str(g): rows0[g] for g in gammas}, n_pos=0, beta=cfg.beta,
                        mix=list(cfg.mix_start), n_easy=0, n_mid=0, n_frontier=0, loss=float("nan"),
                        field_grad_rms=0.0, enc_grad_rms=0.0, online_SR=0.0, online_CR=0.0,
                        covered={str(g): 0 for g in gammas}))
    if resume_state is None:
        best_sr = sr0 = agg0["SR"]; best_safe_sr = (-1.0, float("inf")); collapse_ct = 0
    legacy_prime_remaining = cfg.legacy_prime_iters if resume_state is None else 0

    def train_state_at(iteration):
        return _capture_train_state(iteration, opt, qbuf, covered, pile, teacher, history,
                                    roll_reached, roll_coll, last, best_sr, sr0, best_safe_sr,
                                    collapse_ct, best_probe, best_probe_cov, cooled,
                                    resume_signature=resume_sig)

    final_iter = cfg.start_iter
    for local_t in range(1, cfg.iters + 1):
        t = cfg.start_iter + local_t                         # EVERY schedule and artifact uses absolute t
        final_iter = t
        beta = cfg.beta
        cfg.active_quantile = _quantile_at(cfg.quantile_schedule, t)
        cfg.active_min_modes = _int_schedule_at(
            getattr(cfg, "min_modes_schedule", ()), t, cfg.min_modes_per_gamma)
        K = cfg.rollouts_per_iter
        K_eff = int(np.ceil(K / 2)) if (t <= cfg.early_until or t >= cfg.cooldown_from) else K
        a = float(np.clip((t - cfg.early_until) /
                          max(cfg.cooldown_from - cfg.early_until, 1), 0, 1))
        frontier_ready = _phased_frontier_ready(history, cfg)
        cfg.phase_uniform_active = bool(getattr(cfg, "phased_curriculum", False) and not frontier_ready)
        mix = ((1.0, 0.0) if cfg.phase_uniform_active else
               tuple(float(s0 * (1 - a) + e0 * a) for s0, e0 in zip(cfg.mix_start, cfg.mix_end)))
        ndf = int(round(cfg.demo_frac * cfg.batch_cap)) if (cfg.demo_frac > 0 and demo is not None) else 0
        fresh_target = cfg.batch_cap - ndf                 # fresh quota for the early-stop gather
        tgt_e = int(round(mix[0] * fresh_target)); tgt_f = fresh_target - tgt_e
        fresh, qbuf, rr, rc, vr, att, gather_audit = _gather_fresh(
            policy, unc, env, cfg, gammas, beta, K_eff, tgt_e, tgt_f, qbuf, covered, device,
            gamma_offset=t - 1)
        roll_reached.extend(rr); roll_coll.extend(rc)
        n_valid = 0 if fresh is None else fresh["U"].shape[0]

        inner = (cfg.early_inner if t <= cfg.early_until else
                 cfg.cooldown_inner if t >= cfg.cooldown_from else cfg.inner_steps)
        it_batch, it_pile = (0, 0, 0), 0                   # THIS iter's actual batch draw (0s if no update)
        used_e_this = used_f_this = None                    # THIS iter's batch-used fresh rows (None if skipped)
        if fresh is not None:
            easy_idx, frontier_idx, scores = label_fresh(policy, unc, fresh, env, cfg, device)
        prime_only = legacy_prime_remaining > 0
        if prime_only:
            legacy_prime_remaining -= 1
        if prime_only or t <= cfg.warmup_gather:           # WARM-UP: query/GP memory only, NO gradient step
            if fresh is not None and pile is not None:     # (GP σ-buffer fills before the first update)
                pile.add(fresh, easy_idx, frontier_idx, scores, t)
            if prime_only:
                log(f"it{t:05d} LEGACY PRIME: qbuf {0 if qbuf is None else qbuf['U'].shape[0]} windows; "
                    "no gradient step", flush=True)
            if t == cfg.warmup_gather and pile is not None:
                log(f"it{t:05d} WARM-UP done: pile {len(pile)} windows "
                    f"({pile.count('easy')}e/{pile.count('frontier')}f, "
                    f"{len(set(pile.rid.tolist()))} rollouts)", flush=True)
        elif fresh is not None:
            if t >= cfg.cooldown_from and not cooled:
                for grp in opt.param_groups:
                    grp["lr"] *= cfg.cooldown_lr_mult
                cooled = True
            upd = update_flow_fresh(policy, opt, fresh, easy_idx, frontier_idx, mix, inner, cfg,
                                    field_params, enc_params, device, demo=demo, teacher=teacher, pile=pile,
                                    trust_anchor=trust_anchor, env=env,
                                    escape_replay=escape_replay)
            if upd is not None:
                last = upd
                it_batch, it_pile = upd["batch"], upd.get("n_pile", 0)
                used_e_this = upd.get("used_easy_idx"); used_f_this = upd.get("used_frontier_idx")
            if pile is not None:                           # add AFTER the update: the pile stays strictly older
                pile.add(fresh, easy_idx, frontier_idx, scores, t)
        elif pile is not None and len(pile) > 0:           # gather starved -> train on the (recent) pile
            upd = update_flow_fresh(policy, opt, None, np.array([], int), np.array([], int), mix, inner, cfg,
                                    field_params, enc_params, device, demo=demo, teacher=teacher, pile=pile,
                                    trust_anchor=trust_anchor, env=env,
                                    escape_replay=escape_replay)
            if upd is not None:
                last = upd
                it_batch, it_pile = upd["batch"], upd.get("n_pile", 0)
        # No demo-only recovery: if valid gathering or either class is starved, this iteration is skipped.
        if pile is not None and cfg.pile_relabel_every and t % cfg.pile_relabel_every == 0:
            pile.relabel(policy, unc, cfg, device)         # labels REFRESH with the current policy's σ

        if cfg.log_comp_every and t % cfg.log_comp_every == 0:   # micro diagnostics (user 2026-07-09 pattern)
            near0_e = w2_e = sig_e = sig_f = float("nan")
            if fresh is not None and scores is not None:
                n_all = fresh["U"].shape[0]
                em = np.zeros(n_all, bool); em[easy_idx] = True
                r0 = np.array([np.linalg.norm(np.asarray(GX2.state_from_low5(l), float)[:2])
                               for l in fresh["low5"].numpy()])
                near0_e = float((r0[em] < 1.0).mean()) if em.any() else float("nan")
                w2_e = float((fresh["widx"][em] < 2).mean()) if em.any() else float("nan")
                sig_e = float(scores["sigma"][em].mean()) if em.any() else float("nan")
                sig_f = float(scores["sigma"][~em].mean()) if (~em).any() else float("nan")
                comp = (f"e{len(easy_idx)}/f{len(frontier_idx)} easy(near0 {near0_e:.2f} w<2 {w2_e:.2f} "
                        f"σ {sig_e:.2f}) frontσ {sig_f:.2f} | batch rids {last.get('rid_n', float('nan')):.1f} "
                        f"dom {last.get('rid_dom', float('nan')):.2f} fstep {last.get('functional_step', 0.0):.4f} "
                        f"anchor {last.get('anchor_drift', 0.0):.4f} rollback {int(last.get('rollback', False))}")
            else:
                comp = "e0/f0 (no fresh)"
            rec = dict(iter=t, beta=beta, quantile=cfg.active_quantile,
                       sigma_plane=(scores.get("sigma_plane") if scores else None),
                       margin_plane=(scores.get("margin_plane") if scores else None),
                       prog_plane=(scores.get("prog_plane") if scores else None),
                       n_easy=len(easy_idx), n_frontier=len(frontier_idx),
                       near0_e=near0_e, w2_e=w2_e, sig_e=sig_e, sig_f=sig_f,
                       rid_n=last.get("rid_n", float("nan")), rid_dom=last.get("rid_dom", float("nan")),
                       vr=vr, att=att, loss=last["loss"], fld=last["field_grad_rms"],
                       enc=last["enc_grad_rms"], functional_step=last.get("functional_step", 0.0),
                       anchor_drift=last.get("anchor_drift", 0.0),
                       rollback=last.get("rollback", False), lr=float(opt.param_groups[0]["lr"]),
                       batch_e=it_batch[0], batch_f=it_batch[1], batch_d=it_batch[2],
                       batch_pile=(it_pile if isinstance(it_pile, int) else 0),
                       batch_pe=(last.get("pile_batch", (0, 0))[0] if it_batch != (0, 0, 0) else 0),
                       batch_pf=(last.get("pile_batch", (0, 0))[1] if it_batch != (0, 0, 0) else 0),
                       batch_gamma_counts=(last.get("batch_gamma_counts", {})
                                           if it_batch != (0, 0, 0) else {}),
                       batch_mode_counts=(last.get("batch_mode_counts", {})
                                          if it_batch != (0, 0, 0) else {}),
                       mix_e=float(mix[0]), mix_f=float(mix[1]),
                       phase_uniform=bool(getattr(cfg, "phase_uniform_active", False)),
                       phase_frontier_ready=bool(frontier_ready),
                       active_min_modes=int(getattr(cfg, "active_min_modes", cfg.min_modes_per_gamma)),
                       mode_hit_gate=bool(getattr(cfg, "mode_hit_gate", False)),
                       demo_req=int(round(cfg.demo_frac * cfg.batch_cap)) if demo is not None else 0,
                       batch_hard=last.get("batch_hard", 0), hard_x0=last.get("hard_x0", 0),
                       batch_guard=last.get("batch_guard", 0), guard_pool=last.get("guard_pool", 0),
                       guard_x0=last.get("guard_x0", 0),
                       strip_pool=last.get("strip_pool", 0),
                       strip_oob_frac=last.get("strip_oob_frac", float("nan")),
                       gather_audit=gather_audit)
            if int(getattr(cfg, "strip_probe_every", 0)) and t % int(cfg.strip_probe_every) == 0:
                sp = _strip_probe(policy, env, cfg, device)
                rec.update({f"strip_probe_{k}": v for k, v in sp.items()})
                comp += (f" | strip win-OOB o{sp['origin']:.2f}/{sp.get('origin_mild', float('nan')):.2f}"
                         f" g{sp['goal']:.2f}/{sp.get('goal_mild', float('nan')):.2f}")
            if fresh is not None:
                nrec = int((fresh.get("rkind") == "recovery_origin").sum() +
                           (fresh.get("rkind") == "recovery_goal").sum()) if "rkind" in fresh else 0
                rec["n_strip_windows"] = int((fresh.get("strip") != "").sum()) if "strip" in fresh else 0
                rec["n_recovery_windows"] = nrec
                ug, cg = np.unique(fresh["gamma"].numpy(), return_counts=True)
                rec["gamma_counts"] = {str(float(g)): int(c) for g, c in zip(ug, cg)}
                rug, ruc = np.unique(fresh.get("rollout_gamma", np.array([])), return_counts=True)
                aug, auc = np.unique(fresh.get("attempted_gamma", np.array([])), return_counts=True)
                rec["gamma_rollout_counts"] = {str(float(g)): int(c) for g, c in zip(rug, ruc)}
                rec["gamma_attempt_counts"] = {str(float(g)): int(c) for g, c in zip(aug, auc)}
                rec["gamma_ready"] = bool(fresh.get("gamma_ready", False))
                rec["classes_ready"] = bool(fresh.get("classes_ready", False))
                rec["gamma_class_ready"] = bool(fresh.get("gamma_class_ready", False))
                rec["mode_ready"] = bool(fresh.get("mode_ready", False))
                rec["target_hit_ready"] = bool(fresh.get("target_hit_ready", False))
                rec["target_hits"] = int(gather_audit.get("target_hits", 0))
                rec["modes_per_gamma"] = fresh.get("modes_per_gamma", {})
            if pile is not None:
                rec.update(pile_e=pile.count("easy"), pile_f=pile.count("frontier"),
                           pile_rollouts=len(set(pile.rid.tolist())) if len(pile) else 0,
                           batch_pile=last.get("n_pile", 0), warmup=t <= cfg.warmup_gather)
                comp += f" | pile {pile.count('easy')}e/{pile.count('frontier')}f b{last.get('n_pile', 0)}"
            pr = ""
            if cfg.probe_escape and t % cfg.probe_escape == 0:
                pe, ph, pp = _escape_probe(policy, env, cfg, device)
                rec.update(esc=pe, hstd=ph, eprog=pp)
                pr = f" | esc {pe:.2f} hstd {ph:.2f}"
            if cfg.probe_cov and t % cfg.probe_cov == 0:
                s50, c50, k50, ids50 = _cov_probe(policy, env, cfg, device)
                rec.update(sr50=s50, cr50=c50, cov50=k50, ids50=ids50)
                pr += f" | SR50 {s50:.2f} CR50 {c50:.2f} cov {k50}"
                if c50 == 0.0 and (s50, k50) > best_probe:
                    best_probe = (s50, k50)
                    if outdir:
                        _save_hp_atomic(policy, os.path.join(outdir, "probe_best.pt"),
                                        extra={"iter": t, "SR50": s50, "CR50": c50,
                                               "coverage50": k50, "ids50": ids50, "recipe": recipe,
                                               "resumable": False})
                if c50 == 0.0 and (k50, s50) > best_probe_cov:
                    best_probe_cov = (k50, s50)
                    if outdir:
                        _save_hp_atomic(policy, os.path.join(outdir, "probe_best_coverage.pt"),
                                        extra={"iter": t, "SR50": s50, "CR50": c50,
                                               "coverage50": k50, "ids50": ids50, "recipe": recipe,
                                               "resumable": False})
            log(f"it{t:05d} COMP β {beta:.2f} q {cfg.active_quantile:.2f} {comp} | vr {vr}/{att}{pr}", flush=True)
            if outdir:
                with open(os.path.join(outdir, "probe.jsonl"), "a") as f:
                    f.write(json.dumps({k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                                        for k, v in rec.items()}) + "\n")

        if outdir and cfg.viz_db_every and t % cfg.viz_db_every == 0 and fresh is not None and n_valid >= 8:
            with _preserve_torch_rng():
                _save_viz_db(fresh, scores, easy_idx, frontier_idx, mix,
                             os.path.join(outdir, "viz_db", f"it{t}.pt"), t,
                             used_easy_idx=used_e_this, used_frontier_idx=used_f_this)
        if t % cfg.measure_every == 0 or local_t == cfg.iters:
            rows, agg = _measure(policy, env, cfg, device)
            osr = float(np.mean(roll_reached)) if roll_reached else 0.0
            ocr = float(np.mean(roll_coll)) if roll_coll else 0.0
            ne, nf = len(easy_idx), len(frontier_idx)
            be, bf, bd = last.get("batch", (0, 0, 0))
            log(f"it{t:05d} SR {agg['SR']:.2f} CR {agg['CR']:.2f} | loss {last['loss']:.3f} "
                f"gRMS(fld {last['field_grad_rms']:.3f} enc {last['enc_grad_rms']:.3f}) | "
                f"β {beta:.2f} q {cfg.active_quantile:.2f} mix {mix[0]:.2f}/{mix[1]:.2f} lbl {ne}e/{nf}f | "
                f"batch {be}e+{bf}f+{bd}d nvalid {n_valid} vr {vr}/{att} | "
                f"on(SR {osr:.2f} CR {ocr:.2f})", flush=True)
            history.append(dict(iter=t, SR=agg["SR"], CR=agg["CR"], gdist=agg["mean_goal_dist"],
                                rows={str(g): rows[g] for g in gammas}, n_pos=n_valid, beta=beta,
                                quantile=cfg.active_quantile,
                                lr=float(opt.param_groups[0]["lr"]),
                                mix=list(mix), n_easy=ne, n_mid=0, n_frontier=nf, loss=last["loss"],
                                field_grad_rms=last["field_grad_rms"], enc_grad_rms=last["enc_grad_rms"],
                                online_SR=osr, online_CR=ocr, n_valid=n_valid, valid_rollouts=vr,
                                rid_n=last.get("rid_n", float("nan")), rid_dom=last.get("rid_dom", float("nan")),
                                active_min_modes=int(getattr(cfg, "active_min_modes", cfg.min_modes_per_gamma)),
                                target_hits=int(gather_audit.get("target_hits", 0)),
                                target_hit_ready=bool(gather_audit.get("target_hit_ready", False)),
                                covered={str(gg): len(covered[gg]) for gg in gammas}))
            if outdir:                                     # live history so sweep_watch can see progress
                with open(os.path.join(outdir, "history.json"), "w") as f:
                    json.dump(history, f)
            if agg["SR"] > best_sr:
                best_sr = agg["SR"]
                if outdir:
                    _save_hp_atomic(policy, os.path.join(outdir, "best.pt"),
                                    extra={"iter": t, "SR": agg["SR"], "CR": agg["CR"],
                                           "resumable": False})
            if agg["CR"] == 0.0 and (agg["SR"], -agg["mean_goal_dist"]) > best_safe_sr:
                best_safe_sr = (agg["SR"], -agg["mean_goal_dist"])
                if outdir:
                    _save_hp_atomic(policy, os.path.join(outdir, "safe_best.pt"),
                                    extra={"iter": t, "SR": agg["SR"], "CR": agg["CR"],
                                           "mean_goal_dist": agg["mean_goal_dist"], "rows": history[-1]["rows"],
                                           "recipe": recipe, "resumable": False})
            collapse_ct = (collapse_ct + 1 if (t >= cfg.collapse_min_iter and
                           agg["SR"] < cfg.collapse_frac * max(sr0, best_sr)) else 0)
            if collapse_ct >= cfg.collapse_patience:
                log(f"it{t:05d} COLLAPSED (SR {agg['SR']:.2f} < {cfg.collapse_frac}·max(SR0 {sr0:.2f}, "
                    f"best {best_sr:.2f})) — terminating early", flush=True)
                break

        # Commit resumable state only after this iteration's measurements and counters are final.
        if outdir and t % cfg.ckpt_every == 0:
            _save_hp_atomic(policy, os.path.join(outdir, f"ckpt_{t}.pt"),
                            extra={"iter": t, "srcr": history[-1], "recipe": recipe,
                                   "train_state": train_state_at(t), "resumable": True})

    if outdir:
        _save_hp_atomic(policy, os.path.join(outdir, "final.pt"),
                        extra={"iter": final_iter, "covered": {str(g): sorted(covered[g]) for g in gammas},
                               "history_tail": history[-1], "recipe": recipe,
                               "train_state": train_state_at(final_iter), "resumable": True})
        with open(os.path.join(outdir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
    return dict(history=history, covered={str(g): sorted(covered[g]) for g in gammas})


def _parse_quantile_schedule(items):
    schedule = []
    for item in items:
        try:
            start, quantile = item.split(":", 1)
            schedule.append((int(start), float(quantile)))
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"bad schedule item {item!r}; expected START:Q") from exc
    schedule.sort()
    if not schedule or schedule[0][0] != 0:
        raise argparse.ArgumentTypeError("quantile schedule must begin at absolute iteration 0")
    if any(not 0.0 < q < 1.0 for _, q in schedule):
        raise argparse.ArgumentTypeError("all quantiles must lie in (0,1)")
    return tuple(schedule)


def _parse_int_schedule(items):
    schedule = []
    for item in items:
        try:
            start, value = item.split(":", 1)
            schedule.append((int(start), int(value)))
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"bad integer schedule item {item!r}; expected START:N") from exc
    schedule.sort()
    if any(start < 0 or value < 0 for start, value in schedule):
        raise argparse.ArgumentTypeError("integer schedule starts and values must be nonnegative")
    if len({start for start, _ in schedule}) != len(schedule):
        raise argparse.ArgumentTypeError("integer schedule starts must be unique")
    return tuple(schedule)


def main():
    import grid_scene as GS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--start-iter", type=int, default=None,
                    help="absolute checkpoint iteration (default: read from checkpoint metadata)")
    ap.add_argument("--freeze", dest="freeze", action="store_true", default=True)
    ap.add_argument("--no-freeze", dest="freeze", action="store_false")
    ap.add_argument("--enc-lr-mult", type=float, default=0.3)
    ap.add_argument("--m-measure", type=int, default=25)
    ap.add_argument("--measure-every", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    # fresh-only knobs
    ap.add_argument("--rollouts-per-iter", type=int, default=10, help="maximum valid rollouts while populating both classes")
    ap.add_argument("--gather-attempt-cap", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32, help="total batch (demo + fresh)")
    ap.add_argument("--gp-buf", type=int, default=384,
                    help="GP novelty buffer size. Default 384 < qbuf-cap 500 => _buffer_feat draws a "
                         "RANDOM 384-of-500 subset EVERY call, so sigma (and the easy/frontier split) "
                         "jitters ~15%% across draws. Set >= qbuf-cap for a deterministic buffer.")
    ap.add_argument("--qbuf-cap", type=int, default=500,
                    help="GP query-buffer FIFO depth (windows retained for sigma). Set == gp-buf for a "
                         "fully deterministic small buffer (e.g. --gp-buf 200 --qbuf-cap 200).")
    ap.add_argument("--valid-prog-floor", type=float, default=0.15, help="reject windows below this net-progress (safe-stationary trap; 0=off)")
    ap.add_argument("--min-rollouts", type=int, default=1, help="gather >= this many valid rollouts (LOCKED=1; 4 was the failed uni_C knob)")
    ap.add_argument("--traj-prog-min", type=float, default=0.0, help="dither gate (LOCKED=0/off; 1.0 was the failed uni_C knob)")
    ap.add_argument("--strat-rid", action="store_true", help="batch draw round-robins across source rollouts")
    ap.add_argument("--probe-escape", type=int, default=0, help="origin-escape probe every N iters (0=off)")
    ap.add_argument("--probe-cov", type=int, default=0, help="M=50 faithful SR/CR/staircase-coverage probe every N iters (0=off)")
    ap.add_argument("--fresh-frac", type=float, default=1.0, help="fresh share of the fresh-part batch; rest from the pile (1.0=fresh-only)")
    ap.add_argument("--warmup-gather", type=int, default=0, help="first N iters gather->pile only, no gradient step")
    ap.add_argument("--pile-cap", type=int, default=3000, help="pile FIFO cap (staleness bound)")
    ap.add_argument("--pile-replace", action="store_true", help="pile draws WITH replacement (ablation; default LRU without-replacement)")
    ap.add_argument("--pile-relabel-every", type=int, default=10, help="recompute pile σ-labels every N iters (0=never)")
    ap.add_argument("--log-comp-every", type=int, default=0, help="composition/rid log line every N iters (0=off)")
    ap.add_argument("--quantile-schedule", nargs="+", default=["0:0.50", "200:0.60", "400:0.70"],
                    metavar="START:Q", help="piecewise constant AND-plane quantile by absolute iteration")
    ap.add_argument("--mix-start", type=float, nargs=2, default=None, help="easy/frontier initial mix")
    ap.add_argument("--mix-end", type=float, nargs=2, default=None, help="easy/frontier final mix")
    ap.add_argument("--beta", type=float, choices=[0.2, 0.3, 0.4, 0.5, 0.7, 1.0], default=0.3,
                    help="tilt softmax temperature w=exp((sig-max)/beta): higher = flatter = exploit")
    ap.add_argument("--resume-allow-recipe-drift", action="store_true",
                    help="on stateful resume, tolerate a changed beta/mix/quantile signature (the swept "
                         "knobs do not alter optimizer topology); everything structural stays strict")
    ap.add_argument("--early-until", type=int, default=100)
    ap.add_argument("--cooldown-from", type=int, default=400)
    ap.add_argument("--inner-steps", type=int, default=None, help="mid-phase inner steps (default 4)")
    ap.add_argument("--early-inner", type=int, default=None, help="early/warmup-phase inner steps (default 2)")
    ap.add_argument("--cooldown-inner", type=int, default=None, help="cooldown-phase inner steps (default 2)")
    ap.add_argument("--demo-frac", type=float, default=0.0)
    ap.add_argument("--lwf-eta", type=float, default=0.0)
    ap.add_argument("--teacher-ckpt", default=None,
                    help="fixed original LwF anchor; restored train-state teacher takes precedence")
    ap.add_argument("--easy-strict", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--nfe-explore", type=int, default=8, help="exploration sampler NFE (match evaluation=8)")
    ap.add_argument("--T", type=int, default=250, help="rollout horizon / step budget (raise to give strict "
                    "low-γ policies more time to reach before timeout)")
    ap.add_argument("--field-grad-clip", type=float, default=1.0)
    ap.add_argument("--max-functional-step", type=float, default=0.025,
                    help="rollback one update above this relative fixed-panel vector-field displacement")
    ap.add_argument("--max-anchor-drift", type=float, default=0.016,
                    help="rollback above this cumulative OOD-origin drift from the fixed lineage teacher")
    ap.add_argument("--targeted-frac", type=float, default=0.5,
                    help="fraction of attempts with one fixed uncovered-neighbor staircase target")
    ap.add_argument("--n-target", type=int, default=40)
    ap.add_argument("--align-temp", type=float, default=0.45)
    ap.add_argument("--target-perp-brake", action="store_true",
                    help="target proposals brake the non-requested grid-crossing axis")
    ap.add_argument("--min-modes-per-gamma", type=int, default=2)
    ap.add_argument("--min-modes-schedule", nargs="*", default=[], metavar="START:N",
                    help="absolute certified-mode quota schedule, e.g. 160:2 180:4 200:8 220:12 240:14")
    ap.add_argument("--mode-hit-gate", action="store_true",
                    help="require exact hits on targeted staircase proposals before a coverage update")
    ap.add_argument("--min-target-hits", type=int, default=1)
    ap.add_argument("--recovery-frac", type=float, default=0.0,
                    help="fraction of gather attempts starting ON a failure strip (hard-tail arm; 0=off)")
    ap.add_argument("--recovery-origin-band", type=float, nargs=8, default=None,
                    metavar="V", help="x0 x1 y0 y1 vx0 vx1 vy0 vy1")
    ap.add_argument("--recovery-goal-band", type=float, nargs=8, default=None,
                    metavar="V", help="x0 x1 y0 y1 vx0 vx1 vy0 vy1")
    ap.add_argument("--hard-quota", type=int, default=0,
                    help="fresh-batch slots reserved for strip-context certified windows (0=off)")
    ap.add_argument("--guard-quota", type=int, default=0,
                    help="fresh slots reserved for exact-certified interior boundary approaches")
    ap.add_argument("--guard-x0", choices=["", "inbounds", "near-hard"], default="",
                    help="pair guard targets with latents whose current map remains in bounds")
    ap.add_argument("--hard-x0", choices=["", "oob"], default="",
                    help="'oob' pairs sub-quota targets with harvested OOB-mapping base latents")
    ap.add_argument("--hard-x0-cand", type=int, default=32)
    ap.add_argument("--strip-probe-every", type=int, default=1,
                    help="per-iter absorber probe (win-OOB at 2 fixed failing contexts; 0=off)")
    ap.add_argument("--ablate-curriculum", action="store_true")
    ap.add_argument("--ablate-socp", action="store_true")
    ap.add_argument("--ablate-progress", action="store_true")
    ap.add_argument("--phased-curriculum", action="store_true",
                    help="uniform certified training until sustained SR competence, then enable frontier curriculum")
    ap.add_argument("--phase-sr-threshold", type=float, default=0.85)
    ap.add_argument("--phase-sr-patience", type=int, default=2,
                    help="consecutive measurement gates at/above threshold required before frontier activation")
    ap.add_argument("--wall-plugs", type=int, choices=[0, 2, 4, 8], default=0,
                    help="extend the obstacle perimeter into the corner openings (scene ablation)")
    ap.add_argument("--start-eps", type=float, default=0.0,
                    help="start the robot at (eps,eps) instead of the origin — on the walled scene the "
                         "origin corner is tight (0.083 m); a small +eps puts it clearly in free space")
    ap.add_argument("--reach", type=float, default=0.1,
                    help="goal-reach radius for 'reached' (gather + measure). A decent positive value "
                         "(e.g. 0.2) counts a near-goal trajectory as success, esp. on the walled scene "
                         "where the goal sits at the plug boundary. Does NOT change valid2 (uses GM.REACH).")
    ap.add_argument("--gammas", nargs="+", type=float, default=None,
                    help="restrict the gather/measure safety levels. Default = all 7. Prefer "
                         "--emergent-gamma over restricting this (uniform is cleaner for the paper).")
    ap.add_argument("--emergent-gamma", action="store_true",
                    help="EMERGENT gamma-curriculum: keep all 7 gammas but don't BLOCK the update on gammas "
                         "with zero certified windows this iter (from the pretrained, gamma 0.1/0.2 are 0%% "
                         "SOCP-valid). Gathers every certified window uniformly; low gammas join as the "
                         "frontier lifts interior clearance. The curriculum emerges from the certificate.")
    ap.add_argument("--demo-prefix", default="dr05_",
                    help="dataset prefix for the demo anchor windows (dataset/<prefix>windows_g*.pt). "
                         "w8d_ = WALLED-scene expert demos (2026-07-14): the anchor then teaches the "
                         "walled-expert clearance/goal-stop behavior in-scene.")
    ap.add_argument("--window-level", action="store_true",
                    help="harvest per-WINDOW valid2 samples from every rollout (not only whole-valid "
                         "trajectories). Makes valid samples plentiful -> no low-gamma starvation, no "
                         "emergent-gamma/recovery needed. THE faithful gather (user 2026-07-14).")
    ap.add_argument("--goal-xy", type=float, nargs=2, default=None,
                    help="move the goal to (gx,gy) (default (5,5)). Cleared goal (4.7,4.7) has 0.3 m "
                         "clearance; propagates to env.goal, the relgoal conditioning, valid2 progress, "
                         "and the reach check.")
    ap.add_argument("--escape-replay", default=None,
                    help="immutable certified replay file (endpoint-distillation rows; recipe change)")
    ap.add_argument("--escape-quota", type=int, default=0)
    ap.add_argument("--escape-eta", type=float, default=1.0)
    ap.add_argument("--legacy-prime-iters", type=int, default=1,
                    help="query-memory-only iterations when input lacks a full train_state")
    ap.add_argument("--drop-train-state", action="store_true",
                    help="explicit model-only branch for a changed recipe; forces legacy query prime")
    ap.add_argument("--viz-db-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100)
    args = ap.parse_args()
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Load metadata on CPU so qbuf/pile/RNG tensors in a full checkpoint stay concatenable with CPU gathers.
    pol, ck = HP.load_hp(args.ckpt, device="cpu")
    pol = pol.to(dev)
    env = GS.make_grid()
    env = _apply_wall_plugs(env, args.wall_plugs)
    if args.start_eps > 0.0:
        env.x0 = torch.tensor([args.start_eps, args.start_eps, 0.0, 0.0], dtype=env.x0.dtype)
    if args.goal_xy is not None:                         # move the goal (cleared (4.7,4.7)); propagate to
        env.goal = torch.tensor([float(args.goal_xy[0]), float(args.goal_xy[1])], dtype=env.goal.dtype)
        GM2.GOAL_XY = np.array([float(args.goal_xy[0]), float(args.goal_xy[1])], dtype=float)  # valid2 progress
    ck_iter = ck.get("iter", ck.get("history_tail", {}).get("iter", 0))
    start_iter = int(ck_iter if args.start_iter is None else args.start_iter)
    cfg = CurConfig(iters=args.iters, M_measure=args.m_measure, measure_every=args.measure_every,
                    start_iter=start_iter, rollouts_per_iter=args.rollouts_per_iter,
                    gather_attempt_cap=args.gather_attempt_cap,
                    valid_prog_floor=args.valid_prog_floor, min_rollouts=args.min_rollouts,
                    traj_prog_min=args.traj_prog_min, batch_cap=args.batch,
                    quantile_schedule=_parse_quantile_schedule(args.quantile_schedule), beta=args.beta,
                    early_until=args.early_until, cooldown_from=args.cooldown_from,
                    ckpt_every=args.ckpt_every)
    if args.mix_start:
        cfg.mix_start = tuple(args.mix_start)
    if args.mix_end:
        cfg.mix_end = tuple(args.mix_end)
    if args.inner_steps is not None:
        cfg.inner_steps = args.inner_steps
    if args.early_inner is not None:
        cfg.early_inner = args.early_inner
    if args.cooldown_inner is not None:
        cfg.cooldown_inner = args.cooldown_inner
    cfg.demo_frac = args.demo_frac
    cfg.lwf_eta = args.lwf_eta
    cfg.nfe_explore = args.nfe_explore
    cfg.field_grad_clip = args.field_grad_clip
    cfg.max_functional_step = args.max_functional_step
    cfg.max_anchor_drift = args.max_anchor_drift
    cfg.targeted_frac = args.targeted_frac
    cfg.n_target = args.n_target
    cfg.align_temp = args.align_temp
    cfg.target_perp_brake = args.target_perp_brake
    cfg.gp_buf = args.gp_buf
    cfg.qbuf_cap = args.qbuf_cap
    cfg.reach = args.reach
    cfg.T = int(args.T)
    if args.gammas:
        cfg.gammas = tuple(float(g) for g in args.gammas)
    cfg.emergent_gamma = bool(args.emergent_gamma)
    cfg.demo_prefix = args.demo_prefix
    cfg.window_level = bool(args.window_level)
    cfg.resume_allow_recipe_drift = args.resume_allow_recipe_drift
    cfg.min_modes_per_gamma = args.min_modes_per_gamma
    cfg.min_modes_schedule = _parse_int_schedule(args.min_modes_schedule)
    cfg.mode_hit_gate = args.mode_hit_gate
    cfg.min_target_hits = args.min_target_hits
    if cfg.min_target_hits < 1:
        raise ValueError("--min-target-hits must be >= 1")
    cfg.recovery_frac = args.recovery_frac
    if args.recovery_origin_band is not None:
        cfg.recovery_origin_band = tuple(args.recovery_origin_band)
    if args.recovery_goal_band is not None:
        cfg.recovery_goal_band = tuple(args.recovery_goal_band)
    cfg.hard_quota = args.hard_quota
    cfg.guard_quota = args.guard_quota
    cfg.guard_x0 = args.guard_x0
    cfg.hard_x0 = args.hard_x0
    cfg.hard_x0_cand = args.hard_x0_cand
    cfg.strip_probe_every = args.strip_probe_every
    cfg.escape_quota = args.escape_quota
    cfg.escape_eta = args.escape_eta
    cfg.wall_plugs = args.wall_plugs
    cfg.ablate_curriculum = args.ablate_curriculum
    cfg.ablate_socp = args.ablate_socp
    cfg.ablate_progress = args.ablate_progress
    cfg.phased_curriculum = args.phased_curriculum
    cfg.phase_sr_threshold = args.phase_sr_threshold
    cfg.phase_sr_patience = args.phase_sr_patience
    if cfg.phased_curriculum and cfg.ablate_curriculum:
        raise ValueError("--phased-curriculum and --ablate-curriculum are mutually exclusive")
    if not 0.0 <= cfg.phase_sr_threshold <= 1.0:
        raise ValueError("--phase-sr-threshold must lie in [0,1]")
    if cfg.phase_sr_patience < 1:
        raise ValueError("--phase-sr-patience must be >= 1")
    if args.ablate_curriculum:
        cfg.mix_start = (1.0, 0.0); cfg.mix_end = (1.0, 0.0)
    escape_replay = (torch.load(args.escape_replay, map_location="cpu", weights_only=False)
                     if args.escape_replay else None)
    if not 0.0 <= cfg.recovery_frac <= 0.5:
        raise ValueError("--recovery-frac must lie in [0,0.5] (strip starts never dominate the gather)")
    if not 0.0 <= cfg.targeted_frac <= 1.0:
        raise ValueError("--targeted-frac must lie in [0,1]")
    cfg.legacy_prime_iters = args.legacy_prime_iters
    cfg.easy_strict = args.easy_strict
    if args.lr is not None:
        cfg.lr = args.lr
    cfg.viz_db_every = args.viz_db_every
    cfg.strat_rid = args.strat_rid
    cfg.probe_escape = args.probe_escape
    cfg.probe_cov = args.probe_cov
    cfg.log_comp_every = args.log_comp_every
    cfg.fresh_frac = args.fresh_frac
    cfg.warmup_gather = args.warmup_gather
    cfg.pile_cap = args.pile_cap
    cfg.pile_replace = args.pile_replace
    cfg.pile_relabel_every = args.pile_relabel_every
    print(f"[main] ckpt {os.path.basename(args.ckpt)} repr {ck['config'].get('repr_dim')} "
          f"freeze={args.freeze} enc_lr_mult={args.enc_lr_mult} abs={start_iter}+{args.iters} tag={args.tag}", flush=True)
    run_expand_cur(pol, env, cfg, device=dev, outdir=args.outdir, log=print,
                   freeze_enc=args.freeze, enc_lr_mult=args.enc_lr_mult, tag=args.tag,
                   resume_state=(None if args.drop_train_state else ck.get("train_state")),
                   teacher_ckpt=args.teacher_ckpt,
                   train_seed=args.seed, escape_replay=escape_replay)


if __name__ == "__main__":
    main()
