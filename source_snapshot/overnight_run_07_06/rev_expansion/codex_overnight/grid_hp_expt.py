"""H_P inductive-bias experiment on the 0702 CHESSBOARD (user 2026-07-04 spec, verbatim):
  model ctx = raw low5(5) ⊕ E_g([1,16,12] H_P channel → shallow CNN → AdaptiveAvgPool → 32);
  trunk input = [U(20) + ctx(37) + fourier-t(32)]. NO E_l mixing the raw conditions, NO hist, NO GRU.
Hypothesis: the reduced models' validity JIGGLED because ctx was OOD each iteration and the optimizer remaps the
encoder instead of refining p(U|ctx); the H_P grid channel is the inductive bias that should let coverage(252)
and validity2 SATURATE. Protocol: pretrain on the 0702 demo windows, then `grid_expand2.run_expand2` UNTOUCHED
(validity2 gate + 252 coverage + varσ + probes, SFG2Config defaults, iters=2000 as run_reduced_0703). No control
arm (user: don't compare — watch reliability of coverage/validity2/varσ intermittently)."""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import _paths  # noqa: F401  (0704 bootstrap; not needed when run from 0702)
except ImportError:
    pass
import grid_policy2 as GP2
import grid_expand2 as GX2
import grid_feats as GF
import grid_scene as GS
import wandb_utils as W

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.join(os.path.dirname(HERE), "overnight_run_2026-07-02")
OUT = os.path.join(HERE, "results", "hp_chessboard")
os.makedirs(OUT, exist_ok=True)


class GridHPFlowPolicy(GP2.GridGRUFlowPolicy2):
    """ctx = raw low5(5) ⊕ E_g(H_P[1,16,12]→32). Slices channel 2 (clipped H_P) from the standard 3-ch grid."""
    def __init__(self, width=256, depth=2, u_max=1.0, use_gru=False, repr_dim=None, grid_hw=(16, 12),
                 trunk_hidden=(128, 64), enc_depth=2, raw_condition_dim=5,
                 conditioning_schema="low5", boundary_adapter=False,
                 boundary_adapter_hidden=0, boundary_origin_gate=(1.25, 0.65, 0.50, 0.47),
                 boundary_goal_gate=(3.95, 4.05, 0.55, 0.55),
                 reflection_group_average=False, **kw):
        super().__init__(grid_shape=(1, grid_hw[0], grid_hw[1]), width=width, depth=depth, u_max=u_max,
                         use_gru=use_gru, encode_low=False, use_grid=True, **kw)
        self.repr_dim = repr_dim
        self.grid_hw = tuple(grid_hw)
        self.trunk_hidden = tuple(trunk_hidden)
        self.enc_depth = int(enc_depth)
        self.raw_condition_dim = int(raw_condition_dim)
        if self.raw_condition_dim < 5:
            raise ValueError("raw_condition_dim must include at least low5")
        self.conditioning_schema = str(conditioning_schema)
        if (self.raw_condition_dim, self.conditioning_schema) not in {
            (5, "low5"),
            (7, "low7_closest_boundary"),
            (7, "low7_closest_boundary_tie_mean"),
        }:
            raise ValueError(
                "conditioning dimension and schema must declare low5 or "
                "low7_closest_boundary"
            )
        self.reflection_group_average = bool(reflection_group_average)
        if self.reflection_group_average and (
            self.conditioning_schema != "low7_closest_boundary_tie_mean"
            or use_gru
            or boundary_adapter
        ):
            raise ValueError(
                "reflection group averaging requires tie-mean low7 conditioning "
                "conditioning without a GRU or boundary adapter"
            )
        # 1-ch CNN (enc_depth conv layers) + AdaptiveAvgPool → 32 H_P token; pool scales with grid resolution
        ph, pw = (8, 8) if max(grid_hw) >= 24 else (4, 3)        # 32x32 -> (8,8); 16x12 -> (4,3)
        chs = [1, 8, 16]
        while len(chs) - 1 < self.enc_depth:                    # deeper encoder = more 16-ch conv layers
            chs.append(16)
        conv = []
        for ci in range(self.enc_depth):
            conv += [nn.Conv2d(chs[ci], chs[ci + 1], 3, padding=1), nn.SiLU()]
        self.enc_grid = nn.Sequential(*conv, nn.AdaptiveAvgPool2d((ph, pw)), nn.Flatten(),
                                      nn.Linear(chs[self.enc_depth] * ph * pw, 32), nn.SiLU())
        gd = self.gru_dim if use_gru else 0                      # GRU(16) over past controls → curvature
        self.ctx_dim = self.raw_condition_dim + gd + 32           # raw condition (+ GRU token) + H_P token
        in_dim = self.d + self.ctx_dim + self.t_dim              # 20 + 37 + 32 = 89 (105 with GRU)
        if repr_dim is None:                                     # legacy: plain MLP trunk (→ width), head width→d
            layers = [nn.Linear(in_dim, width), nn.SiLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.SiLU()]
            self.trunk = nn.Sequential(*layers)                  # head (width→20) unchanged
        else:                                                    # repr trunk: in_dim → *trunk_hidden → repr; head repr→d
            dims = [in_dim] + list(self.trunk_hidden) + [repr_dim]
            layers = []
            for a_, b_ in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a_, b_), nn.SiLU()]
            self.trunk = nn.Sequential(*layers)
            self.head = nn.Linear(repr_dim, self.d)              # override inherited Linear(width, d)
        self.boundary_adapter = False
        self.boundary_adapter_hidden = int(boundary_adapter_hidden)
        self.boundary_origin_gate = tuple(float(v) for v in boundary_origin_gate)
        self.boundary_goal_gate = tuple(float(v) for v in boundary_goal_gate)
        if boundary_adapter:
            self.enable_boundary_adapter(self.boundary_adapter_hidden)

    def enable_boundary_adapter(self, hidden=0):
        """Add zero-initialized, compact-support residual heads for evidenced empty strips.

        The original encoder/trunk/head stay unchanged.  Compact support makes the residual
        exactly zero in the task interior; this is a learned training branch, not a safety filter.
        """
        if self.boundary_adapter:
            return
        feat_dim = self.repr_dim if self.repr_dim is not None else self.width
        self.boundary_adapter_hidden = int(hidden)
        if self.boundary_adapter_hidden > 0:
            def make_adapter():
                m = nn.Sequential(nn.Linear(feat_dim, self.boundary_adapter_hidden), nn.SiLU(),
                                  nn.Linear(self.boundary_adapter_hidden, self.d, bias=False))
                nn.init.zeros_(m[-1].weight)
                return m
            self.adapter_origin = make_adapter(); self.adapter_goal = make_adapter()
        else:
            self.adapter_origin = nn.Linear(feat_dim, self.d, bias=False)
            self.adapter_goal = nn.Linear(feat_dim, self.d, bias=False)
            nn.init.zeros_(self.adapter_origin.weight); nn.init.zeros_(self.adapter_goal.weight)
        self.boundary_adapter = True

    def _boundary_gates(self, ctx):
        # ctx[:2] is normalized relative goal; this experiment's fixed scene goal is (5,5).
        x = 5.0 - ctx[:, 0] * GF.R_GOAL
        y = 5.0 - ctx[:, 1] * GF.R_GOAL
        xmax, ymax, xwidth, ywidth = self.boundary_origin_gate
        origin = ((ymax - y) / ywidth).clamp(0.0, 1.0) * ((xmax - x) / xwidth).clamp(0.0, 1.0)
        xmin, ymin, xwidth, ywidth = self.boundary_goal_gate
        goal = ((y - ymin) / ywidth).clamp(0.0, 1.0) * ((x - xmin) / xwidth).clamp(0.0, 1.0)
        return origin, goal

    def forward(self, x, tau, ctx, return_features=False):
        if self.reflection_group_average:
            if ctx.ndim != 2 or ctx.shape[1] != 2 * self.ctx_dim:
                raise ValueError(
                    "group-averaged context must contain original and reflected branches"
                )
            original_context, reflected_context = ctx.split(self.ctx_dim, dim=1)
            reflected_x = x.reshape(len(x), self.T, 2).flip(-1).reshape_as(x)
            combined_features = self.features(
                torch.cat((x, reflected_x), dim=0),
                torch.cat((tau, tau), dim=0),
                torch.cat((original_context, reflected_context), dim=0),
            )
            original_features, reflected_features = combined_features.split(len(x))
            original_velocity = self.head(original_features)
            reflected_velocity = self.head(reflected_features).reshape(
                len(x), self.T, 2
            ).flip(-1).reshape_as(original_velocity)
            velocity = 0.5 * (original_velocity + reflected_velocity)
            features = 0.5 * (original_features + reflected_features)
            return (velocity, features) if return_features else velocity
        h = self.features(x, tau, ctx)
        v = self.head(h)
        if self.boundary_adapter:
            go, gg = self._boundary_gates(ctx)
            v = v + go[:, None] * self.adapter_origin(h) + gg[:, None] * self.adapter_goal(h)
        return (v, h) if return_features else v

    def ctx_from(self, grid, low5, hist):
        hp = grid[..., 2:3, :, :]                                 # H_P channel from the standard [.,3,16,12]
        context = super().ctx_from(hp, low5, hist)
        if not self.reflection_group_average:
            return context
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if low5.dim() == 1:
            low5 = low5.unsqueeze(0)
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)
        n_theta = int(grid.shape[-2])
        if n_theta % 4:
            raise ValueError("x/y reflection requires a polar grid divisible by four")
        indices = torch.remainder(
            n_theta // 4 - torch.arange(n_theta, device=grid.device) - 1,
            n_theta,
        )
        reflected_grid = grid.index_select(-2, indices)
        reflected_low = low5[:, (1, 0, 3, 2, 5, 4, 6)]
        reflected_hp = reflected_grid[..., 2:3, :, :]
        reflected_context = super().ctx_from(
            reflected_hp, reflected_low, hist.flip(-1)
        )
        return torch.cat((context, reflected_context), dim=1)

    @torch.no_grad()
    def phi_s(self, controls, ctx, s=0.9):
        if not self.reflection_group_average:
            return super().phi_s(controls, ctx, s=s)
        batch = controls.shape[0]
        x1 = (controls / self.u_max).reshape(batch, self.d)
        ctx = self._expand_ctx(ctx, batch)
        if len(self.noise_templates) % 2:
            raise RuntimeError("group-averaged feature templates must have even size")
        base_templates = self.noise_templates[: len(self.noise_templates) // 2]
        templates = torch.cat(
            (
                base_templates,
                base_templates.reshape(-1, self.T, 2).flip(-1).reshape(
                    -1, self.d
                ),
            ),
            dim=0,
        )
        features = []
        for template in templates:
            x_s = (1.0 - s) * template[None] + s * x1
            tau = torch.full((batch,), s, device=x1.device, dtype=x1.dtype)
            features.append(self.forward(x_s, tau, ctx, return_features=True)[1])
        return torch.stack(features, dim=0).mean(dim=0)

    def _low_raw(self, low, hist):
        """Retain every declared raw condition; the inherited method drops extras."""
        if low.dim() == 1:
            low = low.unsqueeze(0)
        if low.dim() != 2 or low.shape[1] != self.raw_condition_dim:
            raise ValueError(
                f"raw condition must have shape [B,{self.raw_condition_dim}], got {tuple(low.shape)}"
            )
        low = low.float()
        if not self.use_gru:
            return low
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)
        _, hidden = self.gru(hist.float())
        return torch.cat((low[:, :4], hidden[-1], low[:, 4:]), dim=1)

    def config(self):
        return dict(arch="hp-repr" if self.repr_dim else "hp-reduced-32",
                    schema_version=("w8sg-hp-v4-low7-closest-boundary-tie-mean"
                                    if self.conditioning_schema == "low7_closest_boundary_tie_mean"
                                    else "w8sg-hp-v3-low7-closest-boundary"
                                    if self.conditioning_schema == "low7_closest_boundary"
                                    else "w8sg-hp-v2-low5-only"),
                    raw_start_goal=False,
                    H_pred=self.H_pred,
                    grid_shape=(1, self.grid_hw[0], self.grid_hw[1]), K_hist=self.K_hist,
                    width=self.width, depth=2, u_max=self.u_max, ctx_dim=self.ctx_dim,
                    use_gru=self.use_gru,
                    repr_dim=self.repr_dim, grid_hw=list(self.grid_hw),
                    trunk_hidden=list(self.trunk_hidden), enc_depth=self.enc_depth,
                    raw_condition_dim=self.raw_condition_dim,
                    conditioning_schema=self.conditioning_schema,
                    boundary_adapter=bool(self.boundary_adapter),
                    boundary_adapter_hidden=int(self.boundary_adapter_hidden),
                    boundary_origin_gate=list(self.boundary_origin_gate),
                    boundary_goal_gate=list(self.boundary_goal_gate),
                    reflection_group_average=self.reflection_group_average)


def save_hp(policy, path, extra=None):
    d = {"state_dict": policy.state_dict(), "config": policy.config()}
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_hp(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    pol = GridHPFlowPolicy(width=c["width"], depth=c.get("depth", 2), u_max=c["u_max"],
                           use_gru=c.get("use_gru", False),
                           repr_dim=c.get("repr_dim"), grid_hw=tuple(c.get("grid_hw", (16, 12))),
                           trunk_hidden=tuple(c.get("trunk_hidden", (128, 64))), enc_depth=c.get("enc_depth", 2),
                           raw_condition_dim=c.get("raw_condition_dim", 5),
                           conditioning_schema=c.get("conditioning_schema", "low5"),
                           boundary_adapter=c.get("boundary_adapter", False),
                           boundary_adapter_hidden=c.get("boundary_adapter_hidden", 0),
                           boundary_origin_gate=tuple(c.get("boundary_origin_gate", (1.25, .65, .50, .47))),
                           boundary_goal_gate=tuple(c.get("boundary_goal_gate", (3.95, 4.05, .55, .55))),
                           reflection_group_average=c.get("reflection_group_average", False))
    pol.load_state_dict(ck["state_dict"])
    return pol.to(device).eval(), ck


def pretrain(dev, epochs=120, batch=256, lr=3e-4, warmup=5):
    G, L, Hh, U = [], [], [], []
    for g in ("0.1", "0.5", "1.0"):
        d = torch.load(os.path.join(R2, "dataset", f"windows_g{g}.pt"))
        G.append(d["grid"]); L.append(d["low5"]); Hh.append(d["hist"]); U.append(d["U"])
    G, L, Hh, U = (torch.cat(x) for x in (G, L, Hh, U))
    n = G.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    G, L, Hh, U = G[perm], L[perm], Hh[perm], U[perm]
    nval = max(2048, n // 10)
    tr = TensorDataset(G[nval:], L[nval:], Hh[nval:], U[nval:])
    va = (G[:nval].to(dev), L[:nval].to(dev), Hh[:nval].to(dev), U[:nval].to(dev))
    dl = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True)
    pol = GridHPFlowPolicy().to(dev)
    npar = sum(p.numel() for p in pol.parameters())
    print(f"[pretrain] {n} windows ({n-nval}/{nval}) · HP model {npar/1e3:.1f}k params "
          f"(E_hp {sum(p.numel() for p in pol.enc_grid.parameters())/1e3:.1f}k)", flush=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warmup if ep < warmup else
        0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))))
    best = (float("inf"), None)
    for ep in range(epochs):
        pol.train()
        tot = nb = 0
        for gb, lb, hb, ub in dl:
            loss = pol.cfm_loss(ub.to(dev), pol.ctx_from(gb.to(dev), lb.to(dev), hb.to(dev)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        pol.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            v = float(pol.cfm_loss(va[3], pol.ctx_from(va[0], va[1], va[2])))
        if v < best[0]:
            best = (v, {k: x.detach().cpu().clone() for k, x in pol.state_dict().items()})
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[pretrain] ep {ep:03d} train {tot/nb:.4f} val {v:.4f}", flush=True)
    pol.load_state_dict(best[1])
    save_hp(pol, os.path.join(OUT, "pretrained_hp.pt"), extra={"best_val": best[0]})
    print(f"[pretrain] saved pretrained_hp.pt (val {best[0]:.4f})", flush=True)
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)            # run_reduced_0703 protocol; FULL run = 20000
    ap.add_argument("--skip-pretrain", action="store_true")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--name", default="hp-chessboard")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--enc-lr-mult", type=float, default=None)
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--ell", type=float, default=None)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--measure-every", type=int, default=None)
    ap.add_argument("--n-measure", type=int, default=None)
    ap.add_argument("--s", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--demo-frac", type=float, default=None)
    ap.add_argument("--lwf-eta", type=float, default=None)
    ap.add_argument("--arch-ckpt", default=None, help="start from an hp_arch checkpoint (ResTrunk-aware)")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    if args.arch_ckpt:
        import hp_arch_sweep as ARCH
        pol, _ = ARCH.load_arch(args.arch_ckpt, device=dev)
        print(f"[main] loaded arch ckpt {args.arch_ckpt}", flush=True)
    elif args.skip_pretrain and os.path.exists(os.path.join(OUT, "pretrained_hp.pt")):
        pol, _ = load_hp(os.path.join(OUT, "pretrained_hp.pt"), device=dev)
        print("[main] loaded existing pretrained_hp.pt", flush=True)
    else:
        pol = pretrain(dev)
    env = GS.make_grid()
    cfg = GX2.SFG2Config(iters=args.iters)                        # 0702 defaults; sweep overrides only if given
    for k in ("lr", "alpha", "beta", "enc_lr_mult", "inner_steps", "ell", "temp", "measure_every", "n_measure", "demo_frac", "lwf_eta", "s", "grad_clip", "ckpt_every"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
            print(f"[main] override {k}={v}", flush=True)
    run = W.init_run(args, name=args.name, config={**vars(args), **pol.config()}, group="sfm-0704")
    print(f"[main] EXPANSION: iters={cfg.iters} temp={cfg.temp} ell={cfg.ell} s={cfg.s} beta={cfg.beta} "
          f"N={cfg.N} gp_buf={cfg.gp_buf} (positive-only, validity2 gate, 252 coverage)", flush=True)
    GX2.run_expand2(pol, env, cfg, device=dev, run=run, outdir=args.outdir, log=print)
    W.finish(run)


if __name__ == "__main__":
    main()
