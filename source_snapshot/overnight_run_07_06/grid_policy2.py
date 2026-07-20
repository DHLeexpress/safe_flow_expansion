"""GridGRUFlowPolicy2 — v2 model (2026-07-03): NO aux safety decoder (saves 164,672 params = 55% of v1),
trunk width parametrized, and ABLATABLE conditioning (2026-07-03 reduced-model study).

Three flags carve down the context (default = full, backward-compatible):
  use_gru    : GRU(16) over past executed controls in the low vector       (drop -> no history)
  encode_low : pass the low vector through E_l (21|5 -> 64 -> 48)           (drop -> raw low straight to trunk)
  use_grid   : grid CNN E_g (encoded safety, 64)                            (drop -> policy blind to obstacles;
                                                                             only the verifier enforces safety)
  low vector = relgoal(2) + vel(2) + [GRU(16)] + γ(1)  ->  21 (with GRU) or 5 (without).
  REDUCED model (user 2026-07-03): use_gru=F, encode_low=F, use_grid=F  ->  ctx = raw 5; trunk in = 20+5+32 = 57;
  ZERO learned context encoder — only the velocity field learns; the entangled learned representation is gone.
  FULL model: all True -> ctx = E_l(48)+E_g(64) = 112 (the pretrained2_w* checkpoints). φ_s = trunk (dim=width).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import _paths  # noqa: F401
from flow_policy import FlowPolicy
import grid_feats as GF


class GridGRUFlowPolicy2(FlowPolicy):
    def __init__(self, H_pred=GF.H_PRED, grid_shape=(3, GF.N_THETA, GF.N_R), K_hist=GF.K_HIST,
                 gru_dim=16, low_token=48, grid_token=64, width=256, depth=2, u_max=GF.U_MAX,
                 use_gru=True, encode_low=True, use_grid=True, raw_hist=False, raw_hist_k=10, dropout=0.0,
                 enc_hist=False):
        gd = gru_dim if use_gru else 0
        raw_low = 4 + gd + 1                                    # relgoal2 + vel2 + [GRU] + γ
        rawh = raw_hist_k * 2 if raw_hist else 0
        enc_hist_eff = bool(enc_hist and raw_hist and encode_low)   # route the raw history THROUGH E_l
        enc_in = raw_low + (rawh if enc_hist_eff else 0)      # E_l input dim (25 for the 2nd reduced model)
        low_out = low_token if encode_low else raw_low
        rawh_out = rawh if (raw_hist and not enc_hist_eff) else 0    # appended raw ONLY when not encoded
        grid_out = grid_token if use_grid else 0
        ctx_dim = low_out + grid_out + rawh_out
        super().__init__(T=H_pred, ctx_dim=ctx_dim, width=width, depth=depth, u_max=u_max)
        self.H_pred = H_pred
        self.grid_shape = tuple(grid_shape)
        self.K_hist = K_hist
        self.gru_dim = gru_dim
        self.use_gru = use_gru
        self.encode_low = encode_low
        self.use_grid = use_grid
        self.raw_hist = raw_hist
        self.raw_hist_k = raw_hist_k
        self.dropout = dropout
        self.enc_hist = enc_hist_eff
        self.enc_in = enc_in
        self.raw_low = raw_low
        if dropout > 0:                                        # rebuild trunk WITH dropout (regularizer tweak);
            in_dim = self.d + ctx_dim + self.t_dim            # SiLU kept (subsumes ReLU, better for flow trunks)
            layers = [nn.Linear(in_dim, width), nn.SiLU(), nn.Dropout(dropout)]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout)]
            self.trunk = nn.Sequential(*layers)
        if use_gru:
            self.gru = nn.GRU(input_size=2, hidden_size=gru_dim, num_layers=1, batch_first=True)
        if encode_low:
            self.enc_low = nn.Sequential(nn.Linear(enc_in, 64), nn.SiLU(), nn.Linear(64, low_token), nn.SiLU())
        if use_grid:
            self.enc_grid = nn.Sequential(
                nn.Conv2d(grid_shape[0], 8, 3, padding=1), nn.SiLU(),
                nn.Conv2d(8, 16, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(),
                nn.Linear(16 * 4 * 3, grid_token), nn.SiLU())
        # NO safety_decoder in v2 (loss = cfm only + signed negative term in expansion).

    # ---- context ---------------------------------------------------------
    def _low_raw(self, low5, hist):
        """[relgoal2, vel2, (GRU16), γ1] -> [B, raw_low]. GRU runs here when enabled (grads flow)."""
        if low5.dim() == 1:
            low5 = low5.unsqueeze(0)
        if self.use_gru:
            if hist.dim() == 2:
                hist = hist.unsqueeze(0)
            _, h_n = self.gru(hist.float())
            return torch.cat([low5[:, :4], h_n[-1], low5[:, 4:5]], dim=1)
        return torch.cat([low5[:, :4], low5[:, 4:5]], dim=1)

    def _raw_hist(self, hist):
        h = hist.unsqueeze(0) if hist.dim() == 2 else hist
        return h[:, -self.raw_hist_k:, :].reshape(h.shape[0], -1).float()

    def ctx_from(self, grid, low5, hist):
        """grid [B,3,16,12], low5 [B,5], hist [B,K,2] -> ctx [B,ctx_dim]."""
        low_raw = self._low_raw(low5, hist)
        rh = self._raw_hist(hist) if self.raw_hist else None
        if self.encode_low:
            enc_input = torch.cat([low_raw, rh], dim=1) if self.enc_hist else low_raw
            low_part = self.enc_low(enc_input)               # 2nd reduced model: E_l(25)->48
        else:
            low_part = low_raw
        parts = [low_part]
        if self.raw_hist and not self.enc_hist:              # raw last-k actions appended UNENCODED
            parts.append(rh)
        if self.use_grid:
            if grid.dim() == 3:
                grid = grid.unsqueeze(0)
            parts.append(self.enc_grid(grid.float()))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def encoder_tokens(self, grid, low5, hist):
        """Diagnostics: dict of the learned-encoder outputs that EXIST (collapse check). Empty for reduced."""
        out = {}
        lr = self._low_raw(low5, hist)
        if self.use_gru:
            out["gru"] = lr[:, 4:4 + self.gru_dim]
        if self.encode_low:
            enc_input = torch.cat([lr, self._raw_hist(hist)], dim=1) if self.enc_hist else lr
            out["e_l"] = self.enc_low(enc_input)
        if self.use_grid:
            g = grid.unsqueeze(0) if grid.dim() == 3 else grid
            out["e_g"] = self.enc_grid(g.float())
        return out

    # ---- convenience for rollout / expansion (same interface) ------------
    @torch.no_grad()
    def sample_window(self, grid, low5, hist, n=1, temp=1.0, nfe=12, churn=0.0):
        ctx = self.ctx_from(grid, low5, hist)
        if ctx.shape[0] == 1:
            ctx = ctx[0]
        return self.sample(n, ctx, nfe=nfe, temp=temp, churn=churn)

    def phi_s_at(self, U, grid, low5, hist, s=0.9):
        ctx = self.ctx_from(grid, low5, hist)
        if ctx.shape[0] == 1:
            ctx = ctx[0]
        return self.phi_s(U, ctx, s=s)

    def module_groups(self):
        """Named module groups that EXIST, for per-module gradient-flow diagnostics."""
        g = dict(trunk=self.trunk, head=self.head)
        if self.use_grid:
            g["E_g"] = self.enc_grid
        if self.encode_low:
            g["E_l"] = self.enc_low
        if self.use_gru:
            g["GRU"] = self.gru
        return g

    def encoder_modules(self):
        """Learned context-encoder modules (for the per-group optimizer). Empty for the reduced model."""
        m = []
        if self.use_grid:
            m += list(self.enc_grid.parameters())
        if self.encode_low:
            m += list(self.enc_low.parameters())
        if self.use_gru:
            m += list(self.gru.parameters())
        return m

    def config(self):
        return dict(arch="v2", H_pred=self.H_pred, grid_shape=self.grid_shape, K_hist=self.K_hist,
                    gru_dim=self.gru_dim, width=self.width,
                    depth=len([m for m in self.trunk if isinstance(m, nn.Linear)]), u_max=self.u_max,
                    use_gru=self.use_gru, encode_low=self.encode_low, use_grid=self.use_grid,
                    raw_hist=self.raw_hist, raw_hist_k=self.raw_hist_k, dropout=self.dropout,
                    enc_hist=self.enc_hist)


def build_policy2(width=256, depth=2, gru_dim=16, K_hist=GF.K_HIST, u_max=GF.U_MAX,
                  use_gru=True, encode_low=True, use_grid=True, raw_hist=False, raw_hist_k=10, dropout=0.0,
                  enc_hist=False, device="cpu"):
    return GridGRUFlowPolicy2(width=width, depth=depth, gru_dim=gru_dim, K_hist=K_hist, u_max=u_max,
                              use_gru=use_gru, encode_low=encode_low, use_grid=use_grid,
                              raw_hist=raw_hist, raw_hist_k=raw_hist_k, dropout=dropout, enc_hist=enc_hist).to(device)


def save_policy2(policy, path, extra=None):
    d = {"state_dict": policy.state_dict(), "config": policy.config()}
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_policy2(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    pol = GridGRUFlowPolicy2(H_pred=c["H_pred"], grid_shape=tuple(c["grid_shape"]), K_hist=c["K_hist"],
                             gru_dim=c["gru_dim"], width=c["width"], depth=c["depth"], u_max=c["u_max"],
                             use_gru=c.get("use_gru", True), encode_low=c.get("encode_low", True),
                             use_grid=c.get("use_grid", True), raw_hist=c.get("raw_hist", False),
                             raw_hist_k=c.get("raw_hist_k", 10), dropout=c.get("dropout", 0.0),
                             enc_hist=c.get("enc_hist", False))
    pol.load_state_dict(ck["state_dict"]); pol.to(device).eval()
    return pol, ck


def param_report(policy):
    groups = policy.module_groups()
    rep = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
    rep["total"] = sum(p.numel() for p in policy.parameters())
    return rep


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8
    grid = torch.rand(B, 3, 16, 12); low5 = torch.randn(B, 5); hist = torch.randn(B, GF.K_HIST, 2)
    U = torch.randn(B, GF.H_PRED, 2).clamp(-1, 1)
    for name, kw in (("FULL", {}), ("reduced(5-d)", dict(use_gru=False, encode_low=False, use_grid=False)),
                     ("no-grid(21-d)", dict(use_grid=False))):
        pol = build_policy2(width=256, **kw)
        ctx = pol.ctx_from(grid, low5, hist)
        rep = param_report(pol)
        loss = pol.cfm_loss(U, ctx); loss.backward()
        gn = {k: round(float(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None) ** 0.5), 4)
              for k, m in pol.module_groups().items()}
        print(f"{name:14s} ctx={tuple(ctx.shape)} total={rep['total']:,} grads={gn}")
