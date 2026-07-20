"""GridGRUFlowPolicy — lighter γ-conditioned windowed FM policy for the 5x5 grid.

Changes vs the 07-01 GridLowFlowPolicy (per user):
  (1) low-dim = relgoal-vec(2) + vel(2) + GRU embed(d=16) + γ(1) = 21  (GRU over past executed controls
      replaces [prev_action, prev_valid]; captures long temporal history compactly);
  (2) grid encoder = CNN (was MLP): Conv 3->8->16, AdaptiveAvgPool (4,3), flatten 192 -> Linear 64;
  (3) context = [E_l(48); E_g(64)] = 112;
  (4) velocity trunk shallower (depth 2, was 3): input 20+112+32=164 -> 256 -> 256 -> head 20; phi_s = trunk(256).
Inherits cfm_loss / sample / phi_s / features from the base FlowPolicy (overnight_run_today/src/flow_policy.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import _paths  # noqa: F401
from flow_policy import FlowPolicy
import grid_feats as GF


class GridGRUFlowPolicy(FlowPolicy):
    def __init__(self, H_pred=GF.H_PRED, grid_shape=(3, GF.N_THETA, GF.N_R), K_hist=GF.K_HIST,
                 gru_dim=16, low_token=48, grid_token=64, width=256, depth=2, u_max=GF.U_MAX):
        ctx_dim = low_token + grid_token                       # 48 + 64 = 112
        super().__init__(T=H_pred, ctx_dim=ctx_dim, width=width, depth=depth, u_max=u_max)
        self.H_pred = H_pred
        self.grid_shape = tuple(grid_shape)
        self.K_hist = K_hist
        self.gru_dim = gru_dim
        # (1) GRU over past executed controls (2-D per step) -> gru_dim
        self.gru = nn.GRU(input_size=2, hidden_size=gru_dim, num_layers=1, batch_first=True)
        # low-dim encoder E_l : 21 -> 64 -> 48
        self.enc_low = nn.Sequential(nn.Linear(4 + gru_dim + 1, 64), nn.SiLU(), nn.Linear(64, low_token), nn.SiLU())
        # (2) grid CNN encoder E_g : [3,16,12] -> 64
        self.enc_grid = nn.Sequential(
            nn.Conv2d(grid_shape[0], 8, 3, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(),
            nn.Linear(16 * 4 * 3, grid_token), nn.SiLU())
        # aux safety decoder : grid token(64) -> reconstruct flattened grid (polytope->context signal)
        self.g_in = int(np.prod(grid_shape))
        self.safety_decoder = nn.Sequential(nn.Linear(grid_token, 256), nn.SiLU(), nn.Linear(256, self.g_in))

    # ---- context ---------------------------------------------------------
    def _grid_token(self, grid):
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        return self.enc_grid(grid.float())

    def ctx_from(self, grid, low5, hist):
        """grid [B,3,16,12], low5 [B,5], hist [B,K,2]  ->  ctx [B,112] (GRU runs here, grads flow)."""
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if low5.dim() == 1:
            low5 = low5.unsqueeze(0)
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)
        _, h_n = self.gru(hist.float())                        # h_n [1,B,gru_dim]
        h = h_n[-1]                                            # [B,gru_dim]
        low21 = torch.cat([low5[:, :4], h, low5[:, 4:5]], dim=1)   # relgoal2+vel2+GRU+γ
        e_l = self.enc_low(low21)                             # [B,48]
        e_g = self.enc_grid(grid.float())                    # [B,64]
        return torch.cat([e_l, e_g], dim=1)                  # [B,112]

    def aux_safety_loss(self, grid):
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        B = grid.shape[0]
        return F.mse_loss(self.safety_decoder(self._grid_token(grid)), grid.float().reshape(B, -1))

    # ---- convenience for rollout / expansion -----------------------------
    @torch.no_grad()
    def sample_window(self, grid, low5, hist, n=1, temp=1.0, nfe=12, churn=0.0):
        """Sample n candidate windows at ONE conditioning state -> U [n,H_pred,2] (raw world controls)."""
        ctx = self.ctx_from(grid, low5, hist)                # [1,112]
        if ctx.shape[0] == 1:
            ctx = ctx[0]                                     # -> [112] so base sample broadcasts to n
        return self.sample(n, ctx, nfe=nfe, temp=temp, churn=churn)

    def phi_s_at(self, U, grid, low5, hist, s=0.9):
        """Noised-flow feature φ_s for windows U at ONE state -> [n,width] (for GP uncertainty, Eq.10)."""
        ctx = self.ctx_from(grid, low5, hist)                # [1,112]
        if ctx.shape[0] == 1:
            ctx = ctx[0]                                     # -> [112] so base phi_s broadcasts to n
        return self.phi_s(U, ctx, s=s)

    def config(self):
        return dict(H_pred=self.H_pred, grid_shape=self.grid_shape, K_hist=self.K_hist,
                    gru_dim=self.gru_dim, width=self.width, depth=len([m for m in self.trunk if isinstance(m, nn.Linear)]),
                    u_max=self.u_max)


def build_policy(depth=2, gru_dim=16, K_hist=GF.K_HIST, width=256, u_max=GF.U_MAX, device="cpu"):
    return GridGRUFlowPolicy(depth=depth, gru_dim=gru_dim, K_hist=K_hist, width=width, u_max=u_max).to(device)


def save_policy(policy, path, extra=None):
    d = {"state_dict": policy.state_dict(), "config": policy.config()}
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    c = ck["config"]
    pol = GridGRUFlowPolicy(H_pred=c["H_pred"], grid_shape=tuple(c["grid_shape"]), K_hist=c["K_hist"],
                            gru_dim=c["gru_dim"], width=c["width"], depth=c["depth"], u_max=c["u_max"])
    pol.load_state_dict(ck["state_dict"]); pol.to(device).eval()
    return pol, ck


if __name__ == "__main__":
    torch.manual_seed(0)
    pol = build_policy()
    B = 8
    grid = torch.rand(B, 3, 16, 12)
    low5 = torch.randn(B, 5)
    hist = torch.randn(B, GF.K_HIST, 2)
    U = torch.randn(B, GF.H_PRED, 2).clamp(-1, 1)
    ctx = pol.ctx_from(grid, low5, hist)
    print("ctx", tuple(ctx.shape), "(expect (8,112))")
    phi = pol.phi_s(U, ctx, s=0.9)
    print("phi_s", tuple(phi.shape), "(expect (8,256))")
    loss = pol.cfm_loss(U, ctx) + 0.3 * pol.aux_safety_loss(grid)
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in pol.gru.parameters() if p.grad is not None)
    print("cfm+aux loss", round(float(loss), 4), "| GRU grad-norm", round(gnorm, 5),
          "(must be > 0 => GRU trains end-to-end)")
    n_par = sum(p.numel() for p in pol.parameters())
    print("params", n_par, "| sample_window", tuple(pol.sample_window(grid[0], low5[0], hist[0], n=4).shape))
