"""Conditional Flow-Matching policy over WHOLE control sequences (spec 3.1: not delta-U).

Cond-OT path:  x_tau = (1-tau) x0 + tau x1,  x0~N(0,I), x1 = U_norm,  target velocity = x1 - x0.
The design is the full sequence U in R^{T x 2}. A shallow MLP velocity field models multi-modal
p(U | c). `phi_s` exposes the penultimate hidden feature at noise level s for the ACTFLOW uncertainty
(Eq. 10).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_time(tau: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """tau [B] in [0,1] -> [B, dim] Fourier features."""
    freqs = torch.arange(1, dim // 2 + 1, device=tau.device).float() * torch.pi
    ang = tau[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class FlowPolicy(nn.Module):
    def __init__(self, T: int, ctx_dim: int, width: int = 256, depth: int = 3,
                 u_max: float = 3.0, t_dim: int = 32, n_noise_feat: int = 4):
        super().__init__()
        self.T = T
        self.d = T * 2
        self.u_max = u_max
        self.width = width
        self.ctx_dim = ctx_dim
        in_dim = self.d + ctx_dim + t_dim
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)        # ends at penultimate features (width)
        self.head = nn.Linear(width, self.d)
        self.t_dim = t_dim
        # fixed noise templates so phi_s is a deterministic, consistent function of U within a call
        g = torch.Generator().manual_seed(20260627)
        self.register_buffer("noise_templates",
                             torch.randn(n_noise_feat, self.d, generator=g))

    # ------------------------------------------------------------------ core
    def features(self, x: torch.Tensor, tau: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        te = fourier_time(tau, self.t_dim)
        inp = torch.cat([x, ctx, te], dim=1)
        return self.trunk(inp)                       # [B, width] penultimate

    def forward(self, x, tau, ctx, return_features=False):
        h = self.features(x, tau, ctx)
        v = self.head(h)
        return (v, h) if return_features else v

    # ------------------------------------------------------------------ helpers
    def _expand_ctx(self, ctx: torch.Tensor, B: int) -> torch.Tensor:
        if ctx.dim() == 1:
            ctx = ctx[None].expand(B, -1)
        return ctx

    def cfm_loss(self, U_controls: torch.Tensor, ctx: torch.Tensor,
                 weights: torch.Tensor | None = None) -> torch.Tensor:
        B = U_controls.shape[0]
        x1 = (U_controls / self.u_max).reshape(B, self.d)
        x0 = torch.randn_like(x1)
        tau = torch.rand(B, device=x1.device).clamp(1e-4, 1.0)
        x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
        target = x1 - x0
        pred = self.forward(x_tau, tau, self._expand_ctx(ctx, B))
        per = ((pred - target) ** 2).mean(dim=1)
        if weights is not None:
            per = per * weights
        return per.mean()

    @torch.no_grad()
    def sample(self, n: int, ctx: torch.Tensor, nfe: int = 12,
               temp: float = 1.0, churn: float = 0.0,
               initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        """temp scales the initial-noise spread; churn injects per-step noise.
        temp>1 / churn>0 fatten the proposal's tails so active exploration can reach
        disconnected valid modes (e.g. the opposite homotopy leaf). Use defaults for eval."""
        device = self.head.weight.device
        ctx = self._expand_ctx(ctx, n)
        if initial_noise is None:
            initial_noise = torch.randn(n, self.d, device=device)
        elif tuple(initial_noise.shape) != (n, self.d):
            raise ValueError(
                f"initial_noise shape {tuple(initial_noise.shape)} != {(n, self.d)}"
            )
        else:
            initial_noise = initial_noise.to(device=device, dtype=self.head.weight.dtype)
        x = temp * initial_noise
        for i in range(nfe):
            tau = torch.full((n,), i / nfe, device=device)
            x = x + (1.0 / nfe) * self.forward(x, tau, ctx)
            if churn > 0 and i < nfe - 1:
                x = x + churn * (1.0 / nfe) ** 0.5 * torch.randn_like(x)
        U = (x.reshape(n, self.T, 2) * self.u_max).clamp(-self.u_max, self.u_max)
        return U

    @torch.no_grad()
    def phi_s(self, U_controls: torch.Tensor, ctx: torch.Tensor, s: float = 0.9) -> torch.Tensor:
        """Noised-flow representation at level s, averaged over fixed noise templates -> [B, width]."""
        B = U_controls.shape[0]
        x1 = (U_controls / self.u_max).reshape(B, self.d)
        ctx = self._expand_ctx(ctx, B)
        feats = []
        for k in range(self.noise_templates.shape[0]):
            x0 = self.noise_templates[k][None].expand(B, -1)
            x_s = (1 - s) * x0 + s * x1
            tau = torch.full((B,), s, device=x1.device)
            feats.append(self.features(x_s, tau, ctx))
        return torch.stack(feats, 0).mean(0)         # [B, width]


def env_context(env, device) -> torch.Tensor:
    """Constant context vector c = normalized [start, goal, obstacles(cx,cy,r)*N]."""
    parts = [env.x0[:2] / 6.0, env.goal / 6.0]
    obs = env.obstacles.clone()
    obs[:, :2] = obs[:, :2] / 6.0
    parts.append(obs.reshape(-1))
    return torch.cat(parts).to(device).float()
