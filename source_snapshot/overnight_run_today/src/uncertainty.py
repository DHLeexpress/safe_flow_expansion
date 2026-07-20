"""ACTFLOW Eq. (10): posterior-variance uncertainty over the noised-flow representation phi_s.

    sigma_t^2(x) = k(x,x) - k(x,X_t) (K_t + lambda I)^{-1} k(X_t,x)

Generic GP form with a selectable kernel on (L2-normalized) phi_s features:
  - 'linear' : k(a,b) = <a,b>  (paper-faithful; with normalization this is cosine, k(x,x)=1)
  - 'rbf'    : k(a,b) = exp(-||a-b||^2 / (2 ell^2))  (the 2-D toy used RBF, ell=0.08)

Normalization makes k(x,x)=1, so the cold-start (empty buffer) uncertainty is the constant 1 for
ALL x -- exactly the chessboard iteration-0 behavior (diagnosis D1).  The buffer features must be
RE-EXTRACTED from the current policy every round (phi_s^t co-evolves with theta_t); this class only
holds the current snapshot.
"""
from __future__ import annotations

import torch


def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class GPUncertainty:
    def __init__(self, kernel: str = "linear", lengthscale: float = 1.0,
                 lam: float = 1e-2, normalize: bool = True):
        assert kernel in ("linear", "rbf")
        self.kernel = kernel
        self.ell = lengthscale
        self.lam = lam
        self.normalize = normalize
        self.X: torch.Tensor | None = None       # [t,D] buffer features (normalized)
        self._L: torch.Tensor | None = None       # chol of (K_XX + lam I)

    def _prep(self, phi: torch.Tensor) -> torch.Tensor:
        return _l2norm(phi) if self.normalize else phi

    def _k(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if self.kernel == "linear":
            return A @ B.T
        d2 = torch.cdist(A, B) ** 2
        return torch.exp(-d2 / (2 * self.ell ** 2))

    def _kxx(self, A: torch.Tensor) -> torch.Tensor:
        if self.kernel == "linear":
            return (A * A).sum(-1)               # =1 if normalized
        return torch.ones(A.shape[0], device=A.device)

    def set_buffer(self, phi_buffer: torch.Tensor | None):
        if phi_buffer is None or phi_buffer.shape[0] == 0:
            self.X, self._L = None, None
            return
        X = self._prep(phi_buffer)
        self.X = X
        K = self._k(X, X).double()                 # float64 for a stable Cholesky
        I = torch.eye(K.shape[0], device=K.device, dtype=torch.float64)
        jit = float(self.lam)
        for _ in range(6):                          # jitter retry: greedy queries can make K near-singular
            try:
                self._L = torch.linalg.cholesky(K + jit * I)
                return
            except Exception:
                jit *= 10.0
        self._L = torch.linalg.cholesky(K + jit * I)

    @torch.no_grad()
    def sigma(self, phi_query: torch.Tensor) -> torch.Tensor:
        """phi_query [M,D] -> sigma [M] (>=0)."""
        q = self._prep(phi_query)
        kxx = self._kxx(q)                        # [M]
        if self.X is None:                        # cold start: sigma = sqrt(k(x,x))
            return kxx.clamp_min(0).sqrt()
        kxX = self._k(q, self.X)                  # [M,t]
        # solve (K+lam I) a = kXx  via chol (float64):  a = L^{-T} L^{-1} kXx
        v = torch.cholesky_solve(kxX.T.double(), self._L)  # [t,M]
        reduction = (kxX * v.T.to(kxX.dtype)).sum(dim=1)   # [M]
        var = (kxx - reduction).clamp_min(0.0)
        return var.sqrt()

    # ----------------------------------------------------------- diagnostics (D1..D6)
    @torch.no_grad()
    def diagnostics(self, phi_buffer: torch.Tensor | None, phi_fresh: torch.Tensor) -> dict:
        out = {}
        sig_fresh = self.sigma(phi_fresh)
        out["sigma_fresh_mean"] = float(sig_fresh.mean())
        out["sigma_fresh_std"] = float(sig_fresh.std())
        # D6 representation collapse guard
        out["sigma_cov"] = float(sig_fresh.std() / sig_fresh.mean().clamp_min(1e-9))
        if phi_buffer is not None and phi_buffer.shape[0] > 0:
            sig_buf = self.sigma(phi_buffer)
            out["sigma_buffer_mean"] = float(sig_buf.mean())      # D2: should be << fresh
            # D4 novelty correlation: sigma vs min-dist-to-buffer (Spearman via rank corr)
            q = self._prep(phi_fresh)
            X = self._prep(phi_buffer)
            mind = torch.cdist(q, X).amin(dim=1)
            out["D4_novelty_corr"] = float(_spearman(sig_fresh, mind))
            out["buffer_rank"] = int(torch.linalg.matrix_rank(self.X).item()) if self.X is not None else 0
        return out


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 3:
        return float("nan")
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-9)
    return float((ra @ rb) / denom)
