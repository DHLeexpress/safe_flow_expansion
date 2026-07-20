"""NN-based verifier-uncertainty estimator — the estimator ACTFLOW (arXiv 2606.08802) actually uses in its
continuous experiments (Apx. experimental details): a deep bootstrapped ensemble of 5 MLPs, each 2 hidden
layers × 100 units, ReLU, 10% dropout; each member trained independently on a 90% bootstrap subsample of the
accumulated (φ_s, y) feature–label pairs (Adam 1e-3, up to `max_steps`); the ensemble standard deviation
across members is the uncertainty signal.

Contrast with `uncertainty.GPUncertainty` (the linear/RBF-kernel posterior std we currently use): the GP is
fit on QUERIED features only and its σ is pure NOVELTY (distance-to-buffer). This ensemble is a validity
CLASSIFIER fit on BOTH accepted (y=1) and rejected (y=0) windows, so its σ (member disagreement) peaks at the
decision boundary of the valid set — informative-query uncertainty, which is the term Eq-9 actually maximizes.
That boundary-seeking signal is what turns a unimodal prior multimodal (it points at *new valid* regions, not
just far ones). Kept as a SEPARATE module so gp-vs-nn is a clean A/B; same set_buffer/sigma call surface as the
GP so `grid_rollout.fm_deploy` needs no change, plus a `fit(phi, y)` entry point for the labels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _l2norm(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class _Member(nn.Module):
    def __init__(self, d_in, hidden=100, p_drop=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logit


class NNUncertainty:
    """Bootstrapped deep ensemble uncertainty over the L2-normalized flow representation φ_s.

    Interface parity with GPUncertainty:
      - sigma(phi_query) -> [M] uncertainty (ensemble std of predicted validity prob), used by fm_deploy;
      - set_buffer(feats) is a NO-OP shim so a caller that only has features (no labels) does not crash —
        the ensemble is (re)fit via fit(phi, y). Before any fit, sigma returns 1 (cold start, matches GP).
    """

    def __init__(self, n_members=5, hidden=100, p_drop=0.10, lr=1e-3, max_steps=1000,
                 batch=256, bootstrap=0.9, normalize=True, warm_start=False, device="cpu",
                 patience=60, weight_decay=0.0):
        self.n_members = n_members
        self.hidden = hidden
        self.p_drop = p_drop
        self.lr = lr
        self.max_steps = max_steps
        self.batch = batch
        self.bootstrap = bootstrap
        self.normalize = normalize
        self.warm_start = warm_start
        self.device = device
        self.patience = patience
        self.weight_decay = weight_decay
        self.members: list[_Member] | None = None
        self.d_in: int | None = None

    def _prep(self, phi):
        return _l2norm(phi) if self.normalize else phi

    # kept so a GP-style caller (features only) is harmless; real fitting is fit(phi, y)
    def set_buffer(self, _feats):
        return

    def fit(self, phi, y, seed=0):
        """phi [N,D] features, y [N] in {0,1}. Trains n_members classifiers on 90% bootstraps.
        Skips (keeps prior members) if a class is absent — a single-class buffer has no boundary to learn."""
        phi = self._prep(phi.detach()).to(self.device)
        y = y.detach().float().to(self.device)
        N, D = phi.shape
        if N < 8 or float(y.min()) == float(y.max()):
            return False                                    # need both classes to define uncertainty
        self.d_in = D
        g = torch.Generator(device="cpu").manual_seed(seed)
        pos_w = ((y == 0).sum().clamp_min(1) / (y == 1).sum().clamp_min(1)).to(self.device)  # class imbalance
        new_members = []
        for m in range(self.n_members):
            mem = (self.members[m] if (self.warm_start and self.members and m < len(self.members))
                   else _Member(D, self.hidden, self.p_drop).to(self.device))
            idx_all = torch.randperm(N, generator=g)[:max(8, int(self.bootstrap * N))]
            opt = torch.optim.Adam(mem.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            mem.train()
            best = float("inf"); bad = 0
            for step in range(self.max_steps):
                bi = idx_all[torch.randint(0, len(idx_all), (min(self.batch, len(idx_all)),), generator=g)]
                logit = mem(phi[bi])
                loss = lossfn(logit, y[bi])
                opt.zero_grad(); loss.backward(); opt.step()
                lv = float(loss)
                if lv < best - 1e-4:
                    best = lv; bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            mem.eval()
            new_members.append(mem)
        self.members = new_members
        return True

    @torch.no_grad()
    def sigma(self, phi_query):
        """phi_query [M,D] -> [M] ensemble std of predicted validity probability. Cold start (unfit) -> 1."""
        q = self._prep(phi_query).to(self.device)
        if self.members is None:
            return torch.ones(q.shape[0], device=q.device)
        preds = torch.stack([torch.sigmoid(m(q)) for m in self.members], 0)   # [n_members, M]
        return preds.std(dim=0)                                                # member disagreement

    @torch.no_grad()
    def mean_prob(self, phi_query):
        """Ensemble mean predicted validity prob (diagnostic; not used for tilting)."""
        q = self._prep(phi_query).to(self.device)
        if self.members is None:
            return torch.full((q.shape[0],), 0.5, device=q.device)
        return torch.stack([torch.sigmoid(m(q)) for m in self.members], 0).mean(0)


if __name__ == "__main__":
    torch.manual_seed(0)
    # Toy that survives L2-normalization: validity depends on DIRECTION (angle), not radius. valid = the
    # design's first coordinate (after normalizing to the sphere) exceeds a threshold -> a boundary at
    # cos=THR that the ensemble should be UNSURE about (high disagreement) and confident away from.
    D = 16
    THR = 0.35
    X = torch.randn(800, D)
    Xn = X / X.norm(dim=1, keepdim=True)
    y = (Xn[:, 0] > THR).float()
    unc = NNUncertainty(max_steps=400, device="cpu")
    ok = unc.fit(X, y)
    print("fit ok:", ok, "| valid fraction", round(float(y.mean()), 2))

    def probe(c0):                      # a query whose normalized 1st-coord = c0
        v = torch.zeros(D); v[0] = c0; v[1] = (max(1e-4, 1 - c0 * c0)) ** 0.5
        return v

    q = torch.stack([probe(0.9), probe(-0.9), probe(THR)])   # deep-valid, deep-invalid, boundary
    s = unc.sigma(q); p = unc.mean_prob(q)
    for lbl, i in (("deep-valid", 0), ("deep-invalid", 1), ("boundary", 2)):
        print(f"  {lbl:12s}: mean_prob {float(p[i]):.2f}  sigma(disagreement) {float(s[i]):.3f}")
    assert s[2] > s[0] and s[2] > s[1], "boundary should have the highest ensemble disagreement"
    print("  -> boundary-peaking confirmed (σ_boundary > σ_deep)")
    print("cold-start sigma (unfit):", float(NNUncertainty().sigma(torch.randn(4, D))[0]), "(expect 1.0)")
