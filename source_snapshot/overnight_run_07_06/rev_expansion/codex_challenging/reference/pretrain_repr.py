"""Pretrain the repr-reduced 32x32 H_P flow policy on off-diagonal DR data (2026-07-07).

trunk 89 -> 128 -> 64 -> repr -> [head repr->20]; phi_s = repr feeds the GP sigma-regression.
Fast loop: off-diagonal `dr05_` (32x32), ~100 trajs/gamma. Sweep repr in {10,15,20}.
"""
from __future__ import annotations

import argparse
import math
import os

import torch
from torch.utils.data import TensorDataset, DataLoader

import _paths  # noqa: F401
import grid_hp_expt as HP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
OUT = os.path.join(HERE, "results", "hp_repr"); os.makedirs(OUT, exist_ok=True)
GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]


def load_data(prefix, gammas, cap, seed=0):
    G, L, Hh, U = [], [], [], []
    gen = torch.Generator().manual_seed(seed)
    for g in gammas:
        d = torch.load(os.path.join(DATA, f"{prefix}windows_g{g}.pt"), weights_only=False)
        n = d["grid"].shape[0]
        idx = torch.randperm(n, generator=gen)[:cap] if (cap and n > cap) else torch.arange(n)
        G.append(d["grid"][idx]); L.append(d["low5"][idx]); Hh.append(d["hist"][idx]); U.append(d["U"][idx])
    return (torch.cat(x) for x in (G, L, Hh, U))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", type=int, default=20)
    ap.add_argument("--prefix", default="dr05_")
    ap.add_argument("--per-gamma-cap", type=int, default=0, help="0 = use all windows")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--grid-hw", type=int, default=32)
    ap.add_argument("--trunk-hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--enc-depth", type=int, default=2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    G, L, Hh, U = load_data(args.prefix, GAMMAS, args.per_gamma_cap)
    n = G.shape[0]
    print(f"[pretrain_repr] repr={args.repr} grid={args.grid_hw} data '{args.prefix}' {n} windows "
          f"(cap {args.per_gamma_cap}/γ) gshape {tuple(G.shape[1:])}", flush=True)
    assert tuple(G.shape[1:]) == (3, args.grid_hw, args.grid_hw), \
        f"grid {tuple(G.shape[1:])} != (3,{args.grid_hw},{args.grid_hw}) — regenerate the data at this resolution!"
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    G, L, Hh, U = G[perm], L[perm], Hh[perm], U[perm]
    nval = max(2048, n // 10)
    tr = TensorDataset(G[nval:], L[nval:], Hh[nval:], U[nval:])
    va = tuple(x[:nval].to(dev) for x in (G, L, Hh, U))
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)

    pol = HP.GridHPFlowPolicy(repr_dim=args.repr, grid_hw=(args.grid_hw, args.grid_hw),
                              trunk_hidden=tuple(args.trunk_hidden), enc_depth=args.enc_depth).to(dev)
    npar = sum(p.numel() for p in pol.parameters())
    print(f"[pretrain_repr] model {npar/1e3:.1f}k params ({n-nval} train / {nval} val)", flush=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / args.warmup if ep < args.warmup else
        0.5 * (1 + math.cos(math.pi * (ep - args.warmup) / max(1, args.epochs - args.warmup))))
    best = (float("inf"), None)
    for ep in range(args.epochs):
        pol.train(); tot = nb = 0
        for gb, lb, hb, ub in dl:
            loss = pol.cfm_loss(ub.to(dev), pol.ctx_from(gb.to(dev), lb.to(dev), hb.to(dev)))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        sched.step(); pol.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            v = float(pol.cfm_loss(va[3], pol.ctx_from(va[0], va[1], va[2])))
        if v < best[0]:
            best = (v, {k: x.detach().cpu().clone() for k, x in pol.state_dict().items()})
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"[pretrain_repr] ep {ep:03d} train {tot/nb:.4f} val {v:.4f}", flush=True)
    pol.load_state_dict(best[1])
    tag = args.tag or f"repr{args.repr}_{args.prefix.rstrip('_')}"
    out = os.path.join(OUT, f"pretrained_{tag}.pt")
    HP.save_hp(pol, out, extra={"best_val": best[0], "data": args.prefix, "per_gamma_cap": args.per_gamma_cap})
    print(f"[pretrain_repr] saved {out} (val {best[0]:.4f})", flush=True)


if __name__ == "__main__":
    main()
