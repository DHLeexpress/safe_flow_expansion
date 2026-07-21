#!/usr/bin/env python3
"""M50 Kazuki CFM-MPPI tests on the canonical giant-obstacle scene.

Three questions, three arms (all with OUR B1 pretrained FM as the proposal
model, tie-mean low7 conditioning, gamma_ctx=0.5, T=300, reach=0.15, the
gallery's named-seed convention extended from 10 to M):

  zero        safe_coef=0, goal_coef=0 (the gallery diagnostic at M50):
              does zero-coefficient Kazuki collapse into raw pretrained
              sampling?  Elite selection, MPPI costs, refit, and warm-start
              dilution all remain active, so the executed distribution can
              still diverge from the raw generator.
  zero_nocost safe_coef=0, goal_coef=0, COLL_W=0, GOAL_W=0: even the MPPI
              stage cost removed.  Elite topk over constant costs plus the
              per-mode softmax refit (uniform weights -> perturbation mean)
              and warm-start dilution remain -- there is no coefficient
              setting that turns the pipeline into the identity map.
  faithful    their actual deployment protocol: mixed safe-coef batch
              {0.1,0.3,0.5,0.7,0.9} x 40 samples, goal_coef=0.1 (their
              GOAL_COEF), full FlowMPPI refinement and warm start.  Does the
              faithful method escape the local minimum around the giant
              obstacle?

Reference for raw pretrained behavior: the retained M50 confirmation cells
r000_g*.npz (no controller, temperature 1).  Kazuki runs use gamma_ctx=0.5,
so r000_g0.5 is the matched raw row.

Writes per-arm npz (paths + outcomes) and a summary JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKBOOK = Path(__file__).resolve().parents[1]
SNAP = WORKBOOK / "source_snapshot/overnight_run_07_06/rev_expansion/codex_overnight"
sys.path.insert(0, str(SNAP.parents[1]))          # overnight_run_07_06
sys.path.insert(0, str(SNAP.parent))              # rev_expansion
sys.path.insert(0, str(SNAP))                     # codex_overnight

import kazuki_baseline as KB                       # noqa: E402
from afe2_scene_profiles import build_scene, get_scene_profile  # noqa: E402
from grid_hp_expt import load_hp                   # noqa: E402

METRIC_VERSION = "b1_current_best_gallery_v1"      # keep the gallery seed stream
SCHEMA = "low7_closest_boundary_tie_mean"
GAMMA_CTX = 0.5

ARMS = {
    "zero": dict(safe_coefs=[0.0], goal_coef=0.0, coll_w=100.0, goal_w=0.1),
    "zero_nocost": dict(safe_coefs=[0.0], goal_coef=0.0, coll_w=0.0, goal_w=0.0),
    "faithful": dict(
        safe_coefs=[0.1, 0.3, 0.5, 0.7, 0.9], goal_coef=0.1,
        coll_w=100.0, goal_w=0.1,
    ),
}


def named_seed(*parts) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def classify_path(path: np.ndarray, env, reach: float) -> str:
    obs = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    rr = float(env.r_robot)
    dmin = (
        np.linalg.norm(path[:, None, :2] - obs[None, :, :2], axis=2)
        - obs[None, :, 2] - rr
    ).min()
    if dmin < 0.0:
        return "CR"
    if np.linalg.norm(path[-1] - goal) < reach:
        return "SR"
    return "TO"


def run_arm(arm: str, policy, env, m: int, t_cap: int, reach: float, device: str):
    spec = ARMS[arm]
    KB.GOAL_COEF = spec["goal_coef"]
    KB.COLL_W = spec["coll_w"]
    KB.GOAL_W = spec["goal_w"]
    KB.BETA_MPPI = 20.0
    KB.MPPI_LAMBDA = 0.1
    KB.MPPI_SIGMA = 0.2
    KB.R_MARGIN = 0.05
    KB.N_SAMPLE = 200
    KB.N_ELITE = 10
    KB.N_COPY = 200
    paths, outcomes = [], []
    t0 = time.time()
    for index in range(m):
        seed = named_seed(METRIC_VERSION, "kazuki", index)
        seed_all(seed)
        out = KB.kazuki_deploy(
            policy, env, spec["safe_coefs"], gamma_ctx=GAMMA_CTX, T=t_cap,
            reach=reach, device=device, seed=seed, conditioning_schema=SCHEMA,
        )
        path = np.asarray(out["path"], dtype=np.float32)
        paths.append(path)
        outcomes.append(classify_path(path, env, reach))
        done = index + 1
        counts = {k: outcomes.count(k) for k in ("SR", "CR", "TO")}
        print(
            f"[{arm}] ep{index:03d} {outcomes[-1]} steps={out['steps']} | "
            f"SR {counts['SR']/done:.2f} CR {counts['CR']/done:.2f} "
            f"TO {counts['TO']/done:.2f} ({(time.time()-t0)/done:.1f}s/ep)",
            flush=True,
        )
    return paths, outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-ckpt", type=Path,
        default=Path("/home/dohyun/projects/afe2_runs/low7_groupavg_tiemean_r0_pair_0f0c128/"
                     "seed_20260718_eq_0_ga_1/pretrain/data/checkpoint_candidate.pt"),
    )
    parser.add_argument("--expected-ckpt-sha256",
                        default="524c9c0a4fd071221ac509b9d8e6fbbfb85fdf1811aa04160317f2a9e2d3ef90")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--reach", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    digest = hashlib.sha256(args.pretrained_ckpt.read_bytes()).hexdigest()
    if digest != args.expected_ckpt_sha256:
        raise RuntimeError(f"checkpoint hash mismatch: {digest}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    profile = get_scene_profile("low7_radius1_canonical_v1")
    env = build_scene(profile)
    policy, _ = load_hp(str(args.pretrained_ckpt), device="cpu")
    policy = policy.to(args.device).eval()

    summary = {
        "metric_version": METRIC_VERSION,
        "scene": "low7_radius1_canonical_v1",
        "schema": SCHEMA,
        "gamma_ctx": GAMMA_CTX,
        "checkpoint_sha256": digest,
        "M": args.m, "T": args.T, "reach": args.reach,
        "arms": {},
    }
    for arm in args.arms:
        paths, outcomes = run_arm(
            arm, policy, env, args.m, args.T, args.reach, args.device
        )
        pa = np.empty(len(paths), dtype=object)
        for i, p in enumerate(paths):
            pa[i] = p
        np.savez_compressed(
            args.outdir / f"kazuki_{arm}_m{args.m}.npz",
            paths=pa, outcomes=np.array(outcomes),
            config_json=json.dumps(ARMS[arm]),
        )
        counts = {k: outcomes.count(k) for k in ("SR", "CR", "TO")}
        summary["arms"][arm] = {
            **ARMS[arm],
            "SR": counts["SR"] / args.m,
            "CR": counts["CR"] / args.m,
            "timeout": counts["TO"] / args.m,
        }
        print(f"[{arm}] FINAL {summary['arms'][arm]}", flush=True)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
