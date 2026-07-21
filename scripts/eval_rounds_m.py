#!/usr/bin/env python3
"""Raw per-round checkpoint evaluation at configurable M with (a) per-gamma
evaluation-temperature overrides and (b) an adaptive-gamma scheduler mode.

Mirrors the canonical raw evaluator (paper_results/low7_raw_m50_eval.py):
  - bare H=10 receding-horizon policy, no GP/tilt/verifier in the loop,
  - T=300, reach=0.15, NFE=8, round-independent pinned noise bank,
  - metrics per episode from afe_m20_eval._trajectory_metrics_worker
    (v_safe = taskspace AND sliding-window SOCP at the episode gamma,
    clearance over all states/obstacles, CR = collision or OOB,
    time-to-goal on successes only).

Additions over the canonical evaluator:
  - --m: any per-gamma episode count (bank regenerated for (scene, m)).
  - --temp-map: per-gamma, round-gated sampling-temperature overrides,
    e.g. '{"0.1": [[10, 0.5]]}' = gamma 0.1 evaluates at temperature 0.5
    from round 10 on (temp scales the initial flow noise; sample() computes
    x = temp * initial_noise). All overrides are recorded in the output.
  - --adaptive "alpha,beta": per-step gamma scheduler
        gamma_{k+1} = clip(gamma_k + alpha*(beta + dmargin/dgamma)*dt, 0, 1)
    with gamma_0 = 0.5. The margin is BEHAVIORAL: the min obstacle clearance
    of the planned window the policy proposes at a given conditioning gamma;
    dmargin/dgamma is the common-noise central difference over probe plans
    sampled at clip(gamma_k +/- 0.05) (the certificate slack is structurally
    monotone in gamma and cannot push gamma down; the behavioral clearance
    is gamma-sensitive exactly where safety must win: measured on r19, grad
    q10 = -0.099 near obstacles, q90 = +0.009 in open space). beta > 0 is the
    aggressiveness drift. Validity for the adaptive rollout re-runs the
    sliding-window SOCP with each window certified at the gamma active at the
    window's start step.

Output: <outdir>/rounds.jsonl, one row per (round, gamma-or-adaptive) with
per-episode arrays and mean/1-sigma-SE aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import multiprocessing as mp

import numpy as np

WORKBOOK = Path(__file__).resolve().parents[1]
SNAP = WORKBOOK / "source_snapshot/overnight_run_07_06/rev_expansion/codex_overnight"
for entry in (str(SNAP.parents[1]), str(SNAP.parent), str(SNAP),
              str(SNAP / "paper_results")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

BANK_VERSION = "b1_round_eval_m_v1"
SCENE = "low7_radius1_canonical_v1"
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
T = 300
REACH = 0.15
NFE = 8
N_THETA = 180
SOCP_R = 2.5
H_WIN = 10
STRIDE = 2
DELTA_GAMMA = 0.05
INFEASIBLE_MARGIN = -0.05


def sha_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def make_bank(n_gamma: int, m: int, d: int, split: str = "legacy") -> np.ndarray:
    seed_parts = (
        (BANK_VERSION, SCENE, n_gamma, m, d)
        if split == "legacy"
        else (BANK_VERSION, split, SCENE, n_gamma, m, d)
    )
    rng = np.random.default_rng(sha_seed(*seed_parts))
    return rng.standard_normal((n_gamma, m, T, d), dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pool workers (spawn-imported; keep import-safe at module level).
# ---------------------------------------------------------------------------

def _adaptive_metrics_worker(task):
    """Metrics for one adaptive-gamma episode: canonical geometry metrics plus
    v_safe with each sliding window certified at its start-step gamma."""
    import afe_m20_eval as M20
    import grid_metrics as GM
    from verifier_polytope import certify_window

    path, gamma_trace, status, dt, reach = task
    env = M20._WORKER_ENV
    points = np.asarray(path, dtype=np.float64)
    obstacles = env.obstacles.detach().cpu().numpy()
    clearance = float(
        (np.linalg.norm(points[:, None, :] - obstacles[None, :, :2], axis=2)
         - obstacles[None, :, 2] - float(env.r_robot)).min()
    ) if obstacles.size else float("inf")
    collision = bool(clearance < 0.0)
    oob = bool((points < -GM.EPS_TASK).any() or (points > GM.GRID_M + GM.EPS_TASK).any())
    reached = bool(status == "reached")
    taskspace = bool(GM.in_taskspace(points))
    n_steps = len(points) - 1
    socp = True
    if n_steps < H_WIN:
        socp = False
    else:
        for k in range(0, n_steps, STRIDE):
            span = min(H_WIN, n_steps - k)
            if span < 1:
                break
            gamma_k = float(gamma_trace[min(k, len(gamma_trace) - 1)])
            ok, *_ = certify_window(
                points[k:k + span + 1], obstacles, float(env.r_robot), gamma_k,
                R=SOCP_R, n_theta=N_THETA,
            )
            if not ok:
                socp = False
                break
    cr = bool(collision or oob)
    return {
        "status": str(status),
        "success": bool(reached and not cr),
        "cr": cr,
        "timeout": bool(status == "timeout" and not cr),
        "v_safe": bool(taskspace and socp),
        "minimum_clearance": clearance,
        "steps": int(n_steps),
        "time_to_goal": float(n_steps * dt) if reached and not cr else None,
        "gamma_final": float(gamma_trace[-1]) if len(gamma_trace) else 0.5,
        "gamma_mean": float(np.mean(gamma_trace)) if len(gamma_trace) else 0.5,
    }


def window_clearance_margin(states, controls, obstacles, r_robot, dt):
    """Vectorized min obstacle clearance of planned windows.
    states [B,4]; controls [B,H,2] -> [B] clearance of the H rolled positions."""
    B, H, _ = controls.shape
    p = states[:, :2].copy()
    v = states[:, 2:].copy()
    clear = np.full(B, np.inf)
    for t in range(H):
        a = controls[:, t]
        p = p + dt * v + 0.5 * dt * dt * a
        v = v + dt * a
        d = (np.linalg.norm(p[:, None, :] - obstacles[None, :, :2], axis=2)
             - obstacles[None, :, 2] - r_robot)
        clear = np.minimum(clear, d.min(axis=1))
    return clear


# ---------------------------------------------------------------------------


def resolve_temp(temp_map: dict, gamma: float, round_i: int) -> float:
    entries = temp_map.get(f"{gamma:g}", [])
    temp = 1.0
    for from_round, value in entries:
        if round_i >= int(from_round):
            temp = float(value)
    return temp


def run_fixed(policy, env, device, bank, m, gammas, temps, seed_round):
    """Batched raw receding-horizon rollout; per-episode temperature."""
    import torch
    import afe_context as CX

    start = env.x0.detach().cpu().numpy().astype(np.float32)
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    schema = "low7_closest_boundary_tie_mean"
    episodes = []
    for gi, gamma in enumerate(gammas):
        for ri in range(m):
            episodes.append(dict(
                gamma_index=gi, rollout_index=ri, gamma=float(gamma),
                temp=float(temps[gi]), state=start.copy(), history=[],
                path=[start[:2].copy()], status=None,
            ))
    with torch.no_grad():
        for control_t in range(T):
            active = [e for e in episodes if e["status"] is None]
            if not active:
                break
            grids, lows, hists, noises = [], [], [], []
            for e in active:
                rec = CX.build_context(e["state"], goal, e["gamma"], e["history"], env, schema)
                grids.append(rec.grid)
                lows.append(rec.low5)
                hists.append(rec.hist)
                noises.append(bank[e["gamma_index"], e["rollout_index"], control_t] * e["temp"])
            ctx = policy.ctx_from(
                torch.as_tensor(np.asarray(grids, np.float32), device=device),
                torch.as_tensor(np.asarray(lows, np.float32), device=device),
                torch.as_tensor(np.asarray(hists, np.float32), device=device),
            )
            controls = policy.sample(
                len(active), ctx, nfe=NFE, temp=1.0,
                initial_noise=torch.as_tensor(np.asarray(noises), device=device),
            ).detach().cpu().numpy()
            for e, window in zip(active, controls):
                a = np.asarray(window[0], np.float32)
                s = e["state"]
                dt = float(env.dt)
                s = np.array([s[0] + dt * s[2] + 0.5 * dt * dt * a[0],
                              s[1] + dt * s[3] + 0.5 * dt * dt * a[1],
                              s[2] + dt * a[0], s[3] + dt * a[1]], np.float32)
                e["state"] = s
                e["history"].append(a)
                e["path"].append(s[:2].copy())
                p = s[:2]
                import grid_metrics as GM
                if np.linalg.norm(p - goal) < REACH:
                    e["status"] = "reached"
                elif (p < -GM.EPS_TASK).any() or (p > GM.GRID_M + GM.EPS_TASK).any():
                    e["status"] = "oob"
                elif obstacles.size and (
                    np.linalg.norm(p[None] - obstacles[:, :2], axis=1)
                    - obstacles[:, 2] - rr
                ).min() < 0.0:
                    e["status"] = "collision"
    return episodes


def run_adaptive(policy, env, device, bank, m, alpha, beta, executor=None):
    """Adaptive-gamma rollout.

    Per step: sample the executed plan at gamma_k, plus two common-noise
    probe plans at clip(gamma_k +/- DELTA_GAMMA); the margin is the planned
    window's min obstacle clearance and its gamma-gradient is the central
    difference over the probes (behavioral sensitivity: aggressive
    conditioning cuts closer to obstacles, so the gradient is negative
    exactly where safety must win over the beta aggressiveness drift).
    """
    import torch
    import afe_context as CX
    import grid_metrics as GM

    start = env.x0.detach().cpu().numpy().astype(np.float32)
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    dt = float(env.dt)
    schema = "low7_closest_boundary_tie_mean"
    episodes = [dict(rollout_index=ri, gamma=0.5, gamma_trace=[], grad_trace=[],
                     state=start.copy(), history=[], path=[start[:2].copy()],
                     status=None) for ri in range(m)]
    with torch.no_grad():
        for control_t in range(T):
            active = [e for e in episodes if e["status"] is None]
            if not active:
                break
            n = len(active)
            gam = np.array([e["gamma"] for e in active], dtype=np.float64)
            g_hi = np.clip(gam + DELTA_GAMMA, 0.0, 1.0)
            g_lo = np.clip(gam - DELTA_GAMMA, 0.0, 1.0)
            grids, lows, hists, noises = [], [], [], []
            for variant_gammas in (gam, g_hi, g_lo):
                for e, g in zip(active, variant_gammas):
                    rec = CX.build_context(e["state"], goal, float(g),
                                           e["history"], env, schema)
                    grids.append(rec.grid)
                    lows.append(rec.low5)
                    hists.append(rec.hist)
                    noises.append(bank[0, e["rollout_index"], control_t])
            ctx = policy.ctx_from(
                torch.as_tensor(np.asarray(grids, np.float32), device=device),
                torch.as_tensor(np.asarray(lows, np.float32), device=device),
                torch.as_tensor(np.asarray(hists, np.float32), device=device),
            )
            controls = policy.sample(
                3 * n, ctx, nfe=NFE, temp=1.0,
                initial_noise=torch.as_tensor(np.asarray(noises), device=device),
            ).detach().cpu().numpy()
            exec_plans = controls[:n]
            states = np.stack([e["state"] for e in active])
            clr_hi = window_clearance_margin(states, controls[n:2 * n],
                                             obstacles, rr, dt)
            clr_lo = window_clearance_margin(states, controls[2 * n:],
                                             obstacles, rr, dt)
            span = np.maximum(g_hi - g_lo, 1e-6)
            grads = (clr_hi - clr_lo) / span
            for idx, (e, window) in enumerate(zip(active, exec_plans)):
                e["gamma_trace"].append(e["gamma"])
                e["grad_trace"].append(float(grads[idx]))
                gnew = e["gamma"] + alpha * (beta + grads[idx]) * dt
                e["gamma"] = float(np.clip(gnew, 0.0, 1.0))
                a = np.asarray(window[0], np.float32)
                s = e["state"]
                s = np.array([s[0] + dt * s[2] + 0.5 * dt * dt * a[0],
                              s[1] + dt * s[3] + 0.5 * dt * dt * a[1],
                              s[2] + dt * a[0], s[3] + dt * a[1]], np.float32)
                e["state"] = s
                e["history"].append(a)
                e["path"].append(s[:2].copy())
                p = s[:2]
                if np.linalg.norm(p - goal) < REACH:
                    e["status"] = "reached"
                elif (p < -GM.EPS_TASK).any() or (p > GM.GRID_M + GM.EPS_TASK).any():
                    e["status"] = "oob"
                elif obstacles.size and (
                    np.linalg.norm(p[None] - obstacles[:, :2], axis=1)
                    - obstacles[:, 2] - rr
                ).min() < 0.0:
                    e["status"] = "collision"
    return episodes


def aggregate(rows, key_bool=("cr", "v_safe")):
    n = len(rows)
    out = {"n": n}
    successes = sum(1 for row in rows if row["success"])
    success_rate = successes / n
    out["SR"] = {
        "mean": success_rate,
        "se": float(np.sqrt(success_rate * (1 - success_rate) / n)),
    }
    for key in key_bool:
        k = sum(1 for r in rows if r[key])
        p = k / n
        out[key.upper() if key == "cr" else key] = {
            "mean": p, "se": float(np.sqrt(p * (1 - p) / n)),
        }
    clear = np.array([r["minimum_clearance"] for r in rows], dtype=float)
    out["clearance"] = {"mean": float(clear.mean()),
                        "se": float(clear.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0}
    times = np.array([r["time_to_goal"] for r in rows if r["time_to_goal"] is not None])
    out["time"] = {
        "mean": float(times.mean()) if times.size else None,
        "se": float(times.std(ddof=1) / np.sqrt(times.size)) if times.size > 1 else 0.0,
        "n_success": int(times.size),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", type=Path, required=True,
                        help="directory containing ckpt_<round>.pt")
    parser.add_argument("--rounds", default="0-20")
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--gammas", default=",".join(f"{g:g}" for g in GAMMAS))
    parser.add_argument("--temp-map", default="{}",
                        help='JSON {"0.1": [[10, 0.5]]}: gamma 0.1 uses temp 0.5 from round 10')
    parser.add_argument("--adaptive", default=None, help="alpha,beta -> adaptive mode")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tag", default="eval")
    parser.add_argument(
        "--bank-split", default="legacy",
        help="declared CRN split name; use different names for calibration and evaluation",
    )
    parser.add_argument(
        "--fail-closed", action="store_true",
        help="require absent metric/contract outputs and write them exactly once",
    )
    args = parser.parse_args()

    import torch  # noqa: F401  (after CUDA_VISIBLE_DEVICES is set by caller)
    import afe_m20_eval as M20
    import grid_hp_expt as HP
    from afe2_scene_profiles import build_scene, get_scene_profile

    lo, hi = (args.rounds.split("-") + [args.rounds])[:2]
    rounds = list(range(int(lo), int(hi) + 1))
    gammas = tuple(float(g) for g in args.gammas.split(","))
    temp_map = json.loads(args.temp_map)
    adaptive = None
    if args.adaptive:
        a, b = args.adaptive.split(",")
        adaptive = (float(a), float(b))

    env = build_scene(get_scene_profile(SCENE))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"{args.tag}.jsonl"
    contract_path = args.outdir / f"{args.tag}.contract.json"
    if args.fail_closed and (out_path.exists() or contract_path.exists()):
        raise FileExistsError(f"refusing stale evaluation output: {out_path}")
    probe_policy, _ = HP.load_hp(str(args.arm_dir / f"ckpt_{rounds[0]}.pt"), device="cpu")
    d = int(probe_policy.d)
    if adaptive:
        bank = make_bank(1, args.m, d, args.bank_split)
    else:
        # Always generate the full 7-gamma bank and slice canonical slots so a
        # single-gamma re-evaluation (temperature overrides) reuses exactly the
        # same noise as the full run -- no seed discontinuity when splicing.
        full = make_bank(len(GAMMAS), args.m, d, args.bank_split)
        bank = full[[GAMMAS.index(g) for g in gammas]]

    checkpoint_hashes = {
        str(round_i): sha256_file(args.arm_dir / f"ckpt_{round_i}.pt")
        for round_i in rounds
    }

    context = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context,
        initializer=M20._worker_init, initargs=(SCENE, REACH, N_THETA),
    ) as executor, out_path.open("x" if args.fail_closed else "a") as stream:
        for round_i in rounds:
            policy, _ = HP.load_hp(str(args.arm_dir / f"ckpt_{round_i}.pt"), device="cpu")
            policy = policy.to(args.device).eval()
            if adaptive is None:
                temps = [resolve_temp(temp_map, g, round_i) for g in gammas]
                episodes = run_fixed(policy, env, args.device, bank, args.m,
                                     gammas, temps, round_i)
                tasks = [
                    (np.asarray(e["path"], np.float32), e["gamma"],
                     "timeout" if e["status"] is None else e["status"],
                     float(env.dt), REACH)
                    for e in episodes
                ]
                rows = list(executor.map(M20._trajectory_metrics_worker, tasks, chunksize=4))
                for gi, gamma in enumerate(gammas):
                    cell = [r for e, r in zip(episodes, rows) if e["gamma_index"] == gi]
                    record = {
                        "round": round_i, "gamma": gamma, "mode": "fixed",
                        "temp": temps[gi], "m": args.m,
                        **aggregate(cell),
                    }
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    print(f"[{args.tag}] r{round_i:02d} g{gamma:g} temp={temps[gi]:g} "
                          f"CR {record['CR']['mean']:.3f} Vsafe {record['v_safe']['mean']:.3f} "
                          f"clr {record['clearance']['mean']:.4f} "
                          f"time {record['time']['mean']}", flush=True)
            else:
                alpha, beta = adaptive
                episodes = run_adaptive(policy, env, args.device, bank, args.m,
                                        alpha, beta)
                tasks = [
                    (np.asarray(e["path"], np.float32),
                     np.asarray(e["gamma_trace"], np.float32),
                     "timeout" if e["status"] is None else e["status"],
                     float(env.dt), REACH)
                    for e in episodes
                ]
                rows = list(executor.map(_adaptive_metrics_worker, tasks, chunksize=4))
                gmeans = np.array([r["gamma_mean"] for r in rows])
                gfinal = np.array([r["gamma_final"] for r in rows])
                all_grads = np.concatenate(
                    [np.asarray(e["grad_trace"], float) for e in episodes if e["grad_trace"]]
                ) if any(e["grad_trace"] for e in episodes) else np.zeros(1)
                grad_stats = {
                    "mean": float(all_grads.mean()),
                    "q10": float(np.quantile(all_grads, 0.1)),
                    "q50": float(np.quantile(all_grads, 0.5)),
                    "q90": float(np.quantile(all_grads, 0.9)),
                }
                record = {
                    "round": round_i, "gamma": None, "mode": "adaptive",
                    "alpha": alpha, "beta": beta, "m": args.m,
                    "gamma_mean": float(gmeans.mean()),
                    "gamma_final_mean": float(gfinal.mean()),
                    "grad_stats": grad_stats,
                    **aggregate(rows),
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"[{args.tag}] r{round_i:02d} adaptive(a={alpha:g},b={beta:g}) "
                      f"CR {record['CR']['mean']:.3f} Vsafe {record['v_safe']['mean']:.3f} "
                      f"clr {record['clearance']['mean']:.4f} time {record['time']['mean']} "
                      f"gamma_mean {record['gamma_mean']:.3f}", flush=True)
            print(f"[{args.tag}] r{round_i:02d} done at {time.time()-t0:.0f}s", flush=True)
    contract = {
        "status": "B1_METRICS_ONLY_EVALUATION_COMPLETE",
        "raw_policy": "bare receding-horizon flow; no GP, tilt, verifier, or fallback",
        "arm_dir": str(args.arm_dir.resolve()),
        "rounds": rounds,
        "gammas": list(gammas),
        "M_per_gamma": args.m,
        "NFE": NFE,
        "temperature_map": temp_map,
        "adaptive": adaptive,
        "bank_version": BANK_VERSION,
        "bank_split": args.bank_split,
        "bank_sha256": hashlib.sha256(bank.tobytes(order="C")).hexdigest(),
        "checkpoint_sha256": checkpoint_hashes,
        "metrics": str(out_path.resolve()),
        "metrics_sha256": sha256_file(out_path),
        "elapsed_seconds": time.time() - t0,
        "trajectories_persisted": False,
    }
    mode = "x" if args.fail_closed else "w"
    with contract_path.open(mode) as stream:
        json.dump(contract, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
