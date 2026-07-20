"""Metrics v2 (2026-07-03) — window-level validity + coverage_cumulative / coverage_final.

validity2 (user 2b): the ONLY criteria are window-level, over every sliding 10-window (stride 2):
  (1) SOCP-certified (verifier polytope, unchanged);
  (2) in task space;
  (3) GOAL-APPROACH (replaces monotone-staircase + reaches-goal): with d_t = ||x_t - goal|| over the window,
      max_t d_t <= d_0 + SLACK   (never retreats away from the goal beyond slack)  AND
      d_H  <= d_0 - min(DELTA, 0.5*d_0)   (net progress; DELTA calibrated on demo windows below);
      a window starting inside the reach disk auto-passes (arrived).
coverage_cumulative = old coverage (distinct staircases ever realized by a VALID deploy / 252, monotone up).
coverage_final     = distinct staircases realized by VALID deploys within ONE measurement batch / 252
                     (non-cumulative, can decrease; ceiling = n_deploys/252).
Existing grid_metrics.py is untouched; staircase_id / in_taskspace / socp_ok are reused from it.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
import grid_metrics as GM
import grid_feats as GF
import verifier_polytope as VP

GOAL_XY = np.array([GM.GRID_M, GM.GRID_M], dtype=float)


def staircase_id_goal(path, goal=None, reach=0.15):
    """Goal-relative coverage id (user 2026-07-14). `GM.staircase_id` hardcodes the (5,5) corner: it counts
    R/U boundary crossings toward (GRID_M,GRID_M), fires its reach-check on the distance to (5,5), and
    "completes to (5,5)". At a MOVED goal (e.g. cleared (4.7,4.7)) the path stops ~0.42 short, so that
    reach-check and completion misfire and coverage collapses to ~0 for every method (a metric artifact,
    not a real zero). Fix: rescale the path per-axis so `goal` maps exactly to (GRID_M,GRID_M), then reuse
    the canonical 252-staircase discretizer. The R/U crossing TOPOLOGY is now goal-relative (boundaries sit
    at evenly-spaced sub-goals goal*k/5), the 252 denominator is preserved, and (5,5) is a no-op."""
    g = np.asarray(GOAL_XY if goal is None else goal, dtype=float)[:2]
    p = np.asarray(path, dtype=float)[:, :2]
    if np.allclose(g, GM.GRID_M):
        return GM.staircase_id(p, reach=reach)
    if not np.all(g > 1e-6):
        return None                                  # degenerate goal on an axis -> no staircase frame
    scale = GM.GRID_M / g                            # per-axis map goal -> (GRID_M, GRID_M)
    return GM.staircase_id(p * scale[None, :], reach=reach * float(scale.mean()))
# calibrated on demo windows (calibrate_approach). Three conditions per window:
#   (net)  net progress start->end >= DELTA_PROG
#   (slack) never retreats from the window start by more than SLACK_RETREAT
#   (step) gets closer (nearly) EVERY step: no single step increases distance by more than STEP_TOL
#          (user 2026-07-03: check per-step approach, not just start-vs-end; STEP_TOL allows weaving detours)
DELTA_PROG = 0.10   # rev_expansion 2026-07-08: net-progress bar lowered 0.12 -> 0.10 (user)
SLACK_RETREAT = 0.15
STEP_TOL = 0.16


def approach_ok(dists, delta=DELTA_PROG, reach=GM.REACH):
    """rev_expansion FIX (user 2026-07-08): dual goal = performant + safe(SOCP). SOCP owns safety, so the
    over-restrictive bounded-retreat and per-step-tolerance stages are DROPPED — keep ONLY net progress
    (they rejected safe weaving/detour windows). dists: ||x_t - goal|| for t = 0..H (window start included)."""
    d = np.asarray(dists, dtype=float)
    d0 = d[0]
    if d0 < reach:                                   # already arrived -> auto-pass
        return True
    return d[-1] <= d0 - min(delta, 0.5 * d0)        # net progress over the window: d0 - dH >= min(0.10, 0.5 d0)


def criteria_status(path, env, gamma, H=10, stride=2):
    """Independent PASS/FAIL of each of the THREE validity criteria (for violation statistics, user
    2026-07-03): taskspace, approach (goal-distance every step), socp (verifier polytope). PASS = True."""
    p = np.asarray(path, dtype=float)
    task = GM.in_taskspace(p)
    if len(p) >= H + 1:
        D = np.linalg.norm(p - GOAL_XY[None], axis=1)
        appr = all(approach_ok(D[t:t + H + 1]) for t in range(0, len(p) - H, stride))
    else:
        appr = False
    socp = GM.socp_ok(p, env, gamma)
    return dict(taskspace=bool(task), approach=bool(appr), socp=bool(socp))


def traj_valid2(path, env, gamma, H=10, stride=2, check_socp=True):
    """validity2 of an executed trajectory: every sliding window passes approach ∧ taskspace ∧ SOCP."""
    p = np.asarray(path, dtype=float)
    if len(p) < H + 1:
        return False
    if not GM.in_taskspace(p):
        return False
    D = np.linalg.norm(p - GOAL_XY[None], axis=1)
    for t in range(0, len(p) - H, stride):
        if not approach_ok(D[t:t + H + 1]):
            return False
    if check_socp and not GM.socp_ok(p, env, gamma):
        return False
    return True


def traj_breakdown(path, env, gamma, H=10, stride=2):
    """Which criteria fail for this trajectory (for stats). Returns (valid: bool, status: dict of PASS bools)."""
    p = np.asarray(path, dtype=float)
    if len(p) < H + 1:
        return False, dict(taskspace=GM.in_taskspace(p), approach=False, socp=False)
    st = criteria_status(path, env, gamma, H, stride)
    return (st["taskspace"] and st["approach"] and st["socp"]), st


def window_label_cheap(state, U, env, gamma):
    """Cheap per-window buffer label during exploration (NO SOCP — that runs once per trajectory):
    planned window in task space ∧ approaches the goal. Signature matches grid_rollout verify_fn."""
    import grid_rollout as GR
    seg = GR.window_positions(state, U, env.dt)
    if not GM.in_taskspace(seg):
        return False
    d = np.linalg.norm(np.vstack([np.asarray(state, float)[None, :2], seg]) - GOAL_XY[None], axis=1)
    return approach_ok(d)


def window_min_clearance(state, U, env):
    """Min obstacle clearance of the planned window (for the pos_margin data-hygiene gate)."""
    import grid_rollout as GR
    seg = GR.window_positions(state, U, env.dt)
    obs = env.obstacles.detach().cpu().numpy()
    if not obs.size:
        return np.inf
    d = np.linalg.norm(seg[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - float(env.r_robot)
    return float(d.min())


def window_socp_stats(state, U, env, gamma, R=2.5, n_theta=180):
    """Return ``(ok, face_margin, cert_residual)`` for one planned control window.

    The minimum fitted-polytope level-set residual is *not* a useful continuous margin: fitting a face to
    the same trajectory normally leaves an active constraint, so that statistic is zero up to roundoff for
    more than half of feasible windows.  ``face_margin`` is instead the minimum optimized ``Face.m`` over
    sensed real obstacles (or ``R_eff`` when no real obstacle is sensed).  It is the verifier's literal
    nondegenerate SOCP decision margin.  Feasibility still comes from the unchanged certificate check.
    """
    import grid_rollout as GR
    st = np.asarray(state, dtype=float)
    seg = GR.window_positions(st, U, env.dt)
    path = np.vstack([st[:2][None], seg])
    obs = env.obstacles.detach().cpu().numpy()
    ok, faces, _raw, reff = VP.certify_window(path, obs, float(env.r_robot), float(gamma),
                                              R=R, n_theta=n_theta)
    alpha = (1.0 - float(gamma)) ** np.arange(len(path), dtype=float)
    ok2, cert_residual, _worst_t = VP.check_certificate(
        faces, path - path[0], alpha, include_start=False)
    if ok != ok2:
        raise RuntimeError(f"certificate result mismatch: certify_window={ok}, check={ok2}")
    if not ok2:
        return False, float("nan"), float(cert_residual)
    real_m = [float(f.m) for f in faces
              if getattr(f, "kind", None) == "real" and bool(getattr(f, "feasible", True))]
    face_margin = min(real_m) if real_m else float(reff)
    if not np.isfinite(face_margin) or face_margin <= 0.0:
        raise RuntimeError(f"invalid feasible real-face margin: {face_margin}")
    return True, float(face_margin), float(cert_residual)


def window_socp_margin(state, U, env, gamma, R=2.5, n_theta=180):
    """Minimum feasible real-face SOCP margin (low = tight but certified)."""
    return window_socp_stats(state, U, env, gamma, R=R, n_theta=n_theta)[1]


def measure2(paths, env, gamma, covered):
    """Batch measurement: validity2 %, coverage_cumulative (updates `covered` in place), coverage_final,
    reach-rate, avg steps of valid trajectories, and per-criterion VIOLATION fractions (fraction of the
    batch failing each of taskspace / approach / socp; a trajectory can fail several) — user 2026-07-03."""
    n_valid = 0
    final_set = set()
    steps = []
    n_reach = 0
    viol = dict(taskspace=0, approach=0, socp=0)
    goal = env.goal.detach().cpu().numpy()
    for path in paths:
        reached = GM.reaches_goal(path, goal)
        n_reach += int(reached)
        valid, st = traj_breakdown(path, env, gamma)
        for k, ok in st.items():
            if not ok:
                viol[k] += 1
        if valid:
            n_valid += 1
            steps.append(len(path))
            if reached:
                sid = GM.staircase_id(path)
                if sid is not None:
                    covered.add(sid)
                    final_set.add(sid)
    n = max(len(paths), 1)
    return dict(validity=n_valid / n,
                coverage_cum=len(covered) / GM.N_STAIR,
                coverage_final=len(final_set) / GM.N_STAIR,
                n_final=len(final_set),
                reach_rate=n_reach / n,
                avg_steps=float(np.mean(steps)) if steps else 0.0,
                violations={k: v / n for k, v in viol.items()})


def wilson_band(p, n, z=1.0):
    """+-z-sigma Wilson interval half-widths for a proportion (for validity plots)."""
    if n <= 0:
        return 0.0, 0.0
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, p - (ctr - hw)), max(0.0, (ctr + hw) - p)


def calibrate_approach(demo, dt=0.1, n_max=8000, seed=0):
    """Pass-rate of the approach test on expert demo windows, reconstructed goal-relative from low5:
    p = goal - relgoal*R_GOAL, v = low5[2:4]*V_SCALE, DI-roll U -> distances. Returns stats + suggested DELTA
    (1st percentile of demo net progress, floored at 0.05)."""
    import grid_rollout as GR
    rng = np.random.default_rng(seed)
    n = demo["U"].shape[0]
    idx = rng.permutation(n)[:min(n, n_max)]
    L = demo["low5"][idx].numpy()
    Uw = demo["U"][idx].numpy()
    prog, step_inc, ok = [], [], 0
    for i in range(len(idx)):
        p = GOAL_XY - L[i, :2].astype(float) * GF.R_GOAL
        v = L[i, 2:4].astype(float) * GF.V_SCALE
        st = np.array([p[0], p[1], v[0], v[1]], np.float32)
        seg = GR.window_positions(st, Uw[i], dt)
        d = np.linalg.norm(np.vstack([p[None], seg]) - GOAL_XY[None], axis=1)
        prog.append(d[0] - d[-1])
        step_inc.append(float(np.diff(d).max()) if len(d) > 1 else 0.0)
        ok += int(approach_ok(d))
    prog = np.asarray(prog); step_inc = np.asarray(step_inc)
    return dict(pass_rate=ok / len(idx), n=len(idx),
                prog_p01=float(np.percentile(prog, 1)), prog_p05=float(np.percentile(prog, 5)),
                prog_med=float(np.median(prog)),
                step_inc_p95=float(np.percentile(step_inc, 95)), step_inc_p99=float(np.percentile(step_inc, 99)),
                step_inc_max=float(step_inc.max()),
                suggested_delta=float(max(0.05, np.percentile(prog, 1))),
                suggested_step_tol=float(np.percentile(step_inc, 99)))


if __name__ == "__main__":
    import grid_scene as GS
    import grid_expand as GE
    env = GS.make_grid()
    # unit checks
    diag = np.array([[i * 0.1, i * 0.1] for i in range(51)])
    stall = np.vstack([np.array([[i * 0.1, i * 0.1] for i in range(20)]), np.tile([2.0, 2.0], (40, 1))])
    back = np.vstack([np.array([[i * 0.1, i * 0.1] for i in range(30)]),
                      np.array([[3.0 - i * 0.05, 3.0 - i * 0.05] for i in range(20)])])
    print("diag  valid2 (no socp):", traj_valid2(diag, env, 0.5, check_socp=False), "(expect True)")
    print("stall valid2 (no socp):", traj_valid2(stall, env, 0.5, check_socp=False), "(expect False)")
    print("back  valid2 (no socp):", traj_valid2(back, env, 0.5, check_socp=False), "(expect False)")
    cov = set()
    m = measure2([diag, stall], env, 0.5, cov)
    print("measure2 sanity:", {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in m.items() if k not in ("n_final",)})
    # demo calibration per gamma
    for g in (0.1, 0.5, 1.0):
        demo = GE.load_demo(g)
        c = calibrate_approach(demo)
        print(f"γ{g}: demo approach pass {c['pass_rate']*100:.1f}% (n={c['n']})  "
              f"net-prog p01={c['prog_p01']:.3f} med={c['prog_med']:.3f}  "
              f"step-inc p95={c['step_inc_p95']:.3f} p99={c['step_inc_p99']:.3f} max={c['step_inc_max']:.3f}  "
              f"suggest DELTA={c['suggested_delta']:.3f} STEP_TOL={c['suggested_step_tol']:.3f}  "
              f"(using DELTA={DELTA_PROG} STEP_TOL={STEP_TOL})")
