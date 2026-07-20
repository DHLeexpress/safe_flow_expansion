"""Stage D — coverage (252 staircases) + window verifier for the 5x5 grid.

Coverage denominator = C(10,5) = 252 monotone right/up lattice paths from (0,0)->(5,5): 5 R + 5 U unit
moves on the 6x6 vertex grid (a design "mode"). A deployed trajectory is discretized to its round(x),round(y)
crossing sequence; if it is strictly monotone (never crosses a boundary backward) and completes all 5 R + 5 U
it realizes ONE of the 252 staircases. A staircase is COVERED only when a VALID trajectory realizes it.

Validity of a trajectory = AND of the three window criteria along the whole path:
  (1) LOCAL   : the SOCP verifier certifies every sliding 10-window (verifier_polytope.certify_trajectory);
  (2) GLOBAL  : every point stays in the task space [0,5]^2 (off-grid -> invalid);
  (3) GLOBAL2 : the path is monotone right/up (staircase_id is not None) — moves toward the goal;
plus it must reach the goal. measure(paths,...) -> (validity, coverage, avg_steps).
"""
from __future__ import annotations

import itertools

import numpy as np

import _paths  # noqa: F401
import verifier_polytope as VP

GRID_M = 5.0
EPS_TASK = 0.12          # task-space tolerance (matches is_success margin)
REACH = 0.45


def enumerate_staircases():
    """All 252 = C(10,5) monotone R/U words of length 10 with 5 R and 5 U."""
    S = set()
    for rpos in itertools.combinations(range(10), 5):
        w = ["U"] * 10
        for i in rpos:
            w[i] = "R"
        S.add("".join(w))
    return S


STAIRCASES = enumerate_staircases()      # 252
N_STAIR = len(STAIRCASES)                # 252


def neighbors(word):
    """1-swap adjacent staircases (swap a differing R,U pair) — the reachability frontier of `word`."""
    out = set()
    for i in range(len(word) - 1):
        if word[i] != word[i + 1]:
            out.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])
    return out


def staircase_id(path, back_tol=0.5, reach=REACH):
    """Discretize a path to its right/up crossing word (boundaries at integers 1..5; streets sit at
    half-integers so weaving stays mid-cell). Commit each boundary monotonically; tolerate sub-cell
    weaving up to `back_tol`; a backward crossing beyond that -> non-monotone (None). On reaching the goal
    the remaining moves complete to (5,5). Returns the R/U string (one of the 252) or None."""
    p = np.asarray(path, dtype=float)
    cx = cy = 0
    seq = []
    reached = False
    for (x, y) in p:
        while cx < 5 and x >= cx + 1 - 1e-6:
            cx += 1; seq.append("R")
        while cy < 5 and y >= cy + 1 - 1e-6:
            cy += 1; seq.append("U")
        if x < cx - back_tol or y < cy - back_tol:      # retreated a boundary backward -> non-monotone
            return None
        if (x - GRID_M) ** 2 + (y - GRID_M) ** 2 < reach * reach:
            reached = True
            break
    if not reached:
        return None
    while cx < 5:                                        # reaching (5,5) completes the last moves
        cx += 1; seq.append("R")
    while cy < 5:
        cy += 1; seq.append("U")
    return "".join(seq) if len(seq) == 10 else None


def in_taskspace(path, eps=EPS_TASK):
    p = np.asarray(path, dtype=float)
    return bool((p >= -eps).all() and (p <= GRID_M + eps).all())


def reaches_goal(path, goal, reach=REACH):
    return bool(np.linalg.norm(np.asarray(path, float) - np.asarray(goal, float), axis=1).min() < reach)


def socp_ok(path, env, gamma, R=2.5, n_theta=180, stride=2, H_win=10):
    """LOCAL validity: every sliding 10-window is SOCP-certified (verifier_polytope)."""
    obs = env.obstacles.detach().cpu().numpy()
    return bool(VP.certify_trajectory(np.asarray(path, float), obs, float(env.r_robot), float(gamma),
                                      H_win=H_win, stride=stride, R=R, n_theta=n_theta))


def is_valid_traj(path, env, gamma, check_socp=True):
    """Binary validity = reach ∧ taskspace ∧ monotone-staircase ∧ (SOCP)."""
    goal = env.goal.detach().cpu().numpy()
    if not reaches_goal(path, goal):
        return False
    if not in_taskspace(path):
        return False
    if staircase_id(path) is None:
        return False
    if check_socp and not socp_ok(path, env, gamma):
        return False
    return True


def measure(paths, env, gamma, covered=None, check_socp=True):
    """paths: list of executed [n+1,2]. Returns (validity, coverage, avg_steps, covered).
    `covered` (set of staircase ids) is updated in place & cumulative across calls."""
    if covered is None:
        covered = set()
    n_valid = 0
    steps = []
    for path in paths:
        if is_valid_traj(path, env, gamma, check_socp=check_socp):
            n_valid += 1
            sid = staircase_id(path)
            if sid is not None:
                covered.add(sid)
            steps.append(len(path))
    validity = n_valid / max(len(paths), 1)
    coverage = len(covered) / N_STAIR
    avg_steps = float(np.mean(steps)) if steps else 0.0
    return validity, coverage, avg_steps, covered


if __name__ == "__main__":
    print("n staircases =", N_STAIR, "(expect 252)")
    # a hand staircase along the anti-diagonal
    good = np.array([[i * 0.1, i * 0.1] for i in range(51)])       # (0,0)->(5,5) monotone
    print("diag staircase_id:", staircase_id(good), "| valid taskspace", in_taskspace(good))
    back = np.array([[0, 0], [1, 0], [0.4, 0], [2, 1]])            # goes backward in x
    print("backward -> id (expect None):", staircase_id(back))
    off = np.array([[0, 0], [1, -0.5], [5, 5]])                    # leaves task space
    print("off-grid in_taskspace (expect False):", in_taskspace(off))
