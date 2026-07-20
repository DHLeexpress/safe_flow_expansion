"""Core objects of the minimal AFE-style Safe Flow Expansion (user redesign 2026-07-16).

One object identity: the PLANNED action window U_t = (u_{t|t},...,u_{t+9|t}) with its context
c_t = (grid, low5, hist) is the single thing that is (1) sampled from the flow, (2) uncertainty-scored,
(3) FULLY verified (whole predicted DI rollout in-bounds AND SOCP-certified at its gamma) BEFORE
execution, (4) stored in D_n / A_n regardless of label, and (5) replayed for training when positive.
Safety margin m and progress r are stored SEPARATELY; progress is never part of the safety label y.

Pieces here:
  BLRSigma        A_n = I + lam^-1 sum z z^T on the normalized representation supplied by the
                  caller; sigma^2 = z^T A_n^-1 z. AFE2 rebuilds it after each representation
                  update. Rank-1 Sherman-Morrison uses float64; there is no eviction.
  DStore          append-only query store, normalized by control step (B queries share one context).
  verify_plan     the deterministic full verifier on ONE planned window (in-bounds + SOCP + m, r).
  SafeMPPIFallback certified backup controller (the SafeMPPI expert generator, one plan per step).
  build_audit_contexts / run_audit
                  fixed held-out rho_eval (positions x gammas, zero vel, empty hist); UNTILTED plan
                  samples, fully verified -> per-gamma V_hat and V_hat^prog. Never added to buffers.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))          # codex_overnight/
_REV = os.path.dirname(_HERE)                               # rev_expansion/
_WORK = os.path.dirname(_REV)                               # overnight_run_07_06/
sys.path.insert(0, _WORK)                                   # shared grid code
sys.path.insert(0, _REV)                                    # rev_expansion helpers
sys.path.insert(0, _HERE)                                   # local copies ALWAYS win (grid_metrics2!)

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import afe_context as CX
import grid_rollout as GR
import grid_scene as GS


@contextmanager
def isolated_random_state(seed):
    """Seed a diagnostic/evaluation block without perturbing gathering RNG.

    ``torch.random.fork_rng`` restores the CPU generator and every initialized
    CUDA generator.  NumPy and Python's generator need explicit snapshots.
    """

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        random.seed(int(seed))
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


# ------------------------------------------------------------------ uncertainty
class BLRSigma:
    """sigma_n(U,c) from the design matrix on caller-supplied normalized features.

    A_n = I + lam^-1 sum_i z_i z_i^T over every window actually submitted to the full verifier
    (positive or negative). sigma^2 = z^T A_n^-1 z, matching the linear-kernel GP posterior std.
    No cap, eviction, or decimation is applied within one fixed representation.
    """

    def __init__(self, dim=32, lam=1e-2):
        self.dim = int(dim)
        self.lam = float(lam)
        self.A_inv = torch.eye(self.dim, dtype=torch.float64)
        self.n = 0

    @torch.no_grad()
    def sigma(self, Z):
        """Z [B,dim] (normalized, any device/dtype) -> posterior std [B] float32 (cpu)."""
        Zd = Z.detach().to("cpu", torch.float64)
        q = torch.einsum("bi,ij,bj->b", Zd, self.A_inv, Zd).clamp_min(0.0)
        return q.sqrt().to(torch.float32)

    @torch.no_grad()
    def update(self, Z):
        """Rank-1 Sherman-Morrison for each row of Z: A += lam^-1 z z^T."""
        for z in Z.detach().to("cpu", torch.float64):
            Av = self.A_inv @ z
            self.A_inv -= torch.outer(Av, Av) / (self.lam + float(z @ Av))
            self.n += 1

    def state_dict(self):
        return dict(A_inv=self.A_inv.clone(), n=int(self.n), lam=self.lam, dim=self.dim)

    def load_state_dict(self, st):
        self.A_inv = st["A_inv"].clone()
        self.n = int(st["n"])
        self.lam = float(st["lam"])
        self.dim = int(st["dim"])


@torch.no_grad()
def frozen_feat(model, U, gT, lT, hT, s=0.9):
    """Normalized representation feature z = phi_s(U,c)/||.|| -> [B,32]."""
    f = model.phi_s_at(U, gT, lT, hT, s=s)
    return f / f.norm(dim=1, keepdim=True).clamp_min(1e-9)


# ------------------------------------------------------------------ verifier
def verify_plan(state4, U_np, env, gamma, goal_np, n_theta=180):
    """Full deterministic verifier on ONE planned window, called BEFORE execution.

    y = 1  iff the whole predicted DI rollout of the plan is in task space AND SOCP-certified at
    this gamma (alpha_t = (1-gamma)^t).  Progress r = d0 - dH (net approach over the plan) and the
    SOCP face margin m are returned separately and are NOT part of y.
    """
    st = np.asarray(state4, dtype=np.float32)
    seg = GR.window_positions(st, U_np, env.dt)
    d = np.linalg.norm(np.vstack([st[:2][None], seg]) - goal_np[None], axis=1)
    prog = float(d[0] - d[-1])
    d0 = float(d[0])
    if not GM.in_taskspace(seg):
        return dict(y=0, margin=float("nan"), resid=float("nan"), prog=prog, d0=d0,
                    reason="oob", n_socp_solve=0, verifier_seconds=0.0)
    started = time.perf_counter()
    try:
        ok, margin, resid = GM2.window_socp_stats(st, U_np, env, float(gamma), n_theta=n_theta)
    except RuntimeError:
        # Defined-but-conservative behavior on verifier edge cases: count as unsafe, never crash
        # an overnight run. The counter surfaces in probe.jsonl so a systematic issue is visible.
        return dict(y=0, margin=float("nan"), resid=float("nan"), prog=prog, d0=d0,
                    reason="socp_error", n_socp_solve=1,
                    verifier_seconds=time.perf_counter() - started)
    return dict(y=int(bool(ok)), margin=float(margin), resid=float(resid), prog=prog, d0=d0,
                reason=("ok" if ok else "socp_fail"), n_socp_solve=1,
                verifier_seconds=time.perf_counter() - started)


def verify_plan_with_terminal(
    state4,
    U_np,
    env,
    gamma,
    goal_np,
    *,
    reach,
    n_theta=180,
):
    """Keep the full-window training label, but stop verification at goal for execution.

    ``y`` remains the original full-H label consumed by ``D+``. ``exec_y`` may
    additionally accept a plan whose certified prefix reaches the absorbing
    goal set; its unexecuted suffix is never relabeled as safe training data.
    """

    controls = np.asarray(U_np, dtype=np.float32)
    full = verify_plan(state4, controls, env, gamma, goal_np, n_theta=n_theta)
    seg = GR.window_positions(np.asarray(state4, dtype=np.float32), controls, env.dt)
    distances = np.linalg.norm(seg - np.asarray(goal_np)[None], axis=1)
    hits = np.flatnonzero(distances < float(reach))
    tau = int(hits[0] + 1) if hits.size else None  # number of controls through first hit
    terminal_prog = (
        float(full["d0"] - distances[tau - 1]) if tau is not None else None
    )

    full.update(
        exec_y=int(full["y"]),
        exec_prog=(terminal_prog if terminal_prog is not None else float(full["prog"])),
        exec_margin=float(full["margin"]),
        terminal_prog=terminal_prog,
        terminal_resid=None,
        terminal_hit=bool(tau is not None),
        terminal_tau=tau,
        terminal_rescue=False,
        terminal_reason=None,
        terminal_reverify=False,
    )
    if full["y"] or tau is None or full["reason"] == "socp_error":
        return full

    prefix = verify_plan(
        state4,
        controls[:tau],
        env,
        gamma,
        goal_np,
        n_theta=n_theta,
    )
    full.update(
        exec_y=int(prefix["y"]),
        exec_prog=float(prefix["prog"]),
        exec_margin=float(prefix["margin"]),
        terminal_prog=float(prefix["prog"]),
        terminal_resid=float(prefix["resid"]),
        terminal_rescue=bool(prefix["y"]),
        terminal_reason=prefix["reason"],
        terminal_reverify=True,
        n_socp_solve=int(full["n_socp_solve"] + prefix["n_socp_solve"]),
        verifier_seconds=float(full["verifier_seconds"] + prefix["verifier_seconds"]),
    )
    return full


def prog_bar_ok(prog, d0, delta=GM2.DELTA_PROG):
    """The calibrated net-progress bar used by V_hat^prog: r >= min(delta, 0.5 d0)."""
    return bool(prog >= min(delta, 0.5 * d0))


# ------------------------------------------------------------------ query store
class DStore:
    """Append-only D_n = {(c_i, U_i, y_i, m_i, r_i)} over fully-verified planned windows.

    Normalized by control step: the B queries of one step share one stored context. The H_P grid
    channel and action history are kept in float32, exactly matching the tensors embedded during
    acquisition. Positives never leave the archive. Replay is cumulative by
    default and may opt into an explicit recent-round window.
    """

    def __init__(self, *, conditioning_schema=CX.LOW5_SCHEMA, condition_dim=5):
        self.conditioning_schema = str(conditioning_schema)
        self.condition_dim = int(condition_dim)
        expected_dim = CX.SCHEMA_DIMS.get(self.conditioning_schema)
        if expected_dim is None or self.condition_dim != expected_dim:
            raise ValueError(
                "query-store conditioning contract is invalid: "
                f"{(self.conditioning_schema, self.condition_dim)}"
            )
        # per-step context tables
        self.ctx_state = []      # np float32 [4]
        self.ctx_hp = []         # np float32 [1,32,32]  (grid channel 2)
        self.ctx_low5 = []       # legacy wire name; np float32 [condition_dim]
        self.ctx_hist = []       # np float32 [K_HIST,2]
        self.ctx_meta = []       # (round, episode, t)
        # per-query tables
        self.q_sid = []          # context row of each query
        self.q_U = []            # np float32 [H,2]
        self.q_y = []
        self.q_margin = []
        self.q_resid = []
        self.q_prog = []
        self.q_d0 = []
        self.q_sigma = []
        self.q_gamma = []
        self.q_round = []
        self.q_exec = []         # 1 if this plan's first action was executed
        self.q_exec_y = []       # execution admissibility (full-H or certified terminal prefix)
        self.q_exec_prog = []
        self.q_exec_margin = []
        self.q_nvp_negative = [] # 1 only for queries at a terminal NVP context
        self.q_terminal_hit = []
        self.q_terminal_rescue = []
        self.q_terminal_tau = []
        self.q_terminal_prog = []
        self.q_terminal_resid = []
        self.q_terminal_reason = []
        self.q_terminal_reverify = []
        self.q_seg = []          # np float16 [H,2] planned positions (viz)
        self.pos_ids = []        # indices into the query tables with y==1

    def add_step_ctx(self, state4, grid_np, low5_np, hist_np, meta):
        condition = np.asarray(low5_np, np.float32)
        if condition.shape != (self.condition_dim,) or not np.isfinite(condition).all():
            raise RuntimeError(
                f"stored {self.conditioning_schema} condition has shape {condition.shape}; "
                f"expected {(self.condition_dim,)}"
            )
        self.ctx_state.append(np.asarray(state4, np.float32).copy())
        self.ctx_hp.append(np.asarray(grid_np[2:3], np.float32).copy())
        self.ctx_low5.append(condition.copy())
        self.ctx_hist.append(np.asarray(hist_np, np.float32).copy())
        self.ctx_meta.append(tuple(meta))
        return len(self.ctx_state) - 1

    def add_query(self, sid, U_np, v, sigma, gamma, round_i, seg):
        qid = len(self.q_sid)
        self.q_sid.append(int(sid))
        self.q_U.append(np.asarray(U_np, np.float32).copy())
        self.q_y.append(int(v["y"]))
        self.q_margin.append(float(v["margin"]))
        self.q_resid.append(float(v["resid"]))
        self.q_prog.append(float(v["prog"]))
        self.q_d0.append(float(v["d0"]))
        self.q_sigma.append(float(sigma))
        self.q_gamma.append(float(gamma))
        self.q_round.append(int(round_i))
        self.q_exec.append(0)
        self.q_exec_y.append(int(v.get("exec_y", v["y"])))
        self.q_exec_prog.append(float(v.get("exec_prog", v["prog"])))
        self.q_exec_margin.append(float(v.get("exec_margin", v["margin"])))
        self.q_nvp_negative.append(0)
        self.q_terminal_hit.append(int(bool(v.get("terminal_hit", False))))
        self.q_terminal_rescue.append(int(bool(v.get("terminal_rescue", False))))
        self.q_terminal_tau.append(int(v.get("terminal_tau") or -1))
        self.q_terminal_prog.append(
            float(v["terminal_prog"]) if v.get("terminal_prog") is not None else float("nan")
        )
        self.q_terminal_resid.append(
            float(v["terminal_resid"]) if v.get("terminal_resid") is not None else float("nan")
        )
        self.q_terminal_reason.append(v.get("terminal_reason"))
        self.q_terminal_reverify.append(int(bool(v.get("terminal_reverify", False))))
        self.q_seg.append(np.asarray(seg, np.float16).copy())
        if v["y"]:
            self.pos_ids.append(qid)
        return qid

    def mark_executed(self, qid):
        full_witness = bool(self.q_y[qid])
        prefix_witness = (
            bool(self.q_exec_y[qid])
            and bool(self.q_terminal_rescue[qid])
            and self.q_terminal_tau[qid] >= 1
            and self.q_terminal_reason[qid] == "ok"
            and np.isfinite(self.q_exec_margin[qid])
            and self.q_exec_margin[qid] > 0.0
        )
        if not (full_witness or prefix_witness):
            raise RuntimeError("executed query has no persisted full-H or terminal-prefix certificate")
        self.q_exec[qid] = 1

    def mark_nvp_negative(self, query_ids):
        """Persist the queried plans that jointly produced a terminal NVP.

        The marker is deliberately independent of the full-window verifier label:
        a plan may remain in ``D+`` because it is SOCP-positive while also carrying
        evidence against closed-loop viability at the terminal NVP context.
        """

        for query_id in query_ids:
            query_id = int(query_id)
            if query_id < 0:
                continue
            if query_id >= len(self.q_nvp_negative):
                raise IndexError(f"NVP-negative query id is out of range: {query_id}")
            if self.q_exec[query_id]:
                raise RuntimeError("an executed query cannot be marked as terminal-NVP negative")
            self.q_nvp_negative[query_id] = 1

    def validate_execution_witnesses(self):
        for qid, executed in enumerate(self.q_exec):
            if not executed:
                continue
            full_witness = bool(self.q_y[qid])
            prefix_witness = (
                bool(self.q_exec_y[qid])
                and bool(self.q_terminal_rescue[qid])
                and self.q_terminal_tau[qid] >= 1
                and self.q_terminal_reason[qid] == "ok"
                and np.isfinite(self.q_exec_margin[qid])
                and self.q_exec_margin[qid] > 0.0
            )
            if not (full_witness or prefix_witness):
                raise RuntimeError(f"query {qid} has no persisted execution certificate")

    def __len__(self):
        return len(self.q_sid)

    def n_pos(self):
        return len(self.pos_ids)

    def grid3_of(self, sids):
        """Reconstruct [B,3,32,32] float32 grids (channels 0/1 zero; the model reads only ch2)."""
        hp = torch.stack([torch.from_numpy(self.ctx_hp[s].astype(np.float32)) for s in sids])
        B = hp.shape[0]
        g = torch.zeros(B, 3, hp.shape[2], hp.shape[3], dtype=torch.float32)
        g[:, 2:3] = hp
        return g

    def positive_ids(self, *, round_i=None, replay_window=None):
        """Return the positive replay population, optionally limited to recent rounds."""
        if replay_window is None:
            return list(self.pos_ids)
        if round_i is None:
            raise ValueError("windowed positive replay requires the current round")
        replay_window = int(replay_window)
        if replay_window < 1:
            raise ValueError("positive replay window must be at least one round")
        first_round = max(1, int(round_i) - replay_window + 1)
        return [
            query_id for query_id in self.pos_ids
            if first_round <= int(self.q_round[query_id]) <= int(round_i)
        ]

    def positive_replay_hierarchy(self, *, eligible_ids=None):
        """Index positives by round, gamma, replica, context for balanced replay."""

        population = self.pos_ids if eligible_ids is None else eligible_ids
        if not population:
            return {}
        hierarchy = {}
        for query_id in population:
            query_id = int(query_id)
            context_id = int(self.q_sid[query_id])
            if not 0 <= context_id < len(self.ctx_meta):
                raise IndexError(f"positive query has invalid context id: {context_id}")
            meta = self.ctx_meta[context_id]
            if len(meta) != 3:
                raise RuntimeError(f"stored context metadata is malformed: {meta}")
            query_round = int(self.q_round[query_id])
            if int(meta[0]) != query_round:
                raise RuntimeError(
                    f"query/context round mismatch: query={query_round}, context={meta[0]}"
                )
            gamma = round(float(self.q_gamma[query_id]), 8)
            replica = int(meta[1])
            hierarchy.setdefault(query_round, {}).setdefault(gamma, {}).setdefault(
                replica, {}
            ).setdefault(context_id, []).append(query_id)
        return hierarchy

    def positive_hierarchy_equal_mass(self, declared_gammas, *, eligible_ids=None):
        """Return gamma/episode/context/query-equal mass on positive support.

        Episode instances are keyed by ``(round, stored episode id)`` because
        episode ids repeat across expansion rounds.  Empty groups cannot carry
        positive CFM mass and are reported rather than invented.
        """

        population = [
            int(query_id)
            for query_id in (self.pos_ids if eligible_ids is None else eligible_ids)
        ]
        if not population:
            return {}, {
                "declared_gammas": [round(float(value), 8) for value in declared_gammas],
                "active_gammas": [],
                "missing_declared_gammas": [
                    round(float(value), 8) for value in declared_gammas
                ],
                "raw_mass_sum": 0.0,
                "episode_mass_max_error": None,
                "context_mass_max_error": None,
                "within_context_query_mass_max_error": None,
            }
        if len(set(population)) != len(population):
            raise ValueError("positive replay mass population contains duplicate query ids")
        storage_map = CX.declared_gamma_storage_map(declared_gammas)
        declared = sorted(storage_map.values())
        groups = {}
        for query_id in population:
            context_id = int(self.q_sid[query_id])
            if not 0 <= context_id < len(self.ctx_meta):
                raise IndexError(f"positive query has invalid context id: {context_id}")
            meta = self.ctx_meta[context_id]
            if len(meta) != 3:
                raise RuntimeError(f"stored context metadata is malformed: {meta}")
            query_round = int(self.q_round[query_id])
            if int(meta[0]) != query_round:
                raise RuntimeError(
                    f"query/context round mismatch: query={query_round}, context={meta[0]}"
                )
            gamma = CX.canonical_declared_gamma(
                self.q_gamma[query_id], storage_map
            )
            episode_instance = (query_round, int(meta[1]))
            groups.setdefault(gamma, {}).setdefault(episode_instance, {}).setdefault(
                context_id, []
            ).append(query_id)

        mass = {}
        gamma_mass = 1.0 / len(groups)
        for episode_map in groups.values():
            episode_mass = gamma_mass / len(episode_map)
            for context_map in episode_map.values():
                context_mass = episode_mass / len(context_map)
                for query_ids in context_map.values():
                    query_mass = context_mass / len(query_ids)
                    for query_id in query_ids:
                        mass[query_id] = float(query_mass)
        if set(mass) != set(population):
            raise RuntimeError("hierarchical replay mass does not cover the population")
        total = float(sum(mass.values()))
        if not np.isclose(total, 1.0, rtol=0.0, atol=1.0e-10):
            raise RuntimeError(f"hierarchical replay mass sums to {total}, not one")
        active = sorted(groups)
        episode_mass_errors = []
        context_mass_errors = []
        within_context_query_mass_errors = []
        for gamma, episode_map in groups.items():
            target_episode_mass = gamma_mass / len(episode_map)
            for context_map in episode_map.values():
                observed_episode_mass = sum(
                    mass[query_id]
                    for query_ids in context_map.values()
                    for query_id in query_ids
                )
                episode_mass_errors.append(
                    abs(observed_episode_mass - target_episode_mass)
                )
                target_context_mass = target_episode_mass / len(context_map)
                for query_ids in context_map.values():
                    observed_context_mass = sum(mass[query_id] for query_id in query_ids)
                    context_mass_errors.append(
                        abs(observed_context_mass - target_context_mass)
                    )
                    target_query_mass = target_context_mass / len(query_ids)
                    within_context_query_mass_errors.extend(
                        abs(mass[query_id] - target_query_mass)
                        for query_id in query_ids
                    )
        return mass, {
            "declared_gammas": declared,
            "active_gammas": active,
            "missing_declared_gammas": [
                gamma for gamma in declared if gamma not in groups
            ],
            "positive_episode_instances": int(sum(len(value) for value in groups.values())),
            "positive_contexts": int(sum(
                len(context_map)
                for episode_map in groups.values()
                for context_map in episode_map.values()
            )),
            "mass_by_gamma": {
                str(gamma): float(sum(
                    mass[query_id]
                    for context_map in groups[gamma].values()
                    for query_ids in context_map.values()
                    for query_id in query_ids
                ))
                for gamma in active
            },
            "raw_mass_sum": total,
            "episode_mass_max_error": float(max(episode_mass_errors, default=0.0)),
            "context_mass_max_error": float(max(context_mass_errors, default=0.0)),
            "within_context_query_mass_max_error": float(max(
                within_context_query_mass_errors, default=0.0
            )),
        }

    def sample_positive_ids(
        self,
        nb,
        rng,
        *,
        eligible_ids=None,
        sampling="query_uniform",
        hierarchy=None,
    ):
        """Draw positive query ids using the declared replay population measure."""

        population = self.pos_ids if eligible_ids is None else eligible_ids
        if not population:
            return None
        if sampling == "query_uniform":
            return [population[i] for i in rng.integers(0, len(population), nb)]
        if sampling != "round_gamma_replica_context":
            raise ValueError(f"unknown positive replay sampling rule: {sampling}")
        hierarchy = (
            self.positive_replay_hierarchy(eligible_ids=population)
            if hierarchy is None
            else hierarchy
        )
        if not hierarchy:
            return None

        def choose(values):
            ordered = sorted(values)
            return ordered[int(rng.integers(0, len(ordered)))]

        ids = []
        for _ in range(int(nb)):
            query_round = choose(hierarchy)
            gamma = choose(hierarchy[query_round])
            replica = choose(hierarchy[query_round][gamma])
            context_id = choose(hierarchy[query_round][gamma][replica])
            queries = hierarchy[query_round][gamma][replica][context_id]
            ids.append(int(queries[int(rng.integers(0, len(queries)))]))
        return ids

    def positive_epoch_ids(
        self,
        rng,
        *,
        eligible_ids=None,
        sampling="query_uniform",
    ):
        """Order every eligible positive exactly once for one replay epoch.

        The hierarchical option interleaves round/gamma/replica/context cells
        while they remain nonempty.  It balances *order*, not total cell mass:
        an exact epoch necessarily retains every query from longer cells.
        """

        population = list(self.pos_ids if eligible_ids is None else eligible_ids)
        if not population:
            return []
        if len({int(query_id) for query_id in population}) != len(population):
            raise ValueError("positive replay epoch population contains duplicate query ids")
        if sampling == "query_uniform":
            order = [population[int(index)] for index in rng.permutation(len(population))]
        elif sampling == "round_gamma_replica_context":
            hierarchy = self.positive_replay_hierarchy(eligible_ids=population)

            def shuffled(values):
                ordered = sorted(values)
                return [ordered[int(index)] for index in rng.permutation(len(ordered))]

            def interleave(generators):
                active = [generators[key] for key in shuffled(generators)]
                while active:
                    next_active = []
                    for generator in active:
                        try:
                            yield next(generator)
                            next_active.append(generator)
                        except StopIteration:
                            pass
                    active = next_active

            def queries(query_ids):
                yield from shuffled(query_ids)

            def contexts(context_map):
                yield from interleave({
                    context_id: queries(query_ids)
                    for context_id, query_ids in context_map.items()
                })

            def replicas(replica_map):
                yield from interleave({
                    replica: contexts(context_map)
                    for replica, context_map in replica_map.items()
                })

            def gammas(gamma_map):
                yield from interleave({
                    gamma: replicas(replica_map)
                    for gamma, replica_map in gamma_map.items()
                })

            order = list(interleave({
                query_round: gammas(gamma_map)
                for query_round, gamma_map in hierarchy.items()
            }))
        else:
            raise ValueError(f"unknown positive replay sampling rule: {sampling}")

        order = [int(query_id) for query_id in order]
        if len(order) != len(population) or set(order) != {int(q) for q in population}:
            raise RuntimeError("positive replay epoch did not cover its population exactly once")
        return order

    def positive_batch(self, query_ids):
        """Reconstruct one explicit positive-query batch without resampling."""

        ids = [int(query_id) for query_id in query_ids]
        if not ids:
            raise ValueError("positive replay batch cannot be empty")
        if any(query_id < 0 or query_id >= len(self.q_sid) for query_id in ids):
            raise IndexError("positive replay query id is out of range")
        sids = [self.q_sid[q] for q in ids]
        G = self.grid3_of(sids)
        L = torch.stack([torch.from_numpy(self.ctx_low5[s]) for s in sids])
        H = torch.stack([torch.from_numpy(self.ctx_hist[s].astype(np.float32)) for s in sids])
        U = torch.stack([torch.from_numpy(self.q_U[q]) for q in ids])
        return G, L, H, U, ids

    def sample_pos(
        self,
        nb,
        rng,
        *,
        eligible_ids=None,
        sampling="query_uniform",
        hierarchy=None,
    ):
        """Reconstruct a batch drawn from the declared positive replay measure."""

        ids = self.sample_positive_ids(
            nb,
            rng,
            eligible_ids=eligible_ids,
            sampling=sampling,
            hierarchy=hierarchy,
        )
        if ids is None:
            return None
        return self.positive_batch(ids)

    def round_slice(self, round_i):
        """Query ids belonging to one round (for per-round stats/viz)."""
        return [q for q in range(len(self.q_sid)) if self.q_round[q] == round_i]

    def save(self, path):
        self.validate_execution_witnesses()
        torch.save(dict(
            conditioning_schema=self.conditioning_schema,
            condition_dim=self.condition_dim,
            ctx_state=np.stack(self.ctx_state) if self.ctx_state else np.zeros((0, 4), np.float32),
            ctx_hp=np.stack(self.ctx_hp) if self.ctx_hp else np.zeros((0, 1, 32, 32), np.float32),
            ctx_low5=(
                np.stack(self.ctx_low5)
                if self.ctx_low5
                else np.zeros((0, self.condition_dim), np.float32)
            ),
            ctx_hist=np.stack(self.ctx_hist) if self.ctx_hist else np.zeros((0, GF.K_HIST, 2), np.float32),
            ctx_meta=np.asarray(self.ctx_meta, np.int32) if self.ctx_meta else np.zeros((0, 3), np.int32),
            q_sid=np.asarray(self.q_sid, np.int64), q_U=np.stack(self.q_U) if self.q_U else np.zeros((0, GF.H_PRED, 2), np.float32),
            q_y=np.asarray(self.q_y, np.int8), q_margin=np.asarray(self.q_margin, np.float32),
            q_resid=np.asarray(self.q_resid, np.float32), q_prog=np.asarray(self.q_prog, np.float32),
            q_d0=np.asarray(self.q_d0, np.float32), q_sigma=np.asarray(self.q_sigma, np.float32),
            q_gamma=np.asarray(self.q_gamma, np.float32), q_round=np.asarray(self.q_round, np.int32),
            q_exec=np.asarray(self.q_exec, np.int8),
            q_exec_y=np.asarray(self.q_exec_y, np.int8),
            q_exec_prog=np.asarray(self.q_exec_prog, np.float32),
            q_exec_margin=np.asarray(self.q_exec_margin, np.float32),
            q_nvp_negative=np.asarray(self.q_nvp_negative, np.int8),
            q_terminal_hit=np.asarray(self.q_terminal_hit, np.int8),
            q_terminal_rescue=np.asarray(self.q_terminal_rescue, np.int8),
            q_terminal_tau=np.asarray(self.q_terminal_tau, np.int16),
            q_terminal_prog=np.asarray(self.q_terminal_prog, np.float32),
            q_terminal_resid=np.asarray(self.q_terminal_resid, np.float32),
            q_terminal_reason=list(self.q_terminal_reason),
            q_terminal_reverify=np.asarray(self.q_terminal_reverify, np.int8),
            q_seg=np.stack(self.q_seg) if self.q_seg else np.zeros((0, GF.H_PRED, 2), np.float16),
        ), path)


# ------------------------------------------------------------------ certified fallback
class SafeMPPIFallback:
    """Certified SafeMPPI backup: when no drawn plan verifies safe, plan ONE action with the same
    SafeMPPI controller that generated the pretraining demos (mode1 config, walls included via
    planner_obstacles).  Fresh adapter per episode (its internal warm start is trajectory-local)."""

    def __init__(self, env):
        self.cfg = GS.mode1_config()
        self.goal_t = env.goal.detach().cpu().float()
        self.obs_plan = GS.planner_obstacles(env)
        self.ad = None

    def reset(self):
        from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
        self.ad = SafeMPPIAdapter(**self.cfg)

    def plan(self, state4, gamma, seed):
        a, _ = self.ad.plan(torch.tensor(np.asarray(state4, np.float32)), self.goal_t,
                            self.obs_plan, gamma=float(gamma), seed=int(seed))
        return a.detach().cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------ audit (rho_eval)
def build_audit_contexts(
    env,
    gammas,
    n_pos=12,
    seed=20260716,
    min_clear=0.05,
    v_adverse=0.65,
    conditioning_schema=CX.LOW5_SCHEMA,
):
    """Fixed held-out rho_eval: n_pos free-space positions (position 0 = the episode start), each in
    TWO velocity conditions -- rest (v=0) and ADVERSE (moving at v_adverse toward the nearest
    obstacle: where window certification is actually hard; rest-only audits sit at ~99% ceiling) --
    crossed with every gamma, empty history.  Fixed across the whole run and across arms/seeds;
    audit samples are NEVER added to D_n or A_n."""
    rng = np.random.default_rng(seed)
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    x0 = env.x0.detach().cpu().numpy()
    pos = [np.array([x0[0], x0[1]], np.float32)]
    while len(pos) < n_pos:
        p = rng.uniform(0.1, 4.9, 2).astype(np.float32)
        clr = (np.linalg.norm(p[None] - obs[:, :2], axis=1) - obs[:, 2] - rr).min()
        if clr > min_clear:
            pos.append(p)
    goal_np = env.goal.detach().cpu().numpy()
    ctxs = []
    for pi, p in enumerate(pos):
        j = int(np.argmin(np.linalg.norm(obs[:, :2] - p[None], axis=1) - obs[:, 2]))
        d = obs[j, :2] - p
        d = d / (np.linalg.norm(d) + 1e-9)
        for vk, v in (("rest", np.zeros(2, np.float32)),
                      ("adverse", (v_adverse * d).astype(np.float32))):
            st = np.array([p[0], p[1], v[0], v[1]], np.float32)
            for g in gammas:
                record = CX.build_context(
                    st, goal_np, float(g), [], env, conditioning_schema
                )
                ctxs.append(dict(
                    pos_id=pi,
                    vel=vk,
                    state=st,
                    gamma=float(g),
                    grid=np.asarray(record.grid, dtype=np.float32),
                    low5=np.asarray(record.low5, dtype=np.float32),
                    hist=np.asarray(record.hist, dtype=np.float32),
                    conditioning_schema=conditioning_schema,
                ))
    return ctxs


@torch.no_grad()
def run_audit(policy, ctxs, env, goal_np, device, n_plans=4, nfe=8, n_theta=180, seed=0):
    """UNTILTED audit: for each rho_eval context sample n_plans plans from the CURRENT flow at
    temp=1 and fully verify each.  Returns per-gamma V_hat (mean y), V_hat^prog (y AND calibrated
    net-progress bar), pooled numbers, and the per-context matrix for viz."""
    per = []
    with isolated_random_state(seed):
        for c in ctxs:
            gT = torch.tensor(c["grid"], device=device)
            lT = torch.tensor(c["low5"], device=device)
            hT = torch.tensor(c["hist"], device=device)
            U = policy.sample_window(gT, lT, hT, n=n_plans, temp=1.0, nfe=nfe)
            ys, ps = [], []
            for j in range(U.shape[0]):
                v = verify_plan(c["state"], U[j].detach().cpu().numpy(), env, c["gamma"], goal_np,
                                n_theta=n_theta)
                ys.append(v["y"])
                ps.append(int(v["y"] and prog_bar_ok(v["prog"], v["d0"])))
            per.append(dict(pos_id=c["pos_id"], vel=c.get("vel", "rest"), gamma=c["gamma"],
                            V=float(np.mean(ys)), Vprog=float(np.mean(ps)),
                            k=int(np.sum(ys)), n=int(len(ys)),
                            k_prog=int(np.sum(ps)), n_prog=int(len(ps))))
    out = dict(per=per)
    gammas = sorted({p["gamma"] for p in per})
    out["V_gamma"] = {str(g): float(np.mean([p["V"] for p in per if p["gamma"] == g])) for g in gammas}
    out["Vprog_gamma"] = {str(g): float(np.mean([p["Vprog"] for p in per if p["gamma"] == g]))
                          for g in gammas}
    out["counts_gamma"] = {
        str(g): {
            "k": int(sum(p["k"] for p in per if p["gamma"] == g)),
            "n": int(sum(p["n"] for p in per if p["gamma"] == g)),
        }
        for g in gammas
    }
    out["V"] = float(np.mean([p["V"] for p in per]))
    out["Vprog"] = float(np.mean([p["Vprog"] for p in per]))
    # Explicit paper-facing names.  Keep the legacy keys above for historical plots.
    out["V_safe"] = out["V"]
    out["V_full"] = out["Vprog"]
    out["V_safe_gamma"] = out["V_gamma"]
    out["V_full_gamma"] = out["Vprog_gamma"]
    out["counts_gamma_full"] = {
        str(g): {
            "k": int(sum(p["k_prog"] for p in per if p["gamma"] == g)),
            "n": int(sum(p["n_prog"] for p in per if p["gamma"] == g)),
        }
        for g in gammas
    }
    for vk in ("rest", "adverse"):
        sub = [p for p in per if p.get("vel", "rest") == vk]
        if sub:
            out[f"V_{vk}"] = float(np.mean([p["V"] for p in sub]))
            out[f"V_gamma_{vk}"] = {str(g): float(np.mean([p["V"] for p in sub if p["gamma"] == g]))
                                    for g in gammas}
            out[f"counts_gamma_{vk}"] = {
                str(g): {
                    "k": int(sum(p["k"] for p in sub if p["gamma"] == g)),
                    "n": int(sum(p["n"] for p in sub if p["gamma"] == g)),
                }
                for g in gammas
            }
    return out
