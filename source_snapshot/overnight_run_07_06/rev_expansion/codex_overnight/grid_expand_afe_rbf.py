"""Single-arm RBF Safe Flow Expansion with synchronous parallel rollouts.

This is a task-specific AFE adaptation, not a claim that the main AFE theorem
requires an RBF kernel.  It follows the peptide experiment's RBF choices while
making the control-specific memory semantics explicit:

* exact RBF-GP on at most 512 recent full-H positives;
* append-only D+ with cumulative replay by default and an opt-in round window;
* multiple closed-loop replicas gathered synchronously; the GP is frozen for
  the whole round, so replicas do not depend on an arbitrary execution order;
* B-budget sequential acquisition: only already-selected pending locations,
  never the unqueried remainder of K, condition the next posterior variance;
* fixed pretrained-only beta by default, with an opt-in round-local ESS target;
* one AFE update arm only (batch 128, lr 1e-4, 250 steps, no proximal term);
* deterministic full verifier before execution and expert-free NVP termination.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
for _path in (_WORK, _REV, _HERE):
    sys.path.insert(0, _path)

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_rollout as GR
import grid_hp_expt as HP
import grid_expand_hardtail as HT
from di_grid_viz import di_step

import afe_core as AC
import afe_context as CX
import afe_adaptive as AD
import afe2_calibration as BC
import afe_rbf_core as RC
import afe_route_metrics as RM
import afe_execution as EX
import afe_signed_update as SU
import afe_demo_support as DS
import grid_expand_afe2 as AFE2
from afe2_scene_profiles import (
    SCENE_PROFILES,
    assert_scene_snapshot,
    build_scene,
    get_scene_profile,
    scene_snapshot,
)


@dataclass
class AFERBFConfig(AFE2.AFE2Config):
    protocol_profile: str = "v1"
    arm: str = "afe"
    replicas: int = 2
    gp_cap: int = 512
    gp_lam: float = 1.0e-2
    verifier_workers: int = 16
    lengthscale_samples: int = 50
    acquisition_mode: str = "sequential"
    adaptive_ess_target: float | None = None
    adaptive_beta_contexts_per_gamma: int | None = None
    adaptive_beta_equalize_gammas: bool = False
    replay_window: int | None = None
    replay_sampling: str = "query_uniform"
    replay_update_mode: str = "fixed_steps_with_replacement"
    replay_loss_weighting: str = "query_uniform"
    gp_replay_window: int = 1
    gp_replay_sampling: str = "round_gamma"
    lengthscale_multiplier: float = 1.0
    negative_alpha: float = 0.0
    execution_rule: str = "legacy_max_horizon_progress"
    training_probes: bool = True
    calibration_replicas: int | None = None
    calibration_control_steps: int | None = None
    sweep_compact_artifacts: bool = False
    compact_checkpoint_every: int = 10
    route_metric_steps: int = 0
    route_ambiguity_band: float = RM.DEFAULT_AMBIGUITY_BAND
    rbf_offline_sweep: bool = False
    nvp_audit_all_k: bool = False
    optimizer_steps_per_round: int = 0
    demo_frac: float = 0.0


def configure_policy_trainability(policy, freeze_visual_encoder: bool) -> dict:
    """Freeze exactly the visual grid encoder when requested."""

    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    if freeze_visual_encoder:
        for parameter in policy.enc_grid.parameters():
            parameter.requires_grad_(False)
    frozen = {
        name for name, parameter in policy.named_parameters()
        if not parameter.requires_grad
    }
    expected = {
        f"enc_grid.{name}" for name, _ in policy.enc_grid.named_parameters()
    } if freeze_visual_encoder else set()
    if frozen != expected:
        raise RuntimeError(
            f"unexpected frozen parameters: got {sorted(frozen)}, expected {sorted(expected)}"
        )
    trainable = [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    if not any(name.startswith("trunk.") for name in trainable):
        raise RuntimeError("flow trunk is not trainable")
    if not any(name.startswith("head.") for name in trainable):
        raise RuntimeError("flow output head is not trainable")
    return {"frozen": sorted(frozen), "trainable": trainable}


def _episode(state, gamma, replica, episode_id, env, cfg):
    obstacles = env.obstacles.detach().cpu().numpy()
    clearance = float(
        (np.linalg.norm(state[:2][None] - obstacles[:, :2], axis=1)
         - obstacles[:, 2] - float(env.r_robot)).min()
    )
    collision = bool(clearance < 0.0)
    oob = bool(
        (state[:2] < -cfg.taskspace_epsilon).any()
        or (state[:2] > GM.GRID_M + cfg.taskspace_epsilon).any()
    )
    goal = env.goal.detach().cpu().numpy()
    status = None
    if collision or oob or np.linalg.norm(state[:2] - goal) < cfg.reach:
        status = "collision" if collision else ("oob" if oob else "reached")
    return {
        "episode_id": int(episode_id),
        "replica": int(replica),
        "gamma": float(gamma),
        "state": state.copy(),
        "hist": [],
        "path": [state[:2].copy()],
        "clear_min": clearance,
        "collision": collision,
        "oob": oob,
        "status": status,
        "term_t": (0 if status is not None else None),
        "step_stats": [],
    }


def _context_arrays(episodes, env, cfg):
    return CX.arrays_for_episodes(episodes, env, cfg.conditioning_schema)


def query_has_socp_error(result):
    """Any full-H or terminal verifier SOCP error makes the query unobserved."""

    return (
        result.get("reason") == "socp_error"
        or result.get("terminal_reason") == "socp_error"
    )


def classify_nvp_all_k(all_k_counts: dict, socp_errors: int) -> str:
    """Classify an audit-only all-K check at an unchanged NVP context."""

    if int(all_k_counts["nominal_hp_eligible"]) > 0:
        return "selected_B_acquisition_miss"
    if int(socp_errors) > 0:
        return "indeterminate_socp_error"
    if int(all_k_counts["execution_verifier_positive"]) > 0:
        return "all_K_nominal_hp_gate_failure"
    return "finite_K_no_execution_candidate"


def run_all_k_nvp_audit_only(
    executor,
    *,
    episode_id,
    state,
    candidate_controls,
    gamma,
    query_rows,
    execution_selection,
    execution_rule,
    env,
):
    """Verify unselected K-B plans without access to training or RNG objects.

    The deliberately narrow signature is the isolation contract: policy, GP,
    beta, D/D+, optimizer, acquisition RNG and execution state are not mutable
    inputs.  The returned dictionary is observational metadata only.
    """

    selected_ids = {int(row[0]) for row in query_rows}
    audit_ids = [
        candidate_id for candidate_id in range(len(candidate_controls))
        if candidate_id not in selected_ids
    ]
    tasks = [
        (episode_id, candidate_id, state, candidate_controls[candidate_id], gamma)
        for candidate_id in audit_ids
    ]
    audit_start = time.perf_counter()
    audit_results = list(executor.map(RC.verify_in_worker, tasks, chunksize=1))
    audit_wall = time.perf_counter() - audit_start
    result_by_id = {
        int(candidate_id): result for candidate_id, _, result in query_rows
    }
    for _, candidate_id, result in audit_results:
        result_by_id[int(candidate_id)] = result
    if set(result_by_id) != set(range(len(candidate_controls))):
        raise RuntimeError("all-K NVP audit did not cover every candidate")
    all_k_results = [
        result_by_id[candidate_id] for candidate_id in range(len(candidate_controls))
    ]
    all_k_selection = EX.select_nominal_hp_execution(
        state,
        candidate_controls,
        all_k_results,
        gamma,
        env,
        candidate_ids=list(range(len(candidate_controls))),
        selector=execution_rule,
    )
    all_k_errors = int(sum(query_has_socp_error(result) for result in all_k_results))
    result = {
        "status": "audit_only_no_execution_no_storage",
        "classification": classify_nvp_all_k(
            all_k_selection["counts"], all_k_errors
        ),
        "selected_B_counts": execution_selection["counts"],
        "selected_B_socp_errors": int(sum(
            query_has_socp_error(row[2]) for row in query_rows
        )),
        "all_K_counts": all_k_selection["counts"],
        "all_K_socp_errors": all_k_errors,
        "audit_extra_verifications": len(audit_results),
        "audit_socp_solves": int(sum(
            int(item[2]["n_socp_solve"]) for item in audit_results
        )),
        "audit_verifier_seconds": float(sum(
            float(item[2]["verifier_seconds"]) for item in audit_results
        )),
        "audit_wall_seconds": audit_wall,
    }
    if result["classification"] == "selected_B_acquisition_miss":
        if int(execution_selection["counts"]["nominal_hp_eligible"]) != 0:
            raise RuntimeError("acquisition-miss audit contradicts selected-B NVP")
        chosen_id = int(all_k_selection["chosen"]["candidate_id"])
        if chosen_id in selected_ids:
            raise RuntimeError("acquisition-miss witness was already in selected B")
    return result, audit_wall


def _proposal_noise(policy, active, cfg, purpose, round_i, control_t, device):
    """Stable per-episode proposal streams, batched only after noise generation."""

    seed_round = 0 if purpose == "controller_eval" else int(round_i)
    chunks = []
    for episode in active:
        generator = torch.Generator(device=device)
        generator.manual_seed(AFE2.named_seed(
            cfg.seed,
            "proposal",
            purpose,
            seed_round,
            episode["episode_id"],
            control_t,
        ))
        chunks.append(torch.randn(
            cfg.K, policy.d, device=device, generator=generator
        ))
    return torch.cat(chunks, dim=0)


def _acquisition_stats(
    sig,
    selected,
    features,
    controls,
    cfg,
    marginal_sigma=None,
    sequential_trace=None,
):
    if sequential_trace:
        ess_by_step = [float(row["ess_norm"]) for row in sequential_trace]
        entropy_by_step = [float(row["entropy_norm"]) for row in sequential_trace]
        ess_norm = float(np.median(ess_by_step))
        ess_first = ess_by_step[0]
        entropy = float(np.median(entropy_by_step))
        pool_vectors = [row["scores"].detach().cpu().numpy() for row in sequential_trace]
        selected_values = np.asarray([
            row["chosen_score"] for row in sequential_trace
        ], dtype=np.float64)
        pool_values = np.concatenate(pool_vectors)
        step_spans = np.asarray([np.ptp(row) for row in pool_vectors])
        step_iqrs = np.asarray([
            np.quantile(row, 0.75) - np.quantile(row, 0.25)
            for row in pool_vectors
        ])
        uplift = float(np.median([
            float(row["chosen_score"]) - float(row["scores"].mean())
            for row in sequential_trace
        ]))
    else:
        weights = torch.exp(((sig - sig.max()) / max(cfg.beta, 1.0e-9)).clamp(-30, 30))
        probability = (weights / weights.sum()).to(torch.float64)
        ess_norm = float(1.0 / (probability.square().sum() * probability.numel()))
        ess_first = ess_norm
        ess_by_step = [ess_norm]
        entropy = float(
            -(probability * (probability + 1.0e-30).log()).sum() / np.log(cfg.K)
        )
        entropy_by_step = [entropy]
        pool_values = sig.detach().cpu().numpy()
        selected_values = sig[selected].detach().cpu().numpy()
        step_spans = np.asarray([np.ptp(pool_values)])
        step_iqrs = np.asarray([
            np.quantile(pool_values, 0.75) - np.quantile(pool_values, 0.25)
        ])
        uplift = float(selected_values.mean() - pool_values.mean())
    quantiles = np.quantile(pool_values, [0.1, 0.25, 0.5, 0.75, 0.9])
    normalized = features.detach().cpu().to(torch.float64)
    cosine_distance = (1.0 - normalized @ normalized.T).clamp_min(0.0)
    pairs = torch.triu_indices(cfg.K, cfg.K, offset=1)
    feature_distance = cosine_distance[pairs[0], pairs[1]].numpy()
    plan_distance = torch.pdist(
        controls.detach().cpu().to(torch.float64).reshape(cfg.K, -1)
    ).numpy()
    correlation = (
        float(np.corrcoef(feature_distance, plan_distance)[0, 1])
        if np.std(feature_distance) > 0.0 and np.std(plan_distance) > 0.0
        else float("nan")
    )
    output = {
        "ess_norm": ess_norm,
        "ess_first": ess_first,
        "ess_by_step": ess_by_step,
        "ent": entropy,
        "entropy_by_step": entropy_by_step,
        "uplift": uplift,
        "sig_span": float(np.median(step_spans)),
        "sig_iqr": float(np.median(step_iqrs)),
        "sig_all": [float(quantiles[index]) for index in (0, 2, 4)],
        "sig_sel": [
            float(value)
            for value in np.quantile(selected_values, [0.1, 0.5, 0.9])
        ],
        "feature_cosine_distance_q": [
            float(value) for value in np.quantile(feature_distance, [0.1, 0.5, 0.9])
        ],
        "feature_plan_distance_corr": correlation,
    }
    if marginal_sigma is not None:
        output["marginal_sigma_med"] = float(marginal_sigma.median())
        output["marginal_sigma_iqr"] = float(
            torch.quantile(marginal_sigma, 0.75) - torch.quantile(marginal_sigma, 0.25)
        )
    return output


@torch.no_grad()
def run_parallel_episodes(
    policy,
    gp,
    env,
    cfg,
    store,
    round_i,
    replicas,
    device,
    executor,
    *,
    collect,
    viz,
    purpose,
    acquisition_mode="sequential",
    max_control_steps=None,
):
    """Advance all gamma x replica episodes in lockstep with batched GPU proposals."""

    start = env.x0.detach().cpu().numpy().astype(np.float32)
    episodes = []
    for gamma_index, gamma in enumerate(cfg.gammas):
        for replica in range(replicas):
            episode_id = gamma_index * replicas + replica
            episodes.append(_episode(start, gamma, replica, episode_id, env, cfg))
    timings = {
        "sampling": 0.0,
        "verifier_wall": 0.0,
        "nvp_audit_verifier_wall": 0.0,
        "bookkeeping": 0.0,
    }
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    robot_radius = float(env.r_robot)
    route_metric_steps = int(getattr(cfg, "route_metric_steps", 0))
    route_ambiguity_band = float(getattr(
        cfg,
        "route_ambiguity_band",
        RM.DEFAULT_AMBIGUITY_BAND,
    ))

    control_horizon = cfg.T if max_control_steps is None else min(
        cfg.T, int(max_control_steps)
    )
    for control_t in range(control_horizon):
        active = [episode for episode in episodes if episode["status"] is None]
        if not active:
            break
        grid_np, low_np, hist_np = _context_arrays(active, env, cfg)
        grid = torch.as_tensor(grid_np, device=device)
        low = torch.as_tensor(low_np, device=device)
        hist = torch.as_tensor(hist_np, device=device)
        sampling_start = time.perf_counter()
        context = policy.ctx_from(grid, low, hist)
        repeated_context = context.repeat_interleave(cfg.K, dim=0)
        initial_noise = _proposal_noise(
            policy, active, cfg, purpose, round_i, control_t, device
        )
        candidates = policy.sample(
            len(active) * cfg.K,
            repeated_context,
            nfe=cfg.nfe,
            temp=cfg.temp,
            initial_noise=initial_noise,
        ).reshape(len(active), cfg.K, policy.H_pred, 2)
        features = policy.phi_s(
            candidates.reshape(len(active) * cfg.K, policy.H_pred, 2),
            repeated_context,
            s=cfg.s,
        )
        features = RC.l2_normalize(features).reshape(len(active), cfg.K, -1)
        marginal_sigma = gp.sigma(
            features.reshape(len(active) * cfg.K, -1)
        ).reshape(
            len(active), cfg.K
        )
        selected = []
        traces = []
        first_scores = []
        seed_round = 0 if purpose == "controller_eval" else int(round_i)
        for episode_index, episode in enumerate(active):
            with AC.isolated_random_state(AFE2.named_seed(
                cfg.seed,
                "acquisition",
                purpose,
                seed_round,
                episode["episode_id"],
                control_t,
            )):
                if acquisition_mode == "uniform":
                    order = torch.randperm(cfg.K, device=features.device)
                    vectors = gp.sequential_score_vectors(
                        features[episode_index], order, min(cfg.B, cfg.K)
                    )
                    chosen = order[: min(cfg.B, cfg.K)].tolist()
                    pending = list(range(cfg.K))
                    trace = []
                    for step, vector in enumerate(vectors):
                        chosen_global = int(chosen[step])
                        chosen_local = pending.index(chosen_global)
                        trace.append({
                            "scores": vector,
                            "remaining": None,
                            "chosen": chosen_global,
                            "chosen_score": float(vector[chosen_local]),
                            "ess_norm": 1.0,
                            "entropy_norm": 1.0,
                        })
                        pending.pop(chosen_local)
                elif acquisition_mode == "sequential":
                    chosen, trace = gp.sequential_acquire(
                        features[episode_index], min(cfg.B, cfg.K), cfg.beta
                    )
                else:
                    raise ValueError(f"unknown acquisition mode: {acquisition_mode}")
            selected.append(chosen)
            traces.append(trace)
            first_scores.append(trace[0]["scores"])
        sigma = torch.stack(first_scores)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings["sampling"] += time.perf_counter() - sampling_start

        candidate_cpu = candidates.detach().cpu().numpy()
        sigma_cpu = sigma.detach().cpu()
        marginal_sigma_cpu = marginal_sigma.detach().cpu()
        feature_cpu = features.detach().cpu()
        step_context_ids = {}
        if collect:
            for local_index, episode in enumerate(active):
                step_context_ids[episode["episode_id"]] = store.add_step_ctx(
                    episode["state"],
                    grid_np[local_index],
                    low_np[local_index],
                    hist_np[local_index],
                    (round_i, episode["episode_id"], control_t),
                )
        tasks = []
        for local_index, episode in enumerate(active):
            for candidate_id in selected[local_index]:
                tasks.append((
                    episode["episode_id"],
                    candidate_id,
                    episode["state"],
                    candidate_cpu[local_index, candidate_id],
                    episode["gamma"],
                ))
        verifier_start = time.perf_counter()
        results = list(executor.map(RC.verify_in_worker, tasks, chunksize=1))
        timings["verifier_wall"] += time.perf_counter() - verifier_start
        by_episode = {episode["episode_id"]: [] for episode in active}
        for episode_id, candidate_id, result in results:
            by_episode[episode_id].append((candidate_id, result))

        bookkeeping_start = time.perf_counter()
        audit_wall_this_step = 0.0
        for local_index, episode in enumerate(active):
            need_segments = (
                viz is not None
                or (
                    route_metric_steps > 0
                    and control_t < route_metric_steps
                )
            )
            segments_all = (
                GR.di_rollout_batch(
                    episode["state"], candidate_cpu[local_index], env.dt
                )
                if need_segments else None
            )
            episode_results = by_episode[episode["episode_id"]]
            acquired_scores = {
                int(row["chosen"]): float(row["chosen_score"])
                for row in traces[local_index]
            }
            best = None
            execution_selection = None
            query_rows = []
            verifier_cpu_seconds = 0.0
            for candidate_id, result in episode_results:
                verifier_cpu_seconds += float(result["verifier_seconds"])
                query_id = -1
                controls = candidate_cpu[local_index, candidate_id]
                segment = GR.window_positions(episode["state"], controls, env.dt)
                if not query_has_socp_error(result) and collect:
                    query_id = store.add_query(
                        step_context_ids[episode["episode_id"]],
                        controls,
                        result,
                        acquired_scores[candidate_id],
                        episode["gamma"],
                        round_i,
                        segment,
                    )
                query_rows.append((candidate_id, query_id, result))
                if cfg.execution_rule == "legacy_max_horizon_progress" and result["exec_y"] and (
                    best is None or result["exec_prog"] > best[0]
                ):
                    best = (float(result["exec_prog"]), query_id, controls, candidate_id, result)

            if cfg.execution_rule != "legacy_max_horizon_progress":
                queried_ids = [row[0] for row in query_rows]
                queried_controls = np.stack([
                    candidate_cpu[local_index, candidate_id]
                    for candidate_id in queried_ids
                ])
                execution_selection = EX.select_nominal_hp_execution(
                    episode["state"],
                    queried_controls,
                    [row[2] for row in query_rows],
                    episode["gamma"],
                    env,
                    candidate_ids=queried_ids,
                    selector=cfg.execution_rule,
                )
                chosen = execution_selection["chosen"]
                if chosen is not None:
                    local_choice = int(chosen["local_index"])
                    candidate_id, query_id, result = query_rows[local_choice]
                    controls = candidate_cpu[local_index, candidate_id]
                    primary = (
                        chosen["step_progress"]
                        if cfg.execution_rule == EX.MAX_STEP_PROGRESS
                        else chosen["nominal_hp_step_margin"]
                    )
                    best = (float(primary), query_id, controls, candidate_id, result)

            drawn = selected[local_index]
            stats = _acquisition_stats(
                sigma_cpu[local_index],
                drawn,
                feature_cpu[local_index],
                torch.from_numpy(candidate_cpu[local_index]),
                cfg,
                marginal_sigma=marginal_sigma_cpu[local_index],
                sequential_trace=traces[local_index],
            )
            full_positive_available = any(row[2]["y"] == 1 for row in query_rows)
            selected_rescue = bool(best is not None and best[4]["terminal_rescue"])
            stats.update(
                n_err=sum(row[2]["reason"] == "socp_error" for row in query_rows),
                n_socp_solve=sum(int(row[2]["n_socp_solve"]) for row in query_rows),
                verifier_seconds=verifier_cpu_seconds,
                n_terminal_error=sum(
                    row[2]["terminal_reason"] == "socp_error" for row in query_rows
                ),
                n_pos=sum(row[2]["y"] == 1 for row in query_rows),
                n_exec_pos=sum(row[2]["exec_y"] == 1 for row in query_rows),
                n_full_socp_positive=sum(row[2]["y"] == 1 for row in query_rows),
                n_exec_verified_hp_positive=(
                    None if execution_selection is None
                    else execution_selection["counts"]["nominal_hp_eligible"]
                ),
                execution_failure=(
                    None if execution_selection is None
                    else execution_selection["failure"]
                ),
                execution_rule=cfg.execution_rule,
                n_terminal_rescue=sum(bool(row[2]["terminal_rescue"]) for row in query_rows),
                n_terminal_reverify=sum(bool(row[2]["terminal_reverify"]) for row in query_rows),
                selected_terminal_rescue=selected_rescue,
                selected_terminal_required=bool(selected_rescue and not full_positive_available),
                full_positive_available=full_positive_available,
                n_drawn=len(query_rows),
            )
            nvp_audit = None
            if (
                best is None
                and getattr(cfg, "nvp_audit_all_k", False)
                and purpose == "gather"
            ):
                if execution_selection is None:
                    raise RuntimeError("all-K NVP audit requires nominal-Hp execution")
                nvp_audit, audit_wall = run_all_k_nvp_audit_only(
                    executor,
                    episode_id=episode["episode_id"],
                    state=episode["state"],
                    candidate_controls=candidate_cpu[local_index],
                    gamma=episode["gamma"],
                    query_rows=query_rows,
                    execution_selection=execution_selection,
                    execution_rule=cfg.execution_rule,
                    env=env,
                )
                audit_wall_this_step += audit_wall
                timings["nvp_audit_verifier_wall"] += audit_wall
                stats["nvp_audit"] = nvp_audit
                episode["nvp_audit"] = nvp_audit
            if route_metric_steps > 0 and control_t < route_metric_steps:
                if segments_all is None:
                    raise RuntimeError("route diagnostics require candidate segments")
                route_labels = RM.classify_plan_endpoints(
                    segments_all[:, -1, :],
                    start=env.x0.detach().cpu().numpy()[:2],
                    goal=goal,
                    ambiguity_band=route_ambiguity_band,
                )
                queried_labels = route_labels[np.asarray(drawn, dtype=np.int64)]
                positive_labels = np.asarray([
                    route_labels[candidate_id]
                    for candidate_id, _, result in query_rows
                    if result["y"] == 1
                ], dtype=np.int8)
                executed_labels = np.asarray(
                    [] if best is None else [route_labels[best[3]]],
                    dtype=np.int8,
                )
                stats["route_modes"] = {
                    "all_K": route_labels.astype(np.int8).tolist(),
                    "selected_B": queried_labels.astype(np.int8).tolist(),
                    "full_H_positive": positive_labels.tolist(),
                    "executed": executed_labels.tolist(),
                }
            episode["step_stats"].append(stats)

            if viz is not None:
                if segments_all is None:
                    raise RuntimeError("visualization requires candidate segments")
                segments = segments_all.astype(np.float16)
                admissible = [row[2] for row in query_rows if row[2]["exec_y"]]
                viz.append({
                    "t": control_t,
                    "episode": episode["episode_id"],
                    "replica": episode["replica"],
                    "gamma": episode["gamma"],
                    "state": episode["state"].copy(),
                    "segsK": segments,
                    "drawn": [row[0] for row in query_rows],
                    "y": [(-1 if row[2]["reason"] == "socp_error" else row[2]["y"])
                          for row in query_rows],
                    "exec_y": [row[2]["exec_y"] for row in query_rows],
                    "exec_verified_hp_y": (
                        None if execution_selection is None
                        else [
                            int(row["eligible"])
                            for row in execution_selection["per_candidate"]
                        ]
                    ),
                    "step_progress": (
                        None if execution_selection is None
                        else [
                            float(row["step_progress"])
                            for row in execution_selection["per_candidate"]
                        ]
                    ),
                    "nominal_hp_step_margin": (
                        None if execution_selection is None
                        else [
                            float(row["nominal_hp_step_margin"])
                            for row in execution_selection["per_candidate"]
                        ]
                    ),
                    "execution_rule": cfg.execution_rule,
                    "terminal_rescue": [bool(row[2]["terminal_rescue"]) for row in query_rows],
                    "terminal_tau": [row[2]["terminal_tau"] for row in query_rows],
                    "n_socp_solve": stats["n_socp_solve"],
                    "sel": (-1 if best is None else best[3]),
                    "sig_q": stats["sig_all"],
                    "sigB_q": stats["sig_sel"],
                    "min_margin": (
                        float(np.nanmin([row["exec_margin"] for row in admissible]))
                        if admissible else float("nan")
                    ),
                    "nvp_audit": nvp_audit,
                })

            if best is None:
                if collect:
                    store.mark_nvp_negative(row[1] for row in query_rows)
                episode["status"] = "nvp"
                episode["nvp_reason"] = (
                    execution_selection["failure"]
                    if execution_selection is not None
                    else "no_verified_execution_candidate"
                )
                episode["term_t"] = control_t
                continue
            if collect and best[1] >= 0:
                store.mark_executed(best[1])
            action = np.asarray(best[2][0], dtype=np.float32)
            episode["state"] = di_step(episode["state"], action, dt=env.dt)
            episode["hist"].append(action)
            episode["path"].append(episode["state"][:2].copy())
            episode["clear_min"] = min(
                episode["clear_min"],
                float(
                    (np.linalg.norm(episode["state"][:2][None] - obstacles[:, :2], axis=1)
                     - obstacles[:, 2] - robot_radius).min()
                ),
            )
            episode["collision"] = bool(episode["clear_min"] < 0.0)
            episode["oob"] = bool(
                (episode["state"][:2] < -cfg.taskspace_epsilon).any()
                or (episode["state"][:2] > GM.GRID_M + cfg.taskspace_epsilon).any()
            )
            if episode["collision"] or episode["oob"]:
                episode["status"] = "collision" if episode["collision"] else "oob"
                episode["term_t"] = control_t + 1
            elif np.linalg.norm(episode["state"][:2] - goal) < cfg.reach:
                episode["status"] = "reached"
                episode["term_t"] = control_t + 1
        timings["bookkeeping"] += (
            time.perf_counter() - bookkeeping_start - audit_wall_this_step
        )

    output = []
    for episode in episodes:
        if episode["status"] is None:
            episode["status"] = "timeout"
        output.append({
            "episode_id": episode["episode_id"],
            "replica": episode["replica"],
            "gamma": episode["gamma"],
            "path": np.asarray(episode["path"], dtype=np.float32),
            "status": episode["status"],
            "term_t": episode["term_t"],
            "steps": len(episode["path"]) - 1,
            "clear_min": episode["clear_min"],
            "collision": episode["collision"],
            "oob": episode["oob"],
            "nvp_reason": episode.get("nvp_reason"),
            "nvp_audit": episode.get("nvp_audit"),
            "step_stats": episode["step_stats"],
        })
    return output, timings


def _per_gamma_episode_stats(episodes, cfg):
    output = {}
    for gamma in cfg.gammas:
        records = [record for record in episodes if record["gamma"] == float(gamma)]
        steps = [item for record in records for item in record["step_stats"]]
        row = {
            "episodes": len(records),
            "status_counts": {
                name: sum(record["status"] == name for record in records)
                for name in ("reached", "nvp", "timeout", "collision", "oob")
            },
            "steps": int(sum(record["steps"] for record in records)),
            "ess_med": (float(np.median([item["ess_norm"] for item in steps]))
                        if steps else None),
            "ess_first_med": (float(np.median([item["ess_first"] for item in steps]))
                              if steps else None),
            "ent_med": (float(np.median([item["ent"] for item in steps])) if steps else None),
            "uplift_med": (float(np.median([item["uplift"] for item in steps])) if steps else None),
            "sig_iqr_med": (float(np.median([item["sig_iqr"] for item in steps])) if steps else None),
            "sig_span_med": (float(np.median([item["sig_span"] for item in steps])) if steps else None),
            "n_q": int(sum(item["n_drawn"] for item in steps)),
            "n_pos": int(sum(item["n_pos"] for item in steps)),
            "n_exec_pos": int(sum(item["n_exec_pos"] for item in steps)),
            "n_full_socp_positive": int(sum(
                item["n_full_socp_positive"] for item in steps
            )),
            "n_exec_verified_hp_positive": int(sum(
                item["n_exec_verified_hp_positive"] or 0 for item in steps
            )),
            "n_socp_solve": int(sum(item["n_socp_solve"] for item in steps)),
            "verifier_cpu_seconds": float(sum(item["verifier_seconds"] for item in steps)),
            "n_err": int(sum(item["n_err"] for item in steps)),
        }
        row["nvp_audit"] = _summarize_nvp_audits(steps)
        if getattr(cfg, "nvp_audit_all_k", False) and (
            row["nvp_audit"]["count"] != row["status_counts"]["nvp"]
        ):
            raise RuntimeError("per-gamma NVP audit count does not match NVP episodes")
        route_steps = [item for item in steps if "route_modes" in item]
        if route_steps:
            row["route_modes_early"] = {
                population: RM.summarize_modes([
                    label
                    for item in route_steps
                    for label in item["route_modes"][population]
                ])
                for population in (
                    "all_K",
                    "selected_B",
                    "full_H_positive",
                    "executed",
                )
            }
        output[str(gamma)] = row
    return output


def _summarize_nvp_audits(step_stats):
    audits = [
        item["nvp_audit"] for item in step_stats
        if item.get("nvp_audit") is not None
    ]
    classes = (
        "selected_B_acquisition_miss",
        "all_K_nominal_hp_gate_failure",
        "finite_K_no_execution_candidate",
        "indeterminate_socp_error",
    )
    counts = {
        label: sum(audit["classification"] == label for audit in audits)
        for label in classes
    }
    total = len(audits)
    if sum(counts.values()) != total:
        raise RuntimeError("NVP audit classifications are not exhaustive and exclusive")
    return {
        "count": total,
        "class_counts": counts,
        "class_rates": {
            label: (float(count / total) if total else None)
            for label, count in counts.items()
        },
        "extra_verifications": int(sum(
            audit["audit_extra_verifications"] for audit in audits
        )),
        "socp_solves": int(sum(audit["audit_socp_solves"] for audit in audits)),
        "verifier_seconds": float(sum(
            audit["audit_verifier_seconds"] for audit in audits
        )),
    }


def _controller_summary(episodes, cfg, env):
    rows = {}
    for gamma in cfg.gammas:
        records = [record for record in episodes if record["gamma"] == float(gamma)]
        count = len(records)
        rows[str(gamma)] = {
            "SR": sum(record["status"] == "reached" for record in records) / count,
            "CR": sum(record["collision"] or record["oob"] for record in records) / count,
            "collision": sum(record["collision"] for record in records) / count,
            "OOB": sum(record["oob"] for record in records) / count,
            "NVP": sum(record["status"] == "nvp" for record in records) / count,
            "TO": sum(record["status"] == "timeout" for record in records) / count,
            "clear": float(np.nanmean([record["clear_min"] for record in records])),
            "time": (
                float(np.mean([
                    record["steps"] * env.dt
                    for record in records if record["status"] == "reached"
                ]))
                if any(record["status"] == "reached" for record in records)
                else float("nan")
            ),
            "clear_values": [float(record["clear_min"]) for record in records],
            "time_success_values": [
                float(record["steps"] * env.dt)
                for record in records if record["status"] == "reached"
            ],
            "status_values": [record["status"] for record in records],
            "nvp_t": [
                int(record["term_t"])
                for record in records if record["status"] == "nvp"
            ],
        }
    pooled = {
        key: float(np.mean([row[key] for row in rows.values()]))
        for key in ("SR", "CR", "NVP")
    }
    return rows, pooled


@torch.no_grad()
def _calibration_score_vectors(policy, gp, store, cfg, device, context_ids=None):
    """Build beta-neutral B-step score vectors at disjoint rollout contexts."""

    vectors = []
    pools = []
    gamma_counts = {}
    chunk_size = 16
    legacy_all_contexts = context_ids is None
    if legacy_all_contexts:
        context_ids = list(range(len(store.ctx_state)))
    else:
        context_ids = [int(context_id) for context_id in context_ids]
    for begin in range(0, len(context_ids), chunk_size):
        sids = context_ids[begin:begin + chunk_size]
        grid = store.grid3_of(sids).to(device)
        low = torch.stack([torch.from_numpy(store.ctx_low5[sid]) for sid in sids]).to(device)
        hist = torch.stack([
            torch.from_numpy(store.ctx_hist[sid].astype(np.float32)) for sid in sids
        ]).to(device)
        context = policy.ctx_from(grid, low, hist)
        repeated = context.repeat_interleave(cfg.K, dim=0)
        with AC.isolated_random_state(
            AFE2.named_seed(
                cfg.seed,
                "rbf_operational_beta_candidates",
                begin if legacy_all_contexts else tuple(sids),
            )
        ):
            controls = policy.sample(
                len(sids) * cfg.K, repeated, nfe=cfg.nfe, temp=cfg.temp
            )
        features = RC.l2_normalize(
            policy.phi_s(controls, repeated, s=cfg.s)
        ).reshape(len(sids), cfg.K, -1)
        for local_index, sid in enumerate(sids):
            pools.append(features[local_index].detach())
            order_rng = np.random.default_rng(
                AFE2.named_seed(cfg.seed, "rbf_operational_beta_order", sid)
            )
            order = torch.as_tensor(
                order_rng.permutation(cfg.K), device=device, dtype=torch.long
            )
            vectors.extend([
                score.detach().cpu().numpy()
                for score in gp.sequential_score_vectors(
                    features[local_index], order, min(cfg.B, cfg.K)
                )
            ])
            gamma = str(round(float(store.ctx_low5[sid][-1]), 2))
            gamma_counts[gamma] = gamma_counts.get(gamma, 0) + 1
    if not vectors:
        raise RuntimeError("operational beta calibration produced no rollout-context scores")
    return vectors, gamma_counts, pools


@torch.no_grad()
def calibrate_rbf(policy, env, cfg, device, executor):
    """Calibrate ell and one fixed beta at the declared operational GP size."""

    calibration_start = time.perf_counter()
    state = env.x0.detach().cpu().numpy().astype(np.float32)
    synthetic = [
        _episode(state, gamma, 0, index, env, cfg)
        for index, gamma in enumerate(cfg.gammas)
    ]
    grid_np, low_np, hist_np = _context_arrays(synthetic, env, cfg)
    grid = torch.as_tensor(grid_np, device=device)
    low = torch.as_tensor(low_np, device=device)
    hist = torch.as_tensor(hist_np, device=device)
    context = policy.ctx_from(grid, low, hist)
    base, extra = divmod(cfg.lengthscale_samples, len(cfg.gammas))
    context_indices = [
        index
        for index in range(len(cfg.gammas))
        for _ in range(base + int(index < extra))
    ]
    context_index = torch.as_tensor(context_indices, device=device)
    with AC.isolated_random_state(AFE2.named_seed(cfg.seed, "rbf_lengthscale")):
        controls = policy.sample(
            cfg.lengthscale_samples,
            context[context_index],
            nfe=cfg.nfe,
            temp=cfg.temp,
        )
        features = RC.l2_normalize(
            policy.phi_s(controls, context[context_index], s=cfg.s)
        )
    base_lengthscale = RC.mean_pairwise_lengthscale(features)
    lengthscale = float(base_lengthscale * cfg.lengthscale_multiplier)
    lengthscale_segments = GR.di_rollout_batch(
        state,
        controls.detach().cpu().numpy().reshape(
            cfg.lengthscale_samples, policy.H_pred, 2
        ),
        env.dt,
    )
    lengthscale_seed_route_modes = RM.summarize_modes(
        RM.classify_plan_endpoints(
            lengthscale_segments[:, -1, :],
            start=env.x0.detach().cpu().numpy()[:2],
            goal=env.goal.detach().cpu().numpy(),
            ambiguity_band=float(getattr(
                cfg, "route_ambiguity_band", RM.DEFAULT_AMBIGUITY_BAND
            )),
        )
    )

    # A beta calibrated on the 50-sample length-scale seed is not operational:
    # the expansion GP has `gp_cap` points and sees rollout contexts.  Build a
    # separate pretrained-only archive using uniform B-budget acquisition, then
    # discard it from CFM training and evaluation.
    empty_gp = RC.RBFGPSigma(lengthscale, cfg.gp_lam)
    seed_store = AC.DStore(
        conditioning_schema=cfg.conditioning_schema,
        condition_dim=cfg.raw_condition_dim,
    )
    calibration_replicas = cfg.calibration_replicas or cfg.replicas
    seed_episodes, seed_timing = run_parallel_episodes(
        policy, empty_gp, env, cfg, seed_store, 0, calibration_replicas,
        device, executor, collect=True, viz=None,
        purpose="rbf_operational_seed", acquisition_mode="uniform",
        max_control_steps=cfg.calibration_control_steps,
    )
    seed_buffer_seed = AFE2.named_seed(cfg.seed, "rbf_operational_seed_buffer")
    seed_ids = (
        RC.previous_round_positive_ids(
            seed_store, 0, cfg.gp_cap, cfg.gammas, seed_buffer_seed
        )
        if cfg.gp_replay_sampling == "round_gamma"
        else RC.recent_round_positive_ids_hierarchical(
            seed_store, 0, 1, cfg.gp_cap, seed_buffer_seed
        )
    )
    if len(seed_ids) != cfg.gp_cap:
        raise RuntimeError(
            f"operational GP calibration requires exactly {cfg.gp_cap} verified positives; "
            f"found {len(seed_ids)}"
        )
    operational_features = AFE2.embed_queries(
        policy, seed_store, cfg, device, ids=seed_ids
    ).to(device)
    gp = RC.RBFGPSigma(lengthscale, cfg.gp_lam)
    gp.set_buffer(operational_features)

    # Use an independent uniform-acquisition rollout archive for contexts.  The
    # query plans in this archive are not GP points and never enter D+.
    beta_store = AC.DStore(
        conditioning_schema=cfg.conditioning_schema,
        condition_dim=cfg.raw_condition_dim,
    )
    beta_episodes, beta_timing = run_parallel_episodes(
        policy, gp, env, cfg, beta_store, 0, calibration_replicas,
        device, executor, collect=True, viz=None,
        purpose="rbf_operational_beta_contexts", acquisition_mode="uniform",
        max_control_steps=cfg.calibration_control_steps,
    )
    score_vector_start = time.perf_counter()
    beta_context_ids = None
    if cfg.adaptive_beta_contexts_per_gamma is not None:
        beta_context_ids = AD.round_gamma_episode_balanced_context_ids(
            beta_store,
            0,
            cfg.gammas,
            cfg.adaptive_beta_contexts_per_gamma,
            cfg.seed,
            equalize_gammas=cfg.adaptive_beta_equalize_gammas,
        )
    score_vectors, context_gamma_counts, feature_pools = _calibration_score_vectors(
        policy,
        gp,
        beta_store,
        cfg,
        device,
        context_ids=beta_context_ids,
    )
    score_vector_seconds = time.perf_counter() - score_vector_start
    target = (
        BC.ESS_TARGET
        if cfg.adaptive_ess_target is None
        else float(cfg.adaptive_ess_target)
    )
    solution = BC.solve_beta_ragged(score_vectors, target=target)
    offline_sweep = None
    if cfg.rbf_offline_sweep:
        offline_sweep = AD.rbf_counterfactual_sweep(
            feature_pools,
            operational_features,
            cfg,
            round_i=0,
            target=target,
            lengthscale=lengthscale,
        )
    seed_steps = [item for row in seed_episodes for item in row["step_stats"]]
    beta_steps = [item for row in beta_episodes for item in row["step_stats"]]

    def verifier_budget(steps):
        return {
            "queries": int(sum(item["n_drawn"] for item in steps)),
            "positives": int(sum(item["n_pos"] for item in steps)),
            "socp_solves": int(sum(item["n_socp_solve"] for item in steps)),
            "socp_errors": int(sum(item["n_err"] for item in steps)),
            "verifier_cpu_seconds": float(sum(item["verifier_seconds"] for item in steps)),
        }

    seed_budget = verifier_budget(seed_steps)
    beta_budget = verifier_budget(beta_steps)
    calibration_budget = {
        "seed_archive": seed_budget,
        "disjoint_context_archive": beta_budget,
        "total_queries": seed_budget["queries"] + beta_budget["queries"],
        "total_positives": seed_budget["positives"] + beta_budget["positives"],
        "total_socp_solves": seed_budget["socp_solves"] + beta_budget["socp_solves"],
        "total_socp_errors": seed_budget["socp_errors"] + beta_budget["socp_errors"],
        "score_vector_seconds": float(score_vector_seconds),
        "total_wall_seconds": float(time.perf_counter() - calibration_start),
        "enters_training_Dplus": False,
        "enters_round1_GP": "exactly the declared operational_gp_size seed positives",
    }
    return {
        "base_lengthscale": float(base_lengthscale),
        "lengthscale_multiplier": float(cfg.lengthscale_multiplier),
        "lengthscale": float(lengthscale),
        "lengthscale_samples": cfg.lengthscale_samples,
        "lengthscale_seed_route_modes": lengthscale_seed_route_modes,
        "lengthscale_seed_route_intervention": False,
        "calibration_replicas": int(calibration_replicas),
        "calibration_control_steps": cfg.calibration_control_steps,
        "operational_gp_size": gp.n,
        "operational_seed_queries": len(seed_store),
        "operational_seed_positives": seed_store.n_pos(),
        "operational_seed_status_counts": {
            name: sum(row["status"] == name for row in seed_episodes)
            for name in ("reached", "nvp", "timeout", "collision", "oob")
        },
        "operational_seed_timing": seed_timing,
        "beta_context_queries": len(beta_store),
        "beta_context_count": len(beta_store.ctx_state),
        "beta_context_selected_count": (
            len(beta_store.ctx_state)
            if beta_context_ids is None else len(beta_context_ids)
        ),
        "beta_context_cap_per_gamma": cfg.adaptive_beta_contexts_per_gamma,
        "beta_context_gamma_counts": context_gamma_counts,
        "beta_context_status_counts": {
            name: sum(row["status"] == name for row in beta_episodes)
            for name in ("reached", "nvp", "timeout", "collision", "oob")
        },
        "beta_context_timing": beta_timing,
        "calibration_budget": calibration_budget,
        "bootstrap_features": operational_features,
        "beta": float(solution["beta"]),
        "beta_solution": solution,
        "score_vector_sha256": BC.score_vectors_sha256(score_vectors),
        "score_vectors": score_vectors,
        "offline_sweep": offline_sweep,
    }


def _gp_from_query_ids(policy, store, query_ids, cfg, device, lengthscale):
    gp = RC.RBFGPSigma(lengthscale, cfg.gp_lam)
    features = AFE2.embed_queries(policy, store, cfg, device, ids=query_ids)
    gp.set_buffer(features.to(device))
    counts = {}
    for query_id in query_ids:
        key = str(round(float(store.q_gamma[query_id]), 2))
        counts[key] = counts.get(key, 0) + 1
    diagnostics = gp.diagnostics()
    diagnostics.update(
        source_query_ids=[int(value) for value in query_ids],
        gamma_counts=counts,
    )
    return gp, diagnostics


def _aggregate_step_stats(episodes, cfg):
    values = [item for record in episodes for item in record["step_stats"]]
    if not values:
        return {}
    correlations = [
        item["feature_plan_distance_corr"]
        for item in values if np.isfinite(item["feature_plan_distance_corr"])
    ]
    nvp_audit = _summarize_nvp_audits(values)
    if getattr(cfg, "nvp_audit_all_k", False) and (
        nvp_audit["count"]
        != sum(record["status"] == "nvp" for record in episodes)
    ):
        raise RuntimeError("aggregate NVP audit count does not match NVP episodes")
    output = {
        "ess_med": float(np.median([item["ess_norm"] for item in values])),
        "ess_first_med": float(np.median([item["ess_first"] for item in values])),
        "ess_by_step_med": [
            float(np.median([item["ess_by_step"][step] for item in values]))
            for step in range(min(cfg.B, cfg.K))
        ],
        "ent_med": float(np.median([item["ent"] for item in values])),
        "uplift_med": float(np.median([item["uplift"] for item in values])),
        "sig_span_med": float(np.median([item["sig_span"] for item in values])),
        "sig_iqr_med": float(np.median([item["sig_iqr"] for item in values])),
        "sig_all_med": float(np.median([item["sig_all"][1] for item in values])),
        "sig_sel_med": float(np.median([item["sig_sel"][1] for item in values])),
        "feature_plan_distance_corr_med": (
            float(np.median(correlations)) if correlations else None
        ),
        "verifier_cpu_seconds": float(sum(item["verifier_seconds"] for item in values)),
        "marginal_sigma_med": float(np.median([
            item["marginal_sigma_med"] for item in values
        ])),
        "marginal_sigma_iqr_med": float(np.median([
            item["marginal_sigma_iqr"] for item in values
        ])),
        "nvp_audit": nvp_audit,
    }
    route_values = [item for item in values if "route_modes" in item]
    if route_values:
        output["route_modes_early"] = {
            population: RM.summarize_modes([
                label
                for item in route_values
                for label in item["route_modes"][population]
            ])
            for population in (
                "all_K",
                "selected_B",
                "full_H_positive",
                "executed",
            )
        }
        output["route_metric_contexts"] = len(route_values)
    return output


def run(policy, env, cfg, device, outdir, checkpoint_path, checkpoint_sha256,
        checkpoint_model_sha256, checkpoint_contract, checkpoint_contract_sha256,
        source_git_state):
    if os.path.exists(outdir) and (not os.path.isdir(outdir) or os.listdir(outdir)):
        raise RuntimeError(f"single-arm run requires a new or empty output directory: {outdir}")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "viz_db"), exist_ok=True)
    profile = get_scene_profile(cfg.scene_profile)
    scene = scene_snapshot(env, profile)
    assert_scene_snapshot(scene)
    demo_reference = None
    demo_provenance = None
    if cfg.protocol_profile in {"v3_support_sweep", "v3_support_preflight"}:
        demo_reference, demo_provenance = DS.load_authenticated_demo_reference(
            checkpoint_path,
            checkpoint_sha256,
            load_tensors=cfg.demo_frac > 0.0,
        )
    store = AC.DStore(
        conditioning_schema=cfg.conditioning_schema,
        condition_dim=cfg.raw_condition_dim,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=cfg.afe_lr,
    )
    audit_contexts = (
        AC.build_audit_contexts(
            env,
            cfg.gammas,
            n_pos=cfg.audit_pos,
            conditioning_schema=cfg.conditioning_schema,
        )
        if cfg.training_probes else None
    )
    representation_probe = AFE2.rep_probe_build(policy, env, cfg, device)
    goal = env.goal.detach().cpu().numpy()

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=cfg.verifier_workers,
        mp_context=context,
        initializer=RC.initialize_verifier_worker,
        initargs=(cfg.scene_profile, cfg.reach, cfg.n_theta),
    ) as executor:
        calibration = calibrate_rbf(policy, env, cfg, device, executor)
        cfg.beta = calibration["beta"]
        calibration_public = {
            key: AFE2._json_safe(value)
            for key, value in calibration.items()
            if key not in {"bootstrap_features", "score_vectors"}
        }
        calibration_public.update({
            "status": "CALIBRATED_AFE_RBF_SEQUENTIAL_OPERATIONAL_V3",
            "kernel": "RBF on L2-normalized phi_s",
            "lengthscale_rule": (
                "declared multiplier times the mean pairwise embedding distance of exactly "
                "50 samples from the pretrained model"
            ),
            "gp_buffer_label": "full-H verifier positive only",
            "acquisition_statistic": (
                "normalized GP posterior variance conditioned on the GP buffer and only "
                "the already-selected pending locations in a B-step acquisition"
            ),
            "ess_target": calibration["beta_solution"]["target"],
            "scene_sha256": scene["sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "source_git_commit": source_git_state["commit"],
        })
        calibration_path = os.path.join(outdir, "rbf_calibration.json")
        with open(calibration_path, "w") as stream:
            json.dump(calibration_public, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

        if cfg.protocol_profile in {"v3_support_sweep", "v3_support_preflight"}:
            algorithm = "afe_rbf_low7_v3_optimizer_demo_support_v1"
        elif cfg.protocol_profile == "v2_lineage_mass_smoke":
            algorithm = "afe_rbf_low7_v2_lineage_mass_smoke_v1"
        elif cfg.protocol_profile == "v2_smoke":
            algorithm = "afe_rbf_low7_v2_sample_complete_smoke_v2"
        elif cfg.execution_rule != "legacy_max_horizon_progress":
            algorithm = "afe_rbf_low7_signed_execution_sweep_v1"
        elif cfg.acquisition_mode == "uniform":
            algorithm = "afe_uniform_parallel_v1"
        elif cfg.adaptive_ess_target is not None:
            algorithm = "afe_rbf_adaptive_ess_parallel_v4"
        else:
            algorithm = "afe_rbf_sequential_operational_parallel_v3"
        beta_protocol = (
            "uniform B-without-replacement; RBF beta is diagnostic only"
            if cfg.acquisition_mode == "uniform"
            else (
                f"round-1 operational calibration followed after every round by a beta-neutral "
                f"current-policy calibration targeting median ESS={cfg.adaptive_ess_target:g}; "
                "beta_n is frozen during round n"
                if cfg.adaptive_ess_target is not None
                else (
                    f"one pretrained-only continuous ESS calibration against an operational "
                    f"{cfg.gp_cap}-positive GP and disjoint rollout-context B-step score vectors; "
                    "uniform beta-neutral pending orders; fixed for every expansion round"
                )
            )
        )
        replay_population = (
            "the complete cumulative full-H D+ archive"
            if cfg.replay_window is None
            else (
                f"full-H positives from the current and previous "
                f"{cfg.replay_window - 1} rounds; cumulative D+ archive is retained"
            )
        )
        learning_memory = (
            f"uniform query replay over {replay_population}"
            if cfg.replay_sampling == "query_uniform"
            else (
                "hierarchically round/gamma/replica/context/query-interleaved replay over "
                f"{replay_population}"
            )
        )
        if cfg.replay_update_mode == "fixed_macro_steps_exact_epoch":
            update_description = (
                f"positive-only gamma/episode/context/query equal-mass CFM lr "
                f"{cfg.afe_lr:g}; one exact W2 D+ epoch partitioned into exactly "
                f"{cfg.optimizer_steps_per_round} mass-balanced macro Adam updates; "
                f"demo objective mass {cfg.demo_frac:g}; alpha 0, no prox"
            )
            optimizer_draws_per_round = None
            optimizer_draw_interpretation = (
                "exactly |eligible D+| unique positive draws; all positives appear once; "
                "macro-batches are balanced by hierarchical loss mass and each macro-batch "
                "causes exactly one Adam update"
            )
        elif cfg.replay_update_mode == "one_epoch_without_replacement":
            weighting_text = (
                "gamma/episode/context/query equal-mass weighted"
                if cfg.replay_loss_weighting
                == "gamma_episode_context_query_equal_mass"
                else "query-uniform"
            )
            update_description = (
                f"positive-only {weighting_text} CFM lr {cfg.afe_lr:g}, batch {cfg.batch}, one exact "
                "without-replacement epoch over eligible D+; dynamic optimizer steps "
                "ceil(|eligible D+|/batch), alpha 0, no prox"
            )
            optimizer_draws_per_round = None
            optimizer_draw_interpretation = (
                "exactly |eligible D+| unique positive draws; every eligible positive appears "
                "once; samples are partitioned into ceil(|eligible D+|/batch) minibatches whose "
                "sizes differ by at most one; the declared replay-loss measure, not draw count, "
                "sets each positive's gradient coefficient"
            )
        else:
            update_description = (
                f"signed CFM lr {cfg.afe_lr:g}, separate batch {cfg.batch}, "
                f"{cfg.afe_steps} steps, alpha {cfg.negative_alpha:g}, no prox"
            )
            optimizer_draws_per_round = int(cfg.batch * cfg.afe_steps)
            optimizer_draw_interpretation = (
                "stochastic replay draws, not an epoch over the eligible archive"
            )
        recipe = {
            "algorithm": algorithm,
            "protocol_profile": cfg.protocol_profile,
            "arm": "afe",
            "single_arm": True,
            "kernel": "RBF",
            "base_lengthscale": calibration["base_lengthscale"],
            "lengthscale_multiplier": cfg.lengthscale_multiplier,
            "lengthscale": calibration["lengthscale"],
            "lengthscale_protocol": calibration_public["lengthscale_rule"],
            "beta": cfg.beta,
            "beta_protocol": beta_protocol,
            "adaptive_ess_target": cfg.adaptive_ess_target,
            "adaptive_beta_contexts_per_gamma": (
                cfg.adaptive_beta_contexts_per_gamma
            ),
            "adaptive_beta_equalize_gammas": cfg.adaptive_beta_equalize_gammas,
            "acquisition_mode": cfg.acquisition_mode,
            "acquisition_memory": (
                "round 1: verified-positive pretrained calibration seed; later rounds: at most "
                f"{cfg.gp_cap} full-H positives from the current and previous "
                f"{cfg.gp_replay_window - 1} rounds, selected without replacement by "
                f"{cfg.gp_replay_sampling}; re-embedded with current phi; frozen within round"
            ),
            "calibration_budget": calibration_public["calibration_budget"],
            "calibration_replicas": calibration["calibration_replicas"],
            "calibration_control_steps": calibration["calibration_control_steps"],
            "calibration_scope": (
                "round-0 acquisition-only verifier budget; the seed archive supplies the "
                "declared round-1 GP but neither archive enters cumulative training D+ or audit"
            ),
            "calibration_limitation": (
                "beta is solved on beta-neutral random pending orders; realized first-step "
                "ESS/K and stage-normalized ESS/M_remaining are logged during expansion"
            ),
            "learning_memory": learning_memory,
            "replay_window": cfg.replay_window,
            "replay_sampling": cfg.replay_sampling,
            "replay_update_mode": cfg.replay_update_mode,
            "replay_loss_weighting": cfg.replay_loss_weighting,
            "replay_loss_measure": (
                "equal nested mass over active positive-support gamma, (round, episode), "
                "context, and positive query; all eligible positives remain in the exact epoch; "
                "a size-b minibatch in an S-step epoch applies weight b*S*mu_q before clipping"
                if cfg.replay_loss_weighting
                == "gamma_episode_context_query_equal_mass"
                else "one equal loss weight per eligible positive query"
            ),
            "replay_loss_measure_limitation": (
                "gamma/episode/context groups with zero eligible full-H positives cannot carry "
                "positive CFM mass and are logged as missing support"
                if cfg.replay_loss_weighting
                == "gamma_episode_context_query_equal_mass"
                else None
            ),
            "replay_epochs": (
                1 if cfg.replay_update_mode in {
                    "one_epoch_without_replacement", "fixed_macro_steps_exact_epoch"
                } else None
            ),
            "gp_replay_window": cfg.gp_replay_window,
            "gp_replay_sampling": cfg.gp_replay_sampling,
            "rbf_offline_sweep": (
                "one pretrained-policy counterfactual sweep stored in rbf_calibration.json"
                if cfg.rbf_offline_sweep else False
            ),
            "uncertainty_meaning": (
                "RBF posterior variance conditioned on the acquisition buffer and only the "
                "locations already selected within the same B-budget query; not validity "
                "probability and not a safety certificate"
            ),
            "parallel_sampling": (
                f"{cfg.replicas} closed-loop replicas per gamma advanced synchronously; one GPU "
                f"proposal batch per control tick; {cfg.verifier_workers} persistent spawned CPU "
                "verifier workers; no within-round GP update"
            ),
            "verifier_workers": cfg.verifier_workers,
            "execution": (
                f"{cfg.execution_rule}; execution-verified (full-H or certified goal prefix) "
                "and, for nominal_hp rules, the first executed state satisfies the nominal "
                "SafeMPPI DTCBF level-set condition; NVP terminates; no expert/fallback"
            ),
            "execution_rule": cfg.execution_rule,
            "nvp_all_k_audit": (
                "at the unchanged NVP context only, verify unselected K-B candidates and "
                "classify acquisition miss versus nominal-Hp gate versus finite-K failure; "
                "audit candidates never enter D/D+, GP, beta, execution, or training"
                if cfg.nvp_audit_all_k else False
            ),
            "update": update_description,
            "optimizer_draws_per_round": optimizer_draws_per_round,
            "optimizer_draws_formula": (
                "|eligible D+|" if optimizer_draws_per_round is None else None
            ),
            "optimizer_steps_formula": (
                str(cfg.optimizer_steps_per_round)
                if cfg.replay_update_mode == "fixed_macro_steps_exact_epoch"
                else "ceil(|eligible D+|/batch)"
                if cfg.replay_update_mode == "one_epoch_without_replacement"
                else str(cfg.afe_steps)
            ),
            "optimizer_draw_interpretation": optimizer_draw_interpretation,
            "rounds": cfg.rounds,
            "rollout_replicas": cfg.replicas,
            "T": cfg.T,
            "K": cfg.K,
            "B": cfg.B,
            "batch": cfg.batch,
            "afe_lr": cfg.afe_lr,
            "afe_steps": cfg.afe_steps,
            "optimizer_steps_per_round": cfg.optimizer_steps_per_round,
            "demo_frac": cfg.demo_frac,
            "demo_objective_semantics": (
                "L=(1-demo_frac)*L_pos + demo_frac*L_demo; both source losses "
                "independently normalized; no update when W2 D+ is empty"
                if demo_provenance is not None else None
            ),
            "demo_reference": (
                demo_provenance.record() if demo_provenance is not None else None
            ),
            "demo_buffer_isolation": (
                "authenticated TRAIN reference never enters D/D+, RBF GP, beta, "
                "acquisition, verifier counts, execution, or raw evaluation"
                if demo_provenance is not None else None
            ),
            "grad_clip": cfg.grad_clip,
            "gradient_clip_semantics": (
                "the declared replay measure defines the pre-clip objective; existing global "
                "gradient clipping is then applied and its active-step fraction is logged"
            ),
            "negative_alpha": cfg.negative_alpha,
            "negative_alpha_semantics": (
                "paper gradient-norm target: g = g_pos - rho*g_neg, "
                "rho=alpha*||g_pos||/||g_neg||; alpha=0 is exact positive-only update"
            ),
            "negative_population": (
                "all successfully observed B queries at a terminal NVP context from the same "
                "replay window; this task-specific closed-loop viability signal may overlap "
                "full-H D+ and excludes SOCP errors"
            ),
            "artifact_profile": (
                "sweep_compact" if cfg.sweep_compact_artifacts else "full"
            ),
            "artifact_profile_description": (
                "omit final DStore; retain checkpoints at the declared compact interval and "
                "training-viz rounds 1..10/every10; probe, calibration, and completion hashes "
                "remain complete"
                if cfg.sweep_compact_artifacts
                else "retain complete DStore, every checkpoint, and every training-viz round"
            ),
            "compact_checkpoint_every": cfg.compact_checkpoint_every,
            "route_diagnostics": {
                "intervention": False,
                "early_control_steps": cfg.route_metric_steps,
                "ambiguity_band_m": cfg.route_ambiguity_band,
                "mode_definition": (
                    "U/R is the sign of plan-endpoint cross-track displacement from the "
                    "oriented canonical start-goal line; ambiguous values are retained"
                ),
                "training_populations": [
                    "all_K", "selected_B", "full_H_positive", "executed"
                ],
            },
            "gp_cap": cfg.gp_cap,
            "gp_lam": cfg.gp_lam,
            "s": cfg.s,
            "nfe": cfg.nfe,
            "M_eval": cfg.M_eval,
            "training_probes": cfg.training_probes,
            "outcome_evaluation": (
                "not produced by the trainer; stored checkpoints require a separate raw, "
                "untilted temperature-1 evaluation"
            ),
            "gammas": list(cfg.gammas),
            "reach": cfg.reach,
            "seed": cfg.seed,
            "scene": scene,
            "source_checkpoint": os.path.abspath(checkpoint_path),
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_checkpoint_model_sha256": checkpoint_model_sha256,
            "source_checkpoint_contract": checkpoint_contract,
            "source_checkpoint_contract_sha256": checkpoint_contract_sha256,
            "source_git_commit": source_git_state["commit"],
            "runtime": AFE2._runtime_provenance(device),
            "methodological_scope": (
                "task-specific peptide-style RBF AFE adaptation; previous-round cap and parallel "
                "frozen acquisition are explicit computational assumptions"
            ),
            "reference_code_semantics": (
                "sequential Schur complements are the B<K budget-consistent adaptation of "
                "the public peptide implementation's batch-conditional covariance"
            ),
            "no_curriculum": True,
            "no_anchor": True,
            "no_prox": True,
            "no_fallback": True,
            "conditioning_schema": cfg.conditioning_schema,
            "raw_condition_dim": cfg.raw_condition_dim,
            "freeze_visual_encoder": cfg.freeze_visual_encoder,
        }
        recipe_path = os.path.join(outdir, "recipe.json")
        with open(recipe_path, "w") as stream:
            json.dump(AFE2._json_safe(recipe), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

        probe_path = os.path.join(outdir, "probe.jsonl")

        def write_probe(record):
            with open(probe_path, "a") as stream:
                stream.write(json.dumps(AFE2._json_safe(record), allow_nan=False) + "\n")

        bootstrap_gp = RC.RBFGPSigma(calibration["lengthscale"], cfg.gp_lam)
        bootstrap_gp.set_buffer(calibration["bootstrap_features"].to(device))
        if cfg.training_probes:
            audit0 = AC.run_audit(
                policy, audit_contexts, env, goal, device,
                n_plans=cfg.audit_plans, nfe=cfg.nfe, n_theta=cfg.n_theta,
                seed=AFE2.named_seed(cfg.seed, "audit"),
            )
            eval0, eval0_timing = run_parallel_episodes(
                policy, bootstrap_gp, env, cfg, store, 0, cfg.M_eval, device, executor,
                collect=False, viz=None, purpose="controller_eval",
            )
            rows0, pooled0 = _controller_summary(eval0, cfg, env)
        else:
            audit0 = {
                "V": None, "V_safe": None, "V_full": None,
                "V_gamma": {}, "V_safe_gamma": {}, "V_full_gamma": {},
            }
            rows0, pooled0 = {}, {"SR": None, "CR": None, "NVP": None}
            eval0_timing = None
        write_probe({
            "round": 0,
            "arm": "afe",
            "negative_alpha": cfg.negative_alpha,
            "execution_rule": cfg.execution_rule,
            "acquisition_mode": cfg.acquisition_mode,
            "beta_used": cfg.beta,
            "beta_next": cfg.beta,
            "V": audit0["V"],
            "V_safe": audit0["V_safe"],
            "V_full": audit0["V_full"],
            "V_gamma": audit0["V_gamma"],
            "V_safe_gamma": audit0["V_safe_gamma"],
            "V_full_gamma": audit0["V_full_gamma"],
            "ctrl": rows0,
            "ctrl_pooled": pooled0,
            "n_D": 0,
            "n_Dpos": 0,
            "n_Dneg": 0,
            "n_Dneutral": 0,
            "gp_buffer": bootstrap_gp.diagnostics(),
            "calibration_budget": calibration_public["calibration_budget"],
            "rbf_offline_sweep": calibration_public["offline_sweep"],
            "rep_cos": 1.0,
            "evaluation_timing": eval0_timing,
        })
        HT._save_hp_atomic(
            policy, os.path.join(outdir, "ckpt_0.pt"),
            extra={"iter": 0, "recipe": recipe, "resumable": False},
        )
        if cfg.training_probes:
            print(
                f"[afe-rbf] r000 V {audit0['V']:.3f} ctrl SR {pooled0['SR']:.2f} "
                f"NVP {pooled0['NVP']:.2f} beta/ell "
                f"{cfg.beta:.4g}/{calibration['lengthscale']:.4g}",
                flush=True,
            )
        else:
            print(
                f"[afe-rbf] r000 training probes skipped; beta/ell "
                f"{cfg.beta:.4g}/{calibration['lengthscale']:.4g}",
                flush=True,
            )

        gp_for_gather = bootstrap_gp
        gp_start_diagnostics = bootstrap_gp.diagnostics()
        for round_i in range(1, cfg.rounds + 1):
            round_start = time.perf_counter()
            beta_used = float(cfg.beta)
            policy.eval()
            viz = []
            episodes, gather_timing = run_parallel_episodes(
                policy, gp_for_gather, env, cfg, store, round_i, cfg.replicas,
                device, executor, collect=True, viz=viz, purpose="gather",
                acquisition_mode=cfg.acquisition_mode,
            )
            gather_seconds = time.perf_counter() - round_start
            per_gamma = _per_gamma_episode_stats(episodes, cfg)
            acquisition = _aggregate_step_stats(episodes, cfg)

            update_start = time.perf_counter()
            replay_rng = np.random.default_rng(AFE2.named_seed(cfg.seed, "replay", round_i))
            with AC.isolated_random_state(AFE2.named_seed(cfg.seed, "update", round_i)):
                if cfg.protocol_profile in {"v3_support_sweep", "v3_support_preflight"}:
                    update = DS.update_round_support(
                        policy, optimizer, store, cfg, device, replay_rng, round_i,
                        demo_reference,
                    )
                else:
                    update = SU.update_round_signed(
                        policy, optimizer, store, cfg, device, replay_rng, round_i,
                        alpha=cfg.negative_alpha,
                    )
            update_seconds = time.perf_counter() - update_start
            policy.eval()

            gp_buffer_seed = AFE2.named_seed(cfg.seed, "gp_buffer", round_i)
            query_ids = (
                RC.recent_round_positive_ids(
                    store,
                    round_i,
                    cfg.gp_replay_window,
                    cfg.gp_cap,
                    cfg.gammas,
                    gp_buffer_seed,
                )
                if cfg.gp_replay_sampling == "round_gamma"
                else RC.recent_round_positive_ids_hierarchical(
                    store,
                    round_i,
                    cfg.gp_replay_window,
                    cfg.gp_cap,
                    gp_buffer_seed,
                )
            )
            gp_post, gp_post_diagnostics = _gp_from_query_ids(
                policy, store, query_ids, cfg, device, calibration["lengthscale"]
            )
            adaptive_calibration = None
            beta_next = beta_used
            if cfg.adaptive_ess_target is not None:
                calibration_pools, calibration_gamma_counts = AD.feature_pools(
                    policy, store, cfg, device, round_i
                )
                adaptive_calibration = AD.calibrate_from_pools(
                    gp_post,
                    calibration_pools,
                    cfg,
                    round_i,
                    cfg.adaptive_ess_target,
                )
                adaptive_calibration["context_gamma_counts"] = calibration_gamma_counts
                beta_next = float(adaptive_calibration["beta"])
                cfg.beta = beta_next
            if cfg.training_probes:
                audit = AC.run_audit(
                    policy, audit_contexts, env, goal, device,
                    n_plans=cfg.audit_plans, nfe=cfg.nfe, n_theta=cfg.n_theta,
                    seed=AFE2.named_seed(cfg.seed, "audit"),
                )
                evaluation, evaluation_timing = run_parallel_episodes(
                    policy, gp_post, env, cfg, store, round_i, cfg.M_eval,
                    device, executor, collect=False, viz=None, purpose="controller_eval",
                    acquisition_mode=cfg.acquisition_mode,
                )
                rows, pooled = _controller_summary(evaluation, cfg, env)
            else:
                audit = {
                    "V": None, "V_safe": None, "V_full": None,
                    "V_gamma": {}, "V_safe_gamma": {}, "V_full_gamma": {},
                    "counts_gamma": {},
                }
                rows, pooled = {}, {"SR": None, "CR": None, "NVP": None}
                evaluation_timing = None
            drawn = (update or {}).get("drawn_ids", {})
            trained_gamma = {}
            distinct_gamma = {}
            for query_id, count in drawn.items():
                key = str(round(float(store.q_gamma[query_id]), 2))
                trained_gamma[key] = trained_gamma.get(key, 0) + int(count)
                distinct_gamma[key] = distinct_gamma.get(key, 0) + 1
            record = {
                "round": round_i,
                "arm": "afe",
                "negative_alpha": cfg.negative_alpha,
                "execution_rule": cfg.execution_rule,
                "acquisition_mode": cfg.acquisition_mode,
                "beta_used": beta_used,
                "beta_next": beta_next,
                "adaptive_beta_calibration": adaptive_calibration,
                "rbf_offline_sweep": None,
                "n_D": len(store),
                "n_Dpos": store.n_pos(),
                "n_Dneg": int(sum(int(value) == 1 for value in store.q_nvp_negative)),
                "n_Doverlap": int(sum(
                    int(y) == 1 and int(nvp_negative) == 1
                    for y, nvp_negative in zip(store.q_y, store.q_nvp_negative)
                )),
                "n_Dneutral": int(sum(
                    int(y) == 0 and int(nvp_negative) == 0
                    for y, nvp_negative in zip(store.q_y, store.q_nvp_negative)
                )),
                "n_Dterminal_rescue_neutral": int(sum(
                    int(y) == 0 and int(exec_y) == 1 and int(nvp_negative) == 0
                    for y, exec_y, nvp_negative in zip(
                        store.q_y, store.q_exec_y, store.q_nvp_negative
                    )
                )),
                "per_gamma": per_gamma,
                **acquisition,
                "gp_round_start": gp_start_diagnostics,
                "gp_buffer": gp_post_diagnostics,
                "rep_cos": AFE2.rep_cos_drift(policy, representation_probe, cfg),
                "V": audit["V"],
                "V_safe": audit["V_safe"],
                "V_full": audit["V_full"],
                "V_gamma": audit["V_gamma"],
                "V_safe_gamma": audit["V_safe_gamma"],
                "V_full_gamma": audit["V_full_gamma"],
                "V_counts_gamma": audit["counts_gamma"],
                "ctrl": rows,
                "ctrl_pooled": pooled,
                "trained_draws_gamma": trained_gamma,
                "trained_distinct_gamma": distinct_gamma,
                "n_train_distinct": 0 if update is None else update["n_distinct"],
                "t_gather": gather_seconds,
                "t_update": update_seconds,
                "gather_timing": gather_timing,
                "evaluation_timing": evaluation_timing,
            }
            if update is not None:
                record.update({
                    key: value for key, value in update.items()
                    if key not in {"drawn_ids", "negative_drawn_ids"}
                })
            write_probe(record)
            keep_viz = (
                not cfg.sweep_compact_artifacts
                or round_i <= 10
                or round_i % 10 == 0
            )
            if keep_viz:
                torch.save({
                    "round": round_i,
                    "viz": viz,
                    "eps": [
                        {key: value for key, value in episode.items() if key != "step_stats"}
                        for episode in episodes
                    ],
                    "gp_buffer_query_ids": np.asarray(query_ids, dtype=np.int64),
                    "gp_diagnostics": gp_post_diagnostics,
                    "scene": scene,
                    "audit": audit,
                    "train_ids": np.asarray(sorted(drawn), dtype=np.int64),
                    "train_counts": np.asarray(
                        [drawn[key] for key in sorted(drawn)], dtype=np.int64
                    ),
                    "goal": goal,
                    "x0": env.x0.detach().cpu().numpy(),
                }, os.path.join(outdir, "viz_db", f"round{round_i}.pt"))
            keep_checkpoint = (
                not cfg.sweep_compact_artifacts
                or round_i % cfg.compact_checkpoint_every == 0
                or round_i == cfg.rounds
            )
            if keep_checkpoint:
                HT._save_hp_atomic(
                    policy, os.path.join(outdir, f"ckpt_{round_i}.pt"),
                    extra={"iter": round_i, "recipe": recipe, "resumable": False},
                )
            outcome = (
                f"V {audit['V']:.3f} SR {pooled['SR']:.2f} NVP {pooled['NVP']:.2f}"
                if cfg.training_probes else "training probes skipped"
            )
            print(
                f"[afe-rbf] r{round_i:03d} D {len(store)} D+ {store.n_pos()} "
                f"GP {gp_post.n}/{cfg.gp_cap} ESS/M {record.get('ess_med', float('nan')):.3f} "
                f"beta {beta_used:.4g}->{beta_next:.4g} "
                f"uplift {record.get('uplift_med', float('nan')):.4f} {outcome} "
                f"gather {gather_seconds:.1f}s update {update_seconds:.1f}s",
                flush=True,
            )
            gp_for_gather = gp_post
            gp_start_diagnostics = gp_post_diagnostics

    final_path = os.path.join(outdir, "final.pt")
    store_path = os.path.join(outdir, "dstore.pt")
    HT._save_hp_atomic(
        policy, final_path,
        extra={"iter": cfg.rounds, "recipe": recipe, "resumable": False},
    )
    if not cfg.sweep_compact_artifacts:
        store.save(store_path)
    else:
        negative_ids = np.flatnonzero(
            np.asarray(store.q_nvp_negative, dtype=np.int8)
        ).astype(np.int64)
        np.savez_compressed(
            os.path.join(outdir, "nvp_negative_archive.npz"),
            query_id=negative_ids,
            step_context_id=np.asarray(
                [store.q_sid[index] for index in negative_ids], dtype=np.int64
            ),
            round=np.asarray(
                [store.q_round[index] for index in negative_ids], dtype=np.int32
            ),
            gamma=np.asarray(
                [store.q_gamma[index] for index in negative_ids], dtype=np.float32
            ),
            full_y=np.asarray(
                [store.q_y[index] for index in negative_ids], dtype=np.int8
            ),
            terminal_exec_y=np.asarray(
                [store.q_exec_y[index] for index in negative_ids], dtype=np.int8
            ),
            controls=(
                np.stack([store.q_U[index] for index in negative_ids])
                if len(negative_ids)
                else np.zeros((0, GF.H_PRED, 2), dtype=np.float32)
            ),
            state=(
                np.stack([
                    store.ctx_state[store.q_sid[index]] for index in negative_ids
                ])
                if len(negative_ids)
                else np.zeros((0, 4), dtype=np.float32)
            ),
        )
    viz_rounds = (
        list(range(1, cfg.rounds + 1))
        if not cfg.sweep_compact_artifacts
        else [
            index for index in range(1, cfg.rounds + 1)
            if index <= 10 or index % 10 == 0
        ]
    )
    checkpoint_rounds = (
        list(range(cfg.rounds + 1))
        if not cfg.sweep_compact_artifacts
        else sorted({
            0,
            cfg.rounds,
            *range(
                cfg.compact_checkpoint_every,
                cfg.rounds + 1,
                cfg.compact_checkpoint_every,
            ),
        })
    )
    required = [
        "recipe.json",
        "rbf_calibration.json",
        "probe.jsonl",
        "final.pt",
        *([] if cfg.sweep_compact_artifacts else ["dstore.pt"]),
        *(["nvp_negative_archive.npz"] if cfg.sweep_compact_artifacts else []),
        *[f"ckpt_{index}.pt" for index in checkpoint_rounds],
        *[f"viz_db/round{index}.pt" for index in viz_rounds],
    ]
    inventory = {}
    for relative in required:
        path = os.path.join(outdir, relative)
        if not os.path.isfile(path):
            raise RuntimeError(f"completion artifact is missing: {relative}")
        inventory[relative] = AFE2._sha256_file(path)
    complete = {
        "status": "COMPLETE",
        "algorithm": recipe["algorithm"],
        "completed_round": cfg.rounds,
        "scene_sha256": scene["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_model_sha256": checkpoint_model_sha256,
        "checkpoint_contract_sha256": checkpoint_contract_sha256,
        "source_git_commit": source_git_state["commit"],
        "artifact_sha256": inventory,
    }
    with open(os.path.join(outdir, "COMPLETE.json"), "w") as stream:
        json.dump(complete, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"[afe-rbf] COMPLETE: {outdir}", flush=True)


def validate_protocol_args(args) -> None:
    """Fail closed on study-specific contracts without changing V1 defaults."""

    if args.protocol_profile == "v1":
        if args.K != 64 or args.B != 8 or args.batch != 128:
            raise ValueError("the first RBF study holds K=64, B=8, and batch=128 fixed")
        return
    support_profile = args.protocol_profile in {
        "v3_support_sweep", "v3_support_preflight"
    }
    if args.protocol_profile not in {
        "v2_smoke", "v2_lineage_mass_smoke",
        "v3_support_sweep", "v3_support_preflight",
    }:
        raise ValueError(f"unknown RBF protocol profile: {args.protocol_profile}")

    exact = {
        "scene_profile": "low7_radius1_canonical_v1",
        "rounds": (
            1 if args.protocol_profile == "v3_support_preflight"
            else 100 if args.protocol_profile == "v3_support_sweep"
            else 10
        ),
        "rollout_replicas": 8,
        "K": 16,
        "B": 4,
        "T": 300,
        "M_eval": 0,
        "batch": 128,
        "afe_steps": 0,
        "afe_lr": 1.0e-5,
        "gp_cap": 512,
        "gp_lam": 1.0e-2,
        "acquisition_mode": "sequential",
        "adaptive_ess_target": 0.5,
        "adaptive_beta_contexts_per_gamma": 64,
        "adaptive_beta_equalize_gammas": True,
        "replay_window": 2,
        "replay_sampling": "round_gamma_replica_context",
        "replay_update_mode": (
            "fixed_macro_steps_exact_epoch"
            if support_profile else "one_epoch_without_replacement"
        ),
        "replay_loss_weighting": (
            "gamma_episode_context_query_equal_mass"
            if args.protocol_profile in {
                "v2_lineage_mass_smoke", "v3_support_sweep", "v3_support_preflight"
            }
            else "query_uniform"
        ),
        "gp_replay_window": 2,
        "gp_replay_sampling": "round_gamma_replica_context",
        "lengthscale_multiplier": 1.0,
        "negative_alpha": 0.0,
        "execution_rule": (
            "nominal_hp_max_step_margin"
            if args.protocol_profile in {
                "v2_lineage_mass_smoke", "v3_support_sweep", "v3_support_preflight"
            }
            else "nominal_hp_max_step_margin_only"
        ),
        "conditioning_schema": CX.LOW7_SCHEMA,
        "freeze_visual_encoder": True,
        "skip_training_probes": True,
        "calibration_replicas": 8,
        "calibration_control_steps": 4,
        "sweep_compact_artifacts": True,
        "compact_checkpoint_every": 1,
        "route_metric_steps": 10,
        "route_ambiguity_band": RM.DEFAULT_AMBIGUITY_BAND,
        "nvp_audit_all_k": args.protocol_profile == "v2_lineage_mass_smoke",
    }
    if support_profile:
        if getattr(args, "optimizer_steps_per_round", 0) not in {16, 32}:
            raise ValueError("V3 support optimizer dose must be 16 or 32")
        if getattr(args, "demo_frac", 0.0) not in {0.0, 0.125, 0.25}:
            raise ValueError("V3 support demo objective mass must be 0, 0.125, or 0.25")
    elif (
        getattr(args, "optimizer_steps_per_round", 0) != 0
        or getattr(args, "demo_frac", 0.0) != 0.0
    ):
        raise ValueError("optimizer-dose/demo options are exclusive to V3 support profiles")
    mismatches = []
    for name, expected in exact.items():
        actual = getattr(args, name)
        if isinstance(expected, float):
            matches = np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-12)
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError("V2 smoke protocol mismatch: " + "; ".join(mismatches))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-profile",
        choices=(
            "v1", "v2_smoke", "v2_lineage_mass_smoke",
            "v3_support_sweep", "v3_support_preflight",
        ),
        default="v1",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--expected-ckpt-sha256", required=True)
    parser.add_argument("--scene-profile", choices=sorted(SCENE_PROFILES), required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rollout-replicas", type=int, default=2)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--M-eval", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--afe-steps", type=int, default=250)
    parser.add_argument("--afe-lr", type=float, default=1.0e-4)
    parser.add_argument("--gp-cap", type=int, default=512)
    parser.add_argument("--gp-lam", type=float, default=1.0e-2)
    parser.add_argument(
        "--acquisition-mode",
        choices=("sequential", "uniform"),
        default="sequential",
    )
    parser.add_argument("--adaptive-ess-target", type=float, default=None)
    parser.add_argument("--adaptive-beta-contexts-per-gamma", type=int, default=None)
    parser.add_argument("--adaptive-beta-equalize-gammas", action="store_true")
    parser.add_argument("--replay-window", type=int, default=None)
    parser.add_argument(
        "--replay-sampling",
        choices=("query_uniform", "round_gamma_replica_context"),
        default="query_uniform",
    )
    parser.add_argument(
        "--replay-update-mode",
        choices=(
            "fixed_steps_with_replacement", "one_epoch_without_replacement",
            "fixed_macro_steps_exact_epoch",
        ),
        default="fixed_steps_with_replacement",
    )
    parser.add_argument(
        "--replay-loss-weighting",
        choices=("query_uniform", "gamma_episode_context_query_equal_mass"),
        default="query_uniform",
    )
    parser.add_argument("--gp-replay-window", type=int, default=1)
    parser.add_argument(
        "--gp-replay-sampling",
        choices=("round_gamma", "round_gamma_replica_context"),
        default="round_gamma",
    )
    parser.add_argument("--lengthscale-multiplier", type=float, default=1.0)
    parser.add_argument("--negative-alpha", type=float, default=0.0)
    parser.add_argument(
        "--execution-rule",
        choices=(
            "legacy_max_horizon_progress",
            "nominal_hp_max_step_progress",
            "nominal_hp_max_step_margin",
            "nominal_hp_max_step_margin_only",
        ),
        default="legacy_max_horizon_progress",
    )
    parser.add_argument(
        "--conditioning-schema",
        choices=(CX.LOW5_SCHEMA, CX.LOW7_SCHEMA),
        default=CX.LOW5_SCHEMA,
    )
    parser.add_argument("--freeze-visual-encoder", action="store_true")
    parser.add_argument("--skip-training-probes", action="store_true")
    parser.add_argument("--calibration-replicas", type=int, default=None)
    parser.add_argument("--calibration-control-steps", type=int, default=None)
    parser.add_argument("--sweep-compact-artifacts", action="store_true")
    parser.add_argument("--compact-checkpoint-every", type=int, default=10)
    parser.add_argument("--route-metric-steps", type=int, default=0)
    parser.add_argument(
        "--route-ambiguity-band",
        type=float,
        default=RM.DEFAULT_AMBIGUITY_BAND,
    )
    parser.add_argument("--rbf-offline-sweep", action="store_true")
    parser.add_argument("--nvp-audit-all-k", action="store_true")
    parser.add_argument("--optimizer-steps-per-round", type=int, default=0)
    parser.add_argument("--demo-frac", type=float, default=0.0)
    parser.add_argument("--verifier-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=910)
    args = parser.parse_args()
    validate_protocol_args(args)
    if args.afe_lr <= 0.0:
        raise ValueError("AFE learning rate must be positive")
    if args.replay_update_mode == "fixed_steps_with_replacement":
        if args.afe_steps < 1:
            raise ValueError("fixed-step replay requires positive AFE steps")
    elif args.afe_steps != 0:
        raise ValueError("exact-epoch replay requires --afe-steps 0 as a non-operative sentinel")
    if args.replay_update_mode in {
        "one_epoch_without_replacement", "fixed_macro_steps_exact_epoch"
    } and args.negative_alpha != 0.0:
        raise ValueError("exact positive replay epoch currently requires negative alpha zero")
    if args.replay_loss_weighting != "query_uniform" and (
        args.replay_update_mode not in {
            "one_epoch_without_replacement", "fixed_macro_steps_exact_epoch"
        }
        or args.replay_sampling != "round_gamma_replica_context"
        or args.negative_alpha != 0.0
    ):
        raise ValueError(
            "hierarchical equal-mass loss requires hierarchical exact positive replay"
        )
    if args.nvp_audit_all_k and args.execution_rule == "legacy_max_horizon_progress":
        raise ValueError("all-K NVP audit requires nominal-Hp execution")
    if args.rounds < 1 or args.rollout_replicas < 1 or args.M_eval < 0:
        raise ValueError("rounds and rollout replicas must be positive; M-eval cannot be negative")
    if not args.skip_training_probes and args.M_eval < 1:
        raise ValueError("M-eval must be positive unless training probes are skipped")
    if args.verifier_workers < 1:
        raise ValueError("verifier worker count must be positive")
    if args.adaptive_ess_target is not None and not 0.0 < args.adaptive_ess_target < 1.0:
        raise ValueError("adaptive ESS target must lie strictly between zero and one")
    if (
        args.adaptive_beta_contexts_per_gamma is not None
        and args.adaptive_beta_contexts_per_gamma < 1
    ):
        raise ValueError("adaptive-beta context cap per gamma must be positive")
    if args.acquisition_mode == "uniform" and args.adaptive_ess_target is not None:
        raise ValueError("uniform acquisition does not use adaptive beta")
    if args.replay_window is not None and args.replay_window < 1:
        raise ValueError("replay window must be at least one round")
    if args.gp_replay_window < 1:
        raise ValueError("GP replay window must be at least one round")
    if not np.isfinite(args.lengthscale_multiplier) or args.lengthscale_multiplier <= 0.0:
        raise ValueError("length-scale multiplier must be finite and positive")
    if not np.isfinite(args.negative_alpha) or args.negative_alpha < 0.0:
        raise ValueError("negative alpha must be finite and nonnegative")
    if args.calibration_replicas is not None and args.calibration_replicas < 1:
        raise ValueError("calibration replicas must be positive")
    if args.calibration_control_steps is not None and args.calibration_control_steps < 1:
        raise ValueError("calibration control steps must be positive")
    if args.compact_checkpoint_every < 1:
        raise ValueError("compact checkpoint interval must be positive")
    if args.route_metric_steps < 0:
        raise ValueError("route metric step count must be nonnegative")
    if (
        not np.isfinite(args.route_ambiguity_band)
        or args.route_ambiguity_band < 0.0
    ):
        raise ValueError("route ambiguity band must be finite and nonnegative")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_sha256 = AFE2._sha256_file(args.ckpt)
    if checkpoint_sha256 != args.expected_ckpt_sha256.lower():
        raise ValueError(
            f"checkpoint hash {checkpoint_sha256} != expected {args.expected_ckpt_sha256.lower()}"
        )
    policy, checkpoint = HP.load_hp(args.ckpt, device="cpu")
    policy = policy.to(device)
    profile = get_scene_profile(args.scene_profile)
    checkpoint_model_sha256, checkpoint_contract, checkpoint_contract_sha256 = (
        AFE2.validate_checkpoint_contract(
            profile.name, policy, checkpoint, checkpoint_sha256
        )
    )
    policy_contract = CX.require_declared_contract(
        policy,
        args.conditioning_schema,
        7 if args.conditioning_schema == CX.LOW7_SCHEMA else 5,
    )
    if profile.name in {
        "low7_radius1_canonical_v1",
        "low7_radius03_canonical_v1",
    }:
        if args.conditioning_schema != CX.LOW7_SCHEMA or not args.freeze_visual_encoder:
            raise ValueError(
                "low7 OOD expansion requires low7 closest-boundary conditioning "
                "and a frozen visual encoder"
            )
    elif args.conditioning_schema != CX.LOW5_SCHEMA or args.freeze_visual_encoder:
        raise ValueError("legacy scenes retain low5 conditioning and a trainable encoder")
    trainability = configure_policy_trainability(policy, args.freeze_visual_encoder)
    source_git_state = AFE2._git_state()
    if (
        source_git_state["commit"] is None
        or source_git_state["tracked_dirty"] is not False
        or source_git_state["untracked_runtime_sources"] != []
    ):
        raise RuntimeError(
            "AFE-RBF requires committed clean source; "
            f"untracked runtime sources={source_git_state['untracked_runtime_sources']}"
        )
    env = build_scene(profile)
    GM2.GOAL_XY = np.asarray(profile.goal, dtype=float)
    cfg = AFERBFConfig(
        protocol_profile=args.protocol_profile,
        rounds=args.rounds,
        T=args.T,
        K=args.K,
        B=args.B,
        arm="afe",
        batch=args.batch,
        afe_steps=args.afe_steps,
        afe_lr=args.afe_lr,
        M_eval=args.M_eval,
        wall_plugs=profile.wall_plugs,
        start_eps=profile.start[0],
        goal_xy=profile.goal,
        scene_profile=profile.name,
        seed=args.seed,
        replicas=args.rollout_replicas,
        gp_cap=args.gp_cap,
        gp_lam=args.gp_lam,
        verifier_workers=args.verifier_workers,
        acquisition_mode=args.acquisition_mode,
        adaptive_ess_target=args.adaptive_ess_target,
        adaptive_beta_contexts_per_gamma=(
            args.adaptive_beta_contexts_per_gamma
        ),
        adaptive_beta_equalize_gammas=args.adaptive_beta_equalize_gammas,
        replay_window=args.replay_window,
        replay_sampling=args.replay_sampling,
        replay_update_mode=args.replay_update_mode,
        replay_loss_weighting=args.replay_loss_weighting,
        gp_replay_window=args.gp_replay_window,
        gp_replay_sampling=args.gp_replay_sampling,
        lengthscale_multiplier=args.lengthscale_multiplier,
        negative_alpha=args.negative_alpha,
        execution_rule=args.execution_rule,
        training_probes=not args.skip_training_probes,
        calibration_replicas=args.calibration_replicas,
        calibration_control_steps=args.calibration_control_steps,
        sweep_compact_artifacts=args.sweep_compact_artifacts,
        compact_checkpoint_every=args.compact_checkpoint_every,
        route_metric_steps=args.route_metric_steps,
        route_ambiguity_band=args.route_ambiguity_band,
        rbf_offline_sweep=args.rbf_offline_sweep,
        nvp_audit_all_k=args.nvp_audit_all_k,
        optimizer_steps_per_round=args.optimizer_steps_per_round,
        demo_frac=args.demo_frac,
        conditioning_schema=policy_contract.schema,
        raw_condition_dim=policy_contract.raw_condition_dim,
        freeze_visual_encoder=args.freeze_visual_encoder,
    )
    print(
        f"[afe-rbf] scene={profile.name} rounds={cfg.rounds} replicas/gamma={cfg.replicas} "
        f"K={cfg.K} B={cfg.B} GPcap={cfg.gp_cap} workers={cfg.verifier_workers} "
        f"acquisition={cfg.acquisition_mode} adaptive_ESS={cfg.adaptive_ess_target} "
        f"replay_W={cfg.replay_window} replay_mode={cfg.replay_update_mode} "
        f"replay_weighting={cfg.replay_loss_weighting} "
        f"gp_W={cfg.gp_replay_window} "
        f"ell_mult={cfg.lengthscale_multiplier:g} alpha={cfg.negative_alpha:g} "
        f"steps={cfg.optimizer_steps_per_round if cfg.optimizer_steps_per_round else ('dynamic' if cfg.afe_steps == 0 else cfg.afe_steps)} "
        f"exec={cfg.execution_rule} nvp_allK_audit={cfg.nvp_audit_all_k} "
        f"frozen={len(trainability['frozen'])}",
        flush=True,
    )
    run(
        policy, env, cfg, device, args.outdir,
        args.ckpt, checkpoint_sha256, checkpoint_model_sha256,
        checkpoint_contract, checkpoint_contract_sha256, source_git_state,
    )


if __name__ == "__main__":
    main()
