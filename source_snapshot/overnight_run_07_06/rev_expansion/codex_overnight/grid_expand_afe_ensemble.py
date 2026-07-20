"""Single-arm neural-ensemble Safe Flow Expansion.

This runner changes the uncertainty estimator and its necessary cold start;
the validated AFE control, verifier, execution, and CFM pipeline is shared.
Round 1 is an explicit uniform-query bootstrap.  After
each flow update, all cumulative successful verifier queries are re-embedded
with the current flow representation and the reference AFE five-MLP ensemble
is refit from scratch. Positive queries alone enter the append-only D+ archive;
CFM replay is cumulative by default and may use an explicit recent-round window.
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
import grid_hp_expt as HP
import grid_metrics2 as GM2
import grid_expand_hardtail as HT

import afe_core as AC
import afe_context as CX
import afe_adaptive as AD
import afe2_calibration as BC
import afe_ensemble_core as EC
import afe_rbf_core as RC
import grid_expand_afe2 as AFE2
import grid_expand_afe_rbf as RBF
from afe2_scene_profiles import (
    SCENE_PROFILES,
    assert_scene_snapshot,
    build_scene,
    get_scene_profile,
    scene_snapshot,
)


@dataclass
class AFEEnsembleConfig(RBF.AFERBFConfig):
    ensemble_members: int = 5
    ensemble_hidden: int = 100
    ensemble_dropout: float = 0.1
    ensemble_train_fraction: float = 0.9
    ensemble_lr: float = 1.0e-3
    ensemble_steps: int = 1000
    ensemble_early_window: int = 30
    plumbing_smoke: bool = False


def configure_policy_trainability(policy, freeze_visual_encoder: bool) -> dict:
    """Freeze exactly E(H_P), leaving the flow trunk and head trainable."""

    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    if freeze_visual_encoder:
        for parameter in policy.enc_grid.parameters():
            parameter.requires_grad_(False)
    frozen = {
        name for name, parameter in policy.named_parameters() if not parameter.requires_grad
    }
    encoder_names = {f"enc_grid.{name}" for name, _ in policy.enc_grid.named_parameters()}
    if freeze_visual_encoder:
        if frozen != encoder_names:
            raise RuntimeError(
                f"only the visual grid encoder may be frozen; got {sorted(frozen)}"
            )
    elif frozen:
        raise RuntimeError(f"unexpected frozen policy parameters: {sorted(frozen)}")
    trainable = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    if not any(name.startswith("trunk.") for name in trainable):
        raise RuntimeError("flow trunk is not trainable")
    if not any(name.startswith("head.") for name in trainable):
        raise RuntimeError("flow output head is not trainable")
    return {"frozen": sorted(frozen), "trainable": sorted(trainable)}


def _new_estimator(cfg, device):
    return EC.DeepEnsembleSigma(
        feature_dim=32,
        members=cfg.ensemble_members,
        hidden_dim=cfg.ensemble_hidden,
        dropout=cfg.ensemble_dropout,
        train_fraction=cfg.ensemble_train_fraction,
        learning_rate=cfg.ensemble_lr,
        max_steps=cfg.ensemble_steps,
        early_window=cfg.ensemble_early_window,
        device=device,
    )


def _label_counts(store, cfg):
    output = {}
    gamma_storage_map = CX.declared_gamma_storage_map(cfg.gammas)
    for gamma in cfg.gammas:
        gamma_key = CX.canonical_declared_gamma(gamma, gamma_storage_map)
        ids = [
            index for index, value in enumerate(store.q_gamma)
            if CX.canonical_declared_gamma(value, gamma_storage_map) == gamma_key
        ]
        positives = int(sum(store.q_y[index] for index in ids))
        output[str(gamma)] = {
            "total": len(ids),
            "positive": positives,
            "negative": len(ids) - positives,
        }
    return output


def _fit_estimator(policy, store, cfg, device, round_i):
    if len(store) < 2:
        raise RuntimeError("ensemble fit requires at least two verifier queries")
    labels = torch.as_tensor(store.q_y, dtype=torch.float32, device=device)
    reembed_started = time.perf_counter()
    features = AFE2.embed_queries(
        policy, store, cfg, device, ids=list(range(len(store)))
    ).to(device)
    reembed_seconds = time.perf_counter() - reembed_started
    estimator = _new_estimator(cfg, device)
    with AC.isolated_random_state(AFE2.named_seed(
        cfg.seed, "ensemble_refit", round_i
    )):
        diagnostics = estimator.fit(features, labels)
    diagnostics["reembed_seconds"] = float(reembed_seconds)
    diagnostics["per_gamma_labels"] = _label_counts(store, cfg)
    estimator.fit_diagnostics = dict(diagnostics)
    return estimator, diagnostics


@torch.no_grad()
def _beta_score_vectors(policy, estimator, store, cfg, device):
    """Unverified, beta-neutral candidate pools at round-1 bootstrap contexts."""

    vectors = []
    gamma_counts = {}
    chunk_size = 16
    for begin in range(0, len(store.ctx_state), chunk_size):
        sids = list(range(begin, min(begin + chunk_size, len(store.ctx_state))))
        grid = store.grid3_of(sids).to(device)
        low = torch.stack([
            torch.from_numpy(store.ctx_low5[sid]) for sid in sids
        ]).to(device)
        hist = torch.stack([
            torch.from_numpy(store.ctx_hist[sid].astype(np.float32)) for sid in sids
        ]).to(device)
        context = policy.ctx_from(grid, low, hist)
        repeated = context.repeat_interleave(cfg.K, dim=0)
        with AC.isolated_random_state(AFE2.named_seed(
            cfg.seed, "ensemble_beta_candidates", begin
        )):
            controls = policy.sample(
                len(sids) * cfg.K, repeated, nfe=cfg.nfe, temp=cfg.temp
            )
        features = EC.l2_normalize(
            policy.phi_s(controls, repeated, s=cfg.s)
        ).reshape(len(sids), cfg.K, -1)
        for local_index, sid in enumerate(sids):
            rng = np.random.default_rng(AFE2.named_seed(
                cfg.seed, "ensemble_beta_order", sid
            ))
            order = torch.as_tensor(
                rng.permutation(cfg.K), device=device, dtype=torch.long
            )
            vectors.extend([
                score.detach().cpu().numpy()
                for score in estimator.sequential_score_vectors(
                    features[local_index], order, min(cfg.B, cfg.K)
                )
            ])
            gamma = str(round(float(store.ctx_low5[sid][-1]), 2))
            gamma_counts[gamma] = gamma_counts.get(gamma, 0) + 1
    if not vectors:
        raise RuntimeError("ensemble beta calibration produced no score vectors")
    return vectors, gamma_counts


def _write_json(path, value):
    with open(path, "w") as stream:
        json.dump(AFE2._json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _save_estimator(path, estimator, round_i, beta, source_git_commit):
    temporary = f"{path}.tmp"
    torch.save({
        "round": int(round_i),
        "beta": (None if beta is None else float(beta)),
        "source_git_commit": source_git_commit,
        "estimator": estimator.state_dict(),
    }, temporary)
    os.replace(temporary, path)


def _recipe(cfg, env, checkpoint_path, checkpoint_sha256, checkpoint_model_sha256,
            checkpoint_contract, checkpoint_contract_sha256, source_git_state, device):
    profile = get_scene_profile(cfg.scene_profile)
    scene = scene_snapshot(env, profile)
    adaptive = cfg.adaptive_ess_target is not None
    learning_memory = (
        "uniform replay over complete cumulative full-H D+ only"
        if cfg.replay_window is None
        else (
            f"uniform replay over full-H positives from the current and previous "
            f"{cfg.replay_window - 1} rounds; cumulative D+ archive is retained"
        )
    )
    return {
        "algorithm": (
            "afe_low7_deep_ensemble_adaptive_ess_parallel_v1"
            if cfg.conditioning_schema == CX.LOW7_SCHEMA
            else (
                "afe_deep_ensemble_adaptive_ess_parallel_v2"
                if adaptive else "afe_deep_ensemble_parallel_v1"
            )
        ),
        "arm": "afe",
        "single_arm": True,
        "kernel": None,
        "uncertainty_estimator": "deep_bootstrapped_ensemble",
        "ensemble": {
            "members": cfg.ensemble_members,
            "architecture": [32, cfg.ensemble_hidden, cfg.ensemble_hidden, 1],
            "activation": "ReLU",
            "dropout": cfg.ensemble_dropout,
            "train_fraction": cfg.ensemble_train_fraction,
            "subsample_semantics": "independent randperm 90% without replacement",
            "objective": "MSE on globally standardized deterministic verifier labels",
            "optimizer": "Adam",
            "learning_rate": cfg.ensemble_lr,
            "max_steps": cfg.ensemble_steps,
            "early_stopping_window": cfg.ensemble_early_window,
            "refit": "from scratch after every CFM update on current-phi cumulative D",
        },
        "bootstrap": (
            "round 1 uses uniform B-without-replacement acquisition; its normal verifier "
            "queries enter cumulative D and positives enter D+; no hidden verifier archive"
        ),
        "beta": None,
        "beta_protocol": (
            f"after every refit, calibrate beta_(n+1) on beta-neutral current-policy pools at "
            f"all contexts visited in round n for median ESS={cfg.adaptive_ess_target:g}; "
            "no verifier labels or verifier calls; beta is frozen during the next round"
            if adaptive else (
                "after round-1 refit, solve once for stage-normalized ESS/M_remaining=0.375 "
                "using beta-neutral random removal orders on unverified candidate pools at stored "
                "bootstrap contexts; fixed thereafter; realized Gibbs ESS is logged separately"
            )
        ),
        "adaptive_ess_target": cfg.adaptive_ess_target,
        "acquisition_memory": (
            "all successful full-verifier queries with binary labels; cumulative, re-embedded "
            "under current phi; frozen ensemble during each parallel gather"
        ),
        "learning_memory": learning_memory,
        "replay_window": cfg.replay_window,
        "conditioning": {
            "schema": cfg.conditioning_schema,
            "raw_condition_dim": cfg.raw_condition_dim,
            "ctx_dim": 39 if cfg.conditioning_schema == CX.LOW7_SCHEMA else 37,
            "trunk_input_dim": (
                91 if cfg.conditioning_schema == CX.LOW7_SCHEMA else 89
            ),
            "gamma_last": True,
            "builder": (
                "afe_restart.scene.context_from_state_low7"
                if cfg.conditioning_schema == CX.LOW7_SCHEMA
                else "afe_restart.scene.context_from_state"
            ),
        },
        "freeze_visual_encoder": bool(cfg.freeze_visual_encoder),
        "trainable_modules": (
            ["trunk", "head"]
            if cfg.freeze_visual_encoder else ["visual_grid_encoder", "trunk", "head"]
        ),
        "video_include_round0": bool(cfg.conditioning_schema == CX.LOW7_SCHEMA),
        "plumbing_smoke": bool(cfg.plumbing_smoke),
        "uncertainty_meaning": (
            "population standard deviation of five raw verifier-label regressors; epistemic "
            "ensemble disagreement, not validity probability and not a safety certificate"
        ),
        "selection": (
            "K=64 scored once; B=8 Gibbs draws without replacement; no GP-style posterior "
            "conditioning after a selected but still-unlabeled candidate"
        ),
        "sigma_usage": "verifier-query acquisition only; no replay reweighting",
        "label_separation": (
            "safety is the deterministic full verifier label; progress ranks only "
            "execution-admissible queries; training accepts full-H positives only"
        ),
        "socp_error": (
            "any full-H or terminal-prefix SOCP error updates neither D nor D+"
        ),
        "parallel_sampling": (
            f"{cfg.replicas} closed-loop replicas per gamma advanced synchronously; one GPU "
            f"proposal batch per control tick; {cfg.verifier_workers} persistent CPU verifiers"
        ),
        "execution": (
            "maximum-progress terminal-aware verified plan; execute first action; absorbing "
            "goal prefix allowed only for execution; NVP terminates; no expert/fallback"
        ),
        "update": f"CFM lr {cfg.afe_lr:g}, batch {cfg.batch}, {cfg.afe_steps} steps, no prox",
        "rounds": cfg.rounds,
        "rollout_replicas": cfg.replicas,
        "T": cfg.T,
        "K": cfg.K,
        "B": cfg.B,
        "batch": cfg.batch,
        "afe_lr": cfg.afe_lr,
        "afe_steps": cfg.afe_steps,
        "s": cfg.s,
        "nfe": cfg.nfe,
        "M_eval": cfg.M_eval,
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
            "faithful to the public AFE molecule/protein neural uncertainty estimator; "
            "finite-K verifier acquisition is the control-specific adaptation"
        ),
        "theory_scope": (
            "does not inherit exact GP posterior or information-gain guarantees"
        ),
        "complexity": (
            "cumulative storage/re-embedding O(N); reference full-batch ensemble refit linear "
            "in N per gradient step and O(R^2) total when N grows linearly with rounds; query "
            "cost independent of N"
        ),
        "no_curriculum": True,
        "no_anchor": True,
        "no_prox": True,
        "no_fallback": True,
        "no_expert_replay": True,
        "no_rollback": True,
        "no_curated_recovery_starts": True,
    }


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
    recipe = _recipe(
        cfg, env, checkpoint_path, checkpoint_sha256, checkpoint_model_sha256,
        checkpoint_contract, checkpoint_contract_sha256, source_git_state, device,
    )
    recipe_path = os.path.join(outdir, "recipe.json")
    _write_json(recipe_path, recipe)

    CX.require_declared_contract(
        policy, cfg.conditioning_schema, cfg.raw_condition_dim
    )
    store = AC.DStore(
        conditioning_schema=cfg.conditioning_schema,
        condition_dim=cfg.raw_condition_dim,
    )
    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("expansion has no trainable policy parameters")
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.afe_lr)
    audit_contexts = AC.build_audit_contexts(
        env,
        cfg.gammas,
        n_pos=cfg.audit_pos,
        conditioning_schema=cfg.conditioning_schema,
    )
    representation_probe = AFE2.rep_probe_build(policy, env, cfg, device)
    goal = env.goal.detach().cpu().numpy()
    estimator = _new_estimator(cfg, device)
    probe_path = os.path.join(outdir, "probe.jsonl")

    def write_probe(record):
        with open(probe_path, "a") as stream:
            stream.write(json.dumps(AFE2._json_safe(record), allow_nan=False) + "\n")

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=cfg.verifier_workers,
        mp_context=context,
        initializer=RC.initialize_verifier_worker,
        initargs=(cfg.scene_profile, cfg.reach, cfg.n_theta),
    ) as executor:
        audit0 = AC.run_audit(
            policy, audit_contexts, env, goal, device,
            n_plans=cfg.audit_plans, nfe=cfg.nfe, n_theta=cfg.n_theta,
            seed=AFE2.named_seed(cfg.seed, "audit"),
        )
        viz0 = []
        eval0, eval0_timing = RBF.run_parallel_episodes(
            policy, estimator, env, cfg, store, 0, cfg.M_eval, device, executor,
            collect=False, viz=viz0, purpose="controller_eval", acquisition_mode="uniform",
        )
        rows0, pooled0 = RBF._controller_summary(eval0, cfg, env)
        write_probe({
            "round": 0,
            "arm": "afe",
            "acquisition_mode": "uniform_unfit",
            "beta_used": None,
            "beta_next": None,
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
            "ensemble": estimator.diagnostics(),
            "rep_cos": 1.0,
            "evaluation_timing": eval0_timing,
        })
        HT._save_hp_atomic(
            policy, os.path.join(outdir, "ckpt_0.pt"),
            extra={"iter": 0, "recipe": recipe, "resumable": False},
        )
        _save_estimator(
            os.path.join(outdir, "ensemble_round0.pt"), estimator, 0, None,
            source_git_state["commit"],
        )
        if recipe["video_include_round0"]:
            torch.save({
                "round": 0,
                "viz": viz0,
                "eps": [
                    {key: value for key, value in episode.items() if key != "step_stats"}
                    for episode in eval0
                ],
                "ensemble_diagnostics": estimator.diagnostics(),
                "acquisition_ensemble_checkpoint": "ensemble_round0.pt",
                "post_update_ensemble_checkpoint": "ensemble_round0.pt",
                "scene": scene,
                "audit": audit0,
                "train_ids": np.zeros(0, dtype=np.int64),
                "train_counts": np.zeros(0, dtype=np.int64),
                "goal": goal,
                "x0": env.x0.detach().cpu().numpy(),
            }, os.path.join(outdir, "viz_db", "round0.pt"))
        print(
            f"[afe-ensemble] r000 V {audit0['V']:.3f} ctrl SR {pooled0['SR']:.2f} "
            f"NVP {pooled0['NVP']:.2f} estimator=unfit",
            flush=True,
        )

        estimator_start_diagnostics = estimator.diagnostics()
        for round_i in range(1, cfg.rounds + 1):
            round_started = time.perf_counter()
            beta_used = None if round_i == 1 else float(cfg.beta)
            policy.eval()
            viz = []
            acquisition_mode = "uniform" if round_i == 1 else "sequential"
            episodes, gather_timing = RBF.run_parallel_episodes(
                policy, estimator, env, cfg, store, round_i, cfg.replicas,
                device, executor, collect=True, viz=viz, purpose="gather",
                acquisition_mode=acquisition_mode,
            )
            gather_seconds = time.perf_counter() - round_started
            per_gamma = RBF._per_gamma_episode_stats(episodes, cfg)
            acquisition = RBF._aggregate_step_stats(episodes, cfg)

            update_started = time.perf_counter()
            replay_rng = np.random.default_rng(AFE2.named_seed(cfg.seed, "replay", round_i))
            with AC.isolated_random_state(AFE2.named_seed(cfg.seed, "update", round_i)):
                update = AFE2.update_round(
                    policy, optimizer, store, cfg, device, replay_rng, round_i
                )
            update_seconds = time.perf_counter() - update_started
            policy.eval()

            estimator, estimator_diagnostics = _fit_estimator(
                policy, store, cfg, device, round_i
            )
            beta_calibration = None
            if cfg.adaptive_ess_target is not None:
                calibration_started = time.perf_counter()
                calibration_pools, context_gamma_counts = AD.feature_pools(
                    policy, store, cfg, device, round_i
                )
                beta_calibration = AD.calibrate_from_pools(
                    estimator,
                    calibration_pools,
                    cfg,
                    round_i,
                    cfg.adaptive_ess_target,
                )
                beta_calibration["context_gamma_counts"] = context_gamma_counts
                beta_calibration["candidate_pools_enter_D_or_Dplus"] = False
                beta_calibration["seconds"] = float(
                    time.perf_counter() - calibration_started
                )
                cfg.beta = float(beta_calibration["beta"])
            elif round_i == 1:
                calibration_started = time.perf_counter()
                score_vectors, context_gamma_counts = _beta_score_vectors(
                    policy, estimator, store, cfg, device
                )
                solution = BC.solve_beta_ragged(score_vectors)
                cfg.beta = float(solution["beta"])
                beta_calibration = {
                    "status": "CALIBRATED_AFE_DEEP_ENSEMBLE_V1",
                    "beta": cfg.beta,
                    "solution": solution,
                    "score_vector_sha256": BC.score_vectors_sha256(score_vectors),
                    "score_vector_count": len(score_vectors),
                    "context_count": len(store.ctx_state),
                    "context_gamma_counts": context_gamma_counts,
                    "verifier_queries": 0,
                    "candidate_pools_enter_D_or_Dplus": False,
                    "seconds": float(time.perf_counter() - calibration_started),
                }
            if round_i == 1:
                recipe["beta"] = cfg.beta
                recipe["beta_calibration"] = beta_calibration
                _write_json(os.path.join(outdir, "ensemble_calibration.json"), beta_calibration)
                _write_json(recipe_path, recipe)
            beta_next = float(cfg.beta)

            _save_estimator(
                os.path.join(outdir, f"ensemble_round{round_i}.pt"),
                estimator, round_i, cfg.beta, source_git_state["commit"],
            )

            audit_started = time.perf_counter()
            audit = AC.run_audit(
                policy, audit_contexts, env, goal, device,
                n_plans=cfg.audit_plans, nfe=cfg.nfe, n_theta=cfg.n_theta,
                seed=AFE2.named_seed(cfg.seed, "audit"),
            )
            audit_seconds = time.perf_counter() - audit_started
            controller_eval_started = time.perf_counter()
            evaluation, evaluation_timing = RBF.run_parallel_episodes(
                policy, estimator, env, cfg, store, round_i, cfg.M_eval,
                device, executor, collect=False, viz=None, purpose="controller_eval",
                acquisition_mode="sequential",
            )
            controller_eval_seconds = time.perf_counter() - controller_eval_started
            rows, pooled = RBF._controller_summary(evaluation, cfg, env)
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
                "acquisition_mode": (
                    "uniform_bootstrap" if round_i == 1 else "ensemble_tilt"
                ),
                "beta_used": beta_used,
                "beta_next": beta_next,
                "n_D": len(store),
                "n_Dpos": store.n_pos(),
                "per_gamma": per_gamma,
                **acquisition,
                "ensemble_round_start": estimator_start_diagnostics,
                "ensemble": estimator_diagnostics,
                "beta": cfg.beta,
                "beta_calibration": beta_calibration,
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
                "t_ensemble": (
                    estimator_diagnostics["reembed_seconds"]
                    + estimator_diagnostics["fit_seconds"]
                ),
                "t_beta_calibration": (
                    0.0 if beta_calibration is None else beta_calibration["seconds"]
                ),
                "t_audit": audit_seconds,
                "t_controller_eval": controller_eval_seconds,
                "gather_timing": gather_timing,
                "evaluation_timing": evaluation_timing,
            }
            if update is not None:
                record.update({
                    "steps": update["steps"],
                    "stop": update["stop"],
                    "cfm": update["cfm"],
                    "cfm_first": update["cfm_first"],
                    "cfm_last": update["cfm_last"],
                    "fstep_final": update["fstep_final"],
                    "fstep_max": update["fstep_max"],
                    "grad_norm": update["grad_norm"],
                    "rel_param_change": update["rel_param_change"],
                    "replay_window": update["replay_window"],
                    "replay_eligible": update["replay_eligible"],
                    "replay_fresh_draws": update["replay_fresh_draws"],
                    "replay_fresh_distinct": update["replay_fresh_distinct"],
                    "replay_fresh_fraction": update["replay_fresh_fraction"],
                    "replay_eligible_round_counts": update[
                        "replay_eligible_round_counts"
                    ],
                    "replay_draw_round_counts": update["replay_draw_round_counts"],
                })
            checkpoint_started = time.perf_counter()
            torch.save({
                "round": round_i,
                "viz": viz,
                "eps": [
                    {key: value for key, value in episode.items() if key != "step_stats"}
                    for episode in episodes
                ],
                "ensemble_diagnostics": estimator_diagnostics,
                "acquisition_ensemble_checkpoint": f"ensemble_round{round_i - 1}.pt",
                "post_update_ensemble_checkpoint": f"ensemble_round{round_i}.pt",
                "scene": scene,
                "audit": audit,
                "train_ids": np.asarray(sorted(drawn), dtype=np.int64),
                "train_counts": np.asarray(
                    [drawn[key] for key in sorted(drawn)], dtype=np.int64
                ),
                "goal": goal,
                "x0": env.x0.detach().cpu().numpy(),
            }, os.path.join(outdir, "viz_db", f"round{round_i}.pt"))
            HT._save_hp_atomic(
                policy, os.path.join(outdir, f"ckpt_{round_i}.pt"),
                extra={"iter": round_i, "recipe": recipe, "resumable": False},
            )
            record["t_checkpoint"] = time.perf_counter() - checkpoint_started
            record["t_round_total"] = time.perf_counter() - round_started
            write_probe(record)
            print(
                f"[afe-ensemble] r{round_i:03d} D {len(store)} D+ {store.n_pos()} "
                f"labels+ {estimator_diagnostics['positive_fraction']:.3f} "
                f"ESS/M {record.get('ess_med', float('nan')):.3f} "
                f"beta {beta_used}->{beta_next:.4g} "
                f"uplift {record.get('uplift_med', float('nan')):.4f} V {audit['V']:.3f} "
                f"SR {pooled['SR']:.2f} NVP {pooled['NVP']:.2f} "
                f"gather {gather_seconds:.1f}s CFM {update_seconds:.1f}s "
                f"ensemble {estimator_diagnostics['reembed_seconds'] + estimator_diagnostics['fit_seconds']:.1f}s",
                flush=True,
            )
            estimator_start_diagnostics = estimator_diagnostics

    final_path = os.path.join(outdir, "final.pt")
    store_path = os.path.join(outdir, "dstore.pt")
    HT._save_hp_atomic(
        policy, final_path,
        extra={"iter": cfg.rounds, "recipe": recipe, "resumable": False},
    )
    store.save(store_path)
    required = [
        "recipe.json",
        "ensemble_calibration.json",
        "probe.jsonl",
        "final.pt",
        "dstore.pt",
        *[f"ckpt_{index}.pt" for index in range(cfg.rounds + 1)],
        *[f"ensemble_round{index}.pt" for index in range(cfg.rounds + 1)],
        *[
            f"viz_db/round{index}.pt"
            for index in range(
                0 if recipe["video_include_round0"] else 1,
                cfg.rounds + 1,
            )
        ],
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
    _write_json(os.path.join(outdir, "COMPLETE.json"), complete)
    print(f"[afe-ensemble] COMPLETE: {outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--adaptive-ess-target", type=float, default=None)
    parser.add_argument("--replay-window", type=int, default=None)
    parser.add_argument("--verifier-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=910)
    parser.add_argument(
        "--conditioning-schema",
        choices=(CX.LOW5_SCHEMA, CX.LOW7_SCHEMA),
        default=CX.LOW5_SCHEMA,
    )
    parser.add_argument("--freeze-visual-encoder", action="store_true")
    parser.add_argument("--plumbing-smoke", action="store_true")
    args = parser.parse_args()
    if args.K != 64 or args.B != 8 or args.batch != 128:
        raise ValueError("the first ensemble study holds K=64, B=8, and batch=128 fixed")
    if args.afe_steps != 250 or args.afe_lr != 1.0e-4:
        raise ValueError("the first ensemble study holds CFM at 250 steps and lr=1e-4")
    if args.rounds < 1 or args.rollout_replicas < 1 or args.M_eval < 1:
        raise ValueError("rounds, rollout replicas, and M-eval must be positive")
    if args.verifier_workers < 1:
        raise ValueError("verifier worker count must be positive")
    if args.adaptive_ess_target is not None and not 0.0 < args.adaptive_ess_target < 1.0:
        raise ValueError("adaptive ESS target must lie strictly between zero and one")
    if args.replay_window is not None and args.replay_window < 1:
        raise ValueError("replay window must be at least one round")
    if args.conditioning_schema == CX.LOW7_SCHEMA:
        expected_rounds = 1 if args.plumbing_smoke else 100
        exact = {
            "rounds": (args.rounds, expected_rounds),
            "rollout_replicas": (args.rollout_replicas, 2),
            "M_eval": (args.M_eval, 2),
            "verifier_workers": (args.verifier_workers, 16),
            "T": (args.T, 300),
            "adaptive_ess_target": (args.adaptive_ess_target, 0.5),
            "replay_window": (args.replay_window, 5),
            "seed": (args.seed, 910),
        }
        mismatches = {
            key: values for key, values in exact.items() if values[0] != values[1]
        }
        if mismatches:
            raise ValueError(f"low7 canonical recipe mismatch: {mismatches}")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    if profile.name == "low7_radius1_canonical_v1":
        if args.conditioning_schema != CX.LOW7_SCHEMA or not args.freeze_visual_encoder:
            raise ValueError(
                "low7 radius-1 expansion requires exact low7 conditioning and a frozen visual encoder"
            )
    elif args.conditioning_schema != CX.LOW5_SCHEMA or args.freeze_visual_encoder:
        raise ValueError("legacy scenes retain their declared low5/all-trainable contract")
    configure_policy_trainability(policy, args.freeze_visual_encoder)
    source_git_state = AFE2._git_state()
    if (
        source_git_state["commit"] is None
        or source_git_state["tracked_dirty"] is not False
        or source_git_state["untracked_runtime_sources"] != []
    ):
        raise RuntimeError(
            "AFE-ensemble requires committed clean source; "
            f"untracked runtime sources={source_git_state['untracked_runtime_sources']}"
        )
    env = build_scene(profile)
    GM2.GOAL_XY = np.asarray(profile.goal, dtype=float)
    cfg = AFEEnsembleConfig(
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
        verifier_workers=args.verifier_workers,
        adaptive_ess_target=args.adaptive_ess_target,
        replay_window=args.replay_window,
        conditioning_schema=policy_contract.schema,
        raw_condition_dim=policy_contract.raw_condition_dim,
        freeze_visual_encoder=args.freeze_visual_encoder,
        plumbing_smoke=args.plumbing_smoke,
    )
    print(
        f"[afe-ensemble] scene={profile.name} rounds={cfg.rounds} "
        f"replicas/gamma={cfg.replicas} K={cfg.K} B={cfg.B} "
        f"ensemble=5x100x100 workers={cfg.verifier_workers} "
        f"adaptive_ESS={cfg.adaptive_ess_target} replay_W={cfg.replay_window}",
        flush=True,
    )
    run(
        policy, env, cfg, device, args.outdir,
        args.ckpt, checkpoint_sha256, checkpoint_model_sha256,
        checkpoint_contract, checkpoint_contract_sha256, source_git_state,
    )


if __name__ == "__main__":
    main()
