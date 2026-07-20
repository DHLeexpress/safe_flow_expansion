#!/usr/bin/env python3
"""Stage 05: minimal planned-window AFE expansion with fixed held-out audits."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

import grid_hp_expt as HP

from .ablations import MatchedProtocol
from .config import (
    AFEConfig,
    FeatureConfig,
    SamplingConfig,
    clean_method_absence_manifest,
)
from .controller import EpisodeResult, PlannedWindowAFEController
from .deps import sha256_file
from .evaluation import rollout_ordinary_flow, summarize_rollouts
from .fallback import SafeMPPIBackup
from .policy import (
    FrozenFeatureModel,
    ledger_cfm_loss,
    model_state_hash,
    require_promoted_fresh_pretrain,
)
from .proximal_update import ProximalConfig, solve_proximal_update
from .scene import GAMMAS, make_ood_scene, verifier_spec_fingerprint
from .schemas import QuerySource
from .stage4_baseline import audit_model, load_audit_bank
from .store import VerificationStore
from .uncertainty import CumulativeLinearUncertainty


STAGE = Path(__file__).resolve().parent / "stage_results/05_afe_expansion"
TARGET_GAMMAS = (0.1, 0.5, 1.0)
CHECKPOINT_SCHEMA = "planned_window_afe_v5_compact_replay_evidence"
PROXIMAL_OBJECTIVE_FORMULA = (
    "uniform-positive CFM + ||theta-theta_n||^2/(2 eta_n)"
)
FULL_REPLAY_DESCRIPTION = (
    "uniform full-verifier-positive FLOW queries only; SafeMPPI "
    "backup updates A but is excluded from training"
)
USABLE_PROXIMAL_STOPPING_REASONS = frozenset({
    "gradient_tolerance",
    "relative_loss_tolerance",
    "update_norm_bound",
    "no_positive_records",
})


# Round checkpoints restore these values authoritatively.  The parser defaults
# are retained here so an explicit conflicting resume flag can fail loudly,
# while omitted flags inherit the saved protocol rather than its CLI defaults.
RUN_DEFAULTS: dict[str, object] = {
    "seed": 105000,
    "episodes_per_gamma": 1,
    "episode_max_steps": 240,
    "candidate_count": 64,
    "verifier_budget": 8,
    "fallback_verifier_budget": 8,
    "backup_smooth_weight": 0.12,
    "backup_noise_var_mult": 3.0,
    "backup_retreat_weight": 0.0,
    "beta": 0.2,
    "ridge_lambda": 0.01,
    "nfe": 8,
    "acquisition": "afe",
    "progress_ranking": True,
    "prox_eta": 0.05,
    "learning_rate": 2e-5,
    "microbatch": 256,
    "solver_max_steps": 12,
    "solver_min_steps": 2,
    "update_norm_limit": 0.12,
    "relative_loss_tolerance": 2e-3,
    "gradient_tolerance": 1e-5,
    "audit_plans_per_context": 4,
    "audit_progress_threshold": 0.10,
    "eval_rollouts": 6,
}


def _matched_protocol(args: argparse.Namespace) -> MatchedProtocol:
    """Materialize the fields that must match every single-axis control."""

    return MatchedProtocol(
        seed=args.seed,
        candidate_count=args.candidate_count,
        verifier_budget=args.verifier_budget,
        fallback_verifier_budget=args.fallback_verifier_budget,
        beta=args.beta,
        backup_smooth_weight=args.backup_smooth_weight,
        backup_noise_var_mult=args.backup_noise_var_mult,
        backup_retreat_weight=args.backup_retreat_weight,
        rounds=args.rounds,
        episodes_per_gamma=args.episodes_per_gamma,
        episode_max_steps=args.episode_max_steps,
        expansion_temperature=1.0,
        nfe=args.nfe,
        ridge_lambda=args.ridge_lambda,
        prox_eta=args.prox_eta,
        learning_rate=args.learning_rate,
        microbatch=args.microbatch,
        solver_max_steps=args.solver_max_steps,
        solver_min_steps=args.solver_min_steps,
        update_norm_limit=args.update_norm_limit,
        relative_loss_tolerance=args.relative_loss_tolerance,
        gradient_tolerance=args.gradient_tolerance,
        audit_plans_per_context=args.audit_plans_per_context,
        audit_progress_threshold=args.audit_progress_threshold,
        eval_rollouts=args.eval_rollouts,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(path)


def _solver_payload(value: object) -> dict[str, object]:
    """Materialize proximal telemetry without accepting an implicit result."""

    if isinstance(value, dict):
        return copy.deepcopy(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return copy.deepcopy(payload)
    raise RuntimeError("proximal solver did not return structured telemetry")


def _proximal_unusable_reason(value: object) -> str | None:
    """Return why a numerical solve is unusable, or ``None`` when admissible.

    A maximum-step cap is only a resource bound.  Reaching it is not evidence
    that the declared proximal objective was solved, so production must stop
    rather than silently turning the cap into a fixed update count.
    """

    payload = _solver_payload(value)
    stopping_reason = str(payload.get("stopping_reason", ""))
    if stopping_reason not in USABLE_PROXIMAL_STOPPING_REASONS:
        return (
            f"stopping_reason={stopping_reason!r} is not a declared "
            "convergence/update-norm/no-positive condition"
        )
    positive_count = int(payload.get("positive_count", -1))
    optimizer_steps = int(payload.get("optimizer_steps", -1))
    if stopping_reason == "no_positive_records":
        if positive_count != 0 or optimizer_steps != 0:
            return "no_positive_records telemetry is internally inconsistent"
        return None
    if positive_count <= 0:
        return f"{stopping_reason} requires a nonempty positive replay ledger"
    if stopping_reason in {"gradient_tolerance", "relative_loss_tolerance"}:
        if payload.get("converged") is not True:
            return f"{stopping_reason} is not marked converged"
        return None
    trace = payload.get("trace")
    if not isinstance(trace, (tuple, list)) or not trace:
        return "update_norm_bound is missing its projected solver trace"
    final_step = trace[-1]
    if not isinstance(final_step, dict):
        try:
            final_step = asdict(final_step)
        except (TypeError, ValueError):
            return "update_norm_bound has malformed final-step telemetry"
    if final_step.get("projected_to_update_bound") is not True:
        return "update_norm_bound was not produced by hard-bound projection"
    return None


def _require_usable_proximal_solve(
    value: object,
    *,
    label: str,
    round_index: int,
    output_dir: Path | None = None,
) -> None:
    """Fail closed on an incomplete/invalid solve and optionally emit STUCK."""

    reason = _proximal_unusable_reason(value)
    if reason is None:
        return
    payload = _solver_payload(value)
    stuck = {
        "status": "STUCK_UNUSABLE_PROXIMAL_SOLVE",
        "usable_for_checkpoint": False,
        "label": str(label),
        "round": int(round_index),
        "stopping_reason": payload.get("stopping_reason"),
        "reason": reason,
        "solver": payload,
    }
    if output_dir is not None:
        _write_json(
            output_dir / f"logs/round_{int(round_index):03d}_solver_STUCK.json",
            stuck,
        )
    raise RuntimeError(
        f"{label} round {round_index} proximal solve is unusable: {reason}; "
        "no round checkpoint may be emitted"
    )


def _save_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate_ordinary(
    model,
    env,
    *,
    seed: int,
    per_gamma: int,
    nfe: int,
    temperature: float = 1.0,
    max_steps: int | None = None,
) -> tuple[dict[str, object], list[object]]:
    results = []
    for gamma_index, gamma in enumerate(GAMMAS):
        for rollout_index in range(per_gamma):
            results.append(rollout_ordinary_flow(
                model,
                env,
                gamma,
                seed=seed + gamma_index * 10_000 + rollout_index,
                temperature=temperature,
                nfe=nfe,
                max_steps=max_steps,
            ))
    summary = {
        str(gamma): summarize_rollouts([row for row in results if row.gamma == gamma])
        for gamma in GAMMAS
    }
    return summary, results


def behavioral_gate(per_gamma: dict[str, object]) -> tuple[bool, dict[str, bool]]:
    """Return a noisy tuning/early-stop heuristic, never final validity."""

    checks = {
        str(gamma): float(per_gamma[str(gamma)]["success_rate"]) > 0.0
        for gamma in TARGET_GAMMAS
    }
    return all(checks.values()), checks


def _query_round_summary(
    episodes: list[EpisodeResult],
    store: VerificationStore,
    before_queries: int,
) -> dict[str, object]:
    new_records = store.records[before_queries:]
    flow_records = [
        record for record in new_records if record.source is QuerySource.FLOW
    ]
    backup_records = [
        record
        for record in new_records
        if record.source is QuerySource.SAFEMPPI_BACKUP
    ]
    total_steps = sum(len(episode.actions) for episode in episodes)
    control_decisions = sum(len(episode.traces) for episode in episodes)
    fallback_steps = sum(episode.fallback_steps for episode in episodes)

    sigma_spans: list[float] = []
    candidate_sigma_means: list[float] = []
    ess_values: list[float] = []
    ess_fractions: list[float] = []
    probability_ratios: list[float] = []
    queried_sigma_values: list[float] = []
    queried_sigma_ranks: list[float] = []
    for episode in episodes:
        for trace in episode.traces:
            raw_sigmas = getattr(trace, "candidate_sigmas", None)
            raw_probabilities = getattr(
                trace, "acquisition_probabilities", None,
            )
            if raw_sigmas is None or raw_probabilities is None:
                continue
            sigmas = np.asarray(raw_sigmas, dtype=np.float64).reshape(-1)
            probabilities = np.asarray(
                raw_probabilities, dtype=np.float64,
            ).reshape(-1)
            if (
                len(sigmas) == 0
                or len(probabilities) != len(sigmas)
                or not np.isfinite(sigmas).all()
                or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0.0)
            ):
                continue
            sigma_spans.append(float(np.ptp(sigmas)))
            candidate_sigma_means.append(float(sigmas.mean()))
            ess = float(getattr(
                trace,
                "acquisition_ess",
                1.0 / np.square(probabilities).sum(),
            ))
            ess_values.append(ess)
            ess_fractions.append(ess / len(sigmas))
            probability_ratios.append(
                float(probabilities.max() / probabilities.min())
            )
            for query in getattr(trace, "queried", ()):
                if getattr(query, "plan_kind", None) != "flow":
                    continue
                sigma = float(getattr(query, "acquisition_sigma"))
                queried_sigma_values.append(sigma)
                # Mid-rank makes ties neutral and yields 0.5 under uniform
                # acquisition from an exchangeable candidate cloud.
                queried_sigma_ranks.append(float(
                    (np.count_nonzero(sigmas < sigma)
                     + 0.5 * np.count_nonzero(sigmas == sigma))
                    / len(sigmas)
                ))

    def finite_summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "episodes": len(episodes),
        "successful_runtime_episodes": sum(episode.success for episode in episodes),
        "fail_closed_episodes": sum(episode.fail_closed for episode in episodes),
        "executed_steps": total_steps,
        "control_decisions": control_decisions,
        "fallback_steps": fallback_steps,
        "fallback_frequency": fallback_steps / max(control_decisions, 1),
        "new_total_full_verifier_calls": len(new_records),
        "new_verifier_calls": len(flow_records),
        "new_positive_queries": sum(record.safe for record in flow_records),
        "new_negative_queries": sum(not record.safe for record in flow_records),
        "query_acceptance": (
            sum(record.safe for record in flow_records) / len(flow_records)
            if flow_records else math.nan
        ),
        "backup_verifier_calls": len(backup_records),
        "backup_positive_queries": sum(record.safe for record in backup_records),
        "backup_negative_queries": sum(not record.safe for record in backup_records),
        "backup_acceptance": (
            sum(record.safe for record in backup_records) / len(backup_records)
            if backup_records else math.nan
        ),
        "cache_hits": sum(episode.cache_hits for episode in episodes),
        "acquisition_diagnostics": {
            "decision_count": len(sigma_spans),
            "candidate_sigma_span": finite_summary(sigma_spans),
            "candidate_sigma_mean": finite_summary(candidate_sigma_means),
            "effective_sample_size": finite_summary(ess_values),
            "effective_sample_size_fraction": finite_summary(ess_fractions),
            "gibbs_probability_max_min_ratio": finite_summary(
                probability_ratios
            ),
            "queried_flow_sigma": finite_summary(queried_sigma_values),
            "queried_flow_sigma_midrank": finite_summary(
                queried_sigma_ranks
            ),
            "interpretation": (
                "AFE selectivity only; sigma is frozen-feature linear-GP "
                "leverage, not calibrated verifier-error probability"
            ),
        },
    }


def _matrix_summary(store: VerificationStore) -> dict[str, object]:
    eigenvalues = np.linalg.eigvalsh(store.uncertainty.A)
    sign, logdet = np.linalg.slogdet(store.uncertainty.A)
    return {
        "observations": store.uncertainty.count,
        "trace": float(np.trace(store.uncertainty.A)),
        "logdet": float(logdet) if sign > 0 else -math.inf,
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "eigenvalues": eigenvalues.tolist(),
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot weights portably without retaining live GPU storage."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _saved_run_config(args: argparse.Namespace) -> dict[str, object]:
    return {name: getattr(args, name) for name in RUN_DEFAULTS}


def _restore_saved_run_config(
    args: argparse.Namespace, saved: dict[str, object],
) -> None:
    """Restore a round's protocol and reject inferred explicit drift.

    ``argparse`` does not retain whether a default-valued option was written on
    the command line.  A value differing from both the parser default and the
    checkpoint value is therefore certainly an explicit conflicting override;
    it is rejected.  Omitted defaults inherit the checkpoint value.
    """

    missing = sorted(set(RUN_DEFAULTS) - set(saved))
    extra = sorted(set(saved) - set(RUN_DEFAULTS))
    if missing or extra:
        raise RuntimeError(
            f"resume run_config schema mismatch: missing={missing}, extra={extra}"
        )
    for name, default in RUN_DEFAULTS.items():
        current = getattr(args, name)
        checkpoint_value = saved[name]
        if current != default and current != checkpoint_value:
            raise RuntimeError(
                f"resume cannot change scientific option {name}: "
                f"checkpoint={checkpoint_value!r}, requested={current!r}"
            )
        setattr(args, name, checkpoint_value)


def _validate_full_resume_recipe(recipe: dict[str, object]) -> None:
    """Require the exact clean Full arm before a checkpoint can resume.

    A resume checkpoint is executable state, not descriptive metadata.  The
    clean-mechanism claims, arm switches, FLOW-only replay rule, and prescribed
    proximal objective must therefore all agree with its authoritative saved
    run configuration.
    """

    expected_arm = {
        "method": "planned-window AFE",
        "arm": "full",
        "acquisition": "afe",
        "acquisition_mode": "afe",
        "progress_ranking": True,
        "eligibility_mode": "full",
        "replay_eligibility": "full_safe",
        "runtime_safety_claim": True,
        "uncertainty_tilting": True,
        "ordinary_audit_untilted": True,
        "sampling_temperature": 1.0,
        "visualization_temperature": 0.5,
    }
    mismatched = {
        key: {"expected": expected, "observed": recipe.get(key)}
        for key, expected in expected_arm.items()
        if recipe.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(
            f"resume checkpoint is not the clean Full arm: {mismatched}"
        )
    if recipe.get("legacy_mechanisms") != clean_method_absence_manifest():
        raise RuntimeError(
            "resume checkpoint clean-method absence manifest is missing or altered"
        )
    if recipe.get("replay") != FULL_REPLAY_DESCRIPTION:
        raise RuntimeError(
            "resume checkpoint does not declare uniform positive FLOW-only replay "
            "with backup excluded"
        )

    run_config = recipe.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("resume checkpoint is missing its complete run_config")
    missing = sorted(set(RUN_DEFAULTS) - set(run_config))
    extra = sorted(set(run_config) - set(RUN_DEFAULTS))
    if missing or extra:
        raise RuntimeError(
            f"resume run_config schema mismatch: missing={missing}, extra={extra}"
        )
    if run_config["acquisition"] != "afe" or run_config["progress_ranking"] is not True:
        raise RuntimeError("resume run_config does not encode the clean Full arm")
    for recipe_key, config_key in (
        ("prox_eta", "prox_eta"),
        ("learning_rate", "learning_rate"),
        ("solver_max_steps", "solver_max_steps"),
        ("solver_min_steps", "solver_min_steps"),
        ("update_norm_limit", "update_norm_limit"),
    ):
        if recipe.get(recipe_key) != run_config[config_key]:
            raise RuntimeError(
                f"resume recipe {recipe_key} disagrees with saved run_config"
            )

    solver = recipe.get("solver")
    if not isinstance(solver, dict):
        raise RuntimeError("resume checkpoint lacks the proximal solver contract")
    expected_solver = {
        "max_steps": run_config["solver_max_steps"],
        "min_steps": run_config["solver_min_steps"],
        "relative_loss_tolerance": run_config["relative_loss_tolerance"],
        "gradient_tolerance": run_config["gradient_tolerance"],
        "update_norm_limit": run_config["update_norm_limit"],
        "reported_optimizer_steps_are_numerical_outcomes": True,
    }
    solver_mismatch = {
        key: {"expected": expected, "observed": solver.get(key)}
        for key, expected in expected_solver.items()
        if solver.get(key) != expected
    }
    if solver_mismatch:
        raise RuntimeError(
            f"resume checkpoint proximal solver contract changed: {solver_mismatch}"
        )
    if "not a fixed scientific update count" not in str(
        solver.get("max_steps_role", "")
    ):
        raise RuntimeError("resume solver cap is mislabeled as a scientific step count")

    objective = recipe.get("update_objective")
    if not isinstance(objective, dict) or objective != {
        "formula": PROXIMAL_OBJECTIVE_FORMULA,
        "proximal_reference": "theta_n captured at expansion-round entry",
        "proximal_reference_is_data_or_replay_anchoring": False,
        "legacy_anchor_or_recovery_data": False,
    }:
        raise RuntimeError(
            "resume checkpoint update is not the prescribed proximal objective"
        )


def _restore_resume_state(
    model: torch.nn.Module,
    checkpoint: dict[str, object],
    *,
    audit_bank_fingerprint: str,
    verifier_spec: str,
) -> tuple[FrozenFeatureModel, VerificationStore, list[dict[str, object]], int, dict[str, object]]:
    """Restore one completed round without refreezing ``theta_n`` as ``phi^0``."""

    if checkpoint.get("afe_schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(
            "resume requires a self-contained planned-window AFE v5 checkpoint "
            "with exact verifier identity, evaluated solver terminal, and "
            "compact full-replay evidence; "
            "a fresh checkpoint must be passed with --checkpoint"
        )
    round_index = int(checkpoint.get("round", -1))
    if round_index < 0:
        raise RuntimeError("resume checkpoint has an invalid round index")
    current_hash = model_state_hash(model)
    if checkpoint.get("current_model_hash") != current_hash:
        raise RuntimeError("resume checkpoint current-model hash mismatch")

    recipe = copy.deepcopy(checkpoint.get("recipe"))
    if not isinstance(recipe, dict):
        raise RuntimeError("resume checkpoint is missing its recipe")
    _validate_full_resume_recipe(recipe)
    if recipe.get("audit_bank_fingerprint") != audit_bank_fingerprint:
        raise RuntimeError("resume audit bank differs from the checkpoint audit bank")
    if recipe.get("audit_bank_role") != "round_monitoring":
        raise RuntimeError("resume checkpoint did not use the round-monitoring bank")
    if recipe.get("verifier_spec_fingerprint") != verifier_spec:
        raise RuntimeError(
            "resume scene/goal/dynamics/verifier specification differs from "
            "the checkpoint query ledger"
        )
    frozen_state = checkpoint.get("frozen_feature_state_dict")
    if not isinstance(frozen_state, dict):
        raise RuntimeError("resume checkpoint is missing frozen phi0 weights")
    frozen_model = copy.deepcopy(model)
    frozen_model.load_state_dict(frozen_state, strict=True)
    frozen = FrozenFeatureModel(
        frozen_model,
        s=float(recipe.get("feature_time", 0.9)),
        expected_dim=32,
    )
    frozen_hash = frozen.state_hash
    if checkpoint.get("frozen_feature_hash") != frozen_hash:
        raise RuntimeError("resume frozen-feature hash mismatch")
    if recipe.get("frozen_feature_hash") != frozen_hash:
        raise RuntimeError("resume recipe and frozen-feature weights disagree")
    if recipe.get("source_model_hash") != frozen_hash:
        raise RuntimeError("resume source model is not the frozen phi0 model")

    raw_store = checkpoint.get("verification_store_state")
    if not isinstance(raw_store, dict):
        raise RuntimeError("resume checkpoint is missing the cumulative query store")
    store = VerificationStore.from_state_dict(raw_store)
    if not math.isclose(
        store.uncertainty.lambda_,
        float(recipe["run_config"]["ridge_lambda"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("resume store and recipe ridge lambda disagree")

    raw_history = checkpoint.get("history")
    if not isinstance(raw_history, list) or not raw_history:
        raise RuntimeError("resume checkpoint is missing round history")
    history = copy.deepcopy(raw_history)
    final = history[-1]
    if int(final.get("round", -1)) != round_index:
        raise RuntimeError("resume history does not end at the checkpoint round")
    if final.get("model_hash") != current_hash:
        raise RuntimeError("resume history and checkpoint model hashes disagree")
    for row in history:
        if not isinstance(row, dict):
            raise RuntimeError("resume history contains a malformed round row")
        history_round = int(row.get("round", -1))
        if history_round <= 0:
            continue
        if row.get("solver") is None:
            raise RuntimeError(
                f"resume history round {history_round} lacks proximal telemetry"
            )
        _require_usable_proximal_solve(
            row["solver"],
            label="Stage 05 resume history",
            round_index=history_round,
        )
    matrix = final.get("matrix")
    if not isinstance(matrix, dict) or int(matrix.get("observations", -1)) != store.query_count:
        raise RuntimeError("resume history and cumulative query count disagree")
    store.assert_exact_accounting()
    return frozen, store, history, round_index, recipe


def _round_checkpoint(
    output: Path,
    model,
    frozen: FrozenFeatureModel,
    store: VerificationStore,
    round_index: int,
    recipe: dict[str, object],
    history: list[dict[str, object]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    HP.save_hp(model, temporary, extra={
        "afe_schema": CHECKPOINT_SCHEMA,
        "round": round_index,
        "recipe": recipe,
        "history": history,
        "frozen_feature_hash": frozen.state_hash,
        "frozen_feature_state_dict": _cpu_state_dict(frozen.model),
        "current_model_hash": model_state_hash(model),
        "verification_store_state": store.state_dict(),
    })
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume a completed v2 round; --rounds is the target total round",
    )
    parser.add_argument("--audit-bank", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--giant-radius", type=float, default=1.2)
    parser.add_argument(
        "--evaluation-max-steps",
        type=int,
        default=None,
        help="ordinary-flow evaluation horizon; defaults to the environment horizon",
    )
    parser.add_argument("--seed", type=int, default=RUN_DEFAULTS["seed"])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--episodes-per-gamma", type=int, default=RUN_DEFAULTS["episodes_per_gamma"])
    parser.add_argument("--episode-max-steps", type=int, default=RUN_DEFAULTS["episode_max_steps"])
    parser.add_argument("--candidate-count", type=int, default=RUN_DEFAULTS["candidate_count"])
    parser.add_argument("--verifier-budget", type=int, default=RUN_DEFAULTS["verifier_budget"])
    parser.add_argument("--fallback-verifier-budget", type=int, default=RUN_DEFAULTS["fallback_verifier_budget"])
    parser.add_argument("--backup-smooth-weight", type=float, default=RUN_DEFAULTS["backup_smooth_weight"])
    parser.add_argument("--backup-noise-var-mult", type=float, default=RUN_DEFAULTS["backup_noise_var_mult"])
    parser.add_argument("--backup-retreat-weight", type=float, default=RUN_DEFAULTS["backup_retreat_weight"])
    parser.add_argument("--beta", type=float, default=RUN_DEFAULTS["beta"])
    parser.add_argument("--ridge-lambda", type=float, default=RUN_DEFAULTS["ridge_lambda"])
    parser.add_argument("--nfe", type=int, default=RUN_DEFAULTS["nfe"])
    parser.add_argument("--acquisition", choices=("afe", "uniform"), default=RUN_DEFAULTS["acquisition"])
    parser.add_argument("--progress-ranking", action=argparse.BooleanOptionalAction, default=RUN_DEFAULTS["progress_ranking"])
    parser.add_argument("--prox-eta", type=float, default=RUN_DEFAULTS["prox_eta"])
    parser.add_argument("--learning-rate", type=float, default=RUN_DEFAULTS["learning_rate"])
    parser.add_argument("--microbatch", type=int, default=RUN_DEFAULTS["microbatch"])
    parser.add_argument(
        "--solver-max-steps", type=int, default=RUN_DEFAULTS["solver_max_steps"],
        help="numerical proximal-solver cap; not a fixed scientific update count",
    )
    parser.add_argument(
        "--solver-min-steps", type=int, default=RUN_DEFAULTS["solver_min_steps"],
        help="minimum numerical steps before tolerance-based convergence",
    )
    parser.add_argument("--update-norm-limit", type=float, default=RUN_DEFAULTS["update_norm_limit"])
    parser.add_argument("--relative-loss-tolerance", type=float, default=RUN_DEFAULTS["relative_loss_tolerance"])
    parser.add_argument("--gradient-tolerance", type=float, default=RUN_DEFAULTS["gradient_tolerance"])
    parser.add_argument("--audit-plans-per-context", type=int, default=RUN_DEFAULTS["audit_plans_per_context"])
    parser.add_argument("--audit-progress-threshold", type=float, default=RUN_DEFAULTS["audit_progress_threshold"])
    parser.add_argument("--eval-rollouts", type=int, default=RUN_DEFAULTS["eval_rollouts"])
    parser.add_argument("--continue-after-gate", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.giant_radius) or args.giant_radius <= 0.0:
        raise ValueError("--giant-radius must be finite and positive")
    if args.evaluation_max_steps is not None and args.evaluation_max_steps <= 0:
        raise ValueError("--evaluation-max-steps must be positive")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    input_checkpoint = args.resume_checkpoint or args.checkpoint
    assert input_checkpoint is not None
    model, source_checkpoint = HP.load_hp(input_checkpoint, device=device)
    if args.checkpoint is not None:
        require_promoted_fresh_pretrain(model, source_checkpoint)
    bank = load_audit_bank(args.audit_bank)
    if bank.role != "round_monitoring":
        raise RuntimeError(
            "Stage 05 may use only the round-monitoring bank; the sealed final "
            "bank must remain outcome-uninspected until final evaluation"
        )
    resuming = args.resume_checkpoint is not None
    env = make_ood_scene(radius=args.giant_radius)
    current_verifier_spec = verifier_spec_fingerprint(env, env.goal)
    if resuming:
        frozen, store, history, completed_round, recipe = _restore_resume_state(
            model,
            source_checkpoint,
            audit_bank_fingerprint=bank.fingerprint,
            verifier_spec=current_verifier_spec,
        )
        _restore_saved_run_config(args, recipe["run_config"])
        if args.rounds < completed_round:
            raise RuntimeError(
                f"--rounds={args.rounds} precedes resumed round {completed_round}"
            )
        saved_outdir = recipe.get("output_dir")
        if not isinstance(saved_outdir, str):
            raise RuntimeError("resume checkpoint is missing its output directory")
        expected_outdir = Path(saved_outdir).resolve()
        if args.outdir is not None and args.outdir.resolve() != expected_outdir:
            raise RuntimeError(
                "resume must continue in the original output directory so prior "
                "round bundles are not orphaned"
            )
        args.outdir = expected_outdir
        frozen_hash = frozen.state_hash
    else:
        if "afe_schema" in source_checkpoint:
            raise RuntimeError(
                "an AFE round checkpoint cannot start a fresh run; use "
                "--resume-checkpoint so phi0 and cumulative A are preserved"
            )
        completed_round = 0
        history: list[dict[str, object]] = []
        args.outdir = args.outdir or STAGE
        frozen = FrozenFeatureModel.from_pretrained(model, s=0.9, expected_dim=32)
        frozen_hash = frozen.state_hash
        expected_phi0_hash = source_checkpoint.get("model_state_sha256")
        if expected_phi0_hash is None or str(expected_phi0_hash) != frozen_hash:
            raise RuntimeError(
                "source checkpoint is not the hash-locked Stage-03 pretrained/phi0 state"
            )
        store = VerificationStore(CumulativeLinearUncertainty(lambda_=args.ridge_lambda))
    afe_config = AFEConfig(
        sampling=SamplingConfig(
            candidate_count=args.candidate_count,
            verifier_budget=args.verifier_budget,
            beta=args.beta,
            expansion_temperature=1.0,
            audit_temperature=1.0,
            visualization_temperature=0.5,
            nfe=args.nfe,
        ),
        features=FeatureConfig(
            representation_dim=32,
            feature_time=0.9,
            ridge_lambda=args.ridge_lambda,
        ),
    )
    controller = PlannedWindowAFEController(
        model,
        frozen,
        store,
        config=afe_config,
        backup=SafeMPPIBackup(
            smooth_weight=args.backup_smooth_weight,
            noise_var_mult=args.backup_noise_var_mult,
            retreat_weight=args.backup_retreat_weight,
            max_debug_candidates=0,
        ),
        device=device,
        fallback_verifier_budget=args.fallback_verifier_budget,
        acquisition_mode=args.acquisition,
        progress_ranking=args.progress_ranking,
    )
    if not resuming:
        source_model_hash = model_state_hash(model)
        matched_protocol = _matched_protocol(args)
        recipe = {
            "method": "planned-window AFE" if args.acquisition == "afe" else "-AFE uniform acquisition",
            "arm": "full" if args.acquisition == "afe" and args.progress_ranking else "custom",
            "display_name": "Our approach",
            "acquisition": args.acquisition,
            "acquisition_mode": args.acquisition,
            "progress_ranking": args.progress_ranking,
            "eligibility_mode": "full",
            "replay_eligibility": "full_safe",
            "runtime_safety_claim": True,
            "uncertainty_tilting": args.acquisition == "afe",
            "ordinary_audit_untilted": True,
            "sampling_temperature": 1.0,
            "visualization_temperature": 0.5,
            "scene": {
                "giant_radius": float(args.giant_radius),
                "start": [0.5, 0.5],
                "goal": [4.5, 4.5],
                "expansion_max_steps": int(args.episode_max_steps),
                "evaluation_max_steps": (
                    None
                    if args.evaluation_max_steps is None
                    else int(args.evaluation_max_steps)
                ),
            },
            "candidate_count": args.candidate_count,
            "verifier_budget": args.verifier_budget,
            "fallback_verifier_budget": args.fallback_verifier_budget,
            "backup_planner": {
                "smooth_weight": args.backup_smooth_weight,
                "noise_var_mult": args.backup_noise_var_mult,
                "retreat_weight": args.backup_retreat_weight,
                "cost_selected_proposals_only": True,
                "raw_debug_candidates": "not proposed or executed",
            },
            "beta": args.beta,
            "ridge_lambda": args.ridge_lambda,
            "feature_time": 0.9,
            "frozen_feature_hash": frozen_hash,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_sha256": sha256_file(args.checkpoint),
            "source_model_hash": source_model_hash,
            "audit_bank": str(args.audit_bank.resolve()),
            "audit_bank_fingerprint": bank.fingerprint,
            "audit_bank_role": bank.role,
            "verifier_spec_fingerprint": current_verifier_spec,
            "output_dir": str(args.outdir.resolve()),
            "run_config": _saved_run_config(args),
            "matched_protocol": asdict(matched_protocol),
            "legacy_mechanisms": clean_method_absence_manifest(),
            "prox_eta": args.prox_eta,
            "learning_rate": args.learning_rate,
            "solver_max_steps": args.solver_max_steps,
            "solver_min_steps": args.solver_min_steps,
            "update_norm_limit": args.update_norm_limit,
            "solver": {
                "max_steps": args.solver_max_steps,
                "max_steps_role": "numerical cap, not a fixed scientific update count",
                "min_steps": args.solver_min_steps,
                "min_steps_role": "minimum before tolerance-based convergence",
                "relative_loss_tolerance": args.relative_loss_tolerance,
                "gradient_tolerance": args.gradient_tolerance,
                "update_norm_limit": args.update_norm_limit,
                "reported_optimizer_steps_are_numerical_outcomes": True,
            },
            "update_objective": {
                "formula": PROXIMAL_OBJECTIVE_FORMULA,
                "proximal_reference": "theta_n captured at expansion-round entry",
                "proximal_reference_is_data_or_replay_anchoring": False,
                "legacy_anchor_or_recovery_data": False,
            },
            "gamma_distribution": "fixed uniform episode allocation over seven gammas; no schedule",
            "replay": FULL_REPLAY_DESCRIPTION,
        }
    if resuming:
        # ``--rounds`` is the only intentionally extendable protocol field.
        # Preserve all saved settings while making the new final budget
        # explicit for cross-arm artifact validation.
        matched_protocol = _matched_protocol(args)
        recipe["matched_protocol"] = asdict(matched_protocol)
    args.outdir.mkdir(parents=True, exist_ok=True)
    _write_json(args.outdir / "logs/recipe.json", recipe)

    if not resuming:
        baseline_audit = audit_model(
            model,
            env,
            bank,
            plans_per_context=args.audit_plans_per_context,
            seed=args.seed - 1,
            nfe=args.nfe,
            progress_threshold=args.audit_progress_threshold,
        )
        baseline_eval, baseline_rollouts = evaluate_ordinary(
            model,
            env,
            seed=args.seed - 10_000,
            per_gamma=args.eval_rollouts,
            nfe=args.nfe,
            max_steps=args.evaluation_max_steps,
        )
        history = [{
            "round": 0,
            "audit": baseline_audit.to_dict(),
            "ordinary_per_gamma": baseline_eval,
            "matrix": _matrix_summary(store),
            "query": None,
            "solver": None,
            "model_hash": model_state_hash(model),
        }]
        _save_torch(args.outdir / "data/round_000_bundle.pt", {
            "schema_version": "afe_expansion_round_v1",
            "round": 0,
            "recipe": recipe,
            "episodes": [],
            "ordinary_rollouts": [asdict(row) for row in baseline_rollouts],
            "audit": baseline_audit.to_dict(),
            "store_state": store.state_dict(),
        })
        _round_checkpoint(
            args.outdir / "checkpoints/round_000.pt",
            model, frozen, store, 0, recipe, history,
        )

    started = time.perf_counter()
    wall_offset = float(history[-1].get("wall_seconds_total", 0.0))
    gate_passed = bool(
        history[-1].get("tuning_gate", {}).get("passed", False)
    )
    first_round = completed_round + 1
    round_range = (
        range(first_round, args.rounds + 1)
        if args.continue_after_gate or not gate_passed
        else range(0)
    )
    for round_index in round_range:
        before_queries = store.query_count
        episodes: list[EpisodeResult] = []
        for gamma_index, gamma in enumerate(GAMMAS):
            for episode_index in range(args.episodes_per_gamma):
                episode_seed = (
                    args.seed
                    + round_index * 1_000_000
                    + gamma_index * 10_000
                    + episode_index
                )
                episodes.append(controller.run_episode(
                    env,
                    gamma,
                    seed=episode_seed,
                    max_steps=args.episode_max_steps,
                    reach=0.20,
                ))
        query_summary = _query_round_summary(episodes, store, before_queries)
        solver = solve_proximal_update(
            model,
            store.uniform_positive_view(source=QuerySource.FLOW),
            ledger_cfm_loss,
            ProximalConfig(
                eta=args.prox_eta,
                learning_rate=args.learning_rate,
                batch_size=args.microbatch,
                max_steps=args.solver_max_steps,
                min_steps=args.solver_min_steps,
                update_norm_limit=args.update_norm_limit,
                relative_loss_tolerance=args.relative_loss_tolerance,
                gradient_tolerance=args.gradient_tolerance,
                tolerance_patience=2,
                seed=args.seed + round_index,
            ),
        )
        _require_usable_proximal_solve(
            solver,
            label="Stage 05 Full",
            round_index=round_index,
            output_dir=args.outdir,
        )
        if frozen.state_hash != frozen_hash:
            raise RuntimeError("frozen uncertainty representation changed")
        audit = audit_model(
            model,
            env,
            bank,
            plans_per_context=args.audit_plans_per_context,
            seed=args.seed + round_index * 100,
            nfe=args.nfe,
            progress_threshold=args.audit_progress_threshold,
        )
        ordinary, ordinary_rollouts = evaluate_ordinary(
            model,
            env,
            seed=args.seed + round_index * 100_000,
            per_gamma=args.eval_rollouts,
            nfe=args.nfe,
            max_steps=args.evaluation_max_steps,
        )
        gate_passed, gate_checks = behavioral_gate(ordinary)
        record = {
            "round": round_index,
            "query": query_summary,
            "solver": solver.to_dict(),
            "audit": audit.to_dict(),
            "ordinary_per_gamma": ordinary,
            "tuning_gate": {
                "passed": gate_passed,
                "target_nonzero_sr": gate_checks,
                "role": "checkpoint-selection heuristic only; not final evidence",
                "rollouts_per_gamma": args.eval_rollouts,
            },
            "matrix": _matrix_summary(store),
            "model_hash": model_state_hash(model),
            "wall_seconds_total": wall_offset + time.perf_counter() - started,
        }
        history.append(record)
        _save_torch(args.outdir / f"data/round_{round_index:03d}_bundle.pt", {
            "schema_version": "afe_expansion_round_v1",
            "round": round_index,
            "recipe": recipe,
            "episodes": episodes,
            "ordinary_rollouts": [asdict(row) for row in ordinary_rollouts],
            "audit": audit.to_dict(),
            "query_summary": query_summary,
            "solver": solver.to_dict(),
            "matrix": record["matrix"],
            "store_state": store.state_dict(),
        })
        _round_checkpoint(
            args.outdir / f"checkpoints/round_{round_index:03d}.pt",
            model, frozen, store, round_index, recipe, history,
        )
        _write_json(args.outdir / "logs/history.json", history)
        print(
            f"[round {round_index}] q={query_summary['new_verifier_calls']} "
            f"acc={query_summary['query_acceptance']:.3f} "
            f"flow_pos={len(store.uniform_positive_view(source=QuerySource.FLOW))} "
            f"fallback={query_summary['fallback_frequency']:.3f} "
            f"solver={solver.stopping_reason}/{solver.optimizer_steps} "
            f"update={solver.final_update_norm:.4g} gate={gate_passed}",
            flush=True,
        )
        if gate_passed and not args.continue_after_gate:
            break

    summary = {
        "status": (
            "TUNING_GATE_PASS_REQUIRES_SEALED_FINAL_EVALUATION"
            if gate_passed
            else "BOUNDED_RUN_COMPLETE_TUNING_GATE_NOT_YET_PASS"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(recipe["source_checkpoint"]),
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if resuming else None,
        "source_checkpoint_config": source_checkpoint.get("config", {}),
        "audit_bank": str(args.audit_bank.resolve()),
        "scene": {
            "giant_radius": float(args.giant_radius),
            "start": [0.5, 0.5],
            "goal": [4.5, 4.5],
            "expansion_max_steps": int(args.episode_max_steps),
            "evaluation_max_steps": (
                None
                if args.evaluation_max_steps is None
                else int(args.evaluation_max_steps)
            ),
        },
        "recipe": recipe,
        "rounds_completed": len(history) - 1,
        "final": history[-1],
        "validity_reporting": {
            "query_acceptance": "sigma-tilted acquisition efficiency only",
            "model_validity": "isolated untilted temperature-1 round-monitoring V",
            "performance_validity": "isolated round-monitoring Vprog",
            "confidence_interval_scope": (
                "conditional plan-sampling Wilson interval on one fixed bank "
                "and one trained model; not an independent-training-seed CI"
            ),
            "runtime": "certified fallback and fail-closed frequencies",
            "tuning_gate": (
                "nonzero SR on a small changing-seed rollout batch; checkpoint "
                "selection only, never final validity evidence"
            ),
            "final_evidence_required": (
                "untouched sealed_final_test bank, temperature 1, no tilting, "
                "and aggregation over at least two independently trained models"
            ),
        },
    }
    _write_json(args.outdir / "REPORT.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
