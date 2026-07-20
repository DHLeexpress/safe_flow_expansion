#!/usr/bin/env python3
"""Stage 04: fixed OOD audit bank, expert ceiling, and frozen baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

import grid_hp_expt as HP

from .audit import AuditConfig, ImmutableContextBank, run_independent_audit
from .evaluation import detour_mode, local_plan_mode, rollout_ordinary_flow, summarize_rollouts
from .policy import (
    model_state_hash,
    require_promoted_fresh_pretrain,
    sample_plans,
)
from .scene import GAMMAS, context_from_state, make_ood_scene, verifier_spec_fingerprint
from .stage2_planned_demos import DemoRunConfig, run_expert_rollout
from .verifier import verify_plan


STAGE = Path(__file__).resolve().parent / "stage_results/04_ood_baseline"
EXPERT_SCHEMA = "afe_ood_safemppi_expert_rollouts_v1"
BASELINE_SCHEMA = "afe_ood_baseline_rollouts_v2"
AUDIT_BANK_SCHEMA = "afe_audit_bank_v3_locked_provenance"


@dataclass(frozen=True)
class SafeMPPIExpertSettings:
    """The one selected SafeMPPI recipe used by every Stage-04 producer."""

    smooth_weight: float = 8.0
    noise_var_mult: float = 3.0
    retreat_weight: float = 1.0

    def __post_init__(self) -> None:
        values = (self.smooth_weight, self.noise_var_mult, self.retreat_weight)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SafeMPPI expert settings must be finite")
        if self.smooth_weight < 0.0 or self.retreat_weight < 0.0:
            raise ValueError("SafeMPPI expert cost weights must be nonnegative")
        if self.noise_var_mult <= 0.0:
            raise ValueError("SafeMPPI expert noise variance multiplier must be positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "smooth_weight": float(self.smooth_weight),
            "noise_var_mult": float(self.noise_var_mult),
            "retreat_weight": float(self.retreat_weight),
        }


def _locked_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AuditStateContext:
    state: np.ndarray
    executed_history: np.ndarray
    expert_seed: int
    expert_mode: str
    source_step: int

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32).copy()
        history = np.asarray(self.executed_history, dtype=np.float32).reshape(-1, 2).copy()
        if state.shape != (4,) or not np.isfinite(state).all() or not np.isfinite(history).all():
            raise ValueError("invalid held-out audit state/history")
        state.setflags(write=False)
        history.setflags(write=False)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "executed_history", history)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def build_fixed_audit_bank(
    env,
    *,
    device: torch.device,
    seed0: int,
    contexts_per_mode: int,
    candidate_limit: int,
    planner_settings: SafeMPPIExpertSettings = SafeMPPIExpertSettings(),
) -> tuple[ImmutableContextBank, list[dict[str, Any]]]:
    """Use held-out certified OOD expert paths, balanced by global detour side."""
    accepted: dict[str, dict[str, Any]] = {}
    config = DemoRunConfig(
        max_steps=300,
        reach_m=0.20,
        smooth_weight=planner_settings.smooth_weight,
        noise_var_mult=planner_settings.noise_var_mult,
        retreat_weight=planner_settings.retreat_weight,
        max_debug_candidates=6,
        max_proposals_per_step=8,
        quota_per_direction=1,
        max_candidate_seeds_per_gamma=candidate_limit,
        seed0=seed0,
    )
    from .evaluation import detour_mode

    attempts: list[dict[str, Any]] = []
    for seed in range(seed0, seed0 + candidate_limit):
        episode = run_expert_rollout(
            env=env,
            gamma=0.5,
            seed=seed,
            device=device,
            config=config,
        )
        mode = detour_mode(episode["path"])
        attempts.append({
            "seed": seed,
            "success": bool(episode["success"]),
            "mode": mode,
            "steps": int(episode["steps"]),
            "query_acceptance": float(episode["query_acceptance"]),
        })
        if episode["success"] and mode in {"upper-left", "lower-right"} and mode not in accepted:
            accepted[mode] = episode
        if len(accepted) == 2:
            break
    if len(accepted) != 2:
        raise RuntimeError(f"could not build two-mode OOD expert audit bank: {attempts}")

    contexts: list[AuditStateContext] = []
    for mode in ("upper-left", "lower-right"):
        episode = accepted[mode]
        actions = np.asarray(episode["executed_actions"], dtype=np.float32)
        states = np.asarray(episode["states"], dtype=np.float32)
        # Exclude step 0 (every expansion episode starts there) and terminal
        # states.  Fixed interior quantiles are held out from acquisition.
        high = max(2, len(actions) - 2)
        indices = np.unique(
            np.linspace(1, high, contexts_per_mode, dtype=int)
        ).tolist()
        if len(indices) < contexts_per_mode:
            raise RuntimeError("expert trajectory too short for fixed audit contexts")
        for step in indices[:contexts_per_mode]:
            contexts.append(AuditStateContext(
                state=states[step],
                executed_history=actions[:step],
                expert_seed=int(episode["seed"]),
                expert_mode=mode,
                source_step=int(step),
            ))
    return (
        ImmutableContextBank(contexts, role="round_monitoring"),
        [accepted[mode] for mode in ("upper-left", "lower-right")],
    )


def build_sealed_final_test_bank(
    env,
    *,
    device: torch.device,
    seed0: int,
    interior_contexts_per_mode: int,
    candidate_limit: int,
    planner_settings: SafeMPPIExpertSettings = SafeMPPIExpertSettings(),
) -> tuple[ImmutableContextBank, list[dict[str, Any]]]:
    """Build an outcome-uninspected final bank from disjoint expert seeds.

    The deployment start is included explicitly.  Each interior row comes from
    a different successful SafeMPPI rollout, avoiding the pseudo-replication of
    taking many correlated states from one path.  This constructor never sees a
    flow model and callers must not run round-by-round audits on its output.
    """

    if interior_contexts_per_mode <= 0 or candidate_limit <= 0:
        raise ValueError("sealed-bank context count and candidate limit must be positive")
    config = DemoRunConfig(
        max_steps=300,
        reach_m=0.20,
        smooth_weight=planner_settings.smooth_weight,
        noise_var_mult=planner_settings.noise_var_mult,
        retreat_weight=planner_settings.retreat_weight,
        max_debug_candidates=6,
        max_proposals_per_step=8,
        quota_per_direction=1,
        max_candidate_seeds_per_gamma=candidate_limit,
        seed0=seed0,
    )
    wanted = ("upper-left", "lower-right")
    accepted: dict[str, list[dict[str, Any]]] = {mode: [] for mode in wanted}
    provenance: list[dict[str, Any]] = []
    for seed in range(seed0, seed0 + candidate_limit):
        episode = run_expert_rollout(
            env=env,
            gamma=0.5,
            seed=seed,
            device=device,
            config=config,
        )
        mode = detour_mode(episode["path"])
        row = {
            "seed": int(seed),
            "success": bool(episode["success"]),
            "mode": mode,
            "steps": int(episode["steps"]),
        }
        provenance.append(row)
        if (
            episode["success"]
            and mode in accepted
            and len(accepted[mode]) < interior_contexts_per_mode
        ):
            accepted[mode].append(episode)
        if all(len(accepted[mode]) >= interior_contexts_per_mode for mode in wanted):
            break
    missing = {
        mode: interior_contexts_per_mode - len(accepted[mode])
        for mode in wanted
        if len(accepted[mode]) < interior_contexts_per_mode
    }
    if missing:
        raise RuntimeError(
            f"could not build independently seeded sealed-bank modes {missing}; "
            f"attempted seeds [{seed0},{seed0 + candidate_limit})"
        )

    contexts = [AuditStateContext(
        state=env.x0.detach().cpu().numpy(),
        executed_history=np.empty((0, 2), dtype=np.float32),
        expert_seed=-1,
        expert_mode="deployment_start",
        source_step=0,
    )]
    for mode in wanted:
        for episode_index, episode in enumerate(accepted[mode]):
            actions = np.asarray(episode["executed_actions"], dtype=np.float32)
            states = np.asarray(episode["states"], dtype=np.float32)
            if len(actions) < 4:
                raise RuntimeError("sealed-bank expert trajectory is too short")
            # Spread the one state drawn from each independent trajectory over
            # interior quantiles while excluding both endpoints.
            fraction = (episode_index + 1) / (len(accepted[mode]) + 1)
            step = int(np.clip(round(fraction * (len(actions) - 1)), 1, len(actions) - 2))
            contexts.append(AuditStateContext(
                state=states[step],
                executed_history=actions[:step],
                expert_seed=int(episode["seed"]),
                expert_mode=mode,
                source_step=step,
            ))
    seeds = [context.expert_seed for context in contexts if context.expert_seed >= 0]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("sealed interior contexts must use independent expert seeds")
    return ImmutableContextBank(contexts, role="sealed_final_test"), provenance


def save_audit_bank(
    bank: ImmutableContextBank,
    output: Path,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Save contexts together with tamper-evident expert-source provenance."""

    rows = []
    for context in bank:
        rows.append({
            "state": np.array(context.state, copy=True),
            "executed_history": np.array(context.executed_history, copy=True),
            "expert_seed": context.expert_seed,
            "expert_mode": context.expert_mode,
            "source_step": context.source_step,
        })
    normalized_provenance = json.loads(json.dumps(
        dict(provenance), sort_keys=True, default=_json_default, allow_nan=False,
    ))
    provenance_fingerprint = _locked_fingerprint(normalized_provenance)
    artifact_identity = {
        "schema_version": AUDIT_BANK_SCHEMA,
        "fingerprint": bank.fingerprint,
        "role": bank.role,
        "source_provenance_fingerprint": provenance_fingerprint,
    }
    artifact_fingerprint = _locked_fingerprint(artifact_identity)
    payload = {
        **artifact_identity,
        "artifact_fingerprint": artifact_fingerprint,
        "contexts": rows,
        "training_use_forbidden": True,
        "source_provenance": normalized_provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        key: payload[key]
        for key in (
            "schema_version", "fingerprint", "role",
            "source_provenance_fingerprint", "artifact_fingerprint",
        )
    } | {"source_provenance": normalized_provenance}


def load_audit_bank_artifact(
    path: Path,
    *,
    require_locked_provenance: bool = False,
) -> tuple[ImmutableContextBank, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.get("schema_version")
    if schema not in {"afe_audit_bank_v1", "afe_audit_bank_v2", AUDIT_BANK_SCHEMA}:
        raise ValueError("unsupported audit-bank schema")
    role = payload.get("role", "unspecified")
    bank = ImmutableContextBank(
        (AuditStateContext(**row) for row in payload["contexts"]), role=role,
    )
    if bank.fingerprint != payload["fingerprint"]:
        raise RuntimeError("held-out audit-bank fingerprint mismatch")
    if payload.get("training_use_forbidden") is not True:
        raise RuntimeError("held-out audit bank does not forbid training use")
    if schema == AUDIT_BANK_SCHEMA:
        provenance = payload.get("source_provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("locked audit bank is missing source provenance")
        provenance = dict(provenance)
        provenance_fingerprint = _locked_fingerprint(provenance)
        if provenance_fingerprint != payload.get("source_provenance_fingerprint"):
            raise RuntimeError("audit-bank source provenance fingerprint mismatch")
        artifact_identity = {
            "schema_version": AUDIT_BANK_SCHEMA,
            "fingerprint": bank.fingerprint,
            "role": bank.role,
            "source_provenance_fingerprint": provenance_fingerprint,
        }
        artifact_fingerprint = _locked_fingerprint(artifact_identity)
        if artifact_fingerprint != payload.get("artifact_fingerprint"):
            raise RuntimeError("audit-bank artifact fingerprint mismatch")
        metadata = artifact_identity | {
            "artifact_fingerprint": artifact_fingerprint,
            "source_provenance": provenance,
        }
    else:
        if require_locked_provenance:
            raise RuntimeError(
                "scientific consumer requires a v3 audit bank with locked source provenance"
            )
        metadata = {
            "schema_version": schema,
            "fingerprint": bank.fingerprint,
            "role": bank.role,
            "source_provenance_fingerprint": None,
            "artifact_fingerprint": None,
            "source_provenance": None,
        }
    return bank, metadata


def load_audit_bank(path: Path) -> ImmutableContextBank:
    bank, _metadata = load_audit_bank_artifact(path)
    return bank


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> dict[str, float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one attempted rollout")
    if successes < 0 or successes > trials:
        raise ValueError("Wilson successes must lie in [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    probability = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (probability + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z_squared / (4.0 * trials**2)
        )
        / denominator
    )
    return {
        "confidence": float(confidence),
        "low": max(0.0, center - half_width),
        "high": min(1.0, center + half_width),
    }


def _normalize_expert_episode(episode: Mapping[str, Any], env) -> dict[str, Any]:
    """Convert one attempted SafeMPPI episode to the fixed expert row schema."""

    path = np.asarray(episode.get("path"), dtype=np.float32)
    actions = np.asarray(
        episode.get("executed_actions", episode.get("actions", ())), dtype=np.float32
    ).reshape(-1, 2)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0:
        raise ValueError(f"expert path must have shape [N,2], got {path.shape}")
    if not np.isfinite(path).all() or not np.isfinite(actions).all():
        raise ValueError("expert path/actions contain non-finite values")
    if len(path) != len(actions) + 1:
        raise ValueError("expert path must contain exactly one more state than action")
    gamma = float(episode["gamma"])
    seed = int(episode["seed"])
    reached = bool(episode.get("reached", False))
    collision = bool(episode.get("collision", False))
    out_of_bounds = not bool(episode.get("in_bounds", True))
    dead_reason = episode.get("dead_reason")
    timeout = str(dead_reason) == "timeout"
    success = bool(episode.get("success", False))
    expected_success = reached and not collision and not out_of_bounds and dead_reason is None
    if success != expected_success:
        raise ValueError(
            "expert success must equal reached AND collision-free AND in-bounds without failure"
        )
    minimum_clearance = float(episode["min_clearance_m"])
    wall_seconds = float(episode.get("wall_seconds", math.nan))
    if not math.isfinite(minimum_clearance) or not math.isfinite(wall_seconds):
        raise ValueError("expert clearance and wall time must be finite")
    duration = len(actions) * float(env.dt)
    path_length = float(
        episode.get(
            "path_length_m",
            np.linalg.norm(np.diff(path.astype(np.float64), axis=0), axis=1).sum(),
        )
    )
    return {
        "gamma": gamma,
        "seed": seed,
        "path": path.copy(),
        "actions": actions.copy(),
        "success": success,
        "reached": reached,
        "collision": collision,
        "out_of_bounds": out_of_bounds,
        "timeout": timeout,
        "failure_reason": None if success else str(dead_reason or "unspecified_failure"),
        "min_clearance_m": minimum_clearance,
        "rollout_duration_s": duration,
        "time_to_goal_s": duration if success else None,
        "wall_time_s": wall_seconds,
        "path_length_m": path_length,
        "detour_mode": detour_mode(path),
    }


def summarize_expert_rollouts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize every attempted expert rollout, including all failures."""

    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("cannot summarize an empty expert rollout set")
    n = len(records)
    binary_fields = {
        "success": ("success", "successes"),
        "collision": ("collision", "collisions"),
        "out_of_bounds": ("out_of_bounds", "out_of_bounds_count"),
        "timeout": ("timeout", "timeouts"),
    }
    summary: dict[str, Any] = {"n": n, "attempted_rollouts": n}
    for label, (field, count_key) in binary_fields.items():
        count = sum(bool(row[field]) for row in records)
        summary[count_key] = count
        summary[f"{label}_rate"] = count / n
        summary[f"{label}_rate_wilson_95"] = _wilson_interval(count, n)
    successes = [row for row in records if bool(row["success"])]
    failure_counts: dict[str, int] = {}
    for row in records:
        if not bool(row["success"]):
            reason = str(row.get("failure_reason", "unspecified_failure"))
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    mode_counts: dict[str, int] = {}
    for row in successes:
        mode = str(row["detour_mode"])
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    summary.update(
        {
            "mean_min_clearance_m": float(
                np.mean([float(row["min_clearance_m"]) for row in records])
            ),
            "mean_success_clearance_m": (
                float(np.mean([float(row["min_clearance_m"]) for row in successes]))
                if successes
                else None
            ),
            "mean_time_to_goal_s": (
                float(np.mean([float(row["time_to_goal_s"]) for row in successes]))
                if successes
                else None
            ),
            "mean_wall_time_s": float(
                np.mean([float(row["wall_time_s"]) for row in records])
            ),
            "mean_path_length_m": float(
                np.mean([float(row["path_length_m"]) for row in records])
            ),
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "mode_counts_successes": dict(sorted(mode_counts.items())),
            "successful_mode_coverage": len(mode_counts),
        }
    )
    return summary


def evaluate_fresh_ood_expert(
    env,
    *,
    device: torch.device,
    gammas: Sequence[float],
    attempts_per_gamma: int,
    seed0: int,
    planner_settings: SafeMPPIExpertSettings = SafeMPPIExpertSettings(),
    rollout_fn: Callable[..., Mapping[str, Any]] = run_expert_rollout,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run a fixed, non-stopping SafeMPPI attempt budget at every gamma."""

    if attempts_per_gamma <= 0:
        raise ValueError("attempts_per_gamma must be positive")
    if not gammas:
        raise ValueError("at least one gamma is required")
    config = DemoRunConfig(
        max_steps=300,
        reach_m=0.20,
        smooth_weight=planner_settings.smooth_weight,
        noise_var_mult=planner_settings.noise_var_mult,
        retreat_weight=planner_settings.retreat_weight,
        max_debug_candidates=6,
        max_proposals_per_step=8,
        quota_per_direction=1,
        max_candidate_seeds_per_gamma=attempts_per_gamma,
        seed0=seed0,
    )
    rows: list[dict[str, Any]] = []
    for gamma_index, gamma in enumerate(gammas):
        for attempt in range(attempts_per_gamma):
            seed = int(seed0 + gamma_index * attempts_per_gamma + attempt)
            episode = rollout_fn(
                env=env,
                gamma=float(gamma),
                seed=seed,
                device=device,
                config=config,
            )
            row = _normalize_expert_episode(episode, env)
            if not math.isclose(row["gamma"], float(gamma), abs_tol=0.0, rel_tol=0.0):
                raise RuntimeError("expert rollout returned the wrong gamma")
            if row["seed"] != seed:
                raise RuntimeError("expert rollout returned the wrong source seed")
            row["attempt_index"] = attempt
            rows.append(row)
    per_gamma = {
        f"{float(gamma):g}": summarize_expert_rollouts(
            row for row in rows if row["gamma"] == float(gamma)
        )
        for gamma in gammas
    }
    expected = attempts_per_gamma * len(gammas)
    if len(rows) != expected or any(
        int(per_gamma[f"{float(gamma):g}"]["n"]) != attempts_per_gamma
        for gamma in gammas
    ):
        raise RuntimeError("fresh expert evaluation did not retain every fixed-budget attempt")
    return rows, per_gamma


def save_expert_evaluation(
    output: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    per_gamma: Mapping[str, Mapping[str, Any]],
    attempts_per_gamma: int,
    seed0: int,
    planner_settings: SafeMPPIExpertSettings = SafeMPPIExpertSettings(),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": EXPERT_SCHEMA,
            "controller": "fresh OOD SafeMPPI",
            "attempt_policy": "fixed count per gamma; no success-conditioned stopping",
            "attempted_rollouts_per_gamma": int(attempts_per_gamma),
            "seed0": int(seed0),
            "expert_planner": planner_settings.to_dict(),
            "rollouts": [dict(row) for row in rows],
            "per_gamma": dict(per_gamma),
        },
        output,
    )


def make_baseline_rollout_bundle(
    *,
    rollouts: Sequence[Any],
    expert_rows: Sequence[Mapping[str, Any]],
    expert_per_gamma: Mapping[str, Mapping[str, Any]],
    expert_artifact: Path,
    expert_planner: SafeMPPIExpertSettings = SafeMPPIExpertSettings(),
) -> dict[str, Any]:
    """Build the Stage-04 PT bundle with an explicit ordinary T=1 contract."""

    ordinary_rows = [asdict(row) for row in rollouts]
    if any(float(row.get("temperature", math.nan)) != 1.0 for row in ordinary_rows):
        raise ValueError("Stage-04 baseline metrics accept only ordinary T=1 rollouts")
    expert_attempt_counts = {
        int(metrics["n"]) for metrics in expert_per_gamma.values()
    }
    if len(expert_attempt_counts) > 1:
        raise ValueError("expert evaluation must use one fixed attempt count at every gamma")
    return {
        "schema_version": BASELINE_SCHEMA,
        "temperature": 1.0,
        "ordinary_flow_temperature": 1.0,
        "ordinary_flow_evaluation": {
            "temperature": 1.0,
            "sampling_distribution": "ordinary conditional flow",
            "uncertainty_tilting": False,
            "safety_filter": False,
        },
        "rollouts": ordinary_rows,
        "expert": [dict(row) for row in expert_rows],
        "expert_evaluation": {
            "controller": "fresh OOD SafeMPPI",
            "artifact": str(expert_artifact),
            "expert_planner": expert_planner.to_dict(),
            "attempt_policy": "fixed count per gamma; failures retained",
            "attempted_rollouts_per_gamma": (
                next(iter(expert_attempt_counts)) if expert_attempt_counts else 0
            ),
            "per_gamma": dict(expert_per_gamma),
        },
    }


def audit_model(
    model: torch.nn.Module,
    env,
    bank: ImmutableContextBank,
    *,
    plans_per_context: int,
    seed: int,
    nfe: int,
    progress_threshold: float,
):
    goal = env.goal.detach().cpu().numpy()

    def sampler(model, audit_context, gamma, count, *, temperature, generator):
        context = context_from_state(
            audit_context.state,
            goal,
            gamma,
            audit_context.executed_history,
            env,
        )
        return sample_plans(
            model,
            context,
            count,
            temperature=temperature,
            nfe=nfe,
            generator=generator,
        )

    def verifier(audit_context, gamma, plan):
        result = verify_plan(audit_context.state, plan, env, gamma, goal=goal)
        return {
            "in_bounds": result.in_bounds,
            "socp_ok": result.socp_ok,
            "safe": result.safe,
            "progress_m": result.progress_m,
            "mode": local_plan_mode(audit_context.state, goal, result.positions),
        }

    return run_independent_audit(
        model,
        bank,
        GAMMAS,
        sampler,
        verifier,
        AuditConfig(
            plans_per_context=plans_per_context,
            progress_threshold=progress_threshold,
            seed=seed,
            temperature=1.0,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=STAGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=94000)
    parser.add_argument("--audit-contexts-per-mode", type=int, default=8)
    parser.add_argument("--audit-plans-per-context", type=int, default=4)
    parser.add_argument("--expert-candidate-limit", type=int, default=48)
    parser.add_argument("--sealed-interior-contexts-per-mode", type=int, default=2)
    parser.add_argument("--sealed-candidate-limit", type=int, default=128)
    parser.add_argument("--sealed-seed-offset", type=int, default=10_000_000)
    parser.add_argument("--expert-eval-rollouts-per-gamma", type=int, default=8)
    parser.add_argument("--expert-eval-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--expert-smooth-weight", type=float, default=8.0)
    parser.add_argument("--expert-noise-var-mult", type=float, default=3.0)
    parser.add_argument("--expert-retreat-weight", type=float, default=1.0)
    parser.add_argument("--eval-rollouts", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--progress-threshold", type=float, default=0.10)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    env = make_ood_scene(radius=1.2)
    model, checkpoint = HP.load_hp(args.checkpoint, device=device)
    require_promoted_fresh_pretrain(model, checkpoint)
    expert_planner = SafeMPPIExpertSettings(
        smooth_weight=args.expert_smooth_weight,
        noise_var_mult=args.expert_noise_var_mult,
        retreat_weight=args.expert_retreat_weight,
    )
    scene_verifier_fingerprint = verifier_spec_fingerprint(env, env.goal)

    bank, audit_source_episodes = build_fixed_audit_bank(
        env,
        device=device,
        seed0=args.seed,
        contexts_per_mode=args.audit_contexts_per_mode,
        candidate_limit=args.expert_candidate_limit,
        planner_settings=expert_planner,
    )
    bank_path = args.outdir / "data/fixed_audit_bank.pt"
    monitoring_provenance = {
        "schema_version": "afe_audit_bank_source_provenance_v1",
        "purpose": "round_monitoring",
        "expert_planner": expert_planner.to_dict(),
        "expert_rollout": {
            "gamma": 0.5,
            "max_steps": 300,
            "reach_m": 0.20,
            "max_debug_candidates": 6,
            "max_proposals_per_step": 8,
            "seed0": args.seed,
            "candidate_limit": args.expert_candidate_limit,
        },
        "contexts_per_mode": args.audit_contexts_per_mode,
        "selected_sources": [{
            "seed": int(row["seed"]),
            "mode": detour_mode(row["path"]),
            "steps": int(row["steps"]),
        } for row in audit_source_episodes],
        "scene_verifier_spec_fingerprint": scene_verifier_fingerprint,
    }
    monitoring_bank_metadata = save_audit_bank(
        bank, bank_path, provenance=monitoring_provenance,
    )
    sealed_bank, sealed_provenance = build_sealed_final_test_bank(
        env,
        device=device,
        seed0=args.seed + args.sealed_seed_offset,
        interior_contexts_per_mode=args.sealed_interior_contexts_per_mode,
        candidate_limit=args.sealed_candidate_limit,
        planner_settings=expert_planner,
    )
    sealed_bank_path = args.outdir / "data/sealed_final_test_bank.pt"
    sealed_provenance_payload = {
        "schema_version": "afe_audit_bank_source_provenance_v1",
        "purpose": "sealed_final_test",
        "expert_planner": expert_planner.to_dict(),
        "expert_rollout": {
            "gamma": 0.5,
            "max_steps": 300,
            "reach_m": 0.20,
            "max_debug_candidates": 6,
            "max_proposals_per_step": 8,
            "seed0": args.seed + args.sealed_seed_offset,
            "candidate_limit": args.sealed_candidate_limit,
        },
        "interior_contexts_per_mode": args.sealed_interior_contexts_per_mode,
        "candidate_attempts": sealed_provenance,
        "selected_context_sources": [{
            "expert_seed": int(context.expert_seed),
            "expert_mode": str(context.expert_mode),
            "source_step": int(context.source_step),
        } for context in sealed_bank],
        "scene_verifier_spec_fingerprint": scene_verifier_fingerprint,
    }
    sealed_bank_metadata = save_audit_bank(
        sealed_bank, sealed_bank_path, provenance=sealed_provenance_payload,
    )
    audit = audit_model(
        model,
        env,
        bank,
        plans_per_context=args.audit_plans_per_context,
        seed=args.seed + 1,
        nfe=args.nfe,
        progress_threshold=args.progress_threshold,
    )

    expert_seed0 = args.seed + args.expert_eval_seed_offset
    audit_seed_stop = args.seed + args.expert_candidate_limit
    expert_seed_stop = expert_seed0 + len(GAMMAS) * args.expert_eval_rollouts_per_gamma
    if max(args.seed, expert_seed0) < min(audit_seed_stop, expert_seed_stop):
        raise ValueError("fresh expert-evaluation seeds overlap fixed audit-bank source seeds")
    expert_rows, expert_per_gamma = evaluate_fresh_ood_expert(
        env,
        device=device,
        gammas=GAMMAS,
        attempts_per_gamma=args.expert_eval_rollouts_per_gamma,
        seed0=expert_seed0,
        planner_settings=expert_planner,
    )
    expert_path = args.outdir / "data/ood_safemppi_expert_rollouts.pt"
    save_expert_evaluation(
        expert_path,
        rows=expert_rows,
        per_gamma=expert_per_gamma,
        attempts_per_gamma=args.expert_eval_rollouts_per_gamma,
        seed0=expert_seed0,
        planner_settings=expert_planner,
    )

    rollouts = []
    for gamma_index, gamma in enumerate(GAMMAS):
        for index in range(args.eval_rollouts):
            rollouts.append(rollout_ordinary_flow(
                model,
                env,
                gamma,
                seed=args.seed + 10_000 + gamma_index * 1000 + index,
                temperature=1.0,
                nfe=args.nfe,
            ))
    per_gamma = {
        str(gamma): summarize_rollouts([row for row in rollouts if row.gamma == gamma])
        for gamma in GAMMAS
    }
    args.outdir.joinpath("data").mkdir(parents=True, exist_ok=True)
    baseline_bundle = make_baseline_rollout_bundle(
        rollouts=rollouts,
        expert_rows=expert_rows,
        expert_per_gamma=expert_per_gamma,
        expert_artifact=expert_path,
        expert_planner=expert_planner,
    )
    torch.save(baseline_bundle, args.outdir / "data/baseline_rollouts.pt")
    summary = {
        "status": "OOD_BASELINE_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_schema": checkpoint.get("config", {}),
        "scene": {"giant_radius": 1.2, "start": [0.5, 0.5], "goal": [4.5, 4.5]},
        "safemppi_expert_planner": expert_planner.to_dict(),
        "audit": audit.to_dict(),
        "audit_bank": {
            "path": str(bank_path),
            "fingerprint": bank.fingerprint,
            "role": bank.role,
            "training_use_forbidden": True,
            "source_seeds": [int(row["seed"]) for row in audit_source_episodes],
            "source_provenance_fingerprint": monitoring_bank_metadata[
                "source_provenance_fingerprint"
            ],
            "artifact_fingerprint": monitoring_bank_metadata["artifact_fingerprint"],
        },
        "sealed_final_test_bank": {
            "path": str(sealed_bank_path),
            "fingerprint": sealed_bank.fingerprint,
            "role": sealed_bank.role,
            "context_count": len(sealed_bank),
            "deployment_start_included": True,
            "interior_contexts_use_distinct_seeds": True,
            "flow_outcomes_evaluated_during_round_tuning": False,
            "candidate_attempts": len(sealed_provenance),
            "source_provenance_fingerprint": sealed_bank_metadata[
                "source_provenance_fingerprint"
            ],
            "artifact_fingerprint": sealed_bank_metadata["artifact_fingerprint"],
        },
        "ordinary_flow_temperature": 1.0,
        "ordinary_flow_sampling_distribution": "ordinary conditional flow at T=1",
        "ordinary_rollouts_per_gamma": args.eval_rollouts,
        "ordinary_per_gamma": per_gamma,
        "fresh_ood_safemppi_expert": {
            "artifact": str(expert_path),
            "seed0": expert_seed0,
            "attempted_rollouts_per_gamma": args.expert_eval_rollouts_per_gamma,
            "failures_retained": True,
            "expert_planner": expert_planner.to_dict(),
            "per_gamma": expert_per_gamma,
        },
    }
    _write_json(args.outdir / "logs/summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
