"""Matched single-axis controls for planned-window AFE.

The three controls in this module deliberately leave candidate count, real
verifier calls, expansion rounds, uniform replay, and the proximal numerical
solver untouched:

``-AFE``
    Query candidates uniformly while retaining the real fixed-feature sigma
    in telemetry and updating the same cumulative uncertainty matrix.
``-Progress``
    Select the first verifier-eligible candidate rather than ranking eligible
    candidates by progress.  Progress remains a logged diagnostic.
``-SOCP``
    An OFFLINE-ONLY control whose acquisition/training eligibility predicate
    is strict task-space bounds.  The full SOCP is nevertheless run on every
    query and its actual outcome is retained.  This arm makes no runtime
    certificate claim.

In particular, ``-SOCP`` never fabricates ``socp_certified=True`` and never
mutates a :class:`VerificationRecord`.  Its proximal solver receives a
separate identity-checked view over strict-bounds rows.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from .schemas import (
    QueryContext,
    QuerySource,
    VerificationRecord,
    query_content_hash,
)


class AblationArm(str, Enum):
    MINUS_AFE = "minus_afe"
    MINUS_PROGRESS = "minus_progress"
    MINUS_SOCP = "minus_socp"


@dataclass(frozen=True)
class AblationSpec:
    """The only switches permitted to differ in a matched control."""

    arm: AblationArm
    display_name: str
    acquisition_mode: str
    progress_ranking: bool
    eligibility_mode: str
    replay_eligibility: str
    runtime_safety_claim: bool
    scientific_scope: str

    def __post_init__(self) -> None:
        if self.acquisition_mode not in {"afe", "uniform"}:
            raise ValueError("unknown acquisition mode")
        if self.eligibility_mode not in {"full", "bounds_only_offline"}:
            raise ValueError("unknown eligibility mode")
        if self.replay_eligibility not in {"full_safe", "strict_bounds"}:
            raise ValueError("unknown replay eligibility")
        if self.eligibility_mode == "bounds_only_offline" and self.runtime_safety_claim:
            raise ValueError("bounds-only control cannot carry a runtime-safety claim")


SPECS: dict[AblationArm, AblationSpec] = {
    AblationArm.MINUS_AFE: AblationSpec(
        arm=AblationArm.MINUS_AFE,
        display_name=r"$-\mathrm{AFE}$",
        acquisition_mode="uniform",
        progress_ranking=True,
        eligibility_mode="full",
        replay_eligibility="full_safe",
        runtime_safety_claim=True,
        scientific_scope="remove uncertainty tilting only",
    ),
    AblationArm.MINUS_PROGRESS: AblationSpec(
        arm=AblationArm.MINUS_PROGRESS,
        display_name=r"$-\mathrm{Progress}$",
        acquisition_mode="afe",
        progress_ranking=False,
        eligibility_mode="full",
        replay_eligibility="full_safe",
        runtime_safety_claim=True,
        scientific_scope="remove progress ranking among full-verified-safe plans only",
    ),
    AblationArm.MINUS_SOCP: AblationSpec(
        arm=AblationArm.MINUS_SOCP,
        display_name=r"$-\mathrm{SOCP}$ (offline only)",
        acquisition_mode="afe",
        progress_ranking=True,
        eligibility_mode="bounds_only_offline",
        replay_eligibility="strict_bounds",
        runtime_safety_claim=False,
        scientific_scope=(
            "remove SOCP from control/training eligibility while retaining the "
            "actual full-verifier outcome in telemetry"
        ),
    ),
}


def ablation_spec(arm: AblationArm | str) -> AblationSpec:
    return SPECS[AblationArm(arm)]


@dataclass(frozen=True)
class MatchedProtocol:
    """Settings that must be identical across all three controls."""

    seed: int
    candidate_count: int
    verifier_budget: int
    fallback_verifier_budget: int
    beta: float
    backup_smooth_weight: float
    backup_noise_var_mult: float
    backup_retreat_weight: float
    rounds: int
    episodes_per_gamma: int
    episode_max_steps: int
    expansion_temperature: float
    nfe: int
    ridge_lambda: float
    prox_eta: float
    learning_rate: float
    microbatch: int
    solver_max_steps: int
    solver_min_steps: int
    update_norm_limit: float
    relative_loss_tolerance: float
    gradient_tolerance: float
    audit_plans_per_context: int
    audit_progress_threshold: float
    eval_rollouts: int

    def __post_init__(self) -> None:
        integer_positive = (
            self.candidate_count,
            self.verifier_budget,
            self.fallback_verifier_budget,
            self.rounds,
            self.episodes_per_gamma,
            self.episode_max_steps,
            self.nfe,
            self.microbatch,
            self.solver_max_steps,
            self.audit_plans_per_context,
            self.eval_rollouts,
        )
        if any(value <= 0 for value in integer_positive):
            raise ValueError("matched protocol counts must be positive")
        if self.verifier_budget > self.candidate_count:
            raise ValueError("verifier budget cannot exceed candidate count")
        if not 0 <= self.solver_min_steps <= self.solver_max_steps:
            raise ValueError("solver_min_steps must lie in [0, solver_max_steps]")
        positive_reals = (
            self.expansion_temperature,
            self.ridge_lambda,
            self.prox_eta,
            self.learning_rate,
            self.update_norm_limit,
            self.beta,
            self.backup_noise_var_mult,
            self.audit_progress_threshold,
        )
        if any((not np.isfinite(value)) or value <= 0 for value in positive_reals):
            raise ValueError("matched protocol real-valued scales must be positive")
        if not np.isfinite(self.backup_smooth_weight) or self.backup_smooth_weight < 0:
            raise ValueError("backup smoothness weight must be finite and nonnegative")
        if not np.isfinite(self.backup_retreat_weight) or self.backup_retreat_weight < 0:
            raise ValueError("backup retreat weight must be finite and nonnegative")

    @property
    def scheduled_flow_query_budget(self) -> int:
        """Per-gamma budget absent cache hits, fallback, or early termination."""

        return self.rounds * self.episodes_per_gamma * self.episode_max_steps * self.verifier_budget


def assert_matched_protocols(protocols: Iterable[MatchedProtocol]) -> MatchedProtocol:
    rows = tuple(protocols)
    if not rows:
        raise ValueError("at least one protocol is required")
    reference = rows[0]
    if any(row != reference for row in rows[1:]):
        raise ValueError("ablation protocols are not matched")
    return reference


@dataclass(frozen=True, eq=False)
class OfflineBoundsReplayItem:
    """Identity-preserving strict-bounds target for the offline ``-SOCP`` arm.

    The hashes make this object consumable by the unchanged uniform proximal
    solver without changing the actual safety label in the source ledger.
    ``actual_socp_certified`` remains available for validity reporting.
    """

    context: QueryContext
    gamma: float
    plan: np.ndarray
    source_query_hash: str
    training_target_hash: str
    actual_socp_certified: bool
    actual_full_safe: bool
    source_executed: bool

    @classmethod
    def from_record(cls, record: VerificationRecord) -> "OfflineBoundsReplayItem":
        record.validate_identity()
        if not record.safety.strict_bounds:
            raise ValueError("offline bounds replay requires strict_bounds=True")
        expected = query_content_hash(record.context, record.gamma, record.plan)
        return cls(
            context=record.context,
            gamma=record.gamma,
            plan=np.array(record.plan, copy=True),
            source_query_hash=expected,
            training_target_hash=expected,
            actual_socp_certified=record.safety.socp_certified,
            actual_full_safe=record.safe,
            source_executed=record.executed,
        )

    def __post_init__(self) -> None:
        expected = query_content_hash(self.context, self.gamma, self.plan)
        if self.source_query_hash != expected or self.training_target_hash != expected:
            raise ValueError("offline bounds replay identity mismatch")
        # Preserve the source dtype bit-for-bit: query identity hashes include
        # dtype as well as values, so a float64-to-float32 cast would silently
        # turn the replay target into a different object.
        plan = np.array(self.plan, copy=True)
        if plan.shape != (10, 2) or not np.isfinite(plan).all():
            raise ValueError("offline bounds replay plan must have shape (10,2)")
        plan.setflags(write=False)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "actual_socp_certified", bool(self.actual_socp_certified))
        object.__setattr__(self, "actual_full_safe", bool(self.actual_full_safe))
        object.__setattr__(self, "source_executed", bool(self.source_executed))
        if self.actual_full_safe != self.actual_socp_certified:
            # The constructor is reachable only for strict-bounds records.
            raise ValueError("actual_full_safe must equal actual SOCP for a strict-bounds row")
        if self.source_executed and not self.actual_full_safe:
            raise ValueError("an actual SOCP failure cannot be executed in the certified ledger")

    def validate_identity(self) -> None:
        expected = query_content_hash(self.context, self.gamma, self.plan)
        if expected != self.source_query_hash or expected != self.training_target_hash:
            raise ValueError("offline bounds replay identity mismatch")


class OfflineBoundsReplayView(Sequence[OfflineBoundsReplayItem]):
    """Uniform, immutable strict-bounds view without altered verifier labels."""

    def __init__(self, records: Iterable[VerificationRecord]) -> None:
        self._items = tuple(
            OfflineBoundsReplayItem.from_record(record)
            for record in records
            if record.safety.strict_bounds
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int | slice):
        return self._items[index]

    def __iter__(self) -> Iterator[OfflineBoundsReplayItem]:
        return iter(self._items)


def training_view(
    records: Iterable[VerificationRecord], spec: AblationSpec,
) -> tuple[VerificationRecord, ...] | OfflineBoundsReplayView:
    """Return the uniformly replayed rows for one arm.

    Full-label controls return the immutable ledger rows and rely on the
    proximal solver's normal safety filtering.  ``-SOCP`` returns an explicit
    bounds-only wrapper and leaves the ledger untouched.
    """

    rows = tuple(
        record for record in records if record.source is QuerySource.FLOW
    )
    if spec.replay_eligibility == "strict_bounds":
        return OfflineBoundsReplayView(rows)
    return rows


def eligibility_counts(
    records: Iterable[VerificationRecord], spec: AblationSpec,
) -> dict[str, int | float]:
    all_rows = tuple(records)
    rows = tuple(
        record for record in all_rows if record.source is QuerySource.FLOW
    )
    backup_rows = tuple(
        record
        for record in all_rows
        if record.source is QuerySource.SAFEMPPI_BACKUP
    )
    actual_safe = sum(record.safe for record in rows)
    strict_bounds = sum(record.safety.strict_bounds for record in rows)
    eligible = strict_bounds if spec.replay_eligibility == "strict_bounds" else actual_safe
    return {
        "queried": len(rows),
        "actual_full_safe": actual_safe,
        "strict_bounds": strict_bounds,
        "training_eligible": eligible,
        "actual_socp_failures_inside_bounds": sum(
            record.safety.strict_bounds and not record.safety.socp_certified
            for record in rows
        ),
        "actual_full_acceptance": actual_safe / len(rows) if rows else float("nan"),
        "training_eligibility_rate": eligible / len(rows) if rows else float("nan"),
        "backup_verifier_calls": len(backup_rows),
        "backup_actual_full_safe": sum(record.safe for record in backup_rows),
    }


def arm_manifest(spec: AblationSpec, protocol: MatchedProtocol) -> dict[str, object]:
    return {
        "arm": spec.arm.value,
        "display_name": spec.display_name,
        "scientific_scope": spec.scientific_scope,
        "acquisition_mode": spec.acquisition_mode,
        "progress_ranking": spec.progress_ranking,
        "eligibility_mode": spec.eligibility_mode,
        "replay_eligibility": spec.replay_eligibility,
        "runtime_safety_claim": spec.runtime_safety_claim,
        "actual_socp_always_evaluated_and_logged": True,
        "uncertainty_updated_for_every_real_query_positive_or_negative": True,
        "uniform_replay_no_frontier_weighting": True,
        "matched_protocol": protocol.__dict__,
    }
