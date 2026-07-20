"""Receding-horizon controller for identical generated/queried/trained plans."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from .acquisition import acquire_without_replacement
from .config import AFEConfig, DEFAULT_CONFIG
from .dynamics import execute_first_action
from .fallback import SafeMPPIBackup
from .policy import FrozenFeatureModel, sample_plans
from .scene import context_from_state, verifier_spec_fingerprint
from .schemas import (
    ProgressResult,
    QueryContext,
    QuerySource,
    SafetyResult,
    VerificationRecord,
    query_content_hash,
)
from .store import VerificationStore
from .verifier import PlanVerification, verify_plan


VerifierFunction = Callable[..., PlanVerification]


@dataclass(frozen=True)
class QueriedPlanTrace:
    candidate_index: int
    query_hash: str
    source: str
    plan_kind: str
    acquisition_sigma: float
    safe: bool
    in_bounds: bool
    socp_ok: bool
    progress_m: float
    clearance_m: float
    cache_hit: bool
    executed: bool


@dataclass(frozen=True)
class ControlStepTrace:
    step: int
    gamma: float
    state_before: np.ndarray
    candidate_plans: np.ndarray
    candidate_sigmas: np.ndarray
    acquisition_probabilities: np.ndarray
    acquisition_order: np.ndarray
    queried: tuple[QueriedPlanTrace, ...]
    verifier_calls: int
    cache_hits: int
    selected_query_hash: str | None
    selected_source: str | None
    action: np.ndarray | None
    state_after: np.ndarray
    fallback_used: bool
    fail_closed: bool
    acquisition_entropy: float
    acquisition_ess: float
    eligibility_mode: str = "full"
    runtime_safety_claim: bool = True
    selected_actual_full_safe: bool | None = None

    def to_state_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["queried"] = [asdict(item) for item in self.queried]
        return result


@dataclass(frozen=True)
class EpisodeResult:
    gamma: float
    seed: int
    states: np.ndarray
    actions: np.ndarray
    reached: bool
    collision: bool
    in_bounds: bool
    fail_closed: bool
    fallback_steps: int
    verifier_calls: int
    query_positives: int
    cache_hits: int
    traces: tuple[ControlStepTrace, ...]
    eligibility_mode: str = "full"
    runtime_safety_claim: bool = True

    @property
    def success(self) -> bool:
        return self.reached and not self.collision and self.in_bounds and not self.fail_closed

    @property
    def query_acceptance(self) -> float:
        return self.query_positives / self.verifier_calls if self.verifier_calls else float("nan")


def _safety(result: PlanVerification) -> SafetyResult:
    return SafetyResult(
        strict_bounds=result.in_bounds,
        socp_certified=result.socp_ok,
        min_clearance=result.physical_clearance_m,
        certificate_slack=result.certificate_residual,
        feasible_face_margin=result.face_margin_m,
    )


def _progress(result: PlanVerification) -> ProgressResult:
    return ProgressResult(
        initial_goal_distance=result.start_goal_distance_m,
        terminal_goal_distance=result.terminal_goal_distance_m,
    )


def _runtime_status(position: np.ndarray, env) -> tuple[bool, bool, float]:
    obstacles = env.obstacles.detach().cpu().numpy()
    clearance = (
        np.linalg.norm(position[None] - obstacles[:, :2], axis=1)
        - obstacles[:, 2]
        - float(env.r_robot)
    )
    minimum = float(clearance.min()) if len(clearance) else float("inf")
    in_bounds = bool(((position >= 0.0) & (position <= 5.0)).all())
    return in_bounds, minimum < -1e-7, minimum


class PlannedWindowAFEController:
    """AFE query controller with same-verifier SafeMPPI fallback."""

    def __init__(
        self,
        model: torch.nn.Module,
        frozen_features: FrozenFeatureModel,
        store: VerificationStore,
        *,
        config: AFEConfig = DEFAULT_CONFIG,
        backup: SafeMPPIBackup | None = None,
        verifier_fn: VerifierFunction = verify_plan,
        device: str | torch.device = "cuda:0",
        fallback_verifier_budget: int = 8,
        acquisition_mode: str = "afe",
        progress_ranking: bool = True,
        eligibility_mode: str = "full",
    ) -> None:
        self.model = model
        self.frozen_features = frozen_features
        self.store = store
        self.config = config
        self.backup = backup or SafeMPPIBackup()
        self.verifier_fn = verifier_fn
        self.device = torch.device(device)
        self.fallback_verifier_budget = int(fallback_verifier_budget)
        if acquisition_mode not in {"afe", "uniform"}:
            raise ValueError("acquisition_mode must be 'afe' or 'uniform'")
        self.acquisition_mode = acquisition_mode
        self.progress_ranking = bool(progress_ranking)
        if eligibility_mode not in {"full", "bounds_only_offline"}:
            raise ValueError(
                "eligibility_mode must be 'full' or 'bounds_only_offline'"
            )
        self.eligibility_mode = eligibility_mode
        if self.fallback_verifier_budget <= 0:
            raise ValueError("fallback_verifier_budget must be positive")
        if next(model.parameters()).device != self.device:
            raise ValueError("model and controller must use the same device")
        if self.store.uncertainty.feature_dim != config.features.representation_dim:
            raise ValueError("uncertainty and frozen representation dimensions differ")

    @property
    def runtime_safety_claim(self) -> bool:
        """Whether selected plans must carry the actual full SOCP label."""

        return self.eligibility_mode == "full"

    def _eligible_result(self, result: PlanVerification) -> bool:
        return result.safe if self.runtime_safety_claim else result.in_bounds

    def _eligible_record(self, record: VerificationRecord) -> bool:
        return record.safe if self.runtime_safety_claim else record.safety.strict_bounds

    def _record(
        self,
        context: QueryContext,
        gamma: float,
        plan: np.ndarray,
        source: QuerySource,
        feature: np.ndarray,
        sigma: float,
        result: PlanVerification,
        *,
        executed: bool,
    ) -> VerificationRecord:
        content_hash = query_content_hash(context, gamma, plan)
        return VerificationRecord(
            context=context,
            gamma=gamma,
            plan=plan,
            source=source,
            feature_z=feature,
            acquisition_sigma=sigma,
            safety=_safety(result),
            progress=_progress(result),
            executed=executed,
            generated_hash=content_hash,
            verifier_input_hash=content_hash,
        )

    def _assert_exact_verifier_context(
        self,
        context: QueryContext,
        state: np.ndarray,
        env,
    ) -> None:
        """Reject any cache/query key that is not the literal verifier input."""

        current = np.asarray(state, dtype=np.float64)
        if not np.array_equal(context.verifier_state, current):
            raise RuntimeError(
                "query context verifier_state differs from the state submitted "
                "to the full verifier"
            )
        expected_spec = verifier_spec_fingerprint(
            env,
            env.goal,
            dynamics=self.config.dynamics,
            verifier=self.config.verifier,
        )
        if context.verifier_spec_fingerprint != expected_spec:
            raise RuntimeError(
                "query context scene/goal/dynamics/verifier fingerprint mismatch"
            )

    def _verify_flow_batch(
        self,
        state: np.ndarray,
        context: QueryContext,
        gamma: float,
        env,
        plans: np.ndarray,
        features: np.ndarray,
        sigmas: np.ndarray,
        order: np.ndarray,
    ) -> tuple[
        list[VerificationRecord],
        list[VerificationRecord],
        list[tuple[int, VerificationRecord]],
        int,
        VerificationRecord | None,
    ]:
        """Return ledger rows and the arm-eligible control selection.

        In the offline bounds-only control the returned selection may retain
        an actual failed SOCP label.  Such a row is never marked ``executed``
        in the certified ledger; the enclosing control trace references it.
        """
        self._assert_exact_verifier_context(context, state, env)
        new_results: list[tuple[int, np.ndarray, np.ndarray, float, PlanVerification]] = []
        cached_safe: list[VerificationRecord] = []
        eligible_in_acquisition_order: list[tuple[str, int, float, int]] = []
        seen_this_batch: set[str] = set()
        cache_hits = 0
        for acquisition_rank, candidate_index in enumerate(order):
            plan = plans[int(candidate_index)]
            query_hash = query_content_hash(context, gamma, plan)
            if query_hash in seen_this_batch:
                cache_hits += 1
                continue
            seen_this_batch.add(query_hash)
            cached = self.store.get(query_hash)
            if cached is not None:
                cache_hits += 1
                if self._eligible_record(cached):
                    cached_safe.append(cached)
                    eligible_in_acquisition_order.append(
                        ("cached", len(cached_safe) - 1, cached.progress_value, acquisition_rank)
                    )
                continue
            result = self.verifier_fn(
                context.verifier_state,
                plan,
                env,
                gamma,
                goal=env.goal,
                dynamics=self.config.dynamics,
                verifier=self.config.verifier,
            )
            new_results.append(
                (int(candidate_index), plan, features[int(candidate_index)], float(sigmas[int(candidate_index)]), result)
            )
            if self._eligible_result(result):
                eligible_in_acquisition_order.append(
                    ("new", len(new_results) - 1, result.progress_m, acquisition_rank)
                )
            if len(new_results) >= self.config.sampling.verifier_budget:
                break

        chosen_position: int | None = None
        chosen_kind: str | None = None
        if eligible_in_acquisition_order:
            if self.progress_ranking:
                # Preserve the full method's pre-ablation ranking/tie rule:
                # new rows first, then cached rows, with candidate index as
                # the tie breaker for newly queried plans.
                ranked_eligible = [
                    ("new", position, result.progress_m, index)
                    for position, (index, _plan, _feature, _sigma, result)
                    in enumerate(new_results)
                    if self._eligible_result(result)
                ] + [
                    ("cached", position, cached.progress_value, position)
                    for position, cached in enumerate(cached_safe)
                ]
                chosen_kind, chosen_position, _progress_value, _rank = max(
                    ranked_eligible,
                    key=lambda item: (item[2], -item[3]),
                )
            else:
                chosen_kind, chosen_position, _progress_value, _rank = (
                    eligible_in_acquisition_order[0]
                )

        new_records = [
            self._record(
                context,
                gamma,
                plan,
                QuerySource.FLOW,
                feature,
                sigma,
                result,
                # VerificationRecord.executed denotes certified execution.
                # A bounds-only offline selection with a failed actual SOCP
                # is referenced only by ControlStepTrace.selected_query_hash.
                executed=(
                    chosen_kind == "new"
                    and position == chosen_position
                    and result.safe
                ),
            )
            for position, (index, plan, feature, sigma, result) in enumerate(new_results)
        ]
        indexed = [(row[0], record) for row, record in zip(new_results, new_records)]
        # Every score was computed against the same pre-batch A_n.
        if new_records:
            self.store.append_batch(new_records)
        chosen = (
            new_records[chosen_position]
            if chosen_kind == "new" and chosen_position is not None
            else cached_safe[chosen_position]
            if chosen_kind == "cached" and chosen_position is not None
            else None
        )
        return new_records, cached_safe, indexed, cache_hits, chosen

    def _verify_backup_batch(
        self,
        state: np.ndarray,
        context: QueryContext,
        gamma: float,
        env,
        *,
        seed: int,
    ) -> tuple[list[VerificationRecord], VerificationRecord | None, list[str], int]:
        self._assert_exact_verifier_context(context, state, env)
        proposals, _telemetry = self.backup.propose(
            state,
            env.goal.detach().cpu().numpy(),
            env,
            gamma,
            seed=seed,
            device=self.device,
        )
        plans = np.asarray([proposal.plan for proposal in proposals], dtype=np.float32)
        if len(plans) == 0:
            return [], None, [], 0
        features = self.frozen_features.encode(context, plans)
        sigmas = self.store.uncertainty.sigmas(features)
        new_rows: list[tuple[int, object, PlanVerification]] = []
        cached_safe: list[VerificationRecord] = []
        cache_hits = 0
        kinds: list[str] = []
        eligible_in_proposal_order: list[tuple[str, int, float, int]] = []
        seen: set[str] = set()
        for proposal_rank, proposal in enumerate(proposals):
            index = proposal_rank
            query_hash = query_content_hash(context, gamma, proposal.plan)
            if query_hash in seen:
                cache_hits += 1
                continue
            seen.add(query_hash)
            cached = self.store.get(query_hash)
            if cached is not None:
                cache_hits += 1
                if self._eligible_record(cached):
                    cached_safe.append(cached)
                    eligible_in_proposal_order.append(
                        ("cached", len(cached_safe) - 1, cached.progress_value, proposal_rank)
                    )
                continue
            result = self.verifier_fn(
                context.verifier_state,
                proposal.plan,
                env,
                gamma,
                goal=env.goal,
                dynamics=self.config.dynamics,
                verifier=self.config.verifier,
            )
            new_rows.append((index, proposal, result))
            kinds.append(proposal.kind)
            if self._eligible_result(result):
                eligible_in_proposal_order.append(
                    ("new", len(new_rows) - 1, result.progress_m, proposal_rank)
                )
            if len(new_rows) >= self.fallback_verifier_budget:
                break

        chosen_kind: str | None = None
        chosen_position: int | None = None
        if eligible_in_proposal_order:
            if self.progress_ranking:
                # Same new-then-cached ordering and tie rule as the full
                # controller before the ablation switch was introduced.
                ranked_eligible = [
                    ("new", position, result.progress_m, position)
                    for position, (_index, _proposal, result) in enumerate(new_rows)
                    if self._eligible_result(result)
                ] + [
                    ("cached", position, cached.progress_value, position)
                    for position, cached in enumerate(cached_safe)
                ]
                chosen_kind, chosen_position, _value, _rank = max(
                    ranked_eligible,
                    key=lambda item: (item[2], -item[3]),
                )
            else:
                chosen_kind, chosen_position, _value, _rank = (
                    eligible_in_proposal_order[0]
                )
        records = []
        for position, (index, proposal, result) in enumerate(new_rows):
            records.append(self._record(
                context,
                gamma,
                proposal.plan,
                QuerySource.SAFEMPPI_BACKUP,
                features[index],
                float(sigmas[index]),
                result,
                executed=(
                    chosen_kind == "new"
                    and chosen_position == position
                    and result.safe
                ),
            ))
        if records:
            self.store.append_batch(records)
        chosen = (
            records[chosen_position]
            if chosen_kind == "new" and chosen_position is not None
            else cached_safe[chosen_position]
            if chosen_kind == "cached" and chosen_position is not None
            else None
        )
        return records, chosen, kinds, cache_hits

    def run_episode(
        self,
        env,
        gamma: float,
        *,
        seed: int,
        max_steps: int | None = None,
        reach: float = 0.15,
    ) -> EpisodeResult:
        if float(env.dt) != self.config.dynamics.dt:
            raise ValueError("environment and verifier dynamics dt differ")
        # SafeMPPI uses warm starts within an episode.  Reset its hidden
        # proposal state exactly once here so gamma/episode order cannot leak
        # into the first backup query of a new rollout.
        reset_backup = getattr(self.backup, "reset", None)
        if callable(reset_backup):
            reset_backup()
        steps_limit = int(env.T if max_steps is None else max_steps)
        state = env.x0.detach().cpu().numpy().astype(np.float64)
        goal = env.goal.detach().cpu().numpy().astype(np.float64)
        states = [state.copy()]
        actions: list[np.ndarray] = []
        traces: list[ControlStepTrace] = []
        fallback_steps = verifier_calls = positives = cache_hits_total = 0
        fail_closed = collision = False
        in_bounds = True
        sample_generator = torch.Generator(device=self.device).manual_seed(int(seed))
        acquisition_generator = torch.Generator().manual_seed(int(seed) ^ 0x5AFE2026)

        for step in range(steps_limit):
            context = context_from_state(
                state,
                goal,
                gamma,
                actions,
                env,
                dynamics=self.config.dynamics,
                verifier=self.config.verifier,
            )
            plans = sample_plans(
                self.model,
                context,
                self.config.sampling.candidate_count,
                temperature=self.config.sampling.expansion_temperature,
                nfe=self.config.sampling.nfe,
                generator=sample_generator,
            )
            features = self.frozen_features.encode(context, plans)
            sigmas = self.store.uncertainty.sigmas(features)
            acquisition_scores = (
                sigmas if self.acquisition_mode == "afe" else np.zeros_like(sigmas)
            )
            acquisition = acquire_without_replacement(
                acquisition_scores,
                len(sigmas),
                self.config.sampling.beta,
                generator=acquisition_generator,
            )
            new_flow, _cached_flow, indexed_flow, cache_hits, selected = self._verify_flow_batch(
                state, context, gamma, env, plans, features, sigmas, acquisition.indices,
            )
            verifier_calls += len(new_flow)
            positives += sum(record.safe for record in new_flow)
            cache_hits_total += cache_hits
            queried_traces: list[QueriedPlanTrace] = []
            for candidate_index, record in indexed_flow:
                queried_traces.append(QueriedPlanTrace(
                    candidate_index=candidate_index,
                    query_hash=record.query_hash,
                    source=record.source.value,
                    plan_kind="flow",
                    acquisition_sigma=record.acquisition_sigma,
                    safe=record.safe,
                    in_bounds=record.safety.strict_bounds,
                    socp_ok=record.safety.socp_certified,
                    progress_m=record.progress_value,
                    clearance_m=record.safety.min_clearance,
                    cache_hit=False,
                    executed=bool(selected and selected.query_hash == record.query_hash),
                ))

            fallback_used = selected is None
            if fallback_used:
                fallback_steps += 1
                backup_records, selected, backup_kinds, backup_cache_hits = self._verify_backup_batch(
                    state,
                    context,
                    gamma,
                    env,
                    seed=int(seed) * 100_000 + step,
                )
                verifier_calls += len(backup_records)
                positives += sum(record.safe for record in backup_records)
                cache_hits_total += backup_cache_hits
                for index, record in enumerate(backup_records):
                    queried_traces.append(QueriedPlanTrace(
                        candidate_index=-1,
                        query_hash=record.query_hash,
                        source=record.source.value,
                        plan_kind=backup_kinds[index] if index < len(backup_kinds) else "backup",
                        acquisition_sigma=record.acquisition_sigma,
                        safe=record.safe,
                        in_bounds=record.safety.strict_bounds,
                        socp_ok=record.safety.socp_certified,
                        progress_m=record.progress_value,
                        clearance_m=record.safety.min_clearance,
                        cache_hit=False,
                        executed=bool(selected and selected.query_hash == record.query_hash),
                    ))

            before = state.copy()
            if selected is None:
                fail_closed = True
                after = before.copy()
                action = None
            else:
                if self.runtime_safety_claim and not selected.safe:
                    raise RuntimeError("controller attempted to execute an uncertified plan")
                if not self.runtime_safety_claim and not selected.safety.strict_bounds:
                    raise RuntimeError("offline bounds-only controller selected an out-of-bounds plan")
                action = np.asarray(selected.plan[0], dtype=np.float64).copy()
                state = execute_first_action(
                    state, selected.plan, config=self.config.dynamics,
                )
                if not np.array_equal(action, np.asarray(selected.plan[0], dtype=np.float64)):
                    raise RuntimeError("executed action differs from verified plan[0]")
                actions.append(action.astype(np.float32))
                states.append(state.copy())
                after = state.copy()
                in_bounds, collision, clearance = _runtime_status(state[:2], env)
                if self.runtime_safety_claim and (not in_bounds or collision):
                    raise RuntimeError(
                        "same-verifier runtime invariant failed after certified action: "
                        f"in_bounds={in_bounds}, clearance={clearance}"
                    )

            traces.append(ControlStepTrace(
                step=step,
                gamma=float(gamma),
                state_before=before,
                candidate_plans=plans,
                candidate_sigmas=sigmas,
                acquisition_probabilities=acquisition.probabilities,
                acquisition_order=acquisition.indices,
                queried=tuple(queried_traces),
                verifier_calls=len(new_flow) + (len(backup_records) if fallback_used else 0),
                cache_hits=cache_hits + (backup_cache_hits if fallback_used else 0),
                selected_query_hash=selected.query_hash if selected else None,
                selected_source=selected.source.value if selected else None,
                action=action,
                state_after=after,
                fallback_used=fallback_used,
                fail_closed=selected is None,
                acquisition_entropy=acquisition.entropy,
                acquisition_ess=acquisition.effective_sample_size,
                eligibility_mode=self.eligibility_mode,
                runtime_safety_claim=self.runtime_safety_claim,
                selected_actual_full_safe=selected.safe if selected else None,
            ))
            if (
                selected is None
                or np.linalg.norm(state[:2] - goal) < reach
                or (not self.runtime_safety_claim and (collision or not in_bounds))
            ):
                break

        reached = bool(np.linalg.norm(state[:2] - goal) < reach)
        return EpisodeResult(
            gamma=float(gamma),
            seed=int(seed),
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32).reshape(-1, 2),
            reached=reached,
            collision=collision,
            in_bounds=in_bounds,
            fail_closed=fail_closed,
            fallback_steps=fallback_steps,
            verifier_calls=verifier_calls,
            query_positives=positives,
            cache_hits=cache_hits_total,
            traces=tuple(traces),
            eligibility_mode=self.eligibility_mode,
            runtime_safety_claim=self.runtime_safety_claim,
        )
