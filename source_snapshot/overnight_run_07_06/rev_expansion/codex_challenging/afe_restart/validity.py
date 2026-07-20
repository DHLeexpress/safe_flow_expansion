"""Honest validation and aggregation of sealed final audits.

The replication unit in this module is an independently pretrained and
expanded model.  Plan rows are never pooled across models to manufacture a
larger sample size.  Conditional Wilson intervals remain available in each
single-model :mod:`afe_restart.audit` result; the intervals produced here are
Student-t intervals across the per-model estimates.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"expected an audit mapping/dataclass, got {type(value).__name__}")


VALID_AUDIT_MODES = (
    "left-of-goal-ray",
    "goal-ray",
    "right-of-goal-ray",
)


def _student_t_interval(
    values: Sequence[float],
    confidence: float,
    *,
    bounds: tuple[float, float] | None = None,
) -> dict[str, float | str | int]:
    if len(values) < 2:
        raise ValueError("an independent-training-seed interval requires at least two runs")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("per-seed estimates must be finite")
    if bounds is not None:
        low_bound, high_bound = bounds
        if low_bound > high_bound or np.any(
            (array < low_bound) | (array > high_bound)
        ):
            raise ValueError("per-seed estimates lie outside their declared bounds")
    from scipy.stats import t as student_t

    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, len(array) - 1))
    half_width = critical * standard_deviation / math.sqrt(len(array))
    low = mean - half_width
    high = mean + half_width
    if bounds is not None:
        low = max(bounds[0], low)
        high = min(bounds[1], high)
    return {
        "mean": mean,
        "standard_deviation_across_training_seeds": standard_deviation,
        "low": low,
        "high": high,
        "confidence": float(confidence),
        "independent_training_seed_count": len(array),
        "method": "student_t_across_independent_training_seed_estimates",
    }


def _training_seed_interval(
    values: Sequence[float], confidence: float,
) -> dict[str, float | str | int]:
    return _student_t_interval(values, confidence, bounds=(0.0, 1.0))


def _gamma_rows(audit: Mapping[str, Any], *, seed: int) -> dict[float, dict[str, Any]]:
    materialized = [dict(row) for row in audit.get("per_gamma", ())]
    gammas = [float(row["gamma"]) for row in materialized]
    if not materialized:
        raise ValueError(f"training seed {seed} has no per-gamma audit rows")
    if len(gammas) != len(set(gammas)):
        raise ValueError(f"training seed {seed} has duplicate gamma audit rows")
    if not np.isfinite(np.asarray(gammas, dtype=np.float64)).all():
        raise ValueError(f"training seed {seed} has non-finite gamma values")
    return {gamma: row for gamma, row in zip(gammas, materialized)}


def _mode_coverage_fraction(row: Mapping[str, Any], *, strict: bool) -> float:
    counts = {str(key): int(value) for key, value in row.get("mode_counts", {}).items()}
    if not counts and "safe_mode_coverage" not in row:
        if strict:
            raise ValueError("sealed audit row is missing valid-mode telemetry")
        return math.nan
    if any(value < 0 for value in counts.values()):
        raise ValueError("audit mode counts cannot be negative")
    unknown = set(counts) - set(VALID_AUDIT_MODES)
    if unknown:
        raise ValueError(f"audit contains unregistered local modes: {sorted(unknown)}")
    observed = sum(value > 0 for value in counts.values())
    declared = int(row.get("safe_mode_coverage", observed))
    if declared != observed:
        raise ValueError("safe_mode_coverage disagrees with positive mode counts")
    return observed / len(VALID_AUDIT_MODES)


def validate_sealed_audit_protocol(
    audit_value: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one audit against the complete preregistered sealed protocol.

    The returned mapping is a defensive copy suitable for artifact emission.
    This validator intentionally requires all seven gamma levels and all
    valid-mode fields; it is stricter than the backwards-compatible aggregate
    helper below.
    """

    audit = _mapping(audit_value)
    required_protocol = {
        "context_bank_fingerprint",
        "context_bank_role",
        "context_count",
        "plans_per_context",
        "progress_threshold",
        "nfe",
        "temperature",
        "uncertainty_tilting",
        "sampling_distribution",
        "gammas",
        "audit_seed",
    }
    missing = required_protocol - set(protocol)
    if missing:
        raise ValueError(f"sealed protocol is missing fields: {sorted(missing)}")
    if protocol["context_bank_role"] != "sealed_final_test":
        raise ValueError("sealed protocol must use role='sealed_final_test'")
    if float(protocol["temperature"]) != 1.0 or bool(protocol["uncertainty_tilting"]):
        raise ValueError("sealed protocol must be ordinary, untilted temperature one")
    expected_gammas = tuple(float(value) for value in protocol["gammas"])
    if len(expected_gammas) != 7 or len(set(expected_gammas)) != 7:
        raise ValueError("sealed protocol requires seven unique gamma levels")

    exact_pairs = (
        ("context_bank_fingerprint", str),
        ("context_bank_role", str),
        ("context_count", int),
        ("plans_per_context", int),
        ("progress_threshold", float),
        ("temperature", float),
        ("uncertainty_tilting", bool),
        ("sampling_distribution", str),
        ("seed", int),
    )
    protocol_keys = {
        "seed": "audit_seed",
    }
    for audit_key, caster in exact_pairs:
        protocol_key = protocol_keys.get(audit_key, audit_key)
        if caster(audit.get(audit_key)) != caster(protocol[protocol_key]):
            raise ValueError(
                f"sealed audit {audit_key!r} does not match the exact protocol"
            )
    if int(audit.get("total_verifier_calls", -1)) != (
        int(protocol["context_count"])
        * int(protocol["plans_per_context"])
        * len(expected_gammas)
    ):
        raise ValueError("sealed audit verifier-call count disagrees with its protocol")
    if audit.get("confidence_interval_scope") != (
        "conditional_plan_sampling_wilson_on_fixed_context_bank_single_model"
    ):
        raise ValueError("single-model sealed audit has an incorrect interval scope")
    if int(audit.get("independent_training_seed_count", -1)) != 1 or bool(
        audit.get("independent_training_seed_ci", True)
    ):
        raise ValueError("single-model audit cannot claim an independent-seed interval")

    rows = _gamma_rows(audit, seed=int(audit["seed"]))
    if tuple(rows) != expected_gammas:
        raise ValueError("sealed audit gamma order/content does not match all seven levels")
    expected_n = int(protocol["context_count"]) * int(protocol["plans_per_context"])
    for gamma, row in rows.items():
        sample_count = int(row.get("sample_count", -1))
        safe_count = int(row.get("safe_count", -1))
        progress_count = int(row.get("safe_progress_count", -1))
        if sample_count != expected_n or not 0 <= progress_count <= safe_count <= sample_count:
            raise ValueError(f"gamma {gamma:g} has inconsistent sealed audit counts")
        if not math.isclose(
            float(row.get("validity_mass", math.nan)),
            safe_count / sample_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"gamma {gamma:g} validity mass disagrees with counts")
        if not math.isclose(
            float(row.get("progress_validity", math.nan)),
            progress_count / sample_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"gamma {gamma:g} progress validity disagrees with counts")
        _mode_coverage_fraction(row, strict=True)
        if sum(int(value) for value in row["mode_counts"].values()) != safe_count:
            raise ValueError(f"gamma {gamma:g} valid-mode counts do not sum to safe_count")
    return audit


def aggregate_independent_training_seed_audits(
    audits_by_training_seed: Mapping[int, Any],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Aggregate per-seed sealed-bank estimates without pooling plan rows.

    Each value must be one ordinary-temperature-one audit of an independently
    trained model on the exact same sealed context bank.  The returned interval
    treats the independently trained model as the replication unit.  Conditional
    Wilson intervals inside each audit are intentionally ignored here.
    """

    if len(audits_by_training_seed) < 2:
        raise ValueError("need at least two independent training seeds")
    seed_items = sorted((int(seed), _mapping(audit)) for seed, audit in audits_by_training_seed.items())
    if len({seed for seed, _audit in seed_items}) != len(seed_items):
        raise ValueError("training seed identifiers must be unique")

    fingerprints = {str(audit.get("context_bank_fingerprint", "")) for _seed, audit in seed_items}
    if len(fingerprints) != 1 or "" in fingerprints:
        raise ValueError("all audits must use the exact same fingerprinted context bank")
    for seed, audit in seed_items:
        if audit.get("context_bank_role") != "sealed_final_test":
            raise ValueError(f"training seed {seed} did not use the sealed final-test bank")
        if float(audit.get("temperature", math.nan)) != 1.0:
            raise ValueError(f"training seed {seed} audit is not temperature one")
        if bool(audit.get("uncertainty_tilting", True)):
            raise ValueError(f"training seed {seed} audit used uncertainty tilting")

    rows_by_seed: dict[int, dict[float, dict[str, Any]]] = {}
    for seed, audit in seed_items:
        rows_by_seed[seed] = _gamma_rows(audit, seed=seed)
    gamma_sets = {tuple(sorted(rows)) for rows in rows_by_seed.values()}
    if len(gamma_sets) != 1:
        raise ValueError("sealed audits do not contain the same gamma levels")
    gammas = next(iter(gamma_sets))

    per_gamma = []
    per_seed_aggregate_validity: dict[int, list[float]] = {seed: [] for seed, _ in seed_items}
    per_seed_aggregate_progress: dict[int, list[float]] = {seed: [] for seed, _ in seed_items}
    per_seed_aggregate_coverage: dict[int, list[float]] = {seed: [] for seed, _ in seed_items}
    for gamma in gammas:
        validity = [float(rows_by_seed[seed][gamma]["validity_mass"]) for seed, _ in seed_items]
        progress = [float(rows_by_seed[seed][gamma]["progress_validity"]) for seed, _ in seed_items]
        coverage = [
            _mode_coverage_fraction(rows_by_seed[seed][gamma], strict=False)
            for seed, _ in seed_items
        ]
        for (seed, _audit), value, progress_value in zip(seed_items, validity, progress):
            per_seed_aggregate_validity[seed].append(value)
            per_seed_aggregate_progress[seed].append(progress_value)
        validity_interval = _training_seed_interval(validity, confidence)
        progress_interval = _training_seed_interval(progress, confidence)
        gamma_result = {
            "gamma": gamma,
            "validity": validity_interval,
            "progress_validity": progress_interval,
            "V": validity_interval,
            "Vprog": progress_interval,
            "per_training_seed_validity": {
                str(seed): value for (seed, _audit), value in zip(seed_items, validity)
            },
            "per_training_seed_progress_validity": {
                str(seed): value for (seed, _audit), value in zip(seed_items, progress)
            },
        }
        if not any(math.isnan(value) for value in coverage):
            for (seed, _audit), value in zip(seed_items, coverage):
                per_seed_aggregate_coverage[seed].append(value)
            gamma_result["valid_mode_coverage"] = _training_seed_interval(
                coverage, confidence
            )
            gamma_result["per_training_seed_valid_mode_coverage"] = {
                str(seed): value
                for (seed, _audit), value in zip(seed_items, coverage)
            }
            gamma_result["valid_mode_vocabulary"] = list(VALID_AUDIT_MODES)
        per_gamma.append(gamma_result)

    aggregate_validity = [
        float(np.mean(per_seed_aggregate_validity[seed])) for seed, _ in seed_items
    ]
    aggregate_progress = [
        float(np.mean(per_seed_aggregate_progress[seed])) for seed, _ in seed_items
    ]
    overall_validity_interval = _training_seed_interval(aggregate_validity, confidence)
    overall_progress_interval = _training_seed_interval(aggregate_progress, confidence)
    aggregate_over_gammas: dict[str, Any] = {
        "validity": overall_validity_interval,
        "progress_validity": overall_progress_interval,
        "V": overall_validity_interval,
        "Vprog": overall_progress_interval,
    }
    if all(per_seed_aggregate_coverage[seed] for seed, _audit in seed_items):
        aggregate_coverage = [
            float(np.mean(per_seed_aggregate_coverage[seed]))
            for seed, _audit in seed_items
        ]
        aggregate_over_gammas["valid_mode_coverage"] = _training_seed_interval(
            aggregate_coverage, confidence
        )
        aggregate_over_gammas["per_training_seed_valid_mode_coverage"] = {
            str(seed): value
            for (seed, _audit), value in zip(seed_items, aggregate_coverage)
        }
        aggregate_over_gammas["valid_mode_coverage_definition"] = (
            "mean across gamma of observed safe local modes / 3 preregistered modes"
        )
    return {
        "schema_version": "afe_independent_training_seed_validity_v1",
        "context_bank_fingerprint": next(iter(fingerprints)),
        "context_bank_role": "sealed_final_test",
        "training_seeds": [seed for seed, _audit in seed_items],
        "independent_training_seed_count": len(seed_items),
        "replication_unit": "independently_trained_model",
        "plan_samples_pooled_across_training_seeds": False,
        "confidence_interval_scope": "across_independent_training_seed_estimates",
        "aggregate_over_gammas": aggregate_over_gammas,
        "per_gamma": per_gamma,
    }


def aggregate_sealed_full_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Aggregate strict per-run sealed records for independent Full replicas."""

    materialized = [dict(run) for run in runs]
    if len(materialized) < 2:
        raise ValueError("need at least two independently pretrained+expanded Full runs")
    if any(str(run.get("method")) != "full" for run in materialized):
        raise ValueError("independent sealed aggregate accepts Full runs only")
    seeds = [int(run["expansion_training_seed"]) for run in materialized]
    model_hashes = [str(run["model_state_sha256"]) for run in materialized]
    pretrain_hashes = [str(run["source_pretrain_hash"]) for run in materialized]
    if len(set(seeds)) != len(seeds):
        raise ValueError("independent Full runs have duplicate expansion training seeds")
    if len(set(model_hashes)) != len(model_hashes):
        raise ValueError("independent Full runs have duplicate final model hashes")
    if len(set(pretrain_hashes)) != len(pretrain_hashes):
        raise ValueError("Full replicas are not independently pretrained")
    fingerprints = {str(run.get("protocol_fingerprint", "")) for run in materialized}
    if len(fingerprints) != 1 or "" in fingerprints:
        raise ValueError("independent Full audits have a protocol mismatch")

    audits = {
        seed: _mapping(run["audit"])
        for seed, run in zip(seeds, materialized)
    }
    aggregate = aggregate_independent_training_seed_audits(
        audits, confidence=confidence
    )
    aggregate.update({
        "schema_version": "afe_sealed_independent_full_aggregate_v1",
        "protocol_fingerprint": next(iter(fingerprints)),
        "run_ids": [str(run["run_id"]) for run in materialized],
        "model_state_sha256s": model_hashes,
        "source_pretrain_hashes": pretrain_hashes,
    })

    runtime_rows = [dict(run.get("runtime", {})) for run in materialized]
    available = [row for row in runtime_rows if bool(row.get("available", False))]
    runtime: dict[str, Any] = {
        "source": "expansion_checkpoint_history_not_sealed_audit_execution",
        "available_run_count": len(available),
        "missing_run_count": len(runtime_rows) - len(available),
    }
    if len(available) == len(runtime_rows):
        fallback = [float(row["fallback_frequency"]) for row in available]
        failclosed = [float(row["failclosed_frequency"]) for row in available]
        fallback_interval = _training_seed_interval(fallback, confidence)
        failclosed_interval = _training_seed_interval(failclosed, confidence)
        runtime.update({
            "fallback_frequency": fallback_interval,
            "failclosed_frequency": failclosed_interval,
            "per_training_seed_fallback_frequency": {
                str(seed): value for seed, value in zip(seeds, fallback)
            },
            "per_training_seed_failclosed_frequency": {
                str(seed): value for seed, value in zip(seeds, failclosed)
            },
        })
        aggregate["aggregate_over_gammas"].update({
            "fallback_frequency": fallback_interval,
            "failclosed_frequency": failclosed_interval,
        })
    else:
        runtime["intervals_available"] = False
        runtime["reason"] = (
            "an across-seed runtime interval requires checkpoint history for every Full run"
        )
    aggregate["runtime"] = runtime
    return aggregate


__all__ = [
    "VALID_AUDIT_MODES",
    "aggregate_independent_training_seed_audits",
    "aggregate_sealed_full_runs",
    "validate_sealed_audit_protocol",
]
