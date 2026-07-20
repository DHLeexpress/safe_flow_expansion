from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from afe_restart.schemas import (
    FEATURE_DIM,
    ProgressResult,
    QueryContext,
    QuerySource,
    SafetyResult,
    VerificationRecord,
    query_content_hash,
)
from afe_restart.store import VerificationStore
from afe_restart.uncertainty import CumulativeLinearUncertainty


def _context(offset: float = 0.0) -> QueryContext:
    return QueryContext(
        grid=np.arange(16, dtype=np.float32).reshape(4, 4) + offset,
        low5=np.linspace(0.0, 1.0, 5, dtype=np.float32) + offset,
        hist=np.arange(12, dtype=np.float32).reshape(6, 2) + offset,
        verifier_state=np.asarray([offset, 0.0, 0.0, 0.0], dtype=np.float64),
        verifier_spec_fingerprint="c" * 64,
    )


def _feature(index: int) -> np.ndarray:
    z = np.zeros(FEATURE_DIM, dtype=np.float64)
    z[index] = 1.0
    return z


def _record(
    store: VerificationStore,
    *,
    index: int,
    safe: bool,
    gamma: float = 0.5,
    plan_offset: float = 0.0,
    audit: bool = False,
    feature_z: np.ndarray | None = None,
    source: QuerySource = QuerySource.FLOW,
) -> VerificationRecord:
    z = _feature(index) if feature_z is None else np.asarray(feature_z)
    plan = (
        np.arange(20, dtype=np.float32).reshape(10, 2) / 20.0 + plan_offset
    )
    return VerificationRecord(
        context=_context(10.0 if audit else 0.0),
        gamma=gamma,
        plan=plan,
        source=source,
        feature_z=z,
        acquisition_sigma=store.uncertainty.sigma(z),
        safety=SafetyResult(
            strict_bounds=safe,
            socp_certified=safe,
            min_clearance=0.2 if safe else -0.1,
            certificate_slack=0.03 if safe else -0.02,
            feasible_face_margin=0.01 if safe else -0.04,
        ),
        progress=ProgressResult(
            initial_goal_distance=5.0,
            terminal_goal_distance=4.0 if safe else 5.2,
        ),
    )


def test_cold_sigma_is_one_and_fixed_probe_never_increases() -> None:
    uncertainty = CumulativeLinearUncertainty()
    rng = np.random.default_rng(7)
    probe = rng.normal(size=FEATURE_DIM)
    probe /= np.linalg.norm(probe)

    assert uncertainty.A.dtype == np.float64
    assert uncertainty.A.shape == (32, 32)
    assert uncertainty.sigma(probe) == pytest.approx(1.0, abs=1e-14)

    history = [uncertainty.sigma(probe)]
    features: list[np.ndarray] = []
    for _ in range(50):
        z = rng.normal(size=FEATURE_DIM)
        z /= np.linalg.norm(z)
        features.append(z)
        uncertainty.observe(z)
        history.append(uncertainty.sigma(probe))

    assert np.all(np.diff(history) <= 1e-12)
    assert uncertainty.count == len(features)

    restored = CumulativeLinearUncertainty.from_state_dict(
        uncertainty.state_dict()
    )
    rebuilt = CumulativeLinearUncertainty.from_features(features)
    np.testing.assert_allclose(restored.A, uncertainty.A, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rebuilt.A, uncertainty.A, rtol=1e-14, atol=1e-14)
    assert restored.sigma(probe) == pytest.approx(history[-1], rel=1e-14)


def test_store_has_exact_query_A_replay_and_audit_accounting() -> None:
    store = VerificationStore()
    positive = _record(store, index=0, safe=True, plan_offset=0.0)
    store.append(positive)
    negative = _record(store, index=1, safe=False, plan_offset=1.0)
    store.append(negative)

    expected = np.eye(FEATURE_DIM, dtype=np.float64)
    expected += np.outer(positive.feature_z, positive.feature_z)
    expected += np.outer(negative.feature_z, negative.feature_z)
    np.testing.assert_allclose(store.uncertainty.A, expected, rtol=0.0, atol=0.0)
    assert store.query_count == store.uncertainty.count == 2
    assert store.positive_count == 1
    assert store.negative_count == 1
    assert store.query_acceptance == pytest.approx(0.5)

    replay = store.uniform_positive_view()
    assert len(replay) == 1
    assert replay[0].source_query_hash == positive.query_hash
    assert replay[0].training_target_hash == positive.query_hash
    replay[0].validate_identity()

    matrix_before = store.uncertainty.A
    count_before = store.uncertainty.count
    audit = _record(
        store, index=2, safe=True, plan_offset=2.0, gamma=1.0, audit=True
    )
    store.append_audit(audit)
    assert store.audit_count == 1
    assert store.query_count == 2
    assert store.uncertainty.count == count_before
    np.testing.assert_array_equal(store.uncertainty.A, matrix_before)
    assert len(store.uniform_positive_view()) == 1

    with pytest.raises(ValueError, match="duplicate query hash"):
        store.append(positive)
    assert store.query_count == store.uncertainty.count == 2

    restored = VerificationStore.from_state_dict(store.state_dict())
    assert restored.query_count == 2
    assert restored.audit_count == 1
    np.testing.assert_allclose(
        restored.uncertainty.A, store.uncertainty.A, rtol=0.0, atol=0.0
    )
    restored.assert_exact_accounting()


def test_batch_uses_one_shared_prequery_A_and_is_atomic() -> None:
    store = VerificationStore()
    z0 = _feature(0)
    z1 = _feature(0) + _feature(1)
    z1 /= np.linalg.norm(z1)

    first = _record(
        store, index=0, safe=True, plan_offset=10.0, feature_z=z0
    )
    second = _record(
        store, index=1, safe=False, plan_offset=11.0, feature_z=z1
    )
    assert first.acquisition_sigma == pytest.approx(1.0)
    assert second.acquisition_sigma == pytest.approx(1.0)

    sequential = CumulativeLinearUncertainty()
    sequential.observe(z0)
    assert sequential.sigma(z1) < second.acquisition_sigma

    store.append_batch((first, second))
    assert store.batch_sizes == (2,)
    assert store.query_count == store.uncertainty.count == 2
    expected = np.eye(FEATURE_DIM, dtype=np.float64)
    expected += np.outer(z0, z0) + np.outer(z1, z1)
    np.testing.assert_allclose(store.uncertainty.A, expected, rtol=0.0, atol=1e-15)

    matrix_before = store.uncertainty.A
    records_before = store.records
    batches_before = store.batch_sizes
    duplicate = _record(
        store, index=2, safe=True, plan_offset=12.0, feature_z=_feature(2)
    )
    with pytest.raises(ValueError, match="within acquisition batch"):
        store.append_batch((duplicate, duplicate))
    assert store.records == records_before
    assert store.batch_sizes == batches_before
    assert store.uncertainty.count == 2
    np.testing.assert_array_equal(store.uncertainty.A, matrix_before)

    restored = VerificationStore.from_state_dict(store.state_dict())
    assert restored.batch_sizes == (2,)
    assert restored.query_count == restored.uncertainty.count == 2
    np.testing.assert_allclose(
        restored.uncertainty.A, store.uncertainty.A, rtol=0.0, atol=0.0
    )


def test_exact_query_identity_and_replay_hashes_are_tamper_evident() -> None:
    context = _context()
    plan = np.arange(20, dtype=np.float32).reshape(10, 2)
    baseline = query_content_hash(context, 0.5, plan)

    assert baseline == query_content_hash(context, 0.5, plan[:, ::-1][:, ::-1])
    assert baseline != query_content_hash(context, 1.0, plan)
    assert baseline != query_content_hash(_context(0.1), 0.5, plan)
    changed_state = replace(
        context,
        verifier_state=np.nextafter(
            context.verifier_state, np.ones(4, dtype=np.float64)
        ),
    )
    assert baseline != query_content_hash(changed_state, 0.5, plan)
    changed_spec = replace(context, verifier_spec_fingerprint="f" * 64)
    assert baseline != query_content_hash(changed_spec, 0.5, plan)
    changed_plan = plan.copy()
    changed_plan[0, 0] = np.nextafter(changed_plan[0, 0], np.float32(1.0))
    assert baseline != query_content_hash(context, 0.5, changed_plan)

    store = VerificationStore()
    record = _record(store, index=4, safe=True)
    assert record.generated_hash == record.verifier_input_hash == record.query_hash
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(record, generated_hash="0" * 64)

    store.append(record)
    replay = store.uniform_positive_view()[0]
    assert replay.training_target_hash == record.query_hash
    with pytest.raises(ValueError, match="read-only"):
        record.plan[0, 0] = 999.0
    with pytest.raises(ValueError, match="read-only"):
        replay.plan[0, 0] = 999.0


def test_safe_backup_updates_cumulative_A_but_is_excluded_from_flow_replay() -> None:
    store = VerificationStore()
    backup = _record(
        store,
        index=7,
        safe=True,
        source=QuerySource.SAFEMPPI_BACKUP,
    )
    before = store.uncertainty.A
    store.append(backup)

    assert store.query_count == store.uncertainty.count == 1
    assert not np.array_equal(before, store.uncertainty.A)
    assert len(store.uniform_positive_view()) == 1
    assert len(store.uniform_positive_view(source=QuerySource.FLOW)) == 0
