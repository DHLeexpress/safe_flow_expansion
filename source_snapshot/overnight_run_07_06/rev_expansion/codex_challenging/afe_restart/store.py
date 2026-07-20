"""Append-only query ledger and isolated audit ledger."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .schemas import QuerySource, ReplayItem, VerificationRecord
from .uncertainty import CumulativeLinearUncertainty


class UniformPositiveView(Sequence[ReplayItem]):
    """Immutable positive-replay snapshot with only uniform sampling."""

    def __init__(self, items: Sequence[ReplayItem]) -> None:
        checked: list[ReplayItem] = []
        for item in items:
            if not isinstance(item, ReplayItem):
                raise TypeError("uniform replay accepts ReplayItem objects only")
            item.validate_identity()
            checked.append(item)
        self._items = tuple(checked)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int | slice) -> ReplayItem | tuple[ReplayItem, ...]:
        return self._items[index]

    def __iter__(self) -> Iterator[ReplayItem]:
        return iter(self._items)

    def sample(
        self,
        rng: np.random.Generator,
        size: int,
        *,
        replace: bool = False,
    ) -> tuple[ReplayItem, ...]:
        """Draw indices uniformly; no sigma, margin, or progress weights exist."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if size < 0:
            raise ValueError("size must be nonnegative")
        if not replace and size > len(self):
            raise ValueError("cannot sample more replay rows than exist")
        if size and not self:
            raise ValueError("cannot sample an empty replay view")
        indices = rng.choice(len(self), size=size, replace=replace)
        return tuple(self._items[int(index)] for index in np.atleast_1d(indices))


class VerificationStore:
    """Own the cumulative query ledger, positive view, and isolated audits.

    ``append`` is the only operation that mutates the uncertainty matrix.  It
    rejects duplicate content hashes so a deterministic duplicate can be
    served from the existing row instead of masquerading as a new observation.
    ``append_audit`` is deliberately separate and has no acquisition/training
    side effects.
    """

    # Version 2 requires query-schema-v2 contexts carrying the exact float64
    # verifier state and immutable scene/goal/dynamics/verifier fingerprint.
    STATE_VERSION = 2

    def __init__(
        self,
        uncertainty: CumulativeLinearUncertainty | None = None,
        *,
        sigma_rtol: float = 1e-7,
        sigma_atol: float = 1e-10,
    ) -> None:
        self.uncertainty = uncertainty or CumulativeLinearUncertainty()
        if self.uncertainty.count != 0:
            raise ValueError(
                "a new store requires empty uncertainty; use from_state_dict "
                "or rebuild_uncertainty for existing records"
            )
        if sigma_rtol < 0.0 or sigma_atol < 0.0:
            raise ValueError("sigma tolerances must be nonnegative")
        self.sigma_rtol = float(sigma_rtol)
        self.sigma_atol = float(sigma_atol)
        self._records: list[VerificationRecord] = []
        self._by_hash: dict[str, VerificationRecord] = {}
        self._audit_records: list[VerificationRecord] = []
        self._batch_sizes: list[int] = []

    @property
    def records(self) -> tuple[VerificationRecord, ...]:
        return tuple(self._records)

    @property
    def audit_records(self) -> tuple[VerificationRecord, ...]:
        return tuple(self._audit_records)

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        """Verifier-query batch boundaries in append order."""

        return tuple(self._batch_sizes)

    @property
    def query_count(self) -> int:
        return len(self._records)

    @property
    def audit_count(self) -> int:
        return len(self._audit_records)

    @property
    def positive_count(self) -> int:
        return sum(record.safe for record in self._records)

    @property
    def negative_count(self) -> int:
        return self.query_count - self.positive_count

    @property
    def query_acceptance(self) -> float:
        if not self._records:
            return float("nan")
        return self.positive_count / self.query_count

    def __len__(self) -> int:
        return self.query_count

    def __contains__(self, query_hash: object) -> bool:
        return isinstance(query_hash, str) and query_hash in self._by_hash

    def get(self, query_hash: str) -> VerificationRecord | None:
        return self._by_hash.get(query_hash)

    def append(self, record: VerificationRecord) -> None:
        """Append one real verifier result as a one-element acquisition batch."""

        self.append_batch((record,))

    def append_batch(self, records: Iterable[VerificationRecord]) -> None:
        """Atomically append candidates acquired from one shared ``A_n``.

        Finite Gibbs probabilities and their sigmas are computed before any
        candidate in a verifier-budget batch is observed.  Accordingly every
        row here is checked against the same starting matrix, then all rows are
        committed and observed exactly once.  An invalid row or duplicate
        leaves the ledger and uncertainty state unchanged.
        """

        batch = tuple(records)
        if not batch:
            raise ValueError("an acquisition batch cannot be empty")

        # Complete all fallible record/identity/duplicate/sigma validation
        # before constructing or committing any new state.
        batch_hashes: set[str] = set()
        for record in batch:
            if not isinstance(record, VerificationRecord):
                raise TypeError("every batch row must be a VerificationRecord")
            record.validate_identity()
            if record.query_hash in self._by_hash:
                raise ValueError(
                    "duplicate query hash already exists in the ledger: reuse "
                    "the cached deterministic verifier record"
                )
            if record.query_hash in batch_hashes:
                raise ValueError("duplicate query hash within acquisition batch")
            batch_hashes.add(record.query_hash)

        # No observe occurs during this loop: all sigmas refer to shared A_n.
        for record in batch:
            expected_sigma = self.uncertainty.sigma(record.feature_z)
            if not np.isclose(
                record.acquisition_sigma,
                expected_sigma,
                rtol=self.sigma_rtol,
                atol=self.sigma_atol,
            ):
                raise ValueError(
                    "record acquisition_sigma is not its shared pre-batch "
                    f"uncertainty ({record.acquisition_sigma} != {expected_sigma})"
                )

        # Build the complete post-batch uncertainty off to the side.  This
        # gives atomic behavior even if a future observe implementation adds a
        # fallible check after the validation above.
        updated = CumulativeLinearUncertainty.from_state_dict(
            self.uncertainty.state_dict()
        )
        for record in batch:
            updated.observe(record.feature_z)

        # Commit without replacing the caller-visible uncertainty object.
        self.uncertainty.load_state_dict(updated.state_dict())
        self._records.extend(batch)
        self._by_hash.update((record.query_hash, record) for record in batch)
        self._batch_sizes.append(len(batch))
        self.assert_exact_accounting()

    def append_audit(self, record: VerificationRecord) -> None:
        """Append an evaluation-only result without touching A or replay."""

        if not isinstance(record, VerificationRecord):
            raise TypeError("record must be a VerificationRecord")
        record.validate_identity()
        if record.executed:
            raise ValueError("an independent audit record cannot be executed")
        before_count = self.uncertainty.count
        before_matrix = self.uncertainty.A
        self._audit_records.append(record)
        if self.uncertainty.count != before_count or not np.array_equal(
            self.uncertainty.A, before_matrix
        ):
            raise RuntimeError("audit isolation was violated")

    def uniform_positive_view(
        self,
        *,
        source: QuerySource | str | None = None,
    ) -> UniformPositiveView:
        """Return uniform verified-positive replay, optionally source-scoped.

        Stage 05 passes QuerySource.FLOW: certified backup plans remain real
        verifier observations in cumulative A, but runtime fallback must never
        silently become expert distillation.
        """

        source_value = QuerySource(source) if source is not None else None
        return UniformPositiveView(
            [
                ReplayItem.from_record(record)
                for record in self._records
                if record.safe
                and (source_value is None or record.source is source_value)
            ]
        )

    def assert_exact_accounting(self) -> None:
        if self.query_count != self.uncertainty.count:
            raise RuntimeError(
                "verifier-ledger count and cumulative-A observation count differ"
            )
        if self.query_count != len(self._by_hash):
            raise RuntimeError("query ledger contains a duplicate content hash")
        if any(size <= 0 for size in self._batch_sizes):
            raise RuntimeError("query ledger contains an invalid batch size")
        if sum(self._batch_sizes) != self.query_count:
            raise RuntimeError("query ledger batch boundaries do not cover all rows")
        expected_trace = (
            self.uncertainty.feature_dim * self.uncertainty.prior_precision
            + self.query_count / self.uncertainty.lambda_
        )
        if not np.isclose(
            float(np.trace(self.uncertainty.A)),
            expected_trace,
            rtol=1e-11,
            atol=1e-10,
        ):
            raise RuntimeError("cumulative-A trace does not match verifier accounting")

    def rebuild_uncertainty(self) -> None:
        rebuilt = CumulativeLinearUncertainty.from_features(
            (record.feature_z for record in self._records),
            feature_dim=self.uncertainty.feature_dim,
            lambda_=self.uncertainty.lambda_,
            prior_precision=self.uncertainty.prior_precision,
        )
        self.uncertainty = rebuilt
        self.assert_exact_accounting()

    def state_dict(self) -> dict[str, Any]:
        self.assert_exact_accounting()
        return {
            "version": self.STATE_VERSION,
            "sigma_rtol": self.sigma_rtol,
            "sigma_atol": self.sigma_atol,
            "uncertainty": self.uncertainty.state_dict(),
            "batch_sizes": list(self._batch_sizes),
            "records": [record.to_state_dict() for record in self._records],
            "audit_records": [
                record.to_state_dict() for record in self._audit_records
            ],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "VerificationStore":
        if int(state.get("version", -1)) != cls.STATE_VERSION:
            raise ValueError(
                "unsupported verification-store state version; legacy stores "
                "without exact verifier query identity must not be resumed"
            )
        expected_uncertainty = CumulativeLinearUncertainty.from_state_dict(
            state["uncertainty"]
        )
        empty_uncertainty = CumulativeLinearUncertainty(
            feature_dim=expected_uncertainty.feature_dim,
            lambda_=expected_uncertainty.lambda_,
            prior_precision=expected_uncertainty.prior_precision,
        )
        result = cls(
            empty_uncertainty,
            sigma_rtol=float(state.get("sigma_rtol", 1e-7)),
            sigma_atol=float(state.get("sigma_atol", 1e-10)),
        )
        records = [
            VerificationRecord.from_state_dict(record_state)
            for record_state in state["records"]
        ]
        raw_batch_sizes = state.get("batch_sizes", [1] * len(records))
        batch_sizes: list[int] = []
        for raw_size in raw_batch_sizes:
            size = int(raw_size)
            if size != raw_size or size <= 0:
                raise ValueError("stored batch_sizes must contain positive integers")
            batch_sizes.append(size)
        if sum(batch_sizes) != len(records):
            raise ValueError("stored batch_sizes do not cover every query record")

        # Replay in original batch order independently reconstructs A and
        # rechecks each record's shared *pre-batch* sigma and content identity.
        offset = 0
        for batch_size in batch_sizes:
            result.append_batch(records[offset : offset + batch_size])
            offset += batch_size
        for audit_state in state.get("audit_records", []):
            result.append_audit(VerificationRecord.from_state_dict(audit_state))
        if expected_uncertainty.count != result.uncertainty.count or not np.allclose(
            expected_uncertainty.A,
            result.uncertainty.A,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("stored A does not match the verifier-record rebuild")
        result.assert_exact_accounting()
        return result
