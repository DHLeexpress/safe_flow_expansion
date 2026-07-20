"""Cumulative fixed-feature linear uncertainty for AFE acquisition."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schemas import FEATURE_DIM


class CumulativeLinearUncertainty:
    """Maintain ``A = aI + lambda^-1 sum(z z^T)`` in float64.

    The matrix is cumulative: there is intentionally no capacity, eviction,
    decimation, or raw-feature buffer.  Variances use a Cholesky solve rather
    than maintaining a numerically fragile explicit inverse.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        *,
        lambda_: float = 1.0,
        prior_precision: float = 1.0,
    ) -> None:
        if int(feature_dim) != feature_dim or int(feature_dim) != FEATURE_DIM:
            raise ValueError(
                f"planned-window AFE requires feature_dim={FEATURE_DIM}, "
                f"got {feature_dim}"
            )
        if not np.isfinite(lambda_) or lambda_ <= 0.0:
            raise ValueError("lambda_ must be positive and finite")
        if float(prior_precision) != 1.0:
            raise ValueError("planned-window AFE fixes A_0=I (prior_precision=1)")
        self.feature_dim = int(feature_dim)
        self.lambda_ = float(lambda_)
        self.prior_precision = float(prior_precision)
        self._A = np.eye(self.feature_dim, dtype=np.float64) * self.prior_precision
        self._count = 0
        self._cholesky: NDArray[np.float64] | None = None

    @property
    def count(self) -> int:
        """Number of real verifier observations incorporated exactly once."""

        return self._count

    @property
    def A(self) -> NDArray[np.float64]:
        """A read-only snapshot; callers cannot mutate uncertainty state."""

        snapshot = self._A.copy()
        snapshot.setflags(write=False)
        return snapshot

    def _feature(self, value: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(value, dtype=np.float64)
        if z.shape != (self.feature_dim,):
            raise ValueError(
                f"feature must have shape {(self.feature_dim,)}, got {z.shape}"
            )
        if not np.all(np.isfinite(z)):
            raise ValueError("feature contains a non-finite value")
        norm = float(np.linalg.norm(z))
        if not np.isclose(norm, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(f"feature must be unit normalized, got norm {norm}")
        return z

    def _factor(self) -> NDArray[np.float64]:
        if self._cholesky is None:
            self._cholesky = np.linalg.cholesky(self._A)
        return self._cholesky

    def sigma_squared(self, feature_z: ArrayLike) -> float:
        z = self._feature(feature_z)
        solved = np.linalg.solve(self._factor(), z)
        variance = float(np.dot(solved, solved))
        # Roundoff cannot make a positive quadratic form meaningfully negative.
        return max(0.0, variance)

    def sigma(self, feature_z: ArrayLike) -> float:
        return float(np.sqrt(self.sigma_squared(feature_z)))

    def sigmas(self, features: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [N,{self.feature_dim}], got {values.shape}"
            )
        return np.asarray([self.sigma(row) for row in values], dtype=np.float64)

    def observe(self, feature_z: ArrayLike) -> None:
        """Add one and only one real verifier query to the design matrix."""

        z = self._feature(feature_z)
        self._A += np.outer(z, z) / self.lambda_
        # Explicit symmetrization limits long-run floating-point asymmetry.
        self._A = (self._A + self._A.T) * 0.5
        self._count += 1
        self._cholesky = None

    def reset(self) -> None:
        self._A = np.eye(self.feature_dim, dtype=np.float64) * self.prior_precision
        self._count = 0
        self._cholesky = None

    def rebuild(self, features: Iterable[ArrayLike]) -> None:
        """Rebuild state from the complete append-only verifier ledger."""

        self.reset()
        for feature in features:
            self.observe(feature)

    @classmethod
    def from_features(
        cls,
        features: Iterable[ArrayLike],
        *,
        feature_dim: int = FEATURE_DIM,
        lambda_: float = 1.0,
        prior_precision: float = 1.0,
    ) -> "CumulativeLinearUncertainty":
        result = cls(
            feature_dim=feature_dim,
            lambda_=lambda_,
            prior_precision=prior_precision,
        )
        result.rebuild(features)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "feature_dim": self.feature_dim,
            "lambda_": self.lambda_,
            "prior_precision": self.prior_precision,
            "count": self.count,
            "A": self._A.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported uncertainty state version")
        feature_dim = int(state["feature_dim"])
        lambda_ = float(state["lambda_"])
        prior_precision = float(state["prior_precision"])
        count = int(state["count"])
        matrix = np.array(state["A"], dtype=np.float64, order="C", copy=True)
        if feature_dim != self.feature_dim:
            raise ValueError("state feature_dim does not match this instance")
        if lambda_ != self.lambda_ or prior_precision != self.prior_precision:
            raise ValueError("state uncertainty hyperparameters do not match")
        if count < 0:
            raise ValueError("state count must be nonnegative")
        if matrix.shape != (self.feature_dim, self.feature_dim):
            raise ValueError("state A has the wrong shape")
        if matrix.dtype != np.float64 or not np.all(np.isfinite(matrix)):
            raise ValueError("state A must be finite float64")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
            raise ValueError("state A must be symmetric")
        np.linalg.cholesky(matrix)
        # Unit z vectors make trace an inexpensive exact-accounting checksum.
        expected_trace = (
            self.feature_dim * self.prior_precision + count / self.lambda_
        )
        if not np.isclose(
            float(np.trace(matrix)), expected_trace, rtol=1e-11, atol=1e-10
        ):
            raise ValueError("state A trace is inconsistent with verifier count")
        self._A = matrix
        self._count = count
        self._cholesky = None

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> "CumulativeLinearUncertainty":
        result = cls(
            feature_dim=int(state["feature_dim"]),
            lambda_=float(state["lambda_"]),
            prior_precision=float(state["prior_precision"]),
        )
        result.load_state_dict(state)
        return result
