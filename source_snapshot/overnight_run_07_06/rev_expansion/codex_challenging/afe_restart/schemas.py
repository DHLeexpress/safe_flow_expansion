"""Immutable data objects for the planned-window AFE query ledger.

The hash in this module is deliberately about the *query*, not its verifier
result.  Consequently the same generated plan, verifier input, and replay
target must all have the same hash over the model context, exact verifier
state/specification, ``gamma``, and plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import struct
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


PLAN_HORIZON = 10
ACTION_DIM = 2
FEATURE_DIM = 32
HASH_VERSION = b"afe-restart-query-v2-exact-verifier-input\x00"


class QuerySource(str, Enum):
    """The model that proposed a full window submitted to the verifier."""

    FLOW = "flow"
    SAFEMPPI_BACKUP = "safemppi_backup"


def _immutable_array(
    value: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | type[Any] | None = None,
) -> NDArray[Any]:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise TypeError(f"{name} must be a numeric array, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    array.setflags(write=False)
    return array


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def _margin_float(value: float, *, name: str) -> float:
    """Margins may be infinite for an infeasible verifier, but never NaN."""

    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} must not be NaN")
    return result


def _hash_array(hasher: Any, label: bytes, array: NDArray[Any]) -> None:
    # dtype.str records width and byte order; shape prevents concatenation
    # ambiguities.  C-order canonicalization makes strides irrelevant while
    # retaining every stored value bit.
    contiguous = np.ascontiguousarray(array)
    hasher.update(struct.pack("!I", len(label)))
    hasher.update(label)
    dtype = contiguous.dtype.str.encode("ascii")
    hasher.update(struct.pack("!I", len(dtype)))
    hasher.update(dtype)
    hasher.update(struct.pack("!I", contiguous.ndim))
    for size in contiguous.shape:
        hasher.update(struct.pack("!Q", size))
    raw = contiguous.tobytes(order="C")
    hasher.update(struct.pack("!Q", len(raw)))
    hasher.update(raw)


def _validate_hash(value: str, *, name: str) -> str:
    if len(value) != 64:
        raise ValueError(f"{name} is not a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a SHA-256 hex digest") from exc
    return value.lower()


@dataclass(frozen=True, eq=False)
class QueryContext:
    """Model context plus the exact non-plan verifier inputs.

    ``grid/low5/hist`` are the tensors supplied to the conditional flow.
    They are intentionally float32 and therefore cannot serve as a lossless
    proxy for the float64 state supplied to the full verifier.  The latter and
    a fingerprint of scene, goal, dynamics and verifier configuration are
    carried separately so a cached label is valid only for the identical
    deterministic verifier query.
    """

    grid: NDArray[Any]
    low5: NDArray[Any]
    hist: NDArray[Any]
    verifier_state: NDArray[np.float64]
    verifier_spec_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", _immutable_array(self.grid, name="grid"))
        object.__setattr__(self, "low5", _immutable_array(self.low5, name="low5"))
        object.__setattr__(self, "hist", _immutable_array(self.hist, name="hist"))
        object.__setattr__(
            self,
            "verifier_state",
            _immutable_array(
                self.verifier_state,
                name="verifier_state",
                shape=(4,),
                dtype=np.float64,
            ),
        )
        object.__setattr__(
            self,
            "verifier_spec_fingerprint",
            _validate_hash(
                self.verifier_spec_fingerprint,
                name="verifier_spec_fingerprint",
            ),
        )

    def to_state_dict(self) -> dict[str, NDArray[Any]]:
        return {
            "grid": np.array(self.grid, copy=True),
            "low5": np.array(self.low5, copy=True),
            "hist": np.array(self.hist, copy=True),
            "verifier_state": np.array(self.verifier_state, copy=True),
            "verifier_spec_fingerprint": self.verifier_spec_fingerprint,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "QueryContext":
        missing = {
            "verifier_state",
            "verifier_spec_fingerprint",
        } - set(state)
        if missing:
            raise ValueError(
                "legacy query context lacks exact verifier identity fields: "
                f"{sorted(missing)}; regenerate the artifact under query schema v2"
            )
        return cls(
            grid=state["grid"],
            low5=state["low5"],
            hist=state["hist"],
            verifier_state=state["verifier_state"],
            verifier_spec_fingerprint=state["verifier_spec_fingerprint"],
        )


def query_content_hash(
    context: QueryContext, gamma: float, plan: ArrayLike
) -> str:
    """Return a stable, bit-exact SHA-256 identity for one planned query."""

    if not isinstance(context, QueryContext):
        raise TypeError("context must be a QueryContext")
    gamma_value = _finite_float(gamma, name="gamma")
    plan_array = _immutable_array(
        plan, name="plan", shape=(PLAN_HORIZON, ACTION_DIM)
    )
    hasher = hashlib.sha256()
    hasher.update(HASH_VERSION)
    _hash_array(hasher, b"grid", context.grid)
    _hash_array(hasher, b"low5", context.low5)
    _hash_array(hasher, b"hist", context.hist)
    _hash_array(hasher, b"verifier_state", context.verifier_state)
    hasher.update(b"verifier_spec_fingerprint\x00")
    hasher.update(bytes.fromhex(context.verifier_spec_fingerprint))
    # Gamma is a semantic float scalar, canonicalized as IEEE-754 binary64.
    hasher.update(b"gamma\x00")
    hasher.update(struct.pack("!d", gamma_value))
    _hash_array(hasher, b"plan", plan_array)
    return hasher.hexdigest()


@dataclass(frozen=True)
class SafetyResult:
    """Full-window safety outputs, with no progress criterion mixed in."""

    strict_bounds: bool
    socp_certified: bool
    min_clearance: float
    certificate_slack: float
    feasible_face_margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "strict_bounds", bool(self.strict_bounds))
        object.__setattr__(self, "socp_certified", bool(self.socp_certified))
        object.__setattr__(
            self,
            "min_clearance",
            _margin_float(self.min_clearance, name="min_clearance"),
        )
        object.__setattr__(
            self,
            "certificate_slack",
            _margin_float(self.certificate_slack, name="certificate_slack"),
        )
        object.__setattr__(
            self,
            "feasible_face_margin",
            _margin_float(self.feasible_face_margin, name="feasible_face_margin"),
        )

    @property
    def safe(self) -> bool:
        return self.strict_bounds and self.socp_certified

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "strict_bounds": self.strict_bounds,
            "socp_certified": self.socp_certified,
            "min_clearance": self.min_clearance,
            "certificate_slack": self.certificate_slack,
            "feasible_face_margin": self.feasible_face_margin,
        }


@dataclass(frozen=True)
class ProgressResult:
    """Goal-distance progress for ranking plans already known to be safe."""

    initial_goal_distance: float
    terminal_goal_distance: float

    def __post_init__(self) -> None:
        initial = _finite_float(
            self.initial_goal_distance, name="initial_goal_distance"
        )
        terminal = _finite_float(
            self.terminal_goal_distance, name="terminal_goal_distance"
        )
        if initial < 0.0 or terminal < 0.0:
            raise ValueError("goal distances must be nonnegative")
        object.__setattr__(self, "initial_goal_distance", initial)
        object.__setattr__(self, "terminal_goal_distance", terminal)

    @property
    def value(self) -> float:
        return self.initial_goal_distance - self.terminal_goal_distance

    def to_state_dict(self) -> dict[str, float]:
        return {
            "initial_goal_distance": self.initial_goal_distance,
            "terminal_goal_distance": self.terminal_goal_distance,
        }


@dataclass(frozen=True, eq=False)
class VerificationRecord:
    """One immutable, fully evaluated planned-window verifier query."""

    context: QueryContext
    gamma: float
    plan: NDArray[Any]
    source: QuerySource
    feature_z: NDArray[np.float64]
    acquisition_sigma: float
    safety: SafetyResult
    progress: ProgressResult
    executed: bool = False
    generated_hash: str | None = None
    verifier_input_hash: str | None = None
    query_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.context, QueryContext):
            raise TypeError("context must be a QueryContext")
        gamma = _finite_float(self.gamma, name="gamma")
        plan = _immutable_array(
            self.plan, name="plan", shape=(PLAN_HORIZON, ACTION_DIM)
        )
        feature = _immutable_array(
            self.feature_z,
            name="feature_z",
            shape=(FEATURE_DIM,),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(feature))
        if norm <= np.finfo(np.float64).tiny:
            raise ValueError("feature_z has zero norm")
        # Store the prescribed normalized representation, not an unchecked
        # caller approximation to it.
        feature = _immutable_array(
            feature / norm,
            name="feature_z",
            shape=(FEATURE_DIM,),
            dtype=np.float64,
        )
        sigma = _finite_float(self.acquisition_sigma, name="acquisition_sigma")
        if sigma < 0.0:
            raise ValueError("acquisition_sigma must be nonnegative")
        try:
            source = QuerySource(self.source)
        except ValueError as exc:
            raise ValueError(f"unknown query source: {self.source!r}") from exc
        if not isinstance(self.safety, SafetyResult):
            raise TypeError("safety must be a SafetyResult")
        if not isinstance(self.progress, ProgressResult):
            raise TypeError("progress must be a ProgressResult")
        executed = bool(self.executed)
        if executed and not self.safety.safe:
            raise ValueError("an unsafe verifier record cannot be marked executed")

        expected = query_content_hash(self.context, gamma, plan)
        supplied_query = self.query_hash or expected
        supplied_query = _validate_hash(supplied_query, name="query_hash")
        generated = _validate_hash(
            self.generated_hash or expected, name="generated_hash"
        )
        verifier = _validate_hash(
            self.verifier_input_hash or expected, name="verifier_input_hash"
        )
        if supplied_query != expected or generated != expected or verifier != expected:
            raise ValueError(
                "identity mismatch: generated plan, verifier input, and exact "
                "(model context, exact verifier state/spec, gamma, plan) "
                "content must have the same hash"
            )

        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "feature_z", feature)
        object.__setattr__(self, "acquisition_sigma", sigma)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "executed", executed)
        object.__setattr__(self, "query_hash", expected)
        object.__setattr__(self, "generated_hash", expected)
        object.__setattr__(self, "verifier_input_hash", expected)

    @property
    def safe(self) -> bool:
        return self.safety.safe

    @property
    def progress_value(self) -> float:
        return self.progress.value

    @property
    def executed_action(self) -> NDArray[Any] | None:
        if not self.executed:
            return None
        action = self.plan[0].view()
        action.setflags(write=False)
        return action

    def rehash(self) -> str:
        return query_content_hash(self.context, self.gamma, self.plan)

    def validate_identity(self) -> None:
        expected = self.rehash()
        if not (
            expected == self.query_hash
            and expected == self.generated_hash
            and expected == self.verifier_input_hash
        ):
            raise ValueError("verification-record identity hash mismatch")

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_state_dict(),
            "gamma": self.gamma,
            "plan": np.array(self.plan, copy=True),
            "source": self.source.value,
            "feature_z": np.array(self.feature_z, copy=True),
            "acquisition_sigma": self.acquisition_sigma,
            "safety": self.safety.to_state_dict(),
            "progress": self.progress.to_state_dict(),
            "executed": self.executed,
            "generated_hash": self.generated_hash,
            "verifier_input_hash": self.verifier_input_hash,
            "query_hash": self.query_hash,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "VerificationRecord":
        return cls(
            context=QueryContext.from_state_dict(state["context"]),
            gamma=state["gamma"],
            plan=state["plan"],
            source=state["source"],
            feature_z=state["feature_z"],
            acquisition_sigma=state["acquisition_sigma"],
            safety=SafetyResult(**state["safety"]),
            progress=ProgressResult(**state["progress"]),
            executed=state.get("executed", False),
            generated_hash=state.get("generated_hash"),
            verifier_input_hash=state.get("verifier_input_hash"),
            query_hash=state.get("query_hash", ""),
        )


@dataclass(frozen=True, eq=False)
class ReplayItem:
    """A safe ledger row exposed as a CFM target without replay weights."""

    context: QueryContext
    gamma: float
    plan: NDArray[Any]
    source_query_hash: str
    training_target_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.context, QueryContext):
            raise TypeError("context must be a QueryContext")
        gamma = _finite_float(self.gamma, name="gamma")
        plan = _immutable_array(
            self.plan, name="plan", shape=(PLAN_HORIZON, ACTION_DIM)
        )
        expected = query_content_hash(self.context, gamma, plan)
        source_hash = _validate_hash(
            self.source_query_hash, name="source_query_hash"
        )
        training_hash = _validate_hash(
            self.training_target_hash, name="training_target_hash"
        )
        if source_hash != expected or training_hash != expected:
            raise ValueError(
                "replay identity mismatch: the training target is not the "
                "verified planned query"
            )
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "source_query_hash", expected)
        object.__setattr__(self, "training_target_hash", expected)

    @classmethod
    def from_record(cls, record: VerificationRecord) -> "ReplayItem":
        record.validate_identity()
        if not record.safe:
            raise ValueError("only verified-safe records may enter positive replay")
        return cls(
            context=record.context,
            gamma=record.gamma,
            plan=record.plan,
            source_query_hash=record.query_hash,
            training_target_hash=query_content_hash(
                record.context, record.gamma, record.plan
            ),
        )

    def validate_identity(self) -> None:
        expected = query_content_hash(self.context, self.gamma, self.plan)
        if expected != self.source_query_hash or expected != self.training_target_hash:
            raise ValueError("replay identity hash mismatch")
