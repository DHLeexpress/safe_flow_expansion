#!/usr/bin/env python3
"""Render the planned-window AFE process without inventing training rows.

The two large panels have deliberately different visual semantics:

* acquisition: every ordinary conditional-flow candidate is colored by its
  fixed-feature linear uncertainty (``viridis``), and every plan actually sent
  to the full verifier receives an accepted/rejected endpoint marker;
* replay: only the exact, verifier-positive planned windows used by the
  proximal CFM update are shown, colored by safety level (truncated ``plasma``).

The module accepts either the live controller/store objects through
``build_expansion_frames`` / ``render_from_controller_data`` or the lossless
JSON snapshot emitted by ``save_visualization_data``.  It never reconstructs a
training window from executed first actions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .audit import AuditResult
from .config import DynamicsConfig
from .controller import ControlStepTrace
from .dynamics import planned_positions
from .proximal_update import ProximalUpdateResult
from .schemas import QuerySource, VerificationRecord
from .store import VerificationStore


FORMAT_VERSION = 2
SIGMA_CMAP = "viridis"
GAMMA_CMAP_NAME = "plasma_trunc"
GAMMA_LEVELS = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0), dtype=np.float64)
# Exact discrete samples used by the original all-gamma figure (Image #1).
_GAMMA_RGBA = plt.get_cmap("plasma")(np.linspace(0.02, 0.90, len(GAMMA_LEVELS)))
GAMMA_CMAP = ListedColormap(_GAMMA_RGBA, name=GAMMA_CMAP_NAME)
_GAMMA_EDGES = np.empty(len(GAMMA_LEVELS) + 1, dtype=np.float64)
_GAMMA_EDGES[1:-1] = 0.5 * (GAMMA_LEVELS[:-1] + GAMMA_LEVELS[1:])
_GAMMA_EDGES[0] = GAMMA_LEVELS[0] - (_GAMMA_EDGES[1] - GAMMA_LEVELS[0])
_GAMMA_EDGES[-1] = GAMMA_LEVELS[-1] + (GAMMA_LEVELS[-1] - _GAMMA_EDGES[-2])
GAMMA_NORM = BoundaryNorm(_GAMMA_EDGES, GAMMA_CMAP.N)

# Exposed for tests and downstream report builders.  These strings are also
# used verbatim in the rendered panels and manifest.
RENDER_LABELS = {
    "acquisition": "Plan acquisition — ordinary T=1 flow candidates",
    "replay": "Uniform positive flow-query replay — exact fully verified planned targets",
    "validity": "Query acceptance ≠ held-out validity (audit T=1)",
    "matrix": "Cumulative fixed-feature Aₙ",
    "counts": "FLOW-query, verifier, and backup accounting",
    "proximal": "Proximal CFM solver",
    "coverage": "Exact replay coverage",
}


FloatArray = NDArray[np.float64]


def _array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
    tail: tuple[int, ...] | None = None,
) -> FloatArray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.shape}")
    if tail is not None and result.shape[-len(tail) :] != tail:
        raise ValueError(f"{name} must end in shape {tail}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    result.setflags(write=False)
    return result


def _float(value: Any, *, name: str, finite: bool = True) -> float:
    result = float(value)
    if finite and not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class SceneSnapshot:
    """Everything needed to draw robot-centre trajectories faithfully."""

    obstacles: FloatArray
    robot_radius: float
    start: FloatArray
    goal: FloatArray
    bounds: tuple[float, float, float, float] = (0.0, 5.0, 0.0, 5.0)

    def __post_init__(self) -> None:
        obstacles = _array(self.obstacles, name="obstacles", ndim=2, tail=(3,))
        start = _array(self.start, name="start", ndim=1, tail=(2,))
        goal = _array(self.goal, name="goal", ndim=1, tail=(2,))
        radius = _float(self.robot_radius, name="robot_radius")
        if radius < 0.0 or np.any(obstacles[:, 2] < 0.0):
            raise ValueError("scene radii must be nonnegative")
        bounds = tuple(float(value) for value in self.bounds)
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("bounds must be four finite values")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("scene bounds must be ordered")
        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "robot_radius", radius)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "bounds", bounds)

    @classmethod
    def from_environment(cls, env) -> "SceneSnapshot":
        start_state = np.asarray(env.x0.detach().cpu(), dtype=np.float64)
        return cls(
            obstacles=env.obstacles.detach().cpu().numpy(),
            robot_radius=float(env.r_robot),
            start=start_state[:2],
            goal=env.goal.detach().cpu().numpy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacles": self.obstacles.tolist(),
            "robot_radius": self.robot_radius,
            "start": self.start.tolist(),
            "goal": self.goal.tolist(),
            "bounds": list(self.bounds),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SceneSnapshot":
        return cls(
            obstacles=value["obstacles"],
            robot_radius=value["robot_radius"],
            start=value["start"],
            goal=value["goal"],
            bounds=tuple(value.get("bounds", (0.0, 5.0, 0.0, 5.0))),
        )


@dataclass(frozen=True)
class PlanSnapshot:
    """One exact planned-window path and its immutable query identity."""

    query_hash: str
    path: FloatArray
    gamma: float
    source: str
    safe: bool
    candidate_index: int
    sigma: float
    strict_bounds: bool = True

    def __post_init__(self) -> None:
        if len(self.query_hash) != 64:
            raise ValueError("query_hash must be a SHA-256 digest")
        try:
            bytes.fromhex(self.query_hash)
        except ValueError as exc:
            raise ValueError("query_hash must be a SHA-256 digest") from exc
        path = _array(self.path, name="plan path", ndim=2, tail=(2,))
        if len(path) < 2:
            raise ValueError("plan path must include current and predicted positions")
        gamma = _float(self.gamma, name="gamma")
        sigma = _float(self.sigma, name="sigma")
        if sigma < 0.0:
            raise ValueError("sigma must be nonnegative")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "safe", bool(self.safe))
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "strict_bounds", bool(self.strict_bounds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "path": self.path.tolist(),
            "gamma": self.gamma,
            "source": self.source,
            "safe": self.safe,
            "candidate_index": self.candidate_index,
            "sigma": self.sigma,
            "strict_bounds": self.strict_bounds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanSnapshot":
        return cls(**dict(value))


@dataclass(frozen=True)
class AuditSnapshot:
    gamma: float
    validity: float
    validity_low: float
    validity_high: float
    progress_validity: float
    progress_low: float
    progress_high: float
    sample_count: int

    def __post_init__(self) -> None:
        for name in (
            "validity", "validity_low", "validity_high",
            "progress_validity", "progress_low", "progress_high",
        ):
            value = _float(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "gamma", _float(self.gamma, name="gamma"))
        object.__setattr__(self, "sample_count", int(self.sample_count))
        if self.sample_count <= 0:
            raise ValueError("audit sample_count must be positive")


@dataclass(frozen=True)
class AuditProvenance:
    """Explicit provenance required before audit values may enter a video."""

    temperature: float
    uncertainty_tilting: bool
    sampling_distribution: str
    context_bank_fingerprint: str
    context_bank_role: str

    def __post_init__(self) -> None:
        if _float(self.temperature, name="audit provenance temperature") != 1.0:
            raise ValueError("held-out audit provenance must explicitly use T=1")
        if bool(self.uncertainty_tilting):
            raise ValueError("held-out audit provenance must explicitly be untilted")
        distribution = str(self.sampling_distribution)
        if distribution != "ordinary_conditional_flow_iid":
            raise ValueError(
                "held-out audit must explicitly use ordinary_conditional_flow_iid"
            )
        fingerprint = str(self.context_bank_fingerprint)
        if len(fingerprint) != 64:
            raise ValueError("audit context-bank fingerprint must be SHA-256")
        try:
            bytes.fromhex(fingerprint)
        except ValueError as exc:
            raise ValueError("audit context-bank fingerprint must be SHA-256") from exc
        role = str(self.context_bank_role)
        if role not in {"round_monitoring", "sealed_final_test"}:
            raise ValueError("audit context-bank role is not explicit")
        object.__setattr__(self, "temperature", 1.0)
        object.__setattr__(self, "uncertainty_tilting", False)
        object.__setattr__(self, "sampling_distribution", distribution)
        object.__setattr__(self, "context_bank_fingerprint", fingerprint)
        object.__setattr__(self, "context_bank_role", role)


@dataclass(frozen=True)
class ProximalSnapshot:
    objective: float
    cfm_loss: float
    proximal_penalty: float
    gradient_norm: float
    update_norm: float
    optimizer_steps: int
    positive_coverage: float
    stopping_reason: str

    def __post_init__(self) -> None:
        for name in (
            "objective", "cfm_loss", "proximal_penalty", "gradient_norm", "update_norm",
        ):
            object.__setattr__(self, name, _float(getattr(self, name), name=name))
        coverage = _float(self.positive_coverage, name="positive_coverage")
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("positive_coverage must lie in [0, 1]")
        object.__setattr__(self, "positive_coverage", coverage)
        object.__setattr__(self, "optimizer_steps", int(self.optimizer_steps))
        object.__setattr__(self, "stopping_reason", str(self.stopping_reason))


@dataclass(frozen=True)
class ExpansionVizFrame:
    """A lossless rendering snapshot after one controller query event."""

    frame_index: int
    round_index: int
    control_step: int
    gamma: float
    candidate_paths: FloatArray
    candidate_sigmas: FloatArray
    acquired: tuple[PlanSnapshot, ...]
    executed: PlanSnapshot | None
    replay: tuple[PlanSnapshot, ...]
    A_eigenvalues: FloatArray
    A_logdet: float
    A_observation_count: int
    query_count: int
    positive_count: int
    backup_query_count: int
    backup_positive_count: int
    verifier_calls: int
    fallback_count: int
    cache_hits: int
    audit: tuple[AuditSnapshot, ...] = ()
    audit_provenance: AuditProvenance | None = None
    proximal: ProximalSnapshot | None = None
    expansion_temperature: float = 1.0
    audit_temperature: float = 1.0
    replay_eligibility: str = "full_safe"
    runtime_safety_claim: bool = True
    method_label: str = "Full"
    acquisition_mode: str = "afe"
    progress_ranking: bool = True

    def __post_init__(self) -> None:
        paths = _array(self.candidate_paths, name="candidate_paths", ndim=3, tail=(2,))
        sigmas = _array(self.candidate_sigmas, name="candidate_sigmas", ndim=1)
        if len(paths) != len(sigmas):
            raise ValueError("candidate paths and sigma lengths differ")
        if len(paths) == 0 or paths.shape[1] < 2:
            raise ValueError("each frame requires nonempty full candidate paths")
        if np.any(sigmas < 0.0):
            raise ValueError("candidate sigma must be nonnegative")
        eigenvalues = _array(self.A_eigenvalues, name="A_eigenvalues", ndim=1)
        if len(eigenvalues) == 0 or np.any(eigenvalues <= 0.0):
            raise ValueError("A eigenvalues must be positive")
        if float(self.expansion_temperature) != 1.0:
            raise ValueError("expansion visualization requires ordinary T=1 candidates")
        if float(self.audit_temperature) != 1.0:
            raise ValueError("held-out audit visualization requires ordinary T=1 samples")
        if self.audit_provenance is None:
            raise ValueError("held-out audit provenance must be supplied explicitly")
        if self.audit_provenance.temperature != self.audit_temperature:
            raise ValueError("frame audit temperature disagrees with audit provenance")
        if self.acquisition_mode not in {"afe", "uniform"}:
            raise ValueError("unknown acquisition mode")
        if self.replay_eligibility not in {"full_safe", "strict_bounds"}:
            raise ValueError("unknown replay eligibility")
        if self.runtime_safety_claim:
            if self.executed is not None and not self.executed.safe:
                raise ValueError("certified executed plan must be fully verifier-positive")
            if self.replay_eligibility != "full_safe":
                raise ValueError("a bounds-only replay cannot carry a runtime-safety claim")
        elif self.executed is not None and not self.executed.strict_bounds:
            raise ValueError("offline bounds-only selection must still satisfy strict bounds")
        if self.replay_eligibility == "full_safe" and any(not plan.safe for plan in self.replay):
            raise ValueError("uniform positive replay may contain verifier-positive plans only")
        if self.replay_eligibility == "strict_bounds" and any(
            not plan.strict_bounds for plan in self.replay
        ):
            raise ValueError("offline bounds-only replay contains an out-of-bounds plan")
        replay_hashes = [plan.query_hash for plan in self.replay]
        if len(set(replay_hashes)) != len(replay_hashes):
            raise ValueError("replay snapshot contains a duplicate query hash")
        for name in (
            "frame_index", "round_index", "control_step", "A_observation_count",
            "query_count", "positive_count", "backup_query_count",
            "backup_positive_count", "verifier_calls", "fallback_count", "cache_hits",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.A_observation_count != self.query_count + self.backup_query_count:
            raise ValueError(
                "A observation count must equal FLOW plus backup verifier-query count"
            )
        if self.positive_count > self.query_count:
            raise ValueError("positive_count cannot exceed query_count")
        if self.backup_positive_count > self.backup_query_count:
            raise ValueError("backup_positive_count cannot exceed backup_query_count")
        if self.verifier_calls != self.A_observation_count:
            raise ValueError("verifier calls must equal all unique verifier observations")
        object.__setattr__(self, "candidate_paths", paths)
        object.__setattr__(self, "candidate_sigmas", sigmas)
        object.__setattr__(self, "A_eigenvalues", eigenvalues)
        object.__setattr__(self, "A_logdet", _float(self.A_logdet, name="A_logdet"))
        object.__setattr__(self, "gamma", _float(self.gamma, name="gamma"))
        object.__setattr__(self, "acquired", tuple(self.acquired))
        object.__setattr__(self, "replay", tuple(self.replay))
        object.__setattr__(self, "audit", tuple(self.audit))
        object.__setattr__(self, "runtime_safety_claim", bool(self.runtime_safety_claim))
        object.__setattr__(self, "method_label", str(self.method_label))
        object.__setattr__(self, "progress_ranking", bool(self.progress_ranking))

    @property
    def query_acceptance(self) -> float:
        """Acceptance of ordinary FLOW queries only; backup is separate."""

        return self.positive_count / self.query_count if self.query_count else float("nan")

    @property
    def backup_acceptance(self) -> float:
        return (
            self.backup_positive_count / self.backup_query_count
            if self.backup_query_count else float("nan")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "round_index": self.round_index,
            "control_step": self.control_step,
            "gamma": self.gamma,
            "candidate_paths": self.candidate_paths.tolist(),
            "candidate_sigmas": self.candidate_sigmas.tolist(),
            "acquired": [item.to_dict() for item in self.acquired],
            "executed": self.executed.to_dict() if self.executed else None,
            "replay": [item.to_dict() for item in self.replay],
            "A_eigenvalues": self.A_eigenvalues.tolist(),
            "A_logdet": self.A_logdet,
            "A_observation_count": self.A_observation_count,
            "query_count": self.query_count,
            "positive_count": self.positive_count,
            "backup_query_count": self.backup_query_count,
            "backup_positive_count": self.backup_positive_count,
            "verifier_calls": self.verifier_calls,
            "fallback_count": self.fallback_count,
            "cache_hits": self.cache_hits,
            "audit": [asdict(item) for item in self.audit],
            "audit_provenance": asdict(self.audit_provenance),
            "proximal": asdict(self.proximal) if self.proximal else None,
            "expansion_temperature": self.expansion_temperature,
            "audit_temperature": self.audit_temperature,
            "replay_eligibility": self.replay_eligibility,
            "runtime_safety_claim": self.runtime_safety_claim,
            "method_label": self.method_label,
            "acquisition_mode": self.acquisition_mode,
            "progress_ranking": self.progress_ranking,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpansionVizFrame":
        state = dict(value)
        state["acquired"] = tuple(PlanSnapshot.from_dict(item) for item in state["acquired"])
        state["executed"] = (
            PlanSnapshot.from_dict(state["executed"]) if state.get("executed") else None
        )
        state["replay"] = tuple(PlanSnapshot.from_dict(item) for item in state["replay"])
        state["audit"] = tuple(AuditSnapshot(**item) for item in state.get("audit", ()))
        provenance = state.get("audit_provenance")
        state["audit_provenance"] = (
            AuditProvenance(**provenance) if provenance is not None else None
        )
        state["proximal"] = (
            ProximalSnapshot(**state["proximal"]) if state.get("proximal") else None
        )
        return cls(**state)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


_MISSING = object()


def _required(value: Any, name: str) -> Any:
    result = _get(value, name, _MISSING)
    if result is _MISSING:
        raise ValueError(f"audit result is missing explicit {name!r} provenance")
    return result


def _audit_provenance(value: AuditResult | Mapping[str, Any]) -> AuditProvenance:
    return AuditProvenance(
        temperature=float(_required(value, "temperature")),
        uncertainty_tilting=bool(_required(value, "uncertainty_tilting")),
        sampling_distribution=str(_required(value, "sampling_distribution")),
        context_bank_fingerprint=str(_required(value, "context_bank_fingerprint")),
        context_bank_role=str(_required(value, "context_bank_role")),
    )


def _audit_snapshot(value: AuditResult | Mapping[str, Any] | None) -> tuple[AuditSnapshot, ...]:
    if value is None:
        return ()
    _audit_provenance(value)
    per_gamma = _required(value, "per_gamma")
    if per_gamma is None:
        raise ValueError("audit result is missing per_gamma metrics")
    if isinstance(per_gamma, Mapping):
        rows = list(per_gamma.values())
    else:
        rows = list(per_gamma)
    result: list[AuditSnapshot] = []
    for row in rows:
        validity_interval = _get(row, "validity_interval")
        progress_interval = _get(row, "progress_validity_interval")
        validity = float(_get(row, "validity_mass"))
        progress = float(_get(row, "progress_validity"))
        result.append(AuditSnapshot(
            gamma=float(_get(row, "gamma")),
            validity=validity,
            validity_low=float(_get(validity_interval, "low", validity)),
            validity_high=float(_get(validity_interval, "high", validity)),
            progress_validity=progress,
            progress_low=float(_get(progress_interval, "low", progress)),
            progress_high=float(_get(progress_interval, "high", progress)),
            sample_count=int(_get(row, "sample_count")),
        ))
    return tuple(sorted(result, key=lambda row: row.gamma))


def _proximal_snapshot(value: ProximalUpdateResult | Mapping[str, Any] | None) -> ProximalSnapshot | None:
    if value is None:
        return None
    trace = list(_get(value, "trace", ()))
    if not trace:
        # Exact no-positive or no-trainable-parameter no-op.
        return ProximalSnapshot(
            objective=0.0,
            cfm_loss=0.0,
            proximal_penalty=0.0,
            gradient_norm=0.0,
            update_norm=float(_get(value, "final_update_norm", 0.0)),
            optimizer_steps=int(_get(value, "optimizer_steps", 0)),
            positive_coverage=0.0,
            stopping_reason=str(_get(value, "stopping_reason", "no_update")),
        )
    last = trace[-1]
    return ProximalSnapshot(
        objective=float(_get(last, "objective")),
        cfm_loss=float(_get(last, "cfm_loss")),
        proximal_penalty=float(_get(last, "proximal_penalty")),
        gradient_norm=float(_get(last, "gradient_norm")),
        update_norm=float(_get(value, "final_update_norm", _get(last, "update_norm"))),
        optimizer_steps=int(_get(value, "optimizer_steps")),
        positive_coverage=float(min(_get(row, "positive_coverage", 0.0) for row in trace)),
        stopping_reason=str(_get(value, "stopping_reason")),
    )


def _indexed(values: Any, count: int, *, name: str) -> list[Any]:
    """Normalize optional per-frame data; a scalar belongs to the final frame."""

    result = [None] * count
    if values is None:
        return result
    if isinstance(values, Mapping) and all(isinstance(key, (int, np.integer)) for key in values):
        for key, value in values.items():
            index = int(key)
            if index < 0 or index >= count:
                raise ValueError(f"{name} frame index {index} is out of range")
            result[index] = value
        return result
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != count:
            raise ValueError(f"{name} sequence must have one entry per trace")
        return list(values)
    result[-1] = values
    return result


def _replay_hashes(
    proximal: Any,
    visible: Sequence[VerificationRecord],
    explicit: Sequence[str] | None,
    replay_eligibility: str,
) -> tuple[str, ...]:
    if replay_eligibility == "full_safe":
        eligible = [
            record
            for record in visible
            if record.source is QuerySource.FLOW and record.safe
        ]
    elif replay_eligibility == "strict_bounds":
        eligible = [
            record
            for record in visible
            if record.source is QuerySource.FLOW
            and record.safety.strict_bounds
        ]
    else:
        raise ValueError("unknown replay eligibility")
    eligible_hashes = tuple(record.query_hash for record in eligible)
    if proximal is None:
        if explicit:
            raise ValueError("replay hashes were supplied for a frame with no proximal update")
        return ()
    if explicit is not None:
        hashes = tuple(str(value) for value in explicit)
    else:
        total = int(_get(proximal, "total_record_count"))
        trace = list(_get(proximal, "trace", ()))
        if not trace and not eligible:
            return ()
        indices = sorted({
            int(index)
            for step in trace
            for index in _get(step, "original_record_indices", ())
        })
        if total == len(visible):
            hashes = tuple(visible[index].query_hash for index in indices)
        elif total == len(eligible):
            hashes = tuple(eligible[index].query_hash for index in indices)
        else:
            raise ValueError(
                "cannot tie proximal telemetry to ledger rows; supply exact replay_query_hashes"
            )
    if len(set(hashes)) != len(hashes):
        raise ValueError("an exact replay pass cannot contain duplicate query hashes")
    if set(hashes) != set(eligible_hashes):
        raise ValueError(
            "proximal replay must be the uniform full eligible ledger view; "
            "its hashes differ from the immutable visible training view"
        )
    return hashes


def build_expansion_frames(
    traces: Sequence[ControlStepTrace],
    store: VerificationStore,
    proximal_telemetry: Any = None,
    *,
    audit_results: Any = None,
    replay_query_hashes: Any = None,
    round_indices: Sequence[int] | None = None,
    dynamics: DynamicsConfig = DynamicsConfig(),
    expansion_temperature: float = 1.0,
    audit_temperature: float = 1.0,
    replay_eligibility: str = "full_safe",
    runtime_safety_claim: bool = True,
    method_label: str = "Full",
    acquisition_mode: str = "afe",
    progress_ranking: bool = True,
) -> tuple[ExpansionVizFrame, ...]:
    """Build exact frames from live controller traces and the append-only store.

    A strict ordered event check ensures every store row is referenced by the
    exact controller verifier event that appended it.  This intentionally
    refuses older data in which executed first actions were silently
    reassembled into a different H=10 window.
    """

    trace_rows = tuple(traces)
    if not trace_rows:
        raise ValueError("at least one controller trace is required")
    if float(expansion_temperature) != 1.0 or float(audit_temperature) != 1.0:
        raise ValueError("AFE expansion and independent audit frames must both use T=1")
    if replay_eligibility not in {"full_safe", "strict_bounds"}:
        raise ValueError("unknown replay eligibility")
    if replay_eligibility == "strict_bounds" and runtime_safety_claim:
        raise ValueError("offline bounds-only replay cannot claim runtime safety")
    if acquisition_mode not in {"afe", "uniform"}:
        raise ValueError("acquisition_mode must be 'afe' or 'uniform'")
    records = tuple(store.records)
    by_hash = {record.query_hash: record for record in records}
    if len(by_hash) != len(records):
        raise ValueError("verification store contains duplicate query hashes")
    traced_hashes = tuple(
        query.query_hash for trace in trace_rows for query in trace.queried
    )
    record_hashes = tuple(record.query_hash for record in records)
    if traced_hashes != record_hashes:
        missing = sorted(set(record_hashes) - set(traced_hashes))
        unknown = sorted(set(traced_hashes) - set(record_hashes))
        raise ValueError(
            "trace/store event-order identity mismatch: "
            f"untraced_store={missing[:3]}, unknown_trace={unknown[:3]}"
        )

    proximal_by_frame = _indexed(proximal_telemetry, len(trace_rows), name="proximal telemetry")
    audit_by_frame = _indexed(audit_results, len(trace_rows), name="audit results")
    replay_by_frame = _indexed(replay_query_hashes, len(trace_rows), name="replay hashes")
    rounds = list(range(len(trace_rows))) if round_indices is None else [int(v) for v in round_indices]
    if len(rounds) != len(trace_rows):
        raise ValueError("round_indices must have one entry per trace")
    explicit_audits = [value for value in audit_by_frame if value is not None]
    if not explicit_audits:
        raise ValueError("at least one explicit independent audit is required")
    audit_provenance = _audit_provenance(explicit_audits[0])
    if any(_audit_provenance(value) != audit_provenance for value in explicit_audits[1:]):
        raise ValueError("visualization audits do not share one explicit provenance")

    matrix = (
        np.eye(store.uncertainty.feature_dim, dtype=np.float64)
        * store.uncertainty.prior_precision
    )
    visible: list[VerificationRecord] = []
    visible_hashes: set[str] = set()
    start_state: dict[str, np.ndarray] = {}
    verifier_calls = fallback_count = cache_hits = 0
    flow_query_count = flow_positive_count = 0
    backup_query_count = backup_positive_count = 0
    frames: list[ExpansionVizFrame] = []
    for frame_index, trace in enumerate(trace_rows):
        state = np.asarray(trace.state_before, dtype=np.float64)
        if state.shape != (4,):
            raise ValueError("controller state_before must have shape (4,)")
        candidate_plans = np.asarray(trace.candidate_plans, dtype=np.float64)
        candidate_sigmas = np.asarray(trace.candidate_sigmas, dtype=np.float64)
        acquisition_probabilities = np.asarray(
            trace.acquisition_probabilities, dtype=np.float64,
        )
        acquisition_order = np.asarray(trace.acquisition_order, dtype=np.int64)
        if candidate_plans.ndim != 3 or candidate_plans.shape[1:] != (10, 2):
            raise ValueError("controller candidate_plans must have shape [K,10,2]")
        if candidate_sigmas.shape != (len(candidate_plans),):
            raise ValueError("controller candidate sigma count differs from candidates")
        if acquisition_probabilities.shape != (len(candidate_plans),):
            raise ValueError("controller acquisition probability count differs from candidates")
        if (
            not np.isfinite(acquisition_probabilities).all()
            or np.any(acquisition_probabilities < 0.0)
            or not np.isclose(acquisition_probabilities.sum(), 1.0)
        ):
            raise ValueError("controller acquisition probabilities are invalid")
        if acquisition_mode == "uniform" and not np.allclose(
            acquisition_probabilities, 1.0 / len(candidate_plans),
            rtol=1e-7, atol=1e-9,
        ):
            raise ValueError("-AFE trace is not actually uniform acquisition")
        if acquisition_order.shape != (len(candidate_plans),) or set(
            acquisition_order.tolist()
        ) != set(range(len(candidate_plans))):
            raise ValueError("controller acquisition_order is not a candidate permutation")
        if int(trace.verifier_calls) != len(trace.queried):
            raise ValueError("trace verifier_calls differs from its exact new-query events")
        if str(getattr(trace, "eligibility_mode", "full")) != (
            "bounds_only_offline" if replay_eligibility == "strict_bounds" else "full"
        ):
            raise ValueError("trace eligibility mode disagrees with artifact arm")
        if bool(getattr(trace, "runtime_safety_claim", True)) != bool(runtime_safety_claim):
            raise ValueError("trace runtime-safety claim disagrees with artifact arm")
        candidate_paths = np.asarray([
            planned_positions(state, plan, config=dynamics) for plan in candidate_plans
        ])

        acquired: list[PlanSnapshot] = []
        for query in trace.queried:
            record = by_hash[query.query_hash]
            if query.query_hash in visible_hashes:
                raise ValueError("queried trace repeats a stored verifier observation")
            visible_hashes.add(query.query_hash)
            visible.append(record)
            record.validate_identity()
            if not np.array_equal(record.context.verifier_state, state):
                raise ValueError("query record is associated with the wrong control event state")
            if not np.isclose(record.gamma, trace.gamma, rtol=0.0, atol=0.0):
                raise ValueError("query record gamma differs from its control event")
            if query.source != record.source.value:
                raise ValueError("query trace source differs from its ledger record")
            if (
                bool(query.safe) != record.safe
                or bool(query.in_bounds) != record.safety.strict_bounds
                or bool(query.socp_ok) != record.safety.socp_certified
            ):
                raise ValueError("query trace verifier outcome differs from its ledger record")
            if not np.isclose(query.acquisition_sigma, record.acquisition_sigma):
                raise ValueError("query trace sigma differs from its ledger record")
            if not np.isclose(query.progress_m, record.progress_value):
                raise ValueError("query trace progress differs from its ledger record")
            if not np.isclose(query.clearance_m, record.safety.min_clearance):
                raise ValueError("query trace clearance differs from its ledger record")
            if bool(query.cache_hit):
                raise ValueError("a stored verifier observation cannot be labeled a cache hit")
            if record.source is QuerySource.FLOW:
                index = int(query.candidate_index)
                if index < 0 or index >= len(candidate_plans):
                    raise ValueError("FLOW query candidate index is outside its event")
                if not np.array_equal(
                    np.asarray(record.plan, dtype=np.float64), candidate_plans[index]
                ):
                    raise ValueError("FLOW query plan differs from its event candidate")
                if not np.isclose(record.acquisition_sigma, candidate_sigmas[index]):
                    raise ValueError("FLOW query sigma differs from its candidate sigma")
                flow_query_count += 1
                flow_positive_count += int(record.safe)
            elif record.source is QuerySource.SAFEMPPI_BACKUP:
                if int(query.candidate_index) != -1:
                    raise ValueError("backup query must use candidate_index=-1")
                backup_query_count += 1
                backup_positive_count += int(record.safe)
            else:
                raise ValueError("active-expansion trace contains an unsupported query source")
            start_state[query.query_hash] = state.copy()
            matrix += np.outer(record.feature_z, record.feature_z) / store.uncertainty.lambda_
            path = planned_positions(state, record.plan, config=dynamics)
            acquired.append(PlanSnapshot(
                query_hash=record.query_hash,
                path=path,
                gamma=record.gamma,
                source=record.source.value,
                safe=record.safe,
                candidate_index=int(query.candidate_index),
                sigma=record.acquisition_sigma,
                strict_bounds=record.safety.strict_bounds,
            ))

        executed = None
        if trace.selected_query_hash is not None:
            selected = by_hash.get(trace.selected_query_hash)
            if selected is None:
                raise ValueError("selected plan hash is absent from verification store")
            if trace.selected_source != selected.source.value:
                raise ValueError("selected source differs from its ledger record")
            if bool(trace.fallback_used) != (
                selected.source is QuerySource.SAFEMPPI_BACKUP
            ):
                raise ValueError("fallback flag differs from selected plan source")
            if trace.action is None or not np.array_equal(
                np.asarray(trace.action, dtype=np.float64),
                np.asarray(selected.plan[0], dtype=np.float64),
            ):
                raise ValueError("executed action is not selected verified plan[0]")
            selected_flags = [
                bool(item.executed)
                for item in trace.queried
                if item.query_hash == selected.query_hash
            ]
            if selected_flags and selected_flags != [True]:
                raise ValueError("selected new query lacks exact executed-event association")
            start_state[selected.query_hash] = state.copy()
            executed = PlanSnapshot(
                query_hash=selected.query_hash,
                path=planned_positions(state, selected.plan, config=dynamics),
                gamma=selected.gamma,
                source=selected.source.value,
                safe=selected.safe,
                candidate_index=next(
                    (item.candidate_index for item in acquired if item.query_hash == selected.query_hash), -1
                ),
                sigma=selected.acquisition_sigma,
                strict_bounds=selected.safety.strict_bounds,
            )
        elif (
            trace.action is not None
            or not bool(trace.fail_closed)
            or not bool(trace.fallback_used)
            or trace.selected_source is not None
        ):
            raise ValueError("unselected control event must be explicitly fail-closed")

        proximal_value = proximal_by_frame[frame_index]
        explicit_hashes = replay_by_frame[frame_index]
        hashes = _replay_hashes(
            proximal_value, visible, explicit_hashes, replay_eligibility,
        )
        replay: list[PlanSnapshot] = []
        for query_hash in hashes:
            record = by_hash[query_hash]
            replay.append(PlanSnapshot(
                query_hash=query_hash,
                path=planned_positions(start_state[query_hash], record.plan, config=dynamics),
                gamma=record.gamma,
                source=record.source.value,
                safe=record.safe,
                candidate_index=-1,
                sigma=record.acquisition_sigma,
                strict_bounds=record.safety.strict_bounds,
            ))

        verifier_calls += int(trace.verifier_calls)
        fallback_count += int(trace.fallback_used)
        cache_hits += int(trace.cache_hits)
        if verifier_calls != len(visible):
            raise ValueError("controller verifier-call count differs from exact visible ledger rows")
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) * 0.5)
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0:
            raise ValueError("reconstructed A is not positive definite")
        frames.append(ExpansionVizFrame(
            frame_index=frame_index,
            round_index=rounds[frame_index],
            control_step=int(trace.step),
            gamma=float(trace.gamma),
            candidate_paths=candidate_paths,
            candidate_sigmas=trace.candidate_sigmas,
            acquired=tuple(acquired),
            executed=executed,
            replay=tuple(replay),
            A_eigenvalues=eigenvalues,
            A_logdet=float(logdet),
            A_observation_count=len(visible),
            query_count=flow_query_count,
            positive_count=flow_positive_count,
            backup_query_count=backup_query_count,
            backup_positive_count=backup_positive_count,
            verifier_calls=verifier_calls,
            fallback_count=fallback_count,
            cache_hits=cache_hits,
            audit=_audit_snapshot(audit_by_frame[frame_index]),
            audit_provenance=audit_provenance,
            proximal=_proximal_snapshot(proximal_value),
            expansion_temperature=1.0,
            audit_temperature=1.0,
            replay_eligibility=replay_eligibility,
            runtime_safety_claim=runtime_safety_claim,
            method_label=method_label,
            acquisition_mode=acquisition_mode,
            progress_ranking=progress_ranking,
        ))

    if visible_hashes != set(by_hash):
        raise RuntimeError("not every verification record reached a visualization frame")
    np.testing.assert_allclose(matrix, store.uncertainty.A, rtol=1e-10, atol=1e-11)
    return tuple(frames)


def save_visualization_data(
    path: str | Path,
    scene: SceneSnapshot,
    frames: Sequence[ExpansionVizFrame],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_rows = tuple(frames)
    if not frame_rows:
        raise ValueError("cannot save empty visualization data")
    replay_modes = {frame.replay_eligibility for frame in frame_rows}
    safety_claims = {frame.runtime_safety_claim for frame in frame_rows}
    acquisition_modes = {frame.acquisition_mode for frame in frame_rows}
    progress_modes = {frame.progress_ranking for frame in frame_rows}
    audit_provenance = {frame.audit_provenance for frame in frame_rows}
    if (
        len(replay_modes) != 1
        or len(safety_claims) != 1
        or len(acquisition_modes) != 1
        or len(progress_modes) != 1
        or len(audit_provenance) != 1
    ):
        raise ValueError("one visualization file cannot mix replay/safety semantics")
    payload = {
        "format": "planned_window_afe_expansion_viz",
        "version": FORMAT_VERSION,
        "scene": scene.to_dict(),
        "frames": [frame.to_dict() for frame in frame_rows],
        "metadata": dict(metadata or {}),
        "semantics": {
            "sigma_colormap": SIGMA_CMAP,
            "gamma_colormap": GAMMA_CMAP_NAME,
            "acquisition_temperature": 1.0,
            "audit_temperature": 1.0,
            "replay_distribution": (
                "uniform_positive_flow_query_ledger"
                if frame_rows[0].replay_eligibility == "full_safe"
                else "uniform_strict_bounds_offline_view"
            ),
            "replay_eligibility": frame_rows[0].replay_eligibility,
            "runtime_safety_claim": frame_rows[0].runtime_safety_claim,
            "acquisition_mode": frame_rows[0].acquisition_mode,
            "progress_ranking": frame_rows[0].progress_ranking,
            "query_acceptance_denominator": "FLOW verifier queries only",
            "backup_query_accounting": "separate; still included in cumulative A_n",
            "audit_provenance": asdict(frame_rows[0].audit_provenance),
        },
    }
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return destination


def load_visualization_data(
    path: str | Path,
) -> tuple[SceneSnapshot, tuple[ExpansionVizFrame, ...], dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text())
    if payload.get("format") != "planned_window_afe_expansion_viz":
        raise ValueError("input is not planned-window AFE visualization data")
    if int(payload.get("version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported expansion visualization format version")
    scene = SceneSnapshot.from_dict(payload["scene"])
    frames = tuple(ExpansionVizFrame.from_dict(row) for row in payload["frames"])
    if not frames:
        raise ValueError("saved visualization data has no frames")
    semantics = payload.get("semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("saved visualization data lacks explicit semantics")
    expected_replay = (
        "uniform_positive_flow_query_ledger"
        if frames[0].replay_eligibility == "full_safe"
        else "uniform_strict_bounds_offline_view"
    )
    expected = {
        "acquisition_temperature": 1.0,
        "audit_temperature": 1.0,
        "replay_distribution": expected_replay,
        "replay_eligibility": frames[0].replay_eligibility,
        "runtime_safety_claim": frames[0].runtime_safety_claim,
        "acquisition_mode": frames[0].acquisition_mode,
        "progress_ranking": frames[0].progress_ranking,
        "query_acceptance_denominator": "FLOW verifier queries only",
        "backup_query_accounting": "separate; still included in cumulative A_n",
        "audit_provenance": asdict(frames[0].audit_provenance),
    }
    for key, value in expected.items():
        if semantics.get(key) != value:
            raise ValueError(f"saved visualization semantics disagree on {key!r}")
    if any(
        frame.replay_eligibility != frames[0].replay_eligibility
        or frame.runtime_safety_claim != frames[0].runtime_safety_claim
        or frame.acquisition_mode != frames[0].acquisition_mode
        or frame.progress_ranking != frames[0].progress_ranking
        or frame.audit_provenance != frames[0].audit_provenance
        for frame in frames[1:]
    ):
        raise ValueError("saved visualization frames mix scientific semantics")
    return scene, frames, dict(payload.get("metadata", {}))


def _gamma_color(gamma: float):
    value = float(gamma)
    nearest = int(np.argmin(np.abs(GAMMA_LEVELS - value)))
    if np.isclose(GAMMA_LEVELS[nearest], value, rtol=0.0, atol=1.0e-8):
        return tuple(_GAMMA_RGBA[nearest])
    # Non-canonical values are still deterministic, but primary experiments
    # use only the seven discrete levels above.
    fractional_index = np.interp(value, GAMMA_LEVELS, np.arange(len(GAMMA_LEVELS)))
    low = int(np.floor(fractional_index))
    high = min(low + 1, len(GAMMA_LEVELS) - 1)
    mix = fractional_index - low
    return tuple((1.0 - mix) * _GAMMA_RGBA[low] + mix * _GAMMA_RGBA[high])


def _draw_scene(axis, scene: SceneSnapshot, title: str) -> None:
    axis.set_facecolor("#f8f7f4")
    for obstacle in scene.obstacles:
        radius = float(obstacle[2] + scene.robot_radius)
        giant = obstacle[2] >= 0.9
        axis.add_patch(Circle(
            obstacle[:2], radius,
            facecolor="#686868" if giant else "#c8c8c8",
            edgecolor="#b2182b" if giant else "none",
            linewidth=1.4 if giant else 0.0,
            zorder=1,
        ))
    axis.plot(*scene.start, "ks", ms=6.5, zorder=9)
    axis.plot(*scene.goal, "*", color="gold", mec="black", ms=15, zorder=9)
    xmin, xmax, ymin, ymax = scene.bounds
    pad = 0.32
    axis.set_xlim(xmin - pad, xmax + pad)
    axis.set_ylim(ymin - pad, ymax + pad)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(math.ceil(xmin), math.floor(xmax) + 1))
    axis.set_yticks(np.arange(math.ceil(ymin), math.floor(ymax) + 1))
    axis.grid(color="#ffffff", lw=0.7, alpha=0.55, zorder=0)
    axis.set_title(title, fontsize=14, pad=8)


def _acquisition_title(frame: ExpansionVizFrame) -> str:
    if frame.acquisition_mode == "afe":
        return "AFE acquisition — uncertainty-tilted ordinary T=1 flow candidates"
    return "−AFE acquisition — uniform ordinary T=1 flow candidates; σ diagnostic only"


def _selection_description(frame: ExpansionVizFrame) -> str:
    return (
        "best progress among eligible plans"
        if frame.progress_ranking
        else "first eligible plan in acquisition order (−Progress)"
    )


def _render_frame(
    frame: ExpansionVizFrame,
    history: Sequence[ExpansionVizFrame],
    scene: SceneSnapshot,
    output: Path,
    sigma_norm: Normalize,
    *,
    dpi: int,
) -> None:
    figure = plt.figure(figsize=(24, 11), constrained_layout=True)
    figure.patch.set_facecolor("white")
    grid = figure.add_gridspec(3, 6, height_ratios=(1.0, 1.0, 0.82))
    acquisition_axis = figure.add_subplot(grid[:2, :3])
    replay_axis = figure.add_subplot(grid[:2, 3:])
    compact = grid[2, :].subgridspec(1, 5, wspace=0.33)
    matrix_axis, validity_axis, counts_axis, proximal_axis, coverage_axis = [
        figure.add_subplot(compact[0, index]) for index in range(5)
    ]

    _draw_scene(acquisition_axis, scene, _acquisition_title(frame))
    sigma_map = plt.get_cmap(SIGMA_CMAP)
    for path, sigma in zip(frame.candidate_paths, frame.candidate_sigmas):
        acquisition_axis.plot(
            path[:, 0], path[:, 1], color=sigma_map(sigma_norm(float(sigma))),
            lw=1.0, alpha=0.42, zorder=3,
        )
    for plan in frame.acquired:
        acquisition_axis.plot(
            plan.path[:, 0], plan.path[:, 1], color="#202020", lw=0.8,
            ls="--", alpha=0.56, zorder=4,
        )
        marker = "o" if plan.safe else "x"
        color = "#159447" if plan.safe else "#d7301f"
        acquisition_axis.plot(
            plan.path[-1, 0], plan.path[-1, 1], marker=marker,
            color=color, mec="#083d1c" if plan.safe else color,
            ms=6.5, mew=1.8, zorder=7,
        )
    if frame.executed is not None:
        plan = frame.executed
        linestyle = "--" if plan.source == "safemppi_backup" else "-"
        if not plan.safe:
            linestyle = ":"
        acquisition_axis.plot(
            plan.path[:, 0], plan.path[:, 1], color="black", lw=5.0,
            alpha=0.85, ls=linestyle, zorder=7,
        )
        acquisition_axis.plot(
            plan.path[:, 0], plan.path[:, 1], color=_gamma_color(plan.gamma),
            lw=3.0, ls=linestyle, zorder=8,
        )
        if not plan.safe:
            acquisition_axis.plot(
                plan.path[-1, 0], plan.path[-1, 1], marker="X",
                color="#d7301f", mec="black", ms=9, mew=0.8, zorder=10,
            )
    colorbar = figure.colorbar(
        ScalarMappable(norm=sigma_norm, cmap=SIGMA_CMAP),
        ax=acquisition_axis, fraction=0.035, pad=0.02,
    )
    colorbar.set_label("fixed-feature linear σ (acquisition only)")
    source = frame.executed.source if frame.executed else "fail closed"
    if frame.executed is not None and not frame.executed.safe:
        source += " (OFFLINE; actual SOCP rejected)"
    acquisition_axis.text(
        0.012, 0.015,
        f"queried this event: {len(frame.acquired)}  |  executed: {source}\n"
        f"{'Gibbs' if frame.acquisition_mode == 'afe' else 'uniform'} candidates: "
        f"{len(frame.candidate_paths)}  |  γ={frame.gamma:g}\n"
        f"selection: {_selection_description(frame)}",
        transform=acquisition_axis.transAxes, va="bottom", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#aaaaaa"},
        zorder=10,
    )
    execution_label = (
        "exact certified executed plan"
        if frame.runtime_safety_claim
        else "offline selected plan (not a safety claim)"
    )
    acquisition_axis.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="#159447", label="full verifier accepted"),
        Line2D([], [], marker="x", ls="none", color="#d7301f", label="full verifier rejected"),
        Line2D([], [], color="black", lw=3.2, label=execution_label),
        Line2D(
            [], [], color="black", lw=2.4, ls="--",
            label=(
                "certified backup when used"
                if frame.runtime_safety_claim
                else "offline backup proposal when used"
            ),
        ),
    ], loc="upper left", fontsize=8, framealpha=0.9)

    replay_title = (
        RENDER_LABELS["replay"]
        if frame.replay_eligibility == "full_safe"
        else "OFFLINE -SOCP replay — exact strict-bounds training targets"
    )
    _draw_scene(replay_axis, scene, f"{replay_title} (n={len(frame.replay)})")
    replay_alpha = max(0.10, min(0.72, 4.0 / math.sqrt(max(len(frame.replay), 1))))
    for plan in frame.replay:
        replay_axis.plot(
            plan.path[:, 0], plan.path[:, 1],
            color=_gamma_color(plan.gamma), lw=1.3, alpha=replay_alpha,
            ls="-" if plan.safe else "--", zorder=3,
        )
        replay_axis.plot(
            plan.path[0, 0], plan.path[0, 1], ".", color="#222222",
            ms=2.4, alpha=0.5, zorder=4,
        )
        if not plan.safe:
            replay_axis.plot(
                plan.path[-1, 0], plan.path[-1, 1], marker="x",
                color="#d7301f", ms=4.5, mew=1.0, alpha=0.8, zorder=5,
            )
    gamma_bar = figure.colorbar(
        ScalarMappable(norm=GAMMA_NORM, cmap=GAMMA_CMAP),
        ax=replay_axis, fraction=0.035, pad=0.02,
    )
    gamma_bar.set_label("safety level γ")
    gamma_bar.set_ticks(GAMMA_LEVELS)
    if frame.proximal is None:
        replay_axis.text(
            0.5, 0.5, "No model update at this event",
            transform=replay_axis.transAxes, ha="center", va="center",
            fontsize=13, color="#555555",
            bbox={"facecolor": "white", "alpha": 0.83, "edgecolor": "#bbbbbb"},
        )
    else:
        pass_description = (
            "every positive ledger row exactly once per objective pass"
            if frame.replay_eligibility == "full_safe"
            else "every immutable strict-bounds row once per objective pass; red × = actual SOCP failure"
        )
        replay_axis.text(
            0.012, 0.015,
            f"{pass_description}\n"
            f"minimum pass coverage: {100.0 * frame.proximal.positive_coverage:.1f}%",
            transform=replay_axis.transAxes, va="bottom", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#aaaaaa"},
        )

    # A matrix: observation count plus complete eigen-spectrum.
    spectrum = np.sort(frame.A_eigenvalues)[::-1]
    matrix_axis.semilogy(np.arange(1, len(spectrum) + 1), spectrum, "o-", ms=2.6, lw=1.1)
    matrix_axis.set_title(RENDER_LABELS["matrix"], fontsize=10)
    matrix_axis.set_xlabel("eigenvalue rank")
    matrix_axis.set_ylabel("eigenvalue")
    matrix_axis.grid(alpha=0.25)
    matrix_axis.text(
        0.97, 0.96, f"n={frame.A_observation_count}\nlog det={frame.A_logdet:.2f}",
        transform=matrix_axis.transAxes, ha="right", va="top", fontsize=8,
    )

    x = np.arange(len(history))
    acceptance = np.asarray([row.query_acceptance for row in history])
    audit_v = np.full(len(history), np.nan)
    audit_p = np.full(len(history), np.nan)
    for index, row in enumerate(history):
        if row.audit:
            audit_v[index] = np.mean([metric.validity for metric in row.audit])
            audit_p[index] = np.mean([metric.progress_validity for metric in row.audit])
    validity_axis.plot(
        x, acceptance, "o-", color="#4c78a8", ms=3, lw=1.3,
        label="FLOW query accept",
    )
    validity_axis.plot(x, audit_v, "s-", color="#159447", ms=3, lw=1.3, label="held-out V")
    validity_axis.plot(x, audit_p, "^-", color="#e68613", ms=3, lw=1.3, label="held-out Vprog")
    validity_axis.set_ylim(-0.04, 1.04)
    validity_axis.set_title(RENDER_LABELS["validity"], fontsize=10)
    validity_axis.set_xlabel("event")
    validity_axis.set_ylabel("rate")
    validity_axis.grid(alpha=0.25)
    validity_axis.legend(fontsize=7, loc="best")

    verifier = np.asarray([row.verifier_calls for row in history])
    fallback = np.asarray([row.fallback_count for row in history])
    rejected = np.asarray([row.query_count - row.positive_count for row in history])
    backup_queries = np.asarray([row.backup_query_count for row in history])
    counts_axis.plot(x, verifier, color="#4c78a8", lw=1.5, label="all verifier calls")
    counts_axis.plot(x, rejected, color="#d7301f", lw=1.3, label="rejected FLOW queries")
    counts_axis.plot(x, backup_queries, color="#f28e2b", lw=1.3, label="backup queries")
    counts_axis.plot(x, fallback, color="#7b3294", lw=1.3, label="fallback steps")
    counts_axis.set_title(RENDER_LABELS["counts"], fontsize=10)
    counts_axis.set_xlabel("event")
    counts_axis.set_ylabel("cumulative count")
    counts_axis.grid(alpha=0.25)
    counts_axis.legend(fontsize=7, loc="best")

    objective = np.asarray([
        row.proximal.objective if row.proximal is not None else np.nan for row in history
    ])
    update = np.asarray([
        row.proximal.update_norm if row.proximal is not None else np.nan for row in history
    ])
    proximal_axis.plot(x, objective, "o-", color="#f28e2b", ms=3, lw=1.3, label="objective")
    proximal_twin = proximal_axis.twinx()
    proximal_twin.plot(x, update, "s-", color="#4c78a8", ms=3, lw=1.3, label="update norm")
    proximal_axis.set_title(RENDER_LABELS["proximal"], fontsize=10)
    proximal_axis.set_xlabel("event")
    proximal_axis.set_ylabel("objective", color="#f28e2b")
    proximal_twin.set_ylabel("‖θ−θ₀‖", color="#4c78a8")
    proximal_axis.grid(alpha=0.25)
    if frame.proximal is not None:
        proximal_axis.text(
            0.02, 0.96,
            f"steps={frame.proximal.optimizer_steps}\n{frame.proximal.stopping_reason}",
            transform=proximal_axis.transAxes, va="top", fontsize=7,
        )

    gammas = sorted({plan.gamma for plan in frame.replay})
    counts = [sum(np.isclose(plan.gamma, gamma) for plan in frame.replay) for gamma in gammas]
    if gammas:
        positions = np.arange(len(gammas))
        coverage_axis.bar(positions, counts, color=[_gamma_color(gamma) for gamma in gammas])
        coverage_axis.set_xticks(positions, [f"{gamma:g}" for gamma in gammas], rotation=45)
    else:
        coverage_axis.text(0.5, 0.5, "no update", ha="center", va="center", transform=coverage_axis.transAxes)
    coverage_axis.set_title(RENDER_LABELS["coverage"], fontsize=10)
    coverage_axis.set_xlabel("γ")
    coverage_axis.set_ylabel("unique training targets")
    coverage_axis.grid(axis="y", alpha=0.25)

    safety_banner = (
        ""
        if frame.runtime_safety_claim
        else "  |  NO RUNTIME SAFETY CLAIM — OFFLINE -SOCP CONTROL"
    )
    figure.suptitle(
        f"Planned-window Active Flow Expansion [{frame.method_label}]"
        f"  |  round {frame.round_index}, control step {frame.control_step}, event {frame.frame_index}"
        f"  |  acquisition T=1, independent audit T=1{safety_banner}",
        fontsize=18,
    )
    if not frame.runtime_safety_claim:
        figure.text(
            0.5, 0.958,
            "NO RUNTIME SAFETY CLAIM — OFFLINE -SOCP ABLATION",
            ha="center", va="top", fontsize=15, fontweight="bold", color="white",
            bbox={"facecolor": "#b2182b", "edgecolor": "#67001f", "pad": 5.0},
            zorder=100,
        )
    if frame.replay_eligibility == "full_safe":
        progress_text = (
            "progress ranks accepted plans only"
            if frame.progress_ranking
            else "−Progress: first accepted plan is selected; progress is logged only"
        )
        footer = (
            "generated plan = full-verifier query = buffered object = CFM target; "
            + progress_text
        )
    else:
        footer = (
            "OFFLINE ABLATION: generated/query/buffer identities stay exact; "
            "training eligibility is strict bounds, while actual SOCP labels remain unchanged"
        )
    figure.text(
        0.5, 0.003,
        footer,
        ha="center", fontsize=9, color="#444444",
    )
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def render_expansion_video(
    scene: SceneSnapshot,
    frames: Sequence[ExpansionVizFrame],
    output_mp4: str | Path,
    *,
    preview_png: str | Path | None = None,
    fps: int = 2,
    seconds_per_event: float = 1.0,
    dpi: int = 90,
    keep_frames: bool = False,
) -> dict[str, Any]:
    """Render MP4, final-frame PNG, JSONL frame log, and manifest."""

    rows = tuple(frames)
    if not rows:
        raise ValueError("at least one expansion frame is required")
    replay_modes = {row.replay_eligibility for row in rows}
    safety_claims = {row.runtime_safety_claim for row in rows}
    acquisition_modes = {row.acquisition_mode for row in rows}
    progress_modes = {row.progress_ranking for row in rows}
    audit_provenance = {row.audit_provenance for row in rows}
    if (
        len(replay_modes) != 1
        or len(safety_claims) != 1
        or len(acquisition_modes) != 1
        or len(progress_modes) != 1
        or len(audit_provenance) != 1
    ):
        raise ValueError("one video cannot mix replay or runtime-safety semantics")
    if fps <= 0 or seconds_per_event <= 0.0 or dpi <= 0:
        raise ValueError("fps, seconds_per_event, and dpi must be positive")
    output = Path(output_mp4)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = Path(preview_png) if preview_png else output.with_name(f"{output.stem}_preview.png")
    preview.parent.mkdir(parents=True, exist_ok=True)
    frame_directory = output.with_name(f"{output.stem}_frames")
    frame_directory.mkdir(parents=True, exist_ok=True)
    for old in frame_directory.glob("frame_*.png"):
        old.unlink()

    all_sigma = np.concatenate([frame.candidate_sigmas for frame in rows])
    sigma_min = float(np.min(all_sigma))
    sigma_max = float(np.max(all_sigma))
    if math.isclose(sigma_min, sigma_max):
        delta = max(1.0e-6, abs(sigma_min) * 1.0e-3)
        sigma_min -= delta
        sigma_max += delta
    sigma_norm = Normalize(vmin=sigma_min, vmax=sigma_max)
    rendered: list[Path] = []
    for index, frame in enumerate(rows):
        destination = frame_directory / f"frame_{index:06d}.png"
        _render_frame(frame, rows[: index + 1], scene, destination, sigma_norm, dpi=dpi)
        rendered.append(destination)
    shutil.copyfile(rendered[-1], preview)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the expansion MP4")
    input_rate = 1.0 / float(seconds_per_event)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{input_rate:.12g}",
        "-i", str(frame_directory / "frame_%06d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-r", str(int(fps)), "-pix_fmt", "yuv420p", "-c:v", "libx264",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    frame_log = output.with_name(f"{output.stem}_frames.jsonl")
    with frame_log.open("w") as stream:
        for frame in rows:
            stream.write(json.dumps({
                "frame_index": frame.frame_index,
                "round_index": frame.round_index,
                "control_step": frame.control_step,
                "gamma": frame.gamma,
                "candidate_count": len(frame.candidate_paths),
                "acquired_hashes": [plan.query_hash for plan in frame.acquired],
                "accepted_hashes": [plan.query_hash for plan in frame.acquired if plan.safe],
                "rejected_hashes": [plan.query_hash for plan in frame.acquired if not plan.safe],
                "executed_hash": frame.executed.query_hash if frame.executed else None,
                "executed_source": frame.executed.source if frame.executed else None,
                "replay_hashes": [plan.query_hash for plan in frame.replay],
                "A_observation_count": frame.A_observation_count,
                "A_logdet": frame.A_logdet,
                "query_acceptance": frame.query_acceptance,
                "query_acceptance_scope": "FLOW_only",
                "flow_query_count": frame.query_count,
                "flow_positive_count": frame.positive_count,
                "backup_query_count": frame.backup_query_count,
                "backup_positive_count": frame.backup_positive_count,
                "backup_acceptance": frame.backup_acceptance,
                "acquisition_mode": frame.acquisition_mode,
                "progress_ranking": frame.progress_ranking,
                "audit_provenance": asdict(frame.audit_provenance),
                "audit": [asdict(metric) for metric in frame.audit],
                "proximal": asdict(frame.proximal) if frame.proximal else None,
            }, separators=(",", ":")) + "\n")

    probe: dict[str, Any] | None = None
    if shutil.which("ffprobe") is not None:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=width,height,avg_frame_rate,nb_frames",
            "-of", "json", str(output),
        ], check=True, capture_output=True, text=True)
        probe = json.loads(result.stdout)
    manifest = {
        "status": "PASS",
        "method": "planned_window_afe",
        "output_mp4": str(output.resolve()),
        "preview_png": str(preview.resolve()),
        "frame_log": str(frame_log.resolve()),
        "source_event_count": len(rows),
        "fps": int(fps),
        "seconds_per_event": float(seconds_per_event),
        "temperatures": {"expansion": 1.0, "independent_audit": 1.0},
        "colormaps": {"sigma": SIGMA_CMAP, "gamma": GAMMA_CMAP_NAME},
        "replay_distribution": (
            "uniform_positive_flow_query_ledger"
            if rows[-1].replay_eligibility == "full_safe"
            else "uniform_strict_bounds_offline_view"
        ),
        "replay_eligibility": rows[-1].replay_eligibility,
        "runtime_safety_claim": rows[-1].runtime_safety_claim,
        "method_label": rows[-1].method_label,
        "acquisition_mode": rows[-1].acquisition_mode,
        "progress_ranking": rows[-1].progress_ranking,
        "query_acceptance_scope": "FLOW_only",
        "audit_provenance": asdict(rows[-1].audit_provenance),
        "no_runtime_safety_claim_banner": not rows[-1].runtime_safety_claim,
        "labels": RENDER_LABELS,
        "final_frame": {
            "query_count": rows[-1].query_count,
            "positive_count": rows[-1].positive_count,
            "query_acceptance": rows[-1].query_acceptance,
            "backup_query_count": rows[-1].backup_query_count,
            "backup_positive_count": rows[-1].backup_positive_count,
            "backup_acceptance": rows[-1].backup_acceptance,
            "fallback_count": rows[-1].fallback_count,
            "actual_socp_failures_in_training_replay": sum(
                not plan.safe for plan in rows[-1].replay
            ),
            "audit": [asdict(metric) for metric in rows[-1].audit],
        },
        "sha256": {"mp4": _sha256(output), "preview": _sha256(preview)},
        "ffprobe": probe,
    }
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest["manifest"] = str(manifest_path.resolve())
    if not keep_frames:
        shutil.rmtree(frame_directory)
    else:
        manifest["rendered_frames"] = str(frame_directory.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def render_from_controller_data(
    traces: Sequence[ControlStepTrace],
    store: VerificationStore,
    proximal_telemetry: Any,
    env,
    output_mp4: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience entry point for live expansion code."""

    build_keys = {
        "audit_results", "replay_query_hashes", "round_indices", "dynamics",
        "expansion_temperature", "audit_temperature", "replay_eligibility",
        "runtime_safety_claim", "method_label", "acquisition_mode",
        "progress_ranking",
    }
    build_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in build_keys}
    frames = build_expansion_frames(
        traces, store, proximal_telemetry, **build_kwargs,
    )
    return render_expansion_video(
        SceneSnapshot.from_environment(env), frames, output_mp4, **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="saved visualization JSON")
    parser.add_argument("--output", type=Path, required=True, help="output MP4")
    parser.add_argument("--preview", type=Path, default=None, help="optional preview PNG")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seconds-per-event", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--keep-frames", action="store_true")
    arguments = parser.parse_args()
    scene, frames, _metadata = load_visualization_data(arguments.input)
    manifest = render_expansion_video(
        scene, frames, arguments.output,
        preview_png=arguments.preview,
        fps=arguments.fps,
        seconds_per_event=arguments.seconds_per_event,
        dpi=arguments.dpi,
        keep_frames=arguments.keep_frames,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
