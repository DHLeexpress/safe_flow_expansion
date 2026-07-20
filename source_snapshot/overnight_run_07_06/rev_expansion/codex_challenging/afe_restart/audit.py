"""Independent, untilted temperature-one validity audit.

The API deliberately has no acquisition ledger, replay store, or uncertainty
matrix argument.  Audit plans are sampled from a deep-snapshotted held-out
context bank and summarized in place; they have no route into AFE state.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import inspect
import math
from statistics import NormalDist
import struct
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import nn


class PlanSampler(Protocol):
    """Ordinary conditional-flow sampler used by the independent audit."""

    def __call__(
        self,
        model: nn.Module,
        context: Any,
        gamma: float,
        count: int,
        *,
        temperature: float,
        generator: torch.Generator,
    ) -> Any: ...


class FullWindowVerifier(Protocol):
    """Deterministic full-window verifier adapter.

    The result must expose ``safe``/``y`` and ``progress``/``r``.  If both
    ``strict_bounds`` and ``socp_success`` are exposed, their conjunction is
    used as the safety label in preference to a redundant combined field.
    """

    def __call__(self, context: Any, gamma: float, plan: Any) -> Any: ...


@dataclass(frozen=True)
class AuditConfig:
    plans_per_context: int
    progress_threshold: float
    seed: int = 0
    temperature: float = 1.0
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.plans_per_context <= 0:
            raise ValueError("plans_per_context must be positive")
        if self.temperature != 1.0:
            raise ValueError("independent model-validity audit must use temperature 1.0")
        if not math.isfinite(self.progress_threshold):
            raise ValueError("progress_threshold must be finite")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one")


@dataclass(frozen=True)
class BinomialInterval:
    low: float
    high: float
    confidence: float
    method: str = "wilson_conditional_plan_sampling"


@dataclass(frozen=True)
class GammaAuditMetrics:
    gamma: float
    sample_count: int
    safe_count: int
    safe_progress_count: int
    validity_mass: float
    validity_interval: BinomialInterval
    progress_validity: float
    progress_validity_interval: BinomialInterval
    mean_progress: float
    mean_safe_progress: float | None
    mode_counts: dict[str, int]
    safe_mode_coverage: int


@dataclass(frozen=True)
class AuditResult:
    context_count: int
    plans_per_context: int
    total_verifier_calls: int
    seed: int
    temperature: float
    progress_threshold: float
    context_bank_fingerprint: str
    context_bank_role: str
    sampling_distribution: str
    uncertainty_tilting: bool
    confidence_interval_scope: str
    independent_training_seed_count: int
    independent_training_seed_ci: bool
    per_gamma: tuple[GammaAuditMetrics, ...]

    def by_gamma(self) -> dict[float, GammaAuditMetrics]:
        return {metric.gamma: metric for metric in self.per_gamma}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _digest_part(hasher: Any, value: Any) -> None:
    """Hash nested numerical contexts without depending on object identity."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"torch")
        hasher.update(str(tensor.dtype).encode())
        hasher.update(repr(tuple(tensor.shape)).encode())
        # Flatten first because PyTorch's dtype-view rejects zero-dimensional
        # tensors even though their single scalar has an unambiguous byte form.
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"numpy")
        hasher.update(array.dtype.str.encode())
        hasher.update(repr(array.shape).encode())
        hasher.update(array.tobytes())
    elif is_dataclass(value) and not isinstance(value, type):
        hasher.update(b"dataclass")
        hasher.update(type(value).__qualname__.encode())
        for field in fields(value):
            hasher.update(field.name.encode())
            _digest_part(hasher, getattr(value, field.name))
    elif isinstance(value, Mapping):
        hasher.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _digest_part(hasher, key)
            _digest_part(hasher, value[key])
    elif isinstance(value, (tuple, list)):
        hasher.update(b"tuple" if isinstance(value, tuple) else b"list")
        hasher.update(struct.pack("!Q", len(value)))
        for item in value:
            _digest_part(hasher, item)
    elif isinstance(value, (str, bytes, int, bool, type(None))):
        hasher.update(type(value).__name__.encode())
        hasher.update(repr(value).encode())
    elif isinstance(value, float):
        hasher.update(b"float")
        hasher.update(struct.pack("!d", value))
    elif hasattr(value, "__dict__"):
        hasher.update(type(value).__qualname__.encode())
        _digest_part(hasher, vars(value))
    else:
        # Contexts should normally consist of mappings and numerical arrays.
        # The fallback remains deterministic for ordinary immutable scalars.
        hasher.update(type(value).__qualname__.encode())
        hasher.update(repr(value).encode())


def _fingerprint(contexts: Sequence[Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"afe-independent-context-bank-v1")
    _digest_part(hasher, tuple(contexts))
    return hasher.hexdigest()


class ImmutableContextBank(Sequence[Any]):
    """Deep-snapshotted held-out contexts returned only through fresh copies."""

    def __init__(self, contexts: Iterable[Any], *, role: str = "unspecified") -> None:
        self._contexts = tuple(copy.deepcopy(context) for context in contexts)
        if not self._contexts:
            raise ValueError("the independent audit context bank cannot be empty")
        if role not in {"unspecified", "round_monitoring", "sealed_final_test"}:
            raise ValueError(f"unsupported audit context-bank role: {role!r}")
        self._role = role
        self._fingerprint = _fingerprint(self._contexts)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def role(self) -> str:
        return self._role

    def assert_integrity(self) -> None:
        if _fingerprint(self._contexts) != self._fingerprint:
            raise RuntimeError("immutable context bank was modified")

    def __len__(self) -> int:
        return len(self._contexts)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(copy.deepcopy(item) for item in self._contexts[index])
        return copy.deepcopy(self._contexts[index])

    def __iter__(self) -> Iterator[Any]:
        for context in self._contexts:
            yield copy.deepcopy(context)


def _field(record: Any, names: Sequence[str]) -> tuple[bool, Any]:
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return True, record[name]
    for name in names:
        if hasattr(record, name):
            return True, getattr(record, name)
    return False, None


def _verification_values(result: Any) -> tuple[bool, float, str | None]:
    got_bounds, strict_bounds = _field(
        result, ("strict_bounds", "bounds_ok", "in_bounds")
    )
    got_socp, socp = _field(
        result, ("socp_success", "socp_ok", "socp_certified", "certified")
    )
    if got_bounds and got_socp:
        safe = bool(strict_bounds) and bool(socp)
    else:
        got_safe, safe_value = _field(
            result, ("safe", "y", "is_safe", "verifier_safe")
        )
        if not got_safe:
            raise ValueError(
                "full verifier result must expose safe/y or bounds and SOCP predicates"
            )
        safe = bool(safe_value)

    got_progress, progress_value = _field(
        result, ("progress", "r", "progress_value", "progress_m")
    )
    if not got_progress:
        raise ValueError("full verifier result must expose progress/r separately")
    progress = float(progress_value)
    if not math.isfinite(progress):
        raise ValueError("verifier progress must be finite")
    got_mode, mode_value = _field(result, ("mode", "route_mode", "safe_mode"))
    mode = str(mode_value) if got_mode and mode_value is not None else None
    return safe, progress, mode


def _wilson(successes: int, trials: int, confidence: float) -> BinomialInterval:
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return BinomialInterval(
        low=max(0.0, centre - half_width),
        high=min(1.0, centre + half_width),
        confidence=confidence,
    )


def _callback_accepts_generator(callback: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return True
    return "generator" in signature.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _subseed(seed: int, gamma: float, context_index: int) -> int:
    digest = hashlib.sha256(
        f"afe-audit-v1:{seed}:{gamma:.17g}:{context_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _plan_sequence(plans: Any, expected: int) -> list[Any]:
    if isinstance(plans, (torch.Tensor, np.ndarray)):
        if plans.ndim == 0:
            raise ValueError("sampler must return a batch of full plans")
        materialized = [plans[index] for index in range(plans.shape[0])]
    else:
        materialized = list(plans)
    if len(materialized) != expected:
        raise ValueError(
            f"ordinary sampler returned {len(materialized)} plans; expected {expected}"
        )
    return materialized


def run_independent_audit(
    model: nn.Module,
    context_bank: ImmutableContextBank | Iterable[Any],
    gammas: Sequence[float],
    sample_plans_fn: PlanSampler | Callable[..., Any],
    verifier_fn: FullWindowVerifier | Callable[..., Any],
    config: AuditConfig,
) -> AuditResult:
    """Estimate per-gamma model validity using ordinary T=1 flow samples.

    Every sampled plan is passed to ``verifier_fn`` exactly once.  Neither the
    plans nor verifier results are returned, inserted, or replayed.  The input
    API therefore provides no acquisition-store or ``A`` mutation channel.
    """

    bank = (
        context_bank
        if isinstance(context_bank, ImmutableContextBank)
        else ImmutableContextBank(context_bank)
    )
    bank.assert_integrity()
    gamma_values = tuple(float(gamma) for gamma in gammas)
    if not gamma_values:
        raise ValueError("at least one audit gamma is required")
    if any(not math.isfinite(gamma) for gamma in gamma_values):
        raise ValueError("audit gammas must be finite")
    if len(set(gamma_values)) != len(gamma_values):
        raise ValueError("audit gammas must be unique")

    accepts_generator = _callback_accepts_generator(sample_plans_fn)
    metrics: list[GammaAuditMetrics] = []
    original_training_mode = model.training
    model.eval()
    try:
        for gamma in gamma_values:
            safe_count = 0
            safe_progress_count = 0
            progress_values: list[float] = []
            safe_progress_values: list[float] = []
            mode_counts: dict[str, int] = {}
            for context_index in range(len(bank)):
                context = bank[context_index]
                context_before = _fingerprint((context,))
                try:
                    generator = torch.Generator(device=_model_device(model))
                except (RuntimeError, TypeError):  # pragma: no cover
                    generator = torch.Generator()
                generator.manual_seed(_subseed(config.seed, gamma, context_index))
                with torch.inference_mode():
                    if accepts_generator:
                        sampled = sample_plans_fn(
                            model,
                            context,
                            gamma,
                            config.plans_per_context,
                            temperature=1.0,
                            generator=generator,
                        )
                    else:
                        sampled = sample_plans_fn(
                            model,
                            context,
                            gamma,
                            config.plans_per_context,
                            temperature=1.0,
                        )
                if _fingerprint((context,)) != context_before:
                    raise RuntimeError("audit sampler mutated its held-out context input")
                plans = _plan_sequence(sampled, config.plans_per_context)
                for plan in plans:
                    verifier_context = bank[context_index]
                    verifier_context_before = _fingerprint((verifier_context,))
                    result = verifier_fn(verifier_context, gamma, plan)
                    if _fingerprint((verifier_context,)) != verifier_context_before:
                        raise RuntimeError("full verifier mutated its audit context input")
                    safe, progress, mode = _verification_values(result)
                    progress_values.append(progress)
                    if safe:
                        safe_count += 1
                        safe_progress_values.append(progress)
                        if mode is not None:
                            mode_counts[mode] = mode_counts.get(mode, 0) + 1
                        if progress >= config.progress_threshold:
                            safe_progress_count += 1

            sample_count = len(bank) * config.plans_per_context
            metrics.append(
                GammaAuditMetrics(
                    gamma=gamma,
                    sample_count=sample_count,
                    safe_count=safe_count,
                    safe_progress_count=safe_progress_count,
                    validity_mass=safe_count / sample_count,
                    validity_interval=_wilson(
                        safe_count, sample_count, config.confidence
                    ),
                    progress_validity=safe_progress_count / sample_count,
                    progress_validity_interval=_wilson(
                        safe_progress_count, sample_count, config.confidence
                    ),
                    mean_progress=sum(progress_values) / sample_count,
                    mean_safe_progress=(
                        sum(safe_progress_values) / len(safe_progress_values)
                        if safe_progress_values
                        else None
                    ),
                    mode_counts=dict(sorted(mode_counts.items())),
                    safe_mode_coverage=len(mode_counts),
                )
            )
    finally:
        model.train(original_training_mode)
    bank.assert_integrity()
    return AuditResult(
        context_count=len(bank),
        plans_per_context=config.plans_per_context,
        total_verifier_calls=(
            len(bank) * config.plans_per_context * len(gamma_values)
        ),
        seed=config.seed,
        temperature=1.0,
        progress_threshold=config.progress_threshold,
        context_bank_fingerprint=bank.fingerprint,
        context_bank_role=bank.role,
        sampling_distribution="ordinary_conditional_flow_iid",
        uncertainty_tilting=False,
        confidence_interval_scope=(
            "conditional_plan_sampling_wilson_on_fixed_context_bank_single_model"
        ),
        independent_training_seed_count=1,
        independent_training_seed_ci=False,
        per_gamma=tuple(metrics),
    )


__all__ = [
    "AuditConfig",
    "AuditResult",
    "BinomialInterval",
    "FullWindowVerifier",
    "GammaAuditMetrics",
    "ImmutableContextBank",
    "PlanSampler",
    "run_independent_audit",
]
