"""Numerical solver for the planned-window AFE proximal update.

The method-level objective is

    mean_{(c, U) in D+} loss_CFM(theta; c, U)
        + ||theta - theta_round_start||^2 / (2 * eta).

This module intentionally knows nothing about curricula, frontier classes,
uncertainty scores, demonstrations, or negative examples.  It accepts ledger
records, retains only records carrying a positive verifier label, and uses
every one with equal weight on every full objective pass.  ``batch_size`` is a
gradient-accumulation microbatch size, not a replay quota.  The optimizer step
count is a numerical outcome reported in telemetry, not a fixed part of the
method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

import torch
from torch import nn


class CFMLossFunction(Protocol):
    """Loss adapter expected by :func:`solve_proximal_update`.

    ``batch`` is a tuple of the original positive ledger records.  Implementors
    should return the *mean* CFM loss for that batch.  A local seeded generator
    is supplied so flow time/noise draws need not touch global RNG state.
    """

    def __call__(
        self,
        model: nn.Module,
        batch: Sequence[Any],
        *,
        generator: torch.Generator,
    ) -> torch.Tensor: ...


OptimizerFactory = Callable[..., torch.optim.Optimizer]


@dataclass(frozen=True)
class ProximalConfig:
    """Declared numerical tolerances for one expansion-round update."""

    eta: float
    learning_rate: float
    batch_size: int
    max_steps: int
    update_norm_limit: float
    min_steps: int = 1
    relative_loss_tolerance: float | None = 1.0e-4
    gradient_tolerance: float | None = 1.0e-6
    tolerance_patience: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.min_steps < 0 or self.min_steps > self.max_steps:
            raise ValueError("min_steps must lie in [0, max_steps]")
        if not math.isfinite(self.update_norm_limit) or self.update_norm_limit <= 0.0:
            raise ValueError("update_norm_limit must be finite and positive")
        if self.relative_loss_tolerance is not None:
            if (
                not math.isfinite(self.relative_loss_tolerance)
                or self.relative_loss_tolerance < 0.0
            ):
                raise ValueError("relative_loss_tolerance must be nonnegative")
        if self.gradient_tolerance is not None:
            if (
                not math.isfinite(self.gradient_tolerance)
                or self.gradient_tolerance < 0.0
            ):
                raise ValueError("gradient_tolerance must be nonnegative")
        if self.tolerance_patience <= 0:
            raise ValueError("tolerance_patience must be positive")
        if self.relative_loss_tolerance is None and self.gradient_tolerance is None:
            # A max-step/update-bound-only solve is intentional and supported.
            return


@dataclass(frozen=True)
class ProximalStepTelemetry:
    """One objective/gradient evaluation and, usually, one optimizer step."""

    evaluation: int
    optimizer_step: int
    microbatch_count: int
    microbatch_sizes: tuple[int, ...]
    record_order_sha256: str
    unique_record_count: int
    positive_coverage: float
    cfm_loss: float
    proximal_penalty: float
    objective: float
    relative_objective_change: float | None
    gradient_norm: float
    update_norm: float
    projected_to_update_bound: bool


@dataclass(frozen=True)
class ProximalUpdateResult:
    """Complete, JSON-friendly telemetry for one numerical solve."""

    positive_count: int
    total_record_count: int
    trainable_parameter_count: int
    optimizer_steps: int
    objective_evaluations: int
    stopping_reason: str
    converged: bool
    seed: int
    eta: float
    learning_rate: float
    requested_batch_size: int
    max_steps: int
    min_steps: int
    update_norm_limit: float
    final_update_norm: float
    sampling: str
    objective_randomness: str
    trace: tuple[ProximalStepTelemetry, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return telemetry composed only of JSON-compatible containers."""

        return asdict(self)


def _field(record: Any, names: Sequence[str]) -> tuple[bool, Any]:
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return True, record[name]
    for name in names:
        if hasattr(record, name):
            return True, getattr(record, name)
    return False, None


def _positive_label(record: Any) -> bool:
    """Read a verifier label without accepting an unlabeled training row."""

    # ``UniformPositiveView`` exposes identity-checked ReplayItem objects rather
    # than redundantly carrying ``safe=True``.  Their two matching hashes are a
    # structural marker for that already-filtered schema; revalidate before use.
    got_source_hash, _ = _field(record, ("source_query_hash",))
    got_target_hash, _ = _field(record, ("training_target_hash",))
    if got_source_hash and got_target_hash and hasattr(record, "validate_identity"):
        record.validate_identity()
        return True

    # Some verifier schemas retain the two required predicates rather than a
    # redundant combined label.  Prefer their conjunction when both exist so a
    # stale convenience label can never admit a failed certificate.
    got_bounds, strict_bounds = _field(
        record, ("strict_bounds", "bounds_ok", "in_bounds")
    )
    got_socp, socp = _field(
        record, ("socp_success", "socp_ok", "socp_certified", "certified")
    )
    if got_bounds and got_socp:
        return bool(strict_bounds) and bool(socp)

    found, value = _field(record, ("y", "safe", "is_safe", "verifier_safe"))
    if found:
        return bool(value)

    found, verification = _field(record, ("verification", "verifier_result"))
    if found:
        return _positive_label(verification)

    raise ValueError(
        "each replay record must carry y/safe or strict_bounds and socp_success"
    )


def _materialize_records(records: Iterable[Any] | Any) -> list[Any]:
    """Accept an iterable or a ledger exposing a conventional record view."""

    if hasattr(records, "records"):
        view = records.records
        records = view() if callable(view) else view
    return list(records)


def _record_order_sha256(indices: Sequence[int]) -> str:
    """Compact, exact evidence for one uniformly shuffled full-ledger pass."""

    digest = hashlib.sha256(b"afe-uniform-positive-order-v1\x00")
    digest.update(len(indices).to_bytes(8, byteorder="little", signed=False))
    for index in indices:
        value = int(index)
        if value < 0:
            raise ValueError("record order cannot contain a negative index")
        digest.update(value.to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _loss_accepts_generator(loss_fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(loss_fn)
    except (TypeError, ValueError):
        return True
    return "generator" in signature.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


def _extract_loss_tensor(raw: Any) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw
    if isinstance(raw, Mapping):
        for key in ("loss", "cfm_loss"):
            if key in raw and isinstance(raw[key], torch.Tensor):
                return raw[key]
    if isinstance(raw, (tuple, list)) and raw and isinstance(raw[0], torch.Tensor):
        return raw[0]
    raise TypeError("cfm_loss_fn must return a scalar Tensor (or loss-first tuple/mapping)")


def _make_loss_generator(parameters: Sequence[nn.Parameter], seed: int) -> torch.Generator:
    device = parameters[0].device
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):  # pragma: no cover - exotic torch devices
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _squared_update(
    parameters: Sequence[nn.Parameter], reference_values: Sequence[torch.Tensor]
) -> torch.Tensor:
    terms = [
        torch.sum((parameter - reference) ** 2)
        for parameter, reference in zip(parameters, reference_values)
    ]
    # ``parameters`` is nonempty at every call site.
    return torch.stack(terms).sum()


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().double().pow(2).sum()
    return float(torch.sqrt(squared).cpu())


def _update_norm(
    parameters: Sequence[nn.Parameter],
    reference_values: Sequence[torch.Tensor],
) -> float:
    with torch.no_grad():
        return float(
            torch.sqrt(
                _squared_update(parameters, reference_values).double()
            ).cpu()
        )


def _restore(parameters: Sequence[nn.Parameter], values: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            parameter.copy_(value)


def _project_update(
    parameters: Sequence[nn.Parameter],
    reference_values: Sequence[torch.Tensor],
    limit: float,
) -> float:
    norm = _update_norm(parameters, reference_values)
    if norm <= limit:
        return norm
    scale = limit / norm
    with torch.no_grad():
        for parameter, reference in zip(parameters, reference_values):
            parameter.copy_(reference + scale * (parameter - reference))
    # Return the measured value rather than assuming floating-point projection
    # lands exactly on ``limit``.
    return _update_norm(parameters, reference_values)


class _UniformFullPass:
    """All rows exactly once per seeded, reshuffled objective pass."""

    def __init__(self, size: int, batch_size: int, generator: torch.Generator) -> None:
        self.size = size
        self.batch_size = min(size, batch_size)
        self.generator = generator

    def batches(self) -> Iterator[list[int]]:
        permutation = torch.randperm(self.size, generator=self.generator)
        for start in range(0, self.size, self.batch_size):
            yield permutation[start : start + self.batch_size].tolist()


def solve_proximal_update(
    model: nn.Module,
    records: Iterable[Any] | Any,
    cfm_loss_fn: CFMLossFunction | Callable[..., Any],
    config: ProximalConfig,
    *,
    optimizer_factory: OptimizerFactory | None = None,
) -> ProximalUpdateResult:
    """Optimize one bounded proximal CFM update over uniform positive replay.

    The proximal reference theta_n is captured at function entry and remains
    fixed for the whole solve. Every positive row contributes with equal weight before *each*
    optimizer step; shuffling and microbatching only bound memory.  No record
    metadata other than the deterministic verifier label is inspected.  A
    zero-positive call is an exact parameter and optimizer no-op.
    """

    all_records = _materialize_records(records)
    indexed_positives = [
        (index, record)
        for index, record in enumerate(all_records)
        if _positive_label(record)
    ]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_count = sum(parameter.numel() for parameter in parameters)

    common = dict(
        positive_count=len(indexed_positives),
        total_record_count=len(all_records),
        trainable_parameter_count=parameter_count,
        seed=config.seed,
        eta=config.eta,
        learning_rate=config.learning_rate,
        requested_batch_size=config.batch_size,
        max_steps=config.max_steps,
        min_steps=config.min_steps,
        update_norm_limit=config.update_norm_limit,
        sampling="uniform_full_positive_pass_seeded_reshuffle",
        objective_randomness=str(
            getattr(
                cfm_loss_fn,
                "objective_randomness",
                "caller_managed_unspecified",
            )
        ),
    )
    if not indexed_positives:
        return ProximalUpdateResult(
            **common,
            optimizer_steps=0,
            objective_evaluations=0,
            stopping_reason="no_positive_records",
            converged=False,
            final_update_norm=0.0,
            trace=(),
        )
    if not parameters:
        return ProximalUpdateResult(
            **common,
            optimizer_steps=0,
            objective_evaluations=0,
            stopping_reason="no_trainable_parameters",
            converged=False,
            final_update_norm=0.0,
            trace=(),
        )

    round_start = [parameter.detach().clone() for parameter in parameters]
    cpu_generator = torch.Generator().manual_seed(config.seed)
    loss_generator = _make_loss_generator(parameters, config.seed)
    batches = _UniformFullPass(
        len(indexed_positives), config.batch_size, cpu_generator
    )
    if optimizer_factory is None:
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    else:
        optimizer = optimizer_factory(parameters, lr=config.learning_rate)

    accepts_generator = _loss_accepts_generator(cfm_loss_fn)
    trace: list[ProximalStepTelemetry] = []
    previous_objective: float | None = None
    stable_evaluations = 0
    optimizer_steps = 0
    stopping_reason = "max_steps"
    converged = False
    original_training_mode = model.training
    model.train(True)

    try:
        for evaluation in range(1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            original_indices_list: list[int] = []
            microbatch_sizes: list[int] = []
            microbatch_count = 0
            cfm_value = 0.0
            finite_cfm = True
            for positive_indices in batches.batches():
                microbatch_count += 1
                original_microbatch = tuple(
                    indexed_positives[index][0] for index in positive_indices
                )
                microbatch_sizes.append(len(original_microbatch))
                original_indices_list.extend(original_microbatch)
                batch = tuple(indexed_positives[index][1] for index in positive_indices)
                if accepts_generator:
                    raw_loss = cfm_loss_fn(model, batch, generator=loss_generator)
                else:
                    raw_loss = cfm_loss_fn(model, batch)
                microbatch_loss = _extract_loss_tensor(raw_loss)
                if microbatch_loss.numel() != 1:
                    raise ValueError("cfm_loss_fn must return a scalar mean loss")
                # Weight by row count so a short final microbatch does not get
                # the same mass as a full one.  The accumulated gradient is
                # exactly the empirical mean over every positive ledger row.
                weight = len(positive_indices) / len(indexed_positives)
                weighted_loss = microbatch_loss * weight
                weighted_value = float(weighted_loss.detach().cpu())
                cfm_value += weighted_value
                if not math.isfinite(weighted_value):
                    finite_cfm = False
                    break
                weighted_loss.backward()

            original_indices = tuple(original_indices_list)
            unique_record_count = len(set(original_indices))
            record_order_sha256 = _record_order_sha256(original_indices)
            squared_update = _squared_update(parameters, round_start)
            proximal_penalty = squared_update / (2.0 * config.eta)
            proximal_value = float(proximal_penalty.detach().cpu())
            objective_value = cfm_value + proximal_value
            relative_change = None
            if previous_objective is not None:
                denominator = max(abs(previous_objective), 1.0e-12)
                relative_change = abs(objective_value - previous_objective) / denominator

            if not finite_cfm or not math.isfinite(objective_value):
                stopping_reason = "nonfinite_objective"
                trace.append(
                    ProximalStepTelemetry(
                        evaluation=evaluation,
                        optimizer_step=optimizer_steps,
                        microbatch_count=microbatch_count,
                        microbatch_sizes=tuple(microbatch_sizes),
                        record_order_sha256=record_order_sha256,
                        unique_record_count=unique_record_count,
                        positive_coverage=(
                            unique_record_count / len(indexed_positives)
                        ),
                        cfm_loss=cfm_value,
                        proximal_penalty=proximal_value,
                        objective=objective_value,
                        relative_objective_change=relative_change,
                        gradient_norm=math.nan,
                        update_norm=_update_norm(parameters, round_start),
                        projected_to_update_bound=False,
                    )
                )
                break

            # CFM gradients were accumulated microbatch-by-microbatch above;
            # The proximal-reference term is differentiated exactly once.
            proximal_penalty.backward()
            gradient_norm = _gradient_norm(parameters)
            if not math.isfinite(gradient_norm):
                stopping_reason = "nonfinite_gradient"
                trace.append(
                    ProximalStepTelemetry(
                        evaluation=evaluation,
                        optimizer_step=optimizer_steps,
                        microbatch_count=microbatch_count,
                        microbatch_sizes=tuple(microbatch_sizes),
                        record_order_sha256=record_order_sha256,
                        unique_record_count=unique_record_count,
                        positive_coverage=1.0,
                        cfm_loss=cfm_value,
                        proximal_penalty=proximal_value,
                        objective=objective_value,
                        relative_objective_change=relative_change,
                        gradient_norm=gradient_norm,
                        update_norm=_update_norm(parameters, round_start),
                        projected_to_update_bound=False,
                    )
                )
                break

            gradient_converged = (
                config.gradient_tolerance is not None
                and optimizer_steps >= config.min_steps
                and gradient_norm <= config.gradient_tolerance
            )
            if (
                config.relative_loss_tolerance is not None
                and relative_change is not None
                and optimizer_steps >= config.min_steps
                and relative_change <= config.relative_loss_tolerance
            ):
                stable_evaluations += 1
            else:
                stable_evaluations = 0
            objective_converged = (
                stable_evaluations >= config.tolerance_patience
            )

            # The objective and gradient above describe the *current* model.
            # Stop at that evaluated point.  Taking one more Adam step here
            # would make the saved parameters differ from the parameters whose
            # convergence was actually measured.
            if gradient_converged or objective_converged:
                stopping_reason = (
                    "gradient_tolerance"
                    if gradient_converged
                    else "relative_loss_tolerance"
                )
                converged = True
                trace.append(
                    ProximalStepTelemetry(
                        evaluation=evaluation,
                        optimizer_step=optimizer_steps,
                        microbatch_count=microbatch_count,
                        microbatch_sizes=tuple(microbatch_sizes),
                        record_order_sha256=record_order_sha256,
                        unique_record_count=unique_record_count,
                        positive_coverage=1.0,
                        cfm_loss=cfm_value,
                        proximal_penalty=proximal_value,
                        objective=objective_value,
                        relative_objective_change=relative_change,
                        gradient_norm=gradient_norm,
                        update_norm=_update_norm(parameters, round_start),
                        projected_to_update_bound=False,
                    )
                )
                break

            # This objective becomes the reference for the next evaluated
            # point.  It must be recorded before taking the numerical step.
            previous_objective = objective_value

            before_step = [parameter.detach().clone() for parameter in parameters]
            optimizer.step()
            optimizer_steps += 1
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                _restore(parameters, before_step)
                optimizer_steps -= 1
                stopping_reason = "nonfinite_parameter_update"
                trace.append(
                    ProximalStepTelemetry(
                        evaluation=evaluation,
                        optimizer_step=optimizer_steps,
                        microbatch_count=microbatch_count,
                        microbatch_sizes=tuple(microbatch_sizes),
                        record_order_sha256=record_order_sha256,
                        unique_record_count=unique_record_count,
                        positive_coverage=1.0,
                        cfm_loss=cfm_value,
                        proximal_penalty=proximal_value,
                        objective=objective_value,
                        relative_objective_change=relative_change,
                        gradient_norm=gradient_norm,
                        update_norm=_update_norm(parameters, round_start),
                        projected_to_update_bound=False,
                    )
                )
                break

            update_norm_before_projection = _update_norm(
                parameters, round_start
            )
            projected = update_norm_before_projection >= config.update_norm_limit
            update_norm = _project_update(
                parameters, round_start, config.update_norm_limit
            ) if projected else update_norm_before_projection
            trace.append(
                ProximalStepTelemetry(
                    evaluation=evaluation,
                    optimizer_step=optimizer_steps,
                    microbatch_count=microbatch_count,
                    microbatch_sizes=tuple(microbatch_sizes),
                    record_order_sha256=record_order_sha256,
                    unique_record_count=unique_record_count,
                    positive_coverage=1.0,
                    cfm_loss=cfm_value,
                    proximal_penalty=proximal_value,
                    objective=objective_value,
                    relative_objective_change=relative_change,
                    gradient_norm=gradient_norm,
                    update_norm=update_norm,
                    projected_to_update_bound=projected,
                )
            )

            if projected:
                stopping_reason = "update_norm_bound"
                break
    finally:
        model.train(original_training_mode)

    return ProximalUpdateResult(
        **common,
        optimizer_steps=optimizer_steps,
        objective_evaluations=len(trace),
        stopping_reason=stopping_reason,
        converged=converged,
        final_update_norm=_update_norm(parameters, round_start),
        trace=tuple(trace),
    )


__all__ = [
    "CFMLossFunction",
    "ProximalConfig",
    "ProximalStepTelemetry",
    "ProximalUpdateResult",
    "solve_proximal_update",
]
