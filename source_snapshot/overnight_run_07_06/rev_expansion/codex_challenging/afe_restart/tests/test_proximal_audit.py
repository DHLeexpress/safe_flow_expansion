from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

from afe_restart.audit import AuditConfig, ImmutableContextBank, run_independent_audit
from afe_restart.proximal_update import ProximalConfig, solve_proximal_update
from afe_restart.schemas import QueryContext, ReplayItem, query_content_hash


class ScalarModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))


def mse_loss(
    model: ScalarModel,
    batch: tuple[dict[str, float], ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    del generator
    targets = torch.tensor(
        [row["target"] for row in batch], device=model.weight.device
    )
    return ((model.weight - targets) ** 2).mean()


def test_proximal_zero_positive_is_exact_noop() -> None:
    model = ScalarModel(0.75)
    before = model.weight.detach().clone()

    def must_not_run(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("loss must not be evaluated with zero positives")

    result = solve_proximal_update(
        model,
        [{"y": False, "target": 9.0}, {"safe": False, "target": -3.0}],
        must_not_run,
        ProximalConfig(
            eta=1.0,
            learning_rate=0.1,
            batch_size=2,
            max_steps=20,
            update_norm_limit=1.0,
        ),
    )

    assert torch.equal(model.weight.detach(), before)
    assert result.optimizer_steps == 0
    assert result.objective_evaluations == 0
    assert result.stopping_reason == "no_positive_records"


def test_proximal_uses_uniform_epochs_and_configured_not_fixed_steps() -> None:
    records = [
        {"y": True, "target": float(index), "frontier_weight": 1000.0 - index}
        for index in range(6)
    ] + [{"y": False, "target": 100.0}]
    config = ProximalConfig(
        eta=100.0,
        learning_rate=1.0e-3,
        batch_size=2,
        max_steps=7,
        update_norm_limit=100.0,
        relative_loss_tolerance=None,
        gradient_tolerance=None,
        seed=17,
    )
    model_a = ScalarModel()
    model_b = ScalarModel()

    result_a = solve_proximal_update(model_a, records, mse_loss, config)
    result_b = solve_proximal_update(model_b, records, mse_loss, config)

    # Every optimizer step is the exact uniform empirical objective over all
    # six positives, split into three memory microbatches.  Arbitrary frontier
    # metadata has no effect.  Seven requested steps also proves that four is
    # not embedded in the implementation.
    for step in result_a.trace:
        assert step.microbatch_count == 3
        assert step.microbatch_sizes == (2, 2, 2)
        assert step.unique_record_count == 6
        assert len(step.record_order_sha256) == 64
        assert step.positive_coverage == 1.0
    assert result_a.optimizer_steps == 7
    assert result_a.stopping_reason == "max_steps"
    assert result_a.sampling == "uniform_full_positive_pass_seeded_reshuffle"
    assert [s.record_order_sha256 for s in result_a.trace] == [
        s.record_order_sha256 for s in result_b.trace
    ]
    assert torch.equal(model_a.weight.detach(), model_b.weight.detach())


def test_proximal_penalty_and_hard_update_norm_bound() -> None:
    model = ScalarModel(0.0)
    result = solve_proximal_update(
        model,
        [{"strict_bounds": True, "socp_success": True, "target": 10.0}],
        mse_loss,
        ProximalConfig(
            eta=0.5,
            learning_rate=1.0,
            batch_size=1,
            max_steps=50,
            update_norm_limit=0.05,
            relative_loss_tolerance=None,
            gradient_tolerance=None,
        ),
        optimizer_factory=lambda params, lr: torch.optim.SGD(params, lr=lr),
    )

    assert result.optimizer_steps == 1
    assert result.stopping_reason == "update_norm_bound"
    assert result.trace[0].projected_to_update_bound
    assert result.trace[0].proximal_penalty == pytest.approx(0.0)
    assert result.final_update_norm == pytest.approx(0.05, abs=1.0e-6)
    assert abs(float(model.weight.detach())) == pytest.approx(0.05, abs=1.0e-6)


def test_relative_tolerance_stops_at_the_evaluated_model() -> None:
    """Convergence telemetry and saved parameters must name one point."""

    model = ScalarModel(0.0)
    result = solve_proximal_update(
        model,
        [{"y": True, "target": 1.0}],
        mse_loss,
        ProximalConfig(
            eta=100.0,
            learning_rate=0.1,
            batch_size=1,
            max_steps=5,
            min_steps=1,
            update_norm_limit=10.0,
            relative_loss_tolerance=1.0,
            gradient_tolerance=None,
            tolerance_patience=1,
        ),
        optimizer_factory=lambda params, lr: torch.optim.SGD(params, lr=lr),
    )

    # One step moves w=0 to w=.2.  The next full objective evaluation meets
    # the deliberately loose tolerance and must stop *before* another step.
    assert result.stopping_reason == "relative_loss_tolerance"
    assert result.optimizer_steps == 1
    assert result.objective_evaluations == 2
    assert float(model.weight.detach()) == pytest.approx(0.2, abs=1.0e-7)
    expected_objective = (0.2 - 1.0) ** 2 + 0.2**2 / (2.0 * 100.0)
    assert result.trace[-1].objective == pytest.approx(expected_objective)
    assert result.trace[-1].optimizer_step == result.optimizer_steps
    assert result.trace[-1].update_norm == pytest.approx(result.final_update_norm)


def test_proximal_accepts_identity_checked_positive_replay_items() -> None:
    context = QueryContext(
        grid=np.zeros((2, 2), dtype=np.float32),
        low5=np.zeros(5, dtype=np.float32),
        hist=np.zeros((2, 2), dtype=np.float32),
        verifier_state=np.zeros(4, dtype=np.float64),
        verifier_spec_fingerprint="d" * 64,
    )
    plan = np.zeros((10, 2), dtype=np.float32)
    content_hash = query_content_hash(context, 0.5, plan)
    item = ReplayItem(
        context=context,
        gamma=0.5,
        plan=plan,
        source_query_hash=content_hash,
        training_target_hash=content_hash,
    )
    model = ScalarModel()

    def replay_loss(
        model: ScalarModel,
        batch: tuple[ReplayItem, ...],
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del generator
        assert batch == (item,)
        return (model.weight - 1.0) ** 2

    result = solve_proximal_update(
        model,
        [item],
        replay_loss,
        ProximalConfig(
            eta=1.0,
            learning_rate=0.01,
            batch_size=1,
            max_steps=1,
            update_norm_limit=1.0,
            relative_loss_tolerance=None,
            gradient_tolerance=None,
        ),
    )
    assert result.positive_count == 1
    assert result.optimizer_steps == 1


@dataclass(frozen=True)
class Verification:
    safe: bool
    progress: float
    mode: str


def test_independent_audit_is_untilted_isolated_and_per_gamma() -> None:
    source_contexts = [
        {"id": 0, "grid": np.array([0.0, 1.0])},
        {"id": 1, "grid": np.array([2.0, 3.0])},
    ]
    bank = ImmutableContextBank(source_contexts)
    original_fingerprint = bank.fingerprint
    acquisition_state = {
        "ledger_size": 11,
        "A": np.eye(2),
    }
    acquisition_before = acquisition_state["A"].copy()
    observed_temperatures: list[float] = []
    verifier_calls = 0

    def sample_plans(
        model: nn.Module,
        context: dict[str, object],
        gamma: float,
        count: int,
        *,
        temperature: float,
        generator: torch.Generator,
    ) -> np.ndarray:
        del model, context, gamma, generator
        observed_temperatures.append(temperature)
        return np.arange(count, dtype=np.float32).reshape(count, 1, 1)

    def verify(
        context: dict[str, object], gamma: float, plan: np.ndarray
    ) -> Verification:
        nonlocal verifier_calls
        verifier_calls += 1
        index = int(plan[0, 0])
        return Verification(
            safe=index % 2 == 0,
            progress=float(index),
            mode=("upper" if context["id"] == 0 else "right") + f"-{gamma}",
        )

    model = ScalarModel()
    result = run_independent_audit(
        model,
        bank,
        (0.1, 1.0),
        sample_plans,
        verify,
        AuditConfig(plans_per_context=3, progress_threshold=1.5, seed=91),
    )

    assert observed_temperatures == [1.0] * 4
    assert result.temperature == 1.0
    assert result.uncertainty_tilting is False
    assert result.sampling_distribution == "ordinary_conditional_flow_iid"
    assert verifier_calls == result.total_verifier_calls == 12
    for metric in result.per_gamma:
        assert metric.sample_count == 6
        assert metric.safe_count == 4
        assert metric.safe_progress_count == 2
        assert metric.validity_mass == pytest.approx(4 / 6)
        assert metric.progress_validity == pytest.approx(2 / 6)
        assert metric.safe_mode_coverage == 2

    # Audit has no store/A parameter or write path; source and bank are intact.
    assert acquisition_state["ledger_size"] == 11
    assert np.array_equal(acquisition_state["A"], acquisition_before)
    assert bank.fingerprint == original_fingerprint
    bank.assert_integrity()
    assert np.array_equal(source_contexts[0]["grid"], np.array([0.0, 1.0]))


def test_audit_prefers_explicit_bounds_and_socp_and_rejects_t05() -> None:
    with pytest.raises(ValueError, match="temperature 1.0"):
        AuditConfig(
            plans_per_context=1,
            progress_threshold=0.0,
            temperature=0.5,
        )

    def sample(
        model: nn.Module,
        context: object,
        gamma: float,
        count: int,
        *,
        temperature: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del model, context, gamma, temperature, generator
        return torch.zeros(count, 10, 2)

    def verifier(context: object, gamma: float, plan: torch.Tensor) -> dict[str, object]:
        del context, gamma, plan
        return {
            "safe": True,  # A stale/redundant label must not override evidence.
            "strict_bounds": True,
            "socp_success": False,
            "progress": 10.0,
        }

    result = run_independent_audit(
        ScalarModel(),
        [{"context": 1}],
        [0.5],
        sample,
        verifier,
        AuditConfig(plans_per_context=2, progress_threshold=1.0),
    )
    metric = result.per_gamma[0]
    assert metric.safe_count == 0
    assert metric.safe_progress_count == 0
    assert metric.validity_mass == 0.0
