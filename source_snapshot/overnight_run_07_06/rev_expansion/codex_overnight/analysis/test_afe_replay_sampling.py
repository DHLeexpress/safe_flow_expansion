from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_core as AC
import grid_expand_afe2 as AFE2


def _imbalanced_store() -> AC.DStore:
    store = AC.DStore()
    # round, gamma, episode(replica), number of positive queries at one context
    cells = (
        (1, 0.1, 0, 100),
        (1, 0.1, 1, 1),
        (1, 0.5, 2, 1),
        (2, 0.1, 0, 1),
        (2, 0.5, 2, 1),
    )
    for query_round, gamma, episode, count in cells:
        context_id = len(store.ctx_meta)
        store.ctx_meta.append((query_round, episode, 0))
        for _ in range(count):
            query_id = len(store.q_sid)
            store.q_sid.append(context_id)
            store.q_round.append(query_round)
            store.q_gamma.append(gamma)
            store.pos_ids.append(query_id)
    return store


def test_query_uniform_positive_replay_preserves_the_legacy_rng_draw() -> None:
    store = _imbalanced_store()
    population = list(store.pos_ids)
    expected_rng = np.random.default_rng(17)
    expected = [
        population[index]
        for index in expected_rng.integers(0, len(population), 200)
    ]

    actual = store.sample_positive_ids(
        200,
        np.random.default_rng(17),
        eligible_ids=population,
    )

    assert actual == expected


def test_hierarchical_positive_replay_neutralizes_query_count_dominance() -> None:
    store = _imbalanced_store()
    population = list(store.pos_ids)
    hierarchy = store.positive_replay_hierarchy(eligible_ids=population)

    draws = store.sample_positive_ids(
        20_000,
        np.random.default_rng(23),
        eligible_ids=population,
        sampling="round_gamma_replica_context",
        hierarchy=hierarchy,
    )

    rounds = np.asarray([store.q_round[query_id] for query_id in draws])
    # The first context owns 100/104 queries, but only one of the equally sampled
    # round->gamma->replica->context leaves.
    dominant = np.mean(np.asarray(draws) < 100)
    assert np.mean(rounds == 1) == pytest.approx(0.5, abs=0.02)
    assert dominant == pytest.approx(0.125, abs=0.02)


def test_hierarchical_positive_replay_respects_the_eligible_window() -> None:
    store = _imbalanced_store()
    eligible = [
        query_id for query_id in store.pos_ids
        if store.q_round[query_id] == 2
    ]
    hierarchy = store.positive_replay_hierarchy(eligible_ids=eligible)

    draws = store.sample_positive_ids(
        100,
        np.random.default_rng(9),
        eligible_ids=eligible,
        sampling="round_gamma_replica_context",
        hierarchy=hierarchy,
    )

    assert {store.q_round[query_id] for query_id in draws} == {2}
    assert set(draws).issubset(set(eligible))


@pytest.mark.parametrize(
    "sampling", ("query_uniform", "round_gamma_replica_context")
)
def test_positive_epoch_is_deterministic_and_uses_every_query_once(sampling) -> None:
    store = _imbalanced_store()
    eligible = list(store.pos_ids)

    first = store.positive_epoch_ids(
        np.random.default_rng(31), eligible_ids=eligible, sampling=sampling
    )
    second = store.positive_epoch_ids(
        np.random.default_rng(31), eligible_ids=eligible, sampling=sampling
    )

    assert first == second
    assert len(first) == len(eligible)
    assert len(set(first)) == len(eligible)
    assert set(first) == set(eligible)


def test_hierarchical_positive_epoch_interleaves_before_long_cell_tail() -> None:
    store = _imbalanced_store()
    order = store.positive_epoch_ids(
        np.random.default_rng(7),
        eligible_ids=store.pos_ids,
        sampling="round_gamma_replica_context",
    )

    # All four singleton cells appear before the 100-query cell is exhausted.
    assert {100, 101, 102, 103}.issubset(set(order[:12]))


def test_hierarchical_positive_replay_rejects_query_context_round_mismatch() -> None:
    store = _imbalanced_store()
    store.ctx_meta[0] = (99, 0, 0)
    with pytest.raises(RuntimeError, match="query/context round mismatch"):
        store.positive_replay_hierarchy(eligible_ids=store.pos_ids)


def test_hierarchical_equal_mass_uses_every_query_and_equalizes_nested_cells() -> None:
    store = _imbalanced_store()
    # Add a second context with two positives to the already dominant
    # (round 1, episode 0) lineage. This exercises both the context and query
    # levels, not just gamma/episode balancing.
    second_context = len(store.ctx_meta)
    store.ctx_meta.append((1, 0, 1))
    second_context_ids = []
    for _ in range(2):
        query_id = len(store.q_sid)
        store.q_sid.append(second_context)
        store.q_round.append(1)
        store.q_gamma.append(0.1)
        store.pos_ids.append(query_id)
        second_context_ids.append(query_id)
    mass, diagnostics = store.positive_hierarchy_equal_mass(
        (0.1, 0.5),
        eligible_ids=store.pos_ids
    )

    assert set(mass) == set(store.pos_ids)
    assert sum(mass.values()) == pytest.approx(1.0)
    assert diagnostics["missing_declared_gammas"] == []
    assert diagnostics["mass_by_gamma"] == pytest.approx({"0.1": 0.5, "0.5": 0.5})
    # Gamma .1 has three active episode instances, so each episode receives
    # 1/6 total mass. The first episode has two contexts, hence 1/12 each,
    # regardless of their 100-versus-2 positive query counts.
    dominant = sum(mass[query_id] for query_id in range(100))
    second = sum(mass[query_id] for query_id in second_context_ids)
    singleton = mass[100]
    assert dominant == pytest.approx(second)
    assert dominant == pytest.approx(1.0 / 12.0)
    assert singleton == pytest.approx(1.0 / 6.0)
    assert mass[second_context_ids[0]] == pytest.approx(
        mass[second_context_ids[1]]
    )
    assert diagnostics["episode_mass_max_error"] == pytest.approx(0.0)
    assert diagnostics["context_mass_max_error"] == pytest.approx(0.0)
    assert diagnostics["within_context_query_mass_max_error"] == pytest.approx(0.0)


def test_hierarchical_equal_mass_canonicalizes_float32_gamma_and_reports_empty() -> None:
    store = _imbalanced_store()
    store.q_gamma[0] = float(np.float32(0.1))
    mass, diagnostics = store.positive_hierarchy_equal_mass(
        (0.1, 0.5, 0.7), eligible_ids=store.pos_ids
    )
    assert sum(mass.values()) == pytest.approx(1.0)
    assert diagnostics["missing_declared_gammas"] == [0.7]
    store.q_gamma[0] = 0.6
    with pytest.raises(ValueError, match="not declared"):
        store.positive_hierarchy_equal_mass(
            (0.1, 0.5, 0.7), eligible_ids=store.pos_ids
        )


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.u_max = 1.0
        self.d = 2
        self.seen_weight_batches = []

    def module_groups(self):
        return {"trunk": self}

    def ctx_from(self, grid, low, hist):
        del grid, hist
        return low[:, :1]

    def _expand_ctx(self, context, count):
        assert len(context) == count
        return context

    def forward(self, values, time, context):
        del time
        return values + self.weight + context

    def cfm_loss(self, controls, context, weights=None):
        target = controls.mean(dim=1)
        per = (self.weight + context - target).square().mean(dim=1)
        if weights is not None:
            self.seen_weight_batches.append(
                [float(value) for value in weights.detach()]
            )
            per = per * weights
        return per.mean()


def _trainable_store() -> AC.DStore:
    store = AC.DStore()
    for query_round, gamma, episode in ((1, 0.1, 0), (1, 0.5, 1), (2, 0.1, 0)):
        context_id = len(store.ctx_meta)
        store.ctx_meta.append((query_round, episode, 0))
        store.ctx_hp.append(np.zeros((1, 2, 2), np.float32))
        store.ctx_low5.append(np.asarray((1.0, 0.0, 0.0, 0.0, gamma), np.float32))
        store.ctx_hist.append(np.zeros((1, 2), np.float32))
        query_id = len(store.q_sid)
        store.q_sid.append(context_id)
        store.q_round.append(query_round)
        store.q_gamma.append(gamma)
        store.q_U.append(np.asarray([[1.0, 0.0]], np.float32))
        store.pos_ids.append(query_id)
    return store


def test_update_round_routes_the_hierarchical_replay_option(monkeypatch) -> None:
    store = _trainable_store()
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    cfg = SimpleNamespace(
        arm="afe",
        replay_window=2,
        replay_sampling="round_gamma_replica_context",
        batch=2,
        afe_steps=1,
        grad_clip=0.0,
    )
    original = store.sample_pos
    calls = []

    def sample_spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "sample_pos", sample_spy)
    result = AFE2.update_round(
        policy,
        optimizer,
        store,
        cfg,
        torch.device("cpu"),
        np.random.default_rng(3),
        round_i=2,
    )

    assert result["replay_sampling"] == "round_gamma_replica_context"
    assert calls[0]["sampling"] == "round_gamma_replica_context"
    assert calls[0]["hierarchy"]


def test_update_round_exact_epoch_has_dynamic_tail_and_full_unique_coverage(monkeypatch) -> None:
    store = _trainable_store()
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    cfg = SimpleNamespace(
        arm="afe",
        replay_window=2,
        replay_sampling="round_gamma_replica_context",
        replay_update_mode="one_epoch_without_replacement",
        batch=2,
        afe_steps=0,
        grad_clip=0.0,
    )
    batch_sizes = []
    original = store.positive_batch

    def batch_spy(query_ids):
        batch_sizes.append(len(query_ids))
        return original(query_ids)

    monkeypatch.setattr(store, "positive_batch", batch_spy)
    result = AFE2.update_round(
        policy,
        optimizer,
        store,
        cfg,
        torch.device("cpu"),
        np.random.default_rng(3),
        round_i=2,
    )

    assert batch_sizes == [2, 1]
    assert result["steps"] == 2
    assert result["stop"] == "one_epoch_complete"
    assert result["optimizer_draws"] == 3
    assert result["n_distinct"] == 3
    assert result["replay_duplicate_draws"] == 0
    assert result["replay_epoch_coverage"] == 1.0
    assert result["replay_batch_sizes"] == [2, 1]
    assert set(result["drawn_ids"]) == set(store.pos_ids)


def test_update_round_exact_epoch_applies_hierarchical_equal_mass_weights() -> None:
    store = _trainable_store()
    # Add nine extra positives to the first context so query-uniform replay
    # would give that single lineage ten times the loss mass.
    first_context = store.q_sid[0]
    for _ in range(9):
        query_id = len(store.q_sid)
        store.q_sid.append(first_context)
        store.q_round.append(1)
        store.q_gamma.append(0.1)
        store.q_U.append(np.asarray([[1.0, 0.0]], np.float32))
        store.pos_ids.append(query_id)
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    cfg = SimpleNamespace(
        arm="afe",
        replay_window=2,
        replay_sampling="round_gamma_replica_context",
        replay_update_mode="one_epoch_without_replacement",
        replay_loss_weighting="gamma_episode_context_query_equal_mass",
        gammas=(0.1, 0.5),
        batch=4,
        afe_steps=0,
        grad_clip=0.0,
    )

    result = AFE2.update_round(
        policy,
        optimizer,
        store,
        cfg,
        torch.device("cpu"),
        np.random.default_rng(5),
        round_i=2,
    )

    assert result["replay_epoch_coverage"] == 1.0
    assert result["replay_loss_weighting"] == (
        "gamma_episode_context_query_equal_mass"
    )
    assert sum(len(batch) for batch in policy.seen_weight_batches) == len(store.pos_ids)
    assert result["replay_population_weight_max"] > (
        result["replay_population_weight_min"]
    )
    assert result["replay_applied_weight_max"] > result["replay_applied_weight_min"]
    assert 0.0 < result["replay_weight_ess_fraction"] < 1.0


def test_equal_mass_batch_weights_preserve_global_coefficients_with_tail(
    monkeypatch,
) -> None:
    store = _trainable_store()
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    cfg = SimpleNamespace(
        arm="afe",
        replay_window=2,
        replay_sampling="round_gamma_replica_context",
        replay_update_mode="one_epoch_without_replacement",
        replay_loss_weighting="gamma_episode_context_query_equal_mass",
        gammas=(0.1, 0.5),
        batch=2,
        afe_steps=0,
        grad_clip=0.0,
    )
    mass, _ = store.positive_hierarchy_equal_mass(
        cfg.gammas, eligible_ids=store.pos_ids
    )
    batch_ids = []
    original = store.positive_batch

    def batch_spy(query_ids):
        batch_ids.append(list(query_ids))
        return original(query_ids)

    monkeypatch.setattr(store, "positive_batch", batch_spy)
    AFE2.update_round(
        policy,
        optimizer,
        store,
        cfg,
        torch.device("cpu"),
        np.random.default_rng(3),
        round_i=2,
    )

    assert [len(ids) for ids in batch_ids] == [2, 1]
    steps = len(batch_ids)
    for ids, weights in zip(batch_ids, policy.seen_weight_batches):
        batch_size = len(ids)
        for query_id, weight in zip(ids, weights):
            assert weight / batch_size == pytest.approx(
                steps * mass[query_id]
            )
