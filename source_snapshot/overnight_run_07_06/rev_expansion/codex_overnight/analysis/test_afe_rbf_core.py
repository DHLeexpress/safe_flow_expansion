from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_rbf_core as RC
import afe2_calibration as BC


def test_rbf_sigma_is_small_on_buffer_and_large_for_a_distant_feature() -> None:
    gp = RC.RBFGPSigma(lengthscale=0.25, lam=1.0e-4)
    buffer = torch.tensor([[1.0, 0.0], [0.98, 0.2]], dtype=torch.float32)
    gp.set_buffer(buffer)
    values = gp.sigma(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))

    assert float(values[0]) < 0.02
    assert float(values[1]) > 0.95
    assert gp.diagnostics()["kernel_effective_rank"] > 1.0


def test_lengthscale_is_mean_pairwise_normalized_distance() -> None:
    features = torch.tensor([[2.0, 0.0], [0.0, 3.0], [-4.0, 0.0]])
    expected = (np.sqrt(2.0) + 2.0 + np.sqrt(2.0)) / 3.0
    assert RC.mean_pairwise_lengthscale(features) == pytest.approx(expected)


def test_batch_conditional_variance_penalizes_near_duplicates() -> None:
    gp = RC.RBFGPSigma(lengthscale=0.25, lam=1.0e-3)
    candidates = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    conditional = gp.conditional_variance(candidates)

    assert float(conditional[0]) == pytest.approx(float(conditional[1]), rel=1.0e-5)
    assert float(conditional[0]) < 0.01
    assert float(conditional[2]) > 0.9


def test_sequential_scores_condition_only_on_already_selected_locations() -> None:
    gp = RC.RBFGPSigma(lengthscale=0.2, lam=1.0e-3)
    candidates = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    scores = gp.sequential_score_vectors(
        candidates, torch.tensor([0, 1, 2]), steps=2
    )

    assert torch.allclose(scores[0], torch.ones(3), atol=2.0e-3)
    assert float(scores[1][0]) < 0.01
    assert float(scores[1][1]) > 0.9


def test_sequential_acquisition_suppresses_a_selected_duplicate() -> None:
    gp = RC.RBFGPSigma(lengthscale=0.2, lam=1.0e-3)
    candidates = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    torch.manual_seed(4)
    selected, trace = gp.sequential_acquire(candidates, steps=2, beta=1.0e-4)

    assert len(selected) == len(set(selected)) == 2
    assert not torch.equal(candidates[selected[0]], candidates[selected[1]])
    assert len(trace) == 2
    assert all(0.0 < row["ess_norm"] <= 1.0 for row in trace)


def test_ragged_beta_solver_hits_the_declared_normalized_ess() -> None:
    vectors = [
        np.asarray([0.0, 0.1, 0.4, 0.8]),
        np.asarray([0.0, 0.2, 0.7]),
        np.asarray([0.1, 0.5]),
    ]
    solution = BC.solve_beta_ragged(vectors)
    summary = BC.ess_summary_ragged(vectors, solution["beta"])

    assert summary["ess_med"] == pytest.approx(BC.ESS_TARGET, abs=BC.ESS_TOLERANCE)
    assert solution["score_vector_lengths"] == [4, 3, 2]
    assert len(BC.score_vectors_sha256(vectors)) == 64


class _Store:
    def __init__(self):
        self.pos_ids = list(range(36))
        self.q_round = [1] * 28 + [2] * 8
        self.q_gamma = [0.1] * 20 + [0.5] * 8 + [0.1] * 4 + [0.5] * 4


def test_previous_round_buffer_is_capped_balanced_and_round_local() -> None:
    store = _Store()
    selected = RC.previous_round_positive_ids(
        store, round_i=1, cap=10, gammas=(0.1, 0.5), seed=7
    )
    selected_gamma = [store.q_gamma[index] for index in selected]

    assert len(selected) == len(set(selected)) == 10
    assert all(store.q_round[index] == 1 for index in selected)
    assert selected_gamma.count(0.1) == 5
    assert selected_gamma.count(0.5) == 5


def test_previous_round_buffer_balances_float32_conditioning_gammas() -> None:
    store = _Store()
    store.pos_ids = list(range(20))
    store.q_round = [1] * 20
    store.q_gamma = [np.float32(0.3)] * 10 + [np.float32(0.7)] * 10

    selected = RC.previous_round_positive_ids(
        store, round_i=1, cap=8, gammas=(0.3, 0.7), seed=7
    )

    assert len(selected) == len(set(selected)) == 8
    assert sum(np.isclose(store.q_gamma[index], 0.3) for index in selected) == 4
    assert sum(np.isclose(store.q_gamma[index], 0.7) for index in selected) == 4


def test_one_round_recent_buffer_preserves_previous_round_selection() -> None:
    store = _Store()
    expected = RC.previous_round_positive_ids(
        store, round_i=1, cap=10, gammas=(0.1, 0.5), seed=17
    )
    actual = RC.recent_round_positive_ids(
        store,
        round_i=1,
        replay_window=1,
        cap=10,
        gammas=(0.1, 0.5),
        seed=17,
    )

    assert actual == expected


class _RecentStore:
    def __init__(self):
        self.pos_ids = list(range(32))
        self.q_round = [round_i for round_i in range(1, 5) for _ in range(8)]
        self.q_gamma = [gamma for _ in range(4) for gamma in (0.1, 0.5) for _ in range(4)]


def test_recent_round_buffer_is_deterministic_capped_and_cell_balanced() -> None:
    store = _RecentStore()
    selected = RC.recent_round_positive_ids(
        store,
        round_i=4,
        replay_window=2,
        cap=8,
        gammas=(0.1, 0.5),
        seed=23,
    )
    repeated = RC.recent_round_positive_ids(
        store,
        round_i=4,
        replay_window=2,
        cap=8,
        gammas=(0.1, 0.5),
        seed=23,
    )

    assert selected == repeated
    assert len(selected) == len(set(selected)) == 8
    counts = {}
    for query_id in selected:
        cell = (store.q_round[query_id], store.q_gamma[query_id])
        counts[cell] = counts.get(cell, 0) + 1
    assert counts == {(3, 0.1): 2, (3, 0.5): 2, (4, 0.1): 2, (4, 0.5): 2}


class _HierarchicalStore:
    def __init__(self):
        self.pos_ids = []
        self.q_round = []
        self.q_gamma = []
        self.q_sid = []
        self.ctx_meta = []
        for round_i in (1, 2):
            for gamma_index, gamma in enumerate((0.1, 0.5)):
                for replica, context_count in enumerate((8, 2)):
                    episode = gamma_index * 2 + replica
                    for control_t in range(context_count):
                        context_id = len(self.ctx_meta)
                        self.ctx_meta.append((round_i, episode, control_t))
                        for _ in range(2):
                            query_id = len(self.pos_ids)
                            self.pos_ids.append(query_id)
                            self.q_round.append(round_i)
                            self.q_gamma.append(gamma)
                            self.q_sid.append(context_id)

    def positive_replay_hierarchy(self, *, eligible_ids=None):
        import afe_core as AC

        proxy = AC.DStore()
        proxy.pos_ids = list(self.pos_ids)
        proxy.q_round = list(self.q_round)
        proxy.q_gamma = list(self.q_gamma)
        proxy.q_sid = list(self.q_sid)
        proxy.ctx_meta = list(self.ctx_meta)
        return proxy.positive_replay_hierarchy(eligible_ids=eligible_ids)


def test_hierarchical_recent_gp_buffer_balances_replicas_without_replacement() -> None:
    store = _HierarchicalStore()
    selected = RC.recent_round_positive_ids_hierarchical(
        store, round_i=2, replay_window=2, cap=16, seed=31
    )
    repeated = RC.recent_round_positive_ids_hierarchical(
        store, round_i=2, replay_window=2, cap=16, seed=31
    )

    assert selected == repeated
    assert len(selected) == len(set(selected)) == 16
    cell_counts = {}
    for query_id in selected:
        context_id = store.q_sid[query_id]
        round_i, episode, _ = store.ctx_meta[context_id]
        key = (round_i, store.q_gamma[query_id], episode)
        cell_counts[key] = cell_counts.get(key, 0) + 1
    assert set(cell_counts.values()) == {2}
