from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import afe_adaptive as AD
import afe_core as AC


def test_positive_replay_window_preserves_archive_and_limits_population() -> None:
    store = AC.DStore()
    store.pos_ids = list(range(12))
    store.q_round = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]

    assert store.positive_ids() == list(range(12))
    assert store.positive_ids(round_i=6, replay_window=1) == [10, 11]
    assert store.positive_ids(round_i=6, replay_window=5) == list(range(2, 12))
    assert store.pos_ids == list(range(12))


def test_round_local_beta_calibration_hits_requested_target() -> None:
    generator = torch.Generator().manual_seed(7)
    buffer = torch.randn(64, 8, generator=generator)
    pools = [torch.randn(64, 8, generator=generator) for _ in range(4)]
    from afe_rbf_core import RBFGPSigma

    gp = RBFGPSigma(lengthscale=0.7, lam=1.0e-2)
    gp.set_buffer(buffer)
    cfg = SimpleNamespace(K=64, B=8, seed=910)
    result = AD.calibrate_from_pools(gp, pools, cfg, round_i=3, target=0.5)

    assert result["target"] == 0.5
    assert result["solution"]["achieved"]["ess_med"] == pytest.approx(
        0.5, abs=1.0e-4
    )
    assert result["verifier_queries"] == 0


def test_rbf_counterfactual_sweep_is_read_only_and_complete() -> None:
    generator = torch.Generator().manual_seed(11)
    buffer = torch.randn(512, 8, generator=generator)
    before = buffer.clone()
    pools = [torch.randn(64, 8, generator=generator) for _ in range(3)]
    cfg = SimpleNamespace(K=64, B=8, seed=910, gp_lam=1.0e-2)

    rows = AD.rbf_counterfactual_sweep(
        pools,
        buffer,
        cfg,
        round_i=2,
        target=0.5,
        lengthscale=0.7,
    )

    assert len(rows) == 6
    assert {(row["cap"], row["lengthscale_multiplier"]) for row in rows} == {
        (cap, multiplier)
        for cap in (128, 512)
        for multiplier in (0.5, 1.0, 2.0)
    }
    assert all(row["achieved"]["ess_med"] == pytest.approx(0.5, abs=1.0e-4)
               for row in rows)
    assert torch.equal(buffer, before)


def _context_store(rows):
    store = AC.DStore()
    for round_i, gamma, episode_id, control_t in rows:
        store.ctx_meta.append((round_i, episode_id, control_t))
        store.ctx_low5.append(np.asarray((0.0, 0.0, 0.0, 0.0, gamma), np.float32))
    return store


def test_adaptive_context_cap_is_round_gamma_episode_balanced_and_deterministic() -> None:
    rows = []
    for gamma_index, gamma in enumerate((0.1, 0.5)):
        for replica, count in enumerate((20, 5, 2)):
            episode_id = gamma_index * 3 + replica
            rows.extend((4, gamma, episode_id, step) for step in range(count))
    rows.extend((3, 0.1, 99, step) for step in range(20))
    store = _context_store(rows)

    selected = AD.round_gamma_episode_balanced_context_ids(
        store, 4, (0.1, 0.5), cap_per_gamma=12, seed=910
    )
    repeated = AD.round_gamma_episode_balanced_context_ids(
        store, 4, (0.1, 0.5), cap_per_gamma=12, seed=910
    )

    assert selected == repeated
    assert len(selected) == len(set(selected)) == 24
    assert all(store.ctx_meta[context_id][0] == 4 for context_id in selected)
    for gamma in (0.1, 0.5):
        gamma_ids = [
            context_id for context_id in selected
            if float(store.ctx_low5[context_id][-1]) == pytest.approx(gamma)
        ]
        assert len(gamma_ids) == 12
        counts = {}
        for context_id in gamma_ids:
            episode_id = store.ctx_meta[context_id][1]
            counts[episode_id] = counts.get(episode_id, 0) + 1
        # The two short replicas are exhausted; the remaining quota comes from
        # the long replica without dropping either short trajectory.
        assert sorted(counts.values()) == [2, 5, 5]


def test_adaptive_context_cap_keeps_underfilled_gamma_cells() -> None:
    store = _context_store([
        (2, 0.1, 0, 0),
        (2, 0.1, 1, 0),
        (2, 0.5, 2, 0),
    ])

    selected = AD.round_gamma_episode_balanced_context_ids(
        store, 2, (0.1, 0.5), cap_per_gamma=8, seed=5
    )

    assert selected == [0, 1, 2]


def test_adaptive_context_cap_accepts_all_declared_float32_gammas() -> None:
    gammas = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
    store = _context_store([
        (0, gamma, index, 0) for index, gamma in enumerate(gammas)
    ])

    selected = AD.round_gamma_episode_balanced_context_ids(
        store,
        0,
        gammas,
        cap_per_gamma=1,
        seed=910,
        equalize_gammas=True,
    )

    assert selected == list(range(len(gammas)))


def test_adaptive_context_cap_can_equalize_to_the_hardest_gamma() -> None:
    store = _context_store([
        *((2, 0.1, 0, step) for step in range(8)),
        *((2, 0.5, 1, step) for step in range(3)),
    ])

    selected = AD.round_gamma_episode_balanced_context_ids(
        store,
        2,
        (0.1, 0.5),
        cap_per_gamma=8,
        seed=5,
        equalize_gammas=True,
    )

    assert len(selected) == 6
    assert sum(store.ctx_low5[index][-1] == pytest.approx(0.1) for index in selected) == 3
    assert sum(store.ctx_low5[index][-1] == pytest.approx(0.5) for index in selected) == 3


def test_adaptive_context_cap_rejects_undeclared_gamma() -> None:
    store = _context_store([(1, 0.6, 0, 0)])
    with pytest.raises(ValueError, match="undeclared conditioning gamma"):
        AD.round_gamma_episode_balanced_context_ids(
            store, 1, (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0), cap_per_gamma=2, seed=1
        )


def test_adaptive_context_cap_rejects_nearby_undeclared_gamma() -> None:
    store = _context_store([(1, 0.3001, 0, 0)])
    with pytest.raises(ValueError, match="undeclared conditioning gamma"):
        AD.round_gamma_episode_balanced_context_ids(
            store, 1, (0.3,), cap_per_gamma=2, seed=1
        )
