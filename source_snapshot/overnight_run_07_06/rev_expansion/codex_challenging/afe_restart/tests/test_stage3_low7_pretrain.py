from __future__ import annotations

import torch

from afe_restart.stage3_low7_pretrain import (
    GAMMAS,
    Low7Pool,
    _canonical_gamma,
    _objective_weights,
    paired_split,
)


def _pool() -> Low7Pool:
    pairs = 4
    gamma = torch.tensor(
        [value for value in GAMMAS for _ in range(pairs)], dtype=torch.float64
    )
    pair_ids = torch.tensor(list(range(pairs)) * len(GAMMAS), dtype=torch.long)
    trajectory_ids = torch.arange(len(gamma), dtype=torch.long)
    count = len(gamma)
    return Low7Pool(
        grid=torch.zeros(count, 3, 32, 32),
        low7=torch.zeros(count, 7),
        hist=torch.zeros(count, 16, 2),
        plans=torch.zeros(count, 10, 2),
        gamma=gamma,
        pair_ids=pair_ids,
        trajectory_ids=trajectory_ids,
        trajectory_weight=torch.ones(count, dtype=torch.float64),
        trajectory_rows=tuple(
            {"trajectory_id": index} for index in range(count)
        ),
        query_hashes=tuple("0" * 64 for _ in range(count)),
        declared_pair_ids=tuple(range(pairs)),
        source={},
    )


def test_pair_split_is_disjoint_across_every_gamma() -> None:
    pool = _pool()
    train, validation, audit = paired_split(
        pool, validation_pairs=1, seed=17
    )

    train_pairs = set(pool.pair_ids[train].tolist())
    validation_pairs = set(pool.pair_ids[validation].tolist())
    assert train_pairs.isdisjoint(validation_pairs)
    assert audit["pair_leakage"] == 0
    for gamma in GAMMAS:
        entry = audit["per_gamma"][f"{gamma:g}"]
        assert entry == {
            "train_trajectories": 3,
            "validation_trajectories": 1,
        }


def test_objective_has_equal_gamma_mass() -> None:
    pool = _pool()
    rows = torch.arange(len(pool))
    weights = _objective_weights(pool, rows)

    masses = []
    for gamma in GAMMAS:
        mask = torch.isclose(
            pool.gamma, torch.tensor(gamma, dtype=torch.float64), atol=5e-7
        )
        masses.append(float(weights[mask].sum()))
    assert masses == [1.0] * len(GAMMAS)


def test_float32_gamma_is_canonicalized_before_identity_hashing() -> None:
    assert _canonical_gamma(float(torch.tensor(0.1, dtype=torch.float32))) == 0.1


def test_pair_split_uses_bank_before_success_missingness() -> None:
    pool = _pool()
    pool = Low7Pool(
        **{
            **pool.__dict__,
            "declared_pair_ids": tuple(range(6)),
        }
    )
    _train, _validation, audit = paired_split(pool, validation_pairs=2, seed=17)

    assert len(audit["train_pair_ids"]) == 4
    assert len(audit["validation_pair_ids"]) == 2
    assert set(audit["train_pairs_without_targets"]) | set(
        audit["validation_pairs_without_targets"]
    ) == {4, 5}
