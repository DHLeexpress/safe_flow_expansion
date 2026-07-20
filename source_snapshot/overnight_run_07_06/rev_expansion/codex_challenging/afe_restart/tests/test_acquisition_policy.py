from __future__ import annotations

import numpy as np
import torch

from afe_restart.acquisition import acquire_without_replacement, gibbs_probabilities


def test_gibbs_is_stable_and_monotone() -> None:
    sigma = np.asarray([1e9, 1e9 + 1.0, 1e9 + 2.0])
    probability = gibbs_probabilities(sigma, beta=0.5)
    assert np.isclose(probability.sum(), 1.0)
    assert np.all(np.diff(probability) > 0)


def test_acquisition_unique_seeded_and_only_sigma_dependent() -> None:
    sigma = np.linspace(0.0, 1.0, 32)
    first = acquire_without_replacement(
        sigma, 8, 0.2, generator=torch.Generator().manual_seed(4),
    )
    second = acquire_without_replacement(
        sigma, 8, 0.2, generator=torch.Generator().manual_seed(4),
    )
    assert len(np.unique(first.indices)) == 8
    assert np.array_equal(first.indices, second.indices)
    assert np.allclose(first.probabilities, second.probabilities)
    assert first.effective_sample_size <= len(sigma)

