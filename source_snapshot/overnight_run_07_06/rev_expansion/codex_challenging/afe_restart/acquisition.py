"""Finite-candidate AFE acquisition.

Sigma is consumed exactly here.  This module does not know labels, margins,
progress, replay classes, or optimizer weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AcquisitionBatch:
    indices: np.ndarray
    probabilities: np.ndarray
    entropy: float
    effective_sample_size: float


def gibbs_probabilities(sigmas: np.ndarray, beta: float) -> np.ndarray:
    values = np.asarray(sigmas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("sigmas must be a nonempty 1-D array")
    if not np.isfinite(values).all():
        raise ValueError("sigmas must be finite")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    logits = (values - values.max()) / float(beta)
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    return probabilities


def acquire_without_replacement(
    sigmas: np.ndarray,
    budget: int,
    beta: float,
    *,
    generator: torch.Generator,
) -> AcquisitionBatch:
    probabilities = gibbs_probabilities(sigmas, beta)
    if budget <= 0 or budget > len(probabilities):
        raise ValueError(f"budget must be in [1,{len(probabilities)}]")
    tensor = torch.as_tensor(probabilities, dtype=torch.float64)
    selected = torch.multinomial(tensor, budget, replacement=False, generator=generator)
    entropy = -float(np.sum(probabilities * np.log(probabilities + np.finfo(float).tiny)))
    ess = float(1.0 / np.sum(probabilities**2))
    return AcquisitionBatch(
        indices=selected.cpu().numpy().astype(np.int64, copy=False),
        probabilities=probabilities,
        entropy=entropy,
        effective_sample_size=ess,
    )

