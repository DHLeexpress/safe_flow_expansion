"""Reference-faithful deep-ensemble uncertainty for Safe Flow Expansion.

This module mirrors the neural uncertainty estimator used by the AFE molecule
and protein experiments: five independently initialized MLPs regress
standardized verifier labels from current-flow features, and their prediction
standard deviation is the uncertainty signal.  It deliberately has no kernel
matrix and no analytic within-batch posterior conditioning.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn


def l2_normalize(values: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


class EnsembleMember(nn.Module):
    """The exact 100-100 scalar MLP used by the public AFE implementation."""

    def __init__(self, feature_dim: int, hidden_dim: int = 100, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class DeepEnsembleSigma:
    """Five-member AFE ensemble with a GP-compatible acquisition surface.

    ``fit`` consumes every successful verifier query, positive and negative.
    The flow replay buffer remains positive-only elsewhere.  Before the first
    fit, scores are constant so the explicit warm-up acquisition is uniform.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        members: int = 5,
        hidden_dim: int = 100,
        dropout: float = 0.1,
        train_fraction: float = 0.9,
        learning_rate: float = 1.0e-3,
        max_steps: int = 1000,
        early_window: int = 30,
        device: str | torch.device = "cpu",
    ):
        if members < 2:
            raise ValueError("deep ensemble requires at least two members")
        if not 0.0 < train_fraction <= 1.0:
            raise ValueError("ensemble train fraction must lie in (0, 1]")
        if max_steps < 1 or early_window < 1:
            raise ValueError("ensemble fit lengths must be positive")
        self.feature_dim = int(feature_dim)
        self.member_count = int(members)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.train_fraction = float(train_fraction)
        self.learning_rate = float(learning_rate)
        self.max_steps = int(max_steps)
        self.early_window = int(early_window)
        self.device = torch.device(device)
        self.models: nn.ModuleList | None = None
        self.fit_diagnostics: dict[str, object] = {
            "n": 0,
            "positive_fraction": None,
            "member_steps": [],
            "fit_seconds": 0.0,
        }

    @property
    def n(self) -> int:
        return int(self.fit_diagnostics["n"])

    @property
    def is_fit(self) -> bool:
        return self.models is not None

    def _construct_models(self) -> nn.ModuleList:
        return nn.ModuleList([
            EnsembleMember(self.feature_dim, self.hidden_dim, self.dropout).to(self.device)
            for _ in range(self.member_count)
        ])

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
        """Reinitialize and refit all members on cumulative current-flow features."""

        started = time.perf_counter()
        values = l2_normalize(features.detach()).to(self.device)
        targets = labels.detach().flatten().to(self.device, torch.float32)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError("ensemble features have the wrong shape")
        if values.shape[0] != targets.shape[0] or values.shape[0] < 2:
            raise ValueError("ensemble fit requires matching feature/label rows")
        if not torch.isfinite(values).all() or not torch.isfinite(targets).all():
            raise ValueError("ensemble fit data must be finite")

        # This is the public AFE normalization, including its epsilon convention.
        standardized = (targets - targets.mean()) / (targets.std() + 1.0e-8)
        models = self._construct_models()
        criterion = nn.MSELoss()
        sample_count = values.shape[0]
        train_count = max(1, int(self.train_fraction * sample_count))
        member_steps: list[int] = []
        member_final_losses: list[float] = []

        for model in models:
            indices = torch.randperm(sample_count, device=self.device)[:train_count]
            train_x = values[indices]
            train_y = standardized[indices]
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            losses = torch.zeros(self.max_steps, dtype=torch.float32)
            steps_taken = self.max_steps
            model.train()
            for step in range(self.max_steps):
                optimizer.zero_grad()
                prediction = model(train_x).squeeze(-1)
                loss = criterion(prediction, train_y)
                loss.backward()
                optimizer.step()
                losses[step] = loss.detach().cpu()
                if step > self.early_window:
                    recent_min = losses[step - self.early_window + 1:step + 1].min()
                    overall_min = losses[:step - self.early_window + 1].min()
                    if overall_min <= recent_min:
                        steps_taken = step + 1
                        break
            model.eval()
            member_steps.append(int(steps_taken))
            member_final_losses.append(float(losses[steps_taken - 1]))

        self.models = models
        with torch.no_grad():
            train_predictions = self._predictions(values)
            train_sigma = train_predictions.std(dim=0, correction=0)
            member_correlation = _mean_member_correlation(train_predictions)
        self.fit_diagnostics = {
            "n": int(sample_count),
            "positive_fraction": float(targets.mean()),
            "label_unique_count": int(torch.unique(targets).numel()),
            "label_mean": float(targets.mean()),
            "label_std": float(targets.std(correction=0)),
            "member_steps": member_steps,
            "member_final_losses": member_final_losses,
            "train_sigma_mean": float(train_sigma.mean()),
            "train_sigma_q": [
                float(value) for value in torch.quantile(train_sigma, torch.tensor(
                    [0.1, 0.5, 0.9], device=train_sigma.device
                ))
            ],
            "member_prediction_correlation": member_correlation,
            "fit_seconds": float(time.perf_counter() - started),
        }
        return dict(self.fit_diagnostics)

    @torch.no_grad()
    def _predictions(self, features: torch.Tensor) -> torch.Tensor:
        if self.models is None:
            raise RuntimeError("ensemble has not been fit")
        return torch.stack([model(features).squeeze(-1) for model in self.models], dim=0)

    @torch.no_grad()
    def mean_and_sigma(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = l2_normalize(features.detach()).to(self.device)
        if self.models is None:
            zeros = torch.zeros(query.shape[0], dtype=query.dtype, device=query.device)
            return zeros, zeros
        predictions = self._predictions(query)
        std, mean = torch.std_mean(predictions, dim=0, correction=0)
        return mean, std

    @torch.no_grad()
    def sigma(self, features: torch.Tensor) -> torch.Tensor:
        return self.mean_and_sigma(features)[1]

    @torch.no_grad()
    def sequential_score_vectors(
        self,
        features: torch.Tensor,
        order: torch.Tensor,
        steps: int,
    ) -> list[torch.Tensor]:
        """Return fixed ensemble scores with selected indices removed.

        Unlike an exact GP, a frozen deep ensemble has no posterior update after
        selecting an unlabeled point.  The only sequential operation here is
        sampling candidate indices without replacement.
        """

        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("ensemble acquisition requires a K-by-D feature matrix")
        if steps < 1 or steps > features.shape[0]:
            raise ValueError("ensemble acquisition steps must lie in [1, K]")
        order = order.to(device=features.device, dtype=torch.long).flatten()
        if order.numel() != features.shape[0] or sorted(order.tolist()) != list(
            range(features.shape[0])
        ):
            raise ValueError("ensemble calibration order must be a permutation of K")
        all_scores = self.sigma(features)
        remaining = torch.arange(features.shape[0], device=features.device)
        vectors: list[torch.Tensor] = []
        for step in range(steps):
            vectors.append(all_scores[remaining])
            chosen = int(order[step])
            keep = remaining != chosen
            if int((~keep).sum()) != 1:
                raise RuntimeError("ensemble calibration selected a missing candidate")
            remaining = remaining[keep]
        return vectors

    @torch.no_grad()
    def sequential_acquire(
        self,
        features: torch.Tensor,
        steps: int,
        beta: float,
    ) -> tuple[list[int], list[dict[str, object]]]:
        """Gibbs-sample B indices without replacement from ensemble disagreement."""

        if not self.is_fit:
            raise RuntimeError("guided ensemble acquisition requires a fitted estimator")
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("ensemble beta must be finite and positive")
        if steps < 1 or steps > features.shape[0]:
            raise ValueError("ensemble acquisition steps must lie in [1, K]")
        all_scores = self.sigma(features)
        remaining = torch.arange(features.shape[0], device=features.device)
        selected: list[int] = []
        trace: list[dict[str, object]] = []
        for _ in range(steps):
            scores = all_scores[remaining]
            weights = torch.exp(((scores - scores.max()) / beta).clamp(-30.0, 30.0))
            probability = weights / weights.sum()
            chosen_local = int(torch.multinomial(probability, 1).item())
            chosen_global = int(remaining[chosen_local])
            normalized_ess = float(
                1.0 / (probability.to(torch.float64).square().sum() * probability.numel())
            )
            normalized_entropy = (
                float(
                    -(probability.to(torch.float64)
                      * (probability.to(torch.float64) + 1.0e-30).log()).sum()
                    / math.log(probability.numel())
                )
                if probability.numel() > 1
                else 1.0
            )
            trace.append({
                "scores": scores,
                "remaining": remaining,
                "chosen": chosen_global,
                "chosen_score": float(scores[chosen_local]),
                "ess_norm": normalized_ess,
                "entropy_norm": normalized_entropy,
            })
            selected.append(chosen_global)
            remaining = remaining[remaining != chosen_global]
        return selected, trace

    def diagnostics(self) -> dict[str, object]:
        return {
            "estimator": "deep_ensemble",
            "is_fit": self.is_fit,
            **self.fit_diagnostics,
        }

    def state_dict(self) -> dict[str, object]:
        """Serializable estimator state for exact acquisition replay."""

        model_states = None
        if self.models is not None:
            model_states = [
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                for model in self.models
            ]
        return {
            "version": 1,
            "config": {
                "feature_dim": self.feature_dim,
                "members": self.member_count,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "train_fraction": self.train_fraction,
                "learning_rate": self.learning_rate,
                "max_steps": self.max_steps,
                "early_window": self.early_window,
            },
            "fit_diagnostics": dict(self.fit_diagnostics),
            "model_states": model_states,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        device: str | torch.device = "cpu",
    ) -> "DeepEnsembleSigma":
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported deep-ensemble state version")
        config = dict(state["config"])
        estimator = cls(device=device, **config)
        model_states = state.get("model_states")
        if model_states is not None:
            if len(model_states) != estimator.member_count:
                raise ValueError("deep-ensemble state has the wrong member count")
            estimator.models = estimator._construct_models()
            for model, model_state in zip(estimator.models, model_states):
                model.load_state_dict(model_state)
                model.eval()
        estimator.fit_diagnostics = dict(state["fit_diagnostics"])
        return estimator


def _mean_member_correlation(predictions: torch.Tensor) -> float | None:
    """Mean off-diagonal member correlation, undefined for constant predictions."""

    centered = predictions - predictions.mean(dim=1, keepdim=True)
    norms = centered.norm(dim=1)
    valid = norms > 1.0e-12
    if int(valid.sum()) < 2:
        return None
    normalized = centered[valid] / norms[valid, None]
    correlation = normalized @ normalized.T
    pairs = torch.triu_indices(correlation.shape[0], correlation.shape[0], offset=1)
    return float(correlation[pairs[0], pairs[1]].mean())
