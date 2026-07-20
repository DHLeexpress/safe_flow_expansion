"""Validated constants for the clean planned-window AFE restart.

The defaults intentionally distinguish scientific sampling from rendering:
expansion and model-validity audits use temperature 1.0, while temperature
0.5 is reserved for the rollout figure.  No value in this module introduces
a gamma curriculum or an easy/frontier replay split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


GAMMAS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def clean_method_absence_manifest() -> dict[str, object]:
    """Explicitly serialize mechanisms removed from the clean restart."""

    return {
        "demo_frac": "absent",
        "lwf_or_functional_data_anchor": "absent",
        "legacy_anchor_or_recovery_replay": "absent",
        "easy_frontier_split_or_weighting": "absent",
        "negative_sample_alpha_objective": "absent",
        "uncertainty_weighted_replay": "absent",
        "fixed_optimizer_step_count_as_scientific_rule": False,
    }


@dataclass(frozen=True)
class DynamicsConfig:
    """Double-integrator and task-space constants."""

    horizon: int = 10
    state_dim: int = 4
    action_dim: int = 2
    dt: float = 0.1
    u_max: float = 1.0
    workspace_low: float = 0.0
    workspace_high: float = 5.0

    def __post_init__(self) -> None:
        if self.horizon != 10:
            raise ValueError("the AFE query object is fixed to an H=10 plan")
        if self.state_dim != 4 or self.action_dim != 2:
            raise ValueError("expected a 4-D double-integrator state and 2-D action")
        if self.dt <= 0.0 or self.u_max <= 0.0:
            raise ValueError("dt and u_max must be positive")
        if self.workspace_low >= self.workspace_high:
            raise ValueError("workspace_low must be smaller than workspace_high")


@dataclass(frozen=True)
class VerifierConfig:
    """Arguments to the unchanged full fitted-polytope SOCP verifier."""

    sensing_radius: float = 2.5
    artificial_faces: int = 12
    artificial_radius: float = 0.16
    minimum_face_margin: float = 1.0e-4
    maximum_face_margin: float | None = None
    angle_samples: int = 180
    rollout_padding_factor: float = 1.3

    def __post_init__(self) -> None:
        if self.sensing_radius <= 0.0:
            raise ValueError("sensing_radius must be positive")
        if self.artificial_faces < 3:
            raise ValueError("at least three artificial faces are required")
        if self.artificial_radius <= 0.0 or self.minimum_face_margin <= 0.0:
            raise ValueError("artificial_radius and minimum_face_margin must be positive")
        if (self.maximum_face_margin is not None
                and self.maximum_face_margin < self.minimum_face_margin):
            raise ValueError("maximum_face_margin cannot be below the minimum")
        if self.angle_samples < 8:
            raise ValueError("angle_samples is too small for the face search")
        if self.rollout_padding_factor <= 0.0:
            raise ValueError("rollout_padding_factor must be positive")


@dataclass(frozen=True)
class SamplingConfig:
    """Finite-candidate AFE acquisition settings."""

    candidate_count: int = 64
    verifier_budget: int = 8
    beta: float = 0.2
    expansion_temperature: float = 1.0
    audit_temperature: float = 1.0
    visualization_temperature: float = 0.5
    nfe: int = 8

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not 0 < self.verifier_budget <= self.candidate_count:
            raise ValueError("verifier_budget must be in [1, candidate_count]")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if min(self.expansion_temperature, self.audit_temperature,
               self.visualization_temperature) <= 0.0:
            raise ValueError("sampling temperatures must be positive")
        if self.nfe <= 0:
            raise ValueError("nfe must be positive")


@dataclass(frozen=True)
class FeatureConfig:
    """Frozen fixed-feature linear uncertainty model settings."""

    representation_dim: int = 32
    feature_time: float = 0.9
    ridge_lambda: float = 1.0e-2

    def __post_init__(self) -> None:
        if self.representation_dim != 32:
            raise ValueError("the restart contract fixes the uncertainty feature at 32 dimensions")
        if not 0.0 <= self.feature_time <= 1.0:
            raise ValueError("feature_time must lie in [0, 1]")
        if self.ridge_lambda <= 0.0:
            raise ValueError("ridge_lambda must be positive")


@dataclass(frozen=True)
class AFEConfig:
    """Top-level immutable configuration shared by restart stages."""

    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gammas: tuple[float, ...] = GAMMAS

    def __post_init__(self) -> None:
        if not self.gammas:
            raise ValueError("at least one gamma is required")
        if any(not 0.0 < float(gamma) <= 1.0 for gamma in self.gammas):
            raise ValueError("each gamma must lie in (0, 1]")
        if len(set(float(gamma) for gamma in self.gammas)) != len(self.gammas):
            raise ValueError("gammas must be unique")


DEFAULT_CONFIG: Final[AFEConfig] = AFEConfig()
