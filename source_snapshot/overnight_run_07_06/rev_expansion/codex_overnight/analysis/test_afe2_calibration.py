from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "afe2_calibration_test_local", _ROOT / "afe2_calibration.py"
)
assert _SPEC is not None and _SPEC.loader is not None
BC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BC)


def _pools() -> np.ndarray:
    """Deterministic non-tied K=64 pools with a well-defined ESS root."""

    spans = np.linspace(0.02, 0.20, 31, dtype=np.float64)
    return np.stack([np.linspace(0.0, span, 64) for span in spans])


def _success_payload(pools: np.ndarray) -> tuple[dict, dict]:
    solution = BC.solve_beta(pools)
    expected = {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_model_sha256": "b" * 64,
        "scene_sha256": "c" * 64,
        "source_git_commit": "d" * 40,
        "lam": 1.0,
        "K": 64,
        "B": 8,
        "seed": 20260716,
    }
    payload = {
        **expected,
        "status": BC.SUCCESS_STATUS,
        "chosen": solution["beta"],
        "ess_target": BC.ESS_TARGET,
        "ess_tolerance": BC.ESS_TOLERANCE,
        "solver": BC.SOLVER,
        "acquisition": BC.ACQUISITION,
        "pool_weighting": BC.POOL_WEIGHTING,
        "solution": solution,
        "failure_reason": None,
        "n_pools": len(pools),
        "sigma_pool_sha256": BC.sigma_pool_sha256(pools),
    }
    return payload, expected


def test_continuous_solver_attains_predeclared_median_ess_target() -> None:
    solution = BC.solve_beta(_pools())
    achieved = solution["achieved"]

    assert solution["target"] == BC.ESS_TARGET
    assert solution["tolerance"] == BC.ESS_TOLERANCE
    assert abs(achieved["ess_med"] - BC.ESS_TARGET) <= BC.ESS_TOLERANCE
    assert 0.0 < achieved["ess_p10"] <= achieved["ess_med"] <= achieved["ess_p90"] <= 1.0
    assert solution["beta"] > 0.0


@pytest.mark.parametrize("target", [0.25, 0.5, 0.75])
def test_continuous_solver_accepts_explicit_ess_targets(target: float) -> None:
    solution = BC.solve_beta(_pools(), target=target)
    ragged = BC.solve_beta_ragged(list(_pools()), target=target)

    assert solution["target"] == target
    assert ragged["target"] == target
    assert solution["achieved"]["ess_med"] == pytest.approx(
        target, abs=BC.ESS_TOLERANCE
    )
    assert ragged["achieved"]["ess_med"] == pytest.approx(
        target, abs=BC.ESS_TOLERANCE
    )


@pytest.mark.parametrize("target", [0.0, 1.0, np.nan])
def test_continuous_solver_rejects_invalid_ess_targets(target: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        BC.solve_beta(_pools(), target=target)


@pytest.mark.parametrize("scale", [0.01, 7.0, 100.0])
def test_continuous_solver_is_scale_equivariant(scale: float) -> None:
    base = BC.solve_beta(_pools())
    scaled = BC.solve_beta(_pools() * scale)

    assert scaled["beta"] == pytest.approx(base["beta"] * scale, rel=1e-12)
    for name in ("ess_p10", "ess_med", "ess_p90"):
        assert scaled["achieved"][name] == pytest.approx(
            base["achieved"][name], abs=1e-12
        )


def test_continuous_solver_fails_closed_for_flat_sigma_pools() -> None:
    with pytest.raises(ValueError, match="sigma pools are flat"):
        BC.solve_beta(np.ones((20, 64), dtype=np.float64))


def test_continuous_solver_and_pool_digest_are_repeatable() -> None:
    pools = _pools()
    assert BC.solve_beta(pools) == BC.solve_beta(pools.copy())
    assert BC.sigma_pool_sha256(pools) == BC.sigma_pool_sha256(pools.copy())


def test_success_artifact_validates_solver_witness_and_provenance() -> None:
    payload, expected = _success_payload(_pools())
    assert BC.validate_success(payload, expected) == payload["chosen"]

    wrong_provenance = dict(expected, scene_sha256="e" * 64)
    with pytest.raises(ValueError, match="provenance mismatch"):
        BC.validate_success(payload, wrong_provenance)

    corrupt_digest = copy.deepcopy(payload)
    corrupt_digest["sigma_pool_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="sigma-pool digest"):
        BC.validate_success(corrupt_digest, expected)

    missed_target = copy.deepcopy(payload)
    missed_target["solution"]["achieved"]["ess_med"] = BC.ESS_TARGET + 2 * BC.ESS_TOLERANCE
    with pytest.raises(ValueError, match="did not attain"):
        BC.validate_success(missed_target, expected)
