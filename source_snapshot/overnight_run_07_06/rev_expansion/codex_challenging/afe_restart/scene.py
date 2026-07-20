"""Exact ID/OOD stadium construction and policy contexts."""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

import grid_feats as grid_features

from .config import DynamicsConfig, VerifierConfig
from . import seed_geometry
from .schemas import QueryContext
from . import verifier as full_verifier


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
START = np.asarray((0.5, 0.5), dtype=np.float32)
GOAL = np.asarray((4.5, 4.5), dtype=np.float32)
GIANT_CENTER = np.asarray((2.5, 2.5), dtype=np.float32)
CENTRAL_CENTERS = np.asarray(
    ((2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)), dtype=np.float32,
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def verifier_implementation_fingerprint() -> str:
    """Hash the local implementation that can change a verifier label.

    Numerical inputs are fingerprinted separately for each scene.  This digest
    prevents a saved deterministic-result cache from being resumed after an
    implementation edit without regenerating its verifier observations.
    """

    sources = {
        Path(full_verifier.__file__).resolve(),
        Path(__file__).with_name("dynamics.py").resolve(),
        Path(__file__).with_name("config.py").resolve(),
        Path(full_verifier.VP.__file__).resolve(),
    }
    for function_name in (
        "check_certificate",
        "make_variable_faces",
        "artificial_obstacles",
    ):
        source = inspect.getsourcefile(getattr(full_verifier.VP, function_name))
        if source is None:
            raise RuntimeError(
                f"cannot resolve verifier dependency source for {function_name}"
            )
        sources.add(Path(source).resolve())
    payload = {
        "schema": "afe-full-verifier-implementation-v1",
        "numpy_version": np.__version__,
        "sources": [
            {"path": str(path), "sha256": _sha256_path(path)}
            for path in sorted(sources, key=str)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fingerprint_array(digest, label: str, value: np.ndarray) -> None:
    """Add one canonical numerical array to a verifier-spec digest."""

    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8") + b"\x00")
    digest.update(array.dtype.str.encode("ascii") + b"\x00")
    digest.update(repr(array.shape).encode("ascii") + b"\x00")
    digest.update(array.tobytes(order="C"))


def verifier_spec_fingerprint(
    env,
    goal: np.ndarray | None = None,
    *,
    dynamics: DynamicsConfig = DynamicsConfig(),
    verifier: VerifierConfig = VerifierConfig(),
) -> str:
    """Fingerprint every fixed input governing one full-verifier label.

    State and plan vary per query and are hashed separately.  This digest locks
    obstacle geometry, robot radius, goal, environment step, dynamics and
    fitted-polytope verifier configuration, plus byte hashes of the local
    implementation files that can change a label.
    """

    obstacles = (
        env.obstacles.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    goal_source = env.goal if goal is None else goal
    if hasattr(goal_source, "detach"):
        goal_source = goal_source.detach().cpu().numpy()
    goal_xy = np.asarray(goal_source, dtype=np.float64).reshape(-1)[:2]
    if obstacles.ndim != 2 or obstacles.shape[1] != 3:
        raise ValueError("verifier scene obstacles must have shape (N,3)")
    if goal_xy.shape != (2,) or not np.isfinite(goal_xy).all():
        raise ValueError("verifier goal must contain two finite coordinates")
    digest = hashlib.sha256()
    digest.update(b"afe-fitted-polytope-full-window-verifier-spec-v2\x00")
    digest.update(verifier_implementation_fingerprint().encode("ascii") + b"\x00")
    _fingerprint_array(digest, "obstacles_float64", obstacles)
    _fingerprint_array(digest, "goal_float64", goal_xy)
    scalars = {
        "robot_radius": float(env.r_robot),
        "environment_dt": float(env.dt),
        "dynamics": asdict(dynamics),
        "verifier": asdict(verifier),
    }
    digest.update(
        json.dumps(
            scalars,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _set_endpoints(env, start: np.ndarray = START, goal: np.ndarray = GOAL):
    env.x0 = torch.as_tensor(
        [float(start[0]), float(start[1]), 0.0, 0.0], dtype=env.x0.dtype, device=env.x0.device,
    )
    env.goal = torch.as_tensor(goal, dtype=env.goal.dtype, device=env.goal.device)
    return env


def make_id_scene(*, start: np.ndarray = START, goal: np.ndarray = GOAL):
    """Ordinary symmetric 4x4 ID stadium with the established eight plugs."""
    return _set_endpoints(seed_geometry.make_walled_env(8), start, goal)


def make_ood_scene(
    radius: float = 1.2, *, start: np.ndarray = START, goal: np.ndarray = GOAL,
):
    """Replace exactly the four central ID circles by one giant obstacle."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    env = make_id_scene(start=start, goal=goal)
    obstacles = env.obstacles.detach().cpu().numpy()
    central = np.zeros(len(obstacles), dtype=bool)
    for center in CENTRAL_CENTERS:
        central |= np.all(np.isclose(obstacles[:, :2], center[None], atol=1e-7), axis=1)
    if int(central.sum()) != 4:
        raise RuntimeError(f"expected exactly four central obstacles, got {int(central.sum())}")
    giant = np.asarray([[*GIANT_CENTER, float(radius)]], dtype=np.float32)
    replaced = np.concatenate((obstacles[~central], giant), axis=0)
    env.obstacles = torch.as_tensor(replaced, dtype=env.obstacles.dtype, device=env.obstacles.device)
    env.obs_vel = torch.zeros(len(replaced), 2, dtype=env.obstacles.dtype, device=env.obstacles.device)
    return env


def context_from_state(
    state: np.ndarray,
    goal: np.ndarray,
    gamma: float,
    executed_controls: list[np.ndarray] | np.ndarray,
    env,
    *,
    dynamics: DynamicsConfig = DynamicsConfig(),
    verifier: VerifierConfig = VerifierConfig(),
) -> QueryContext:
    """Build the original endpoint-free ``low5 + E(H_P)`` context."""
    verifier_state = np.asarray(state, dtype=np.float64).reshape(-1)
    if verifier_state.shape != (4,) or not np.isfinite(verifier_state).all():
        raise ValueError("verifier state must be a finite length-four vector")
    state = verifier_state.astype(np.float32)
    controls = np.asarray(executed_controls, dtype=np.float32).reshape(-1, 2)
    obstacle_array = env.obstacles.detach().cpu().numpy()
    grid, low5, history = grid_features.featurize(
        state,
        np.asarray(goal, dtype=np.float32),
        float(gamma),
        controls,
        obstacle_array,
        float(env.r_robot),
        K=grid_features.K_HIST,
    )
    return QueryContext(
        grid=grid,
        low5=low5,
        hist=history,
        verifier_state=verifier_state,
        verifier_spec_fingerprint=verifier_spec_fingerprint(
            env,
            np.asarray(goal, dtype=np.float64),
            dynamics=dynamics,
            verifier=verifier,
        ),
    )


def context_from_state_low7(
    state: np.ndarray,
    goal: np.ndarray,
    gamma: float,
    executed_controls: list[np.ndarray] | np.ndarray,
    env,
    *,
    dynamics: DynamicsConfig = DynamicsConfig(),
    verifier: VerifierConfig = VerifierConfig(),
) -> QueryContext:
    """Build ``low7 + E(H_P)`` without exposing absolute start or goal.

    ``QueryContext.low5`` retains its legacy wire name, but this schema stores
    seven values: relative goal, velocity, closest-boundary vector, and gamma.
    Keeping gamma last preserves downstream grouping. Keeping the field name
    avoids changing existing query identity and
    ledger code; the checkpoint schema records the actual dimension.
    """

    verifier_state = np.asarray(state, dtype=np.float64).reshape(-1)
    if verifier_state.shape != (4,) or not np.isfinite(verifier_state).all():
        raise ValueError("verifier state must be a finite length-four vector")
    state32 = verifier_state.astype(np.float32)
    controls = np.asarray(executed_controls, dtype=np.float32).reshape(-1, 2)
    obstacle_array = env.obstacles.detach().cpu().numpy()
    grid, low7, history = grid_features.featurize_low7(
        state32,
        np.asarray(goal, dtype=np.float32),
        float(gamma),
        controls,
        obstacle_array,
        float(env.r_robot),
        K=grid_features.K_HIST,
    )
    return QueryContext(
        grid=grid,
        low5=low7,
        hist=history,
        verifier_state=verifier_state,
        verifier_spec_fingerprint=verifier_spec_fingerprint(
            env,
            np.asarray(goal, dtype=np.float64),
            dynamics=dynamics,
            verifier=verifier,
        ),
    )


def minimum_endpoint_clearance(env) -> dict[str, float]:
    obstacles = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)

    def clearance(point: np.ndarray) -> float:
        return float(
            (np.linalg.norm(obstacles[:, :2] - point[None], axis=1) - obstacles[:, 2] - rr).min()
        )

    return {"start": clearance(START), "goal": clearance(GOAL)}
