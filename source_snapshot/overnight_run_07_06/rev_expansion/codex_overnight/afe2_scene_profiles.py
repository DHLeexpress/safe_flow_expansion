"""Explicit scene adapters for running one AFE2 engine on declared tasks.

The profile is the only task-specific input to the shared AFE2 implementation.
It fixes geometry and endpoints; the pretrained checkpoint remains an explicit
CLI input and is hash-recorded by the trainer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
for _path in (_WORK, _REV, _HERE):
    sys.path.insert(0, _path)


CENTRAL_CENTERS = np.asarray(
    ((2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)),
    dtype=np.float32,
)
GIANT_CENTER = np.asarray((2.5, 2.5), dtype=np.float32)


@dataclass(frozen=True)
class AFE2SceneProfile:
    name: str
    start: tuple[float, float]
    goal: tuple[float, float]
    wall_plugs: int
    center_replacement_radius: float | None
    description: str
    interior_disk_radius: float | None = None


CLAUDE_GRID_V1 = AFE2SceneProfile(
    name="claude_grid_v1",
    start=(0.3, 0.3),
    goal=(4.7, 4.7),
    wall_plugs=8,
    center_replacement_radius=None,
    description=(
        "Claude AFE2 reference: 4x4 radius-0.2 interior grid, established "
        "boundary walls plus eight plugs, start (0.3,0.3), goal (4.7,4.7)."
    ),
)

CODEX_RADIUS1_V1 = AFE2SceneProfile(
    name="codex_radius1_v1",
    start=(0.5, 0.5),
    goal=(4.5, 4.5),
    wall_plugs=8,
    center_replacement_radius=1.0,
    description=(
        "Codex OOD task: replace exactly the four disks at (2,2), (2,3), "
        "(3,2), (3,3) by one physical disk at (2.5,2.5), radius 1.0; "
        "retain the other interior disks, boundary walls, and eight plugs."
    ),
)

CODEX_RADIUS04_V1 = AFE2SceneProfile(
    name="codex_radius04_v1",
    start=(0.5, 0.5),
    goal=(4.5, 4.5),
    wall_plugs=8,
    center_replacement_radius=None,
    interior_disk_radius=0.4,
    description=(
        "Codex obstacle-size OOD task: retain the 4x4 interior grid but double "
        "exactly those sixteen physical disk radii from 0.2 to 0.4; retain the "
        "radius-0.2 boundary walls and eight plugs."
    ),
)

CODEX_RADIUS03_V1 = AFE2SceneProfile(
    name="codex_radius03_v1",
    start=(0.5, 0.5),
    goal=(4.5, 4.5),
    wall_plugs=8,
    center_replacement_radius=None,
    interior_disk_radius=0.3,
    description=(
        "Obstacle-size OOD task: retain the 4x4 interior grid but change exactly "
        "those sixteen physical disk radii from 0.2 to 0.3; retain the "
        "radius-0.2 boundary walls and eight plugs."
    ),
)

# Endpoint-matched profiles for the low7 pretraining/OOD comparison.  The old
# profiles above remain immutable so historical run hashes stay meaningful.
LOW7_ID_CANONICAL_V1 = AFE2SceneProfile(
    name="low7_id_canonical_v1",
    start=(0.3, 0.3),
    goal=(4.7, 4.7),
    wall_plugs=8,
    center_replacement_radius=None,
    description=(
        "Endpoint-matched low7 ID task: ordinary 4x4 radius-0.2 interior grid, "
        "start (0.3,0.3), goal (4.7,4.7)."
    ),
)

LOW7_RADIUS1_CANONICAL_V1 = AFE2SceneProfile(
    name="low7_radius1_canonical_v1",
    start=(0.3, 0.3),
    goal=(4.7, 4.7),
    wall_plugs=8,
    center_replacement_radius=1.0,
    description=(
        "Endpoint-matched giant-obstacle OOD task: replace the four central "
        "radius-0.2 disks by one radius-1.0 disk at (2.5,2.5)."
    ),
)

LOW7_RADIUS03_CANONICAL_V1 = AFE2SceneProfile(
    name="low7_radius03_canonical_v1",
    start=(0.3, 0.3),
    goal=(4.7, 4.7),
    wall_plugs=8,
    center_replacement_radius=None,
    interior_disk_radius=0.3,
    description=(
        "Endpoint-matched obstacle-size OOD task: change all sixteen interior "
        "disk radii from 0.2 to 0.3."
    ),
)

SCENE_PROFILES = {
    profile.name: profile
    for profile in (
        CLAUDE_GRID_V1,
        CODEX_RADIUS1_V1,
        CODEX_RADIUS03_V1,
        CODEX_RADIUS04_V1,
        LOW7_ID_CANONICAL_V1,
        LOW7_RADIUS1_CANONICAL_V1,
        LOW7_RADIUS03_CANONICAL_V1,
    )
}


def get_scene_profile(name: str) -> AFE2SceneProfile:
    try:
        return SCENE_PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown AFE2 scene profile {name!r}; expected one of "
            f"{sorted(SCENE_PROFILES)}"
        ) from exc


def replace_four_central_disks(
    obstacles: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Return the declared four-to-one center replacement geometry."""

    values = np.asarray(obstacles, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("obstacles must have shape [N,3]")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("center replacement radius must be finite and positive")
    central = np.zeros(len(values), dtype=bool)
    for center in CENTRAL_CENTERS:
        central |= np.all(
            np.isclose(values[:, :2], center[None], rtol=0.0, atol=1.0e-7),
            axis=1,
        )
    if int(central.sum()) != 4:
        raise RuntimeError(
            "codex_radius1_v1 requires exactly four central source disks; "
            f"found {int(central.sum())}"
        )
    giant = np.asarray([[2.5, 2.5, float(radius)]], dtype=np.float32)
    replaced = np.concatenate((values[~central], giant), axis=0)
    for center in CENTRAL_CENTERS:
        if np.any(
            np.all(
                np.isclose(replaced[:, :2], center[None], rtol=0.0, atol=1.0e-7),
                axis=1,
            )
        ):
            raise RuntimeError("a replaced central disk remains in the OOD scene")
    return replaced


def replace_interior_disk_radii(
    obstacles: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Change exactly the canonical 4x4 interior disks, leaving walls/plugs intact."""

    values = np.asarray(obstacles, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("obstacles must have shape [N,3]")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("interior disk radius must be finite and positive")
    centers = np.asarray(
        [(x, y) for x in (1.0, 2.0, 3.0, 4.0) for y in (1.0, 2.0, 3.0, 4.0)],
        dtype=np.float32,
    )
    interior = np.zeros(len(values), dtype=bool)
    for center in centers:
        matches = np.all(
            np.isclose(values[:, :2], center[None], rtol=0.0, atol=1.0e-7),
            axis=1,
        )
        if int(matches.sum()) != 1:
            raise RuntimeError(
                "the interior-radius scene requires one canonical interior disk at "
                f"{tuple(float(v) for v in center)}; found {int(matches.sum())}"
            )
        interior |= matches
    if int(interior.sum()) != 16:
        raise RuntimeError(f"expected 16 interior disks; found {int(interior.sum())}")
    replaced = values.copy()
    replaced[interior, 2] = float(radius)
    return replaced


def build_scene(profile: AFE2SceneProfile):
    """Build one profile using the same base scene implementation as AFE2."""

    # Imports are deliberately local.  Pure geometry tests can import this
    # module without resolving the experiment's several same-named modules.
    import torch
    import grid_scene as GS
    import grid_expand_hardtail as HT

    env = HT._apply_wall_plugs(GS.make_grid(), profile.wall_plugs)
    if profile.interior_disk_radius is not None:
        replaced = replace_interior_disk_radii(
            env.obstacles.detach().cpu().numpy(),
            radius=profile.interior_disk_radius,
        )
        env.obstacles = torch.as_tensor(
            replaced,
            dtype=env.obstacles.dtype,
            device=env.obstacles.device,
        )
        env.obs_vel = torch.zeros(
            len(replaced),
            2,
            dtype=env.obstacles.dtype,
            device=env.obstacles.device,
        )
    if profile.center_replacement_radius is not None:
        replaced = replace_four_central_disks(
            env.obstacles.detach().cpu().numpy(),
            radius=profile.center_replacement_radius,
        )
        env.obstacles = torch.as_tensor(
            replaced,
            dtype=env.obstacles.dtype,
            device=env.obstacles.device,
        )
        env.obs_vel = torch.zeros(
            len(replaced),
            2,
            dtype=env.obstacles.dtype,
            device=env.obstacles.device,
        )
    env.x0 = torch.as_tensor(
        [profile.start[0], profile.start[1], 0.0, 0.0],
        dtype=env.x0.dtype,
        device=env.x0.device,
    )
    env.goal = torch.as_tensor(
        profile.goal,
        dtype=env.goal.dtype,
        device=env.goal.device,
    )
    return env


def scene_snapshot(env, profile: AFE2SceneProfile) -> dict[str, Any]:
    """Serialize the exact geometry used by training and visualization."""

    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64)
    start = env.x0.detach().cpu().numpy().astype(np.float64)
    goal = env.goal.detach().cpu().numpy().astype(np.float64)
    profile_payload = asdict(profile)
    # Preserve the hashes of the two already-published profiles.  This field did
    # not exist when their artifacts were frozen and is meaningful only when set.
    if profile_payload["interior_disk_radius"] is None:
        profile_payload.pop("interior_disk_radius")
    payload: dict[str, Any] = {
        "profile": profile_payload,
        "obstacles": obstacles.tolist(),
        "start_state": start.tolist(),
        "goal": goal.tolist(),
        "robot_radius": float(env.r_robot),
        "dt": float(env.dt),
        "u_max": float(env.u_max),
        "workspace_bounds": [0.0, 5.0, 0.0, 5.0],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def assert_scene_snapshot(snapshot: dict[str, Any]) -> None:
    """Fail unless a snapshot is intact and exactly equals its named profile."""

    copied = dict(snapshot)
    expected = str(copied.pop("sha256"))
    canonical = json.dumps(
        copied,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != expected:
        raise ValueError("AFE2 scene snapshot fingerprint mismatch")
    profile_payload = copied.get("profile") or {}
    profile = get_scene_profile(profile_payload.get("name"))
    expected_snapshot = scene_snapshot(build_scene(profile), profile)
    if expected != expected_snapshot["sha256"]:
        raise ValueError(
            f"AFE2 scene snapshot does not equal declared profile {profile.name!r}"
        )
