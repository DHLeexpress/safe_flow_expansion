from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parents[1]
_REV = _HERE.parent
_WORK = _REV.parent
_RESTART_PARENT = _REV / "codex_challenging"
_COLLIDING = {
    "_paths", "grid_feats", "grid_metrics", "grid_metrics2", "grid_rollout",
    "grid_scene", "grid_hp_expt", "grid_expand_hardtail", "di_grid_viz",
    "afe_core", "grid_expand_afe2", "afe2_scene_profiles", "afe2_calibration",
    "verifier_polytope",
}


@contextmanager
def _isolated_import(paths, module_name):
    names = {
        name for name in sys.modules
        if name in _COLLIDING or name == "afe_restart" or name.startswith("afe_restart.")
    }
    saved = {name: sys.modules.pop(name) for name in names}
    old_path = list(sys.path)
    sys.path[:0] = [str(path) for path in paths]
    try:
        yield importlib.import_module(module_name)
    finally:
        for name in list(sys.modules):
            if name in _COLLIDING or name == "afe_restart" or name.startswith("afe_restart."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = old_path


def _overnight_scene():
    return _isolated_import((_HERE, _REV, _WORK), "afe2_scene_profiles")


def _base_obstacles() -> np.ndarray:
    interior = np.asarray(
        [[x, y, 0.2] for x in (1.0, 2.0, 3.0, 4.0) for y in (1.0, 2.0, 3.0, 4.0)],
        dtype=np.float32,
    )
    walls = np.asarray(((-0.2, 1.0, 0.2), (5.2, 4.0, 0.2)), dtype=np.float32)
    return np.concatenate((interior, walls), axis=0)


def test_radius1_replaces_exactly_four_center_disks() -> None:
    base = _base_obstacles()
    with _overnight_scene() as scene:
        replaced = scene.replace_four_central_disks(base, radius=1.0)
        central_centers = scene.CENTRAL_CENTERS.copy()

    assert len(replaced) == len(base) - 3
    assert np.sum(
        np.all(np.isclose(replaced, np.asarray((2.5, 2.5, 1.0))), axis=1)
    ) == 1
    for center in central_centers:
        assert not np.any(
            np.all(np.isclose(replaced[:, :2], center[None]), axis=1)
        )

    retained_base = [
        row for row in base
        if not any(np.allclose(row[:2], center) for center in central_centers)
    ]
    for row in retained_base:
        assert np.any(np.all(np.isclose(replaced, row[None]), axis=1))


def test_radius1_replacement_rejects_ambiguous_source_geometry() -> None:
    base = _base_obstacles()
    with _overnight_scene() as scene:
        incomplete = base[
            ~np.all(np.isclose(base[:, :2], scene.CENTRAL_CENTERS[0][None]), axis=1)
        ]
        with pytest.raises(RuntimeError, match="exactly four"):
            scene.replace_four_central_disks(incomplete, radius=1.0)


def test_scene_snapshot_is_bound_to_geometry() -> None:
    with _overnight_scene() as scene:
        env = scene.build_scene(scene.CODEX_RADIUS1_V1)
        snapshot = scene.scene_snapshot(env, scene.CODEX_RADIUS1_V1)
        scene.assert_scene_snapshot(snapshot)

        tampered = copy.deepcopy(snapshot)
        tampered["obstacles"][-1][2] = 1.1
        with pytest.raises(ValueError, match="fingerprint"):
            scene.assert_scene_snapshot(tampered)

        self_consistent_wrong = copy.deepcopy(tampered)
        self_consistent_wrong.pop("sha256")
        canonical = json.dumps(
            self_consistent_wrong,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        self_consistent_wrong["sha256"] = hashlib.sha256(canonical).hexdigest()
        with pytest.raises(ValueError, match="declared profile"):
            scene.assert_scene_snapshot(self_consistent_wrong)


def test_radius1_profile_builds_declared_exact_geometry() -> None:
    with _overnight_scene() as scene:
        env = scene.build_scene(scene.CODEX_RADIUS1_V1)
        snapshot = scene.scene_snapshot(env, scene.CODEX_RADIUS1_V1)
        obstacles = np.asarray(snapshot["obstacles"])
        central_centers = scene.CENTRAL_CENTERS.copy()

    assert obstacles.shape == (69, 3)
    assert np.sum(np.all(np.isclose(obstacles, (2.5, 2.5, 1.0)), axis=1)) == 1
    for center in central_centers:
        assert not np.any(np.all(np.isclose(obstacles[:, :2], center[None]), axis=1))
    assert snapshot["start_state"][:2] == [0.5, 0.5]
    assert snapshot["goal"] == [4.5, 4.5]
    assert snapshot["sha256"] == "9b12258fc4c9a3631e3fdc2fccf0fe54dbe54fdc7d46e677dd7e8e360b32cf37"


def test_radius04_changes_only_the_sixteen_interior_disks() -> None:
    with _overnight_scene() as scene:
        base = scene.build_scene(scene.AFE2SceneProfile(
            name="test_base",
            start=(0.5, 0.5),
            goal=(4.5, 4.5),
            wall_plugs=8,
            center_replacement_radius=None,
            description="test only",
        ))
        ood = scene.build_scene(scene.CODEX_RADIUS04_V1)
        base_obstacles = base.obstacles.detach().cpu().numpy()
        ood_obstacles = ood.obstacles.detach().cpu().numpy()
        snapshot = scene.scene_snapshot(ood, scene.CODEX_RADIUS04_V1)
        scene.assert_scene_snapshot(snapshot)

    assert np.array_equal(base_obstacles[:, :2], ood_obstacles[:, :2])
    changed = ~np.isclose(base_obstacles[:, 2], ood_obstacles[:, 2])
    assert int(changed.sum()) == 16
    assert np.allclose(base_obstacles[changed, 2], 0.2)
    assert np.allclose(ood_obstacles[changed, 2], 0.4)
    assert np.array_equal(base_obstacles[~changed], ood_obstacles[~changed])
    assert snapshot["profile"]["interior_disk_radius"] == 0.4
    assert snapshot["start_state"][:2] == [0.5, 0.5]
    assert snapshot["goal"] == [4.5, 4.5]


def test_radius03_changes_only_the_sixteen_interior_disks() -> None:
    with _overnight_scene() as scene:
        base = scene.build_scene(scene.AFE2SceneProfile(
            name="test_base",
            start=(0.5, 0.5),
            goal=(4.5, 4.5),
            wall_plugs=8,
            center_replacement_radius=None,
            description="test only",
        ))
        ood = scene.build_scene(scene.CODEX_RADIUS03_V1)
        base_obstacles = base.obstacles.detach().cpu().numpy()
        ood_obstacles = ood.obstacles.detach().cpu().numpy()
        snapshot = scene.scene_snapshot(ood, scene.CODEX_RADIUS03_V1)
        scene.assert_scene_snapshot(snapshot)

    assert np.array_equal(base_obstacles[:, :2], ood_obstacles[:, :2])
    changed = ~np.isclose(base_obstacles[:, 2], ood_obstacles[:, 2])
    assert int(changed.sum()) == 16
    assert np.allclose(base_obstacles[changed, 2], 0.2)
    assert np.allclose(ood_obstacles[changed, 2], 0.3)
    assert np.array_equal(base_obstacles[~changed], ood_obstacles[~changed])
    assert snapshot["profile"]["interior_disk_radius"] == 0.3
    assert snapshot["start_state"][:2] == [0.5, 0.5]
    assert snapshot["goal"] == [4.5, 4.5]


def test_scene_profile_matches_canonical_restart_radius1() -> None:
    with _overnight_scene() as scene:
        ours = scene.build_scene(scene.CODEX_RADIUS1_V1)
        ours_values = (
            ours.obstacles.detach().cpu().numpy().copy(),
            ours.x0.detach().cpu().numpy().copy(),
            ours.goal.detach().cpu().numpy().copy(),
            float(ours.r_robot), float(ours.dt), float(ours.u_max),
        )
    with _isolated_import(
        (_RESTART_PARENT, _WORK, _REV), "afe_restart.scene"
    ) as canonical:
        reference = canonical.make_ood_scene(radius=1.0)
        reference_values = (
            reference.obstacles.detach().cpu().numpy().copy(),
            reference.x0.detach().cpu().numpy().copy(),
            reference.goal.detach().cpu().numpy().copy(),
            float(reference.r_robot), float(reference.dt), float(reference.u_max),
        )

    for actual, expected in zip(ours_values[:3], reference_values[:3]):
        assert np.array_equal(actual, expected)
    assert ours_values[3:] == reference_values[3:]
