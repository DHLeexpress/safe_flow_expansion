"""Package-local construction of the established eight-plug ID stadium."""
from __future__ import annotations

import torch

import grid_scene as GS


_WALL_STEP = 5.0 / 13.0
_WALL_PLUGS4 = (
    (_WALL_STEP, -0.2, 0.2),
    (5.0 - _WALL_STEP, 5.2, 0.2),
    (-0.2, _WALL_STEP, 0.2),
    (5.2, 5.0 - _WALL_STEP, 0.2),
)
_WALL_PLUGS8 = _WALL_PLUGS4 + (
    (0.0, -0.2, 0.2),
    (-0.2, 0.0, 0.2),
    (5.2, 5.0, 0.2),
    (5.0, 5.2, 0.2),
)


def make_walled_env(wall_plugs: int = 8):
    """Apply the historical plug geometry without a bare-module import."""

    env = GS.make_grid()
    if not wall_plugs:
        return env
    plugs = (
        _WALL_PLUGS4[:2]
        if wall_plugs == 2
        else _WALL_PLUGS8
        if wall_plugs == 8
        else _WALL_PLUGS4
    )
    extra = torch.tensor(
        plugs, dtype=env.obstacles.dtype, device=env.obstacles.device
    )
    env.obstacles = torch.cat((env.obstacles, extra), dim=0)
    return env
