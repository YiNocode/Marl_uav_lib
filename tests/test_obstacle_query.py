"""Tests for corridor / CBF obstacle queries."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.geometry.obstacle_query import (
    obstacles_in_corridor,
    select_cbf_obstacles,
    select_plan_obstacles,
    select_validation_obstacles,
)
from marl_uav.framework.geometry.obstacle_query import ObstacleQueryConfig
from marl_uav.framework.planning.path_validation import validate_planned_path


def _obs(cx: float, cy: float, r: float) -> Obstacle:
    return Obstacle(kind="circle", center=np.array([cx, cy], dtype=np.float64), radius=r)


def test_corridor_includes_obstacle_on_segment() -> None:
    obs = [_obs(0.0, 0.0, 0.5), _obs(10.0, 0.0, 0.5)]
    corridor = obstacles_in_corridor(
        np.array([-2.0, 0.0]), np.array([2.0, 0.0]), obs, half_width=1.0,
    )
    assert len(corridor) == 1
    assert float(corridor[0].center[0]) == 0.0


def test_validation_full_mode() -> None:
    obs = [_obs(0.0, 0.0, 0.5), _obs(10.0, 0.0, 0.5)]
    cfg = ObstacleQueryConfig(validation_mode="full")
    out = select_validation_obstacles(np.zeros(2), np.ones(2), obs, cfg)
    assert len(out) == 2


def test_cbf_forward_query_filters_behind_agent() -> None:
    obs = [_obs(2.0, 0.0, 0.5), _obs(-2.0, 0.0, 0.5)]
    picked, idx = select_cbf_obstacles(
        np.zeros(2), 0.0, np.array([0.25, 0.0]), obs,
        activation_radius=0.5,
        forward_range=3.0,
        forward_cone_half_deg=90.0,
        mode="radius_forward",
    )
    assert 0 in idx
    assert -2.0 not in [float(o.center[0]) for o in picked if o.center[0] < 0]


def test_path_validation_rejects_through_obstacle() -> None:
    obs = [_obs(0.0, 0.0, 1.0)]
    path = [np.array([-3.0, 0.0]), np.array([3.0, 0.0])]
    assert not validate_planned_path(path, obs, safety_margin=0.1, uav_radius=0.15)
