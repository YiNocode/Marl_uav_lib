"""Tests for slot projection -> path replan signal wiring."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.planning.path_cache import AgentPathState, DeployPathCache, PathCacheConfig
from marl_uav.framework.reachability.slot_projection import detect_slot_projection_moves


def test_detect_slot_projection_moves() -> None:
    nominal = np.array([[0.0, 4.0, 1.0], [3.0, 0.0, 1.0], [-3.0, 0.0, 1.0]], dtype=np.float32)
    effective = nominal.copy()
    effective[0, 0] = 1.5
    moved = detect_slot_projection_moves(nominal, effective, None, threshold=0.05)
    assert moved[0]
    assert not moved[1]
    assert not moved[2]


def test_invalidate_slot_clears_cached_path() -> None:
    cache = DeployPathCache(PathCacheConfig())
    obs = [Obstacle(kind="circle", center=np.array([0.0, 0.0]), radius=1.0)]
    start = np.array([0.0, -4.0])
    goal = np.array([0.0, 4.0])
    cache.get_or_replan_assigned_path(
        0, 0, start, goal, obs, obs, obs,
        (-10.0, -10.0, 10.0, 10.0), {"planner_timeout_ms": 5.0},
        step_count=1, safety_margin=0.1, uav_radius=0.1, force=True,
    )
    assert cache.get_agent_path(0) is not None
    cache.invalidate_slot(0)
    assert cache.get_agent_path(0) is None


def test_invalid_cached_path_forces_replan_guard() -> None:
    cache = DeployPathCache(PathCacheConfig(replan_interval=100))
    obs = [Obstacle(kind="circle", center=np.array([0.0, 0.0]), radius=1.0)]
    cache.agent_paths[0] = AgentPathState(
        path=[np.array([-2.0, 0.0]), np.array([2.0, 0.0])],
        slot_id=0,
        start_xy=np.array([-2.0, 0.0]),
        goal_xy=np.array([2.0, 0.0]),
        target_xy=np.array([0.0, 0.0]),
        path_length=4.0,
        min_clearance=-1.0,
        mean_clearance=-1.0,
        planned_step=1,
        last_replan_step=1,
        los_blocked=True,
        feasible=True,
    )

    assert cache.should_replan(
        0, 0, np.array([-2.0, 0.0]), np.array([2.0, 0.0]), obs, 2,
        safety_margin=0.1, uav_radius=0.1,
    )
    assert cache.get_agent_path(0) is None


def test_cache_replans_when_target_or_slot_moves_or_endpoint_stale() -> None:
    cfg = PathCacheConfig(
        replan_interval=1000,
        replan_slot_move_thresh=1.0,
        replan_target_move_thresh=1.0,
        replan_endpoint_error_thresh=1.0,
        replan_min_clearance=-10.0,
        replan_tracking_error_thresh=10.0,
    )
    cache = DeployPathCache(cfg)
    cache.agent_paths[0] = AgentPathState(
        path=[np.array([0.0, 0.0]), np.array([1.0, 0.0])],
        slot_id=0,
        start_xy=np.array([0.0, 0.0]),
        goal_xy=np.array([1.0, 0.0]),
        target_xy=np.array([0.0, 0.0]),
        path_length=1.0,
        min_clearance=10.0,
        mean_clearance=10.0,
        planned_step=1,
        last_replan_step=1,
        feasible=True,
    )
    assert cache.should_replan(
        0, 0, np.array([0.0, 0.0]), np.array([2.2, 0.0]), [], 2,
        safety_margin=0.0, uav_radius=0.0, target_xy=np.array([0.0, 0.0]), speed=1.0,
    )
    cache.agent_paths[0].goal_xy = np.array([1.0, 0.0])
    assert cache.should_replan(
        0, 0, np.array([0.0, 0.0]), np.array([1.0, 0.0]), [], 2,
        safety_margin=0.0, uav_radius=0.0, target_xy=np.array([2.2, 0.0]), speed=1.0,
    )
    assert cache.should_replan(
        0, 0, np.array([0.0, 0.0]), np.array([2.2, 0.0]), [], 2,
        safety_margin=0.0, uav_radius=0.0, target_xy=np.array([0.0, 0.0]), speed=1.0,
    )


def test_cache_replans_when_cbf_active_too_long() -> None:
    cfg = PathCacheConfig(
        replan_interval=1000,
        replan_cbf_active_steps=3,
        replan_min_clearance=-10.0,
        replan_tracking_error_thresh=10.0,
    )
    cache = DeployPathCache(cfg)
    cache.agent_paths[0] = AgentPathState(
        path=[np.array([0.0, 0.0]), np.array([1.0, 0.0])],
        slot_id=0,
        start_xy=np.array([0.0, 0.0]),
        goal_xy=np.array([1.0, 0.0]),
        target_xy=np.array([0.0, 0.0]),
        path_length=1.0,
        min_clearance=10.0,
        mean_clearance=10.0,
        planned_step=1,
        last_replan_step=1,
        feasible=True,
    )
    assert cache.should_replan(
        0, 0, np.array([0.0, 0.0]), np.array([1.0, 0.0]), [], 2,
        safety_margin=0.0, uav_radius=0.0, target_xy=np.array([0.0, 0.0]),
        cbf_active_steps=3, speed=1.0,
    )


def test_failed_plan_does_not_store_direct_collision_path() -> None:
    cache = DeployPathCache(PathCacheConfig())
    obs = [Obstacle(kind="circle", center=np.array([0.0, 0.0]), radius=1.0)]
    path, replanned, _ms = cache.get_or_replan_assigned_path(
        0, 0, np.array([-0.5, 0.0]), np.array([0.5, 0.0]), obs, obs, obs,
        (-10.0, -10.0, 10.0, 10.0), {"planner_timeout_ms": 5.0},
        step_count=1, safety_margin=0.1, uav_radius=0.1, force=True,
    )
    assert path is None
    assert not replanned
    assert cache.get_agent_path(0) is None
