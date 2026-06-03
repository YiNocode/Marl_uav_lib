"""Tests for turn-radius-aware path tracking safety."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.planning.path_tracking import (
    adjust_lookahead_for_turn_safety,
    points_min_boundary_clearance,
    sampled_turn_arc,
)


def _circle(cx: float, cy: float, r: float) -> Obstacle:
    return Obstacle(kind="circle", center=np.array([cx, cy], dtype=np.float64), radius=r)


def test_sampled_turn_arc_has_expected_quarter_turn() -> None:
    pts, angle = sampled_turn_arc(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        turn_radius=1.0,
        num_samples=5,
    )
    assert pts.shape == (5, 2)
    assert abs(angle - np.pi / 2.0) < 1e-6
    np.testing.assert_allclose(pts[0], np.array([0.0, 0.0]), atol=1e-6)


def test_turn_safety_keeps_previous_tangent_when_arc_hits_obstacle() -> None:
    adjusted, diag = adjust_lookahead_for_turn_safety(
        np.array([0.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([1.0, 0.0]),
        [_circle(0.7, 0.3, 0.25)],
        uav_radius=0.15,
        turn_radius=1.0,
        min_turn_clearance=0.4,
        safe_forward_dist=0.5,
        num_samples=11,
    )
    assert diag["turn_safety_active"]
    np.testing.assert_allclose(adjusted, np.array([0.5, 0.0]), atol=1e-6)


def test_turn_safety_allows_clear_turn() -> None:
    desired = np.array([0.0, 2.0])
    adjusted, diag = adjust_lookahead_for_turn_safety(
        np.array([0.0, 0.0]),
        desired,
        np.array([1.0, 0.0]),
        [_circle(5.0, 5.0, 0.25)],
        uav_radius=0.15,
        turn_radius=1.0,
        min_turn_clearance=0.4,
        safe_forward_dist=0.5,
        num_samples=11,
    )
    assert not diag["turn_safety_active"]
    np.testing.assert_allclose(adjusted, desired, atol=1e-6)


def test_turn_safety_blocks_turn_arc_that_crosses_boundary() -> None:
    adjusted, diag = adjust_lookahead_for_turn_safety(
        np.array([9.4, 0.0]),
        np.array([10.5, 0.0]),
        np.array([0.0, 1.0]),
        [],
        uav_radius=0.15,
        turn_radius=1.0,
        min_turn_clearance=0.4,
        safe_forward_dist=0.5,
        num_samples=11,
        world_xy=10.0,
        boundary_margin=0.3,
        min_boundary_clearance=0.4,
    )
    assert diag["turn_safety_active"]
    assert diag["turn_boundary_unsafe"]
    np.testing.assert_allclose(adjusted, np.array([9.4, 0.5]), atol=1e-6)


def test_points_min_boundary_clearance_uses_usable_margin() -> None:
    clear = points_min_boundary_clearance(
        np.array([[9.5, 0.0], [9.8, 0.0]]),
        world_xy=10.0,
        boundary_margin=0.3,
    )
    assert abs(clear - (-0.1)) < 1e-6
