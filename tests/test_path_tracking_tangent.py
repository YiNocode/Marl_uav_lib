"""Tests for path tangent bearing selection."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.planning.path_tracking import path_tangent_bearing


def test_path_tangent_uses_forward_segment() -> None:
    path = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    bearing = path_tangent_bearing(path, np.array([0.1, 0.0]))
    assert abs(bearing - 0.0) < 1e-5

    bearing_turn = path_tangent_bearing(path, np.array([0.95, 0.05]))
    assert abs(bearing_turn - np.pi / 2) < 0.2 or abs(bearing_turn - 0.0) < 0.2


def test_path_tangent_accepts_xyz_waypoints() -> None:
    path = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    bearing = path_tangent_bearing(path, np.array([0.1, 0.0]))
    assert abs(bearing - 0.0) < 1e-5
