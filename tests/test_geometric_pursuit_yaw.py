"""Tests for bearing-aligned heuristic pursuit control."""

from __future__ import annotations

import numpy as np

from marl_uav.control.geometric_pursuit_baselines import proportional_actions_to_targets


def test_legacy_decoupled_xy_when_yaw_disabled() -> None:
    low = np.array([-0.25, -0.25, -0.01, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.01, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    g = p.copy()
    g[:, 0] = 5.0
    acts = proportional_actions_to_targets(p, g, low, high, xy_gain=0.25, z_gain=0.20)
    np.testing.assert_allclose(acts[:, 0], 0.25)
    np.testing.assert_allclose(acts[:, 1], 0.0)
    np.testing.assert_allclose(acts[:, 2], 0.0)


def test_bearing_aligned_xy_matches_axis_target() -> None:
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    g = np.zeros((3, 3), dtype=np.float32)
    g[:, 0] = 5.0
    yaw = np.zeros((3,), dtype=np.float32)
    acts = proportional_actions_to_targets(
        p,
        g,
        low,
        high,
        xy_gain=0.25,
        z_gain=0.20,
        pursuer_yaw=yaw,
        yaw_gain=0.25,
    )
    np.testing.assert_allclose(acts[:, 0], 0.25, atol=1e-5)
    np.testing.assert_allclose(acts[:, 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(acts[:, 2], 0.0, atol=1e-5)


def test_yaw_tracks_bearing_error() -> None:
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    g = np.zeros((3, 3), dtype=np.float32)
    g[:, 1] = 5.0
    yaw = np.zeros((3,), dtype=np.float32)
    acts = proportional_actions_to_targets(
        p,
        g,
        low,
        high,
        xy_gain=0.25,
        z_gain=0.20,
        pursuer_yaw=yaw,
        yaw_gain=0.25,
    )
    expected_yaw = min(0.25 * (np.pi / 2.0), 0.25)
    np.testing.assert_allclose(acts[:, 2], expected_yaw, atol=1e-5)


def test_path_following_full_world_speed_despite_yaw_misalign() -> None:
    """Holonomic body cmd: world speed stays at cap even when yaw != path tangent."""
    from marl_uav.control.geometric_pursuit_baselines import path_following_body_actions

    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    g = p.copy()
    g[:, 0] = 5.0
    yaw = np.full((3,), np.pi / 4, dtype=np.float32)
    path = [[[0.0, 0.0], [5.0, 0.0]]] * 3
    acts = path_following_body_actions(
        p, g, low, high, path,
        xy_gain=0.25, z_gain=0.20, pursuer_yaw=yaw, yaw_gain=0.25,
        approach_dist=0.5,
    )
    world_vx = float(acts[0, 0]) * np.cos(np.pi / 4) - float(acts[0, 1]) * np.sin(np.pi / 4)
    np.testing.assert_allclose(world_vx, 0.25, atol=0.02)


def test_path_cruise_full_speed_at_lookahead() -> None:
    """Path tracking: at lookahead distance, command full speed (not xy_gain * dist)."""
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    g = np.zeros((3, 3), dtype=np.float32)
    g[:, 0] = 0.8
    yaw = np.zeros((3,), dtype=np.float32)
    acts_old = proportional_actions_to_targets(
        p, g, low, high, xy_gain=0.25, z_gain=0.20, pursuer_yaw=yaw, yaw_gain=0.25,
        path_cruise=False,
    )
    acts_cruise = proportional_actions_to_targets(
        p, g, low, high, xy_gain=0.25, z_gain=0.20, pursuer_yaw=yaw, yaw_gain=0.25,
        path_cruise=True, path_cruise_dist=0.8,
    )
    assert float(acts_old[0, 0]) < 0.21
    np.testing.assert_allclose(acts_cruise[:, 0], 0.25, atol=1e-5)
