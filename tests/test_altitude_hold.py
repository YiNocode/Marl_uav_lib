"""Tests for hard altitude hold execution layer."""

from __future__ import annotations

import pytest
import numpy as np

from marl_uav.control.altitude_hold import apply_hard_altitude_to_action_row, hard_altitude_hold
from marl_uav.control.geometric_pursuit_baselines import (
    proportional_actions_to_targets,
    tube_tracking_body_actions,
)


def _with_hard_altitude(acts, p, g, low, high, *, gate_horizontal=True):
    out = np.asarray(acts, dtype=np.float32).copy()
    for i in range(3):
        apply_hard_altitude_to_action_row(
            out[i], float(p[i, 2]), float(g[i, 2]), low, high,
            gate_horizontal=gate_horizontal,
        )
    return np.clip(out, low[None, :], high[None, :])


def test_hard_altitude_saturates_vz_and_gates_yaw() -> None:
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    p[:, 2] = 0.80
    g = p.copy()
    g[:, 0] = 2.0
    g[:, 2] = 1.05
    yaw = np.zeros((3,), dtype=np.float32)
    acts = proportional_actions_to_targets(
        p, g, low, high, xy_gain=0.25, z_gain=0.20, pursuer_yaw=yaw, yaw_gain=0.25,
    )
    acts = _with_hard_altitude(acts, p, g, low, high)
    np.testing.assert_allclose(acts[:, 3], 0.15, atol=1e-5)
    assert float(np.max(np.abs(acts[:, 2]))) < 1e-5
    assert float(np.max(np.abs(acts[:, 0]))) < 1e-5


def test_hard_altitude_no_vz_inside_deadband() -> None:
    vz, gate = hard_altitude_hold(1.00, 1.01, -0.15, 0.15, deadband=0.025)
    assert vz == 0.0
    assert gate == 1.0


def test_hard_altitude_floor_margin_overrides_low_target() -> None:
    vz, gate = hard_altitude_hold(
        0.56,
        0.52,
        -0.15,
        0.15,
        z_floor=0.5,
        floor_margin=0.25,
    )
    assert vz == pytest.approx(0.15)
    assert gate < 1.0


def test_apply_hard_altitude_floor_margin() -> None:
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    row = np.array([0.20, 0.0, 0.10, -0.10], dtype=np.float32)
    apply_hard_altitude_to_action_row(
        row,
        z=0.58,
        z_hold=0.52,
        action_low=low,
        action_high=high,
        z_floor=0.5,
        floor_margin=0.25,
    )
    assert float(row[3]) == pytest.approx(0.15)


def test_obstacle_priority_skips_horizontal_gate() -> None:
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    p[:, 2] = 0.80
    g = p.copy()
    g[:, 2] = 1.05
    row = np.array([0.20, 0.0, 0.10, 0.0], dtype=np.float32)
    apply_hard_altitude_to_action_row(
        row, float(p[0, 2]), float(g[0, 2]), low, high, gate_horizontal=False,
    )
    assert float(row[0]) == pytest.approx(0.20)
    assert float(row[2]) == pytest.approx(0.10)
    np.testing.assert_allclose(row[3], 0.15, atol=1e-5)


def test_tube_tracking_hard_altitude() -> None:
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    p[:, 2] = 0.75
    g = p.copy()
    g[:, 2] = 1.05
    path = [[[0.0, 0.0], [4.0, 0.0]]] * 3
    acts = tube_tracking_body_actions(
        p, g, low, high, path,
        xy_gain=0.25, z_gain=0.20, pursuer_yaw=np.zeros(3, dtype=np.float32), yaw_gain=0.25,
    )
    acts = _with_hard_altitude(acts, p, g, low, high)
    np.testing.assert_allclose(acts[:, 3], 0.15, atol=1e-5)
    assert float(np.max(np.abs(acts[:, 2]))) < 1e-5
