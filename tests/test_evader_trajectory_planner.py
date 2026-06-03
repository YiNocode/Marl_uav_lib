"""Tests for escape-oriented evader trajectory planner."""

from __future__ import annotations

import numpy as np

from marl_uav.envs.tasks.evader_trajectory_planner import (
    _min_pursuer_dist,
    path_hold_altitude,
    plan_sharp_turn_evader_path,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex3 import PursuitEvasion3v1Task


def test_escape_path_increases_pursuer_distance() -> None:
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pursuers = np.array(
        [
            [-6.0, 0.0, 1.0],
            [-5.0, 2.0, 1.0],
            [-5.0, -2.0, 1.0],
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(0)
    path = plan_sharp_turn_evader_path(
        evader,
        pursuers,
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        world_xy=20.0,
        rng=rng,
        num_legs=3,
        min_leg_m=6.0,
        min_turn_deg=45.0,
        arena_margin_ratio=0.1,
    )
    start_d = _min_pursuer_dist(evader[:2], pursuers[:, :2])
    end_d = _min_pursuer_dist(path[-1, :2], pursuers[:, :2])
    assert end_d > start_d
    # net motion should be away from pursuer cluster (+x)
    assert float(path[-1, 0] - path[0, 0]) > 2.0


def test_planned_path_holds_constant_altitude() -> None:
    evader = np.array([0.0, 0.0, 1.05], dtype=np.float64)
    pursuers = np.array([[-6.0, 0.0, 1.0], [-5.0, 2.0, 1.0], [-5.0, -2.0, 1.0]], dtype=np.float64)
    rng = np.random.default_rng(7)
    path = plan_sharp_turn_evader_path(
        evader,
        pursuers,
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        world_xy=20.0,
        rng=rng,
        z_min=0.95,
        z_max=1.10,
        num_legs=3,
        min_leg_m=6.0,
    )
    hold_z = path_hold_altitude(path)
    assert np.isclose(hold_z, 1.05, rtol=0.0, atol=1e-5)
    assert np.allclose(path[:, 2], hold_z)


def test_ex3_sharp_turn_plans_path_at_reset() -> None:
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        obstacle_grid_spacing=8.0,
        evader_sharp_turn_enabled=True,
        init_random_bias_enabled=False,
        evader_planned_path_num_legs=3,
        evader_planned_path_min_leg_m=6.0,
    )
    rng = np.random.default_rng(1)
    start_pos, _, state = task.sample_initial_conditions(4, rng)
    assert state.evader_planned_path.shape[0] >= 2
    evader_z = float(start_pos[state.evader_id, 2])
    assert np.allclose(state.evader_planned_path[:, 2], evader_z)
    pursuer_xy = start_pos[state.pursuer_ids, :2]
    evader_xy = start_pos[state.evader_id, :2]
    d0 = _min_pursuer_dist(evader_xy, pursuer_xy)
    d1 = _min_pursuer_dist(state.evader_planned_path[-1, :2], pursuer_xy)
    assert d1 >= d0 - 0.5
