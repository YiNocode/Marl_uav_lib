"""Tests for ex3 sharp-turn path and random initial bias."""

from __future__ import annotations

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex3 import PursuitEvasion3v1Task


def test_ex3_both_flags_off_matches_ex2_reset_shape() -> None:
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        obstacle_grid_spacing=8.0,
        evader_sharp_turn_enabled=False,
        init_random_bias_enabled=False,
    )
    rng = np.random.default_rng(0)
    start_pos, _, state = task.sample_initial_conditions(4, rng)
    assert start_pos.shape == (4, 3)
    assert state.obstacle_xy.shape[0] > 0
    assert state.evader_planned_path.shape[0] == 0


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
    _, _, state = task.sample_initial_conditions(4, rng)
    assert state.evader_planned_path.shape[0] >= 2


def test_ex3_random_bias_spreads_agents() -> None:
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        obstacle_grid_spacing=8.0,
        init_random_bias_enabled=True,
        evader_sharp_turn_enabled=False,
    )
    rng = np.random.default_rng(2)
    start_pos, _, state = task.sample_initial_conditions(4, rng)
    xs = start_pos[:, 0]
    assert float(np.std(xs)) > 1.0
    assert state.obstacle_xy.shape[0] > 0


def test_ex3_random_bias_keeps_pursuers_near_evader_altitude() -> None:
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        z_min=0.5,
        z_max=5.0,
        obstacle_grid_spacing=8.0,
        init_random_bias_enabled=True,
    )
    rng = np.random.default_rng(42)
    start_pos, _, _ = task.sample_initial_conditions(4, rng)
    evader_z = float(start_pos[3, 2])
    pursuer_z = start_pos[:3, 2]
    assert float(np.max(np.abs(pursuer_z - evader_z))) <= task.init_pursuer_noise_z + 0.05
    assert float(np.min(start_pos[:, 2])) >= 2.0
    assert float(np.max(start_pos[:, 2])) <= 3.0


def test_ex3_default_reset_initializes_agents_at_two_to_three_meters() -> None:
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        z_min=0.5,
        z_max=5.0,
        obstacle_grid_spacing=8.0,
        init_random_bias_enabled=False,
    )
    rng = np.random.default_rng(7)
    start_pos, _, _ = task.sample_initial_conditions(4, rng)
    assert float(np.min(start_pos[:, 2])) >= 2.0
    assert float(np.max(start_pos[:, 2])) <= 3.0


def test_ex3_sharp_turn_and_random_bias_together() -> None:
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        obstacle_grid_spacing=8.0,
        evader_sharp_turn_enabled=True,
        init_random_bias_enabled=True,
    )
    rng = np.random.default_rng(3)
    start_pos, _, state = task.sample_initial_conditions(4, rng)
    assert start_pos.shape == (4, 3)
    assert state.evader_planned_path.shape[0] >= 2
