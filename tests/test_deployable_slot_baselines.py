"""Deployable slot baseline controllers (Hungarian / OT)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from marl_uav.control.geometric_pursuit_baselines import (
    hungarian_slot_actions_from_state,
    ot_slot_actions_from_state,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import PursuitEvasion3v1Task


def _mock_env(task: PursuitEvasion3v1Task, pursuer_pos: np.ndarray, evader_pos: np.ndarray):
    state = SimpleNamespace(
        pursuer_ids=np.array([0, 1, 2], dtype=np.int64),
        evader_id=3,
        assigned_target_indices=np.array([0, 1, 2], dtype=np.int64),
    )
    lin_pos = np.zeros((4, 3), dtype=np.float32)
    lin_pos[:3] = pursuer_pos
    lin_pos[3] = evader_pos
    return SimpleNamespace(task=task, task_state=state, prev_backend_state=None), lin_pos


def test_hungarian_and_ot_controllers_force_assignment_mode() -> None:
    pursuer_pos = np.array(
        [
            [0.0, 3.0, 1.0],
            [2.5, -1.0, 1.0],
            [-2.0, -2.0, 1.0],
        ],
        dtype=np.float32,
    )
    evader_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    task_h = PursuitEvasion3v1Task(
        world_xy=20.0,
        role_assignment_mode="entropic_ot",
        assignment_inertia_margin=0.0,
    )
    task_o = PursuitEvasion3v1Task(
        world_xy=20.0,
        role_assignment_mode="nearest",
        assignment_inertia_margin=0.0,
    )
    env_h, lin_pos = _mock_env(task_h, pursuer_pos, evader_pos)
    env_o, _ = _mock_env(task_o, pursuer_pos, evader_pos)

    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    hungarian_slot_actions_from_state(
        env_h, lin_pos, low, high, xy_gain=0.25, z_gain=0.20, yaw_gain=0.0
    )
    ot_slot_actions_from_state(
        env_o, lin_pos, low, high, xy_gain=0.25, z_gain=0.20, yaw_gain=0.0
    )

    _, assign_h, _ = task_h._assigned_targets_from_state(
        pursuer_pos, evader_pos, task_state=env_h.task_state, role_assignment_mode="nearest"
    )
    _, assign_o, _ = task_o._assigned_targets_from_state(
        pursuer_pos, evader_pos, task_state=env_o.task_state, role_assignment_mode="entropic_ot"
    )
    np.testing.assert_array_equal(env_h.task_state.assigned_target_indices, assign_h)
    np.testing.assert_array_equal(env_o.task_state.assigned_target_indices, assign_o)


def test_hungarian_slot_matches_task_nearest_assignment() -> None:
    pursuer_pos = np.array(
        [
            [0.0, 3.0, 1.0],
            [2.5, -1.0, 1.0],
            [-2.0, -2.0, 1.0],
        ],
        dtype=np.float32,
    )
    evader_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        role_assignment_mode="entropic_ot",
        assignment_inertia_margin=0.0,
    )
    env, lin_pos = _mock_env(task, pursuer_pos, evader_pos)
    low = np.full(4, -1.0, dtype=np.float32)
    high = np.full(4, 1.0, dtype=np.float32)

    hungarian_slot_actions_from_state(
        env, lin_pos, low, high, xy_gain=0.1, z_gain=0.1, yaw_gain=0.0
    )
    _, expected, _ = task._assigned_targets_from_state(
        pursuer_pos,
        evader_pos,
        task_state=env.task_state,
        role_assignment_mode="nearest",
    )
    np.testing.assert_array_equal(env.task_state.assigned_target_indices, expected)
