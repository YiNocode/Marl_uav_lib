from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task import (
    PursuitEvasion3v1Task as BasePursuitEvasion3v1Task,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import PursuitEvasion3v1Task


def test_legacy_residual_control_kwargs_are_accepted_but_not_task_control():
    task = PursuitEvasion3v1Task(
        residual_control_gain=0.5,
        residual_control_gain_final=0.1,
        residual_control_gain_decay_epochs=5,
    )

    assert not hasattr(task, "set_training_progress")
    assert not hasattr(task, "residual_control_gain")


def test_ex1_action_to_setpoint_does_not_add_slot_residual():
    task = PursuitEvasion3v1Task()
    rng = np.random.default_rng(123)
    start_pos, _, task_state = task.sample_initial_conditions(num_agents=4, rng=rng)
    states = np.zeros((4, 4, 3), dtype=np.float32)
    states[:, 3, :] = start_pos
    backend_state = SimpleNamespace(
        states=states,
        contact_array=np.zeros((4, 4), dtype=np.int8),
    )
    actions = np.array(
        [
            [0.05, 0.01, 0.0, 0.02],
            [-0.04, 0.02, 0.0, -0.01],
            [0.03, -0.02, 0.0, 0.00],
        ],
        dtype=np.float32,
    )

    setpoints = task.action_to_setpoint(
        actions,
        backend_state,
        task_state,
        action_space_type="continuous",
        action_dim=4,
    )

    expected = np.zeros_like(actions)
    expected[:, 0] = actions[:, 0] / task.continuous_action_xy_ref * task.pursuer_speed_xy
    expected[:, 1] = actions[:, 1] / task.continuous_action_xy_ref * task.pursuer_speed_xy
    expected[:, 2] = actions[:, 2]
    expected[:, 3] = actions[:, 3] / task.continuous_action_z_ref * task.pursuer_speed_z
    np.testing.assert_allclose(setpoints[:3], expected, atol=1e-6)


def test_base_pursuit_action_to_setpoint_scales_continuous_actions():
    task = BasePursuitEvasion3v1Task()
    rng = np.random.default_rng(123)
    start_pos, _, task_state = task.sample_initial_conditions(num_agents=4, rng=rng)
    states = np.zeros((4, 4, 3), dtype=np.float32)
    states[:, 3, :] = start_pos
    backend_state = SimpleNamespace(
        states=states,
        contact_array=np.zeros((4, 4), dtype=np.int8),
    )
    actions = np.array(
        [
            [0.05, 0.01, 0.0, 0.02],
            [-0.04, 0.02, 0.0, -0.01],
            [0.03, -0.02, 0.0, 0.00],
        ],
        dtype=np.float32,
    )

    setpoints = task.action_to_setpoint(
        actions,
        backend_state,
        task_state,
        action_space_type="continuous",
        action_dim=4,
    )

    expected = np.zeros_like(actions)
    expected[:, 0] = actions[:, 0] / task.continuous_action_xy_ref * task.pursuer_speed_xy
    expected[:, 1] = actions[:, 1] / task.continuous_action_xy_ref * task.pursuer_speed_xy
    expected[:, 2] = actions[:, 2]
    expected[:, 3] = actions[:, 3] / task.continuous_action_z_ref * task.pursuer_speed_z
    np.testing.assert_allclose(setpoints[:3], expected, atol=1e-6)
