from __future__ import annotations

import numpy as np

from marl_uav.envs.backends.genesis_backend import GenesisBackend


def test_genesis_velocity_setpoints_map_to_xyz_velocity():
    backend = GenesisBackend.__new__(GenesisBackend)
    backend.velocity_low = np.array([-1.0, -1.0, -0.1, -0.5], dtype=np.float32)
    backend.velocity_high = np.array([1.0, 1.0, 0.1, 0.5], dtype=np.float32)

    actions = np.array(
        [
            [
                [0.25, -0.5, 0.07, 0.15],
                [2.0, -2.0, 0.2, -1.0],
            ]
        ],
        dtype=np.float32,
    )

    cmd = backend._actions_to_velocity_commands(actions)

    expected = np.array(
        [
            [
                [0.25, -0.5, 0.15],
                [1.0, -1.0, -0.5],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(cmd, expected)


def test_genesis_set_pos_command_integrates_velocity_with_dt():
    backend = GenesisBackend.__new__(GenesisBackend)
    backend.dt = 0.01
    backend.world_xy = 2.0
    backend.z_min = 0.5
    backend.z_max = 1.5

    current_pos = np.array([[0.0, 0.0, 1.0], [1.99, -1.99, 0.51]], dtype=np.float32)
    vel_cmd = np.array([[0.25, -0.5, 0.15], [2.0, -2.0, -5.0]], dtype=np.float32)

    cmd = backend._velocity_to_set_pos_command(current_pos, vel_cmd)

    expected = np.array([[0.0025, -0.005, 1.0015], [2.0, -2.0, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(cmd, expected)
