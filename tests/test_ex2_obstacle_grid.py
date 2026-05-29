from __future__ import annotations

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import PursuitEvasion3v1Task


def test_obstacle_grid_covers_full_arena():
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        obstacle_grid_spacing=4.0,
    )
    r_lo, r_hi = task._obstacle_r_min_max()
    r = 0.5 * (r_lo + r_hi)
    grid = task._grid_obstacle_centers(r)

    # world_xy=20, spacing=4 → 11×11 网格
    assert len(grid) == 11 * 11
    assert grid[0] == (-20.0, 20.0)
    assert grid[1] == (-16.0, 20.0)
    assert grid[10] == (20.0, 20.0)
    assert grid[11] == (-20.0, 16.0)
    assert grid[-1] == (20.0, -20.0)

    rng = np.random.default_rng(0)
    xy, radii = task._sample_obstacles(rng)
    assert xy.shape == (121, 2)
    assert radii.shape == (121,)
    assert task.num_obstacles_min == 121
    assert task.num_obstacles_max >= 121
    np.testing.assert_allclose(xy[0], [-20.0, 20.0])
    np.testing.assert_allclose(xy[11], [-20.0, 16.0])
    assert np.allclose(radii, r)
