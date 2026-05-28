from __future__ import annotations

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import PursuitEvasion3v1Task


def _task(mode: str) -> PursuitEvasion3v1Task:
    return PursuitEvasion3v1Task(
        world_xy=20.0,
        capture_dist=1.0,
        episode_limit=100,
        pursuer_speed=0.15,
        evader_speed=0.2,
        role_assignment_mode=mode,
        ot_epsilon=0.05,
        ot_epsilon_scale=0.25,
        ot_sinkhorn_iterations=30,
        assignment_inertia_margin=0.0,
        progress_reward_scale=0.0,
        capture_bonus=0.0,
    )


def test_entropic_ot_can_differ_from_nearest_assignment() -> None:
    """Sanity: OT and brute-force nearest need not always pick the same perm."""
    pursuer_pos = np.array(
        [
            [0.0, 3.0, 1.0],
            [2.5, -1.0, 1.0],
            [-2.0, -2.0, 1.0],
        ],
        dtype=np.float32,
    )
    evader_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    task_ot = _task("entropic_ot")
    task_nn = _task("nearest")
    _, assign_ot, _ = task_ot._assigned_targets_from_state(pursuer_pos, evader_pos)
    _, assign_nn, _ = task_nn._assigned_targets_from_state(pursuer_pos, evader_pos)

    assert assign_ot.shape == (3,)
    assert assign_nn.shape == (3,)
    assert len(np.unique(assign_ot)) == 3
    assert len(np.unique(assign_nn)) == 3
