from __future__ import annotations

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import (
    PursuitEvasion3v1Task,
    PursuitEvasion3v1TaskEx2State,
)


def _make_state(obstacle_xy: np.ndarray, obstacle_r: np.ndarray) -> PursuitEvasion3v1TaskEx2State:
    return PursuitEvasion3v1TaskEx2State(
        pursuer_ids=np.array([0, 1, 2], dtype=np.int64),
        evader_id=3,
        captured=False,
        capture_agent=-1,
        prev_pursuer_dists=np.ones((3,), dtype=np.float32),
        obstacle_xy=np.asarray(obstacle_xy, dtype=np.float32).reshape(-1, 2),
        obstacle_r=np.asarray(obstacle_r, dtype=np.float32).reshape(-1),
    )


def test_ex2_reference_targets_match_ex1_when_no_obstacles():
    task = PursuitEvasion3v1Task()
    pursuer_pos = np.array(
        [[-4.0, -1.0, 1.0], [-4.0, 0.0, 1.0], [-4.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    evader_pos = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    task_state = _make_state(np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32))

    targets_ex2 = task._reference_manifold_targets(pursuer_pos, evader_pos, task_state=task_state)
    targets_ex1 = super(PursuitEvasion3v1Task, task)._reference_manifold_targets(
        pursuer_pos,
        evader_pos,
        task_state=task_state,
    )

    np.testing.assert_allclose(targets_ex2, targets_ex1, atol=1e-5, rtol=1e-5)


def test_ex2_obstacle_aware_manifold_pushes_slot_away_from_blocked_ray():
    task = PursuitEvasion3v1Task(
        manifold_target_phase=0.0,
        obstacle_manifold_top_k=1,
        obstacle_manifold_fourier_scale=0.0,
        obstacle_manifold_bump_scale=0.0,
    )
    pursuer_pos = np.array(
        [[-4.0, -1.0, 1.0], [-4.0, 0.0, 1.0], [-4.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    evader_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    task_state = _make_state(np.array([[1.6, 0.0]], dtype=np.float32), np.array([0.45], dtype=np.float32))

    targets = task._reference_manifold_targets(pursuer_pos, evader_pos, task_state=task_state)
    slot_radii = np.linalg.norm(targets[:, :2] - evader_pos[None, :2], axis=1)
    rho_base = task._compute_target_radius_xy(pursuer_pos, evader_pos, task_state=task_state)

    assert slot_radii[0] > rho_base
    assert slot_radii[0] > slot_radii[1]
    assert slot_radii[0] > slot_radii[2]
