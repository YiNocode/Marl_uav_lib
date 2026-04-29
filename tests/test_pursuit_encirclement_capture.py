import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import (
    PursuitEvasion3v1Task,
    compute_pursuit_structure_metrics_3v1,
)


def _pursuers_from_angles(radius: float, angles_deg: list[float]) -> np.ndarray:
    pursuers = []
    for ang_deg in angles_deg:
        ang = np.deg2rad(float(ang_deg))
        pursuers.append([radius * np.cos(ang), radius * np.sin(ang), 1.0])
    return np.asarray(pursuers, dtype=np.float32)


def test_encirclement_capture_accepts_tight_ring():
    task = PursuitEvasion3v1Task(capture_dist=1.0, world_xy=20.0, reference_world_xy=2.0)
    evader_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pursuer_pos = _pursuers_from_angles(radius=1.4, angles_deg=[0.0, 120.0, 240.0])
    metrics = compute_pursuit_structure_metrics_3v1(pursuer_pos, evader_pos)

    assert task._encirclement_capture_satisfied(
        struct_metrics=metrics,
        hold_steps=task.encirclement_capture_hold_steps,
        min_dist=1.4,
        mean_radius_xy=1.4,
        target_radius_xy=1.0,
    )


def test_encirclement_capture_rejects_large_escape_gap():
    task = PursuitEvasion3v1Task(capture_dist=1.0, world_xy=20.0, reference_world_xy=2.0)
    evader_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pursuer_pos = _pursuers_from_angles(radius=1.4, angles_deg=[0.0, 90.0, 240.0])
    metrics = compute_pursuit_structure_metrics_3v1(pursuer_pos, evader_pos)

    assert not task._encirclement_capture_satisfied(
        struct_metrics=metrics,
        hold_steps=task.encirclement_capture_hold_steps,
        min_dist=1.4,
        mean_radius_xy=1.4,
        target_radius_xy=1.0,
    )
