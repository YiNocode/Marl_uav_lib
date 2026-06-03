"""Tests for structure slot selection and evader obstacle APF."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import PursuitEvasion3v1Task
from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.reachability.structure_assignment_cost import StructureAssignmentConfig
from marl_uav.framework.reachability.structure_slot_selection import (
    StructureSlotSelectionConfig,
    select_structure_manifold_slots,
)


class _ObstacleTask(PursuitEvasion3v1Task):
    """Minimal ex1-like hooks for manifold geometry in tests."""

    def __init__(self) -> None:
        super().__init__(world_xy=20.0)
        self.manifold_target_phase = 0.0
        self.manifold_curve_num_samples = 48
        self.manifold_target_radius_scale = 1.0
        self.manifold_contraction_rate = 0.0
        self.manifold_structure_gate_scale = 0.0
        self.manifold_target_rho_min = 1.0
        self.manifold_target_rho_max = None
        self.capture_dist = 1.0

    def _compute_target_radius_xy(self, pursuer_pos, evader_pos, task_state=None) -> float:
        del pursuer_pos, evader_pos, task_state
        return 3.0

    def _obstacle_aware_radius(self, ang, rho_base, evader_pos, task_state=None):
        del evader_pos, task_state
        return np.full(np.asarray(ang).shape, np.float32(rho_base), dtype=np.float32)


def test_structure_slot_selection_expands_radius_when_needed() -> None:
    task = _ObstacleTask()
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pursuers = np.array(
        [
            [5.0, 0.0, 1.0],
            [-2.5, 4.33, 1.0],
            [-2.5, -4.33, 1.0],
        ],
        dtype=np.float32,
    )
    obstacles = [Obstacle(kind="circle", center=np.array([2.0, 0.0]), radius=1.5)]
    cfg = StructureSlotSelectionConfig(
        enabled=True,
        radius_scales=(1.0, 1.5, 2.0),
        max_radius_scale=2.0,
        w_radius_penalty=0.0,
    )
    struct_cfg = StructureAssignmentConfig(structure_scale=10.0, w_travel=0.0)
    targets, _curve, diag = select_structure_manifold_slots(
        task, pursuers, evader, obstacles, None, cfg, struct_cfg,
        previous_assignment=None, safety_margin=0.2, uav_radius=0.15, switch_penalty=0.0,
    )
    assert targets.shape == (3, 3)
    assert float(diag.get("manifold_radius_scale", 1.0)) >= 1.0
    radii = np.linalg.norm(targets[:, :2] - evader[:2], axis=1)
    assert float(np.mean(radii)) >= 2.5


def test_evader_obstacle_repulsion_points_away() -> None:
    task = PursuitEvasion3v1Task(world_xy=20.0, evader_apf_pursuer_gain=2.0)
    state = SimpleNamespace(
        obstacle_xy=np.array([[0.0, 0.0]], dtype=np.float32),
        obstacle_r=np.array([2.0], dtype=np.float32),
    )
    evader = np.array([1.5, 0.0, 1.0], dtype=np.float32)
    force, threat = task._evader_apf_obstacle_repulsion(evader, state)
    assert threat > 0.0
    assert float(force[0]) > 0.0
