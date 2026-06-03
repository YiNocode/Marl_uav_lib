"""Tests for structure-first SCE assignment costs."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.reachability.structure_assignment_cost import (
    StructureAssignmentConfig,
    assignment_structure_score,
    select_structure_assignment,
)


def _circle(cx: float, cy: float, r: float) -> Obstacle:
    return Obstacle(kind="circle", center=np.array([cx, cy], dtype=np.float64), radius=r)


def test_structure_assignment_prefers_uniform_encirclement() -> None:
    """Balanced slot assignment should beat collapsed geometry when all pairs reachable."""
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pursuers = np.array(
        [
            [3.0, 0.0, 1.0],
            [-1.5, 2.6, 1.0],
            [-1.5, -2.6, 1.0],
        ],
        dtype=np.float64,
    )
    slots = np.array(
        [
            [2.0, 0.0, 1.0],
            [-1.0, 1.73, 1.0],
            [-1.0, -1.73, 1.0],
        ],
        dtype=np.float64,
    )
    cfg = StructureAssignmentConfig(w_cov=1.0, w_col=1.0, w_ang=1.0, structure_scale=10.0, w_travel=0.0)
    assign, diag = select_structure_assignment(
        pursuers, slots, evader, [], None, cfg,
        safety_margin=0.1, uav_radius=0.1, switch_penalty=0.0,
        exclude_unreachable=False,
    )
    identity_score = assignment_structure_score(pursuers, slots, np.arange(3), evader, cfg)
    assigned_score = float(diag["assigned_structure_score"])
    assert assigned_score >= identity_score - 1e-6
    assert assign.shape == (3,)


def test_unreachable_pair_excluded_from_permutation() -> None:
    """Slot behind obstacle should not be assigned to pursuer without LOS/path."""
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pursuers = np.array(
        [
            [-4.0, 0.0, 1.0],
            [0.0, -4.0, 1.0],
            [4.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    slots = np.array(
        [
            [0.0, 4.0, 1.0],
            [4.0, 0.0, 1.0],
            [0.0, 3.5, 1.0],
        ],
        dtype=np.float64,
    )
    obstacles = [_circle(0.0, 0.0, 1.2)]
    cfg = StructureAssignmentConfig(structure_scale=10.0, w_travel=0.0)
    assign, diag = select_structure_assignment(
        pursuers, slots, evader, obstacles, None, cfg,
        safety_margin=0.1, uav_radius=0.1, switch_penalty=0.0,
        exclude_unreachable=True,
    )
    reachable = diag["pair_reachable_matrix"]
    for i in range(3):
        assert bool(reachable[i, int(assign[i])])
