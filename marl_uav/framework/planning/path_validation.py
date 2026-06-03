"""Post-planner path validation against obstacle sets."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle, collision_check_path


def validate_planned_path(
    path: list[np.ndarray] | np.ndarray,
    obstacles: list[Obstacle],
    *,
    safety_margin: float,
    uav_radius: float,
) -> bool:
    """Return True if ``path`` is collision-free w.r.t. ``obstacles``."""
    if not obstacles:
        return True
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return True
    return collision_check_path(
        pts, obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
    )
