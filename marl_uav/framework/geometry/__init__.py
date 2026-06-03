"""2D obstacle geometry utilities for deployable pursuit baselines."""

from marl_uav.framework.geometry.obstacle_adapter import (
    local_obstacles_for_pursuers,
    nearest_obstacles,
    obstacles_from_task_state,
)
from marl_uav.framework.geometry.obstacle_geometry import (
    CircleObstacle,
    Obstacle,
    collision_check_path,
    has_line_of_sight,
    inflate_obstacle,
    line_segment_intersects_obstacle,
    path_length,
)

__all__ = [
    "CircleObstacle",
    "Obstacle",
    "collision_check_path",
    "has_line_of_sight",
    "inflate_obstacle",
    "line_segment_intersects_obstacle",
    "local_obstacles_for_pursuers",
    "nearest_obstacles",
    "obstacles_from_task_state",
    "path_length",
]
