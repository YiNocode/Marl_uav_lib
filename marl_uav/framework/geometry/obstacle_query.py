"""Spatial obstacle queries for planning corridors and CBF activation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle, _segment_point_distance_sq


@dataclass
class ObstacleQueryConfig:
    """How deployable baselines subset obstacles for planning vs validation."""

    plan_mode: str = "corridor"  # corridor | all
    validation_mode: str = "full"  # full | corridor
    corridor_half_width: float = 2.5
    cbf_mode: str = "radius_forward"  # all | radius | radius_forward
    cbf_forward_range: float = 3.0
    cbf_forward_cone_half_deg: float = 90.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ObstacleQueryConfig:
        d = dict(raw or {})
        return cls(
            plan_mode=str(d.get("plan_mode", d.get("obstacle_query_mode", "corridor"))).strip().lower(),
            validation_mode=str(d.get("validation_mode", "full")).strip().lower(),
            corridor_half_width=float(d.get("corridor_half_width", 2.5)),
            cbf_mode=str(d.get("cbf_mode", "radius_forward")).strip().lower(),
            cbf_forward_range=float(d.get("cbf_forward_range", 3.0)),
            cbf_forward_cone_half_deg=float(d.get("cbf_forward_cone_half_deg", 90.0)),
        )


def segment_point_distance(p0: np.ndarray, p1: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(_segment_point_distance_sq(
        np.asarray(p0, dtype=np.float64).reshape(2),
        np.asarray(p1, dtype=np.float64).reshape(2),
        np.asarray(q, dtype=np.float64).reshape(2),
    )))


def obstacles_in_corridor(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[Obstacle],
    half_width: float,
) -> list[Obstacle]:
    """Obstacles whose inflated disc intersects the start→goal corridor tube."""
    if not obstacles:
        return []
    p0 = np.asarray(start_xy, dtype=np.float64).reshape(2)
    p1 = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    hw = max(float(half_width), 0.0)
    out: list[Obstacle] = []
    for obs in obstacles:
        c = np.asarray(obs.center, dtype=np.float64).reshape(2)
        r = float(obs.radius)
        d = segment_point_distance(p0, p1, c)
        if d <= hw + r:
            out.append(obs)
    return out


def select_plan_obstacles(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    all_obstacles: list[Obstacle],
    cfg: ObstacleQueryConfig,
) -> list[Obstacle]:
    if cfg.plan_mode == "all" or not all_obstacles:
        return list(all_obstacles)
    return obstacles_in_corridor(start_xy, goal_xy, all_obstacles, cfg.corridor_half_width)


def select_validation_obstacles(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    all_obstacles: list[Obstacle],
    cfg: ObstacleQueryConfig,
) -> list[Obstacle]:
    if cfg.validation_mode == "full" or not all_obstacles:
        return list(all_obstacles)
    return obstacles_in_corridor(start_xy, goal_xy, all_obstacles, cfg.corridor_half_width)


def select_cbf_obstacles(
    agent_pos_xy: np.ndarray,
    agent_yaw: float,
    u_nominal_body: np.ndarray,
    all_obstacles: list[Obstacle],
    *,
    activation_radius: float,
    forward_range: float,
    forward_cone_half_deg: float,
    mode: str = "radius_forward",
) -> tuple[list[Obstacle], list[int]]:
    """
    Return obstacles relevant to CBF with global indices into ``all_obstacles``.

    ``radius_forward``: union of within ``activation_radius`` and forward cone.
    """
    if not all_obstacles or mode == "all":
        return list(all_obstacles), list(range(len(all_obstacles)))

    p = np.asarray(agent_pos_xy, dtype=np.float64).reshape(2)
    u = np.asarray(u_nominal_body, dtype=np.float64).reshape(2)
    speed = float(np.linalg.norm(u))
    if speed > 1e-3:
        forward = u / speed
    else:
        forward = np.array([np.cos(float(agent_yaw)), np.sin(float(agent_yaw))], dtype=np.float64)
    cos_half = float(np.cos(np.deg2rad(max(float(forward_cone_half_deg), 0.0))))

    picked: list[Obstacle] = []
    indices: list[int] = []
    act_r = float(activation_radius)
    fwd_r = float(forward_range)

    for i, obs in enumerate(all_obstacles):
        c = np.asarray(obs.center, dtype=np.float64).reshape(2)
        rel = c - p
        dist = float(np.linalg.norm(rel))
        surface = max(0.0, dist - float(obs.radius))
        in_radius = surface <= act_r

        in_forward = False
        if dist > 1e-6 and fwd_r > 0.0:
            rel_u = rel / dist
            if float(np.dot(rel_u, forward)) >= cos_half and dist <= fwd_r + float(obs.radius):
                in_forward = True

        if mode == "radius":
            keep = in_radius
        else:
            keep = in_radius or in_forward

        if keep:
            picked.append(obs)
            indices.append(int(i))

    return picked, indices
