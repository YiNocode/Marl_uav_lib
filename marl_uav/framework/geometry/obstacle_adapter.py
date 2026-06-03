"""Adapt environment task state obstacles to unified geometry objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle


def obstacles_from_task_state(
    task_state: Any,
    *,
    uav_radius: float | None = None,
    task: Any | None = None,
) -> list[Obstacle]:
    """
    Build obstacle list from ex2 ``obstacle_xy`` / ``obstacle_r`` on task_state.

    Currently E2 uses vertical cylinders → 2D circles. Rect/polygon support is
    reserved via ``Obstacle.kind`` for future env extensions.
    """
    xy = getattr(task_state, "obstacle_xy", None)
    radii = getattr(task_state, "obstacle_r", None)
    if xy is None or radii is None:
        return []
    xy_arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    r_arr = np.asarray(radii, dtype=np.float64).reshape(-1)
    n = min(int(xy_arr.shape[0]), int(r_arr.shape[0]))
    if n == 0:
        return []

    del uav_radius, task  # inflation applied downstream via safety_margin
    out: list[Obstacle] = []
    for i in range(n):
        r = float(r_arr[i])
        if r <= 0.0:
            continue
        out.append(
            Obstacle(
                kind="circle",
                center=xy_arr[i].copy(),
                radius=r,
            )
        )
    return out


def nearest_obstacles(
    position_xy: np.ndarray,
    obstacles: list[Obstacle],
    k: int = 2,
) -> list[Obstacle]:
    """
    Return the ``k`` obstacles closest to ``position_xy`` (by surface clearance).

    Used by deployable cached-path baselines to avoid global obstacle queries.
    """
    if not obstacles or k <= 0:
        return []
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    scored: list[tuple[float, Obstacle]] = []
    for obs in obstacles:
        c = np.asarray(obs.center, dtype=np.float64).reshape(2)
        clearance = max(0.0, float(np.linalg.norm(p - c)) - float(obs.radius))
        scored.append((clearance, obs))
    scored.sort(key=lambda x: x[0])
    return [obs for _, obs in scored[: int(k)]]


def local_obstacles_for_pursuers(
    pursuer_positions_xy: np.ndarray,
    obstacles: list[Obstacle],
    k: int = 2,
) -> list[list[Obstacle]]:
    """Per-pursuer nearest-``k`` obstacle lists."""
    p = np.asarray(pursuer_positions_xy, dtype=np.float64).reshape(-1, 2)
    return [nearest_obstacles(p[i], obstacles, k) for i in range(int(p.shape[0]))]


def obstacle_index(obstacle: Obstacle, obstacles: list[Obstacle]) -> int | None:
    """Map a local obstacle back to its index in the global list."""
    c = np.asarray(obstacle.center, dtype=np.float64).reshape(2)
    for i, obs in enumerate(obstacles):
        oc = np.asarray(obs.center, dtype=np.float64).reshape(2)
        if obs.kind == obstacle.kind and abs(float(obs.radius) - float(obstacle.radius)) < 1e-6:
            if float(np.linalg.norm(oc - c)) < 1e-4:
                return i
    return None


def local_obstacle_indices(
    local_obs: list[Obstacle],
    obstacles: list[Obstacle],
) -> list[int]:
    """Global obstacle indices for a per-agent local list."""
    out: list[int] = []
    for lo in local_obs:
        idx = obstacle_index(lo, obstacles)
        if idx is not None:
            out.append(idx)
    return out


def manifold_influencing_obstacles(
    evader_pos_xy: np.ndarray,
    obstacles: list[Obstacle],
    *,
    capture_dist: float,
    target_radius_xy: float,
    top_k: int = 4,
    influence_radius_scale: float = 2.5,
    clearance_margin_scale: float = 0.35,
) -> list[dict[str, float]]:
    """Obstacles currently deforming the capture manifold (for debug viz)."""
    if not obstacles:
        return []
    ev = np.asarray(evader_pos_xy, dtype=np.float64).reshape(2)
    influence_r = float(influence_radius_scale) * max(float(capture_dist), float(target_radius_xy))
    extra = float(clearance_margin_scale) * float(capture_dist)
    scored: list[tuple[float, int, float]] = []
    for i, obs in enumerate(obstacles):
        c = np.asarray(obs.center, dtype=np.float64).reshape(2)
        dist = float(np.linalg.norm(ev - c))
        clear_r = float(obs.radius) + extra
        surface = max(0.0, dist - clear_r)
        if surface <= influence_r:
            closeness = float(np.clip(1.0 - surface / max(influence_r, 1e-6), 0.0, 1.0))
            scored.append((closeness, i, surface))
    scored.sort(key=lambda x: (-x[0], x[2]))
    out: list[dict[str, float]] = []
    for closeness, idx, surface in scored[: max(int(top_k), 0)]:
        obs = obstacles[idx]
        out.append(
            {
                "index": float(idx),
                "center_x": float(obs.center[0]),
                "center_y": float(obs.center[1]),
                "radius": float(obs.radius),
                "surface_dist": float(surface),
                "influence_weight": float(closeness),
            }
        )
    return out


def obstacle_version_key(obstacles: list[Obstacle]) -> tuple:
    """Hashable key for path-cache invalidation when obstacles change."""
    parts: list[tuple] = []
    for obs in obstacles:
        c = np.round(np.asarray(obs.center, dtype=np.float64).reshape(2), decimals=2)
        parts.append((obs.kind, float(obs.radius), float(c[0]), float(c[1])))
    return tuple(sorted(parts))
