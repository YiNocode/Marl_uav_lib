"""Visibility-graph path planner for sparse 2D obstacle fields."""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import (
    Obstacle,
    collision_check_path,
    has_line_of_sight,
    inflate_obstacle,
    path_length,
)

INF = 1e18


def _default_bounds(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    pad: float = 2.0,
) -> tuple[float, float, float, float]:
    pts = [np.asarray(start, dtype=np.float64).reshape(2), np.asarray(goal, dtype=np.float64).reshape(2)]
    for obs in obstacles:
        pts.append(np.asarray(obs.center, dtype=np.float64).reshape(2))
    arr = np.stack(pts, axis=0)
    xmin, ymin = np.min(arr, axis=0) - pad
    xmax, ymax = np.max(arr, axis=0) + pad
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _circle_sample_points(
    obs: Obstacle,
    *,
    num_samples: int,
    clearance: float,
) -> list[np.ndarray]:
    c = np.asarray(obs.center, dtype=np.float64).reshape(2)
    r = float(obs.radius) + float(clearance)
    angles = np.linspace(0.0, 2.0 * np.pi, int(num_samples), endpoint=False, dtype=np.float64)
    return [c + r * np.array([np.cos(a), np.sin(a)], dtype=np.float64) for a in angles]


def _shortcut_smooth(
    path: list[np.ndarray],
    obstacles: list[Obstacle],
    *,
    safety_margin: float,
    uav_radius: float,
) -> list[np.ndarray]:
    if len(path) <= 2:
        return path
    out = [np.asarray(path[0], dtype=np.float64).reshape(2).copy()]
    i = 0
    n = len(path)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if has_line_of_sight(
                out[-1],
                np.asarray(path[j], dtype=np.float64).reshape(2),
                obstacles,
                safety_margin=safety_margin,
                uav_radius=uav_radius,
                treat_tangent_as_blocked=True,
            ):
                break
            j -= 1
        out.append(np.asarray(path[j], dtype=np.float64).reshape(2).copy())
        i = j
    return out


def plan_path(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float] | None = None,
    cfg: dict[str, Any] | None = None,
    *,
    safety_margin: float = 0.3,
    uav_radius: float = 0.15,
) -> list[np.ndarray] | None:
    """
    Plan a collision-free polyline from start to goal using a visibility graph.

    Parameters
    ----------
    start_xy, goal_xy : [2] world coordinates
    obstacles : list of (possibly uninflated) obstacles
    bounds : optional (xmin, ymin, xmax, ymax) clipping for sampling
    cfg : path_planner config dict

    Returns
    -------
    list of [2] waypoints including start and goal, or None if planning fails.
    """
    raw_cfg = dict(cfg or {})
    planner_type = str(raw_cfg.get("type", "visibility_graph")).strip().lower()
    if planner_type not in ("visibility_graph", "grid_astar"):
        planner_type = "visibility_graph"

    start = np.asarray(start_xy, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    clearance = float(raw_cfg.get("clearance", 0.2))
    num_samples = int(raw_cfg.get("num_obstacle_samples", 24))
    shortcut = bool(raw_cfg.get("shortcut_smoothing", True))

    inflated = [
        inflate_obstacle(o, uav_radius=uav_radius, safety_margin=safety_margin) for o in obstacles
    ]

    if has_line_of_sight(
        start,
        goal,
        obstacles,
        safety_margin=safety_margin,
        uav_radius=uav_radius,
        treat_tangent_as_blocked=True,
    ):
        return [start.copy(), goal.copy()]

    if planner_type == "grid_astar":
        path = _grid_astar(
            start,
            goal,
            inflated,
            bounds=bounds or _default_bounds(start, goal, inflated),
            resolution=float(raw_cfg.get("grid_resolution", 0.25)),
        )
    else:
        path = _visibility_graph_path(
            start,
            goal,
            inflated,
            num_samples=num_samples,
            clearance=clearance,
            max_nodes=int(raw_cfg.get("max_nodes", 256)),
        )

    if path is None:
        return None

    if shortcut:
        path = _shortcut_smooth(
            path,
            obstacles,
            safety_margin=safety_margin,
            uav_radius=uav_radius,
        )

    if not collision_check_path(
        path,
        obstacles,
        safety_margin=safety_margin,
        uav_radius=uav_radius,
        treat_tangent_as_blocked=True,
    ):
        return None
    return path


def _visibility_graph_path(
    start: np.ndarray,
    goal: np.ndarray,
    inflated: list[Obstacle],
    *,
    num_samples: int,
    clearance: float,
    max_nodes: int,
) -> list[np.ndarray] | None:
    nodes: list[np.ndarray] = [start.copy(), goal.copy()]
    for obs in inflated:
        if obs.kind == "circle":
            nodes.extend(_circle_sample_points(obs, num_samples=num_samples, clearance=clearance))
        elif obs.kind == "aabb" and obs.half_extents is not None:
            he = np.asarray(obs.half_extents, dtype=np.float64).reshape(2) + clearance
            c = np.asarray(obs.center, dtype=np.float64).reshape(2)
            corners = [
                c + np.array([sx, sy], dtype=np.float64)
                for sx in (-he[0], he[0])
                for sy in (-he[1], he[1])
            ]
            nodes.extend(corners)
        # TODO: polygon boundary sampling

    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]

    n = len(nodes)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if has_line_of_sight(
                nodes[i],
                nodes[j],
                inflated,
                safety_margin=0.0,
                uav_radius=0.0,
                treat_tangent_as_blocked=True,
            ):
                w = float(np.linalg.norm(nodes[j] - nodes[i]))
                adj[i].append((j, w))
                adj[j].append((i, w))

    dist = [INF] * n
    prev = [-1] * n
    dist[0] = 0.0
    heap: list[tuple[float, int]] = [(0.0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == 1:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if dist[1] >= INF * 0.5:
        return None

    path: list[np.ndarray] = []
    cur = 1
    while cur >= 0:
        path.append(nodes[cur].copy())
        cur = prev[cur]
    path.reverse()
    return path


def _grid_astar(
    start: np.ndarray,
    goal: np.ndarray,
    inflated: list[Obstacle],
    *,
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> list[np.ndarray] | None:
    """Simple grid A* fallback when visibility graph is insufficient."""
    res = max(float(resolution), 0.05)
    xmin, ymin, xmax, ymax = bounds
    nx = max(2, int((xmax - xmin) / res) + 1)
    ny = max(2, int((ymax - ymin) / res) + 1)

    def to_cell(p: np.ndarray) -> tuple[int, int]:
        ix = int(np.clip(round((p[0] - xmin) / res), 0, nx - 1))
        iy = int(np.clip(round((p[1] - ymin) / res), 0, ny - 1))
        return ix, iy

    def to_xy(cell: tuple[int, int]) -> np.ndarray:
        return np.array([xmin + cell[0] * res, ymin + cell[1] * res], dtype=np.float64)

    def free(cell: tuple[int, int]) -> bool:
        p = to_xy(cell)
        for obs in inflated:
            if obs.kind == "circle":
                if np.linalg.norm(p - obs.center.reshape(2)) <= float(obs.radius):
                    return False
            elif obs.kind == "aabb" and obs.half_extents is not None:
                he = np.asarray(obs.half_extents, dtype=np.float64).reshape(2)
                c = np.asarray(obs.center, dtype=np.float64).reshape(2)
                if np.all(np.abs(p - c) <= he):
                    return False
        return True

    start_c = to_cell(start)
    goal_c = to_cell(goal)
    if not free(start_c) or not free(goal_c):
        return None

    open_set: list[tuple[float, tuple[int, int]]] = [(0.0, start_c)]
    g_score = {start_c: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_c:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return [to_xy(c) for c in cells]

        cx, cy = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            nb = (cx + dx, cy + dy)
            if nb[0] < 0 or nb[0] >= nx or nb[1] < 0 or nb[1] >= ny:
                continue
            if not free(nb):
                continue
            tentative = g_score[current] + float(np.hypot(dx, dy)) * res
            if tentative < g_score.get(nb, INF):
                came_from[nb] = current
                g_score[nb] = tentative
                h = float(np.linalg.norm(to_xy(nb) - to_xy(goal_c)))
                heapq.heappush(open_set, (tentative + h, nb))

    return None
