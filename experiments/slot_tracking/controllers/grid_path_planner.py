"""Lightweight grid A* planner for obstacle slot-tracking diagnostics."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import has_line_of_sight


def point_clearance(position_xy: np.ndarray, obstacle: Any, *, uav_radius: float, safety_margin: float) -> float:
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    c = np.asarray(getattr(obstacle, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
    pad = float(uav_radius) + float(safety_margin)
    if getattr(obstacle, "kind", "circle") == "aabb" and getattr(obstacle, "half_extents", None) is not None:
        half = np.asarray(getattr(obstacle, "half_extents"), dtype=np.float64).reshape(2)
        outside = np.maximum(np.abs(p - c) - half, 0.0)
        return float(np.linalg.norm(outside) - pad)
    return float(np.linalg.norm(p - c) - float(getattr(obstacle, "radius", 0.0)) - pad)


def point_is_free(position_xy: np.ndarray, obstacles: list[Any], *, world_xy: float, uav_radius: float, safety_margin: float) -> bool:
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    if np.max(np.abs(p)) > float(world_xy) - float(uav_radius) - float(safety_margin):
        return False
    return all(point_clearance(p, obs, uav_radius=uav_radius, safety_margin=safety_margin) >= 0.0 for obs in obstacles)


def plan_grid_astar(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[Any],
    *,
    world_xy: float,
    uav_radius: float,
    safety_margin: float,
    resolution: float = 0.4,
    max_expansions: int = 20000,
    clearance_weight: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Plan a collision-free 2D path from start to goal using grid A*."""
    start = np.asarray(start_xy, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    if has_line_of_sight(start, goal, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
        return np.stack([start, goal], axis=0), {"planner_success": True, "planner_expansions": 0, "planner_reason": "los_clear"}

    res = max(float(resolution), 0.05)
    limit = max(float(world_xy) - float(uav_radius) - float(safety_margin), 0.0)
    n = int(np.floor((2.0 * limit) / res)) + 1

    def to_idx(p: np.ndarray) -> tuple[int, int]:
        clipped = np.clip(np.asarray(p, dtype=np.float64).reshape(2), -limit, limit)
        ij = np.rint((clipped + limit) / res).astype(int)
        return int(np.clip(ij[0], 0, n - 1)), int(np.clip(ij[1], 0, n - 1))

    def to_xy(idx: tuple[int, int]) -> np.ndarray:
        return np.array([idx[0] * res - limit, idx[1] * res - limit], dtype=np.float64)

    start_idx = _nearest_free_idx(to_idx(start), to_xy, obstacles, world_xy=world_xy, uav_radius=uav_radius, safety_margin=safety_margin, n=n)
    goal_idx = _nearest_free_idx(to_idx(goal), to_xy, obstacles, world_xy=world_xy, uav_radius=uav_radius, safety_margin=safety_margin, n=n)
    if start_idx is None or goal_idx is None:
        return np.stack([start, goal], axis=0), {"planner_success": False, "planner_expansions": 0, "planner_reason": "start_or_goal_not_free"}

    neighbors = [
        (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, np.sqrt(2.0)),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, np.sqrt(2.0)), (1, 0, 1.0), (1, 1, np.sqrt(2.0)),
    ]
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    gscore = {start_idx: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    heappush(open_heap, (_heuristic(start_idx, goal_idx), counter, start_idx))
    closed: set[tuple[int, int]] = set()
    expansions = 0

    while open_heap and expansions < int(max_expansions):
        _f, _c, current = heappop(open_heap)
        if current in closed:
            continue
        if current == goal_idx:
            idx_path = _reconstruct(came_from, current)
            path = np.asarray([to_xy(idx) for idx in idx_path], dtype=np.float64)
            path[0] = start
            path[-1] = goal
            return smooth_path(path, obstacles, uav_radius=uav_radius, safety_margin=safety_margin), {
                "planner_success": True,
                "planner_expansions": int(expansions),
                "planner_reason": "astar",
            }
        closed.add(current)
        expansions += 1
        cur_xy = to_xy(current)
        for dx, dy, cost in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt[0] < 0 or nxt[0] >= n or nxt[1] < 0 or nxt[1] >= n or nxt in closed:
                continue
            nxt_xy = to_xy(nxt)
            if not point_is_free(nxt_xy, obstacles, world_xy=world_xy, uav_radius=uav_radius, safety_margin=safety_margin):
                continue
            if not has_line_of_sight(cur_xy, nxt_xy, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
                continue
            clearance = point_clearance(nxt_xy, obstacles, uav_radius=uav_radius, safety_margin=0.0)
            clearance_penalty = max(float(clearance_weight), 0.0) / max(float(clearance), 1e-3)
            tentative = gscore[current] + cost * res + clearance_penalty * res
            if tentative + 1e-9 < gscore.get(nxt, float("inf")):
                came_from[nxt] = current
                gscore[nxt] = tentative
                counter += 1
                heappush(open_heap, (tentative + res * _heuristic(nxt, goal_idx), counter, nxt))

    return np.stack([start, goal], axis=0), {
        "planner_success": False,
        "planner_expansions": int(expansions),
        "planner_reason": "astar_failed",
    }


def smooth_path(path: np.ndarray, obstacles: list[Any], *, uav_radius: float, safety_margin: float) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] <= 2:
        return pts
    out = [pts[0]]
    i = 0
    while i < pts.shape[0] - 1:
        j = pts.shape[0] - 1
        while j > i + 1:
            if has_line_of_sight(pts[i], pts[j], obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
                break
            j -= 1
        out.append(pts[j])
        i = j
    return np.asarray(out, dtype=np.float64)


def path_length(path: np.ndarray) -> float:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def select_lookahead_subgoal(path: np.ndarray, position_xy: np.ndarray, *, lookahead_distance: float) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    pos = np.asarray(position_xy, dtype=np.float64).reshape(2)
    if pts.shape[0] == 0:
        return pos.copy()
    if pts.shape[0] == 1:
        return pts[0].copy()

    seg_starts = pts[:-1]
    seg_ends = pts[1:]
    best_dist = float("inf")
    best_idx = 0
    best_t = 0.0
    for i, (a, b) in enumerate(zip(seg_starts, seg_ends)):
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(pos - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(pos - proj))
        if d < best_dist:
            best_dist = d
            best_idx = i
            best_t = t

    remaining = max(float(lookahead_distance), 0.0)
    cur = pts[best_idx] + best_t * (pts[best_idx + 1] - pts[best_idx])
    for i in range(best_idx, pts.shape[0] - 1):
        a = cur if i == best_idx else pts[i]
        b = pts[i + 1]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len >= remaining:
            return a + (remaining / max(seg_len, 1e-9)) * (b - a)
        remaining -= seg_len
    return pts[-1].copy()


def _nearest_free_idx(
    center: tuple[int, int],
    to_xy,
    obstacles: list[Any],
    *,
    world_xy: float,
    uav_radius: float,
    safety_margin: float,
    n: int,
) -> tuple[int, int] | None:
    if point_is_free(to_xy(center), obstacles, world_xy=world_xy, uav_radius=uav_radius, safety_margin=safety_margin):
        return center
    max_r = min(max(n, 1), 20)
    for radius in range(1, max_r + 1):
        candidates: list[tuple[float, tuple[int, int]]] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                idx = (center[0] + dx, center[1] + dy)
                if idx[0] < 0 or idx[0] >= n or idx[1] < 0 or idx[1] >= n:
                    continue
                if point_is_free(to_xy(idx), obstacles, world_xy=world_xy, uav_radius=uav_radius, safety_margin=safety_margin):
                    candidates.append((float(dx * dx + dy * dy), idx))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    return None


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _reconstruct(came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]) -> list[tuple[int, int]]:
    out = [current]
    while current in came_from:
        current = came_from[current]
        out.append(current)
    out.reverse()
    return out
