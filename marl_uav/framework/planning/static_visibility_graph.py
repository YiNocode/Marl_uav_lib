"""Static/local visibility graph for deployable path queries."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_adapter import obstacle_version_key
from marl_uav.framework.geometry.obstacle_geometry import (
    Obstacle,
    has_line_of_sight,
    inflate_obstacle,
    path_length,
)

INF = 1e18


@dataclass
class StaticVisibilityGraph:
    """
    Obstacle boundary samples indexed in a spatial grid.

    Full adjacency is NOT built globally; each query builds a local subgraph
    inside a start-goal bounding box (keeps replans fast in dense E2 fields).
    """

    nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    obstacle_version: tuple = field(default_factory=tuple)
    cell_size: float = 4.0
    _grid: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    inflated_obstacles: list[Obstacle] = field(default_factory=list)
    build_time_ms: float = 0.0

    def build_static_graph_once(
        self,
        obstacles: list[Obstacle],
        bounds: tuple[float, float, float, float],
        cfg: dict[str, Any],
        *,
        safety_margin: float,
        uav_radius: float,
    ) -> None:
        """Build sample nodes once per episode / obstacle change."""
        t0 = time.perf_counter()
        ver = obstacle_version_key(obstacles)
        if ver == self.obstacle_version and self.nodes.shape[0] > 0:
            return

        num_samples = int(cfg.get("num_obstacle_samples", 12))
        clearance = float(cfg.get("clearance", 0.2))
        self.inflated_obstacles = [
            inflate_obstacle(o, uav_radius=uav_radius, safety_margin=safety_margin) for o in obstacles
        ]
        node_list: list[np.ndarray] = []
        for obs in self.inflated_obstacles:
            if obs.kind != "circle":
                continue
            c = np.asarray(obs.center, dtype=np.float64).reshape(2)
            r = float(obs.radius) + clearance
            angles = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
            for a in angles:
                node_list.append(c + r * np.array([np.cos(a), np.sin(a)], dtype=np.float64))

        if node_list:
            self.nodes = np.stack(node_list, axis=0)
        else:
            self.nodes = np.zeros((0, 2), dtype=np.float64)

        self._grid.clear()
        cs = float(self.cell_size)
        for idx in range(int(self.nodes.shape[0])):
            p = self.nodes[idx]
            key = (int(np.floor(p[0] / cs)), int(np.floor(p[1] / cs)))
            self._grid.setdefault(key, []).append(idx)

        self.obstacle_version = ver
        self.build_time_ms = (time.perf_counter() - t0) * 1000.0

    def _local_node_indices(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        margin: float,
        max_nodes: int,
    ) -> list[int]:
        p0 = np.asarray(start, dtype=np.float64).reshape(2)
        p1 = np.asarray(goal, dtype=np.float64).reshape(2)
        lo = np.minimum(p0, p1) - margin
        hi = np.maximum(p0, p1) + margin
        cs = float(self.cell_size)
        ix0, iy0 = int(np.floor(lo[0] / cs)), int(np.floor(lo[1] / cs))
        ix1, iy1 = int(np.floor(hi[0] / cs)), int(np.floor(hi[1] / cs))
        out: list[int] = []
        for ix in range(ix0 - 1, ix1 + 2):
            for iy in range(iy0 - 1, iy1 + 2):
                out.extend(self._grid.get((ix, iy), []))
        if len(out) > max_nodes:
            out = out[:max_nodes]
        return out

    def query_path(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        obstacles: list[Obstacle],
        cfg: dict[str, Any],
        *,
        safety_margin: float,
        uav_radius: float,
        timeout_ms: float = 5.0,
    ) -> tuple[list[np.ndarray] | None, float, bool]:
        """
        Local visibility-graph shortest path.

        Returns (path, time_ms, timed_out).
        """
        t0 = time.perf_counter()
        deadline = t0 + timeout_ms / 1000.0
        start = np.asarray(start_xy, dtype=np.float64).reshape(2)
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)

        if has_line_of_sight(
            start, goal, obstacles,
            safety_margin=safety_margin, uav_radius=uav_radius,
        ):
            return [start.copy(), goal.copy()], (time.perf_counter() - t0) * 1000.0, False

        max_nodes = int(cfg.get("max_nodes", 128))
        margin = float(cfg.get("local_margin", 6.0))
        local_idx = self._local_node_indices(start, goal, margin, max_nodes)
        nodes = [start.copy(), goal.copy()]
        for idx in local_idx:
            nodes.append(self.nodes[idx].copy())
        if len(nodes) > max_nodes + 2:
            nodes = nodes[: max_nodes + 2]

        n = len(nodes)
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        inflated = self.inflated_obstacles

        for i in range(n):
            if time.perf_counter() > deadline:
                return None, (time.perf_counter() - t0) * 1000.0, True
            for j in range(i + 1, n):
                if has_line_of_sight(
                    nodes[i], nodes[j], inflated,
                    safety_margin=0.0, uav_radius=0.0,
                ):
                    w = float(np.linalg.norm(nodes[j] - nodes[i]))
                    adj[i].append((j, w))
                    adj[j].append((i, w))

        dist = [INF] * n
        prev = [-1] * n
        dist[0] = 0.0
        heap: list[tuple[float, int]] = [(0.0, 0)]
        while heap:
            if time.perf_counter() > deadline:
                return None, (time.perf_counter() - t0) * 1000.0, True
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

        elapsed = (time.perf_counter() - t0) * 1000.0
        if dist[1] >= INF * 0.5:
            return None, elapsed, False

        path: list[np.ndarray] = []
        cur = 1
        while cur >= 0:
            path.append(nodes[cur].copy())
            cur = prev[cur]
        path.reverse()

        if bool(cfg.get("shortcut_smoothing", True)):
            path = _shortcut(path, obstacles, safety_margin=safety_margin, uav_radius=uav_radius)

        return path, elapsed, False


def _shortcut(
    path: list[np.ndarray],
    obstacles: list[Obstacle],
    *,
    safety_margin: float,
    uav_radius: float,
) -> list[np.ndarray]:
    if len(path) <= 2:
        return path
    out = [np.asarray(path[0], dtype=np.float64).reshape(2).copy()]
    i, n = 0, len(path)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if has_line_of_sight(
                out[-1], np.asarray(path[j]).reshape(2), obstacles,
                safety_margin=safety_margin, uav_radius=uav_radius,
            ):
                break
            j -= 1
        out.append(np.asarray(path[j], dtype=np.float64).reshape(2).copy())
        i = j
    return out
