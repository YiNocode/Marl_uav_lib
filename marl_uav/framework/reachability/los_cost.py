"""Fast LOS-only assignment cost matrix (no path planner)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle, inflate_obstacle

INF_COST = 1e9


def _segment_circle_blocked(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius: float) -> bool:
    """True if segment p0->p1 intersects circle (tangent = blocked)."""
    c = np.asarray(center, dtype=np.float64).reshape(2)
    v = p1 - p0
    w = c - p0
    c1 = float(np.dot(w, v))
    if c1 <= 0.0:
        dist_sq = float(np.dot(w, w))
    else:
        c2 = float(np.dot(v, v))
        if c2 <= c1:
            d = c - p1
            dist_sq = float(np.dot(d, d))
        else:
            t = c1 / c2
            proj = p0 + t * v
            d = c - proj
            dist_sq = float(np.dot(d, d))
    r = float(radius)
    return dist_sq <= r * r


def _precompute_inflated_circles(
    obstacles: list[Obstacle],
    *,
    uav_radius: float,
    safety_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    radii: list[float] = []
    for obs in obstacles:
        if obs.kind != "circle":
            continue
        infl = inflate_obstacle(obs, uav_radius=uav_radius, safety_margin=safety_margin)
        centers.append(np.asarray(infl.center, dtype=np.float64).reshape(2))
        radii.append(float(infl.radius))
    if not centers:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.stack(centers, axis=0), np.asarray(radii, dtype=np.float64)


def build_los_cost_matrix(
    pursuer_positions: np.ndarray,
    slots: np.ndarray,
    obstacles: list[Obstacle],
    previous_assignment: np.ndarray | None,
    *,
    safety_margin: float = 0.3,
    uav_radius: float = 0.15,
    los_block_penalty: float = 100.0,
    use_infinite_los_block: bool = False,
    switch_penalty: float = 0.2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build C_ij = ||p_i - s_j|| + lambda_los * I[blocked] + lambda_switch * I[switch].

    Uses segment-circle tests only; no path planner. Target < 1 ms for 3×3 × ~100 obstacles.
    """
    t0 = time.perf_counter()
    p = np.asarray(pursuer_positions, dtype=np.float64).reshape(-1, 3)[:, :2]
    s = np.asarray(slots, dtype=np.float64).reshape(-1, 3)[:, :2]
    n_p, n_s = int(p.shape[0]), int(s.shape[0])
    euclid = np.linalg.norm(p[:, None, :] - s[None, :, :], axis=2).astype(np.float64)
    cost = euclid.copy()
    los_blocked = np.zeros((n_p, n_s), dtype=bool)

    centers, radii = _precompute_inflated_circles(
        obstacles, uav_radius=uav_radius, safety_margin=safety_margin
    )
    t_los_start = time.perf_counter()

    if centers.shape[0] > 0:
        for i in range(n_p):
            p0 = p[i]
            for j in range(n_s):
                p1 = s[j]
                seg_min = np.minimum(p0, p1) - radii.max()
                seg_max = np.maximum(p0, p1) + radii.max()
                for k in range(int(centers.shape[0])):
                    c = centers[k]
                    if c[0] < seg_min[0] - radii[k] or c[0] > seg_max[0] + radii[k]:
                        continue
                    if c[1] < seg_min[1] - radii[k] or c[1] > seg_max[1] + radii[k]:
                        continue
                    if _segment_circle_blocked(p0, p1, c, radii[k]):
                        los_blocked[i, j] = True
                        break

    for i in range(n_p):
        for j in range(n_s):
            if los_blocked[i, j]:
                if use_infinite_los_block:
                    cost[i, j] = INF_COST
                else:
                    cost[i, j] = euclid[i, j] + los_block_penalty

    for i in range(n_p):
        if np.all(cost[i] >= INF_COST * 0.5):
            cost[i, :] = euclid[i, :] + los_block_penalty

    if previous_assignment is not None:
        prev = np.asarray(previous_assignment, dtype=np.int64).reshape(-1)
        if prev.shape[0] == n_p:
            for i in range(n_p):
                for j in range(n_s):
                    if int(prev[i]) != j and cost[i, j] < INF_COST * 0.5:
                        cost[i, j] += switch_penalty

    los_ms = (time.perf_counter() - t_los_start) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    assigned_los_blocked = [
        bool(los_blocked[i, j])
        for i in range(n_p)
        for j in range(n_s)
    ]

    diagnostics: dict[str, Any] = {
        "assignment_cost_matrix": cost.copy(),
        "los_blocked_matrix": los_blocked.copy(),
        "num_los_blocked_pairs": int(np.sum(los_blocked)),
        "assigned_los_blocked": assigned_los_blocked,
        "los_cost_time_ms": total_ms,
        "los_check_time_ms": los_ms,
    }
    return cost, diagnostics
