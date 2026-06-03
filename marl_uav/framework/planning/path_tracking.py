"""Path tracking utilities (O(n_waypoints) lookahead selection)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _unit2(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(x))
    if n > 1e-9:
        return x / n
    if fallback is not None:
        return _unit2(fallback)
    return np.array([1.0, 0.0], dtype=np.float64)


def select_lookahead_waypoint(
    path: list[np.ndarray] | np.ndarray,
    current_position: np.ndarray,
    lookahead_dist: float,
    *,
    waypoint_accept_radius: float = 0.2,
    terminal_goal: np.ndarray | None = None,
) -> np.ndarray:
    """
    Select a forward waypoint on ``path`` at least ``lookahead_dist`` ahead.

    O(number of waypoints); does not invoke any planner.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        if terminal_goal is not None:
            return np.asarray(terminal_goal, dtype=np.float64).reshape(-1).copy()
        return np.asarray(current_position, dtype=np.float64).reshape(-1)[:2]

    cur = np.asarray(current_position, dtype=np.float64).reshape(-1)[:2]
    goal_xy = pts[-1, :2]
    if terminal_goal is not None:
        goal_xy = np.asarray(terminal_goal, dtype=np.float64).reshape(-1)[:2]

    if float(np.linalg.norm(cur - goal_xy)) <= float(waypoint_accept_radius):
        if terminal_goal is not None:
            return np.asarray(terminal_goal, dtype=np.float64).reshape(-1).copy()
        out = np.zeros(3, dtype=np.float64)
        out[:2] = goal_xy
        return out

    best_i = 0
    best_d = np.inf
    for i in range(int(pts.shape[0])):
        d = float(np.linalg.norm(pts[i, :2] - cur))
        if d < best_d:
            best_d = d
            best_i = i

    for j in range(best_i, int(pts.shape[0])):
        d = float(np.linalg.norm(pts[j, :2] - cur))
        if d >= float(lookahead_dist):
            out = np.zeros(3, dtype=np.float64)
            out[:2] = pts[j, :2]
            return out

    if terminal_goal is not None:
        return np.asarray(terminal_goal, dtype=np.float64).reshape(-1).copy()
    out = np.zeros(3, dtype=np.float64)
    out[:2] = goal_xy
    return out


def path_tangent_bearing(
    path: list[np.ndarray] | np.ndarray,
    current_position: np.ndarray,
) -> float:
    """Unit direction of the path segment nearest ``current_position`` (world frame)."""
    pts_raw = np.asarray(path, dtype=np.float64)
    if pts_raw.ndim != 2 or pts_raw.shape[0] == 0:
        return 0.0
    pts = pts_raw[:, :2]
    if pts.shape[0] < 2:
        cur = np.asarray(current_position, dtype=np.float64).reshape(2)
        if pts.shape[0] == 1:
            delta = pts[0] - cur
        else:
            return 0.0
        nrm = float(np.linalg.norm(delta))
        if nrm < 1e-9:
            return 0.0
        return float(np.arctan2(delta[1], delta[0]))

    cur = np.asarray(current_position, dtype=np.float64).reshape(2)
    best_i = 0
    best_d = np.inf
    for i in range(int(pts.shape[0])):
        d = float(np.linalg.norm(pts[i] - cur))
        if d < best_d:
            best_d = d
            best_i = i

    if best_i < int(pts.shape[0]) - 1:
        seg = pts[best_i + 1] - pts[best_i]
    else:
        seg = pts[best_i] - pts[best_i - 1]
    nrm = float(np.linalg.norm(seg))
    if nrm < 1e-9:
        return float(np.arctan2(seg[1], seg[0])) if np.any(seg) else 0.0
    return float(np.arctan2(seg[1] / nrm, seg[0] / nrm))


def closest_point_on_polyline(
    path: list[np.ndarray] | np.ndarray,
    current_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Closest point on ``path`` polyline to ``current_position``.

    Returns (closest_xy, unit_tangent, cross_track_signed, along_track_dist).
    """
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    cur = np.asarray(current_position, dtype=np.float64).reshape(2)
    if pts.shape[0] == 0:
        return cur.copy(), np.array([1.0, 0.0], dtype=np.float64), 0.0, 0.0
    if pts.shape[0] == 1:
        delta = pts[0] - cur
        nrm = float(np.linalg.norm(delta))
        if nrm < 1e-9:
            return pts[0].copy(), np.array([1.0, 0.0], dtype=np.float64), 0.0, 0.0
        tang = delta / nrm
        return pts[0].copy(), tang, -nrm, 0.0

    best_dist_sq = np.inf
    best_pt = pts[0].copy()
    best_tangent = np.array([1.0, 0.0], dtype=np.float64)
    along = 0.0
    seg_start = 0.0

    for i in range(int(pts.shape[0]) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        v = p1 - p0
        c2 = float(np.dot(v, v))
        if c2 < 1e-12:
            proj = p0
            t_param = 0.0
        else:
            t_param = float(np.clip(np.dot(cur - p0, v) / c2, 0.0, 1.0))
            proj = p0 + t_param * v
        d_sq = float(np.dot(cur - proj, cur - proj))
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best_pt = proj.copy()
            seg_len = float(np.sqrt(c2)) if c2 >= 1e-12 else 0.0
            if seg_len > 1e-9:
                best_tangent = (v / seg_len).astype(np.float64)
            along = seg_start + t_param * seg_len
        seg_start += float(np.sqrt(c2)) if c2 >= 1e-12 else 0.0

    normal = np.array([-best_tangent[1], best_tangent[0]], dtype=np.float64)
    cross_signed = float(np.dot(cur - best_pt, normal))

    return best_pt, best_tangent, cross_signed, along


def sampled_turn_arc(
    position_xy: np.ndarray,
    from_dir_xy: np.ndarray,
    to_dir_xy: np.ndarray,
    *,
    turn_radius: float,
    num_samples: int = 9,
) -> tuple[np.ndarray, float]:
    """Sample the swept centerline arc for a bounded-radius heading change.

    A line segment can be obstacle-free while the executed vehicle trajectory
    cuts a corner.  This helper approximates the swept turn from the previous
    tangent/command direction to the next desired direction; callers can then
    clearance-check the sampled arc with the same obstacle set used by planning.
    """
    pos = np.asarray(position_xy, dtype=np.float64).reshape(2)
    d0 = _unit2(from_dir_xy)
    d1 = _unit2(to_dir_xy, fallback=d0)
    cross = float(d0[0] * d1[1] - d0[1] * d1[0])
    dot = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
    delta = float(np.arctan2(cross, dot))
    if abs(delta) < 1e-4:
        horizon = max(float(turn_radius), 1e-6)
        return np.stack([pos, pos + d1 * horizon], axis=0), 0.0

    radius = max(float(turn_radius), 1e-6)
    side = 1.0 if delta > 0.0 else -1.0
    left0 = np.array([-d0[1], d0[0]], dtype=np.float64)
    center = pos + side * radius * left0
    start_angle = float(np.arctan2(pos[1] - center[1], pos[0] - center[0]))
    ts = np.linspace(0.0, delta, max(int(num_samples), 3))
    pts = np.stack([
        center + radius * np.array([np.cos(start_angle + t), np.sin(start_angle + t)], dtype=np.float64)
        for t in ts
    ], axis=0)
    return pts, abs(delta)


def points_min_obstacle_clearance(
    points_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
) -> float:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0 or not obstacles:
        return float("inf")
    best = float("inf")
    for p in pts:
        for obs in obstacles:
            if getattr(obs, "kind", None) != "circle":
                continue
            c = np.asarray(obs.center, dtype=np.float64).reshape(2)
            clear = float(np.linalg.norm(p - c)) - float(obs.radius) - float(uav_radius)
            best = min(best, clear)
    return best


def points_min_boundary_clearance(
    points_xy: np.ndarray,
    *,
    world_xy: float | None,
    boundary_margin: float = 0.0,
) -> float:
    """Minimum distance from sampled points to the usable square boundary."""
    if world_xy is None or not np.isfinite(float(world_xy)):
        return float("inf")
    w = max(float(world_xy), 1e-6)
    margin = max(float(boundary_margin), 0.0)
    usable = max(w - margin, 1e-6)
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return float("inf")
    clear = usable - np.max(np.abs(pts), axis=1)
    return float(np.min(clear))


def adjust_lookahead_for_turn_safety(
    current_xy: np.ndarray,
    desired_xy: np.ndarray,
    previous_dir_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    turn_radius: float,
    min_turn_clearance: float,
    safe_forward_dist: float,
    num_samples: int = 9,
    world_xy: float | None = None,
    boundary_margin: float = 0.0,
    min_boundary_clearance: float | None = None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Keep a slot/path command from cutting an unsafe turn near obstacles.

    If the predicted swept arc is too close to an obstacle, the returned target
    stays on the previous tangent for a short distance.  That gives the high
    level replanner another cycle to pick a smoother/safer slot instead of
    asking CBF to fight a globally bad command.
    """
    cur = np.asarray(current_xy, dtype=np.float64).reshape(2)
    desired = np.asarray(desired_xy, dtype=np.float64).reshape(2)
    prev_dir = _unit2(previous_dir_xy)
    desired_dir = _unit2(desired - cur, fallback=prev_dir)
    arc, angle = sampled_turn_arc(
        cur, prev_dir, desired_dir,
        turn_radius=turn_radius, num_samples=num_samples,
    )
    clear = points_min_obstacle_clearance(arc, obstacles, uav_radius=uav_radius)
    boundary_clear = points_min_boundary_clearance(
        arc, world_xy=world_xy, boundary_margin=boundary_margin,
    )
    boundary_thresh = (
        float(min_turn_clearance)
        if min_boundary_clearance is None else float(min_boundary_clearance)
    )
    obstacle_unsafe = bool(np.isfinite(clear) and clear < float(min_turn_clearance))
    boundary_unsafe = bool(np.isfinite(boundary_clear) and boundary_clear < boundary_thresh)
    unsafe = bool((obstacle_unsafe or boundary_unsafe) and angle > 0.05)
    if not unsafe:
        return desired.copy(), {
            "turn_safety_active": False,
            "turn_angle_rad": float(angle),
            "turn_arc_min_clearance": float(clear),
            "turn_boundary_min_clearance": float(boundary_clear),
            "turn_obstacle_unsafe": False,
            "turn_boundary_unsafe": False,
        }
    adjusted = cur + prev_dir * max(float(safe_forward_dist), 0.0)
    return adjusted, {
        "turn_safety_active": True,
        "turn_angle_rad": float(angle),
        "turn_arc_min_clearance": float(clear),
        "turn_boundary_min_clearance": float(boundary_clear),
        "turn_obstacle_unsafe": bool(obstacle_unsafe),
        "turn_boundary_unsafe": bool(boundary_unsafe),
    }
