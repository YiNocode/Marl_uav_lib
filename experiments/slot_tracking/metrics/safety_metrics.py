"""Safety metrics and geometric checks for slot-tracking benchmark."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle


def obstacle_clearance(position_xy: np.ndarray, obstacles: list[Obstacle], *, uav_radius: float) -> float:
    """Nearest surface clearance after accounting for UAV radius."""
    if not obstacles:
        return float("inf")
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    vals = []
    for obs in obstacles:
        c = np.asarray(obs.center, dtype=np.float64).reshape(2)
        vals.append(float(np.linalg.norm(p - c) - float(obs.radius) - float(uav_radius)))
    return float(min(vals))


def boundary_margin(position_xy: np.ndarray, *, world_xy: float, uav_radius: float) -> float:
    """Minimum signed distance to square boundary after accounting for UAV radius."""
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    return float(float(world_xy) - float(uav_radius) - np.max(np.abs(p)))


def outside_boundary(position_xy: np.ndarray, *, world_xy: float, uav_radius: float) -> bool:
    return bool(boundary_margin(position_xy, world_xy=world_xy, uav_radius=uav_radius) < 0.0)


def obstacle_collision(position_xy: np.ndarray, obstacles: list[Obstacle], *, uav_radius: float) -> bool:
    return bool(obstacle_clearance(position_xy, obstacles, uav_radius=uav_radius) < 0.0)


def inter_agent_collision(positions_xy: np.ndarray, *, min_distance: float) -> bool:
    pts = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            if float(np.linalg.norm(pts[i] - pts[j])) < float(min_distance):
                return True
    return False


def outward_velocity_ratio(rows: list[dict], *, world_xy: float, near_boundary_distance: float) -> float:
    """Fraction of near-boundary steps moving farther outward."""
    outward = 0
    near = 0
    for row in rows:
        p = np.array([row["uav_x"], row["uav_y"]], dtype=np.float64)
        v = np.array([row["uav_vx"], row["uav_vy"]], dtype=np.float64)
        margin = float(world_xy) - float(np.max(np.abs(p)))
        if margin <= float(near_boundary_distance):
            near += 1
            axis = int(np.argmax(np.abs(p)))
            if p[axis] * v[axis] > 0.0:
                outward += 1
    return 0.0 if near == 0 else float(outward / near)


def compute_safety_metrics(rows: list[dict], *, cfg: dict) -> dict[str, float | bool]:
    """Compute per-episode safety aggregates from per-step rows."""
    dyn = cfg["dynamics"]
    world_xy = float(cfg["world"]["world_xy"])
    safety_margin = float(dyn["safety_margin"])
    min_obs = min(float(r["nearest_obstacle_distance"]) for r in rows) if rows else float("inf")
    obs_vals = np.asarray([float(r["nearest_obstacle_distance"]) for r in rows], dtype=np.float64) if rows else np.asarray([], dtype=np.float64)
    pred_vals = np.asarray([float(r.get("min_predicted_next_clearance", np.nan)) for r in rows], dtype=np.float64) if rows else np.asarray([], dtype=np.float64)
    pred_vals = pred_vals[np.isfinite(pred_vals)]
    min_bound = min(float(r["boundary_margin"]) for r in rows) if rows else float("inf")
    below_margin = [
        float(r["nearest_obstacle_distance"]) < safety_margin or float(r["boundary_margin"]) < safety_margin
        for r in rows
    ]
    near_boundary = float(cfg["failure_classifier"]["near_boundary_distance"])
    return {
        "obstacle_collision": bool(any(bool(r.get("obstacle_collision", False)) for r in rows)),
        "boundary_violation": bool(any(bool(r.get("boundary_violation", False)) for r in rows)),
        "inter_agent_collision": bool(any(bool(r.get("inter_agent_collision", False)) for r in rows)),
        "min_obstacle_clearance": float(min_obs),
        "episode_min_clearance": float(min_obs),
        "p5_obstacle_clearance": float(np.percentile(obs_vals[np.isfinite(obs_vals)], 5)) if np.any(np.isfinite(obs_vals)) else float("inf"),
        "step_clearance_p5": float(np.percentile(obs_vals[np.isfinite(obs_vals)], 5)) if np.any(np.isfinite(obs_vals)) else float("inf"),
        "min_boundary_margin": float(min_bound),
        "time_below_safety_margin": float(np.mean(below_margin)) if below_margin else 0.0,
        "min_predicted_next_clearance": float(np.min(pred_vals)) if pred_vals.size else float("inf"),
        "predicted_next_clearance_min": float(np.min(pred_vals)) if pred_vals.size else float("inf"),
        "predictive_filter_active_ratio": float(np.mean([bool(r.get("predictive_filter_active", False)) for r in rows])) if rows else 0.0,
        "outward_velocity_ratio": outward_velocity_ratio(
            rows,
            world_xy=world_xy,
            near_boundary_distance=near_boundary,
        ),
    }
