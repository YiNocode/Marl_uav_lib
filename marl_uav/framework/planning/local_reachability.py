"""Lightweight local rollout reachability probe for slot assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.framework.planning.path_tracking import points_min_boundary_clearance
from marl_uav.framework.planning.turn_radius_obstacle_query import (
    TurnRadiusObstacleQueryConfig,
    batch_path_min_clearance,
    default_amax_xy,
    effective_omega_max,
    path_min_clearance,
    query_turn_radius_obstacles,
    resolve_speed_samples,
    rollout_unicycle_paths_batch,
    rollout_unicycle_paths_batch_inertial,
    wrap_to_pi,
)


@dataclass(frozen=True)
class LocalReachabilityScoringConfig:
    w_goal: float = 1.0
    w_heading: float = 0.2
    w_obstacle: float = 5.0
    w_smooth: float = 0.0
    w_speed: float = 0.0
    w_vel_smooth: float = 0.0
    vmax: float = 0.25
    collision_large_penalty: float = 1_000_000.0
    world_xy: float | None = None
    boundary_margin: float = 0.30
    amax_xy: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "LocalReachabilityScoringConfig":
        d = dict(raw or {})
        world_xy = d.get("world_xy", cls.world_xy)
        return cls(
            w_goal=float(d.get("w_goal", cls.w_goal)),
            w_heading=float(d.get("w_heading", cls.w_heading)),
            w_obstacle=float(d.get("w_obstacle", cls.w_obstacle)),
            w_smooth=float(d.get("w_smooth", cls.w_smooth)),
            w_speed=float(d.get("w_speed", cls.w_speed)),
            w_vel_smooth=float(d.get("w_vel_smooth", cls.w_vel_smooth)),
            vmax=float(d.get("vmax", cls.vmax)),
            collision_large_penalty=float(
                d.get("collision_large_penalty", cls.collision_large_penalty)
            ),
            world_xy=None if world_xy is None else float(world_xy),
            boundary_margin=float(d.get("boundary_margin", cls.boundary_margin)),
            amax_xy=None if d.get("amax_xy") is None else float(d["amax_xy"]),
        )


@dataclass(frozen=True)
class LocalReachabilityResult:
    blocked: bool
    min_clearance: float
    best_cost: float
    best_idx: int = 0
    best_path: np.ndarray | None = None
    best_speed: float = 0.0
    best_yaw_rate: float = 0.0
    local_obstacles: tuple[Any, ...] = ()
    candidate_count: int = 0
    valid_candidate_count: int = 0
    actual_speed_xy: float = 0.0
    heading_err_now: float = 0.0
    speed_gate: str = "full"
    out_of_bounds_candidate_count: int = 0
    emergency_stop_reason: str | None = None


def _heading_speed_gate(
    speeds: np.ndarray,
    heading_err_now: float,
) -> tuple[np.ndarray, str]:
    """Gate forward speed when the pursuer is not yet aligned with the goal."""
    deg60 = float(np.pi / 3.0)
    deg30 = float(np.pi / 6.0)
    if heading_err_now > deg60:
        return np.zeros_like(speeds), "stop"
    if heading_err_now > deg30:
        return speeds * 0.3, "slow"
    return speeds, "full"


def _paths_out_of_bounds(
    cand_paths: np.ndarray,
    bounds_xy: tuple[float, float, float, float],
    boundary_margin: float,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds_xy
    margin = max(float(boundary_margin), 0.0)
    x_lo = float(x_min) + margin
    x_hi = float(x_max) - margin
    y_lo = float(y_min) + margin
    y_hi = float(y_max) - margin
    xs = cand_paths[:, :, 0]
    ys = cand_paths[:, :, 1]
    return np.any((xs < x_lo) | (xs > x_hi) | (ys < y_lo) | (ys > y_hi), axis=1)


def _current_position_clearance(
    pos_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
    bounds_xy: tuple[float, float, float, float] | None,
    boundary_margin: float,
    world_xy: float | None,
) -> float:
    pos = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    stub = np.stack([pos, pos], axis=0)
    obs_clear = path_min_clearance(
        stub,
        obstacles,
        uav_radius=uav_radius,
        safety_margin=safety_margin,
    )
    if bounds_xy is not None:
        x_min, x_max, y_min, y_max = bounds_xy
        margin = max(float(boundary_margin), 0.0)
        bnd_clear = min(
            float(pos[0] - (float(x_min) + margin)),
            float((float(x_max) - margin) - pos[0]),
            float(pos[1] - (float(y_min) + margin)),
            float((float(y_max) - margin) - pos[1]),
        )
    elif world_xy is not None and np.isfinite(float(world_xy)):
        bnd_clear = points_min_boundary_clearance(
            stub,
            world_xy=world_xy,
            boundary_margin=boundary_margin,
        )
    else:
        bnd_clear = float("inf")
    return float(min(obs_clear, bnd_clear))


def _merge_boundary_clearance(
    clearances: np.ndarray,
    cand_paths: np.ndarray,
    *,
    world_xy: float | None,
    boundary_margin: float,
) -> np.ndarray:
    if world_xy is None or not np.isfinite(float(world_xy)):
        return clearances
    out = np.asarray(clearances, dtype=np.float64).copy()
    for k in range(cand_paths.shape[0]):
        bnd = points_min_boundary_clearance(
            cand_paths[k],
            world_xy=world_xy,
            boundary_margin=boundary_margin,
        )
        out[k] = min(float(out[k]), bnd)
    return out


def local_reachability_probe(
    pos: np.ndarray,
    yaw: float,
    goal: np.ndarray,
    obstacles: list[Any],
    *,
    cfg: TurnRadiusObstacleQueryConfig,
    scoring: LocalReachabilityScoringConfig | None = None,
    prev_action: np.ndarray | None = None,
    current_velocity_xy: np.ndarray | None = None,
    current_vel_xy: np.ndarray | None = None,
    bounds_xy: tuple[float, float, float, float] | None = None,
    executable_safety: bool = False,
) -> LocalReachabilityResult:
    """Sample unicycle rollouts and score reachability toward a local goal."""
    score_cfg = scoring or LocalReachabilityScoringConfig()
    pos_xy = np.asarray(pos, dtype=np.float64).reshape(2)
    goal_xy = np.asarray(goal, dtype=np.float64).reshape(2)
    rel = goal_xy - pos_xy
    goal_dist = float(np.linalg.norm(rel))
    goal_bearing = float(np.arctan2(rel[1], rel[0])) if goal_dist > 1e-9 else float(yaw)
    heading_err_now = float(abs(wrap_to_pi(goal_bearing - float(yaw))))

    v_in = current_vel_xy if current_vel_xy is not None else current_vel_xy
    v0 = (
        np.asarray(v_in, dtype=np.float64).reshape(2)
        if v_in is not None
        else np.zeros(2, dtype=np.float64)
    )
    actual_speed_xy = float(np.linalg.norm(v0))
    speed_gate = "full"
    emergency_stop_reason: str | None = None

    omega = effective_omega_max(cfg.vmax, cfg.omega_max, cfg.min_turn_radius)
    yaw_rates = np.linspace(-omega, omega, max(int(cfg.num_yaw_samples), 1), dtype=np.float64)
    use_inertial = executable_safety or float(np.linalg.norm(v0)) > 1e-3
    speeds = resolve_speed_samples(
        cfg.vmax,
        cfg.speed_samples,
        include_brake=use_inertial and 0.0 not in cfg.speed_samples,
    )
    if executable_safety:
        speeds, speed_gate = _heading_speed_gate(speeds, heading_err_now)
    speed_grid, yaw_rate_grid = np.meshgrid(speeds, yaw_rates, indexing="ij")
    cand_speeds = speed_grid.reshape(-1)
    cand_yaw_rates = yaw_rate_grid.reshape(-1)
    amax_xy = score_cfg.amax_xy if score_cfg.amax_xy is not None else cfg.amax_xy
    if amax_xy is None:
        amax_xy = default_amax_xy(cfg.vmax, cfg.horizon_s, cfg.dt)
    if use_inertial:
        cand_paths, cand_final_yaws = rollout_unicycle_paths_batch_inertial(
            pos_xy,
            float(yaw),
            v0,
            cand_speeds,
            cand_yaw_rates,
            horizon_s=cfg.horizon_s,
            dt=cfg.dt,
            amax_xy=float(amax_xy),
        )
    else:
        cand_paths, cand_final_yaws = rollout_unicycle_paths_batch(
            pos_xy,
            float(yaw),
            cand_speeds,
            cand_yaw_rates,
            horizon_s=cfg.horizon_s,
            dt=cfg.dt,
        )
    local_obstacles = query_turn_radius_obstacles(
        pos_xy,
        float(yaw),
        goal_xy,
        obstacles,
        cfg,
        candidate_paths=cand_paths,
    )
    clearances = batch_path_min_clearance(
        cand_paths,
        local_obstacles,
        uav_radius=cfg.uav_radius,
        safety_margin=cfg.safety_margin,
    )
    clearances = _merge_boundary_clearance(
        clearances,
        cand_paths,
        world_xy=score_cfg.world_xy,
        boundary_margin=score_cfg.boundary_margin,
    )
    finite_clearance = np.where(np.isfinite(clearances), clearances, np.inf)
    collides = finite_clearance <= 0.0
    out_of_bounds_candidate_count = 0
    if bounds_xy is not None:
        out_of_bounds = _paths_out_of_bounds(
            cand_paths,
            bounds_xy,
            score_cfg.boundary_margin,
        )
        out_of_bounds_candidate_count = int(np.sum(out_of_bounds))
        collides = collides | out_of_bounds
    elif executable_safety and score_cfg.world_xy is not None and np.isfinite(float(score_cfg.world_xy)):
        w = float(score_cfg.world_xy)
        derived_bounds = (-w, w, -w, w)
        out_of_bounds = _paths_out_of_bounds(
            cand_paths,
            derived_bounds,
            score_cfg.boundary_margin,
        )
        out_of_bounds_candidate_count = int(np.sum(out_of_bounds))
        collides = collides | out_of_bounds
    final_goal = np.linalg.norm(cand_paths[:, -1, :] - goal_xy[None, :], axis=1)
    heading_err = np.abs(np.asarray([wrap_to_pi(goal_bearing - y) for y in cand_final_yaws]))
    obstacle_penalty = np.zeros_like(final_goal)
    finite_mask = np.isfinite(finite_clearance)
    obstacle_penalty[finite_mask] = 1.0 / np.maximum(finite_clearance[finite_mask], 1e-3)
    speed_penalty = cand_speeds / max(float(score_cfg.vmax), 1e-6)
    smooth_penalty = np.zeros_like(final_goal)
    if prev_action is not None:
        prev = np.asarray(prev_action, dtype=np.float64).reshape(-1)
        if prev.shape[0] >= 2:
            smooth_penalty = np.linalg.norm(
                np.stack([cand_speeds - prev[0], cand_yaw_rates - prev[1]], axis=1),
                axis=1,
            )
    vel_smooth_penalty = np.zeros_like(final_goal)
    v_cur = v0
    if v_in is not None and score_cfg.w_vel_smooth > 0.0:
        step = max(float(cfg.dt), 1e-3)
        if cand_paths.shape[1] >= 2:
            cand_vel = (cand_paths[:, 1, :] - cand_paths[:, 0, :]) / step
        else:
            cand_vel = np.zeros((cand_paths.shape[0], 2), dtype=np.float64)
        vel_smooth_penalty = np.linalg.norm(cand_vel - v_cur[None, :], axis=1)
    costs = (
        score_cfg.w_goal * final_goal
        + score_cfg.w_heading * heading_err
        + score_cfg.w_obstacle * obstacle_penalty
        + score_cfg.w_smooth * smooth_penalty
        + score_cfg.w_speed * speed_penalty
        + score_cfg.w_vel_smooth * vel_smooth_penalty
    )
    if executable_safety and actual_speed_xy > max(float(cfg.vmax) * 0.5, 0.08):
        clearance_now = _current_position_clearance(
            pos_xy,
            obstacles,
            uav_radius=cfg.uav_radius,
            safety_margin=cfg.safety_margin,
            bounds_xy=bounds_xy,
            boundary_margin=score_cfg.boundary_margin,
            world_xy=score_cfg.world_xy,
        )
        low_clearance = float(cfg.uav_radius + cfg.safety_margin)
        if clearance_now < low_clearance:
            emergency_stop_reason = "high_speed_low_clearance"
            costs = np.where(
                cand_speeds > 1e-6,
                score_cfg.collision_large_penalty,
                costs,
            )
    costs = np.where(collides, score_cfg.collision_large_penalty, costs)
    best_idx = int(np.argmin(costs)) if costs.size else 0
    valid_count = int(np.sum(~collides))
    blocked = valid_count == 0

    if blocked:
        best_cost = float(score_cfg.collision_large_penalty)
        min_clear = float(np.min(finite_clearance)) if finite_clearance.size else float("inf")
        best_path = np.stack([pos_xy, pos_xy], axis=0)
    else:
        best_cost = float(costs[best_idx])
        best_path = np.asarray(cand_paths[best_idx], dtype=np.float64)
        min_clear = path_min_clearance(
            best_path,
            local_obstacles,
            uav_radius=cfg.uav_radius,
            safety_margin=cfg.safety_margin,
        )
        if score_cfg.world_xy is not None:
            min_clear = min(
                float(min_clear),
                points_min_boundary_clearance(
                    best_path,
                    world_xy=score_cfg.world_xy,
                    boundary_margin=score_cfg.boundary_margin,
                ),
            )

    return LocalReachabilityResult(
        blocked=blocked,
        min_clearance=float(min_clear),
        best_cost=best_cost,
        best_idx=best_idx,
        best_path=best_path,
        best_speed=float(cand_speeds[best_idx]) if not blocked else 0.0,
        best_yaw_rate=float(cand_yaw_rates[best_idx]) if not blocked else 0.0,
        local_obstacles=tuple(local_obstacles),
        candidate_count=int(costs.size),
        valid_candidate_count=valid_count,
        actual_speed_xy=actual_speed_xy,
        heading_err_now=heading_err_now,
        speed_gate=speed_gate,
        out_of_bounds_candidate_count=out_of_bounds_candidate_count,
        emergency_stop_reason=emergency_stop_reason,
    )
