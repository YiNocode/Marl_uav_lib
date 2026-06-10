"""Local sampled obstacle avoidance for the E2 debug trajectory planner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.framework.planning.local_reachability import (
    LocalReachabilityScoringConfig,
    local_reachability_probe,
)
from marl_uav.framework.planning.turn_radius_obstacle_query import (
    TurnRadiusObstacleQueryConfig,
    effective_omega_max,
    wrap_to_pi,
)


@dataclass(frozen=True)
class ObstacleAvoidanceConfig:
    horizon_s: float = 2.5
    dt: float = 0.1
    vmax: float = 0.25
    omega_max: float = 1.0
    num_yaw_samples: int = 21
    speed_samples: tuple[float, ...] = (1.0, 0.6, 0.3, 0.0)
    min_turn_radius: float = 0.25
    lookahead_dist: float = 3.0
    uav_radius: float = 0.15
    safety_margin: float = 0.60
    query_extra_margin: float = 0.50
    collision_large_penalty: float = 1_000_000.0
    w_goal: float = 0.5
    w_heading: float = 1.0
    w_obstacle: float = 20.0
    w_smooth: float = 0.1
    w_speed: float = 0.0
    w_vel_smooth: float = 0.0
    heading_slowdown_angle: float = 0.85
    min_turn_speed_ratio: float = 0.15
    goal_slowdown_radius: float = 0.60
    direct_los_enabled: bool = True
    # Default planner for multirotor UAVs: body-frame vx/vy is holonomic, so
    # translation must not be constrained to the current yaw direction.
    prefer_holonomic_tracking: bool = True
    use_sampled_planner: bool = False
    position_kp: float = 1.0
    velocity_kd: float = 0.0
    arrival_radius: float = 0.08
    yaw_time_constant: float = 0.35
    obstacle_influence_radius: float = 2.0
    obstacle_barrier_gain: float = 1.5
    obstacle_repulse_gain: float = 0.35
    fallback_zero_if_blocked: bool = True
    use_candidate_rollout_filter: bool = True
    world_xy: float | None = None
    boundary_margin: float = 0.30
    boundary_activation_distance: float = 0.60
    boundary_hard_margin: float = 0.05
    boundary_braking_margin: float = 0.30
    boundary_braking_gain: float = 1.0
    max_inward_correction: float = 1.0
    amax_xy: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ObstacleAvoidanceConfig":
        d = dict(raw or {})
        speeds = d.get("speed_samples", cls.speed_samples)
        world_xy = d.get("world_xy", cls.world_xy)
        return cls(
            horizon_s=float(d.get("horizon_s", cls.horizon_s)),
            dt=float(d.get("dt", cls.dt)),
            vmax=float(d.get("vmax", cls.vmax)),
            omega_max=float(d.get("omega_max", cls.omega_max)),
            num_yaw_samples=int(d.get("num_yaw_samples", cls.num_yaw_samples)),
            speed_samples=tuple(float(x) for x in speeds),
            min_turn_radius=float(d.get("min_turn_radius", cls.min_turn_radius)),
            lookahead_dist=float(d.get("lookahead_dist", cls.lookahead_dist)),
            uav_radius=float(d.get("uav_radius", cls.uav_radius)),
            safety_margin=float(d.get("safety_margin", cls.safety_margin)),
            query_extra_margin=float(d.get("query_extra_margin", cls.query_extra_margin)),
            collision_large_penalty=float(
                d.get("collision_large_penalty", cls.collision_large_penalty)
            ),
            w_goal=float(d.get("w_goal", cls.w_goal)),
            w_heading=float(d.get("w_heading", cls.w_heading)),
            w_obstacle=float(d.get("w_obstacle", cls.w_obstacle)),
            w_smooth=float(d.get("w_smooth", cls.w_smooth)),
            w_speed=float(d.get("w_speed", cls.w_speed)),
            w_vel_smooth=float(d.get("w_vel_smooth", cls.w_vel_smooth)),
            heading_slowdown_angle=float(d.get("heading_slowdown_angle", cls.heading_slowdown_angle)),
            min_turn_speed_ratio=float(d.get("min_turn_speed_ratio", cls.min_turn_speed_ratio)),
            goal_slowdown_radius=float(d.get("goal_slowdown_radius", cls.goal_slowdown_radius)),
            direct_los_enabled=bool(d.get("direct_los_enabled", cls.direct_los_enabled)),
            prefer_holonomic_tracking=bool(d.get("prefer_holonomic_tracking", cls.prefer_holonomic_tracking)),
            use_sampled_planner=bool(d.get("use_sampled_planner", cls.use_sampled_planner)),
            position_kp=float(d.get("position_kp", cls.position_kp)),
            velocity_kd=float(d.get("velocity_kd", d.get("kd", cls.velocity_kd))),
            arrival_radius=float(d.get("arrival_radius", cls.arrival_radius)),
            yaw_time_constant=float(d.get("yaw_time_constant", cls.yaw_time_constant)),
            obstacle_influence_radius=float(d.get("obstacle_influence_radius", cls.obstacle_influence_radius)),
            obstacle_barrier_gain=float(d.get("obstacle_barrier_gain", cls.obstacle_barrier_gain)),
            obstacle_repulse_gain=float(d.get("obstacle_repulse_gain", cls.obstacle_repulse_gain)),
            fallback_zero_if_blocked=bool(
                d.get("fallback_zero_if_blocked", cls.fallback_zero_if_blocked)
            ),
            use_candidate_rollout_filter=bool(
                d.get("use_candidate_rollout_filter", cls.use_candidate_rollout_filter)
            ),
            world_xy=None if world_xy is None else float(world_xy),
            boundary_margin=float(d.get("boundary_margin", cls.boundary_margin)),
            boundary_activation_distance=float(
                d.get("boundary_activation_distance", cls.boundary_activation_distance)
            ),
            boundary_hard_margin=float(d.get("boundary_hard_margin", cls.boundary_hard_margin)),
            boundary_braking_margin=float(
                d.get("boundary_braking_margin", d.get("boundary_hard_margin", cls.boundary_braking_margin))
            ),
            boundary_braking_gain=float(d.get("boundary_braking_gain", cls.boundary_braking_gain)),
            max_inward_correction=float(d.get("max_inward_correction", cls.max_inward_correction)),
            amax_xy=None if d.get("amax_xy") is None else float(d["amax_xy"]),
        )

    def query_cfg(self) -> TurnRadiusObstacleQueryConfig:
        return TurnRadiusObstacleQueryConfig(
            horizon_s=self.horizon_s,
            dt=self.dt,
            vmax=self.vmax,
            omega_max=self.omega_max,
            num_yaw_samples=self.num_yaw_samples,
            speed_samples=self.speed_samples,
            min_turn_radius=self.min_turn_radius,
            lookahead_dist=self.lookahead_dist,
            uav_radius=self.uav_radius,
            safety_margin=self.safety_margin,
            query_extra_margin=self.query_extra_margin,
            use_candidate_rollout_filter=self.use_candidate_rollout_filter,
            amax_xy=self.amax_xy,
        )

    def scoring_cfg(self) -> LocalReachabilityScoringConfig:
        return LocalReachabilityScoringConfig(
            w_goal=self.w_goal,
            w_heading=self.w_heading,
            w_obstacle=self.w_obstacle,
            w_smooth=self.w_smooth,
            w_speed=self.w_speed,
            w_vel_smooth=self.w_vel_smooth,
            vmax=self.vmax,
            collision_large_penalty=self.collision_large_penalty,
            world_xy=self.world_xy,
            boundary_margin=self.boundary_margin,
            amax_xy=self.amax_xy,
        )


class ObstacleAvoidanceController:
    """Pick one local unicycle rollout and expose its path for debug drawing."""

    def __init__(self, cfg: dict[str, Any] | ObstacleAvoidanceConfig | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, ObstacleAvoidanceConfig) else ObstacleAvoidanceConfig.from_dict(cfg)

    @staticmethod
    def _as_xy_radius(obstacle: Any) -> tuple[np.ndarray, float] | None:
        """Best-effort obstacle parser for the holonomic safety layer.

        Unknown formats are ignored by the vector tracker instead of forcing the
        controller into the slow sampled rollout branch. The original obstacle
        objects are still available to the sampled planner when that mode is
        explicitly enabled.
        """
        if isinstance(obstacle, dict):
            xy_obj = None
            for key in ("xy", "center", "centre", "pos", "position", "center_xy"):
                if key in obstacle:
                    xy_obj = obstacle[key]
                    break
            if xy_obj is None and "x" in obstacle and "y" in obstacle:
                xy_obj = (obstacle["x"], obstacle["y"])
            if xy_obj is None:
                return None
            try:
                xy = np.asarray(xy_obj, dtype=np.float64).reshape(-1)
            except Exception:
                return None
            if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
                return None
            radius = 0.0
            for key in ("radius", "r", "rad", "obstacle_radius", "cylinder_radius"):
                if key in obstacle:
                    try:
                        radius = float(obstacle[key])
                    except Exception:
                        radius = 0.0
                    break
            return xy[:2].copy(), max(radius, 0.0)

        if isinstance(obstacle, (tuple, list)):
            arr = np.asarray(obstacle, dtype=object).reshape(-1)
            if arr.size >= 3:
                try:
                    return np.asarray([arr[0], arr[1]], dtype=np.float64), max(float(arr[2]), 0.0)
                except Exception:
                    return None
            if arr.size >= 2:
                try:
                    return np.asarray([arr[0], arr[1]], dtype=np.float64), 0.0
                except Exception:
                    return None

        xy_obj = None
        for key in ("xy", "center", "centre", "pos", "position", "center_xy"):
            if hasattr(obstacle, key):
                xy_obj = getattr(obstacle, key)
                break
        if xy_obj is None and hasattr(obstacle, "x") and hasattr(obstacle, "y"):
            xy_obj = (getattr(obstacle, "x"), getattr(obstacle, "y"))
        if xy_obj is None:
            return None
        try:
            xy = np.asarray(xy_obj, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
            return None
        radius = 0.0
        for key in ("radius", "r", "rad", "obstacle_radius", "cylinder_radius"):
            if hasattr(obstacle, key):
                try:
                    radius = float(getattr(obstacle, key))
                except Exception:
                    radius = 0.0
                break
        return xy[:2].copy(), max(radius, 0.0)

    @staticmethod
    def _clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64).reshape(2)
        n = float(np.linalg.norm(v))
        if limit > 0.0 and n > limit:
            return v * (float(limit) / max(n, 1e-9))
        return v

    @staticmethod
    def _boundary_records(
        pos_xy: np.ndarray,
        cfg: ObstacleAvoidanceConfig,
        bounds_xy: tuple[float, float, float, float] | None,
    ) -> list[tuple[str, np.ndarray, float]]:
        p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
        if bounds_xy is not None:
            xmin, xmax, ymin, ymax = (float(x) for x in bounds_xy)
        elif cfg.world_xy is not None and np.isfinite(float(cfg.world_xy)):
            half = max(float(cfg.world_xy) - max(float(cfg.boundary_margin), 0.0), 0.0)
            xmin, xmax, ymin, ymax = -half, half, -half, half
        else:
            return []
        return [
            ("x_min", np.array([-1.0, 0.0], dtype=np.float64), float(p[0] - xmin)),
            ("x_max", np.array([1.0, 0.0], dtype=np.float64), float(xmax - p[0])),
            ("y_min", np.array([0.0, -1.0], dtype=np.float64), float(p[1] - ymin)),
            ("y_max", np.array([0.0, 1.0], dtype=np.float64), float(ymax - p[1])),
        ]

    @staticmethod
    def _add_inward_correction_preserve_tangent(
        action_xy: np.ndarray,
        normal: np.ndarray,
        *,
        correction: float,
        vmax: float,
    ) -> tuple[np.ndarray, float]:
        action = np.asarray(action_xy, dtype=np.float64).reshape(2)
        n = np.asarray(normal, dtype=np.float64).reshape(2)
        requested = max(float(correction), 0.0)
        if requested <= 0.0:
            return action.copy(), 0.0
        tangent = action - float(np.dot(action, n)) * n
        tangent_norm = float(np.linalg.norm(tangent))
        limit = max(float(vmax), 0.0)
        if limit <= 0.0:
            return action.copy(), 0.0
        max_normal = float(np.sqrt(max(limit * limit - tangent_norm * tangent_norm, 0.0)))
        current_outward = float(np.dot(action, n))
        desired_outward = current_outward - requested
        limited_outward = max(desired_outward, -max_normal)
        applied = max(current_outward - limited_outward, 0.0)
        return tangent + limited_outward * n, float(applied)

    @staticmethod
    def _componentwise_boundary_filter(
        pos_xy: np.ndarray,
        action_xy: np.ndarray,
        current_velocity_xy: np.ndarray,
        *,
        cfg: ObstacleAvoidanceConfig,
        bounds_xy: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        u = np.asarray(action_xy, dtype=np.float64).reshape(2).copy()
        before = u.copy()
        vel = np.asarray(current_velocity_xy, dtype=np.float64).reshape(2)
        records = ObstacleAvoidanceController._boundary_records(pos_xy, cfg, bounds_xy)
        if not records:
            return u, False, {"boundary_active_names": [], "boundary_projected_action_failed": False}

        active_names: list[str] = []
        details: dict[str, Any] = {}
        projected_failed = False
        activation = max(float(cfg.boundary_activation_distance), 0.0)
        braking_margin_threshold = max(float(cfg.boundary_braking_margin), 0.0)
        amax = (
            max(float(cfg.amax_xy), 1e-9)
            if cfg.amax_xy is not None
            else max(float(cfg.vmax) / max(float(cfg.dt), 1e-6), 1e-9)
        )

        for name, normal, distance in records:
            if distance > activation:
                continue
            active_names.append(name)
            action_before_axis = u.copy()
            v_out = float(np.dot(vel, normal))
            action_out_before = float(np.dot(u, normal))
            braking_distance = (max(v_out, 0.0) ** 2) / max(2.0 * amax, 1e-9)
            braking_margin = float(distance - braking_distance)
            inward_correction = 0.0

            if action_out_before > 0.0:
                u = u - action_out_before * normal
            if braking_margin < braking_margin_threshold:
                requested = float(cfg.boundary_braking_gain) * (braking_margin_threshold - braking_margin)
                requested = min(max(requested, 0.0), max(float(cfg.max_inward_correction), 0.0))
                u, inward_correction = ObstacleAvoidanceController._add_inward_correction_preserve_tangent(
                    u,
                    normal,
                    correction=requested,
                    vmax=float(cfg.vmax),
                )
            action_out_mid = float(np.dot(u, normal))
            if action_out_mid > 0.0:
                u = u - action_out_mid * normal

            action_out_after = float(np.dot(u, normal))
            if distance < float(cfg.boundary_hard_margin) and action_out_after > 1e-6:
                projected_failed = True
            normal_before = float(np.dot(before, normal))
            normal_after = float(np.dot(u, normal))
            details[name] = {
                "distance_to_boundary": float(distance),
                "velocity_outward_projection": float(v_out),
                "action_outward_projection_before": float(action_out_before),
                "action_outward_projection_after": float(action_out_after),
                "estimated_braking_distance": float(braking_distance),
                "braking_margin": float(braking_margin),
                "inward_correction": float(inward_correction),
                "normal_component_before": float(normal_before),
                "normal_component_after": float(normal_after),
                "tangential_norm_before": float(np.linalg.norm(before - normal_before * normal)),
                "tangential_norm_after": float(np.linalg.norm(u - normal_after * normal)),
                "axis_delta_norm": float(np.linalg.norm(u - action_before_axis)),
            }

        u = ObstacleAvoidanceController._clip_norm(u, float(cfg.vmax))
        for name, normal, distance in records:
            if name not in active_names:
                continue
            action_out_after = float(np.dot(u, normal))
            if distance < float(cfg.boundary_hard_margin) and action_out_after > 1e-6:
                projected_failed = True
            if name in details:
                details[name]["action_outward_projection_after"] = action_out_after

        if projected_failed:
            print("BOUNDARY_PROJECTED_ACTION_FAILED")

        return u, bool(np.linalg.norm(u - before) > 1e-9), {
            "boundary_active_names": active_names,
            "boundary_filter_details": details,
            "boundary_projected_action_failed": bool(projected_failed),
        }

    @staticmethod
    def _holonomic_barrier_tracking_action(
        pos: np.ndarray,
        goal: np.ndarray,
        yaw: float,
        obstacles: list[Any],
        cfg: ObstacleAvoidanceConfig,
        *,
        feedforward_velocity_xy: np.ndarray | None = None,
        current_velocity_xy: np.ndarray | None = None,
        bounds_xy: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
        """Pure-pursuit-like slot tracking with a local safety projection.

        This is the default branch for multirotors. It keeps the strong direct
        target tracking behavior of pure pursuit, then projects the velocity
        away from nearby obstacle/boundary hazards. It avoids the main failure
        mode of the sampled unicycle rollout: reducing a holonomic UAV to
        current-yaw forward motion.
        """
        p = np.asarray(pos, dtype=np.float64).reshape(2)
        g = np.asarray(goal, dtype=np.float64).reshape(2)
        vel = (
            np.zeros(2, dtype=np.float64)
            if current_velocity_xy is None
            else np.asarray(current_velocity_xy, dtype=np.float64).reshape(2)
        )
        delta = g - p
        dist = float(np.linalg.norm(delta))
        if dist <= max(float(cfg.arrival_radius), 1e-6):
            feedback = np.zeros(2, dtype=np.float64)
            bearing = float(yaw)
        else:
            direction = delta / max(dist, 1e-9)
            feedback = float(cfg.position_kp) * delta
            bearing = float(np.arctan2(direction[1], direction[0]))

        ff = (
            np.zeros(2, dtype=np.float64)
            if feedforward_velocity_xy is None
            else np.asarray(feedforward_velocity_xy, dtype=np.float64).reshape(2)
        )
        u_raw = feedback + ff - float(cfg.velocity_kd) * vel
        u = ObstacleAvoidanceController._clip_norm(u_raw, float(cfg.vmax))

        parsed_count = 0
        active_obstacle_count = 0
        min_clearance = 1e9
        safety = float(cfg.uav_radius) + float(cfg.safety_margin)
        influence = max(float(cfg.obstacle_influence_radius), safety + 1e-6)
        repulse_gain = max(float(cfg.obstacle_repulse_gain), 0.0)
        barrier_gain = max(float(cfg.obstacle_barrier_gain), 0.0)

        for obs in list(obstacles or []):
            parsed = ObstacleAvoidanceController._as_xy_radius(obs)
            if parsed is None:
                continue
            center, radius = parsed
            parsed_count += 1
            rel = p - center
            d = float(np.linalg.norm(rel))
            if d <= 1e-9:
                # Degenerate case: choose a deterministic outward normal.
                n = np.array([1.0, 0.0], dtype=np.float64)
            else:
                n = rel / d
            clearance = d - float(radius) - safety
            min_clearance = min(min_clearance, clearance)
            if clearance < influence:
                active_obstacle_count += 1
                # Soft repulsion improves clearance when there is room.
                strength = repulse_gain * (influence - clearance) / max(influence, 1e-6)
                u = u + strength * n
                # First-order CBF-style velocity projection: do not move inward
                # faster than gamma * clearance. Negative clearance makes the
                # constraint actively push outward.
                inward = float(np.dot(n, u))
                lower_bound = -barrier_gain * clearance
                if inward < lower_bound:
                    u = u + (lower_bound - inward) * n
                u = ObstacleAvoidanceController._clip_norm(u, float(cfg.vmax))

        u_before_boundary = u.copy()
        u, boundary_active, boundary_diag = ObstacleAvoidanceController._componentwise_boundary_filter(
            p,
            u,
            vel,
            cfg=cfg,
            bounds_xy=bounds_xy,
        )

        speed = float(np.linalg.norm(u))
        if speed > 1e-9:
            yaw_target = float(np.arctan2(u[1], u[0]))
        else:
            yaw_target = bearing
        omega = effective_omega_max(cfg.vmax, cfg.omega_max, cfg.min_turn_radius)
        yaw_err = wrap_to_pi(yaw_target - float(yaw))
        yaw_rate = float(np.clip(yaw_err / max(float(cfg.yaw_time_constant), 1e-3), -omega, omega))
        best_path = np.stack([p, p + max(float(cfg.horizon_s), 1e-3) * u], axis=0)
        diag = {
            "guidance_mode": "holonomic_barrier_tracking",
            "local_obstacle_count": int(parsed_count),
            "active_obstacle_count": int(active_obstacle_count),
            "candidate_count": 1,
            "valid_candidate_count": 1,
            "best_candidate_cost": float(dist),
            "best_candidate_speed": float(speed),
            "best_candidate_yaw_rate": float(yaw_rate),
            "min_predicted_clearance": float(min_clearance),
            "local_planner_blocked": False,
            "assigned_slot_distance": float(dist),
            "selected_action_norm": float(speed),
            "raw_selected_speed": float(np.linalg.norm(u_raw)),
            "nominal_cmd_xy": u_raw.astype(float).tolist(),
            "heading_speed_scale": 1.0,
            "goal_speed_scale": 1.0,
            "feedforward_speed_norm": float(np.linalg.norm(ff)),
            "best_path_xy": best_path.astype(float).tolist(),
            "local_obstacles": [],
            "actual_speed_xy": 0.0,
            "heading_err_now": float(wrap_to_pi(bearing - float(yaw))),
            "speed_gate": "holonomic_barrier",
            "out_of_bounds_candidate_count": 0,
            "boundary_margin": float(cfg.boundary_margin),
            "boundary_filter_active": bool(boundary_active),
            "action_before_boundary_filter": u_before_boundary.astype(float).tolist(),
            "action_after_boundary_filter": u.astype(float).tolist(),
            "emergency_stop_reason": None,
        }
        diag.update(boundary_diag)
        return u.astype(np.float64), yaw_rate, best_path.astype(np.float64), diag

    @staticmethod
    def _direct_vector_tracking_action(
        pos: np.ndarray,
        goal: np.ndarray,
        yaw: float,
        cfg: ObstacleAvoidanceConfig,
        feedforward_velocity_xy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, dict[str, float]]:
        """Holonomic point tracking used when the straight segment is clear.

        The environment action already accepts body-frame vx/vy.  Therefore the
        shortest correct controller for an unobstructed assigned slot is a world
        vector velocity toward the slot, later transformed to body frame by the
        trajectory planner.  Yaw is only aligned for attitude/debugging; it must
        not gate translation.
        """
        delta = np.asarray(goal, dtype=np.float64).reshape(2) - np.asarray(pos, dtype=np.float64).reshape(2)
        dist = float(np.linalg.norm(delta))
        ff = (
            np.zeros(2, dtype=np.float64)
            if feedforward_velocity_xy is None
            else np.asarray(feedforward_velocity_xy, dtype=np.float64).reshape(2)
        )
        if dist <= max(float(cfg.arrival_radius), 1e-6):
            feedback = np.zeros(2, dtype=np.float64)
            bearing = float(yaw)
        else:
            direction = delta / max(dist, 1e-9)
            feedback = min(float(cfg.vmax), float(cfg.position_kp) * dist) * direction
            bearing = float(np.arctan2(direction[1], direction[0]))
        action_xy = feedback + ff
        norm = float(np.linalg.norm(action_xy))
        if norm > float(cfg.vmax) > 0.0:
            action_xy = action_xy * (float(cfg.vmax) / max(norm, 1e-9))
        speed = float(np.linalg.norm(action_xy))
        omega = effective_omega_max(cfg.vmax, cfg.omega_max, cfg.min_turn_radius)
        yaw_err = wrap_to_pi(bearing - float(yaw))
        yaw_rate = float(np.clip(yaw_err / max(float(cfg.yaw_time_constant), 1e-3), -omega, omega))
        return action_xy, yaw_rate, speed, {
            "raw_selected_speed": float(speed),
            "heading_speed_scale": 1.0,
            "goal_speed_scale": 1.0,
            "feedforward_speed_norm": float(np.linalg.norm(ff)),
        }

    @staticmethod
    def _scale_speed_for_tracking(
        speed: float,
        *,
        goal_dist: float,
        heading_err: float,
        cfg: ObstacleAvoidanceConfig,
    ) -> tuple[float, dict[str, float]]:
        """Apply executable speed gates that the rollout score alone cannot guarantee.

        The local rollout may choose a high forward speed while also requesting
        a large yaw-rate. In a velocity+yaw-rate interface, that means the UAV
        still moves along its current heading during the next control tick. This
        gate slows down when the goal/waypoint is far off the current heading
        and near the final waypoint, reducing boundary overshoot and late turns.
        """
        raw_speed = max(float(speed), 0.0)
        abs_heading = abs(float(heading_err))
        if cfg.heading_slowdown_angle <= 0.0:
            heading_scale = 1.0
        else:
            # 1 before the threshold; then smoothly decay to min_turn_speed_ratio at pi rad.
            denom = max(np.pi - cfg.heading_slowdown_angle, 1e-6)
            decay = np.clip((abs_heading - cfg.heading_slowdown_angle) / denom, 0.0, 1.0)
            heading_scale = (1.0 - decay) + decay * float(cfg.min_turn_speed_ratio)

        if cfg.goal_slowdown_radius <= 1e-6:
            goal_scale = 1.0
        else:
            goal_scale = np.clip(float(goal_dist) / float(cfg.goal_slowdown_radius), 0.20, 1.0)

        scaled = raw_speed * float(min(heading_scale, goal_scale))
        return scaled, {
            "raw_selected_speed": raw_speed,
            "heading_speed_scale": float(heading_scale),
            "goal_speed_scale": float(goal_scale),
        }

    def compute_action(
        self,
        pos_xy: np.ndarray,
        yaw: float,
        goal_xy: np.ndarray,
        obstacles: list[Any],
        prev_action: np.ndarray | None = None,
        current_velocity_xy: np.ndarray | None = None,
        *,
        current_vel_xy: np.ndarray | None = None,
        bounds_xy: tuple[float, float, float, float] | None = None,
        feedforward_velocity_xy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
        t0 = time.perf_counter()
        cfg = self.cfg
        pos = np.asarray(pos_xy, dtype=np.float64).reshape(2)
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        goal_dist = float(np.linalg.norm(goal - pos))
        goal_bearing = float(np.arctan2(goal[1] - pos[1], goal[0] - pos[0])) if goal_dist > 1e-9 else float(yaw)
        vel_xy = current_vel_xy if current_vel_xy is not None else current_velocity_xy

        if cfg.prefer_holonomic_tracking and not cfg.use_sampled_planner:
            action_xy, best_yaw_rate, best_path, diag = self._holonomic_barrier_tracking_action(
                pos,
                goal,
                float(yaw),
                list(obstacles or []),
                cfg,
                feedforward_velocity_xy=feedforward_velocity_xy,
                current_velocity_xy=vel_xy,
                bounds_xy=bounds_xy,
            )
            diag["local_planner_time_ms"] = float((time.perf_counter() - t0) * 1000.0)
            diag["actual_speed_xy"] = 0.0 if vel_xy is None else float(
                np.linalg.norm(np.asarray(vel_xy, dtype=np.float64).reshape(2))
            )
            return action_xy.astype(np.float64), float(best_yaw_rate), best_path.astype(np.float64), diag

        if cfg.direct_los_enabled and not obstacles:
            action_xy, best_yaw_rate, best_speed, speed_diag = self._direct_vector_tracking_action(
                pos, goal, float(yaw), cfg, feedforward_velocity_xy=feedforward_velocity_xy
            )
            best_path = np.stack([pos, goal], axis=0)
            diag = {
                "guidance_mode": "direct_los",
                "local_obstacle_count": 0,
                "candidate_count": 1,
                "valid_candidate_count": 1,
                "best_candidate_cost": float(goal_dist),
                "best_candidate_speed": float(best_speed),
                "best_candidate_yaw_rate": float(best_yaw_rate),
                "min_predicted_clearance": 1e9,
                "local_planner_blocked": False,
                "local_planner_time_ms": float((time.perf_counter() - t0) * 1000.0),
                "assigned_slot_distance": float(goal_dist),
                "selected_action_norm": float(np.linalg.norm(action_xy)),
                **speed_diag,
                "best_path_xy": best_path.astype(float).tolist(),
                "local_obstacles": [],
                "actual_speed_xy": 0.0 if vel_xy is None else float(np.linalg.norm(np.asarray(vel_xy, dtype=np.float64).reshape(2))),
                "heading_err_now": float(wrap_to_pi(goal_bearing - float(yaw))),
                "speed_gate": "direct_los",
                "out_of_bounds_candidate_count": 0,
                "boundary_margin": float(cfg.boundary_margin),
                "emergency_stop_reason": None,
            }
            return action_xy.astype(np.float64), float(best_yaw_rate), best_path.astype(np.float64), diag

        reach = local_reachability_probe(
            pos,
            float(yaw),
            goal,
            obstacles,
            cfg=cfg.query_cfg(),
            scoring=cfg.scoring_cfg(),
            prev_action=prev_action,
            current_vel_xy=vel_xy,
            bounds_xy=bounds_xy,
            executable_safety=True,
        )

        emergency_stop_reason = reach.emergency_stop_reason
        speed_diag = {
            "raw_selected_speed": 0.0,
            "heading_speed_scale": 0.0,
            "goal_speed_scale": 0.0,
        }
        if reach.blocked:
            omega = effective_omega_max(cfg.vmax, cfg.omega_max, cfg.min_turn_radius)
            yaw_rates = np.linspace(-omega, omega, max(int(cfg.num_yaw_samples), 1), dtype=np.float64)
            yaw_err = wrap_to_pi(goal_bearing - float(yaw))
            max_yaw = float(yaw_rates[-1]) if yaw_rates.size else 0.0
            best_speed = 0.0
            best_yaw_rate = float(np.clip(yaw_err / max(cfg.horizon_s, 1e-3), -max_yaw, max_yaw))
            if cfg.fallback_zero_if_blocked:
                action_xy = np.zeros(2, dtype=np.float64)
                best_path = np.stack([pos, pos], axis=0)
                if emergency_stop_reason is None:
                    emergency_stop_reason = "all_candidates_blocked"
            else:
                action_xy = 0.05 * cfg.vmax * np.array([np.cos(float(yaw)), np.sin(float(yaw))])
                best_path = np.stack([pos, pos + cfg.horizon_s * action_xy], axis=0)
        else:
            raw_best_speed = float(reach.best_speed)
            best_yaw_rate = float(reach.best_yaw_rate)
            best_speed, speed_diag = self._scale_speed_for_tracking(
                raw_best_speed,
                goal_dist=goal_dist,
                heading_err=wrap_to_pi(goal_bearing - float(yaw)),
                cfg=cfg,
            )
            action_xy = best_speed * np.array([np.cos(float(yaw)), np.sin(float(yaw))], dtype=np.float64)
            best_path = np.asarray(reach.best_path, dtype=np.float64)

        diag = {
            "guidance_mode": "sampled_local_rollout",
            "local_obstacle_count": int(len(reach.local_obstacles)),
            "candidate_count": int(reach.candidate_count),
            "valid_candidate_count": int(reach.valid_candidate_count),
            "best_candidate_cost": float(reach.best_cost),
            "best_candidate_speed": float(best_speed),
            "best_candidate_yaw_rate": float(best_yaw_rate),
            "min_predicted_clearance": float(reach.min_clearance),
            "local_planner_blocked": bool(reach.blocked),
            "local_planner_time_ms": float((time.perf_counter() - t0) * 1000.0),
            "assigned_slot_distance": float(goal_dist),
            "selected_action_norm": float(np.linalg.norm(action_xy)),
            **speed_diag,
            "best_path_xy": best_path.astype(float).tolist(),
            "local_obstacles": list(reach.local_obstacles),
            "actual_speed_xy": float(reach.actual_speed_xy),
            "heading_err_now": float(reach.heading_err_now),
            "speed_gate": str(reach.speed_gate),
            "out_of_bounds_candidate_count": int(reach.out_of_bounds_candidate_count),
            "boundary_margin": float(cfg.boundary_margin),
            "emergency_stop_reason": emergency_stop_reason,
        }
        return action_xy.astype(np.float64), float(best_yaw_rate), best_path.astype(np.float64), diag
