"""Lightweight turn-radius local planner for assigned SCE slots."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.framework.planning.turn_radius_obstacle_query import (
    TurnRadiusObstacleQueryConfig,
    batch_path_min_clearance,
    effective_omega_max,
    path_min_clearance,
    query_turn_radius_obstacles,
    resolve_speed_samples,
    rollout_unicycle_paths_batch,
    wrap_to_pi,
)


@dataclass(frozen=True)
class TurnRadiusSlotPlannerConfig:
    enabled: bool = True
    horizon_s: float = 1.5
    dt: float = 0.1
    vmax: float = 0.25
    omega_max: float = 1.0
    num_yaw_samples: int = 11
    speed_samples: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    min_turn_radius: float = 0.25
    lookahead_dist: float = 1.5
    uav_radius: float = 0.15
    safety_margin: float = 0.30
    query_extra_margin: float = 0.20
    collision_large_penalty: float = 1_000_000.0
    w_goal: float = 1.0
    w_heading: float = 0.2
    w_obstacle: float = 5.0
    w_smooth: float = 0.1
    w_speed: float = -0.05
    fallback_zero_if_blocked: bool = True
    use_candidate_rollout_filter: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TurnRadiusSlotPlannerConfig":
        d = dict(raw or {})
        speeds = d.get("speed_samples", cls.speed_samples)
        return cls(
            enabled=bool(d.get("enabled", cls.enabled)),
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
            collision_large_penalty=float(d.get("collision_large_penalty", cls.collision_large_penalty)),
            w_goal=float(d.get("w_goal", cls.w_goal)),
            w_heading=float(d.get("w_heading", cls.w_heading)),
            w_obstacle=float(d.get("w_obstacle", cls.w_obstacle)),
            w_smooth=float(d.get("w_smooth", cls.w_smooth)),
            w_speed=float(d.get("w_speed", cls.w_speed)),
            fallback_zero_if_blocked=bool(d.get("fallback_zero_if_blocked", cls.fallback_zero_if_blocked)),
            use_candidate_rollout_filter=bool(
                d.get("use_candidate_rollout_filter", cls.use_candidate_rollout_filter)
            ),
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
        )


class TurnRadiusSlotPlanner:
    """Candidate-rollout local planner from current UAV state to one slot."""

    def __init__(self, cfg: dict[str, Any] | TurnRadiusSlotPlannerConfig | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, TurnRadiusSlotPlannerConfig) else TurnRadiusSlotPlannerConfig.from_dict(cfg)

    def _yaw_rate_candidates(self) -> np.ndarray:
        omega = effective_omega_max(self.cfg.vmax, self.cfg.omega_max, self.cfg.min_turn_radius)
        return np.linspace(-omega, omega, max(int(self.cfg.num_yaw_samples), 1), dtype=np.float64)

    def _speed_candidates(self) -> np.ndarray:
        return resolve_speed_samples(self.cfg.vmax, self.cfg.speed_samples)

    def compute_action(
        self,
        pos_xy: np.ndarray,
        yaw: float,
        vel_xy: np.ndarray,
        assigned_slot_xy: np.ndarray,
        obstacles: list[Any],
        prev_action: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return world-frame XY velocity and diagnostics.

        The planner cost only contains slot-reaching, heading, obstacle
        clearance, smoothness, and speed terms.  Encirclement metrics such as
        D_ang, C_cov, and C_col are intentionally absent from this local layer.
        """
        del vel_xy
        t0 = time.perf_counter()
        cfg = self.cfg
        pos = np.asarray(pos_xy, dtype=np.float64).reshape(2)
        slot = np.asarray(assigned_slot_xy, dtype=np.float64).reshape(2)
        goal_vec = slot - pos
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_bearing = float(np.arctan2(goal_vec[1], goal_vec[0])) if goal_dist > 1e-9 else float(yaw)

        yaw_rates = self._yaw_rate_candidates()
        speeds = self._speed_candidates()
        speed_grid, yaw_rate_grid = np.meshgrid(speeds, yaw_rates, indexing="ij")
        cand_speeds = speed_grid.reshape(-1)
        cand_yaw_rates = yaw_rate_grid.reshape(-1)
        cand_paths, cand_final_yaws = rollout_unicycle_paths_batch(
            pos,
            float(yaw),
            cand_speeds,
            cand_yaw_rates,
            horizon_s=cfg.horizon_s,
            dt=cfg.dt,
        )

        local_obstacles = query_turn_radius_obstacles(
            pos,
            float(yaw),
            slot,
            obstacles,
            cfg.query_cfg(),
            candidate_paths=cand_paths,
        )
        prev = None if prev_action is None else np.asarray(prev_action, dtype=np.float64).reshape(-1)

        clearances = batch_path_min_clearance(
            cand_paths,
            local_obstacles,
            uav_radius=cfg.uav_radius,
            safety_margin=cfg.safety_margin,
        )
        finite_clearance = np.where(np.isfinite(clearances), clearances, np.inf)
        collides = finite_clearance <= 0.0
        final_goal = np.linalg.norm(cand_paths[:, -1, :] - slot[None, :], axis=1)
        heading_err = np.abs(np.asarray([wrap_to_pi(goal_bearing - fy) for fy in cand_final_yaws], dtype=np.float64))
        obstacle_penalty = np.zeros_like(final_goal)
        finite_mask = np.isfinite(finite_clearance)
        obstacle_penalty[finite_mask] = 1.0 / np.maximum(finite_clearance[finite_mask], 1e-3)
        speed_penalty = cand_speeds / max(float(cfg.vmax), 1e-6)
        smooth_penalty = np.zeros_like(final_goal)
        if prev is not None and prev.shape[0] >= 2:
            smooth_penalty = np.linalg.norm(
                np.stack([cand_speeds - prev[0], cand_yaw_rates - prev[1]], axis=1),
                axis=1,
            )
        costs = (
            cfg.w_goal * final_goal
            + cfg.w_heading * heading_err
            + cfg.w_obstacle * obstacle_penalty
            + cfg.w_smooth * smooth_penalty
            + cfg.w_speed * speed_penalty
        )
        costs = np.where(collides, cfg.collision_large_penalty, costs)
        best_idx = int(np.argmin(costs)) if costs.size else 0
        best_cost = float(costs[best_idx]) if costs.size else float("inf")
        best_speed = float(cand_speeds[best_idx]) if costs.size else 0.0
        best_yaw_rate = float(cand_yaw_rates[best_idx]) if costs.size else 0.0
        best_path = cand_paths[best_idx] if costs.size else None
        best_final_yaw = float(cand_final_yaws[best_idx]) if costs.size else float(yaw)
        valid_count = int(np.sum(~collides))
        min_predicted_clearance = float(np.min(finite_clearance)) if finite_clearance.size else float("inf")
        candidate_count = int(costs.size)

        blocked = valid_count == 0
        if blocked:
            yaw_err = wrap_to_pi(goal_bearing - float(yaw))
            best_speed = 0.0
            best_yaw_rate = float(np.clip(yaw_err / max(cfg.horizon_s, 1e-3), -self._yaw_rate_candidates()[-1], self._yaw_rate_candidates()[-1]))
            if cfg.fallback_zero_if_blocked:
                action_xy = np.zeros(2, dtype=np.float64)
            else:
                action_xy = 0.05 * cfg.vmax * np.array([np.cos(float(yaw)), np.sin(float(yaw))], dtype=np.float64)
            best_cost = cfg.collision_large_penalty
        else:
            action_xy = best_speed * np.array([np.cos(float(yaw)), np.sin(float(yaw))], dtype=np.float64)

        if best_path is not None and not blocked:
            min_clear = path_min_clearance(
                best_path,
                local_obstacles,
                uav_radius=cfg.uav_radius,
                safety_margin=cfg.safety_margin,
            )
        else:
            min_clear = min_predicted_clearance

        diag = {
            "local_obstacle_count": int(len(local_obstacles)),
            "candidate_count": int(candidate_count),
            "valid_candidate_count": int(valid_count),
            "best_candidate_cost": float(best_cost),
            "best_candidate_speed": float(best_speed),
            "best_candidate_yaw_rate": float(best_yaw_rate),
            "best_candidate_final_yaw": float(best_final_yaw),
            "min_predicted_clearance": float(min_clear),
            "local_planner_blocked": bool(blocked),
            "local_planner_time_ms": float((time.perf_counter() - t0) * 1000.0),
            "assigned_slot_distance": float(goal_dist),
            "selected_action_norm": float(np.linalg.norm(action_xy)),
            "local_obstacles": local_obstacles,
        }
        return action_xy.astype(np.float64), diag


class TurnRadiusSlotController:
    """Thin controller wrapper around ``TurnRadiusSlotPlanner``."""

    def __init__(self, cfg: dict[str, Any] | TurnRadiusSlotPlannerConfig | None = None) -> None:
        self.planner = TurnRadiusSlotPlanner(cfg)

    def compute_action(
        self,
        pos_xy: np.ndarray,
        yaw: float,
        vel_xy: np.ndarray,
        assigned_slot_xy: np.ndarray,
        obstacles: list[Any],
        prev_action: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        action_xy, diag = self.planner.compute_action(
            pos_xy,
            yaw,
            vel_xy,
            assigned_slot_xy,
            obstacles,
            prev_action=prev_action,
        )
        return action_xy, float(diag["best_candidate_yaw_rate"]), diag
