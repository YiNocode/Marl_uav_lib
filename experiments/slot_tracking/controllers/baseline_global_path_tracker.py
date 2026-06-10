"""Global path plus lookahead tracking baseline for obstacle slot tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.slot_tracking.controllers.baseline_pure_pursuit import ControllerObservation, clip_norm
from experiments.slot_tracking.controllers.grid_path_planner import (
    path_length,
    plan_grid_astar,
    select_lookahead_subgoal,
)
from experiments.slot_tracking.controllers.nominal_slot_tracker import NominalSlotTracker
from marl_uav.framework.geometry.obstacle_geometry import has_line_of_sight


class GlobalPathPurePursuitTracker:
    """Plan to the proxy slot and track a configurable lookahead waypoint."""

    name = "global_path_pure_pursuit"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        raw = dict(cfg or {})
        self.resolution = float(raw.get("grid_resolution", 0.4))
        self.lookahead_distance = float(raw.get("lookahead_distance", 1.4))
        self.tracking_margin = float(raw.get("tracking_margin", raw.get("planning_safety_margin_extra", 0.0)))
        self.planning_safety_margin_extra = self.tracking_margin
        self.replan_goal_delta = float(raw.get("replan_goal_delta", 0.6))
        self.replan_path_deviation = float(raw.get("replan_path_deviation", 1.2))
        self.max_expansions = int(raw.get("max_expansions", 20000))
        self.clearance_weight = float(raw.get("clearance_weight", 0.0))
        self.planner_enter_inflation = float(raw.get("planner_enter_inflation", self.tracking_margin))
        self.planner_exit_inflation = float(raw.get("planner_exit_inflation", max(self.tracking_margin, 0.18)))
        self.clear_los_hold_steps = int(raw.get("clear_los_hold_steps", 8))
        self.nominal_rollout_horizon = int(raw.get("nominal_rollout_horizon", 16))
        self.nominal_rollout_clearance = float(raw.get("nominal_rollout_clearance", 0.05))
        self.min_replan_interval_steps = int(raw.get("min_replan_interval_steps", 8))
        self.proxy_replan_threshold = float(raw.get("proxy_replan_threshold", self.replan_goal_delta))
        self.proxy_slot_smoothing_alpha = float(raw.get("proxy_slot_smoothing_alpha", 1.0))
        self.target_lead_time_s = float(raw.get("target_lead_time_s", 0.0))
        self.path_deviation_threshold = float(raw.get("path_deviation_threshold", self.replan_path_deviation))
        self.path_commit_horizon = int(raw.get("path_commit_horizon", self.min_replan_interval_steps))
        self.stuck_window = int(raw.get("stuck_window", 12))
        self.subgoal_visibility_required = bool(raw.get("subgoal_visibility_required", True))
        self.max_subgoal_shift_per_step = float(raw.get("max_subgoal_shift_per_step", 0.45))
        self.subgoal_hold_steps = int(raw.get("subgoal_hold_steps", 5))
        self.allow_backward_progress = bool(raw.get("allow_backward_progress", False))
        self.mode = str(raw.get("tracker", "nominal")).strip().lower()
        self.kp = float(raw.get("kp", 1.4))
        self.nominal = NominalSlotTracker(raw.get("nominal_slot_tracker", raw))
        self.cached_path: np.ndarray | None = None
        self.cached_goal: np.ndarray | None = None
        self.replan_count = 0
        self.step_count = 0
        self.path_id = 0
        self.planner_active = False
        self.clear_hold_count = 0
        self.last_replan_step = -10**9
        self.current_progress_index = 0
        self.current_arclength_progress = 0.0
        self.prev_subgoal: np.ndarray | None = None
        self.smoothed_goal: np.ndarray | None = None
        self.subgoal_hold_remaining = 0
        self.progress_history: list[float] = []
        self.target_mode = "PROXY_TRACKING"

    def reset(self) -> None:
        self.cached_path = None
        self.cached_goal = None
        self.replan_count = 0
        self.step_count = 0
        self.path_id = 0
        self.planner_active = False
        self.clear_hold_count = 0
        self.last_replan_step = -10**9
        self.current_progress_index = 0
        self.current_arclength_progress = 0.0
        self.prev_subgoal = None
        self.smoothed_goal = None
        self.subgoal_hold_remaining = 0
        self.progress_history = []
        self.target_mode = "PROXY_TRACKING"
        self.nominal.reset()

    def compute_action(self, obs: ControllerObservation) -> tuple[np.ndarray, dict[str, Any]]:
        self.step_count += 1
        pos = np.asarray(obs.position[:2], dtype=np.float64).reshape(2)
        goal_vel = np.asarray(obs.slot_velocity[:2], dtype=np.float64).reshape(2)
        goal = np.asarray(obs.slot_position[:2], dtype=np.float64).reshape(2) + max(self.target_lead_time_s, 0.0) * goal_vel
        goal = _project_tracking_goal(
            goal,
            obstacles=list(obs.obstacles),
            world_xy=obs.world_xy,
            uav_radius=obs.uav_radius,
            safety_margin=obs.safety_margin,
        )
        if self.smoothed_goal is None:
            self.smoothed_goal = goal.copy()
        else:
            alpha = float(np.clip(self.proxy_slot_smoothing_alpha, 0.0, 1.0))
            self.smoothed_goal = alpha * goal + (1.0 - alpha) * self.smoothed_goal
        planning_goal = self.smoothed_goal.copy()
        plan_margin = float(obs.safety_margin) + max(float(self.planning_safety_margin_extra), 0.0)
        planning_inflation_radius = float(obs.uav_radius) + plan_margin
        obstacles = list(obs.obstacles)
        enter_margin = float(obs.safety_margin) + max(float(self.planner_enter_inflation), 0.0)
        exit_margin = float(obs.safety_margin) + max(float(self.planner_exit_inflation), 0.0)
        los_enter_clear = _segment_visible(pos, planning_goal, obstacles, safety_margin=enter_margin, uav_radius=obs.uav_radius) if obstacles else True
        los_exit_clear = _segment_visible(pos, planning_goal, obstacles, safety_margin=exit_margin, uav_radius=obs.uav_radius) if obstacles else True
        nominal_safe, nominal_diag = _nominal_rollout_safe(
            pos,
            np.asarray(obs.velocity[:2], dtype=np.float64).reshape(2),
            planning_goal,
            goal_vel,
            obstacles,
            world_xy=obs.world_xy,
            dt=obs.dt,
            vmax=obs.uav_vmax,
            amax=obs.uav_amax,
            uav_radius=obs.uav_radius,
            safety_margin=max(float(self.nominal_rollout_clearance), 0.0),
            horizon=self.nominal_rollout_horizon,
            kp=float(getattr(self.nominal, "kp", 1.6)),
            kd=float(getattr(self.nominal, "kd", 0.45)),
        )
        planner_called = False
        planner_success = True
        planner_reason = "NOMINAL"
        replan_reason = "NONE"
        replan_needed = False
        safety_bypass = False
        path_valid = self.cached_path is not None and _path_collision_free(self.cached_path, obstacles, safety_margin=plan_margin, uav_radius=obs.uav_radius)
        path_deviation = _distance_to_polyline(pos, self.cached_path) if self.cached_path is not None else float("inf")

        if self.planner_active:
            if nominal_safe:
                self.clear_hold_count += 1
            else:
                self.clear_hold_count = 0
            if self.clear_hold_count >= self.clear_los_hold_steps:
                self.planner_active = False
                self.clear_hold_count = 0
        elif not nominal_safe:
            self.planner_active = True
            self.clear_hold_count = 0

        if not self.planner_active:
            path = np.stack([pos, planning_goal], axis=0)
            self.current_progress_index = 0
            self.current_arclength_progress = 0.0
        else:
            replan_needed, replan_reason = self._replan_decision(
                pos,
                planning_goal,
                obstacles,
                plan_margin=plan_margin,
                uav_radius=obs.uav_radius,
                path_valid=path_valid,
                path_deviation=path_deviation,
            )
            cooldown_ready = (self.step_count - self.last_replan_step) >= self.min_replan_interval_steps
            safety_bypass = replan_reason in ("NO_PATH", "PATH_BLOCKED", "SUBGOAL_NOT_VISIBLE")
            if replan_needed and (cooldown_ready or safety_bypass):
                path, pdiag = plan_grid_astar(
                    pos,
                    planning_goal,
                    obstacles,
                    world_xy=obs.world_xy,
                    uav_radius=obs.uav_radius,
                    safety_margin=plan_margin,
                    resolution=self.resolution,
                    max_expansions=self.max_expansions,
                    clearance_weight=self.clearance_weight,
                )
                self.cached_path = path
                self.cached_goal = planning_goal.copy()
                self.replan_count += 1
                self.path_id += 1
                self.last_replan_step = self.step_count
                self.current_progress_index = 0
                self.current_arclength_progress = 0.0
                path_valid = _path_collision_free(self.cached_path, obstacles, safety_margin=plan_margin, uav_radius=obs.uav_radius)
                planner_success = bool(pdiag.get("planner_success", False))
                planner_reason = str(pdiag.get("planner_reason", replan_reason))
                planner_called = True
            elif replan_needed:
                replan_reason = "COOLDOWN_BYPASS_FOR_SAFETY" if safety_bypass else "COOLDOWN"
                planner_success = bool(path_valid)
                planner_reason = replan_reason
            path = self.cached_path if self.cached_path is not None else np.stack([pos, planning_goal], axis=0)
        steps_since_last_replan = int(self.step_count - self.last_replan_step) if self.last_replan_step > -10**8 else 10**9
        replan_allowed_by_cooldown = bool(steps_since_last_replan >= self.min_replan_interval_steps)
        replan_blocked_by_cooldown = bool(self.planner_active and replan_needed and not replan_allowed_by_cooldown and not safety_bypass)

        lookahead = self._lookahead(obs)
        self.current_progress_index, self.current_arclength_progress = _path_progress_state(
            path,
            pos,
            previous_index=self.current_progress_index,
            previous_arclength=self.current_arclength_progress,
            allow_backward=self.allow_backward_progress or planner_called,
        )
        if not self.planner_active:
            raw_subgoal = planning_goal.copy()
        else:
            raw_subgoal = _select_lookahead_by_arclength(
                path,
                self.current_arclength_progress + lookahead,
                fallback=select_lookahead_subgoal(path, pos, lookahead_distance=lookahead),
            )
        if not self.planner_active:
            subgoal = planning_goal.copy()
            visibility_diag = {
                "selected_subgoal_visible": bool(los_enter_clear),
                "subgoal_backtracked": False,
                "backtracked_waypoint_index": -1,
                "raw_lookahead_subgoal": goal.astype(float).tolist(),
            }
        else:
            subgoal, visibility_diag = _visible_subgoal(
                path,
                pos,
                raw_subgoal,
                list(obs.obstacles),
                uav_radius=obs.uav_radius,
                safety_margin=plan_margin,
                lookahead_distance=lookahead,
                desired_goal=planning_goal,
            )
        target_mode_previous = self.target_mode
        selected_visible = bool(visibility_diag.get("selected_subgoal_visible", True))
        no_visible_subgoal = bool(visibility_diag.get("no_visible_subgoal", False))
        if self.planner_active:
            self.target_mode = "PATH_SUBGOAL_TRACKING" if selected_visible and path_valid and not no_visible_subgoal else "RECOVERY"
        else:
            self.target_mode = "PROXY_TRACKING"
        if not np.all(np.isfinite(subgoal)) or no_visible_subgoal or (self.planner_active and not path_valid):
            self.target_mode = "SAFE_STOP" if no_visible_subgoal or (self.planner_active and not path_valid) else self.target_mode
        target_mode_transition_reason = "UNCHANGED" if self.target_mode == target_mode_previous else f"{target_mode_previous}_TO_{self.target_mode}"
        raw_selected_subgoal = subgoal.copy()
        subgoal_shift_norm = float(np.linalg.norm(subgoal - self.prev_subgoal)) if self.prev_subgoal is not None else 0.0
        subgoal_held = False
        if (
            self.planner_active
            and
            self.prev_subgoal is not None
            and subgoal_shift_norm > self.max_subgoal_shift_per_step
            and _segment_visible(pos, self.prev_subgoal, obstacles, safety_margin=plan_margin, uav_radius=obs.uav_radius)
            and self.subgoal_hold_remaining <= 0
            and path_valid
        ):
            self.subgoal_hold_remaining = self.subgoal_hold_steps
        if self.planner_active and path_valid and self.prev_subgoal is not None and self.subgoal_hold_remaining > 0:
            subgoal = self.prev_subgoal.copy()
            self.subgoal_hold_remaining -= 1
            subgoal_held = True
            subgoal_shift_norm = 0.0
        else:
            if not self.planner_active:
                self.subgoal_hold_remaining = 0
            self.prev_subgoal = subgoal.copy()
        stale_subgoal_used = bool(subgoal_held and not path_valid)
        subgoal_vel = np.zeros(2, dtype=np.float64) if np.linalg.norm(subgoal - planning_goal) > 0.35 else goal_vel

        if self.mode in ("nominal", "nominal_slot_tracker"):
            sub_obs = ControllerObservation(
                position=obs.position,
                velocity=obs.velocity,
                slot_position=np.array([subgoal[0], subgoal[1], obs.slot_position[2]], dtype=np.float64),
                slot_velocity=np.array([subgoal_vel[0], subgoal_vel[1], 0.0], dtype=np.float64),
                obstacles=obs.obstacles,
                world_xy=obs.world_xy,
                dt=obs.dt,
                uav_vmax=obs.uav_vmax,
                uav_amax=obs.uav_amax,
                uav_radius=obs.uav_radius,
                safety_margin=obs.safety_margin,
                peer_positions=obs.peer_positions,
                controller_clip_enabled=obs.controller_clip_enabled,
            )
            action, diag = self.nominal.compute_action(sub_obs)
        else:
            delta = subgoal - pos
            action = clip_norm(self.kp * delta, obs.uav_vmax) if obs.controller_clip_enabled else self.kp * delta
            diag = {
                "v_goal": (self.kp * delta).astype(float).tolist(),
                "v_final_before_clip": (self.kp * delta).astype(float).tolist(),
                "v_final_after_clip": action.astype(float).tolist(),
                "raw_cmd_norm": float(np.linalg.norm(self.kp * delta)),
                "clip_flag": bool(np.linalg.norm(self.kp * delta) > np.linalg.norm(action) + 1e-9),
            }
        progress_to_subgoal = _progress_fields(pos, subgoal, action, prefix="subgoal").get("progress_to_subgoal", 0.0)
        self.progress_history.append(float(progress_to_subgoal))
        if len(self.progress_history) > max(self.stuck_window * 3, 32):
            self.progress_history = self.progress_history[-max(self.stuck_window * 3, 32):]

        diag.update(
            _planner_diag(
                pos=pos,
                raw_goal=planning_goal,
                subgoal=subgoal,
                action=action,
                path=path,
                planner_called=planner_called,
                planner_success=planner_success,
                planner_reason=planner_reason,
                replan_count=self.replan_count,
                lookahead=lookahead,
                los_blocked=not los_enter_clear,
                visibility_diag=visibility_diag,
                obstacles=obstacles,
                safety_margin=plan_margin,
                collision_radius=obs.uav_radius,
                configured_safety_margin=obs.safety_margin,
                tracking_margin=self.tracking_margin,
                planning_inflation_radius=planning_inflation_radius,
                planner_mode_active=self.planner_active,
                replan_reason=replan_reason,
                path_valid=path_valid,
                path_deviation=path_deviation,
                current_path_id=self.path_id,
                current_path_progress_index=self.current_progress_index,
                current_path_arclength_progress=self.current_arclength_progress,
                nominal_rollout_safe=nominal_safe,
                nominal_rollout_diag=nominal_diag,
                los_enter_clear=los_enter_clear,
                los_exit_clear=los_exit_clear,
                subgoal_shift_norm=subgoal_shift_norm,
                raw_selected_subgoal=raw_selected_subgoal,
                subgoal_held=subgoal_held,
                stale_subgoal_used=stale_subgoal_used,
                target_mode=self.target_mode,
                target_mode_previous=target_mode_previous,
                target_mode_transition_reason=target_mode_transition_reason,
                steps_since_last_replan=steps_since_last_replan,
                replan_allowed_by_cooldown=replan_allowed_by_cooldown,
                replan_blocked_by_cooldown=replan_blocked_by_cooldown,
                proxy_slot_smoothing_alpha=self.proxy_slot_smoothing_alpha,
                path_commit_horizon=self.path_commit_horizon,
                target_lead_time_s=self.target_lead_time_s,
            )
        )
        return action, diag

    def _lookahead(self, obs: ControllerObservation) -> float:
        speed = float(np.linalg.norm(np.asarray(obs.velocity[:2], dtype=np.float64)))
        return float(self.lookahead_distance + 0.4 * speed)

    def _needs_replan(self, pos: np.ndarray, goal: np.ndarray, obs: ControllerObservation) -> bool:
        if self.cached_path is None or self.cached_goal is None:
            return True
        if float(np.linalg.norm(goal - self.cached_goal)) > self.replan_goal_delta:
            return True
        if _distance_to_polyline(pos, self.cached_path) > self.replan_path_deviation:
            return True
        pts = np.asarray(self.cached_path, dtype=np.float64).reshape(-1, 2)
        for i in range(pts.shape[0] - 1):
            plan_margin = float(obs.safety_margin) + max(float(self.planning_safety_margin_extra), 0.0)
            if not has_line_of_sight(pts[i], pts[i + 1], list(obs.obstacles), safety_margin=plan_margin, uav_radius=obs.uav_radius):
                return True
        return False

    def _replan_decision(
        self,
        pos: np.ndarray,
        goal: np.ndarray,
        obstacles: list[Any],
        *,
        plan_margin: float,
        uav_radius: float,
        path_valid: bool,
        path_deviation: float,
    ) -> tuple[bool, str]:
        if self.cached_path is None or self.cached_goal is None:
            return True, "NO_PATH"
        if not path_valid:
            return True, "PATH_BLOCKED"
        if (self.step_count - self.last_replan_step) < self.path_commit_horizon and path_valid:
            return False, "PATH_COMMITTED"
        if float(np.linalg.norm(goal - self.cached_goal)) > self.proxy_replan_threshold:
            return True, "PROXY_MOVED_BEYOND_THRESHOLD"
        if path_deviation > self.path_deviation_threshold:
            return True, "PATH_DEVIATION"
        if self.subgoal_visibility_required and self.prev_subgoal is not None:
            if not _segment_visible(pos, self.prev_subgoal, obstacles, safety_margin=plan_margin, uav_radius=uav_radius):
                return True, "SUBGOAL_NOT_VISIBLE"
        if len(self.progress_history) >= max(self.stuck_window, 1):
            recent = self.progress_history[-self.stuck_window:]
            if recent and float(np.mean(recent)) < 0.03:
                return True, "STUCK"
        return False, "NONE"


def _planner_diag(
    *,
    pos: np.ndarray,
    raw_goal: np.ndarray,
    subgoal: np.ndarray,
    action: np.ndarray,
    path: np.ndarray,
    planner_called: bool,
    planner_success: bool,
    planner_reason: str,
    replan_count: int,
    lookahead: float,
    los_blocked: bool,
    visibility_diag: dict[str, Any],
    obstacles: list[Any],
    safety_margin: float,
    collision_radius: float,
    configured_safety_margin: float,
    tracking_margin: float,
    planning_inflation_radius: float,
    planner_mode_active: bool,
    replan_reason: str,
    path_valid: bool,
    path_deviation: float,
    current_path_id: int,
    current_path_progress_index: int,
    current_path_arclength_progress: float,
    nominal_rollout_safe: bool,
    nominal_rollout_diag: dict[str, Any],
    los_enter_clear: bool,
    los_exit_clear: bool,
    subgoal_shift_norm: float,
    raw_selected_subgoal: np.ndarray,
    subgoal_held: bool,
    stale_subgoal_used: bool,
    target_mode: str,
    target_mode_previous: str,
    target_mode_transition_reason: str,
    steps_since_last_replan: int,
    replan_allowed_by_cooldown: bool,
    replan_blocked_by_cooldown: bool,
    proxy_slot_smoothing_alpha: float,
    path_commit_horizon: int,
    target_lead_time_s: float,
) -> dict[str, Any]:
    progress_sub = _progress_fields(pos, subgoal, action, prefix="subgoal")
    progress_raw = _progress_fields(pos, raw_goal, action, prefix="raw_slot")
    current_segment = _current_path_segment(pos, path)
    min_path = _min_polyline_clearance(path, obstacles, uav_radius=collision_radius)
    mean_path = _mean_polyline_clearance(path, obstacles, uav_radius=collision_radius)
    min_current = _min_polyline_clearance(current_segment, obstacles, uav_radius=collision_radius)
    min_subgoal = _min_polyline_clearance(np.stack([pos, subgoal], axis=0), obstacles, uav_radius=collision_radius)
    los_proxy = _segment_visible(pos, raw_goal, obstacles, safety_margin=safety_margin, uav_radius=collision_radius) if obstacles else True
    los_subgoal = _segment_visible(pos, subgoal, obstacles, safety_margin=safety_margin, uav_radius=collision_radius) if obstacles else True
    return {
        "planner_called": bool(planner_called),
        "planner_success": bool(planner_success),
        "planner_reason": str(planner_reason),
        "replan_reason": str(replan_reason),
        "planner_path_length": path_length(path),
        "planner_replan_count": int(replan_count),
        "planner_mode_active": bool(planner_mode_active),
        "target_mode": str(target_mode),
        "target_mode_previous": str(target_mode_previous),
        "target_mode_transition_reason": str(target_mode_transition_reason),
        "path_valid": bool(path_valid),
        "path_blocked": bool(not _path_collision_free(path, obstacles, safety_margin=safety_margin, uav_radius=collision_radius)),
        "path_deviation": float(path_deviation),
        "steps_since_last_replan": int(steps_since_last_replan),
        "replan_allowed_by_cooldown": bool(replan_allowed_by_cooldown),
        "replan_blocked_by_cooldown": bool(replan_blocked_by_cooldown),
        "proxy_slot_smoothing_alpha": float(proxy_slot_smoothing_alpha),
        "path_commit_horizon": int(path_commit_horizon),
        "target_lead_time_s": float(target_lead_time_s),
        "current_path_id": int(current_path_id),
        "current_path_waypoints": np.asarray(path, dtype=np.float64).reshape(-1, 2).astype(float).tolist(),
        "current_path_progress_index": int(current_path_progress_index),
        "current_path_arclength_progress": float(current_path_arclength_progress),
        "lookahead_distance": float(lookahead),
        "current_subgoal": subgoal.astype(float).tolist(),
        "selected_subgoal_path": subgoal.astype(float).reshape(1, 2).tolist(),
        "planned_path": np.asarray(path, dtype=np.float64).reshape(-1, 2).astype(float).tolist(),
        "current_path_segment": np.asarray(current_segment, dtype=np.float64).reshape(-1, 2).astype(float).tolist(),
        "min_distance_to_planned_path": float(min_path),
        "planned_path_min_clearance": float(min_path),
        "planned_path_mean_clearance": float(mean_path),
        "min_distance_to_current_segment": float(min_current),
        "min_distance_to_subgoal_segment": float(min_subgoal),
        "lookahead_segment_clearance": float(min_subgoal),
        "line_of_sight_to_proxy": bool(los_proxy),
        "line_of_sight_to_proxy_slot": bool(los_proxy),
        "line_of_sight_enter_clear": bool(los_enter_clear),
        "line_of_sight_exit_clear": bool(los_exit_clear),
        "line_of_sight_to_subgoal": bool(los_subgoal),
        "nominal_rollout_safe": bool(nominal_rollout_safe),
        "nominal_rollout_min_clearance": float(nominal_rollout_diag.get("nominal_rollout_min_clearance", np.nan)),
        "nominal_rollout_boundary_min": float(nominal_rollout_diag.get("nominal_rollout_boundary_min", np.nan)),
        "subgoal_visible": bool(visibility_diag.get("selected_subgoal_visible", los_subgoal)),
        "selected_subgoal_visible": bool(visibility_diag.get("selected_subgoal_visible", los_subgoal)),
        "no_visible_subgoal": bool(visibility_diag.get("no_visible_subgoal", False)),
        "subgoal_backtracked": bool(visibility_diag.get("subgoal_backtracked", False)),
        "backtracked_waypoint_index": int(visibility_diag.get("backtracked_waypoint_index", -1)),
        "raw_selected_subgoal": np.asarray(raw_selected_subgoal, dtype=np.float64).reshape(2).astype(float).tolist(),
        "raw_lookahead_subgoal": np.asarray(visibility_diag.get("raw_lookahead_subgoal", subgoal), dtype=np.float64).reshape(2).astype(float).tolist(),
        "subgoal_shift_norm": float(subgoal_shift_norm),
        "subgoal_held": bool(subgoal_held),
        "stale_subgoal_used": bool(stale_subgoal_used),
        "collision_radius": float(collision_radius),
        "safety_margin": float(configured_safety_margin),
        "tracking_margin": float(tracking_margin),
        "planning_inflation_radius": float(planning_inflation_radius),
        "v_planner_subgoal": np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist(),
        "v_after_planner": np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist(),
        "line_of_sight_blocked": bool(los_blocked),
        **progress_sub,
        **progress_raw,
    }


def _progress_fields(pos: np.ndarray, goal: np.ndarray, action: np.ndarray, *, prefix: str) -> dict[str, float]:
    delta = np.asarray(goal, dtype=np.float64).reshape(2) - np.asarray(pos, dtype=np.float64).reshape(2)
    dist = float(np.linalg.norm(delta))
    act = np.asarray(action, dtype=np.float64).reshape(2)
    if dist <= 1e-9:
        return {f"cos_to_{prefix}": float("nan"), f"progress_to_{prefix}": 0.0}
    progress = float(np.dot(act, delta / dist))
    return {
        f"cos_to_{prefix}": float(progress / max(float(np.linalg.norm(act)), 1e-9)),
        f"progress_to_{prefix}": progress,
    }


def _distance_to_polyline(pos: np.ndarray, path: np.ndarray) -> float:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return float("inf")
    if pts.shape[0] == 1:
        return float(np.linalg.norm(np.asarray(pos, dtype=np.float64).reshape(2) - pts[0]))
    p = np.asarray(pos, dtype=np.float64).reshape(2)
    best = float("inf")
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _nominal_rollout_safe(
    pos_xy: np.ndarray,
    vel_xy: np.ndarray,
    goal_xy: np.ndarray,
    goal_vel_xy: np.ndarray,
    obstacles: list[Any],
    *,
    world_xy: float,
    dt: float,
    vmax: float,
    amax: float,
    uav_radius: float,
    safety_margin: float,
    horizon: int,
    kp: float,
    kd: float,
) -> tuple[bool, dict[str, float]]:
    p = np.asarray(pos_xy, dtype=np.float64).reshape(2).copy()
    v = np.asarray(vel_xy, dtype=np.float64).reshape(2).copy()
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    goal_vel = np.asarray(goal_vel_xy, dtype=np.float64).reshape(2)
    dt_eff = max(float(dt), 1e-9)
    min_clear = float("inf")
    min_boundary = float("inf")
    for _ in range(max(int(horizon), 1)):
        cmd = goal_vel + float(kp) * (goal - p) - float(kd) * v
        desired = clip_norm(cmd, vmax)
        dv = clip_norm(desired - v, float(amax) * dt_eff)
        v = clip_norm(v + dv, vmax)
        nxt = p + v * dt_eff
        if obstacles:
            clear = _segment_min_clearance_xy(p, nxt, obstacles, uav_radius=uav_radius)
            min_clear = min(min_clear, clear)
            if clear < float(safety_margin):
                return False, {
                    "nominal_rollout_min_clearance": float(min_clear),
                    "nominal_rollout_boundary_min": float(min_boundary),
                }
        b = float(world_xy) - float(uav_radius) - np.max(np.abs(nxt))
        min_boundary = min(min_boundary, b)
        if b < float(safety_margin):
            return False, {
                "nominal_rollout_min_clearance": float(min_clear),
                "nominal_rollout_boundary_min": float(min_boundary),
            }
        p = nxt
    return True, {
        "nominal_rollout_min_clearance": float(min_clear),
        "nominal_rollout_boundary_min": float(min_boundary),
    }


def _segment_min_clearance_xy(
    p0: np.ndarray,
    p1: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    samples: int = 8,
) -> float:
    if not obstacles:
        return float("inf")
    a = np.asarray(p0, dtype=np.float64).reshape(2)
    b = np.asarray(p1, dtype=np.float64).reshape(2)
    best = float("inf")
    for t in np.linspace(0.0, 1.0, max(int(samples), 2)):
        p = a + float(t) * (b - a)
        for obs in obstacles:
            c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
            best = min(best, float(np.linalg.norm(p - c) - float(getattr(obs, "radius", 0.0)) - float(uav_radius)))
    return float(best)


def _path_progress_state(
    path: np.ndarray,
    pos: np.ndarray,
    *,
    previous_index: int,
    previous_arclength: float,
    allow_backward: bool,
) -> tuple[int, float]:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0, 0.0
    p = np.asarray(pos, dtype=np.float64).reshape(2)
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    prefix = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    best_i = 0
    best_s = 0.0
    best_d = float("inf")
    start_i = 0 if allow_backward else min(max(int(previous_index), 0), pts.shape[0] - 2)
    for i in range(start_i, pts.shape[0] - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(p - proj))
        s = float(prefix[i] + t * seg_lengths[i])
        if d < best_d:
            best_i = i
            best_s = s
            best_d = d
    if not allow_backward and best_s < float(previous_arclength):
        best_s = float(previous_arclength)
        best_i = min(max(int(previous_index), 0), pts.shape[0] - 2)
    return int(best_i), float(best_s)


def _select_lookahead_by_arclength(path: np.ndarray, target_s: float, *, fallback: np.ndarray) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return np.asarray(fallback, dtype=np.float64).reshape(2).copy()
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    prefix = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(prefix[-1])
    s = min(max(float(target_s), 0.0), total)
    for i, seg_len in enumerate(seg_lengths):
        if s <= prefix[i + 1] + 1e-9:
            t = 0.0 if float(seg_len) <= 1e-12 else float((s - prefix[i]) / seg_len)
            return pts[i] + t * (pts[i + 1] - pts[i])
    return pts[-1].copy()


def _visible_subgoal(
    path: np.ndarray,
    pos: np.ndarray,
    subgoal: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
    lookahead_distance: float,
    desired_goal: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(pos, dtype=np.float64).reshape(2)
    sg = np.asarray(subgoal, dtype=np.float64).reshape(2)
    desired_vec = (
        np.asarray(desired_goal, dtype=np.float64).reshape(2) - p
        if desired_goal is not None
        else sg - p
    )
    if not obstacles or _segment_visible(p, sg, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
        return sg, {
            "selected_subgoal_visible": True,
            "subgoal_backtracked": False,
            "backtracked_waypoint_index": -1,
            "no_visible_subgoal": False,
            "raw_lookahead_subgoal": sg.astype(float).tolist(),
        }
    if pts.shape[0] == 0:
        return p.copy(), {
            "selected_subgoal_visible": False,
            "subgoal_backtracked": True,
            "backtracked_waypoint_index": -1,
            "no_visible_subgoal": True,
            "raw_lookahead_subgoal": sg.astype(float).tolist(),
        }

    closest_idx = int(np.argmin(np.linalg.norm(pts - p.reshape(1, 2), axis=1)))
    selected_idx = int(np.argmin(np.linalg.norm(pts - sg.reshape(1, 2), axis=1)))
    lo = min(closest_idx, selected_idx)
    hi = max(closest_idx, selected_idx)
    for idx in range(hi, lo - 1, -1):
        cand = pts[idx]
        progress_ok = float(np.dot(cand - p, desired_vec)) >= -1e-6
        if progress_ok and _segment_visible(p, cand, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
            return cand.copy(), {
                "selected_subgoal_visible": True,
                "subgoal_backtracked": True,
                "backtracked_waypoint_index": int(idx),
                "no_visible_subgoal": False,
                "raw_lookahead_subgoal": sg.astype(float).tolist(),
            }

    tangent = _path_tangent_at(pos, pts)
    for scale in (0.5, 0.25, 0.1):
        cand = p + tangent * max(float(lookahead_distance) * scale, 0.05)
        if _segment_visible(p, cand, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
            return cand, {
                "selected_subgoal_visible": True,
                "subgoal_backtracked": True,
                "backtracked_waypoint_index": -1,
                "no_visible_subgoal": False,
                "raw_lookahead_subgoal": sg.astype(float).tolist(),
            }
    escape = _obstacle_tangent_escape(
        p,
        desired_goal if desired_goal is not None else sg,
        obstacles,
        uav_radius=uav_radius,
        safety_margin=safety_margin,
        step=max(float(lookahead_distance) * 0.5, 0.15),
    )
    if escape is not None and _segment_visible(p, escape, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
        return escape, {
            "selected_subgoal_visible": True,
            "subgoal_backtracked": True,
            "backtracked_waypoint_index": -1,
            "no_visible_subgoal": False,
            "raw_lookahead_subgoal": sg.astype(float).tolist(),
        }
    return p.copy(), {
        "selected_subgoal_visible": False,
        "subgoal_backtracked": True,
        "backtracked_waypoint_index": -1,
        "no_visible_subgoal": True,
        "raw_lookahead_subgoal": sg.astype(float).tolist(),
    }


def _path_tangent_at(pos: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64).reshape(2)
    best_i = 0
    best_d = float("inf")
    for i in range(pts.shape[0] - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        d = float(np.linalg.norm(p - (a + t * ab)))
        if d < best_d:
            best_d = d
            best_i = i
    tangent = pts[best_i + 1] - pts[best_i]
    n = float(np.linalg.norm(tangent))
    if n <= 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return tangent / n


def _current_path_segment(pos: np.ndarray, path: np.ndarray) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        p = np.asarray(pos, dtype=np.float64).reshape(2)
        return np.stack([p, p], axis=0)
    p = np.asarray(pos, dtype=np.float64).reshape(2)
    best_i = 0
    best_d = float("inf")
    for i in range(pts.shape[0] - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        d = float(np.linalg.norm(p - (a + t * ab)))
        if d < best_d:
            best_d = d
            best_i = i
    return pts[best_i:best_i + 2].copy()


def _min_polyline_clearance(path: np.ndarray, obstacles: list[Any], *, uav_radius: float) -> float:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or not obstacles:
        return float("inf")
    if pts.shape[0] == 1:
        samples = pts
    else:
        samples = []
        for a, b in zip(pts[:-1], pts[1:]):
            seg_len = float(np.linalg.norm(b - a))
            n = max(int(np.ceil(seg_len / 0.05)), 1)
            for t in np.linspace(0.0, 1.0, n + 1):
                samples.append(a + t * (b - a))
        samples = np.asarray(samples, dtype=np.float64)
    best = float("inf")
    for p in np.asarray(samples, dtype=np.float64).reshape(-1, 2):
        for obs in obstacles:
            c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
            best = min(best, float(np.linalg.norm(p - c) - float(getattr(obs, "radius", 0.0)) - float(uav_radius)))
    return float(best)


def _mean_polyline_clearance(path: np.ndarray, obstacles: list[Any], *, uav_radius: float) -> float:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or not obstacles:
        return float("inf")
    samples = []
    if pts.shape[0] == 1:
        samples = [pts[0]]
    else:
        for a, b in zip(pts[:-1], pts[1:]):
            seg_len = float(np.linalg.norm(b - a))
            n = max(int(np.ceil(seg_len / 0.1)), 1)
            for t in np.linspace(0.0, 1.0, n + 1):
                samples.append(a + t * (b - a))
    vals = []
    for p in np.asarray(samples, dtype=np.float64).reshape(-1, 2):
        best = float("inf")
        for obs in obstacles:
            c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
            best = min(best, float(np.linalg.norm(p - c) - float(getattr(obs, "radius", 0.0)) - float(uav_radius)))
        vals.append(best)
    return float(np.mean(vals)) if vals else float("inf")


def _path_collision_free(path: np.ndarray, obstacles: list[Any], *, safety_margin: float, uav_radius: float) -> bool:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2 or not obstacles:
        return True
    for a, b in zip(pts[:-1], pts[1:]):
        if not has_line_of_sight(a, b, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
            return False
    return True


def _project_tracking_goal(
    goal: np.ndarray,
    obstacles: list[Any],
    *,
    world_xy: float,
    uav_radius: float,
    safety_margin: float,
) -> np.ndarray:
    out = np.asarray(goal, dtype=np.float64).reshape(2).copy()
    limit = max(float(world_xy) - float(uav_radius) - max(float(safety_margin), 0.0), 0.0)
    out = np.clip(out, -limit, limit)
    for _ in range(3):
        changed = False
        for obs in obstacles:
            c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
            radius = float(getattr(obs, "radius", 0.0))
            rel = out - c
            dist = float(np.linalg.norm(rel))
            required = radius + float(uav_radius) + max(float(safety_margin), 0.0)
            if dist < required:
                normal = np.array([1.0, 0.0], dtype=np.float64) if dist <= 1e-9 else rel / dist
                out = c + required * normal
                out = np.clip(out, -limit, limit)
                changed = True
        if not changed:
            break
    return out


def _obstacle_tangent_escape(
    pos: np.ndarray,
    desired_goal: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
    step: float,
) -> np.ndarray | None:
    if not obstacles:
        return None
    p = np.asarray(pos, dtype=np.float64).reshape(2)
    goal = np.asarray(desired_goal, dtype=np.float64).reshape(2)
    desired = goal - p
    best: tuple[float, np.ndarray] | None = None
    for obs in obstacles:
        c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
        rel = p - c
        dist = float(np.linalg.norm(rel))
        if dist <= 1e-9:
            continue
        clearance = dist - float(getattr(obs, "radius", 0.0)) - float(uav_radius)
        if clearance > float(safety_margin) + 0.2:
            continue
        normal = rel / dist
        deficit = max(float(safety_margin) - clearance + 0.02, 0.0)
        tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
        for sign in (1.0, -1.0):
            direction = sign * tangent
            cand = p + direction * max(float(step), 0.05) + normal * deficit
            progress = float(np.dot(direction, desired))
            clear_score = _segment_clearance_future(cand, obstacles, uav_radius=uav_radius + safety_margin)
            score = progress + 0.25 * clear_score
            if best is None or score > best[0]:
                best = (score, cand)
    return None if best is None else best[1]


def _segment_clearance_future(point: np.ndarray, obstacles: list[Any], *, uav_radius: float) -> float:
    p = np.asarray(point, dtype=np.float64).reshape(2)
    best = float("inf")
    for obs in obstacles:
        c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
        best = min(best, float(np.linalg.norm(p - c) - float(getattr(obs, "radius", 0.0)) - float(uav_radius)))
    return float(best)


def _segment_visible(
    p0: np.ndarray,
    p1: np.ndarray,
    obstacles: list[Any],
    *,
    safety_margin: float,
    uav_radius: float,
    samples: int = 16,
) -> bool:
    if not obstacles:
        return True
    a = np.asarray(p0, dtype=np.float64).reshape(2)
    b = np.asarray(p1, dtype=np.float64).reshape(2)
    pad = max(float(uav_radius), 0.0) + max(float(safety_margin), 0.0)
    start_clear = _segment_clearance_future(a, obstacles, uav_radius=pad)
    end_clear = _segment_clearance_future(b, obstacles, uav_radius=pad)
    allow_escape = bool(start_clear < 0.0 and end_clear > start_clear + 1e-6)
    min_inflated = float("inf")
    for t in np.linspace(0.0, 1.0, max(int(samples), 2))[1:]:
        p = a + float(t) * (b - a)
        for obs in obstacles:
            c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
            clear = float(np.linalg.norm(p - c) - float(getattr(obs, "radius", 0.0)) - pad)
            actual_clear = float(np.linalg.norm(p - c) - float(getattr(obs, "radius", 0.0)) - max(float(uav_radius), 0.0))
            min_inflated = min(min_inflated, clear)
            if actual_clear < -1e-9:
                return False
            if clear < -1e-9 and not allow_escape:
                return False
    if allow_escape and end_clear + 1e-9 < min(start_clear, min_inflated):
        return False
    return True
