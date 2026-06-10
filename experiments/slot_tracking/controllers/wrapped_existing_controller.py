"""Wrapper around the repository's existing low-level obstacle avoidance controller."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.slot_tracking.controllers.baseline_global_path_tracker import GlobalPathPurePursuitTracker
from experiments.slot_tracking.controllers.baseline_pure_pursuit import ControllerObservation, clip_norm
from experiments.slot_tracking.controllers.nominal_slot_tracker import NominalSlotTracker
from marl_uav.framework.geometry.obstacle_geometry import has_line_of_sight
from marl_uav.control.obstacle_avoidance_controller import ObstacleAvoidanceController


class ExistingControllerWrapper:
    """Expose ``ObstacleAvoidanceController`` through the benchmark controller API.

    Assumption: the existing controller's holonomic branch returns a world-frame
    XY velocity command toward ``goal_xy`` with local obstacle/boundary safety
    projection. The benchmark still applies the shared dynamics and collision
    checker after this command, so this wrapper does not grant privileged state.
    """

    name = "existing"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.raw_cfg = dict(cfg or {})
        self.controller = ObstacleAvoidanceController(self.raw_cfg)
        nominal_cfg = {
            **dict(self.raw_cfg.get("nominal_slot_tracker") or {}),
            "kp": self.raw_cfg.get("nominal_kp", (self.raw_cfg.get("nominal_slot_tracker") or {}).get("kp", 1.6)),
            "kd": self.raw_cfg.get("nominal_kd", (self.raw_cfg.get("nominal_slot_tracker") or {}).get("kd", 0.45)),
            "vmax": self.raw_cfg.get("vmax", (self.raw_cfg.get("nominal_slot_tracker") or {}).get("vmax")),
            "amax": self.raw_cfg.get("amax_xy", (self.raw_cfg.get("nominal_slot_tracker") or {}).get("amax")),
            "dt": self.raw_cfg.get("dt", (self.raw_cfg.get("nominal_slot_tracker") or {}).get("dt")),
        }
        self.nominal = NominalSlotTracker(nominal_cfg)
        self.free_space_fallback = bool(self.raw_cfg.get("free_space_fallback", False))
        self.obstacle_activation_distance = float(self.raw_cfg.get("obstacle_activation_distance", 2.0))
        self.boundary_activation_distance = float(self.raw_cfg.get("boundary_activation_distance", 2.0))
        self.boundary_hard_margin = float(self.raw_cfg.get("boundary_hard_margin", self.raw_cfg.get("safety_margin", 0.25)))
        self.boundary_braking_margin = float(
            self.raw_cfg.get("boundary_braking_margin", self.raw_cfg.get("boundary_hard_margin", self.raw_cfg.get("safety_margin", 0.25)))
        )
        self.boundary_braking_gain = float(self.raw_cfg.get("boundary_braking_gain", 1.0))
        self.boundary_alpha = float(self.raw_cfg.get("boundary_alpha", self.boundary_braking_gain))
        self.boundary_filter_mode = str(self.raw_cfg.get("boundary_filter_mode", "minimal_projection")).strip().lower()
        self.boundary_projection_enabled = bool(self.raw_cfg.get("boundary_projection_enabled", True))
        self.max_inward_correction = float(self.raw_cfg.get("max_inward_correction", self.raw_cfg.get("vmax", 1.0)))
        self.inter_agent_activation_distance = float(self.raw_cfg.get("inter_agent_activation_distance", 0.75))
        self.line_of_sight_shortcut = bool(self.raw_cfg.get("line_of_sight_shortcut", True))
        self.line_of_sight_safety_margin = float(self.raw_cfg.get("line_of_sight_safety_margin", self.raw_cfg.get("safety_margin", 0.25)))
        self.planner_subgoal_enabled = bool(self.raw_cfg.get("planner_subgoal_enabled", False))
        planner_cfg = dict(self.raw_cfg.get("planner_subgoal") or self.raw_cfg.get("global_path_pure_pursuit") or {})
        planner_cfg.setdefault("nominal_slot_tracker", nominal_cfg)
        self.path_tracker = GlobalPathPurePursuitTracker(planner_cfg)
        self.final_obstacle_filter_enabled = bool(self.raw_cfg.get("final_obstacle_filter_enabled", True))
        self.prev_action = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.nominal.reset()
        self.path_tracker.reset()

    def compute_action(self, obs: ControllerObservation) -> tuple[np.ndarray, dict[str, Any]]:
        v_nom = self._nominal_desired_velocity(obs)
        _nominal_action, nominal_diag = self.nominal.compute_action(obs)
        nominal_diag["v_nom"] = v_nom.astype(float).tolist()
        nominal_diag["v_goal"] = v_nom.astype(float).tolist()
        nominal_diag["v_final_before_clip"] = v_nom.astype(float).tolist()
        nominal_diag["nominal_tracker_final_clipped"] = True
        risk = self._risk_flags(obs)
        slot_inside_boundary = bool(np.max(np.abs(np.asarray(obs.slot_position[:2], dtype=np.float64))) <= float(obs.world_xy))
        if self.planner_subgoal_enabled and obs.obstacles:
            return self._compute_planner_subgoal_action(obs, v_nom, risk)
        if (
            self.free_space_fallback
            and not risk["obstacle_risk"]
            and not risk["inter_agent_risk"]
            and slot_inside_boundary
        ):
            action, boundary_diag = self._apply_boundary_filter(obs, v_nom)
            reason = "nominal_boundary_filter" if risk["boundary_risk"] else "free_space_fallback"
            out = self._fallback_diag(nominal_diag, action, risk, reason=reason)
            out.update(boundary_diag)
            out["v_boundary"] = np.asarray(action - v_nom, dtype=np.float64).astype(float).tolist()
            out["v_after_obstacle_filter"] = np.asarray(v_nom, dtype=np.float64).astype(float).tolist()
            out["v_after_planner"] = np.asarray(v_nom, dtype=np.float64).astype(float).tolist()
            out["v_final_before_clip"] = v_nom.astype(float).tolist()
            out["v_final_after_clip"] = np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist()
            out["final_action"] = np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist()
            out["raw_cmd_norm"] = float(np.linalg.norm(v_nom))
            out["clip_flag"] = False
            out["speed_saturation_flag"] = False
            out["acceleration_saturation_flag"] = False
            self.prev_action = np.asarray(action, dtype=np.float64).reshape(2).copy()
            return action, out

        if (
            self.line_of_sight_shortcut
            and not risk["inter_agent_risk"]
            and slot_inside_boundary
            and self._line_of_sight_clear(obs)
        ):
            action, boundary_diag = self._apply_boundary_filter(obs, v_nom)
            out = self._fallback_diag(nominal_diag, action, risk, reason="line_of_sight_shortcut")
            out.update(boundary_diag)
            out["v_boundary"] = np.asarray(action - v_nom, dtype=np.float64).astype(float).tolist()
            out["v_after_obstacle_filter"] = np.asarray(v_nom, dtype=np.float64).astype(float).tolist()
            out["v_after_planner"] = np.asarray(v_nom, dtype=np.float64).astype(float).tolist()
            out["v_final_before_clip"] = v_nom.astype(float).tolist()
            out["v_final_after_clip"] = np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist()
            out["final_action"] = np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist()
            out["raw_cmd_norm"] = float(np.linalg.norm(v_nom))
            out["clip_flag"] = False
            out["speed_saturation_flag"] = False
            out["acceleration_saturation_flag"] = False
            self.prev_action = np.asarray(action, dtype=np.float64).reshape(2).copy()
            return action, out

        bounds = (-float(obs.world_xy), float(obs.world_xy), -float(obs.world_xy), float(obs.world_xy))
        action_xy, yaw_rate, _path, diag = self.controller.compute_action(
            np.asarray(obs.position[:2], dtype=np.float64),
            0.0,
            np.asarray(obs.slot_position[:2], dtype=np.float64),
            list(obs.obstacles),
            prev_action=self.prev_action,
            current_velocity_xy=np.asarray(obs.velocity[:2], dtype=np.float64),
            bounds_xy=bounds,
            feedforward_velocity_xy=None,
        )
        action_before_boundary = np.asarray(action_xy, dtype=np.float64).reshape(2).copy()
        action_xy, boundary_diag = self._apply_boundary_filter(obs, action_before_boundary)
        self.prev_action = np.asarray(action_xy, dtype=np.float64).reshape(2).copy()
        out = dict(diag)
        out.update(self._unavailable_component_fields())
        out["v_goal"] = list(nominal_diag.get("v_goal", [float("nan"), float("nan")]))
        nominal_cmd = np.asarray(diag.get("nominal_cmd_xy", v_nom), dtype=np.float64).reshape(2)
        out["v_obstacle"] = np.asarray(action_before_boundary - nominal_cmd, dtype=np.float64).astype(float).tolist()
        out["v_after_obstacle_filter"] = action_before_boundary.astype(float).tolist()
        out["v_after_planner"] = action_before_boundary.astype(float).tolist()
        out["v_final_after_clip"] = np.asarray(action_xy, dtype=np.float64).reshape(2).astype(float).tolist()
        out["final_action"] = np.asarray(action_xy, dtype=np.float64).reshape(2).astype(float).tolist()
        out["v_boundary"] = np.asarray(action_xy - action_before_boundary, dtype=np.float64).astype(float).tolist()
        out["clip_flag"] = bool(float(out.get("raw_selected_speed", np.linalg.norm(action_xy))) > np.linalg.norm(action_xy) + 1e-9)
        out["yaw_rate"] = float(yaw_rate)
        out["raw_cmd_norm"] = float(np.linalg.norm(action_xy))
        out["existing_bypass_reason"] = "existing_safety_path_modules"
        out["used_existing_safety_modules"] = True
        out.update(risk)
        out.update(boundary_diag)
        return np.asarray(action_xy, dtype=np.float64).reshape(2), out

    def _compute_planner_subgoal_action(
        self,
        obs: ControllerObservation,
        v_nom: np.ndarray,
        risk: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        planner_action, planner_diag = self.path_tracker.compute_action(obs)
        subgoal = np.asarray(planner_diag.get("current_subgoal", obs.slot_position[:2]), dtype=np.float64).reshape(2)
        action_before_boundary = np.asarray(planner_action, dtype=np.float64).reshape(2)
        obstacle_diag: dict[str, Any] = {}
        if self.final_obstacle_filter_enabled:
            bounds = (-float(obs.world_xy), float(obs.world_xy), -float(obs.world_xy), float(obs.world_xy))
            filtered, yaw_rate, _path, obstacle_diag = self.controller.compute_action(
                np.asarray(obs.position[:2], dtype=np.float64),
                0.0,
                subgoal,
                list(obs.obstacles),
                prev_action=self.prev_action,
                current_velocity_xy=np.asarray(obs.velocity[:2], dtype=np.float64),
                bounds_xy=bounds,
                feedforward_velocity_xy=None,
            )
            action_before_boundary = np.asarray(filtered, dtype=np.float64).reshape(2)
        else:
            yaw_rate = 0.0
        action_xy, boundary_diag = self._apply_boundary_filter(obs, action_before_boundary)
        self.prev_action = np.asarray(action_xy, dtype=np.float64).reshape(2).copy()

        nominal_cmd = np.asarray(obstacle_diag.get("nominal_cmd_xy", planner_action), dtype=np.float64).reshape(2)
        out = dict(obstacle_diag)
        out.update(planner_diag)
        out.update(boundary_diag)
        out.update(risk)
        out["v_nom"] = np.asarray(v_nom, dtype=np.float64).reshape(2).astype(float).tolist()
        out["v_goal"] = np.asarray(v_nom, dtype=np.float64).reshape(2).astype(float).tolist()
        out["v_planner_subgoal"] = np.asarray(planner_action, dtype=np.float64).reshape(2).astype(float).tolist()
        out["v_obstacle"] = np.asarray(action_before_boundary - nominal_cmd, dtype=np.float64).astype(float).tolist()
        out["v_after_obstacle_filter"] = action_before_boundary.astype(float).tolist()
        out["v_after_planner"] = np.asarray(planner_action, dtype=np.float64).reshape(2).astype(float).tolist()
        out["v_boundary"] = np.asarray(action_xy - action_before_boundary, dtype=np.float64).astype(float).tolist()
        out["v_final_before_clip"] = np.asarray(planner_action, dtype=np.float64).reshape(2).astype(float).tolist()
        out["v_final_after_clip"] = np.asarray(action_xy, dtype=np.float64).reshape(2).astype(float).tolist()
        out["final_action"] = np.asarray(action_xy, dtype=np.float64).reshape(2).astype(float).tolist()
        out["raw_cmd_norm"] = float(np.linalg.norm(planner_action))
        out["clip_flag"] = bool(float(out.get("raw_selected_speed", np.linalg.norm(action_xy))) > np.linalg.norm(action_xy) + 1e-9)
        out["yaw_rate"] = float(yaw_rate)
        out["existing_bypass_reason"] = "planner_subgoal"
        out["used_existing_safety_modules"] = True
        out["planner_subgoal_enabled"] = True
        return np.asarray(action_xy, dtype=np.float64).reshape(2), out

    def _nominal_desired_velocity(self, obs: ControllerObservation) -> np.ndarray:
        pos = np.asarray(obs.position[:2], dtype=np.float64).reshape(2)
        vel = np.asarray(obs.velocity[:2], dtype=np.float64).reshape(2)
        slot = np.asarray(obs.slot_position[:2], dtype=np.float64).reshape(2)
        slot_vel = np.asarray(obs.slot_velocity[:2], dtype=np.float64).reshape(2)
        return slot_vel + float(self.nominal.kp) * (slot - pos) - float(self.nominal.kd) * vel

    def _risk_flags(self, obs: ControllerObservation) -> dict[str, Any]:
        p = np.asarray(obs.position[:2], dtype=np.float64)
        nearest = float("inf")
        for obstacle in obs.obstacles:
            center = np.asarray(getattr(obstacle, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
            radius = float(getattr(obstacle, "radius", 0.0))
            nearest = min(nearest, float(np.linalg.norm(p - center) - radius - obs.uav_radius - obs.safety_margin))
        bmargin = float(obs.world_xy) - float(obs.uav_radius) - float(np.max(np.abs(p)))
        inter_agent_min = float("inf")
        if obs.peer_positions is not None:
            peers = np.asarray(obs.peer_positions, dtype=np.float64).reshape(-1, 3)
            for peer in peers:
                d = float(np.linalg.norm(peer[:2] - p))
                if d > 1e-9:
                    inter_agent_min = min(inter_agent_min, d)
        return {
            "nearest_obstacle_distance_est": nearest,
            "boundary_margin_est": bmargin,
            "inter_agent_min_distance_est": inter_agent_min,
            "obstacle_risk": bool(nearest <= self.obstacle_activation_distance),
            "boundary_risk": bool(bmargin <= self.boundary_activation_distance),
            "inter_agent_risk": bool(inter_agent_min <= self.inter_agent_activation_distance),
        }

    def _line_of_sight_clear(self, obs: ControllerObservation) -> bool:
        if not obs.obstacles:
            return True
        return bool(
            has_line_of_sight(
                np.asarray(obs.position[:2], dtype=np.float64),
                np.asarray(obs.slot_position[:2], dtype=np.float64),
                list(obs.obstacles),
                safety_margin=self.line_of_sight_safety_margin,
                uav_radius=obs.uav_radius,
            )
        )

    def _fallback_diag(self, diag: dict[str, Any], action: np.ndarray, risk: dict[str, Any], *, reason: str) -> dict[str, Any]:
        out = dict(diag)
        out["existing_bypass_reason"] = reason
        out["used_existing_safety_modules"] = False
        out["v_obstacle"] = [0.0, 0.0]
        out["v_boundary"] = [0.0, 0.0]
        out["v_path"] = [0.0, 0.0]
        out["v_final_after_clip"] = np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist()
        out["v_after_obstacle_filter"] = np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist()
        out["v_after_planner"] = np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist()
        out["final_action"] = np.asarray(action, dtype=np.float64).reshape(2).astype(float).tolist()
        out.update(risk)
        return out

    def _apply_boundary_filter(self, obs: ControllerObservation, action_xy: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        before = np.asarray(action_xy, dtype=np.float64).reshape(2).copy()
        p = np.asarray(obs.position[:2], dtype=np.float64).reshape(2)
        v = np.asarray(obs.velocity[:2], dtype=np.float64).reshape(2)
        if self.boundary_filter_mode == "minimal_projection":
            return self._apply_minimal_projection_boundary_filter(obs, before, p, v)
        after = before.copy()
        boundary_records = _all_boundaries(p, obs.world_xy)
        active_records = [
            (name, normal, raw_distance, float(raw_distance - obs.uav_radius))
            for name, normal, raw_distance in boundary_records
            if float(raw_distance - obs.uav_radius) <= self.boundary_activation_distance
        ]
        details: dict[str, Any] = {}
        projected_failed = False
        changed = False
        eps = 1e-9

        for name, normal, raw_distance, boundary_margin in active_records:
            action_before_axis = after.copy()
            velocity_out = float(np.dot(v, normal))
            action_out_before = float(np.dot(action_before_axis, normal))
            v_out_pos = max(velocity_out, 0.0)
            braking_distance = (v_out_pos * v_out_pos) / max(2.0 * float(obs.uav_amax), eps)
            braking_margin = float(boundary_margin - braking_distance)
            inward_correction = 0.0
            if self.boundary_projection_enabled:
                if action_out_before > 0.0:
                    after = after - action_out_before * normal
                if braking_margin < self.boundary_braking_margin:
                    requested = self.boundary_braking_gain * (self.boundary_braking_margin - braking_margin)
                    inward_correction = min(max(float(requested), 0.0), self.max_inward_correction)
                    after, inward_correction = _add_inward_correction_preserve_tangent(
                        after,
                        normal,
                        correction=inward_correction,
                        vmax=float(obs.uav_vmax),
                    )
                action_out_mid = float(np.dot(after, normal))
                if action_out_mid > 0.0:
                    after = after - action_out_mid * normal
            action_out_after = float(np.dot(after, normal))
            if boundary_margin < self.boundary_hard_margin and action_out_after > 1e-6:
                projected_failed = True
            changed = bool(changed or np.linalg.norm(after - action_before_axis) > 1e-9)
            normal_before = float(np.dot(before, normal))
            normal_after = float(np.dot(after, normal))
            tangent_before = before - normal_before * normal
            tangent_after = after - normal_after * normal
            details[name] = {
                "distance_to_boundary": float(raw_distance),
                "boundary_margin": float(boundary_margin),
                "velocity_outward_projection": float(velocity_out),
                "action_outward_projection_before": float(action_out_before),
                "action_outward_projection_after": float(action_out_after),
                "estimated_braking_distance": float(braking_distance),
                "braking_margin": float(braking_margin),
                "inward_correction": float(inward_correction),
                "normal_component_before": float(normal_before),
                "normal_component_after": float(normal_after),
                "tangential_norm_before": float(np.linalg.norm(tangent_before)),
                "tangential_norm_after": float(np.linalg.norm(tangent_after)),
            }

        after = clip_norm(after, obs.uav_vmax) if obs.controller_clip_enabled else after
        for name, normal, raw_distance, boundary_margin in active_records:
            action_out_after = float(np.dot(after, normal))
            if boundary_margin < self.boundary_hard_margin and action_out_after > 1e-6:
                projected_failed = True
            if name in details:
                details[name]["action_outward_projection_after"] = action_out_after

        if projected_failed:
            print("BOUNDARY_PROJECTED_ACTION_FAILED")

        normal, boundary_name, raw_distance = _nearest_boundary(p, obs.world_xy)
        boundary_margin = float(raw_distance - obs.uav_radius)
        velocity_out = float(np.dot(v, normal))
        action_out_before = float(np.dot(before, normal))
        action_out_after = float(np.dot(after, normal))
        v_out_pos = max(velocity_out, 0.0)
        braking_distance = (v_out_pos * v_out_pos) / max(2.0 * float(obs.uav_amax), 1e-9)
        braking_margin = boundary_margin - braking_distance
        return after, {
            "nearest_boundary_name": boundary_name,
            "outward_normal": normal.astype(float).tolist(),
            "velocity_outward_projection": velocity_out,
            "action_outward_projection": action_out_after,
            "action_outward_projection_before_filter": action_out_before,
            "distance_to_boundary": float(raw_distance),
            "estimated_braking_distance": float(braking_distance),
            "braking_margin": float(braking_margin),
            "boundary_filter_active": bool(self.boundary_projection_enabled and changed),
            "boundary_active_names": ",".join(name for name, *_ in active_records),
            "boundary_projected_action_failed": bool(projected_failed),
            "action_before_boundary_filter": before.astype(float).tolist(),
            "action_after_boundary_filter": after.astype(float).tolist(),
            "boundary_filter_details": details,
        }

    def _apply_minimal_projection_boundary_filter(
        self,
        obs: ControllerObservation,
        before: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        boundary_records = _all_boundaries(p, obs.world_xy)
        constraints: list[tuple[str, np.ndarray, float]] = []
        details: dict[str, Any] = {}
        d_safe = max(float(self.boundary_hard_margin), float(obs.safety_margin))
        for name, normal, raw_distance in boundary_records:
            margin = float(raw_distance - obs.uav_radius)
            active = bool(margin <= self.boundary_activation_distance)
            b = float(self.boundary_alpha * max(margin - d_safe, 0.0))
            velocity_out = float(np.dot(v, normal))
            braking_distance = (max(velocity_out, 0.0) ** 2) / max(2.0 * float(obs.uav_amax), 1e-9)
            braking_margin = float(margin - braking_distance)
            action_out_before = float(np.dot(before, normal))
            if active:
                constraints.append((name, normal, b))
            details[name] = {
                "distance_to_boundary": float(raw_distance),
                "boundary_margin": float(margin),
                "constraint_bound": float(b),
                "active": bool(active),
                "velocity_outward_projection": float(velocity_out),
                "action_outward_projection_before": float(action_out_before),
                "action_outward_projection_after": float(action_out_before),
                "estimated_braking_distance": float(braking_distance),
                "braking_margin": float(braking_margin),
                "inward_correction": 0.0,
                "normal_component_before": float(action_out_before),
                "normal_component_after": float(action_out_before),
                "tangential_norm_before": float(np.linalg.norm(before - action_out_before * normal)),
                "tangential_norm_after": float(np.linalg.norm(before - action_out_before * normal)),
                "tangential_retention_ratio": 1.0,
            }

        projected, projection_diag = _project_velocity_to_halfspaces(before, constraints)
        after_projection = projected.copy()
        after_braking = after_projection.copy()
        projected_failed = False
        for name, normal, raw_distance in boundary_records:
            margin = float(raw_distance - obs.uav_radius)
            action_out_after = float(np.dot(after_projection, normal))
            if margin < self.boundary_hard_margin and action_out_after > 1e-6:
                projected_failed = True
            if name in details:
                normal_before = float(np.dot(before, normal))
                normal_after = float(np.dot(after_projection, normal))
                tangent_before = before - normal_before * normal
                tangent_after = after_projection - normal_after * normal
                before_norm = float(np.linalg.norm(tangent_before))
                after_norm = float(np.linalg.norm(tangent_after))
                details[name].update(
                    {
                        "action_outward_projection_after": action_out_after,
                        "normal_component_after": normal_after,
                        "tangential_norm_after": after_norm,
                        "tangential_retention_ratio": (
                            float(after_norm / before_norm) if before_norm > 1e-9 else 1.0
                        ),
                    }
                )

        if projected_failed:
            print("BOUNDARY_PROJECTED_ACTION_FAILED")

        normal, boundary_name, raw_distance = _nearest_boundary(p, obs.world_xy)
        boundary_margin = float(raw_distance - obs.uav_radius)
        velocity_out = float(np.dot(v, normal))
        action_out_before = float(np.dot(before, normal))
        action_out_after = float(np.dot(after_projection, normal))
        braking_distance = (max(velocity_out, 0.0) ** 2) / max(2.0 * float(obs.uav_amax), 1e-9)
        braking_margin = boundary_margin - braking_distance
        changed = bool(np.linalg.norm(after_projection - before) > 1e-9)
        return after_projection, {
            "nearest_boundary_name": boundary_name,
            "outward_normal": normal.astype(float).tolist(),
            "velocity_outward_projection": velocity_out,
            "action_outward_projection": action_out_after,
            "action_outward_projection_before_filter": action_out_before,
            "distance_to_boundary": float(raw_distance),
            "estimated_braking_distance": float(braking_distance),
            "braking_margin": float(braking_margin),
            "boundary_filter_active": bool(self.boundary_projection_enabled and changed),
            "boundary_active_names": ",".join(name for name, *_ in constraints),
            "boundary_projected_action_failed": bool(projected_failed),
            "boundary_filter_mode": "minimal_projection",
            "boundary_projection_case": str(projection_diag.get("projection_case", "")),
            "action_before_boundary_filter": before.astype(float).tolist(),
            "action_after_boundary_filter": after_projection.astype(float).tolist(),
            "v_nom": before.astype(float).tolist(),
            "v_before_boundary_filter": before.astype(float).tolist(),
            "v_after_boundary_projection": after_projection.astype(float).tolist(),
            "v_after_braking_correction": after_braking.astype(float).tolist(),
            "boundary_filter_details": details,
        }

    @staticmethod
    def _unavailable_component_fields() -> dict[str, Any]:
        nan_vec = [float("nan"), float("nan")]
        return {
            "v_goal": nan_vec,
            "v_obstacle": nan_vec,
            "v_boundary": nan_vec,
            "v_path": nan_vec,
            "v_smooth": nan_vec,
            "v_final_before_clip": nan_vec,
            "v_after_obstacle_filter": nan_vec,
            "v_after_planner": nan_vec,
        }


def make_controller(name: str, cfg: dict[str, Any] | None = None):
    """Factory for benchmark controller names."""
    key = str(name).strip().lower()
    if key in ("pure_pursuit", "pure", "pp"):
        from experiments.slot_tracking.controllers.baseline_pure_pursuit import PurePursuitTracker

        return PurePursuitTracker(cfg)
    if key in ("pd", "pd_tracker"):
        from experiments.slot_tracking.controllers.baseline_pd_tracker import PDTracker

        return PDTracker(cfg)
    if key in ("apf", "apf_tracker"):
        from experiments.slot_tracking.controllers.baseline_apf_tracker import APFTracker

        return APFTracker(cfg)
    if key in ("nominal_slot_tracker", "nominal", "slot_tracker"):
        from experiments.slot_tracking.controllers.nominal_slot_tracker import NominalSlotTracker

        return NominalSlotTracker(cfg)
    if key in ("global_path_pure_pursuit", "astar_nominal_tracker", "global_path", "astar"):
        from experiments.slot_tracking.controllers.baseline_global_path_tracker import GlobalPathPurePursuitTracker

        return GlobalPathPurePursuitTracker(cfg)
    if key in ("existing_no_planner", "wrapped_existing_no_planner"):
        data = dict(cfg or {})
        data["planner_subgoal_enabled"] = False
        return ExistingControllerWrapper(data)
    if key in ("existing_planner_subgoal", "existing_with_planner", "wrapped_existing_planner_subgoal"):
        data = dict(cfg or {})
        data["planner_subgoal_enabled"] = True
        return ExistingControllerWrapper(data)
    if key in ("existing", "wrapped_existing_controller"):
        return ExistingControllerWrapper(cfg)
    raise ValueError(f"Unknown controller: {name}")


def _nearest_boundary(pos_xy: np.ndarray, world_xy: float) -> tuple[np.ndarray, str, float]:
    records = _all_boundaries(pos_xy, world_xy)
    name, normal, raw_distance = min(records, key=lambda x: x[2])
    return normal, name, float(raw_distance)


def _all_boundaries(pos_xy: np.ndarray, world_xy: float) -> list[tuple[str, np.ndarray, float]]:
    p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    w = float(world_xy)
    return [
        ("x_min", np.array([-1.0, 0.0], dtype=np.float64), float(p[0] + w)),
        ("x_max", np.array([1.0, 0.0], dtype=np.float64), float(w - p[0])),
        ("y_min", np.array([0.0, -1.0], dtype=np.float64), float(p[1] + w)),
        ("y_max", np.array([0.0, 1.0], dtype=np.float64), float(w - p[1])),
    ]


def _add_inward_correction_preserve_tangent(
    action_xy: np.ndarray,
    normal: np.ndarray,
    *,
    correction: float,
    vmax: float,
) -> tuple[np.ndarray, float]:
    action = np.asarray(action_xy, dtype=np.float64).reshape(2).copy()
    n = np.asarray(normal, dtype=np.float64).reshape(2)
    requested = max(float(correction), 0.0)
    if requested <= 0.0:
        return action, 0.0
    tangential = action - float(np.dot(action, n)) * n
    tangential_norm = float(np.linalg.norm(tangential))
    limit = max(float(vmax), 0.0)
    if limit <= 0.0:
        return action, 0.0
    max_normal_mag = float(np.sqrt(max(limit * limit - tangential_norm * tangential_norm, 0.0)))
    current_outward = float(np.dot(action, n))
    desired_outward = current_outward - requested
    limited_outward = max(desired_outward, -max_normal_mag)
    applied = max(current_outward - limited_outward, 0.0)
    return tangential + limited_outward * n, float(applied)


def _project_velocity_to_halfspaces(
    velocity_xy: np.ndarray,
    constraints: list[tuple[str, np.ndarray, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a 2D velocity onto active boundary half-spaces."""
    v = np.asarray(velocity_xy, dtype=np.float64).reshape(2)
    if not constraints:
        return v.copy(), {"projection_case": "none"}

    def feasible(x: np.ndarray) -> bool:
        return all(float(np.dot(n, x)) <= float(b) + 1e-9 for _name, n, b in constraints)

    if feasible(v):
        return v.copy(), {"projection_case": "already_feasible"}

    candidates: list[tuple[float, str, np.ndarray]] = []
    for name, n, b in constraints:
        violation = float(np.dot(n, v) - float(b))
        x = v - max(violation, 0.0) * np.asarray(n, dtype=np.float64).reshape(2)
        if feasible(x):
            candidates.append((float(np.linalg.norm(x - v)), f"one_constraint:{name}", x))

    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            name_i, n_i, b_i = constraints[i]
            name_j, n_j, b_j = constraints[j]
            a = np.stack([np.asarray(n_i, dtype=np.float64), np.asarray(n_j, dtype=np.float64)], axis=0)
            bvec = np.array([float(b_i), float(b_j)], dtype=np.float64)
            det = float(np.linalg.det(a))
            if abs(det) <= 1e-9:
                continue
            x = np.linalg.solve(a, bvec)
            if feasible(x):
                candidates.append((float(np.linalg.norm(x - v)), f"two_constraints:{name_i},{name_j}", x))

    if candidates:
        _dist, case, best = min(candidates, key=lambda item: item[0])
        return np.asarray(best, dtype=np.float64).reshape(2), {"projection_case": case}

    # The square-boundary constraints should be feasible, but fall back to a
    # deterministic sequential projection if floating point degeneracy appears.
    x = v.copy()
    for name, n, b in constraints:
        del name
        violation = float(np.dot(n, x) - float(b))
        if violation > 0.0:
            x = x - violation * np.asarray(n, dtype=np.float64).reshape(2)
    return x, {"projection_case": "sequential_fallback"}
