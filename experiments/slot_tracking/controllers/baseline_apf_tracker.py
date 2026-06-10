"""Artificial-potential-field baseline for obstacle-aware slot tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.slot_tracking.controllers.baseline_pure_pursuit import (
    ControllerObservation,
    clip_norm,
)


class APFTracker:
    """Attractive slot tracker plus obstacle and boundary repulsion."""

    name = "apf"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        raw = dict(cfg or {})
        self.kp = float(raw.get("kp", 1.2))
        self.kd = float(raw.get("kd", 0.25))
        self.obstacle_gain = float(raw.get("obstacle_gain", 0.8))
        self.boundary_gain = float(raw.get("boundary_gain", 1.5))
        self.influence_radius = float(raw.get("influence_radius", 2.4))

    def reset(self) -> None:
        pass

    def compute_action(self, obs: ControllerObservation) -> tuple[np.ndarray, dict[str, Any]]:
        p = np.asarray(obs.position[:2], dtype=np.float64)
        v = np.asarray(obs.velocity[:2], dtype=np.float64)
        goal = np.asarray(obs.slot_position[:2], dtype=np.float64)
        v_goal = self.kp * (goal - p) - self.kd * v
        v_obstacle = np.zeros(2, dtype=np.float64)
        v_boundary = np.zeros(2, dtype=np.float64)
        cmd = v_goal.copy()
        active = 0
        min_clearance = 1e9
        for obstacle in obs.obstacles:
            center = np.asarray(getattr(obstacle, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
            radius = float(getattr(obstacle, "radius", 0.0))
            rel = p - center
            dist = float(np.linalg.norm(rel))
            normal = np.array([1.0, 0.0], dtype=np.float64) if dist < 1e-9 else rel / dist
            clearance = dist - radius - obs.uav_radius - obs.safety_margin
            min_clearance = min(min_clearance, clearance)
            if clearance < self.influence_radius:
                active += 1
                c = max(clearance, 1e-3)
                strength = self.obstacle_gain * (1.0 / c - 1.0 / self.influence_radius) / (c * c)
                term = min(strength, 4.0 * obs.uav_vmax) * normal
                v_obstacle = v_obstacle + term
                cmd = cmd + term

        w = float(obs.world_xy)
        margin = max(0.8, obs.uav_radius + obs.safety_margin)
        for axis in range(2):
            lower_clear = p[axis] - (-w)
            upper_clear = w - p[axis]
            if lower_clear < margin:
                term = self.boundary_gain * (margin - lower_clear)
                v_boundary[axis] += term
                cmd[axis] += term
            if upper_clear < margin:
                term = -self.boundary_gain * (margin - upper_clear)
                v_boundary[axis] += term
                cmd[axis] += term

        before_clip = cmd.copy()
        cmd = clip_norm(cmd, obs.uav_vmax) if obs.controller_clip_enabled else cmd
        return cmd, {
            "v_goal": v_goal.astype(float).tolist(),
            "v_obstacle": v_obstacle.astype(float).tolist(),
            "v_boundary": v_boundary.astype(float).tolist(),
            "v_final_before_clip": before_clip.astype(float).tolist(),
            "v_final_after_clip": cmd.astype(float).tolist(),
            "raw_cmd_norm": float(np.linalg.norm(before_clip)),
            "clip_flag": bool(np.linalg.norm(before_clip) > np.linalg.norm(cmd) + 1e-9),
            "active_obstacle_count": int(active),
            "min_obstacle_clearance_est": float(min_clearance),
        }
