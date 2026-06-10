"""Nominal dynamic slot tracker used as the free-space reference."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.slot_tracking.controllers.baseline_pure_pursuit import (
    ControllerObservation,
    clip_norm,
)


class NominalSlotTracker:
    """Dynamic slot tracker with explicit speed and acceleration limits."""

    name = "nominal_slot_tracker"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        raw = dict(cfg or {})
        self.kp = float(raw.get("kp", 1.6))
        self.kd = float(raw.get("kd", 0.45))
        self.vmax = None if raw.get("vmax") is None else float(raw["vmax"])
        self.amax = None if raw.get("amax") is None else float(raw["amax"])
        self.dt = None if raw.get("dt") is None else float(raw["dt"])

    def reset(self) -> None:
        pass

    def compute_action(self, obs: ControllerObservation) -> tuple[np.ndarray, dict[str, Any]]:
        pos = np.asarray(obs.position[:2], dtype=np.float64)
        vel = np.asarray(obs.velocity[:2], dtype=np.float64)
        slot = np.asarray(obs.slot_position[:2], dtype=np.float64)
        slot_vel = np.asarray(obs.slot_velocity[:2], dtype=np.float64)
        vmax = float(obs.uav_vmax if self.vmax is None else self.vmax)
        amax = float(obs.uav_amax if self.amax is None else self.amax)
        dt = float(obs.dt if self.dt is None else self.dt)

        v_nom = slot_vel + self.kp * (slot - pos) - self.kd * vel
        if obs.controller_clip_enabled:
            v_speed_limited = clip_norm(v_nom, vmax)
            dv = v_speed_limited - vel
            dv_limited = clip_norm(dv, amax * dt)
            action = clip_norm(vel + dv_limited, vmax)
        else:
            v_speed_limited = v_nom.copy()
            dv = v_speed_limited - vel
            dv_limited = dv.copy()
            action = v_nom.copy()
        speed_clip = bool(np.linalg.norm(v_nom) > np.linalg.norm(v_speed_limited) + 1e-9)
        accel_clip = bool(np.linalg.norm(dv) > np.linalg.norm(dv_limited) + 1e-9)
        return action, {
            "v_goal": v_nom.astype(float).tolist(),
            "v_obstacle": [0.0, 0.0],
            "v_boundary": [0.0, 0.0],
            "v_path": [0.0, 0.0],
            "v_smooth": v_speed_limited.astype(float).tolist(),
            "v_final_before_clip": v_nom.astype(float).tolist(),
            "v_final_after_clip": action.astype(float).tolist(),
            "final_action": action.astype(float).tolist(),
            "raw_cmd_norm": float(np.linalg.norm(v_nom)),
            "clip_flag": bool(speed_clip or accel_clip),
            "speed_saturation_flag": speed_clip,
            "acceleration_saturation_flag": accel_clip,
            "nominal_tracker_kp": float(self.kp),
            "nominal_tracker_kd": float(self.kd),
        }
