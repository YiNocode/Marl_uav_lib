"""PD/velocity tracking baseline for dynamic slot tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.slot_tracking.controllers.baseline_pure_pursuit import (
    ControllerObservation,
    clip_norm,
)


class PDTracker:
    """PD tracker with optional slot velocity feed-forward."""

    name = "pd"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        raw = dict(cfg or {})
        self.kp = float(raw.get("kp", 1.6))
        self.kd = float(raw.get("kd", 0.45))
        self.slot_velocity_gain = float(raw.get("slot_velocity_gain", 0.6))

    def reset(self) -> None:
        pass

    def compute_action(self, obs: ControllerObservation) -> tuple[np.ndarray, dict[str, Any]]:
        pos_err = np.asarray(obs.slot_position[:2], dtype=np.float64) - np.asarray(obs.position[:2], dtype=np.float64)
        vel_err = (
            self.slot_velocity_gain * np.asarray(obs.slot_velocity[:2], dtype=np.float64)
            - np.asarray(obs.velocity[:2], dtype=np.float64)
        )
        cmd = self.kp * pos_err + self.kd * vel_err
        before_clip = cmd.copy()
        cmd = clip_norm(cmd, obs.uav_vmax) if obs.controller_clip_enabled else cmd
        return cmd, {
            "v_goal": before_clip.astype(float).tolist(),
            "v_final_before_clip": before_clip.astype(float).tolist(),
            "v_final_after_clip": cmd.astype(float).tolist(),
            "raw_cmd_norm": float(np.linalg.norm(before_clip)),
            "clip_flag": bool(np.linalg.norm(before_clip) > np.linalg.norm(cmd) + 1e-9),
        }
