"""Pure-pursuit velocity baseline for slot tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ControllerObservation:
    """Per-agent observation supplied to all benchmark controllers."""

    position: np.ndarray
    velocity: np.ndarray
    slot_position: np.ndarray
    slot_velocity: np.ndarray
    obstacles: list[Any]
    world_xy: float
    dt: float
    uav_vmax: float
    uav_amax: float
    uav_radius: float
    safety_margin: float
    peer_positions: np.ndarray | None = None
    controller_clip_enabled: bool = True


def clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if limit > 0.0 and n > limit:
        return v * (float(limit) / max(n, 1e-9))
    return v


class PurePursuitTracker:
    """Command velocity directly toward the current slot position."""

    name = "pure_pursuit"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        raw = dict(cfg or {})
        self.kp = float(raw.get("kp", 1.4))
        self.arrival_radius = float(raw.get("arrival_radius", 0.05))

    def reset(self) -> None:
        pass

    def compute_action(self, obs: ControllerObservation) -> tuple[np.ndarray, dict[str, Any]]:
        delta = np.asarray(obs.slot_position[:2], dtype=np.float64) - np.asarray(obs.position[:2], dtype=np.float64)
        dist = float(np.linalg.norm(delta))
        if dist <= self.arrival_radius:
            cmd = np.zeros(2, dtype=np.float64)
        else:
            cmd = self.kp * delta
        before_clip = cmd.copy()
        cmd = clip_norm(cmd, obs.uav_vmax) if obs.controller_clip_enabled else cmd
        return cmd, {
            "v_goal": before_clip.astype(float).tolist(),
            "v_final_before_clip": before_clip.astype(float).tolist(),
            "v_final_after_clip": cmd.astype(float).tolist(),
            "raw_cmd_norm": float(np.linalg.norm(before_clip)),
            "clip_flag": bool(np.linalg.norm(before_clip) > np.linalg.norm(cmd) + 1e-9),
        }
