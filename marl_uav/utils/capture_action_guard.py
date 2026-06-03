"""Optional rollout-time capture protection (action_guard / mixed_action)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CaptureProtectionStats:
    """Per-episode accumulators for protection diagnostics."""

    steps: int = 0
    near_capture_steps: int = 0
    protection_active_steps: int = 0
    override_count: int = 0
    override_norm_sum: float = 0.0
    false_positive_approx: int = 0
    false_negative_approx: int = 0

    def to_dict(self) -> dict[str, float]:
        n = max(self.steps, 1)
        return {
            "capture_protection_active_rate": float(self.protection_active_steps / n),
            "near_capture_state_rate": float(self.near_capture_steps / n),
            "capture_protection_override_count": float(self.override_count),
            "capture_protection_override_norm": float(
                self.override_norm_sum / max(self.override_count, 1)
            ),
            "protected_action_fraction": float(self.protection_active_steps / n),
            # Approximate: protection fired but episode did not capture
            "capture_protection_false_positive_rate": float(self.false_positive_approx / n),
            # Approximate: near capture, no protection, episode ended without capture
            "capture_protection_false_negative_rate": float(self.false_negative_approx / n),
        }


@dataclass
class CaptureActionGuard:
    """Apply action_guard or mixed_action when near capture."""

    mode: str = "action_guard"
    enabled: bool = True
    near_capture_dist: float = 2.0
    max_action_deviation: float = 0.5
    protect_if_sce_action_improves_capture: bool = True
    mix_beta: float = 0.5
    stats: CaptureProtectionStats = field(default_factory=CaptureProtectionStats)

    def apply(
        self,
        action_mappo: np.ndarray,
        action_sce: np.ndarray,
        *,
        min_pursuer_evader_dist: float,
        sce_improves: bool = True,
    ) -> np.ndarray:
        """Return possibly modified action; update stats."""
        self.stats.steps += 1
        mappo = np.asarray(action_mappo, dtype=np.float32)
        sce = np.asarray(action_sce, dtype=np.float32)
        if mappo.shape != sce.shape:
            sce = sce.reshape(mappo.shape)

        if not self.enabled:
            return mappo.astype(np.float32)

        if float(min_pursuer_evader_dist) < float(self.near_capture_dist):
            self.stats.near_capture_steps += 1

        active = float(min_pursuer_evader_dist) < float(self.near_capture_dist)
        if self.protect_if_sce_action_improves_capture and not sce_improves:
            active = False

        if not active:
            if (
                float(min_pursuer_evader_dist) < float(self.near_capture_dist) * 1.2
                and not sce_improves
            ):
                self.stats.false_negative_approx += 1
            return mappo.astype(np.float32)

        self.stats.protection_active_steps += 1
        mode = str(self.mode).lower()

        if mode == "mixed_action":
            beta = float(np.clip(self.mix_beta, 0.0, 1.0))
            out = beta * sce + (1.0 - beta) * mappo
        else:
            # action_guard: clamp deviation per element
            delta = mappo - sce
            max_dev = float(self.max_action_deviation)
            delta = np.clip(delta, -max_dev, max_dev)
            out = sce + delta

        diff = float(np.linalg.norm(out - mappo))
        if diff > 1e-6:
            self.stats.override_count += 1
            self.stats.override_norm_sum += diff

        return np.asarray(out, dtype=np.float32)

    def episode_end(self, *, captured: bool) -> None:
        if self.stats.protection_active_steps > 0 and not captured:
            self.stats.false_positive_approx += self.stats.protection_active_steps

    def reset_stats(self) -> None:
        self.stats = CaptureProtectionStats()
