"""XY arena boundary barriers for low-level pursuit controllers."""

from __future__ import annotations

import numpy as np


def apply_xy_boundary_barrier(
    pos_xy: np.ndarray,
    u_world: np.ndarray,
    *,
    world_xy: float | None,
    boundary_margin: float,
    boundary_alpha: float,
    action_low_xy: np.ndarray,
    action_high_xy: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Clamp world-frame XY velocity so the agent stays inside the usable arena."""
    if world_xy is None or not np.isfinite(float(world_xy)):
        return np.asarray(u_world, dtype=np.float64).reshape(2).copy(), False

    w = max(float(world_xy), 1e-6)
    margin = float(np.clip(boundary_margin, 0.0, max(w - 1e-6, 0.0)))
    alpha = max(float(boundary_alpha), 0.0)
    safe_min = -w + margin
    safe_max = w - margin
    p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    u = np.asarray(u_world, dtype=np.float64).reshape(2).copy()
    low = np.asarray(action_low_xy, dtype=np.float64).reshape(2).copy()
    high = np.asarray(action_high_xy, dtype=np.float64).reshape(2).copy()

    u_min = low.copy()
    u_max = high.copy()
    for ax in range(2):
        lower_h = float(p[ax] - safe_min)
        upper_h = float(safe_max - p[ax])
        u_min[ax] = max(float(u_min[ax]), -alpha * lower_h)
        u_max[ax] = min(float(u_max[ax]), alpha * upper_h)
        if u_min[ax] > u_max[ax]:
            center_cmd = float(np.clip(-alpha * p[ax], low[ax], high[ax]))
            u_min[ax] = center_cmd
            u_max[ax] = center_cmd

    safe = np.minimum(np.maximum(u, u_min), u_max)
    return safe, bool(np.linalg.norm(safe - u) > 1e-6)
