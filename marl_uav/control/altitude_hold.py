"""Hard altitude hold for deployable execution (vz saturation + yaw/xy gating)."""

from __future__ import annotations

import numpy as np


def hard_altitude_hold(
    z: float,
    z_hold: float,
    vz_min: float,
    vz_max: float,
    *,
    deadband: float = 0.025,
    priority_band: float = 0.10,
    z_floor: float | None = None,
    z_ceiling: float | None = None,
    floor_margin: float = 0.25,
    ceiling_margin: float = 0.10,
) -> tuple[float, float]:
    """
    Hard altitude execution layer.

    Returns ``(vz_cmd, horizontal_gate)``:

    - ``vz_cmd``: saturated vertical speed toward ``z_hold`` whenever outside deadband.
    - ``horizontal_gate``: scale in ``[0, 1]`` applied to yaw rate and horizontal body
      velocity; zero when altitude error exceeds ``priority_band``.
    """
    target = float(z_hold)
    if z_floor is not None:
        target = max(target, float(z_floor) + max(float(floor_margin), 0.0))
    if z_ceiling is not None:
        target = min(target, float(z_ceiling) - max(float(ceiling_margin), 0.0))

    err = target - float(z)
    abs_err = abs(err)
    if abs_err <= float(deadband):
        return 0.0, 1.0

    sign = 1.0 if err > 0.0 else -1.0
    vmax_z = max(abs(float(vz_min)), abs(float(vz_max)))
    vz = float(np.clip(sign * vmax_z, float(vz_min), float(vz_max)))

    span = max(float(priority_band) - float(deadband), 1e-6)
    if abs_err >= float(priority_band):
        gate = 0.0
    else:
        gate = 1.0 - (abs_err - float(deadband)) / span
    return vz, float(np.clip(gate, 0.0, 1.0))


def apply_hard_altitude_to_action_row(
    row: np.ndarray,
    z: float,
    z_hold: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    deadband: float = 0.025,
    priority_band: float = 0.10,
    gate_horizontal: bool = True,
    z_floor: float | None = None,
    z_ceiling: float | None = None,
    floor_margin: float = 0.25,
    ceiling_margin: float = 0.10,
) -> None:
    """In-place: enforce hard vz; optionally gate yaw + horizontal body commands."""
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    out = np.asarray(row, dtype=np.float32).reshape(-1)
    if out.shape[0] < 4:
        return

    vz, gate = hard_altitude_hold(
        z,
        z_hold,
        float(low[3]),
        float(high[3]),
        deadband=deadband,
        priority_band=priority_band,
        z_floor=z_floor,
        z_ceiling=z_ceiling,
        floor_margin=floor_margin,
        ceiling_margin=ceiling_margin,
    )
    if not gate_horizontal:
        gate = 1.0
    out[0] = np.float32(float(out[0]) * gate)
    out[1] = np.float32(float(out[1]) * gate)
    if out.shape[0] >= 3:
        out[2] = np.float32(float(out[2]) * gate)
    out[3] = np.float32(vz)
