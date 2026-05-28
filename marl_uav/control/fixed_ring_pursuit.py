"""Fixed-ring encirclement controller for 3v1 pursuit baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from marl_uav.control.geometric_pursuit_baselines import (
    default_proportional_gains,
    proportional_actions_to_targets,
    pursuer_yaws_from_backend,
)


def fixed_ring_actions_from_state(
    lin_pos: np.ndarray,
    pursuer_ids: np.ndarray,
    evader_id: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    ring_radius: float,
    approach_gain: float,
    z_gain: float,
    phase: float = 0.0,
    assignment: str = "fixed",
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
) -> np.ndarray:
    """Return continuous [vx, vy, yaw_rate, vz] actions for a fixed ring.

    The controller places three target points around the evader at 120 degree
    spacing and moves each pursuer toward its assigned point. ``assignment`` can
    be ``fixed`` for stable agent ids or ``angle_order`` for angular sorting.
    """
    pos = np.asarray(lin_pos, dtype=np.float32)
    pids = np.asarray(pursuer_ids, dtype=np.int64).reshape(3)
    eid = int(evader_id)

    evader = pos[eid]
    angles = float(phase) + np.arange(3, dtype=np.float32) * (2.0 * np.pi / 3.0)
    targets = np.zeros((3, 3), dtype=np.float32)
    targets[:, 0] = evader[0] + float(ring_radius) * np.cos(angles)
    targets[:, 1] = evader[1] + float(ring_radius) * np.sin(angles)
    targets[:, 2] = evader[2]

    if assignment == "angle_order":
        rel = pos[pids, :2] - evader[None, :2]
        pursuer_order = np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))
        target_order = np.arange(3, dtype=np.int64)
        assigned_targets = np.zeros_like(targets)
        assigned_targets[pursuer_order] = targets[target_order]
    elif assignment == "fixed":
        assigned_targets = targets
    else:
        raise ValueError(f"Unsupported fixed-ring assignment={assignment!r}")

    return proportional_actions_to_targets(
        pos[pids],
        assigned_targets,
        action_low,
        action_high,
        xy_gain=approach_gain,
        z_gain=z_gain,
        pursuer_yaw=pursuer_yaw,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )


def make_fixed_ring_get_actions_fn(
    env: Any,
    *,
    ring_radius: float = 1.6,
    approach_gain: float | None = 0.25,
    z_gain: float | None = 0.20,
    yaw_gain: float | None = 0.25,
    yaw_align_min_speed: float = 0.25,
    phase: float = 0.0,
    assignment: str = "fixed",
):
    """Build a RolloutWorker ``get_actions_fn`` for the fixed-ring baseline."""
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError("fixed-ring baseline requires a continuous action-space env")
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)
    approach_gain, z_gain, yaw_gain = default_proportional_gains(
        low, high, xy_gain=approach_gain, z_gain=z_gain, yaw_gain=yaw_gain
    )

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError("Environment must be reset before selecting fixed-ring actions.")
        backend = env.prev_backend_state
        lin_pos = np.asarray(backend.states[:, 3, :], dtype=np.float32)
        yaws = pursuer_yaws_from_backend(backend, env.task_state.pursuer_ids)
        return fixed_ring_actions_from_state(
            lin_pos,
            env.task_state.pursuer_ids,
            env.task_state.evader_id,
            low,
            high,
            ring_radius=ring_radius,
            approach_gain=approach_gain,
            z_gain=z_gain,
            phase=phase,
            assignment=assignment,
            pursuer_yaw=yaws,
            yaw_gain=yaw_gain,
            yaw_align_min_speed=yaw_align_min_speed,
        )

    return get_actions
