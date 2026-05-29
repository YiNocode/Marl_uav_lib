"""SCE framework: deformable manifold + OT roles + minimal proportional execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from marl_uav.control.geometric_pursuit_baselines import (
    default_proportional_gains,
    proportional_actions_to_targets,
    pursuer_yaws_from_backend,
)
from marl_uav.utils.control_timing import publish_control_timing, should_record_control_timing


def sce_proportional_actions_from_state(
    env: Any,
    lin_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
) -> np.ndarray:
    """
    Minimal execution backend: track OT-assigned manifold slot targets.

    Requires ``pursuit_evasion_3v1_ex1`` with deformable manifold and
    ``role_assignment_mode: entropic_ot``.
    """
    pos = np.asarray(lin_pos, dtype=np.float32)
    task_state = env.task_state
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
    pursuer_pos = pos[pursuer_ids]
    evader_pos = pos[int(task_state.evader_id)]

    task = env.task
    if not hasattr(task, "_assigned_targets_from_state"):
        raise TypeError("SCE controller requires pursuit_evasion_3v1_ex1 task with slot targets.")
    mode = str(getattr(task, "role_assignment_mode", "")).strip().lower()
    if mode != "entropic_ot":
        raise ValueError(
            f"SCE controller expects role_assignment_mode='entropic_ot', got {mode!r}"
        )

    import time

    record = should_record_control_timing(env)
    _, assignment, assigned_targets = task._assigned_targets_from_state(
        pursuer_pos,
        evader_pos,
        task_state=task_state,
        record_timing=record,
    )
    t_after_role = time.perf_counter()
    task_state.assigned_target_indices = np.asarray(assignment, dtype=np.int64).copy()
    actions = proportional_actions_to_targets(
        pursuer_pos,
        assigned_targets,
        action_low,
        action_high,
        xy_gain=xy_gain,
        z_gain=z_gain,
        pursuer_yaw=pursuer_yaw,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )
    if record:
        publish_control_timing(env, action_mapping_time=time.perf_counter() - t_after_role)
    return actions


def make_sce_get_actions_fn(
    env: Any,
    *,
    xy_gain: float | None = 0.25,
    z_gain: float | None = 0.20,
    yaw_gain: float | None = 0.25,
    yaw_align_min_speed: float = 0.25,
):
    """Build a RolloutWorker ``get_actions_fn`` for SCE (proportional execution backend)."""
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError("SCE proportional backend requires a continuous action-space env")
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)
    xy_gain, z_gain, yaw_gain = default_proportional_gains(
        low, high, xy_gain=xy_gain, z_gain=z_gain, yaw_gain=yaw_gain
    )

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError("Environment must be reset before selecting SCE actions.")
        backend = env.prev_backend_state
        lin_pos = np.asarray(backend.states[:, 3, :], dtype=np.float32)
        yaws = pursuer_yaws_from_backend(backend, env.task_state.pursuer_ids)
        return sce_proportional_actions_from_state(
            env,
            lin_pos,
            low,
            high,
            xy_gain=xy_gain,
            z_gain=z_gain,
            pursuer_yaw=yaws,
            yaw_gain=yaw_gain,
            yaw_align_min_speed=yaw_align_min_speed,
        )

    return get_actions
