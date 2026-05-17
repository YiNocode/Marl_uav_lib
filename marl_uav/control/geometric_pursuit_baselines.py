"""Geometric non-learning baselines for 3v1 pursuit."""

from __future__ import annotations

from typing import Any

import numpy as np


def _empty_actions(action_low: np.ndarray) -> np.ndarray:
    adim = int(np.asarray(action_low).reshape(-1).shape[0])
    return np.zeros((3, adim), dtype=np.float32)


def _clip_actions(actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    return np.clip(np.asarray(actions, dtype=np.float32), low[None, :], high[None, :]).astype(np.float32)


def proportional_actions_to_targets(
    pursuer_pos: np.ndarray,
    target_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
) -> np.ndarray:
    """Map pursuer-to-target position error to [vx, vy, yaw_rate, vz] actions."""
    p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    g = np.asarray(target_pos, dtype=np.float32).reshape(3, 3)
    out = _empty_actions(action_low)
    err = g - p
    out[:, 0] = float(xy_gain) * err[:, 0]
    out[:, 1] = float(xy_gain) * err[:, 1]
    if out.shape[1] >= 3:
        out[:, 2] = 0.0
    if out.shape[1] >= 4:
        out[:, 3] = float(z_gain) * err[:, 2]
    return _clip_actions(out, action_low, action_high)


def pure_pursuit_actions_from_state(
    lin_pos: np.ndarray,
    pursuer_ids: np.ndarray,
    evader_id: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
) -> np.ndarray:
    """Pure pursuit: each pursuer directly moves toward the evader."""
    pos = np.asarray(lin_pos, dtype=np.float32)
    pids = np.asarray(pursuer_ids, dtype=np.int64).reshape(3)
    evader = pos[int(evader_id)]
    targets = np.repeat(evader[None, :], 3, axis=0)
    return proportional_actions_to_targets(
        pos[pids],
        targets,
        action_low,
        action_high,
        xy_gain=xy_gain,
        z_gain=z_gain,
    )


def oracle_slot_actions_from_state(
    env: Any,
    lin_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
) -> np.ndarray:
    """Oracle slot controller: move directly toward the task's assigned slot targets."""
    pos = np.asarray(lin_pos, dtype=np.float32)
    task_state = env.task_state
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
    pursuer_pos = pos[pursuer_ids]
    evader_pos = pos[int(task_state.evader_id)]

    if not hasattr(env.task, "_assigned_targets_from_state"):
        raise TypeError("Oracle slot baseline requires a pursuit task with slot targets.")

    _, assignment, assigned_targets = env.task._assigned_targets_from_state(
        pursuer_pos,
        evader_pos,
        task_state=task_state,
    )
    task_state.assigned_target_indices = np.asarray(assignment, dtype=np.int64).copy()
    return proportional_actions_to_targets(
        pursuer_pos,
        assigned_targets,
        action_low,
        action_high,
        xy_gain=xy_gain,
        z_gain=z_gain,
    )


def make_pure_pursuit_get_actions_fn(
    env: Any,
    *,
    xy_gain: float = 0.25,
    z_gain: float = 0.20,
):
    """Build a RolloutWorker ``get_actions_fn`` for pure pursuit."""
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError("pure-pursuit baseline requires a continuous action-space env")
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError("Environment must be reset before selecting pure-pursuit actions.")
        lin_pos = np.asarray(env.prev_backend_state.states[:, 3, :], dtype=np.float32)
        return pure_pursuit_actions_from_state(
            lin_pos,
            env.task_state.pursuer_ids,
            env.task_state.evader_id,
            low,
            high,
            xy_gain=xy_gain,
            z_gain=z_gain,
        )

    return get_actions


def make_oracle_slot_get_actions_fn(
    env: Any,
    *,
    xy_gain: float = 0.25,
    z_gain: float = 0.20,
):
    """Build a RolloutWorker ``get_actions_fn`` for the oracle slot controller."""
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError("oracle-slot baseline requires a continuous action-space env")
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError("Environment must be reset before selecting oracle-slot actions.")
        lin_pos = np.asarray(env.prev_backend_state.states[:, 3, :], dtype=np.float32)
        return oracle_slot_actions_from_state(
            env,
            lin_pos,
            low,
            high,
            xy_gain=xy_gain,
            z_gain=z_gain,
        )

    return get_actions

