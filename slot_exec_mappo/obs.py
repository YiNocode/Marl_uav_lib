"""Execution-layer observation and global state builders."""

from __future__ import annotations

from typing import Any

import numpy as np

from slot_exec_mappo.config import ExecObsConfig


def local_obs_dim(cfg: ExecObsConfig) -> int:
    base = 9 + 7 + 4 * int(cfg.obstacle_slots) + 4
    if cfg.include_prev_action:
        base += 4
    return base


def _normalize_position(task: Any, pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    world_xy = float(getattr(task, "world_xy", 20.0))
    z_min = float(getattr(task, "z_min", 0.0))
    z_max = float(getattr(task, "z_max", 5.0))
    z_center = 0.5 * (z_min + z_max)
    z_half = max(0.5 * (z_max - z_min), 1e-3)
    out = np.array(
        [pos[0] / world_xy, pos[1] / world_xy, (pos[2] - z_center) / z_half],
        dtype=np.float32,
    )
    return np.clip(out, -2.0, 2.0)


def _normalize_velocity(task: Any, vel: np.ndarray) -> np.ndarray:
    vel = np.asarray(vel, dtype=np.float32).reshape(3)
    vxy = max(float(getattr(task, "pursuer_speed_xy", 0.25)), 1e-6)
    vz = max(float(getattr(task, "pursuer_speed_z", 0.15)), 1e-6)
    out = np.array([vel[0] / vxy, vel[1] / vxy, vel[2] / vz], dtype=np.float32)
    return np.clip(out, -2.0, 2.0)


def _normalize_angle(ang: np.ndarray) -> np.ndarray:
    ang = np.asarray(ang, dtype=np.float32).reshape(3)
    return np.clip(ang / np.pi, -1.0, 1.0).astype(np.float32)


def _normalize_delta(task: Any, delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32).reshape(3)
    world_xy = float(getattr(task, "world_xy", 20.0))
    z_min = float(getattr(task, "z_min", 0.0))
    z_max = float(getattr(task, "z_max", 5.0))
    z_half = max(0.5 * (z_max - z_min), 1e-3)
    out = np.array([delta[0] / world_xy, delta[1] / world_xy, delta[2] / z_half], dtype=np.float32)
    return np.clip(out, -2.0, 2.0)


def _role_feature_block(task: Any, pursuer_pos: np.ndarray, assigned_target: np.ndarray) -> np.ndarray:
    rel_target = _normalize_delta(task, assigned_target - pursuer_pos)
    norm = float(np.linalg.norm(rel_target))
    if norm < 1e-6:
        slot_dir = np.zeros((3,), dtype=np.float32)
    else:
        slot_dir = (rel_target / norm).astype(np.float32)
    return np.concatenate([rel_target, slot_dir, np.array([1.0], dtype=np.float32)]).astype(np.float32)


def _boundary_block(task: Any, pos_xy: np.ndarray) -> np.ndarray:
    world_xy = float(getattr(task, "world_xy", 20.0))
    half = 0.5 * world_xy
    margin = float(getattr(task, "evader_margin_xy_ratio", 0.25)) * world_xy
    usable = max(half - margin, 1e-3)
    x, y = float(pos_xy[0]), float(pos_xy[1])
    return np.array(
        [
            (x + usable) / world_xy,
            (usable - x) / world_xy,
            (y + usable) / world_xy,
            (usable - y) / world_xy,
        ],
        dtype=np.float32,
    )


def _obstacle_block(task: Any, task_state: Any, pursuer_xy: np.ndarray, slots: int) -> np.ndarray:
    if hasattr(task, "_pursuer_obstacle_obs_block") and hasattr(task_state, "obstacle_xy"):
        obs_xy = getattr(task_state, "obstacle_xy", None)
        obs_r = getattr(task_state, "obstacle_r", None)
        if obs_xy is not None and obs_r is not None and np.asarray(obs_xy).size > 0:
            block = task._pursuer_obstacle_obs_block(pursuer_xy, obs_xy, obs_r)
            want = int(slots) * 4
            if block.size >= want:
                return np.asarray(block[:want], dtype=np.float32)
            out = np.zeros((want,), dtype=np.float32)
            out[: block.size] = block
            return out
    return np.zeros((int(slots) * 4,), dtype=np.float32)


def _global_obstacle_block(task: Any, task_state: Any, slots: int) -> np.ndarray:
    if hasattr(task, "_global_obstacle_state_block") and hasattr(task_state, "obstacle_xy"):
        obs_xy = getattr(task_state, "obstacle_xy", None)
        obs_r = getattr(task_state, "obstacle_r", None)
        if obs_xy is not None and obs_r is not None and np.asarray(obs_xy).size > 0:
            block = task._global_obstacle_state_block(obs_xy, obs_r)
            want = int(slots) * 4
            if block.size >= want:
                return np.asarray(block[:want], dtype=np.float32)
            out = np.zeros((want,), dtype=np.float32)
            out[: block.size] = block
            return out
    return np.zeros((int(slots) * 4,), dtype=np.float32)


def resolve_assigned_targets(task: Any, task_state: Any, backend_state: Any) -> np.ndarray:
    states = backend_state.states
    lin_pos = states[:, 3, :]
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64)
    pursuer_pos = lin_pos[pursuer_ids]
    evader_pos = lin_pos[int(task_state.evader_id)]
    if hasattr(task, "_assigned_targets_from_state"):
        _slots, _assignment, assigned = task._assigned_targets_from_state(
            pursuer_pos,
            evader_pos,
            task_state=task_state,
        )
        task_state.assigned_target_indices = np.asarray(_assignment, dtype=np.int64).copy()
        return np.asarray(assigned, dtype=np.float32).reshape(3, 3)
    return pursuer_pos.copy()


def build_local_obs(
    env: Any,
    *,
    cfg: ExecObsConfig,
    prev_actions: np.ndarray | None = None,
) -> np.ndarray:
    task = env.task
    task_state = env.task_state
    backend = env.prev_backend_state
    if backend is None or task_state is None:
        raise RuntimeError("Env must be reset before building execution observations.")

    states = backend.states
    lin_pos = states[:, 3, :]
    lin_vel = states[:, 2, :]
    ang_pos = states[:, 1, :]
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64)
    assigned = resolve_assigned_targets(task, task_state, backend)

    if prev_actions is None:
        prev_actions = np.zeros((len(pursuer_ids), 4), dtype=np.float32)
    prev_actions = np.asarray(prev_actions, dtype=np.float32).reshape(len(pursuer_ids), -1)

    obs_list: list[np.ndarray] = []
    for row, pid in enumerate(pursuer_ids):
        parts = [
            _normalize_position(task, lin_pos[int(pid)]),
            _normalize_velocity(task, lin_vel[int(pid)]),
            _normalize_angle(ang_pos[int(pid)]),
            _role_feature_block(task, lin_pos[int(pid)], assigned[row]),
            _obstacle_block(task, task_state, lin_pos[int(pid), :2], cfg.obstacle_slots),
            _boundary_block(task, lin_pos[int(pid), :2]),
        ]
        if cfg.include_prev_action:
            pa = prev_actions[row]
            if pa.size >= 4:
                parts.append(pa[:4].astype(np.float32))
            else:
                pad = np.zeros((4,), dtype=np.float32)
                pad[: pa.size] = pa
                parts.append(pad)
        obs_list.append(np.concatenate(parts, axis=0).astype(np.float32))
    return np.stack(obs_list, axis=0)


def global_state_dim(cfg: ExecObsConfig) -> int:
    # 3 pursuers × (pos3 + vel3 + yaw1)
    return 21 + 9 + 4 * int(cfg.obstacle_slots) + 3


def build_global_state(
    env: Any,
    *,
    cfg: ExecObsConfig,
    slot_dists: np.ndarray | None = None,
    min_clearance: float | None = None,
) -> np.ndarray:
    task = env.task
    task_state = env.task_state
    backend = env.prev_backend_state
    if backend is None or task_state is None:
        raise RuntimeError("Env must be reset before building execution state.")

    states = backend.states
    lin_pos = states[:, 3, :]
    lin_vel = states[:, 2, :]
    ang_pos = states[:, 1, :]
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64)
    assigned = resolve_assigned_targets(task, task_state, backend)

    pursuer_parts: list[np.ndarray] = []
    for pid in pursuer_ids:
        yaw = float(ang_pos[int(pid), 2]) if ang_pos.shape[1] >= 3 else 0.0
        pursuer_parts.append(
            np.concatenate(
                [
                    _normalize_position(task, lin_pos[int(pid)]),
                    _normalize_velocity(task, lin_vel[int(pid)]),
                    np.array([yaw / np.pi], dtype=np.float32),
                ]
            )
        )
    pursuer_flat = np.concatenate(pursuer_parts, axis=0)

    slot_flat = _normalize_position(task, assigned[0])
    for i in range(1, 3):
        slot_flat = np.concatenate([slot_flat, _normalize_position(task, assigned[i])], axis=0)

    obs_global = _global_obstacle_block(task, task_state, cfg.obstacle_slots)

    if slot_dists is None:
        dists = np.linalg.norm(assigned[:, :2] - lin_pos[pursuer_ids][:, :2], axis=1)
        mean_slot_dist = float(np.mean(dists))
    else:
        mean_slot_dist = float(np.mean(np.asarray(slot_dists, dtype=np.float64)))

    if min_clearance is None:
        min_clearance = 0.0

    ep_limit = max(int(getattr(task, "episode_limit", 400)), 1)
    step_frac = float(getattr(env, "step_count", 0)) / float(ep_limit)
    scalars = np.array(
        [
            mean_slot_dist / float(getattr(task, "world_xy", 20.0)),
            float(min_clearance) / float(getattr(task, "world_xy", 20.0)),
            np.clip(step_frac, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([pursuer_flat, slot_flat, obs_global, scalars], axis=0).astype(np.float32)
