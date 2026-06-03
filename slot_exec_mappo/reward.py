"""Execution-layer reward for slot navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from slot_exec_mappo.config import ExecRewardConfig
from slot_exec_mappo.obs import resolve_assigned_targets


@dataclass
class ExecRewardState:
    prev_slot_dists: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    prev_actions: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), dtype=np.float32))
    arrived: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=bool))

    def reset(self) -> None:
        self.prev_slot_dists = np.zeros(3, dtype=np.float32)
        self.prev_actions = np.zeros((3, 4), dtype=np.float32)
        self.arrived = np.zeros(3, dtype=bool)


def _pursuer_oob_mask(task: Any, pursuer_pos: np.ndarray) -> np.ndarray:
    if hasattr(task, "_get_oob_mask"):
        return np.asarray(task._get_oob_mask(pursuer_pos), dtype=bool).reshape(-1)
    return np.zeros((pursuer_pos.shape[0],), dtype=bool)


def _obstacle_hit_mask(task: Any, task_state: Any, pursuer_pos: np.ndarray) -> np.ndarray:
    if hasattr(task, "_pursuer_obstacle_collision_mask"):
        return np.asarray(task._pursuer_obstacle_collision_mask(pursuer_pos, task_state), dtype=bool)
    return np.zeros((pursuer_pos.shape[0],), dtype=bool)


def _min_clearance(task: Any, task_state: Any, pursuer_xy: np.ndarray) -> float:
    if not hasattr(task_state, "obstacle_xy"):
        return float("inf")
    obs_xy = np.asarray(task_state.obstacle_xy, dtype=np.float64).reshape(-1, 2)
    obs_r = np.asarray(task_state.obstacle_r, dtype=np.float64).reshape(-1)
    if obs_xy.size == 0:
        return float("inf")
    if hasattr(task, "_pursuer_obstacle_hit_radius"):
        hit_r = float(task._pursuer_obstacle_hit_radius())
    else:
        hit_r = 0.15
    d = np.linalg.norm(obs_xy - pursuer_xy.reshape(1, 2), axis=1)
    surface = d - obs_r - hit_r
    return float(np.min(surface))


def compute_exec_rewards(
    env: Any,
    actions: np.ndarray,
    *,
    cfg: ExecRewardConfig,
    rw_state: ExecRewardState,
    step_info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    task = env.task
    task_state = env.task_state
    backend = env.prev_backend_state
    if backend is None or task_state is None:
        raise RuntimeError("Env must be stepped before computing execution rewards.")

    lin_pos = backend.states[:, 3, :]
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64)
    pursuer_pos = lin_pos[pursuer_ids]
    assigned = resolve_assigned_targets(task, task_state, backend)

    dists = np.linalg.norm(assigned[:, :2] - pursuer_pos[:, :2], axis=1).astype(np.float32)
    prev = np.asarray(rw_state.prev_slot_dists, dtype=np.float32).reshape(3)
    if float(np.sum(prev)) <= 1e-9:
        prev = dists.copy()

    progress_norm = max(float(cfg.progress_dist_norm), 1e-6)
    delta = np.clip((prev - dists) / progress_norm, -1.0, 1.0)

    acts = np.asarray(actions, dtype=np.float32).reshape(3, -1)
    if acts.shape[1] < 4:
        pad = np.zeros((3, 4), dtype=np.float32)
        pad[:, : acts.shape[1]] = acts
        acts = pad

    oob_mask = _pursuer_oob_mask(task, pursuer_pos)
    hit_mask = _obstacle_hit_mask(task, task_state, pursuer_pos)
    any_hit = bool(np.any(hit_mask))

    rewards = np.zeros((3,), dtype=np.float32)
    diag: dict[str, Any] = {
        "slot_dist_xy": dists.astype(float).tolist(),
        "slot_progress": delta.astype(float).tolist(),
        "obstacle_hit": hit_mask.astype(bool).tolist(),
        "oob": oob_mask.astype(bool).tolist(),
    }

    for i in range(3):
        clearance = _min_clearance(task, task_state, pursuer_pos[i, :2])
        r = float(cfg.w_progress) * float(delta[i])
        r += float(cfg.w_alive)
        r -= float(cfg.w_time)
        margin = float(cfg.clearance_margin)
        r += float(cfg.w_clearance) * float(np.log1p(max(clearance - margin, 0.0)))
        if dists[i] < float(cfg.arrive_dist) and not bool(rw_state.arrived[i]):
            r += float(cfg.w_arrive)
            rw_state.arrived[i] = True
        if hit_mask[i]:
            r -= float(cfg.w_collision)
        if oob_mask[i]:
            r -= float(cfg.w_oob)
        diff = acts[i, :4] - rw_state.prev_actions[i, :4]
        r -= float(cfg.w_smooth) * float(np.dot(diff, diff))
        if any_hit:
            r -= float(cfg.w_team_collision)
        rewards[i] = np.float32(r)

    rw_state.prev_slot_dists = dists.copy()
    rw_state.prev_actions = acts[:, :4].copy()
    diag["mean_slot_dist"] = float(np.mean(dists))
    diag["min_clearance"] = float(min(_min_clearance(task, task_state, pursuer_pos[j, :2]) for j in range(3)))
    if step_info is not None:
        diag["termination_reason"] = step_info.get("termination_reason", "running")
    return rewards, diag
