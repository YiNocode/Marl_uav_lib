"""Obstacle-aware APF-style pursuit baselines."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np

from marl_uav.control.altitude_hold import apply_hard_altitude_to_action_row
from marl_uav.control.boundary_utils import apply_xy_boundary_barrier
from marl_uav.control.geometric_pursuit_baselines import pursuer_yaws_from_backend
from marl_uav.control.obstacle_avoidance_controller import ObstacleAvoidanceController
from marl_uav.framework.geometry.obstacle_adapter import obstacles_from_task_state
from marl_uav.utils.control_timing import publish_control_timing, should_record_control_timing


def _ring_targets(evader_pos: np.ndarray, *, radius: float, phase: float) -> np.ndarray:
    e = np.asarray(evader_pos, dtype=np.float32).reshape(3)
    theta = float(phase) + np.arange(3, dtype=np.float32) * np.float32(2.0 * np.pi / 3.0)
    targets = np.zeros((3, 3), dtype=np.float32)
    targets[:, 0] = e[0] + float(radius) * np.cos(theta)
    targets[:, 1] = e[1] + float(radius) * np.sin(theta)
    targets[:, 2] = e[2]
    return targets


def _targets_for_kind(
    kind: Literal["pure_pursuit_apf", "fixed_ring_apf"],
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    *,
    ring_radius: float,
    phase: float,
    assignment: str,
) -> np.ndarray:
    if kind == "pure_pursuit_apf":
        return np.repeat(np.asarray(evader_pos, dtype=np.float32).reshape(1, 3), 3, axis=0)

    targets = _ring_targets(evader_pos, radius=ring_radius, phase=phase)
    if assignment == "fixed":
        return targets
    if assignment == "angle_order":
        rel = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)[:, :2] - np.asarray(
            evader_pos, dtype=np.float32
        ).reshape(1, 3)[:, :2]
        order = np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))
        assigned = np.zeros_like(targets)
        assigned[order] = targets
        return assigned
    raise ValueError(f"Unsupported fixed-ring APF assignment={assignment!r}")


def _make_apf_fn(
    env: Any,
    *,
    kind: Literal["pure_pursuit_apf", "fixed_ring_apf"],
    ring_radius: float = 1.6,
    phase: float = 0.0,
    assignment: str = "fixed",
    obstacle_avoidance: dict[str, Any] | None = None,
    boundary_margin: float = 0.30,
    boundary_alpha: float = 3.0,
    altitude_floor_margin: float = 0.25,
    altitude_ceiling_margin: float = 0.10,
):
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError(f"{kind} baseline requires a continuous action-space env")
    low = np.asarray(env.action_low_np, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_high_np, dtype=np.float32).reshape(-1)
    ctrl = ObstacleAvoidanceController(obstacle_avoidance)

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError(f"Environment must be reset before selecting {kind} actions.")

        record = should_record_control_timing(env)
        t0 = time.perf_counter() if record else None
        backend = env.prev_backend_state
        task = env.task
        task_state = env.task_state
        lin_pos = np.asarray(backend.states[:, 3, :], dtype=np.float32)
        lin_vel = np.asarray(backend.states[:, 2, :], dtype=np.float32)
        pids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
        pursuer_pos = lin_pos[pids]
        evader_pos = lin_pos[int(task_state.evader_id)]
        yaws = pursuer_yaws_from_backend(backend, pids)
        obstacles = obstacles_from_task_state(task_state, task=task)
        world_xy = float(getattr(task, "world_xy", 0.0))
        bounds = None
        if np.isfinite(world_xy) and world_xy > 0:
            half = max(world_xy - float(boundary_margin), 0.0)
            bounds = (-half, half, -half, half)

        targets = _targets_for_kind(
            kind,
            pursuer_pos,
            evader_pos,
            ring_radius=ring_radius,
            phase=phase,
            assignment=assignment,
        )
        actions = np.zeros((3, low.shape[0]), dtype=np.float32)
        for i in range(3):
            u_xy, yaw_rate, _path, _diag = ctrl.compute_action(
                pursuer_pos[i, :2],
                float(yaws[i]),
                targets[i, :2],
                obstacles,
                current_velocity_xy=lin_vel[pids[i], :2],
                bounds_xy=bounds,
            )
            safe_xy, _active = apply_xy_boundary_barrier(
                pursuer_pos[i, :2],
                u_xy,
                world_xy=world_xy,
                boundary_margin=boundary_margin,
                boundary_alpha=boundary_alpha,
                action_low_xy=low[:2],
                action_high_xy=high[:2],
            )
            actions[i, 0] = np.float32(safe_xy[0])
            actions[i, 1] = np.float32(safe_xy[1])
            if actions.shape[1] >= 3:
                actions[i, 2] = np.float32(yaw_rate)
            if actions.shape[1] >= 4:
                apply_hard_altitude_to_action_row(
                    actions[i],
                    float(pursuer_pos[i, 2]),
                    float(targets[i, 2]),
                    low,
                    high,
                    z_floor=float(getattr(task, "z_min", 0.0)),
                    z_ceiling=float(getattr(task, "z_max", 5.0)),
                    floor_margin=float(altitude_floor_margin),
                    ceiling_margin=float(altitude_ceiling_margin),
                )

        if record and t0 is not None:
            publish_control_timing(env, total_decision_latency=time.perf_counter() - t0)
        return np.clip(actions, low[None, :], high[None, :]).astype(np.float32)

    return get_actions


def make_pure_pursuit_apf_get_actions_fn(env: Any, **kwargs: Any):
    return _make_apf_fn(env, kind="pure_pursuit_apf", **kwargs)


def make_fixed_ring_apf_get_actions_fn(env: Any, **kwargs: Any):
    return _make_apf_fn(env, kind="fixed_ring_apf", **kwargs)
