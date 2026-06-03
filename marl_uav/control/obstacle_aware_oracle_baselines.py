"""Offline upper-bound SCE baselines — NOT real-time deployable.

These methods re-plan paths for all pursuer-slot pairs each assignment step.
Use only for research upper bounds; do NOT use in E2 real-time ablation tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from marl_uav.control.geometric_pursuit_baselines import (
    default_proportional_gains,
    proportional_actions_to_targets,
    pursuer_yaws_from_backend,
)
from marl_uav.framework.geometry.obstacle_adapter import obstacles_from_task_state
from marl_uav.framework.planning.visibility_path_planner import plan_path
from marl_uav.framework.reachability.los_cost import build_los_cost_matrix
from marl_uav.framework.role_allocation import default_ot_epsilon, entropic_ot_assignment
from marl_uav.framework.geometry.obstacle_geometry import path_length


def _exact_path_cost_matrix(pursuer_pos, targets, obstacles, prev, rcfg, bounds):
    """Full pairwise path planning — offline oracle only."""
    import time

    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(-1, 3)[:, :2]
    s = np.asarray(targets, dtype=np.float64).reshape(-1, 3)[:, :2]
    n = int(p.shape[0])
    cost = np.zeros((n, n), dtype=np.float64)
    planner_ms = 0.0
    los_blocked = np.zeros((n, n), dtype=bool)

    _, los_diag = build_los_cost_matrix(
        pursuer_pos, targets, obstacles, prev,
        safety_margin=rcfg.get("safety_margin", 0.3),
        uav_radius=rcfg.get("uav_radius", 0.15),
        los_block_penalty=0.0,
    )
    los_blocked = los_diag["los_blocked_matrix"]

    for i in range(n):
        for j in range(n):
            if not los_blocked[i, j]:
                cost[i, j] = float(np.linalg.norm(s[j] - p[i]))
                continue
            t0 = time.perf_counter()
            path = plan_path(
                p[i], s[j], obstacles, bounds=bounds,
                cfg=rcfg.get("path_planner", {}),
                safety_margin=float(rcfg.get("safety_margin", 0.3)),
                uav_radius=float(rcfg.get("uav_radius", 0.15)),
            )
            planner_ms += (time.perf_counter() - t0) * 1000.0
            cost[i, j] = path_length(path) if path is not None else 1e9

    return cost, {"path_planner_time_ms": planner_ms, "offline_oracle": True}


def oracle_exact_path_actions_from_state(
    env, lin_pos, action_low, action_high, **kwargs
) -> np.ndarray:
    """Offline: exact path cost for all pairs + slot tracking."""
    reach = dict(kwargs.pop("reachability", {}) or {})
    xy_gain = float(kwargs.get("xy_gain", 0.25))
    z_gain = float(kwargs.get("z_gain", 0.20))
    yaw_gain = float(kwargs.get("yaw_gain", 0.25))

    pos = np.asarray(lin_pos, dtype=np.float32)
    task = env.task
    ts = env.task_state
    pursuer_ids = np.asarray(ts.pursuer_ids, dtype=np.int64).reshape(3)
    pursuer_pos = pos[pursuer_ids]
    evader_pos = pos[int(ts.evader_id)]
    w = float(getattr(task, "world_xy", 20.0))
    bounds = (-w, -w, w, w)
    obstacles = obstacles_from_task_state(ts, task=task)
    targets = task._reference_manifold_targets(pursuer_pos, evader_pos, task_state=ts)
    prev = getattr(ts, "assigned_target_indices", None)
    cost, _ = _exact_path_cost_matrix(pursuer_pos, targets, obstacles, prev, reach, bounds)
    eps = default_ot_epsilon(cost, task.ot_epsilon, task.ot_epsilon_scale)
    assignment = entropic_ot_assignment(
        cost, epsilon=eps, num_iters=task.ot_sinkhorn_iterations,
        prev_assignment=prev, inertia_margin=task.assignment_inertia_margin,
    )
    ts.assigned_target_indices = assignment.copy()
    yaws = kwargs.get("pursuer_yaw")
    return proportional_actions_to_targets(
        pursuer_pos, targets[assignment], action_low, action_high,
        xy_gain=xy_gain, z_gain=z_gain, pursuer_yaw=yaws, yaw_gain=yaw_gain,
    )


def make_sce_exact_path_oracle_get_actions_fn(env: Any, **kwargs: Any):
    """Offline upper-bound only, not deployable."""
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)
    xy, zg, yg = default_proportional_gains(
        low, high, xy_gain=kwargs.get("xy_gain"), z_gain=kwargs.get("z_gain"), yaw_gain=kwargs.get("yaw_gain"),
    )
    reach = dict(kwargs.get("reachability") or {})

    def get_actions(obs_list, state, avail_actions):
        del obs_list, state, avail_actions
        backend = env.prev_backend_state
        lin_pos = np.asarray(backend.states[:, 3, :], dtype=np.float32)
        yaws = pursuer_yaws_from_backend(backend, env.task_state.pursuer_ids)
        return oracle_exact_path_actions_from_state(
            env, lin_pos, low, high,
            reachability=reach, xy_gain=xy, z_gain=zg, yaw_gain=yg, pursuer_yaw=yaws, **kwargs,
        )

    return get_actions
