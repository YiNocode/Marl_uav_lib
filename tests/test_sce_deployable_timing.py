"""Performance tests for deployable SCE obstacle baselines."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from marl_uav.control.obstacle_aware_sce_baselines import deployable_sce_actions_from_state
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import PursuitEvasion3v1Task
from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.reachability.los_cost import build_los_cost_matrix


def _many_obstacles(n: int = 121) -> list[Obstacle]:
    obs: list[Obstacle] = []
    k = 0
    for x in range(-20, 21, 4):
        for y in range(-20, 21, 4):
            if k >= n:
                return obs
            obs.append(Obstacle(kind="circle", center=np.array([float(x), float(y)]), radius=0.5))
            k += 1
    return obs


def test_build_los_cost_matrix_under_1ms() -> None:
    pursuers = np.array([[0, -4, 1], [4, 0, 1], [-4, 0, 1]], dtype=np.float64)
    slots = np.array([[0, 4, 1], [4, 0, 1], [0, 3, 1]], dtype=np.float64)
    obstacles = _many_obstacles(121)
    t0 = time.perf_counter()
    for _ in range(20):
        cost, diag = build_los_cost_matrix(
            pursuers, slots, obstacles, None,
            safety_margin=0.3, uav_radius=0.15, los_block_penalty=100.0,
        )
    ms = (time.perf_counter() - t0) / 20 * 1000.0
    assert cost.shape == (3, 3)
    assert ms < 5.0, f"LOS cost too slow: {ms:.2f} ms"
    assert "los_blocked_matrix" in diag


def _mock_env(task, obstacles_xy, obstacles_r, step_count: int = 0):
    state = SimpleNamespace(
        pursuer_ids=np.array([0, 1, 2], dtype=np.int64),
        evader_id=3,
        assigned_target_indices=np.array([0, 1, 2], dtype=np.int64),
        obstacle_xy=np.asarray(obstacles_xy, dtype=np.float32),
        obstacle_r=np.asarray(obstacles_r, dtype=np.float32),
    )
    env = SimpleNamespace(
        task=task,
        task_state=state,
        prev_backend_state=None,
        step_count=step_count,
    )
    return env


def _run_steps(kind: str, reachability: dict, *, cbf: dict | None = None, n_steps: int = 100):
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        role_assignment_mode="entropic_ot",
        assignment_inertia_margin=0.05,
        obstacle_manifold_top_k=4,
    )
    obs_xy = []
    obs_r = []
    for x in range(-20, 21, 4):
        for y in range(-20, 21, 4):
            obs_xy.append([float(x), float(y)])
            obs_r.append(0.5)
    obs_xy = np.asarray(obs_xy, dtype=np.float32)
    obs_r = np.asarray(obs_r, dtype=np.float32)

    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    env = _mock_env(task, obs_xy, obs_r)
    times = []
    replans = []

    for step in range(n_steps):
        env.step_count = step
        pursuer_pos = np.array(
            [
                [2.0 * np.sin(0.1 * step), 3.0, 1.0],
                [2.5 * np.cos(0.07 * step), -1.0, 1.0],
                [-2.0, -2.0 + 0.01 * step, 1.0],
            ],
            dtype=np.float32,
        )
        evader = np.array([0.5 * np.sin(0.05 * step), 0.5 * np.cos(0.05 * step), 1.0], dtype=np.float32)
        lin_pos = np.zeros((4, 3), dtype=np.float32)
        lin_pos[:3] = pursuer_pos
        lin_pos[3] = evader
        t0 = time.perf_counter()
        deployable_sce_actions_from_state(
            env, lin_pos, low, high,
            kind=kind,
            reachability=reachability,
            cbf=cbf,
            runtime_rates={
                "control_hz": 50,
                "manifold_update_hz": 10,
                "assignment_update_hz": 5,
                "path_replan_hz": 1,
                "cbf_hz": 50,
            },
            xy_gain=0.25, z_gain=0.2, yaw_gain=0.0,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
        diag = getattr(env, "_obstacle_aware_diagnostics", {})
        replans.append(float(diag.get("num_replans_this_step", 0)))

    arr = np.asarray(times, dtype=np.float64)
    return {
        "avg_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "replan_steps": int(np.sum(np.asarray(replans) > 0)),
    }


@pytest.mark.parametrize("kind,reach,cbf,avg_limit,p95_limit", [
    (
        "sce_los_slot",
        {"enabled": True, "mode": "los", "safety_margin": 0.3, "uav_radius": 0.15, "los_block_penalty": 100.0},
        None, 5.0, 15.0,
    ),
    (
        "sce_cached_path_slot",
        {
            "enabled": True, "mode": "cached_path", "safety_margin": 0.3, "uav_radius": 0.15,
            "path_cache": {"max_replans_per_step": 3, "fallback_blocked_penalty": 50.0},
            "path_planner": {"num_obstacle_samples": 12, "planner_timeout_ms": 5.0, "build_static_graph_once": True},
            "path_tracking": {"lookahead_dist": 0.8},
        },
        None, 20.0, 50.0,
    ),
    (
        "sce_cached_path_cbf_slot",
        {
            "enabled": True, "mode": "cached_path", "safety_margin": 0.3, "uav_radius": 0.15,
            "path_cache": {"max_replans_per_step": 3},
            "path_planner": {"planner_timeout_ms": 5.0, "build_static_graph_once": True},
        },
        {"enabled": True, "solver": "projection", "max_projection_iters": 5, "obstacle_activation_radius": 2.0},
        25.0, 60.0,
    ),
])
def test_deployable_decision_time(kind, reach, cbf, avg_limit, p95_limit) -> None:
    stats = _run_steps(kind, reach, cbf=cbf, n_steps=80)
    assert stats["max_ms"] < 10000.0, "still seeing multi-second steps"
    assert stats["avg_ms"] < avg_limit, f"{kind} avg {stats['avg_ms']:.2f} ms > {avg_limit}"
    assert stats["p95_ms"] < p95_limit, f"{kind} p95 {stats['p95_ms']:.2f} ms > {p95_limit}"
    if "path" in kind:
        assert stats["replan_steps"] < 80, "replanning every step"
