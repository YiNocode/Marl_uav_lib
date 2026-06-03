"""Turn-radius local slot planner behavior tests."""

from __future__ import annotations

import time

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.planning.turn_radius_obstacle_query import query_turn_radius_obstacles
from marl_uav.framework.planning.turn_radius_slot_planner import TurnRadiusSlotPlanner


def _obs(x: float, y: float, r: float) -> Obstacle:
    return Obstacle(kind="circle", center=np.array([x, y], dtype=np.float64), radius=float(r))


def _cfg(**overrides):
    cfg = {
        "horizon_s": 1.5,
        "dt": 0.1,
        "vmax": 0.25,
        "omega_max": 1.0,
        "num_yaw_samples": 11,
        "speed_samples": [1.0, 0.75, 0.5, 0.25],
        "min_turn_radius": 0.25,
        "lookahead_dist": 1.5,
        "uav_radius": 0.10,
        "safety_margin": 0.10,
        "query_extra_margin": 0.20,
        "collision_large_penalty": 1_000_000.0,
        "w_goal": 1.0,
        "w_heading": 0.2,
        "w_obstacle": 2.0,
        "w_smooth": 0.1,
        "w_speed": -0.05,
        "fallback_zero_if_blocked": True,
    }
    cfg.update(overrides)
    return cfg


def test_no_obstacle_moves_straight_at_high_speed() -> None:
    planner = TurnRadiusSlotPlanner(_cfg())

    action, diag = planner.compute_action(
        np.array([0.0, 0.0]),
        0.0,
        np.zeros(2),
        np.array([2.0, 0.0]),
        [],
    )

    assert diag["local_obstacle_count"] == 0
    assert diag["valid_candidate_count"] == diag["candidate_count"]
    assert diag["best_candidate_speed"] == 0.25
    assert abs(float(diag["best_candidate_yaw_rate"])) < 1e-6
    np.testing.assert_allclose(action, np.array([0.25, 0.0]), atol=1e-6)


def test_obstacle_outside_turn_radius_corridor_is_not_selected() -> None:
    cfg = _cfg()
    obstacle = _obs(0.2, 1.2, 0.10)

    local = query_turn_radius_obstacles(
        np.array([0.0, 0.0]),
        0.0,
        np.array([2.0, 0.0]),
        [obstacle],
        cfg,
    )
    planner = TurnRadiusSlotPlanner(cfg)
    action, diag = planner.compute_action(
        np.array([0.0, 0.0]),
        0.0,
        np.zeros(2),
        np.array([2.0, 0.0]),
        [obstacle],
    )

    assert local == []
    assert diag["local_obstacle_count"] == 0
    assert abs(float(diag["best_candidate_yaw_rate"])) < 1e-6
    assert action[0] > 0.20


def test_obstacle_on_direct_path_selects_turn_candidate() -> None:
    cfg = _cfg(horizon_s=2.0, safety_margin=0.10, uav_radius=0.10, w_obstacle=4.0)
    planner = TurnRadiusSlotPlanner(cfg)

    action, diag = planner.compute_action(
        np.array([0.0, 0.0]),
        0.0,
        np.zeros(2),
        np.array([2.0, 0.0]),
        [_obs(0.5, 0.0, 0.05)],
    )

    assert diag["local_obstacle_count"] == 1
    assert diag["valid_candidate_count"] > 0
    assert abs(float(diag["best_candidate_yaw_rate"])) > 0.1
    assert action[0] > 0.0


def test_all_candidates_blocked_falls_back_without_crashing() -> None:
    planner = TurnRadiusSlotPlanner(_cfg())
    obstacles = [
        _obs(0.20, 0.0, 0.25),
        _obs(-0.20, 0.0, 0.25),
        _obs(0.0, 0.20, 0.25),
        _obs(0.0, -0.20, 0.25),
    ]

    action, diag = planner.compute_action(
        np.array([0.0, 0.0]),
        0.0,
        np.zeros(2),
        np.array([2.0, 0.0]),
        obstacles,
    )

    assert diag["local_planner_blocked"] is True
    np.testing.assert_allclose(action, np.zeros(2), atol=1e-9)


def test_runtime_under_e2_like_121_cylinders() -> None:
    cfg = _cfg(safety_margin=0.30, uav_radius=0.15)
    planner = TurnRadiusSlotPlanner(cfg)
    xs = np.linspace(-20.0, 20.0, 11)
    obstacles = [_obs(float(x), float(y), 0.20) for x in xs for y in xs]
    states = [
        (np.array([1.0, 1.0]), 0.0, np.array([3.0, 1.0])),
        (np.array([1.0, -1.0]), 0.2, np.array([3.0, -0.8])),
        (np.array([-1.0, 1.0]), -0.2, np.array([1.5, 1.1])),
    ]

    totals_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        for pos, yaw, slot in states:
            planner.compute_action(pos, yaw, np.zeros(2), slot, obstacles)
        totals_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(totals_ms))
    p95_ms = float(np.percentile(totals_ms, 95))
    assert avg_ms < 5.0, f"avg 3-UAV local planner time {avg_ms:.3f} ms"
    assert p95_ms < 10.0, f"p95 3-UAV local planner time {p95_ms:.3f} ms"
