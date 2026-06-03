"""Tests for obstacle-aware SCE deployable baselines."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from marl_uav.control.geometric_pursuit_baselines import ot_slot_actions_from_state
from marl_uav.control.obstacle_aware_sce_baselines import (
    _apply_control_priority_layers,
    _apply_safety_speed_constraints,
    _apply_turn_slowdown_actions,
    _resolve_cbf_config,
)
from marl_uav.framework.geometry.obstacle_geometry import (
    Obstacle,
    collision_check_path,
    has_line_of_sight,
    line_segment_intersects_obstacle,
    path_length,
)
from marl_uav.framework.planning.path_tracking import select_lookahead_waypoint
from marl_uav.framework.planning.visibility_path_planner import plan_path
from marl_uav.framework.reachability.assignment_cost import ReachabilityConfig
from marl_uav.framework.reachability.los_cost import build_los_cost_matrix
from marl_uav.framework.reachability.reachability_slot_selection import StructureSelectionConfig
from marl_uav.framework.safety.cbf_filter import CBFConfig, LightweightCBFFilter, apply_cbf_filter


def _circle_obs(cx: float, cy: float, r: float) -> Obstacle:
    return Obstacle(kind="circle", center=np.array([cx, cy], dtype=np.float64), radius=r)


def test_los_cost_matrix_blocked_pair() -> None:
    """Test 1: p2->s1 segment through obstacle increases cost."""
    pursuers = np.array(
        [
            [-5.0, 0.0, 1.0],
            [0.0, -4.0, 1.0],
            [5.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    slots = np.array(
        [
            [0.0, 4.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 1.0],
        ],
        dtype=np.float64,
    )
    obstacles = [_circle_obs(0.0, 0.0, 1.0)]
    cost, diag = build_los_cost_matrix(
        pursuers, slots, obstacles, None,
        safety_margin=0.1, uav_radius=0.1, los_block_penalty=50.0,
    )
    euclid = np.linalg.norm(pursuers[:, None, :2] - slots[None, :, :2], axis=2)
    assert diag["los_blocked_matrix"][1, 0] is True or diag["los_blocked_matrix"][1, 0] == True
    assert cost[1, 0] > euclid[1, 0]


def test_nearest_obstacles_top_k() -> None:
    from marl_uav.framework.geometry.obstacle_adapter import nearest_obstacles

    obs = [
        _circle_obs(0.0, 0.0, 1.0),
        _circle_obs(10.0, 0.0, 1.0),
        _circle_obs(0.0, 10.0, 1.0),
    ]
    near = nearest_obstacles(np.array([0.5, 0.0]), obs, k=2)
    assert len(near) == 2
    centers = {tuple(np.round(o.center, 1)) for o in near}
    assert (0.0, 0.0) in centers or np.linalg.norm(near[0].center) < 2.0


def test_geometry_los_clear_vs_blocked() -> None:
    obs = _circle_obs(0.0, 0.0, 1.0)
    inflated = Obstacle(kind="circle", center=np.array([0.0, 0.0]), radius=1.3)
    assert line_segment_intersects_obstacle(
        np.array([-2.0, 0.0]), np.array([2.0, 0.0]), inflated
    )
    assert has_line_of_sight(
        np.array([-3.0, 3.0]),
        np.array([3.0, 3.0]),
        [obs],
        safety_margin=0.0,
        uav_radius=0.0,
    )
    assert not has_line_of_sight(
        np.array([-1.0, 0.0]),
        np.array([1.0, 0.0]),
        [obs],
        safety_margin=0.2,
        uav_radius=0.1,
    )
    # tangent treated as blocked
    p0 = np.array([1.3, 0.0])
    p1 = np.array([1.3, 2.0])
    assert line_segment_intersects_obstacle(p0, p1, inflated, treat_tangent_as_blocked=True)


def test_path_planner_around_obstacle() -> None:
    """Test 2: plan_path detours with >=3 points."""
    start = np.array([-3.0, 0.0])
    goal = np.array([3.0, 0.0])
    obstacles = [_circle_obs(0.0, 0.0, 1.0)]
    path = plan_path(
        start,
        goal,
        obstacles,
        bounds=(-10.0, -10.0, 10.0, 10.0),
        cfg={"num_obstacle_samples": 24, "clearance": 0.2},
        safety_margin=0.2,
        uav_radius=0.1,
    )
    assert path is not None
    assert len(path) >= 3
    assert collision_check_path(path, obstacles, safety_margin=0.2, uav_radius=0.1)
    assert path_length(path) > float(np.linalg.norm(goal - start))


def test_collision_check_path_fails_on_segment_through_obstacle() -> None:
    path = [np.array([-2.0, 0.0]), np.array([2.0, 0.0])]
    obstacles = [_circle_obs(0.0, 0.0, 1.0)]
    assert not collision_check_path(path, obstacles, safety_margin=0.2, uav_radius=0.1)


def test_select_lookahead_waypoint() -> None:
    """Test 3: lookahead returns forward waypoint, not terminal goal."""
    start = np.array([0.0, 0.0])
    waypoint = np.array([1.0, 0.0])
    goal = np.array([3.0, 0.0])
    path = [start, waypoint, goal]
    cur = np.array([0.05, 0.0, 1.0])
    local = select_lookahead_waypoint(
        path, cur, lookahead_dist=0.5, waypoint_accept_radius=0.1, terminal_goal=goal
    )
    assert float(local[0]) < goal[0] - 0.5


def test_cbf_filter_deflects_toward_obstacle() -> None:
    """Test 4: CBF changes nominal velocity near obstacle."""
    obs = [_circle_obs(0.0, 0.0, 1.0)]
    p = np.array([1.32, 0.0])
    u_nom = np.array([-0.25, 0.0])
    u_safe, diag = apply_cbf_filter(
        p,
        u_nom,
        obs,
        np.zeros((2, 2)),
        cfg=CBFConfig(enabled=True, safety_margin=0.2, min_agent_sep=0.6),
        uav_radius=0.1,
        action_low_xy=np.array([-0.25, -0.25]),
        action_high_xy=np.array([0.25, 0.25]),
    )
    assert diag["cbf_action_delta_norm"] > 1e-4 or not diag["qp_success"]
    u_far, diag_far = apply_cbf_filter(
        np.array([5.0, 5.0]),
        u_nom,
        obs,
        np.zeros((2, 2)),
        cfg=CBFConfig(enabled=True),
        uav_radius=0.1,
    )
    assert float(np.linalg.norm(u_far - u_nom)) < 1e-3


def test_cbf_predicts_current_velocity_obstacle_hit() -> None:
    obs = [_circle_obs(0.7, 0.0, 0.1)]
    u_safe, diag = apply_cbf_filter(
        np.array([0.0, 0.0]),
        np.array([0.25, 0.0]),
        obs,
        np.zeros((0, 2)),
        cfg=CBFConfig(
            enabled=True,
            safety_margin=0.1,
            uav_radius=0.1,
            obstacle_activation_radius=0.05,
            forward_range=0.1,
            predictive_enabled=True,
            prediction_horizon_s=3.0,
            predictive_extra_margin=0.2,
        ),
        uav_radius=0.1,
        action_low_xy=np.array([-0.25, -0.25]),
        action_high_xy=np.array([0.25, 0.25]),
    )
    assert diag["cbf_action_delta_norm"] > 1e-4


def test_explicit_cbf_config_keeps_runtime_task_radius() -> None:
    cfg = _resolve_cbf_config(
        CBFConfig(enabled=True, uav_radius=0.15, safety_margin=0.3),
        ReachabilityConfig(),
        "sce_cached_path_cbf_slot",
        [_circle_obs(0.0, 0.0, 1.0)],
        uav_r=0.30,
    )
    assert cfg is not None
    assert cfg.uav_radius == 0.30


def test_cbf_outward_fallback_pushes_away_from_obstacle() -> None:
    obs = [_circle_obs(0.0, 0.0, 1.0)]
    pos = np.array([1.35, 0.0], dtype=np.float64)
    u_nom = np.array([-0.25, 0.0], dtype=np.float64)
    u_safe, diag = apply_cbf_filter(
        pos,
        u_nom,
        obs,
        np.zeros((0, 2), dtype=np.float64),
        cfg=CBFConfig(
            enabled=True,
            safety_margin=0.3,
            uav_radius=0.3,
            infeasible_fallback="outward_velocity",
            emergency_escape_speed=0.25,
            max_projection_iters=1,
        ),
        uav_radius=0.3,
        action_low_xy=np.array([-0.25, -0.25], dtype=np.float64),
        action_high_xy=np.array([0.25, 0.25], dtype=np.float64),
    )
    assert diag["cbf_timeout_or_fallback"]
    assert float(u_safe[0]) > 0.0


def test_control_priority_local_escape_overrides_obstacle_approach() -> None:
    obs = [_circle_obs(0.0, 0.0, 1.0)]
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[0, 0] = -0.25
    pursuer_pos = np.array(
        [[1.35, 0.0, 2.0], [5.0, 5.0, 2.0], [-5.0, -5.0, 2.0]],
        dtype=np.float32,
    )
    targets = pursuer_pos.copy()
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    out, debug, _ms, _active, _delta = _apply_control_priority_layers(
        actions,
        pursuer_pos,
        targets,
        low,
        high,
        all_obstacles=obs,
        other_agents_xy=[pursuer_pos[1:, :2], pursuer_pos[[0, 2], :2], pursuer_pos[:2, :2]],
        yaws=None,
        ccfg=CBFConfig(
            enabled=True,
            safety_margin=0.3,
            uav_radius=0.3,
            emergency_escape_speed=0.25,
            local_escape_enabled=True,
            local_escape_extra_clearance=0.15,
        ),
        cbf_filter=LightweightCBFFilter(),
        cbf_every=100,
        step_counter=2,
    )
    assert debug[0]["local_escape_active"]
    assert float(out[0, 0]) > 0.0


def test_cbf_uses_world_velocity_when_actions_are_body_frame() -> None:
    obs = [_circle_obs(0.0, 0.7, 0.1)]
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[0, 0] = 0.25  # body-frame forward; with yaw=90deg this is +world-y.
    pursuer_pos = np.array(
        [[0.0, 0.0, 1.0], [5.0, 5.0, 1.0], [-5.0, -5.0, 1.0]],
        dtype=np.float32,
    )
    targets = pursuer_pos.copy()
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    out, debug, _ms, _active, _delta = _apply_control_priority_layers(
        actions,
        pursuer_pos,
        targets,
        low,
        high,
        all_obstacles=obs,
        other_agents_xy=[pursuer_pos[1:, :2], pursuer_pos[[0, 2], :2], pursuer_pos[:2, :2]],
        yaws=np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float32),
        ccfg=CBFConfig(
            enabled=True,
            safety_margin=0.1,
            uav_radius=0.1,
            obstacle_activation_radius=0.05,
            forward_range=0.1,
            predictive_enabled=True,
            prediction_horizon_s=3.0,
            predictive_extra_margin=0.2,
        ),
        cbf_filter=LightweightCBFFilter(),
        cbf_every=1,
        step_counter=1,
    )
    assert debug[0]["obstacle_threat"]
    assert float(out[0, 0]) <= 1e-6


def test_control_priority_applies_altitude_floor_guard() -> None:
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[:, 3] = -0.15
    pursuer_pos = np.array(
        [[0.0, 0.0, 0.56], [1.0, 0.0, 0.58], [2.0, 0.0, 0.60]],
        dtype=np.float32,
    )
    targets = pursuer_pos.copy()
    targets[:, 2] = 0.52
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    out, _debug, _ms, _active, _delta = _apply_control_priority_layers(
        actions,
        pursuer_pos,
        targets,
        low,
        high,
        all_obstacles=[],
        other_agents_xy=[pursuer_pos[1:, :2], pursuer_pos[[0, 2], :2], pursuer_pos[:2, :2]],
        yaws=np.zeros(3, dtype=np.float32),
        ccfg=None,
        cbf_filter=LightweightCBFFilter(),
        cbf_every=1,
        step_counter=1,
        z_floor=0.5,
        altitude_floor_margin=0.25,
    )
    np.testing.assert_allclose(out[:, 3], 0.15, atol=1e-6)


def test_control_priority_keeps_pursuer_inside_xy_boundary() -> None:
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[0, 0] = 0.25
    pursuer_pos = np.array(
        [[19.9, 0.0, 2.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    targets = pursuer_pos.copy()
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    out, debug, _ms, _active, _delta = _apply_control_priority_layers(
        actions,
        pursuer_pos,
        targets,
        low,
        high,
        all_obstacles=[],
        other_agents_xy=[pursuer_pos[1:, :2], pursuer_pos[[0, 2], :2], pursuer_pos[:2, :2]],
        yaws=None,
        ccfg=None,
        cbf_filter=LightweightCBFFilter(),
        cbf_every=1,
        step_counter=1,
        world_xy=20.0,
        boundary_margin=0.3,
        boundary_alpha=2.0,
    )
    assert debug[0]["boundary_active"]
    assert float(out[0, 0]) < 0.0


def test_control_priority_xy_boundary_uses_world_velocity_with_yaw() -> None:
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[0, 0] = 0.25
    pursuer_pos = np.array(
        [[0.0, 19.9, 2.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    targets = pursuer_pos.copy()
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    out, debug, _ms, _active, _delta = _apply_control_priority_layers(
        actions,
        pursuer_pos,
        targets,
        low,
        high,
        all_obstacles=[],
        other_agents_xy=[pursuer_pos[1:, :2], pursuer_pos[[0, 2], :2], pursuer_pos[:2, :2]],
        yaws=np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float32),
        ccfg=None,
        cbf_filter=LightweightCBFFilter(),
        cbf_every=1,
        step_counter=1,
        world_xy=20.0,
        boundary_margin=0.3,
        boundary_alpha=2.0,
    )
    wy = out[0, 0] * np.sin(np.pi / 2.0) + out[0, 1] * np.cos(np.pi / 2.0)
    assert debug[0]["boundary_active"]
    assert float(wy) < 0.0


def test_turn_safety_slowdown_immediately_points_to_new_direction() -> None:
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[0, 0] = 0.25
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    debug = [
        {"turn_safety_active": True, "turn_desired_dir_xy": [0.0, 1.0]},
        {},
        {},
    ]

    out = _apply_turn_slowdown_actions(
        actions,
        debug,
        low,
        high,
        yaws=None,
        slow_speed=0.08,
        yaw_gain=0.8,
    )

    assert debug[0]["turn_slowdown_active"]
    np.testing.assert_allclose(out[0, :2], np.array([0.0, 0.08], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out[1:, :2], 0.0, atol=1e-6)


def test_structure_selection_default_does_not_optimize_slot_risk() -> None:
    cfg = StructureSelectionConfig.from_dict(None)
    assert cfg.w_slot_risk == 0.0


def test_safety_speed_constraint_caps_full_speed_before_obstacle() -> None:
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[0, 0] = 0.25
    pursuer_pos = np.array(
        [[0.0, 0.0, 2.0], [5.0, 5.0, 2.0], [-5.0, -5.0, 2.0]],
        dtype=np.float32,
    )
    assigned_targets = np.array(
        [[3.0, 0.0, 2.0], [5.0, 5.0, 2.0], [-5.0, -5.0, 2.0]],
        dtype=np.float32,
    )
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    obs = [_circle_obs(0.55, 0.0, 0.1)]

    out, debug = _apply_safety_speed_constraints(
        actions,
        pursuer_pos,
        assigned_targets,
        low,
        high,
        obs,
        yaws=None,
        world_xy=20.0,
        boundary_margin=0.3,
        uav_radius=0.1,
        cfg={
            "safety_speed_free_clearance": 2.0,
            "safety_speed_slow_clearance": 1.0,
            "safety_speed_stop_clearance": 0.45,
            "safety_speed_min": 0.04,
            "safety_speed_cruise": 0.16,
            "safety_speed_max": 0.22,
            "safety_speed_prediction_horizon": 2.0,
            "structure_speed_enabled": True,
            "structure_cruise_speed": 0.16,
            "structure_capture_speed": 0.22,
            "structure_capture_dist": 0.75,
        },
    )

    assert debug[0]["safety_speed_active"]
    assert float(np.linalg.norm(out[0, :2])) < 0.25
    assert float(np.linalg.norm(out[0, :2])) <= float(debug[0]["safety_speed_cap"]) + 1e-6


def test_safety_speed_constraint_cruises_below_full_speed_until_slot_formed() -> None:
    actions = np.zeros((3, 4), dtype=np.float32)
    actions[0, 0] = 0.25
    pursuer_pos = np.array(
        [[0.0, 0.0, 2.0], [5.0, 5.0, 2.0], [-5.0, -5.0, 2.0]],
        dtype=np.float32,
    )
    assigned_targets = np.array(
        [[4.0, 0.0, 2.0], [5.0, 5.0, 2.0], [-5.0, -5.0, 2.0]],
        dtype=np.float32,
    )
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    out, debug = _apply_safety_speed_constraints(
        actions,
        pursuer_pos,
        assigned_targets,
        low,
        high,
        [],
        yaws=None,
        world_xy=20.0,
        boundary_margin=0.3,
        uav_radius=0.1,
        cfg={
            "safety_speed_cruise": 0.16,
            "safety_speed_max": 0.25,
            "structure_speed_enabled": True,
            "structure_cruise_speed": 0.16,
            "structure_capture_speed": 0.25,
            "structure_capture_dist": 0.75,
        },
    )

    assert debug[0]["safety_speed_active"]
    np.testing.assert_allclose(np.linalg.norm(out[0, :2]), 0.16, atol=1e-6)


def test_regression_ot_slot_unchanged_without_reachability() -> None:
    """Test 5: ot_slot without reachability matches task OT assignment."""
    from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import PursuitEvasion3v1Task

    pursuer_pos = np.array(
        [
            [0.0, 3.0, 1.0],
            [2.5, -1.0, 1.0],
            [-2.0, -2.0, 1.0],
        ],
        dtype=np.float32,
    )
    evader_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        role_assignment_mode="entropic_ot",
        assignment_inertia_margin=0.0,
    )
    state = SimpleNamespace(
        pursuer_ids=np.array([0, 1, 2], dtype=np.int64),
        evader_id=3,
        assigned_target_indices=np.array([0, 1, 2], dtype=np.int64),
    )
    lin_pos = np.zeros((4, 3), dtype=np.float32)
    lin_pos[:3] = pursuer_pos
    lin_pos[3] = evader_pos
    env = SimpleNamespace(task=task, task_state=state, prev_backend_state=None)
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    ot_slot_actions_from_state(env, lin_pos, low, high, xy_gain=0.25, z_gain=0.20, yaw_gain=0.0)
    _, expected, _ = task._assigned_targets_from_state(
        pursuer_pos, evader_pos, task_state=state, role_assignment_mode="entropic_ot"
    )
    np.testing.assert_array_equal(state.assigned_target_indices, expected)
