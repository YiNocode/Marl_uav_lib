from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from marl_uav.control.boundary_utils import apply_xy_boundary_barrier
from marl_uav.control.manifold_generator import (
    ManifoldGenerator,
    ManifoldSignature,
    _radius_at_angle,
    build_anchored_manifold_curve,
    build_manifold_signature,
    build_pursuer_manifold_path,
    build_shared_manifold_curve,
    should_replan_manifold_paths,
)
from marl_uav.control.obstacle_avoidance_controller import ObstacleAvoidanceController
from marl_uav.control.slot_allocator import SlotAllocator
from marl_uav.control.trajectory_planner import (
    SlotTargetStabilizer,
    SlotTargetStabilizerConfig,
    _apply_predictive_xy_boundary_guard,
    trajectory_planner_actions_from_state,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import PursuitEvasion3v1Task
from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.planning.local_reachability import local_reachability_probe
from marl_uav.framework.planning.turn_radius_obstacle_query import TurnRadiusObstacleQueryConfig
from marl_uav.utils.config import load_config
from marl_uav.utils.debug_browser import build_debug_frame
from marl_uav.utils.debug_viz import resolve_viz_profile


def _obs(x: float, y: float, r: float) -> Obstacle:
    return Obstacle(kind="circle", center=np.array([x, y], dtype=np.float64), radius=float(r))


def _task_state(obstacle_xy=None, obstacle_r=None):
    return SimpleNamespace(
        pursuer_ids=np.array([0, 1, 2], dtype=np.int64),
        evader_id=3,
        assigned_target_indices=np.array([0, 1, 2], dtype=np.int64),
        obstacle_xy=np.asarray(obstacle_xy if obstacle_xy is not None else np.zeros((0, 2)), dtype=np.float32),
        obstacle_r=np.asarray(obstacle_r if obstacle_r is not None else np.zeros((0,)), dtype=np.float32),
        elapsed_steps=0,
    )


def test_manifold_generator_matches_task_without_obstacles() -> None:
    task = PursuitEvasion3v1Task()
    state = _task_state()
    pursuers = np.array([[-4.0, -1.0, 1.0], [-4.0, 0.0, 1.0], [-4.0, 1.0, 1.0]], dtype=np.float32)
    evader = np.array([2.0, 0.0, 1.0], dtype=np.float32)

    slots, curve, diag = ManifoldGenerator({"curve_num_samples": 33}).generate(task, pursuers, evader, state)
    expected = task._reference_manifold_targets(pursuers, evader, task_state=state)

    np.testing.assert_allclose(slots, expected, atol=1e-5)
    assert curve is not None
    assert curve.shape == (33, 3)
    assert diag["curve_num_samples"] == 33


def test_manifold_generator_pushes_slot_away_from_obstacle_ray() -> None:
    task = PursuitEvasion3v1Task(
        manifold_target_phase=0.0,
        obstacle_manifold_top_k=1,
        obstacle_manifold_fourier_scale=0.0,
        obstacle_manifold_bump_scale=0.0,
    )
    state = _task_state([[3.5, 0.0]], [0.8])
    pursuers = np.array([[-4.0, -1.0, 1.0], [-4.0, 0.0, 1.0], [-4.0, 1.0, 1.0]], dtype=np.float32)
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    rho_base = task._compute_target_radius_xy(pursuers, evader, task_state=state)
    slots, _curve, _diag = ManifoldGenerator().generate(task, pursuers, evader, state)
    radii = np.linalg.norm(slots[:, :2] - evader[None, :2], axis=1)

    assert radii[0] > rho_base
    assert radii[0] > radii[1]
    assert radii[0] > radii[2]


def test_pursuer_manifold_path_passes_through_pursuer_and_slot() -> None:
    task = PursuitEvasion3v1Task()
    state = _task_state()
    pursuers = np.array([[-4.0, -1.0, 1.0], [-4.0, 0.0, 1.0], [-4.0, 1.0, 1.0]], dtype=np.float32)
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    slots, _curve, _diag = ManifoldGenerator({"curve_num_samples": 64}).generate(
        task, pursuers, evader, state
    )

    anchored = build_anchored_manifold_curve(
        task, pursuers[0, :2], pursuers, evader, state, num_samples=64
    )
    path = build_pursuer_manifold_path(
        pursuers[0, :2],
        slots[0],
        anchored,
        evader_xy=evader[:2],
    )
    np.testing.assert_allclose(path[0, :2], pursuers[0, :2], atol=1e-5)
    np.testing.assert_allclose(path[-1, :2], slots[0, :2], atol=1e-5)
    assert path.shape[0] >= 3

    gen = ManifoldGenerator()
    _slots, curve, _diag = gen.generate(task, pursuers, evader, state)
    paths = gen.generate_pursuer_paths(pursuers, _slots, curve, evader_pos=evader)
    assert len(paths) == 3
    for i, pth in enumerate(paths):
        np.testing.assert_allclose(pth[0, :2], pursuers[i, :2], atol=1e-5)
        np.testing.assert_allclose(pth[-1, :2], _slots[i, :2], atol=1e-5)


def test_shared_manifold_curve_passes_through_pursuers() -> None:
    task = PursuitEvasion3v1Task()
    state = _task_state([[1.2, 0.0]], [0.25])
    pursuers = np.array([[-3.0, 1.5, 1.0], [-3.0, 0.0, 1.0], [-3.0, -1.5, 1.0]], dtype=np.float32)
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    curve, _meta = build_shared_manifold_curve(task, pursuers, evader, state, num_samples=128)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=True) + float(task.manifold_target_phase)
    rho = np.linalg.norm(curve[:, :2] - evader[:2], axis=1)
    for i in range(3):
        ang = float(np.arctan2(float(pursuers[i, 1] - evader[1]), float(pursuers[i, 0] - evader[0])))
        r_p = float(np.linalg.norm(pursuers[i, :2] - evader[:2]))
        rho_at = _radius_at_angle(theta, rho, ang)
        assert rho_at >= r_p - 1e-6


def test_shared_manifold_radius_contracts_over_time() -> None:
    task = PursuitEvasion3v1Task(manifold_contraction_rate=0.05)
    pursuers = np.array([[-4.0, 0.0, 1.0], [-4.0, 2.0, 1.0], [-4.0, -2.0, 1.0]], dtype=np.float32)
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    state_early = _task_state()
    state_early.elapsed_steps = 0
    state_late = _task_state()
    state_late.elapsed_steps = 40
    _, meta_early = build_shared_manifold_curve(task, pursuers, evader, state_early, num_samples=64)
    _, meta_late = build_shared_manifold_curve(task, pursuers, evader, state_late, num_samples=64)
    assert meta_late["rho_base"] < meta_early["rho_base"]
    assert meta_late["contraction_decay"] < meta_early["contraction_decay"]


def test_manifold_generator_produces_single_shared_curve() -> None:
    task = PursuitEvasion3v1Task()
    state = _task_state()
    pursuers = np.array([[-4.0, -1.0, 1.0], [-4.0, 0.0, 1.0], [-4.0, 1.0, 1.0]], dtype=np.float32)
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gen = ManifoldGenerator({"curve_num_samples": 64})
    _slots, curve, diag = gen.generate(task, pursuers, evader, state)
    assert curve is not None
    assert gen._last_manifold_curve is not None
    paths = gen.generate_pursuer_paths(pursuers, _slots, curve, evader_pos=evader)
    assert len(paths) == 3
    assert "rho_base" in diag
    assert "contraction_decay" in diag


def test_anchored_manifold_curve_bulges_around_obstacle() -> None:
    task = PursuitEvasion3v1Task(
        obstacle_manifold_top_k=1,
        obstacle_manifold_bump_scale=0.35,
        obstacle_manifold_fourier_scale=0.0,
    )
    state = _task_state([[1.0, 0.0]], [0.30])
    pursuers = np.array([[-3.0, 0.0, 1.0], [-3.0, 2.0, 1.0], [-3.0, -2.0, 1.0]], dtype=np.float32)
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    curve, _meta = build_shared_manifold_curve(task, pursuers, evader, state, num_samples=128)
    rho_before = np.asarray(
        task._obstacle_aware_radius(
            np.linspace(0.0, 2.0 * np.pi, 128, endpoint=True, dtype=np.float32)
            + np.float32(task.manifold_target_phase),
            np.float32(
                task._compute_target_radius_xy(pursuers, evader.astype(np.float32), task_state=state)
            ),
            evader.astype(np.float32),
            task_state=state,
        ),
        dtype=np.float64,
    )
    curve = build_anchored_manifold_curve(
        task, pursuers[0, :2], pursuers, evader, state, num_samples=128
    )
    radii = np.linalg.norm(curve[:, :2] - evader[:2], axis=1)
    ang = np.arctan2(curve[:, 1] - evader[1], curve[:, 0] - evader[0])
    obs_idx = int(np.argmin(np.abs(ang - 0.0)))
    away_idx = int(np.argmin(np.abs(ang - np.pi)))
    assert radii[obs_idx] >= float(rho_before[obs_idx]) - 1e-6
    assert radii[obs_idx] >= radii[away_idx] - 1e-6


def test_pursuer_manifold_path_sweeps_toward_slot_not_ring_tangent() -> None:
    task = PursuitEvasion3v1Task()
    state = _task_state()
    pursuers = np.array([[-4.0, 0.0, 1.0], [-4.0, 2.0, 1.0], [-4.0, 4.0, 1.0]], dtype=np.float32)
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    _slots, curve, _diag = ManifoldGenerator({"curve_num_samples": 64}).generate(
        task, pursuers, evader, state
    )
    curve, _meta = build_shared_manifold_curve(task, pursuers, evader, state, num_samples=64)
    slot = np.array([2.0, -1.5, 1.0], dtype=np.float32)
    path = build_pursuer_manifold_path(
        pursuers[0, :2],
        slot,
        curve,
        evader_xy=evader[:2],
        num_samples=16,
    )
    early = path[4, :2] - path[0, :2]
    to_slot = slot[:2] - pursuers[0, :2]
    # Early segment should move toward slot vertically, not stay purely horizontal.
    assert early[1] * to_slot[1] > 0.0
    assert abs(float(early[1])) > 0.05


def test_slot_allocator_returns_unique_assignment_and_honors_inertia() -> None:
    pursuers = np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 4.0, 1.0]], dtype=np.float32)
    slots = np.array([[0.0, 0.1, 1.0], [0.0, 2.1, 1.0], [0.0, 4.1, 1.0]], dtype=np.float32)
    alloc = SlotAllocator({"assignment_inertia_margin": 0.0})

    assignment, assigned, diag = alloc.allocate(pursuers, slots, [])
    assert sorted(assignment.tolist()) == [0, 1, 2]
    np.testing.assert_allclose(assigned, slots[assignment], atol=1e-6)
    assert np.asarray(diag["cost_matrix"]).shape == (3, 3)

    alloc.previous_assignment = np.array([2, 1, 0], dtype=np.int64)
    sticky, _assigned, _diag = alloc.allocate(pursuers, slots, [])
    assert sorted(sticky.tolist()) == [0, 1, 2]
    alloc.cfg = type(alloc.cfg)(assignment_inertia_margin=1_000_000.0)
    alloc.previous_assignment = np.array([2, 1, 0], dtype=np.int64)
    sticky, _assigned, _diag = alloc.allocate(pursuers, slots, [])
    np.testing.assert_array_equal(sticky, [2, 1, 0])


def test_slot_target_stabilizer_filters_and_caps_slot_motion() -> None:
    stab = SlotTargetStabilizer()
    cfg = SlotTargetStabilizerConfig(
        assignment_switch_margin=0.5,
        min_assignment_hold_steps=20,
        slot_filter_alpha=0.5,
        slot_target_vmax_ratio=0.5,
        control_dt=0.1,
    )
    slots0 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64)
    assignment0 = np.array([0, 1, 2], dtype=np.int64)
    stab.update(
        slots=slots0,
        raw_assignment=assignment0,
        raw_assigned_targets=slots0[assignment0],
        current_step=0,
        cfg=cfg,
        tracking_vmax=1.0,
    )

    slots1 = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64)
    assignment, stabilized, diag = stab.update(
        slots=slots1,
        raw_assignment=assignment0,
        raw_assigned_targets=slots1[assignment0],
        current_step=1,
        cfg=cfg,
        tracking_vmax=1.0,
    )

    np.testing.assert_array_equal(assignment, assignment0)
    np.testing.assert_allclose(stabilized[0, :2], [0.05, 0.0], atol=1e-9)
    assert diag[0]["slot_target_shift"] == 0.05
    assert diag[0]["slot_target_speed"] == 0.5


def test_slot_target_stabilizer_rejects_switch_during_hold() -> None:
    stab = SlotTargetStabilizer()
    cfg = SlotTargetStabilizerConfig(
        assignment_switch_margin=0.5,
        min_assignment_hold_steps=20,
        slot_filter_alpha=1.0,
        slot_target_vmax_ratio=100.0,
        control_dt=0.1,
    )
    slots0 = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [20.0, 0.0, 1.0]], dtype=np.float64)
    assignment0 = np.array([0, 1, 2], dtype=np.int64)
    stab.update(
        slots=slots0,
        raw_assignment=assignment0,
        raw_assigned_targets=slots0[assignment0],
        current_step=0,
        cfg=cfg,
        tracking_vmax=1.0,
    )

    raw_assignment = np.array([1, 0, 2], dtype=np.int64)
    assignment, stabilized, diag = stab.update(
        slots=slots0,
        raw_assignment=raw_assignment,
        raw_assigned_targets=slots0[raw_assignment],
        current_step=1,
        cfg=cfg,
        tracking_vmax=1.0,
    )

    np.testing.assert_array_equal(assignment, assignment0)
    np.testing.assert_allclose(stabilized[:, :2], slots0[:, :2], atol=1e-9)
    assert diag[0]["assignment_changed"] is False
    assert diag[0]["assignment_hold_age"] == 1


def test_slot_target_stabilizer_freeze_modes() -> None:
    slots0 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64)
    slots1 = np.array([[0.5, 0.0, 1.0], [1.5, 0.0, 1.0], [2.5, 0.0, 1.0]], dtype=np.float64)
    assignment0 = np.array([0, 1, 2], dtype=np.int64)
    assignment1 = np.array([2, 1, 0], dtype=np.int64)

    freeze_slots = SlotTargetStabilizer()
    freeze_slots_cfg = SlotTargetStabilizerConfig(
        slot_filter_alpha=1.0,
        slot_target_vmax_ratio=100.0,
        freeze_slots_after_first_step=True,
    )
    freeze_slots.update(
        slots=slots0,
        raw_assignment=assignment0,
        raw_assigned_targets=slots0[assignment0],
        current_step=0,
        cfg=freeze_slots_cfg,
        tracking_vmax=1.0,
    )
    assignment, stabilized, _diag = freeze_slots.update(
        slots=slots1,
        raw_assignment=assignment1,
        raw_assigned_targets=slots1[assignment1],
        current_step=1,
        cfg=freeze_slots_cfg,
        tracking_vmax=1.0,
    )
    np.testing.assert_array_equal(assignment, assignment0)
    np.testing.assert_allclose(stabilized[:, :2], slots0[:, :2], atol=1e-9)

    freeze_assignment = SlotTargetStabilizer()
    freeze_assignment_cfg = SlotTargetStabilizerConfig(
        slot_filter_alpha=1.0,
        slot_target_vmax_ratio=100.0,
        freeze_assignment_after_first_step=True,
    )
    freeze_assignment.update(
        slots=slots0,
        raw_assignment=assignment0,
        raw_assigned_targets=slots0[assignment0],
        current_step=0,
        cfg=freeze_assignment_cfg,
        tracking_vmax=1.0,
    )
    assignment, stabilized, _diag = freeze_assignment.update(
        slots=slots1,
        raw_assignment=assignment1,
        raw_assigned_targets=slots1[assignment1],
        current_step=1,
        cfg=freeze_assignment_cfg,
        tracking_vmax=1.0,
    )
    np.testing.assert_array_equal(assignment, assignment0)
    np.testing.assert_allclose(stabilized[:, :2], slots1[assignment0, :2], atol=1e-9)


def test_obstacle_avoidance_no_obstacle_moves_straight_and_reports_path() -> None:
    ctrl = ObstacleAvoidanceController({"vmax": 0.25, "omega_max": 1.0})

    action, yaw_rate, path, diag = ctrl.compute_action(
        np.array([0.0, 0.0]),
        0.0,
        np.array([2.0, 0.0]),
        [],
    )

    np.testing.assert_allclose(action, [0.25, 0.0], atol=1e-6)
    assert abs(yaw_rate) < 1e-6
    assert path.shape[1] == 2
    assert len(diag["best_path_xy"]) == path.shape[0]


def test_obstacle_avoidance_vel_smooth_prefers_braking_with_momentum() -> None:
    ctrl_free = ObstacleAvoidanceController(
        {
            "horizon_s": 1.5,
            "vmax": 0.25,
            "amax_xy": 0.5,
            "w_vel_smooth": 0.0,
            "w_speed": 0.0,
            "world_xy": 5.0,
            "boundary_margin": 0.30,
        }
    )
    ctrl_smooth = ObstacleAvoidanceController(
        {
            "horizon_s": 1.5,
            "vmax": 0.25,
            "amax_xy": 0.5,
            "w_vel_smooth": 8.0,
            "w_speed": 0.0,
            "world_xy": 5.0,
            "boundary_margin": 0.30,
        }
    )
    pos = np.array([3.8, 0.0])
    current_v = np.array([0.25, 0.0])
    _a_free, _, _, diag_free = ctrl_free.compute_action(
        pos, 0.0, np.array([0.0, -2.0]), [], current_velocity_xy=current_v
    )
    _a_smooth, _, _, diag_smooth = ctrl_smooth.compute_action(
        pos, 0.0, np.array([0.0, -2.0]), [], current_velocity_xy=current_v
    )
    assert diag_smooth["best_candidate_speed"] <= diag_free["best_candidate_speed"] + 1e-6


def test_obstacle_avoidance_turns_when_direct_path_is_blocked() -> None:
    ctrl = ObstacleAvoidanceController(
        {"horizon_s": 2.0, "uav_radius": 0.10, "safety_margin": 0.10, "w_obstacle": 4.0}
    )

    action, yaw_rate, path, diag = ctrl.compute_action(
        np.array([0.0, 0.0]),
        0.0,
        np.array([2.0, 0.0]),
        [_obs(0.5, 0.0, 0.05)],
    )

    assert diag["valid_candidate_count"] > 0
    assert float(diag["best_candidate_speed"]) < 0.25 - 1e-6


def test_executable_safety_heading_gate_stops_large_misalignment() -> None:
    ctrl = ObstacleAvoidanceController(
        {
            "vmax": 0.25,
            "speed_samples": [1.0, 0.6, 0.3, 0.0],
            "prefer_holonomic_tracking": False,
            "use_sampled_planner": True,
            "direct_los_enabled": False,
        }
    )
    _action, _yaw_rate, _path, diag = ctrl.compute_action(
        np.array([0.0, 0.0]),
        0.0,
        np.array([0.0, 2.0]),
        [],
    )
    assert diag["heading_err_now"] > np.pi / 3.0
    assert diag["speed_gate"] == "stop"
    assert float(diag["best_candidate_speed"]) < 1e-6


def test_executable_safety_rejects_out_of_bounds_rollouts() -> None:
    ctrl = ObstacleAvoidanceController(
        {
            "horizon_s": 1.0,
            "vmax": 0.25,
            "safety_margin": 0.10,
            "uav_radius": 0.10,
            "boundary_margin": 0.30,
            "prefer_holonomic_tracking": False,
            "use_sampled_planner": True,
            "direct_los_enabled": False,
        }
    )
    pos = np.array([4.5, 0.0])
    bounds = (-5.0, 5.0, -5.0, 5.0)
    _action, _yaw_rate, path, diag = ctrl.compute_action(
        pos,
        0.0,
        np.array([4.8, 0.0]),
        [],
        current_vel_xy=np.array([0.25, 0.0]),
        bounds_xy=bounds,
    )
    assert diag["out_of_bounds_candidate_count"] > 0
    assert np.all(np.abs(path[:, 0]) <= 5.0 - 0.30 + 1e-6)


def test_obstacle_avoidance_respects_boundary() -> None:
    world_xy = 5.0
    margin = 0.30
    ctrl = ObstacleAvoidanceController(
        {
            "horizon_s": 1.0,
            "vmax": 0.20,
            "world_xy": world_xy,
            "boundary_margin": margin,
        }
    )
    pos = np.array([3.5, 0.0])
    _action, _yaw_rate, path, diag = ctrl.compute_action(
        pos,
        0.0,
        np.array([4.0, 0.0]),
        [],
    )
    assert diag["valid_candidate_count"] > 0
    usable = world_xy - margin
    assert np.all(np.abs(path[:, 0]) <= usable + 1e-6)


def test_obstacle_avoidance_boundary_filter_preserves_tangent() -> None:
    ctrl = ObstacleAvoidanceController(
        {
            "vmax": 1.0,
            "position_kp": 1.0,
            "boundary_activation_distance": 0.6,
            "boundary_braking_margin": 0.0,
            "world_xy": 5.0,
            "boundary_margin": 0.0,
        }
    )

    action, _yaw_rate, _path, diag = ctrl.compute_action(
        np.array([4.6, 0.0]),
        0.0,
        np.array([6.0, -1.0]),
        [],
        bounds_xy=(-5.0, 5.0, -5.0, 5.0),
    )

    assert diag["boundary_filter_active"] is True
    assert "x_max" in diag["boundary_active_names"]
    assert action[0] <= 1e-9
    assert action[1] < -0.55


def test_obstacle_avoidance_boundary_filter_handles_corner_axes() -> None:
    ctrl = ObstacleAvoidanceController(
        {
            "vmax": 1.0,
            "position_kp": 1.0,
            "boundary_activation_distance": 0.6,
            "boundary_braking_margin": 0.0,
            "world_xy": 5.0,
            "boundary_margin": 0.0,
        }
    )

    action, _yaw_rate, _path, diag = ctrl.compute_action(
        np.array([4.6, 4.6]),
        0.0,
        np.array([6.0, 6.0]),
        [],
        bounds_xy=(-5.0, 5.0, -5.0, 5.0),
    )

    assert set(diag["boundary_active_names"]) == {"x_max", "y_max"}
    assert action[0] <= 1e-9
    assert action[1] <= 1e-9


def test_trajectory_planner_boundary_barrier_clamps_outward_motion() -> None:
    world_xy = 5.0
    margin = 0.30
    pos = np.array([world_xy - 0.20, 0.0])
    u_world = np.array([0.25, 0.0])
    low = np.array([-0.25, -0.25], dtype=np.float64)
    high = np.array([0.25, 0.25], dtype=np.float64)
    safe, active = apply_xy_boundary_barrier(
        pos,
        u_world,
        world_xy=world_xy,
        boundary_margin=margin,
        boundary_alpha=2.0,
        action_low_xy=low,
        action_high_xy=high,
    )
    assert active is True
    assert safe[0] < u_world[0]


def test_predictive_boundary_guard_limits_outward_speed_by_braking_distance() -> None:
    safe, active, diag = _apply_predictive_xy_boundary_guard(
        np.array([4.6, 0.0]),
        np.array([0.5, 0.0]),
        np.array([0.0, 0.0]),
        world_xy=5.0,
        boundary_margin=0.30,
        boundary_gain=3.0,
        amax_xy=0.5,
        action_low_xy=np.array([-0.5, -0.5]),
        action_high_xy=np.array([0.5, 0.5]),
    )

    assert active is True
    np.testing.assert_allclose(safe, [0.3, 0.0], atol=1e-6)
    assert diag["predictive_boundary_active_axes"] == ["x"]


def test_predictive_boundary_guard_commands_inward_velocity_when_stopping_too_late() -> None:
    safe_upper, active_upper, diag_upper = _apply_predictive_xy_boundary_guard(
        np.array([4.65, 0.0]),
        np.array([0.1, 0.0]),
        np.array([0.4, 0.0]),
        world_xy=5.0,
        boundary_margin=0.30,
        boundary_gain=3.0,
        amax_xy=0.5,
        action_low_xy=np.array([-0.5, -0.5]),
        action_high_xy=np.array([0.5, 0.5]),
    )
    safe_lower, active_lower, diag_lower = _apply_predictive_xy_boundary_guard(
        np.array([-4.65, 0.0]),
        np.array([-0.1, 0.0]),
        np.array([-0.4, 0.0]),
        world_xy=5.0,
        boundary_margin=0.30,
        boundary_gain=3.0,
        amax_xy=0.5,
        action_low_xy=np.array([-0.5, -0.5]),
        action_high_xy=np.array([0.5, 0.5]),
    )

    assert active_upper is True
    assert active_lower is True
    assert safe_upper[0] < 0.0
    assert safe_lower[0] > 0.0
    assert diag_upper["predictive_boundary_details"]["x"]["reason"] == "upper_braking_override"
    assert diag_lower["predictive_boundary_details"]["x"]["reason"] == "lower_braking_override"


def test_slot_allocator_reachability_cost_prefers_clear_slot() -> None:
    obstacles = [_obs(0.0, 0.0, 0.20)]
    query_cfg = TurnRadiusObstacleQueryConfig(
        horizon_s=2.0,
        vmax=0.25,
        num_yaw_samples=11,
        uav_radius=0.10,
        safety_margin=0.10,
    )
    blocked = local_reachability_probe(
        np.array([-1.5, 0.0]),
        0.0,
        np.array([1.5, 0.0]),
        obstacles,
        cfg=query_cfg,
    )
    clear = local_reachability_probe(
        np.array([-1.5, 0.0]),
        0.0,
        np.array([-1.5, 2.5]),
        obstacles,
        cfg=query_cfg,
    )
    assert clear.valid_candidate_count > 0
    assert blocked.min_clearance <= clear.min_clearance

    pursuers = np.array([[-1.5, 0.0, 1.0], [0.0, 5.0, 1.0], [0.0, -5.0, 1.0]], dtype=np.float32)
    slots = np.array([[1.5, 0.0, 1.0], [-1.5, 2.5, 1.0], [1.5, -2.5, 1.0]], dtype=np.float32)
    alloc = SlotAllocator(
        {
            "assignment_inertia_margin": 0.0,
            "los_penalty": 0.0,
            "w_reach_block": 100.0,
            "w_clearance": 1.0,
            "w_path": 0.5,
            "uav_radius": 0.10,
            "safety_margin": 0.10,
            "reachability": {
                "horizon_s": 2.0,
                "vmax": 0.25,
                "num_yaw_samples": 11,
                "uav_radius": 0.10,
                "safety_margin": 0.10,
            },
        }
    )
    _assignment, _assigned, diag = alloc.allocate(
        pursuers,
        slots,
        obstacles,
        pursuer_yaws=np.zeros(3, dtype=np.float64),
        world_xy=20.0,
    )
    cost = np.asarray(diag["cost_matrix"], dtype=np.float64)
    assert cost[0, 0] > cost[0, 1]


def test_obstacle_avoidance_all_blocked_falls_back_with_path() -> None:
    ctrl = ObstacleAvoidanceController(
        {
            "uav_radius": 0.10,
            "safety_margin": 0.10,
            "prefer_holonomic_tracking": False,
            "use_sampled_planner": True,
        }
    )
    obstacles = [_obs(0.2, 0.0, 0.25), _obs(-0.2, 0.0, 0.25), _obs(0.0, 0.2, 0.25), _obs(0.0, -0.2, 0.25)]

    action, _yaw_rate, path, diag = ctrl.compute_action(
        np.array([0.0, 0.0]),
        0.0,
        np.array([2.0, 0.0]),
        obstacles,
    )

    assert diag["local_planner_blocked"] is True
    np.testing.assert_allclose(action, [0.0, 0.0], atol=1e-9)
    assert len(diag["best_path_xy"]) == path.shape[0]


def test_inertial_rollout_brakes_near_boundary_with_momentum() -> None:
    ctrl = ObstacleAvoidanceController(
        {
            "horizon_s": 1.5,
            "dt": 0.1,
            "vmax": 0.25,
            "amax_xy": 0.5,
            "world_xy": 5.0,
            "boundary_margin": 0.30,
            "w_speed": 0.0,
            "w_vel_smooth": 0.0,
            "prefer_holonomic_tracking": False,
            "use_sampled_planner": True,
            "direct_los_enabled": False,
        }
    )
    pos = np.array([3.8, 2.0])
    yaw = 0.0
    v0 = np.array([0.25, 0.0])
    goal = np.array([0.0, -2.0])

    _action, _yaw_rate, path, diag = ctrl.compute_action(
        pos,
        yaw,
        goal,
        [],
        current_velocity_xy=v0,
    )

    assert diag["valid_candidate_count"] > 0
    assert np.all(np.abs(path[:, 0]) <= 5.0 - 0.30 + 1e-6)
    assert float(diag["best_candidate_speed"]) < 0.25 - 1e-6


def test_inertial_rollout_rejects_coasting_into_obstacle() -> None:
    obstacles = [_obs(0.35, 0.0, 0.10)]
    query_cfg = TurnRadiusObstacleQueryConfig(
        horizon_s=1.0,
        dt=0.1,
        vmax=0.25,
        amax_xy=0.5,
        uav_radius=0.10,
        safety_margin=0.10,
    )
    reach = local_reachability_probe(
        np.array([0.0, 0.0]),
        0.0,
        np.array([0.0, 2.0]),
        obstacles,
        cfg=query_cfg,
        current_velocity_xy=np.array([0.25, 0.0]),
    )
    assert reach.blocked or reach.best_speed < 0.25 - 1e-6


def test_e2_suite_uses_sce_trajectory_planner_main_method() -> None:
    suite = load_config("configs/benchmark/e2_obstacle_field_suite.yaml")
    methods = suite["methods"]
    assert "SCE" in methods
    assert methods["SCE"]["config"].endswith("configs/experiment/e2/sce.yaml")
    cfg = load_config(methods["SCE"]["config"])
    assert "trajectory_planner" in cfg


def test_should_replan_on_rho_change() -> None:
    prev = ManifoldSignature(rho_base=2.0, contraction_decay=1.0, max_curve_displacement=0.0)
    new = ManifoldSignature(rho_base=1.9, contraction_decay=1.0, max_curve_displacement=0.0)
    assert should_replan_manifold_paths(
        prev,
        new,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        curve_tol=0.05,
        rho_tol=0.001,
    )


def test_should_not_replan_when_manifold_unchanged() -> None:
    prev = ManifoldSignature(rho_base=2.0, contraction_decay=1.0, max_curve_displacement=0.01)
    new = ManifoldSignature(rho_base=2.0, contraction_decay=1.0, max_curve_displacement=0.01)
    assert not should_replan_manifold_paths(
        prev,
        new,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        curve_tol=0.05,
        rho_tol=0.001,
    )


def _trajectory_planner_env(*, elapsed_steps: int = 0, step_count: int = 1):
    task = PursuitEvasion3v1Task(
        world_xy=20.0,
        role_assignment_mode="entropic_ot",
        manifold_contraction_rate=0.05,
    )
    state = _task_state([[0.8, 0.0]], [0.1])
    state.elapsed_steps = int(elapsed_steps)
    backend = SimpleNamespace(states=np.zeros((4, 4, 3), dtype=np.float32))
    backend.states[:, 3, :] = np.array(
        [[-2.0, -1.0, 1.0], [-2.0, 0.0, 1.0], [-2.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    backend.states[:, 1, 2] = 0.0
    return SimpleNamespace(
        task=task,
        task_state=state,
        prev_backend_state=backend,
        step_count=int(step_count),
        _episode_return=0.0,
        _episode_len=1,
    )


def test_manifold_replan_on_rho_shrink() -> None:
    env = _trajectory_planner_env(elapsed_steps=0, step_count=1)
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    cfg = {
        "manifold_replan_curve_tol": 0.05,
        "manifold_replan_rho_tol": 0.001,
        "obstacle_avoidance": {"horizon_s": 0.5, "dt": 0.1},
    }
    trajectory_planner_actions_from_state(
        env,
        env.prev_backend_state.states[:, 3, :],
        low,
        high,
        cfg=cfg,
        pursuer_yaw=np.zeros(3, dtype=np.float32),
    )
    st = env._trajectory_planner_state
    first_paths = st.cached_pursuer_paths
    first_version = st.manifold_version

    env.task_state.elapsed_steps = 50
    trajectory_planner_actions_from_state(
        env,
        env.prev_backend_state.states[:, 3, :],
        low,
        high,
        cfg=cfg,
        pursuer_yaw=np.zeros(3, dtype=np.float32),
    )
    diag = env._obstacle_aware_diagnostics["trajectory_planner"]["manifold"]
    assert diag["manifold_replanned"] is True
    assert st.manifold_version == first_version + 1
    assert st.cached_pursuer_paths is not first_paths


def test_manifold_no_replan_when_unchanged() -> None:
    env = _trajectory_planner_env(elapsed_steps=10, step_count=2)
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    cfg = {
        "manifold_replan_curve_tol": 0.05,
        "manifold_replan_rho_tol": 0.001,
        "obstacle_avoidance": {"horizon_s": 0.5, "dt": 0.1},
    }
    trajectory_planner_actions_from_state(
        env,
        env.prev_backend_state.states[:, 3, :],
        low,
        high,
        cfg=cfg,
        pursuer_yaw=np.zeros(3, dtype=np.float32),
    )
    st = env._trajectory_planner_state
    first_paths = st.cached_pursuer_paths
    first_version = st.manifold_version

    trajectory_planner_actions_from_state(
        env,
        env.prev_backend_state.states[:, 3, :],
        low,
        high,
        cfg=cfg,
        pursuer_yaw=np.zeros(3, dtype=np.float32),
    )
    diag = env._obstacle_aware_diagnostics["trajectory_planner"]["manifold"]
    assert diag["manifold_replanned"] is False
    assert st.manifold_version == first_version
    assert st.cached_pursuer_paths is first_paths


def test_trajectory_planner_tracks_stabilized_slots_not_raw_slots() -> None:
    env = _trajectory_planner_env(elapsed_steps=0, step_count=1)
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    cfg = {
        "freeze_slots_after_first_step": True,
        "obstacle_avoidance": {"horizon_s": 0.5, "dt": 0.1},
    }

    trajectory_planner_actions_from_state(
        env,
        env.prev_backend_state.states[:, 3, :],
        low,
        high,
        cfg=cfg,
        pursuer_yaw=np.zeros(3, dtype=np.float32),
    )
    first_stable = np.asarray(
        env._obstacle_aware_diagnostics["deploy_control"]["pursuers"][0][
            "stabilized_slot_target_xy"
        ],
        dtype=np.float64,
    )

    env.step_count = 2
    env.prev_backend_state.states[3, 3, 0] = 1.0
    trajectory_planner_actions_from_state(
        env,
        env.prev_backend_state.states[:, 3, :],
        low,
        high,
        cfg=cfg,
        pursuer_yaw=np.zeros(3, dtype=np.float32),
    )
    pursuer_diag = env._obstacle_aware_diagnostics["deploy_control"]["pursuers"][0]
    tp_diag = pursuer_diag["trajectory_planner"]
    raw = np.asarray(pursuer_diag["raw_slot_target_xy"], dtype=np.float64)
    stable = np.asarray(pursuer_diag["stabilized_slot_target_xy"], dtype=np.float64)
    waypoint = np.asarray(tp_diag["tracking_waypoint_xy"], dtype=np.float64)

    np.testing.assert_allclose(stable, first_stable, atol=1e-6)
    assert float(np.linalg.norm(raw - stable)) > 1e-3
    np.testing.assert_allclose(waypoint, stable, atol=1e-6)
    assert tp_diag["slot_velocity_ff_applied"] is False
    assert len(pursuer_diag["backend_cmd_ground_xy"]) == 2
    assert len(pursuer_diag["backend_cmd_world_xy"]) == 2
    assert pursuer_diag["backend_cmd_action_layout"] == "[vx_ground, vy_ground, vr, vz_ground]"


def test_trajectory_planner_sends_ground_xy_velocity_without_yaw_rotation() -> None:
    env = _trajectory_planner_env(elapsed_steps=0, step_count=1)
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
    yaws = np.full(3, np.pi / 2.0, dtype=np.float32)

    actions = trajectory_planner_actions_from_state(
        env,
        env.prev_backend_state.states[:, 3, :],
        low,
        high,
        cfg={"obstacle_avoidance": {"horizon_s": 0.5, "dt": 0.1}},
        pursuer_yaw=yaws,
    )

    pursuer_diag = env._obstacle_aware_diagnostics["deploy_control"]["pursuers"][0]
    cmd_ground = np.asarray(pursuer_diag["backend_cmd_ground_xy"], dtype=np.float64)
    cmd_world = np.asarray(pursuer_diag["backend_cmd_world_xy"], dtype=np.float64)
    np.testing.assert_allclose(actions[0, :2], cmd_ground, atol=1e-6)
    np.testing.assert_allclose(cmd_world, cmd_ground, atol=1e-6)
    assert float(np.linalg.norm(cmd_ground)) > 1e-6


def test_debug_frame_includes_trajectory_planner_manifold_curve() -> None:
    env = _trajectory_planner_env(elapsed_steps=0, step_count=1)
    low = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)

    trajectory_planner_actions_from_state(
        env,
        env.prev_backend_state.states[:, 3, :],
        low,
        high,
        cfg={"obstacle_avoidance": {"horizon_s": 0.5, "dt": 0.1}},
        pursuer_yaw=np.zeros(3, dtype=np.float32),
    )
    info = {
        "termination_reason": "running",
        "obstacle_aware_diagnostics": env._obstacle_aware_diagnostics,
        "reference_manifold_curve": env.task_state.reference_manifold_curve,
        "reference_manifold_targets": env.task_state.reference_manifold_targets,
    }
    frame = build_debug_frame(
        env,
        info,
        event="step",
        extra={"viz": resolve_viz_profile({"trajectory_planner": {}})},
    )

    assert frame["viz"]["method"] == "trajectory_planner"
    assert frame["viz"]["manifold_only"] is True
    assert frame["deploy_control"]["local_planner"] == "trajectory_planner"
    assert "assigned_path_xy" not in frame["deploy_control"]["pursuers"][0]
    assert frame["role"]["assigned_targets"]
    assert len(frame["manifold"]["curve"]) >= 2
    assert "pursuer_curves" not in frame["manifold"]
    assert "pursuer_closed_curves" not in frame["manifold"]
