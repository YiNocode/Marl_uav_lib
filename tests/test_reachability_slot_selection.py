"""Tests for reachability-aware candidate slot selection."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.planning.path_cache import DeployPathCache
from marl_uav.framework.reachability.reachability_slot_selection import (
    generate_candidate_slots,
    select_reachability_aware_slots,
)


def _circle(cx: float, cy: float, r: float) -> Obstacle:
    return Obstacle(kind="circle", center=np.array([cx, cy], dtype=np.float64), radius=r)


def test_slot_inside_obstacle_is_not_selected_as_candidate() -> None:
    evader = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    pursuers = np.array([[0.0, -5.0, 2.0], [5.0, 0.0, 2.0], [-5.0, 0.0, 2.0]], dtype=np.float32)
    candidates = generate_candidate_slots(
        evader,
        pursuers,
        [_circle(3.0, 0.0, 1.0)],
        {"num_candidate_slots": 4, "slot_radius_candidates": [3.0], "slot_min_clearance": 0.2},
        world_xy=20.0,
        safety_margin=0.1,
        uav_radius=0.15,
    )
    assert candidates
    assert all(not np.allclose(c.pos[:2], np.array([3.0, 0.0]), atol=1e-4) for c in candidates)
    assert all(not c.is_inside_obstacle for c in candidates)


def test_candidate_generation_preserves_all_scanned_angles_under_cap() -> None:
    evader = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    pursuers = np.array([[0.0, -5.0, 2.0], [5.0, 0.0, 2.0], [-5.0, 0.0, 2.0]], dtype=np.float32)
    candidates = generate_candidate_slots(
        evader,
        pursuers,
        [],
        {
            "num_candidate_slots": 12,
            "slot_radius_candidates": [2.5, 3.0, 3.5, 4.0],
            "max_candidates": 12,
            "preserve_all_angles": True,
        },
        world_xy=20.0,
        safety_margin=0.1,
        uav_radius=0.15,
    )
    assert len(candidates) == 12
    assert {c.angle_index for c in candidates} == set(range(12))


def test_too_few_safe_candidates_trigger_fallback() -> None:
    evader = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    pursuers = np.array([[-6.0, 0.0, 2.0], [0.0, -6.0, 2.0], [6.0, 0.0, 2.0]], dtype=np.float32)
    targets, assignment, diag = select_reachability_aware_slots(
        pursuers,
        evader,
        [_circle(0.0, 0.0, 10.0)],
        None,
        DeployPathCache(),
        (-20.0, -20.0, 20.0, 20.0),
        {"max_nodes": 0, "planner_timeout_ms": 5.0, "num_obstacle_samples": 8},
        world_xy=20.0,
        pursuer_speed=0.25,
        safety_margin=0.2,
        uav_radius=0.15,
        slot_cfg={
            "num_candidate_slots": 6,
            "slot_radius_candidates": [3.0],
            "allow_los_blocked_slots": True,
            "slot_min_clearance": 0.2,
        },
        assignment_cfg={"infeasible_cost": 1.0e6},
        structure_cfg={"max_slot_combinations": 20},
        switch_penalty=0.5,
    )
    assert targets.shape == (3, 3)
    assert assignment.shape == (3,)
    assert bool(diag["fallback_slot_selection"])


def test_infeasible_pairs_receive_large_assignment_cost() -> None:
    evader = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    pursuers = np.array([[-6.0, 0.0, 2.0], [0.0, -6.0, 2.0], [6.0, 0.0, 2.0]], dtype=np.float32)
    targets, assignment, diag = select_reachability_aware_slots(
        pursuers,
        evader,
        [_circle(0.0, 0.0, 2.0)],
        None,
        DeployPathCache(),
        (-20.0, -20.0, 20.0, 20.0),
        {"max_nodes": 0, "planner_timeout_ms": 5.0, "num_obstacle_samples": 8},
        world_xy=20.0,
        pursuer_speed=0.25,
        safety_margin=0.2,
        uav_radius=0.15,
        slot_cfg={
            "num_candidate_slots": 6,
            "slot_radius_candidates": [3.0],
            "allow_los_blocked_slots": True,
            "slot_min_clearance": 0.2,
        },
        assignment_cfg={"infeasible_cost": 1.0e6},
        structure_cfg={"max_slot_combinations": 20},
        switch_penalty=0.5,
    )
    assert targets.shape == (3, 3)
    assert assignment.shape == (3,)
    assert np.max(diag["reachability_assignment_cost_matrix"]) >= 1.0e6


def test_actual_slots_selected_before_pair_assignment() -> None:
    evader = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    actual = np.array(
        [
            [3.0, 0.0, 2.0],
            [-1.5, 2.598076, 2.0],
            [-1.5, -2.598076, 2.0],
        ],
        dtype=np.float32,
    )
    pursuers = actual.copy()
    pursuers[:, :2] *= 1.15
    targets, assignment, diag = select_reachability_aware_slots(
        pursuers,
        evader,
        [],
        None,
        DeployPathCache(),
        (-20.0, -20.0, 20.0, 20.0),
        {"max_nodes": 0, "planner_timeout_ms": 5.0, "num_obstacle_samples": 8},
        world_xy=20.0,
        pursuer_speed=0.25,
        safety_margin=0.2,
        uav_radius=0.15,
        slot_cfg={
            "num_candidate_slots": 3,
            "slot_radius_candidates": [3.0],
            "allow_los_blocked_slots": True,
            "slot_min_clearance": 0.2,
        },
        assignment_cfg={"infeasible_cost": 1.0e6, "w_risk": 0.0, "w_turn": 0.0},
        structure_cfg={"max_slot_combinations": 20, "w_slot_risk": 0.0},
        switch_penalty=0.5,
    )
    assert targets.shape == (3, 3)
    assert diag["actual_slot_positions"] == targets.tolist()
    assert diag["selected_assignment_cost_matrix"].shape == (3, 3)
    np.testing.assert_array_equal(assignment, np.array([0, 1, 2], dtype=np.int64))


def test_actual_slot_selection_evaluates_combinations_beyond_old_prefix_cap() -> None:
    evader = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    pursuers = np.array([[0.0, -6.0, 2.0], [5.0, 3.0, 2.0], [-5.0, 3.0, 2.0]], dtype=np.float32)
    targets, _assignment, diag = select_reachability_aware_slots(
        pursuers,
        evader,
        [],
        None,
        DeployPathCache(),
        (-20.0, -20.0, 20.0, 20.0),
        {"max_nodes": 0, "planner_timeout_ms": 5.0, "num_obstacle_samples": 8},
        world_xy=20.0,
        pursuer_speed=0.25,
        safety_margin=0.2,
        uav_radius=0.15,
        slot_cfg={
            "num_candidate_slots": 6,
            "slot_radius_candidates": [3.0],
            "max_candidates": 6,
            "preserve_all_angles": True,
        },
        assignment_cfg={"infeasible_cost": 1.0e6, "w_risk": 0.0, "w_turn": 0.0},
        structure_cfg={
            "max_slot_combinations": 1,
            "w_slot_risk": 0.0,
            "w_assignment_feasibility": 0.0,
        },
        switch_penalty=0.5,
    )
    assert int(diag["num_evaluated_slot_combinations"]) > 1
    assert targets.shape == (3, 3)
    assert float(diag["candidate_D_ang"]) > 0.9
    assert float(diag["candidate_C_col"]) < 0.2


def test_reachability_assignment_lock_preserves_reachable_slots() -> None:
    evader = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    pursuers = np.array([[0.0, -6.0, 2.0], [6.0, 0.0, 2.0], [0.0, 6.0, 2.0]], dtype=np.float32)
    slot_desc = [
        {"angle_index": 0, "radius": 3.0},
        {"angle_index": 2, "radius": 3.0},
        {"angle_index": 4, "radius": 3.0},
    ]
    prev_assignment = np.array([2, 0, 1], dtype=np.int64)
    targets, assignment, diag = select_reachability_aware_slots(
        pursuers,
        evader,
        [],
        None,
        DeployPathCache(),
        (-20.0, -20.0, 20.0, 20.0),
        {"max_nodes": 0, "planner_timeout_ms": 5.0, "num_obstacle_samples": 8},
        world_xy=20.0,
        pursuer_speed=0.25,
        safety_margin=0.2,
        uav_radius=0.15,
        slot_cfg={
            "num_candidate_slots": 6,
            "slot_radius_candidates": [3.0],
            "max_candidates": 6,
            "preserve_all_angles": True,
        },
        assignment_cfg={"infeasible_cost": 1.0e6, "w_risk": 0.0, "w_turn": 0.0},
        structure_cfg={
            "preserve_assignment_until_unreachable": True,
            "w_assignment_feasibility": 10.0,
        },
        switch_penalty=0.5,
        previous_slot_descriptors=slot_desc,
        previous_slot_assignment=prev_assignment,
    )

    assert targets.shape == (3, 3)
    np.testing.assert_array_equal(assignment, prev_assignment)
    assert bool(diag["assignment_lock_preserved"])
    assert diag["selected_candidate_descriptors"] == slot_desc
