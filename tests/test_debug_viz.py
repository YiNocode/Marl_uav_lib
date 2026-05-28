"""Tests for debug visualization capability profiles."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from marl_uav.utils.debug_viz import (
    build_controller_targets,
    filter_algorithm_fields,
    resolve_viz_profile,
)


def test_pure_pursuit_profile_disables_manifold():
    cfg = {"pure_pursuit": {"xy_gain": 0.25}, "benchmark": {"method": "pure_pursuit"}}
    viz = resolve_viz_profile(cfg)
    assert viz["method"] == "pure_pursuit"
    assert viz["manifold_curve"] is False
    assert viz["role_allocation"] is False
    assert viz["pursuit_targets"] is True


def test_sce_profile_enables_manifold_and_ot():
    cfg = {"sce": {"xy_gain": 0.25}}
    viz = resolve_viz_profile(cfg)
    assert viz["manifold_curve"] is True
    assert viz["ot_details"] is True


def test_build_controller_targets_pure_pursuit():
    viz = resolve_viz_profile({"pure_pursuit": {}})
    task_state = SimpleNamespace(pursuer_ids=np.array([0, 1, 2]), evader_id=3)
    positions = np.array(
        [[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [5.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    backend_state = SimpleNamespace(states=np.zeros((4, 4, 3), dtype=np.float32))
    backend_state.states[:, 3, :] = positions
    env = SimpleNamespace(task_state=task_state, prev_backend_state=backend_state)
    out = build_controller_targets(env, viz=viz, controller_cfg={})
    assert out is not None
    assert out["kind"] == "evader"
    assert len(out["targets"]) == 3


def test_filter_algorithm_fields_for_pure_pursuit():
    viz = resolve_viz_profile({"pure_pursuit": {}})
    algo = {
        "method": "pure_pursuit",
        "task_name": "Task",
        "capture_dist": 1.0,
        "pursuer_speed_xy": 1.5,
        "evader_speed_xy": 2.0,
        "manifold_contraction_rate": 0.01,
        "ot_epsilon": 0.05,
        "role_assignment_mode": "entropic_ot",
    }
    filtered = filter_algorithm_fields(algo, viz)
    assert "manifold_contraction_rate" not in filtered
    assert "ot_epsilon" not in filtered
    assert filtered["capture_dist"] == 1.0
