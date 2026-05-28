"""Tests for browser debug visualization helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from marl_uav.utils.debug_browser import (
    DebugBrowserHub,
    build_debug_frame,
    configure_debug_browser,
    get_debug_browser_hub,
    publish_episode_marker,
)
from marl_uav.utils.debug_viz import resolve_viz_profile


class _FakeTask:
    world_xy = 20.0
    z_min = 0.5
    z_max = 5.0
    role_assignment_mode = "entropic_ot"
    manifold_target_phase = 0.0
    manifold_target_radius_scale = 1.0
    manifold_contraction_rate = 0.01
    manifold_structure_gate_scale = 0.75
    ot_epsilon = 0.05
    ot_epsilon_scale = 0.25
    ot_sinkhorn_iterations = 25
    assignment_inertia_margin = 0.05
    capture_dist = 1.0
    pursuer_speed_xy = 1.5
    evader_speed_xy = 2.0

    def _assigned_targets_from_state(self, pursuer_pos, evader_pos, task_state=None):
        targets = np.stack(
            [
                evader_pos + np.array([2.0, 0.0, 0.0], dtype=np.float32),
                evader_pos + np.array([0.0, 2.0, 0.0], dtype=np.float32),
                evader_pos + np.array([-2.0, 0.0, 0.0], dtype=np.float32),
            ],
            axis=0,
        )
        assignment = np.array([0, 1, 2], dtype=np.int64)
        return targets, assignment, targets[assignment]


def _fake_env():
    pursuer_ids = np.array([0, 1, 2], dtype=np.int64)
    task_state = SimpleNamespace(
        pursuer_ids=pursuer_ids,
        evader_id=3,
        latest_target_radius_xy=2.0,
        prev_mean_radius_xy=2.1,
        structure_hold_steps=3,
    )
    positions = np.array(
        [
            [-1.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    backend_state = SimpleNamespace(states=np.zeros((4, 4, 3), dtype=np.float32))
    backend_state.states[:, 3, :] = positions
    backend_state.states[:, 2, :] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.8, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    env = SimpleNamespace(
        task=_FakeTask(),
        task_state=task_state,
        prev_backend_state=backend_state,
        step_count=5,
        _episode_return=1.5,
        _episode_len=5,
    )
    return env


def test_build_debug_frame_includes_manifold_and_ot():
    env = _fake_env()
    info = {
        "pursuit_structure": {"C_cov": 0.8, "C_col": 0.2, "D_ang": 0.7},
        "reference_manifold_curve": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        "reference_manifold_targets": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        "termination_reason": "running",
    }
    viz = {
        "method": "sce",
        "manifold_curve": True,
        "slot_targets": True,
        "role_allocation": True,
        "ot_details": True,
        "structure_metrics": True,
        "obstacles": True,
    }
    frame = build_debug_frame(env, info, event="step", extra={"viz": viz})
    assert frame["event"] == "step"
    assert frame["viz"]["method"] == "sce"
    assert "manifold" in frame
    assert "role" in frame
    assert frame["role"]["role_assignment"] == [0, 1, 2]
    assert "ot" in frame["role"]


def test_build_debug_frame_pure_pursuit_omits_manifold():
    env = _fake_env()
    info = {
        "pursuit_structure": {"C_cov": 0.8, "C_col": 0.2, "D_ang": 0.7},
        "reference_manifold_curve": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        "reference_manifold_targets": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        "termination_reason": "running",
    }
    viz = resolve_viz_profile({"pure_pursuit": {}})
    frame = build_debug_frame(env, info, event="step", extra={"viz": viz})
    assert "manifold" not in frame
    assert "role" not in frame
    assert "pursuit_structure" not in frame
    assert frame["controller_targets"]["kind"] == "evader"


def test_build_debug_frame_includes_kinematics():
    env = _fake_env()
    info = {"termination_reason": "running"}
    frame = build_debug_frame(env, info, event="step")
    assert "kinematics" in frame
    agents = frame["kinematics"]["agents"]
    assert len(agents) == 4
    assert agents[0]["label"] == "P0"
    assert agents[0]["speed_xy"] == 1.0
    assert agents[1]["speed_xy"] == 1.5
    assert agents[3]["label"] == "E"


def test_hub_capture_rate_stats():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18767,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    hub.set_run_plan(total_episodes=4)
    publish_episode_marker("episode_end", capture=True, episode=1)
    stats = hub.get_run_stats()
    assert stats["captured_episodes"] == 1
    assert stats["completed_episodes"] == 1
    assert stats["capture_rate"] == 1.0
    publish_episode_marker("episode_end", capture=False, episode=2)
    stats = hub.get_run_stats()
    assert stats["captured_episodes"] == 1
    assert stats["completed_episodes"] == 2
    assert stats["capture_rate"] == 0.5
    state = hub.get_control_state()
    assert state["run_stats"]["capture_rate"] == 0.5
    configure_debug_browser(enabled=False)


def test_hub_publish_roundtrip():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18766,
        autostart=False,
        playback_speed=0.5,
        start_paused=True,
    )
    assert hub is not None
    state = hub.get_control_state()
    assert state["awaiting_start"] is True
    assert state["paused"] is True
    sub = hub.subscribe()
    hub.publish({"event": "test", "step": 1})
    msg = sub.get(timeout=1.0)
    payload = json.loads(msg)
    assert payload["step"] == 1
    hub.start_run()
    state = hub.get_control_state()
    assert state["awaiting_start"] is False
    assert state["paused"] is False
    hub.set_run_plan(total_episodes=10)
    assert hub.get_control_state()["total_episodes"] == 10
    configure_debug_browser(enabled=False)
    assert get_debug_browser_hub() is None
