"""Tests for browser debug visualization helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle
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


def test_build_debug_frame_includes_control_timing():
    env = _fake_env()
    env._last_control_timing = {
        "manifold_update_time": 0.0012,
        "slot_assignment_time": 0.0008,
        "total_decision_latency": 0.0045,
        "control_frequency": 222.2,
        "nominal_control_hz": 50.0,
    }
    info = {"termination_reason": "running"}
    viz = resolve_viz_profile({"sce": {}})
    frame = build_debug_frame(env, info, event="step", extra={"viz": viz})
    ct = frame["control_timing"]
    assert ct["slot_assignment_time"] == 0.0008
    assert ct["manifold_update_time"] == 0.0012
    assert ct["total_decision_latency"] == 0.0045
    assert ct["control_frequency"] == 222.2
    assert ct["nominal_control_hz"] == 50.0


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
    env.prev_backend_state.states[0, 1, 2] = np.pi / 2.0
    info = {"termination_reason": "running"}
    frame = build_debug_frame(env, info, event="step")
    assert "kinematics" in frame
    agents = frame["kinematics"]["agents"]
    assert len(agents) == 4
    assert agents[0]["label"] == "P0"
    assert agents[0]["speed_xy"] == 1.0
    np.testing.assert_allclose(agents[0]["linear_ground"][:2], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(agents[0]["linear_world_xy"], [1.0, 0.0], atol=1e-6)
    assert abs(float(agents[0]["yaw_rad"]) - np.pi / 2.0) < 1e-6
    assert frame["kinematics"]["frame"] == "ground"
    assert agents[1]["speed_xy"] == 1.5
    assert agents[3]["label"] == "E"


def test_build_debug_frame_includes_obstacles_from_task_state():
    env = _fake_env()
    env.task_state.obstacle_xy = np.array([[1.0, 2.0], [-3.0, 0.5]], dtype=np.float32)
    env.task_state.obstacle_r = np.array([0.4, 0.6], dtype=np.float32)
    info = {"termination_reason": "running"}
    viz = resolve_viz_profile({"pure_pursuit": {}})
    frame = build_debug_frame(env, info, event="step", extra={"viz": viz})
    assert "obstacles" in frame
    assert len(frame["obstacles"]["xy"]) == 2
    np.testing.assert_allclose(frame["obstacles"]["r"], [0.4, 0.6], rtol=1e-5)


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
    hub.arm_start_gate()
    hub.start_run()
    state = hub.get_control_state()
    assert state["awaiting_start"] is False
    assert state["paused"] is False
    hub.set_run_plan(total_episodes=10)
    assert hub.get_control_state()["total_episodes"] == 10
    configure_debug_browser(enabled=False)
    assert get_debug_browser_hub() is None


def test_hub_publish_serializes_obstacle_diagnostics():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18775,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    sub = hub.subscribe()
    obs = Obstacle(
        kind="circle",
        center=np.array([1.0, 2.0], dtype=np.float64),
        radius=0.5,
    )
    hub.publish(
        {
            "event": "step",
            "step": 7,
            "deploy_control": {"pursuers": [{"local_obstacles": [obs]}]},
        }
    )
    payload = json.loads(sub.get(timeout=1.0))
    local_obs = payload["deploy_control"]["pursuers"][0]["local_obstacles"][0]
    assert local_obs == {
        "kind": "circle",
        "center": [1.0, 2.0],
        "radius": 0.5,
        "half_extents": None,
        "vertices": None,
    }
    configure_debug_browser(enabled=False)


def test_episode_marker_includes_step():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18774,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    sub = hub.subscribe()
    publish_episode_marker("episode_start", episode=2, total_episodes=5, seed=102)
    start = json.loads(sub.get(timeout=1.0))
    assert start["step"] == 0
    publish_episode_marker(
        "episode_end",
        episode=2,
        total_episodes=5,
        episode_len=120,
        capture=False,
    )
    end = json.loads(sub.get(timeout=1.0))
    assert end["step"] == 120
    configure_debug_browser(enabled=False)


def test_hub_episode_marker_carries_visual_snapshot():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18768,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    sub = hub.subscribe()
    hub.publish(
        {
            "event": "reset",
            "step": 0,
            "positions": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            "agent_labels": ["P0", "E"],
        }
    )
    sub.get(timeout=1.0)
    hub.publish({"event": "episode_start", "step": 0, "episode": 1})
    marker = json.loads(sub.get(timeout=1.0))
    assert marker["event"] == "episode_start"
    assert marker["positions"] == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    assert marker["agent_labels"] == ["P0", "E"]
    configure_debug_browser(enabled=False)


def test_hub_second_episode_auto_continues():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18772,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    hub.set_run_plan(total_episodes=3)
    hub.next_episode()
    hub.arm_start_gate()
    hub.start_run()
    hub.wait_if_blocked()
    assert hub._run_armed is True
    assert hub._episode_run_started is True

    hub.next_episode()
    state = hub.get_control_state()
    assert state["needs_start_click"] is False
    assert state["awaiting_start"] is False
    assert state["run_armed"] is True
    assert hub._episode_run_started is True
    hub.arm_start_gate()
    hub.wait_if_blocked()
    assert hub._is_blocked() is False
    configure_debug_browser(enabled=False)


def test_hub_second_episode_waits_when_gate_cleared():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18773,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    hub.set_run_plan(total_episodes=2)
    hub.next_episode()
    hub.arm_start_gate()
    hub.start_run()
    hub.wait_if_blocked()
    with hub._control_lock:
        hub._start_gate.clear()
        hub._run_armed = False
    hub.next_episode()
    hub.arm_start_gate()
    state = hub.get_control_state()
    assert state["needs_start_click"] is True
    assert state["awaiting_start"] is True
    hub.start_run()
    hub.wait_if_blocked()
    assert hub._episode_run_started is True
    configure_debug_browser(enabled=False)


def test_hub_arm_start_gate():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18769,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    hub.next_episode()
    hub.start_run()
    assert hub._start_gate.is_set()
    hub.arm_start_gate()
    state = hub.get_control_state()
    assert state["awaiting_start"] is False
    assert state["needs_start_click"] is False
    hub.wait_if_blocked()
    configure_debug_browser(enabled=False)


def test_hub_arm_start_gate_without_early_click():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18770,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    hub.next_episode()
    hub.arm_start_gate()
    state = hub.get_control_state()
    assert state["awaiting_start"] is True
    assert state["needs_start_click"] is True
    hub.start_run()
    state = hub.get_control_state()
    assert state["awaiting_start"] is False
    hub.wait_if_blocked()
    configure_debug_browser(enabled=False)


def test_hub_early_start_before_wait():
    configure_debug_browser(enabled=False)
    hub = configure_debug_browser(
        enabled=True,
        port=18771,
        autostart=False,
        start_paused=True,
    )
    assert hub is not None
    hub.start_run()
    hub.next_episode()
    hub.arm_start_gate()
    hub.wait_if_blocked()
    assert hub.get_control_state()["needs_start_click"] is False
    configure_debug_browser(enabled=False)
