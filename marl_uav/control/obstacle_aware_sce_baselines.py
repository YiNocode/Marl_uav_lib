"""Real-time deployable obstacle-aware SCE baselines.

Deployable methods (real-time):
- sce_los_slot: LOS-aware OT assignment + direct slot tracking
- sce_cached_path_slot: cached/event-triggered path + waypoint tracking
- sce_cached_path_cbf_slot: cached path + lightweight CBF projection
- sce_reachability_slot: candidate slots + reachability-aware assignment + path tracking
- sce_reachability_cbf_slot: reachability-aware path tracking + local CBF safety filter

Offline upper-bound only (NOT deployable):
- sce_exact_path_oracle / sce_exact_path_qp_oracle — see obstacle_aware_oracle_baselines.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from marl_uav.control.altitude_hold import apply_hard_altitude_to_action_row
from marl_uav.control.boundary_utils import apply_xy_boundary_barrier
from marl_uav.control.geometric_pursuit_baselines import (
    analyze_tube_tracking_speed,
    default_proportional_gains,
    proportional_actions_to_targets,
    pursuer_yaws_from_backend,
    tube_tracking_body_actions,
)
from marl_uav.framework.geometry.obstacle_adapter import (
    manifold_influencing_obstacles,
    obstacle_index,
    obstacles_from_task_state,
)
from marl_uav.framework.geometry.obstacle_query import (
    ObstacleQueryConfig,
    obstacles_in_corridor,
    select_plan_obstacles,
    select_validation_obstacles,
)
from marl_uav.framework.geometry.obstacle_geometry import (
    collision_check_path,
    has_line_of_sight,
    path_length,
)
from marl_uav.framework.planning.path_cache import DeployPathCache, PathCacheConfig
from marl_uav.framework.planning.path_tracking import (
    adjust_lookahead_for_turn_safety,
    points_min_boundary_clearance,
    points_min_obstacle_clearance,
    select_lookahead_waypoint,
)
from marl_uav.framework.planning.turn_radius_slot_planner import TurnRadiusSlotController
from marl_uav.framework.reachability.assignment_cost import (
    ReachabilityConfig,
    build_reachability_cost_matrix,
)
from marl_uav.framework.reachability.reachability_slot_selection import (
    select_reachability_aware_slots,
)
from marl_uav.framework.reachability.slot_projection import (
    SlotProjectionConfig,
    detect_slot_projection_moves,
    project_manifold_slots,
)
from marl_uav.framework.reachability.structure_slot_selection import (
    StructureSlotSelectionConfig,
    select_structure_manifold_slots,
)
from marl_uav.framework.role_allocation import default_ot_epsilon, entropic_ot_assignment
from marl_uav.framework.safety.cbf_filter import CBFConfig, LightweightCBFFilter
from marl_uav.framework.utils.step_timing import StepTimingRecorder
from marl_uav.utils.control_timing import should_record_control_timing

DeployableKind = Literal[
    "sce_los_slot",
    "sce_turn_radius_slot",
    "sce_cached_path_slot",
    "sce_cached_path_cbf_slot",
    "sce_reachability_slot",
    "sce_reachability_cbf_slot",
]


@dataclass
class RuntimeRatesConfig:
    control_hz: float = 50.0
    manifold_update_hz: float = 10.0
    assignment_update_hz: float = 5.0
    path_replan_hz: float = 1.0
    cbf_hz: float = 50.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RuntimeRatesConfig:
        d = dict(raw or {})
        return cls(
            control_hz=float(d.get("control_hz", 50.0)),
            manifold_update_hz=float(d.get("manifold_update_hz", 10.0)),
            assignment_update_hz=float(d.get("assignment_update_hz", 5.0)),
            path_replan_hz=float(d.get("path_replan_hz", 1.0)),
            cbf_hz=float(d.get("cbf_hz", 50.0)),
        )

    def every_n_steps(self, rate_hz: float, control_hz: float) -> int:
        if rate_hz <= 0:
            return 1
        return max(1, int(round(control_hz / rate_hz)))


@dataclass
class DeployControllerState:
    path_cache: DeployPathCache = field(default_factory=DeployPathCache)
    timing: StepTimingRecorder = field(default_factory=StepTimingRecorder)
    cbf_filter: LightweightCBFFilter = field(default_factory=LightweightCBFFilter)
    step_counter: int = 0
    cached_targets: np.ndarray | None = None
    cached_manifold_curve: np.ndarray | None = None
    cached_slot_selection_diag: dict[str, Any] | None = None
    last_manifold_radius_scale: float | None = None
    last_effective_targets: np.ndarray | None = None
    slot_projection_moved: np.ndarray | None = None
    cached_assignment: np.ndarray | None = None
    episode_stats: dict[str, list[float]] = field(default_factory=dict)
    replans_this_step: int = 0
    cbf_consecutive_steps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    assignment_switch_count: int = 0
    cached_candidate_assignment: np.ndarray | None = None
    cached_candidate_descriptors: list[dict[str, Any]] | None = None
    cached_slot_descriptors: list[dict[str, Any]] | None = None
    cached_reachability_diag: dict[str, Any] | None = None
    last_world_cmd_dirs: np.ndarray = field(default_factory=lambda: np.full((3, 2), np.nan, dtype=np.float64))
    turn_radius_prev_actions: np.ndarray = field(default_factory=lambda: np.zeros((3, 2), dtype=np.float64))

    def reset_episode(self) -> None:
        self.path_cache.clear()
        self.timing.reset_episode()
        self.step_counter = 0
        self.cached_targets = None
        self.cached_manifold_curve = None
        self.cached_slot_selection_diag = None
        self.last_manifold_radius_scale = None
        self.last_effective_targets = None
        self.slot_projection_moved = None
        self.cached_assignment = None
        self.replans_this_step = 0
        self.cbf_consecutive_steps = np.zeros(3, dtype=np.int64)
        self.assignment_switch_count = 0
        self.cached_candidate_assignment = None
        self.cached_candidate_descriptors = None
        self.cached_slot_descriptors = None
        self.cached_reachability_diag = None
        self.last_world_cmd_dirs = np.full((3, 2), np.nan, dtype=np.float64)
        self.turn_radius_prev_actions = np.zeros((3, 2), dtype=np.float64)
        self.episode_stats = _empty_episode_stats()


def _empty_episode_stats() -> dict[str, list[float]]:
    return {
        "assigned_pair_blocked_los": [],
        "path_cache_hit_rate": [],
        "num_replans_this_step": [],
        "cbf_active": [],
        "cbf_active_rate": [],
        "cbf_active_consecutive_steps": [],
        "cbf_correction_norm": [],
        "nominal_action_norm": [],
        "filtered_action_norm": [],
        "cbf_filter_time_ms": [],
        "decision_total_ms": [],
        "slot_reachable_rate": [],
        "mean_time_to_slot": [],
        "max_time_to_slot": [],
        "path_clearance_min": [],
        "path_clearance_mean": [],
        "path_risk_integral": [],
        "slot_behind_obstacle_rate": [],
        "los_blocked_slot_rate": [],
        "stale_path_rate": [],
        "path_endpoint_error": [],
        "path_min_clearance": [],
        "path_tracking_error": [],
        "turn_safety_active_rate": [],
        "turn_arc_min_clearance": [],
        "turn_boundary_min_clearance": [],
        "turn_boundary_unsafe_rate": [],
        "turn_angle_rad": [],
        "safety_speed_limit_rate": [],
        "safety_speed_cap_mean": [],
        "safety_speed_clearance_min": [],
        "assignment_lock_preserved_rate": [],
        "assignment_switch_count": [],
        "fallback_slot_selection_rate": [],
        "unreachable_slot_rate": [],
        "local_obstacle_count": [],
        "candidate_count": [],
        "valid_candidate_count": [],
        "local_planner_blocked": [],
        "local_planner_time_ms": [],
        "best_candidate_cost": [],
        "best_candidate_speed": [],
        "best_candidate_yaw_rate": [],
        "min_predicted_clearance": [],
        "assigned_slot_distance": [],
        "selected_action_norm": [],
    }


def _get_state(env: Any, path_cache_cfg: dict[str, Any]) -> DeployControllerState:
    st = getattr(env, "_deploy_sce_state", None)
    if st is None:
        st = DeployControllerState(
            path_cache=DeployPathCache(path_cache_cfg),
            episode_stats=_empty_episode_stats(),
        )
        env._deploy_sce_state = st
    return st


def _maybe_reset(env: Any, st: DeployControllerState) -> None:
    ep_step = int(getattr(env, "step_count", 0))
    last = getattr(env, "_deploy_sce_last_step", None)
    if ep_step == 0 or last is None or ep_step < int(last):
        st.reset_episode()
    env._deploy_sce_last_step = ep_step


def _world_bounds(task: Any) -> tuple[float, float, float, float]:
    w = float(getattr(task, "world_xy", 20.0))
    return (-w, -w, w, w)


def _uav_radius(task: Any, rcfg: ReachabilityConfig) -> float:
    if hasattr(task, "_pursuer_obstacle_hit_radius"):
        return float(task._pursuer_obstacle_hit_radius())
    return float(rcfg.uav_radius)


def _perceived_obstacles(
    all_obstacles: list,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    rcfg: ReachabilityConfig,
) -> tuple[list, dict[str, float | str]]:
    """Return the single obstacle set used by planning, LOS, scoring, and CBF.

    In global mode the controller sees all obstacles.  In local mode every
    high-level method shares the same perceived subset; only environment
    termination remains based on the true global set.
    """
    mode = str(getattr(rcfg, "obstacle_mode", "global")).strip().lower()
    diag: dict[str, float | str] = {
        "obstacle_mode": mode,
        "num_global_obstacles": float(len(all_obstacles)),
        "num_planning_obstacles": float(len(all_obstacles)),
    }
    if mode != "local":
        diag["obstacle_mode"] = "global"
        return list(all_obstacles), diag

    anchors = np.vstack([
        np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)[:, :2],
        np.asarray(evader_pos, dtype=np.float64).reshape(3)[None, :2],
    ])
    radius = float(max(getattr(rcfg, "local_obstacle_radius", 20.0), 0.0))
    scored: list[tuple[float, int, Any]] = []
    for idx, obs in enumerate(all_obstacles):
        if getattr(obs, "kind", None) != "circle":
            scored.append((0.0, idx, obs))
            continue
        center = np.asarray(obs.center, dtype=np.float64).reshape(2)
        surface = float(np.min(np.linalg.norm(anchors - center[None, :], axis=1))) - float(obs.radius)
        if surface <= radius:
            scored.append((surface, idx, obs))
    scored.sort(key=lambda x: (x[0], x[1]))
    top_k = getattr(rcfg, "local_obstacle_top_k", None)
    if top_k is not None:
        scored = scored[: max(int(top_k), 0)]
    perceived = [obs for _d, _idx, obs in scored]
    diag.update({
        "num_planning_obstacles": float(len(perceived)),
        "local_obstacle_radius": radius,
        "local_obstacle_top_k": -1.0 if top_k is None else float(top_k),
    })
    return perceived, diag


def _ot_assign(cost: np.ndarray, task: Any, prev: np.ndarray | None) -> np.ndarray:
    eps = default_ot_epsilon(cost, task.ot_epsilon, task.ot_epsilon_scale)
    return entropic_ot_assignment(
        cost,
        epsilon=eps,
        num_iters=task.ot_sinkhorn_iterations,
        prev_assignment=prev,
        inertia_margin=task.assignment_inertia_margin,
    )


def _body_vel_to_world(vx_b: float, vy_b: float, yaw: float) -> tuple[float, float]:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return vx_b * c - vy_b * s, vx_b * s + vy_b * c


def _world_vel_to_body(vx_w: float, vy_w: float, yaw: float) -> tuple[float, float]:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return vx_w * c + vy_w * s, -vx_w * s + vy_w * c


def _wrap_to_pi_scalar(x: float) -> float:
    return float((float(x) + np.pi) % (2.0 * np.pi) - np.pi)


def _apply_turn_slowdown_actions(
    actions: np.ndarray,
    turn_agent_debug: list[dict[str, Any]],
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    yaws: np.ndarray | None,
    slow_speed: float,
    yaw_gain: float,
) -> np.ndarray:
    """Immediately slow and steer when a path angle switch is unsafe.

    The high-level planner decides that a new slot/path direction is needed, but
    tube tracking otherwise cruises along the old path tangent at full speed.
    During that transition we command a small velocity in the new direction so
    the UAV turns before it spends lateral radius near obstacles or boundaries.
    CBF/boundary layers still run after this; this hook only prevents the
    nominal controller from fighting the planned turn.
    """
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    out = np.asarray(actions, dtype=np.float32).reshape(3, -1).copy()
    vmax = float(np.max(np.abs(high[:2]))) if high.size >= 2 else float(slow_speed)
    speed = float(np.clip(float(slow_speed), 0.0, max(vmax, 1e-6)))
    if speed <= 1e-9:
        return out

    for i, dbg in enumerate(turn_agent_debug[:3]):
        if not bool(dbg.get("turn_safety_active", False)):
            continue
        desired = np.asarray(dbg.get("turn_desired_dir_xy", []), dtype=np.float64).reshape(-1)
        if desired.size != 2 or not np.all(np.isfinite(desired)):
            desired = np.asarray(dbg.get("turn_adjusted_dir_xy", []), dtype=np.float64).reshape(-1)
        if desired.size != 2 or not np.all(np.isfinite(desired)):
            continue
        nrm = float(np.linalg.norm(desired))
        if nrm < 1e-9:
            continue
        desired = desired / nrm
        u_world = desired * speed
        if yaws is not None:
            yaw_i = float(yaws[i])
            bx, by = _world_vel_to_body(float(u_world[0]), float(u_world[1]), yaw_i)
            out[i, 0] = float(bx)
            out[i, 1] = float(by)
            if out.shape[1] >= 3 and float(yaw_gain) > 0.0:
                yaw_ref = float(np.arctan2(float(desired[1]), float(desired[0])))
                out[i, 2] = float(yaw_gain) * _wrap_to_pi_scalar(yaw_ref - yaw_i)
        else:
            out[i, 0] = float(u_world[0])
            out[i, 1] = float(u_world[1])
        dbg["turn_slowdown_active"] = True
        dbg["turn_slowdown_speed"] = speed
        dbg["turn_slowdown_dir_xy"] = desired.astype(float).tolist()

    return np.clip(out, low[None, :], high[None, :]).astype(np.float32)


def _speed_cap_from_clearance(
    clearance: float,
    *,
    stop_clearance: float,
    slow_clearance: float,
    free_clearance: float,
    min_speed: float,
    cruise_speed: float,
    max_speed: float,
) -> float:
    if not np.isfinite(float(clearance)):
        return float(max_speed)
    c = float(clearance)
    stop = float(stop_clearance)
    slow = max(float(slow_clearance), stop + 1e-6)
    free = max(float(free_clearance), slow + 1e-6)
    if c <= stop:
        return max(float(min_speed), 0.0)
    if c <= slow:
        t = (c - stop) / (slow - stop)
        return float(min_speed) + t * (float(cruise_speed) - float(min_speed))
    if c <= free:
        t = (c - slow) / (free - slow)
        return float(cruise_speed) + t * (float(max_speed) - float(cruise_speed))
    return float(max_speed)


def _apply_safety_speed_constraints(
    actions: np.ndarray,
    pursuer_pos: np.ndarray,
    assigned_targets: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    obstacles: list,
    *,
    yaws: np.ndarray | None,
    world_xy: float | None,
    boundary_margin: float,
    uav_radius: float,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Cap nominal speed from predicted obstacle/boundary clearance.

    CBF is a local safety filter; it should not be the first component that
    notices a full-speed command is about to cut a corner.  This layer samples a
    short forward tube in the commanded direction and lowers speed before the
    vehicle reaches the obstacle/boundary, while still allowing faster motion
    once the structured slot formation is nearly established.
    """
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    out = np.asarray(actions, dtype=np.float32).reshape(3, -1).copy()
    pos = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
    slots = np.asarray(assigned_targets, dtype=np.float64).reshape(3, 3)
    vmax = float(np.max(np.abs(high[:2]))) if high.size >= 2 else 1.0

    free_clearance = float(cfg.get("safety_speed_free_clearance", 2.0))
    slow_clearance = float(cfg.get("safety_speed_slow_clearance", 1.0))
    stop_clearance = float(cfg.get("safety_speed_stop_clearance", max(0.35, float(uav_radius) + 0.15)))
    min_speed = float(cfg.get("safety_speed_min", min(0.04, vmax)))
    safety_max = float(cfg.get("safety_speed_max", vmax))
    cruise_speed = float(cfg.get("safety_speed_cruise", min(vmax, max(min_speed, vmax * 0.65))))
    horizon = float(cfg.get("safety_speed_prediction_horizon", 2.0))
    min_forward = float(cfg.get("safety_speed_min_forward_dist", 0.45))
    samples = max(int(cfg.get("safety_speed_samples", 9)), 3)
    structure_enabled = bool(cfg.get("structure_speed_enabled", True))
    structure_cruise = float(cfg.get("structure_cruise_speed", cruise_speed))
    structure_capture = float(cfg.get("structure_capture_speed", min(vmax, safety_max)))
    structure_dist = float(cfg.get("structure_capture_dist", 0.75))

    debug: list[dict[str, Any]] = [{}, {}, {}]
    for i in range(3):
        yaw_i = float(yaws[i]) if yaws is not None else 0.0
        u_world = np.asarray(out[i, :2], dtype=np.float64)
        if yaws is not None:
            wx, wy = _body_vel_to_world(float(out[i, 0]), float(out[i, 1]), yaw_i)
            u_world = np.array([wx, wy], dtype=np.float64)
        speed = float(np.linalg.norm(u_world))
        slot_dist = float(np.linalg.norm(slots[i, :2] - pos[i, :2]))
        structure_cap = min(vmax, safety_max)
        if structure_enabled:
            structure_cap = structure_capture if slot_dist <= structure_dist else structure_cruise
        max_allowed = float(np.clip(structure_cap, 0.0, vmax))
        if speed <= 1e-9:
            debug[i] = {
                "safety_speed_active": False,
                "safety_speed_cap": max_allowed,
                "slot_dist_xy": slot_dist,
            }
            continue

        direction = u_world / speed
        forward_dist = max(min_forward, speed * max(horizon, 0.0))
        ts = np.linspace(0.0, 1.0, samples)
        pts = pos[i, :2][None, :] + ts[:, None] * forward_dist * direction[None, :]
        obs_clear = points_min_obstacle_clearance(pts, obstacles, uav_radius=uav_radius)
        boundary_clear = points_min_boundary_clearance(
            pts, world_xy=world_xy, boundary_margin=boundary_margin,
        )
        forward_clear = min(float(obs_clear), float(boundary_clear))
        clearance_cap = _speed_cap_from_clearance(
            forward_clear,
            stop_clearance=stop_clearance,
            slow_clearance=slow_clearance,
            free_clearance=free_clearance,
            min_speed=min_speed,
            cruise_speed=min(cruise_speed, max_allowed),
            max_speed=max_allowed,
        )
        cap = float(np.clip(clearance_cap, 0.0, max_allowed))
        active = bool(speed > cap + 1e-6)
        if active:
            u_world = direction * cap
            if yaws is not None:
                bx, by = _world_vel_to_body(float(u_world[0]), float(u_world[1]), yaw_i)
                out[i, 0] = float(bx)
                out[i, 1] = float(by)
            else:
                out[i, 0] = float(u_world[0])
                out[i, 1] = float(u_world[1])
        debug[i] = {
            "safety_speed_active": active,
            "safety_speed_cap": cap,
            "safety_speed_nominal": speed,
            "safety_speed_filtered": float(np.linalg.norm(u_world)),
            "safety_speed_forward_clearance": forward_clear,
            "safety_speed_obstacle_clearance": float(obs_clear),
            "safety_speed_boundary_clearance": float(boundary_clear),
            "slot_dist_xy": slot_dist,
            "structure_speed_cap": max_allowed,
        }

    return np.clip(out, low[None, :], high[None, :]).astype(np.float32), debug


def _apply_local_obstacle_escape(
    pos_xy: np.ndarray,
    u_world: np.ndarray,
    obstacles: list,
    *,
    uav_radius: float,
    safety_margin: float,
    extra_clearance: float,
    escape_speed: float,
    action_low_xy: np.ndarray,
    action_high_xy: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    u = np.asarray(u_world, dtype=np.float64).reshape(2).copy()
    low = np.asarray(action_low_xy, dtype=np.float64).reshape(2)
    high = np.asarray(action_high_xy, dtype=np.float64).reshape(2)
    trigger_clearance = max(float(uav_radius) + float(safety_margin) + float(extra_clearance), 1e-6)
    collision_clearance = max(float(uav_radius), 1e-6)

    best_clear = float("inf")
    best_dir: np.ndarray | None = None
    for obs in obstacles:
        if getattr(obs, "kind", None) != "circle":
            continue
        center = np.asarray(obs.center, dtype=np.float64).reshape(2)
        diff = p - center
        dist = float(np.linalg.norm(diff))
        surface_clear = dist - float(obs.radius)
        if surface_clear >= best_clear:
            continue
        best_clear = surface_clear
        if dist > 1e-9:
            best_dir = diff / dist
        else:
            fallback = -u
            nrm = float(np.linalg.norm(fallback))
            best_dir = fallback / nrm if nrm > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)

    if best_dir is None or best_clear >= trigger_clearance:
        return u, False, best_clear

    denom = max(trigger_clearance - collision_clearance, 1e-6)
    severity = float(np.clip((trigger_clearance - best_clear) / denom, 0.0, 1.0))
    min_outward = max(float(escape_speed), 0.0) * severity
    outward = float(np.dot(u, best_dir))
    if outward < min_outward:
        u = u + (min_outward - outward) * best_dir

    u = np.clip(u, low, high)
    vmax = float(np.max(np.abs(high))) if np.all(np.isfinite(high)) else float("inf")
    nrm = float(np.linalg.norm(u))
    if np.isfinite(vmax) and nrm > vmax and nrm > 1e-9:
        u = u * (vmax / nrm)
    return u, True, best_clear


def _resolve_cbf_config(
    ccfg: CBFConfig | None,
    rcfg: ReachabilityConfig,
    kind: DeployableKind,
    obstacles: list,
    uav_r: float,
) -> CBFConfig | None:
    """Obstacle avoidance config: explicit CBF yaml or default when obstacles present."""
    if ccfg is not None and ccfg.enabled:
        return replace(ccfg, uav_radius=max(float(ccfg.uav_radius), float(uav_r)))
    if len(obstacles) == 0:
        return None
    if kind not in ("sce_cached_path_cbf_slot", "sce_reachability_cbf_slot"):
        return None
    return CBFConfig(
        enabled=True,
        safety_margin=float(rcfg.safety_margin),
        uav_radius=float(uav_r),
    )


def _cbf_obstacle_threat(diag: dict[str, Any]) -> bool:
    if not bool(diag.get("cbf_active", False)):
        return False
    if int(diag.get("num_violated_before", 0)) > 0:
        return True
    if float(diag.get("cbf_delta_norm", 0.0)) > 1e-4:
        return True
    if bool(diag.get("cbf_timeout_or_fallback", False)):
        return True
    return len(diag.get("cbf_active_obstacle_indices", [])) > 0


def _apply_control_priority_layers(
    actions: np.ndarray,
    pursuer_pos: np.ndarray,
    track_targets: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    all_obstacles: list,
    other_agents_xy: list[np.ndarray],
    yaws: np.ndarray | None,
    ccfg: CBFConfig | None,
    cbf_filter: LightweightCBFFilter,
    cbf_every: int,
    step_counter: int,
    z_floor: float | None = None,
    z_ceiling: float | None = None,
    altitude_floor_margin: float = 0.25,
    altitude_ceiling_margin: float = 0.10,
    world_xy: float | None = None,
    boundary_margin: float = 0.30,
    boundary_alpha: float = 2.0,
) -> tuple[np.ndarray, list[dict[str, Any]], float, float, float]:
    """
    Execution priority (high → low):

    1. Obstacle avoidance (CBF on horizontal velocity)
    2. Slot / path tracking nominal (already in ``actions``)
    3. Hard altitude hold on vz (+ yaw/xy gate only when not avoiding)
    """
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    out = np.asarray(actions, dtype=np.float32).reshape(3, -1).copy()
    cbf_agent_debug: list[dict[str, Any]] = [{}, {}, {}]
    cbf_ms_total = 0.0
    cbf_active = 0.0
    cbf_delta = 0.0
    obstacle_threat = [False, False, False]
    boundary_threat = [False, False, False]
    local_escape_threat = [False, False, False]

    if ccfg is not None and ccfg.enabled:
        if step_counter % cbf_every == 1 or cbf_every == 1:
            deltas, actives, times = [], [], []
            for i in range(3):
                yaw_i = float(yaws[i]) if yaws is not None else 0.0
                u_world = np.asarray(out[i, :2], dtype=np.float64)
                if yaws is not None:
                    wx, wy = _body_vel_to_world(float(out[i, 0]), float(out[i, 1]), yaw_i)
                    u_world = np.array([wx, wy], dtype=np.float64)
                u_safe, cbf_diag = cbf_filter.filter(
                    pursuer_pos[i, :2], u_world, all_obstacles, other_agents_xy[i], ccfg,
                    uav_radius=ccfg.uav_radius,
                    action_low_xy=low[:2], action_high_xy=high[:2],
                    agent_yaw=yaw_i,
                    all_obstacles=all_obstacles,
                )
                if yaws is not None:
                    bx, by = _world_vel_to_body(float(u_safe[0]), float(u_safe[1]), yaw_i)
                    out[i, 0] = float(bx)
                    out[i, 1] = float(by)
                else:
                    out[i, 0] = float(u_safe[0])
                    out[i, 1] = float(u_safe[1])
                threat = _cbf_obstacle_threat(cbf_diag)
                obstacle_threat[i] = threat
                deltas.append(float(cbf_diag["cbf_delta_norm"]))
                actives.append(float(cbf_diag["cbf_active"]))
                times.append(float(cbf_diag["cbf_filter_time_ms"]))
                cbf_agent_debug[i] = {
                    "active_obstacle_indices": [
                        int(x) for x in cbf_diag.get("cbf_active_obstacle_indices", [])
                    ],
                    "cbf_delta_norm": float(cbf_diag["cbf_delta_norm"]),
                    "nominal_speed_xy": float(cbf_diag.get("cbf_nominal_speed_xy", 0.0)),
                    "safe_speed_xy": float(cbf_diag.get("cbf_safe_speed_xy", 0.0)),
                    "obstacle_threat": bool(threat),
                }
            cbf_ms_total = float(np.sum(times))
            cbf_active = float(np.mean(actives))
            cbf_delta = float(np.mean(deltas))

    for i in range(3):
        yaw_i = float(yaws[i]) if yaws is not None else 0.0
        u_world = np.asarray(out[i, :2], dtype=np.float64)
        if yaws is not None:
            wx, wy = _body_vel_to_world(float(out[i, 0]), float(out[i, 1]), yaw_i)
            u_world = np.array([wx, wy], dtype=np.float64)
        if ccfg is not None and ccfg.enabled and bool(getattr(ccfg, "local_escape_enabled", True)):
            u_escape, escape_active, nearest_clear = _apply_local_obstacle_escape(
                pursuer_pos[i, :2],
                u_world,
                all_obstacles,
                uav_radius=float(ccfg.uav_radius),
                safety_margin=float(ccfg.safety_margin),
                extra_clearance=float(getattr(ccfg, "local_escape_extra_clearance", 0.15)),
                escape_speed=float(getattr(ccfg, "emergency_escape_speed", 0.12)),
                action_low_xy=low[:2],
                action_high_xy=high[:2],
            )
            local_escape_threat[i] = bool(escape_active)
            if escape_active:
                u_world = u_escape
                if yaws is not None:
                    bx, by = _world_vel_to_body(float(u_world[0]), float(u_world[1]), yaw_i)
                    out[i, 0] = float(bx)
                    out[i, 1] = float(by)
                else:
                    out[i, 0] = float(u_world[0])
                    out[i, 1] = float(u_world[1])
                cbf_agent_debug[i]["local_escape_active"] = True
                cbf_agent_debug[i]["nearest_obstacle_clearance_xy"] = float(nearest_clear)
        u_safe, b_active = apply_xy_boundary_barrier(
            pursuer_pos[i, :2],
            u_world,
            world_xy=world_xy,
            boundary_margin=boundary_margin,
            boundary_alpha=boundary_alpha,
            action_low_xy=low[:2],
            action_high_xy=high[:2],
        )
        boundary_threat[i] = bool(b_active)
        if b_active:
            if yaws is not None:
                bx, by = _world_vel_to_body(float(u_safe[0]), float(u_safe[1]), yaw_i)
                out[i, 0] = float(bx)
                out[i, 1] = float(by)
            else:
                out[i, 0] = float(u_safe[0])
                out[i, 1] = float(u_safe[1])
            cbf_agent_debug[i]["boundary_active"] = True
            cbf_agent_debug[i]["boundary_safe_speed_xy"] = float(np.linalg.norm(u_safe))

        apply_hard_altitude_to_action_row(
            out[i],
            float(pursuer_pos[i, 2]),
            float(track_targets[i, 2]),
            low,
            high,
            gate_horizontal=not (obstacle_threat[i] or local_escape_threat[i] or boundary_threat[i]),
            z_floor=z_floor,
            z_ceiling=z_ceiling,
            floor_margin=altitude_floor_margin,
            ceiling_margin=altitude_ceiling_margin,
        )

    out = np.clip(out, low[None, :], high[None, :]).astype(np.float32)
    return out, cbf_agent_debug, cbf_ms_total, cbf_active, cbf_delta


def deployable_sce_actions_from_state(
    env: Any,
    lin_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    kind: DeployableKind,
    reachability: dict[str, Any] | ReachabilityConfig,
    cbf: dict[str, Any] | CBFConfig | None = None,
    runtime_rates: dict[str, Any] | RuntimeRatesConfig | None = None,
    turn_radius_planner: dict[str, Any] | None = None,
    lin_vel: np.ndarray | None = None,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
) -> np.ndarray:
    """Real-time deployable SCE baseline action function."""
    t_decision = time.perf_counter()

    rcfg = reachability if isinstance(reachability, ReachabilityConfig) else ReachabilityConfig.from_dict(reachability)
    rates = (
        runtime_rates
        if isinstance(runtime_rates, RuntimeRatesConfig)
        else RuntimeRatesConfig.from_dict(runtime_rates)
    )
    ccfg = None if cbf is None else (cbf if isinstance(cbf, CBFConfig) else CBFConfig.from_dict(cbf))

    pos = np.asarray(lin_pos, dtype=np.float32)
    vel = np.zeros_like(pos, dtype=np.float32) if lin_vel is None else np.asarray(lin_vel, dtype=np.float32)
    task_state = env.task_state
    task = env.task
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
    pursuer_pos = pos[pursuer_ids]
    pursuer_vel = vel[pursuer_ids]
    evader_pos = pos[int(task_state.evader_id)]

    if str(getattr(task, "role_assignment_mode", "")).strip().lower() != "entropic_ot":
        raise ValueError("Deployable SCE baselines require role_assignment_mode='entropic_ot'")

    st = _get_state(env, rcfg.path_cache)
    _maybe_reset(env, st)
    st.step_counter += 1
    st.replans_this_step = 0

    control_hz = float(rates.control_hz)
    manifold_every = rates.every_n_steps(rates.manifold_update_hz, control_hz)
    assign_every = rates.every_n_steps(rates.assignment_update_hz, control_hz)
    replan_every = rates.every_n_steps(rates.path_replan_hz, control_hz)
    cbf_every = rates.every_n_steps(rates.cbf_hz, control_hz)

    uav_r = _uav_radius(task, rcfg)
    all_obstacles = obstacles_from_task_state(task_state, task=task)
    obstacles, obstacle_diag = _perceived_obstacles(all_obstacles, pursuer_pos, evader_pos, rcfg)
    bounds = _world_bounds(task)
    planner_cfg = dict(rcfg.path_planner or {})
    pt_cfg = dict(rcfg.path_tracking or {})
    pc_cfg = PathCacheConfig.from_dict(rcfg.path_cache)
    st.path_cache.cfg = pc_cfg
    oq_cfg = ObstacleQueryConfig.from_dict(rcfg.obstacle_query)
    mode = rcfg.mode if rcfg.enabled else "euclidean"
    use_reachability_slots = kind in ("sce_reachability_slot", "sce_reachability_cbf_slot")
    use_turn_radius_slot = kind == "sce_turn_radius_slot"
    use_path_kind = kind in (
        "sce_cached_path_slot",
        "sce_cached_path_cbf_slot",
        "sce_reachability_slot",
        "sce_reachability_cbf_slot",
    )
    if kind == "sce_los_slot" and mode in ("cached_path", "structure_cached_path"):
        mode = "structure_los"

    # --- manifold (sub-rate) ---
    t0 = time.perf_counter()
    update_manifold = (
        st.cached_targets is None
        or st.step_counter % manifold_every == 1
    )
    slot_sel_cfg = StructureSlotSelectionConfig.from_dict(rcfg.structure_slot_selection)
    slot_sel_diag: dict[str, Any] = {}
    if use_reachability_slots:
        # Candidate-slot SCE chooses slots jointly with path reachability later;
        # fixed manifold projection would reintroduce the old geometry-first failure mode.
        if st.cached_targets is None:
            st.cached_targets = task._reference_manifold_targets(
                pursuer_pos, evader_pos, task_state=task_state,
            )
        st.cached_manifold_curve = None
        st.cached_slot_selection_diag = {}
        slot_sel_diag = {}
    elif update_manifold:
        if slot_sel_cfg.enabled and mode.startswith("structure"):
            st.cached_targets, st.cached_manifold_curve, slot_sel_diag = select_structure_manifold_slots(
                task, pursuer_pos, evader_pos, obstacles, task_state,
                slot_sel_cfg, rcfg.structure_cfg(),
                previous_assignment=getattr(task_state, "assigned_target_indices", None),
                safety_margin=rcfg.safety_margin, uav_radius=uav_r,
                switch_penalty=rcfg.switch_penalty, path_cache=st.path_cache,
            )
            st.cached_slot_selection_diag = dict(slot_sel_diag)
            new_scale = float(slot_sel_diag.get("manifold_radius_scale", 1.0))
            if (
                st.last_manifold_radius_scale is not None
                and abs(new_scale - st.last_manifold_radius_scale) > 1e-4
            ):
                st.path_cache.agent_paths.clear()
                st.path_cache.pair_cost_cache.clear()
            st.last_manifold_radius_scale = new_scale
        else:
            st.cached_targets = task._reference_manifold_targets(
                pursuer_pos, evader_pos, task_state=task_state,
            )
            st.cached_manifold_curve = None
            if hasattr(task, "_reference_manifold_curve"):
                st.cached_manifold_curve = task._reference_manifold_curve(
                    pursuer_pos, evader_pos, task_state=task_state,
                )
            st.cached_slot_selection_diag = None
    else:
        slot_sel_diag = dict(st.cached_slot_selection_diag or {})

    nominal_targets = np.asarray(st.cached_targets, dtype=np.float32).copy()
    slot_proj_cfg = SlotProjectionConfig.from_dict(rcfg.slot_projection)
    proj_diag: list[dict[str, Any]] = []
    if (not use_reachability_slots) and slot_proj_cfg.enabled and st.cached_manifold_curve is not None:
        targets, proj_diag = project_manifold_slots(
            nominal_targets, st.cached_manifold_curve, pursuer_pos, evader_pos, obstacles,
            slot_proj_cfg, safety_margin=rcfg.safety_margin, uav_radius=uav_r,
        )
    else:
        targets = nominal_targets.copy()

    slot_moved_mask = np.zeros(3, dtype=bool)
    if slot_proj_cfg.enabled:
        slot_moved_mask = detect_slot_projection_moves(
            nominal_targets, targets, st.last_effective_targets,
            threshold=slot_proj_cfg.replan_trigger_threshold,
        )
        if slot_proj_cfg.trigger_path_replan and pc_cfg.slot_projection_force_replan:
            for j in range(3):
                if bool(slot_moved_mask[j]):
                    st.path_cache.invalidate_slot(j)
    st.last_effective_targets = np.asarray(targets, dtype=np.float32).copy()
    st.slot_projection_moved = slot_moved_mask.copy()
    st.timing.manifold.record((time.perf_counter() - t0) * 1000.0)

    # --- assignment (sub-rate) ---
    t0 = time.perf_counter()
    prev = getattr(task_state, "assigned_target_indices", None)
    update_assign = (
        st.cached_assignment is None
        or st.step_counter % assign_every == 1
    )
    if update_assign:
        old_assignment = None if st.cached_assignment is None else np.asarray(st.cached_assignment).copy()
        if use_reachability_slots:
            speed_ref = float(np.max(np.abs(action_high[:2]))) if action_high.size >= 2 else 1.0
            slot_cfg_for_select = dict(rcfg.candidate_slots or {})
            structure_cfg_for_select = dict(rcfg.structure_selection or {})
            # Browser debug favors interactive diagnosis over exhaustive search.
            # Normal benchmark/eval uses the YAML values unchanged.
            if bool(getattr(task, "debug", False)):
                slot_cap = int(slot_cfg_for_select.get("debug_max_candidates", 24))
                combo_cap = int(structure_cfg_for_select.get("debug_max_slot_combinations", 160))
                if "max_candidates" in slot_cfg_for_select:
                    slot_cfg_for_select["max_candidates"] = min(int(slot_cfg_for_select["max_candidates"]), slot_cap)
                else:
                    slot_cfg_for_select["max_candidates"] = slot_cap
                if "max_slot_combinations" in structure_cfg_for_select:
                    structure_cfg_for_select["max_slot_combinations"] = min(
                        int(structure_cfg_for_select["max_slot_combinations"]), combo_cap
                    )
                else:
                    structure_cfg_for_select["max_slot_combinations"] = combo_cap
            targets, assignment, cost_diag = select_reachability_aware_slots(
                pursuer_pos,
                evader_pos,
                obstacles,
                st.cached_candidate_assignment,
                st.path_cache,
                bounds,
                planner_cfg,
                world_xy=float(getattr(task, "world_xy", 20.0)),
                pursuer_speed=max(speed_ref, 1e-6),
                safety_margin=rcfg.safety_margin,
                uav_radius=uav_r,
                slot_cfg=slot_cfg_for_select,
                assignment_cfg=rcfg.assignment_cost,
                structure_cfg=structure_cfg_for_select,
                switch_penalty=rcfg.switch_penalty,
                previous_candidate_descriptors=st.cached_candidate_descriptors,
                previous_slot_descriptors=st.cached_slot_descriptors,
                previous_slot_assignment=st.cached_assignment,
            )
            st.cached_targets = np.asarray(targets, dtype=np.float32).copy()
            st.cached_assignment = np.asarray(assignment, dtype=np.int64).reshape(3)
            selected_ids = np.asarray(cost_diag.get("selected_candidate_indices", []), dtype=np.int64)
            if selected_ids.shape[0] == 3:
                st.cached_candidate_assignment = selected_ids[st.cached_assignment]
            assigned_desc = cost_diag.get("assigned_candidate_descriptors")
            if isinstance(assigned_desc, list) and len(assigned_desc) == 3:
                st.cached_candidate_descriptors = [dict(x) for x in assigned_desc]
            slot_desc = cost_diag.get("selected_candidate_descriptors")
            if isinstance(slot_desc, list) and len(slot_desc) == 3:
                st.cached_slot_descriptors = [dict(x) for x in slot_desc]
            st.cached_reachability_diag = dict(cost_diag)
        else:
            cost, cost_diag = build_reachability_cost_matrix(
                pursuer_pos, targets, obstacles, prev, rcfg,
                evader_pos=evader_pos,
                path_cache=st.path_cache,
            )
            struct_assign = cost_diag.pop("_structure_assignment", None)
            if struct_assign is not None:
                st.cached_assignment = np.asarray(struct_assign, dtype=np.int64).reshape(3)
            else:
                st.cached_assignment = _ot_assign(cost, task, prev)
        if old_assignment is not None and st.cached_assignment is not None:
            st.assignment_switch_count += int(np.sum(np.asarray(st.cached_assignment) != old_assignment))
    else:
        cost_diag = {}
        if st.cached_assignment is None:
            st.cached_assignment = np.arange(3, dtype=np.int64)
        if use_reachability_slots and st.cached_targets is not None:
            targets = np.asarray(st.cached_targets, dtype=np.float32).copy()
            cost_diag = dict(st.cached_reachability_diag or {})

    assignment = st.cached_assignment
    assigned_targets = targets[assignment]
    task_state.assigned_target_indices = np.asarray(assignment, dtype=np.int64).copy()
    st.timing.assignment.record((time.perf_counter() - t0) * 1000.0)
    if "los_check_time_ms" in cost_diag:
        st.timing.los_check.record(float(cost_diag["los_check_time_ms"]))
    if "cost_matrix_time_ms" in cost_diag:
        st.timing.cost_matrix.record(float(cost_diag["cost_matrix_time_ms"]))
    elif "los_cost_time_ms" in cost_diag:
        st.timing.cost_matrix.record(float(cost_diag["los_cost_time_ms"]))

    # --- tracking targets ---
    track_targets = assigned_targets.copy()
    agent_paths_xy: list[list[list[float]] | None] = [None, None, None]
    boundary_margin = float(pt_cfg.get("boundary_margin", max(0.0, uav_r)))
    boundary_alpha = float(pt_cfg.get("boundary_alpha", 2.0))
    t0 = time.perf_counter()
    path_endpoint_errors: list[float] = []
    path_tracking_errors: list[float] = []
    path_min_clearances: list[float] = []
    expected_times: list[float] = []
    turn_active_flags: list[float] = []
    turn_arc_clearances: list[float] = []
    turn_boundary_clearances: list[float] = []
    turn_boundary_unsafe_flags: list[float] = []
    turn_angles: list[float] = []
    turn_agent_debug: list[dict[str, Any]] = [{}, {}, {}]
    turn_safety_enabled = False
    turn_slow_speed = 0.0
    turn_yaw_gain = 0.0
    safety_speed_debug: list[dict[str, Any]] = [{}, {}, {}]
    turn_radius_debug: list[dict[str, Any]] = [{}, {}, {}]
    stale_paths = 0
    path_invalid_flags = 0
    if use_path_kind:
        lookahead = float(pt_cfg.get("lookahead_dist", 0.8))
        accept_r = float(pt_cfg.get("waypoint_accept_radius", 0.2))
        max_replans = int(pc_cfg.max_replans_per_step)
        allow_replan = st.step_counter % replan_every == 1
        drift_replan_interval = max(
            1,
            int(pt_cfg.get("drift_replan_interval", replan_every)),
        )
        drift_replan_enabled = bool(
            pt_cfg.get("drift_replan_enabled", "drift_replan_interval" in pt_cfg)
        )
        invalidated_cached_paths = 0
        blocked_without_safe_path = 0
        drift_replan_requests = 0
        turn_safety_enabled = bool(pt_cfg.get("turn_safety_enabled", use_reachability_slots))
        speed_ref_xy = float(np.max(np.abs(action_high[:2]))) if action_high.size >= 2 else 1.0
        turn_slow_speed = float(pt_cfg.get("turn_slow_speed", min(0.08, max(0.02, speed_ref_xy * 0.35))))
        turn_yaw_gain = float(pt_cfg.get("turn_yaw_gain", yaw_gain))
        turn_radius = float(pt_cfg.get(
            "min_turn_radius",
            max(0.6, speed_ref_xy * speed_ref_xy / max(float(pt_cfg.get("max_lateral_accel", 0.25)), 1e-6)),
        ))
        turn_min_clearance = float(pt_cfg.get("turn_min_clearance", max(0.6, rcfg.safety_margin + uav_r)))
        turn_boundary_clearance = float(pt_cfg.get("turn_boundary_min_clearance", max(0.4, boundary_margin)))
        turn_safe_forward = float(pt_cfg.get("turn_safe_forward_dist", min(lookahead, max(0.4, turn_radius * 0.5))))
        turn_samples = int(pt_cfg.get("turn_arc_samples", 11))

        for i in range(3):
            slot_j = int(assignment[i])
            goal_xy = targets[slot_j, :2]
            # A single perceived obstacle set is used for path planning,
            # collision checking, LOS, shortcut validation, and CBF.  Local
            # perception is explicit via rcfg.obstacle_mode; termination still
            # uses the environment's true global obstacles.
            plan_obs = obstacles
            validate_obs = obstacles
            path = st.path_cache.get_agent_path(i)
            st_path = st.path_cache.agent_paths.get(i)
            slot_moved = (
                st_path is not None
                and st_path.goal_xy is not None
                and float(np.linalg.norm(goal_xy - st_path.goal_xy.reshape(2)))
                > pc_cfg.slot_replan_threshold
            )
            slot_projected = (
                st.slot_projection_moved is not None
                and bool(st.slot_projection_moved[slot_j])
                and slot_proj_cfg.trigger_path_replan
                and pc_cfg.slot_projection_force_replan
            )
            force_slot = (pc_cfg.slot_replan_always and slot_moved) or slot_projected
            cache_stale_or_unsafe = False
            if path is not None:
                diag_now = st.path_cache.path_diagnostics(
                    i,
                    pursuer_pos[i, :2],
                    goal_xy,
                    target_xy=evader_pos[:2],
                    speed=float(np.max(np.abs(action_high[:2]))),
                )
                path_endpoint_errors.append(float(diag_now["path_endpoint_error"]))
                path_tracking_errors.append(float(diag_now["path_tracking_error"]))
                path_min_clearances.append(float(diag_now["path_min_clearance"]))
                expected_times.append(float(diag_now["expected_time_to_slot"]))
                path_invalid_flags += int(bool(diag_now["path_is_invalid"]))
                if (
                    float(diag_now["slot_moved_distance"]) > pc_cfg.replan_slot_move_thresh
                    or float(diag_now["target_moved_distance"]) > pc_cfg.replan_target_move_thresh
                    or float(diag_now["path_endpoint_error"]) > pc_cfg.replan_endpoint_error_thresh
                ):
                    stale_paths += 1
                validate_cached_now = (
                    allow_replan
                    or force_slot
                    or st.step_counter % pc_cfg.cache_validation_interval == 1
                )
                if (
                    st_path is None
                    or int(st_path.slot_id) != slot_j
                    or validate_cached_now
                ):
                    cache_stale_or_unsafe = st.path_cache.cached_path_stale_or_unsafe(
                        i, slot_j, pursuer_pos[i, :2], goal_xy, validate_obs,
                        safety_margin=rcfg.safety_margin, uav_radius=uav_r,
                    )
                if cache_stale_or_unsafe:
                    path = None
                    invalidated_cached_paths += 1
            drift_high = False
            if path is not None and drift_replan_enabled:
                path_dev = st.path_cache._path_deviation(path, pursuer_pos[i, :2])
                drift_exceeds = path_dev > pc_cfg.path_deviation_threshold
                if drift_exceeds:
                    drift_replan_requests += 1
                drift_ready = (
                    st_path is None
                    or st_path.last_replan_step < 0
                    or st.step_counter - int(st_path.last_replan_step) >= drift_replan_interval
                )
                drift_high = bool(drift_exceeds and drift_ready)
            check_replan = allow_replan or force_slot or cache_stale_or_unsafe or drift_high
            need_replan = (
                st.replans_this_step < max_replans
                and check_replan
                and (
                    path is None
                    or slot_projected
                    or st.path_cache.should_replan(
                        i, slot_j, pursuer_pos[i, :2], goal_xy,
                        validate_obs, st.step_counter,
                        safety_margin=rcfg.safety_margin, uav_radius=uav_r,
                        force_slot=force_slot or slot_moved or drift_high,
                        force_projection=slot_projected,
                        target_xy=evader_pos[:2],
                        cbf_active_steps=int(st.cbf_consecutive_steps[i]),
                        speed=float(np.max(np.abs(action_high[:2]))),
                    )
                )
            )
            if need_replan:
                t_plan = time.perf_counter()
                path, replanned, plan_ms = st.path_cache.get_or_replan_assigned_path(
                    i, slot_j, pursuer_pos[i, :2], goal_xy,
                    plan_obs, obstacles, validate_obs,
                    bounds, planner_cfg, st.step_counter,
                    safety_margin=rcfg.safety_margin, uav_radius=uav_r,
                    force=True,
                    target_xy=evader_pos[:2],
                    cbf_active_steps=int(st.cbf_consecutive_steps[i]),
                    speed=float(np.max(np.abs(action_high[:2]))),
                )
                st.timing.path_planning.record(plan_ms if plan_ms > 0 else (time.perf_counter() - t_plan) * 1000.0)
                if replanned:
                    st.replans_this_step += 1

            if path is not None:
                agent_paths_xy[i] = [
                    [float(pt[0]), float(pt[1])] for pt in np.asarray(path, dtype=np.float64).reshape(-1, 2)
                ]
                local = select_lookahead_waypoint(
                    path, pursuer_pos[i], lookahead,
                    waypoint_accept_radius=accept_r,
                    terminal_goal=assigned_targets[i],
                )
                if turn_safety_enabled:
                    desired_dir = np.asarray(local[:2] - pursuer_pos[i, :2], dtype=np.float64)
                    desired_nrm = float(np.linalg.norm(desired_dir))
                    if desired_nrm > 1e-9:
                        desired_dir = desired_dir / desired_nrm
                    else:
                        desired_dir = np.zeros(2, dtype=np.float64)
                    prev_dir = st.last_world_cmd_dirs[i]
                    if not np.all(np.isfinite(prev_dir)):
                        pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
                        if pts.shape[0] >= 2:
                            nearest = int(np.argmin(np.linalg.norm(pts - pursuer_pos[i, :2], axis=1)))
                            if nearest < pts.shape[0] - 1:
                                prev_dir = pts[nearest + 1] - pts[nearest]
                            else:
                                prev_dir = pts[nearest] - pts[max(nearest - 1, 0)]
                        else:
                            prev_dir = local[:2] - pursuer_pos[i, :2]
                    adjusted_xy, turn_diag = adjust_lookahead_for_turn_safety(
                        pursuer_pos[i, :2],
                        local[:2],
                        prev_dir,
                        obstacles,
                        uav_radius=uav_r,
                        turn_radius=turn_radius,
                        min_turn_clearance=turn_min_clearance,
                        safe_forward_dist=turn_safe_forward,
                        num_samples=turn_samples,
                        world_xy=float(getattr(task, "world_xy", np.inf)),
                        boundary_margin=boundary_margin,
                        min_boundary_clearance=turn_boundary_clearance,
                    )
                    local[:2] = adjusted_xy[:2]
                    adjusted_dir = np.asarray(adjusted_xy[:2] - pursuer_pos[i, :2], dtype=np.float64)
                    adjusted_nrm = float(np.linalg.norm(adjusted_dir))
                    if adjusted_nrm > 1e-9:
                        adjusted_dir = adjusted_dir / adjusted_nrm
                    else:
                        adjusted_dir = np.zeros(2, dtype=np.float64)
                    turn_active_flags.append(1.0 if bool(turn_diag["turn_safety_active"]) else 0.0)
                    turn_arc_clearances.append(float(turn_diag["turn_arc_min_clearance"]))
                    turn_boundary_clearances.append(float(turn_diag["turn_boundary_min_clearance"]))
                    turn_boundary_unsafe_flags.append(1.0 if bool(turn_diag["turn_boundary_unsafe"]) else 0.0)
                    turn_angles.append(float(turn_diag["turn_angle_rad"]))
                    turn_diag["turn_desired_dir_xy"] = desired_dir.astype(float).tolist()
                    turn_diag["turn_adjusted_dir_xy"] = adjusted_dir.astype(float).tolist()
                    turn_agent_debug[i] = dict(turn_diag)
                track_targets[i, :2] = local[:2]
            elif not has_line_of_sight(
                pursuer_pos[i, :2], goal_xy, validate_obs,
                safety_margin=rcfg.safety_margin, uav_radius=uav_r,
            ):
                # No certified path: hold horizontal position instead of driving through an obstacle.
                track_targets[i, :2] = pursuer_pos[i, :2]
                blocked_without_safe_path += 1
    else:
        invalidated_cached_paths = 0
        blocked_without_safe_path = 0
        drift_replan_requests = 0
    st.timing.waypoint.record((time.perf_counter() - t0) * 1000.0)

    # --- proportional / path-following control ---
    t0 = time.perf_counter()
    yaws = pursuer_yaw
    use_path_follow = use_path_kind
    tube_radius = float(pt_cfg.get("tube_radius", 0.35))
    cross_track_gain = float(pt_cfg.get("cross_track_gain", 1.0))
    approach_dist = float(pt_cfg.get("approach_dist", 0.5))
    altitude_floor_margin = float(pt_cfg.get("altitude_floor_margin", 0.25))
    altitude_ceiling_margin = float(pt_cfg.get("altitude_ceiling_margin", 0.10))
    safety_speed_enabled = bool(pt_cfg.get("safety_speed_enabled", use_reachability_slots))
    if use_turn_radius_slot:
        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        planner_cfg_local = dict(turn_radius_planner or {})
        if "vmax" not in planner_cfg_local and high.size >= 2:
            planner_cfg_local["vmax"] = float(np.max(np.abs(high[:2])))
        if "omega_max" not in planner_cfg_local and high.size >= 3:
            planner_cfg_local["omega_max"] = float(max(abs(float(low[2])), abs(float(high[2]))))
        if "uav_radius" not in planner_cfg_local:
            planner_cfg_local["uav_radius"] = float(uav_r)
        if "safety_margin" not in planner_cfg_local:
            planner_cfg_local["safety_margin"] = float(rcfg.safety_margin)
        controller = TurnRadiusSlotController(planner_cfg_local)
        actions = np.zeros((3, low.shape[0]), dtype=np.float32)
        prev_turn_actions = np.asarray(st.turn_radius_prev_actions, dtype=np.float64).reshape(3, 2)
        yaws_local = (
            np.asarray(yaws, dtype=np.float64).reshape(3)
            if yaws is not None
            else np.zeros(3, dtype=np.float64)
        )
        next_prev = np.zeros((3, 2), dtype=np.float64)
        for i in range(3):
            world_xy, yaw_rate_cmd, diag_i = controller.compute_action(
                pursuer_pos[i, :2],
                float(yaws_local[i]),
                pursuer_vel[i, :2],
                assigned_targets[i, :2],
                all_obstacles,
                prev_action=prev_turn_actions[i],
            )
            bx, by = _world_vel_to_body(float(world_xy[0]), float(world_xy[1]), float(yaws_local[i]))
            actions[i, 0] = np.float32(bx)
            actions[i, 1] = np.float32(by)
            if actions.shape[1] >= 3:
                actions[i, 2] = np.float32(yaw_rate_cmd)
            if actions.shape[1] >= 4:
                actions[i, 3] = np.float32(float(z_gain) * float(assigned_targets[i, 2] - pursuer_pos[i, 2]))
            next_prev[i, 0] = float(diag_i.get("best_candidate_speed", np.linalg.norm(world_xy)))
            next_prev[i, 1] = float(diag_i.get("best_candidate_yaw_rate", yaw_rate_cmd))
            diag_i = dict(diag_i)
            diag_i.pop("local_obstacles", None)
            turn_radius_debug[i] = diag_i
        st.turn_radius_prev_actions = next_prev
        actions = np.clip(actions, low[None, :], high[None, :]).astype(np.float32)
        speed_breakdown = []
        safety_speed_enabled = False
    elif use_path_follow:
        actions = tube_tracking_body_actions(
            pursuer_pos, track_targets, action_low, action_high,
            agent_paths_xy,
            xy_gain=xy_gain, z_gain=z_gain,
            pursuer_yaw=yaws, yaw_gain=yaw_gain,
            tube_radius=tube_radius,
            cross_track_gain=cross_track_gain,
            approach_dist=approach_dist,
        )
        if turn_safety_enabled:
            actions = _apply_turn_slowdown_actions(
                actions,
                turn_agent_debug,
                action_low,
                action_high,
                yaws=yaws,
                slow_speed=turn_slow_speed,
                yaw_gain=turn_yaw_gain,
            )
    else:
        actions = proportional_actions_to_targets(
            pursuer_pos, track_targets, action_low, action_high,
            xy_gain=xy_gain, z_gain=z_gain,
            pursuer_yaw=yaws, yaw_gain=yaw_gain,
            yaw_align_min_speed=yaw_align_min_speed,
        )
        speed_breakdown = []
    if safety_speed_enabled:
        actions, safety_speed_debug = _apply_safety_speed_constraints(
            actions,
            pursuer_pos,
            assigned_targets,
            action_low,
            action_high,
            obstacles,
            yaws=yaws,
            world_xy=float(getattr(task, "world_xy", np.inf)),
            boundary_margin=boundary_margin,
            uav_radius=uav_r,
            cfg=pt_cfg,
        )
    st.timing.action_post.record((time.perf_counter() - t0) * 1000.0)

    # --- priority: obstacle CBF > slot nominal > altitude hold ---
    nominal_actions = np.asarray(actions, dtype=np.float32).copy()
    effective_ccfg = _resolve_cbf_config(ccfg, rcfg, kind, obstacles, uav_r)
    other_agents_xy = [np.delete(pursuer_pos[:, :2], i, axis=0) for i in range(3)]
    actions, cbf_agent_debug, cbf_ms_total, cbf_active, cbf_delta = _apply_control_priority_layers(
        actions,
        pursuer_pos,
        track_targets,
        action_low,
        action_high,
        all_obstacles=obstacles,
        other_agents_xy=other_agents_xy,
        yaws=yaws,
        ccfg=effective_ccfg,
        cbf_filter=st.cbf_filter,
        cbf_every=cbf_every,
        step_counter=st.step_counter,
        z_floor=float(getattr(task, "z_min", 0.0)),
        z_ceiling=float(getattr(task, "z_max", np.inf)),
        altitude_floor_margin=altitude_floor_margin,
        altitude_ceiling_margin=altitude_ceiling_margin,
        world_xy=float(getattr(task, "world_xy", np.inf)),
        boundary_margin=boundary_margin,
        boundary_alpha=boundary_alpha,
    )
    if cbf_ms_total > 0.0:
        st.timing.cbf_filter.record(cbf_ms_total)
    for i in range(3):
        dbg = cbf_agent_debug[i] if i < len(cbf_agent_debug) else {}
        active_now = bool(dbg.get("obstacle_threat", False)) or float(dbg.get("cbf_delta_norm", 0.0)) > 1e-4
        active_now = active_now or bool(dbg.get("local_escape_active", False)) or bool(dbg.get("boundary_active", False))
        st.cbf_consecutive_steps[i] = int(st.cbf_consecutive_steps[i] + 1) if active_now else 0

    decision_ms = (time.perf_counter() - t_decision) * 1000.0
    st.timing.decision_total.record(decision_ms)

    los_blocked = cost_diag.get("los_blocked_matrix")
    if los_blocked is None:
        los_blocked = cost_diag.get("pair_reachable_matrix")
        if los_blocked is not None:
            los_blocked = ~np.asarray(los_blocked, dtype=bool)
    assigned_blocked = False
    if los_blocked is not None:
        assigned_blocked = any(
            bool(los_blocked[i, int(assignment[i])]) for i in range(3)
        )

    def _mean_finite(vals: list[float], default: float = float("nan")) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else default

    def _min_finite(vals: list[float], default: float = float("nan")) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.min(arr)) if arr.size else default

    nominal_action_norm = float(np.mean(np.linalg.norm(nominal_actions[:, :2], axis=1)))
    filtered_action_norm = float(np.mean(np.linalg.norm(np.asarray(actions)[:, :2], axis=1)))
    safety_active_flags = [
        1.0 if bool(d.get("safety_speed_active", False)) else 0.0
        for d in safety_speed_debug
    ]
    safety_caps = [
        float(d.get("safety_speed_cap", np.nan))
        for d in safety_speed_debug
    ]
    safety_clearances = [
        float(d.get("safety_speed_forward_clearance", np.nan))
        for d in safety_speed_debug
    ]
    local_obstacle_counts = [float(d.get("local_obstacle_count", np.nan)) for d in turn_radius_debug]
    candidate_counts = [float(d.get("candidate_count", np.nan)) for d in turn_radius_debug]
    valid_candidate_counts = [float(d.get("valid_candidate_count", np.nan)) for d in turn_radius_debug]
    local_blocked_flags = [
        1.0 if bool(d.get("local_planner_blocked", False)) else 0.0
        for d in turn_radius_debug
    ]
    local_planner_times = [float(d.get("local_planner_time_ms", np.nan)) for d in turn_radius_debug]
    best_candidate_costs = [float(d.get("best_candidate_cost", np.nan)) for d in turn_radius_debug]
    best_candidate_speeds = [float(d.get("best_candidate_speed", np.nan)) for d in turn_radius_debug]
    best_candidate_yaws = [float(d.get("best_candidate_yaw_rate", np.nan)) for d in turn_radius_debug]
    min_predicted_clearances = [float(d.get("min_predicted_clearance", np.nan)) for d in turn_radius_debug]
    assigned_slot_distances = [float(d.get("assigned_slot_distance", np.nan)) for d in turn_radius_debug]
    selected_action_norms = [float(d.get("selected_action_norm", np.nan)) for d in turn_radius_debug]
    stale_path_rate = float(stale_paths / 3.0) if use_path_kind else 0.0
    unreachable_slot_rate = float(cost_diag.get("unreachable_slot_rate", 0.0))
    if "slot_reachable_rate" not in cost_diag and "pair_feasible_matrix" in cost_diag:
        feasible = np.asarray(cost_diag["pair_feasible_matrix"], dtype=bool)
        cost_diag["slot_reachable_rate"] = float(np.mean(np.any(feasible, axis=0)))
        unreachable_slot_rate = float(1.0 - cost_diag["slot_reachable_rate"])

    step_diag = {
        **obstacle_diag,
        "assigned_pair_blocked_los": float(assigned_blocked),
        "num_los_blocked_pairs": float(cost_diag.get("num_los_blocked_pairs", 0)),
        "num_slot_projection_moves": float(np.sum(st.slot_projection_moved)) if st.slot_projection_moved is not None else 0.0,
        "path_cache_hit_rate": st.path_cache.hit_rate,
        "num_replans_this_step": float(st.replans_this_step),
        "path_replan_count": float(st.path_cache.stats["replan_count"]),
        "num_invalidated_cached_paths": float(invalidated_cached_paths),
        "num_blocked_without_safe_path": float(blocked_without_safe_path),
        "num_drift_replan_requests": float(drift_replan_requests),
        "stale_path_rate": stale_path_rate,
        "path_endpoint_error": _mean_finite(path_endpoint_errors),
        "path_min_clearance": _min_finite(path_min_clearances),
        "path_tracking_error": _mean_finite(path_tracking_errors),
        "turn_safety_active_rate": _mean_finite(turn_active_flags, 0.0),
        "turn_arc_min_clearance": _min_finite(turn_arc_clearances),
        "turn_boundary_min_clearance": _min_finite(turn_boundary_clearances),
        "turn_boundary_unsafe_rate": _mean_finite(turn_boundary_unsafe_flags, 0.0),
        "turn_angle_rad": _mean_finite(turn_angles),
        "safety_speed_limit_rate": _mean_finite(safety_active_flags, 0.0),
        "safety_speed_cap_mean": _mean_finite(safety_caps),
        "safety_speed_clearance_min": _min_finite(safety_clearances),
        "expected_time_to_slot": _mean_finite(expected_times),
        "path_is_invalid_rate": float(path_invalid_flags / 3.0) if use_path_kind else 0.0,
        "slot_reachable_rate": float(cost_diag.get("slot_reachable_rate", 1.0)),
        "mean_time_to_slot": float(cost_diag.get("mean_time_to_slot", _mean_finite(expected_times))),
        "max_time_to_slot": float(cost_diag.get("max_time_to_slot", _mean_finite(expected_times))),
        "path_clearance_min": float(cost_diag.get("path_clearance_min", _min_finite(path_min_clearances))),
        "path_clearance_mean": float(cost_diag.get("path_clearance_mean", _mean_finite(path_min_clearances))),
        "path_risk_integral": float(cost_diag.get("path_risk_integral", 0.0)),
        "slot_behind_obstacle_rate": float(cost_diag.get("slot_behind_obstacle_rate", 0.0)),
        "los_blocked_slot_rate": float(cost_diag.get("los_blocked_slot_rate", 0.0)),
        "fallback_slot_selection_rate": float(cost_diag.get("fallback_slot_selection_rate", 0.0)),
        "unreachable_slot_rate": unreachable_slot_rate,
        "assignment_lock_preserved_rate": 1.0 if bool(cost_diag.get("assignment_lock_preserved", False)) else 0.0,
        "assignment_switch_count": float(st.assignment_switch_count),
        "altitude_floor_margin": float(altitude_floor_margin),
        "xy_boundary_margin": float(boundary_margin),
        "path_planner_timeout_count": float(st.path_cache.stats["timeout_count"]),
        "cbf_active": cbf_active,
        "cbf_active_rate": cbf_active,
        "cbf_active_consecutive_steps": float(np.max(st.cbf_consecutive_steps)),
        "cbf_filter_time_ms": cbf_ms_total,
        "cbf_delta_norm": cbf_delta,
        "cbf_correction_norm": cbf_delta,
        "nominal_action_norm": nominal_action_norm,
        "filtered_action_norm": filtered_action_norm,
        "local_obstacle_count": _mean_finite(local_obstacle_counts),
        "candidate_count": _mean_finite(candidate_counts),
        "valid_candidate_count": _mean_finite(valid_candidate_counts),
        "local_planner_blocked": _mean_finite(local_blocked_flags, 0.0),
        "local_planner_time_ms": _mean_finite(local_planner_times),
        "best_candidate_cost": _mean_finite(best_candidate_costs),
        "best_candidate_speed": _mean_finite(best_candidate_speeds),
        "best_candidate_yaw_rate": _mean_finite(best_candidate_yaws),
        "min_predicted_clearance": _min_finite(min_predicted_clearances),
        "assigned_slot_distance": _mean_finite(assigned_slot_distances),
        "selected_action_norm": _mean_finite(selected_action_norms),
        "decision_time_ms": decision_ms,
        "decision_total_ms": decision_ms,
        "waypoint_tracking_time_ms": st.timing.waypoint.samples_ms[-1] if st.timing.waypoint.samples_ms else 0.0,
    }
    if use_path_kind and should_record_control_timing(env):
        plens, wps, cf = [], [], []
        for i in range(3):
            goal_xy = targets[int(assignment[i]), :2]
            validate_obs = obstacles
            path = st.path_cache.get_agent_path(i)
            if path is not None:
                plens.append(path_length(path))
                wps.append(len(path))
                cf.append(collision_check_path(
                    path, validate_obs, safety_margin=rcfg.safety_margin, uav_radius=uav_r,
                ))
        if plens:
            step_diag["assigned_path_length"] = float(np.mean(plens))
            step_diag["assigned_path_collision_free"] = float(np.mean(cf))

    manifold_obs = manifold_influencing_obstacles(
        evader_pos[:2],
        obstacles,
        capture_dist=float(getattr(task, "capture_dist", 1.0)),
        target_radius_xy=float(getattr(task_state, "latest_target_radius_xy", getattr(task, "capture_dist", 1.0))),
        top_k=int(getattr(task, "obstacle_manifold_top_k", 4)),
        influence_radius_scale=float(getattr(task, "obstacle_manifold_influence_radius_scale", 2.5)),
        clearance_margin_scale=float(getattr(task, "obstacle_manifold_clearance_margin_scale", 0.35)),
    )
    if use_path_follow:
        speed_breakdown = analyze_tube_tracking_speed(
            pursuer_pos, actions, agent_paths_xy, action_high,
            tube_radius=tube_radius, approach_dist=approach_dist,
            xy_gain=xy_gain, pursuer_yaw=yaws,
        )
    pursuer_debug: list[dict[str, Any]] = []
    if use_turn_radius_slot:
        for i in range(3):
            dbg = dict(turn_radius_debug[i])
            entry = {
                "slot_id": int(assignment[i]),
                "track_target_xy": assigned_targets[i, :2].astype(float).tolist(),
                "slot_target_xy": assigned_targets[i, :2].astype(float).tolist(),
                "assigned_path_xy": None,
                "corridor_obstacle_indices": [],
                "speed_cmd_xy": float(np.hypot(actions[i, 0], actions[i, 1])),
                "world_speed_cmd_xy": float(dbg.get("selected_action_norm", np.nan)),
                "speed_cap_xy": float(np.max(np.abs(action_high[:2]))) if action_high.size >= 2 else 0.0,
                "limit_reason": "turn_radius_blocked" if bool(dbg.get("local_planner_blocked", False)) else "turn_radius_slot",
                "track_dist_xy": float(dbg.get("assigned_slot_distance", np.nan)),
                "turn_radius_local_planner": dbg,
            }
            pursuer_debug.append(entry)
        env._deploy_sce_debug = {
            "pursuers": pursuer_debug,
            "manifold_obstacles": manifold_obs,
            "lookahead_dist": float((turn_radius_planner or {}).get("lookahead_dist", 1.5)),
            "local_planner": "turn_radius_swept_corridor",
        }
    elif use_path_follow:
        for i in range(3):
            goal_xy = targets[int(assignment[i]), :2]
            corridor_obs = obstacles_in_corridor(
                pursuer_pos[i, :2], goal_xy, obstacles, oq_cfg.corridor_half_width,
            )
            corridor_idx = [
                idx for c in corridor_obs
                if (idx := obstacle_index(c, all_obstacles)) is not None
            ]
            br = speed_breakdown[i]
            entry: dict[str, Any] = {
                "slot_id": int(assignment[i]),
                "track_target_xy": track_targets[i, :2].astype(float).tolist(),
                "slot_target_xy": assigned_targets[i, :2].astype(float).tolist(),
                "assigned_path_xy": agent_paths_xy[i],
                "corridor_obstacle_indices": corridor_idx,
                "speed_cmd_xy": float(br["speed_cmd_xy"]),
                "world_speed_cmd_xy": float(br.get("world_speed_cmd_xy", br["speed_cmd_xy"])),
                "cross_track_xy": float(br.get("cross_track_xy", 0.0)),
                "tube_radius": float(br.get("tube_radius", tube_radius)),
                "speed_cap_xy": float(br["speed_cap_xy"]),
                "limit_reason": str(br["limit_reason"]),
                "align_factor": float(br["align_factor"]),
                "track_dist_xy": float(br.get("dist_xy", 0.0)),
                "yaw_err_deg": float(br["yaw_err_deg"]),
            }
            if i < len(cbf_agent_debug) and cbf_agent_debug[i]:
                dbg = cbf_agent_debug[i]
                entry["cbf"] = dbg
                entry["speed_cmd_xy"] = float(np.hypot(actions[i, 0], actions[i, 1]))
                if dbg.get("obstacle_threat") or float(dbg.get("cbf_delta_norm", 0.0)) > 1e-4:
                    entry["limit_reason"] = "cbf_obstacle"
                elif dbg.get("local_escape_active"):
                    entry["limit_reason"] = "local_obstacle_escape"
                elif dbg.get("boundary_active"):
                    entry["limit_reason"] = "xy_boundary"
            if i < len(turn_agent_debug) and turn_agent_debug[i]:
                entry["turn_safety"] = turn_agent_debug[i]
                if turn_agent_debug[i].get("turn_safety_active"):
                    entry["limit_reason"] = "turn_safety"
            if i < len(safety_speed_debug) and safety_speed_debug[i]:
                entry["safety_speed"] = safety_speed_debug[i]
                entry["speed_cmd_xy"] = float(np.hypot(actions[i, 0], actions[i, 1]))
                entry["speed_cap_xy"] = float(safety_speed_debug[i].get("safety_speed_cap", entry["speed_cap_xy"]))
                if safety_speed_debug[i].get("safety_speed_active"):
                    entry["limit_reason"] = "safety_speed"
            pursuer_debug.append(entry)

        env._deploy_sce_debug = {
            "pursuers": pursuer_debug,
            "manifold_obstacles": manifold_obs,
            "lookahead_dist": float(pt_cfg.get("lookahead_dist", 0.8)),
        }
        if use_reachability_slots:
            env._deploy_sce_debug["candidate_slots"] = {
                "positions": cost_diag.get("candidate_slot_positions", []),
                "reachable": cost_diag.get("candidate_slot_reachable", []),
                "los_blocked": cost_diag.get("candidate_slot_los_blocked", []),
                "clearance": cost_diag.get("candidate_slot_clearance", []),
                "risk": cost_diag.get("candidate_slot_risk", []),
                "selected_indices": cost_diag.get("selected_candidate_indices", []),
                "fallback": bool(cost_diag.get("fallback_slot_selection", False)),
            }

    for k, v in step_diag.items():
        if k in st.episode_stats and v is not None:
            st.episode_stats[k].append(float(v))

    env._obstacle_aware_diagnostics = {
        **cost_diag,
        **step_diag,
        **st.timing.episode_summary(),
        "slot_projection": proj_diag,
        "slot_projection_moved": (
            st.slot_projection_moved.astype(int).tolist()
            if st.slot_projection_moved is not None else [0, 0, 0]
        ),
        "structure_slot_selection": slot_sel_diag,
    }
    if use_path_kind or use_turn_radius_slot:
        env._obstacle_aware_diagnostics["deploy_control"] = env._deploy_sce_debug
    if should_record_control_timing(env):
        pass  # no step-level publish to avoid viz overhead

    for i in range(3):
        uxy = np.asarray(actions[i, :2], dtype=np.float64)
        if yaws is not None:
            wx, wy = _body_vel_to_world(float(actions[i, 0]), float(actions[i, 1]), float(yaws[i]))
            uxy = np.array([wx, wy], dtype=np.float64)
        nrm = float(np.linalg.norm(uxy))
        if nrm > 1e-6:
            st.last_world_cmd_dirs[i] = uxy / nrm
    return actions


def _make_fn(env, kind, label, **kwargs):
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)
    xy, zg, yg = default_proportional_gains(
        low, high,
        xy_gain=kwargs.pop("xy_gain", 0.25),
        z_gain=kwargs.pop("z_gain", 0.20),
        yaw_gain=kwargs.pop("yaw_gain", 0.25),
    )
    reach = dict(kwargs.pop("reachability", {}) or {})
    cbf = dict(kwargs.pop("cbf", {}) or {})
    rates = dict(kwargs.pop("runtime_rates", {}) or {})
    turn_radius = dict(kwargs.pop("turn_radius_planner", {}) or {})
    obstacle_query_cfg = dict(kwargs.pop("obstacle_query", {}) or {})
    if obstacle_query_cfg:
        if "use_candidate_rollout_filter" in obstacle_query_cfg:
            turn_radius.setdefault(
                "use_candidate_rollout_filter",
                bool(obstacle_query_cfg["use_candidate_rollout_filter"]),
            )

    def get_actions(obs_list, state, avail_actions):
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError(f"Reset env before {label}")
        backend = env.prev_backend_state
        lin_pos = np.asarray(backend.states[:, 3, :], dtype=np.float32)
        lin_vel = np.asarray(backend.states[:, 2, :], dtype=np.float32)
        yaws = pursuer_yaws_from_backend(backend, env.task_state.pursuer_ids)
        return deployable_sce_actions_from_state(
            env, lin_pos, low, high, kind=kind,
            reachability=reach,
            cbf=cbf if kind in ("sce_cached_path_cbf_slot", "sce_reachability_cbf_slot") else None,
            runtime_rates=rates,
            turn_radius_planner=turn_radius,
            lin_vel=lin_vel,
            xy_gain=xy, z_gain=zg, yaw_gain=yg,
            pursuer_yaw=yaws,
            yaw_align_min_speed=float(kwargs.get("yaw_align_min_speed", 0.25)),
        )

    return get_actions


def make_sce_los_slot_get_actions_fn(env: Any, **kwargs: Any):
    return _make_fn(env, "sce_los_slot", "sce-los-slot", **kwargs)


def make_sce_turn_radius_slot_get_actions_fn(env: Any, **kwargs: Any):
    return _make_fn(env, "sce_turn_radius_slot", "sce-turn-radius-slot", **kwargs)


def make_sce_cached_path_slot_get_actions_fn(env: Any, **kwargs: Any):
    return _make_fn(env, "sce_cached_path_slot", "sce-cached-path-slot", **kwargs)


def make_sce_cached_path_cbf_slot_get_actions_fn(env: Any, **kwargs: Any):
    return _make_fn(env, "sce_cached_path_cbf_slot", "sce-cached-path-cbf-slot", **kwargs)


def make_sce_reachability_slot_get_actions_fn(env: Any, **kwargs: Any):
    return _make_fn(env, "sce_reachability_slot", "sce-reachability-slot", **kwargs)


def make_sce_reachability_cbf_slot_get_actions_fn(env: Any, **kwargs: Any):
    return _make_fn(env, "sce_reachability_cbf_slot", "sce-reachability-cbf-slot", **kwargs)


# Backward-compatible aliases (deprecated deployable names)
def make_sce_path_slot_get_actions_fn(env: Any, **kwargs: Any):
    return make_sce_cached_path_slot_get_actions_fn(env, **kwargs)


def make_sce_path_cbf_slot_get_actions_fn(env: Any, **kwargs: Any):
    return make_sce_cached_path_cbf_slot_get_actions_fn(env, **kwargs)


def episode_timing_summary(env: Any) -> dict[str, float]:
    st = getattr(env, "_deploy_sce_state", None)
    if st is None:
        return {}
    out = st.timing.episode_summary()
    out["path_replan_count"] = float(st.path_cache.stats["replan_count"])
    if st.path_cache.stats["replan_time_ms"]:
        out["avg_replan_time_ms"] = float(np.mean(st.path_cache.stats["replan_time_ms"]))
    for key, vals in st.episode_stats.items():
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[f"mean_{key}"] = float(np.mean(arr))
                if key == "local_planner_time_ms":
                    out["avg_local_planner_time_ms"] = float(np.mean(arr))
                    out["p95_local_planner_time_ms"] = float(np.percentile(arr, 95))
                if key == "local_obstacle_count":
                    out["avg_local_obstacle_count"] = float(np.mean(arr))
                if key == "valid_candidate_count":
                    out["avg_valid_candidate_count"] = float(np.mean(arr))
                if key == "local_planner_blocked":
                    out["blocked_step_rate"] = float(np.mean(arr))
    return out
