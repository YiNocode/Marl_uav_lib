"""E2 debug trajectory planner: manifold -> slots -> obstacle avoidance -> actions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from marl_uav.control.altitude_hold import apply_hard_altitude_to_action_row
from marl_uav.control.geometric_pursuit_baselines import (
    pursuer_yaws_from_backend,
)
from marl_uav.control.manifold_generator import (
    ManifoldGenerator,
    ManifoldSignature,
    build_manifold_signature,
    should_replan_manifold_paths,
)
from marl_uav.control.obstacle_avoidance_controller import ObstacleAvoidanceController
from marl_uav.control.slot_allocator import SlotAllocator
from marl_uav.control.slot_transition_manager import SlotTransitionManager
from marl_uav.framework.geometry.obstacle_adapter import manifold_influencing_obstacles, obstacles_from_task_state
from marl_uav.utils.control_timing import publish_control_timing, should_record_control_timing


@dataclass(frozen=True)
class SlotTargetStabilizerConfig:
    assignment_switch_margin: float = 0
    min_assignment_hold_steps: int = 500
    slot_filter_alpha: float = 1
    slot_target_vmax_ratio: float = 50
    control_dt: float = 0.04
    freeze_slots_after_first_step: bool = False
    freeze_assignment_after_first_step: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SlotTargetStabilizerConfig":
        d = dict(raw or {})
        return cls(
            assignment_switch_margin=float(
                d.get("assignment_switch_margin", cls.assignment_switch_margin)
            ),
            min_assignment_hold_steps=int(
                d.get("min_assignment_hold_steps", cls.min_assignment_hold_steps)
            ),
            slot_filter_alpha=float(d.get("slot_filter_alpha", cls.slot_filter_alpha)),
            slot_target_vmax_ratio=float(
                d.get("slot_target_vmax_ratio", cls.slot_target_vmax_ratio)
            ),
            control_dt=float(d.get("control_dt", cls.control_dt)),
            freeze_slots_after_first_step=bool(
                d.get("freeze_slots_after_first_step", cls.freeze_slots_after_first_step)
            ),
            freeze_assignment_after_first_step=bool(
                d.get("freeze_assignment_after_first_step", cls.freeze_assignment_after_first_step)
            ),
        )


class SlotTargetStabilizer:
    """Stateful smoothing/hysteresis layer between slot allocation and tracking."""

    def __init__(self) -> None:
        self.filtered_targets: np.ndarray | None = None
        self.prev_raw_assigned_targets: np.ndarray | None = None
        self.prev_assignment: np.ndarray | None = None
        self.assignment_age: np.ndarray | None = None
        self.prev_step: int | None = None

    def reset(self) -> None:
        self.filtered_targets = None
        self.prev_raw_assigned_targets = None
        self.prev_assignment = None
        self.assignment_age = None
        self.prev_step = None

    @staticmethod
    def _clip_xy_step(prev: np.ndarray, target: np.ndarray, max_step: float) -> tuple[np.ndarray, float]:
        out = np.asarray(target, dtype=np.float64).reshape(3).copy()
        delta_xy = out[:2] - np.asarray(prev, dtype=np.float64).reshape(3)[:2]
        shift = float(np.linalg.norm(delta_xy))
        if max_step > 0.0 and shift > max_step:
            out[:2] = prev[:2] + delta_xy * (max_step / max(shift, 1e-9))
            shift = float(max_step)
        return out, shift

    def update(
        self,
        *,
        slots: np.ndarray,
        raw_assignment: np.ndarray,
        raw_assigned_targets: np.ndarray,
        current_step: int,
        cfg: SlotTargetStabilizerConfig,
        tracking_vmax: float,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        slot_arr = np.asarray(slots, dtype=np.float64).reshape(3, 3)
        raw_assignment_arr = np.asarray(raw_assignment, dtype=np.int64).reshape(3)
        raw_targets_arr = np.asarray(raw_assigned_targets, dtype=np.float64).reshape(3, 3)
        dt = max(float(cfg.control_dt), 1e-6)
        alpha = float(np.clip(cfg.slot_filter_alpha, 0.0, 1.0))
        slot_target_vmax = max(float(cfg.slot_target_vmax_ratio), 0.0) * max(float(tracking_vmax), 0.0)
        step_delta = 1
        if self.prev_step is not None:
            step_delta = max(int(current_step) - int(self.prev_step), 1)
        dt_eff = dt * float(step_delta)

        if self.filtered_targets is None or self.prev_assignment is None:
            self.filtered_targets = raw_targets_arr.copy()
            self.prev_raw_assigned_targets = raw_targets_arr.copy()
            self.prev_assignment = raw_assignment_arr.copy()
            self.assignment_age = np.zeros(3, dtype=np.int64)
            self.prev_step = int(current_step)
            diagnostics = []
            for i in range(3):
                diagnostics.append(
                    {
                        "raw_slot_target_xy": raw_targets_arr[i, :2].astype(float).tolist(),
                        "stabilized_slot_target_xy": self.filtered_targets[i, :2].astype(float).tolist(),
                        "slot_target_speed": 0.0,
                        "slot_target_shift": 0.0,
                        "assignment_changed": False,
                        "assignment_hold_age": 0,
                        "slot_filter_alpha": float(alpha),
                        "slot_target_vmax": float(slot_target_vmax),
                    }
                )
            return self.prev_assignment.copy(), self.filtered_targets.copy(), diagnostics

        prev_filtered = np.asarray(self.filtered_targets, dtype=np.float64).reshape(3, 3)
        prev_assignment = np.asarray(self.prev_assignment, dtype=np.int64).reshape(3)
        prev_age = (
            np.zeros(3, dtype=np.int64)
            if self.assignment_age is None
            else np.asarray(self.assignment_age, dtype=np.int64).reshape(3)
        )
        effective_assignment = prev_assignment.copy()
        target_for_filter = np.zeros((3, 3), dtype=np.float64)
        new_age = prev_age + int(step_delta)
        assignment_changed = np.zeros(3, dtype=bool)
        margin = max(float(cfg.assignment_switch_margin), 0.0)
        hold_steps = max(int(cfg.min_assignment_hold_steps), 0)

        for i in range(3):
            old_slot = int(prev_assignment[i])
            new_slot = int(raw_assignment_arr[i])
            old_valid = 0 <= old_slot < slot_arr.shape[0]
            new_valid = 0 <= new_slot < slot_arr.shape[0]

            if bool(cfg.freeze_slots_after_first_step):
                chosen_slot = old_slot if old_valid else new_slot
                target_for_filter[i] = prev_filtered[i]
            elif bool(cfg.freeze_assignment_after_first_step):
                chosen_slot = old_slot if old_valid else new_slot
                target_for_filter[i] = slot_arr[chosen_slot] if 0 <= chosen_slot < slot_arr.shape[0] else raw_targets_arr[i]
            elif not new_valid:
                chosen_slot = old_slot
                target_for_filter[i] = slot_arr[old_slot] if old_valid else prev_filtered[i]
            elif (not old_valid) or new_slot == old_slot:
                chosen_slot = new_slot
                target_for_filter[i] = slot_arr[new_slot]
            else:
                old_target = slot_arr[old_slot]
                new_target = slot_arr[new_slot]
                hold_expired = bool(prev_age[i] >= hold_steps)
                old_dist = float(np.linalg.norm(old_target[:2] - prev_filtered[i, :2]))
                new_dist = float(np.linalg.norm(new_target[:2] - prev_filtered[i, :2]))
                accept_switch = hold_expired or (new_dist + margin < old_dist)
                chosen_slot = new_slot if accept_switch else old_slot
                target_for_filter[i] = slot_arr[chosen_slot]
                if accept_switch:
                    assignment_changed[i] = True
                    new_age[i] = 0

            effective_assignment[i] = int(chosen_slot)
            if int(chosen_slot) == int(prev_assignment[i]) and not bool(assignment_changed[i]):
                new_age[i] = int(prev_age[i]) + int(step_delta)

        valid_unique = (
            np.all(effective_assignment >= 0)
            and np.all(effective_assignment < slot_arr.shape[0])
            and len(set(int(x) for x in effective_assignment.tolist())) == effective_assignment.size
        )
        prev_valid_unique = (
            np.all(prev_assignment >= 0)
            and np.all(prev_assignment < slot_arr.shape[0])
            and len(set(int(x) for x in prev_assignment.tolist())) == prev_assignment.size
        )
        if not valid_unique and prev_valid_unique:
            effective_assignment = prev_assignment.copy()
            target_for_filter = slot_arr[effective_assignment].copy()
            assignment_changed[:] = False
            new_age = prev_age + int(step_delta)

        filtered = np.zeros_like(prev_filtered)
        diagnostics: list[dict[str, Any]] = []
        max_step = slot_target_vmax * dt_eff
        for i in range(3):
            low_passed = (1.0 - alpha) * prev_filtered[i] + alpha * target_for_filter[i]
            filtered[i], shift = self._clip_xy_step(prev_filtered[i], low_passed, max_step)
            speed = shift / max(dt_eff, 1e-6)
            diagnostics.append(
                {
                    "raw_slot_target_xy": raw_targets_arr[i, :2].astype(float).tolist(),
                    "stabilized_slot_target_xy": filtered[i, :2].astype(float).tolist(),
                    "slot_target_speed": float(speed),
                    "slot_target_shift": float(shift),
                    "assignment_changed": bool(assignment_changed[i]),
                    "assignment_hold_age": int(new_age[i]),
                    "slot_filter_alpha": float(alpha),
                    "slot_target_vmax": float(slot_target_vmax),
                }
            )

        self.filtered_targets = filtered.copy()
        self.prev_raw_assigned_targets = raw_targets_arr.copy()
        self.prev_assignment = effective_assignment.copy()
        self.assignment_age = new_age.astype(np.int64).copy()
        self.prev_step = int(current_step)
        return effective_assignment.copy(), filtered.copy(), diagnostics


@dataclass
class TrajectoryPlannerState:
    manifold: ManifoldGenerator
    allocator: SlotAllocator
    avoidance: ObstacleAvoidanceController
    slot_stabilizer: SlotTargetStabilizer = field(default_factory=SlotTargetStabilizer)
    prev_local_actions: np.ndarray = field(default_factory=lambda: np.zeros((3, 2), dtype=np.float64))
    last_step: int | None = None
    cached_manifold_curve: np.ndarray | None = None
    cached_pursuer_paths: list[np.ndarray] | None = None
    cached_assignment: np.ndarray | None = None
    cached_assigned_targets: np.ndarray | None = None
    prev_assigned_targets_for_ff: np.ndarray | None = None
    prev_slot_ff_step: int | None = None
    manifold_signature: ManifoldSignature | None = None
    manifold_version: int = 0
    slot_transition_managers: list[SlotTransitionManager] = field(default_factory=list)
    last_commanded_targets: np.ndarray | None = None
    last_proxy_targets: np.ndarray | None = None
    last_slot_transition_diag: list[dict[str, Any]] = field(default_factory=list)

    def reset_episode(self) -> None:
        self.allocator.reset()
        self.slot_stabilizer.reset()
        self.prev_local_actions = np.zeros((3, 2), dtype=np.float64)
        self.cached_manifold_curve = None
        self.cached_pursuer_paths = None
        self.cached_assignment = None
        self.cached_assigned_targets = None
        self.prev_assigned_targets_for_ff = None
        self.prev_slot_ff_step = None
        self.manifold_signature = None
        self.manifold_version = 0
        self.slot_transition_managers = []
        self.last_commanded_targets = None
        self.last_proxy_targets = None
        self.last_slot_transition_diag = []


def _get_state(env: Any, cfg: dict[str, Any]) -> TrajectoryPlannerState:
    st = getattr(env, "_trajectory_planner_state", None)
    if st is None:
        st = TrajectoryPlannerState(
            manifold=ManifoldGenerator(cfg.get("manifold_generator")),
            allocator=SlotAllocator(cfg.get("slot_allocator")),
            avoidance=ObstacleAvoidanceController(cfg.get("obstacle_avoidance")),
        )
        env._trajectory_planner_state = st
    return st


def _maybe_reset(env: Any, st: TrajectoryPlannerState) -> None:
    step = int(getattr(env, "step_count", 0))
    if step == 0 or st.last_step is None or step < int(st.last_step):
        st.reset_episode()
    st.last_step = step


def _fixed_ring_targets(
    evader_pos: np.ndarray,
    *,
    radius: float,
    z: float,
    phase: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    r = max(float(radius), 1e-6)
    theta = np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=np.float64)
    theta = theta + float(phase)
    slots = np.zeros((3, 3), dtype=np.float32)
    slots[:, 0] = np.float32(e[0]) + np.float32(r) * np.cos(theta).astype(np.float32)
    slots[:, 1] = np.float32(e[1]) + np.float32(r) * np.sin(theta).astype(np.float32)
    slots[:, 2] = np.float32(z)

    curve_theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=True, dtype=np.float64) + float(phase)
    curve = np.zeros((curve_theta.shape[0], 3), dtype=np.float32)
    curve[:, 0] = np.float32(e[0]) + np.float32(r) * np.cos(curve_theta).astype(np.float32)
    curve[:, 1] = np.float32(e[1]) + np.float32(r) * np.sin(curve_theta).astype(np.float32)
    curve[:, 2] = np.float32(z)
    diag = {
        "target_radius_xy_mean": float(r),
        "target_radius_xy_min": float(r),
        "target_radius_xy_max": float(r),
        "curve_num_samples": int(curve.shape[0]),
        "rho_base": float(r),
        "rho_max": float(r),
        "rho_min": float(r),
        "contraction_decay": 1.0,
        "manifold_contraction_rate": 0.0,
        "manifold_generation_disabled": True,
    }
    return slots, curve, diag




def _planner_bounds_xy(world_xy: float, boundary_margin: float) -> tuple[float, float, float, float] | None:
    """Return explicit XY bounds for the rollout filter.

    The rest of this stack already uses ``world_xy`` as the XY half-extent in
    ``apply_xy_boundary_barrier`` and in the local reachability scoring config.
    Passing the same convention here makes out-of-bounds candidates infeasible
    before the post-hoc boundary barrier has to override the selected action.
    """
    if not np.isfinite(world_xy) or world_xy <= 0.0:
        return None
    half = max(float(world_xy) - float(boundary_margin), 0.0)
    return (-half, half, -half, half)


def _apply_predictive_xy_boundary_guard(
    pos_xy: np.ndarray,
    u_world: np.ndarray,
    current_vel_xy: np.ndarray,
    *,
    world_xy: float | None,
    boundary_margin: float,
    boundary_gain: float,
    amax_xy: float,
    action_low_xy: np.ndarray,
    action_high_xy: np.ndarray,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """Axis-wise velocity guard that accounts for remaining braking distance."""
    if world_xy is None or not np.isfinite(float(world_xy)):
        u = np.asarray(u_world, dtype=np.float64).reshape(2).copy()
        return u, False, {"predictive_boundary_active_axes": []}

    w = max(float(world_xy), 1e-6)
    margin = float(np.clip(boundary_margin, 0.0, max(w - 1e-6, 0.0)))
    safe_min = -w + margin
    safe_max = w - margin
    k = max(float(boundary_gain), 0.0)
    amax = max(float(amax_xy), 1e-9)
    p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    u = np.asarray(u_world, dtype=np.float64).reshape(2).copy()
    u_before = u.copy()
    v = np.asarray(current_vel_xy, dtype=np.float64).reshape(2)
    low = np.asarray(action_low_xy, dtype=np.float64).reshape(2)
    high = np.asarray(action_high_xy, dtype=np.float64).reshape(2)

    active_axes: list[str] = []
    details: dict[str, Any] = {}
    axis_names = ("x", "y")
    for ax, name in enumerate(axis_names):
        d_upper = max(float(safe_max - p[ax]), 0.0)
        d_lower = max(float(p[ax] - safe_min), 0.0)

        upper_limit = min(k * d_upper, float(np.sqrt(2.0 * amax * d_upper)))
        lower_limit = min(k * d_lower, float(np.sqrt(2.0 * amax * d_lower)))
        stopping_upper = (max(float(v[ax]), 0.0) ** 2) / (2.0 * amax)
        stopping_lower = (max(float(-v[ax]), 0.0) ** 2) / (2.0 * amax)
        reason = None

        if u[ax] > 0.0:
            limited = min(float(u[ax]), upper_limit)
            if limited < u[ax] - 1e-9:
                u[ax] = limited
                reason = "outward_upper_limited"
        if v[ax] > 0.0 and stopping_upper > d_upper + 1e-9:
            inward_cap = max(abs(float(low[ax])), 0.0)
            brake_speed = min(inward_cap, max(abs(float(v[ax])), upper_limit))
            if brake_speed > 0.0:
                u[ax] = min(float(u[ax]), -brake_speed)
                reason = "upper_braking_override"

        if u[ax] < 0.0:
            limited = max(float(u[ax]), -lower_limit)
            if limited > u[ax] + 1e-9:
                u[ax] = limited
                reason = "outward_lower_limited"
        if v[ax] < 0.0 and stopping_lower > d_lower + 1e-9:
            inward_cap = max(float(high[ax]), 0.0)
            brake_speed = min(inward_cap, max(abs(float(v[ax])), lower_limit))
            if brake_speed > 0.0:
                u[ax] = max(float(u[ax]), brake_speed)
                reason = "lower_braking_override"

        u[ax] = float(np.clip(u[ax], low[ax], high[ax]))
        if abs(float(u[ax] - u_before[ax])) > 1e-9:
            active_axes.append(name)
        details[name] = {
            "distance_to_min": float(d_lower),
            "distance_to_max": float(d_upper),
            "allowed_outward_to_min": float(lower_limit),
            "allowed_outward_to_max": float(upper_limit),
            "stopping_distance_to_min": float(stopping_lower),
            "stopping_distance_to_max": float(stopping_upper),
            "reason": reason,
        }

    return u, bool(active_axes), {
        "predictive_boundary_active_axes": active_axes,
        "predictive_boundary_safe_min": float(safe_min),
        "predictive_boundary_safe_max": float(safe_max),
        "predictive_boundary_amax_xy": float(amax),
        "predictive_boundary_gain": float(k),
        "predictive_boundary_details": details,
    }


def _as_xy_radius(obstacle: Any) -> tuple[np.ndarray, float] | None:
    """Best-effort parser for obstacle records used only by the fast LOS branch.

    The controller still passes the original obstacle objects to the sampled
    local planner.  This parser only decides whether a straight line to the
    slot is obviously clear.  If an unknown obstacle format is encountered, the
    function returns None and the planner conservatively falls back to the
    sampled obstacle-aware branch.
    """
    if isinstance(obstacle, dict):
        xy_obj = None
        for key in ("xy", "center", "centre", "pos", "position", "center_xy"):
            if key in obstacle:
                xy_obj = obstacle[key]
                break
        if xy_obj is None and "x" in obstacle and "y" in obstacle:
            xy_obj = (obstacle["x"], obstacle["y"])
        if xy_obj is None:
            return None
        xy = np.asarray(xy_obj, dtype=np.float64).reshape(-1)
        if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
            return None
        radius = 0.0
        for key in ("radius", "r", "rad", "obstacle_radius", "cylinder_radius"):
            if key in obstacle:
                radius = float(obstacle[key])
                break
        return xy[:2].copy(), max(radius, 0.0)

    if isinstance(obstacle, (tuple, list)):
        arr = np.asarray(obstacle, dtype=object).reshape(-1)
        if arr.size >= 3:
            try:
                return np.asarray([arr[0], arr[1]], dtype=np.float64), max(float(arr[2]), 0.0)
            except Exception:
                return None
        if arr.size >= 2:
            try:
                return np.asarray([arr[0], arr[1]], dtype=np.float64), 0.0
            except Exception:
                return None

    xy_obj = None
    for key in ("xy", "center", "centre", "pos", "position", "center_xy"):
        if hasattr(obstacle, key):
            xy_obj = getattr(obstacle, key)
            break
    if xy_obj is None and hasattr(obstacle, "x") and hasattr(obstacle, "y"):
        xy_obj = (getattr(obstacle, "x"), getattr(obstacle, "y"))
    if xy_obj is None:
        return None
    xy = np.asarray(xy_obj, dtype=np.float64).reshape(-1)
    if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
        return None
    radius = 0.0
    for key in ("radius", "r", "rad", "obstacle_radius", "cylinder_radius"):
        if hasattr(obstacle, key):
            radius = float(getattr(obstacle, key))
            break
    return xy[:2].copy(), max(radius, 0.0)


def _point_segment_distance_xy(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    tau = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    proj = a + tau * ab
    return float(np.linalg.norm(point - proj))


def _control_dt_from_env(env: Any, task: Any, raw_cfg: dict[str, Any]) -> float:
    """Best-effort control interval for slot-velocity feed-forward.

    If this cannot be inferred from the environment, use top-level
    ``control_dt`` in the config.  A wrong dt only affects the optional
    feed-forward term; the feedback term still remains active.
    """
    for key in ("control_dt", "dt", "sim_dt", "physics_dt"):
        if key in raw_cfg:
            try:
                value = float(raw_cfg[key])
                if np.isfinite(value) and value > 0.0:
                    return value
            except Exception:
                pass
    for obj in (env, task):
        for key in ("control_dt", "dt", "sim_dt", "physics_dt", "time_step"):
            if hasattr(obj, key):
                try:
                    value = float(getattr(obj, key))
                    if np.isfinite(value) and value > 0.0:
                        return value
                except Exception:
                    pass
    return 1.0


def _slot_transition_enabled(raw_cfg: dict[str, Any]) -> bool:
    cfg = dict(raw_cfg.get("slot_transition", {}) or {})
    return bool(cfg.get("enabled", True))


def _apply_slot_transition_layer(
    st: TrajectoryPlannerState,
    *,
    raw_cfg: dict[str, Any],
    task: Any,
    assigned_targets: np.ndarray,
    pursuer_pos: np.ndarray,
    obstacles: list[Any],
    current_step: int,
    control_dt: float,
    tracking_vmax: float,
    world_xy: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    raw = np.asarray(assigned_targets, dtype=np.float64).reshape(3, 3)
    if not _slot_transition_enabled(raw_cfg):
        st.last_commanded_targets = raw.copy()
        st.last_proxy_targets = raw.copy()
        st.last_slot_transition_diag = []
        return raw.copy(), raw.copy(), []

    cfg = dict(raw_cfg.get("slot_transition", {}) or {})
    safety_cfg = dict(raw_cfg.get("obstacle_avoidance", {}) or {})
    uav_radius = float(cfg.get("uav_radius", safety_cfg.get("uav_radius", 0.15)))
    safety_margin = float(cfg.get("safety_margin", safety_cfg.get("safety_margin", 0.30)))
    vmax = float(cfg.get("slot_ref_vmax", max(float(tracking_vmax), 1e-6)))
    amax = float(cfg.get("slot_ref_amax", max(2.0 * vmax, 1e-6)))
    planner_cfg = dict(cfg.get("planner") or {})
    desired = {
        "world_xy": float(world_xy if np.isfinite(world_xy) and world_xy > 0.0 else getattr(task, "world_xy", 20.0)),
        "uav_radius": uav_radius,
        "safety_margin": safety_margin,
        "dt": max(float(control_dt), 1e-6),
        "slot_ref_vmax": vmax,
        "slot_ref_amax": amax,
        "jump_detection_threshold": float(cfg.get("jump_detection_threshold", 0.5)),
        "frequent_jump_min_interval_steps": int(cfg.get("frequent_jump_min_interval_steps", 20)),
        "high_freq_factor": float(cfg.get("high_freq_factor", 1.25)),
        "planner_cfg": planner_cfg,
    }
    if len(st.slot_transition_managers) != 3:
        st.slot_transition_managers = [SlotTransitionManager(**desired) for _ in range(3)]

    commanded = np.zeros_like(raw, dtype=np.float64)
    proxy = np.zeros_like(raw, dtype=np.float64)
    diags: list[dict[str, Any]] = []
    previous = st.last_commanded_targets
    ppos = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
    for i, mgr in enumerate(st.slot_transition_managers):
        prev_i = None
        if previous is not None and np.asarray(previous).shape == raw.shape:
            prev_i = np.asarray(previous, dtype=np.float64)[i]
        out = mgr.update(
            raw_slot_pos=raw[i],
            previous_commanded_slot_pos=prev_i,
            uav_pos=ppos[i],
            obstacles=obstacles,
            step=int(current_step),
        )
        commanded[i] = np.asarray(out["commanded_slot_pos"], dtype=np.float64).reshape(3)
        proxy[i] = np.asarray(out["proxy_slot_pos"], dtype=np.float64).reshape(3)
        diags.append(dict(out))

    st.last_commanded_targets = commanded.copy()
    st.last_proxy_targets = proxy.copy()
    st.last_slot_transition_diag = diags
    return commanded.copy(), proxy.copy(), diags


def _segment_is_clear_xy(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    clearance_margin: float,
    bounds_xy: tuple[float, float, float, float] | None,
) -> tuple[bool, float, str]:
    """Return whether the straight segment can be used as a fast path.

    If the line is clear, chasing the final slot directly is better than
    following a stale manifold path waypoint.  If an obstacle blocks the line or
    the obstacle format is unknown, we do not guess; the sampled local planner
    is used instead.
    """
    a = np.asarray(start_xy, dtype=np.float64).reshape(2)
    b = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    if bounds_xy is not None:
        xmin, xmax, ymin, ymax = bounds_xy
        for p in (a, b):
            if p[0] < xmin or p[0] > xmax or p[1] < ymin or p[1] > ymax:
                return False, 0.0, "segment_out_of_bounds"

    min_clearance = float("inf")
    for obs in list(obstacles or []):
        parsed = _as_xy_radius(obs)
        if parsed is None:
            return False, 0.0, "unknown_obstacle_format"
        center, radius = parsed
        clearance = _point_segment_distance_xy(center, a, b) - float(radius) - float(uav_radius)
        min_clearance = min(min_clearance, clearance)
        if clearance < float(clearance_margin):
            return False, float(min_clearance), "segment_blocked_by_obstacle"
    if min_clearance == float("inf"):
        min_clearance = 1e9
    return True, float(min_clearance), "segment_clear"


def _max_xy_displacement(a: np.ndarray | None, b: np.ndarray) -> float:
    """Maximum XY displacement between two target arrays."""
    if a is None:
        return float("inf")
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return float("inf")
    if aa.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(aa[:, :2] - bb[:, :2], axis=1)))


def _select_path_waypoint(
    path: np.ndarray | None,
    pos_xy: np.ndarray,
    final_goal_xy: np.ndarray,
    lookahead: float,
) -> np.ndarray:
    """Pick a reachable sub-goal on the cached path instead of always chasing the final slot.

    For obstacle cases, the final slot can be behind a cylinder. A single-step
    local planner should therefore track the next path waypoint; otherwise it
    repeatedly aims through the obstacle and only reacts when the obstacle is
    already inside the short rollout horizon.
    """
    goal = np.asarray(final_goal_xy, dtype=np.float64).reshape(2)
    if path is None:
        return goal
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return goal
    pts_xy = pts[:, :2]
    pos = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    finite = np.all(np.isfinite(pts_xy), axis=1)
    pts_xy = pts_xy[finite]
    if pts_xy.shape[0] < 2:
        return goal

    nearest = int(np.argmin(np.linalg.norm(pts_xy - pos[None, :], axis=1)))
    lookahead = max(float(lookahead), 1e-3)
    acc = 0.0
    prev = pos
    for j in range(nearest, pts_xy.shape[0]):
        seg = float(np.linalg.norm(pts_xy[j] - prev))
        acc += seg
        if acc >= lookahead:
            return pts_xy[j].copy()
        prev = pts_xy[j]
    return goal


def trajectory_planner_actions_from_state(
    env: Any,
    lin_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    lin_vel: np.ndarray | None = None,
    pursuer_yaw: np.ndarray | None = None,
) -> np.ndarray:
    raw_cfg = dict(cfg or {})
    st = _get_state(env, raw_cfg)
    _maybe_reset(env, st)

    t_total = time.perf_counter()
    task = env.task
    task_state = env.task_state
    pos = np.asarray(lin_pos, dtype=np.float32)
    pids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
    pursuer_pos = pos[pids]
    evader_pos = pos[int(task_state.evader_id)]
    yaws = (
        np.asarray(pursuer_yaw, dtype=np.float64).reshape(3)
        if pursuer_yaw is not None
        else np.zeros(3, dtype=np.float64)
    )
    # PyFlyt velocity mode consumes [vx, vy, vr, vz] where vx/vy/vz are
    # ground-frame linear velocities. Backend linear velocity is used in the
    # same ground/world frame for boundary prediction.
    vel_ground = (
        np.asarray(lin_vel, dtype=np.float64).reshape(-1, 3)[pids]
        if lin_vel is not None
        else np.zeros((3, 3), dtype=np.float64)
    )
    vel_world = np.asarray(vel_ground[:, :2], dtype=np.float64).reshape(3, 2).copy()

    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    action_cap_xy = float(np.max(np.abs(high[:2]))) if high.size >= 2 else float(st.avoidance.cfg.vmax)
    # The local planner must use the actual action envelope by default.
    # A stale hard-coded vmax=0.25 makes the UAVs crawl even when the action
    # space permits faster ground-frame XY commands.  Use top-level tracking_vmax
    # to intentionally cap the speed for ablation.
    tracking_vmax = float(raw_cfg.get("tracking_vmax", raw_cfg.get("vmax_xy", action_cap_xy)))
    if np.isfinite(action_cap_xy) and action_cap_xy > 0.0:
        tracking_vmax = min(max(tracking_vmax, 1e-6), action_cap_xy)
    obstacles = obstacles_from_task_state(task_state, task=task)
    world_xy = float(getattr(task, "world_xy", 0.0))
    boundary_margin = float(raw_cfg.get("boundary_margin", 0.30))
    boundary_alpha = float(raw_cfg.get("boundary_alpha", 2.0))
    boundary_activation_distance = float(raw_cfg.get("boundary_activation_distance", max(boundary_margin, 0.60)))
    boundary_hard_margin = float(raw_cfg.get("boundary_hard_margin", 0.05))
    boundary_braking_margin = float(raw_cfg.get("boundary_braking_margin", max(boundary_margin, boundary_hard_margin)))
    boundary_braking_gain = float(raw_cfg.get("boundary_braking_gain", boundary_alpha))
    max_inward_correction = float(raw_cfg.get("max_inward_correction", tracking_vmax))
    velocity_kd = float(raw_cfg.get("tracking_kd", raw_cfg.get("velocity_kd", raw_cfg.get("kd", 0.45))))
    disable_slot_allocation = bool(raw_cfg.get("disable_slot_allocation", False))
    disable_manifold_generation = bool(raw_cfg.get("disable_manifold_generation", False))
    disable_obstacle_avoidance = bool(raw_cfg.get("disable_obstacle_avoidance", False))
    curve_tol = float(raw_cfg.get("manifold_replan_curve_tol", 0.05))
    rho_tol = float(raw_cfg.get("manifold_replan_rho_tol", 0.001))
    current_step = int(getattr(env, "step_count", 0))
    path_target_replan_tol = float(raw_cfg.get("path_target_replan_tol", 0.25))
    path_replan_period = int(raw_cfg.get("path_replan_period", 5))
    path_waypoint_lookahead = float(raw_cfg.get("path_waypoint_lookahead", raw_cfg.get("lookahead_dist", 1.0)))
    bounds_xy = _planner_bounds_xy(world_xy, boundary_margin)
    stabilizer_raw = dict(raw_cfg.get("slot_target_stabilizer", {}) or {})
    stabilizer_cfg = SlotTargetStabilizerConfig.from_dict(
        {
            **raw_cfg,
            **stabilizer_raw,
            "control_dt": stabilizer_raw.get(
                "control_dt",
                raw_cfg.get("control_dt", raw_cfg.get("slot_control_dt", SlotTargetStabilizerConfig.control_dt)),
            ),
        }
    )
    st.avoidance.cfg = replace(
        st.avoidance.cfg,
        world_xy=world_xy,
        boundary_margin=boundary_margin,
        boundary_activation_distance=boundary_activation_distance,
        boundary_hard_margin=boundary_hard_margin,
        boundary_braking_margin=boundary_braking_margin,
        boundary_braking_gain=boundary_braking_gain,
        max_inward_correction=max_inward_correction,
        vmax=tracking_vmax,
        velocity_kd=velocity_kd,
        prefer_holonomic_tracking=True,
        use_sampled_planner=False,
    )
    lookahead_dist = float(raw_cfg.get("lookahead_dist", st.avoidance.cfg.lookahead_dist))
    st.avoidance.cfg = replace(st.avoidance.cfg, lookahead_dist=lookahead_dist)

    t_manifold = time.perf_counter()
    if disable_manifold_generation:
        fixed_radius = float(raw_cfg.get(
            "fixed_ring_radius",
            raw_cfg.get("ring_radius", 1.6 * float(getattr(task, "capture_dist", 1.0))),
        ))
        fixed_z = float(raw_cfg.get("fixed_ring_z", evader_pos[2]))
        fixed_phase = float(raw_cfg.get("fixed_ring_phase", getattr(task, "manifold_target_phase", 0.0)))
        slots, curve_new, manifold_diag = _fixed_ring_targets(
            evader_pos,
            radius=fixed_radius,
            z=fixed_z,
            phase=fixed_phase,
        )
    else:
        slots, curve_new, manifold_diag = st.manifold.generate(task, pursuer_pos, evader_pos, task_state)
    manifold_dt = time.perf_counter() - t_manifold

    t_assign = time.perf_counter()
    if disable_slot_allocation:
        raw_assignment = np.arange(3, dtype=np.int64)
        raw_assigned_targets = np.asarray(slots, dtype=np.float32).reshape(3, 3)[raw_assignment]
        allocation_diag = {
            "role_assignment": raw_assignment.astype(int).tolist(),
            "slot_allocation_disabled": True,
            "cost_matrix": np.zeros((3, 3), dtype=float).tolist(),
            "transport_plan": np.eye(3, dtype=float).tolist(),
            "ot_epsilon": 0.0,
            "los_blocked_matrix": np.zeros((3, 3), dtype=bool).tolist(),
            "reach_blocked_matrix": np.zeros((3, 3), dtype=bool).tolist(),
            "reachability_cost_matrix": np.zeros((3, 3), dtype=float).tolist(),
            "reach_min_clearance_matrix": np.full((3, 3), np.inf, dtype=float).tolist(),
        }
    else:
        raw_assignment, raw_assigned_targets, allocation_diag = st.allocator.allocate(
            pursuer_pos,
            slots,
            obstacles,
            pursuer_yaws=yaws,
            world_xy=world_xy,
        )
    assignment_dt = time.perf_counter() - t_assign

    assignment, assigned_targets, slot_stabilizer_diags = st.slot_stabilizer.update(
        slots=slots,
        raw_assignment=raw_assignment,
        raw_assigned_targets=raw_assigned_targets,
        current_step=current_step,
        cfg=stabilizer_cfg,
        tracking_vmax=tracking_vmax,
    )
    stabilized_assigned_targets = np.asarray(assigned_targets, dtype=np.float64).copy()
    control_dt = _control_dt_from_env(env, task, raw_cfg)
    assigned_targets, proxy_assigned_targets, slot_transition_diags = _apply_slot_transition_layer(
        st,
        raw_cfg=raw_cfg,
        task=task,
        assigned_targets=stabilized_assigned_targets,
        pursuer_pos=pursuer_pos,
        obstacles=[] if disable_obstacle_avoidance else obstacles,
        current_step=current_step,
        control_dt=control_dt,
        tracking_vmax=tracking_vmax,
        world_xy=world_xy,
    )
    task_state.assigned_target_indices = np.asarray(assignment, dtype=np.int64).copy()
    allocation_diag = {
        **allocation_diag,
        "slot_allocation_disabled": bool(disable_slot_allocation),
        "raw_assignment": raw_assignment.astype(int).tolist(),
        "raw_assigned_targets": np.asarray(raw_assigned_targets, dtype=np.float64).astype(float).tolist(),
        "stabilized_assignment": assignment.astype(int).tolist(),
        "stabilized_assigned_targets": stabilized_assigned_targets.astype(float).tolist(),
        "proxy_assigned_targets": np.asarray(proxy_assigned_targets, dtype=np.float64).astype(float).tolist(),
        "commanded_assigned_targets": np.asarray(assigned_targets, dtype=np.float64).astype(float).tolist(),
        "slot_transition": slot_transition_diags,
    }

    # Default to pure feedback. Slot-velocity feed-forward is useful only when
    # assignment is stable and the real control dt is known; otherwise it can
    # inject noisy target jumps and degrade tracking relative to pure pursuit.
    slot_velocity_ff_gain = float(raw_cfg.get("slot_velocity_ff_gain", 0.0))
    slot_velocity_ff_max = float(raw_cfg.get("slot_velocity_ff_max", 0.5 * tracking_vmax))
    predictive_boundary_guard_enabled = bool(raw_cfg.get("predictive_boundary_guard_enabled", False))
    predictive_boundary_amax_raw = raw_cfg.get(
        "predictive_boundary_amax_xy",
        raw_cfg.get("boundary_amax_xy", raw_cfg.get("amax_xy", st.avoidance.cfg.amax_xy)),
    )
    predictive_boundary_amax = (
        max(float(tracking_vmax) / max(float(control_dt), 1e-6), 1e-6)
        if predictive_boundary_amax_raw is None
        else max(float(predictive_boundary_amax_raw), 1e-6)
    )
    slot_velocity_ff = np.zeros((assigned_targets.shape[0], 2), dtype=np.float64)
    if (
        slot_velocity_ff_gain != 0.0
        and st.prev_assigned_targets_for_ff is not None
        and st.prev_assigned_targets_for_ff.shape == assigned_targets.shape
    ):
        step_delta = max(current_step - int(st.prev_slot_ff_step or current_step), 1)
        dt_ff = max(control_dt * float(step_delta), 1e-6)
        slot_velocity_ff = slot_velocity_ff_gain * (
            np.asarray(assigned_targets[:, :2], dtype=np.float64)
            - np.asarray(st.prev_assigned_targets_for_ff[:, :2], dtype=np.float64)
        ) / dt_ff
        if slot_velocity_ff_max > 0.0:
            for _j in range(slot_velocity_ff.shape[0]):
                _n = float(np.linalg.norm(slot_velocity_ff[_j]))
                if _n > slot_velocity_ff_max:
                    slot_velocity_ff[_j] *= slot_velocity_ff_max / max(_n, 1e-9)
    st.prev_assigned_targets_for_ff = np.asarray(assigned_targets, dtype=np.float64).copy()
    st.prev_slot_ff_step = int(current_step)

    new_sig = build_manifold_signature(st.cached_manifold_curve, curve_new, manifold_diag)
    replanned = should_replan_manifold_paths(
        st.manifold_signature,
        new_sig,
        st.cached_assignment,
        assignment,
        curve_tol=curve_tol,
        rho_tol=rho_tol,
    )
    target_shift = _max_xy_displacement(st.cached_assigned_targets, assigned_targets)
    if target_shift > path_target_replan_tol:
        replanned = True
    if path_replan_period > 0 and current_step > 0 and current_step % path_replan_period == 0:
        replanned = True
    if replanned or st.cached_pursuer_paths is None:
        st.cached_manifold_curve = np.asarray(curve_new, dtype=np.float64).copy()
        st.cached_pursuer_paths = st.manifold.generate_pursuer_paths(
            pursuer_pos,
            assigned_targets,
            curve_new,
            evader_pos=evader_pos,
        )
        st.cached_assignment = np.asarray(assignment, dtype=np.int64).copy()
        st.cached_assigned_targets = np.asarray(assigned_targets, dtype=np.float64).copy()
        st.manifold_signature = new_sig
        st.manifold_version += 1

    assert st.cached_pursuer_paths is not None

    t_avoid = time.perf_counter()
    actions = np.zeros((3, low.shape[0]), dtype=np.float32)
    follow_diags: list[dict[str, Any]] = []
    for i in range(3):
        final_goal_xy = assigned_targets[i, :2]
        direct_clear, direct_clearance, direct_reason = _segment_is_clear_xy(
            pursuer_pos[i, :2],
            final_goal_xy,
            [] if disable_obstacle_avoidance else obstacles,
            uav_radius=float(st.avoidance.cfg.uav_radius),
            clearance_margin=float(st.avoidance.cfg.safety_margin),
            bounds_xy=bounds_xy,
        )
        # The local obstacle module is used as a safety layer only. The nominal
        # objective remains the stabilized assigned slot, never a replacement
        # waypoint from an obstacle/path planner.
        waypoint_xy = np.asarray(final_goal_xy, dtype=np.float64).reshape(2)
        waypoint_mode = "stabilized_final_slot"
        action_xy_w, yaw_rate, _best_path, diag = st.avoidance.compute_action(
            pursuer_pos[i, :2],
            float(yaws[i]),
            waypoint_xy,
            [] if disable_obstacle_avoidance else obstacles,
            prev_action=st.prev_local_actions[i],
            current_velocity_xy=vel_world[i],
            bounds_xy=bounds_xy,
            feedforward_velocity_xy=slot_velocity_ff[i],
        )
        diag["final_slot_distance"] = float(np.linalg.norm(assigned_targets[i, :2] - pursuer_pos[i, :2]))
        diag["tracking_waypoint_xy"] = np.asarray(waypoint_xy, dtype=np.float64).astype(float).tolist()
        diag["direct_slot_los"] = bool(direct_clear)
        diag["direct_slot_los_reason"] = str(direct_reason)
        diag["direct_slot_clearance"] = float(direct_clearance)
        diag["tracking_vmax"] = float(tracking_vmax)
        diag["action_cap_xy"] = float(action_cap_xy)
        diag["slot_velocity_ff_xy"] = np.asarray(slot_velocity_ff[i], dtype=np.float64).astype(float).tolist()
        diag["slot_velocity_ff_candidate_xy"] = np.asarray(slot_velocity_ff[i], dtype=np.float64).astype(float).tolist()
        diag["slot_velocity_ff_gain"] = float(slot_velocity_ff_gain)
        diag["slot_velocity_ff_max"] = float(slot_velocity_ff_max)
        diag["slot_velocity_ff_applied"] = bool(np.linalg.norm(slot_velocity_ff[i]) > 1e-9)
        diag["waypoint_mode"] = str(waypoint_mode)
        diag["obstacle_avoidance_disabled"] = bool(disable_obstacle_avoidance)
        diag["control_dt"] = float(control_dt)
        diag["raw_slot_id"] = int(raw_assignment[i])
        diag["raw_slot_target_xy"] = np.asarray(raw_assigned_targets[i, :2], dtype=np.float64).astype(float).tolist()
        diag["stabilized_slot_id"] = int(assignment[i])
        diag["proxy_slot_target_xy"] = np.asarray(proxy_assigned_targets[i, :2], dtype=np.float64).astype(float).tolist()
        diag["commanded_slot_target_xy"] = np.asarray(assigned_targets[i, :2], dtype=np.float64).astype(float).tolist()
        diag["slot_transition"] = slot_transition_diags[i] if i < len(slot_transition_diags) else {}
        diag.update(slot_stabilizer_diags[i])
        diag["nominal_tracking_mode"] = "kp_to_commanded_slot_then_safety_layer"
        u_safe = np.asarray(action_xy_w, dtype=np.float64).reshape(2).copy()
        boundary_active = bool(diag.get("boundary_filter_active", False))
        if predictive_boundary_guard_enabled:
            u_safe, predictive_boundary_active, predictive_boundary_diag = _apply_predictive_xy_boundary_guard(
                pursuer_pos[i, :2],
                u_safe,
                vel_world[i],
                world_xy=world_xy,
                boundary_margin=boundary_margin,
                boundary_gain=boundary_alpha,
                amax_xy=predictive_boundary_amax,
                action_low_xy=low[:2],
                action_high_xy=high[:2],
            )
            boundary_active = bool(boundary_active or predictive_boundary_active)
            diag.update(predictive_boundary_diag)
            diag["predictive_boundary_active"] = bool(predictive_boundary_active)
        else:
            diag["predictive_boundary_active"] = False
            diag["predictive_boundary_active_axes"] = []
            diag["predictive_boundary_amax_xy"] = float(predictive_boundary_amax)
            diag["predictive_boundary_gain"] = float(boundary_alpha)
        diag["componentwise_boundary_filter_active"] = bool(boundary_active)
        diag["componentwise_boundary_active_names"] = list(diag.get("boundary_active_names", []))
        actions[i, 0] = np.float32(float(u_safe[0]))
        actions[i, 1] = np.float32(float(u_safe[1]))
        if actions.shape[1] >= 3:
            actions[i, 2] = np.float32(yaw_rate)
        if actions.shape[1] >= 4:
            actions[i, 3] = np.float32(0.0)

        limit_reason = "obstacle_avoidance"
        if bool(diag.get("local_planner_blocked")):
            limit_reason = "local_planner_blocked"
        elif boundary_active:
            limit_reason = "xy_boundary"

        fd = dict(diag)
        fd["limit_reason"] = limit_reason
        fd["track_dist_xy"] = float(diag.get("assigned_slot_distance", 0.0))
        follow_diags.append(fd)
        st.prev_local_actions[i, 0] = float(diag.get("best_candidate_speed", np.linalg.norm(u_safe)))
        st.prev_local_actions[i, 1] = float(diag.get("best_candidate_yaw_rate", yaw_rate))
    avoid_dt = time.perf_counter() - t_avoid

    debug_pursuers: list[dict[str, Any]] = []
    for i in range(3):
        if actions.shape[1] >= 4:
            z_hold = float(assigned_targets[i, 2])
            apply_hard_altitude_to_action_row(
                actions[i],
                float(pursuer_pos[i, 2]),
                z_hold,
                low,
                high,
                z_floor=float(getattr(task, "z_min", 0.0)),
                z_ceiling=float(getattr(task, "z_max", 5.0)),
                floor_margin=float(raw_cfg.get("altitude_floor_margin", 0.25)),
                ceiling_margin=float(raw_cfg.get("altitude_ceiling_margin", 0.10)),
            )

        fd = dict(follow_diags[i])
        debug_pursuers.append(
            {
                "slot_id": int(assignment[i]),
                "slot_target_xy": stabilized_assigned_targets[i, :2].astype(float).tolist(),
                "raw_slot_id": int(raw_assignment[i]),
                "raw_slot_target_xy": np.asarray(raw_assigned_targets[i, :2], dtype=np.float64).astype(float).tolist(),
                "stabilized_slot_target_xy": stabilized_assigned_targets[i, :2].astype(float).tolist(),
                "proxy_slot_target_xy": np.asarray(proxy_assigned_targets[i, :2], dtype=np.float64).astype(float).tolist(),
                "commanded_slot_target_xy": assigned_targets[i, :2].astype(float).tolist(),
                "slot_transition": slot_transition_diags[i] if i < len(slot_transition_diags) else {},
                "speed_cmd_xy": float(np.hypot(actions[i, 0], actions[i, 1])),
                "world_speed_cmd_xy": float(fd.get("selected_action_norm", 0.0)),
                "speed_cap_xy": float(action_cap_xy),
                "tracking_vmax_xy": float(tracking_vmax),
                "limit_reason": str(fd.get("limit_reason", "obstacle_avoidance")),
                "track_dist_xy": float(fd.get("track_dist_xy", 0.0)),
                "trajectory_planner": {
                    **fd,
                    "manifold_version": int(st.manifold_version),
                    "manifold_replanned": bool(replanned),
                    "manifold_change_metric": float(new_sig.max_curve_displacement),
                    "path_target_shift": float(target_shift),
                    "path_waypoint_lookahead": float(path_waypoint_lookahead),
                },
            }
        )

    actions = np.clip(actions, low[None, :], high[None, :]).astype(np.float32)
    for i in range(3):
        vx = float(actions[i, 0]) if actions.shape[1] >= 1 else 0.0
        vy = float(actions[i, 1]) if actions.shape[1] >= 2 else 0.0
        cmd_diag = {
            "backend_cmd_ground_xy": [float(vx), float(vy)],
            "backend_cmd_world_xy": [float(vx), float(vy)],
            "backend_cmd_speed_ground_xy": float(np.hypot(vx, vy)),
            "backend_cmd_speed_world_xy": float(np.hypot(vx, vy)),
            "backend_cmd_action_xy_indices": [0, 1],
            "backend_cmd_action_layout": "[vx_ground, vy_ground, vr, vz_ground]",
        }
        debug_pursuers[i].update(cmd_diag)
        tp = debug_pursuers[i].get("trajectory_planner")
        if isinstance(tp, dict):
            tp.update(cmd_diag)
            tp["sent_action"] = actions[i].astype(float).tolist()

    target_radius = float(manifold_diag.get("target_radius_xy_mean", getattr(task, "capture_dist", 1.0)))
    manifold_obs = manifold_influencing_obstacles(
        evader_pos[:2],
        obstacles,
        capture_dist=float(getattr(task, "capture_dist", 1.0)),
        target_radius_xy=target_radius,
        top_k=int(getattr(task, "obstacle_manifold_top_k", 4)),
        influence_radius_scale=float(getattr(task, "obstacle_manifold_influence_radius_scale", 2.5)),
        clearance_margin_scale=float(getattr(task, "obstacle_manifold_clearance_margin_scale", 0.35)),
    )
    display_curve = np.asarray(curve_new, dtype=np.float64)
    deploy_control = {
        "pursuers": debug_pursuers,
        "manifold_obstacles": manifold_obs,
        "lookahead_dist": float(lookahead_dist),
        "local_planner": "trajectory_planner",
        "manifold_curve_xy": display_curve[:, :2].astype(float).tolist(),
        "manifold_version": int(st.manifold_version),
        "manifold_replanned": bool(replanned),
    }
    total_dt = time.perf_counter() - t_total
    diagnostics = {
        "deploy_control": deploy_control,
        "trajectory_planner": {
            "manifold": {
                **manifold_diag,
                "manifold_version": int(st.manifold_version),
                "manifold_replanned": bool(replanned),
                "manifold_change_metric": float(new_sig.max_curve_displacement),
                "path_target_shift": float(target_shift),
                "manifold_generation_disabled": bool(disable_manifold_generation),
            },
            "allocation": allocation_diag,
            "ablation": {
                "disable_slot_allocation": bool(disable_slot_allocation),
                "disable_manifold_generation": bool(disable_manifold_generation),
                "disable_obstacle_avoidance": bool(disable_obstacle_avoidance),
            },
        },
        "mean_follow_time_ms": float(avoid_dt * 1000.0),
        "decision_total_ms": float(total_dt * 1000.0),
    }
    env._trajectory_planner_debug = {
        "slot_targets": slots.astype(float).tolist(),
        "manifold_curve": display_curve.astype(float).tolist(),
        "assignment": assignment.astype(int).tolist(),
        "raw_assignment": raw_assignment.astype(int).tolist(),
        "raw_assigned_targets": np.asarray(raw_assigned_targets, dtype=np.float64).astype(float).tolist(),
        "stabilized_assigned_targets": stabilized_assigned_targets.astype(float).tolist(),
        "proxy_assigned_targets": np.asarray(proxy_assigned_targets, dtype=np.float64).astype(float).tolist(),
        "commanded_assigned_targets": np.asarray(assigned_targets, dtype=np.float64).astype(float).tolist(),
        "slot_transition": slot_transition_diags,
        "manifold_version": int(st.manifold_version),
        "manifold_replanned": bool(replanned),
        "deploy_control": deploy_control,
    }
    env._obstacle_aware_diagnostics = diagnostics
    if display_curve is not None:
        task_state.reference_manifold_curve = display_curve.astype(np.float32)
    task_state.reference_manifold_targets = slots

    if should_record_control_timing(env):
        publish_control_timing(
            env,
            manifold_update_time=manifold_dt,
            slot_assignment_time=assignment_dt,
            action_mapping_time=max(avoid_dt, 0.0),
            total_decision_latency=total_dt,
        )
    return actions


def make_trajectory_planner_get_actions_fn(env: Any, **kwargs: Any):
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError("trajectory_planner requires a continuous action-space env")
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)
    cfg = dict(kwargs or {})

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError("Environment must be reset before selecting trajectory_planner actions.")
        backend = env.prev_backend_state
        lin_pos = np.asarray(backend.states[:, 3, :], dtype=np.float32)
        lin_vel = np.asarray(backend.states[:, 2, :], dtype=np.float32)
        yaws = pursuer_yaws_from_backend(backend, env.task_state.pursuer_ids)
        return trajectory_planner_actions_from_state(
            env,
            lin_pos,
            low,
            high,
            cfg=cfg,
            lin_vel=lin_vel,
            pursuer_yaw=yaws,
        )

    return get_actions
