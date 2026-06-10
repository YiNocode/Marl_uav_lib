"""Run the independent low-level slot-tracking benchmark."""

from __future__ import annotations

import argparse
import ast
import csv
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from experiments.slot_tracking.controllers.baseline_pure_pursuit import ControllerObservation, clip_norm
from experiments.slot_tracking.controllers.slot_transition_manager import SlotTransitionManager
from experiments.slot_tracking.controllers.wrapped_existing_controller import make_controller
from experiments.slot_tracking.metrics.failure_classifier import classify_failure, classify_failure_subtype, episode_success
from experiments.slot_tracking.metrics.safety_metrics import (
    boundary_margin,
    compute_safety_metrics,
    inter_agent_collision,
    obstacle_clearance,
    obstacle_collision,
    outside_boundary,
)
from experiments.slot_tracking.metrics.tracking_metrics import compute_tracking_metrics
from experiments.slot_tracking.scenarios.obstacle_maps import obstacle_records
from experiments.slot_tracking.scenarios.slot_scenarios import (
    ScenarioInstance,
    ScenarioSpec,
    instantiate_scenario,
    scenario_specs,
)
from marl_uav.framework.geometry.obstacle_geometry import has_line_of_sight


RAW_COLUMNS = [
    "t",
    "agent_id",
    "uav_x",
    "uav_y",
    "uav_z",
    "slot_x",
    "slot_y",
    "slot_z",
    "commanded_slot_x",
    "commanded_slot_y",
    "commanded_slot_z",
    "raw_slot_x",
    "raw_slot_y",
    "raw_slot_z",
    "proxy_slot_x",
    "proxy_slot_y",
    "proxy_slot_z",
    "uav_vx",
    "uav_vy",
    "uav_vz",
    "action_x",
    "action_y",
    "action_z",
    "tracking_error",
    "failure_step",
    "episode_end_step",
    "simulation_continues_after_failure",
    "post_failure_state_included",
    "tracking_error_at_failure_step",
    "max_tracking_error_before_failure_step",
    "p95_error_pre_failure",
    "p95_error_full_episode",
    "tracking_error_to_raw_slot",
    "tracking_error_to_proxy_slot",
    "tracking_error_to_subgoal",
    "nearest_obstacle_distance",
    "nearest_obstacle_id",
    "nearest_obstacle_name",
    "raw_slot_inside_obstacle",
    "raw_slot_too_close_obstacle",
    "proxy_slot_adjusted",
    "boundary_margin",
    "decision_time_ms",
    "controller_base",
    "target_source",
    "shared_predictive_safety",
    "nearest_boundary_name",
    "outward_normal_x",
    "outward_normal_y",
    "velocity_outward_projection",
    "action_outward_projection",
    "distance_to_boundary",
    "estimated_braking_distance",
    "braking_margin",
    "boundary_filter_active",
    "boundary_active_names",
    "boundary_projected_action_failed",
    "action_before_boundary_filter_x",
    "action_before_boundary_filter_y",
    "action_after_boundary_filter_x",
    "action_after_boundary_filter_y",
    "action_outward_projection_before_filter",
    "v_nom_x",
    "v_nom_y",
    "v_before_boundary_filter_x",
    "v_before_boundary_filter_y",
    "v_after_boundary_projection_x",
    "v_after_boundary_projection_y",
    "v_after_obstacle_filter_x",
    "v_after_obstacle_filter_y",
    "v_after_predictive_safety_x",
    "v_after_predictive_safety_y",
    "v_after_planner_x",
    "v_after_planner_y",
    "v_planner_subgoal_x",
    "v_planner_subgoal_y",
    "v_after_braking_correction_x",
    "v_after_braking_correction_y",
    "v_after_speed_clip_x",
    "v_after_speed_clip_y",
    "v_after_accel_clip_x",
    "v_after_accel_clip_y",
    "v_final_x",
    "v_final_y",
    "action_final_x",
    "action_final_y",
    "norm_v_nom",
    "norm_v_before_boundary_filter",
    "norm_v_after_boundary_projection",
    "norm_v_after_braking_correction",
    "norm_v_after_speed_clip",
    "norm_v_after_accel_clip",
    "norm_v_final",
    "angle_nom_to_boundary_projection",
    "angle_nom_to_final",
    "norm_ratio_final_to_nom",
    "speed_clip_applied",
    "accel_clip_applied",
    "post_boundary_projection_applied",
    "x_min_action_outward_projection_before",
    "x_min_action_outward_projection_after",
    "x_min_velocity_outward_projection",
    "x_min_boundary_margin",
    "x_min_constraint_bound",
    "x_min_braking_margin",
    "x_min_normal_component_before",
    "x_min_normal_component_after",
    "x_min_tangential_norm_before",
    "x_min_tangential_norm_after",
    "x_min_tangential_retention_ratio",
    "x_min_inward_correction",
    "x_max_action_outward_projection_before",
    "x_max_action_outward_projection_after",
    "x_max_velocity_outward_projection",
    "x_max_boundary_margin",
    "x_max_constraint_bound",
    "x_max_braking_margin",
    "x_max_normal_component_before",
    "x_max_normal_component_after",
    "x_max_tangential_norm_before",
    "x_max_tangential_norm_after",
    "x_max_tangential_retention_ratio",
    "x_max_inward_correction",
    "y_min_action_outward_projection_before",
    "y_min_action_outward_projection_after",
    "y_min_velocity_outward_projection",
    "y_min_boundary_margin",
    "y_min_constraint_bound",
    "y_min_braking_margin",
    "y_min_normal_component_before",
    "y_min_normal_component_after",
    "y_min_tangential_norm_before",
    "y_min_tangential_norm_after",
    "y_min_tangential_retention_ratio",
    "y_min_inward_correction",
    "y_max_action_outward_projection_before",
    "y_max_action_outward_projection_after",
    "y_max_velocity_outward_projection",
    "y_max_boundary_margin",
    "y_max_constraint_bound",
    "y_max_braking_margin",
    "y_max_normal_component_before",
    "y_max_normal_component_after",
    "y_max_tangential_norm_before",
    "y_max_tangential_norm_after",
    "y_max_tangential_retention_ratio",
    "y_max_inward_correction",
    "v_goal_x",
    "v_goal_y",
    "v_obstacle_x",
    "v_obstacle_y",
    "obstacle_repulsion_norm",
    "v_boundary_x",
    "v_boundary_y",
    "v_path_x",
    "v_path_y",
    "v_smooth_x",
    "v_smooth_y",
    "v_final_before_clip_x",
    "v_final_before_clip_y",
    "v_final_after_clip_x",
    "v_final_after_clip_y",
    "final_action_x",
    "final_action_y",
    "clip_flag",
    "speed_saturation_flag",
    "acceleration_saturation_flag",
    "double_clip_warning",
    "cos_to_goal",
    "progress_projection",
    "distance_delta",
    "existing_bypass_reason",
    "used_existing_safety_modules",
    "obstacle_risk",
    "boundary_risk",
    "inter_agent_risk",
    "line_of_sight_blocked",
    "planner_called",
    "planner_success",
    "planner_mode_active",
    "replan_reason",
    "planner_path_length",
    "planner_replan_count",
    "current_path_id",
    "current_path_waypoints",
    "current_path_progress_index",
    "current_path_arclength_progress",
    "path_valid",
    "path_deviation",
    "lookahead_distance",
    "target_mode",
    "target_mode_previous",
    "target_mode_transition_reason",
    "raw_slot_valid",
    "commanded_slot_valid",
    "commanded_slot_inside_obstacle",
    "commanded_slot_outside_boundary",
    "commanded_slot_too_close_obstacle",
    "commanded_transition_segment_safe",
    "commanded_slot_invalid_reason",
    "safe_hold_active",
    "safe_hold_reason",
    "proxy_slot_used",
    "slot_transition_mode",
    "slot_transition_mode_previous",
    "jump_detected",
    "jump_distance",
    "minimum_reach_time",
    "jump_interval_steps",
    "raw_slot_too_unstable",
    "transition_progress",
    "commanded_slot_lag_to_raw",
    "commanded_slot_step_norm",
    "desired_velocity_before_final_clip_x",
    "desired_velocity_before_final_clip_y",
    "final_velocity_after_clip_x",
    "final_velocity_after_clip_y",
    "double_clip_flag",
    "clip_stage_names",
    "proxy_slot_valid",
    "los_to_raw",
    "los_to_proxy",
    "selected_subgoal_valid",
    "stale_subgoal_used",
    "target_mode_invariant_violation",
    "target_mode_invariant",
    "current_subgoal_x",
    "current_subgoal_y",
    "current_subgoal_z",
    "planned_path",
    "current_path_segment",
    "selected_subgoal_path",
    "min_distance_to_planned_path",
    "min_distance_to_current_segment",
    "min_distance_to_subgoal_segment",
    "line_of_sight_to_proxy",
    "line_of_sight_to_raw_slot",
    "line_of_sight_to_proxy_slot",
    "nominal_rollout_safe",
    "nominal_rollout_min_clearance",
    "line_of_sight_to_subgoal",
    "path_blocked",
    "subgoal_visible",
    "selected_subgoal_visible",
    "lookahead_segment_clearance",
    "no_visible_subgoal",
    "planned_path_min_clearance",
    "planned_path_mean_clearance",
    "subgoal_backtracked",
    "backtracked_waypoint_index",
    "proxy_slot_shift_norm",
    "subgoal_shift_norm",
    "subgoal_held",
    "collision_radius",
    "safety_margin",
    "tracking_margin",
    "planning_inflation_radius",
    "steps_since_last_replan",
    "replan_allowed_by_cooldown",
    "replan_blocked_by_cooldown",
    "predicted_next_pos_x",
    "predicted_next_pos_y",
    "segment_collision_current_to_next",
    "min_predicted_next_clearance",
    "predictive_filter_active",
    "predictive_filter_subtype",
    "stuck_detector_active",
    "cos_to_subgoal",
    "cos_to_raw_slot",
    "cos_to_proxy_slot",
    "cos_to_commanded_slot",
    "progress_to_subgoal",
    "progress_to_raw_slot",
    "progress_to_proxy_slot",
    "progress_to_commanded_slot",
    "obstacle_collision",
    "boundary_violation",
    "inter_agent_collision",
    "slot_outside_boundary",
    "failure_type",
    "failure_subtype",
]


class ProgressBar:
    """Small dependency-free terminal progress bar for benchmark episodes."""

    def __init__(self, total: int, *, width: int = 28, enabled: bool = True) -> None:
        self.total = max(int(total), 0)
        self.width = max(int(width), 8)
        self.enabled = bool(enabled)
        self.start = time.perf_counter()
        self.last_render = 0.0
        self.count = 0

    def update(self, count: int, *, label: str = "") -> None:
        if not self.enabled or self.total <= 0:
            return
        self.count = min(max(int(count), 0), self.total)
        now = time.perf_counter()
        if self.count < self.total and now - self.last_render < 0.2:
            return
        self.last_render = now
        frac = self.count / max(float(self.total), 1.0)
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        rate = self.count / max(elapsed, 1e-9)
        remaining = (self.total - self.count) / max(rate, 1e-9)
        suffix = f" | {label}" if label else ""
        sys.stdout.write(
            "\r"
            f"[{bar}] {self.count}/{self.total} "
            f"{100.0 * frac:5.1f}% elapsed {format_duration(elapsed)} "
            f"eta {format_duration(remaining)}{suffix}"
        )
        sys.stdout.flush()

    def close(self) -> None:
        if self.enabled and self.total > 0:
            self.update(self.total)
            sys.stdout.write("\n")
            sys.stdout.flush()


def format_duration(seconds: float) -> str:
    """Format seconds for compact progress output."""
    if not np.isfinite(seconds) or seconds < 0.0:
        return "--:--"
    total = int(round(float(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _diag_vec(diag: dict[str, Any], key: str) -> np.ndarray:
    value = diag.get(key, [float("nan"), float("nan")])
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 2:
            return arr[:2].copy()
    except Exception:
        pass
    return np.array([float("nan"), float("nan")], dtype=np.float64)


def _vec_columns(prefix: str, vec: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vec, dtype=np.float64).reshape(2)
    return {f"{prefix}_x": float(arr[0]), f"{prefix}_y": float(arr[1])}


def _boundary_detail_columns(diag: dict[str, Any]) -> dict[str, float]:
    details = diag.get("boundary_filter_details", {})
    out: dict[str, float] = {}
    fields = [
        "action_outward_projection_before",
        "action_outward_projection_after",
        "velocity_outward_projection",
        "boundary_margin",
        "constraint_bound",
        "braking_margin",
        "normal_component_before",
        "normal_component_after",
        "tangential_norm_before",
        "tangential_norm_after",
        "tangential_retention_ratio",
        "inward_correction",
    ]
    for name in ["x_min", "x_max", "y_min", "y_max"]:
        raw = details.get(name, {}) if isinstance(details, dict) else {}
        for field in fields:
            out[f"{name}_{field}"] = float(raw.get(field, np.nan)) if isinstance(raw, dict) else float("nan")
    return out


def _angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float64).reshape(2)
    b = np.asarray(vec_b, dtype=np.float64).reshape(2)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return float("nan")
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.arccos(cos))


def _stage_norm_columns(stages: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in stages.items():
        out[f"norm_{name}"] = float(np.linalg.norm(np.asarray(value, dtype=np.float64).reshape(2)))
    return out


def _goal_progress_fields(pos_xy: np.ndarray, slot_xy: np.ndarray, action_xy: np.ndarray) -> dict[str, float]:
    goal = np.asarray(slot_xy, dtype=np.float64).reshape(2) - np.asarray(pos_xy, dtype=np.float64).reshape(2)
    dist = float(np.linalg.norm(goal))
    if dist <= 1e-9:
        return {"cos_to_goal": float("nan"), "progress_projection": 0.0}
    goal_dir = goal / dist
    action = np.asarray(action_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(action))
    progress = float(np.dot(action, goal_dir))
    cos = float(progress / max(norm, 1e-9))
    return {"cos_to_goal": cos, "progress_projection": progress}


def _nearest_boundary_diag(
    pos_xy: np.ndarray,
    vel_xy: np.ndarray,
    action_xy: np.ndarray,
    *,
    world_xy: float,
    uav_radius: float,
    amax: float,
) -> dict[str, Any]:
    p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    v = np.asarray(vel_xy, dtype=np.float64).reshape(2)
    action = np.asarray(action_xy, dtype=np.float64).reshape(2)
    distances = {
        "x_min": float(p[0] + world_xy),
        "x_max": float(world_xy - p[0]),
        "y_min": float(p[1] + world_xy),
        "y_max": float(world_xy - p[1]),
    }
    name = min(distances, key=distances.get)
    normals = {
        "x_min": np.array([-1.0, 0.0], dtype=np.float64),
        "x_max": np.array([1.0, 0.0], dtype=np.float64),
        "y_min": np.array([0.0, -1.0], dtype=np.float64),
        "y_max": np.array([0.0, 1.0], dtype=np.float64),
    }
    normal = normals[name]
    velocity_out = float(np.dot(v, normal))
    action_out = float(np.dot(action, normal))
    v_out_pos = max(velocity_out, 0.0)
    braking_distance = (v_out_pos * v_out_pos) / max(2.0 * float(amax), 1e-9)
    margin = float(distances[name] - uav_radius)
    return {
        "nearest_boundary_name": name,
        "outward_normal": normal,
        "velocity_outward_projection": velocity_out,
        "action_outward_projection": action_out,
        "distance_to_boundary": float(distances[name]),
        "estimated_braking_distance": float(braking_distance),
        "braking_margin": float(margin - braking_distance),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config_snapshot(cfg: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--controllers", default="pure_pursuit,pd,apf,nominal_slot_tracker,existing")
    parser.add_argument("--scenario_group", default="all")
    parser.add_argument("--num_seeds", type=int, default=None)
    parser.add_argument("--output_dir", default="experiments/slot_tracking/outputs/default_run")
    parser.add_argument("--max_cases_per_group", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true", help="Disable terminal progress bar.")
    return parser.parse_args()


def _controller_base_name(name: str) -> str:
    key = str(name).strip()
    aliases = {
        "pd_raw": "pd",
        "pd_safety": "pd",
        "pure_pursuit_raw": "pure_pursuit",
        "pure_pursuit_safety": "pure_pursuit",
        "nominal_raw": "nominal_slot_tracker",
        "nominal_safety": "nominal_slot_tracker",
    }
    return aliases.get(key, key)


def _controller_uses_raw_slot(name: str) -> bool:
    return str(name).strip().endswith("_raw")


def _controller_uses_predictive_safety(name: str, cfg: dict[str, Any]) -> bool:
    key = str(name).strip()
    if key.endswith("_raw"):
        return False
    if key.endswith("_safety"):
        return True
    return bool(cfg.get("safety", {}).get("predictive_obstacle_filter", True))


def _scenario_groups_arg(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return ["all"]
    return [x.strip().upper() for x in value.split(",") if x.strip()]


def _maybe_limit_specs(specs: list[ScenarioSpec], limit: int | None) -> list[ScenarioSpec]:
    if limit is None or limit <= 0:
        return specs
    counts: Counter[str] = Counter()
    out: list[ScenarioSpec] = []
    for spec in specs:
        if counts[spec.group] < limit:
            out.append(spec)
            counts[spec.group] += 1
    return out


def _observed_obstacles(instance: ScenarioInstance, spec: ScenarioSpec, rng: np.random.Generator) -> list[Any]:
    obstacles = instance.obstacle_map.obstacles
    if not obstacles:
        return []
    if spec.obstacle_dropout_prob <= 0.0 and spec.obstacle_noise_std <= 0.0:
        return obstacles
    observed = []
    from marl_uav.framework.geometry.obstacle_geometry import Obstacle

    for obs in obstacles:
        if spec.obstacle_dropout_prob > 0.0 and rng.random() < spec.obstacle_dropout_prob:
            continue
        center = np.asarray(obs.center, dtype=np.float64).reshape(2)
        if spec.obstacle_noise_std > 0.0:
            center = center + rng.normal(0.0, spec.obstacle_noise_std, size=2)
        observed.append(Obstacle(kind=obs.kind, center=center, radius=float(obs.radius)))
    return observed


def _obstacle_clearance_any(position_xy: np.ndarray, obstacle: Any, *, uav_radius: float) -> float:
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    c = np.asarray(getattr(obstacle, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
    if getattr(obstacle, "kind", "circle") == "aabb" and getattr(obstacle, "half_extents", None) is not None:
        half = np.asarray(getattr(obstacle, "half_extents"), dtype=np.float64).reshape(2)
        outside = np.maximum(np.abs(p - c) - half, 0.0)
        return float(np.linalg.norm(outside) - float(uav_radius))
    return float(np.linalg.norm(p - c) - float(getattr(obstacle, "radius", 0.0)) - float(uav_radius))


def _nearest_obstacle_info(position_xy: np.ndarray, obstacles: list[Any], *, uav_radius: float) -> tuple[float, str]:
    if not obstacles:
        return float("inf"), ""
    best = (float("inf"), "")
    for idx, obs in enumerate(obstacles):
        clear = _obstacle_clearance_any(position_xy, obs, uav_radius=uav_radius)
        label = f"{getattr(obs, 'kind', 'obstacle')}:{idx}"
        if clear < best[0]:
            best = (float(clear), label)
    return best


def _nearest_obstacle_id(label: str) -> int:
    try:
        return int(str(label).rsplit(":", 1)[-1])
    except Exception:
        return -1


def _segment_min_clearance(
    p0: np.ndarray,
    p1: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    samples: int = 12,
) -> float:
    if not obstacles:
        return float("inf")
    a = np.asarray(p0, dtype=np.float64).reshape(2)
    b = np.asarray(p1, dtype=np.float64).reshape(2)
    best = float("inf")
    for t in np.linspace(0.0, 1.0, max(int(samples), 2)):
        p = a + float(t) * (b - a)
        clear, _ = _nearest_obstacle_info(p, obstacles, uav_radius=uav_radius)
        best = min(best, float(clear))
    return float(best)


def _path_min_clearance(path_value: Any, obstacles: list[Any], *, uav_radius: float) -> float:
    if not obstacles:
        return float("inf")
    try:
        pts = np.asarray(path_value, dtype=np.float64).reshape(-1, 2)
    except Exception:
        try:
            pts = np.asarray(ast.literal_eval(str(path_value)), dtype=np.float64).reshape(-1, 2)
        except Exception:
            return float("nan")
    if pts.shape[0] == 0:
        return float("nan")
    if pts.shape[0] == 1:
        return _nearest_obstacle_info(pts[0], obstacles, uav_radius=uav_radius)[0]
    return float(min(_segment_min_clearance(a, b, obstacles, uav_radius=uav_radius) for a, b in zip(pts[:-1], pts[1:])))


def _target_mode_invariant(
    *,
    target_mode: str,
    planner_mode_active: bool,
    raw_slot_valid: bool,
    proxy_slot_valid: bool,
    los_to_proxy: bool,
    path_valid: bool,
    selected_subgoal_valid: bool,
    selected_subgoal_visible: bool,
    stale_subgoal_used: bool,
) -> tuple[bool, str]:
    mode = str(target_mode)
    if not los_to_proxy and mode == "RAW_TRACKING":
        return True, "LOS_PROXY_BLOCKED_BUT_RAW_TRACKING"
    if planner_mode_active and mode == "PATH_SUBGOAL_TRACKING" and (not path_valid or not selected_subgoal_valid):
        return True, "PLANNER_ACTIVE_WITHOUT_VALID_PATH_SUBGOAL"
    if planner_mode_active and mode not in ("PATH_SUBGOAL_TRACKING", "RECOVERY", "SAFE_STOP"):
        return True, "PLANNER_ACTIVE_WITH_WRONG_TARGET_MODE"
    if not selected_subgoal_visible and mode == "PATH_SUBGOAL_TRACKING":
        return True, "INVISIBLE_SUBGOAL_TRACKED"
    if not path_valid and stale_subgoal_used:
        return True, "STALE_SUBGOAL_AFTER_INVALID_PATH"
    if not raw_slot_valid and mode == "RAW_TRACKING":
        return True, "INVALID_RAW_SLOT_TRACKED"
    if not proxy_slot_valid and mode == "PROXY_TRACKING":
        return True, "INVALID_PROXY_SLOT_TRACKED"
    return False, ""


def _proxy_slot_for_safety(
    raw_slot: np.ndarray,
    obstacles: list[Any],
    *,
    world_xy: float,
    uav_radius: float,
    safety_margin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_slot, dtype=np.float64).reshape(3)
    proxy = raw.copy()
    limit = max(float(world_xy) - float(uav_radius) - float(safety_margin), 0.0)
    before_boundary = proxy[:2].copy()
    proxy[0] = float(np.clip(proxy[0], -limit, limit))
    proxy[1] = float(np.clip(proxy[1], -limit, limit))
    boundary_adjusted = bool(np.linalg.norm(proxy[:2] - before_boundary) > 1e-9)

    inside = False
    too_close = False
    for _ in range(3):
        changed = False
        for obs in obstacles:
            center = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
            radius = float(getattr(obs, "radius", 0.0))
            clearance = _obstacle_clearance_any(proxy[:2], obs, uav_radius=uav_radius)
            if clearance < 0.0:
                inside = True
            if clearance < safety_margin:
                too_close = True
                rel = proxy[:2] - center
                dist = float(np.linalg.norm(rel))
                normal = np.array([1.0, 0.0], dtype=np.float64) if dist <= 1e-9 else rel / dist
                if getattr(obs, "kind", "circle") == "aabb" and getattr(obs, "half_extents", None) is not None:
                    radius = float(getattr(obs, "radius", radius))
                desired = radius + float(uav_radius) + float(safety_margin)
                proxy[:2] = center + desired * normal
                proxy[0] = float(np.clip(proxy[0], -limit, limit))
                proxy[1] = float(np.clip(proxy[1], -limit, limit))
                changed = True
        if not changed:
            break

    final_clear, nearest_name = _nearest_obstacle_info(proxy[:2], obstacles, uav_radius=uav_radius)
    raw_clear, _ = _nearest_obstacle_info(raw[:2], obstacles, uav_radius=uav_radius)
    raw_outside = bool(np.max(np.abs(raw[:2])) > float(world_xy))
    return proxy, {
        "proxy_slot_adjusted": bool(boundary_adjusted or np.linalg.norm(proxy[:2] - raw[:2]) > 1e-9),
        "raw_slot_inside_obstacle": bool(raw_clear < 0.0),
        "raw_slot_too_close_obstacle": bool(raw_clear < safety_margin),
        "raw_slot_outside_boundary": raw_outside,
        "proxy_slot_min_obstacle_clearance": float(final_clear),
        "proxy_slot_nearest_obstacle_name": nearest_name,
    }


def _integrate_point_mass(
    pos: np.ndarray,
    vel: np.ndarray,
    cmd_xy: np.ndarray,
    *,
    dt: float,
    vmax: float,
    amax: float,
    wind_xy: np.ndarray,
    world_xy: float | None = None,
    uav_radius: float = 0.0,
    boundary_hard_margin: float = 0.0,
    post_boundary_check: bool = False,
    obstacles: list[Any] | None = None,
    safety_margin: float = 0.0,
    predictive_obstacle_check: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cmd = np.asarray(cmd_xy, dtype=np.float64).reshape(2)
    desired_speed = clip_norm(cmd, vmax)
    dv = desired_speed - vel[:2]
    dv = clip_norm(dv, amax * dt)
    new_vel = vel.copy()
    new_vel[:2] = clip_norm(vel[:2] + dv, vmax)
    post_projected = False
    predictive_projected = False
    predictive_subtype = ""
    if post_boundary_check and world_xy is not None and np.isfinite(float(world_xy)):
        checked_vel, post_projected = _project_final_velocity_for_next_step(
            pos[:2],
            new_vel[:2],
            dt=dt,
            world_xy=float(world_xy),
            uav_radius=float(uav_radius),
            boundary_hard_margin=float(boundary_hard_margin),
        )
        new_vel[:2] = checked_vel
    if predictive_obstacle_check and obstacles:
        checked_vel, predictive_projected, predictive_subtype = _project_final_velocity_for_obstacles(
            pos[:2],
            new_vel[:2],
            list(obstacles),
            dt=dt,
            uav_radius=float(uav_radius),
            safety_margin=float(safety_margin),
            vmax=float(vmax),
        )
        new_vel[:2] = checked_vel
    new_pos = pos.copy()
    new_pos[:2] = new_pos[:2] + (new_vel[:2] + np.asarray(wind_xy, dtype=np.float64).reshape(2)) * dt
    predicted_next = np.asarray(pos[:2], dtype=np.float64).reshape(2) + new_vel[:2] * dt
    segment_unsafe = bool(
        obstacles
        and _future_segment_min_clearance(
            np.asarray(pos[:2], dtype=np.float64).reshape(2),
            predicted_next,
            list(obstacles),
            uav_radius=float(uav_radius) + max(float(safety_margin), 0.0),
        )
        < 0.0
    )
    diag = {
        "dynamics_speed_clip": bool(np.linalg.norm(cmd) > np.linalg.norm(desired_speed) + 1e-9),
        "dynamics_acceleration_clip": bool(np.linalg.norm(desired_speed - vel[:2]) > np.linalg.norm(dv) + 1e-9),
        "post_boundary_projection_applied": bool(post_projected),
        "predictive_filter_active": bool(predictive_projected),
        "predictive_filter_subtype": str(predictive_subtype),
        "dynamics_command_norm_before": float(np.linalg.norm(cmd)),
        "dynamics_command_norm_after_speed_clip": float(np.linalg.norm(desired_speed)),
        "dynamics_velocity_norm_after_accel": float(np.linalg.norm(new_vel[:2])),
        "v_after_speed_clip": desired_speed.astype(float).tolist(),
        "v_after_accel_clip": new_vel[:2].astype(float).tolist(),
        "v_after_predictive_safety": new_vel[:2].astype(float).tolist(),
        "v_final": new_vel[:2].astype(float).tolist(),
        "action_final": new_vel[:2].astype(float).tolist(),
        "predicted_next_pos": predicted_next.astype(float).tolist(),
        "segment_collision_current_to_next": bool(segment_unsafe),
        "min_predicted_next_clearance": _segment_min_clearance(
            np.asarray(pos[:2], dtype=np.float64).reshape(2),
            predicted_next,
            list(obstacles or []),
            uav_radius=float(uav_radius),
        ),
    }
    return new_pos, new_vel, new_vel[:2].copy(), diag


def _project_final_velocity_for_next_step(
    pos_xy: np.ndarray,
    velocity_xy: np.ndarray,
    *,
    dt: float,
    world_xy: float,
    uav_radius: float,
    boundary_hard_margin: float,
) -> tuple[np.ndarray, bool]:
    p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    v = np.asarray(velocity_xy, dtype=np.float64).reshape(2)
    dt_eff = max(float(dt), 1e-9)
    limit = max(float(world_xy) - float(uav_radius) - max(float(boundary_hard_margin), 0.0), 0.0)
    constraints: list[tuple[np.ndarray, float]] = [
        (np.array([1.0, 0.0], dtype=np.float64), (limit - p[0]) / dt_eff),
        (np.array([-1.0, 0.0], dtype=np.float64), (limit + p[0]) / dt_eff),
        (np.array([0.0, 1.0], dtype=np.float64), (limit - p[1]) / dt_eff),
        (np.array([0.0, -1.0], dtype=np.float64), (limit + p[1]) / dt_eff),
    ]
    if all(float(np.dot(n, v)) <= float(b) + 1e-9 for n, b in constraints):
        return v.copy(), False

    candidates: list[tuple[float, np.ndarray]] = []

    def feasible(x: np.ndarray) -> bool:
        return all(float(np.dot(n, x)) <= float(b) + 1e-9 for n, b in constraints)

    for n, b in constraints:
        violation = float(np.dot(n, v) - b)
        x = v - max(violation, 0.0) * n
        if feasible(x):
            candidates.append((float(np.linalg.norm(x - v)), x))
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            n_i, b_i = constraints[i]
            n_j, b_j = constraints[j]
            a = np.stack([n_i, n_j], axis=0)
            if abs(float(np.linalg.det(a))) <= 1e-9:
                continue
            x = np.linalg.solve(a, np.array([b_i, b_j], dtype=np.float64))
            if feasible(x):
                candidates.append((float(np.linalg.norm(x - v)), x))
    if not candidates:
        x = v.copy()
        for n, b in constraints:
            violation = float(np.dot(n, x) - b)
            if violation > 0.0:
                x = x - violation * n
        return x, True
    return min(candidates, key=lambda item: item[0])[1], True


def _project_final_velocity_for_obstacles(
    pos_xy: np.ndarray,
    velocity_xy: np.ndarray,
    obstacles: list[Any],
    *,
    dt: float,
    uav_radius: float,
    safety_margin: float,
    vmax: float,
) -> tuple[np.ndarray, bool, str]:
    p = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    original = np.asarray(velocity_xy, dtype=np.float64).reshape(2)
    v = original.copy()
    dt_eff = max(float(dt), 1e-9)
    inflated_clearance = _future_segment_min_clearance(
        p,
        p + v * dt_eff,
        obstacles,
        uav_radius=float(uav_radius) + max(float(safety_margin), 0.0),
    )
    if not np.isfinite(inflated_clearance) or inflated_clearance >= 0.0:
        return v, False, ""

    active: list[tuple[float, np.ndarray, float]] = []
    for obs in obstacles:
        c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
        rel = p - c
        dist = float(np.linalg.norm(rel))
        n = rel / dist if dist > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
        required = float(getattr(obs, "radius", 0.0)) + float(uav_radius) + max(float(safety_margin), 0.0)
        bound = (required - float(np.dot(n, rel))) / dt_eff
        if float(np.dot(n, v)) < bound:
            active.append((bound - float(np.dot(n, v)), n, bound))

    changed = False
    for _violation, n, bound in sorted(active, key=lambda item: item[0], reverse=True):
        normal_component = float(np.dot(n, v))
        if normal_component < bound:
            v = v + (bound - normal_component) * n
            changed = True
    v = clip_norm(v, vmax)

    if _future_segment_min_clearance(
        p,
        p + v * dt_eff,
        obstacles,
        uav_radius=float(uav_radius) + max(float(safety_margin), 0.0),
    ) >= 0.0:
        return v, True, "PREDICTIVE_SAFETY_PROJECT"

    best = np.zeros(2, dtype=np.float64)
    best_progress = -float("inf")
    for obs in obstacles:
        c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
        rel = p - c
        dist = float(np.linalg.norm(rel))
        n = rel / dist if dist > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
        tangent = np.array([-n[1], n[0]], dtype=np.float64)
        for sign in (1.0, -1.0):
            cand = sign * tangent * min(float(vmax), float(np.linalg.norm(original)))
            for scale in (1.0, 0.5, 0.25):
                trial = cand * scale
                if _future_segment_min_clearance(
                    p,
                    p + trial * dt_eff,
                    obstacles,
                    uav_radius=float(uav_radius) + max(float(safety_margin), 0.0),
                ) >= 0.0:
                    progress = float(np.dot(trial, original))
                    if progress > best_progress:
                        best_progress = progress
                        best = trial
    if best_progress > -float("inf"):
        return best, True, "PREDICTIVE_SAFETY_TANGENT"
    return np.zeros(2, dtype=np.float64), True, "PREDICTIVE_SAFETY_STOP"


def _future_segment_min_clearance(
    p0: np.ndarray,
    p1: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    samples: int = 12,
) -> float:
    if not obstacles:
        return float("inf")
    a = np.asarray(p0, dtype=np.float64).reshape(2)
    b = np.asarray(p1, dtype=np.float64).reshape(2)
    best = float("inf")
    for t in np.linspace(0.0, 1.0, max(int(samples), 2))[1:]:
        p = a + float(t) * (b - a)
        clear, _ = _nearest_obstacle_info(p, obstacles, uav_radius=uav_radius)
        best = min(best, float(clear))
    return float(best)


def run_episode(
    *,
    controller_name: str,
    controller_cfg: dict[str, Any],
    instance: ScenarioInstance,
    cfg: dict[str, Any],
    seed: int,
    raw_path: Path | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + 17)
    base_controller_name = _controller_base_name(controller_name)
    use_raw_slot_target = _controller_uses_raw_slot(controller_name)
    use_predictive_safety = _controller_uses_predictive_safety(controller_name, cfg)
    controller = make_controller(base_controller_name, controller_cfg)
    controller.reset()
    dyn = cfg["dynamics"]
    world = cfg["world"]
    dt = float(dyn["dt"])
    steps = int(dyn["episode_steps"])
    world_xy = float(world["world_xy"])
    base_vmax = float(dyn["uav_vmax"]) * float(instance.spec.actual_vmax_scale)
    amax = float(dyn["uav_amax"])
    uav_radius = float(dyn["uav_radius"])
    safety_margin = float(dyn["safety_margin"])
    proxy_slot_safety_margin = safety_margin + max(float(cfg.get("safety", {}).get("proxy_slot_margin_extra", 0.0)), 0.0)
    transition_cfg = dict(cfg.get("slot_transition") or {})
    use_slot_transition = bool(transition_cfg.get("enabled", instance.spec.group == "D"))
    controller_clip_enabled = bool(cfg.get("dynamics", {}).get("controller_action_clip_enabled", True))
    n = int(instance.spec.num_agents)
    pos = instance.init_positions.copy()
    vel = instance.init_velocities.copy()
    initial_invalid, initial_margin, initial_outward_velocity = _initial_boundary_state(
        pos,
        vel,
        world_xy=world_xy,
        uav_radius=uav_radius,
        safety_margin=safety_margin,
        amax=amax,
    )
    initial_proxy = np.zeros_like(instance.slot_positions[0])
    transition_managers: list[SlotTransitionManager] = []
    initial_commanded = np.zeros_like(instance.slot_positions[0])
    for j in range(n):
        initial_proxy[j], _ = _proxy_slot_for_safety(
            instance.slot_positions[0, j],
            instance.obstacle_map.obstacles,
            world_xy=world_xy,
            uav_radius=uav_radius,
            safety_margin=proxy_slot_safety_margin,
        )
        manager = SlotTransitionManager(
            world_xy=world_xy,
            uav_radius=uav_radius,
            safety_margin=proxy_slot_safety_margin,
            dt=dt,
            slot_ref_vmax=float(transition_cfg.get("slot_ref_vmax", dyn.get("slot_ref_vmax", dyn["uav_vmax"]))),
            slot_ref_amax=float(transition_cfg.get("slot_ref_amax", dyn.get("slot_ref_amax", dyn["uav_amax"]))),
            jump_detection_threshold=float(transition_cfg.get("jump_detection_threshold", dyn.get("jump_detection_threshold", 0.75))),
            frequent_jump_min_interval_steps=int(transition_cfg.get("frequent_jump_min_interval_steps", 20)),
            high_freq_factor=float(transition_cfg.get("high_freq_factor", 1.25)),
        )
        reset_diag = manager.reset(instance.slot_positions[0, j], instance.obstacle_map.obstacles)
        initial_commanded[j] = np.asarray(reset_diag["commanded_slot_pos"], dtype=np.float64).reshape(3)
        transition_managers.append(manager)
    reference_initial = initial_commanded if use_slot_transition else initial_proxy
    prev_errors = np.linalg.norm(pos[:, :2] - reference_initial[:, :2], axis=1)
    prev_proxy_slot_xy = initial_proxy[:, :2].copy()
    prev_commanded_slot = initial_commanded.copy()
    prev_slot_transition_modes = ["INIT" for _ in range(n)]
    prev_subgoal_xy = reference_initial[:, :2].copy()
    delay = max(int(instance.spec.action_delay_steps), 0)
    action_queues = [deque([np.zeros(2, dtype=np.float64)] * (delay + 1), maxlen=delay + 1) for _ in range(n)]
    rows: list[dict[str, Any]] = []

    for k in range(steps):
        raw_slot = instance.slot_positions[k]
        raw_slot_vel = instance.slot_velocities[k]
        proxy_slot = raw_slot.copy()
        commanded_slot = raw_slot.copy()
        commanded_slot_vel = raw_slot_vel.copy()
        proxy_diags: list[dict[str, Any]] = []
        transition_diags: list[dict[str, Any]] = []
        for i in range(n):
            proxy, proxy_diag = _proxy_slot_for_safety(
                raw_slot[i],
                instance.obstacle_map.obstacles,
                world_xy=world_xy,
                uav_radius=uav_radius,
                safety_margin=proxy_slot_safety_margin,
            )
            proxy_slot[i] = proxy
            if use_slot_transition:
                tdiag = transition_managers[i].update(
                    raw_slot_pos=raw_slot[i],
                    previous_commanded_slot_pos=prev_commanded_slot[i],
                    uav_pos=pos[i],
                    obstacles=instance.obstacle_map.obstacles,
                    step=k,
                )
                commanded_slot[i] = np.asarray(tdiag["commanded_slot_pos"], dtype=np.float64).reshape(3)
                commanded_slot_vel[i] = np.asarray(tdiag["commanded_slot_vel"], dtype=np.float64).reshape(3)
                proxy_slot[i] = np.asarray(tdiag["proxy_slot_pos"], dtype=np.float64).reshape(3)
                proxy_diag = {**proxy_diag, **tdiag}
            else:
                commanded_slot[i] = (raw_slot if use_raw_slot_target else proxy_slot)[i]
                commanded_slot_vel[i] = raw_slot_vel[i]
                proxy_diag = {
                    **proxy_diag,
                    "commanded_slot_pos": commanded_slot[i].copy(),
                    "commanded_slot_vel": commanded_slot_vel[i].copy(),
                    "slot_transition_mode": "RAW_TRACKING" if use_raw_slot_target else "PROXY_TRACKING",
                    "jump_detected": False,
                    "jump_distance": 0.0,
                    "minimum_reach_time": 0.0,
                    "jump_interval_steps": 0,
                    "raw_slot_too_unstable": False,
                    "raw_slot_valid": not bool(proxy_diag.get("raw_slot_outside_boundary", False))
                    and not bool(proxy_diag.get("raw_slot_inside_obstacle", False))
                    and not bool(proxy_diag.get("raw_slot_too_close_obstacle", False)),
                    "commanded_slot_valid": True,
                    "commanded_slot_inside_obstacle": False,
                    "commanded_slot_outside_boundary": False,
                    "commanded_slot_too_close_obstacle": False,
                    "commanded_transition_segment_safe": True,
                    "commanded_slot_invalid_reason": "",
                    "safe_hold_active": False,
                    "safe_hold_reason": "",
                    "proxy_slot_valid": True,
                    "proxy_slot_used": bool(proxy_diag.get("proxy_slot_adjusted", False)),
                    "transition_progress": 1.0,
                    "commanded_slot_lag_to_raw": float(np.linalg.norm(commanded_slot[i, :2] - raw_slot[i, :2])),
                    "commanded_slot_step_norm": float(np.linalg.norm(commanded_slot[i, :2] - prev_commanded_slot[i, :2])),
                }
            proxy_diags.append(proxy_diag)
            transition_diags.append(proxy_diag)
        commands = np.zeros((n, 2), dtype=np.float64)
        decision_ms = np.zeros(n, dtype=np.float64)
        controller_diags: list[dict[str, Any]] = [{} for _ in range(n)]
        pre_pos = pos.copy()
        pre_vel = vel.copy()
        for i in range(n):
            obs_noise = rng.normal(0.0, instance.spec.noise_std, size=2) if instance.spec.noise_std > 0.0 else np.zeros(2)
            observed_obstacles = _observed_obstacles(instance, instance.spec, rng)
            observation = ControllerObservation(
                position=np.array([pos[i, 0] + obs_noise[0], pos[i, 1] + obs_noise[1], pos[i, 2]], dtype=np.float64),
                velocity=vel[i].copy(),
                slot_position=np.array([
                    commanded_slot[i, 0] + obs_noise[0],
                    commanded_slot[i, 1] + obs_noise[1],
                    commanded_slot[i, 2],
                ], dtype=np.float64),
                slot_velocity=commanded_slot_vel[i].copy(),
                obstacles=observed_obstacles,
                world_xy=world_xy,
                dt=dt,
                uav_vmax=float(dyn.get("nominal_controller_vmax", dyn["uav_vmax"])),
                uav_amax=amax,
                uav_radius=uav_radius,
                safety_margin=safety_margin,
                peer_positions=np.delete(pos, i, axis=0) if n > 1 else None,
                controller_clip_enabled=controller_clip_enabled,
            )
            t0 = time.perf_counter()
            cmd, diag = controller.compute_action(observation)
            decision_ms[i] = (time.perf_counter() - t0) * 1000.0
            controller_diags[i] = dict(diag)
            action_queues[i].append(np.asarray(cmd, dtype=np.float64).reshape(2))
            commands[i] = action_queues[i][0]

        next_pos = pos.copy()
        next_vel = vel.copy()
        desired_actions = np.zeros((n, 2), dtype=np.float64)
        dynamics_diags: list[dict[str, Any]] = [{} for _ in range(n)]
        for i in range(n):
            wind = rng.normal(0.0, instance.spec.wind_std, size=2) if instance.spec.wind_std > 0.0 else np.zeros(2)
            next_pos[i], next_vel[i], desired_actions[i], dynamics_diags[i] = _integrate_point_mass(
                pos[i],
                vel[i],
                commands[i],
                dt=dt,
                vmax=base_vmax,
                amax=amax,
                wind_xy=wind,
                world_xy=world_xy,
                uav_radius=uav_radius,
                boundary_hard_margin=float(controller_cfg.get("boundary_hard_margin", safety_margin)),
                post_boundary_check=bool(controller_name == "existing"),
                obstacles=instance.obstacle_map.obstacles,
                safety_margin=safety_margin,
                predictive_obstacle_check=bool(use_predictive_safety),
            )

        coll_agent = inter_agent_collision(next_pos[:, :2], min_distance=float(dyn["inter_agent_min_distance"])) if n > 1 else False
        for i in range(n):
            cdiag = controller_diags[i]
            pdiag = proxy_diags[i]
            ddiag = dynamics_diags[i]
            raw_err = float(np.linalg.norm(next_pos[i, :2] - raw_slot[i, :2]))
            proxy_err = float(np.linalg.norm(next_pos[i, :2] - proxy_slot[i, :2]))
            err = float(np.linalg.norm(next_pos[i, :2] - commanded_slot[i, :2]))
            current_subgoal = _diag_vec(cdiag, "current_subgoal")
            if not np.all(np.isfinite(current_subgoal)):
                current_subgoal = commanded_slot[i, :2].copy()
            subgoal_err = float(np.linalg.norm(next_pos[i, :2] - current_subgoal))
            clear = obstacle_clearance(next_pos[i, :2], instance.obstacle_map.obstacles, uav_radius=uav_radius)
            nearest_clear, nearest_name = _nearest_obstacle_info(next_pos[i, :2], instance.obstacle_map.obstacles, uav_radius=uav_radius)
            bmargin = boundary_margin(next_pos[i, :2], world_xy=world_xy, uav_radius=uav_radius)
            v_after = desired_actions[i]
            progress = _goal_progress_fields(pre_pos[i, :2], commanded_slot[i, :2], v_after)
            proxy_progress = _goal_progress_fields(pre_pos[i, :2], proxy_slot[i, :2], v_after)
            raw_progress = _goal_progress_fields(pre_pos[i, :2], raw_slot[i, :2], v_after)
            controller_clip = bool(cdiag.get("clip_flag", False))
            speed_clip = bool(cdiag.get("speed_saturation_flag", False) or ddiag.get("dynamics_speed_clip", False))
            accel_clip = bool(cdiag.get("acceleration_saturation_flag", False) or ddiag.get("dynamics_acceleration_clip", False))
            before_norm = float(cdiag.get("raw_cmd_norm", ddiag.get("dynamics_command_norm_before", np.linalg.norm(commands[i]))))
            after_norm = float(np.linalg.norm(v_after))
            multi_clip = bool(controller_clip and (ddiag.get("dynamics_speed_clip", False) or ddiag.get("dynamics_acceleration_clip", False)))
            double_clip_warning = bool(multi_clip and before_norm > 1e-9 and after_norm < 0.5 * before_norm)
            v_goal = _diag_vec(cdiag, "v_goal")
            v_obstacle = _diag_vec(cdiag, "v_obstacle")
            v_boundary = _diag_vec(cdiag, "v_boundary")
            v_path = _diag_vec(cdiag, "v_path")
            v_smooth = _diag_vec(cdiag, "v_smooth")
            v_before = _diag_vec(cdiag, "v_final_before_clip")
            bdiag_base = _nearest_boundary_diag(
                next_pos[i, :2],
                next_vel[i, :2],
                desired_actions[i],
                world_xy=world_xy,
                uav_radius=uav_radius,
                amax=amax,
            )
            outward_normal = np.asarray(cdiag.get("outward_normal", bdiag_base["outward_normal"]), dtype=np.float64).reshape(2)
            action_before_boundary = _diag_vec(cdiag, "action_before_boundary_filter")
            action_after_boundary = _diag_vec(cdiag, "action_after_boundary_filter")
            if not np.all(np.isfinite(action_before_boundary)):
                action_before_boundary = desired_actions[i].copy()
            if not np.all(np.isfinite(action_after_boundary)):
                action_after_boundary = desired_actions[i].copy()
            v_nom = _diag_vec(cdiag, "v_nom")
            if not np.all(np.isfinite(v_nom)):
                v_nom = v_before.copy() if np.all(np.isfinite(v_before)) else action_before_boundary.copy()
            v_before_boundary = _diag_vec(cdiag, "v_before_boundary_filter")
            if not np.all(np.isfinite(v_before_boundary)):
                v_before_boundary = action_before_boundary.copy()
            v_after_projection = _diag_vec(cdiag, "v_after_boundary_projection")
            if not np.all(np.isfinite(v_after_projection)):
                v_after_projection = action_after_boundary.copy()
            v_after_obstacle = _diag_vec(cdiag, "v_after_obstacle_filter")
            if not np.all(np.isfinite(v_after_obstacle)):
                v_after_obstacle = action_before_boundary.copy()
            v_after_planner = _diag_vec(cdiag, "v_after_planner")
            if not np.all(np.isfinite(v_after_planner)):
                v_after_planner = action_before_boundary.copy()
            v_planner_subgoal = _diag_vec(cdiag, "v_planner_subgoal")
            if not np.all(np.isfinite(v_planner_subgoal)):
                v_planner_subgoal = v_after_planner.copy()
            v_after_braking = _diag_vec(cdiag, "v_after_braking_correction")
            if not np.all(np.isfinite(v_after_braking)):
                v_after_braking = v_after_projection.copy()
            v_after_speed = _diag_vec(ddiag, "v_after_speed_clip")
            v_after_accel = _diag_vec(ddiag, "v_after_accel_clip")
            v_after_predictive = _diag_vec(ddiag, "v_after_predictive_safety")
            v_final = _diag_vec(ddiag, "v_final")
            action_final = _diag_vec(ddiag, "action_final")
            predicted_next = _diag_vec(ddiag, "predicted_next_pos")
            stages = {
                "v_nom": v_nom,
                "v_before_boundary_filter": v_before_boundary,
                "v_after_boundary_projection": v_after_projection,
                "v_after_obstacle_filter": v_after_obstacle,
                "v_after_planner": v_after_planner,
                "v_planner_subgoal": v_planner_subgoal,
                "v_after_braking_correction": v_after_braking,
                "v_after_speed_clip": v_after_speed,
                "v_after_accel_clip": v_after_accel,
                "v_after_predictive_safety": v_after_predictive,
                "v_final": v_final,
            }
            planned_path_value = cdiag.get("planned_path", "")
            current_segment_value = cdiag.get("current_path_segment", "")
            min_path = float(cdiag.get("min_distance_to_planned_path", _path_min_clearance(planned_path_value, instance.obstacle_map.obstacles, uav_radius=uav_radius)))
            min_current_segment = float(cdiag.get("min_distance_to_current_segment", _path_min_clearance(current_segment_value, instance.obstacle_map.obstacles, uav_radius=uav_radius)))
            min_subgoal_segment = float(
                cdiag.get(
                    "min_distance_to_subgoal_segment",
                    _segment_min_clearance(pre_pos[i, :2], current_subgoal, instance.obstacle_map.obstacles, uav_radius=uav_radius),
                )
            )
            los_proxy = bool(
                cdiag.get(
                    "line_of_sight_to_proxy",
                    has_line_of_sight(pre_pos[i, :2], proxy_slot[i, :2], instance.obstacle_map.obstacles, safety_margin=safety_margin, uav_radius=uav_radius)
                    if instance.obstacle_map.obstacles
                    else True,
                )
            )
            los_raw = bool(
                has_line_of_sight(pre_pos[i, :2], raw_slot[i, :2], instance.obstacle_map.obstacles, safety_margin=safety_margin, uav_radius=uav_radius)
                if instance.obstacle_map.obstacles
                else True
            )
            proxy_shift_norm = float(np.linalg.norm(proxy_slot[i, :2] - prev_proxy_slot_xy[i]))
            commanded_shift_norm = float(pdiag.get("commanded_slot_step_norm", np.linalg.norm(commanded_slot[i, :2] - prev_commanded_slot[i, :2])))
            subgoal_shift_norm = float(cdiag.get("subgoal_shift_norm", np.linalg.norm(current_subgoal - prev_subgoal_xy[i])))
            los_subgoal = bool(
                cdiag.get(
                    "line_of_sight_to_subgoal",
                    has_line_of_sight(pre_pos[i, :2], current_subgoal, instance.obstacle_map.obstacles, safety_margin=safety_margin, uav_radius=uav_radius)
                    if instance.obstacle_map.obstacles
                    else True,
                )
            )
            subgoal_progress = _goal_progress_fields(pre_pos[i, :2], current_subgoal, v_after)
            raw_clear_at_slot, _ = _nearest_obstacle_info(raw_slot[i, :2], instance.obstacle_map.obstacles, uav_radius=uav_radius)
            proxy_clear_at_slot, _ = _nearest_obstacle_info(proxy_slot[i, :2], instance.obstacle_map.obstacles, uav_radius=uav_radius)
            commanded_clear_at_slot, _ = _nearest_obstacle_info(commanded_slot[i, :2], instance.obstacle_map.obstacles, uav_radius=uav_radius)
            valid_eps = 1e-5
            raw_slot_valid = bool(pdiag.get("raw_slot_valid", np.max(np.abs(raw_slot[i, :2])) <= world_xy and raw_clear_at_slot + valid_eps >= safety_margin))
            proxy_slot_valid = bool(pdiag.get("proxy_slot_valid", np.max(np.abs(proxy_slot[i, :2])) <= world_xy and proxy_clear_at_slot + valid_eps >= safety_margin))
            commanded_slot_valid = bool(pdiag.get("commanded_slot_valid", np.max(np.abs(commanded_slot[i, :2])) <= world_xy and commanded_clear_at_slot + valid_eps >= safety_margin))
            target_mode = str(cdiag.get("target_mode", "COMMANDED_SLOT_TRACKING" if use_slot_transition else ("RAW_TRACKING" if use_raw_slot_target else "PROXY_TRACKING")))
            target_prev = str(cdiag.get("target_mode_previous", target_mode))
            target_reason = str(cdiag.get("target_mode_transition_reason", ""))
            selected_subgoal_valid = bool(np.all(np.isfinite(current_subgoal)))
            stale_subgoal_used = bool(cdiag.get("stale_subgoal_used", False))
            invariant_violation, invariant_name = _target_mode_invariant(
                target_mode=target_mode,
                planner_mode_active=bool(cdiag.get("planner_mode_active", False)),
                raw_slot_valid=raw_slot_valid,
                proxy_slot_valid=proxy_slot_valid,
                los_to_proxy=bool(cdiag.get("line_of_sight_to_proxy_slot", los_proxy)),
                path_valid=bool(cdiag.get("path_valid", False)),
                selected_subgoal_valid=selected_subgoal_valid,
                selected_subgoal_visible=bool(cdiag.get("selected_subgoal_visible", los_subgoal)),
                stale_subgoal_used=stale_subgoal_used,
            )
            row = {
                "t": float((k + 1) * dt),
                "agent_id": int(i),
                "controller_base": str(base_controller_name),
                "target_source": "commanded_slot" if use_slot_transition else ("raw_slot" if use_raw_slot_target else "proxy_slot"),
                "shared_predictive_safety": bool(use_predictive_safety),
                "uav_x": float(next_pos[i, 0]),
                "uav_y": float(next_pos[i, 1]),
                "uav_z": float(next_pos[i, 2]),
                "slot_x": float(commanded_slot[i, 0]),
                "slot_y": float(commanded_slot[i, 1]),
                "slot_z": float(commanded_slot[i, 2]),
                "commanded_slot_x": float(commanded_slot[i, 0]),
                "commanded_slot_y": float(commanded_slot[i, 1]),
                "commanded_slot_z": float(commanded_slot[i, 2]),
                "raw_slot_x": float(raw_slot[i, 0]),
                "raw_slot_y": float(raw_slot[i, 1]),
                "raw_slot_z": float(raw_slot[i, 2]),
                "proxy_slot_x": float(proxy_slot[i, 0]),
                "proxy_slot_y": float(proxy_slot[i, 1]),
                "proxy_slot_z": float(proxy_slot[i, 2]),
                "uav_vx": float(next_vel[i, 0]),
                "uav_vy": float(next_vel[i, 1]),
                "uav_vz": float(next_vel[i, 2]),
                "action_x": float(desired_actions[i, 0]),
                "action_y": float(desired_actions[i, 1]),
                "action_z": 0.0,
                "tracking_error": err,
                "tracking_error_to_raw_slot": raw_err,
                "tracking_error_to_proxy_slot": proxy_err,
                "tracking_error_to_subgoal": subgoal_err,
                "nearest_obstacle_distance": float(clear),
                "nearest_obstacle_id": int(_nearest_obstacle_id(nearest_name)),
                "nearest_obstacle_name": str(nearest_name),
                "raw_slot_inside_obstacle": bool(pdiag.get("raw_slot_inside_obstacle", False)),
                "raw_slot_too_close_obstacle": bool(pdiag.get("raw_slot_too_close_obstacle", False)),
                "proxy_slot_adjusted": bool(pdiag.get("proxy_slot_adjusted", False)),
                "boundary_margin": float(bmargin),
                "decision_time_ms": float(decision_ms[i]),
                "nearest_boundary_name": str(cdiag.get("nearest_boundary_name", bdiag_base["nearest_boundary_name"])),
                **_vec_columns("outward_normal", outward_normal),
                "velocity_outward_projection": float(cdiag.get("velocity_outward_projection", bdiag_base["velocity_outward_projection"])),
                "action_outward_projection": float(cdiag.get("action_outward_projection", bdiag_base["action_outward_projection"])),
                "distance_to_boundary": float(cdiag.get("distance_to_boundary", bdiag_base["distance_to_boundary"])),
                "estimated_braking_distance": float(cdiag.get("estimated_braking_distance", bdiag_base["estimated_braking_distance"])),
                "braking_margin": float(cdiag.get("braking_margin", bdiag_base["braking_margin"])),
                "boundary_filter_active": bool(cdiag.get("boundary_filter_active", False)),
                "boundary_active_names": str(cdiag.get("boundary_active_names", "")),
                "boundary_projected_action_failed": bool(cdiag.get("boundary_projected_action_failed", False)),
                "action_outward_projection_before_filter": float(
                    cdiag.get("action_outward_projection_before_filter", bdiag_base["action_outward_projection"])
                ),
                **_vec_columns("action_before_boundary_filter", action_before_boundary),
                **_vec_columns("action_after_boundary_filter", action_after_boundary),
                **_vec_columns("v_nom", v_nom),
                **_vec_columns("v_before_boundary_filter", v_before_boundary),
                **_vec_columns("v_after_boundary_projection", v_after_projection),
                **_vec_columns("v_after_obstacle_filter", v_after_obstacle),
                **_vec_columns("v_after_predictive_safety", v_after_predictive),
                **_vec_columns("v_after_planner", v_after_planner),
                **_vec_columns("v_planner_subgoal", v_planner_subgoal),
                **_vec_columns("v_after_braking_correction", v_after_braking),
                **_vec_columns("v_after_speed_clip", v_after_speed),
                **_vec_columns("v_after_accel_clip", v_after_accel),
                **_vec_columns("v_final", v_final),
                **_vec_columns("action_final", action_final),
                **_stage_norm_columns(stages),
                "angle_nom_to_boundary_projection": _angle_between(v_nom, v_after_projection),
                "angle_nom_to_final": _angle_between(v_nom, v_final),
                "norm_ratio_final_to_nom": float(np.linalg.norm(v_final) / max(np.linalg.norm(v_nom), 1e-9)),
                "speed_clip_applied": bool(ddiag.get("dynamics_speed_clip", False)),
                "accel_clip_applied": bool(ddiag.get("dynamics_acceleration_clip", False)),
                "post_boundary_projection_applied": bool(ddiag.get("post_boundary_projection_applied", False)),
                **_boundary_detail_columns(cdiag),
                **_vec_columns("v_goal", v_goal),
                **_vec_columns("v_obstacle", v_obstacle),
                "obstacle_repulsion_norm": float(np.linalg.norm(v_obstacle)) if np.all(np.isfinite(v_obstacle)) else float("nan"),
                **_vec_columns("v_boundary", v_boundary),
                **_vec_columns("v_path", v_path),
                **_vec_columns("v_smooth", v_smooth),
                **_vec_columns("v_final_before_clip", v_before),
                **_vec_columns("v_final_after_clip", v_after),
                **_vec_columns("final_action", next_vel[i, :2]),
                **_vec_columns("desired_velocity_before_final_clip", commands[i]),
                **_vec_columns("final_velocity_after_clip", next_vel[i, :2]),
                "clip_flag": bool(controller_clip or speed_clip or accel_clip),
                "speed_saturation_flag": bool(speed_clip),
                "acceleration_saturation_flag": bool(accel_clip),
                "double_clip_warning": bool(double_clip_warning),
                "double_clip_flag": bool(double_clip_warning),
                "clip_stage_names": ",".join(
                    name
                    for name, active in [
                        ("controller", controller_clip),
                        ("dynamics_speed", bool(ddiag.get("dynamics_speed_clip", False))),
                        ("dynamics_accel", bool(ddiag.get("dynamics_acceleration_clip", False))),
                        ("boundary", bool(ddiag.get("post_boundary_projection_applied", False))),
                        ("predictive", bool(ddiag.get("predictive_filter_active", False))),
                    ]
                    if active
                ),
                "distance_delta": float(prev_errors[i] - err),
                **progress,
                "existing_bypass_reason": str(cdiag.get("existing_bypass_reason", "")),
                "used_existing_safety_modules": bool(cdiag.get("used_existing_safety_modules", False)),
                "obstacle_risk": bool(cdiag.get("obstacle_risk", False)),
                "boundary_risk": bool(cdiag.get("boundary_risk", False)),
                "inter_agent_risk": bool(cdiag.get("inter_agent_risk", False)),
                "line_of_sight_blocked": bool(
                    cdiag.get(
                        "line_of_sight_blocked",
                        not has_line_of_sight(
                            pre_pos[i, :2],
                            proxy_slot[i, :2],
                            instance.obstacle_map.obstacles,
                            safety_margin=safety_margin,
                            uav_radius=uav_radius,
                        ),
                    )
                )
                if instance.obstacle_map.obstacles
                else False,
                "planner_called": bool(
                    cdiag.get("planner_called", cdiag.get("used_existing_safety_modules", False) or cdiag.get("local_planner_blocked", False))
                    or pdiag.get("slot_planner_called", False)
                ),
                "planner_success": bool(cdiag.get("planner_success", False) or pdiag.get("slot_planner_success", False)),
                "planner_mode_active": bool(cdiag.get("planner_mode_active", False) or pdiag.get("slot_planner_called", False)),
                "replan_reason": str(pdiag.get("slot_replan_reason", "") or cdiag.get("replan_reason", cdiag.get("planner_reason", ""))),
                "planner_path_length": float(cdiag.get("planner_path_length", np.nan)),
                "planner_replan_count": int(cdiag.get("planner_replan_count", 0)),
                "current_path_id": int(cdiag.get("current_path_id", 0)),
                "current_path_waypoints": str(cdiag.get("current_path_waypoints", planned_path_value)),
                "current_path_progress_index": int(cdiag.get("current_path_progress_index", -1)),
                "current_path_arclength_progress": float(cdiag.get("current_path_arclength_progress", np.nan)),
                "path_valid": bool(cdiag.get("path_valid", False) or pdiag.get("slot_planner_path_valid", False)),
                "path_deviation": float(cdiag.get("path_deviation", np.nan)),
                "lookahead_distance": float(cdiag.get("lookahead_distance", np.nan)),
                "target_mode": target_mode,
                "target_mode_previous": target_prev,
                "target_mode_transition_reason": target_reason,
                "raw_slot_valid": bool(raw_slot_valid),
                "commanded_slot_valid": bool(commanded_slot_valid),
                "commanded_slot_inside_obstacle": bool(pdiag.get("commanded_slot_inside_obstacle", commanded_clear_at_slot < 0.0)),
                "commanded_slot_outside_boundary": bool(pdiag.get("commanded_slot_outside_boundary", np.max(np.abs(commanded_slot[i, :2])) > world_xy)),
                "commanded_slot_too_close_obstacle": bool(pdiag.get("commanded_slot_too_close_obstacle", commanded_clear_at_slot + valid_eps < safety_margin)),
                "commanded_transition_segment_safe": bool(pdiag.get("commanded_transition_segment_safe", True)),
                "commanded_slot_invalid_reason": str(pdiag.get("commanded_slot_invalid_reason", "")),
                "safe_hold_active": bool(pdiag.get("safe_hold_active", False)),
                "safe_hold_reason": str(pdiag.get("safe_hold_reason", "")),
                "proxy_slot_used": bool(pdiag.get("proxy_slot_used", False)),
                "slot_transition_mode": str(pdiag.get("slot_transition_mode", "DISABLED")),
                "slot_transition_mode_previous": str(prev_slot_transition_modes[i]),
                "jump_detected": bool(pdiag.get("jump_detected", False)),
                "jump_distance": float(pdiag.get("jump_distance", 0.0)),
                "minimum_reach_time": float(pdiag.get("minimum_reach_time", 0.0)),
                "jump_interval_steps": int(pdiag.get("jump_interval_steps", 0)),
                "raw_slot_too_unstable": bool(pdiag.get("raw_slot_too_unstable", False)),
                "transition_progress": float(pdiag.get("transition_progress", 1.0)),
                "commanded_slot_lag_to_raw": float(pdiag.get("commanded_slot_lag_to_raw", np.linalg.norm(commanded_slot[i, :2] - raw_slot[i, :2]))),
                "commanded_slot_step_norm": float(commanded_shift_norm),
                "proxy_slot_valid": bool(proxy_slot_valid),
                "los_to_raw": bool(los_raw),
                "los_to_proxy": bool(cdiag.get("line_of_sight_to_proxy_slot", los_proxy)),
                "selected_subgoal_valid": bool(selected_subgoal_valid),
                "stale_subgoal_used": bool(stale_subgoal_used),
                "target_mode_invariant_violation": bool(invariant_violation),
                "target_mode_invariant": str(invariant_name),
                "current_subgoal_x": float(current_subgoal[0]),
                "current_subgoal_y": float(current_subgoal[1]),
                "current_subgoal_z": float(commanded_slot[i, 2]),
                "planned_path": str(planned_path_value),
                "current_path_segment": str(current_segment_value),
                "selected_subgoal_path": str(cdiag.get("selected_subgoal_path", "")),
                "min_distance_to_planned_path": float(min_path),
                "min_distance_to_current_segment": float(min_current_segment),
                "min_distance_to_subgoal_segment": float(min_subgoal_segment),
                "line_of_sight_to_proxy": bool(los_proxy),
                "line_of_sight_to_raw_slot": bool(los_raw),
                "line_of_sight_to_proxy_slot": bool(cdiag.get("line_of_sight_to_proxy_slot", los_proxy)),
                "nominal_rollout_safe": bool(cdiag.get("nominal_rollout_safe", False)),
                "nominal_rollout_min_clearance": float(cdiag.get("nominal_rollout_min_clearance", np.nan)),
                "line_of_sight_to_subgoal": bool(los_subgoal),
                "path_blocked": bool(cdiag.get("path_blocked", False)),
                "subgoal_visible": bool(cdiag.get("subgoal_visible", los_subgoal)),
                "selected_subgoal_visible": bool(cdiag.get("selected_subgoal_visible", los_subgoal)),
                "lookahead_segment_clearance": float(cdiag.get("lookahead_segment_clearance", min_subgoal_segment)),
                "no_visible_subgoal": bool(cdiag.get("no_visible_subgoal", False)),
                "planned_path_min_clearance": float(cdiag.get("planned_path_min_clearance", min_path)),
                "planned_path_mean_clearance": float(cdiag.get("planned_path_mean_clearance", np.nan)),
                "subgoal_backtracked": bool(cdiag.get("subgoal_backtracked", False)),
                "backtracked_waypoint_index": int(cdiag.get("backtracked_waypoint_index", -1)),
                "proxy_slot_shift_norm": float(proxy_shift_norm),
                "subgoal_shift_norm": float(subgoal_shift_norm),
                "subgoal_held": bool(cdiag.get("subgoal_held", False)),
                "collision_radius": float(cdiag.get("collision_radius", uav_radius)),
                "safety_margin": float(cdiag.get("safety_margin", safety_margin)),
                "tracking_margin": float(cdiag.get("tracking_margin", controller_cfg.get("planner_subgoal", controller_cfg).get("tracking_margin", controller_cfg.get("planner_subgoal", controller_cfg).get("planning_safety_margin_extra", 0.0)) if isinstance(controller_cfg, dict) else 0.0)),
                "planning_inflation_radius": float(cdiag.get("planning_inflation_radius", uav_radius + safety_margin)),
                "steps_since_last_replan": int(cdiag.get("steps_since_last_replan", 0)),
                "replan_allowed_by_cooldown": bool(cdiag.get("replan_allowed_by_cooldown", True)),
                "replan_blocked_by_cooldown": bool(cdiag.get("replan_blocked_by_cooldown", False)),
                "predicted_next_pos_x": float(predicted_next[0]),
                "predicted_next_pos_y": float(predicted_next[1]),
                "segment_collision_current_to_next": bool(ddiag.get("segment_collision_current_to_next", False)),
                "min_predicted_next_clearance": float(ddiag.get("min_predicted_next_clearance", np.nan)),
                "predictive_filter_active": bool(ddiag.get("predictive_filter_active", False)),
                "predictive_filter_subtype": str(ddiag.get("predictive_filter_subtype", "")),
                "stuck_detector_active": bool(err > float(dyn["lost_threshold"]) and nearest_clear < safety_margin),
                "cos_to_subgoal": float(subgoal_progress["cos_to_goal"]),
                "cos_to_raw_slot": float(raw_progress["cos_to_goal"]),
                "cos_to_proxy_slot": float(proxy_progress["cos_to_goal"]),
                "cos_to_commanded_slot": float(progress["cos_to_goal"]),
                "progress_to_subgoal": float(subgoal_progress["progress_projection"]),
                "progress_to_raw_slot": float(raw_progress["progress_projection"]),
                "progress_to_proxy_slot": float(proxy_progress["progress_projection"]),
                "progress_to_commanded_slot": float(progress["progress_projection"]),
                "obstacle_collision": obstacle_collision(next_pos[i, :2], instance.obstacle_map.obstacles, uav_radius=uav_radius),
                "boundary_violation": outside_boundary(next_pos[i, :2], world_xy=world_xy, uav_radius=uav_radius),
                "inter_agent_collision": bool(coll_agent),
                "slot_outside_boundary": bool(np.max(np.abs(raw_slot[i, :2])) > world_xy),
                "failure_type": "",
                "failure_subtype": "",
            }
            rows.append(row)
            prev_errors[i] = err
            prev_proxy_slot_xy[i] = proxy_slot[i, :2].copy()
            prev_commanded_slot[i] = commanded_slot[i].copy()
            prev_slot_transition_modes[i] = str(pdiag.get("slot_transition_mode", prev_slot_transition_modes[i]))
            prev_subgoal_xy[i] = current_subgoal.copy()
        pos = next_pos
        vel = next_vel

    tracking = compute_tracking_metrics(rows, cfg=cfg, jump_steps=instance.jump_steps)
    safety = compute_safety_metrics(rows, cfg=cfg)
    failure_timing = compute_failure_timing_metrics(rows, cfg=cfg)
    d_group_metrics = compute_d_group_metrics(rows, cfg=cfg) if instance.spec.group == "D" else {}
    decision = np.asarray([r["decision_time_ms"] for r in rows], dtype=np.float64)
    v_obs_norm = np.asarray([np.hypot(r.get("v_obstacle_x", np.nan), r.get("v_obstacle_y", np.nan)) for r in rows], dtype=np.float64)
    v_bound_norm = np.asarray([np.hypot(r.get("v_boundary_x", np.nan), r.get("v_boundary_y", np.nan)) for r in rows], dtype=np.float64)
    used_existing = np.asarray([bool(r.get("used_existing_safety_modules", False)) for r in rows], dtype=bool)
    los_blocked = np.asarray([bool(r.get("line_of_sight_blocked", False)) for r in rows], dtype=bool)
    planner_called = np.asarray([bool(r.get("planner_called", False)) for r in rows], dtype=bool)
    planner_mode_active = np.asarray([bool(r.get("planner_mode_active", False)) for r in rows], dtype=bool)
    stuck_active = np.asarray([bool(r.get("stuck_detector_active", False)) for r in rows], dtype=bool)
    los_shortcut = np.asarray([str(r.get("existing_bypass_reason", "")) == "line_of_sight_shortcut" for r in rows], dtype=bool)
    proxy_adjusted = np.asarray([bool(r.get("proxy_slot_adjusted", False)) for r in rows], dtype=bool)
    raw_inside_obstacle = np.asarray([bool(r.get("raw_slot_inside_obstacle", False)) for r in rows], dtype=bool)
    raw_too_close_obstacle = np.asarray([bool(r.get("raw_slot_too_close_obstacle", False)) for r in rows], dtype=bool)
    raw_errors = np.asarray([float(r.get("tracking_error_to_raw_slot", np.nan)) for r in rows], dtype=np.float64)
    subgoal_errors = np.asarray([float(r.get("tracking_error_to_subgoal", np.nan)) for r in rows], dtype=np.float64)
    planner_success = np.asarray([bool(r.get("planner_success", False)) for r in rows], dtype=bool)
    planner_replans = np.asarray([float(r.get("planner_replan_count", 0.0)) for r in rows], dtype=np.float64)
    replan_steps = np.asarray([idx for idx, r in enumerate(rows) if bool(r.get("planner_called", False))], dtype=np.float64)
    planner_path_lengths = np.asarray([float(r.get("planner_path_length", np.nan)) for r in rows], dtype=np.float64)
    subgoal_shift = np.asarray([float(r.get("subgoal_shift_norm", np.nan)) for r in rows], dtype=np.float64)
    proxy_shift = np.asarray([float(r.get("proxy_slot_shift_norm", np.nan)) for r in rows], dtype=np.float64)
    progress_subgoal = np.asarray([float(r.get("progress_to_subgoal", np.nan)) for r in rows], dtype=np.float64)
    progress_raw_slot = np.asarray([float(r.get("progress_to_raw_slot", np.nan)) for r in rows], dtype=np.float64)
    progress_proxy_slot = np.asarray([float(r.get("progress_to_proxy_slot", np.nan)) for r in rows], dtype=np.float64)
    progress_commanded_slot = np.asarray([float(r.get("progress_to_commanded_slot", np.nan)) for r in rows], dtype=np.float64)
    cos_subgoal = np.asarray([float(r.get("cos_to_subgoal", np.nan)) for r in rows], dtype=np.float64)
    cos_raw_slot = np.asarray([float(r.get("cos_to_raw_slot", np.nan)) for r in rows], dtype=np.float64)
    cos_proxy_slot = np.asarray([float(r.get("cos_to_proxy_slot", np.nan)) for r in rows], dtype=np.float64)
    cos_commanded_slot = np.asarray([float(r.get("cos_to_commanded_slot", np.nan)) for r in rows], dtype=np.float64)
    angle_nom_final = np.asarray([r.get("angle_nom_to_final", np.nan) for r in rows], dtype=np.float64)
    norm_ratio_final = np.asarray([r.get("norm_ratio_final_to_nom", np.nan) for r in rows], dtype=np.float64)
    tangential_vals = []
    for r in rows:
        for boundary_name in ["x_min", "x_max", "y_min", "y_max"]:
            val = float(r.get(f"{boundary_name}_tangential_retention_ratio", np.nan))
            if np.isfinite(val):
                tangential_vals.append(val)
    tangential_retention = np.asarray(tangential_vals, dtype=np.float64)
    post_boundary = np.asarray([bool(r.get("post_boundary_projection_applied", False)) for r in rows], dtype=bool)
    metrics = {
        **tracking,
        **safety,
        **failure_timing,
        **d_group_metrics,
        **{k: v for k, v in instance.feasibility_metrics.items() if k not in ("target_infeasible", "infeasible_reason")},
        "invalid_initial_state": bool(initial_invalid),
        "initial_boundary_margin": float(initial_margin),
        "initial_velocity_outward_projection": float(initial_outward_velocity),
        "decision_time_ms_mean": float(np.mean(decision)) if decision.size else float("nan"),
        "decision_time_ms_p95": float(np.percentile(decision, 95)) if decision.size else float("nan"),
        "decision_time_ms_p99": float(np.percentile(decision, 99)) if decision.size else float("nan"),
        "decision_time_p95": float(np.percentile(decision, 95)) if decision.size else float("nan"),
        "decision_time_p99": float(np.percentile(decision, 99)) if decision.size else float("nan"),
        "stuck_ratio": float(np.mean(stuck_active)) if stuck_active.size else 0.0,
        "line_of_sight_blocked_ratio": float(np.mean(los_blocked)) if los_blocked.size else 0.0,
        "planner_called_ratio": float(np.mean(planner_called)) if planner_called.size else 0.0,
        "planner_mode_active_ratio": float(np.mean(planner_mode_active)) if planner_mode_active.size else 0.0,
        "planner_success_ratio": float(np.mean(planner_success[planner_called])) if np.any(planner_called) else 0.0,
        "planner_replan_count": float(np.max(planner_replans[np.isfinite(planner_replans)])) if np.any(np.isfinite(planner_replans)) else 0.0,
        "mean_replan_interval": float(np.mean(np.diff(replan_steps))) if replan_steps.size > 1 else float("inf"),
        "planner_path_length_mean": _finite_mean_local(planner_path_lengths),
        "subgoal_shift_norm_mean": _finite_mean_local(subgoal_shift),
        "subgoal_shift_norm_p95": float(np.percentile(subgoal_shift[np.isfinite(subgoal_shift)], 95)) if np.any(np.isfinite(subgoal_shift)) else float("nan"),
        "proxy_slot_shift_norm_mean": _finite_mean_local(proxy_shift),
        "proxy_slot_shift_norm_p95": float(np.percentile(proxy_shift[np.isfinite(proxy_shift)], 95)) if np.any(np.isfinite(proxy_shift)) else float("nan"),
        "mean_tracking_error_to_subgoal": _finite_mean_local(subgoal_errors),
        "p95_tracking_error_to_subgoal": float(np.percentile(subgoal_errors[np.isfinite(subgoal_errors)], 95)) if np.any(np.isfinite(subgoal_errors)) else float("nan"),
        "mean_progress_to_subgoal": _finite_mean_local(progress_subgoal),
        "mean_progress_to_raw_slot": _finite_mean_local(progress_raw_slot),
        "mean_progress_to_proxy_slot": _finite_mean_local(progress_proxy_slot),
        "mean_progress_to_commanded_slot": _finite_mean_local(progress_commanded_slot),
        "mean_cos_to_subgoal": _finite_mean_local(cos_subgoal),
        "mean_cos_to_raw_slot": _finite_mean_local(cos_raw_slot),
        "mean_cos_to_proxy_slot": _finite_mean_local(cos_proxy_slot),
        "mean_cos_to_commanded_slot": _finite_mean_local(cos_commanded_slot),
        "stuck_detector_active_ratio": float(np.mean(stuck_active)) if stuck_active.size else 0.0,
        "line_of_sight_shortcut_ratio": float(np.mean(los_shortcut)) if los_shortcut.size else 0.0,
        "proxy_slot_adjusted_ratio": float(np.mean(proxy_adjusted)) if proxy_adjusted.size else 0.0,
        "raw_slot_inside_obstacle_runtime_ratio": float(np.mean(raw_inside_obstacle)) if raw_inside_obstacle.size else 0.0,
        "raw_slot_too_close_obstacle_runtime_ratio": float(np.mean(raw_too_close_obstacle)) if raw_too_close_obstacle.size else 0.0,
        "p95_error_to_raw_slot": float(np.percentile(raw_errors[np.isfinite(raw_errors)], 95)) if np.any(np.isfinite(raw_errors)) else float("nan"),
        "mean_v_obstacle_norm": _finite_mean_local(v_obs_norm),
        "mean_v_boundary_norm": _finite_mean_local(v_bound_norm),
        "used_existing_safety_modules_ratio": float(np.mean(used_existing)) if used_existing.size else 0.0,
        "mean_angle_nom_to_final": _finite_mean_local(angle_nom_final),
        "mean_norm_ratio_final_to_nom": _finite_mean_local(norm_ratio_final),
        "mean_tangential_retention_ratio": _finite_mean_local(tangential_retention),
        "post_boundary_projection_ratio": float(np.mean(post_boundary)) if post_boundary.size else 0.0,
        "target_mode_invariant_violation_count": int(sum(bool(r.get("target_mode_invariant_violation", False)) for r in rows)),
        "lookahead_segment_collision_count": int(sum(float(r.get("lookahead_segment_clearance", float("inf"))) < 0.0 for r in rows)),
        "no_visible_subgoal_count": int(sum(bool(r.get("no_visible_subgoal", False)) for r in rows)),
        "planned_path_below_required_count": int(
            sum(
                float(r.get("planned_path_min_clearance", float("inf"))) < float(r.get("safety_margin", dyn.get("safety_margin", 0.0))) + float(r.get("tracking_margin", 0.0))
                and float(r.get("nearest_obstacle_distance", float("inf"))) >= float(r.get("safety_margin", dyn.get("safety_margin", 0.0))) + float(r.get("tracking_margin", 0.0))
                and str(r.get("target_mode", "")) == "PATH_SUBGOAL_TRACKING"
                for r in rows
            )
        ),
    }
    metrics["safe_rejection_success"] = bool(
        instance.spec.group == "D"
        and not instance.feasible
        and int(metrics.get("number_of_jump_events", 0)) > 0
        and not bool(metrics.get("obstacle_collision", False))
        and not bool(metrics.get("boundary_violation", False))
        and not bool(metrics.get("inter_agent_collision", False))
        and float(metrics.get("invalid_commanded_slot_ratio", 0.0)) <= 0.0
        and int(metrics.get("target_mode_invariant_violation_count", 0)) == 0
    )
    metrics["safe_stabilization_success"] = bool(
        instance.spec.group == "D"
        and float(metrics.get("raw_slot_too_unstable_ratio", 0.0)) > 0.0
        and not bool(metrics.get("obstacle_collision", False))
        and not bool(metrics.get("boundary_violation", False))
        and not bool(metrics.get("inter_agent_collision", False))
        and float(metrics.get("invalid_commanded_slot_ratio", 0.0)) <= 0.0
        and int(metrics.get("target_mode_invariant_violation_count", 0)) == 0
        and bool(metrics.get("reacquired_within_budget", True))
    )
    success = episode_success(metrics, feasible=instance.feasible, cfg=cfg)
    if success and instance.spec.group == "C":
        success_cfg = cfg.get("success", {})
        p95_threshold = float(success_cfg.get("p95_error_threshold", float("inf")))
        if float(metrics.get("p95_error", float("inf"))) > p95_threshold or float(metrics.get("slot_lost_ratio", float("inf"))) > float(success_cfg.get("max_lost_ratio", 0.0)):
            print("WARNING: success criterion inconsistent with tracking quality.")
    failure_type = classify_failure(rows, metrics, feasible=instance.feasible, cfg=cfg)
    failure_subtype = classify_failure_subtype(rows, metrics, feasible=instance.feasible, cfg=cfg)
    collision_root_cause = classify_collision_root_cause(rows, metrics, cfg=cfg)
    divergence_root_cause = classify_sparse_divergence_root_cause(rows, metrics, cfg=cfg)
    if int(metrics.get("target_mode_invariant_violation_count", 0)) > 0:
        failure_subtype = "WRONG_TARGET_MODE"
    for row in rows:
        row["failure_type"] = failure_type
        row["failure_subtype"] = failure_subtype
        row["failure_step"] = int(metrics.get("failure_step", -1))
        row["episode_end_step"] = int(metrics.get("episode_end_step", len(rows) - 1))
        row["simulation_continues_after_failure"] = bool(metrics.get("simulation_continues_after_failure", False))
        row["post_failure_state_included"] = bool(metrics.get("post_failure_state_included", False))
        row["tracking_error_at_failure_step"] = float(metrics.get("tracking_error_at_failure_step", np.nan))
        row["max_tracking_error_before_failure_step"] = float(metrics.get("max_tracking_error_before_failure_step", np.nan))
        row["p95_error_pre_failure"] = float(metrics.get("p95_error_pre_failure", np.nan))
        row["p95_error_full_episode"] = float(metrics.get("p95_error_full_episode", np.nan))
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
            writer.writeheader()
            writer.writerows({k: r.get(k, "") for k in RAW_COLUMNS} for r in rows)
    return {
        "controller": controller_name,
        "controller_base": base_controller_name,
        "target_source": "commanded_slot" if use_slot_transition else ("raw_slot" if use_raw_slot_target else "proxy_slot"),
        "shared_predictive_safety": bool(use_predictive_safety),
        "scenario": instance.spec.name,
        "scenario_group": instance.spec.group,
        "obstacle_type": instance.obstacle_map.name,
        "speed_level": float(instance.spec.speed_scale),
        "noise_level": float(instance.spec.noise_std),
        "action_delay_steps": int(instance.spec.action_delay_steps),
        "actual_vmax_scale": float(instance.spec.actual_vmax_scale),
        "wind_std": float(instance.spec.wind_std),
        "obstacle_dropout_prob": float(instance.spec.obstacle_dropout_prob),
        "seed": int(seed),
        "num_agents": int(n),
        "feasible": bool(instance.feasible),
        "infeasible_reason": instance.infeasible_reason,
        "success": bool(success),
        "failure_type": failure_type,
        "failure_subtype": failure_subtype,
        "collision_root_cause": collision_root_cause,
        "divergence_root_cause": divergence_root_cause,
        "evaluation_feasible": bool(
            instance.feasible
            and not initial_invalid
            and float(metrics.get("slot_out_of_bounds_ratio", 0.0)) <= float(
                cfg.get("failure_classifier", {}).get("slot_out_of_bounds_threshold", 0.35)
            )
        ),
        "obstacle_count": int(len(instance.obstacle_map.obstacles)),
        "obstacles": str(obstacle_records(instance.obstacle_map.obstacles)),
        "raw_path": "" if raw_path is None else str(raw_path),
        **metrics,
    }


def classify_collision_root_cause(rows: list[dict[str, Any]], metrics: dict[str, Any], *, cfg: dict[str, Any]) -> str:
    if not bool(metrics.get("obstacle_collision", False)):
        return "NONE"
    collision_rows = [r for r in rows if bool(r.get("obstacle_collision", False))]
    row = collision_rows[0] if collision_rows else (rows[-1] if rows else {})
    safety_margin = float(row.get("safety_margin", cfg.get("dynamics", {}).get("safety_margin", 0.0)))
    tracking_margin = float(row.get("tracking_margin", 0.0))
    required_clearance = safety_margin + max(tracking_margin, 0.0)
    min_planned = float(row.get("min_distance_to_planned_path", np.nan))
    min_subgoal = float(row.get("min_distance_to_subgoal_segment", np.nan))
    min_current = float(row.get("min_distance_to_current_segment", np.nan))
    pred_clear = float(row.get("min_predicted_next_clearance", np.nan))
    if np.isfinite(min_planned) and min_planned < required_clearance:
        return "A.PLANNED_PATH_TOO_CLOSE"
    if (
        bool(row.get("line_of_sight_to_subgoal", True)) is False
        or bool(row.get("selected_subgoal_visible", True)) is False
        or (np.isfinite(min_subgoal) and min_subgoal < required_clearance)
    ):
        return "B.LOOKAHEAD_SEGMENT_COLLISION"
    if bool(row.get("segment_collision_current_to_next", False)) or (np.isfinite(pred_clear) and pred_clear < 0.0):
        return "D.FINAL_STEP_PREDICTION_FAILURE"
    if (
        np.isfinite(min_planned)
        and np.isfinite(min_subgoal)
        and min_planned >= required_clearance
        and min_subgoal >= required_clearance
    ):
        return "C.TRACKING_DEVIATION_FROM_PATH"
    if bool(row.get("path_blocked", False)) is False and np.isfinite(min_current) and min_current < 0.0:
        return "E.OBSTACLE_MAP_OR_COLLISION_CHECK_MISMATCH"
    return "E.OBSTACLE_MAP_OR_COLLISION_CHECK_MISMATCH"


def compute_failure_timing_metrics(rows: list[dict[str, Any]], *, cfg: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {
            "failure_step": -1,
            "episode_end_step": -1,
            "simulation_continues_after_failure": False,
            "post_failure_state_included": False,
            "tracking_error_at_failure_step": float("nan"),
            "max_tracking_error_before_failure_step": float("nan"),
            "p95_error_pre_failure": float("nan"),
            "p95_error_full_episode": float("nan"),
        }
    errors = np.asarray([float(r.get("tracking_error", np.nan)) for r in rows], dtype=np.float64)
    dt = float(cfg.get("dynamics", {}).get("dt", 1.0))
    step_numbers = np.asarray([max(int(round(float(r.get("t", 0.0)) / max(dt, 1e-9))) - 1, 0) for r in rows], dtype=np.int64)
    hard_idx = [
        idx for idx, r in enumerate(rows)
        if bool(r.get("obstacle_collision", False))
        or bool(r.get("boundary_violation", False))
        or bool(r.get("inter_agent_collision", False))
    ]
    if hard_idx:
        failure_idx = int(hard_idx[0])
    else:
        lost_threshold = float(cfg.get("dynamics", {}).get("lost_threshold", float("inf")))
        lost = np.where(errors > lost_threshold)[0]
        failure_idx = int(lost[0]) if lost.size else len(rows) - 1
    failure_step = int(step_numbers[failure_idx]) if step_numbers.size else -1
    episode_end_step = int(np.max(step_numbers)) if step_numbers.size else -1
    pre = errors[: failure_idx + 1]
    pre = pre[np.isfinite(pre)]
    full = errors[np.isfinite(errors)]
    continues = failure_step < episode_end_step
    return {
        "failure_step": int(failure_step),
        "episode_end_step": int(episode_end_step),
        "simulation_continues_after_failure": bool(continues),
        "post_failure_state_included": bool(continues),
        "tracking_error_at_failure_step": float(errors[failure_idx]) if np.isfinite(errors[failure_idx]) else float("nan"),
        "max_tracking_error_before_failure_step": float(np.max(pre)) if pre.size else float("nan"),
        "p95_error_pre_failure": float(np.percentile(pre, 95)) if pre.size else float("nan"),
        "p95_error_full_episode": float(np.percentile(full, 95)) if full.size else float("nan"),
    }


def compute_d_group_metrics(rows: list[dict[str, Any]], *, cfg: dict[str, Any]) -> dict[str, Any]:
    """Jump-aware D-group episode metrics."""
    if not rows:
        return {}
    dyn = cfg.get("dynamics", {})
    success_cfg = cfg.get("success", {})
    dt = float(dyn.get("dt", 0.05))
    uav_vmax = max(float(dyn.get("uav_vmax", 1.0)), 1e-9)
    errors = np.asarray([float(r.get("tracking_error", np.nan)) for r in rows], dtype=np.float64)
    raw_errors = np.asarray([float(r.get("tracking_error_to_raw_slot", np.nan)) for r in rows], dtype=np.float64)
    lag = np.asarray([float(r.get("commanded_slot_lag_to_raw", np.nan)) for r in rows], dtype=np.float64)
    commanded_step = np.asarray([float(r.get("commanded_slot_step_norm", np.nan)) for r in rows], dtype=np.float64)
    jump_idx = [idx for idx, r in enumerate(rows) if bool(r.get("jump_detected", False))]
    raw_xy = np.asarray([[float(r.get("raw_slot_x", np.nan)), float(r.get("raw_slot_y", np.nan))] for r in rows], dtype=np.float64)
    commanded_valid = np.asarray([bool(r.get("commanded_slot_valid", True)) for r in rows], dtype=bool)
    raw_valid = np.asarray([bool(r.get("raw_slot_valid", True)) for r in rows], dtype=bool)
    raw_unstable = np.asarray([bool(r.get("raw_slot_too_unstable", False)) for r in rows], dtype=bool)
    safe_hold = np.asarray([bool(r.get("safe_hold_active", False)) for r in rows], dtype=bool)
    transition_safe = np.asarray([bool(r.get("commanded_transition_segment_safe", True)) for r in rows], dtype=bool)
    proxy_valid = np.asarray([bool(r.get("proxy_slot_valid", True)) for r in rows], dtype=bool)
    cmd_inside = np.asarray([bool(r.get("commanded_slot_inside_obstacle", False)) for r in rows], dtype=bool)
    cmd_outside = np.asarray([bool(r.get("commanded_slot_outside_boundary", False)) for r in rows], dtype=bool)
    planner_called = np.asarray([bool(r.get("planner_called", False)) for r in rows], dtype=bool)
    planner_success = np.asarray([bool(r.get("planner_success", False)) for r in rows], dtype=bool)
    subgoal_visible = np.asarray([bool(r.get("selected_subgoal_visible", False)) for r in rows], dtype=bool)
    path_valid_rows = np.asarray([bool(r.get("path_valid", False)) for r in rows], dtype=bool)
    planner_replans = np.asarray([float(r.get("planner_replan_count", 0.0)) for r in rows], dtype=np.float64)
    target_modes = [str(r.get("target_mode", "")) for r in rows]
    transition_modes = [str(r.get("slot_transition_mode", "")) for r in rows]
    target_switches = sum(1 for a, b in zip(target_modes[:-1], target_modes[1:]) if a != b)
    transition_switches = sum(1 for a, b in zip(transition_modes[:-1], transition_modes[1:]) if a != b)
    jump_distances: list[float] = []
    minimum_reach_times: list[float] = []
    available_windows: list[float] = []
    pre_errors: list[float] = []
    post_max_errors: list[float] = []
    reacq_times: list[float] = []
    allowed_times: list[float] = []
    reacquired_flags: list[bool] = []
    post_reacq_errors: list[float] = []
    post_reacq_lost: list[float] = []
    oscillation_flags: list[bool] = []
    planner_called_after_jump_flags: list[bool] = []
    planner_success_after_jump_flags: list[bool] = []
    selected_subgoal_after_jump_flags: list[bool] = []
    subgoal_visible_after_jump_flags: list[bool] = []
    path_valid_after_jump_flags: list[bool] = []
    replan_reasons_after_jump: list[str] = []
    replan_counts_after_jump: list[float] = []
    threshold = float(success_cfg.get("reacquisition_threshold", success_cfg.get("steady_state_error_threshold", 0.9)))
    lock_threshold = float(success_cfg.get("lock_threshold", threshold))
    lock_window = max(int(success_cfg.get("lock_window", 20)), 1)
    factor = float(success_cfg.get("jump_reacquisition_budget_factor", 1.6))
    min_budget = float(success_cfg.get("jump_reacquisition_min_s", 0.8))
    lost_threshold = float(dyn.get("lost_threshold", 1.5))
    post_window_s = float(success_cfg.get("post_reacquisition_window_s", 2.0))
    post_window = max(int(post_window_s / max(dt, 1e-9)), 1)
    for event_pos, js in enumerate(jump_idx):
        prev = max(js - 1, 0)
        jump_distance = float(np.linalg.norm(raw_xy[js] - raw_xy[prev])) if raw_xy.shape[0] > prev else float("nan")
        next_event = jump_idx[event_pos + 1] if event_pos + 1 < len(jump_idx) else len(rows) - 1
        available = max(float(next_event - js) * dt, 0.0)
        minimum = jump_distance / uav_vmax if np.isfinite(jump_distance) else float("inf")
        allowed = min(available, max(minimum * factor, min_budget))
        deadline = min(len(rows) - 1, js + int(np.ceil(allowed / max(dt, 1e-9))))
        found = None
        for k in range(js, deadline + 1):
            if np.isfinite(errors[k]) and errors[k] <= threshold:
                found = k
                break
        reacq_time = float("inf") if found is None else float(found - js) * dt
        post_start = found if found is not None else deadline
        post_end = min(len(rows), post_start + post_window)
        post_errors = errors[post_start:post_end]
        jump_distances.append(jump_distance)
        minimum_reach_times.append(minimum)
        available_windows.append(available)
        pre_slice = errors[max(0, js - lock_window):js]
        pre_errors.append(float(np.percentile(pre_slice[np.isfinite(pre_slice)], 95)) if np.any(np.isfinite(pre_slice)) else float("nan"))
        post_slice = errors[js:deadline + 1]
        post_max_errors.append(float(np.nanmax(post_slice)) if post_slice.size else float("nan"))
        reacq_times.append(reacq_time)
        allowed_times.append(float(allowed))
        reacquired = bool(np.isfinite(reacq_time) and reacq_time <= allowed + 1e-9)
        reacquired_flags.append(reacquired)
        post_reacq_errors.append(float(np.percentile(post_errors[np.isfinite(post_errors)], 95)) if np.any(np.isfinite(post_errors)) else float("nan"))
        post_reacq_lost.append(float(np.mean(post_errors > lost_threshold)) if post_errors.size else float("nan"))
        actions = np.asarray([[float(r.get("action_x", 0.0)), float(r.get("action_y", 0.0))] for r in rows[post_start:post_end]], dtype=np.float64)
        reversals = 0.0
        if actions.shape[0] > 2:
            norms = np.linalg.norm(actions, axis=1)
            valid = norms > 1e-6
            unit = actions[valid] / norms[valid].reshape(-1, 1) if np.any(valid) else np.zeros((0, 2))
            if unit.shape[0] > 1:
                reversals = float(np.mean(np.sum(unit[1:] * unit[:-1], axis=1) < -0.2))
        oscillation_flags.append(bool(post_errors.size and np.nanstd(post_errors) > float(success_cfg.get("post_jump_oscillation_std_threshold", 0.45)) and reversals > 0.20))
        event_window = slice(js, min(deadline + 1, len(rows)))
        planner_called_after_jump_flags.append(bool(np.any(planner_called[event_window])) if planner_called.size else False)
        planner_success_after_jump_flags.append(bool(np.any(planner_called[event_window] & planner_success[event_window])) if planner_called.size else False)
        selected_subgoal_after_jump_flags.append(bool(np.any(subgoal_visible[event_window] | path_valid_rows[event_window])) if path_valid_rows.size else False)
        subgoal_visible_after_jump_flags.append(bool(np.any(subgoal_visible[event_window])) if subgoal_visible.size else False)
        path_valid_after_jump_flags.append(bool(np.any(path_valid_rows[event_window])) if path_valid_rows.size else False)
        reasons = [str(rows[idx].get("replan_reason", "")) for idx in range(event_window.start or 0, event_window.stop or 0) if str(rows[idx].get("replan_reason", ""))]
        if reasons:
            replan_reasons_after_jump.append(reasons[0])
        if planner_replans.size and np.any(np.isfinite(planner_replans[event_window])):
            start_count = float(planner_replans[js - 1]) if js > 0 and np.isfinite(planner_replans[js - 1]) else 0.0
            end_count = float(np.nanmax(planner_replans[event_window]))
            replan_counts_after_jump.append(max(end_count - start_count, 0.0))
        else:
            replan_counts_after_jump.append(0.0)

    jump_dist_arr = np.asarray(jump_distances, dtype=np.float64)
    reacq_arr = np.asarray(reacq_times, dtype=np.float64)
    allowed_arr = np.asarray(allowed_times, dtype=np.float64)
    post_p95_arr = np.asarray(post_reacq_errors, dtype=np.float64)
    post_lost_arr = np.asarray(post_reacq_lost, dtype=np.float64)
    finite_commanded_step = commanded_step[np.isfinite(commanded_step)]
    return {
        "number_of_jump_events": int(len(jump_idx)),
        "jump_times": ";".join(f"{idx * dt:.3f}" for idx in jump_idx),
        "jump_distance_mean": float(np.nanmean(jump_dist_arr)) if jump_dist_arr.size else 0.0,
        "jump_distance_p95": float(np.percentile(jump_dist_arr[np.isfinite(jump_dist_arr)], 95)) if np.any(np.isfinite(jump_dist_arr)) else 0.0,
        "jump_distance_max": float(np.nanmax(jump_dist_arr)) if jump_dist_arr.size else 0.0,
        "minimum_reach_time": float(np.nanmean(minimum_reach_times)) if minimum_reach_times else 0.0,
        "available_reacquisition_window": float(np.nanmean(available_windows)) if available_windows else 0.0,
        "jump_too_large_for_window": bool(
            any(float(m) > float(w) * factor for m, w in zip(minimum_reach_times, available_windows))
        ),
        "pre_jump_error": float(np.nanmean(pre_errors)) if pre_errors else float("nan"),
        "pre_jump_locked": bool(not pre_errors or np.nanmax(np.asarray(pre_errors, dtype=np.float64)) <= lock_threshold),
        "post_jump_max_error": float(np.nanmax(np.asarray(post_max_errors, dtype=np.float64))) if post_max_errors else float("nan"),
        "reacquisition_time": float(np.nanmean(reacq_arr[np.isfinite(reacq_arr)])) if np.any(np.isfinite(reacq_arr)) else (float("inf") if reacq_arr.size else 0.0),
        "reacquisition_time_p95": float(np.percentile(reacq_arr[np.isfinite(reacq_arr)], 95)) if np.any(np.isfinite(reacq_arr)) else (float("inf") if reacq_arr.size else 0.0),
        "allowed_reacquisition_time": float(np.nanmean(allowed_arr)) if allowed_arr.size else 0.0,
        "allowed_reacquisition_time_p95": float(np.percentile(allowed_arr[np.isfinite(allowed_arr)], 95)) if np.any(np.isfinite(allowed_arr)) else 0.0,
        "allowed_reacquisition_times": ";".join(f"{x:.3f}" for x in allowed_times),
        "reacquired_within_budget": bool(all(reacquired_flags)) if reacquired_flags else True,
        "reacquired_within_budget_ratio": float(np.mean(reacquired_flags)) if reacquired_flags else 1.0,
        "reacquired_within_budget_rate": float(np.mean(reacquired_flags)) if reacquired_flags else 1.0,
        "post_reacquisition_p95_error": float(np.nanmean(post_p95_arr)) if post_p95_arr.size else float("nan"),
        "post_reacquisition_slot_lost_ratio": float(np.nanmean(post_lost_arr)) if post_lost_arr.size else 0.0,
        "full_episode_raw_slot_p95_error": float(np.percentile(raw_errors[np.isfinite(raw_errors)], 95)) if np.any(np.isfinite(raw_errors)) else float("nan"),
        "full_episode_commanded_slot_p95_error": float(np.percentile(errors[np.isfinite(errors)], 95)) if np.any(np.isfinite(errors)) else float("nan"),
        "raw_slot_p95_error": float(np.percentile(raw_errors[np.isfinite(raw_errors)], 95)) if np.any(np.isfinite(raw_errors)) else float("nan"),
        "commanded_slot_p95_error": float(np.percentile(errors[np.isfinite(errors)], 95)) if np.any(np.isfinite(errors)) else float("nan"),
        "commanded_slot_lag_to_raw": float(np.nanmean(lag)) if lag.size else float("nan"),
        "commanded_slot_lag_to_raw_p95": float(np.percentile(lag[np.isfinite(lag)], 95)) if np.any(np.isfinite(lag)) else float("nan"),
        "commanded_slot_lag_to_raw_max": float(np.nanmax(lag)) if lag.size else float("nan"),
        "target_mode_switch_count": int(target_switches),
        "slot_transition_mode_switch_count": int(transition_switches),
        "commanded_slot_jitter": float(np.std(finite_commanded_step)) if finite_commanded_step.size else 0.0,
        "commanded_slot_jitter_p95": float(np.percentile(finite_commanded_step, 95)) if finite_commanded_step.size else 0.0,
        "planner_called_after_jump": float(np.mean(planner_called_after_jump_flags)) if planner_called_after_jump_flags else 0.0,
        "planner_success_after_jump": float(np.mean(planner_success_after_jump_flags)) if planner_success_after_jump_flags else 0.0,
        "selected_subgoal_after_jump": float(np.mean(selected_subgoal_after_jump_flags)) if selected_subgoal_after_jump_flags else 0.0,
        "subgoal_visible_after_jump": float(np.mean(subgoal_visible_after_jump_flags)) if subgoal_visible_after_jump_flags else 0.0,
        "path_valid_after_jump": float(np.mean(path_valid_after_jump_flags)) if path_valid_after_jump_flags else 0.0,
        "replan_reason_after_jump": ";".join(replan_reasons_after_jump[:8]),
        "replan_count_after_jump": float(np.mean(replan_counts_after_jump)) if replan_counts_after_jump else 0.0,
        "oscillation_after_jump": bool(any(oscillation_flags)),
        "invalid_commanded_slot_ratio": float(np.mean(~commanded_valid)) if commanded_valid.size else 0.0,
        "raw_slot_invalid_ratio": float(np.mean(~raw_valid)) if raw_valid.size else 0.0,
        "raw_slot_too_unstable_ratio": float(np.mean(raw_unstable)) if raw_unstable.size else 0.0,
        "safe_hold_active_ratio": float(np.mean(safe_hold)) if safe_hold.size else 0.0,
        "command_transition_blocked_ratio": float(np.mean(~transition_safe)) if transition_safe.size else 0.0,
        "no_valid_proxy_slot_ratio": float(np.mean(~proxy_valid)) if proxy_valid.size else 0.0,
        "commanded_slot_inside_obstacle_ratio": float(np.mean(cmd_inside)) if cmd_inside.size else 0.0,
        "commanded_slot_outside_boundary_ratio": float(np.mean(cmd_outside)) if cmd_outside.size else 0.0,
    }


def classify_sparse_divergence_root_cause(rows: list[dict[str, Any]], metrics: dict[str, Any], *, cfg: dict[str, Any]) -> str:
    if float(metrics.get("p95_error", 0.0)) <= 5.0 and str(metrics.get("failure_type", "")) != "TRACKING_DIVERGENCE":
        return "NONE"
    nominal_safe = np.asarray([bool(r.get("nominal_rollout_safe", False)) for r in rows], dtype=bool)
    planner_active = np.asarray([bool(r.get("planner_mode_active", False)) for r in rows], dtype=bool)
    planner_called = np.asarray([bool(r.get("planner_called", False)) for r in rows], dtype=bool)
    sub_shift = np.asarray([float(r.get("subgoal_shift_norm", np.nan)) for r in rows], dtype=np.float64)
    proxy_shift = np.asarray([float(r.get("proxy_slot_shift_norm", np.nan)) for r in rows], dtype=np.float64)
    progress_idx = np.asarray([float(r.get("current_path_progress_index", np.nan)) for r in rows], dtype=np.float64)
    los_raw = np.asarray([bool(r.get("line_of_sight_to_raw_slot", True)) for r in rows], dtype=bool)
    los_proxy = np.asarray([bool(r.get("line_of_sight_to_proxy_slot", True)) for r in rows], dtype=bool)
    sub_visible = np.asarray([bool(r.get("subgoal_visible", True)) for r in rows], dtype=bool)
    accel_clip = np.asarray([bool(r.get("accel_clip_applied", False)) for r in rows], dtype=bool)
    speed_clip = np.asarray([bool(r.get("speed_clip_applied", False)) for r in rows], dtype=bool)

    if nominal_safe.size and float(np.mean(nominal_safe)) > 0.75 and float(np.mean(planner_active)) > 0.25:
        return "PLANNER_OVER_INTERVENTION"
    call_steps = np.where(planner_called)[0]
    min_interval = int(cfg.get("controllers", {}).get("global_path_pure_pursuit", {}).get("min_replan_interval_steps", 8))
    if call_steps.size > 4 and float(np.mean(np.diff(call_steps))) < max(float(min_interval), 2.0):
        return "REPLAN_CHATTER"
    finite_sub = sub_shift[np.isfinite(sub_shift)]
    finite_proxy = proxy_shift[np.isfinite(proxy_shift)]
    if finite_sub.size and float(np.mean(finite_sub > 0.5)) > 0.08 and (not finite_proxy.size or float(np.mean(finite_proxy > 0.5)) < 0.03):
        return "SUBGOAL_JITTER"
    if finite_proxy.size and float(np.mean(finite_proxy > 0.5)) > 0.05:
        return "PROXY_SLOT_JITTER"
    finite_idx = progress_idx[np.isfinite(progress_idx)]
    if finite_idx.size > 2 and int(np.sum(np.diff(finite_idx) < -0.5)) >= 2:
        return "PATH_PROGRESS_RESET"
    if (los_raw.size and float(np.mean(~los_raw & los_proxy)) > 0.1) or (sub_visible.size and float(np.mean(~sub_visible)) > 0.15):
        return "WRONG_TARGET_MODE"
    if float(np.mean(accel_clip | speed_clip)) > 0.25 and float(metrics.get("p95_tracking_error_to_subgoal", 0.0)) > 1.0:
        return "DYNAMICS_LAG"
    return "PLANNER_OVER_INTERVENTION" if float(metrics.get("planner_called_ratio", 0.0)) > 0.25 else "DYNAMICS_LAG"


def _finite_mean_local(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _initial_boundary_state(
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    world_xy: float,
    uav_radius: float,
    safety_margin: float,
    amax: float,
) -> tuple[bool, float, float]:
    margins: list[float] = []
    outward: list[float] = []
    invalid = False
    for i in range(np.asarray(pos).reshape(-1, 3).shape[0]):
        diag = _nearest_boundary_diag(
            np.asarray(pos, dtype=np.float64).reshape(-1, 3)[i, :2],
            np.asarray(vel, dtype=np.float64).reshape(-1, 3)[i, :2],
            np.zeros(2, dtype=np.float64),
            world_xy=world_xy,
            uav_radius=uav_radius,
            amax=amax,
        )
        margin = float(diag["distance_to_boundary"] - uav_radius)
        v_out = float(diag["velocity_outward_projection"])
        margins.append(margin)
        outward.append(v_out)
        if margin < 0.0 or (margin < safety_margin and v_out > 1e-6):
            invalid = True
    return bool(invalid), float(min(margins) if margins else float("inf")), float(max(outward) if outward else 0.0)


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["controller", "controller_base", "target_source", "shared_predictive_safety", "scenario", "scenario_group", "obstacle_type", "speed_level", "noise_level"]
    numeric = [
        "success",
        "rmse_error",
        "p95_error",
        "final_error",
        "steady_state_error",
        "slot_lost_ratio",
        "time_to_lock",
        "reacquisition_time",
        "path_length",
        "slot_path_length",
        "detour_ratio",
        "control_effort",
        "acceleration_norm_p95",
        "jerk_norm_p95",
        "mean_cos_to_goal",
        "mean_progress_projection",
        "speed_saturation_ratio",
        "acceleration_saturation_ratio",
        "double_clip_warning_ratio",
        "safe_rejection_success",
        "safe_stabilization_success",
        "number_of_jump_events",
        "jump_distance_mean",
        "jump_distance_p95",
        "jump_distance_max",
        "minimum_reach_time",
        "available_reacquisition_window",
        "pre_jump_error",
        "post_jump_max_error",
        "allowed_reacquisition_time",
        "allowed_reacquisition_time_p95",
        "reacquired_within_budget_ratio",
        "reacquired_within_budget_rate",
        "post_reacquisition_p95_error",
        "post_reacquisition_slot_lost_ratio",
        "full_episode_raw_slot_p95_error",
        "full_episode_commanded_slot_p95_error",
        "raw_slot_p95_error",
        "commanded_slot_p95_error",
        "commanded_slot_lag_to_raw",
        "commanded_slot_lag_to_raw_p95",
        "commanded_slot_lag_to_raw_max",
        "target_mode_switch_count",
        "slot_transition_mode_switch_count",
        "commanded_slot_jitter",
        "commanded_slot_jitter_p95",
        "planner_called_after_jump",
        "planner_success_after_jump",
        "selected_subgoal_after_jump",
        "subgoal_visible_after_jump",
        "path_valid_after_jump",
        "replan_count_after_jump",
        "invalid_commanded_slot_ratio",
        "raw_slot_invalid_ratio",
        "raw_slot_too_unstable_ratio",
        "safe_hold_active_ratio",
        "command_transition_blocked_ratio",
        "no_valid_proxy_slot_ratio",
        "commanded_slot_inside_obstacle_ratio",
        "commanded_slot_outside_boundary_ratio",
        "mean_v_obstacle_norm",
        "mean_v_boundary_norm",
        "used_existing_safety_modules_ratio",
        "mean_angle_nom_to_final",
        "mean_norm_ratio_final_to_nom",
        "mean_tangential_retention_ratio",
        "post_boundary_projection_ratio",
        "p5_obstacle_clearance",
        "episode_min_clearance",
        "step_clearance_p5",
        "predicted_next_clearance_min",
        "time_below_safety_margin",
        "min_predicted_next_clearance",
        "predictive_filter_active_ratio",
        "p95_error_pre_failure",
        "p95_error_full_episode",
        "target_mode_invariant_violation_count",
        "lookahead_segment_collision_count",
        "planned_path_below_required_count",
        "actual_slot_speed_mean",
        "actual_slot_speed_p95",
        "actual_slot_speed_max",
        "slot_out_of_bounds_ratio",
        "slot_inside_obstacle_ratio",
        "slot_min_obstacle_clearance",
        "slot_obstacle_too_close_ratio",
        "line_of_sight_blocked_ratio",
        "planner_called_ratio",
        "planner_mode_active_ratio",
        "planner_success_ratio",
        "planner_replan_count",
        "mean_replan_interval",
        "planner_path_length_mean",
        "subgoal_shift_norm_mean",
        "subgoal_shift_norm_p95",
        "proxy_slot_shift_norm_mean",
        "proxy_slot_shift_norm_p95",
        "line_of_sight_shortcut_ratio",
        "stuck_ratio",
        "stuck_detector_active_ratio",
        "mean_tracking_error_to_subgoal",
        "p95_tracking_error_to_subgoal",
        "mean_progress_to_subgoal",
        "mean_progress_to_raw_slot",
        "mean_cos_to_subgoal",
        "mean_cos_to_raw_slot",
        "mean_progress_to_commanded_slot",
        "mean_cos_to_commanded_slot",
        "proxy_slot_adjusted_ratio",
        "raw_slot_inside_obstacle_runtime_ratio",
        "raw_slot_too_close_obstacle_runtime_ratio",
        "p95_error_to_raw_slot",
        "slot_min_boundary_margin",
        "slot_feasible_ratio",
        "slot_too_close_ratio",
        "slot_unreachable_ratio",
        "invalid_initial_state",
        "obstacle_collision",
        "boundary_violation",
        "inter_agent_collision",
        "decision_time_ms_p95",
        "decision_time_ms_p99",
        "decision_time_p95",
        "decision_time_p99",
    ]
    work = df.copy()
    for col in ["success", "safe_rejection_success", "safe_stabilization_success", "obstacle_collision", "boundary_violation", "inter_agent_collision", "invalid_initial_state", "shared_predictive_safety"]:
        if col in work.columns:
            work[col] = work[col].astype(float)
    numeric = [col for col in numeric if col in work.columns]
    agg = work.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std", "median"]).reset_index()
    agg.columns = ["_".join([x for x in col if x]) for col in agg.columns.to_flat_index()]
    counts = work.groupby(group_cols, dropna=False).size().reset_index(name="episodes")
    return counts.merge(agg, on=group_cols, how="left")


def a_group_subscenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed A-group summary split by scenario, speed, feasibility, and controller."""
    work = df[df["scenario_group"] == "A"].copy()
    if work.empty:
        return pd.DataFrame()
    for col in ["success"]:
        work[col] = work[col].astype(float)
    group_cols = ["controller", "scenario", "speed_level", "feasible"]
    metric_cols = [
        "success",
        "rmse_error",
        "p95_error",
        "slot_lost_ratio",
        "time_to_lock",
        "mean_cos_to_goal",
        "mean_progress_projection",
        "speed_saturation_ratio",
        "acceleration_saturation_ratio",
    ]
    summary = work.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    summary = summary.rename(columns={"success": "success_rate"})
    failure_counts = (
        work.groupby(group_cols + ["failure_type"], dropna=False)
        .size()
        .reset_index(name="failure_count")
    )
    packed = (
        failure_counts.assign(part=lambda x: x["failure_type"].astype(str) + ":" + x["failure_count"].astype(str))
        .groupby(group_cols, dropna=False)["part"]
        .apply(lambda x: ";".join(x))
        .reset_index(name="failure_type_counts")
    )
    return summary.merge(packed, on=group_cols, how="left")


def feasible_only_summary(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Primary feasible-only success table grouped by controller/scenario/speed."""
    threshold = float(cfg.get("failure_classifier", {}).get("slot_out_of_bounds_threshold", 0.35))
    work = df[
        (df["failure_type"] != "TARGET_INFEASIBLE")
        & (df["invalid_initial_state"] == False)  # noqa: E712
        & (df["slot_out_of_bounds_ratio"] <= threshold)
        & (df["evaluation_feasible"] == True)  # noqa: E712
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work["success"] = work["success"].astype(float)
    return (
        work.groupby(["controller", "controller_base", "target_source", "shared_predictive_safety", "scenario", "speed_level"], dropna=False)
        .agg(
            episodes=("success", "size"),
            success_rate=("success", "mean"),
            rmse_error=("rmse_error", "mean"),
            p95_error=("p95_error", "mean"),
            slot_lost_ratio=("slot_lost_ratio", "mean"),
            time_to_lock=("time_to_lock", "mean"),
            actual_slot_speed_p95=("actual_slot_speed_p95", "mean"),
        )
        .reset_index()
    )


def a_group_failure_counts(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["scenario_group"] == "A"].copy()
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby(["controller", "scenario", "speed_level", "feasible", "failure_type"], dropna=False)
        .size()
        .reset_index(name="count")
    )


def infeasible_stress_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df[(df["feasible"] == False) | (df["failure_type"] == "TARGET_INFEASIBLE")].copy()  # noqa: E712
    if work.empty:
        return pd.DataFrame()
    work["avoids_obstacle_collision"] = ~work["obstacle_collision"].astype(bool)
    work["avoids_boundary_violation"] = ~work["boundary_violation"].astype(bool)
    if "safe_rejection_success" in work.columns:
        work["safe_rejection_success"] = work["safe_rejection_success"].astype(float)
    return (
        work.groupby(["controller", "controller_base", "target_source", "shared_predictive_safety", "scenario_group", "scenario", "speed_level", "infeasible_reason"], dropna=False)
        .agg(
            episodes=("success", "size"),
            success_rate=("success", "mean"),
            safe_rejection_success_rate=("safe_rejection_success", "mean"),
            avoids_obstacle_collision_rate=("avoids_obstacle_collision", "mean"),
            avoids_boundary_violation_rate=("avoids_boundary_violation", "mean"),
            rmse_error=("rmse_error", "mean"),
            p95_error=("p95_error", "mean"),
            slot_lost_ratio=("slot_lost_ratio", "mean"),
            actual_slot_speed_p95=("actual_slot_speed_p95", "mean"),
            slot_out_of_bounds_ratio=("slot_out_of_bounds_ratio", "mean"),
        )
        .reset_index()
    )


def boundary_failure_subtype_counts(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["failure_subtype"].astype(str).str.startswith("BOUNDARY_")].copy()
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby(["controller", "scenario_group", "scenario", "failure_subtype"], dropna=False)
        .size()
        .reset_index(name="count")
    )


def c_group_obstacle_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["scenario_group"] == "C"].copy()
    if work.empty:
        return pd.DataFrame()
    for col in ["success", "obstacle_collision", "boundary_violation"]:
        work[col] = work[col].astype(float)
    return (
        work.groupby(["controller", "scenario", "speed_level", "evaluation_feasible"], dropna=False)
        .agg(
            episodes=("success", "size"),
            success_rate=("success", "mean"),
            obstacle_collision_rate=("obstacle_collision", "mean"),
            boundary_violation_rate=("boundary_violation", "mean"),
            episode_min_clearance_mean=("episode_min_clearance", "mean"),
            episode_min_clearance_p5=("episode_min_clearance", lambda x: float(np.percentile(np.asarray(x, dtype=np.float64), 5))),
            step_clearance_p5=("step_clearance_p5", "mean"),
            time_below_safety_margin=("time_below_safety_margin", "mean"),
            predicted_next_clearance_min=("predicted_next_clearance_min", "mean"),
            predictive_filter_active_ratio=("predictive_filter_active_ratio", "mean"),
            p95_error_pre_failure=("p95_error_pre_failure", "mean"),
            p95_error_full_episode=("p95_error_full_episode", "mean"),
            target_mode_invariant_violation_count=("target_mode_invariant_violation_count", "sum"),
            lookahead_segment_collision_count=("lookahead_segment_collision_count", "sum"),
            planned_path_below_required_count=("planned_path_below_required_count", "sum"),
            p95_error=("p95_error", "mean"),
            slot_lost_ratio=("slot_lost_ratio", "mean"),
            stuck_ratio=("stuck_ratio", "mean"),
            stuck_detector_active_ratio=("stuck_detector_active_ratio", "mean"),
            line_of_sight_blocked_ratio=("line_of_sight_blocked_ratio", "mean"),
            planner_called_ratio=("planner_called_ratio", "mean"),
            planner_mode_active_ratio=("planner_mode_active_ratio", "mean"),
            planner_success_ratio=("planner_success_ratio", "mean"),
            planner_replan_count=("planner_replan_count", "mean"),
            mean_replan_interval=("mean_replan_interval", "mean"),
            subgoal_shift_norm_mean=("subgoal_shift_norm_mean", "mean"),
            subgoal_shift_norm_p95=("subgoal_shift_norm_p95", "mean"),
            proxy_slot_shift_norm_mean=("proxy_slot_shift_norm_mean", "mean"),
            proxy_slot_shift_norm_p95=("proxy_slot_shift_norm_p95", "mean"),
            tracking_error_to_subgoal=("p95_tracking_error_to_subgoal", "mean"),
            planner_path_length_mean=("planner_path_length_mean", "mean"),
            line_of_sight_shortcut_ratio=("line_of_sight_shortcut_ratio", "mean"),
            decision_time_ms_p95=("decision_time_ms_p95", "median"),
            decision_time_ms_p99=("decision_time_ms_p99", "median"),
        )
        .reset_index()
    )


def d_group_jump_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["scenario_group"] == "D"].copy()
    if work.empty:
        return pd.DataFrame()
    for col in ["success", "safe_rejection_success", "safe_stabilization_success", "reacquired_within_budget", "double_clip_warning_ratio", "obstacle_collision", "boundary_violation"]:
        if col in work.columns:
            work[col] = work[col].astype(float)
    metric_cols = [
        "success",
        "safe_rejection_success",
        "safe_stabilization_success",
        "obstacle_collision",
        "boundary_violation",
        "number_of_jump_events",
        "jump_distance_mean",
        "jump_distance_p95",
        "jump_distance_max",
        "reacquisition_time",
        "reacquisition_time_p95",
        "allowed_reacquisition_time",
        "allowed_reacquisition_time_p95",
        "reacquired_within_budget",
        "reacquired_within_budget_rate",
        "full_episode_raw_slot_p95_error",
        "full_episode_commanded_slot_p95_error",
        "raw_slot_p95_error",
        "commanded_slot_p95_error",
        "post_reacquisition_p95_error",
        "post_reacquisition_slot_lost_ratio",
        "commanded_slot_lag_to_raw",
        "commanded_slot_lag_to_raw_p95",
        "target_mode_switch_count",
        "slot_transition_mode_switch_count",
        "commanded_slot_jitter",
        "planner_called_after_jump",
        "planner_success_after_jump",
        "selected_subgoal_after_jump",
        "subgoal_visible_after_jump",
        "path_valid_after_jump",
        "replan_count_after_jump",
        "double_clip_warning_ratio",
        "invalid_commanded_slot_ratio",
        "raw_slot_invalid_ratio",
        "raw_slot_too_unstable_ratio",
        "safe_hold_active_ratio",
        "command_transition_blocked_ratio",
        "no_valid_proxy_slot_ratio",
        "commanded_slot_inside_obstacle_ratio",
        "commanded_slot_outside_boundary_ratio",
        "target_mode_invariant_violation_count",
        "decision_time_p95",
        "decision_time_p99",
    ]
    metric_cols = [c for c in metric_cols if c in work.columns]
    summary = (
        work.groupby(["controller", "controller_base", "target_source", "shared_predictive_safety", "scenario", "speed_level", "evaluation_feasible"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    if "success" in summary.columns:
        summary = summary.rename(columns={"success": "success_rate"})
    summary = summary.rename(
        columns={
            "safe_rejection_success": "safe_rejection_success_rate",
            "safe_stabilization_success": "safe_stabilization_success_rate",
            "obstacle_collision": "collision_rate",
            "boundary_violation": "boundary_violation_rate",
        }
    )
    failures = (
        work.groupby(["controller", "scenario", "failure_type"], dropna=False)
        .size()
        .reset_index(name="failure_count")
    )
    packed = (
        failures.assign(part=lambda x: x["failure_type"].astype(str) + ":" + x["failure_count"].astype(str))
        .groupby(["controller", "scenario"], dropna=False)["part"]
        .apply(lambda x: ";".join(x))
        .reset_index(name="failure_type_counts")
    )
    return summary.merge(packed, on=["controller", "scenario"], how="left")


def generate_existing_debug_plots(df: pd.DataFrame, output_dir: Path, *, max_plots: int = 20) -> None:
    """Generate focused plots for failed no-obstacle existing-controller episodes."""
    failed = df[
        (df["controller"] == "existing")
        & (df["scenario_group"] == "A")
        & (df["success"] == False)  # noqa: E712
        & (df["raw_path"].astype(str) != "")
    ].copy()
    if failed.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping existing debug plots: matplotlib import failed: {exc}")
        return

    debug_dir = output_dir / "figures" / "debug_existing"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for _, episode in failed.sort_values("p95_error", ascending=False).head(max(int(max_plots), 0)).iterrows():
        raw_path = Path(str(episode["raw_path"]))
        if not raw_path.exists():
            continue
        raw = pd.read_csv(raw_path)
        stem = raw_path.stem
        plt.figure(figsize=(7, 7))
        plt.plot(raw["uav_x"], raw["uav_y"], label="uav")
        plt.plot(raw["slot_x"], raw["slot_y"], "--", label="slot")
        plt.scatter(raw["uav_x"].iloc[0], raw["uav_y"].iloc[0], marker="o", label="uav start")
        plt.scatter(raw["uav_x"].iloc[-1], raw["uav_y"].iloc[-1], marker="x", label="uav end")
        plt.scatter(raw["slot_x"].iloc[0], raw["slot_y"].iloc[0], marker="s", label="slot start")
        plt.scatter(raw["slot_x"].iloc[-1], raw["slot_y"].iloc[-1], marker="*", label="slot end")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(debug_dir / f"{stem}_trajectory.png", dpi=160)
        plt.close()

        _line_plot(raw, ["tracking_error"], debug_dir / f"{stem}_tracking_error.png", "tracking error [m]")
        _line_plot(raw, ["cos_to_goal"], debug_dir / f"{stem}_cos_to_goal.png", "cos_to_goal")
        _line_plot(raw, ["progress_projection"], debug_dir / f"{stem}_progress_projection.png", "progress projection [m/s]")

        norm_df = pd.DataFrame({"t": raw["t"]})
        for prefix in ["v_goal", "v_obstacle", "v_boundary", "v_path", "v_smooth", "v_final_after_clip"]:
            x_col = f"{prefix}_x"
            y_col = f"{prefix}_y"
            if x_col in raw.columns and y_col in raw.columns:
                norm_df[f"norm({prefix})"] = np.hypot(raw[x_col], raw[y_col])
        _line_plot(norm_df, [c for c in norm_df.columns if c != "t"], debug_dir / f"{stem}_velocity_component_norms.png", "velocity norm [m/s]")
        _line_plot(
            raw,
            ["clip_flag", "speed_saturation_flag", "acceleration_saturation_flag", "double_clip_warning"],
            debug_dir / f"{stem}_saturation_flags.png",
            "flag",
        )


def generate_boundary_debug_plots(df: pd.DataFrame, output_dir: Path, cfg: dict[str, Any], *, max_plots: int = 20) -> None:
    """Generate boundary-focused plots for failed feasible existing episodes."""
    failed = df[
        (df["controller"] == "existing")
        & (df["scenario"].isin(["boundary_corner_turn", "boundary_parallel_slot"]))
        & (df["evaluation_feasible"] == True)  # noqa: E712
        & (df["success"] == False)  # noqa: E712
        & (df["raw_path"].astype(str) != "")
    ].copy()
    if failed.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping boundary debug plots: matplotlib import failed: {exc}")
        return

    debug_dir = output_dir / "figures" / "debug_boundary"
    debug_dir.mkdir(parents=True, exist_ok=True)
    world_xy = float(cfg["world"]["world_xy"])
    for _, episode in failed.sort_values("p95_error", ascending=False).head(max(int(max_plots), 0)).iterrows():
        raw_path = Path(str(episode["raw_path"]))
        if not raw_path.exists():
            continue
        raw = pd.read_csv(raw_path)
        stem = raw_path.stem

        plt.figure(figsize=(7, 7))
        plt.plot(raw["uav_x"], raw["uav_y"], label="uav")
        plt.plot(raw["slot_x"], raw["slot_y"], "--", label="slot")
        plt.plot(
            [-world_xy, world_xy, world_xy, -world_xy, -world_xy],
            [-world_xy, -world_xy, world_xy, world_xy, -world_xy],
            color="black",
            linewidth=1.0,
            label="boundary",
        )
        plt.scatter(raw["uav_x"].iloc[0], raw["uav_y"].iloc[0], marker="o", label="uav start")
        plt.scatter(raw["uav_x"].iloc[-1], raw["uav_y"].iloc[-1], marker="x", label="uav end")
        plt.scatter(raw["slot_x"].iloc[0], raw["slot_y"].iloc[0], marker="s", label="slot start")
        plt.scatter(raw["slot_x"].iloc[-1], raw["slot_y"].iloc[-1], marker="*", label="slot end")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(debug_dir / f"{stem}_trajectory_boundary.png", dpi=160)
        plt.close()

        _line_plot(raw, ["tracking_error"], debug_dir / f"{stem}_tracking_error.png", "tracking error [m]")
        _line_plot(
            raw,
            [f"{b}_action_outward_projection_after" for b in ["x_min", "x_max", "y_min", "y_max"]],
            debug_dir / f"{stem}_action_outward_projection.png",
            "action outward projection [m/s]",
        )
        _line_plot(
            raw,
            [f"{b}_velocity_outward_projection" for b in ["x_min", "x_max", "y_min", "y_max"]],
            debug_dir / f"{stem}_velocity_outward_projection.png",
            "velocity outward projection [m/s]",
        )
        _line_plot(
            raw,
            [f"{b}_braking_margin" for b in ["x_min", "x_max", "y_min", "y_max"]],
            debug_dir / f"{stem}_braking_margin.png",
            "braking margin [m]",
        )
        _line_plot(
            raw,
            [f"{b}_tangential_norm_before" for b in ["x_min", "x_max", "y_min", "y_max"]]
            + [f"{b}_tangential_norm_after" for b in ["x_min", "x_max", "y_min", "y_max"]],
            debug_dir / f"{stem}_tangential_norms.png",
            "tangential norm [m/s]",
        )
        _line_plot(
            raw,
            [f"{b}_normal_component_before" for b in ["x_min", "x_max", "y_min", "y_max"]]
            + [f"{b}_normal_component_after" for b in ["x_min", "x_max", "y_min", "y_max"]],
            debug_dir / f"{stem}_normal_components.png",
            "normal component [m/s]",
        )
        _line_plot(
            raw,
            [
                "action_before_boundary_filter_x",
                "action_before_boundary_filter_y",
                "action_after_boundary_filter_x",
                "action_after_boundary_filter_y",
            ],
            debug_dir / f"{stem}_cmd_before_after_boundary_filter.png",
            "command [m/s]",
        )


def generate_obstacle_debug_plots(df: pd.DataFrame, output_dir: Path, cfg: dict[str, Any], *, max_plots: int = 20) -> None:
    """Generate obstacle-focused plots for failed existing C-group episodes."""
    failed = df[
        (df["controller"].isin(["existing", "global_path_pure_pursuit"]))
        & (df["scenario_group"] == "C")
        & (df["success"] == False)  # noqa: E712
        & (df["raw_path"].astype(str) != "")
    ].copy()
    if failed.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping obstacle debug plots: matplotlib import failed: {exc}")
        return

    debug_dir = output_dir / "figures" / "debug_obstacles"
    debug_dir.mkdir(parents=True, exist_ok=True)
    safety_margin = float(cfg["dynamics"]["safety_margin"])
    uav_radius = float(cfg["dynamics"]["uav_radius"])
    for _, episode in failed.sort_values("p95_error", ascending=False).head(max(int(max_plots), 0)).iterrows():
        raw_path = Path(str(episode["raw_path"]))
        if not raw_path.exists():
            continue
        raw = pd.read_csv(raw_path)
        stem = raw_path.stem
        obstacles = _parse_obstacle_records(str(episode.get("obstacles", "[]")))

        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        for obs in obstacles:
            x = float(obs.get("x", 0.0))
            y = float(obs.get("y", 0.0))
            r = float(obs.get("radius", 0.0))
            ax.add_patch(plt.Circle((x, y), r, color="tab:red", alpha=0.25, label="obstacle" if obs is obstacles[0] else None))
            ax.add_patch(plt.Circle((x, y), r + uav_radius + safety_margin, color="tab:red", fill=False, alpha=0.35, linestyle=":"))
        plt.plot(raw["uav_x"], raw["uav_y"], label="uav")
        raw_x = "raw_slot_x" if "raw_slot_x" in raw.columns else "slot_x"
        raw_y = "raw_slot_y" if "raw_slot_y" in raw.columns else "slot_y"
        proxy_x = "proxy_slot_x" if "proxy_slot_x" in raw.columns else "slot_x"
        proxy_y = "proxy_slot_y" if "proxy_slot_y" in raw.columns else "slot_y"
        plt.plot(raw[raw_x], raw[raw_y], "--", label="raw slot")
        plt.plot(raw[proxy_x], raw[proxy_y], ":", label="proxy slot")
        planned = _first_nonempty_path(raw.get("planned_path"))
        if planned is not None and planned.shape[0] > 1:
            plt.plot(planned[:, 0], planned[:, 1], "-.", color="tab:purple", label="planned path")
        if {"current_subgoal_x", "current_subgoal_y"}.issubset(set(raw.columns)):
            plt.plot(raw["current_subgoal_x"], raw["current_subgoal_y"], ".", markersize=2, label="selected subgoals")
        collision_rows = raw[raw.get("obstacle_collision", False).astype(bool)] if "obstacle_collision" in raw.columns else pd.DataFrame()
        if not collision_rows.empty:
            plt.scatter(
                collision_rows["uav_x"].iloc[0],
                collision_rows["uav_y"].iloc[0],
                marker="X",
                s=90,
                color="black",
                label="collision point",
            )
        plt.scatter(raw["uav_x"].iloc[0], raw["uav_y"].iloc[0], marker="o", label="uav start")
        plt.scatter(raw["uav_x"].iloc[-1], raw["uav_y"].iloc[-1], marker="x", label="uav end")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(debug_dir / f"{stem}_trajectory_obstacles.png", dpi=160)
        plt.close()

        _line_plot(raw, ["tracking_error", "tracking_error_to_raw_slot", "tracking_error_to_proxy_slot", "tracking_error_to_subgoal"], debug_dir / f"{stem}_tracking_error.png", "tracking error [m]")
        _line_plot(raw, ["nearest_obstacle_distance"], debug_dir / f"{stem}_nearest_obstacle_distance.png", "clearance [m]")
        norm_df = pd.DataFrame({"t": raw["t"]})
        for prefix in ["v_nom", "v_after_boundary_projection", "v_after_obstacle_filter", "v_after_planner", "v_final", "v_obstacle"]:
            x_col = f"{prefix}_x"
            y_col = f"{prefix}_y"
            if x_col in raw.columns and y_col in raw.columns:
                norm_df[f"norm({prefix})"] = np.hypot(raw[x_col], raw[y_col])
        _line_plot(norm_df, [c for c in norm_df.columns if c != "t"], debug_dir / f"{stem}_velocity_component_norms.png", "velocity norm [m/s]")
        _line_plot(raw, ["selected_subgoal_visible", "subgoal_visible", "subgoal_backtracked"], debug_dir / f"{stem}_subgoal_visibility.png", "flag")
        if "replan_reason" in raw.columns:
            reason_codes = {name: idx for idx, name in enumerate(sorted(set(raw["replan_reason"].astype(str))))}
            reason_df = pd.DataFrame({"t": raw["t"], "replan_reason_code": raw["replan_reason"].astype(str).map(reason_codes).astype(float)})
            _line_plot(reason_df, ["replan_reason_code"], debug_dir / f"{stem}_replan_reason_codes.png", "code")
        _line_plot(raw, ["planner_mode_active", "planner_called", "planner_success", "line_of_sight_blocked", "path_blocked", "stuck_detector_active"], debug_dir / f"{stem}_planner_los_flags.png", "flag")
        _line_plot(raw, ["current_path_progress_index", "current_path_arclength_progress"], debug_dir / f"{stem}_path_progress.png", "progress")
        _line_plot(raw, ["subgoal_shift_norm", "proxy_slot_shift_norm"], debug_dir / f"{stem}_target_shift_norms.png", "shift [m]")
        _line_plot(raw, ["progress_to_subgoal", "progress_to_raw_slot"], debug_dir / f"{stem}_progress_to_targets.png", "progress [m/s]")
        _line_plot(raw, ["cos_to_subgoal", "cos_to_raw_slot", "cos_to_goal"], debug_dir / f"{stem}_cos_to_targets.png", "cos")
        pred_norm = pd.DataFrame({"t": raw["t"]})
        for prefix in ["v_final", "v_after_predictive_safety"]:
            x_col = f"{prefix}_x"
            y_col = f"{prefix}_y"
            if x_col in raw.columns and y_col in raw.columns:
                pred_norm[f"norm({prefix})"] = np.hypot(raw[x_col], raw[y_col])
        _line_plot(pred_norm, [c for c in pred_norm.columns if c != "t"], debug_dir / f"{stem}_predictive_velocity_norms.png", "velocity norm [m/s]")
        _line_plot(raw, ["segment_collision_current_to_next", "predictive_filter_active"], debug_dir / f"{stem}_predicted_next_collision.png", "flag")

        region = raw.copy()
        region["collision_region"] = region["nearest_obstacle_distance"].astype(float) < 0.0
        region["safety_margin_region"] = region["nearest_obstacle_distance"].astype(float) < safety_margin
        _line_plot(region, ["collision_region", "safety_margin_region", "stuck_detector_active"], debug_dir / f"{stem}_failure_regions.png", "flag")


def generate_d_group_debug_plots(df: pd.DataFrame, output_dir: Path, *, max_plots: int = 20) -> None:
    failed = df[
        (df["scenario_group"] == "D")
        & (df["success"] == False)  # noqa: E712
        & (df["raw_path"].astype(str) != "")
    ].copy()
    if failed.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping D debug plots: matplotlib import failed: {exc}")
        return

    debug_dir = output_dir / "figures" / "debug_d_group"
    debug_dir.mkdir(parents=True, exist_ok=True)
    sort_col = "full_episode_commanded_slot_p95_error" if "full_episode_commanded_slot_p95_error" in failed.columns else "p95_error"
    for _, episode in failed.sort_values(sort_col, ascending=False).head(max(int(max_plots), 0)).iterrows():
        raw_path = Path(str(episode["raw_path"]))
        if not raw_path.exists():
            continue
        raw = pd.read_csv(raw_path)
        stem = raw_path.stem
        plt.figure(figsize=(7, 7))
        plt.plot(raw["raw_slot_x"], raw["raw_slot_y"], ":", label="raw slot")
        plt.plot(raw["commanded_slot_x"], raw["commanded_slot_y"], "--", label="commanded slot")
        plt.plot(raw["uav_x"], raw["uav_y"], label="uav")
        jumps = raw[raw["jump_detected"].astype(bool)] if "jump_detected" in raw.columns else pd.DataFrame()
        if not jumps.empty:
            plt.scatter(jumps["raw_slot_x"], jumps["raw_slot_y"], marker="x", label="jump")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(debug_dir / f"{stem}_d_paths.png", dpi=160)
        plt.close()

        _line_plot(raw, ["tracking_error_to_raw_slot", "tracking_error", "tracking_error_to_proxy_slot"], debug_dir / f"{stem}_d_tracking_errors.png", "tracking error [m]")
        _line_plot(raw, ["jump_detected", "transition_progress"], debug_dir / f"{stem}_d_jump_windows.png", "jump/window")
        _line_plot(raw, ["commanded_slot_lag_to_raw"], debug_dir / f"{stem}_d_commanded_lag.png", "lag [m]")
        for col in ["target_mode", "slot_transition_mode"]:
            if col in raw.columns:
                codes = {name: idx for idx, name in enumerate(sorted(set(raw[col].astype(str))))}
                mode_df = pd.DataFrame({"t": raw["t"], f"{col}_code": raw[col].astype(str).map(codes).astype(float)})
                _line_plot(mode_df, [f"{col}_code"], debug_dir / f"{stem}_d_{col}.png", "mode code")
        vel_df = pd.DataFrame({"t": raw["t"]})
        for prefix in ["desired_velocity_before_final_clip", "final_velocity_after_clip", "v_final_after_clip"]:
            x_col = f"{prefix}_x"
            y_col = f"{prefix}_y"
            if x_col in raw.columns and y_col in raw.columns:
                vel_df[f"norm({prefix})"] = np.hypot(raw[x_col], raw[y_col])
        _line_plot(vel_df, [c for c in vel_df.columns if c != "t"], debug_dir / f"{stem}_d_velocity_norms.png", "velocity norm [m/s]")
        _line_plot(raw, ["clip_flag", "speed_saturation_flag", "acceleration_saturation_flag", "double_clip_flag"], debug_dir / f"{stem}_d_clip_flags.png", "flag")


def _parse_obstacle_records(value: str) -> list[dict[str, float]]:
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, float]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            out.append({"x": float(item.get("x", 0.0)), "y": float(item.get("y", 0.0)), "radius": float(item.get("radius", 0.0))})
        except Exception:
            continue
    return out


def _first_nonempty_path(series: Any) -> np.ndarray | None:
    if series is None:
        return None
    try:
        values = list(series)
    except TypeError:
        values = [series]
    for value in values:
        if value is None or str(value).strip() in ("", "nan"):
            continue
        try:
            parsed = ast.literal_eval(str(value))
        except Exception:
            continue
        arr = np.asarray(parsed, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
            return arr[:, :2]
    return None


def _line_plot(df: pd.DataFrame, columns: list[str], output: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    plt.figure(figsize=(9, 4))
    for col in cols:
        plt.plot(df["t"], df[col].astype(float), label=col)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def print_terminal_report(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    print("\n=== Slot Tracking Benchmark Report ===")
    print("Overall success rate per controller:")
    rates = df.groupby("controller")["success"].mean().sort_values(ascending=False)
    for name, rate in rates.items():
        print(f"  {name}: {100.0 * float(rate):.1f}%")

    feasible = df[df["evaluation_feasible"] == True].copy()  # noqa: E712
    print("\nFeasible-only success rate per controller:")
    if feasible.empty:
        print("  none")
    else:
        frates = feasible.groupby("controller")["success"].mean().sort_values(ascending=False)
        for name, rate in frates.items():
            print(f"  {name}: {100.0 * float(rate):.1f}%")

    print("\nController target/safety labeling:")
    label_cols = [c for c in ["controller", "controller_base", "target_source", "shared_predictive_safety"] if c in df.columns]
    if len(label_cols) >= 4:
        labels = df[label_cols].drop_duplicates().sort_values("controller")
        for _, row in labels.iterrows():
            print(
                f"  {row['controller']}: base={row['controller_base']} "
                f"target={row['target_source']} shared_safety={bool(row['shared_predictive_safety'])}"
            )

    print("\nSuccess rate per scenario and speed level:")
    if feasible.empty:
        print("  none")
    else:
        grouped = feasible.groupby(["scenario", "speed_level", "controller"])["success"].mean().reset_index()
        for _, row in grouped.sort_values(["scenario", "speed_level", "controller"]).iterrows():
            print(f"  {row['scenario']} speed={float(row['speed_level']):.2f} {row['controller']}: {100.0 * float(row['success']):.1f}%")

    print("\nTARGET_INFEASIBLE episodes by scenario:")
    target_counts = df[df["failure_type"] == "TARGET_INFEASIBLE"].groupby("scenario").size()
    if target_counts.empty:
        print("  none")
    else:
        for scenario, count in target_counts.items():
            print(f"  {scenario}: {int(count)}")

    d = df[df["scenario_group"] == "D"].copy()
    if not d.empty:
        print("\nD-group jump/deformation report:")
        feasible_d = d[d["evaluation_feasible"] == True].copy()  # noqa: E712
        if feasible_d.empty:
            print("  feasible-only success: none")
        else:
            for name, rate in feasible_d.groupby("controller")["success"].mean().sort_values(ascending=False).items():
                print(f"  feasible-only {name}: {100.0 * float(rate):.1f}%")
        print("  success by D scenario:")
        by_scenario = d.groupby("scenario")["success"].mean().sort_index()
        for scenario, rate in by_scenario.items():
            print(f"    {scenario}: {100.0 * float(rate):.1f}%")
        print("  TARGET_INFEASIBLE by D scenario:")
        d_target = d[d["failure_type"] == "TARGET_INFEASIBLE"].groupby("scenario").size()
        if d_target.empty:
            print("    none")
        else:
            for scenario, count in d_target.items():
                print(f"    {scenario}: {int(count)}")
        smooth = d[d["scenario"].astype(str).str.startswith("smooth_slot_deformation")]
        if not smooth.empty:
            smooth_bad = smooth[
                (smooth.get("number_of_generator_jump_events", pd.Series(0, index=smooth.index)).fillna(0).astype(float) > 0)
                | (smooth["slot_out_of_bounds_ratio"].fillna(0.0).astype(float) > 0.0)
                | (smooth["actual_slot_speed_p95"].fillna(0.0).astype(float) > float(cfg.get("failure_classifier", {}).get("smooth_speed_limit", cfg.get("dynamics", {}).get("uav_vmax", 1.0))))
            ]
            print(f"  smooth_slot_deformation generator warnings: {int(smooth_bad.shape[0])}")
        jump = d[d.get("number_of_jump_events", pd.Series(0, index=d.index)).fillna(0).astype(float) > 0].copy()
        if not jump.empty:
            print("  reacquisition by jump scenario:")
            grouped = jump.groupby("scenario").agg(
                reacquisition_time_mean=("reacquisition_time", "mean"),
                reacquisition_time_p95=("reacquisition_time_p95", "mean"),
                allowed_reacquisition_time_mean=("allowed_reacquisition_time", "mean"),
                reacquired_within_budget=("reacquired_within_budget_ratio", "mean"),
                raw_slot_p95_error=("full_episode_raw_slot_p95_error", "mean"),
                commanded_slot_p95_error=("full_episode_commanded_slot_p95_error", "mean"),
                commanded_slot_lag_to_raw=("commanded_slot_lag_to_raw", "mean"),
                double_clip_rate=("double_clip_warning_ratio", "mean"),
            )
            for scenario, row in grouped.sort_index().iterrows():
                print(
                    f"    {scenario}: reacq_mean={float(row['reacquisition_time_mean']):.3f}s "
                    f"reacq_p95={float(row['reacquisition_time_p95']):.3f}s "
                    f"allowed_mean={float(row['allowed_reacquisition_time_mean']):.3f}s "
                    f"within_budget={float(row['reacquired_within_budget']):.3f} "
                    f"raw_p95={float(row['raw_slot_p95_error']):.3f} "
                    f"cmd_p95={float(row['commanded_slot_p95_error']):.3f} "
                    f"lag={float(row['commanded_slot_lag_to_raw']):.3f} "
                    f"double_clip={float(row['double_clip_rate']):.4f}"
                )
        print("  D failure type counts:")
        d_failures = d["failure_type"].value_counts()
        for failure_type, count in d_failures.items():
            print(f"    {failure_type}: {int(count)}")

    static_nonzero = int(
        df[(df["scenario"] == "static_slot") & (df["actual_slot_speed_p95"].fillna(0.0) > float(cfg.get("failure_classifier", {}).get("static_slot_speed_epsilon", 1e-6)))].shape[0]
    )
    print(f"\nStatic-slot episodes with nonzero actual speed: {static_nonzero}")

    print("\nBoundary failure subtype counts:")
    boundary_counts = df[df["failure_subtype"].astype(str).str.startswith("BOUNDARY_")]["failure_subtype"].value_counts()
    if boundary_counts.empty:
        print("  none")
    else:
        for subtype, count in boundary_counts.items():
            print(f"  {subtype}: {int(count)}")

    existing = df[df["controller"] == "existing"]
    if not existing.empty:
        print("\nWorst 5 scenarios for existing controller:")
        worst = existing.groupby("scenario")["success"].mean().sort_values().head(5)
        for scenario, rate in worst.items():
            print(f"  {scenario}: {100.0 * float(rate):.1f}% success")

    print("\nMost frequent failure types:")
    failures = df[df["failure_type"] != "SUCCESS"]["failure_type"].value_counts().head(8)
    if failures.empty:
        print("  none")
    else:
        for failure_type, count in failures.items():
            print(f"  {failure_type}: {int(count)}")

    a = df[df["scenario_group"] == "A"]
    if {"existing", "pure_pursuit"}.issubset(set(a["controller"])):
        rates_a = a.groupby("controller")["p95_error"].median()
        verdict = bool(rates_a.get("existing", np.inf) <= rates_a.get("pure_pursuit", np.inf) + 1e-9)
        print(f"\nExisting beats pure pursuit in no-obstacle tracking: {verdict}")
    c = df[df["scenario_group"] == "C"]
    if {"existing", "apf"}.issubset(set(c["controller"])):
        feasible_c = c[c["evaluation_feasible"] == True].copy()  # noqa: E712
        rates_c = feasible_c.groupby("controller")["success"].mean() if not feasible_c.empty else c.groupby("controller")["success"].mean()
        p95_c = feasible_c.groupby("controller")["p95_error"].median() if not feasible_c.empty else c.groupby("controller")["p95_error"].median()
        existing_rate = float(rates_c.get("existing", -np.inf))
        apf_rate = float(rates_c.get("apf", -np.inf))
        verdict = bool(existing_rate > apf_rate or (existing_rate >= apf_rate and float(p95_c.get("existing", np.inf)) <= float(p95_c.get("apf", np.inf))))
        print(f"Existing beats APF in feasible obstacle scenarios: {verdict}")

    print_c_group_report(df, cfg)

    rt = float(cfg["success"]["realtime_p95_ms"])
    p95 = df.groupby("controller")["decision_time_ms_p95"].median()
    print(f"Decision p95 below {rt:.2f} ms:")
    for name, value in p95.items():
        print(f"  {name}: {bool(float(value) <= rt)} ({float(value):.3f} ms)")
    print_terminal_diagnosis(df, cfg)
    print_boundary_corner_report(df)


def print_c_group_report(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    c = df[df["scenario_group"] == "C"].copy()
    if c.empty:
        return
    print("\n=== C-Group Obstacle Report ===")
    feasible = c[c["evaluation_feasible"] == True].copy()  # noqa: E712
    source = feasible if not feasible.empty else c

    print("Success rate by obstacle scenario:")
    grouped = source.groupby(["scenario", "controller"])["success"].mean().reset_index()
    for _, row in grouped.sort_values(["scenario", "controller"]).iterrows():
        print(f"  {row['scenario']} {row['controller']}: {100.0 * float(row['success']):.1f}%")

    print("Collision rate by controller:")
    collision = c.groupby("controller")["obstacle_collision"].mean().sort_index()
    for name, value in collision.items():
        print(f"  {name}: {100.0 * float(value):.1f}%")

    print("Clearance metric definitions:")
    print("  episode_min_clearance_mean: mean of each episode's minimum true obstacle clearance.")
    print("  episode_min_clearance_p5: 5th percentile across episode-level minimum true obstacle clearances.")
    print("  step_clearance_p5: mean per-episode 5th percentile over logged step clearances.")
    print("  predicted_next_clearance_min: mean per-episode minimum predicted next-step clearance.")
    print("  time_below_safety_margin: fraction of logged steps below configured safety margin.")

    print("Clearance metrics by controller:")
    for name, part in c.groupby("controller"):
        vals = np.asarray(part.get("episode_min_clearance", part["min_obstacle_clearance"]), dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            episode_mean = float(np.mean(vals))
            episode_p5 = float(np.percentile(vals, 5))
        else:
            episode_mean = float("nan")
            episode_p5 = float("nan")
        p5_vals = np.asarray(part.get("step_clearance_p5", part.get("p5_obstacle_clearance", pd.Series(dtype=float))), dtype=np.float64)
        below_vals = np.asarray(part.get("time_below_safety_margin", pd.Series(dtype=float)), dtype=np.float64)
        pred_vals = np.asarray(part.get("predicted_next_clearance_min", part.get("min_predicted_next_clearance", pd.Series(dtype=float))), dtype=np.float64)
        print(
            f"  {name}: "
            f"episode_min_clearance_mean={episode_mean:.3f}, "
            f"episode_min_clearance_p5={episode_p5:.3f}, "
            f"step_clearance_p5={_finite_mean_local(p5_vals):.3f}, "
            f"predicted_next_clearance_min={_finite_mean_local(pred_vals):.3f}, "
            f"time_below_safety_margin={_finite_mean_local(below_vals):.3f}"
        )

    collisions = c[c["obstacle_collision"].astype(bool)].copy()
    if not collisions.empty and "collision_root_cause" in collisions.columns:
        print("Collision root-cause labels:")
        for label, count in collisions["collision_root_cause"].value_counts().items():
            print(f"  {label}: {int(count)}")
    divergent = c[(c["p95_error"].astype(float) > 5.0) | (c["failure_type"].astype(str) == "TRACKING_DIVERGENCE")].copy()
    if not divergent.empty and "divergence_root_cause" in divergent.columns:
        print("Sparse divergence root-cause labels:")
        for label, count in divergent["divergence_root_cause"].value_counts().items():
            print(f"  {label}: {int(count)}")

    print("P95 tracking error and slot_lost_ratio by scenario:")
    scen = source.groupby("scenario")[["p95_error", "slot_lost_ratio"]].mean().reset_index()
    for _, row in scen.sort_values("scenario").iterrows():
        print(f"  {row['scenario']}: p95={float(row['p95_error']):.3f}, lost={float(row['slot_lost_ratio']):.3f}")

    print("Planner/subgoal diagnostics by controller:")
    diag_cols = [
        "p95_tracking_error_to_subgoal",
        "planner_called_ratio",
        "planner_mode_active_ratio",
        "planner_success_ratio",
        "planner_replan_count",
        "mean_replan_interval",
        "subgoal_shift_norm_p95",
        "proxy_slot_shift_norm_p95",
        "planner_path_length_mean",
        "stuck_detector_active_ratio",
    ]
    available_diag_cols = [col for col in diag_cols if col in c.columns]
    if available_diag_cols:
        planner_diag = c.groupby("controller")[available_diag_cols].mean().reset_index()
        for _, row in planner_diag.sort_values("controller").iterrows():
            print(
                f"  {row['controller']}: "
                f"subgoal_p95={float(row.get('p95_tracking_error_to_subgoal', np.nan)):.3f}, "
                f"called={float(row.get('planner_called_ratio', 0.0)):.3f}, "
                f"active={float(row.get('planner_mode_active_ratio', 0.0)):.3f}, "
                f"success={float(row.get('planner_success_ratio', 0.0)):.3f}, "
                f"replans={float(row.get('planner_replan_count', 0.0)):.2f}, "
                f"interval={float(row.get('mean_replan_interval', np.nan)):.2f}, "
                f"sub_shift_p95={float(row.get('subgoal_shift_norm_p95', np.nan)):.3f}, "
                f"proxy_shift_p95={float(row.get('proxy_slot_shift_norm_p95', np.nan)):.3f}, "
                f"path_len={float(row.get('planner_path_length_mean', np.nan)):.3f}, "
                f"stuck={float(row.get('stuck_detector_active_ratio', 0.0)):.3f}"
            )

    for failure_type in ["STUCK_NEAR_OBSTACLE", "NO_PROGRESS", "UNKNOWN_FAILURE"]:
        count = int((c["failure_type"] == failure_type).sum())
        print(f"{failure_type.lower()} count: {count}")

    p95 = c.groupby("controller")["decision_time_ms_p95"].median().sort_index()
    p99 = c.groupby("controller")["decision_time_ms_p99"].median().sort_index() if "decision_time_ms_p99" in c.columns else pd.Series(dtype=float)
    print("decision_time_p95/p99:")
    for name, value in p95.items():
        print(f"  {name}: p95={float(value):.3f} ms, p99={float(p99.get(name, np.nan)):.3f} ms")

    if {"existing", "apf"}.issubset(set(source["controller"])):
        rates = source.groupby("controller")["success"].mean()
        p95err = source.groupby("controller")["p95_error"].median()
        verdict = bool(
            float(rates.get("existing", -np.inf)) > float(rates.get("apf", -np.inf))
            or (
                float(rates.get("existing", -np.inf)) >= float(rates.get("apf", -np.inf))
                and float(p95err.get("existing", np.inf)) <= float(p95err.get("apf", np.inf))
            )
        )
        print(f"existing beats APF or matches with lower tracking error: {verdict}")

    if {"existing", "existing_planner_subgoal"}.issubset(set(source["controller"])):
        rates = source.groupby("controller")["success"].mean()
        p95err = source.groupby("controller")["p95_error"].median()
        verdict = bool(
            float(rates.get("existing_planner_subgoal", -np.inf)) > float(rates.get("existing", -np.inf))
            or (
                float(rates.get("existing_planner_subgoal", -np.inf)) >= float(rates.get("existing", -np.inf))
                and float(p95err.get("existing_planner_subgoal", np.inf)) < float(p95err.get("existing", np.inf))
            )
        )
        print(f"existing_planner_subgoal beats existing: {verdict}")

    blocked = c[c["line_of_sight_blocked_ratio"] > 0.05].copy()
    baseline_names = ["nominal_slot_tracker", "pd", "pure_pursuit"]
    if "existing" in set(blocked["controller"]) and any(name in set(blocked["controller"]) for name in baseline_names):
        existing_collision = float(blocked[blocked["controller"] == "existing"]["obstacle_collision"].mean())
        base = blocked[blocked["controller"].isin(baseline_names)]
        baseline_collision = float(base["obstacle_collision"].mean()) if not base.empty else float("nan")
        print(f"existing beats nominal/pd/pure_pursuit in collision safety: {bool(existing_collision < baseline_collision)}")

    existing = source[source["controller"] == "existing"].copy()
    if not existing.empty:
        feasible_success = float(existing["success"].mean())
        collision_rate = float(c[c["controller"] == "existing"]["obstacle_collision"].mean())
        boundary_rate = float(c[c["controller"] == "existing"]["boundary_violation"].mean())
        unknown_failures = int((c["failure_type"] == "UNKNOWN_FAILURE").sum())
        failures = int((c["failure_type"] != "SUCCESS").sum())
        unknown_ratio = 0.0 if failures == 0 else unknown_failures / failures
        rt = float(cfg["success"]["realtime_p95_ms"])
        decision_ok = bool(float(existing["decision_time_ms_p95"].median()) < rt)
        print("Initial C criteria:")
        print(f"  existing feasible-only success >= 85%: {bool(feasible_success >= 0.85)} ({100.0 * feasible_success:.1f}%)")
        print(f"  existing obstacle collision <= 1%: {bool(collision_rate <= 0.01)} ({100.0 * collision_rate:.1f}%)")
        print(f"  existing boundary violation <= 1%: {bool(boundary_rate <= 0.01)} ({100.0 * boundary_rate:.1f}%)")
        print(f"  UNKNOWN_FAILURE < 5% of failures: {bool(unknown_ratio < 0.05)} ({100.0 * unknown_ratio:.1f}%)")
        print(f"  decision_time_p95 < {rt:.1f} ms: {decision_ok}")
        configured_clearance = float(cfg.get("success", {}).get("min_obstacle_clearance_p5", cfg.get("dynamics", {}).get("safety_margin", 0.0)))
        existing_p5 = float(existing["p5_obstacle_clearance"].mean()) if "p5_obstacle_clearance" in existing else float("nan")
        existing_below = float(existing["time_below_safety_margin"].mean()) if "time_below_safety_margin" in existing else float("nan")
        print(f"  existing time_below_safety_margin <= 1%: {bool(existing_below <= 0.01)} ({100.0 * existing_below:.1f}%)")
        print(f"  existing p5 obstacle clearance >= {configured_clearance:.2f}: {bool(existing_p5 >= configured_clearance)} ({existing_p5:.3f})")


def print_boundary_corner_report(df: pd.DataFrame) -> None:
    work = df[
        (df["controller"] == "existing")
        & (df["scenario"] == "boundary_corner_turn")
        & (df["speed_level"].isin([0.5, 0.8]))
    ].copy()
    if work.empty:
        return
    print("\n=== Boundary Corner-Turn Focus Report (existing) ===")
    for speed in [0.5, 0.8]:
        part = work[work["speed_level"] == speed].copy()
        if part.empty:
            continue
        progress_nonpos = float(part["failed_step_nonpositive_progress_ratio"].mean()) if "failed_step_nonpositive_progress_ratio" in part else float("nan")
        print(
            f"speed={speed:.2f}: "
            f"success={100.0 * float(part['success'].mean()):.1f}% "
            f"p95_error={float(part['p95_error'].mean()):.3f} "
            f"slot_lost={float(part['slot_lost_ratio'].mean()):.3f} "
            f"double_clip={float(part['double_clip_warning_ratio'].mean()):.4f} "
            f"cos={float(part['mean_cos_to_goal'].mean()):.3f} "
            f"progress<=0={progress_nonpos:.3f} "
            f"angle_nom_final={float(part['mean_angle_nom_to_final'].mean()):.3f} "
            f"norm_ratio={float(part['mean_norm_ratio_final_to_nom'].mean()):.3f} "
            f"tangent_retention={float(part['mean_tangential_retention_ratio'].mean()):.3f}"
        )
        for failure_type in ["BOUNDARY_OUTWARD_ACTION", "TRACKING_DIVERGENCE", "UNKNOWN_FAILURE"]:
            count = int((part["failure_type"] == failure_type).sum())
            print(f"  {failure_type}: {count}")


def _beats_by_metric(df: pd.DataFrame, lhs: str, rhs: str, *, metric: str = "p95_error") -> bool | None:
    if df.empty or not {lhs, rhs}.issubset(set(df["controller"])):
        return None
    med = df.groupby("controller")[metric].median()
    return bool(float(med.get(lhs, np.inf)) <= float(med.get(rhs, np.inf)) + 1e-9)


def print_terminal_diagnosis(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Focused diagnosis for existing controller in free-space slot tracking."""
    print("\n=== Existing Controller Diagnosis ===")
    feasible_a = df[(df["scenario_group"] == "A") & (df["evaluation_feasible"] == True)].copy()  # noqa: E712
    static = feasible_a[feasible_a["scenario"] == "static_slot"]
    linear = feasible_a[(feasible_a["scenario"] == "linear_slot") & (feasible_a["speed_level"] <= 0.8)]
    circular = feasible_a[feasible_a["scenario"] == "circular_slot"]
    sinusoidal = feasible_a[feasible_a["scenario"] == "sinusoidal_slot"]

    checks = [
        ("Does existing beat PD in static_slot?", _beats_by_metric(static, "existing", "pd")),
        ("Does existing beat PD in feasible linear_slot speeds <= 0.8 vmax?", _beats_by_metric(linear, "existing", "pd")),
        ("Does existing beat pure_pursuit in circular_slot?", _beats_by_metric(circular, "existing", "pure_pursuit")),
        ("Does existing beat nominal_slot_tracker in sinusoidal_slot?", _beats_by_metric(sinusoidal, "existing", "nominal_slot_tracker")),
    ]
    for label, value in checks:
        print(f"{label} {value if value is not None else 'n/a'}")

    failed_existing = df[(df["controller"] == "existing") & (df["success"] == False)].copy()  # noqa: E712
    mean_cos = float(failed_existing["mean_cos_to_goal"].mean()) if not failed_existing.empty else float("nan")
    nonpos = float(failed_existing["failed_step_nonpositive_progress_ratio"].mean()) if not failed_existing.empty else float("nan")
    print(f"Mean cos_to_goal for existing in failed episodes: {mean_cos:.4f}" if np.isfinite(mean_cos) else "Mean cos_to_goal for existing in failed episodes: n/a")
    print(f"Fraction of failed existing steps with progress_projection <= 0: {nonpos:.4f}" if np.isfinite(nonpos) else "Fraction of failed existing steps with progress_projection <= 0: n/a")

    existing_a = feasible_a[feasible_a["controller"] == "existing"]
    obs_nonzero = bool(existing_a["mean_v_obstacle_norm"].fillna(0.0).gt(1e-6).any()) if not existing_a.empty else False
    boundary_nonzero = bool(existing_a["mean_v_boundary_norm"].fillna(0.0).gt(1e-6).any()) if not existing_a.empty else False
    double_clip = bool(df[df["controller"] == "existing"]["double_clip_warning_ratio"].fillna(0.0).gt(0.0).any())
    print(f"v_obstacle nonzero in no-obstacle feasible A scenarios: {obs_nonzero}")
    print(f"v_boundary nonzero in safe no-obstacle feasible A scenarios: {boundary_nonzero}")
    print(f"Speed/acceleration appears clipped more than once: {double_clip}")

    if not failed_existing.empty:
        print("\nTop 10 failed existing episodes by p95_error:")
        cols = ["scenario_group", "scenario", "speed_level", "seed", "failure_type", "p95_error", "mean_cos_to_goal", "double_clip_warning_ratio"]
        for _, row in failed_existing.sort_values("p95_error", ascending=False).head(10).iterrows():
            print(
                "  "
                f"{row['scenario_group']}/{row['scenario']} speed={float(row['speed_level']):.2f} "
                f"seed={int(row['seed'])} failure={row['failure_type']} "
                f"p95={float(row['p95_error']):.3f} cos={float(row['mean_cos_to_goal']):.3f} "
                f"double_clip={float(row['double_clip_warning_ratio']):.3f}"
            )
    else:
        print("\nTop 10 failed existing episodes by p95_error: none")

    existing = df[df["controller"] == "existing"].copy()
    if not existing.empty:
        print("\nExisting mean p95_error and slot_lost_ratio by scenario:")
        agg = existing.groupby("scenario")[["p95_error", "slot_lost_ratio"]].mean().reset_index()
        for _, row in agg.sort_values("scenario").iterrows():
            print(
                f"  {row['scenario']}: p95={float(row['p95_error']):.3f}, "
                f"lost={float(row['slot_lost_ratio']):.3f}"
            )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.num_seeds is not None:
        cfg["benchmark"]["num_seeds"] = int(args.num_seeds)
    if args.max_cases_per_group is not None:
        cfg["benchmark"]["max_cases_per_group"] = int(args.max_cases_per_group)
    if args.scenario_group:
        cfg["benchmark"]["scenario_groups"] = _scenario_groups_arg(args.scenario_group)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, output_dir)

    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    specs = scenario_specs(
        cfg["benchmark"].get("scenario_groups", ["all"]),
        [float(x) for x in cfg["benchmark"].get("speed_levels", [1.0])],
        cfg.get("robustness", {}),
        scenario_names=cfg["benchmark"].get("scenario_names"),
    )
    specs = _maybe_limit_specs(specs, cfg["benchmark"].get("max_cases_per_group"))
    if cfg["benchmark"].get("seeds") is not None:
        seeds = [int(x) for x in cfg["benchmark"].get("seeds", [])]
        nseeds = len(seeds)
        cfg["benchmark"]["num_seeds"] = nseeds
    else:
        nseeds = int(cfg["benchmark"]["num_seeds"])
        seeds_start = int(cfg["benchmark"].get("seeds_start", 1001))
        seeds = [seeds_start + i for i in range(nseeds)]
    total_episodes = len(specs) * len(seeds) * len(controllers)
    print(f"Running {len(specs)} scenarios x {len(seeds)} seeds x {len(controllers)} controllers")

    episode_rows: list[dict[str, Any]] = []
    completed = 0
    progress = ProgressBar(total_episodes, enabled=not bool(args.no_progress))
    for spec in specs:
        for seed in seeds:
            instance = instantiate_scenario(spec, cfg, seed)
            for ctrl_name in controllers:
                ctrl_key = "pure_pursuit" if ctrl_name == "pure" else ctrl_name
                ctrl_cfg_key = _controller_base_name(ctrl_key)
                speed_tag = str(f"{float(spec.speed_scale):.2f}").replace(".", "p")
                raw_file = raw_dir / f"{spec.group}_{spec.name}_speed{speed_tag}_{ctrl_key}_seed{seed}.csv"
                row = run_episode(
                    controller_name=ctrl_key,
                    controller_cfg=cfg.get("controllers", {}).get(ctrl_key, cfg.get("controllers", {}).get(ctrl_cfg_key, {})),
                    instance=instance,
                    cfg=cfg,
                    seed=seed,
                    raw_path=raw_file if cfg.get("output", {}).get("save_raw_trajectories", True) else None,
                )
                episode_rows.append(row)
                completed += 1
                progress.update(
                    completed,
                    label=f"{spec.group}/{spec.name} speed={float(spec.speed_scale):.2f} seed={seed} ctrl={ctrl_key}",
                )
    progress.close()

    df = pd.DataFrame(episode_rows)
    metrics_csv = summary_dir / "episode_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    summary = aggregate_summary(df)
    summary_csv = summary_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    feasible_summary = feasible_only_summary(df, cfg)
    if not feasible_summary.empty:
        feasible_summary.to_csv(summary_dir / "feasible_only_summary.csv", index=False)
    a_summary = a_group_subscenario_summary(df)
    if not a_summary.empty:
        a_summary.to_csv(summary_dir / "a_group_subscenario_summary.csv", index=False)
    a_failures = a_group_failure_counts(df)
    if not a_failures.empty:
        a_failures.to_csv(summary_dir / "a_group_failure_counts.csv", index=False)
    infeasible_summary = infeasible_stress_summary(df)
    if not infeasible_summary.empty:
        infeasible_summary.to_csv(summary_dir / "infeasible_stress_summary.csv", index=False)
    boundary_counts = boundary_failure_subtype_counts(df)
    if not boundary_counts.empty:
        boundary_counts.to_csv(summary_dir / "boundary_failure_subtype_counts.csv", index=False)
    c_summary = c_group_obstacle_summary(df)
    if not c_summary.empty:
        c_summary.to_csv(summary_dir / "c_group_obstacle_summary.csv", index=False)
    d_summary = d_group_jump_summary(df)
    if not d_summary.empty:
        d_summary.to_csv(summary_dir / "d_group_jump_summary.csv", index=False)
    generate_existing_debug_plots(
        df,
        output_dir,
        max_plots=int(cfg.get("output", {}).get("debug_existing_max_plots", 20)),
    )
    generate_boundary_debug_plots(
        df,
        output_dir,
        cfg,
        max_plots=int(cfg.get("output", {}).get("debug_boundary_max_plots", cfg.get("output", {}).get("debug_existing_max_plots", 20))),
    )
    generate_obstacle_debug_plots(
        df,
        output_dir,
        cfg,
        max_plots=int(cfg.get("output", {}).get("debug_obstacle_max_plots", cfg.get("output", {}).get("debug_existing_max_plots", 20))),
    )
    generate_d_group_debug_plots(
        df,
        output_dir,
        max_plots=int(cfg.get("output", {}).get("debug_d_max_plots", cfg.get("output", {}).get("debug_existing_max_plots", 20))),
    )
    print(f"Saved episode metrics: {metrics_csv}")
    print(f"Saved grouped summary: {summary_csv}")
    print_terminal_report(df, cfg)


if __name__ == "__main__":
    main()
