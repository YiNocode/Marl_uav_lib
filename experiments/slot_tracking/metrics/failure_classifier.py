"""Rule-based primary failure classification for slot-tracking episodes."""

from __future__ import annotations

import numpy as np


FAILURE_TYPES = [
    "SUCCESS",
    "TARGET_INFEASIBLE",
    "NO_PROGRESS",
    "STUCK_NEAR_OBSTACLE",
    "BOUNDARY_SLOT_OUTSIDE",
    "BOUNDARY_SLOT_TOO_CLOSE",
    "BOUNDARY_INITIAL_STATE_INVALID",
    "BOUNDARY_OUTWARD_ACTION",
    "BOUNDARY_INERTIA_OVERSHOOT",
    "BOUNDARY_PROJECTED_ACTION_FAILED",
    "COLLISION_OBSTACLE",
    "COLLISION_AGENT",
    "OSCILLATION",
    "LATE_RESPONSE",
    "TRACKING_DIVERGENCE",
    "JUMP_REACQUISITION_TIMEOUT",
    "JUMP_TOO_LARGE_FOR_WINDOW",
    "INVALID_RAW_SLOT",
    "INVALID_COMMANDED_SLOT",
    "SAFE_REJECTION_SUCCESS",
    "SAFE_STABILIZATION_SUCCESS",
    "RAW_SLOT_TOO_UNSTABLE",
    "NO_VALID_PROXY_SLOT",
    "COMMAND_TRANSITION_BLOCKED",
    "PLANNER_FAILED_AFTER_JUMP",
    "REACQUISITION_TIMEOUT",
    "INVALID_RAW_SLOT_SAFE_REJECTED",
    "COMMAND_SLOT_LAG_TOO_LARGE",
    "SLOT_TRANSITION_OSCILLATION",
    "TARGET_MODE_SWITCH_CHATTER",
    "UNKNOWN_FAILURE",
]


def episode_success(metrics: dict, *, feasible: bool, cfg: dict) -> bool:
    """Return benchmark success according to configured thresholds."""
    success_cfg = cfg["success"]
    p95_threshold = float(success_cfg.get("p95_error_threshold", float("inf")))
    common_ok = bool(
        feasible
        and not metrics.get("invalid_initial_state", False)
        and not metrics.get("obstacle_collision", False)
        and not metrics.get("boundary_violation", False)
        and not metrics.get("inter_agent_collision", False)
    )
    if bool(metrics.get("safe_stabilization_success", False)):
        return True
    if int(metrics.get("number_of_jump_events", 0)) > 0:
        return bool(
            common_ok
            and float(metrics.get("invalid_commanded_slot_ratio", 0.0)) <= 0.0
            and bool(metrics.get("pre_jump_locked", False))
            and bool(metrics.get("reacquired_within_budget", False))
            and float(metrics.get("post_reacquisition_p95_error", float("inf"))) <= float(success_cfg.get("post_reacquisition_p95_error_threshold", p95_threshold))
            and float(metrics.get("post_reacquisition_slot_lost_ratio", float("inf"))) <= float(success_cfg["max_lost_ratio"])
            and not bool(metrics.get("oscillation_after_jump", False))
        )
    return bool(
        common_ok
        and float(metrics.get("final_error", float("inf"))) <= float(success_cfg["final_error_threshold"])
        and float(metrics.get("steady_state_error", float("inf"))) <= float(success_cfg["steady_state_error_threshold"])
        and float(metrics.get("p95_error", float("inf"))) <= p95_threshold
        and float(metrics.get("slot_lost_ratio", float("inf"))) <= float(success_cfg["max_lost_ratio"])
    )


def classify_failure(rows: list[dict], metrics: dict, *, feasible: bool, cfg: dict) -> str:
    """Assign exactly one primary failure type."""
    if episode_success(metrics, feasible=feasible, cfg=cfg):
        return "SUCCESS"
    fc = cfg["failure_classifier"]
    success_cfg = cfg["success"]
    p95_threshold = float(success_cfg.get("p95_error_threshold", float("inf")))
    if bool(metrics.get("obstacle_collision", False)):
        return "COLLISION_OBSTACLE"
    if bool(metrics.get("inter_agent_collision", False)):
        return "COLLISION_AGENT"
    if bool(metrics.get("safe_rejection_success", False)):
        if float(metrics.get("raw_slot_invalid_ratio", 0.0)) > 0.0:
            return "INVALID_RAW_SLOT_SAFE_REJECTED"
        return "SAFE_REJECTION_SUCCESS"
    if bool(metrics.get("safe_stabilization_success", False)):
        return "SAFE_STABILIZATION_SUCCESS"
    if int(metrics.get("number_of_jump_events", 0)) > 0:
        if float(metrics.get("invalid_commanded_slot_ratio", 0.0)) > 0.0:
            return "INVALID_COMMANDED_SLOT"
        if float(metrics.get("raw_slot_too_unstable_ratio", 0.0)) > 0.0:
            return "RAW_SLOT_TOO_UNSTABLE"
        if float(metrics.get("no_valid_proxy_slot_ratio", 0.0)) > 0.0:
            return "NO_VALID_PROXY_SLOT"
        if float(metrics.get("command_transition_blocked_ratio", 0.0)) > 0.0:
            return "COMMAND_TRANSITION_BLOCKED"
        if bool(metrics.get("jump_too_large_for_window", False)):
            return "JUMP_TOO_LARGE_FOR_WINDOW"
        if float(metrics.get("raw_slot_invalid_ratio", 0.0)) > float(fc.get("slot_out_of_bounds_threshold", fc.get("infeasible_outside_ratio", 0.35))):
            return "INVALID_RAW_SLOT"
        if float(metrics.get("planner_called_after_jump", 0.0)) > 0.0 and float(metrics.get("planner_success_after_jump", 1.0)) <= 0.0:
            return "PLANNER_FAILED_AFTER_JUMP"
        if float(metrics.get("commanded_slot_lag_to_raw_max", 0.0)) > float(fc.get("commanded_slot_lag_threshold", float("inf"))):
            return "COMMAND_SLOT_LAG_TOO_LARGE"
        if int(metrics.get("target_mode_switch_count", 0)) > int(fc.get("target_mode_chatter_switch_threshold", 20)):
            return "TARGET_MODE_SWITCH_CHATTER"
        if bool(metrics.get("oscillation_after_jump", False)):
            return "SLOT_TRANSITION_OSCILLATION"
        if not bool(metrics.get("reacquired_within_budget", True)):
            return "REACQUISITION_TIMEOUT"
        if float(metrics.get("post_reacquisition_p95_error", 0.0)) > float(success_cfg.get("post_reacquisition_p95_error_threshold", p95_threshold)):
            return "REACQUISITION_TIMEOUT"
        if float(metrics.get("post_reacquisition_slot_lost_ratio", 0.0)) > float(success_cfg.get("max_lost_ratio", 0.30)):
            return "REACQUISITION_TIMEOUT"
    if not feasible:
        return "TARGET_INFEASIBLE"
    boundary_subtype = classify_boundary_failure_subtype(rows, metrics, cfg=cfg)
    if boundary_subtype:
        return boundary_subtype

    errors = np.asarray([r["tracking_error"] for r in rows], dtype=np.float64)
    if errors.size < 4:
        return "UNKNOWN_FAILURE"
    outside_ratio = float(metrics.get("slot_out_of_bounds_ratio", np.mean([bool(r.get("slot_outside_boundary", False)) for r in rows])))
    if outside_ratio > float(fc.get("slot_out_of_bounds_threshold", fc.get("infeasible_outside_ratio", 0.35))):
        return "TARGET_INFEASIBLE"
    severe_ratio = float(np.mean(errors > float(fc["severe_lost_threshold"])))
    jump_late_reacquisition = bool(
        int(metrics.get("number_of_jump_events", 0)) > 0
        and float(metrics.get("mean_cos_to_goal", metrics.get("mean_cos_to_commanded_slot", 0.0))) >= float(fc.get("high_cos_not_divergent_threshold", 0.90))
        and float(metrics.get("failed_step_nonpositive_progress_ratio", 1.0)) <= float(fc.get("positive_progress_not_divergent_ratio", 0.25))
    )
    if severe_ratio > float(fc["severe_lost_min_ratio"]) and not jump_late_reacquisition:
        return "TRACKING_DIVERGENCE"
    late_limit = float(cfg["success"]["reacquisition_max_steps"]) * float(cfg["dynamics"]["dt"])
    time_to_lock = float(metrics.get("time_to_lock", float("nan")))
    if float(metrics.get("reacquisition_time", 0.0)) > late_limit:
        return "LATE_RESPONSE"
    if np.isfinite(time_to_lock) and time_to_lock > late_limit:
        return "LATE_RESPONSE"
    if not np.isfinite(time_to_lock) and float(metrics.get("slot_lost_ratio", 1.0)) <= float(cfg["success"].get("max_lost_ratio", 0.30)):
        return "LATE_RESPONSE"

    half = max(errors.size // 2, 1)
    first = float(np.mean(errors[:half]))
    last = float(np.mean(errors[-half:]))
    progress = first - last
    min_clear = float(metrics.get("min_obstacle_clearance", float("inf")))
    if (
        progress < float(fc["stuck_progress_threshold"])
        and min_clear < float(fc["stuck_clearance_threshold"])
    ) or float(metrics.get("stuck_ratio", 0.0)) > float(fc.get("stuck_ratio_threshold", 0.15)):
        return "STUCK_NEAR_OBSTACLE"
    if _oscillates_near_obstacles(rows, errors, cfg=cfg):
        return "OSCILLATION"
    if _should_classify_no_progress(rows, metrics, errors, progress, cfg=cfg):
        return "NO_PROGRESS"
    tail = errors[-max(int(0.3 * errors.size), 1):]
    if float(np.mean(tail)) < float(fc["oscillation_error_threshold"]) and float(np.std(tail)) > float(fc["oscillation_std_threshold"]):
        return "OSCILLATION"
    if _mostly_increasing(errors):
        return "TRACKING_DIVERGENCE"
    return "UNKNOWN_FAILURE"


def classify_failure_subtype(rows: list[dict], metrics: dict, *, feasible: bool, cfg: dict) -> str:
    """Secondary subtype; boundary cases are made explicit here."""
    if feasible and episode_success(metrics, feasible=feasible, cfg=cfg):
        return "SUCCESS"
    boundary = classify_boundary_failure_subtype(rows, metrics, cfg=cfg)
    if boundary:
        return boundary
    if not feasible and float(metrics.get("slot_out_of_bounds_ratio", 0.0)) > float(
        cfg.get("failure_classifier", {}).get("slot_out_of_bounds_threshold", 0.35)
    ):
        return "BOUNDARY_SLOT_OUTSIDE"
    if not feasible:
        return str(metrics.get("infeasible_reason", "TARGET_INFEASIBLE"))
    return ""


def classify_boundary_failure_subtype(rows: list[dict], metrics: dict, *, cfg: dict) -> str:
    """Boundary-specific primary subtype for boundary-related failures."""
    fc = cfg.get("failure_classifier", {})
    dyn = cfg.get("dynamics", {})
    outside_threshold = float(fc.get("slot_out_of_bounds_threshold", fc.get("infeasible_outside_ratio", 0.35)))
    close_threshold = float(fc.get("slot_too_close_ratio_threshold", 0.50))
    safety_margin = float(fc.get("slot_boundary_safety_margin", dyn.get("safety_margin", 0.25)))

    if float(metrics.get("slot_out_of_bounds_ratio", 0.0)) > outside_threshold:
        return "BOUNDARY_SLOT_OUTSIDE"
    if (
        float(metrics.get("slot_out_of_bounds_ratio", 0.0)) <= outside_threshold
        and float(metrics.get("slot_too_close_ratio", 0.0)) > close_threshold
        and float(metrics.get("slot_min_boundary_margin", float("inf"))) < safety_margin
    ):
        return "BOUNDARY_SLOT_TOO_CLOSE"
    if bool(metrics.get("invalid_initial_state", False)):
        return "BOUNDARY_INITIAL_STATE_INVALID"
    if not bool(metrics.get("boundary_violation", False)) and float(metrics.get("outward_velocity_ratio", 0.0)) <= float(
        fc.get("outward_velocity_ratio_threshold", 0.35)
    ):
        return ""

    if any(bool(r.get("boundary_projected_action_failed", False)) for r in rows):
        return "BOUNDARY_PROJECTED_ACTION_FAILED"
    if any(bool(r.get("boundary_violation", False)) and bool(r.get("boundary_filter_active", False)) for r in rows):
        return "BOUNDARY_PROJECTED_ACTION_FAILED"
    if any(
        float(r.get("boundary_margin", float("inf"))) <= float(fc.get("near_boundary_distance", 1.0))
        and float(r.get("action_outward_projection", 0.0)) > 1e-6
        for r in rows
    ):
        return "BOUNDARY_OUTWARD_ACTION"
    if bool(metrics.get("boundary_violation", False)):
        return "BOUNDARY_INERTIA_OVERSHOOT"
    if float(metrics.get("outward_velocity_ratio", 0.0)) > float(fc.get("outward_velocity_ratio_threshold", 0.35)):
        return "BOUNDARY_OUTWARD_ACTION"
    return ""


def _should_classify_no_progress(
    rows: list[dict],
    metrics: dict,
    errors: np.ndarray,
    progress: float,
    *,
    cfg: dict,
) -> bool:
    fc = cfg["failure_classifier"]
    dyn = cfg["dynamics"]
    success_cfg = cfg["success"]
    lost_threshold = float(dyn.get("lost_threshold", fc.get("severe_lost_threshold", 3.0)))
    max_lost_ratio = float(success_cfg.get("max_lost_ratio", 0.25))
    steady_threshold = float(success_cfg.get("steady_state_error_threshold", lost_threshold))
    p95_error = float(metrics.get("p95_error", np.percentile(errors, 95) if errors.size else float("inf")))
    slot_lost_ratio = float(metrics.get("slot_lost_ratio", np.mean(errors > lost_threshold) if errors.size else 1.0))
    steady_state_error = float(metrics.get("steady_state_error", np.mean(errors[-max(int(0.2 * errors.size), 1):])))

    if p95_error < lost_threshold and slot_lost_ratio <= max_lost_ratio and steady_state_error <= steady_threshold:
        return False
    final_error = float(metrics.get("final_error", errors[-1] if errors.size else float("inf")))
    if (
        slot_lost_ratio > float(fc.get("no_progress_lost_ratio_threshold", max(0.60, 2.0 * max_lost_ratio)))
        and (steady_state_error > steady_threshold or final_error > steady_threshold)
    ):
        return True

    window = max(int(float(fc["no_progress_window_fraction"]) * errors.size), 1)
    tail = errors[-window:]
    head = errors[:window]
    tail_lost_ratio = float(np.mean(tail > lost_threshold)) if tail.size else 0.0
    tail_error_high = bool(float(np.mean(tail)) > lost_threshold and tail_lost_ratio > 0.5)
    distance_delta = np.asarray([float(r.get("distance_delta", 0.0)) for r in rows[-window:]], dtype=np.float64)
    progress_projection = np.asarray([float(r.get("progress_projection", 0.0)) for r in rows[-window:]], dtype=np.float64)
    poor_distance_delta = bool(distance_delta.size == 0 or float(np.mean(distance_delta)) < float(fc["stuck_progress_threshold"]))
    poor_projection = bool(progress_projection.size == 0 or float(np.mean(progress_projection)) <= 1e-6)
    no_tail_improvement = bool(float(np.mean(tail)) > 0.85 * float(np.mean(head)) and progress < float(fc["stuck_progress_threshold"]))
    return bool(tail_error_high and poor_distance_delta and poor_projection and no_tail_improvement)


def _mostly_increasing(errors: np.ndarray) -> bool:
    if errors.size < 10:
        return False
    diff = np.diff(errors)
    return bool(np.mean(diff > 0.0) > 0.65 and errors[-1] > errors[0])


def _oscillates_near_obstacles(rows: list[dict], errors: np.ndarray, *, cfg: dict) -> bool:
    if len(rows) < 8:
        return False
    fc = cfg["failure_classifier"]
    tail_n = max(int(0.35 * len(rows)), 8)
    tail_rows = rows[-tail_n:]
    tail_errors = errors[-tail_n:]
    near = np.asarray(
        [
            float(r.get("nearest_obstacle_distance", float("inf"))) < float(fc.get("oscillation_clearance_threshold", fc["stuck_clearance_threshold"]))
            for r in tail_rows
        ],
        dtype=bool,
    )
    if float(np.mean(near)) < float(fc.get("oscillation_near_obstacle_ratio", 0.25)):
        return False
    actions = np.asarray([[float(r.get("action_x", 0.0)), float(r.get("action_y", 0.0))] for r in tail_rows], dtype=np.float64)
    norms = np.linalg.norm(actions, axis=1)
    valid = norms > 1e-6
    if int(np.sum(valid)) < 6:
        return False
    unit = actions[valid] / norms[valid].reshape(-1, 1)
    dots = np.sum(unit[1:] * unit[:-1], axis=1)
    reversals = float(np.mean(dots < -0.2)) if dots.size else 0.0
    high_error = bool(float(np.mean(tail_errors)) > float(fc.get("oscillation_error_threshold", 0.8)))
    error_wobble = bool(float(np.std(tail_errors)) > float(fc.get("oscillation_std_threshold", 0.4)))
    return bool(reversals > float(fc.get("oscillation_action_reversal_ratio", 0.20)) and (high_error or error_wobble))
