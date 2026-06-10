"""Tracking, smoothness, and runtime metrics for slot-tracking episodes."""

from __future__ import annotations

import numpy as np


def _finite_percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def time_to_lock(errors: np.ndarray, *, lock_threshold: float, lock_window: int, dt: float) -> float:
    """First time when rolling-window P95 error remains below threshold."""
    e = np.asarray(errors, dtype=np.float64).reshape(-1)
    w = max(int(lock_window), 1)
    if e.size < w:
        return float("nan")
    for k in range(w - 1, e.size):
        if np.percentile(e[k - w + 1 : k + 1], 95) <= float(lock_threshold):
            return float(k) * float(dt)
    return float("nan")


def reacquisition_time(errors: np.ndarray, jump_steps: list[int], *, threshold: float, dt: float) -> float:
    """Mean time needed to get back below threshold after configured slot jumps."""
    if not jump_steps:
        return 0.0
    e = np.asarray(errors, dtype=np.float64).reshape(-1)
    durations: list[float] = []
    for js in jump_steps:
        found = None
        for k in range(int(js), e.size):
            if e[k] <= float(threshold):
                found = k - int(js)
                break
        durations.append(float("inf") if found is None else float(found) * float(dt))
    finite = [x for x in durations if np.isfinite(x)]
    if not finite:
        return float("inf")
    return float(np.mean(finite))


def compute_tracking_metrics(rows: list[dict], *, cfg: dict, jump_steps: list[int]) -> dict[str, float]:
    """Compute per-episode tracking and smoothness metrics from per-step rows."""
    dt = float(cfg["dynamics"]["dt"])
    success_cfg = cfg["success"]
    dyn = cfg["dynamics"]
    errors = np.asarray([r["tracking_error"] for r in rows], dtype=np.float64)
    action = np.asarray([[r["action_x"], r["action_y"]] for r in rows], dtype=np.float64)
    vel = np.asarray([[r["uav_vx"], r["uav_vy"]] for r in rows], dtype=np.float64)
    uav_xy = np.asarray([[r["uav_x"], r["uav_y"]] for r in rows], dtype=np.float64)
    slot_xy = np.asarray([[r["slot_x"], r["slot_y"]] for r in rows], dtype=np.float64)

    acc = np.diff(vel, axis=0) / max(dt, 1e-9) if vel.shape[0] > 1 else np.zeros((0, 2))
    jerk = np.diff(acc, axis=0) / max(dt, 1e-9) if acc.shape[0] > 1 else np.zeros((0, 2))
    path_len = _polyline_length(uav_xy)
    slot_path_len = _polyline_length(slot_xy)
    tail = max(int(0.2 * errors.size), 1)
    speed = np.linalg.norm(vel, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1) if acc.size else np.zeros(1)
    jerk_norm = np.linalg.norm(jerk, axis=1) if jerk.size else np.zeros(1)
    cos = np.asarray([r.get("cos_to_goal", np.nan) for r in rows], dtype=np.float64)
    progress = np.asarray([r.get("progress_projection", np.nan) for r in rows], dtype=np.float64)
    speed_sat_flags = np.asarray([bool(r.get("speed_saturation_flag", False)) for r in rows], dtype=bool)
    accel_sat_flags = np.asarray([bool(r.get("acceleration_saturation_flag", False)) for r in rows], dtype=bool)
    double_clip_flags = np.asarray([bool(r.get("double_clip_warning", False)) for r in rows], dtype=bool)

    return {
        "rmse_error": float(np.sqrt(np.mean(errors * errors))) if errors.size else float("nan"),
        "mean_error": float(np.mean(errors)) if errors.size else float("nan"),
        "median_error": float(np.median(errors)) if errors.size else float("nan"),
        "p95_error": _finite_percentile(errors, 95),
        "max_error": float(np.max(errors)) if errors.size else float("nan"),
        "final_error": float(errors[-1]) if errors.size else float("nan"),
        "steady_state_error": float(np.mean(errors[-tail:])) if errors.size else float("nan"),
        "slot_lost_ratio": float(np.mean(errors > float(dyn["lost_threshold"]))) if errors.size else float("nan"),
        "time_to_lock": time_to_lock(
            errors,
            lock_threshold=float(success_cfg["lock_threshold"]),
            lock_window=int(success_cfg["lock_window"]),
            dt=dt,
        ),
        "reacquisition_time": reacquisition_time(
            errors,
            jump_steps,
            threshold=float(success_cfg["reacquisition_threshold"]),
            dt=dt,
        ),
        "path_length": path_len,
        "slot_path_length": slot_path_len,
        "detour_ratio": float(path_len / max(slot_path_len, 1e-9)),
        "control_effort": float(np.sum(np.linalg.norm(action, axis=1) ** 2)),
        "acceleration_norm_mean": float(np.mean(acc_norm)),
        "acceleration_norm_p95": _finite_percentile(acc_norm, 95),
        "jerk_norm_mean": float(np.mean(jerk_norm)),
        "jerk_norm_p95": _finite_percentile(jerk_norm, 95),
        "mean_cos_to_goal": _finite_mean(cos),
        "mean_progress_projection": _finite_mean(progress),
        "failed_step_nonpositive_progress_ratio": float(np.mean(progress <= 0.0)) if progress.size else 0.0,
        "speed_saturation_ratio": float(np.mean(speed_sat_flags)) if speed_sat_flags.size else float(np.mean(speed >= 0.98 * float(dyn["uav_vmax"]))),
        "turn_or_acceleration_saturation_ratio": float(np.mean(accel_sat_flags)) if accel_sat_flags.size else float(np.mean(acc_norm >= 0.98 * float(dyn["uav_amax"]))),
        "acceleration_saturation_ratio": float(np.mean(accel_sat_flags)) if accel_sat_flags.size else 0.0,
        "double_clip_warning_ratio": float(np.mean(double_clip_flags)) if double_clip_flags.size else 0.0,
    }


def _polyline_length(xy: np.ndarray) -> float:
    pts = np.asarray(xy, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))
