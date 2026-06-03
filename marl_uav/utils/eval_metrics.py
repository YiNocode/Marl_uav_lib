"""Shared episode-level evaluation metrics (E2 benchmark + BC eval)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def window_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return math.nan
    kk = min(int(k), int(values.size))
    return float(np.mean(values[-kk:]))


def structure_series_from_trajectory(traj_xyz: np.ndarray) -> list[dict[str, float]]:
    from marl_uav.envs.tasks.pursuit_evasion_3v1_task import compute_pursuit_structure_metrics_3v1

    arr = np.asarray(traj_xyz, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] < 4:
        return []
    return [
        compute_pursuit_structure_metrics_3v1(arr[t, :3, :], arr[t, 3, :])
        for t in range(arr.shape[0])
    ]


def value_array(series: list[dict[str, Any]], key: str) -> np.ndarray:
    if not series:
        return np.zeros(0, dtype=np.float64)
    vals: list[float] = []
    for row in series:
        if key in row:
            vals.append(float(row[key]))
    return np.asarray(vals, dtype=np.float64)


def phi_max_array(series: list[dict[str, Any]]) -> np.ndarray:
    if not series:
        return np.zeros(0, dtype=np.float64)
    from marl_uav.envs.tasks.pursuit_evasion_3v1_task import phi_max_from_c_cov

    vals: list[float] = []
    for row in series:
        if "phi_max" in row:
            vals.append(float(row["phi_max"]))
        elif "C_cov" in row:
            vals.append(phi_max_from_c_cov(float(row["C_cov"])))
    return np.asarray(vals, dtype=np.float64)


def episode_metrics_from_info(
    *,
    info: dict[str, Any],
    trajectory: np.ndarray | None = None,
    terminal_window: int = 30,
    control_hz: float = 50.0,
) -> dict[str, Any]:
    """Episode metrics aligned with benchmark E2 summarize keys."""
    traj = np.asarray(trajectory, dtype=np.float32) if trajectory is not None else np.zeros((0, 4, 3))
    series = info.get("pursuit_structure_series")
    if not isinstance(series, list) or not series:
        series = structure_series_from_trajectory(traj)
    cov = value_array(series, "C_cov")
    col = value_array(series, "C_col")
    dang = value_array(series, "D_ang")
    phi_max = phi_max_array(series)

    captured = bool(info.get("capture", False) or info.get("captured", False))
    obstacle_terminal = bool(info.get("obstacle_termination", False))
    timeout = bool(info.get("timeout", False))
    out_of_bounds = bool(info.get("out_of_bounds", False))
    collision = bool(info.get("collision", False) or info.get("has_collision", False))

    if captured:
        terminal_reason = "capture"
    elif obstacle_terminal:
        terminal_reason = "obstacle_collision_terminal"
    elif out_of_bounds:
        terminal_reason = "out_of_bounds_terminal"
    elif timeout:
        terminal_reason = "timeout"
    elif collision:
        terminal_reason = "inter_agent_collision_terminal"
    else:
        terminal_reason = "other_failure"

    capture_step = int(info.get("capture_step", -1))
    if captured and capture_step < 0:
        capture_step = int(info.get("episode_len", 0))
    capture_time_s = (
        float(capture_step / control_hz) if captured and control_hz > 0 else math.nan
    )

    tw = int(terminal_window)
    max_gap = window_mean(phi_max, tw)
    row: dict[str, Any] = {
        "episode_return": float(info.get("episode_return", math.nan)),
        "episode_len": int(info.get("episode_len", 0)),
        "captured": int(captured),
        "capture_step": capture_step if captured else "",
        "capture_time_s": capture_time_s,
        "mean_time_to_capture": capture_time_s,
        "terminal_reason": terminal_reason,
        "terminal_capture": int(terminal_reason == "capture"),
        "terminal_obstacle_collision": int(terminal_reason == "obstacle_collision_terminal"),
        "terminal_inter_agent_collision": int(terminal_reason == "inter_agent_collision_terminal"),
        "terminal_out_of_bounds": int(terminal_reason == "out_of_bounds_terminal"),
        "terminal_timeout": int(terminal_reason == "timeout"),
        "terminal_other_failure": int(terminal_reason == "other_failure"),
        "collision": int(collision),
        "any_collision": int(collision or obstacle_terminal),
        "inter_agent_collision_rate": int(terminal_reason == "inter_agent_collision_terminal"),
        "obstacle_termination": int(obstacle_terminal),
        "timeout": int(timeout),
        "out_of_bounds": int(out_of_bounds),
        "other_failure_rate": int(terminal_reason == "other_failure"),
        f"C_cov_last{tw}": window_mean(cov, tw),
        f"C_col_last{tw}": window_mean(col, tw),
        f"D_ang_last{tw}": window_mean(dang, tw),
        f"max_escape_gap_last{tw}": max_gap,
        f"max_escape_gap_deg_last{tw}": float(np.degrees(max_gap)) if np.isfinite(max_gap) else math.nan,
        "slot_tracking_error": float(info.get("mean_path_tracking_error", math.nan)),
        "path_tracking_error": float(info.get("mean_path_tracking_error", math.nan)),
        "action_smoothness": float(info.get("action_smoothness", math.nan)),
        "mean_action_norm": float(info.get("mean_action_norm", math.nan)),
        "mean_delta_action_norm": float(info.get("mean_delta_action_norm", math.nan)),
    }
    rc = info.get("reward_components")
    if isinstance(rc, dict):
        for key in (
            "reward_capture",
            "reward_capture_progress",
            "reward_slot_tracking",
            "reward_structure",
            "reward_collision",
            "reward_smooth",
            "reward_bc_anchor",
        ):
            if key in rc:
                row[key] = float(rc[key])
    return row


def aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean rates/metrics over episode rows."""
    if not rows:
        return {}
    n = max(len(rows), 1)

    def mean_key(key: str) -> float:
        vals = []
        for r in rows:
            v = r.get(key)
            if v == "" or v is None:
                continue
            try:
                fv = float(v)
                if np.isfinite(fv):
                    vals.append(fv)
            except (TypeError, ValueError):
                continue
        return float(np.mean(vals)) if vals else float("nan")

    out = {
        "num_episodes": float(len(rows)),
        "capture_rate": mean_key("captured"),
        "collision_rate": mean_key("collision"),
        "obstacle_termination_rate": mean_key("obstacle_termination"),
        "timeout_rate": mean_key("timeout"),
        "out_of_bounds_rate": mean_key("out_of_bounds"),
        "bc_action_mse": mean_key("bc_action_mse"),
        "bc_action_cosine_similarity": mean_key("bc_action_cosine_similarity"),
    }
    tw = 30
    for suffix in (f"D_ang_last{tw}", f"C_cov_last{tw}", f"C_col_last{tw}", f"max_escape_gap_last{tw}"):
        out[suffix.replace(f"last{tw}", f"eval_{suffix}")] = mean_key(suffix)
        out[suffix] = mean_key(suffix)
    return out
