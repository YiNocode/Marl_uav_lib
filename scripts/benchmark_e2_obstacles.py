"""E2 obstacle-rich benchmark for heuristic and learned policies in obstacle fields.

Evaluates geometric / slot / trajectory-planner heuristics and checkpointed
MAPPO-style policies on PyFlyt ex2 (cylindrical obstacles), then writes task,
structure, and obstacle metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_uav.control.fixed_ring_pursuit import make_fixed_ring_get_actions_fn
from marl_uav.control.geometric_pursuit_baselines import (
    make_hungarian_slot_get_actions_fn,
    make_oracle_slot_get_actions_fn,
    make_ot_slot_get_actions_fn,
    make_pure_pursuit_get_actions_fn,
)
from marl_uav.control.obstacle_apf_baselines import (
    make_fixed_ring_apf_get_actions_fn,
    make_pure_pursuit_apf_get_actions_fn,
)
from marl_uav.control.trajectory_planner import make_trajectory_planner_get_actions_fn
from marl_uav.agents.mac import MAC
from marl_uav.envs.factories import build_env_from_config
from marl_uav.utils.checkpoint import load_checkpoint
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.config import load_config

from scripts.train import build_learner, build_policy


def load_suite(path: Path) -> dict[str, Any]:
    cfg = load_config(path)
    if "methods" not in cfg:
        raise ValueError(f"Suite config missing 'methods': {path}")
    return cfg


def selected_methods(suite: dict[str, Any], names: list[str] | None) -> list[str]:
    available = list((suite.get("methods") or {}).keys())
    if not names:
        return available
    missing = [m for m in names if m not in available]
    if missing:
        raise ValueError(f"Unknown method(s): {missing}. Available: {available}")
    return names


def _deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    for key, value in dict(updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in str(dotted_key).split(".") if p]
    if not parts:
        raise ValueError("Curriculum parameter path must not be empty.")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _stage_value_label(value: Any) -> str:
    try:
        fv = float(value)
        if fv.is_integer():
            return str(int(fv))
        return ("%g" % fv).replace(".", "p")
    except (TypeError, ValueError):
        return str(value).replace(".", "p")


def _format_path_template(template: str, *, value: Any, seed: int) -> str:
    return template.format(value=_stage_value_label(value), seed=int(seed))


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(str(path_like))
    return p if p.is_absolute() else ROOT / p


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _window_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return math.nan
    kk = min(int(k), int(values.size))
    return float(np.mean(values[-kk:]))


def _structure_series_from_trajectory(traj_xyz: np.ndarray) -> list[dict[str, float]]:
    from marl_uav.envs.tasks.pursuit_evasion_3v1_task import (
        compute_pursuit_structure_metrics_3v1,
    )

    arr = np.asarray(traj_xyz, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] < 4:
        return []
    return [
        compute_pursuit_structure_metrics_3v1(arr[t, :3, :], arr[t, 3, :])
        for t in range(arr.shape[0])
    ]


def _value_array(series: list[dict[str, Any]], key: str) -> np.ndarray:
    if not series:
        return np.zeros(0, dtype=np.float64)
    vals: list[float] = []
    for row in series:
        if key in row:
            vals.append(float(row[key]))
    return np.asarray(vals, dtype=np.float64)


def _xy_velocity_series(traj_xyz: np.ndarray) -> np.ndarray:
    xy = np.asarray(traj_xyz[:, :, :2], dtype=np.float64)
    if xy.shape[0] <= 1:
        return np.zeros_like(xy, dtype=np.float64)
    dxy = np.diff(xy, axis=0)
    vel = np.zeros_like(xy, dtype=np.float64)
    vel[1:] = dxy
    vel[0] = dxy[0]
    return vel


def _fesc_series(traj_xyz: np.ndarray, n_theta: int = 72) -> np.ndarray:
    arr = np.asarray(traj_xyz, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 4:
        return np.zeros(0, dtype=np.float64)
    vel_xy = _xy_velocity_series(arr)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float64)
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    ev_proj = vel_xy[:, 3, :] @ dirs.T
    pu_proj = np.einsum("tad,kd->tak", vel_xy[:, :3, :], dirs)
    return np.max(ev_proj - np.max(pu_proj, axis=1), axis=1).astype(np.float64)


def _role_stability_series(traj_xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj_xyz, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 4:
        return np.zeros(0, dtype=np.float64)
    rel_xy = arr[:, :3, :2] - arr[:, 3:4, :2]
    theta = np.arctan2(rel_xy[:, :, 1], rel_xy[:, :, 0])
    ranks = np.zeros_like(theta, dtype=np.int64)
    for t in range(theta.shape[0]):
        order = np.argsort(theta[t])
        ranks[t, order] = np.arange(3, dtype=np.int64)
    stability = np.ones(theta.shape[0], dtype=np.float64)
    if theta.shape[0] >= 2:
        stability[1:] = 1.0 - np.mean((ranks[1:] != ranks[:-1]).astype(np.float64), axis=1)
    return stability


def _phi_max_array(series: list[dict[str, Any]]) -> np.ndarray:
    """Per-step φ_max; recover from ``C_cov`` when env omitted ``phi_max``."""
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


def _episode_metrics(
    *,
    info: dict[str, Any],
    trajectory: np.ndarray,
    terminal_window: int,
    control_hz: float,
) -> dict[str, Any]:
    series = info.get("pursuit_structure_series")
    if not isinstance(series, list) or not series:
        series = _structure_series_from_trajectory(trajectory)
    cov = _value_array(series, "C_cov")
    col = _value_array(series, "C_col")
    dang = _value_array(series, "D_ang")
    phi_max = _phi_max_array(series)
    fesc = _fesc_series(trajectory)
    role_stability = _role_stability_series(trajectory)

    captured = bool(info.get("capture", False))
    obstacle_terminal = bool(info.get("obstacle_termination", False))
    timeout = bool(info.get("timeout", False))
    out_of_bounds = bool(info.get("out_of_bounds", False))
    collision = bool(info.get("collision", False))
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
    capture_time_s = float(capture_step / control_hz) if captured and control_hz > 0 else math.nan

    max_gap = _window_mean(phi_max, terminal_window)
    row = {
        "episode_return": float(info.get("episode_return", math.nan)),
        "episode_len": int(info.get("episode_len", 0)),
        "captured": int(captured),
        "capture_step": capture_step if captured else "",
        "capture_time_s": capture_time_s,
        "terminal_reason": terminal_reason,
        "terminal_capture": int(terminal_reason == "capture"),
        "terminal_obstacle_collision": int(terminal_reason == "obstacle_collision_terminal"),
        "terminal_inter_agent_collision": int(terminal_reason == "inter_agent_collision_terminal"),
        "terminal_out_of_bounds": int(terminal_reason == "out_of_bounds_terminal"),
        "terminal_timeout": int(terminal_reason == "timeout"),
        "terminal_other_failure": int(terminal_reason == "other_failure"),
        "collision": int(collision),
        "any_collision": int(collision or obstacle_terminal),
        "obstacle_termination": int(obstacle_terminal),
        "timeout": int(timeout),
        "out_of_bounds": int(out_of_bounds),
        "pursuer_oob": int(bool(info.get("pursuer_oob", False))),
        f"C_cov_last{terminal_window}": _window_mean(cov, terminal_window),
        f"C_col_last{terminal_window}": _window_mean(col, terminal_window),
        f"D_ang_last{terminal_window}": _window_mean(dang, terminal_window),
        f"max_escape_gap_last{terminal_window}": max_gap,
        f"max_escape_gap_deg_last{terminal_window}": float(np.degrees(max_gap)) if np.isfinite(max_gap) else math.nan,
        f"F_esc_last{terminal_window}": _window_mean(fesc, terminal_window),
        f"role_stability_last{terminal_window}": _window_mean(role_stability, terminal_window),
        f"role_instability_last{terminal_window}": 1.0 - _window_mean(role_stability, terminal_window),
    }
    oa_keys = (
        "mean_assigned_pair_blocked_los",
        "mean_path_cache_hit_rate",
        "mean_num_replans_this_step",
        "mean_stale_path_rate",
        "mean_path_endpoint_error",
        "mean_path_min_clearance",
        "mean_path_tracking_error",
        "mean_turn_safety_active_rate",
        "mean_turn_arc_min_clearance",
        "mean_turn_boundary_min_clearance",
        "mean_turn_boundary_unsafe_rate",
        "mean_turn_angle_rad",
        "mean_slot_reachable_rate",
        "mean_mean_time_to_slot",
        "mean_max_time_to_slot",
        "mean_path_clearance_min",
        "mean_path_clearance_mean",
        "mean_path_risk_integral",
        "mean_slot_behind_obstacle_rate",
        "mean_los_blocked_slot_rate",
        "mean_unreachable_slot_rate",
        "mean_fallback_slot_selection_rate",
        "mean_assignment_switch_count",
        "mean_cbf_active",
        "mean_cbf_active_rate",
        "mean_cbf_active_consecutive_steps",
        "mean_cbf_correction_norm",
        "mean_nominal_action_norm",
        "mean_filtered_action_norm",
        "mean_local_obstacle_count",
        "mean_candidate_count",
        "mean_valid_candidate_count",
        "mean_local_planner_blocked",
        "mean_local_planner_time_ms",
        "mean_best_candidate_cost",
        "mean_best_candidate_speed",
        "mean_best_candidate_yaw_rate",
        "mean_min_predicted_clearance",
        "mean_assigned_slot_distance",
        "mean_selected_action_norm",
        "avg_local_obstacle_count",
        "avg_valid_candidate_count",
        "blocked_step_rate",
        "avg_local_planner_time_ms",
        "p95_local_planner_time_ms",
        "mean_cbf_filter_time_ms",
        "avg_decision_ms",
        "p95_decision_ms",
        "max_decision_ms",
        "decision_total_avg_ms",
        "decision_total_p95_ms",
        "decision_total_max_ms",
        "los_check_avg_ms",
        "assignment_avg_ms",
        "path_planning_avg_ms",
        "cbf_filter_avg_ms",
        "path_replan_count",
        "avg_path_length",
        "avg_replan_count",
    )
    for key in oa_keys:
        if key in info:
            row[key] = float(info[key])
    return row


def _build_heuristic_worker(
    cfg_path: Path,
    seed: int,
    *,
    cfg_overrides: dict[str, Any] | None = None,
) -> RolloutWorker:
    cfg = load_config(cfg_path)
    cfg = _deep_update(cfg, cfg_overrides)
    env = build_env_from_config(ROOT / str(cfg["env"]), seed=seed, task_cfg=cfg.get("task", {}))
    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        env.reset(seed=seed)

    if "fixed_ring_apf" in cfg:
        get_actions = make_fixed_ring_apf_get_actions_fn(env, **dict(cfg.get("fixed_ring_apf", {}) or {}))
    elif "pure_pursuit_apf" in cfg:
        get_actions = make_pure_pursuit_apf_get_actions_fn(env, **dict(cfg.get("pure_pursuit_apf", {}) or {}))
    elif "fixed_ring" in cfg:
        get_actions = make_fixed_ring_get_actions_fn(env, **dict(cfg.get("fixed_ring", {}) or {}))
    elif "pure_pursuit" in cfg:
        get_actions = make_pure_pursuit_get_actions_fn(env, **dict(cfg.get("pure_pursuit", {}) or {}))
    elif "oracle_slot" in cfg:
        get_actions = make_oracle_slot_get_actions_fn(env, **dict(cfg.get("oracle_slot", {}) or {}))
    elif "hungarian_slot" in cfg:
        get_actions = make_hungarian_slot_get_actions_fn(env, **dict(cfg.get("hungarian_slot", {}) or {}))
    elif "ot_slot" in cfg:
        get_actions = make_ot_slot_get_actions_fn(env, **dict(cfg.get("ot_slot", {}) or {}))
    elif "trajectory_planner" in cfg:
        get_actions = make_trajectory_planner_get_actions_fn(
            env, **dict(cfg.get("trajectory_planner", {}) or {})
        )
    else:
        raise ValueError(
            f"Heuristic config {cfg_path.relative_to(ROOT)} must define one of "
            "fixed_ring_apf, pure_pursuit_apf, fixed_ring, pure_pursuit, oracle_slot, "
            "hungarian_slot, ot_slot, trajectory_planner."
        )
    return RolloutWorker(env=env, policy=object(), get_actions_fn=get_actions)


def _checkpoint_from_method_cfg(meta: dict[str, Any], cfg: dict[str, Any], train_seed: int) -> Path:
    raw = meta.get("checkpoint")
    if raw is None:
        curriculum = dict(cfg.get("curriculum") or {})
        raw = curriculum.get("final_checkpoint")
    if raw is None:
        results_dir = Path(str(cfg.get("train_results_dir", "")))
        raw = results_dir / "checkpoints" / str(train_seed) / "latest.pt"
    raw_s = str(raw).format(seed=int(train_seed))
    return _resolve_path(raw_s)


def _train_seeds_for_method(meta: dict[str, Any], cfg: dict[str, Any], suite: dict[str, Any]) -> list[int]:
    raw = meta.get("train_seeds", suite.get("train_seeds"))
    if raw is None:
        raw = [cfg.get("seed", 101)]
    if isinstance(raw, (int, str)):
        return [int(raw)]
    return [int(x) for x in raw]


def _build_rl_worker(
    cfg_path: Path,
    train_seed: int,
    *,
    checkpoint_path: Path,
    cfg_overrides: dict[str, Any] | None = None,
) -> RolloutWorker:
    cfg = _deep_update(load_config(cfg_path), cfg_overrides)
    env = build_env_from_config(ROOT / str(cfg["env"]), seed=train_seed, task_cfg=cfg.get("task", {}))
    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        env.reset(seed=train_seed)

    policy_core = build_policy(
        ROOT / str(cfg["model"]),
        env,
        ROOT / str(cfg["algo"]),
    )
    learner, _trainer_kwargs = build_learner(ROOT / str(cfg["algo"]), policy=policy_core)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RL checkpoint not found: {checkpoint_path}")
    load_checkpoint(checkpoint_path, learner)
    policy_core.eval()

    n_actions_for_mac = (
        env.n_actions
        if getattr(policy_core, "action_space_type", "discrete") == "discrete"
        else (getattr(policy_core, "action_dim", None) or 0)
    )
    mac = MAC(obs_dim=env.obs_dim, n_actions=n_actions_for_mac, n_agents=env.num_agents)
    mac.policy = policy_core
    mac.set_test_mode(True)
    return RolloutWorker(env=env, policy=mac)


def _generated_stage_config_path(method: str, seed: int, value: Any) -> Path:
    return (
        ROOT
        / "configs"
        / "generated"
        / "e2_bc_curriculum"
        / method
        / f"seed_{int(seed)}_spacing_{_stage_value_label(value)}.yaml"
    )


def _write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _stage_latest_checkpoint(stage_cfg: dict[str, Any], seed: int) -> Path:
    return _resolve_path(
        Path(str(stage_cfg["train_results_dir"])) / "checkpoints" / str(seed) / "latest.pt"
    )


def _stage_count_value(raw: Any, idx: int, default: int) -> int:
    if isinstance(raw, list):
        if idx < len(raw):
            return int(raw[idx])
        return int(raw[-1])
    if raw is None:
        return int(default)
    return int(raw)


def run_training(
    suite: dict[str, Any],
    methods: list[str],
    *,
    skip_existing: bool = False,
) -> None:
    """Train RL methods declared in the E2 suite.

    For curriculum methods, each generated stage config decreases the configured
    spacing and initializes from the previous stage checkpoint.
    """
    for method in methods:
        meta = dict(suite["methods"][method] or {})
        if str(meta.get("kind", "heuristic")) == "heuristic":
            continue
        if not bool(meta.get("train", False)):
            continue

        base_cfg_path = ROOT / str(meta["config"])
        base_cfg = load_config(base_cfg_path)
        seeds = _train_seeds_for_method(meta, base_cfg, suite)
        curriculum = dict(base_cfg.get("curriculum") or {})
        if not bool(curriculum.get("enabled", False)):
            for train_seed in seeds:
                cfg = deepcopy(base_cfg)
                cfg["seed"] = int(train_seed)
                out_path = _generated_stage_config_path(method, train_seed, "single")
                _write_yaml(out_path, cfg)
                cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--train-config", _relpath(out_path)]
                subprocess.run(cmd, cwd=ROOT, check=True)
            continue

        values = list(curriculum.get("values") or [])
        if not values:
            raise ValueError(f"Curriculum method {method!r} has no values.")
        param = str(curriculum.get("parameter", "task.obstacle_grid_spacing"))
        default_epochs = int(base_cfg.get("num_epochs", 1))
        default_rollout_steps = int(base_cfg.get("rollout_steps", 1024))
        default_eval_episodes = int(base_cfg.get("eval_episodes", 20))
        stage_dir_template = str(
            curriculum.get(
                "stage_results_dir_template",
                f"results/e2_obstacle_field_pyflyt/training/{method}/spacing_{{value}}",
            )
        )

        for train_seed in seeds:
            prev_ckpt: Path | None = None
            for idx, value in enumerate(values):
                stage_cfg = deepcopy(base_cfg)
                stage_cfg["seed"] = int(train_seed)
                stage_cfg["train_results_dir"] = _format_path_template(
                    stage_dir_template,
                    value=value,
                    seed=train_seed,
                )
                _set_nested(stage_cfg, param, float(value))
                stage_cfg["num_epochs"] = _stage_count_value(
                    curriculum.get("stage_epochs", curriculum.get("num_epochs_per_stage")),
                    idx,
                    default_epochs,
                )
                stage_cfg["rollout_steps"] = _stage_count_value(
                    curriculum.get("stage_rollout_steps", curriculum.get("rollout_steps_per_stage")),
                    idx,
                    default_rollout_steps,
                )
                stage_cfg["eval_episodes"] = _stage_count_value(
                    curriculum.get("stage_eval_episodes", curriculum.get("eval_episodes_per_stage")),
                    idx,
                    default_eval_episodes,
                )
                bc_cfg = dict(stage_cfg.get("bc_warmstart") or {})
                if idx == 0:
                    bc_cfg["enabled"] = bool(bc_cfg.get("enabled", True))
                    task_override = dict(bc_cfg.get("task") or {})
                    task_override["obstacle_grid_spacing"] = float(value)
                    bc_cfg["task"] = task_override
                    stage_cfg.pop("initial_checkpoint", None)
                    stage_cfg.pop("resume_from_checkpoint", None)
                else:
                    bc_cfg["enabled"] = False
                    if prev_ckpt is None:
                        raise RuntimeError("Missing previous checkpoint before curriculum stage.")
                    stage_cfg["initial_checkpoint"] = _relpath(prev_ckpt)
                stage_cfg["bc_warmstart"] = bc_cfg

                out_path = _generated_stage_config_path(method, train_seed, value)
                _write_yaml(out_path, stage_cfg)
                latest = _stage_latest_checkpoint(stage_cfg, train_seed)
                if skip_existing and latest.is_file():
                    print(f"[train] skip existing stage checkpoint: {latest}")
                else:
                    print(
                        f"[train] curriculum method={method} seed={train_seed} "
                        f"{param}={value} config={_relpath(out_path)}"
                    )
                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "train.py"),
                        "--train-config",
                        _relpath(out_path),
                    ]
                    subprocess.run(cmd, cwd=ROOT, check=True)
                prev_ckpt = latest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        fv = float(value)
        if np.isfinite(fv):
            vals.append(fv)
    return float(np.mean(vals)) if vals else math.nan


def _finite_std(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        fv = float(value)
        if np.isfinite(fv):
            vals.append(fv)
    return float(np.std(vals)) if vals else math.nan


def summarize(records: list[dict[str, Any]], *, terminal_window: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_keys = [
        "episode_return",
        "episode_len",
        "capture_time_s",
        f"C_cov_last{terminal_window}",
        f"C_col_last{terminal_window}",
        f"D_ang_last{terminal_window}",
        f"max_escape_gap_last{terminal_window}",
        f"max_escape_gap_deg_last{terminal_window}",
        f"F_esc_last{terminal_window}",
        f"role_stability_last{terminal_window}",
        f"role_instability_last{terminal_window}",
        "mean_assigned_pair_blocked_los",
        "mean_path_cache_hit_rate",
        "mean_num_replans_this_step",
        "mean_stale_path_rate",
        "mean_path_endpoint_error",
        "mean_path_min_clearance",
        "mean_path_tracking_error",
        "mean_turn_safety_active_rate",
        "mean_turn_arc_min_clearance",
        "mean_turn_boundary_min_clearance",
        "mean_turn_boundary_unsafe_rate",
        "mean_turn_angle_rad",
        "mean_slot_reachable_rate",
        "mean_mean_time_to_slot",
        "mean_max_time_to_slot",
        "mean_path_clearance_min",
        "mean_path_clearance_mean",
        "mean_path_risk_integral",
        "mean_slot_behind_obstacle_rate",
        "mean_los_blocked_slot_rate",
        "mean_unreachable_slot_rate",
        "mean_fallback_slot_selection_rate",
        "mean_assignment_switch_count",
        "mean_cbf_active",
        "mean_cbf_active_rate",
        "mean_cbf_active_consecutive_steps",
        "mean_cbf_correction_norm",
        "mean_nominal_action_norm",
        "mean_filtered_action_norm",
        "mean_local_obstacle_count",
        "mean_candidate_count",
        "mean_valid_candidate_count",
        "mean_local_planner_blocked",
        "mean_local_planner_time_ms",
        "mean_best_candidate_cost",
        "mean_best_candidate_speed",
        "mean_best_candidate_yaw_rate",
        "mean_min_predicted_clearance",
        "mean_assigned_slot_distance",
        "mean_selected_action_norm",
        "avg_local_obstacle_count",
        "avg_valid_candidate_count",
        "blocked_step_rate",
        "avg_local_planner_time_ms",
        "p95_local_planner_time_ms",
        "mean_cbf_filter_time_ms",
        "avg_decision_ms",
        "p95_decision_ms",
        "max_decision_ms",
        "avg_los_ms",
        "avg_assignment_ms",
        "avg_path_planning_ms",
        "avg_cbf_ms",
        "path_replan_count",
        "avg_path_length",
        "terminal_capture",
        "terminal_obstacle_collision",
        "terminal_inter_agent_collision",
        "terminal_out_of_bounds",
        "terminal_timeout",
        "terminal_other_failure",
        "any_collision",
    ]

    by_seed_rows: list[dict[str, Any]] = []
    seed_keys = sorted({(r.get("variant", ""), r["method"], r["train_seed"]) for r in records})
    for variant, method, seed in seed_keys:
        sub = [
            r for r in records
            if r.get("variant", "") == variant and r["method"] == method and r["train_seed"] == seed
        ]
        paper = ""
        if sub:
            paper = str(sub[0].get("paper_name", method))
        row: dict[str, Any] = {
            "variant": variant,
            "obstacle_grid_spacing": sub[0].get("obstacle_grid_spacing", "") if sub else "",
            "method": method,
            "paper_name": paper or method,
            "seed": seed,
            "train_seed": seed,
            "num_episodes": len(sub),
            "capture_rate": _finite_mean(sub, "captured"),
            "collision_rate": _finite_mean(sub, "collision"),
            "any_collision_rate": _finite_mean(sub, "any_collision"),
            "terminal_capture_rate": _finite_mean(sub, "terminal_capture"),
            "terminal_obstacle_collision_rate": _finite_mean(sub, "terminal_obstacle_collision"),
            "obstacle_collision_rate": _finite_mean(sub, "terminal_obstacle_collision"),
            "terminal_inter_agent_collision_rate": _finite_mean(sub, "terminal_inter_agent_collision"),
            "terminal_out_of_bounds_rate": _finite_mean(sub, "terminal_out_of_bounds"),
            "terminal_timeout_rate": _finite_mean(sub, "terminal_timeout"),
            "terminal_other_failure_rate": _finite_mean(sub, "terminal_other_failure"),
            "obstacle_termination_rate": _finite_mean(sub, "obstacle_termination"),
            "timeout_rate": _finite_mean(sub, "timeout"),
            "out_of_bounds_rate": _finite_mean(sub, "out_of_bounds"),
            "inter_agent_collision_rate": _finite_mean(sub, "terminal_inter_agent_collision"),
            "other_failure_rate": _finite_mean(sub, "terminal_other_failure"),
            "mean_time_to_capture": _finite_mean(sub, "capture_time_s"),
            "bc_action_mse": _finite_mean(sub, "bc_action_mse"),
            "policy_bc_kl": _finite_mean(sub, "policy_bc_kl"),
            "capture_protection_active_rate": _finite_mean(sub, "capture_protection_active_rate"),
            "capture_protection_override_count": _finite_mean(sub, "capture_protection_override_count"),
        }
        for key in metric_keys:
            row[f"mean_{key}"] = _finite_mean(sub, key)
            row[f"std_{key}"] = _finite_std(sub, key)
        by_seed_rows.append(row)

    by_method_rows: list[dict[str, Any]] = []
    for variant, method in sorted({(r.get("variant", ""), r["method"]) for r in records}):
        sub = [r for r in records if r.get("variant", "") == variant and r["method"] == method]
        row = {
            "variant": variant,
            "obstacle_grid_spacing": sub[0].get("obstacle_grid_spacing", "") if sub else "",
            "method": method,
            "paper_name": str(sub[0].get("paper_name", method)) if sub else method,
            "num_train_seeds": len({r["train_seed"] for r in sub}),
            "num_episodes": len(sub),
            "capture_rate": _finite_mean(sub, "captured"),
            "collision_rate": _finite_mean(sub, "collision"),
            "any_collision_rate": _finite_mean(sub, "any_collision"),
            "terminal_capture_rate": _finite_mean(sub, "terminal_capture"),
            "terminal_obstacle_collision_rate": _finite_mean(sub, "terminal_obstacle_collision"),
            "obstacle_collision_rate": _finite_mean(sub, "terminal_obstacle_collision"),
            "terminal_inter_agent_collision_rate": _finite_mean(sub, "terminal_inter_agent_collision"),
            "terminal_out_of_bounds_rate": _finite_mean(sub, "terminal_out_of_bounds"),
            "terminal_timeout_rate": _finite_mean(sub, "terminal_timeout"),
            "terminal_other_failure_rate": _finite_mean(sub, "terminal_other_failure"),
            "obstacle_termination_rate": _finite_mean(sub, "obstacle_termination"),
            "timeout_rate": _finite_mean(sub, "timeout"),
            "out_of_bounds_rate": _finite_mean(sub, "out_of_bounds"),
        }
        for key in metric_keys:
            row[f"mean_{key}"] = _finite_mean(sub, key)
            row[f"std_{key}"] = _finite_std(sub, key)
        by_method_rows.append(row)
    return by_seed_rows, by_method_rows


def run_evaluation(
    suite: dict[str, Any],
    methods: list[str],
    *,
    episodes_override: int | None,
    eval_seeds_override: list[int] | None = None,
    strict_missing_rl_checkpoint: bool = True,
) -> None:
    eval_cfg = dict(suite.get("eval", {}) or {})
    episodes = int(episodes_override or eval_cfg.get("episodes_per_seed", 200))
    base_seed = int(eval_cfg.get("base_seed", 7010))
    terminal_window = int(eval_cfg.get("terminal_window", 30))
    eval_seeds = eval_seeds_override or [
        int(s) for s in suite.get("eval_seeds", eval_cfg.get("holdout_seeds", [201, 202, 203]))
    ]
    result_root = ROOT / str(suite.get("result_root", "results/e2_obstacles_pyflyt"))
    record_dir = result_root / "eval_records"
    scenario = str(suite.get("scenario", "e2_obstacles"))
    variants = list(suite.get("variants") or [])
    if not variants:
        variants = [{"name": scenario}]
    all_records: list[dict[str, Any]] = []

    for variant_raw in variants:
        variant = dict(variant_raw or {})
        variant_name = str(variant.get("name", scenario))
        spacing = variant.get("obstacle_grid_spacing", "")
        variant_overrides = dict(variant.get("config_overrides", {}) or {})
        if "task_overrides" in variant:
            variant_overrides = _deep_update(
                variant_overrides,
                {"task": dict(variant.get("task_overrides") or {})},
            )
        if spacing != "":
            variant_overrides = _deep_update(
                variant_overrides,
                {"task": {"obstacle_grid_spacing": float(spacing)}},
            )

        for method in methods:
            meta = dict(suite["methods"][method] or {})
            base_cfg_path = ROOT / str(meta["config"])
            method_kind = str(meta.get("kind", "heuristic"))
            if method_kind not in ("heuristic", "rl", "learned"):
                raise ValueError(
                    f"E2 benchmark method={method!r} has unsupported kind={method_kind!r}."
                )
            method_overrides_all = dict(variant.get("method_overrides", {}) or {})
            cfg_overrides = _deep_update(
                variant_overrides,
                dict(method_overrides_all.get(method, {}) or {}),
            )
            meta_overrides = dict(meta.get("config_overrides", {}) or {})
            cfg_overrides = _deep_update(cfg_overrides, meta_overrides)

            loaded_method_cfg = _deep_update(load_config(base_cfg_path), cfg_overrides)
            env_cfg = load_config(ROOT / str(loaded_method_cfg["env"]))
            control_hz = float((env_cfg.get("backend", {}) or {}).get("control_hz", 60))
            paper = meta.get("paper_name", method)
            train_seeds = [0]
            if method_kind in ("rl", "learned"):
                train_seeds = _train_seeds_for_method(meta, loaded_method_cfg, suite)

            for train_seed in train_seeds:
                checkpoint_path = None
                if method_kind in ("rl", "learned"):
                    checkpoint_path = _checkpoint_from_method_cfg(meta, loaded_method_cfg, train_seed)
                    if not checkpoint_path.is_file():
                        msg = (
                            f"RL checkpoint not found for method={method} "
                            f"train_seed={train_seed}: {checkpoint_path}"
                        )
                        if strict_missing_rl_checkpoint:
                            raise FileNotFoundError(msg)
                        print(f"[eval] skip {msg}")
                        continue
                    print(f"[eval] method={method} train_seed={train_seed} checkpoint={checkpoint_path}")
                for seed in eval_seeds:
                    if method_kind == "heuristic":
                        worker = _build_heuristic_worker(
                            base_cfg_path,
                            seed=base_seed + seed,
                            cfg_overrides=cfg_overrides,
                        )
                    else:
                        assert checkpoint_path is not None
                        worker = _build_rl_worker(
                            base_cfg_path,
                            train_seed=base_seed + train_seed,
                            checkpoint_path=checkpoint_path,
                            cfg_overrides=cfg_overrides,
                        )
                    method_seed_rows: list[dict[str, Any]] = []
                    print(
                        f"[eval] variant={variant_name} method={method} ({paper}) "
                        f"train_seed={train_seed} eval_seed={seed} episodes={episodes}"
                    )
                    try:
                        for ep in range(episodes):
                            ep_seed = base_seed * 1_000_000 + seed * 10_000 + ep
                            _, info = worker.collect_episode(seed=ep_seed, record_trajectory=True)
                            traj = np.asarray(info.get("trajectory"), dtype=np.float32)
                            rec = {
                                "suite": str(suite.get("suite", "E2")),
                                "scenario": scenario,
                                "variant": variant_name,
                                "obstacle_grid_spacing": spacing,
                                "backend": str(suite.get("backend", "pyflyt")),
                                "method": method,
                                "paper_name": paper,
                                "train_seed": train_seed,
                                "eval_seed_label": seed,
                                "eval_episode": ep,
                                "eval_seed": ep_seed,
                            }
                            if checkpoint_path is not None:
                                rec["checkpoint"] = _relpath(checkpoint_path)
                            rec.update(
                                _episode_metrics(
                                    info=info,
                                    trajectory=traj,
                                    terminal_window=terminal_window,
                                    control_hz=control_hz,
                                )
                            )
                            if "policy_bc_action_mse" in info:
                                rec["bc_action_mse"] = float(info["policy_bc_action_mse"])
                            if "policy_bc_kl" in info:
                                rec["policy_bc_kl"] = float(info["policy_bc_kl"])
                            method_seed_rows.append(rec)
                            all_records.append(rec)
                    finally:
                        try:
                            worker.env.close()
                        except Exception:
                            pass
                    csv_name = f"{variant_name}_{method}_train{train_seed}_eval{seed}_records.csv"
                    _write_csv(record_dir / csv_name, method_seed_rows)

    _write_csv(record_dir / f"{scenario}_all_records.csv", all_records)
    by_seed, by_method = summarize(all_records, terminal_window=terminal_window)
    _write_csv(result_root / f"{scenario}_summary_by_seed.csv", by_seed)
    _write_csv(result_root / f"{scenario}_summary_by_method.csv", by_method)
    timing_rows = []
    for row in by_method:
        timing_rows.append({
            "variant": row.get("variant", ""),
            "obstacle_grid_spacing": row.get("obstacle_grid_spacing", ""),
            "method": row["method"],
            "capture_rate": row.get("capture_rate"),
            "collision_rate": row.get("collision_rate"),
            "obstacle_termination_rate": row.get("obstacle_termination_rate"),
            "avg_decision_ms": row.get("mean_avg_decision_ms"),
            "p95_decision_ms": row.get("mean_p95_decision_ms"),
            "max_decision_ms": row.get("mean_max_decision_ms"),
            "avg_los_ms": row.get("mean_avg_los_ms"),
            "avg_assignment_ms": row.get("mean_avg_assignment_ms"),
            "avg_path_planning_ms": row.get("mean_avg_path_planning_ms"),
            "avg_cbf_ms": row.get("mean_avg_cbf_ms"),
            "path_replan_count": row.get("mean_path_replan_count"),
            "cbf_active_rate": row.get("mean_mean_cbf_active"),
        })
    _write_csv(result_root / f"{scenario}_timing_summary.csv", timing_rows)
    result_root.mkdir(parents=True, exist_ok=True)
    with open(result_root / f"{scenario}_eval_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "suite": suite,
                "num_records": len(all_records),
                "terminal_window": terminal_window,
                "episodes_per_seed": episodes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run E2 obstacle-rich PyFlyt benchmark.")
    p.add_argument(
        "--mode",
        choices=["eval", "train", "all"],
        default="eval",
        help="train runs RL curriculum methods; eval writes benchmark CSVs; all does both.",
    )
    p.add_argument(
        "--suite-config",
        type=str,
        default="configs/benchmark/e2_obstacle_field_suite.yaml",
    )
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument(
        "--eval-seeds",
        nargs="*",
        type=int,
        default=None,
        help="Evaluation seeds.",
    )
    p.add_argument(
        "--skip-existing-train",
        action="store_true",
        help="When --mode train/all, skip curriculum stages whose latest.pt already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    suite = load_suite(ROOT / args.suite_config)
    methods = selected_methods(suite, args.methods)

    if args.mode in ("train", "all"):
        run_training(suite, methods, skip_existing=bool(args.skip_existing_train))
    if args.mode in ("eval", "all"):
        run_evaluation(
            suite,
            methods,
            episodes_override=args.episodes,
            eval_seeds_override=args.eval_seeds,
            strict_missing_rl_checkpoint=args.methods is not None,
        )


if __name__ == "__main__":
    main()
