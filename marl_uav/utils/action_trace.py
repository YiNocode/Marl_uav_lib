"""Collect and export single-pursuer action traces for hardware / PX4 replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ACTION_TRACE_FORMAT = "marl_uav.pursuer_action_trace"
ACTION_TRACE_VERSION = 1


@dataclass
class PursuerActionTrace:
    """One episode of actions for a single pursuer."""

    meta: dict[str, Any]
    episode: dict[str, Any]
    control: dict[str, Any]
    initial_state: dict[str, Any]
    steps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ACTION_TRACE_FORMAT,
            "version": ACTION_TRACE_VERSION,
            "meta": self.meta,
            "episode": self.episode,
            "control": self.control,
            "initial_state": self.initial_state,
            "steps": self.steps,
        }

    def save_json(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out


def _resolve_step_dt(env_cfg: dict[str, Any]) -> float:
    backend = dict(env_cfg.get("backend", {}) or {})
    control_hz = float(backend.get("control_hz", 60))
    return 1.0 / max(control_hz, 1e-6)


def _scale_pursuer_setpoints(task: Any, raw_action: np.ndarray) -> np.ndarray:
    scaled = task._scale_continuous_pursuer_actions(raw_action[None, :])
    clipped = task._clip_pursuer_setpoints(scaled)
    return np.asarray(clipped[0], dtype=np.float32)


def collect_pursuer_action_trace(
    env: Any,
    *,
    get_actions_fn: Callable[[Any, Any, Any], np.ndarray],
    seed: int,
    pursuer_index: int = 0,
    meta: dict[str, Any] | None = None,
    env_cfg: dict[str, Any] | None = None,
) -> PursuerActionTrace:
    """Roll out one episode and record one pursuer's policy actions and setpoints."""
    if pursuer_index < 0 or pursuer_index >= 3:
        raise ValueError(f"pursuer_index must be 0..2, got {pursuer_index}")

    obs_state, _ = env.reset(seed=seed)
    obs = obs_state["obs"]
    state = obs_state["state"]
    avail_actions = env.get_avail_actions()

    if env.prev_backend_state is None or env.task_state is None:
        raise RuntimeError("Environment reset did not initialize backend/task state.")

    backend = env.prev_backend_state
    task = env.task
    pursuer_ids = np.asarray(env.task_state.pursuer_ids, dtype=np.int64).reshape(3)
    backend_id = int(pursuer_ids[pursuer_index])

    lin_pos0 = np.asarray(backend.states[:, 3, :], dtype=np.float32)
    lin_vel0 = np.asarray(backend.states[:, 2, :], dtype=np.float32)
    ang_pos0 = np.asarray(backend.states[:, 1, :], dtype=np.float32)

    dt_s = _resolve_step_dt(dict(env_cfg or {}))
    control_hz = 1.0 / dt_s

    action_low = np.asarray(env.action_low_np, dtype=np.float32).reshape(-1).tolist()
    action_high = np.asarray(env.action_high_np, dtype=np.float32).reshape(-1).tolist()

    initial_state = {
        "backend_agent_id": backend_id,
        "position_m": lin_pos0[backend_id].tolist(),
        "velocity_mps": lin_vel0[backend_id].tolist(),
        "orientation_rpy_rad": ang_pos0[backend_id].tolist(),
        "frame": "sim_world_enu",
        "frame_note": (
            "Same ground frame as PyFlyt ex1: +x/+y horizontal, +z up. "
            "Convert to PX4 NED before offboard upload if your stack expects NED."
        ),
    }

    control = {
        "control_hz": control_hz,
        "dt_s": dt_s,
        "action_space": "continuous",
        "action_labels": ["vx_cmd", "vy_cmd", "yaw_rate_cmd", "vz_cmd"],
        "setpoint_labels": ["vx_mps", "vy_mps", "yaw_rate_rads", "vz_mps"],
        "action_low": action_low,
        "action_high": action_high,
        "action_note": (
            "Policy/controller output before task scaling. "
            "Divide xy by continuous_action_xy_ref and z by continuous_action_z_ref to saturate."
        ),
        "setpoint_note": (
            "Physical velocity commands sent to PyFlyt after task._scale_continuous_pursuer_actions. "
            "Use these for PX4 velocity offboard replay."
        ),
        "continuous_action_xy_ref": float(getattr(task, "continuous_action_xy_ref", 0.25)),
        "continuous_action_yaw_ref": float(getattr(task, "continuous_action_yaw_ref", 0.25)),
        "continuous_action_z_ref": float(getattr(task, "continuous_action_z_ref", 0.15)),
        "pursuer_speed_xy_mps": float(getattr(task, "pursuer_speed_xy", 0.25)),
        "pursuer_speed_z_mps": float(getattr(task, "pursuer_speed_z", 0.30)),
    }

    steps: list[dict[str, Any]] = []
    episode_return = 0.0
    step_idx = 0
    capture = False

    while True:
        actions = np.asarray(get_actions_fn(obs, state, avail_actions), dtype=np.float32)
        if actions.ndim != 2 or actions.shape != (3, int(env.action_dim)):
            raise ValueError(f"Expected actions shape (3, {env.action_dim}), got {actions.shape}")

        policy_action = actions[pursuer_index].copy()
        setpoint = _scale_pursuer_setpoints(task, policy_action)

        pos_before = np.asarray(env.prev_backend_state.states[:, 3, :], dtype=np.float32)
        yaw_before = float(
            np.asarray(env.prev_backend_state.states[:, 1, :], dtype=np.float32)[backend_id, 2]
        )

        steps.append(
            {
                "step": step_idx,
                "t_s": round(step_idx * dt_s, 6),
                "policy_action": policy_action.tolist(),
                "setpoint": setpoint.tolist(),
                "position_m": pos_before[backend_id].tolist(),
                "yaw_rad": yaw_before,
            }
        )

        next_obs_state, rewards, terminated, truncated, info = env.step(actions)
        episode_return += float(np.sum(rewards))
        step_idx += 1

        if bool(info.get("capture", False)):
            capture = True

        if terminated or truncated:
            backend_after = env.prev_backend_state
            if backend_after is not None:
                pos_after = np.asarray(backend_after.states[:, 3, :], dtype=np.float32)
                yaw_after = float(np.asarray(backend_after.states[:, 1, :], dtype=np.float32)[backend_id, 2])
                steps.append(
                    {
                        "step": step_idx,
                        "t_s": round(step_idx * dt_s, 6),
                        "event": "terminal",
                        "position_m": pos_after[backend_id].tolist(),
                        "yaw_rad": yaw_after,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
            break

        obs = next_obs_state["obs"]
        state = next_obs_state["state"]
        avail_actions = env.get_avail_actions()

    episode = {
        "seed": int(seed),
        "pursuer_index": int(pursuer_index),
        "backend_agent_id": backend_id,
        "episode_len": int(step_idx),
        "episode_return": float(episode_return),
        "capture": bool(capture),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }

    return PursuerActionTrace(
        meta=dict(meta or {}),
        episode=episode,
        control=control,
        initial_state=initial_state,
        steps=steps,
    )
