"""Native vector environment manager for Genesis DroneEntity scenes.

This manager presents the same small interface as ``VecEnvManager`` but does
not spawn worker subprocesses.  It builds one Genesis scene with
``scene.build(n_envs=num_envs)`` and keeps the existing 3v1 task/reward code as
per-environment Python state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from marl_uav.envs.backends.base_backend import SimBackendState
from marl_uav.envs.backends.genesis_backend import GenesisBackend
from marl_uav.envs.factories import build_pursuit_task_from_config
from marl_uav.envs.tasks.pursuit_evasion_3v1_task import (
    compute_pursuit_structure_metrics_3v1,
    pursuit_structure_from_cached_metrics,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx2,
    PursuitEvasion3v1TaskEx2State,
)
from marl_uav.envs.vec_env_manager import VecEnvStepResult
from marl_uav.utils.config import load_config
from marl_uav.utils.env_action_bounds import parse_continuous_action_bounds_from_env_cfg


class GenesisVecEnvManager:
    """Batched 3v1 pursuit env backed by Genesis native ``n_envs`` replication."""

    def __init__(
        self,
        *,
        env_cfg_path: Path,
        task_cfg: dict[str, Any] | None,
        num_envs: int,
        seed: int,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")

        cfg = load_config(env_cfg_path)
        backend_name = str(cfg.get("backend", "pyflyt")).lower()
        if backend_name != "genesis":
            raise ValueError("GenesisVecEnvManager can only be used with backend: genesis")

        self.num_envs = int(num_envs)
        self.base_seed = int(seed)
        self.rngs = [np.random.default_rng(self.base_seed + i) for i in range(self.num_envs)]
        self.task = build_pursuit_task_from_config(task_cfg, default_name="pursuit_evasion_3v1_ex1")
        self.task_states: list[Any] = []
        self.step_counts = np.zeros((self.num_envs,), dtype=np.int32)
        self.episode_returns = np.zeros((self.num_envs,), dtype=np.float32)
        self.episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)

        action_space_type = str(cfg.get("action_space", "continuous")).lower()
        if action_space_type not in ("discrete", "continuous"):
            raise ValueError(f"Unsupported action_space={action_space_type!r} for Genesis VecEnv")
        self._action_space_type = action_space_type
        self.num_agents = 3
        self.action_dim = int(cfg.get("action_dim", 4))
        self.n_actions = 7 if self._action_space_type == "discrete" else 0

        action_low, action_high = parse_continuous_action_bounds_from_env_cfg(
            cfg,
            action_space=self._action_space_type,
            action_dim=self.action_dim,
        )
        self.action_low_np = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high_np = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if self._action_space_type == "continuous":
            low = np.broadcast_to(self.action_low_np, (self.num_agents, self.action_dim)).astype(np.float32)
            high = np.broadcast_to(self.action_high_np, (self.num_agents, self.action_dim)).astype(np.float32)
            self._action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self._avail_actions = np.ones((self.num_envs, self.num_agents, self.action_dim), dtype=np.float32)
        else:
            self._action_space = spaces.MultiDiscrete(np.full((self.num_agents,), self.n_actions, dtype=np.int64))
            self._avail_actions = np.ones((self.num_envs, self.num_agents, self.n_actions), dtype=np.float32)

        backend_cfg = dict(cfg.get("backend_config", {}) or {})
        backend_cfg["n_envs"] = self.num_envs
        backend_cfg.setdefault("world_xy", float(getattr(self.task, "world_xy", 2.0)))
        backend_cfg.setdefault("z_min", float(getattr(self.task, "z_min", 0.5)))
        backend_cfg.setdefault("z_max", float(getattr(self.task, "z_max", 2.0)))
        backend_cfg.setdefault("episode_limit", int(getattr(self.task, "episode_limit", 400)))
        backend_cfg.setdefault("num_pursuers", 3)
        backend_cfg.setdefault("num_evaders", 1)
        if self._action_space_type == "continuous" and self.action_dim == 4:
            backend_cfg.setdefault("velocity_low", self.action_low_np)
            backend_cfg.setdefault("velocity_high", self.action_high_np)
        backend_cfg.setdefault("seed", seed)
        self.backend = GenesisBackend(**backend_cfg)

        self._prev_backend_states: list[SimBackendState] = []
        self.obs_dim: int | None = None
        self.state_dim: int | None = None

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
        """Reset every native Genesis replica and return batched obs/state."""
        if seed is not None:
            self.base_seed = int(seed)
            self.rngs = [np.random.default_rng(self.base_seed + i) for i in range(self.num_envs)]

        start_pos, start_orn = self._sample_initial_batch(np.arange(self.num_envs))
        batched_state = self.backend.reset_batched(start_pos, start_orn, seed=self.base_seed)
        self._prev_backend_states = [batched_state.env_state(i) for i in range(self.num_envs)]
        self.step_counts.fill(0)
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)
        obs, state = self._build_obs_state_batch(self._prev_backend_states)
        return obs, state, self._avail_actions.copy(), {}

    def step(self, actions: np.ndarray) -> VecEnvStepResult:
        """Step the native Genesis batch once and autoreset completed envs."""
        t0 = time.perf_counter()
        actions = np.asarray(actions)
        setpoints = np.zeros((self.num_envs, self.backend.num_agents, 4), dtype=np.float32)

        for env_idx in range(self.num_envs):
            setpoints[env_idx] = self.task.action_to_setpoint(
                actions[env_idx],
                self._prev_backend_states[env_idx],
                self.task_states[env_idx],
                action_space_type=self._action_space_type,
                action_dim=self.action_dim,
            )
        t_after_action = time.perf_counter()
        batched_state = self.backend.step_batched(setpoints)
        t_after_backend = time.perf_counter()

        next_backend_states = [batched_state.env_state(i) for i in range(self.num_envs)]
        rewards = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=np.bool_)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        infos_list: list[dict[str, Any]] = []
        obs_rows: list[np.ndarray] = []
        state_rows: list[np.ndarray] = []
        gae_obs_rows: list[np.ndarray] = []
        gae_state_rows: list[np.ndarray] = []

        for env_idx in range(self.num_envs):
            task_state = self.task_states[env_idx]
            prev_captured = bool(getattr(task_state, "captured", False))
            rewards[env_idx] = self.task.compute_rewards(
                self._prev_backend_states[env_idx],
                next_backend_states[env_idx],
                task_state,
            )
            self.step_counts[env_idx] += 1
            self.episode_returns[env_idx] += float(np.sum(rewards[env_idx]))
            self.episode_lengths[env_idx] += 1
            term, trunc = self.task.compute_terminated_truncated(
                next_backend_states[env_idx],
                task_state,
                int(self.step_counts[env_idx]),
            )
            terminated[env_idx] = bool(term)
            truncated[env_idx] = bool(trunc)
            terminal_obs = self.task.build_obs(next_backend_states[env_idx], task_state)
            terminal_state = self.task.build_state(next_backend_states[env_idx], task_state)
            info = self._pursuit_info(
                next_backend_states[env_idx],
                env_idx=env_idx,
                terminated=bool(term),
                truncated=bool(trunc),
                prev_captured=prev_captured,
            )
            infos_list.append(info)
            gae_obs_rows.append(terminal_obs)
            gae_state_rows.append(terminal_state)
            obs_rows.append(terminal_obs)
            state_rows.append(terminal_state)

        t_after_task = time.perf_counter()
        dones = np.logical_or(terminated, truncated)
        done_indices = np.flatnonzero(dones)
        if done_indices.size:
            reset_pos, reset_orn = self._sample_initial_batch(done_indices)
            reset_batched = self.backend.reset_envs(done_indices, reset_pos, reset_orn)
            for local_idx, env_idx in enumerate(done_indices):
                reset_state = reset_batched.env_state(int(env_idx))
                self._prev_backend_states[int(env_idx)] = reset_state
                self.step_counts[int(env_idx)] = 0
                self.episode_returns[int(env_idx)] = 0.0
                self.episode_lengths[int(env_idx)] = 0
                obs_rows[int(env_idx)] = self.task.build_obs(reset_state, self.task_states[int(env_idx)])
                state_rows[int(env_idx)] = self.task.build_state(reset_state, self.task_states[int(env_idx)])
        for env_idx in np.flatnonzero(~dones):
            self._prev_backend_states[int(env_idx)] = next_backend_states[int(env_idx)]

        t_after_reset = time.perf_counter()
        infos = self._batch_infos(infos_list)
        infos["timing"] = {
            "total_s": np.full((self.num_envs,), t_after_reset - t0, dtype=np.float32),
            "action_to_setpoint_s": np.full((self.num_envs,), t_after_action - t0, dtype=np.float32),
            "backend_step_s": np.full((self.num_envs,), t_after_backend - t_after_action, dtype=np.float32),
            "compute_rewards_s": np.full((self.num_envs,), t_after_task - t_after_backend, dtype=np.float32),
            "compute_done_s": np.zeros((self.num_envs,), dtype=np.float32),
            "build_obs_state_s": np.zeros((self.num_envs,), dtype=np.float32),
            "build_info_s": np.zeros((self.num_envs,), dtype=np.float32),
        }
        infos["_timing"] = np.ones((self.num_envs,), dtype=np.bool_)

        return VecEnvStepResult(
            obs=np.asarray(obs_rows, dtype=np.float32),
            state=np.asarray(state_rows, dtype=np.float32),
            avail_actions=self._avail_actions.copy(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            dones=dones,
            infos=infos,
            gae_next_obs=np.asarray(gae_obs_rows, dtype=np.float32),
            gae_next_state=np.asarray(gae_state_rows, dtype=np.float32),
        )

    def set_training_progress(self, *, epoch: int, num_epochs: int) -> list[Any]:
        if hasattr(self.task, "set_training_progress"):
            value = self.task.set_training_progress(epoch=epoch, num_epochs=num_epochs)
            return [value for _ in range(self.num_envs)]
        return [None for _ in range(self.num_envs)]

    def close(self) -> None:
        self.backend.close()

    def _sample_initial_batch(self, env_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        start_pos = np.zeros((len(env_indices), self.backend.num_agents, 3), dtype=np.float32)
        start_orn = np.zeros_like(start_pos)
        for local_idx, env_idx in enumerate(env_indices):
            pos, orn, task_state = self.task.sample_initial_conditions(
                self.backend.num_agents,
                self.rngs[int(env_idx)],
            )
            start_pos[local_idx] = pos
            start_orn[local_idx] = orn
            if len(self.task_states) < self.num_envs:
                self.task_states.append(task_state)
            else:
                self.task_states[int(env_idx)] = task_state
        return start_pos, start_orn

    def _build_obs_state_batch(self, states: list[SimBackendState]) -> tuple[np.ndarray, np.ndarray]:
        obs = []
        state = []
        for env_idx, backend_state in enumerate(states):
            obs.append(self.task.build_obs(backend_state, self.task_states[env_idx]))
            state.append(self.task.build_state(backend_state, self.task_states[env_idx]))
        obs_arr = np.asarray(obs, dtype=np.float32)
        state_arr = np.asarray(state, dtype=np.float32)
        self.obs_dim = int(obs_arr.shape[-1])
        self.state_dim = int(state_arr.shape[-1])
        return obs_arr, state_arr

    def _pursuit_info(
        self,
        backend_state: SimBackendState,
        *,
        env_idx: int,
        terminated: bool,
        truncated: bool,
        prev_captured: bool,
    ) -> dict[str, Any]:
        task_state = self.task_states[env_idx]
        lin_pos = backend_state.states[:, 3, :]
        pursuer_pos = lin_pos[task_state.pursuer_ids]
        evader_pos = lin_pos[task_state.evader_id]
        dists = np.linalg.norm(pursuer_pos - evader_pos[None, :], axis=1).astype(np.float32)
        captured = bool(getattr(task_state, "captured", False))
        newly_captured = bool(captured and not prev_captured)
        p_oob_mask = self.task._get_oob_mask(pursuer_pos)
        num_p_oob = int(np.sum(p_oob_mask))
        pursuer_oob = bool(num_p_oob >= 1)
        too_many_pursuers_oob = bool(
            num_p_oob >= int(getattr(self.task, "max_pursuers_oob_before_terminate", 1))
        )
        evader_oob = bool(self.task._get_oob_mask(evader_pos[None, :])[0])
        out_of_bounds = bool(
            np.any(np.abs(lin_pos[:, :2]) > getattr(self.task, "world_xy", 5.0) * 1.2)
            or np.any((lin_pos[:, 2] < 0.1) | (lin_pos[:, 2] > getattr(self.task, "z_max", 2.0) * 1.5))
        )
        has_collision = bool(np.any(backend_state.contact_array))
        latest_struct = np.asarray(
            getattr(task_state, "latest_structure_metrics", None),
            dtype=np.float32,
        ).reshape(-1)
        if latest_struct.shape[0] == 3:
            pursuit_structure = pursuit_structure_from_cached_metrics(
                float(latest_struct[0]),
                float(latest_struct[1]),
                float(latest_struct[2]),
            )
        else:
            pursuit_structure = compute_pursuit_structure_metrics_3v1(pursuer_pos, evader_pos)

        pursuer_obstacle_hit = False
        obstacle_terminated = False
        if isinstance(self.task, PursuitEvasion3v1TaskEx2) and isinstance(task_state, PursuitEvasion3v1TaskEx2State):
            hit_mask = self.task._pursuer_obstacle_collision_mask(pursuer_pos, task_state)
            pursuer_obstacle_hit = bool(np.any(hit_mask))
            obstacle_terminated = bool(terminated and pursuer_obstacle_hit and not newly_captured)

        is_success = bool(terminated and captured)
        timeout = bool(truncated and not (captured or too_many_pursuers_oob or evader_oob))
        termination_reason = "running"
        if terminated:
            if is_success:
                termination_reason = "capture"
            elif obstacle_terminated or has_collision:
                termination_reason = "collision"
            elif too_many_pursuers_oob:
                termination_reason = "pursuer_oob"
            elif evader_oob:
                termination_reason = "evader_oob"
            elif out_of_bounds:
                termination_reason = "out_of_bounds"
            else:
                termination_reason = "terminated"
        elif truncated:
            termination_reason = "timeout" if not is_success else "truncated_success"

        return {
            "all_reached": captured,
            "is_success": is_success,
            "out_of_bounds": out_of_bounds,
            "has_collision": has_collision,
            "mean_goal_distance": float(np.mean(dists)),
            "reward_progress": 0.0,
            "reward_time_penalty": -float(getattr(self.task, "time_penalty", 0.0)) * self.num_agents,
            "reward_reach_bonus": 0.0,
            "reward_collision_penalty": -float(self.num_agents) if has_collision else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": termination_reason,
            "episode_return": float(self.episode_returns[env_idx]),
            "episode_len": int(self.episode_lengths[env_idx]),
            "capture": captured,
            "oob": bool(pursuer_oob or evader_oob or out_of_bounds),
            "captured": captured,
            "capture_step": int(self.step_counts[env_idx]) if newly_captured else -1,
            "pursuer_oob": pursuer_oob,
            "timeout": timeout,
            "pursuit_structure": pursuit_structure,
            "metrics": {"pursuit_structure": pursuit_structure},
            "pursuer_obstacle_hit": pursuer_obstacle_hit,
            "obstacle_terminated": obstacle_terminated,
        }

    def _batch_infos(self, infos: list[dict[str, Any]]) -> dict[str, Any]:
        keys: set[str] = set()
        for info in infos:
            keys.update(info.keys())
        out: dict[str, Any] = {}
        for key in keys:
            values = [info.get(key) for info in infos]
            if all(isinstance(v, dict) for v in values):
                subkeys: set[str] = set()
                for v in values:
                    subkeys.update(v.keys())
                out[key] = {
                    subkey: np.asarray([v.get(subkey) for v in values], dtype=object)
                    for subkey in subkeys
                }
                continue
            first = next((v for v in values if v is not None), None)
            if isinstance(first, (bool, np.bool_)):
                out[key] = np.asarray(values, dtype=np.bool_)
            elif isinstance(first, (int, np.integer)):
                out[key] = np.asarray(values, dtype=np.int64)
            elif isinstance(first, (float, np.floating)):
                out[key] = np.asarray(values, dtype=np.float32)
            else:
                out[key] = np.asarray(values, dtype=object)
        return out
