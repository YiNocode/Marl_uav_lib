"""Async vector environment manager for high-throughput rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from gymnasium import Env, spaces
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from marl_uav.envs.factories import build_env_from_config
from marl_uav.utils.mp_context import default_vec_env_context


def _box_like(shape: tuple[int, ...], *, low: float = -np.inf, high: float = np.inf) -> spaces.Box:
    return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)


class _VectorEnvAdapter(Env):
    """Wrap a single MARL env into a Gymnasium-compatible worker env."""

    metadata = {}

    def __init__(
        self,
        env_cfg_path: str,
        task_cfg: dict[str, Any] | None,
        seed: int,
    ) -> None:
        super().__init__()
        self._env = build_env_from_config(Path(env_cfg_path), seed=seed, task_cfg=task_cfg)
        obs_dict, _ = self._env.reset(seed=seed)

        obs = np.asarray(obs_dict["obs"], dtype=np.float32)
        state = np.asarray(obs_dict["state"], dtype=np.float32)
        self.num_agents = int(obs.shape[0])
        self.obs_dim = int(obs.shape[1])
        self.state_dim = int(state.shape[0])

        if getattr(self._env, "n_actions", 0) > 0:
            n_actions = int(self._env.n_actions)
            self.action_space = spaces.MultiDiscrete(np.full((self.num_agents,), n_actions, dtype=np.int64))
            avail_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_agents, n_actions),
                dtype=np.float32,
            )
        else:
            action_dim = int(self._env.action_dim)
            low = np.broadcast_to(self._env.action_low_np, (self.num_agents, action_dim)).astype(np.float32)
            high = np.broadcast_to(self._env.action_high_np, (self.num_agents, action_dim)).astype(np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            # Continuous envs still return per-agent all-ones masks with shape (N, action_dim).
            avail_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_agents, action_dim),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(
            {
                "obs": _box_like((self.num_agents, self.obs_dim)),
                "state": _box_like((self.state_dim,)),
                "avail_actions": avail_space,
            }
        )

    def _pack_obs(self, obs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        obs = np.asarray(obs_dict["obs"], dtype=np.float32)
        state = np.asarray(obs_dict["state"], dtype=np.float32)
        avail = self._env.get_avail_actions()
        if avail is None:
            avail_arr = np.ones((self.num_agents, 1), dtype=np.float32)
        else:
            avail_arr = np.asarray(avail, dtype=np.float32)
        return {
            "obs": obs,
            "state": state,
            "avail_actions": avail_arr,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs_dict, info = self._env.reset(seed=seed, options=options)
        return self._pack_obs(obs_dict), info

    def step(self, action):
        obs_dict, rewards, terminated, truncated, info = self._env.step(action)
        return self._pack_obs(obs_dict), np.asarray(rewards, dtype=np.float32), terminated, truncated, info

    def close(self) -> None:
        self._env.close()


def make_vec_env_factory(
    env_cfg_path: Path,
    task_cfg: dict[str, Any] | None,
    seed: int,
) -> Callable[[], Env]:
    """Return a top-level pickle-friendly factory for AsyncVectorEnv workers."""

    env_cfg = str(env_cfg_path.resolve())
    task_cfg_copy = dict(task_cfg or {})

    def _make_env() -> Env:
        return _VectorEnvAdapter(env_cfg, task_cfg_copy, seed)

    return _make_env


@dataclass
class VecEnvStepResult:
    obs: np.ndarray
    state: np.ndarray
    avail_actions: np.ndarray | None
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dones: np.ndarray
    infos: dict[str, Any]
    gae_next_obs: np.ndarray
    gae_next_state: np.ndarray


class VecEnvManager:
    """Manage AsyncVectorEnv workers and normalize batched env I/O."""

    def __init__(
        self,
        *,
        env_cfg_path: Path,
        task_cfg: dict[str, Any] | None,
        num_envs: int,
        seed: int,
        context: str | None = None,
        shared_memory: bool = True,
        copy: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")

        mp_context = context if context is not None else default_vec_env_context()

        self.num_envs = int(num_envs)
        self.base_seed = int(seed)
        env_fns = [
            make_vec_env_factory(env_cfg_path, task_cfg, seed + env_idx)
            for env_idx in range(self.num_envs)
        ]
        self._vec_env = AsyncVectorEnv(
            env_fns,
            shared_memory=shared_memory,
            copy=copy,
            context=mp_context,
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        self.single_observation_space = self._vec_env.single_observation_space
        self.single_action_space = self._vec_env.single_action_space

    @property
    def action_space(self):
        return self._vec_env.single_action_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
        """Collect observations from all workers.

        ``AsyncVectorEnv.reset`` waits until every subprocess has finished resetting,
        so slow envs block the whole vector step (barrier); avoid frequent resets in benchmarks.
        """
        base_seed = self.base_seed if seed is None else int(seed)
        seed_list = [base_seed + env_idx for env_idx in range(self.num_envs)]
        obs_dict, infos = self._vec_env.reset(seed=seed_list)
        obs = np.asarray(obs_dict["obs"], dtype=np.float32)
        state = np.asarray(obs_dict["state"], dtype=np.float32)
        avail = obs_dict.get("avail_actions")
        avail_arr = None if avail is None else np.asarray(avail, dtype=np.float32)
        return obs, state, avail_arr, infos

    def step(self, actions: np.ndarray) -> VecEnvStepResult:
        obs_dict, rewards, terminated, truncated, infos = self._vec_env.step(actions)
        obs = np.asarray(obs_dict["obs"], dtype=np.float32)
        state = np.asarray(obs_dict["state"], dtype=np.float32)
        avail = obs_dict.get("avail_actions")
        avail_arr = None if avail is None else np.asarray(avail, dtype=np.float32)
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        terminated_arr = np.asarray(terminated, dtype=np.bool_)
        truncated_arr = np.asarray(truncated, dtype=np.bool_)
        dones_arr = np.logical_or(terminated_arr, truncated_arr)

        gae_next_obs = obs.copy()
        gae_next_state = state.copy()
        final_obs = infos.get("final_obs")
        if final_obs is not None:
            for env_idx in np.flatnonzero(dones_arr):
                terminal_obs = final_obs[env_idx]
                if terminal_obs is not None:
                    gae_next_obs[env_idx] = np.asarray(terminal_obs["obs"], dtype=np.float32)
                    gae_next_state[env_idx] = np.asarray(terminal_obs["state"], dtype=np.float32)

        return VecEnvStepResult(
            obs=obs,
            state=state,
            avail_actions=avail_arr,
            rewards=rewards_arr,
            terminated=terminated_arr,
            truncated=truncated_arr,
            dones=dones_arr,
            infos=infos,
            gae_next_obs=gae_next_obs,
            gae_next_state=gae_next_state,
        )

    def close(self) -> None:
        self._vec_env.close()
