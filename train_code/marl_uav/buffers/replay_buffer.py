"""Replay buffer for off-policy (e.g. MADDPG, QMIX)."""
from __future__ import annotations

from typing import Any

import numpy as np

from marl_uav.buffers.base_buffer import BaseBuffer


class ReplayBuffer(BaseBuffer):
    """Off-policy 经验回放：固定容量、随机采样单步 transition。"""

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
    ) -> None:
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        # 预分配 (capacity, ...)
        self._obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self._state = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self._rewards = np.zeros((capacity, num_agents), dtype=np.float64)
        self._next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self._next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: list[np.ndarray] | np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        rewards: list[float] | np.ndarray,
        next_obs: list[np.ndarray] | np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """添加一步 transition；与 EpisodeBuffer.add 接口一致。"""
        if isinstance(obs, (list, tuple)):
            obs_arr = np.stack(obs, axis=0)
        else:
            obs_arr = np.asarray(obs)
        if isinstance(next_obs, (list, tuple)):
            next_obs_arr = np.stack(next_obs, axis=0)
        else:
            next_obs_arr = np.asarray(next_obs)
        self._obs[self._ptr] = obs_arr
        self._state[self._ptr] = np.asarray(state, dtype=np.float32)
        self._actions[self._ptr] = np.asarray(actions, dtype=np.int64)
        self._rewards[self._ptr] = np.asarray(rewards, dtype=np.float64)
        self._next_obs[self._ptr] = next_obs_arr
        self._next_state[self._ptr] = np.asarray(next_state, dtype=np.float32)
        self._dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_episode(self, episode: dict[str, np.ndarray] | Any) -> None:
        """将一条 episode 的所有 transition 加入 buffer。episode 可为 get_episode() 的返回值或 EpisodeBuffer。"""
        if hasattr(episode, "get_episode"):
            episode = episode.get_episode()
        if not episode:
            return
        T = episode["obs"].shape[0]
        for t in range(T):
            self.add(
                obs=[episode["obs"][t, i] for i in range(self.num_agents)],
                state=episode["state"][t],
                actions=episode["actions"][t],
                rewards=episode["rewards"][t],
                next_obs=[episode["next_obs"][t, i] for i in range(self.num_agents)],
                next_state=episode["next_state"][t],
                done=bool(episode["dones"][t]),
            )

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> dict[str, np.ndarray] | None:
        """均匀随机采样 batch_size 条 transition，返回 dict，数组形状 (batch_size, ...)。"""
        if self._size < batch_size:
            return None
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": self._obs[indices].copy(),
            "state": self._state[indices].copy(),
            "actions": self._actions[indices].copy(),
            "rewards": self._rewards[indices].copy(),
            "next_obs": self._next_obs[indices].copy(),
            "next_state": self._next_state[indices].copy(),
            "dones": self._dones[indices].copy(),
        }
