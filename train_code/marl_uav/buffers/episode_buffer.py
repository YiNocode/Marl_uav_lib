"""Episode buffer for on-policy (e.g. MAPPO)."""
from __future__ import annotations

from typing import Any

import numpy as np

from marl_uav.buffers.base_buffer import BaseBuffer


class EpisodeBuffer(BaseBuffer):
    """存储一条 episode 的 transition，用于 on-policy  rollout。"""

    def __init__(self, num_agents: int, obs_dim: int, state_dim: int) -> None:
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        # lists 中元素可以是 ndarray 或 None（如 log_probs / values 可选）
        self._obs: list[Any] = []
        self._state: list[Any] = []
        self._actions: list[Any] = []
        self._rewards: list[Any] = []
        self._next_obs: list[Any] = []
        self._next_state: list[Any] = []
        self._dones: list[Any] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._log_probs: list[Any] = []
        self._values: list[Any] = []
        self._avail_actions: list[Any] = []

    def clear(self) -> None:
        """清空当前 episode。"""
        self._obs.clear()
        self._state.clear()
        self._actions.clear()
        self._rewards.clear()
        self._next_obs.clear()
        self._next_state.clear()
        self._dones.clear()
        self._terminated.clear()
        self._truncated.clear()
        self._log_probs.clear()
        self._values.clear()
        self._avail_actions.clear()

    def add(
        self,
        obs: list[np.ndarray],
        state: np.ndarray,
        actions: np.ndarray,
        rewards: list[float] | np.ndarray,
        next_obs: list[np.ndarray],
        next_state: np.ndarray,
        done: bool,
        *,
        terminated: bool | None = None,
        truncated: bool | None = None,
        log_probs: np.ndarray | list[float] | None = None,
        values: np.ndarray | list[float] | None = None,
        avail_actions: list[np.ndarray] | np.ndarray | None = None,
    ) -> None:
        """添加一步 transition。terminated/truncated 用于 GAE 的 last_value 计算。"""
        self._obs.append(obs)
        self._state.append(state)
        # 连续动作为 float 保持 float32，离散为 int 保持 int64（由 np.asarray 推断）
        self._actions.append(np.asarray(actions))
        self._rewards.append(np.asarray(rewards, dtype=np.float64))
        self._next_obs.append(next_obs)
        self._next_state.append(next_state)
        self._dones.append(done)
        self._terminated.append(terminated if terminated is not None else done)
        self._truncated.append(truncated if truncated is not None else False)
        self._log_probs.append(
            None if log_probs is None else np.asarray(log_probs, dtype=np.float64)
        )
        self._values.append(
            None if values is None else np.asarray(values, dtype=np.float64)
        )
        # avail_actions 保留原始 list[np.ndarray] 形式，便于后续按 (T,N,A) stack
        self._avail_actions.append(avail_actions)

    def __len__(self) -> int:
        return len(self._obs)

    def sample(self, batch_size: int) -> Any:
        """Episode buffer 不按 batch 采样，返回当前整条 episode。"""
        if len(self._obs) == 0:
            return None
        return self.get_episode()

    def get_episode(self) -> dict[str, np.ndarray]:
        """返回当前 episode 的字典，所有数组为 (T, ...)。"""
        if len(self._obs) == 0:
            return {}
        T = len(self._obs)
        # obs: (T, n_agents, obs_dim)
        obs_arr = np.stack(
            [np.stack(self._obs[t], axis=0) for t in range(T)], axis=0
        )
        state_arr = np.stack(self._state, axis=0)
        actions_arr = np.stack(self._actions, axis=0)
        rewards_arr = np.stack(self._rewards, axis=0)
        next_obs_arr = np.stack(
            [np.stack(self._next_obs[t], axis=0) for t in range(T)], axis=0
        )
        next_state_arr = np.stack(self._next_state, axis=0)
        dones_arr = np.array(self._dones, dtype=np.float32)
        terminated_arr = np.array(self._terminated, dtype=np.float32)
        truncated_arr = np.array(self._truncated, dtype=np.float32)
        out: dict[str, np.ndarray] = {
            "obs": obs_arr,
            "state": state_arr,
            "actions": actions_arr,
            "rewards": rewards_arr,
            "next_obs": next_obs_arr,
            "next_state": next_state_arr,
            "dones": dones_arr,
            "terminated": terminated_arr,
            "truncated": truncated_arr,
        }
        # 可选字段：log_probs / values（若所有时间步都有）
        if self._log_probs and all(x is not None for x in self._log_probs):
            logp_arr = np.stack(self._log_probs, axis=0)
            out["log_probs"] = logp_arr
        if self._values and all(x is not None for x in self._values):
            val_arr = np.stack(self._values, axis=0)
            out["values"] = val_arr
        # 可选字段：avail_actions（若所有时间步都有）
        if self._avail_actions and all(x is not None for x in self._avail_actions):
            avail_arr = np.stack(
                [np.stack(self._avail_actions[t], axis=0) for t in range(T)], axis=0
            )  # (T, N, n_actions)
            out["avail_actions"] = avail_arr
        return out

    def get_episode_return(self) -> float:
        """当前 episode 的累计 reward 和（所有 agent 求和）。"""
        if not self._rewards:
            return 0.0
        return float(np.sum(self._rewards))

    def get_episode_length(self) -> int:
        return len(self._obs)
