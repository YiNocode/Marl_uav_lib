"""Batch data structures for on-policy training (IPPO, MAPPO)."""

from __future__ import annotations

from typing import Any

import numpy as np


class Batch:
    """Container for batched transition data (legacy / generic)."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _add_batch_dim(x: Any) -> Any:
    """(T, ...) -> (1, T, ...); (B, T, ...) unchanged."""
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim == 0:
        return a
    if a.ndim == 1:
        return a[np.newaxis, ...]  # (T,) -> (1, T)
    if a.ndim == 2:
        return a[np.newaxis, ...]  # (T, N) -> (1, T, N)
    if a.ndim == 3:
        return a[np.newaxis, ...]  # (T, N, D) -> (1, T, N, D)
    if a.ndim >= 4:
        return a  # assume already (B, T, ...)
    return a


def _ensure_4d_obs(obs: Any) -> np.ndarray:
    """obs -> [B, T, N, O]. (T, N, O) -> (1, T, N, O)."""
    a = np.asarray(obs, dtype=np.float32)
    if a.ndim == 3:
        return a[np.newaxis, ...]
    return a


def _ensure_3d_state(x: Any) -> np.ndarray | None:
    """state/next_state -> [B, T, S]. (T, S) -> (1, T, S)."""
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 2:
        return a[np.newaxis, ...]
    return a


def _default_agent_masks(B: int, T: int, N: int) -> np.ndarray:
    """All agents active: [B, T, N] ones."""
    return np.ones((B, T, N), dtype=np.float32)


def _default_step_masks(B: int, T: int) -> np.ndarray:
    """All steps valid: [B, T] ones."""
    return np.ones((B, T), dtype=np.float32)


def _default_avail_actions(B: int, T: int, N: int, A: int) -> np.ndarray:
    """All actions available: [B, T, N, A] ones."""
    return np.ones((B, T, N, A), dtype=np.float32)


class EpisodeBatch:
    """On-policy episode batch，统一形状以稳定支持 MAPPO / IPPO。

    形状约定（B=episode 批大小，T=时间步，N=智能体数，O=obs 维，S=state 维，A=动作数）：
        obs:           [B, T, N, O]
        actions:       [B, T, N]
        rewards:       [B, T, N] 或 [B, T]（若为全局 reward 则 (B,T)）
        dones:         [B, T]
        masks:         [B, T] — 步有效掩码（1=有效，0=padding）
        log_probs:     [B, T, N]
        values:        [B, T, N] 或 [B, T]（中心化 critic 时为 (B,T)）
        advantages:    [B, T, N]
        returns:       [B, T, N] 或 [B, T]
        state:         [B, T, S] — 当前步全局 state
        next_state:    [B, T, S] — 执行当前步动作后的全局 state（与 state 对齐为 [B,T,S]）
        agent_masks:   [B, T, N] — 智能体存活/有效掩码（1=有效，0=无效）
        avail_actions: [B, T, N, A] — 可用动作掩码（0/1），可选

    单条 episode（B=1）时，可从 (T,...) 自动扩成 (1,T,...)。
    """

    def __init__(self, **kwargs: Any) -> None:
        # 必选：来自 rollout / get_episode
        obs = kwargs.get("obs")
        actions = kwargs.get("actions")
        rewards = kwargs.get("rewards")
        dones = kwargs.get("dones")

        if obs is None:
            raise ValueError("EpisodeBatch requires 'obs'.")
        if actions is None:
            raise ValueError("EpisodeBatch requires 'actions'.")
        if rewards is None:
            raise ValueError("EpisodeBatch requires 'rewards'.")
        if dones is None:
            raise ValueError("EpisodeBatch requires 'dones'.")

        obs = _ensure_4d_obs(obs)
        B, T, N, O = obs.shape

        self.obs = obs
        self.actions = _add_batch_dim(actions)
        self.actions = np.asarray(self.actions)
        # 连续动作: (B, T, N, action_dim) 保持 float32；离散 (B, T, N) 转为 int64
        if self.actions.ndim == 3:
            self.actions = self.actions.astype(np.int64)
        elif self.actions.ndim == 4:
            self.actions = self.actions.astype(np.float32)
        else:
            self.actions = self.actions.astype(np.int64)

        self.rewards = _add_batch_dim(rewards)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.dones = _add_batch_dim(dones)
        self.dones = np.asarray(self.dones, dtype=np.float32)

        # 步有效掩码（变长时 padding=0）
        masks = kwargs.get("masks")
        if masks is None:
            masks = _default_step_masks(B, T)
        else:
            masks = _add_batch_dim(masks)
            masks = np.asarray(masks, dtype=np.float32)
        self.masks = masks

        # 可选：GAE 前由 rollout 提供
        log_probs = kwargs.get("log_probs")
        values = kwargs.get("values")
        if log_probs is not None:
            log_probs = _add_batch_dim(log_probs)
            self.log_probs = np.asarray(log_probs, dtype=np.float32)
        else:
            self.log_probs = None
        if values is not None:
            values = _add_batch_dim(values)
            self.values = np.asarray(values, dtype=np.float32)
        else:
            self.values = None

        # GAE 后由 trainer 填入
        advantages = kwargs.get("advantages")
        returns = kwargs.get("returns")
        if advantages is not None:
            advantages = _add_batch_dim(advantages)
            self.advantages = np.asarray(advantages, dtype=np.float32)
        else:
            self.advantages = None
        if returns is not None:
            returns = _add_batch_dim(returns)
            self.returns = np.asarray(returns, dtype=np.float32)
        else:
            self.returns = None

        # MAPPO：全局 state
        state = kwargs.get("state")
        next_state = kwargs.get("next_state")
        if state is not None and next_state is not None:
            self.state = _ensure_3d_state(state)
            self.next_state = _ensure_3d_state(next_state)
        else:
            # 未提供时用 obs/next_obs 展平（与 toy_uav get_state 一致）
            self.state = obs.reshape(B, T, -1).astype(np.float32)
            next_obs = kwargs.get("next_obs")
            if next_obs is not None:
                next_obs = _ensure_4d_obs(next_obs)
                self.next_state = next_obs.reshape(B, T, -1).astype(np.float32)
            else:
                self.next_state = self.state

        # MAPPO：智能体掩码与可用动作
        agent_masks = kwargs.get("agent_masks")
        if agent_masks is None:
            agent_masks = _default_agent_masks(B, T, N)
        else:
            agent_masks = _add_batch_dim(agent_masks)
            agent_masks = np.asarray(agent_masks, dtype=np.float32)
        self.agent_masks = agent_masks

        avail_actions = kwargs.get("avail_actions")
        if avail_actions is not None:
            avail_actions = _add_batch_dim(avail_actions)
            self.avail_actions = np.asarray(avail_actions, dtype=np.float32)
        else:
            # 无可用动作信息时置为 None，policy 侧按“全可用”处理
            self.avail_actions = None
        self._B, self._T, self._N = B, T, N

    @property
    def batch_size(self) -> int:
        return int(self._B)

    @property
    def seq_len(self) -> int:
        return int(self._T)

    @property
    def num_agents(self) -> int:
        return int(self._N)
