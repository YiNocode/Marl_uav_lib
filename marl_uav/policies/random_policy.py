"""Random policy for testing rollout."""
from __future__ import annotations

import numpy as np
from typing import Any

from marl_uav.policies.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """随机动作策略，用于测试 rollout 与 env。支持 avail_actions 掩码。"""

    def __init__(
        self,
        num_agents: int,
        n_actions: int,
        seed: int | None = None,
    ) -> None:
        self.num_agents = num_agents
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def forward(
        self,
        obs: list[np.ndarray] | np.ndarray,
        avail_actions: list[np.ndarray] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """根据当前观测（与可选动作掩码）随机采样动作，返回 (num_agents,) int。"""
        if avail_actions is None:
            return self._rng.integers(0, self.n_actions, size=self.num_agents)
        # 在可用动作中采样
        actions = np.zeros(self.num_agents, dtype=np.int64)
        for i in range(self.num_agents):
            mask = np.asarray(avail_actions[i]).flatten()
            avail = np.where(mask > 0.5)[0]
            if len(avail) == 0:
                actions[i] = 0
            else:
                actions[i] = self._rng.choice(avail)
        return actions

    def select_actions(
        self,
        obs: list[np.ndarray] | np.ndarray,
        avail_actions: list[np.ndarray] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """与 forward 一致，供 RolloutWorker 调用。"""
        return self.forward(obs, avail_actions=avail_actions, **kwargs)

    # BasePolicy 新接口（rollout / update 兼容）
    def act(
        self,
        obs: list[np.ndarray] | np.ndarray,
        *,
        state: Any | None = None,
        avail_actions: list[np.ndarray] | np.ndarray | None = None,
        deterministic: bool = False,
        return_entropy: bool = False,
        **kwargs: Any,
    ):
        actions = self.forward(obs, avail_actions=avail_actions, **kwargs)
        # 随机策略没有真实分布信息，这里返回占位张量（形状对齐 actions）
        lp = np.zeros_like(actions, dtype=np.float32)
        ent = np.zeros_like(actions, dtype=np.float32)
        vals = np.zeros_like(actions, dtype=np.float32)
        if return_entropy:
            return actions, lp, vals, ent
        return actions, lp, vals

    def evaluate_actions(
        self,
        obs: list[np.ndarray] | np.ndarray,
        actions: np.ndarray,
        *,
        state: Any | None = None,
        avail_actions: list[np.ndarray] | np.ndarray | None = None,
        **kwargs: Any,
    ):
        # 与 act 类似：返回占位 log_probs/entropy/values，保证 shape 对齐
        act = np.asarray(actions)
        lp = np.zeros_like(act, dtype=np.float32)
        ent = np.zeros_like(act, dtype=np.float32)
        vals = np.zeros_like(act, dtype=np.float32)
        return lp, ent, vals
