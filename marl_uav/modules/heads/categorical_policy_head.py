"""Categorical policy head for discrete actions."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.distributions import Categorical

from marl_uav.modules.heads.base_policy_head import BasePolicyHead


class CategoricalPolicyHead(nn.Module, BasePolicyHead):
    """离散动作策略头：features -> logits -> Categorical 分布。

    职责：
    - 仅负责 actor 分支
    - 根据 features 生成 logits
    - 根据 avail_actions 进行非法动作 mask
    - 提供 rollout 用的 forward()
    - 提供 PPO update 用的 evaluate_actions()

    不再负责 value 输出。
    value 应由独立 critic/value head 负责。
    """

    def __init__(self, input_dim: int, n_actions: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.linear = nn.Linear(self.input_dim, self.n_actions)

    def _build_dist(
        self,
        features: torch.Tensor,
        avail_actions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Categorical]:
        """根据特征构造 logits 和 Categorical 分布。

        Args:
            features: (..., input_dim)
            avail_actions: (..., n_actions) 的 0/1 mask，0 表示不可用

        Returns:
            logits: (..., n_actions)
            dist:   Categorical distribution
        """
        logits = self.linear(features)  # (..., n_actions)

        if avail_actions is not None:
            mask = avail_actions.to(dtype=logits.dtype, device=logits.device)
            invalid = mask <= 0.0
            logits = logits.masked_fill(invalid, -1e9)

        dist = Categorical(logits=logits)
        return logits, dist

    def forward(  # type: ignore[override]
        self,
        features: torch.Tensor,
        *,
        avail_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """前向传播并产生动作。

        Args:
            features: (..., input_dim)
            avail_actions: (..., n_actions) 的动作可用性 mask
            deterministic: True 时 argmax，否则 sample

        Returns:
            {
                "actions": (...,) long
                "log_probs": (...,) float
                "entropy": (...,) float
                "logits": (..., n_actions)
                "dist": Categorical
            }
        """
        logits, dist = self._build_dist(features, avail_actions)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return {
            "actions": actions,
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": logits,
            "dist": dist,
        }

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        *,
        avail_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """给定动作，评估其在当前策略下的 log_prob / entropy。

        这个接口专门供 PPO 类算法 update 时使用。

        Args:
            features: (..., input_dim)
            actions: (...,) long
            avail_actions: (..., n_actions) 的动作可用性 mask

        Returns:
            {
                "log_probs": (...,) float
                "entropy": (...,) float
                "logits": (..., n_actions)
                "dist": Categorical
            }
        """
        logits, dist = self._build_dist(features, avail_actions)

        actions = actions.to(device=logits.device, dtype=torch.long)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": logits,
            "dist": dist,
        }