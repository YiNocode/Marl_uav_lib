"""Base policy interface (actor + critic friendly).

该抽象接口需要能自然表达：actor 与 critic 的输入可能不同。
常见于 MAPPO：actor 用局部 obs；critic 用全局 state（或 state+obs 等）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BasePolicy(ABC):
    """Policy 抽象接口。

    必须支持：
    - act(...): rollout 期间采样/选择动作；critic 可能同时返回 value
    - evaluate_actions(...): update 期间重算给定动作的 log_probs/entropy/values

    其中 `state` 为可选参数，用于支持 centralized critic：
    - actor 输入：obs
    - critic 输入：state（可能与 obs 不同维度）
    """

    @abstractmethod
    def act(
        self,
        obs: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        deterministic: bool = False,
        return_entropy: bool = False,
        **kwargs: Any,
    ) -> Any:
        """用于 rollout：根据 obs（以及可选 state）选择动作，并尽可能返回 log_probs / values。

        Args:
            obs: actor 输入（局部观测），形状由具体实现决定（通常含 agent 维度）。
            state: critic 输入（全局状态），可选；用于 actor/critic 输入不同的场景（如 MAPPO）。
            avail_actions: 可选动作 mask（离散动作常用）。
            deterministic: True 时取 argmax，否则采样。
            return_entropy: True 时额外返回 entropy（便于调试/正则）。

        Returns:
            推荐返回：
              - actions
              - log_probs（与 actions 对齐）
              - values（与 agent 维度对齐的 per-agent value 或 centralized value）
              - 可选 entropy
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self,
        obs: Any,
        actions: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any, Any]:
        """用于 update：给定 obs/actions（以及可选 state）重算 log_probs/entropy/values。

        Returns:
            log_probs, entropy, values
        """
        raise NotImplementedError
