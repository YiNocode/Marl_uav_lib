"""Multi-Agent Controller (MAC)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from marl_uav.agents.base_agent import BaseAgent
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy


class MAC(BaseAgent, nn.Module):
    """Central controller for homogeneous agents with shared policy.

    当前版本只实现参数共享的同构 agent:
        - 单个 `ActorCriticPolicy`，对所有 agent 共享
        - `obs` / `avail_actions` 统一组织成 (batch, n_agents, obs_dim)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        *,
        device: torch.device | str | None = None,
        encoder_hidden_dims: Sequence[int] = (64, 64),
        encoder_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_agents = int(n_agents)

        self.policy = ActorCriticPolicy(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            encoder_hidden_dims=tuple(int(h) for h in encoder_hidden_dims),
            encoder_output_dim=encoder_output_dim,
        )

        if device is not None:
            self.to(device)

        self._test_mode: bool = False

    @property
    def test_mode(self) -> bool:
        return self._test_mode

    def set_test_mode(self, test: bool = True) -> None:
        """切换 train/test 模式，影响是否使用 deterministic 动作。"""
        self._test_mode = bool(test)
        if test:
            self.eval()
        else:
            self.train()

    # BaseAgent 接口
    def parameters(self):
        return self.policy.parameters()

    def _prepare_obs(self, obs: Any) -> torch.Tensor:
        """将 obs 转为 (B, N, obs_dim)。"""
        # env 返回的是 list[np.ndarray] 时，先 stack 为 np.ndarray 再转 tensor 更高效
        if isinstance(obs, (list, tuple)) and len(obs) > 0 and hasattr(
            obs[0], "shape"
        ):
            obs = np.stack(obs, axis=0)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
        if x.ndim == 2:
            # (N, D) -> (1, N, D)
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[1] != self.n_agents or x.shape[2] != self.obs_dim:
            raise ValueError(
                f"obs must have shape (batch, {self.n_agents}, {self.obs_dim}) or "
                f"({self.n_agents}, {self.obs_dim}), got {tuple(x.shape)}"
            )
        return x

    def _prepare_avail(self, avail_actions: Any | None, B: int) -> Optional[torch.Tensor]:
        if avail_actions is None or self.n_actions == 0:
            # 连续动作时 n_actions=0，不传 avail_actions
            return None
        if isinstance(avail_actions, (list, tuple)) and len(avail_actions) > 0 and hasattr(
            avail_actions[0], "shape"
        ):
            avail_actions = np.stack(avail_actions, axis=0)
        m = torch.as_tensor(
            avail_actions, dtype=torch.float32, device=self.policy.device
        )
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.ndim != 3 or m.shape[1] != self.n_agents or m.shape[2] != self.n_actions:
            raise ValueError(
                f"avail_actions must have shape (batch, {self.n_agents}, {self.n_actions}) or "
                f"({self.n_agents}, {self.n_actions}), got {tuple(m.shape)}"
            )
        if m.shape[0] != B:
            if m.shape[0] == 1 and B > 1:
                m = m.expand(B, self.n_agents, self.n_actions)
            else:
                raise ValueError(
                    f"Batch size mismatch between obs (B={B}) and avail_actions (B={m.shape[0]})."
                )
        return m

    def select_actions(  # type: ignore[override]
        self,
        obs: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        deterministic: bool | None = None,
        return_entropy: bool = False,
    ):
        """为所有智能体选择动作。

        Args:
            obs: (batch, n_agents, obs_dim) 或 (n_agents, obs_dim)
            state: 可选全局 state（centralized critic 用）；不影响 actor，仅用于 critic value 估计
            avail_actions: 可选 (batch, n_agents, n_actions) 或 (n_agents, n_actions)
            return_entropy: 是否额外返回 entropy

        Returns:
            actions: (batch, n_agents)
            log_probs: (batch, n_agents)
            values: (batch, n_agents)
            若 return_entropy=True 还会返回 entropy: (batch, n_agents)
        """
        obs_tensor = self._prepare_obs(obs)
        B = obs_tensor.shape[0]
        mask = self._prepare_avail(avail_actions, B)

        # 若显式传入 deterministic，则以其为准；否则使用 MAC 当前的 test_mode
        det = self._test_mode if deterministic is None else bool(deterministic)

        # rollout 时 env 会提供 state；但对 IPPO（critic=obs）应自动忽略 state 以保持兼容
        use_state = False
        if state is not None and getattr(self.policy, "state_dim", None) is not None:
            use_state = True

        if use_state:
            out = self.policy.act(
                obs_tensor,
                state=state,
                avail_actions=mask,
                deterministic=det,
                return_entropy=return_entropy,
            )
        else:
            out = self.policy.act(
                obs_tensor,
                avail_actions=mask,
                deterministic=det,
                return_entropy=return_entropy,
            )

        if return_entropy:
            actions, log_probs, values, entropy = out  # type: ignore[misc]
            return actions, log_probs, values, entropy
        actions, log_probs, values = out  # type: ignore[misc]
        return actions, log_probs, values

