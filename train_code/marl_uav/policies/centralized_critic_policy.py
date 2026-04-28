"""Centralized-critic policy variant (actor obs, critic state).

用于 MAPPO 等 centralized critic 算法：
    - actor 输入：局部 obs
    - critic 输入：全局 state

支持离散动作（Categorical）与连续动作（Gaussian + tanh 仿射，与 GaussianPolicyHead 一致）。
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from marl_uav.modules.encoders.mlp_encoder import MLPEncoder
from marl_uav.modules.heads.categorical_policy_head import CategoricalPolicyHead
from marl_uav.modules.heads.gaussian_policy_head import GaussianPolicyHead
from marl_uav.policies.base_policy import BasePolicy


class CentralizedCriticPolicy(nn.Module, BasePolicy):
    """Policy 变体：actor 用 obs，critic 用 state（不回退 obs）。

    组件拆分（可替换/注入）：
        - actor_encoder
        - critic_encoder
        - policy_head（离散：CategoricalPolicyHead；连续：GaussianPolicyHead）
        - value_head

    形状约定：
        obs:
            - (B, N, O) 或 (N, O)
        state:
            - (S,) 或 (B, S) 或 (B, N, S)
            其中 (S,) / (B, S) 视为 global state，会扩展为 (B, N, S)
        avail_actions（仅离散）:
            - (B, N, A) 或 (N, A)

    输出：
        - values 始终 reshape 为 (B, N)，以对齐 PPO 的 per-agent advantage/value loss 流程。
        - 离散 actions: (B, N)；连续 actions: (B, N, action_dim)
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int | None = None,
        *,
        action_space_type: Literal["discrete", "continuous"] = "discrete",
        action_dim: int | None = None,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
        encoder_hidden_dims: tuple[int, ...] = (64, 64),
        encoder_output_dim: int | None = None,
        actor_encoder: nn.Module | None = None,
        critic_encoder: nn.Module | None = None,
        policy_head: nn.Module | None = None,
        value_head: nn.Module | None = None,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_space_type = str(action_space_type).lower()
        if self.action_space_type not in ("discrete", "continuous"):
            raise ValueError(
                f"action_space_type must be 'discrete' or 'continuous', got {action_space_type!r}"
            )

        if self.action_space_type == "discrete":
            if n_actions is None:
                raise ValueError("n_actions is required when action_space_type is 'discrete'")
            self.n_actions = int(n_actions)
            self.action_dim: int | None = None
        else:
            if action_dim is None:
                raise ValueError("action_dim is required when action_space_type is 'continuous'")
            self.n_actions = 0
            self.action_dim = int(action_dim)
            if action_low is None or action_high is None:
                raise ValueError(
                    "action_low and action_high are required when action_space_type is 'continuous'"
                )
            self._action_low = list(float(x) for x in action_low)
            self._action_high = list(float(x) for x in action_high)
            if len(self._action_low) != self.action_dim or len(self._action_high) != self.action_dim:
                raise ValueError(
                    f"action_low/high length must equal action_dim={self.action_dim}"
                )

        self.actor_encoder: nn.Module = (
            actor_encoder
            if actor_encoder is not None
            else MLPEncoder(
                input_dim=self.obs_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=encoder_output_dim,
            )
        )
        self.critic_encoder: nn.Module = (
            critic_encoder
            if critic_encoder is not None
            else MLPEncoder(
                input_dim=self.state_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=encoder_output_dim,
            )
        )

        actor_feat_dim = int(
            getattr(self.actor_encoder, "output_dim", encoder_output_dim or encoder_hidden_dims[-1])
        )
        critic_feat_dim = int(
            getattr(self.critic_encoder, "output_dim", encoder_output_dim or encoder_hidden_dims[-1])
        )

        if policy_head is not None:
            self.policy_head: nn.Module = policy_head
        elif self.action_space_type == "discrete":
            self.policy_head = CategoricalPolicyHead(actor_feat_dim, self.n_actions)
        else:
            assert self.action_dim is not None
            self.policy_head = GaussianPolicyHead(
                actor_feat_dim,
                self.action_dim,
                log_std_init=float(log_std_init),
                action_low=self._action_low,
                action_high=self._action_high,
            )

        self.value_head: nn.Module = (
            value_head if value_head is not None else nn.Linear(critic_feat_dim, 1)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_obs(self, obs: Any) -> tuple[torch.Tensor, int, int]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (N,O) -> (1,N,O)
        if x.ndim != 3 or x.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs shape must be (batch, n_agents, {self.obs_dim}) "
                f"or (n_agents, {self.obs_dim}), got {tuple(x.shape)}"
            )
        B, N, _ = x.shape
        return x, B, N

    def _prepare_state(self, state: Any, *, B: int, N: int) -> torch.Tensor:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0).unsqueeze(0)  # (S,) -> (1,1,S)
        if s.ndim == 2:
            s = s.unsqueeze(1)  # (B,S) -> (B,1,S)
        if s.ndim != 3 or s.shape[-1] != self.state_dim:
            raise ValueError(
                f"state shape must be (batch, n_agents, {self.state_dim}) or "
                f"(batch, {self.state_dim}) or ({self.state_dim},), got {tuple(s.shape)}"
            )
        if s.shape[0] != B:
            if s.shape[0] == 1 and B > 1:
                s = s.expand(B, s.shape[1], self.state_dim)
            else:
                raise ValueError(
                    f"Batch size mismatch between obs (B={B}) and state (B={s.shape[0]})."
                )
        if s.shape[1] != N:
            if s.shape[1] == 1:
                s = s.expand(B, N, self.state_dim)
            else:
                raise ValueError(
                    f"Agent dim mismatch between obs (N={N}) and state (N={s.shape[1]})."
                )
        return s

    def _prepare_avail(self, avail_actions: Any | None, *, B: int, N: int) -> torch.Tensor | None:
        if self.action_space_type == "continuous":
            return None
        if avail_actions is None:
            return None
        m = torch.as_tensor(avail_actions, dtype=torch.float32, device=self.device)
        if m.ndim == 2:
            m = m.unsqueeze(0)  # (N,A) -> (1,N,A)
        if m.ndim != 3 or m.shape[1] != N or m.shape[2] != self.n_actions:
            raise ValueError(
                f"avail_actions shape must be (batch, {N}, {self.n_actions}) "
                f"or ({N}, {self.n_actions}), got {tuple(m.shape)}"
            )
        if m.shape[0] != B:
            if m.shape[0] == 1 and B > 1:
                m = m.expand(B, N, self.n_actions)
            else:
                raise ValueError(
                    f"Batch size mismatch between obs (B={B}) and avail_actions (B={m.shape[0]})."
                )
        return m

    def forward(  # type: ignore[override]
        self,
        obs: Any,
        state: Any,
        *,
        avail_actions: Any | None = None,
        deterministic: bool = False,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """前向：返回 actor 输出与 critic 输出。

        Returns:
            actor_out: dict 至少含 actions/log_probs/entropy/logits
            critic_out: dict 至少含 values
        """
        obs_tensor, B, N = self._prepare_obs(obs)  # (B,N,O)
        state_tensor = self._prepare_state(state, B=B, N=N)  # (B,N,S)

        flat_obs = obs_tensor.reshape(B * N, self.obs_dim)
        actor_feat_flat = self.actor_encoder(flat_obs)  # (B*N, F_a)
        mask = self._prepare_avail(avail_actions, B=B, N=N)
        mask_flat = (
            None
            if mask is None
            else mask.reshape(B * N, self.n_actions)
        )
        actor_out_flat = self.policy_head(
            actor_feat_flat,
            avail_actions=mask_flat,
            deterministic=deterministic,
        )

        if self.action_space_type == "continuous":
            assert self.action_dim is not None
            adim = self.action_dim
            actions = actor_out_flat["actions"].reshape(B, N, adim)
            log_probs = actor_out_flat["log_probs"].reshape(B, N)
            entropy = actor_out_flat["entropy"].reshape(B, N)
            logits = actor_out_flat["logits"].reshape(B, N, adim)
        else:
            actions = actor_out_flat["actions"].reshape(B, N)
            log_probs = actor_out_flat["log_probs"].reshape(B, N)
            entropy = actor_out_flat["entropy"].reshape(B, N)
            logits = actor_out_flat["logits"].reshape(B, N, self.n_actions)

        actor_out = {
            "actions": actions,
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": logits,
        }

        flat_state = state_tensor.reshape(B * N, self.state_dim)
        critic_feat_flat = self.critic_encoder(flat_state)
        values_flat = self.value_head(critic_feat_flat)  # (B*N,1) or (B*N,)
        if values_flat.ndim > 1:
            values_flat = values_flat.squeeze(-1)
        values = values_flat.reshape(B, N)

        critic_out = {"values": values}
        return actor_out, critic_out

    def act(
        self,
        obs: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        deterministic: bool = False,
        return_entropy: bool = False,
        **kwargs: Any,
    ):
        """rollout：返回 actions/log_probs/values（可选 entropy）。"""
        if state is None:
            raise ValueError("CentralizedCriticPolicy.act requires `state` for critic.")
        actor_out, critic_out = self.forward(
            obs, state, avail_actions=avail_actions, deterministic=deterministic
        )
        actions = actor_out["actions"]
        log_probs = actor_out["log_probs"]
        values = critic_out["values"]
        if return_entropy:
            return actions, log_probs, values, actor_out["entropy"]
        return actions, log_probs, values

    def evaluate_actions(
        self,
        obs: Any,
        actions: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """update：重算 log_probs/entropy/values。"""
        if state is None:
            raise ValueError("CentralizedCriticPolicy.evaluate_actions requires `state` for critic.")

        obs_tensor, B, N = self._prepare_obs(obs)
        state_tensor = self._prepare_state(state, B=B, N=N)

        flat_obs = obs_tensor.reshape(B * N, self.obs_dim)
        actor_feat_flat = self.actor_encoder(flat_obs)

        if self.action_space_type == "continuous":
            assert self.action_dim is not None
            flat_actions = torch.as_tensor(
                actions, dtype=torch.float32, device=self.device
            ).reshape(B * N, self.action_dim)
            flat_avail = None
        else:
            flat_actions = torch.as_tensor(
                actions, dtype=torch.long, device=self.device
            ).reshape(B * N)
            mask = self._prepare_avail(avail_actions, B=B, N=N)
            flat_avail = None if mask is None else mask.reshape(B * N, self.n_actions)

        if hasattr(self.policy_head, "evaluate_actions"):
            actor_eval = self.policy_head.evaluate_actions(
                actor_feat_flat,
                flat_actions,
                avail_actions=flat_avail,
            )
            new_log_probs_flat = actor_eval["log_probs"]
            entropy_flat = actor_eval["entropy"]
        else:
            # 离散且未注入自定义 head：沿用 logits + Categorical
            if not hasattr(self.policy_head, "linear"):
                raise NotImplementedError(
                    "evaluate_actions requires policy_head.evaluate_actions or `.linear` for logits."
                )
            logits = getattr(self.policy_head, "linear")(actor_feat_flat)
            mask = self._prepare_avail(avail_actions, B=B, N=N)
            if mask is not None:
                mask_flat = mask.reshape(B * N, self.n_actions).to(dtype=logits.dtype)
                invalid = mask_flat <= 0.0
                logits = logits.masked_fill(invalid, -1e9)
            dist = Categorical(logits=logits)
            act_tensor = flat_actions.long()
            new_log_probs_flat = dist.log_prob(act_tensor)
            entropy_flat = dist.entropy()

        new_log_probs = new_log_probs_flat.reshape(B, N)
        entropy = entropy_flat.reshape(B, N)

        flat_state = state_tensor.reshape(B * N, self.state_dim)
        critic_feat_flat = self.critic_encoder(flat_state)
        values_flat = self.value_head(critic_feat_flat)
        if values_flat.ndim > 1:
            values_flat = values_flat.squeeze(-1)
        values = values_flat.reshape(B, N)

        return new_log_probs, entropy, values
