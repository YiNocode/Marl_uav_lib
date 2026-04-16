"""Actor-critic policy (for MAPPO, IPPO, etc.)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import torch
from torch import nn

from marl_uav.modules.encoders.mlp_encoder import MLPEncoder
from marl_uav.modules.heads.categorical_policy_head import CategoricalPolicyHead
from marl_uav.modules.heads.gaussian_policy_head import GaussianPolicyHead
from marl_uav.policies.base_policy import BasePolicy


class ActorCriticPolicy(nn.Module, BasePolicy):
    """通用多智能体 actor-critic 策略（支持 actor/critic 输入不同）。

    支持离散动作空间（CategoricalPolicyHead）与连续动作空间（GaussianPolicyHead），
    通过 action_space_type 与 n_actions/action_dim 选择。

    约定:
        - 离散：n_actions 为动作数，avail_actions 形状 (batch, n_agents, n_actions)
        - 连续：action_dim 为动作维度，无 avail_actions，动作形状 (batch, n_agents, action_dim)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int | None = None,
        *,
        action_space_type: Literal["discrete", "continuous"] = "discrete",
        action_dim: int | None = None,
        state_dim: int | None = None,
        encoder_hidden_dims: tuple[int, ...] = (64, 64),
        encoder_output_dim: int | None = None,
        actor_encoder: nn.Module | None = None,
        critic_encoder: nn.Module | None = None,
        actor_head: nn.Module | None = None,
        value_head: nn.Module | None = None,
        log_std_init: float = -0.5,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_space_type = str(action_space_type).lower()
        if self.action_space_type not in ("discrete", "continuous"):
            raise ValueError(f"action_space_type must be 'discrete' or 'continuous', got {action_space_type!r}")

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

        self.state_dim = int(state_dim) if state_dim is not None else None

        # encoders
        self.actor_encoder: nn.Module = (
            actor_encoder
            if actor_encoder is not None
            else MLPEncoder(
                input_dim=self.obs_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=encoder_output_dim,
            )
        )
        critic_in_dim = self.state_dim if self.state_dim is not None else self.obs_dim
        self.critic_encoder: nn.Module = (
            critic_encoder
            if critic_encoder is not None
            else MLPEncoder(
                input_dim=critic_in_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=encoder_output_dim,
            )
        )

        actor_feat_dim = int(getattr(self.actor_encoder, "output_dim", encoder_output_dim or encoder_hidden_dims[-1]))
        critic_feat_dim = int(getattr(self.critic_encoder, "output_dim", encoder_output_dim or encoder_hidden_dims[-1]))

        if actor_head is not None:
            self.actor_head: nn.Module = actor_head
        elif self.action_space_type == "discrete":
            self.actor_head = CategoricalPolicyHead(actor_feat_dim, self.n_actions)
        else:
            assert self.action_dim is not None
            if action_low is None or action_high is None:
                al = [-1.0] * self.action_dim
                ah = [1.0] * self.action_dim
            else:
                al = [float(x) for x in action_low]
                ah = [float(x) for x in action_high]
                if len(al) != self.action_dim or len(ah) != self.action_dim:
                    raise ValueError(
                        f"action_low/high 长度须等于 action_dim={self.action_dim}"
                    )
            self.actor_head = GaussianPolicyHead(
                actor_feat_dim,
                self.action_dim,
                log_std_init=log_std_init,
                action_low=al,
                action_high=ah,
            )

        self.value_head: nn.Module = (
            value_head if value_head is not None else nn.Linear(critic_feat_dim, 1)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_obs(
        self, obs: Any
    ) -> Tuple[torch.Tensor, int, int]:
        """将多种 obs 格式统一为 (B, N, obs_dim)。"""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            # (N, D) -> (1, N, D)
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs shape must be (batch, n_agents, {self.obs_dim}) "
                f"or (n_agents, {self.obs_dim}), got {tuple(x.shape)}"
            )
        B, N, _ = x.shape
        return x, B, N

    def _prepare_state(
        self,
        state: Any,
        *,
        B: int,
        N: int,
        state_dim: int,
    ) -> torch.Tensor:
        """将 state 统一为 (B, N, state_dim)。

        支持:
            - (state_dim,) -> (1, 1, state_dim) -> expand(B,N,state_dim)
            - (B, state_dim) -> (B, 1, state_dim) -> expand(B,N,state_dim)
            - (B, N, state_dim) -> unchanged
        """
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0).unsqueeze(0)
        if s.ndim == 2:
            s = s.unsqueeze(1)
        if s.ndim != 3 or s.shape[-1] != state_dim:
            raise ValueError(
                f"state shape must be (batch, n_agents, {state_dim}) or "
                f"(batch, {state_dim}) or ({state_dim},), got {tuple(s.shape)}"
            )
        if s.shape[0] != B:
            if s.shape[0] == 1 and B > 1:
                s = s.expand(B, s.shape[1], state_dim)
            else:
                raise ValueError(
                    f"Batch size mismatch between obs (B={B}) and state (B={s.shape[0]})."
                )
        if s.shape[1] != N:
            if s.shape[1] == 1:
                s = s.expand(B, N, state_dim)
            else:
                raise ValueError(
                    f"Agent dim mismatch between obs (N={N}) and state (N={s.shape[1]})."
                )
        return s

    def _prepare_avail(
        self,
        avail_actions: Any | None,
        B: int,
        N: int,
    ) -> torch.Tensor | None:
        if self.action_space_type == "continuous":
            return None
        if avail_actions is None:
            return None
        mask = torch.as_tensor(avail_actions, dtype=torch.float32, device=self.device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[1] != N or mask.shape[2] != self.n_actions:
            raise ValueError(
                f"avail_actions shape must be (batch, {N}, {self.n_actions}) "
                f"or ({N}, {self.n_actions}), got {tuple(mask.shape)}"
            )
        if mask.shape[0] != B:
            if mask.shape[0] == 1 and B > 1:
                mask = mask.expand(B, N, self.n_actions)
            else:
                raise ValueError(
                    f"Batch size mismatch between obs (B={B}) and avail_actions "
                    f"(B={mask.shape[0]})."
                )
        return mask

    # BasePolicy 接口
    def forward(  # type: ignore[override]
        self,
        obs: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """前向计算：返回 (actor_features, policy_output, values)。

        policy_output 字典至少包含:
            - actions
            - log_probs
            - entropy
            - logits
        """
        obs_tensor, B, N = self._prepare_obs(obs)  # (B, N, O)

        # -------- actor: obs -> actor_encoder -> actor_head --------
        flat_obs = obs_tensor.reshape(B * N, self.obs_dim)
        actor_features_flat = self.actor_encoder(flat_obs)  # (B*N, F_a)
        actor_feat_dim = actor_features_flat.shape[-1]
        actor_features = actor_features_flat.reshape(B, N, actor_feat_dim)

        mask = self._prepare_avail(avail_actions, B, N)
        mask_flat = None if mask is None else mask.view(B * N, self.n_actions)

        pi_out_flat = self.actor_head(
            actor_features_flat,
            avail_actions=mask_flat,
            deterministic=deterministic,
        )

        # -------- critic: state (if provided) else obs --------
        if state is None:
            critic_in = obs_tensor  # (B,N,O)
            critic_in_dim = self.obs_dim
        else:
            if self.state_dim is None:
                raise ValueError("state_dim must be provided in ActorCriticPolicy when using state input.")
            critic_in = self._prepare_state(state, B=B, N=N, state_dim=self.state_dim)  # (B,N,S)
            critic_in_dim = self.state_dim

        flat_critic_in = critic_in.reshape(B * N, critic_in_dim)
        critic_features_flat = self.critic_encoder(flat_critic_in)
        values_flat = self.value_head(critic_features_flat)
        if values_flat.ndim > 1:
            values_flat = values_flat.squeeze(-1)
        values = values_flat.reshape(B, N)

        # -------- reshape policy outputs (离散 (B,N) / 连续 (B,N,action_dim)) --------
        if self.action_space_type == "continuous":
            adim = self.action_dim
            actions = pi_out_flat["actions"].reshape(B, N, adim)
            log_probs = pi_out_flat["log_probs"].reshape(B, N)
            entropy = pi_out_flat["entropy"].reshape(B, N)
            logits = pi_out_flat["logits"].reshape(B, N, adim)
        else:
            actions = pi_out_flat["actions"].reshape(B, N)
            log_probs = pi_out_flat["log_probs"].reshape(B, N)
            entropy = pi_out_flat["entropy"].reshape(B, N)
            logits = pi_out_flat["logits"].reshape(B, N, self.n_actions)

        policy_output = {
            "actions": actions,
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": logits,
        }
        return actor_features, policy_output, values

    def act(
        self,
        obs: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        deterministic: bool = False,
        return_entropy: bool = False,
    ) -> tuple[Any, Any, Any] | tuple[Any, Any, Any, Any]:
        """采样/选择动作。

        Returns:
            actions, log_probs, values 以及（可选）entropy
        """
        _, pi_out, values = self.forward(
            obs, state=state, avail_actions=avail_actions, deterministic=deterministic
        )
        actions = pi_out["actions"]
        log_probs = pi_out["log_probs"]
        entropy = pi_out["entropy"]
        if return_entropy:
            return actions, log_probs, values, entropy
        return actions, log_probs, values

    def evaluate_actions(
            self,
            obs: Any,
            actions: Any,
            *,
            state: Any | None = None,
            avail_actions: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """给定 obs 和 actions，计算新的 log_probs / entropy / values。

        该接口用于 PPO / IPPO / MAPPO 的 update 阶段。

        约定：
        - obs 可被 _prepare_obs 处理为 (B, N, O)
        - state 若提供，则可被 _prepare_state 处理为 critic 输入
        - actions 形状应与 (B, N) 对齐
        - 返回：
            new_log_probs: (B, N)
            entropy:       (B, N)
            values:        (B, N)
        """
        obs_tensor, B, N = self._prepare_obs(obs)  # -> (B, N, O)

        # ===== Actor branch =====
        flat_obs = obs_tensor.reshape(B * N, self.obs_dim)
        actor_features_flat = self.actor_encoder(flat_obs)

        if self.action_space_type == "continuous":
            flat_actions = torch.as_tensor(
                actions, dtype=torch.float32, device=self.device
            ).reshape(B * N, self.action_dim)
            flat_avail = None
        else:
            flat_actions = torch.as_tensor(
                actions, dtype=torch.long, device=self.device
            ).reshape(B * N)
            mask = self._prepare_avail(avail_actions, B, N)
            flat_avail = None if mask is None else mask.reshape(B * N, self.n_actions)

        actor_eval = self.actor_head.evaluate_actions(
            actor_features_flat,
            flat_actions,
            avail_actions=flat_avail,
        )
        new_log_probs_flat = actor_eval["log_probs"]  # (B*N,)
        entropy_flat = actor_eval["entropy"]  # (B*N,)

        # ===== Critic branch =====
        if state is None:
            critic_in = obs_tensor
            critic_in_dim = self.obs_dim
        else:
            if self.state_dim is None:
                raise ValueError(
                    "state_dim must be provided in ActorCriticPolicy when using state input."
                )
            critic_in = self._prepare_state(
                state,
                B=B,
                N=N,
                state_dim=self.state_dim,
            )
            critic_in_dim = self.state_dim

        flat_critic_in = critic_in.reshape(B * N, critic_in_dim)
        critic_features_flat = self.critic_encoder(flat_critic_in)

        values_flat = self.value_head(critic_features_flat)
        if values_flat.ndim > 1:
            values_flat = values_flat.squeeze(-1)

        # ===== Restore shape =====
        new_log_probs = new_log_probs_flat.reshape(B, N)
        entropy = entropy_flat.reshape(B, N)
        values = values_flat.reshape(B, N)

        return new_log_probs, entropy, values

