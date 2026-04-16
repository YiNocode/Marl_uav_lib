"""Gaussian policy head for continuous actions with bounded env-range outputs.

This implementation outputs FINAL environment actions directly, instead of
normalized actions that are scaled inside env.step().

Action flow:
    raw_action ~ Normal(mean, std)
    squashed = tanh(raw_action)              -> [-1, 1]
    env_action = squashed * scale + bias     -> [action_low, action_high]

Key properties:
- rollout stores env_action directly
- env executes env_action directly
- evaluate_actions(features, env_action) computes log_prob consistently
- no extra scaling should happen inside environment step()

Notes:
- entropy returned here uses base Gaussian entropy as an approximation
  (common practical choice in PPO implementations)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn
from torch.distributions import Normal

from marl_uav.modules.heads.base_policy_head import BasePolicyHead


class GaussianPolicyHead(nn.Module, BasePolicyHead):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        *,
        action_low: Sequence[float],
        action_high: Sequence[float],
        log_std_init: float = -1.0,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.eps = float(eps)

        if len(action_low) != self.action_dim or len(action_high) != self.action_dim:
            raise ValueError(
                f"action_low/high length must equal action_dim={self.action_dim}, "
                f"got {len(action_low)} and {len(action_high)}."
            )

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)

        if torch.any(action_high_t <= action_low_t):
            raise ValueError("Each action_high must be strictly greater than action_low.")

        # Buffers so they move with device automatically
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

        self.mean_linear = nn.Linear(self.input_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))

    def _get_log_std(self) -> torch.Tensor:
        return torch.clamp(self.log_std, self.log_std_min, self.log_std_max)

    def _build_base_dist(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Normal]:
        """Build base diagonal Gaussian in raw (pre-tanh) space."""
        mean = self.mean_linear(features)  # (..., action_dim)
        log_std = self._get_log_std()
        if mean.ndim > 1:
            log_std = log_std.expand_as(mean)
        std = log_std.exp()
        dist = Normal(loc=mean, scale=std)
        return mean, log_std, dist

    def _raw_to_env(self, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map raw action -> squashed [-1,1] -> env action range."""
        squashed = torch.tanh(raw_action)
        env_action = squashed * self.action_scale + self.action_bias
        return squashed, env_action

    def _env_to_raw(self, env_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert env action -> squashed -> raw action using atanh.

        Returns:
            squashed: normalized action in (-1, 1)
            raw_action: pre-tanh raw action
        """
        # Map from env range back to [-1, 1]
        squashed = (env_action - self.action_bias) / torch.clamp(self.action_scale, min=self.eps)
        squashed = torch.clamp(squashed, -1.0 + self.eps, 1.0 - self.eps)

        # atanh(x) = 0.5 * (log(1+x) - log(1-x))
        raw_action = 0.5 * (
            torch.log1p(squashed) - torch.log1p(-squashed)
        )
        return squashed, raw_action

    def _log_prob_from_raw(
        self,
        dist: Normal,
        raw_action: torch.Tensor,
        squashed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log_prob of final env action via change-of-variables.

        env_action = action_bias + action_scale * tanh(raw_action)

        log p(env_action)
            = log p(raw_action)
              - sum log | d env_action / d raw_action |

        d env_action / d raw_action
            = action_scale * (1 - tanh(raw_action)^2)
            = action_scale * (1 - squashed^2)
        """
        base_log_prob = dist.log_prob(raw_action)  # (..., action_dim)

        # log|scale * (1 - tanh(raw)^2)|
        correction = torch.log(
            torch.clamp(self.action_scale * (1.0 - squashed.pow(2)), min=self.eps)
        )

        return (base_log_prob - correction).sum(dim=-1)

    def forward(  # type: ignore[override]
        self,
        features: torch.Tensor,
        *,
        avail_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """Produce FINAL env-range actions directly.

        Args:
            features: (..., input_dim)
            avail_actions: ignored for continuous action
            deterministic: if True, use mean action (after tanh+affine)

        Returns:
            {
                "actions": (..., action_dim)   # FINAL env actions
                "log_probs": (...,)            # log_prob of FINAL env actions
                "entropy": (...,)              # approximate entropy (base Gaussian)
                "logits": mean_raw             # kept for interface compatibility
                "mean": mean_raw
                "log_std": log_std
                "raw_actions": raw_action      # optional debug
            }
        """
        mean, log_std, dist = self._build_base_dist(features)

        if deterministic:
            raw_action = mean
        else:
            # PPO rollout不需要对动作采样反传梯度，sample即可
            raw_action = dist.sample()

        squashed, env_action = self._raw_to_env(raw_action)
        log_probs = self._log_prob_from_raw(dist, raw_action, squashed)

        # 近似熵：使用 base Gaussian 熵
        entropy = dist.entropy().sum(dim=-1)

        return {
            "actions": env_action,   # FINAL env-space action
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": mean,          # compatibility
            "mean": mean,
            "log_std": log_std,
            "raw_actions": raw_action,
        }

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        *,
        avail_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Evaluate FINAL env-range actions under current policy.

        Args:
            features: (..., input_dim)
            actions: (..., action_dim) FINAL env actions, same semantics as rollout-stored actions
            avail_actions: ignored for continuous action

        Returns:
            {
                "log_probs": (...,)
                "entropy": (...,)
                "logits": mean_raw
                "mean": mean_raw
                "log_std": log_std
            }
        """
        mean, log_std, dist = self._build_base_dist(features)

        actions = actions.to(device=mean.device, dtype=mean.dtype)
        squashed, raw_action = self._env_to_raw(actions)
        log_probs = self._log_prob_from_raw(dist, raw_action, squashed)

        # 近似熵：使用 base Gaussian 熵
        entropy = dist.entropy().sum(dim=-1)

        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": mean,
            "mean": mean,
            "log_std": log_std,
        }