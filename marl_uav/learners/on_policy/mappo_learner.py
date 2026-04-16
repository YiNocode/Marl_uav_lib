"""MAPPO learner: centralized critic PPO for multi-agent settings."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from marl_uav.learners.base_learner import BaseLearner


class MAPPOLearner(BaseLearner):
    """Multi-Agent PPO learner with (optional) centralized critic.

    约定：
        - `policy.evaluate_actions(obs, actions, state=state, avail_actions=...)`
          返回 (log_probs, entropy, values)
        - batch 为 on-policy EpisodeBatch 或等价结构，至少包含：
            - obs:           (T, N, obs_dim) 或 (B, T, N, obs_dim)
            - state:         (T, state_dim)  或 (B, T, state_dim)
            - actions:       (T, N)         或 (B, T, N)
            - log_probs:     (T, N)         或 (B, T, N)
            - advantages:    (T, N)         或 (B, T, N)
            - returns:       (T, N)         或 (B, T, N)
            - 可选 avail_actions: (T, N, A) 或 (B, T, N, A)
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
    ) -> None:
        self.policy = policy
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.clip_range = float(clip_range)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.num_epochs = int(num_epochs)

    def _maybe_adjust_advantages(
        self,
        advantages_bt: np.ndarray,
        state_bt: np.ndarray,
        obs_bt: np.ndarray,
    ) -> np.ndarray:
        """子类可覆盖，在展平为 (T_tot, N) 之后、张量化之前调整 advantage。"""
        del state_bt, obs_bt
        return advantages_bt

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def _flatten_bt_n(self, x: np.ndarray) -> np.ndarray:
        """(T, N, ...) 或 (B, T, N, ...) -> (T_tot, N, ...)."""
        if x.ndim == 4:  # (B, T, N, ...)
            B, T, N = x.shape[0], x.shape[1], x.shape[2]
            return x.reshape(B * T, N, *x.shape[3:])
        if x.ndim == 3:  # (T, N, ...)
            return x
        raise ValueError(f"Expected 3D or 4D array for flatten_bt_n, got shape {x.shape}")

    def _flatten_bt(self, x: np.ndarray) -> np.ndarray:
        """(T, S) 或 (B, T, S) -> (T_tot, S)."""
        if x.ndim == 3:  # (B, T, S)
            B, T, S = x.shape
            return x.reshape(B * T, S)
        if x.ndim == 2:  # (T, S)
            return x
        raise ValueError(f"Expected 2D or 3D array for flatten_bt, got shape {x.shape}")

    def update(self, batch: Any) -> Dict[str, Any]:
        """单次 MAPPO 更新（centralized critic）。"""
        required = ("obs", "state", "actions", "log_probs", "advantages", "returns")
        for name in required:
            if not hasattr(batch, name):
                raise ValueError(f"MAPPO requires batch.{name}.")

        obs = np.asarray(batch.obs)
        state = np.asarray(batch.state)
        actions = np.asarray(batch.actions)
        old_log_probs = np.asarray(batch.log_probs)
        advantages = np.asarray(batch.advantages)
        returns = np.asarray(batch.returns)
        # avail_actions 可不存在，或存在但为 None（EpisodeBatch 中字段已声明但值缺失）
        raw_avail = getattr(batch, "avail_actions", None)
        if raw_avail is None:
            avail_actions = None
        else:
            avail_actions = np.asarray(raw_avail)

        # 判定离散 / 连续动作空间
        action_space_type = str(getattr(self.policy, "action_space_type", "discrete")).lower()
        is_continuous = action_space_type == "continuous"

        # 统一为 (T_tot, N, ...) / (T_tot, S)
        obs_bt = self._flatten_bt_n(obs)
        if is_continuous:
            # 连续动作：actions 形状应为 (T, N, A) 或 (B, T, N, A)
            actions_bt = self._flatten_bt_n(actions)
        else:
            # 离散动作：actions 形状 (T, N) / (B, T, N)，先在末尾加 1 维再 squeeze 到 (T_tot, N)
            actions_bt = self._flatten_bt_n(actions[..., np.newaxis]).squeeze(-1)
        old_log_probs_bt = self._flatten_bt_n(old_log_probs[..., np.newaxis]).squeeze(-1)
        advantages_bt = self._flatten_bt_n(advantages[..., np.newaxis]).squeeze(-1)
        returns_bt = self._flatten_bt_n(returns[..., np.newaxis]).squeeze(-1)
        state_bt = self._flatten_bt(state)
        advantages_bt = self._maybe_adjust_advantages(advantages_bt, state_bt, obs_bt)
        if avail_actions is not None and not is_continuous:
            # 连续动作空间下策略会忽略 avail_actions，这里仅在离散动作时展开
            avail_bt = self._flatten_bt_n(avail_actions)
        else:
            avail_bt = None

        T_tot, N = obs_bt.shape[0], obs_bt.shape[1]
        batch_size = T_tot * N

        # 展平成 1D，便于广播与损失计算
        obs_tensor = torch.as_tensor(obs_bt, dtype=torch.float32, device=self.device)
        if is_continuous:
            actions_flat = torch.as_tensor(
                actions_bt.reshape(batch_size, -1), dtype=torch.float32, device=self.device
            )
        else:
            actions_flat = torch.as_tensor(
                actions_bt.reshape(batch_size), dtype=torch.long, device=self.device
            )
        old_log_probs_flat = torch.as_tensor(
            old_log_probs_bt.reshape(batch_size), dtype=torch.float32, device=self.device
        )
        advantages_flat = torch.as_tensor(
            advantages_bt.reshape(batch_size), dtype=torch.float32, device=self.device
        )
        returns_flat = torch.as_tensor(
            returns_bt.reshape(batch_size), dtype=torch.float32, device=self.device
        )
        state_tensor = torch.as_tensor(state_bt, dtype=torch.float32, device=self.device)
        avail_tensor: torch.Tensor | None
        if avail_bt is not None:
            avail_tensor = torch.as_tensor(avail_bt, dtype=torch.float32, device=self.device)
        else:
            avail_tensor = None

        # 归一化 advantage
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std(unbiased=False) + 1e-8
        advantages_flat = (advantages_flat - adv_mean) / adv_std

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.num_epochs):
            # 重新计算当前策略下的 log_probs / entropy / values（centralized critic 用 state）
            new_log_probs, entropy, values = self.policy.evaluate_actions(  # type: ignore[attr-defined]
                obs=obs_bt,
                actions=actions_bt,
                state=state_bt,
                avail_actions=avail_bt,
            )
            new_log_probs_flat = new_log_probs.reshape(batch_size).to(self.device)
            entropy_flat = entropy.reshape(batch_size).to(self.device)
            values_flat = values.reshape(batch_size).to(self.device)

            # PPO ratio
            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
            )

            surr1 = ratio * advantages_flat
            surr2 = clipped_ratio * advantages_flat
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            value_loss = 0.5 * torch.mean((returns_flat - values_flat) ** 2)
            entropy_mean = torch.mean(entropy_flat)

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy_mean
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy_mean.item())

        num_updates = float(self.num_epochs)
        return {
            "loss/policy_loss": total_policy_loss / num_updates,
            "loss/value_loss": total_value_loss / num_updates,
            "loss/entropy": total_entropy / num_updates,
        }

    # BaseLearner 兼容接口
    def train(self, batch: Any) -> dict:  # type: ignore[override]
        return self.update(batch)

    # ---------------------------- checkpoint API ---------------------------- #
    def state_dict(self) -> Dict[str, Any]:
        """返回可用于保存/恢复的完整状态字典."""
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hyperparams": {
                "clip_range": self.clip_range,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
                "num_epochs": self.num_epochs,
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """从 state_dict 恢复 policy 与 optimizer 状态."""
        policy_state = state_dict.get("policy")
        if policy_state is not None:
            self.policy.load_state_dict(policy_state)

        optim_state = state_dict.get("optimizer")
        if optim_state is not None:
            self.optimizer.load_state_dict(optim_state)
