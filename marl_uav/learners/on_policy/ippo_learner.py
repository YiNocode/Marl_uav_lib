"""IPPO learner: independent PPO with shared actor-critic policy."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn
import json
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from marl_uav.data.batch import Batch
from marl_uav.learners.base_learner import BaseLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy


class IPPOLearner(BaseLearner):
    """Independent PPO learner (参数共享版，多智能体视作多个样本)。"""

    def __init__(
        self,
        policy: ActorCriticPolicy,
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

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def _flatten_time_agent(self, x: np.ndarray) -> torch.Tensor:
        """(T, N, ...) -> (T*N, ...) tensor on policy device."""
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if t.ndim < 2:
            raise ValueError(f"Expected at least 2 dims (T, N, ...), got {tuple(t.shape)}")
        return t.reshape(-1, *t.shape[2:])

    def update(self, batch: Batch) -> Dict[str, Any]:
        """单次 PPO 更新。

        期望 Batch 含字段：
            - obs: (T, N, obs_dim)
            - actions: (T, N)
            - log_probs: (T, N)  采样时的旧 log_prob
            - advantages: (T, N)
            - returns: (T, N)
        """
        if not hasattr(batch, "log_probs"):
            raise ValueError("IPPO requires batch.log_probs from rollout.")
        if not hasattr(batch, "advantages") or not hasattr(batch, "returns"):
            raise ValueError("IPPO requires batch.advantages and batch.returns (GAE/returns).")

        obs = np.asarray(batch.obs)  # (T, N, D) 或 (B, T, N, D)
        actions = np.asarray(batch.actions)  # (T, N) 或 (B, T, N)
        old_log_probs = np.asarray(batch.log_probs)  # (T, N) 或 (B, T, N)
        advantages = np.asarray(batch.advantages)  # (T, N) 或 (B, T, N)
        returns = np.asarray(batch.returns)  # (T, N) 或 (B, T, N)
        avail_actions_arr = getattr(batch, "avail_actions", None)

        # 兼容 EpisodeBatch: (B, T, N, ...) -> (B*T, N, ...)
        if obs.ndim == 4:
            B, T, N, D = obs.shape
            obs = obs.reshape(B * T, N, D)
            # 连续动作: actions (B, T, N, action_dim) -> (B*T, N, action_dim)
            if actions.ndim == 4:
                actions = actions.reshape(B * T, N, actions.shape[-1])
            else:
                actions = actions.reshape(B * T, N)
            old_log_probs = old_log_probs.reshape(B * T, N)
            advantages = advantages.reshape(B * T, N)
            returns = returns.reshape(B * T, N)

            if avail_actions_arr is not None:
                aa = np.asarray(avail_actions_arr)
                if aa.ndim == 4 and aa.shape[0] == B and aa.shape[1] == T:
                    aa = aa.reshape(B * T, N, aa.shape[-1])
                avail_actions_arr = aa

        T, N = obs.shape[0], obs.shape[1]
        batch_size = T * N
        is_continuous = getattr(self.policy, "action_space_type", "discrete") == "continuous"

        obs_flat = self._flatten_time_agent(obs)  # (T*N, D)
        # 连续动作不转为 long，且保持 (T*N, action_dim)
        if is_continuous:
            actions_flat = torch.as_tensor(
                actions.reshape(batch_size, -1), dtype=torch.float32, device=self.device
            )
        else:
            actions_flat = torch.as_tensor(
                actions.reshape(batch_size), dtype=torch.long, device=self.device
            )
        old_log_probs_flat = torch.as_tensor(
            old_log_probs.reshape(batch_size), dtype=torch.float32, device=self.device
        )
        advantages_flat = torch.as_tensor(
            advantages.reshape(batch_size), dtype=torch.float32, device=self.device
        )
        returns_flat = torch.as_tensor(
            returns.reshape(batch_size), dtype=torch.float32, device=self.device
        )

        # 归一化 advantage
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std(unbiased=False) + 1e-8
        advantages_flat = (advantages_flat - adv_mean) / adv_std

        # 由于 ActorCriticPolicy.evaluate_actions 接受形状 (T, N, D)/(T, N)
        # 这里直接传原始 np 数组即可，让策略负责转换

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        last_approx_kl = 0.0
        last_clip_fraction = 0.0
        last_grad_norm = 0.0

        for _ in range(self.num_epochs):
            # 重新计算当前策略下的 log_probs / entropy / values
            # new_log_probs, entropy, values = self.policy.evaluate_actions(
            #     obs=obs, actions=actions
            # )
            state = getattr(batch, "state", None)
            avail_actions = avail_actions_arr

            # region agent log: confirm whether state is passed to policy
            try:
                log_entry = {
                    "sessionId": "cd6433",
                    "runId": "pre-fix",
                    "hypothesisId": "ippo_state_pass",
                    "location": "ippo_learner.py:update",
                    "message": "Evaluate actions inputs",
                    "data": {
                        "policy_state_dim": getattr(self.policy, "state_dim", None),
                        "state_is_none": state is None,
                        "has_avail_actions": avail_actions is not None,
                    },
                    "timestamp": int(__import__("time").time() * 1000),
                }

            except Exception:
                pass
            # endregion

            # IPPO 默认不使用全局 state（critic 使用 obs），只有在 policy 明确支持 state_dim 时才传入
            if getattr(self.policy, "state_dim", None) is None:
                state = None

            new_log_probs, entropy, values = self.policy.evaluate_actions(
                obs=obs,
                actions=actions,
                state=state,
                avail_actions=avail_actions,
            )
            new_log_probs_flat = new_log_probs.reshape(batch_size).to(self.device)
            entropy_flat = entropy.reshape(batch_size).to(self.device)
            values_flat = values.reshape(batch_size).to(self.device)

            # PPO ratio
            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
            )
            clipped = (ratio != clipped_ratio).float().mean().item()
            approx_kl = 0.5 * ((new_log_probs_flat - old_log_probs_flat) ** 2).mean().item()

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
            grad_norm = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy_mean.item())
            last_approx_kl = approx_kl
            last_clip_fraction = clipped
            last_grad_norm = grad_norm

        num_updates = float(self.num_epochs)
        return {
            "loss/policy_loss": total_policy_loss / num_updates,
            "loss/value_loss": total_value_loss / num_updates,
            "loss/entropy": total_entropy / num_updates,
            "train/approx_kl": last_approx_kl,
            "train/clip_fraction": last_clip_fraction,
            "train/grad_norm": last_grad_norm,
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

