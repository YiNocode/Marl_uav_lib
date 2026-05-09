"""IPPO learner: independent PPO with shared actor-critic policy."""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from marl_uav.data.batch import Batch
from marl_uav.learners.base_learner import BaseLearner
from marl_uav.learners.tensor_utils import tensor_from_numpy_on_device
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
        minibatch_size: int = 0,
    ) -> None:
        self.policy = policy
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.clip_range = float(clip_range)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.num_epochs = int(num_epochs)
        self.minibatch_size = int(minibatch_size)

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
        state_arr = getattr(batch, "state", None)
        if state_arr is not None:
            state_arr = np.asarray(state_arr)

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
            if state_arr is not None and state_arr.ndim == 3:
                state_arr = state_arr.reshape(B * T, state_arr.shape[-1])

        T, N = obs.shape[0], obs.shape[1]
        batch_size = T * N

        mb_target = self.minibatch_size if self.minibatch_size > 0 else batch_size
        mb_t = max(1, (mb_target + N - 1) // N)
        chunk_starts = list(range(0, T, mb_t))

        adv_np = advantages.astype(np.float64).reshape(-1)
        adv_mean = float(adv_np.mean())
        adv_std = float(adv_np.std()) + 1e-8
        advantages = ((advantages - adv_mean) / adv_std).astype(np.float32, copy=False)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_grad_norm = 0.0
        max_approx_kl = 0.0
        max_clip_fraction = 0.0
        max_grad_norm = 0.0
        n_mb_updates = 0

        use_state = state_arr is not None and getattr(self.policy, "state_dim", None) is not None

        for _ in range(self.num_epochs):
            avail_actions = avail_actions_arr
            random.shuffle(chunk_starts)
            for cs in chunk_starts:
                sl = slice(cs, min(cs + mb_t, T))
                obs_c = obs[sl]
                actions_c = actions[sl]
                old_lp_c = old_log_probs[sl]
                adv_c = advantages[sl].reshape(-1)
                ret_c = returns[sl].reshape(-1)
                avail_c = None if avail_actions is None else np.asarray(avail_actions)[sl]
                state_c = state_arr[sl] if use_state else None

                adv_c_t = tensor_from_numpy_on_device(adv_c, self.device)
                ret_c_t = tensor_from_numpy_on_device(ret_c, self.device)
                csz = obs_c.shape[0] * N
                old_lp_c_t = tensor_from_numpy_on_device(
                    old_lp_c.reshape(-1), self.device
                )

                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    obs=obs_c,
                    actions=actions_c,
                    state=state_c,
                    avail_actions=avail_c,
                )
                new_log_probs_flat = new_log_probs.reshape(csz).to(self.device)
                entropy_flat = entropy.reshape(csz).to(self.device)
                values_flat = values.reshape(csz).to(self.device)

                ratio = torch.exp(new_log_probs_flat - old_lp_c_t)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                clipped = (ratio != clipped_ratio).float().mean().item()
                approx_kl = 0.5 * float(((new_log_probs_flat - old_lp_c_t) ** 2).mean().item())

                surr1 = ratio * adv_c_t
                surr2 = clipped_ratio * adv_c_t
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                value_loss = 0.5 * torch.mean((ret_c_t - values_flat) ** 2)
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
                grad_norm = grad_norm**0.5
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_mean.item())
                total_approx_kl += approx_kl
                total_clip_fraction += clipped
                total_grad_norm += grad_norm
                max_approx_kl = max(max_approx_kl, approx_kl)
                max_clip_fraction = max(max_clip_fraction, clipped)
                max_grad_norm = max(max_grad_norm, grad_norm)
                n_mb_updates += 1

        denom = max(float(n_mb_updates), 1.0)
        return {
            "loss/policy_loss": total_policy_loss / denom,
            "loss/value_loss": total_value_loss / denom,
            "loss/entropy": total_entropy / denom,
            "train/approx_kl": total_approx_kl / denom,
            "train/clip_fraction": total_clip_fraction / denom,
            "train/grad_norm": total_grad_norm / denom,
            "train/max_approx_kl": max_approx_kl,
            "train/max_clip_fraction": max_clip_fraction,
            "train/max_grad_norm": max_grad_norm,
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
                "minibatch_size": self.minibatch_size,
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

