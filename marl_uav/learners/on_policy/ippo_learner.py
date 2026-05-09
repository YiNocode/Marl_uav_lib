"""IPPO learner: independent PPO with optional shared policy."""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from marl_uav.data.batch import Batch
from marl_uav.learners.base_learner import BaseLearner
from marl_uav.learners.tensor_utils import tensor_from_numpy_on_device
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy


class IPPOLearner(BaseLearner):
    """Independent PPO learner."""

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
        target_kl: float | None = None,
    ) -> None:
        self.policy = policy
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.clip_range = float(clip_range)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.num_epochs = int(num_epochs)
        self.minibatch_size = int(minibatch_size)
        self.target_kl = None if target_kl is None else float(target_kl)

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def update(self, batch: Batch) -> Dict[str, Any]:
        if not hasattr(batch, "log_probs"):
            raise ValueError("IPPO requires batch.log_probs from rollout.")
        if not hasattr(batch, "advantages") or not hasattr(batch, "returns"):
            raise ValueError("IPPO requires batch.advantages and batch.returns.")

        obs = np.asarray(batch.obs)
        actions = np.asarray(batch.actions)
        old_log_probs = np.asarray(batch.log_probs)
        advantages = np.asarray(batch.advantages)
        returns = np.asarray(batch.returns)
        avail_actions_arr = getattr(batch, "avail_actions", None)
        state_arr = getattr(batch, "state", None)
        if state_arr is not None:
            state_arr = np.asarray(state_arr)

        if obs.ndim == 4:
            batch_size, time_steps, num_agents, obs_dim = obs.shape
            obs = obs.reshape(batch_size * time_steps, num_agents, obs_dim)
            if actions.ndim == 4:
                actions = actions.reshape(batch_size * time_steps, num_agents, actions.shape[-1])
            else:
                actions = actions.reshape(batch_size * time_steps, num_agents)
            old_log_probs = old_log_probs.reshape(batch_size * time_steps, num_agents)
            advantages = advantages.reshape(batch_size * time_steps, num_agents)
            returns = returns.reshape(batch_size * time_steps, num_agents)

            if avail_actions_arr is not None:
                avail_actions_arr = np.asarray(avail_actions_arr)
                if avail_actions_arr.ndim == 4:
                    avail_actions_arr = avail_actions_arr.reshape(
                        batch_size * time_steps, num_agents, avail_actions_arr.shape[-1]
                    )
            if state_arr is not None and state_arr.ndim == 3:
                state_arr = state_arr.reshape(batch_size * time_steps, state_arr.shape[-1])

        total_steps, num_agents = obs.shape[:2]
        batch_elems = total_steps * num_agents
        minibatch_target = self.minibatch_size if self.minibatch_size > 0 else batch_elems
        minibatch_steps = max(1, (minibatch_target + num_agents - 1) // num_agents)
        chunk_starts = list(range(0, total_steps, minibatch_steps))

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
        total_ratio_mean = 0.0
        total_ratio_max = 0.0
        total_ratio_min = 0.0
        max_approx_kl = 0.0
        max_clip_fraction = 0.0
        max_grad_norm_seen = 0.0
        max_ratio_seen = float("-inf")
        min_ratio_seen = float("inf")
        early_stop = False
        n_mb_updates = 0

        use_state = state_arr is not None and getattr(self.policy, "state_dim", None) is not None

        for _ in range(self.num_epochs):
            random.shuffle(chunk_starts)
            for chunk_start in chunk_starts:
                chunk = slice(chunk_start, min(chunk_start + minibatch_steps, total_steps))
                obs_c = obs[chunk]
                actions_c = actions[chunk]
                old_lp_c = old_log_probs[chunk]
                adv_c = advantages[chunk].reshape(-1)
                ret_c = returns[chunk].reshape(-1)
                avail_c = None if avail_actions_arr is None else np.asarray(avail_actions_arr)[chunk]
                state_c = state_arr[chunk] if use_state else None

                adv_c_t = tensor_from_numpy_on_device(adv_c, self.device)
                ret_c_t = tensor_from_numpy_on_device(ret_c, self.device)
                old_lp_c_t = tensor_from_numpy_on_device(old_lp_c.reshape(-1), self.device)

                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    obs=obs_c,
                    actions=actions_c,
                    state=state_c,
                    avail_actions=avail_c,
                )
                chunk_size = obs_c.shape[0] * num_agents
                new_log_probs_flat = new_log_probs.reshape(chunk_size).to(self.device)
                entropy_flat = entropy.reshape(chunk_size).to(self.device)
                values_flat = values.reshape(chunk_size).to(self.device)

                log_ratio = new_log_probs_flat - old_lp_c_t
                ratio = torch.exp(log_ratio)
                if not torch.isfinite(ratio).all():
                    raise RuntimeError(
                        "Non-finite PPO ratio detected; check old_log_probs alignment "
                        "and continuous-action log_prob consistency."
                    )
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                clip_fraction = float((ratio != clipped_ratio).float().mean().item())
                approx_kl = 0.5 * float(log_ratio.pow(2).mean().item())

                surr1 = ratio * adv_c_t
                surr2 = clipped_ratio * adv_c_t
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = 0.5 * torch.mean((ret_c_t - values_flat) ** 2)
                entropy_mean = torch.mean(entropy_flat)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = 0.0
                for param in self.policy.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm**0.5
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                ratio_mean = float(ratio.mean().item())
                ratio_max = float(ratio.max().item())
                ratio_min = float(ratio.min().item())

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_mean.item())
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                total_grad_norm += grad_norm
                total_ratio_mean += ratio_mean
                total_ratio_max += ratio_max
                total_ratio_min += ratio_min
                max_approx_kl = max(max_approx_kl, approx_kl)
                max_clip_fraction = max(max_clip_fraction, clip_fraction)
                max_grad_norm_seen = max(max_grad_norm_seen, grad_norm)
                max_ratio_seen = max(max_ratio_seen, ratio_max)
                min_ratio_seen = min(min_ratio_seen, ratio_min)
                n_mb_updates += 1

                if self.target_kl is not None and approx_kl > self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        denom = max(float(n_mb_updates), 1.0)
        return {
            "loss/policy_loss": total_policy_loss / denom,
            "loss/value_loss": total_value_loss / denom,
            "loss/entropy": total_entropy / denom,
            "train/approx_kl": total_approx_kl / denom,
            "train/clip_fraction": total_clip_fraction / denom,
            "train/grad_norm": total_grad_norm / denom,
            "train/ratio_mean": total_ratio_mean / denom,
            "train/ratio_max": total_ratio_max / denom,
            "train/ratio_min": total_ratio_min / denom,
            "train/max_approx_kl": max_approx_kl,
            "train/max_clip_fraction": max_clip_fraction,
            "train/max_grad_norm": max_grad_norm_seen,
            "train/max_ratio": max_ratio_seen if n_mb_updates > 0 else 0.0,
            "train/min_ratio": min_ratio_seen if n_mb_updates > 0 else 0.0,
            "train/early_stop": float(1.0 if early_stop else 0.0),
        }

    def train(self, batch: Any) -> dict:  # type: ignore[override]
        return self.update(batch)

    def state_dict(self) -> Dict[str, Any]:
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
                "target_kl": self.target_kl,
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        policy_state = state_dict.get("policy")
        if policy_state is not None:
            self.policy.load_state_dict(policy_state)

        optim_state = state_dict.get("optimizer")
        if optim_state is not None:
            self.optimizer.load_state_dict(optim_state)
