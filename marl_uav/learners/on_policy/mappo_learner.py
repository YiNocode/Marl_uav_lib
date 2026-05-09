"""MAPPO learner: centralized-critic PPO for multi-agent settings."""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from marl_uav.learners.base_learner import BaseLearner
from marl_uav.learners.tensor_utils import tensor_from_numpy_on_device


class MAPPOLearner(BaseLearner):
    """Multi-agent PPO learner with shared policy and centralized critic."""

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

    def _maybe_adjust_advantages(
        self,
        advantages_bt: np.ndarray,
        state_bt: np.ndarray,
        obs_bt: np.ndarray,
    ) -> np.ndarray:
        del state_bt, obs_bt
        return advantages_bt

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def _flatten_bt_n(self, x: np.ndarray) -> np.ndarray:
        """(T, N, ...) or (B, T, N, ...) -> (T_tot, N, ...)."""
        if x.ndim == 4:
            batch_size, time_steps, num_agents = x.shape[:3]
            return x.reshape(batch_size * time_steps, num_agents, *x.shape[3:])
        if x.ndim == 3:
            return x
        raise ValueError(f"Expected 3D or 4D array for flatten_bt_n, got shape {x.shape}")

    def _flatten_bt(self, x: np.ndarray) -> np.ndarray:
        """(T, S) or (B, T, S) -> (T_tot, S)."""
        if x.ndim == 3:
            batch_size, time_steps, state_dim = x.shape
            return x.reshape(batch_size * time_steps, state_dim)
        if x.ndim == 2:
            return x
        raise ValueError(f"Expected 2D or 3D array for flatten_bt, got shape {x.shape}")

    def _validate_batch_alignment(
        self,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray | None,
        avail_actions: np.ndarray | None,
        is_continuous: bool,
    ) -> None:
        leading_btn = obs.shape[:3]
        leading_bt = obs.shape[:2]

        if state.shape[:2] != leading_bt:
            raise ValueError(
                f"state leading dims {state.shape[:2]} do not match obs leading dims {leading_bt}"
            )
        if old_log_probs.shape != leading_btn:
            raise ValueError(
                f"log_probs shape {old_log_probs.shape} does not match obs leading dims {leading_btn}"
            )
        if advantages.shape != leading_btn:
            raise ValueError(
                f"advantages shape {advantages.shape} does not match obs leading dims {leading_btn}"
            )
        if returns.shape != leading_btn:
            raise ValueError(
                f"returns shape {returns.shape} does not match obs leading dims {leading_btn}"
            )
        if masks is not None and masks.shape != leading_bt:
            raise ValueError(
                f"masks shape {masks.shape} does not match obs leading dims {leading_bt}"
            )
        if avail_actions is not None and avail_actions.shape[:3] != leading_btn:
            raise ValueError(
                "avail_actions leading dims "
                f"{avail_actions.shape[:3]} do not match obs leading dims {leading_btn}"
            )

        if is_continuous:
            if actions.ndim != obs.ndim or actions.shape[:3] != leading_btn:
                raise ValueError(
                    "continuous actions shape "
                    f"{actions.shape} does not match obs leading dims {leading_btn}"
                )
        elif actions.shape != leading_btn:
            raise ValueError(
                f"discrete actions shape {actions.shape} does not match obs leading dims {leading_btn}"
            )

    def update(self, batch: Any) -> Dict[str, Any]:
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
        raw_masks = getattr(batch, "masks", None)
        masks = None if raw_masks is None else np.asarray(raw_masks)
        raw_avail = getattr(batch, "avail_actions", None)
        avail_actions = None if raw_avail is None else np.asarray(raw_avail)

        action_space_type = str(getattr(self.policy, "action_space_type", "discrete")).lower()
        is_continuous = action_space_type == "continuous"
        self._validate_batch_alignment(
            obs=obs,
            state=state,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            masks=masks,
            avail_actions=avail_actions,
            is_continuous=is_continuous,
        )

        obs_bt = self._flatten_bt_n(obs)
        if is_continuous:
            actions_bt = self._flatten_bt_n(actions)
        else:
            actions_bt = self._flatten_bt_n(actions[..., np.newaxis]).squeeze(-1)
        old_log_probs_bt = self._flatten_bt_n(old_log_probs[..., np.newaxis]).squeeze(-1)
        advantages_bt = self._flatten_bt_n(advantages[..., np.newaxis]).squeeze(-1)
        returns_bt = self._flatten_bt_n(returns[..., np.newaxis]).squeeze(-1)
        state_bt = self._flatten_bt(state)
        advantages_bt = self._maybe_adjust_advantages(advantages_bt, state_bt, obs_bt)
        if avail_actions is not None and not is_continuous:
            avail_bt = self._flatten_bt_n(avail_actions)
        else:
            avail_bt = None

        total_steps, num_agents = obs_bt.shape[:2]
        batch_size = total_steps * num_agents
        minibatch_target = self.minibatch_size if self.minibatch_size > 0 else batch_size
        minibatch_steps = max(1, (minibatch_target + num_agents - 1) // num_agents)
        chunk_starts = list(range(0, total_steps, minibatch_steps))

        adv_flat = advantages_bt.astype(np.float64).reshape(-1)
        adv_mean = float(adv_flat.mean())
        adv_std = float(adv_flat.std()) + 1e-8
        advantages_bt = ((advantages_bt - adv_mean) / adv_std).astype(np.float32, copy=False)

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

        for _ in range(self.num_epochs):
            random.shuffle(chunk_starts)
            for chunk_start in chunk_starts:
                chunk = slice(chunk_start, min(chunk_start + minibatch_steps, total_steps))
                obs_c = obs_bt[chunk]
                state_c = state_bt[chunk]
                actions_c = actions_bt[chunk]
                old_lp_c = old_log_probs_bt[chunk]
                adv_c = advantages_bt[chunk].reshape(-1)
                ret_c = returns_bt[chunk].reshape(-1)
                avail_c = None if avail_bt is None else avail_bt[chunk]

                adv_c_t = tensor_from_numpy_on_device(adv_c, self.device)
                ret_c_t = tensor_from_numpy_on_device(ret_c, self.device)
                old_lp_c_t = tensor_from_numpy_on_device(old_lp_c.reshape(-1), self.device)

                new_log_probs, entropy, values = self.policy.evaluate_actions(  # type: ignore[attr-defined]
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
                        "Non-finite PPO ratio detected; check rollout flatten alignment, "
                        "old_log_probs shape, and continuous-action log_prob consistency."
                    )
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
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

                approx_kl = 0.5 * float(log_ratio.pow(2).mean().item())
                clip_fraction = float((ratio != clipped_ratio).float().mean().item())
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
