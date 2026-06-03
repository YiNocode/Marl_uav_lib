"""Behavior cloning on the policy actor (Gaussian or discrete head)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


class BCLearner:
    """Supervised actor warm-start via negative log-likelihood of expert actions."""

    def __init__(
        self,
        policy: nn.Module,
        *,
        lr: float = 3e-4,
        max_grad_norm: float = 0.5,
        mse_coef: float = 0.0,
    ) -> None:
        self.policy = policy
        self.max_grad_norm = float(max_grad_norm)
        self.mse_coef = float(mse_coef)

        self._actor_params = self._collect_actor_parameters(policy)
        self.optimizer = Adam(self._actor_params, lr=float(lr))

    @staticmethod
    def _collect_actor_parameters(policy: nn.Module) -> list[nn.Parameter]:
        actor_params: list[nn.Parameter] = []
        if hasattr(policy, "actor_encoder"):
            actor_params.extend(policy.actor_encoder.parameters())
        elif hasattr(policy, "encoder"):
            actor_params.extend(policy.encoder.parameters())
        if hasattr(policy, "policy_head"):
            actor_params.extend(policy.policy_head.parameters())
        elif hasattr(policy, "actor_head"):
            actor_params.extend(policy.actor_head.parameters())
        if not actor_params:
            actor_params = list(policy.parameters())
        return actor_params

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def eval_batch(
        self,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        expert_actions: np.ndarray,
    ) -> dict[str, float]:
        """Forward-only BC loss (no optimizer step)."""
        was_training = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            metrics = self._forward_batch(obs=obs, state=state, expert_actions=expert_actions)
        if was_training:
            self.policy.train()
        return metrics

    def update_batch(
        self,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        expert_actions: np.ndarray,
    ) -> dict[str, float]:
        """One BC gradient step on a batch of transitions."""
        metrics = self._forward_batch(obs=obs, state=state, expert_actions=expert_actions)
        loss = metrics.pop("_loss_tensor")
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = 0.0
        for param in self._actor_params:
            if param.grad is not None:
                grad_norm += float(param.grad.data.norm(2).item() ** 2)
        grad_norm = grad_norm**0.5
        if self.max_grad_norm > 0:
            clip_grad_norm_(self._actor_params, self.max_grad_norm)
        self.optimizer.step()
        metrics["bc/grad_norm"] = float(grad_norm)
        return metrics

    def _forward_batch(
        self,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        expert_actions: np.ndarray,
    ) -> dict[str, float]:
        action_space = str(getattr(self.policy, "action_space_type", "discrete")).lower()
        if action_space != "continuous":
            raise ValueError("BCLearner currently supports continuous action_space only.")

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 2:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.ndim != 3:
            raise ValueError(f"obs must be (B,N,O), got {tuple(obs_t.shape)}")

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        if state_t.ndim == 2:
            state_t = state_t.unsqueeze(1)

        actions_t = torch.as_tensor(expert_actions, dtype=torch.float32, device=self.device)
        if actions_t.ndim == 2:
            actions_t = actions_t.unsqueeze(0)

        new_log_probs, _entropy, _values = self.policy.evaluate_actions(  # type: ignore[attr-defined]
            obs=obs_t,
            actions=actions_t,
            state=state_t,
        )
        nll_loss = -torch.mean(new_log_probs)

        loss = nll_loss
        mse_loss_val = 0.0
        if self.mse_coef > 0.0:
            actor_out, _critic_out = self.policy.forward(  # type: ignore[attr-defined]
                obs_t,
                state_t,
                deterministic=True,
            )
            pred = actor_out["actions"]
            mse_loss = torch.mean((pred - actions_t) ** 2)
            loss = loss + self.mse_coef * mse_loss
            mse_loss_val = float(mse_loss.item())

        with torch.no_grad():
            actor_out, _ = self.policy.forward(  # type: ignore[attr-defined]
                obs_t,
                state_t,
                deterministic=True,
            )
            pred = actor_out["actions"]
            pred_flat = pred.reshape(-1, pred.shape[-1])
            exp_flat = actions_t.reshape(-1, actions_t.shape[-1])
            cos = torch.nn.functional.cosine_similarity(pred_flat, exp_flat, dim=-1)
            cos_sim = float(cos.mean().item())

        return {
            "bc/nll_loss": float(nll_loss.item()),
            "bc/total_loss": float(loss.item()),
            "bc/mse_loss": mse_loss_val,
            "bc/mean_log_prob": float(new_log_probs.mean().item()),
            "bc/action_cosine_similarity": cos_sim,
            "_loss_tensor": loss,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        policy_state = state_dict.get("policy")
        if policy_state is not None:
            self.policy.load_state_dict(policy_state)
        optim_state = state_dict.get("optimizer")
        if optim_state is not None:
            self.optimizer.load_state_dict(optim_state)
