"""Standalone MAPPO trainer (CTDE PPO)."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("slot_exec_mappo requires PyTorch") from exc

from slot_exec_mappo.buffer import RolloutBuffer
from slot_exec_mappo.config import SlotExecConfig
from slot_exec_mappo.policy import SlotExecPolicyBundle


class SlotExecMAPPOTrainer:
    def __init__(
        self,
        env: Any,
        *,
        cfg: SlotExecConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env = env
        self.cfg = cfg or SlotExecConfig()
        self.device = torch.device(device)
        action_dim = int(getattr(env, "action_dim", 4) or 4)
        low = np.asarray(getattr(env, "action_low_np", -0.25), dtype=np.float32).reshape(-1)
        high = np.asarray(getattr(env, "action_high_np", 0.25), dtype=np.float32).reshape(-1)
        self.policy = SlotExecPolicyBundle(
            obs_dim=int(env.obs_dim),
            state_dim=int(env.state_dim),
            action_dim=action_dim,
            hidden_sizes=self.cfg.train.hidden_sizes,
            action_low=low,
            action_high=high,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()),
            lr=float(self.cfg.train.lr),
        )
        self.buffer = RolloutBuffer(
            num_agents=int(env.num_agents),
            obs_dim=int(env.obs_dim),
            state_dim=int(env.state_dim),
            action_dim=action_dim,
            capacity=int(self.cfg.train.rollout_steps),
        )

    def _evaluate_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.policy.actor.net(obs)
        std = torch.exp(self.policy.actor.log_std).expand_as(mean)
        low = self.policy.actor._action_low
        high = self.policy.actor._action_high
        squashed = 2.0 * (actions - low) / (high - low + 1e-8) - 1.0
        squashed = torch.clamp(squashed, -0.999, 0.999)
        raw = 0.5 * torch.log((1.0 + squashed) / (1.0 - squashed + 1e-8))
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        critic_in = torch.cat([state, obs], dim=-1)
        value = self.policy.critic(critic_in)
        return log_prob, entropy, value

    def collect_rollout(self) -> dict[str, float]:
        env = self.env
        cfg = self.cfg.train
        self.buffer.reset()
        obs = env.get_obs()
        state = env.get_state()
        ep_returns: list[float] = []

        while not self.buffer.full:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            state_rep = state_t.expand(obs_t.shape[0], -1)
            actions, log_probs = self.policy.actor(obs_t)
            with torch.no_grad():
                values = self.policy.critic(torch.cat([state_rep, obs_t], dim=-1))
            actions_np = actions.detach().cpu().numpy().astype(np.float32)
            log_probs_np = log_probs.detach().cpu().numpy().astype(np.float32)
            values_np = values.detach().cpu().numpy().astype(np.float32)

            _, reward, terminated, truncated, _info = env.step(actions_np)
            done = bool(terminated or truncated)
            self.buffer.add(
                obs,
                state,
                actions_np,
                log_probs_np,
                reward,
                done,
                values_np,
            )
            ep_returns.append(float(np.sum(reward)))
            if done:
                obs, _info = env.reset()
                state = env.get_state()
            else:
                obs = env.get_obs()
                state = env.get_state()

        last_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        last_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        last_state_rep = last_state.expand(last_obs.shape[0], -1)
        with torch.no_grad():
            last_values = self.policy.critic(torch.cat([last_state_rep, last_obs], dim=-1))
        adv, ret = self.buffer.compute_gae(
            last_values.detach().cpu().numpy(),
            gamma=float(cfg.gamma),
            gae_lambda=float(cfg.gae_lambda),
        )
        metrics = self._update(adv, ret)
        metrics["rollout_return_mean"] = float(np.mean(ep_returns)) if ep_returns else 0.0
        return metrics

    def _update(self, adv: np.ndarray, ret: np.ndarray) -> dict[str, float]:
        cfg = self.cfg.train
        t, n = adv.shape
        obs = self.buffer.obs[:t].reshape(t * n, -1)
        state = np.repeat(self.buffer.state[:t], n, axis=0)
        actions = self.buffer.actions[:t].reshape(t * n, -1)
        old_logp = self.buffer.log_probs[:t].reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv_flat, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret_flat, dtype=torch.float32, device=self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updates = 0
        batch = obs_t.shape[0]
        mb = min(max(batch // 4, 64), batch)

        for _ in range(int(cfg.ppo_epochs)):
            idx = torch.randperm(batch, device=self.device)
            for start in range(0, batch, mb):
                sl = idx[start : start + mb]
                logp, entropy, value = self._evaluate_actions(obs_t[sl], state_t[sl], act_t[sl])
                ratio = torch.exp(logp - old_logp_t[sl])
                surr1 = ratio * adv_t[sl]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_t[sl]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value - ret_t[sl]) ** 2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()),
                    float(cfg.max_grad_norm),
                )
                self.optimizer.step()
                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_entropy += float(entropy.mean().detach().cpu())
                updates += 1

        return {
            "policy_loss": total_policy_loss / max(updates, 1),
            "value_loss": total_value_loss / max(updates, 1),
            "entropy": total_entropy / max(updates, 1),
        }

    def train(self, *, total_updates: int = 100, save_path: str | None = None) -> None:
        for upd in range(int(total_updates)):
            metrics = self.collect_rollout()
            if (upd + 1) % 10 == 0:
                print(
                    f"[slot_exec_mappo] update={upd + 1}/{total_updates} "
                    f"return={metrics.get('rollout_return_mean', 0.0):.3f} "
                    f"pi={metrics.get('policy_loss', 0.0):.4f} "
                    f"v={metrics.get('value_loss', 0.0):.4f}"
                )
            if save_path and (upd + 1) % 50 == 0:
                self.policy.save(save_path)
        if save_path:
            self.policy.save(save_path)
