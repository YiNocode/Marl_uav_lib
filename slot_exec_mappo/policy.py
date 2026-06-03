"""Standalone Gaussian actor and centralized critic for MAPPO."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover
    raise ImportError("slot_exec_mappo requires PyTorch") from exc


def _mlp(sizes: Sequence[int], *, out_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_activation:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.net = _mlp((self.obs_dim, *hidden_sizes, self.action_dim), out_activation=False)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        low = np.asarray(action_low if action_low is not None else -1.0, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high if action_high is not None else 1.0, dtype=np.float32).reshape(-1)
        self.register_buffer("_action_low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("_action_high", torch.as_tensor(high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        action = self._action_low + 0.5 * (squashed + 1.0) * (self._action_high - self._action_low)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.net(obs)
        squashed = torch.tanh(mean)
        return self._action_low + 0.5 * (squashed + 1.0) * (self._action_high - self._action_low)


class CentralCritic(nn.Module):
    def __init__(
        self,
        critic_input_dim: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = _mlp((int(critic_input_dim), *hidden_sizes, 1), out_activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SlotExecPolicyBundle:
    """Actor + critic bundle with save/load helpers."""

    def __init__(
        self,
        *,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.critic_input_dim = self.state_dim + self.obs_dim
        self.device = torch.device(device)
        self.actor = GaussianActor(
            self.obs_dim,
            self.action_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.critic = CentralCritic(self.critic_input_dim, hidden_sizes=hidden_sizes).to(self.device)

    def act(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if deterministic:
                actions = self.actor.deterministic(obs_t)
                log_probs = None
            else:
                actions, log_probs = self.actor(obs_t)
            values = None
        actions_np = actions.cpu().numpy().astype(np.float32)
        log_probs_np = None if log_probs is None else log_probs.cpu().numpy().astype(np.float32)
        return actions_np, log_probs_np, values

    def value(self, state: np.ndarray, obs: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if o.ndim == 1:
            o = o.reshape(1, -1)
        x = torch.cat([s.expand(o.shape[0], -1), o], dim=-1)
        with torch.no_grad():
            v = self.critic(x)
        return v.cpu().numpy().astype(np.float32)

    def save(self, path: str) -> None:
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "critic_input_dim": self.critic_input_dim,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        *,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str | torch.device = "cpu",
    ) -> "SlotExecPolicyBundle":
        payload = torch.load(path, map_location=device, weights_only=False)
        bundle = cls(
            obs_dim=int(payload["obs_dim"]),
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )
        bundle.actor.load_state_dict(payload["actor"])
        bundle.critic.load_state_dict(payload["critic"])
        return bundle
