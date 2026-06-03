"""On-policy rollout buffer for slot execution MAPPO."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBuffer:
    num_agents: int
    obs_dim: int
    state_dim: int
    action_dim: int
    capacity: int
    _ptr: int = 0
    obs: np.ndarray = field(init=False)
    state: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        t = int(self.capacity)
        n = int(self.num_agents)
        self.obs = np.zeros((t, n, self.obs_dim), dtype=np.float32)
        self.state = np.zeros((t, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((t, n, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((t, n), dtype=np.float32)
        self.rewards = np.zeros((t, n), dtype=np.float32)
        self.dones = np.zeros((t,), dtype=np.float32)
        self.values = np.zeros((t, n), dtype=np.float32)
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        values: np.ndarray,
    ) -> None:
        i = self._ptr
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.state[i] = np.asarray(state, dtype=np.float32).reshape(-1)
        self.actions[i] = np.asarray(actions, dtype=np.float32)
        self.log_probs[i] = np.asarray(log_probs, dtype=np.float32).reshape(-1)
        self.rewards[i] = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self.dones[i] = float(done)
        self.values[i] = np.asarray(values, dtype=np.float32).reshape(-1)
        self._ptr += 1

    @property
    def full(self) -> bool:
        return self._ptr >= int(self.capacity)

    @property
    def size(self) -> int:
        return int(self._ptr)

    def reset(self) -> None:
        self._ptr = 0

    def compute_gae(
        self,
        last_values: np.ndarray,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        t = self.size
        n = int(self.num_agents)
        adv = np.zeros((t, n), dtype=np.float32)
        ret = np.zeros((t, n), dtype=np.float32)
        last_gae = np.zeros((n,), dtype=np.float32)
        last_values = np.asarray(last_values, dtype=np.float32).reshape(n)
        for step in reversed(range(t)):
            nonterminal = 1.0 - (self.dones[step] if step < t - 1 else 0.0)
            if step == t - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * nonterminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            adv[step] = last_gae
        ret = adv + self.values[:t]
        return adv, ret
