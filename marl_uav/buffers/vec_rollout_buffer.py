"""Fixed-horizon rollout buffer for vectorized PPO collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.data.batch import Batch


@dataclass
class VecRolloutBatch:
    batch: EpisodeBatch
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    completed_episodes: int


class VecRolloutBuffer:
    """Store a rollout with shape [T, E, ...] and export learner-ready batches."""

    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_shape: tuple[int, ...],
        discrete_actions: bool,
        avail_action_dim: int | None,
    ) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_shape = tuple(action_shape)
        self.discrete_actions = bool(discrete_actions)
        self.avail_action_dim = avail_action_dim
        self.reset()

    def reset(self) -> None:
        T, E, N, O, S = self.num_steps, self.num_envs, self.num_agents, self.obs_dim, self.state_dim
        action_dtype = np.int64 if self.discrete_actions else np.float32
        action_full_shape = (T, E, N, *self.action_shape)
        if self.discrete_actions:
            action_full_shape = (T, E, N)

        self.obs = np.zeros((T, E, N, O), dtype=np.float32)
        self.state = np.zeros((T, E, S), dtype=np.float32)
        self.actions = np.zeros(action_full_shape, dtype=action_dtype)
        self.rewards = np.zeros((T, E, N), dtype=np.float32)
        self.dones = np.zeros((T, E), dtype=np.float32)
        self.terminated = np.zeros((T, E), dtype=np.float32)
        self.truncated = np.zeros((T, E), dtype=np.float32)
        self.log_probs = np.zeros((T, E, N), dtype=np.float32)
        self.values = np.zeros((T, E, N), dtype=np.float32)
        self.next_values = np.zeros((T, E, N), dtype=np.float32)
        self.advantages = np.zeros((T, E, N), dtype=np.float32)
        self.returns = np.zeros((T, E, N), dtype=np.float32)
        if self.avail_action_dim is not None:
            self.avail_actions = np.zeros((T, E, N, self.avail_action_dim), dtype=np.float32)
        else:
            self.avail_actions = None

    def add(
        self,
        step: int,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        avail_actions: np.ndarray | None,
    ) -> None:
        self.obs[step] = obs
        self.state[step] = state
        self.actions[step] = actions
        self.rewards[step] = rewards
        self.dones[step] = dones.astype(np.float32)
        self.terminated[step] = terminated.astype(np.float32)
        self.truncated[step] = truncated.astype(np.float32)
        self.log_probs[step] = log_probs
        self.values[step] = values
        self.next_values[step] = next_values
        if self.avail_actions is not None and avail_actions is not None:
            self.avail_actions[step] = avail_actions

    def compute_returns_and_advantages(self, *, gamma: float, gae_lambda: float) -> None:
        last_adv = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        for step in range(self.num_steps - 1, -1, -1):
            bootstrap_mask = 1.0 - self.terminated[step][:, None]
            nonterminal = 1.0 - self.dones[step][:, None]
            delta = (
                self.rewards[step]
                + gamma * bootstrap_mask * self.next_values[step]
                - self.values[step]
            )
            last_adv = delta + gamma * gae_lambda * nonterminal * last_adv
            self.advantages[step] = last_adv
        self.returns = self.advantages + self.values

    def as_batch(self) -> Batch:
        data: dict[str, Any] = {
            "obs": np.swapaxes(self.obs, 0, 1),
            "state": np.swapaxes(self.state, 0, 1),
            "actions": np.swapaxes(self.actions, 0, 1),
            "rewards": np.swapaxes(self.rewards, 0, 1),
            "dones": np.swapaxes(self.dones, 0, 1),
            "log_probs": np.swapaxes(self.log_probs, 0, 1),
            "values": np.swapaxes(self.values, 0, 1),
            "advantages": np.swapaxes(self.advantages, 0, 1),
            "returns": np.swapaxes(self.returns, 0, 1),
        }
        if self.avail_actions is not None:
            data["avail_actions"] = np.swapaxes(self.avail_actions, 0, 1)
        return Batch(**data)
