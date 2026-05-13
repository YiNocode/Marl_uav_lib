"""Minimal simulation backend interface used by environment adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SimBackendState:
    """Backend state layout consumed by task implementations.

    ``states`` follows the PyFlyt-compatible convention:
    ``[num_agents, 4, 3]`` where rows are angular velocity, Euler angle,
    linear velocity, and world position.
    """

    states: np.ndarray
    aux_states: list[np.ndarray]
    contact_array: np.ndarray
    elapsed_time: float


@dataclass
class BatchedSimBackendState:
    """Vectorized backend state layout for native simulator parallelism.

    ``states`` uses ``[num_envs, num_agents, 4, 3]``.  Each environment slice is
    compatible with :class:`SimBackendState`, which keeps existing task code
    reusable for both single-env and native-vector Genesis rollouts.
    """

    states: np.ndarray
    aux_states: list[list[np.ndarray]]
    contact_array: np.ndarray
    elapsed_time: np.ndarray

    def env_state(self, env_idx: int) -> SimBackendState:
        """Return a single environment view for existing task implementations."""
        return SimBackendState(
            states=np.asarray(self.states[env_idx], dtype=np.float32),
            aux_states=list(self.aux_states[env_idx]),
            contact_array=np.asarray(self.contact_array[env_idx], dtype=np.int8),
            elapsed_time=float(np.asarray(self.elapsed_time).reshape(-1)[env_idx]),
        )


class BaseSimBackend(ABC):
    """Small contract shared by physics backend adapters."""

    num_agents: int

    @abstractmethod
    def reset(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        seed: int | None = None,
    ) -> SimBackendState:
        """Reset the simulator to the requested initial poses."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: Any) -> SimBackendState:
        """Advance one simulator control step."""
        raise NotImplementedError

    @abstractmethod
    def get_agent_state(self) -> dict[str, list[dict[str, np.ndarray]]]:
        """Return role-grouped agent state for diagnostics."""
        raise NotImplementedError

    def close(self) -> None:
        """Release simulator resources."""
        return None
