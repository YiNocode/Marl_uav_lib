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
