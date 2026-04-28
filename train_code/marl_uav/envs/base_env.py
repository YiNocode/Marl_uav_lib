"""Base environment interface for MARL UAV scenarios."""
from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym


class BaseEnv(gym.Env, ABC):
    """Base class for multi-agent UAV environments."""

    @abstractmethod
    def get_obs(self) -> list[Any]:
        """Return list of observations per agent."""
        pass

    @abstractmethod
    def get_avail_actions(self) -> list[Any]:
        """Return list of available action masks per agent."""
        pass

    @abstractmethod
    def get_state(self) -> Any:
        """Return global state (e.g. for centralized critic)."""
        pass
