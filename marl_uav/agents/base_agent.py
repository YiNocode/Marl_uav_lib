"""Base agent interface."""
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base class for multi-agent controller."""

    @abstractmethod
    def select_actions(self, obs: Any, **kwargs) -> Any:
        """Select actions for all agents given observations."""
        pass

    @abstractmethod
    def parameters(self):
        """Return parameters for optimizer."""
        pass
