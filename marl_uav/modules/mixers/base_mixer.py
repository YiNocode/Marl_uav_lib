"""Base mixer for value decomposition."""
from abc import ABC, abstractmethod
from typing import Any


class BaseMixer(ABC):
    """Base mixer: local Qs -> global Q (e.g. QMIX, VDN)."""

    @abstractmethod
    def forward(self, agent_qs: Any, states: Any = None, **kwargs) -> Any:
        """Mix per-agent Q values into global Q."""
        pass
