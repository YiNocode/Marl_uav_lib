"""Base communication module."""
from abc import ABC, abstractmethod
from typing import Any


class BaseComm(ABC):
    """Base inter-agent communication module."""

    @abstractmethod
    def forward(self, agent_embeddings: Any, **kwargs) -> Any:
        """Produce communicated messages/embeddings."""
        pass
