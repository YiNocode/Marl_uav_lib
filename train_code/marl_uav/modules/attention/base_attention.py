"""Base attention module."""
from abc import ABC, abstractmethod
from typing import Any


class BaseAttention(ABC):
    """Base attention over agents/entities."""

    @abstractmethod
    def forward(self, query: Any, key: Any, value: Any, **kwargs) -> Any:
        """Compute attention output."""
        pass
