"""Base memory / recurrent block."""
from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """Base memory module (e.g. RNN hidden state)."""

    @abstractmethod
    def forward(self, x: Any, hidden: Any = None) -> tuple[Any, Any]:
        """Forward; return (output, new_hidden)."""
        pass
