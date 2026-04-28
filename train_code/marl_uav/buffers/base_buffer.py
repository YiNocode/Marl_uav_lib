"""Base buffer interface."""
from abc import ABC, abstractmethod
from typing import Any


class BaseBuffer(ABC):
    """Base experience buffer."""

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Add transition(s)."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Any:
        """Sample batch."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Current size."""
        pass
