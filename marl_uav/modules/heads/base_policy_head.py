"""Base policy head (actor output)."""
from abc import ABC, abstractmethod
from typing import Any


class BasePolicyHead(ABC):
    """Base policy head: hidden -> action logits/distribution."""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Return action logits or distribution params."""
        pass
