"""Base learner interface."""
from abc import ABC, abstractmethod
from typing import Any


class BaseLearner(ABC):
    """Base class for algorithm learner (train step)."""

    @abstractmethod
    def train(self, batch: Any) -> dict:
        """One training step; return loss dict."""
        pass
