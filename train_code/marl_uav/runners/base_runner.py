"""Base runner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRunner(ABC):
    """Base runner (rollout / train / eval)."""

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute run logic."""
        pass

    def __call__(self, **kwargs) -> Any:
        return self.run(**kwargs)

    @staticmethod
    def merge_metrics(*metrics: Dict[str, Any]) -> Dict[str, Any]:
        """简单合并多个指标字典，后者覆盖前者同名键。"""
        out: Dict[str, Any] = {}
        for m in metrics:
            out.update(m)
        return out
