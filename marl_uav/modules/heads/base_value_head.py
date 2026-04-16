"""Base value head (critic output)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch import nn


class _ValueHeadProto(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - protocol
        ...


class BaseValueHead(nn.Module, ABC):
    """价值头接口：输入 feature，输出 value。

    约定：
    - 输入 `x` 形状为 (..., feat_dim)
    - 输出为 (..., value_dim) 或 (..., 1)，由具体子类决定
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features into value(s) V(s) or Q(s,a)."""
        raise NotImplementedError

