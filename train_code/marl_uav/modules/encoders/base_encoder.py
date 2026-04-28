"""Base encoder for observations and global state."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseEncoder(ABC):
    """统一抽象接口：局部观测 encoder 与全局 state encoder 均通过 forward(x) 编码。

    同一接口可接受：
        - obs[..., O]   — 局部观测，最后一维为观测维度 O
        - state[..., S] — 全局 state，最后一维为 state 维度 S

    具体实现时由构造参数决定 input_dim（O 或 S），forward 只要求 x 的最后一维与 input_dim 一致。
    输入形状：(..., input_dim)，输出形状：(..., output_dim)。
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to hidden representation.

        Args:
            x: obs[..., O] 或 state[..., S]，最后一维须等于该 encoder 的 input_dim。

        Returns:
            Encoded tensor, shape (..., output_dim).
        """
        ...
