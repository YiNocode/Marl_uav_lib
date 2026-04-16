"""MLP encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn

from marl_uav.modules.encoders.base_encoder import BaseEncoder
from marl_uav.utils.config import load_config

ActivationName = Literal["relu", "tanh", "gelu", "leaky_relu", "silu", "identity"]


def _make_activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "tanh":
        return nn.Tanh()
    if n == "gelu":
        return nn.GELU()
    if n == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if n in ("silu", "swish"):
        return nn.SiLU()
    if n in ("identity", "linear", "none"):
        return nn.Identity()
    raise ValueError(
        f"Unsupported activation={name!r}. "
        "Use one of: relu, tanh, gelu, leaky_relu, silu, identity."
    )


class MLPEncoder(nn.Module, BaseEncoder):
    """多层感知机编码器：Linear 堆叠 + 激活函数。

    输入形状：(..., input_dim)
    输出形状：(..., output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        output_dim: int | None = None,
        *,
        activation: ActivationName = "relu",
        out_activation: ActivationName = "identity",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim is None:
            if len(hidden_dims) == 0:
                raise ValueError("Either provide output_dim or non-empty hidden_dims.")
            output_dim = int(hidden_dims[-1])
            hidden_dims = tuple(hidden_dims[:-1])

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.output_dim = int(output_dim)

        layers: list[nn.Module] = []
        prev = self.input_dim

        for h in self.hidden_dims:
            if h <= 0:
                raise ValueError("hidden_dims must be positive integers")
            layers.append(nn.Linear(prev, h))
            layers.append(_make_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h

        layers.append(nn.Linear(prev, self.output_dim))
        layers.append(_make_activation(out_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear supports arbitrary leading dims: (..., input_dim) -> (..., output_dim)
        if x.ndim == 0:
            raise ValueError("MLPEncoder.forward expects at least 1D tensor (..., input_dim).")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"MLPEncoder.forward last dim mismatch: got x.shape={tuple(x.shape)}, "
                f"expected (..., {self.input_dim})."
            )
        return self.net(x)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        config: Mapping[str, Any],
        *,
        output_dim: int | None = None,
    ) -> "MLPEncoder":
        """从配置字典构建（如来自 `configs/model/mlp.yaml`）。

        读取字段：
        - hidden_dims: list[int]
        - activation: str
        可选字段：
        - dropout: float
        - out_activation: str
        - output_dim: int（若参数 output_dim 未提供）
        """
        hidden_dims = config.get("hidden_dims", (64, 64))
        activation = config.get("activation", "relu")
        dropout = float(config.get("dropout", 0.0) or 0.0)
        out_activation = config.get("out_activation", "identity")
        if output_dim is None and "output_dim" in config:
            output_dim = int(config["output_dim"])
        return cls(
            input_dim=input_dim,
            hidden_dims=tuple(int(x) for x in hidden_dims),
            output_dim=output_dim,
            activation=activation,
            out_activation=out_activation,
            dropout=dropout,
        )

    @classmethod
    def from_yaml(
        cls,
        input_dim: int,
        yaml_path: str | Path,
        *,
        output_dim: int | None = None,
    ) -> "MLPEncoder":
        """直接读取 YAML 文件并构建。"""
        cfg = load_config(yaml_path)
        return cls.from_config(input_dim=input_dim, config=cfg, output_dim=output_dim)

