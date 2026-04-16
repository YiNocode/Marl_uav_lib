"""从环境 / 配置读取连续动作边界（用于 GaussianPolicyHead 等）。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def parse_continuous_action_bounds_from_env_cfg(
    cfg: Mapping[str, Any],
    *,
    action_space: str,
    action_dim: int,
) -> tuple[list[float] | None, list[float] | None]:
    """从 env YAML（如 pyflyt_3v1.yaml）解析连续动作的 action_low / action_high。

    - 非 continuous：返回 (None, None)，由环境使用默认 Box。
    - continuous 且二者皆缺省：返回 (None, None)，环境侧默认 [-1,1]^action_dim。
    - 仅提供一个：报错。
    - 二者皆提供：校验长度等于 action_dim 后返回列表。
    """
    if str(action_space).lower() != "continuous":
        return None, None
    low = cfg.get("action_low")
    high = cfg.get("action_high")
    if low is None and high is None:
        return None, None
    if low is None or high is None:
        raise ValueError(
            "env 配置为 continuous 时，action_low 与 action_high 必须同时提供，或同时省略（使用默认 [-1,1]）。"
        )
    low_l = [float(x) for x in low]
    high_l = [float(x) for x in high]
    if len(low_l) != action_dim or len(high_l) != action_dim:
        raise ValueError(
            f"action_low / action_high 长度须等于 action_dim={action_dim}，"
            f"当前为 {len(low_l)} 与 {len(high_l)}。"
        )
    return low_l, high_l


def boxed_action_bounds(env: Any, action_dim: int) -> tuple[list[float], list[float]]:
    """优先使用 env 上由配置写入的 action_low_np / action_high_np，否则读 gym Box，再否则 [-1,1]^action_dim。"""
    low_np = getattr(env, "action_low_np", None)
    high_np = getattr(env, "action_high_np", None)
    if low_np is not None and high_np is not None:
        low = np.asarray(low_np, dtype=np.float64).reshape(-1)
        high = np.asarray(high_np, dtype=np.float64).reshape(-1)
        if low.size == action_dim and high.size == action_dim:
            return low.astype(float).tolist(), high.astype(float).tolist()

    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "low") and hasattr(space, "high"):
        low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(space.high, dtype=np.float64).reshape(-1)
        if low.size == action_dim and high.size == action_dim:
            return low.astype(float).tolist(), high.astype(float).tolist()
    return [-1.0] * int(action_dim), [1.0] * int(action_dim)
