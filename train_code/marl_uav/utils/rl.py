"""强化学习常用数学工具（returns / GAE 等）。"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _is_torch(x: Any) -> bool:
    try:
        import torch  # noqa: F401

        import torch as _torch

        return isinstance(x, _torch.Tensor)
    except Exception:
        return False


def _to_float(x: Any) -> Any:
    # dones 可能是 bool/int；统一转 float 便于广播
    if _is_torch(x):
        return x.to(dtype=x.new_tensor(0.0).dtype)
    return np.asarray(x, dtype=np.float32)


def compute_returns(
    rewards: Any,
    dones: Any,
    *,
    gamma: float,
    last_value: Any | float = 0.0,
) -> Any:
    """计算折扣回报（discounted returns）。

    约定：时间维在第 0 维，形状可为 (T, ...)；dones 形状可为 (T,) 或 (T, ...)，可广播。

    公式：
        R_t = r_t + gamma * (1 - done_t) * R_{t+1}
    """
    if _is_torch(rewards):
        import torch

        r = rewards
        d = _to_float(dones).to(device=r.device)
        T = r.shape[0]
        out = torch.zeros_like(r)
        nxt = (
            last_value
            if _is_torch(last_value)
            else torch.as_tensor(last_value, dtype=r.dtype, device=r.device)
        )
        for t in range(T - 1, -1, -1):
            out[t] = r[t] + gamma * (1.0 - d[t]) * nxt
            nxt = out[t]
        return out

    r = np.asarray(rewards)
    d = _to_float(dones)
    T = r.shape[0]
    out = np.zeros_like(r, dtype=np.result_type(r, np.float32))
    nxt = np.asarray(last_value, dtype=out.dtype)
    for t in range(T - 1, -1, -1):
        out[t] = r[t] + gamma * (1.0 - d[t]) * nxt
        nxt = out[t]
    return out


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    *,
    gamma: float,
    gae_lambda: float,
    last_value: Any | float = 0.0,
) -> Tuple[Any, Any]:
    """计算 GAE(Generalized Advantage Estimation) 与 returns。

    约定：时间维在第 0 维，形状可为：
    - rewards: (T, ...)
    - values:  (T, ...) 对应 V(s_t)
    - dones:   (T,) 或 (T, ...) 可广播

    公式：
        delta_t = r_t + gamma * (1-done_t) * V_{t+1} - V_t
        A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
        returns_t = A_t + V_t
    """
    if _is_torch(rewards) or _is_torch(values):
        import torch

        r = rewards if _is_torch(rewards) else torch.as_tensor(rewards)
        v = values if _is_torch(values) else torch.as_tensor(values)
        r = r.to(dtype=v.dtype, device=v.device)
        d = _to_float(dones).to(device=v.device)

        T = v.shape[0]
        adv = torch.zeros_like(v)
        last_adv = torch.zeros_like(v[0])

        next_v = (
            last_value
            if _is_torch(last_value)
            else torch.as_tensor(last_value, dtype=v.dtype, device=v.device)
        )

        for t in range(T - 1, -1, -1):
            nonterminal = 1.0 - d[t]
            delta = r[t] + gamma * nonterminal * next_v - v[t]
            last_adv = delta + gamma * gae_lambda * nonterminal * last_adv
            adv[t] = last_adv
            next_v = v[t]

        ret = adv + v
        return adv, ret

    r = np.asarray(rewards, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    d = _to_float(dones)
    T = v.shape[0]

    adv = np.zeros_like(v, dtype=np.float32)
    last_adv = np.zeros_like(v[0], dtype=np.float32)
    next_v = np.asarray(last_value, dtype=np.float32)

    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - d[t]
        delta = r[t] + gamma * nonterminal * next_v - v[t]
        last_adv = delta + gamma * gae_lambda * nonterminal * last_adv
        adv[t] = last_adv
        next_v = v[t]

    ret = adv + v
    return adv, ret

