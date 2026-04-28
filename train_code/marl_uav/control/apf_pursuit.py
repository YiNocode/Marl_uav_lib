"""追逃场景人工势场（APF）：pursuer 受 evader 吸引、受其他 pursuer 排斥。"""

from __future__ import annotations

import numpy as np


def apf_acceleration_3d(
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    other_pursuer_positions: list[np.ndarray],
    *,
    k_att: float,
    k_rep: float,
    rho0: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    计算单架 pursuer 在 APF 下的等效加速度方向（世界系 xyz）。

    - 吸引力：沿 evader - pursuer 单位方向，系数 k_att（evader 对 pursuer 为吸引目标）。
    - 排斥力：经典 APF，对距离小于 rho0 的其他 pursuer 产生排斥（仅 teammate）。

    Returns:
        (3,) float32，未裁剪；由上层映射到动作空间 Box。
    """
    pursuer_pos = np.asarray(pursuer_pos, dtype=np.float64).reshape(3)
    evader_pos = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    vec = evader_pos - pursuer_pos
    dist = float(np.linalg.norm(vec))
    if dist < eps:
        f_att = np.zeros(3, dtype=np.float64)
    else:
        f_att = float(k_att) * (vec / dist)

    f_rep = np.zeros(3, dtype=np.float64)
    k_rep = float(k_rep)
    rho0 = float(rho0)
    for op in other_pursuer_positions:
        op = np.asarray(op, dtype=np.float64).reshape(3)
        diff = pursuer_pos - op  # 由 teammate 指向自身
        rho = float(np.linalg.norm(diff))
        if rho < eps:
            continue
        if rho < rho0:
            # 排斥项 ∝ (1/ρ - 1/ρ0) / ρ²，方向远离 teammate
            mag = k_rep * (1.0 / rho - 1.0 / rho0) / (rho * rho + eps)
            f_rep += mag * (diff / rho)

    out = (f_att + f_rep).astype(np.float32)
    return out


def apf_action_from_force(
    f_xyz: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """
    将 3D 力/加速度向量映射为连续控制 setpoint [vx, vy, yaw, vz]。

    yaw 通道置 0；vx,vy,vz 分别对应 f 的 x,y,z 并裁剪到 Box。
    """
    f_xyz = np.asarray(f_xyz, dtype=np.float32).reshape(3)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    adim = low.shape[0]
    out = np.zeros((adim,), dtype=np.float32)
    out[0] = np.clip(f_xyz[0], low[0], high[0])
    out[1] = np.clip(f_xyz[1], low[1], high[1])
    if adim >= 3:
        out[2] = float(np.clip(0.0, low[2], high[2]))
    if adim >= 4:
        out[3] = np.clip(f_xyz[2], low[3], high[3])
    return out
