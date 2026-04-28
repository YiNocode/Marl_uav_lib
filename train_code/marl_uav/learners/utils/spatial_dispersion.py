"""空间分散度：多 pursuer 相对 evader 的方向一致性（越一致惩罚越大）。"""

from __future__ import annotations

import numpy as np


def pairwise_mean_cosine_similarity(
    rels: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    根据 pursuer 相对 evader 的位移向量计算每时间步的方向相似度惩罚。

    Args:
        rels: (T, P, D)，约定 rels[t,i] = pursuer_i_pos - evader_pos（与 PursuitEvasion3v1Task.build_state 一致）。
              指向 evader 的方向为 to_evader = -rels。

    Returns:
        penalty: (T,) 各时间步上所有 pursuer 对 (i<j) 的单位方向向量余弦相似度的均值，
                 取值约 [-1, 1]；方向越一致越接近 1，惩罚越大。
    """
    rels = np.asarray(rels, dtype=np.float32)
    if rels.ndim != 3:
        raise ValueError(f"rels must be (T, P, D), got {rels.shape}")
    t, p, d = rels.shape
    if p < 2:
        return np.zeros((t,), dtype=np.float32)

    to_evader = -rels
    norms = np.linalg.norm(to_evader, axis=-1, keepdims=True)  # (T, P, 1)
    norms = np.maximum(norms, eps)
    u = to_evader / norms

    # (T, P, P): dot[i,j] = u_i·u_j，再取上三角 i<j 的均值
    dots = np.einsum("tpd,tqd->tpq", u, u)
    num_pairs = p * (p - 1) // 2
    triu = np.triu(np.ones((p, p), dtype=np.float32), k=1)
    penalty = (dots * triu).sum(axis=(1, 2)) / float(num_pairs)
    return penalty.astype(np.float32)


def extract_rels_from_global_state(
    state_bt: np.ndarray,
    *,
    num_pursuers: int,
    spatial_dim: int = 3,
    rels_from_end: bool = True,
    rels_start_idx: int | None = None,
) -> np.ndarray:
    """
    从展平后的全局 state 中取出 rels，reshape 为 (T, P, D)。

    PursuitEvasion3v1Task.build_state 末尾为 pursuer 相对 evader 的位移，长度 P*D。
    """
    state_bt = np.asarray(state_bt, dtype=np.float32)
    if state_bt.ndim != 2:
        raise ValueError(f"state_bt must be (T, S), got {state_bt.shape}")
    flat = num_pursuers * spatial_dim
    if rels_from_end:
        rel_flat = state_bt[:, -flat:]
    else:
        if rels_start_idx is None:
            raise ValueError("rels_start_idx required when rels_from_end=False")
        rel_flat = state_bt[:, rels_start_idx : rels_start_idx + flat]
    return rel_flat.reshape(-1, num_pursuers, spatial_dim)


def spatial_dispersion_penalty_per_timestep(
    state_bt: np.ndarray,
    *,
    num_pursuers: int,
    spatial_dim: int = 3,
    rels_from_end: bool = True,
    rels_start_idx: int | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """state_bt: (T, S) -> penalty (T,)"""
    rels = extract_rels_from_global_state(
        state_bt,
        num_pursuers=num_pursuers,
        spatial_dim=spatial_dim,
        rels_from_end=rels_from_end,
        rels_start_idx=rels_start_idx,
    )
    return pairwise_mean_cosine_similarity(rels, eps=eps)
