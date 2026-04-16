"""验证多智能体 on-policy batch（EpisodeBatch）形状正确。"""

from __future__ import annotations

import numpy as np
import pytest

from marl_uav.data.batch import EpisodeBatch


def _make_episode_dict(B: int, T: int, N: int, O: int, S: int | None = None, A: int = 4):
    """构造单条或多条 episode 的输入字典（无 B 时均为 T 维）。"""
    S = S or (N * O)
    # 单条 episode: (T, N, O) 等；EpisodeBatch 内部会加 B=1
    obs = np.random.randn(T, N, O).astype(np.float32)
    state = np.random.randn(T, S).astype(np.float32)
    next_state = np.random.randn(T, S).astype(np.float32)
    actions = np.random.randint(0, A, (T, N), dtype=np.int64)
    rewards = np.random.randn(T, N).astype(np.float32)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    values = np.random.randn(T, N).astype(np.float32)
    advantages = np.random.randn(T, N).astype(np.float32)
    returns_ = values + advantages

    return {
        "obs": obs,
        "state": state,
        "next_state": next_state,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "values": values,
        "advantages": advantages,
        "returns": returns_,
    }


# -----------------------------------------------------------------------------
# 1. obs / state / actions / rewards / values / returns 的 shape
# -----------------------------------------------------------------------------


def test_episode_batch_shapes_single_episode():
    """单条 episode (T,N,O) 自动变为 [1,T,N,O]。"""
    T, N, O, S, A = 10, 3, 6, 18, 4
    data = _make_episode_dict(1, T, N, O, S, A)
    batch = EpisodeBatch(**data)

    assert batch.obs.shape == (1, T, N, O), f"obs {batch.obs.shape}"
    assert batch.actions.shape == (1, T, N), f"actions {batch.actions.shape}"
    assert batch.rewards.shape == (1, T, N), f"rewards {batch.rewards.shape}"
    assert batch.dones.shape == (1, T), f"dones {batch.dones.shape}"
    assert batch.values.shape == (1, T, N), f"values {batch.values.shape}"
    assert batch.returns.shape == (1, T, N), f"returns {batch.returns.shape}"


def test_episode_batch_shapes_rewards_global():
    """rewards 可为 (T,) 全局 reward，自动变为 [1, T]。"""
    T, N, O = 5, 2, 4
    data = _make_episode_dict(1, T, N, O)
    data["rewards"] = np.random.randn(T).astype(np.float32)
    batch = EpisodeBatch(**data)

    assert batch.obs.shape == (1, T, N, O)
    assert batch.rewards.shape == (1, T), f"rewards {batch.rewards.shape}"


def test_episode_batch_state_and_next_state_shapes():
    """state / next_state 为 [B, T, S]。"""
    T, N, O, S = 8, 4, 6, 24
    data = _make_episode_dict(1, T, N, O, S)
    batch = EpisodeBatch(**data)

    assert batch.state.shape == (1, T, S), f"state {batch.state.shape}"
    assert batch.next_state.shape == (1, T, S), f"next_state {batch.next_state.shape}"


# -----------------------------------------------------------------------------
# 2. state 的 batch 维与时间维与 obs 对齐
# -----------------------------------------------------------------------------


def test_state_batch_and_time_align_with_obs():
    """state 的 B、T 与 obs 一致。"""
    T, N, O, S = 7, 3, 5, 15
    data = _make_episode_dict(1, T, N, O, S)
    batch = EpisodeBatch(**data)

    B_obs, T_obs, N_obs, _ = batch.obs.shape
    B_s, T_s, S_s = batch.state.shape
    B_ns, T_ns, _ = batch.next_state.shape

    assert B_s == B_obs and T_s == T_obs, "state (B,T) 应与 obs 对齐"
    assert B_ns == B_obs and T_ns == T_obs, "next_state (B,T) 应与 obs 对齐"


def test_state_next_state_same_bt_as_obs():
    """state / next_state 与 obs 的 (B, T) 完全一致，可安全索引。"""
    T, N, O = 4, 2, 6
    data = _make_episode_dict(1, T, N, O)
    batch = EpisodeBatch(**data)

    for b in range(batch.batch_size):
        for t in range(batch.seq_len):
            # 同一 (b,t) 下 obs / state / next_state 应对应同一步
            _ = batch.obs[b, t]  # (N, O)
            _ = batch.state[b, t]  # (S,)
            _ = batch.next_state[b, t]  # (S,)
    assert batch.state.shape[0] == batch.obs.shape[0]
    assert batch.state.shape[1] == batch.obs.shape[1]


# -----------------------------------------------------------------------------
# 3. mask 后维度仍正确
# -----------------------------------------------------------------------------


def test_masks_shape_and_align_with_obs():
    """masks [B,T]、agent_masks [B,T,N] 与 obs 对齐。"""
    T, N, O = 6, 3, 4
    data = _make_episode_dict(1, T, N, O)
    batch = EpisodeBatch(**data)

    assert batch.masks.shape == (1, T), f"masks {batch.masks.shape}"
    assert batch.agent_masks.shape == (1, T, N), f"agent_masks {batch.agent_masks.shape}"


def test_masked_flatten_step_dimension():
    """用 masks 取有效步后，展平的样本数 = 有效步数，维度仍正确。"""
    T, N, O = 5, 2, 6
    data = _make_episode_dict(1, T, N, O)
    # 最后 2 步为 padding（mask=0）
    data["masks"] = np.array([1, 1, 1, 0, 0], dtype=np.float32)
    batch = EpisodeBatch(**data)

    masks = batch.masks  # (1, T)
    obs = batch.obs  # (1, T, N, O)
    state = batch.state  # (1, T, S)

    valid = masks[0] > 0.5
    n_valid = int(valid.sum())
    assert n_valid == 3

    # 取有效步后展平：(n_valid, N, O) / (n_valid, S)
    obs_valid = obs[0][valid]  # (n_valid, N, O)
    state_valid = state[0][valid]  # (n_valid, S)

    assert obs_valid.shape == (n_valid, N, O)
    assert state_valid.shape == (n_valid, state.shape[2])


def test_agent_masks_broadcast_with_obs():
    """agent_masks 与 obs 广播后逐元素乘，形状不变。"""
    T, N, O = 4, 3, 5
    data = _make_episode_dict(1, T, N, O)
    # 部分 agent 无效
    data["agent_masks"] = np.ones((T, N), dtype=np.float32)
    data["agent_masks"][:, 1] = 0.0  # 第 2 个 agent 全为 0
    batch = EpisodeBatch(**data)

    # (1,T,N,O) * (1,T,N,1) 广播
    masked_obs = batch.obs * batch.agent_masks[..., np.newaxis]
    assert masked_obs.shape == batch.obs.shape


def test_avail_actions_shape_when_provided():
    """提供 avail_actions 时为 [B, T, N, A]。"""
    T, N, O, A = 5, 2, 6, 4
    data = _make_episode_dict(1, T, N, O, A=A)
    data["avail_actions"] = np.ones((T, N, A), dtype=np.float32)
    batch = EpisodeBatch(**data)

    assert batch.avail_actions is not None
    assert batch.avail_actions.shape == (1, T, N, A)


def test_batch_properties_match_shape():
    """batch_size / seq_len / num_agents 与数组维度一致。"""
    T, N, O = 10, 4, 8
    data = _make_episode_dict(1, T, N, O)
    batch = EpisodeBatch(**data)

    assert batch.batch_size == 1
    assert batch.seq_len == T
    assert batch.num_agents == N
    assert batch.obs.shape[0] == batch.batch_size
    assert batch.obs.shape[1] == batch.seq_len
    assert batch.obs.shape[2] == batch.num_agents
