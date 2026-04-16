"""Tests for EpisodeBuffer: state storage and shapes."""

from __future__ import annotations

import numpy as np

from marl_uav.buffers.episode_buffer import EpisodeBuffer


def _make_step(t: int, n_agents: int, obs_dim: int, state_dim: int):
    obs = [np.full((obs_dim,), fill_value=t + i, dtype=np.float32) for i in range(n_agents)]
    state = np.full((state_dim,), fill_value=10 * t + 1, dtype=np.float32)
    next_obs = [np.full((obs_dim,), fill_value=t + i + 0.5, dtype=np.float32) for i in range(n_agents)]
    next_state = np.full((state_dim,), fill_value=10 * t + 2, dtype=np.float32)
    actions = np.arange(n_agents, dtype=np.int64) + t
    rewards = np.arange(n_agents, dtype=np.float32) + 0.1 * t
    done = False
    return obs, state, actions, rewards, next_obs, next_state, done


def test_episode_buffer_state_stored_and_stacked_correctly():
    n_agents, obs_dim, state_dim = 3, 4, 5
    buf = EpisodeBuffer(num_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim)

    T = 4
    states = []
    next_states = []
    for t in range(T):
        obs, state, actions, rewards, next_obs, next_state, done = _make_step(
            t, n_agents, obs_dim, state_dim
        )
        buf.add(
            obs=obs,
            state=state,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            next_state=next_state,
            done=done,
        )
        states.append(state)
        next_states.append(next_state)

    episode = buf.get_episode()
    state_arr = episode["state"]
    next_state_arr = episode["next_state"]

    assert state_arr.shape == (T, state_dim)
    assert next_state_arr.shape == (T, state_dim)
    np.testing.assert_allclose(state_arr, np.stack(states, axis=0))
    np.testing.assert_allclose(next_state_arr, np.stack(next_states, axis=0))


def test_episode_buffer_state_unchanged_when_optional_fields_missing():
    """即使部分时间步 log_probs / values 为 None，state 仍按时间正确堆叠。"""
    n_agents, obs_dim, state_dim = 2, 3, 7
    buf = EpisodeBuffer(num_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim)

    T = 3
    states = []
    for t in range(T):
        obs, state, actions, rewards, next_obs, next_state, done = _make_step(
            t, n_agents, obs_dim, state_dim
        )
        # 交替缺失 log_probs / values，模拟“padding”场景
        log_probs = None if t % 2 == 0 else np.zeros((n_agents,), dtype=np.float32)
        values = None if t % 2 == 1 else np.ones((n_agents,), dtype=np.float32)
        buf.add(
            obs=obs,
            state=state,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            next_state=next_state,
            done=done,
            log_probs=log_probs,
            values=values,
        )
        states.append(state)

    episode = buf.get_episode()
    state_arr = episode["state"]
    assert state_arr.shape == (T, state_dim)
    np.testing.assert_allclose(state_arr, np.stack(states, axis=0))


def test_episode_buffer_sample_shapes_align_for_mappo():
    """sample 出来的 episode 字典中 state/obs/values/returns 形状可直接给 MAPPO 用。"""
    n_agents, obs_dim, state_dim = 2, 6, 8
    buf = EpisodeBuffer(num_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim)

    T = 5
    for t in range(T):
        obs, state, actions, rewards, next_obs, next_state, done = _make_step(
            t, n_agents, obs_dim, state_dim
        )
        # 简单填充 log_probs / values，方便后续 GAE/returns 计算
        log_probs = np.full((n_agents,), -0.5, dtype=np.float32)
        values = np.full((n_agents,), 1.0, dtype=np.float32)
        buf.add(
            obs=obs,
            state=state,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            next_state=next_state,
            done=done if t == T - 1 else False,
            log_probs=log_probs,
            values=values,
        )

    episode = buf.sample(batch_size=1)  # type: ignore[assignment]
    assert episode is not None

    obs = episode["obs"]
    state = episode["state"]
    actions = episode["actions"]
    rewards = episode["rewards"]
    dones = episode["dones"]
    log_probs = episode.get("log_probs")
    values = episode.get("values")

    assert obs.shape == (T, n_agents, obs_dim)
    assert state.shape == (T, state_dim)
    assert actions.shape == (T, n_agents)
    assert rewards.shape[0] == T
    assert dones.shape == (T,)
    assert log_probs is not None and log_probs.shape == (T, n_agents)
    assert values is not None and values.shape == (T, n_agents)

