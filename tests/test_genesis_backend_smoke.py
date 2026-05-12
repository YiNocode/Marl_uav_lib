"""Smoke test for the optional Genesis 3v1 backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from marl_uav.envs.factories import build_env_from_config


@pytest.mark.skipif(importlib.util.find_spec("genesis") is None, reason="Genesis is not installed.")
def test_genesis_3v1_smoke():
    """Create Genesis env, reset, and run a few random actions."""
    task_cfg = {
        "name": "pursuit_evasion_3v1_ex2",
        "world_xy": 2.0,
        "z_min": 0.5,
        "z_max": 1.5,
        "episode_limit": 20,
        "num_obstacles_min": 1,
        "num_obstacles_max": 2,
    }
    env = build_env_from_config(Path("configs/env/genesis_3v1.yaml"), seed=0, task_cfg=task_cfg)
    try:
        obs_dict, info = env.reset(seed=0)
        obs = np.asarray(obs_dict["obs"], dtype=np.float32)
        state = np.asarray(obs_dict["state"], dtype=np.float32)
        assert obs.shape[0] == 3
        assert state.ndim == 1
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(state))

        rng = np.random.default_rng(0)
        low = np.broadcast_to(env.action_low_np, (env.num_agents, env.action_dim))
        high = np.broadcast_to(env.action_high_np, (env.num_agents, env.action_dim))
        for _ in range(10):
            actions = rng.uniform(low, high).astype(np.float32)
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            obs = np.asarray(obs_dict["obs"], dtype=np.float32)
            state = np.asarray(obs_dict["state"], dtype=np.float32)
            reward_arr = np.asarray(reward, dtype=np.float32)
            assert np.all(np.isfinite(obs))
            assert np.all(np.isfinite(state))
            assert np.all(np.isfinite(reward_arr))
            assert isinstance(bool(terminated), bool)
            assert isinstance(bool(truncated), bool)
            assert "capture" in info
            assert "oob" in info
            assert "metrics" in info
            if terminated or truncated:
                break
    finally:
        env.close()
