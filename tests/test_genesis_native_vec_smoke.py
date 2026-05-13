from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from marl_uav.envs.genesis_vec_env_manager import GenesisVecEnvManager


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("genesis") is None,
    reason="Genesis is not installed.",
)


def test_genesis_native_vec_smoke():
    """Create one Genesis native-vector scene and step without subprocess workers."""
    manager = GenesisVecEnvManager(
        env_cfg_path=Path("configs/env/genesis_3v1.yaml"),
        task_cfg={"name": "pursuit_evasion_3v1_ex1", "episode_limit": 20},
        num_envs=2,
        seed=7,
    )
    try:
        obs, state, avail, _ = manager.reset(seed=7)
        assert obs.shape[0] == 2
        assert obs.shape[1] == 3
        assert state.shape[0] == 2
        assert avail is not None and avail.shape[:2] == (2, 3)
        for _ in range(3):
            actions = np.random.uniform(-0.05, 0.05, size=(2, 3, 4)).astype(np.float32)
            step = manager.step(actions)
            assert step.obs.shape[:2] == (2, 3)
            assert step.state.shape[0] == 2
            assert step.rewards.shape == (2, 3)
            assert np.all(np.isfinite(step.obs))
            assert np.all(np.isfinite(step.state))
            assert np.all(np.isfinite(step.rewards))
            assert step.dones.dtype == np.bool_
    finally:
        manager.close()
