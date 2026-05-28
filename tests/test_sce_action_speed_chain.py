"""SCE / E1.1: env action box vs task setpoint scaling (legacy speed chain)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marl_uav.envs.factories import build_env_from_config
from marl_uav.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_sce_effective_max_xy_setpoint_matches_env_and_task() -> None:
    cfg = load_config(ROOT / "configs/experiment/e1_1_open_space_pyflyt_sce.yaml")
    env = build_env_from_config(ROOT / cfg["env"], seed=0, task_cfg=cfg["task"])
    try:
        task = env.task
        a_hi = float(env.action_high_np[0])
        ref = float(task.continuous_action_xy_ref)
        expected = min(a_hi / ref, 1.0) * float(task.pursuer_speed_xy)

        assert a_hi == 0.25
        assert ref == 0.25
        assert float(task.pursuer_speed_xy) == 1.5
        assert expected == 1.5
        assert float(task.pursuer_speed_z) == 0.30
        assert float(task.continuous_action_yaw_ref) == 0.25

        sce = cfg.get("sce") or {}
        assert float(sce["xy_gain"]) == a_hi
        assert float(sce["yaw_gain"]) == 0.25
    finally:
        env.close()


def test_sce_proportional_action_saturates_at_env_high() -> None:
    from marl_uav.control.geometric_pursuit_baselines import proportional_actions_to_targets

    low = np.array([-0.25, -0.25, -0.01, -0.15], dtype=np.float32)
    high = np.array([0.25, 0.25, 0.01, 0.15], dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    g = p.copy()
    g[:, 0] = 5.0
    acts = proportional_actions_to_targets(p, g, low, high, xy_gain=0.25, z_gain=0.20)
    np.testing.assert_allclose(acts[:, 0], 0.25)
    np.testing.assert_allclose(acts[:, 1], 0.0)
