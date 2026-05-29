from __future__ import annotations

from pathlib import Path

from marl_uav.envs.factories import build_env_from_config
from marl_uav.utils.env_task_config import merge_task_with_env_defaults, resolve_task_cfg_for_env

ROOT = Path(__file__).resolve().parents[1]


def test_merge_task_with_env_scenario_defaults():
    env_cfg_path = ROOT / "configs/env/e2_pyflyt_3v1_obstacles_heuristic.yaml"
    merged = resolve_task_cfg_for_env(
        env_cfg_path,
        {"pursuer_speed": 0.25, "progress_reward_scale": 0.0},
    )
    assert merged["name"] == "pursuit_evasion_3v1_ex2"
    assert merged["world_xy"] == 20.0
    assert merged["obstacle_grid_spacing"] == 4.0
    assert merged["obstacle_collision_penalty"] == 15.0
    assert merged["pursuer_speed"] == 0.25
    assert merged["progress_reward_scale"] == 0.0


def test_experiment_override_wins_over_env_defaults():
    env_cfg = {
        "scenario_config": "configs/env/e2_obstacle_scenario.yaml",
    }
    merged = merge_task_with_env_defaults(
        env_cfg,
        {"obstacle_grid_spacing": 6.0},
        env_cfg_path=ROOT / "configs/env/e2_pyflyt_3v1_obstacles.yaml",
    )
    assert merged["obstacle_grid_spacing"] == 6.0


def test_build_env_from_e2_config_applies_scenario_defaults():
    env = build_env_from_config(
        ROOT / "configs/env/e2_pyflyt_3v1_obstacles_heuristic.yaml",
        seed=101,
        task_cfg={"pursuer_speed": 0.25, "evader_speed": 0.25},
    )
    try:
        assert env.task.obstacle_grid_spacing == 4.0
        assert env.task.world_xy == 20.0
        assert env.task.num_obstacles_min == 121
    finally:
        env.close()
