from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from marl_uav.utils.config import load_config
from marl_uav.utils.debug_viz import resolve_viz_profile
from marl_uav.utils.eval_metrics import aggregate_eval_rows, episode_metrics_from_info
from marl_uav.runners.bc_pretrainer import _resolve_bc_task_cfg

from scripts.run_debug_browser import _build_get_actions_fn


def test_e2_suite_declares_heuristics_and_bc_curriculum_rl() -> None:
    root = Path(__file__).resolve().parents[1]
    suite = load_config(root / "configs/benchmark/e2_obstacle_field_suite.yaml")

    methods = suite.get("methods") or {}
    assert set(methods) == {
        "pure_pursuit_apf",
        "fixed_ring_apf",
        "SCE",
        "fixed_ring_bc_curriculum_mappo",
    }
    assert suite.get("variants")[0]["obstacle_grid_spacing"] == 5.0

    for name in ("pure_pursuit_apf", "fixed_ring_apf", "SCE"):
        meta = methods[name]
        assert meta.get("kind") == "heuristic", name
        assert meta.get("train") is False, name
        cfg_path = root / str(meta.get("config"))
        assert cfg_path.is_file(), name
        cfg = load_config(cfg_path)
        assert str(cfg.get("env", "")) == "configs/env/pyflyt_obstacles.yaml", name
        assert cfg.get("algo") is None, name
        assert cfg.get("model") is None, name
        assert cfg.get("bc_warmstart") is None, name

    rl_meta = methods["fixed_ring_bc_curriculum_mappo"]
    assert rl_meta.get("kind") == "rl"
    assert rl_meta.get("train") is True
    rl_cfg = load_config(root / str(rl_meta.get("config")))
    assert rl_cfg["algo"] == "configs/algo/mappo.yaml"
    assert rl_cfg["model"] == "configs/model/centralized_critic.yaml"
    assert rl_cfg["bc_warmstart"]["expert"] == "fixed_ring_apf"
    assert rl_cfg["bc_warmstart"]["loss_mode"] == "mse"
    assert rl_cfg["bc_warmstart"]["nll_coef"] == 0.0
    assert rl_cfg["bc_warmstart"]["min_eval_capture_rate"] > 0.0
    assert rl_cfg["bc_warmstart"]["update_epochs_per_batch"] > 1
    assert rl_cfg["bc_warmstart"]["dagger_iterations"] > 0
    assert rl_cfg["bc_warmstart"]["task"]["obstacle_grid_spacing"] == 10.0
    assert "obstacle_avoidance" in rl_cfg["bc_warmstart"]["fixed_ring_apf"]
    assert rl_cfg["curriculum"]["values"] == [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]


def test_bc_task_override_preserves_policy_observation_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiment/e2/bc_curriculum_mappo.yaml")
    bc_task = _resolve_bc_task_cfg(cfg)

    assert bc_task is not None
    assert bc_task["obstacle_grid_spacing"] == 10.0
    assert bc_task["role_features_enabled"] is False
    assert bc_task["encirclement_capture_enabled"] is True


def test_bc_eval_terminal_rates_are_exhaustive_for_obstacle_alias() -> None:
    row = episode_metrics_from_info(
        info={
            "episode_return": -1.0,
            "episode_len": 10,
            "capture": False,
            "obstacle_terminated": True,
            "termination_reason": "collision",
        }
    )
    agg = aggregate_eval_rows([row])
    terminal_sum = (
        agg["terminal_capture_rate"]
        + agg["terminal_obstacle_collision_rate"]
        + agg["terminal_inter_agent_collision_rate"]
        + agg["terminal_out_of_bounds_rate"]
        + agg["terminal_timeout_rate"]
        + agg["terminal_other_failure_rate"]
    )

    assert row["terminal_reason"] == "obstacle_collision_terminal"
    assert agg["terminal_obstacle_collision_rate"] == 1.0
    assert terminal_sum == 1.0


def test_debug_browser_supports_e2_apf_heuristics() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiment/e2/pure_pursuit_apf.yaml")
    env = SimpleNamespace(
        _action_space_type="continuous",
        action_low_np=np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32),
        action_high_np=np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32),
    )

    get_actions = _build_get_actions_fn(env, cfg)
    viz = resolve_viz_profile(cfg)

    assert get_actions is not None
    assert viz["method"] == "pure_pursuit_apf"
    assert viz["pursuit_targets"] is True

    fixed_cfg = {"fixed_ring_apf": {}}
    assert _build_get_actions_fn(env, fixed_cfg) is not None
    fixed_viz = resolve_viz_profile(fixed_cfg)
    assert fixed_viz["method"] == "fixed_ring_apf"
    assert fixed_viz["fixed_ring_targets"] is True
