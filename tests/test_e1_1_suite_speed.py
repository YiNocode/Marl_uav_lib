"""Tests for E1.1 suite speed merge and debug speed bounds."""

from __future__ import annotations

from pathlib import Path

from marl_uav.utils.config import load_config
from marl_uav.utils.e1_1_suite import merge_rl_task_speed, resolve_speed_bounds

ROOT = Path(__file__).resolve().parents[1]


def test_merge_rl_task_speed_injects_from_suite():
    cfg = load_config(ROOT / "configs/experiment/e1_1_open_space_pyflyt_mappo.yaml")
    assert "pursuer_speed" not in cfg["task"]
    merged = merge_rl_task_speed(cfg)
    assert merged["task"]["pursuer_speed"] == 0.25
    assert merged["task"]["evader_speed"] == 0.08
    assert merged["task"]["continuous_action_xy_ref"] == 0.25


def test_heuristic_config_keeps_task_pursuer_speed():
    cfg = load_config(ROOT / "configs/experiment/e1_1_open_space_pyflyt_sce.yaml")
    merged = merge_rl_task_speed(cfg)
    assert merged["task"]["pursuer_speed"] == 0.15


def test_resolve_speed_bounds_heuristic_from_config():
    cfg = load_config(ROOT / "configs/experiment/e1_1_open_space_pyflyt_sce.yaml")
    env_cfg = load_config(ROOT / cfg["env"])
    bounds = resolve_speed_bounds(cfg, env_cfg=env_cfg)
    assert bounds["source"] == "config"
    assert bounds["pursuer_speed_base"] == 0.15
    assert bounds["pursuer_speed_xy"] == 1.5
    assert bounds["pursuer_speed_xy_cap"] == 1.5


def test_resolve_speed_bounds_rl_from_suite():
    cfg = merge_rl_task_speed(load_config(ROOT / "configs/experiment/e1_1_open_space_pyflyt_mappo.yaml"))
    env_cfg = load_config(ROOT / cfg["env"])
    bounds = resolve_speed_bounds(cfg, env_cfg=env_cfg)
    assert bounds["source"] == "suite"
    assert bounds["pursuer_speed_base"] == 0.25
    assert bounds["pursuer_speed_xy_cap"] == 2.5
    assert "e1_1_open_space_suite.yaml" in (bounds["suite_ref"] or "")
