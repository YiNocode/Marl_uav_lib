"""Tests for E1.1 suite speed merge and debug speed bounds."""

from __future__ import annotations

from pathlib import Path

from marl_uav.utils.config import load_config
from marl_uav.utils.e1_1_suite import merge_rl_task_speed, resolve_speed_bounds, suite_ref_for_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_merge_rl_task_speed_noops_for_heuristic_sce():
    cfg = load_config(ROOT / "configs/experiment/e1/sce.yaml")
    merged = merge_rl_task_speed(cfg)
    assert merged["task"]["pursuer_speed"] == cfg["task"]["pursuer_speed"]
    assert merged["task"]["evader_speed"] == cfg["task"]["evader_speed"]


def test_heuristic_config_keeps_task_pursuer_speed():
    cfg = load_config(ROOT / "configs/experiment/e1/sce.yaml")
    merged = merge_rl_task_speed(cfg)
    assert merged["task"]["pursuer_speed"] == 0.25


def test_resolve_speed_bounds_heuristic_from_config():
    cfg = load_config(ROOT / "configs/experiment/e1/sce.yaml")
    env_cfg = load_config(ROOT / cfg["env"])
    bounds = resolve_speed_bounds(cfg, env_cfg=env_cfg)
    assert bounds["source"] == "config"
    assert bounds["pursuer_speed_base"] == 0.25
    assert bounds["pursuer_speed_xy"] == 2.5
    assert bounds["pursuer_speed_xy_cap"] == 2.5


def test_suite_ref_uses_new_e1_suite_path():
    cfg = load_config(ROOT / "configs/experiment/e1/sce.yaml")
    assert suite_ref_for_cfg(cfg) == "configs/benchmark/e1_open_space_suite.yaml"
