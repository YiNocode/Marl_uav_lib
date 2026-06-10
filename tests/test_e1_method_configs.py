from __future__ import annotations

from pathlib import Path

from marl_uav.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_e1_suite_has_only_requested_methods() -> None:
    suite = load_config(ROOT / "configs/benchmark/e1_open_space_suite.yaml")
    methods = suite["methods"]
    assert list(methods) == ["pure_pursuit", "fixed_ring", "SCE"]
    for name, meta in methods.items():
        assert meta["kind"] == "heuristic", name
        assert meta["train"] is False, name
        cfg = load_config(ROOT / str(meta["config"]))
        assert cfg["env"] == "configs/env/pyflyt_open_space.yaml"
        assert "algo" not in cfg
        assert "model" not in cfg


def test_e1_sce_uses_trajectory_planner_components() -> None:
    cfg = load_config(ROOT / "configs/experiment/e1/sce.yaml")
    task = cfg["task"]
    planner = cfg["trajectory_planner"]
    assert cfg["benchmark"]["method"] == "SCE"
    assert task["role_assignment_mode"] == "entropic_ot"
    assert "manifold_generator" in planner
    assert "slot_allocator" in planner
    assert "obstacle_avoidance" in planner
    assert task["encirclement_capture_enabled"] is True
