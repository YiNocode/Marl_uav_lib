from __future__ import annotations

from pathlib import Path

from marl_uav.utils.config import load_config


def test_e2_suite_is_heuristic_only() -> None:
    root = Path(__file__).resolve().parents[1]
    suite = load_config(root / "configs/benchmark/e2_obstacles_suite.yaml")

    methods = suite.get("methods") or {}
    assert methods
    for name, meta in methods.items():
        assert meta.get("kind") == "heuristic", name
        assert meta.get("train") is False, name
        cfg_path = root / str(meta.get("config"))
        assert cfg_path.is_file(), name
        cfg = load_config(cfg_path)
        assert str(cfg.get("env", "")).startswith("configs/env/e2_"), name
        assert cfg.get("algo") is None, name
        assert cfg.get("model") is None, name
        assert cfg.get("bc_warmstart") is None, name
