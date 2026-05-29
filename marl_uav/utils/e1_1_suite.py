"""E1.1 open-space benchmark suite helpers (speed defaults, config merge)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from marl_uav.utils.config import load_config

E1_1_SUITE_REL = Path("configs/benchmark/e1_1_open_space_suite.yaml")
E2_OBSTACLES_SUITE_REL = Path("configs/benchmark/e2_obstacles_suite.yaml")


def load_e1_1_suite(path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return load_config(path or (root / E1_1_SUITE_REL))


def load_e2_obstacles_suite(path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return load_config(path or (root / E2_OBSTACLES_SUITE_REL))


def is_benchmark_scenario_cfg(cfg: dict[str, Any], scenario: str) -> bool:
    bench = cfg.get("benchmark") or {}
    return str(bench.get("scenario", "")).strip() == str(scenario).strip()


def is_rl_experiment_cfg(cfg: dict[str, Any]) -> bool:
    return "algo" in cfg


def merge_rl_task_speed(
    cfg: dict[str, Any],
    suite: dict[str, Any] | None = None,
    *,
    method: str | None = None,
) -> dict[str, Any]:
    """Inject RL task speed fields from a benchmark suite when absent from experiment YAML."""
    if not is_rl_experiment_cfg(cfg):
        return cfg

    suite = suite or load_e1_1_suite()
    expected_scenario = str(suite.get("scenario", "")).strip()
    bench = cfg.get("benchmark") or {}
    cfg_scenario = str(bench.get("scenario", "")).strip()
    if cfg_scenario and expected_scenario and cfg_scenario != expected_scenario:
        return cfg

    method_name = (method or str(bench.get("method", ""))).strip()
    if not method_name:
        return cfg

    method_meta = dict((suite.get("methods") or {}).get(method_name) or {})
    speed_root = dict(suite.get("speed") or {})
    rl_defaults = dict(speed_root.get("rl_defaults") or {})
    task_speed = {**rl_defaults, **dict(method_meta.get("task_speed") or {})}
    if not task_speed:
        return cfg

    out = dict(cfg)
    task = dict(out.get("task") or {})
    for key, value in task_speed.items():
        if key not in task:
            task[key] = value
    out["task"] = task
    return out


def _action_high_xy(*, env: Any | None, env_cfg: dict[str, Any] | None) -> float | None:
    if env is not None:
        high_np = getattr(env, "action_high_np", None)
        if high_np is not None:
            return float(high_np[0])
    if env_cfg:
        high = env_cfg.get("action_high")
        if isinstance(high, (list, tuple)) and high:
            return float(high[0])
    return None


def resolve_speed_bounds(
    cfg: dict[str, Any],
    *,
    env: Any | None = None,
    env_cfg: dict[str, Any] | None = None,
    suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Planar speed caps for debug visualization and sanity checks."""
    suite = suite or load_e1_1_suite()
    speed_root = dict(suite.get("speed") or {})
    ref_world = float(speed_root.get("reference_world_xy", 2.0))
    task = dict(cfg.get("task") or {})
    world_xy = float(task.get("world_xy", 20.0))
    xy_ref = float(
        task.get(
            "continuous_action_xy_ref",
            (speed_root.get("rl_defaults") or {}).get("continuous_action_xy_ref", 0.25),
        )
    )
    action_high_xy = _action_high_xy(env=env, env_cfg=env_cfg)

    is_rl = is_rl_experiment_cfg(cfg)
    if is_rl:
        merged = merge_rl_task_speed(cfg, suite=suite)
        task = dict(merged.get("task") or {})
        base_pursuer = float(task["pursuer_speed"])
        source = "suite"
        suite_ref = str(E1_1_SUITE_REL).replace("\\", "/")
    else:
        if "pursuer_speed" not in task:
            raise ValueError("Non-RL E1.1 config must set task.pursuer_speed")
        base_pursuer = float(task["pursuer_speed"])
        source = "config"
        suite_ref = None

    pursuer_speed_xy = base_pursuer * (world_xy / ref_world)
    if action_high_xy is not None and xy_ref > 0:
        planar_cap = min(action_high_xy / xy_ref, 1.0) * pursuer_speed_xy
    else:
        planar_cap = pursuer_speed_xy

    evader_base = float(task.get("evader_speed", 0.2))
    evader_speed_xy = evader_base * (world_xy / ref_world)

    return {
        "source": source,
        "suite_ref": suite_ref,
        "pursuer_speed_base": base_pursuer,
        "pursuer_speed_xy": pursuer_speed_xy,
        "pursuer_speed_xy_cap": planar_cap,
        "evader_speed_base": evader_base,
        "evader_speed_xy_cap": evader_speed_xy,
        "continuous_action_xy_ref": xy_ref,
        "action_high_xy": action_high_xy,
    }
