"""Merge env-level task defaults with experiment task overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from marl_uav.utils.config import load_config


def merge_task_with_env_defaults(
    env_cfg: dict[str, Any],
    task_cfg: dict[str, Any] | None = None,
    *,
    env_cfg_path: Path | None = None,
) -> dict[str, Any]:
    """Layer experiment ``task`` overrides on top of env ``task_defaults``."""
    merged: dict[str, Any] = {}

    scenario_path = env_cfg.get("scenario_config")
    if scenario_path:
        scenario_cfg = load_config(_resolve_config_path(scenario_path, env_cfg_path))
        merged.update(dict(scenario_cfg.get("task_defaults") or {}))

    merged.update(dict(env_cfg.get("task_defaults") or {}))
    merged.update(dict(task_cfg or {}))
    return merged


def resolve_task_cfg_for_env(
    env_cfg_path: Path,
    task_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env_cfg = load_config(env_cfg_path)
    return merge_task_with_env_defaults(env_cfg, task_cfg, env_cfg_path=env_cfg_path)


def _resolve_config_path(path: str | Path, base: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if base is not None:
        relative = (base.parent / candidate).resolve()
        if relative.is_file():
            return relative
    root = Path(__file__).resolve().parents[2]
    rooted = (root / candidate).resolve()
    if rooted.is_file():
        return rooted
    return candidate
