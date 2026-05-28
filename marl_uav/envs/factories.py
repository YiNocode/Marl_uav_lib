"""Environment factory helpers shared by training, evaluation, and VecEnv workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.utils.config import load_config
from marl_uav.utils.env_action_bounds import parse_continuous_action_bounds_from_env_cfg


def build_pursuit_task_from_config(
    task_cfg: dict[str, Any] | None,
    *,
    default_name: str,
):
    """Build one of the supported 3v1 pursuit task variants."""
    from marl_uav.envs.tasks.pursuit_evasion_3v1_task import PursuitEvasion3v1Task
    from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import (
        PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx1,
    )
    from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import (
        PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx2,
    )

    task_params = dict(task_cfg or {})
    task_params.pop("debug_browser", None)
    task_name = str(task_params.pop("name", default_name))
    if task_name == "pursuit_evasion_3v1":
        return PursuitEvasion3v1Task(**task_params) if task_params else PursuitEvasion3v1Task()
    if task_name == "pursuit_evasion_3v1_ex1":
        return PursuitEvasion3v1TaskEx1(**task_params) if task_params else PursuitEvasion3v1TaskEx1()
    if task_name == "pursuit_evasion_3v1_ex2":
        return PursuitEvasion3v1TaskEx2(**task_params) if task_params else PursuitEvasion3v1TaskEx2()
    raise ValueError(f"Unsupported pursuit task name={task_name!r}")


def env_config_uses_genesis(env_cfg_path: Path) -> bool:
    """Return whether an env YAML requests the Genesis backend."""
    cfg = load_config(env_cfg_path)
    return str(cfg.get("backend", "pyflyt")).lower() == "genesis"


def build_env_from_config(
    env_cfg_path: Path,
    seed: int,
    task_cfg: dict[str, Any] | None = None,
):
    """Build a single environment instance from config."""
    cfg = load_config(env_cfg_path)
    return build_env_from_config_dict(cfg, seed=seed, task_cfg=task_cfg, env_cfg_path=env_cfg_path)


def build_env_from_config_dict(
    cfg: dict[str, Any],
    *,
    seed: int,
    task_cfg: dict[str, Any] | None = None,
    env_cfg_path: Path | None = None,
):
    """Build a single environment from an already loaded env config."""
    env_id = cfg.get("env_id", "toy_uav")
    backend_spec = cfg.get("backend", "pyflyt")
    backend_name = str(backend_spec).lower() if isinstance(backend_spec, str) else "pyflyt"

    if env_id == "toy_uav":
        return ToyUavEnv.from_config(cfg, seed=seed)

    if backend_name == "genesis":
        from marl_uav.envs.adapters.genesis_pursuit_env import GenesisPursuitEvasionEnv
        from marl_uav.envs.backends.genesis_backend import GenesisBackend

        task = build_pursuit_task_from_config(task_cfg, default_name="pursuit_evasion_3v1_ex1")

        action_space = str(cfg.get("action_space", "continuous")).lower()
        action_dim = int(cfg.get("action_dim", 4))
        action_low, action_high = parse_continuous_action_bounds_from_env_cfg(
            cfg,
            action_space=action_space,
            action_dim=action_dim,
        )

        backend_cfg = dict(cfg.get("backend_config", {}) or {})
        backend_cfg.setdefault("world_xy", float(getattr(task, "world_xy", 2.0)))
        backend_cfg.setdefault("z_min", float(getattr(task, "z_min", 0.5)))
        backend_cfg.setdefault("z_max", float(getattr(task, "z_max", 2.0)))
        backend_cfg.setdefault("episode_limit", int(getattr(task, "episode_limit", 400)))
        backend_cfg.setdefault("num_pursuers", 3)
        backend_cfg.setdefault("num_evaders", 1)
        if action_space == "continuous" and action_dim == 4:
            backend_cfg.setdefault("velocity_low", action_low)
            backend_cfg.setdefault("velocity_high", action_high)
        backend_cfg.setdefault("seed", seed)
        backend = GenesisBackend(**backend_cfg)

        return GenesisPursuitEvasionEnv(
            backend=backend,
            task=task,
            seed=seed,
            action_space=action_space,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        )

    if env_id == "pyflyt_navigation":
        from marl_uav.envs.adapters.pyflyt_aviary_env import PyFlytAviaryEnv
        from marl_uav.envs.backends.pyflyt_aviary_backend import PyFlytAviaryBackend
        from marl_uav.envs.tasks.navigation_task import NavigationTask

        if backend_name != "pyflyt":
            raise ValueError(f"Unknown backend: {backend_name}")
        backend_cfg = cfg.get("backend_config", {}) if isinstance(backend_spec, str) else cfg.get("backend", {})
        num_agents = int(backend_cfg.get("num_agents", 1))
        backend = PyFlytAviaryBackend(
            num_agents=num_agents,
            drone_type=backend_cfg.get("drone_type", "quadx"),
            render=bool(backend_cfg.get("render", False)),
            physics_hz=int(backend_cfg.get("physics_hz", 240)),
            control_hz=int(backend_cfg.get("control_hz", 60)),
            world_scale=float(backend_cfg.get("world_scale", 5.0)),
            drone_options=backend_cfg.get("drone_options", {}) or {},
            seed=seed + int(backend_cfg.get("seed_offset", 0)),
            flight_mode=int(backend_cfg.get("flight_mode", 6)),
        )

        task_params = dict(task_cfg or {})
        task_name = str(task_params.get("name", "navigation"))
        if task_name == "navigation":
            task_params.pop("name", None)
            task = NavigationTask(**task_params) if task_params else NavigationTask()
        elif task_name.startswith("pursuit_evasion_3v1"):
            task = build_pursuit_task_from_config(task_cfg, default_name="pursuit_evasion_3v1")
        else:
            raise ValueError(f"Unsupported task name={task_name!r} for env_id={env_id!r}")

        action_space = str(cfg.get("action_space", "discrete")).lower()
        action_dim = int(cfg.get("action_dim", 4))
        action_low, action_high = parse_continuous_action_bounds_from_env_cfg(
            cfg,
            action_space=action_space,
            action_dim=action_dim,
        )
        return PyFlytAviaryEnv(
            backend=backend,
            task=task,
            seed=seed,
            action_space=cfg.get("action_space", "discrete"),
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        )

    source = "" if env_cfg_path is None else f" in {env_cfg_path}"
    raise ValueError(f"Unsupported env_id={env_id!r}{source}")
