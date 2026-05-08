"""Environment factory helpers shared by training, evaluation, and VecEnv workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.utils.config import load_config
from marl_uav.utils.env_action_bounds import parse_continuous_action_bounds_from_env_cfg


def build_env_from_config(
    env_cfg_path: Path,
    seed: int,
    task_cfg: dict[str, Any] | None = None,
):
    """Build a single environment instance from config."""
    cfg = load_config(env_cfg_path)
    env_id = cfg.get("env_id", "toy_uav")

    if env_id == "toy_uav":
        return ToyUavEnv.from_config(cfg, seed=seed)

    if env_id == "pyflyt_navigation":
        from marl_uav.envs.adapters.pyflyt_aviary_env import PyFlytAviaryEnv
        from marl_uav.envs.backends.pyflyt_aviary_backend import PyFlytAviaryBackend
        from marl_uav.envs.tasks.navigation_task import NavigationTask
        from marl_uav.envs.tasks.pursuit_evasion_3v1_task import PursuitEvasion3v1Task
        from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import (
            PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx1,
        )
        from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import (
            PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx2,
        )

        backend_cfg = cfg.get("backend", {})
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
        task_name = str(task_params.pop("name", "navigation"))

        if task_name == "navigation":
            task = NavigationTask(**task_params) if task_params else NavigationTask()
        elif task_name == "pursuit_evasion_3v1":
            task = PursuitEvasion3v1Task(**task_params) if task_params else PursuitEvasion3v1Task()
        elif task_name == "pursuit_evasion_3v1_ex1":
            task = PursuitEvasion3v1TaskEx1(**task_params) if task_params else PursuitEvasion3v1TaskEx1()
        elif task_name == "pursuit_evasion_3v1_ex2":
            task = PursuitEvasion3v1TaskEx2(**task_params) if task_params else PursuitEvasion3v1TaskEx2()
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

    raise ValueError(f"Unsupported env_id={env_id!r} in {env_cfg_path}")
