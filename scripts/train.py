"""Training entry script (on-policy IPPO / MAPPO via config)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.agents.mac import MAC
from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
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
from marl_uav.learners.on_policy import IPPOLearner, MAPPOLearner, SCMAPPOLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy
from marl_uav.policies.dream_mappo_policy import DreamMappoCentralizedCriticPolicy
from marl_uav.runners.evaluator import Evaluator
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.runners.trainer import Trainer
from marl_uav.utils.checkpoint import CheckpointManager
from marl_uav.utils.config import load_config
from marl_uav.utils.env_action_bounds import (
    boxed_action_bounds,
    parse_continuous_action_bounds_from_env_cfg,
)
from marl_uav.utils.logger import Logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_dream_mappo_3v1.yaml"),
        help="顶层训练配置 (包含 env/algo/model/task 子配置路径)",
    )
    return p.parse_args()


def build_env(env_cfg_path: Path, seed: int, task_cfg: dict[str, Any] | None = None):
    """根据 env 配置构建环境实例.

    支持:
        - ToyUavEnv (env_id: toy_uav)
        - PyFlytAviaryEnv + NavigationTask / PursuitEvasion3v1Task / pursuit_evasion_3v1_ex1 / pursuit_evasion_3v1_ex2
          (env_id: pyflyt_navigation)
    """
    cfg = load_config(env_cfg_path)
    env_id = cfg.get("env_id", "toy_uav")

    if env_id == "toy_uav":
        return ToyUavEnv.from_config(cfg, seed=seed)

    if env_id == "pyflyt_navigation":
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

        # Task 参数来自顶层 train config 中的 task 字段
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
        _aspace = str(cfg.get("action_space", "discrete")).lower()
        _adim = int(cfg.get("action_dim", 4))
        _alow, _ahigh = parse_continuous_action_bounds_from_env_cfg(
            cfg, action_space=_aspace, action_dim=_adim
        )
        return PyFlytAviaryEnv(
            backend=backend,
            task=task,
            seed=seed,
            action_space=cfg.get("action_space", "discrete"),
            action_dim=_adim,
            action_low=_alow,
            action_high=_ahigh,
        )

    raise ValueError(f"Unsupported env_id={env_id!r} in {env_cfg_path}")


def build_policy(
    model_cfg_path: Path,
    env: Any,
    algo_cfg_path: Path,
) -> Any:
    """根据 model 与 algo 配置构建策略；algo 中 action_space 决定离散/连续。"""
    cfg = load_config(model_cfg_path)
    algo_cfg = load_config(algo_cfg_path)
    model_type = cfg.get("type", "mlp")
    action_space = str(algo_cfg.get("action_space", "discrete")).lower()

    if action_space not in ("discrete", "continuous"):
        raise ValueError(
            f"action_space must be 'discrete' or 'continuous', got {algo_cfg.get('action_space')!r}"
        )

    # 若 env 提供自身的动作空间类型，则与 algo 配置进行一致性校验
    env_action_space = str(
        getattr(env, "_action_space_type", getattr(env, "action_space_type", "")) or ""
    ).lower()
    if env_action_space in ("discrete", "continuous") and env_action_space != action_space:
        raise ValueError(
            "Mismatch between algo.action_space and env.action_space: "
            f"algo={action_space!r}, env={env_action_space!r}. "
            "请在 env config 和 algo config 中使用一致的 action_space 设置，"
            "否则可能导致 env.n_actions 为 0 或 logits 维度错误。"
        )

    if model_type == "centralized_critic":
        if action_space == "discrete":
            return CentralizedCriticPolicy(
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                n_actions=env.n_actions,
                action_space_type="discrete",
            )
        action_dim = getattr(env, "action_dim", None)
        if action_dim is None and hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
            action_dim = int(env.action_space.shape[0])
        if action_dim is None:
            raise ValueError(
                "For centralized_critic + continuous, env must provide action_dim or action_space.shape."
            )
        low, high = boxed_action_bounds(env, action_dim)
        log_std_init = float(algo_cfg.get("log_std_init", -0.5))
        return CentralizedCriticPolicy(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_space_type="continuous",
            action_dim=action_dim,
            action_low=low,
            action_high=high,
            log_std_init=log_std_init,
        )

    if model_type == "dream_mappo_centralized_critic":
        if action_space != "continuous":
            raise ValueError("dream_mappo_centralized_critic 仅支持 continuous action_space。")
        action_dim = getattr(env, "action_dim", None)
        if action_dim is None and hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
            action_dim = int(env.action_space.shape[0])
        if action_dim is None:
            raise ValueError(
                "For dream_mappo_centralized_critic, env must provide action_dim or action_space.shape."
            )
        low, high = boxed_action_bounds(env, action_dim)
        log_std_init = float(algo_cfg.get("log_std_init", -0.5))
        dream_cfg = cfg.get("dream", {}) or {}
        return DreamMappoCentralizedCriticPolicy(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=action_dim,
            action_low=low,
            action_high=high,
            log_std_init=log_std_init,
            num_pursuers=int(dream_cfg.get("num_pursuers", 3)),
            a_max_geom=float(dream_cfg.get("a_max_geom", 0.15)),
            sigma_p=float(dream_cfg.get("sigma_p", 0.5)),
            rho_scale=float(dream_cfg.get("rho_scale", 0.5)),
            rho_min=float(dream_cfg.get("rho_min", 0.05)),
            psi_scale=float(dream_cfg.get("psi_scale", 3.14159265)),
            a_max_residual=float(dream_cfg.get("a_max_residual", 0.08)),
        )

    # IPPO 的 critic 只用 obs，不传 state_dim，与 checkpoint 加载一致
    algo_name = algo_cfg.get("algo", "ippo").lower()
    state_dim = None if algo_name == "ippo" else getattr(env, "state_dim", None)
    if action_space == "discrete":
        return ActorCriticPolicy(
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
            state_dim=state_dim,
            action_space_type="discrete",
        )
    # 连续动作：从 env 取 action_dim（或 action_space.shape[0]）
    action_dim = getattr(env, "action_dim", None)
    if action_dim is None and hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        action_dim = int(env.action_space.shape[0])
    if action_dim is None:
        raise ValueError(
            "For action_space=continuous, env must provide action_dim or action_space.shape."
        )
    log_std_init = float(algo_cfg.get("log_std_init", -0.5))
    low, high = boxed_action_bounds(env, action_dim)
    return ActorCriticPolicy(
        obs_dim=env.obs_dim,
        action_space_type="continuous",
        action_dim=action_dim,
        state_dim=state_dim,
        log_std_init=log_std_init,
        action_low=low,
        action_high=high,
    )


def build_learner(algo_cfg_path: Path, policy: Any) -> tuple[Any, dict[str, Any]]:
    cfg = load_config(algo_cfg_path)
    algo_name = cfg.get("algo", "ippo").lower()

    # 通用超参数
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    # PPO 相关
    clip_ratio = float(cfg.get("clip_ratio", 0.2))
    value_coef = float(cfg.get("value_coef", cfg.get("vf_coef", 0.5)))
    entropy_coef = float(cfg.get("entropy_coef", cfg.get("ent_coef", 0.01)))
    lr = float(cfg.get("lr", 3e-4))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    num_epochs = int(cfg.get("epochs", 4))

    learner_kwargs = dict(
        lr=lr,
        clip_range=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
    )

    if algo_name == "sc_mappo":
        dispersion_coef = float(cfg.get("dispersion_coef", 0.05))
        num_pursuers = int(cfg.get("num_pursuers", 3))
        spatial_dim = int(cfg.get("spatial_dim", 3))
        rels_from_end = bool(cfg.get("rels_from_end", True))
        rels_start = cfg.get("rels_start_idx")
        rels_start_idx = None if rels_start is None else int(rels_start)
        learner = SCMAPPOLearner(
            policy=policy,
            dispersion_coef=dispersion_coef,
            num_pursuers=num_pursuers,
            spatial_dim=spatial_dim,
            rels_from_end=rels_from_end,
            rels_start_idx=rels_start_idx,
            **learner_kwargs,
        )
    elif algo_name == "mappo":
        learner = MAPPOLearner(policy=policy, **learner_kwargs)
    elif algo_name == "dream_mappo":
        learner = MAPPOLearner(policy=policy, **learner_kwargs)
    elif algo_name == "ippo":
        learner = IPPOLearner(policy=policy, **learner_kwargs)
    else:
        raise ValueError(f"Unsupported algo={algo_name!r} in {algo_cfg_path}")

    trainer_kwargs = dict(gamma=gamma, gae_lambda=gae_lambda)
    return learner, trainer_kwargs


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    train_cfg = load_config(root / args.train_config)

    env_cfg_path = root / train_cfg.get("env", "configs/env/toy_uav.yaml")
    algo_cfg_path = root / train_cfg.get("algo", "configs/algo/ippo.yaml")
    model_cfg_path = root / train_cfg.get("model", "configs/model/.yaml")

    seed = int(train_cfg.get("seed", 42))
    num_epochs = int(train_cfg.get("num_epochs", 10))
    rollout_steps = int(train_cfg.get("rollout_steps", 1024))
    log_interval = int(train_cfg.get("log_interval", 1))
    eval_episodes = int(train_cfg.get("eval_episodes", 5))

    task_cfg = train_cfg.get("task", {})
    env = build_env(env_cfg_path, seed=seed, task_cfg=task_cfg)

    # 若环境还未初始化 obs_dim/state_dim，则在构建 policy 前先 reset 一次
    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        try:
            env.reset(seed=seed)
        except TypeError:
            # 兼容不接受 seed 参数的 reset 签名
            env.reset()

    # 结果与 TensorBoard 日志 / Checkpoint 目录
    # 按 train-config 名区分实验，再按 seed 区分不同随机种子
    exp_name = Path(args.train_config).stem
    results_dir = root / "results" / exp_name
    tb_log_dir = results_dir / "tb_" / str(seed)
    tb_logger = Logger(log_dir=tb_log_dir)

    # 构建 policy & MAC（policy 由 algo 的 action_space 决定离散/连续）
    policy_core = build_policy(model_cfg_path, env, algo_cfg_path)
    n_actions_for_mac = (
        env.n_actions
        if getattr(policy_core, "action_space_type", "discrete") == "discrete"
        else (getattr(policy_core, "action_dim", None) or 0)
    )
    mac = MAC(obs_dim=env.obs_dim, n_actions=n_actions_for_mac, n_agents=env.num_agents)
    mac.policy = policy_core  # 替换为我们构建的 policy

    # 将 Logger 注入 RolloutWorker，使其在每条 episode 完成时记录 train/* 与 env/* 指标
    rollout_worker = RolloutWorker(env=env, policy=mac, logger=tb_logger)
    learner, trainer_kwargs = build_learner(algo_cfg_path, policy=policy_core)

    # Checkpoint 管理：按 train/avg_return 选择最优模型
    # 统一保存到 results/<exp_name>/checkpoints/<seed>/ 下，便于多种子并行
    ckpt_dir = results_dir / "checkpoints" / str(seed)
    ckpt_mgr = CheckpointManager(ckpt_dir, best_metric="train/avg_return", mode="max")

    trainer = Trainer(
        rollout_worker=rollout_worker,
        learner=learner,
        logger=tb_logger,
        checkpoint=ckpt_mgr,
        **trainer_kwargs,
    )

    train_metrics = trainer.run(
        num_epochs=num_epochs,
        rollout_steps=rollout_steps,
        seed=seed,
        log_interval=log_interval,
    )

    evaluator = Evaluator(rollout_worker)
    eval_metrics, _ = evaluator.run(num_episodes=eval_episodes, seed=seed + 10_000)

    # 结束前刷新并关闭 TensorBoard Logger
    tb_logger.flush()
    tb_logger.close()

    print("\n=== Summary ===")
    for k, v in {**train_metrics, **eval_metrics}.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
