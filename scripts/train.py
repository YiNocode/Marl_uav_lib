"""Training entry script (on-policy IPPO / MAPPO via config)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.agents.mac import MAC
from marl_uav.envs.factories import build_env_from_config
from marl_uav.envs.vec_env_manager import VecEnvManager
from marl_uav.learners.on_policy import IPPOLearner, MAPPOLearner, SCMAPPOLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy
from marl_uav.policies.dream_mappo_policy import DreamMappoCentralizedCriticPolicy
from marl_uav.runners.evaluator import Evaluator
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.runners.trainer import Trainer
from marl_uav.runners.vecenv_trainer import VecEnvTrainer
from marl_uav.utils.checkpoint import CheckpointManager
from marl_uav.utils.config import load_config
from marl_uav.utils.device import resolve_train_device
from marl_uav.utils.mp_context import default_vec_env_context
from marl_uav.utils.torch_threading import configure_torch_threads
from marl_uav.utils.env_action_bounds import boxed_action_bounds
from marl_uav.utils.logger import Logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_dream_mappo_3v1.yaml"),
        help="Top-level training config path.",
    )
    return p.parse_args()


def build_env(env_cfg_path: Path, seed: int, task_cfg: dict[str, Any] | None = None):
    return build_env_from_config(env_cfg_path, seed=seed, task_cfg=task_cfg)


def build_policy(
    model_cfg_path: Path,
    env: Any,
    algo_cfg_path: Path,
) -> Any:
    """Build a discrete or continuous policy from model/algo configs."""
    cfg = load_config(model_cfg_path)
    algo_cfg = load_config(algo_cfg_path)
    model_type = cfg.get("type", "mlp")
    action_space = str(algo_cfg.get("action_space", "discrete")).lower()

    if action_space not in ("discrete", "continuous"):
        raise ValueError(
            f"action_space must be 'discrete' or 'continuous', got {algo_cfg.get('action_space')!r}"
        )

    env_action_space = str(
        getattr(env, "_action_space_type", getattr(env, "action_space_type", "")) or ""
    ).lower()
    if env_action_space in ("discrete", "continuous") and env_action_space != action_space:
        raise ValueError(
            "Mismatch between algo.action_space and env.action_space: "
            f"algo={action_space!r}, env={env_action_space!r}."
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
            raise ValueError("dream_mappo_centralized_critic only supports continuous action_space.")
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

    algo_name = algo_cfg.get("algo", "ippo").lower()
    state_dim = None if algo_name == "ippo" else getattr(env, "state_dim", None)
    if action_space == "discrete":
        return ActorCriticPolicy(
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
            state_dim=state_dim,
            action_space_type="discrete",
        )
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


def resolve_train_results_dir(root: Path, train_cfg: dict[str, Any], train_config_arg: str) -> Path:
    """Root directory for TensorBoard logs and checkpoints.

    If ``train_results_dir`` is set in the train YAML, use it (relative to repo root or absolute).
    Otherwise ``results/<stem(train-config)>`` (legacy behaviour).
    """
    override = train_cfg.get("train_results_dir")
    if not override:
        return root / "results" / Path(train_config_arg).stem
    p = Path(str(override))
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def build_learner(algo_cfg_path: Path, policy: Any) -> tuple[Any, dict[str, Any]]:
    cfg = load_config(algo_cfg_path)
    algo_name = cfg.get("algo", "ippo").lower()

    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_ratio = float(cfg.get("clip_ratio", 0.2))
    value_coef = float(cfg.get("value_coef", cfg.get("vf_coef", 0.5)))
    entropy_coef = float(cfg.get("entropy_coef", cfg.get("ent_coef", 0.01)))
    lr = float(cfg.get("lr", 3e-4))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    num_epochs = int(cfg.get("epochs", 4))
    minibatch_size = int(cfg.get("minibatch_size", 0))

    learner_kwargs = dict(
        lr=lr,
        clip_range=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
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
    num_envs = int(train_cfg.get("num_envs", 1))
    log_interval = int(train_cfg.get("log_interval", 1))
    eval_episodes = int(train_cfg.get("eval_episodes", 5))
    vec_env_context = str(train_cfg.get("vec_env_context") or default_vec_env_context())
    vec_env_shared_memory = bool(train_cfg.get("vec_env_shared_memory", True))
    vec_env_copy = bool(train_cfg.get("vec_env_copy", False))
    profile_timing = bool(train_cfg.get("vec_env_profile_timing", False))

    configure_torch_threads(num_envs=num_envs, train_cfg=train_cfg)
    if profile_timing and num_envs > 1:
        os.environ["VEC_ENV_PROFILE_WORKERS"] = "1"
    else:
        os.environ.pop("VEC_ENV_PROFILE_WORKERS", None)

    task_cfg = train_cfg.get("task", {})
    env = build_env(env_cfg_path, seed=seed, task_cfg=task_cfg)

    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

    results_dir = resolve_train_results_dir(root, train_cfg, args.train_config)
    tb_log_dir = results_dir / "tb_" / str(seed)
    tb_logger = Logger(log_dir=tb_log_dir)

    train_device = resolve_train_device(train_cfg)
    print(f"[train] device={train_device} (policy/learner on this device)")

    policy_core = build_policy(model_cfg_path, env, algo_cfg_path)
    policy_core = policy_core.to(train_device)

    n_actions_for_mac = (
        env.n_actions
        if getattr(policy_core, "action_space_type", "discrete") == "discrete"
        else (getattr(policy_core, "action_dim", None) or 0)
    )
    mac = MAC(obs_dim=env.obs_dim, n_actions=n_actions_for_mac, n_agents=env.num_agents)
    mac.policy = policy_core

    learner, trainer_kwargs = build_learner(algo_cfg_path, policy=policy_core)
    ckpt_dir = results_dir / "checkpoints" / str(seed)
    ckpt_mgr = CheckpointManager(ckpt_dir, best_metric="train/avg_return", mode="max")

    rollout_worker = RolloutWorker(env=env, policy=mac, logger=tb_logger)
    vec_env_manager = None
    if num_envs > 1:
        vec_env_manager = VecEnvManager(
            env_cfg_path=env_cfg_path,
            task_cfg=task_cfg,
            num_envs=num_envs,
            seed=seed,
            context=vec_env_context,
            shared_memory=vec_env_shared_memory,
            copy=vec_env_copy,
        )
        trainer = VecEnvTrainer(
            vec_env_manager=vec_env_manager,
            policy=mac,
            learner=learner,
            logger=tb_logger,
            checkpoint=ckpt_mgr,
            **trainer_kwargs,
        )
    else:
        trainer = Trainer(
            rollout_worker=rollout_worker,
            learner=learner,
            logger=tb_logger,
            checkpoint=ckpt_mgr,
            **trainer_kwargs,
        )

    try:
        run_kw: dict[str, Any] = dict(
            num_epochs=num_epochs,
            rollout_steps=rollout_steps,
            seed=seed,
            log_interval=log_interval,
        )
        if num_envs > 1:
            run_kw["profile_timing"] = profile_timing
        train_metrics = trainer.run(**run_kw)
    finally:
        if vec_env_manager is not None:
            vec_env_manager.close()

    evaluator = Evaluator(rollout_worker)
    eval_metrics, _ = evaluator.run(num_episodes=eval_episodes, seed=seed + 10_000)

    tb_logger.flush()
    tb_logger.close()

    print("\n=== Summary ===")
    for k, v in {**train_metrics, **eval_metrics}.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
