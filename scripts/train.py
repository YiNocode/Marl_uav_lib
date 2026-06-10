"""Training entry script (on-policy IPPO / MAPPO via config)."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.agents.mac import MAC
from marl_uav.envs.factories import build_env_from_config, env_config_uses_genesis
from marl_uav.envs.genesis_vec_env_manager import GenesisVecEnvManager
from marl_uav.envs.vec_env_manager import VecEnvManager
from marl_uav.learners.on_policy import IPPOLearner, MAPPOLearner, SCMAPPOLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy
from marl_uav.policies.dream_mappo_policy import DreamMappoCentralizedCriticPolicy
from marl_uav.runners.bc_pretrainer import (
    load_bc_policy_weights,
    run_bc_warmstart,
    set_policy_log_std,
)
from marl_uav.control.expert_policy_factory import make_expert_get_actions_fn
from marl_uav.utils.bc_regression import append_train_log_row, compute_regression_metrics, write_metrics_json
from marl_uav.utils.capture_action_guard import CaptureActionGuard
from marl_uav.utils.experiment_pipeline import (
    resolve_capture_protection_cfg,
    should_attach_bc_anchor,
    skip_mappo_training,
)
from marl_uav.utils.mappo_finetune import (
    attach_bc_anchor_to_learner,
    resolve_mappo_finetune_cfg,
    uses_bc_finetune,
)
from marl_uav.runners.evaluator import Evaluator
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.runners.trainer import Trainer
from marl_uav.runners.vecenv_trainer import VecEnvTrainer
from marl_uav.utils.checkpoint import CheckpointManager, load_checkpoint
from marl_uav.utils.config import load_config
from marl_uav.utils.e1_1_suite import merge_rl_task_speed
from marl_uav.utils.device import resolve_train_device
from marl_uav.utils.mp_context import default_vec_env_context
from marl_uav.utils.torch_threading import configure_torch_threads
from marl_uav.utils.env_action_bounds import boxed_action_bounds
from marl_uav.utils.logger import Logger
from marl_uav.utils.stdio import configure_utf8_stdio


configure_utf8_stdio()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_dream_mappo_3v1.yaml"),
        help="Top-level training config path.",
    )
    p.add_argument(
        "--bc-only",
        action="store_true",
        help="Run behavior-cloning warm-start only (no MAPPO/PPO training).",
    )
    p.add_argument(
        "--skip-bc",
        action="store_true",
        help="Skip BC warm-start even if bc_warmstart.enabled is true in the config.",
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


def resolve_optional_checkpoint(root: Path, train_cfg: dict[str, Any]) -> Path | None:
    """Resolve an optional checkpoint used to initialize training."""
    raw = train_cfg.get("initial_checkpoint", train_cfg.get("resume_from_checkpoint"))
    if not raw:
        return None
    p = Path(str(raw))
    return p if p.is_absolute() else (root / p)


def resolve_vec_rollout_steps(
    *,
    rollout_steps: int,
    num_envs: int,
    train_cfg: dict[str, Any],
) -> int:
    """Resolve rollout length passed to VecEnvTrainer.

    Historically VecEnvTrainer interpreted ``rollout_steps`` as per-env steps,
    so ``num_envs=8, rollout_steps=1024`` collected 8192 transitions before one
    PPO update. For benchmark configs that want the old single-env update
    cadence, set ``vec_rollout_steps_mode: total`` and ``rollout_steps`` is
    treated as the target total env steps per update.
    """
    steps = max(int(rollout_steps), 1)
    envs = max(int(num_envs), 1)
    mode = str(train_cfg.get("vec_rollout_steps_mode", "per_env")).strip().lower()
    if envs <= 1 or mode in ("per_env", "per-env", "worker"):
        return steps
    if mode in ("total", "global", "env_steps", "env-steps"):
        return max(1, int(math.ceil(steps / envs)))
    raise ValueError(
        "vec_rollout_steps_mode must be 'per_env' or 'total', "
        f"got {train_cfg.get('vec_rollout_steps_mode')!r}."
    )


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
    target_kl = cfg.get("target_kl")
    target_kl = None if target_kl is None else float(target_kl)

    learner_kwargs = dict(
        lr=lr,
        clip_range=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        target_kl=target_kl,
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
    train_cfg = merge_rl_task_speed(load_config(root / args.train_config))

    env_cfg_path = root / train_cfg.get("env", "configs/env/toy_uav.yaml")
    algo_cfg_path = root / train_cfg.get("algo", "configs/algo/ippo.yaml")
    model_cfg_path = root / train_cfg.get("model", "configs/model/.yaml")

    seed = int(train_cfg.get("seed", 42))
    num_epochs = int(train_cfg.get("num_epochs", 10))
    rollout_steps = int(train_cfg.get("rollout_steps", 1024))
    num_envs = int(train_cfg.get("num_envs", 1))
    trainer_rollout_steps = resolve_vec_rollout_steps(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        train_cfg=train_cfg,
    )
    log_interval = int(train_cfg.get("log_interval", 1))
    eval_episodes = int(train_cfg.get("eval_episodes", 5))
    vec_env_context = str(train_cfg.get("vec_env_context") or default_vec_env_context())
    vec_env_shared_memory = bool(train_cfg.get("vec_env_shared_memory", True))
    vec_env_copy = bool(train_cfg.get("vec_env_copy", False))
    profile_timing = bool(train_cfg.get("vec_env_profile_timing", False))

    use_genesis_native_vec = env_config_uses_genesis(env_cfg_path) and num_envs > 1
    configure_torch_threads(num_envs=1 if use_genesis_native_vec else num_envs, train_cfg=train_cfg)
    if profile_timing and num_envs > 1:
        os.environ["VEC_ENV_PROFILE_WORKERS"] = "1"
    else:
        os.environ.pop("VEC_ENV_PROFILE_WORKERS", None)

    results_dir = resolve_train_results_dir(root, train_cfg, args.train_config)
    tb_log_dir = results_dir / "tb_" / str(seed)
    tb_logger = Logger(log_dir=tb_log_dir)
    tb_logger.log_scalar("run/alive", 1.0, 0)
    tb_logger.flush()

    task_cfg = train_cfg.get("task", {})
    env = build_env(env_cfg_path, seed=seed, task_cfg=task_cfg)

    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

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

    initial_checkpoint = resolve_optional_checkpoint(root, train_cfg)
    if initial_checkpoint is not None:
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(f"initial checkpoint not found: {initial_checkpoint}")
        load_checkpoint(initial_checkpoint, learner)
        print(f"[train] initialized learner from checkpoint: {initial_checkpoint}")

    bc_cfg = dict(train_cfg.get("bc_warmstart") or {})
    bc_enabled = bool(bc_cfg.get("enabled", False)) and not bool(args.skip_bc)
    bc_ckpt_path = None
    if bc_enabled:
        bc_ckpt_path = run_bc_warmstart(
            env=env,
            policy=policy_core,
            train_cfg=train_cfg,
            results_dir=results_dir,
            seed=seed,
            logger=tb_logger,
            env_cfg_path=env_cfg_path,
            build_env_fn=build_env,
        )
        if bc_ckpt_path is not None and bool(bc_cfg.get("load_for_mappo", True)):
            load_bc_policy_weights(policy_core, bc_ckpt_path)
            log_std_after = bc_cfg.get("log_std_after_bc")
            if log_std_after is not None:
                set_policy_log_std(policy_core, float(log_std_after))
            print(f"[train] loaded BC warm-start weights from {bc_ckpt_path}")

    finetune_cfg = resolve_mappo_finetune_cfg(train_cfg)
    default_entropy_coef = float(getattr(learner, "entropy_coef", 0.0))
    if should_attach_bc_anchor(train_cfg) and isinstance(learner, MAPPOLearner):
        log_std_after = bc_cfg.get("log_std_after_bc")
        attach_bc_anchor_to_learner(
            learner,
            policy=policy_core,
            bc_ckpt_path=bc_ckpt_path,
            device=train_device,
            log_std_after_bc=float(log_std_after) if log_std_after is not None else None,
        )
        if hasattr(learner, "apply_finetune_epoch"):
            learner.apply_finetune_epoch(finetune_cfg, epoch=0)

    import shutil
    import yaml

    config_snapshot = results_dir / "config.yaml"
    with open(config_snapshot, "w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False, allow_unicode=True)

    bc_baseline_metrics: dict[str, float] = {}
    metrics_json_path = results_dir / "metrics.json"
    if (results_dir / "bc_eval_summary.csv").is_file():
        import csv

        with open(results_dir / "bc_eval_summary.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows:
                row0 = rows[0]
                bc_baseline_metrics = {
                    "capture_rate": float(row0.get("bc_eval_capture_rate", 0) or 0),
                    "collision_rate": float(row0.get("bc_eval_collision_rate", 0) or 0),
                }
    if bc_ckpt_path is not None and bc_ckpt_path.is_file():
        shutil.copy2(bc_ckpt_path, results_dir / "bc_checkpoint.pt")

    cp_cfg = resolve_capture_protection_cfg(train_cfg)
    write_metrics_json(
        metrics_json_path,
        {
            "capture_protection_mode": str(
                finetune_cfg.get("_capture_protection_mode", cp_cfg.get("mode", "bc_kl_guard"))
            ),
            "capture_protection_enabled": bool(finetune_cfg.get("_capture_protection_enabled", cp_cfg.get("enabled"))),
            "bc_baseline": bc_baseline_metrics,
        },
    )

    if args.bc_only:
        tb_logger.flush()
        tb_logger.close()
        if bc_ckpt_path is None:
            print("[bc-only] nothing to do (bc_warmstart.enabled is false or run was skipped).")
        else:
            print(f"[bc-only] finished. checkpoint={bc_ckpt_path}")
        return

    if skip_mappo_training(train_cfg):
        print("[train] pipeline stage skips MAPPO fine-tuning (bc_only / bc_eval / num_epochs=0).")
        tb_logger.flush()
        tb_logger.close()
        return

    rollout_worker = RolloutWorker(env=env, policy=mac, logger=tb_logger)
    if bc_cfg.get("enabled") and bc_cfg.get("expert"):
        expert_name = str(bc_cfg.get("expert"))
        expert_params = dict(bc_cfg.get(expert_name) or bc_cfg.get("expert_params") or {})
        rollout_worker._expert_get_actions_fn = make_expert_get_actions_fn(
            env, expert_name, expert_params
        )
        mode = str(cp_cfg.get("mode", "bc_kl_guard"))
        if bool(cp_cfg.get("enabled")) and mode in ("action_guard", "mixed_action"):
            rollout_worker._capture_guard = CaptureActionGuard(
                mode=mode,
                enabled=True,
                near_capture_dist=float(cp_cfg.get("near_capture_dist", 2.0)),
                max_action_deviation=float(cp_cfg.get("max_action_deviation", 0.5)),
                protect_if_sce_action_improves_capture=bool(
                    cp_cfg.get("protect_if_sce_action_improves_capture", True)
                ),
                mix_beta=float(cp_cfg.get("mix_beta", 0.5)),
            )
    if hasattr(learner, "bc_policy_anchor") and learner.bc_policy_anchor is not None:
        rollout_worker._bc_policy_for_diag = learner.bc_policy_anchor
    vec_env_manager = None
    if num_envs > 1:
        if use_genesis_native_vec:
            print(
                "[train] creating GenesisVecEnvManager "
                f"(num_envs={num_envs}, native Genesis scene replication, no subprocess workers)"
            )
            # The single env was only needed to infer obs/state dimensions and
            # action bounds. Close its scene before constructing the native
            # Genesis batch to avoid multiple live scenes in one process.
            env.close()
            vec_env_manager = GenesisVecEnvManager(
                env_cfg_path=env_cfg_path,
                task_cfg=task_cfg,
                num_envs=num_envs,
                seed=seed,
            )
        else:
            print(
                "[train] creating VecEnvManager "
                f"(num_envs={num_envs}, context={vec_env_context}, "
                f"shared_memory={vec_env_shared_memory}, copy={vec_env_copy})"
            )
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
        print("[train] VecEnvTrainer ready")
    else:
        trainer = Trainer(
            rollout_worker=rollout_worker,
            learner=learner,
            logger=tb_logger,
            checkpoint=ckpt_mgr,
            **trainer_kwargs,
        )

    try:
        print(
            f"[train] starting trainer.run "
            f"(num_epochs={num_epochs}, rollout_steps={rollout_steps}, "
            f"trainer_rollout_steps={trainer_rollout_steps}, num_envs={num_envs})"
        )
        if num_envs > 1 and trainer_rollout_steps != rollout_steps:
            print(
                "[train] vec rollout_steps_mode=total: "
                f"requested_total_env_steps={rollout_steps}, "
                f"per_env_rollout_steps={trainer_rollout_steps}, "
                f"actual_total_env_steps={trainer_rollout_steps * num_envs}"
            )
        run_kw: dict[str, Any] = dict(
            num_epochs=num_epochs,
            rollout_steps=trainer_rollout_steps,
            seed=seed,
            log_interval=log_interval,
            finetune_cfg=finetune_cfg,
            default_entropy_coef=default_entropy_coef,
        )
        if finetune_cfg:
            print(
                "[train] mappo_finetune: "
                f"deterministic_rollout_epochs={int(finetune_cfg.get('deterministic_rollout_epochs', 0))} "
                f"protected_epochs={int(finetune_cfg.get('protected_epochs', 0))} "
                f"freeze_actor_epochs={int(finetune_cfg.get('freeze_actor_epochs', 0))} "
                f"bc_kl_coef={float(finetune_cfg.get('bc_kl_coef', 0.0) or 0.0)} "
                f"entropy_coef_start={finetune_cfg.get('entropy_coef_start')}"
            )
        if num_envs > 1:
            run_kw["profile_timing"] = profile_timing
        train_metrics = trainer.run(
            **run_kw,
            train_log_path=results_dir / "train_log.csv",
            bc_baseline_metrics=bc_baseline_metrics,
            early_stop_cfg=dict(train_cfg.get("early_stop") or {}),
        )
    finally:
        if vec_env_manager is not None:
            vec_env_manager.close()

    if use_genesis_native_vec:
        env = build_env(env_cfg_path, seed=seed + 10_000, task_cfg=task_cfg)
    eval_worker = RolloutWorker(env=env, policy=mac, logger=None)
    evaluator = Evaluator(eval_worker)
    eval_metrics, _ = evaluator.run(num_episodes=eval_episodes, seed=seed + 10_000)

    tb_logger.flush()
    tb_logger.close()

    print("\n=== Summary ===")
    for k, v in {**train_metrics, **eval_metrics}.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
