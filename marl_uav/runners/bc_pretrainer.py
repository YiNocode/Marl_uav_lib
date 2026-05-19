"""Roll out geometric experts and behavior-clone the RL policy actor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl_uav.control.expert_policy_factory import make_expert_get_actions_fn
from marl_uav.learners.bc.bc_learner import BCLearner
from marl_uav.utils.logger import Logger


def _resolve_bc_task_cfg(train_cfg: dict[str, Any]) -> dict[str, Any] | None:
    bc_cfg = dict(train_cfg.get("bc_warmstart") or {})
    if bc_cfg.get("task"):
        return dict(bc_cfg["task"])
    if bc_cfg.get("expert_task"):
        return dict(bc_cfg["expert_task"])
    task = train_cfg.get("task")
    return dict(task) if task else None


def collect_expert_transitions(
    env: Any,
    *,
    expert_get_actions,
    num_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (obs, state, expert_action) from expert rollouts."""
    obs_list: list[np.ndarray] = []
    state_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []

    for ep in range(int(num_episodes)):
        ep_seed = int(seed) + ep
        try:
            env.reset(seed=ep_seed)
        except TypeError:
            env.reset()

        while True:
            obs = np.asarray(env.get_obs(), dtype=np.float32)
            state = np.asarray(env.get_state(), dtype=np.float32)
            avail = env.get_avail_actions()
            expert_actions = np.asarray(
                expert_get_actions(obs, state, avail),
                dtype=np.float32,
            ).reshape(env.num_agents, -1)

            obs_list.append(obs)
            state_list.append(state)
            action_list.append(expert_actions)

            _transition, _rewards, terminated, truncated, _info = env.step(expert_actions)
            if bool(terminated) or bool(truncated):
                break

    obs_bt = np.stack(obs_list, axis=0)
    state_bt = np.stack(state_list, axis=0)
    actions_bt = np.stack(action_list, axis=0)
    return obs_bt, state_bt, actions_bt


def run_bc_warmstart(
    *,
    env: Any,
    policy: Any,
    train_cfg: dict[str, Any],
    results_dir: Path,
    seed: int,
    logger: Logger | None = None,
    env_cfg_path: Path | None = None,
    build_env_fn: Any | None = None,
) -> Path | None:
    """Run BC warm-start if enabled in ``train_cfg['bc_warmstart']``.

    Returns the path to the saved BC checkpoint, or ``None`` if skipped/disabled.
    """
    bc_cfg = dict(train_cfg.get("bc_warmstart") or {})
    if not bool(bc_cfg.get("enabled", False)):
        return None

    ckpt_name = str(bc_cfg.get("checkpoint_name", "bc_pretrained.pt"))
    ckpt_path = results_dir / "checkpoints" / str(seed) / ckpt_name
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if bool(bc_cfg.get("skip_if_exists", True)) and ckpt_path.is_file():
        print(f"[bc] skip existing checkpoint: {ckpt_path}")
        return ckpt_path

    expert_name = str(bc_cfg.get("expert", "fixed_ring"))
    expert_params = dict(bc_cfg.get(expert_name) or bc_cfg.get("expert_params") or {})

    task_cfg = _resolve_bc_task_cfg(train_cfg)
    bc_env = env
    if task_cfg is not None and build_env_fn is not None and env_cfg_path is not None:
        if dict(train_cfg.get("task") or {}) != task_cfg:
            print("[bc] building separate env for expert rollouts (bc_warmstart.task override)")
            bc_env = build_env_fn(env_cfg_path, seed=seed + 50_000, task_cfg=task_cfg)
            if getattr(bc_env, "obs_dim", None) is None:
                try:
                    bc_env.reset(seed=seed + 50_000)
                except TypeError:
                    bc_env.reset()

    if getattr(bc_env, "_action_space_type", "") != "continuous":
        raise ValueError("BC warm-start requires continuous action_space.")

    if getattr(policy, "obs_dim", None) != getattr(bc_env, "obs_dim", None):
        raise ValueError(
            "BC env obs_dim must match policy obs_dim: "
            f"policy={getattr(policy, 'obs_dim', None)} env={getattr(bc_env, 'obs_dim', None)}"
        )

    expert_get_actions = make_expert_get_actions_fn(bc_env, expert_name, expert_params)

    num_epochs = int(bc_cfg.get("num_epochs", 50))
    episodes_per_epoch = int(bc_cfg.get("episodes_per_epoch", 32))
    batch_size = int(bc_cfg.get("batch_size", 512))
    log_interval = int(bc_cfg.get("log_interval", 1))
    lr = float(bc_cfg.get("lr", 3e-4))
    max_grad_norm = float(bc_cfg.get("max_grad_norm", 0.5))
    mse_coef = float(bc_cfg.get("mse_coef", 0.0))

    bc_learner = BCLearner(
        policy=policy,
        lr=lr,
        max_grad_norm=max_grad_norm,
        mse_coef=mse_coef,
    )

    print(
        f"[bc] expert={expert_name} epochs={num_epochs} "
        f"episodes_per_epoch={episodes_per_epoch} batch_size={batch_size}"
    )

    rng = np.random.default_rng(seed)
    global_step = 0
    for epoch in range(num_epochs):
        obs_bt, state_bt, actions_bt = collect_expert_transitions(
            bc_env,
            expert_get_actions=expert_get_actions,
            num_episodes=episodes_per_epoch,
            seed=seed + epoch * 10_000,
        )
        total_steps = int(obs_bt.shape[0])
        perm = rng.permutation(total_steps)

        epoch_metrics: dict[str, float] = {}
        num_updates = 0
        for start in range(0, total_steps, batch_size):
            idx = perm[start : start + batch_size]
            metrics = bc_learner.update_batch(
                obs=obs_bt[idx],
                state=state_bt[idx],
                expert_actions=actions_bt[idx],
            )
            for key, val in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + float(val)
            num_updates += 1
            global_step += 1

        if num_updates > 0:
            epoch_metrics = {k: v / num_updates for k, v in epoch_metrics.items()}
        epoch_metrics["bc/num_transitions"] = float(total_steps)

        if logger is not None:
            logger.log_dict(epoch_metrics, step=epoch, prefix="bc")
            logger.flush()

        if log_interval > 0 and (epoch + 1) % log_interval == 0:
            msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(epoch_metrics.items()))
            print(f"[bc] epoch={epoch + 1}/{num_epochs} {msg}")

    save_payload = {
        "epoch": num_epochs - 1,
        "global_step": global_step,
        "expert": expert_name,
        "expert_params": expert_params,
        "policy": policy.state_dict(),
        "bc_learner": bc_learner.state_dict(),
    }
    torch.save(save_payload, ckpt_path)
    print(f"[bc] saved warm-start checkpoint: {ckpt_path}")

    if bc_env is not env:
        try:
            bc_env.close()
        except Exception:
            pass

    return ckpt_path


def load_bc_policy_weights(policy: Any, ckpt_path: Path) -> dict[str, Any]:
    """Load actor weights from a BC checkpoint into ``policy``."""
    data = torch.load(ckpt_path, map_location="cpu")
    policy_state = data.get("policy")
    if policy_state is None:
        raise ValueError(f"BC checkpoint missing 'policy' key: {ckpt_path}")
    policy.load_state_dict(policy_state)
    return data
