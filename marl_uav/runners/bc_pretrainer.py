"""Roll out geometric experts and behavior-clone the RL policy actor."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl_uav.control.expert_policy_factory import make_expert_get_actions_fn
from marl_uav.learners.bc.bc_learner import BCLearner
from marl_uav.utils.config import load_config
from marl_uav.utils.eval_metrics import aggregate_eval_rows, episode_metrics_from_info
from marl_uav.utils.logger import Logger


def _tensor_to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_policy_log_std(policy: Any, log_std: float) -> None:
    """Clamp Gaussian policy exploration after BC (policy_head.log_std)."""
    head = getattr(policy, "policy_head", None) or getattr(policy, "actor_head", None)
    if head is None or not hasattr(head, "log_std"):
        return
    import torch

    with torch.no_grad():
        head.log_std.fill_(float(log_std))


def _action_alignment_metrics(
    policy_actions: np.ndarray, expert_actions: np.ndarray
) -> tuple[float, float]:
    diff = policy_actions.astype(np.float32) - expert_actions.astype(np.float32)
    mse = float(np.mean(diff * diff))
    pa = policy_actions.reshape(-1).astype(np.float64)
    ea = expert_actions.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(pa) * np.linalg.norm(ea))
    cos = float(np.dot(pa, ea) / denom) if denom > 1e-12 else 0.0
    return mse, cos


def evaluate_bc_clone(
    env: Any,
    policy: Any,
    expert_get_actions,
    *,
    num_episodes: int = 20,
    seed: int = 0,
    deterministic: bool = True,
    record_trajectory: bool = False,
    terminal_window: int = 30,
    control_hz: float = 50.0,
) -> dict[str, float]:
    """Evaluate cloned policy vs expert on the same env dynamics."""
    import torch

    policy.eval()
    returns: list[float] = []
    lens: list[int] = []
    action_mse: list[float] = []
    action_cos: list[float] = []
    episode_rows: list[dict[str, Any]] = []

    for ep in range(int(num_episodes)):
        ep_seed = int(seed) + ep
        try:
            env.reset(seed=ep_seed)
        except TypeError:
            env.reset()

        ep_ret = 0.0
        steps = 0
        any_oob = False
        captured = False
        while True:
            obs = np.asarray(env.get_obs(), dtype=np.float32)
            state = np.asarray(env.get_state(), dtype=np.float32)
            avail = env.get_avail_actions()
            expert_actions = np.asarray(
                expert_get_actions(obs, state, avail),
                dtype=np.float32,
            ).reshape(env.num_agents, -1)

            obs_batch = obs[np.newaxis, ...]
            state_batch = state[np.newaxis, ...] if state.ndim == 1 else state
            with torch.no_grad():
                actor_out, _ = policy.forward(  # type: ignore[attr-defined]
                    obs_batch,
                    state_batch,
                    deterministic=deterministic,
                )
            policy_actions = _tensor_to_numpy(actor_out["actions"][0]).astype(np.float32)
            mse, cos = _action_alignment_metrics(policy_actions, expert_actions)
            action_mse.append(mse)
            action_cos.append(cos)

            _transition, rewards, terminated, truncated, info = env.step(policy_actions)
            ep_ret += float(sum(rewards))
            steps += 1
            if info.get("pursuer_oob", False):
                any_oob = True
            if bool(info.get("capture", False) or info.get("captured", False)):
                captured = True
            if bool(terminated) or bool(truncated):
                break

        returns.append(ep_ret)
        lens.append(steps)
        traj = np.asarray(info.get("trajectory", []), dtype=np.float32)
        info_ep = dict(info)
        info_ep.setdefault("capture", captured)
        info_ep.setdefault("captured", captured)
        info_ep.setdefault("pursuer_oob", any_oob)
        info_ep.setdefault("episode_return", ep_ret)
        info_ep.setdefault("episode_len", steps)
        row = episode_metrics_from_info(
            info=info_ep,
            trajectory=traj if record_trajectory and traj.size else None,
            terminal_window=terminal_window,
            control_hz=control_hz,
        )
        row["bc_action_mse"] = float(np.mean(action_mse[-steps:]) if steps else 0.0)
        row["bc_action_cosine_similarity"] = float(np.mean(action_cos[-steps:]) if steps else 0.0)
        episode_rows.append(row)

    policy.train()
    n = max(int(num_episodes), 1)
    agg = aggregate_eval_rows(episode_rows)
    metrics = {
        "bc_eval/capture_rate": float(agg.get("capture_rate", 0.0)),
        "bc_eval/pursuer_oob_episode_rate": float(agg.get("terminal_out_of_bounds_rate", 0.0)),
        "bc_eval/collision_rate": float(agg.get("collision_rate", 0.0)),
        "bc_eval/timeout_rate": float(agg.get("timeout_rate", 0.0)),
        "bc_eval/obstacle_termination_rate": float(agg.get("obstacle_termination_rate", 0.0)),
        "bc_eval/out_of_bounds_rate": float(agg.get("out_of_bounds_rate", 0.0)),
        "bc_eval/other_failure_rate": float(agg.get("other_failure_rate", 0.0)),
        "bc_eval/terminal_capture_rate": float(agg.get("terminal_capture_rate", 0.0)),
        "bc_eval/terminal_obstacle_collision_rate": float(agg.get("terminal_obstacle_collision_rate", 0.0)),
        "bc_eval/terminal_inter_agent_collision_rate": float(agg.get("terminal_inter_agent_collision_rate", 0.0)),
        "bc_eval/terminal_out_of_bounds_rate": float(agg.get("terminal_out_of_bounds_rate", 0.0)),
        "bc_eval/terminal_timeout_rate": float(agg.get("terminal_timeout_rate", 0.0)),
        "bc_eval/terminal_other_failure_rate": float(agg.get("terminal_other_failure_rate", 0.0)),
        "bc_eval/mean_episode_return": float(np.mean(returns)) if returns else 0.0,
        "bc_eval/mean_episode_len": float(np.mean(lens)) if lens else 0.0,
        "bc_eval/mean_action_mse": float(np.mean(action_mse)) if action_mse else 0.0,
        "bc_eval/bc_action_mse": float(np.mean(action_mse)) if action_mse else 0.0,
        "bc_eval/bc_action_cosine_similarity": float(np.mean(action_cos)) if action_cos else 0.0,
    }
    if episode_rows:
        tw = int(terminal_window)
        for key in (
            f"D_ang_last{tw}",
            f"C_cov_last{tw}",
            f"C_col_last{tw}",
            f"max_escape_gap_last{tw}",
        ):
            if key in agg and np.isfinite(agg[key]):
                metrics[f"bc_eval/{key}"] = float(agg[key])
        metrics["_bc_eval_episode_rows"] = episode_rows  # type: ignore[assignment]
    return metrics


def write_bc_eval_summary_csv(path: Path, metrics: dict[str, float], *, seed: int, method: str = "bc") -> None:
    """Write one-row BC evaluation summary for paper tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "bc_train_loss": metrics.get("bc_train_loss", ""),
        "bc_val_loss": metrics.get("bc_val_loss", ""),
        "bc_action_mse": metrics.get("bc_eval/bc_action_mse", metrics.get("bc_eval/mean_action_mse", "")),
        "bc_action_cosine_similarity": metrics.get(
            "bc_eval/bc_action_cosine_similarity", ""
        ),
        "bc_eval_capture_rate": metrics.get("bc_eval/capture_rate", ""),
        "bc_eval_collision_rate": metrics.get("bc_eval/collision_rate", ""),
        "bc_eval_timeout_rate": metrics.get("bc_eval/timeout_rate", ""),
        "bc_eval_obstacle_termination_rate": metrics.get("bc_eval/obstacle_termination_rate", ""),
        "bc_eval_out_of_bounds_rate": metrics.get("bc_eval/out_of_bounds_rate", ""),
        "bc_eval_other_failure_rate": metrics.get("bc_eval/other_failure_rate", ""),
        "bc_eval_terminal_capture_rate": metrics.get("bc_eval/terminal_capture_rate", ""),
        "bc_eval_terminal_obstacle_collision_rate": metrics.get("bc_eval/terminal_obstacle_collision_rate", ""),
        "bc_eval_terminal_inter_agent_collision_rate": metrics.get("bc_eval/terminal_inter_agent_collision_rate", ""),
        "bc_eval_terminal_out_of_bounds_rate": metrics.get("bc_eval/terminal_out_of_bounds_rate", ""),
        "bc_eval_terminal_timeout_rate": metrics.get("bc_eval/terminal_timeout_rate", ""),
        "bc_eval_terminal_other_failure_rate": metrics.get("bc_eval/terminal_other_failure_rate", ""),
    }
    tw = 30
    for suffix in (f"D_ang_last{tw}", f"C_cov_last{tw}", f"C_col_last{tw}", f"max_escape_gap_last{tw}"):
        k = f"bc_eval/{suffix}"
        if k in metrics:
            row[f"bc_eval_{suffix}"] = metrics[k]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _resolve_bc_task_cfg(train_cfg: dict[str, Any]) -> dict[str, Any] | None:
    bc_cfg = dict(train_cfg.get("bc_warmstart") or {})
    task = dict(train_cfg.get("task") or {})
    if bc_cfg.get("task"):
        merged = dict(task)
        merged.update(dict(bc_cfg["task"]))
        return merged
    if bc_cfg.get("expert_task"):
        merged = dict(task)
        merged.update(dict(bc_cfg["expert_task"]))
        return merged
    return task if task else None


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


def collect_policy_labeled_transitions(
    env: Any,
    policy: Any,
    *,
    expert_get_actions,
    num_episodes: int,
    seed: int,
    deterministic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect states visited by ``policy`` and label each with expert actions."""
    import torch

    obs_list: list[np.ndarray] = []
    state_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []

    was_training = bool(getattr(policy, "training", False))
    policy.eval()
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

            obs_batch = obs[np.newaxis, ...]
            state_batch = state[np.newaxis, ...] if state.ndim == 1 else state
            with torch.no_grad():
                actor_out, _ = policy.forward(  # type: ignore[attr-defined]
                    obs_batch,
                    state_batch,
                    deterministic=deterministic,
                )
            policy_actions = _tensor_to_numpy(actor_out["actions"][0]).astype(np.float32)

            _transition, _rewards, terminated, truncated, _info = env.step(policy_actions)
            if bool(terminated) or bool(truncated):
                break

    if was_training:
        policy.train()
    obs_bt = np.stack(obs_list, axis=0)
    state_bt = np.stack(state_list, axis=0)
    actions_bt = np.stack(action_list, axis=0)
    return obs_bt, state_bt, actions_bt


def _fit_bc_arrays(
    *,
    bc_learner: BCLearner,
    obs_bt: np.ndarray,
    state_bt: np.ndarray,
    actions_bt: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    update_epochs: int = 1,
    val_ratio: float = 0.0,
) -> tuple[dict[str, float], float, float]:
    """Fit BC on one labeled dataset and return averaged train/val losses."""
    total_steps = int(obs_bt.shape[0])
    update_epochs = max(int(update_epochs), 1)
    val_ratio = float(np.clip(val_ratio, 0.0, 0.5))
    perm = rng.permutation(total_steps)
    if val_ratio > 0.0 and total_steps > 1:
        n_val = max(1, int(total_steps * val_ratio))
        val_idx = perm[:n_val]
        base_train_idx = perm[n_val:]
    else:
        val_idx = np.array([], dtype=np.int64)
        base_train_idx = perm

    metrics_sum: dict[str, float] = {}
    num_updates = 0
    for _ in range(update_epochs):
        train_idx = rng.permutation(base_train_idx)
        for start in range(0, int(train_idx.size), int(batch_size)):
            idx = train_idx[start : start + int(batch_size)]
            if idx.size == 0:
                continue
            metrics = bc_learner.update_batch(
                obs=obs_bt[idx],
                state=state_bt[idx],
                expert_actions=actions_bt[idx],
            )
            for key, val in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(val)
            num_updates += 1

    metrics_mean = {
        k: v / max(num_updates, 1)
        for k, v in metrics_sum.items()
    }
    train_loss = float(metrics_mean.get("bc/total_loss", 0.0))
    if val_idx.size > 0:
        vm = bc_learner.eval_batch(
            obs=obs_bt[val_idx],
            state=state_bt[val_idx],
            expert_actions=actions_bt[val_idx],
        )
        val_loss = float(vm.get("bc/total_loss", 0.0))
    else:
        val_loss = train_loss
    return metrics_mean, train_loss, val_loss


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

    skipped_training = bool(bc_cfg.get("skip_if_exists", True)) and ckpt_path.is_file()
    if skipped_training:
        print(f"[bc] skip existing checkpoint: {ckpt_path}")
        load_bc_policy_weights(policy, ckpt_path)
        log_std_after = bc_cfg.get("log_std_after_bc")
        if log_std_after is not None:
            set_policy_log_std(policy, float(log_std_after))
        expert_name = str(bc_cfg.get("expert", "fixed_ring"))
        expert_params = dict(bc_cfg.get(expert_name) or bc_cfg.get("expert_params") or {})
        expert_get_actions = make_expert_get_actions_fn(env, expert_name, expert_params)
        eval_metrics = evaluate_bc_clone(
            env,
            policy,
            expert_get_actions,
            num_episodes=int(bc_cfg.get("eval_episodes", 10)),
            seed=seed + 90_000,
        )
        numeric_eval = {
            k: v for k, v in eval_metrics.items()
            if not k.startswith("_") and isinstance(v, (int, float))
        }
        print("[bc] cached checkpoint eval: " + " ".join(f"{k}={v:.4f}" for k, v in numeric_eval.items()))
        if logger is not None:
            logger.log_dict(numeric_eval, step=0, prefix="bc")
            logger.flush()
        min_capture = bc_cfg.get("min_eval_capture_rate")
        if min_capture is not None and eval_metrics["bc_eval/capture_rate"] < float(min_capture):
            print(
                f"[bc] WARNING: cached BC capture_rate={eval_metrics['bc_eval/capture_rate']:.3f} "
                f"< min_eval_capture_rate={float(min_capture):.3f}. "
                "Delete the checkpoint and re-run with --overwrite-configs or set skip_if_exists: false."
            )
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
    nll_coef = float(bc_cfg.get("nll_coef", 1.0))
    loss_mode = str(bc_cfg.get("loss_mode", "nll_mse"))
    update_epochs_per_batch = int(bc_cfg.get("update_epochs_per_batch", 1))
    dagger_iterations = int(bc_cfg.get("dagger_iterations", 0))
    dagger_episodes_per_iter = int(bc_cfg.get("dagger_episodes_per_iter", max(4, episodes_per_epoch // 2)))
    dagger_update_epochs = int(bc_cfg.get("dagger_update_epochs", update_epochs_per_batch))

    bc_learner = BCLearner(
        policy=policy,
        lr=lr,
        max_grad_norm=max_grad_norm,
        mse_coef=mse_coef,
        nll_coef=nll_coef,
        loss_mode=loss_mode,
    )

    print(
        f"[bc] expert={expert_name} epochs={num_epochs} "
        f"episodes_per_epoch={episodes_per_epoch} batch_size={batch_size}"
    )

    val_ratio = float(bc_cfg.get("val_holdout_ratio", 0.0) or 0.0)
    val_ratio = float(np.clip(val_ratio, 0.0, 0.5))
    rng = np.random.default_rng(seed)
    global_step = 0
    last_train_loss = 0.0
    last_val_loss = 0.0
    for epoch in range(num_epochs):
        obs_bt, state_bt, actions_bt = collect_expert_transitions(
            bc_env,
            expert_get_actions=expert_get_actions,
            num_episodes=episodes_per_epoch,
            seed=seed + epoch * 10_000,
        )
        total_steps = int(obs_bt.shape[0])
        epoch_metrics, last_train_loss, last_val_loss = _fit_bc_arrays(
            bc_learner=bc_learner,
            obs_bt=obs_bt,
            state_bt=state_bt,
            actions_bt=actions_bt,
            batch_size=batch_size,
            rng=rng,
            update_epochs=update_epochs_per_batch,
            val_ratio=val_ratio,
        )
        global_step += int(np.ceil(total_steps / max(batch_size, 1))) * max(update_epochs_per_batch, 1)
        epoch_metrics["bc_train_loss"] = last_train_loss
        epoch_metrics["bc_val_loss"] = last_val_loss
        epoch_metrics["bc/num_transitions"] = float(total_steps)

        if logger is not None:
            logger.log_dict(epoch_metrics, step=epoch, prefix="bc")
            logger.flush()

        if log_interval > 0 and (epoch + 1) % log_interval == 0:
            msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(epoch_metrics.items()))
            print(f"[bc] epoch={epoch + 1}/{num_epochs} {msg}")

    for it in range(max(dagger_iterations, 0)):
        obs_bt, state_bt, actions_bt = collect_policy_labeled_transitions(
            bc_env,
            policy,
            expert_get_actions=expert_get_actions,
            num_episodes=dagger_episodes_per_iter,
            seed=seed + 700_000 + it * 10_000,
            deterministic=True,
        )
        dagger_metrics, last_train_loss, last_val_loss = _fit_bc_arrays(
            bc_learner=bc_learner,
            obs_bt=obs_bt,
            state_bt=state_bt,
            actions_bt=actions_bt,
            batch_size=batch_size,
            rng=rng,
            update_epochs=dagger_update_epochs,
            val_ratio=val_ratio,
        )
        dagger_metrics["bc_train_loss"] = last_train_loss
        dagger_metrics["bc_val_loss"] = last_val_loss
        dagger_metrics["bc/num_transitions"] = float(obs_bt.shape[0])
        if logger is not None:
            logger.log_dict(dagger_metrics, step=num_epochs + it, prefix="bc_dagger")
            logger.flush()
        msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(dagger_metrics.items()))
        print(f"[bc-dagger] iter={it + 1}/{dagger_iterations} {msg}")

    log_std_after = bc_cfg.get("log_std_after_bc")
    if log_std_after is not None:
        set_policy_log_std(policy, float(log_std_after))
        print(f"[bc] set policy log_std to {float(log_std_after)}")

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

    eval_episodes = int(bc_cfg.get("eval_episodes", 20))
    exp_cfg = dict(train_cfg.get("experiment") or {})
    eval_seeds = exp_cfg.get("eval_seeds") or [seed + 80_000]
    holdout_seed = int(eval_seeds[0]) if eval_seeds else seed + 80_000
    env_cfg: dict[str, Any] = {}
    if env_cfg_path is not None:
        env_cfg = load_config(env_cfg_path)
    control_hz = float((env_cfg.get("backend", {}) or {}).get("control_hz", 50))

    eval_expert_get_actions = make_expert_get_actions_fn(env, expert_name, expert_params)
    eval_metrics = evaluate_bc_clone(
        env,
        policy,
        eval_expert_get_actions,
        num_episodes=eval_episodes,
        seed=holdout_seed,
        deterministic=True,
        record_trajectory=True,
        terminal_window=int(exp_cfg.get("terminal_window", 30)),
        control_hz=control_hz,
    )
    eval_metrics["bc_train_loss"] = last_train_loss
    eval_metrics["bc_val_loss"] = last_val_loss
    print(
        "[bc] post-train eval: "
        + " ".join(
            f"{k}={v:.4f}"
            for k, v in eval_metrics.items()
            if not k.startswith("_") and isinstance(v, (int, float))
        )
    )
    if logger is not None:
        log_eval = {k: v for k, v in eval_metrics.items() if not k.startswith("_")}
        logger.log_dict(log_eval, step=num_epochs, prefix="bc")
        logger.flush()

    if bool(bc_cfg.get("write_bc_eval_csv", True)):
        csv_path = results_dir / "bc_eval_summary.csv"
        write_bc_eval_summary_csv(csv_path, eval_metrics, seed=seed)
        ep_rows = eval_metrics.pop("_bc_eval_episode_rows", None)
        if ep_rows:
            ep_path = results_dir / "bc_eval_episodes.csv"
            with open(ep_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()))
                writer.writeheader()
                writer.writerows(ep_rows)
        metrics_path = results_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in eval_metrics.items() if not k.startswith("_")},
                f,
                indent=2,
            )

    min_capture = bc_cfg.get("min_eval_capture_rate")
    if min_capture is not None and eval_metrics["bc_eval/capture_rate"] < float(min_capture):
        raise RuntimeError(
            "BC warm-start failed quality gate: "
            f"capture_rate={eval_metrics['bc_eval/capture_rate']:.3f} "
            f"< min_eval_capture_rate={float(min_capture):.3f}. "
            "Try expert=oracle_slot, more bc epochs, or role_features_enabled: false."
        )

    if bc_env is not env:
        try:
            bc_env.close()
        except Exception:
            pass

    return ckpt_path


def _actor_state_prefixes() -> tuple[str, ...]:
    return ("actor_encoder.", "policy_head.", "actor_head.", "dream_actor_head.")


def filter_actor_state_dict(policy_state: dict[str, Any]) -> dict[str, Any]:
    """Keep only actor submodule weights from a full policy state dict."""
    prefixes = _actor_state_prefixes()
    return {k: v for k, v in policy_state.items() if k.startswith(prefixes)}


def load_bc_policy_weights(
    policy: Any,
    ckpt_path: Path,
    *,
    actor_only: bool = True,
) -> dict[str, Any]:
    """Load BC checkpoint into ``policy`` (actor-only by default; critic stays MAPPO-init)."""
    data = torch.load(ckpt_path, map_location="cpu")
    policy_state = data.get("policy")
    if policy_state is None:
        raise ValueError(f"BC checkpoint missing 'policy' key: {ckpt_path}")
    if actor_only:
        actor_state = filter_actor_state_dict(policy_state)
        if not actor_state:
            raise ValueError(f"BC checkpoint has no actor keys: {ckpt_path}")
        incompatible = policy.load_state_dict(actor_state, strict=False)
        unexpected = getattr(incompatible, "unexpected_keys", None)
        if unexpected is None and isinstance(incompatible, (tuple, list)) and len(incompatible) == 2:
            unexpected = incompatible[1]
        if unexpected:
            raise ValueError(f"Unexpected keys when loading BC actor: {list(unexpected)[:5]}")
    else:
        policy.load_state_dict(policy_state)
    return data
