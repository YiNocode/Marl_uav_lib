from __future__ import annotations

"""Compare RandomPolicy baseline vs IPPO on PyFlyt NavigationTask (discrete actions).

记录训练指标、PPO 指标、环境诊断指标，并在结束后绘制三类对比图。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from marl_uav.agents.mac import MAC
from marl_uav.data.batch import Batch
from marl_uav.envs.backends.pyflyt_aviary_backend import PyFlytAviaryBackend
from marl_uav.envs.tasks.navigation_task import NavigationTask
from marl_uav.envs.adapters.pyflyt_aviary_env import PyFlytAviaryEnv
from marl_uav.learners.on_policy import IPPOLearner
from marl_uav.policies.random_policy import RandomPolicy
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.checkpoint import CheckpointManager, load_checkpoint
from marl_uav.utils.logger import Logger
from marl_uav.utils.rl import compute_gae


@dataclass
class EpisodeStats:
    length: int
    total_return: float
    success: bool
    out_of_bounds: bool
    collision: bool
    crash: bool


def collect_episode_eval(
    env: PyFlytAviaryEnv,
    act_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    max_steps: int = 512,
) -> tuple[EpisodeStats, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """用于评估（Random / IPPO）的 rollout，收集统计信息。"""
    obs_state, info = env.reset()
    obs = obs_state["obs"]
    state = obs_state["state"]

    rewards_all: list[np.ndarray] = []
    obs_all: list[np.ndarray] = [obs.copy()]
    actions_all: list[np.ndarray] = []

    total_return = 0.0
    step_count = 0
    done = False
    last_info = info

    while not done and step_count < max_steps:
        actions = act_fn(obs, state)
        next_obs_state, rewards, terminated, truncated, info = env.step(actions)
        next_obs = next_obs_state["obs"]
        next_state = next_obs_state["state"]

        r = np.asarray(rewards, dtype=np.float32)
        rewards_all.append(r)
        actions_all.append(np.asarray(actions, dtype=np.int64).copy())
        obs_all.append(next_obs.copy())

        total_return += float(r.sum())
        step_count += 1
        obs = next_obs
        state = next_state
        last_info = info
        done = bool(terminated or truncated)

    stats = EpisodeStats(
        length=step_count,
        total_return=total_return,
        success=bool(last_info.get("all_reached", False)),
        out_of_bounds=bool(last_info.get("out_of_bounds", False)),
        collision=bool(last_info.get("has_collision", False)),
        crash=bool(last_info.get("crash", False)),
    )
    return stats, obs_all, actions_all, rewards_all


def summarize_returns(
    stats_list: list[EpisodeStats],
    rewards_all_episodes: list[list[np.ndarray]],
) -> None:
    lengths = np.array([s.length for s in stats_list], dtype=np.float32)
    total_returns = np.array([s.total_return for s in stats_list], dtype=np.float32)
    flat_rewards = np.concatenate(
        [np.concatenate(ep, axis=0) for ep in rewards_all_episodes], axis=0
    )
    successes = np.array([s.success for s in stats_list], dtype=np.float32)
    out_of_bounds = np.array([s.out_of_bounds for s in stats_list], dtype=np.float32)
    collisions = np.array([s.collision for s in stats_list], dtype=np.float32)
    crashes = np.array([s.crash for s in stats_list], dtype=np.float32)

    print(f"平均 episode 长度: {lengths.mean():.2f}")
    print(f"平均总回报: {total_returns.mean():.3f}")
    print(f"每步 reward 均值: {flat_rewards.mean():.4f}, 方差: {flat_rewards.var():.4f}")
    print(f"到达目标成功率: {successes.mean():.3f}")
    print(f"出界比例: {out_of_bounds.mean():.3f}")
    print(f"碰撞比例: {collisions.mean():.3f}")
    print(f"坠毁比例: {crashes.mean():.3f}")


def plot_metrics(
    train_hist: dict[str, list[float]],
    ppo_hist: dict[str, list[float]],
    env_hist: dict[str, list[float]],
    save_dir: Path,
) -> None:
    """为训练指标、PPO 指标、环境诊断指标分别绘制对比图。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(next(iter(train_hist.values()))) + 1, dtype=np.float32)

    # 1. 训练指标
    fig1, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    for ax, (key, label) in zip(
        axes,
        [
            ("train/episode_return", "Episode Return"),
            ("train/episode_length", "Episode Length"),
            ("train/success_rate", "Success Rate"),
            ("train/collision_rate", "Collision Rate"),
            ("train/out_of_bounds_rate", "Out-of-Bounds Rate"),
        ],
    ):
        if key in train_hist and train_hist[key]:
            ax.plot(epochs, train_hist[key], color="C0")
            ax.set_ylabel(label)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    fig1.suptitle("Training Metrics (IPPO)")
    fig1.tight_layout()
    fig1.savefig(save_dir / "train_metrics.png", dpi=120)
    plt.close(fig1)

    # 2. PPO 指标
    fig2, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    for ax, (key, label) in zip(
        axes,
        [
            ("loss/policy_loss", "Policy Loss"),
            ("loss/value_loss", "Value Loss"),
            ("loss/entropy", "Entropy"),
            ("train/approx_kl", "Approx KL"),
            ("train/clip_fraction", "Clip Fraction"),
            ("train/grad_norm", "Grad Norm"),
        ],
    ):
        if key in ppo_hist and ppo_hist[key]:
            ax.plot(epochs, ppo_hist[key], color="C1")
            ax.set_ylabel(label)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
    fig2.suptitle("PPO Metrics")
    fig2.tight_layout()
    fig2.savefig(save_dir / "ppo_metrics.png", dpi=120)
    plt.close(fig2)

    # 3. 环境诊断指标
    fig3, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    for ax, (key, label) in zip(
        axes,
        [
            ("env/mean_goal_distance", "Mean Goal Distance"),
            ("env/final_goal_distance", "Final Goal Distance"),
            ("env/reward_progress", "Reward Progress"),
            ("env/reward_time_penalty", "Reward Time Penalty"),
            ("env/reward_collision_penalty", "Reward Collision Penalty"),
            ("env/reward_reach_bonus", "Reward Reach Bonus"),
        ],
    ):
        if key in env_hist and env_hist[key]:
            ax.plot(epochs, env_hist[key], color="C2")
            ax.set_ylabel(label)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
    fig3.suptitle("Environment Diagnostics")
    fig3.tight_layout()
    fig3.savefig(save_dir / "env_diagnostics.png", dpi=120)
    plt.close(fig3)

    print(f"\n图表已保存至: {save_dir.resolve()}")


def main() -> None:
    num_agents = 1
    seed = 0
    num_eval_episodes = 20
    num_epochs = 5000
    rollout_steps = 256
    save_dir = Path(__file__).resolve().parents[1] / "results" / "navigation_ippo_vs_random"

    backend = PyFlytAviaryBackend(
        num_agents=num_agents,
        drone_type="quadx",
        render=False,
        physics_hz=240,
        control_hz=60,
        world_scale=5.0,
        drone_options={},
        seed=seed,
        flight_mode=6,
    )
    task = NavigationTask()
    env_train = PyFlytAviaryEnv(backend=backend, task=task, seed=seed)
    env_eval_rand = PyFlytAviaryEnv(
        backend=PyFlytAviaryBackend(
            num_agents=num_agents,
            drone_type="quadx",
            render=False,
            physics_hz=240,
            control_hz=60,
            world_scale=5.0,
            drone_options={},
            seed=seed + 100,
            flight_mode=6,
        ),
        task=NavigationTask(),
        seed=seed + 100,
    )
    env_eval_ippo = PyFlytAviaryEnv(
        backend=PyFlytAviaryBackend(
            num_agents=num_agents,
            drone_type="quadx",
            render=False,
            physics_hz=240,
            control_hz=60,
            world_scale=5.0,
            drone_options={},
            seed=seed + 200,
            flight_mode=6,
        ),
        task=NavigationTask(),
        seed=seed + 200,
    )

    obs_state, _ = env_train.reset(seed=seed)
    obs = obs_state["obs"]
    obs_dim = int(obs.shape[1])
    n_actions = env_train.n_actions

    # === RandomPolicy baseline ===
    print("=== Evaluate RandomPolicy baseline on NavigationTask ===")
    rand_policy = RandomPolicy(num_agents=num_agents, n_actions=n_actions, seed=seed)

    def rand_act_fn(obs: np.ndarray, state: np.ndarray) -> np.ndarray:
        avail = env_eval_rand.get_avail_actions()
        return rand_policy.select_actions(list(obs), avail_actions=avail)

    rand_stats: list[EpisodeStats] = []
    rand_rewards_all: list[list[np.ndarray]] = []
    for ep in range(num_eval_episodes):
        stats, _, _, rewards_all = collect_episode_eval(
            env_eval_rand, act_fn=rand_act_fn, max_steps=512
        )
        rand_stats.append(stats)
        rand_rewards_all.append(rewards_all)
        print(
            f"[Random] Ep {ep}: len={stats.length}, return={stats.total_return:.3f}, "
            f"success={stats.success}, out_of_bounds={stats.out_of_bounds}, "
            f"collision={stats.collision}, crash={stats.crash}"
        )
    print("\nRandomPolicy 统计：")
    summarize_returns(rand_stats, rand_rewards_all)

    # === IPPO 训练 + 指标记录 ===
    print("\n=== IPPO training on NavigationTask ===")
    mac = MAC(obs_dim=obs_dim, n_actions=n_actions, n_agents=num_agents)
    learner = IPPOLearner(
        policy=mac.policy,
        lr=3e-4,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=4,
    )

    # TensorBoard 日志器：用于测试 Logger/RolloutWorker 的接入是否正常
    tb_log_dir = save_dir / "tb"
    tb_logger = Logger(log_dir=tb_log_dir)

    # 将 Logger 注入 RolloutWorker，使其在每条 episode 完成时记录 train/* 与 env/* 指标
    worker = RolloutWorker(env=env_train, policy=mac, logger=tb_logger)

    # Checkpoint 管理器：保存 latest & best 模型（按 train/episode_return 选择最优）
    ckpt_dir = save_dir / "checkpoints"
    ckpt_mgr = CheckpointManager(ckpt_dir, best_metric="train/episode_return", mode="max")

    gamma = 0.99
    gae_lambda = 0.95

    # 三类指标历史
    train_hist: dict[str, list[float]] = {
        "train/episode_return": [],
        "train/episode_length": [],
        "train/success_rate": [],
        "train/collision_rate": [],
        "train/out_of_bounds_rate": [],
    }
    ppo_hist: dict[str, list[float]] = {
        "loss/policy_loss": [],
        "loss/value_loss": [],
        "loss/entropy": [],
        "train/approx_kl": [],
        "train/clip_fraction": [],
        "train/grad_norm": [],
    }
    env_hist: dict[str, list[float]] = {
        "env/mean_goal_distance": [],
        "env/final_goal_distance": [],
        "env/reward_progress": [],
        "env/reward_time_penalty": [],
        "env/reward_collision_penalty": [],
        "env/reward_reach_bonus": [],
    }

    global_step = 0

    for epoch in range(num_epochs):
        episode_returns: list[float] = []
        episode_lengths: list[float] = []
        success_flags: list[bool] = []
        collision_flags: list[bool] = []
        out_of_bounds_flags: list[bool] = []
        mean_goal_distances: list[float] = []
        final_goal_distances: list[float] = []
        reward_progress_sums: list[float] = []
        reward_time_penalty_sums: list[float] = []
        reward_collision_penalty_sums: list[float] = []
        reward_reach_bonus_sums: list[float] = []

        ppo_policy_losses: list[float] = []
        ppo_value_losses: list[float] = []
        ppo_entropies: list[float] = []
        ppo_approx_kls: list[float] = []
        ppo_clip_fractions: list[float] = []
        ppo_grad_norms: list[float] = []

        steps_collected = 0
        while steps_collected < rollout_steps:
            buf, info = worker.collect_episode(seed=seed + epoch * 100)
            episode_returns.append(info["episode_return"])
            episode_lengths.append(float(info["episode_len"]))
            success_flags.append(info.get("success", False))
            collision_flags.append(info.get("collision", False))
            out_of_bounds_flags.append(info.get("out_of_bounds", False))
            if "env_mean_goal_distance" in info:
                mean_goal_distances.append(info["env_mean_goal_distance"])
            if "env_final_goal_distance" in info:
                final_goal_distances.append(info["env_final_goal_distance"])
            if "env_reward_progress" in info:
                reward_progress_sums.append(info["env_reward_progress"])
            if "env_reward_time_penalty" in info:
                reward_time_penalty_sums.append(info["env_reward_time_penalty"])
            if "env_reward_collision_penalty" in info:
                reward_collision_penalty_sums.append(info["env_reward_collision_penalty"])
            if "env_reward_reach_bonus" in info:
                reward_reach_bonus_sums.append(info["env_reward_reach_bonus"])

            episode = buf.get_episode()
            T = int(episode["obs"].shape[0])
            steps_collected += T

            rewards = episode["rewards"]
            dones = episode["dones"]
            values = episode.get("values")
            if values is None:
                values = np.zeros_like(rewards, dtype=np.float32)

            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                gamma=gamma,
                gae_lambda=gae_lambda,
                last_value=0.0,
            )
            data = dict(episode)
            data["advantages"] = advantages
            data["returns"] = returns
            batch = Batch(**data)
            loss_dict = learner.update(batch)
            # 训练后期减小enrtropy_coef:
            learner.entropy_coef = 0.01 * (1-(0.9*epoch)/4000)

            ppo_policy_losses.append(loss_dict.get("loss/policy_loss", 0.0))
            ppo_value_losses.append(loss_dict.get("loss/value_loss", 0.0))
            ppo_entropies.append(loss_dict.get("loss/entropy", 0.0))
            ppo_approx_kls.append(loss_dict.get("train/approx_kl", 0.0))
            ppo_clip_fractions.append(loss_dict.get("train/clip_fraction", 0.0))
            ppo_grad_norms.append(loss_dict.get("train/grad_norm", 0.0))
            buf.clear()

        # 本 epoch 聚合
        train_hist["train/episode_return"].append(float(np.mean(episode_returns)))
        train_hist["train/episode_length"].append(float(np.mean(episode_lengths)))
        train_hist["train/success_rate"].append(float(np.mean(success_flags)))
        train_hist["train/collision_rate"].append(float(np.mean(collision_flags)))
        train_hist["train/out_of_bounds_rate"].append(float(np.mean(out_of_bounds_flags)))

        ppo_hist["loss/policy_loss"].append(float(np.mean(ppo_policy_losses)))
        ppo_hist["loss/value_loss"].append(float(np.mean(ppo_value_losses)))
        ppo_hist["loss/entropy"].append(float(np.mean(ppo_entropies)))
        ppo_hist["train/approx_kl"].append(float(np.mean(ppo_approx_kls)))
        ppo_hist["train/clip_fraction"].append(float(np.mean(ppo_clip_fractions)))
        ppo_hist["train/grad_norm"].append(float(np.mean(ppo_grad_norms)))

        if mean_goal_distances:
            env_hist["env/mean_goal_distance"].append(float(np.mean(mean_goal_distances)))
        if final_goal_distances:
            env_hist["env/final_goal_distance"].append(float(np.mean(final_goal_distances)))
        if reward_progress_sums:
            env_hist["env/reward_progress"].append(float(np.sum(reward_progress_sums)))
        if reward_time_penalty_sums:
            env_hist["env/reward_time_penalty"].append(float(np.sum(reward_time_penalty_sums)))
        if reward_collision_penalty_sums:
            env_hist["env/reward_collision_penalty"].append(float(np.sum(reward_collision_penalty_sums)))
        if reward_reach_bonus_sums:
            env_hist["env/reward_reach_bonus"].append(float(np.sum(reward_reach_bonus_sums)))

        global_step += steps_collected

        # --- 控制台打印 ---
        print(
            f"[IPPO] epoch {epoch + 1}/{num_epochs} "
            f"return={train_hist['train/episode_return'][-1]:.2f} "
            f"len={train_hist['train/episode_length'][-1]:.1f} "
            f"success={train_hist['train/success_rate'][-1]:.2f}"
        )

        # --- TensorBoard: 记录每个 epoch 的 PPO 指标 ---
        ppo_metrics_epoch = {
            "policy_loss": float(ppo_hist["loss/policy_loss"][-1]),
            "value_loss": float(ppo_hist["loss/value_loss"][-1]),
            "entropy": float(ppo_hist["loss/entropy"][-1]),
            "approx_kl": float(ppo_hist["train/approx_kl"][-1]),
            "clip_fraction": float(ppo_hist["train/clip_fraction"][-1]),
            "grad_norm": float(ppo_hist["train/grad_norm"][-1]),
        }
        tb_logger.log_ppo_metrics(ppo_metrics_epoch, step=epoch)

        # --- Checkpoint: 保存 latest & best learner 状态 ---
        ckpt_metrics = {
            "train/episode_return": train_hist["train/episode_return"][-1],
            "train/episode_length": train_hist["train/episode_length"][-1],
            "loss/policy_loss": ppo_hist["loss/policy_loss"][-1],
            "loss/value_loss": ppo_hist["loss/value_loss"][-1],
            "loss/entropy": ppo_hist["loss/entropy"][-1],
            "train/approx_kl": ppo_hist["train/approx_kl"][-1],
            "train/clip_fraction": ppo_hist["train/clip_fraction"][-1],
            "train/grad_norm": ppo_hist["train/grad_norm"][-1],
        }
        ckpt_mgr.save(
            learner=learner,
            epoch=epoch,
            global_step=global_step,
            metrics=ckpt_metrics,
        )

    # === IPPO 评估 ===
    print("\n=== Evaluate IPPO policy on NavigationTask ===")
    mac.set_test_mode(True)

    def ippo_act_fn(obs: np.ndarray, state: np.ndarray) -> np.ndarray:
        avail = env_eval_ippo.get_avail_actions()
        actions, _, _ = mac.select_actions(
            obs, state=state, avail_actions=avail, deterministic=True
        )
        actions_np = np.asarray(actions)
        if actions_np.ndim > 1 and actions_np.shape[0] == 1:
            actions_np = actions_np[0]
        return actions_np.astype(np.int64)

    ippo_stats: list[EpisodeStats] = []
    ippo_rewards_all: list[list[np.ndarray]] = []
    for ep in range(num_eval_episodes):
        stats, _, _, rewards_all = collect_episode_eval(
            env_eval_ippo, act_fn=ippo_act_fn, max_steps=512
        )
        ippo_stats.append(stats)
        ippo_rewards_all.append(rewards_all)
        print(
            f"[IPPO] Ep {ep}: len={stats.length}, return={stats.total_return:.3f}, "
            f"success={stats.success}, out_of_bounds={stats.out_of_bounds}, "
            f"collision={stats.collision}, crash={stats.crash}"
        )
    print("\nIPPO 统计：")
    summarize_returns(ippo_stats, ippo_rewards_all)

    # 补齐 env_hist 长度（若某 epoch 无 env 诊断则用上一值或 0）
    n_epochs = len(train_hist["train/episode_return"])
    for key in env_hist:
        while len(env_hist[key]) < n_epochs:
            env_hist[key].append(env_hist[key][-1] if env_hist[key] else 0.0)

    plot_metrics(train_hist, ppo_hist, env_hist, save_dir)

    # 结束前刷新并关闭 TensorBoard Logger
    tb_logger.flush()
    tb_logger.close()

    # 测试从 best checkpoint 恢复 learner（可选）
    best_ckpt_path = ckpt_dir / "best.pt"
    if best_ckpt_path.exists():
        _ = load_checkpoint(best_ckpt_path, learner)

    env_train.close()
    env_eval_rand.close()
    env_eval_ippo.close()


if __name__ == "__main__":
    main()
