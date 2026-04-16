"""Compare RandomPolicy baseline vs IPPO on ToyUavEnv."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.agents.mac import MAC
from marl_uav.data.batch import Batch
from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.learners.on_policy import IPPOLearner
from marl_uav.policies.random_policy import RandomPolicy
from marl_uav.runners.evaluator import Evaluator
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.config import load_config
from marl_uav.utils.rl import compute_gae


def eval_random_policy(env_cfg: dict, num_episodes: int = 20, seed: int = 123) -> float:
    """在 ToyUavEnv 上用 RandomPolicy 跑 num_episodes 集，返回平均回报。"""
    env = ToyUavEnv.from_config(env_cfg, seed=seed)
    policy = RandomPolicy(num_agents=env.num_agents, n_actions=env.n_actions, seed=seed)
    worker = RolloutWorker(env, policy)
    evaluator = Evaluator(worker)
    metrics, _ = evaluator.run(num_episodes=num_episodes, seed=seed)
    return float(metrics["eval/avg_return"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    env_cfg = load_config(root / "configs" / "env" / "toy_uav.yaml")
    num_seeds = 3
    num_epochs = 300
    rollout_steps = 128
    eval_episodes = 10

    # 1. 先跑 RandomPolicy baseline
    print("=== Evaluate RandomPolicy baseline on ToyUavEnv ===")
    rand_returns = [
        eval_random_policy(env_cfg, num_episodes=eval_episodes, seed=1000 + i)
        for i in range(num_seeds)
    ]
    rand_mean = float(np.mean(rand_returns))
    print(f"RandomPolicy baseline avg_return over {num_seeds} seeds: {rand_mean:.3f}")

    # 2. IPPO 训练 + 周期 eval
    device = torch.device("cpu")
    steps = np.arange(1, num_epochs + 1, dtype=np.int32) * rollout_steps
    ippo_eval_returns_all = []  # list[ndarray] per seed

    for seed in range(num_seeds):
        print(f"\n=== IPPO training seed {seed} ===")

        # 训练环境 & MAC
        env_train = ToyUavEnv.from_config(env_cfg, seed=seed)
        num_agents = env_train.num_agents
        obs_dim = env_train.obs_dim
        n_actions = env_train.n_actions

        mac = MAC(
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_agents=num_agents,
            device=device,
            encoder_hidden_dims=(64, 64),
        )

        learner = IPPOLearner(
            policy=mac.policy,
            lr=3e-4,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            num_epochs=2,
        )

        worker_train = RolloutWorker(env=env_train, policy=mac)

        # 单独的 eval 环境（避免干扰训练）
        env_eval = ToyUavEnv.from_config(env_cfg, seed=seed + 10_000)
        worker_eval = RolloutWorker(env=env_eval, policy=mac)
        evaluator = Evaluator(worker_eval)

        gamma = 0.99
        gae_lambda = 0.95

        epoch_eval_returns = []
        env_step_seed = seed

        for epoch in range(num_epochs):
            steps_collected = 0

            # === rollout & update ===
            while steps_collected < rollout_steps:
                buf, info = worker_train.collect_episode(seed=env_step_seed)
                env_step_seed += 1

                episode = buf.get_episode()
                T = int(episode["obs"].shape[0])
                steps_collected += T

                rewards = episode["rewards"]  # (T, N)
                dones = episode["dones"]  # (T,)
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
                # 这里只是演示学习流程，loss_dict 可按需记录 / 打印

                buf.clear()

            # === eval current policy ===
            mac.set_test_mode(True)
            metrics, _ = evaluator.run(num_episodes=eval_episodes, seed=seed + epoch * 17)
            mac.set_test_mode(False)

            eval_ret = float(metrics["eval/avg_return"])
            epoch_eval_returns.append(eval_ret)
            print(
                f"[seed {seed}] epoch={epoch+1}/{num_epochs} "
                f"steps~{(epoch+1)*rollout_steps} eval_return={eval_ret:.3f}"
            )

        ippo_eval_returns_all.append(np.array(epoch_eval_returns, dtype=np.float32))

    # 3. 聚合 IPPO 曲线（对多个 seed 取均值）
    ippo_eval_returns_all = np.stack(ippo_eval_returns_all, axis=0)  # (S, E)
    ippo_mean = ippo_eval_returns_all.mean(axis=0)
    ippo_std = ippo_eval_returns_all.std(axis=0)

    # 4. 画出 eval reward vs steps
    plt.figure(figsize=(6, 4))
    plt.plot(steps, ippo_mean, label="IPPO (mean over seeds)")
    plt.fill_between(
        steps,
        ippo_mean - ippo_std,
        ippo_mean + ippo_std,
        color="C0",
        alpha=0.2,
    )
    plt.axhline(rand_mean, color="C1", linestyle="--", label="RandomPolicy baseline")
    plt.xlabel("Env steps (approx)")
    plt.ylabel("Eval average return")
    plt.title("ToyUavEnv: IPPO vs RandomPolicy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nRandomPolicy baseline avg_return:", rand_mean)
    print("IPPO final avg eval_return:", float(ippo_mean[-1]))
    if ippo_mean[-1] > rand_mean:
        print("Conclusion: IPPO is better than RandomPolicy on ToyUavEnv (final performance).")
    else:
        print("Conclusion: IPPO did not clearly outperform RandomPolicy (tuning may be needed).")


if __name__ == "__main__":
    main()

