"""Compare RandomPolicy baseline vs MAPPO (centralized critic) on ToyUavEnv."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.agents.mac import MAC
from marl_uav.data.batch import EpisodeBatch
from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.learners.on_policy import MAPPOLearner
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy
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

    # 1. RandomPolicy baseline
    print("=== Evaluate RandomPolicy baseline on ToyUavEnv ===")
    rand_returns = [
        eval_random_policy(env_cfg, num_episodes=eval_episodes, seed=2000 + i)
        for i in range(num_seeds)
    ]
    rand_mean = float(np.mean(rand_returns))
    print(f"RandomPolicy baseline avg_return over {num_seeds} seeds: {rand_mean:.3f}")

    # 2. MAPPO training + periodic eval
    device = torch.device("cpu")
    steps = np.arange(1, num_epochs + 1, dtype=np.int32) * rollout_steps
    mappo_eval_returns_all: list[np.ndarray] = []

    for seed in range(num_seeds):
        print(f"\n=== MAPPO training seed {seed} ===")

        # training env & centralized critic policy
        env_train = ToyUavEnv.from_config(env_cfg, seed=seed)
        num_agents = env_train.num_agents
        obs_dim = env_train.obs_dim
        n_actions = env_train.n_actions
        state_dim = env_train.state_dim

        # Centralized critic policy: actor obs, critic state
        policy = CentralizedCriticPolicy(
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
        ).to(device)

        mac = MAC(
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_agents=num_agents,
            device=device,
            encoder_hidden_dims=(64, 64),
        )
        mac.policy = policy

        learner = MAPPOLearner(
            policy=policy,
            lr=3e-4,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            num_epochs=2,
        )

        worker_train = RolloutWorker(env=env_train, policy=mac)

        # separate eval env
        env_eval = ToyUavEnv.from_config(env_cfg, seed=seed + 20_000)
        worker_eval = RolloutWorker(env=env_eval, policy=mac)
        evaluator = Evaluator(worker_eval)

        gamma = 0.99
        gae_lambda = 0.95

        epoch_eval_returns: list[float] = []
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
                batch = EpisodeBatch(**data)

                _ = learner.update(batch)
                buf.clear()

            # === eval current policy ===
            mac.set_test_mode(True)
            metrics, _ = evaluator.run(num_episodes=eval_episodes, seed=seed + epoch * 23)
            mac.set_test_mode(False)

            eval_ret = float(metrics["eval/avg_return"])
            epoch_eval_returns.append(eval_ret)
            print(
                f"[MAPPO seed {seed}] epoch={epoch+1}/{num_epochs} "
                f"steps~{(epoch+1)*rollout_steps} eval_return={eval_ret:.3f}"
            )

        mappo_eval_returns_all.append(np.array(epoch_eval_returns, dtype=np.float32))

    # 3. aggregate MAPPO curves
    mappo_eval_returns_all = np.stack(mappo_eval_returns_all, axis=0)  # (S, E)
    mappo_mean = mappo_eval_returns_all.mean(axis=0)
    mappo_std = mappo_eval_returns_all.std(axis=0)

    # 4. plot eval reward vs steps
    plt.figure(figsize=(6, 4))
    plt.plot(steps, mappo_mean, label="MAPPO (mean over seeds)")
    plt.fill_between(
        steps,
        mappo_mean - mappo_std,
        mappo_mean + mappo_std,
        color="C0",
        alpha=0.2,
    )
    plt.axhline(rand_mean, color="C1", linestyle="--", label="RandomPolicy baseline")
    plt.xlabel("Env steps (approx)")
    plt.ylabel("Eval average return")
    plt.title("ToyUavEnv: MAPPO vs RandomPolicy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nRandomPolicy baseline avg_return:", rand_mean)
    print("MAPPO final avg eval_return:", float(mappo_mean[-1]))
    if mappo_mean[-1] > rand_mean:
        print("Conclusion: MAPPO is better than RandomPolicy on ToyUavEnv (final performance).")
    else:
        print("Conclusion: MAPPO did not clearly outperform RandomPolicy (tuning may be needed).")


if __name__ == "__main__":
    main()

