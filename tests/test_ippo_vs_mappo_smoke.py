"""Smoke test: short IPPO vs MAPPO training on ToyUavEnv.

目的：
  - 确认 IPPO 与 MAPPO 都能在 ToyEnv 上完成若干 epoch 训练而不报错
  - MAPPO 使用 centralized critic 的完整流程能跑通（含 state 对齐 / evaluate_actions 调用）
  - 不对性能做强约束，只要求 metrics 为有限值
"""

from __future__ import annotations

import numpy as np
import torch

from marl_uav.agents.mac import MAC
from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.learners.on_policy.ippo_learner import IPPOLearner
from marl_uav.learners.on_policy.mappo_learner import MAPPOLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy
from marl_uav.runners.evaluator import Evaluator
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.runners.trainer import Trainer


def _make_env(seed: int = 0) -> ToyUavEnv:
    return ToyUavEnv(num_agents=3, episode_limit=10, seed=seed)


def _run_short_training_ippo(seed: int = 0) -> dict:
    env = _make_env(seed)
    obs_dim, n_actions, n_agents = env.obs_dim, env.n_actions, env.num_agents

    policy = ActorCriticPolicy(obs_dim=obs_dim, n_actions=n_actions)
    mac = MAC(obs_dim=obs_dim, n_actions=n_actions, n_agents=n_agents)
    mac.policy = policy

    worker = RolloutWorker(env=env, policy=mac)
    learner = IPPOLearner(policy=policy, lr=3e-4, num_epochs=2)
    trainer = Trainer(rollout_worker=worker, learner=learner, gamma=0.99, gae_lambda=0.95)

    train_metrics = trainer.run(num_epochs=2, rollout_steps=64, seed=seed, log_interval=0)
    evaluator = Evaluator(worker)
    eval_metrics, _ = evaluator.run(num_episodes=3, seed=seed + 123)
    return {**train_metrics, **eval_metrics}


def _run_short_training_mappo(seed: int = 1) -> dict:
    env = _make_env(seed)
    obs_dim, n_actions, n_agents, state_dim = (
        env.obs_dim,
        env.n_actions,
        env.num_agents,
        env.state_dim,
    )

    policy = CentralizedCriticPolicy(
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
    )
    mac = MAC(obs_dim=obs_dim, n_actions=n_actions, n_agents=n_agents)
    mac.policy = policy

    worker = RolloutWorker(env=env, policy=mac)
    learner = MAPPOLearner(policy=policy, lr=3e-4, num_epochs=2)
    trainer = Trainer(rollout_worker=worker, learner=learner, gamma=0.99, gae_lambda=0.95)

    train_metrics = trainer.run(num_epochs=2, rollout_steps=64, seed=seed, log_interval=0)
    evaluator = Evaluator(worker)
    eval_metrics, _ = evaluator.run(num_episodes=3, seed=seed + 123)
    return {**train_metrics, **eval_metrics}


def test_ippo_vs_mappo_smoke():
    torch.manual_seed(0)
    np.random.seed(0)

    ippo_metrics = _run_short_training_ippo(seed=0)
    mappo_metrics = _run_short_training_mappo(seed=1)

    # 两者都能跑完，且关键指标为有限值
    for name, metrics in [("ippo", ippo_metrics), ("mappo", mappo_metrics)]:
        assert "train/avg_return" in metrics
        assert "train/avg_len" in metrics
        assert "eval/avg_return" in metrics
        vals = [
            metrics["train/avg_return"],
            metrics["train/avg_len"],
            metrics["eval/avg_return"],
        ]
        assert all(np.isfinite(v) for v in vals), f"{name} metrics contain non-finite values: {metrics}"

    # 简单输出对比结果（pytest -s 时可见）
    print("\n[smoke] IPPO metrics:", ippo_metrics)
    print("[smoke] MAPPO metrics:", mappo_metrics)

