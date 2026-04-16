"""Integration tests for rollout with state: check batch completeness and alignment."""

from __future__ import annotations

import numpy as np
import torch

from marl_uav.agents.mac import MAC
from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.learners.on_policy.ippo_learner import IPPOLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.runners.trainer import Trainer


def _make_mac_and_worker(num_agents: int = 2):
    env = ToyUavEnv(num_agents=num_agents, episode_limit=5)
    obs_dim = env.obs_dim
    n_actions = env.n_actions

    mac = MAC(obs_dim=obs_dim, n_actions=n_actions, n_agents=num_agents)
    worker = RolloutWorker(env=env, policy=mac)
    return env, mac, worker


def test_rollout_episode_buffer_contains_full_state_and_values_log_probs():
    torch.manual_seed(0)
    np.random.seed(0)

    env, mac, worker = _make_mac_and_worker(num_agents=2)

    buf, info = worker.collect_episode(seed=42)
    episode = buf.get_episode()

    # 基本字段存在
    for key in ("obs", "state", "actions", "rewards", "next_obs", "next_state", "dones"):
        assert key in episode, f"episode missing key: {key}"

    obs = episode["obs"]        # (T, N, O)
    state = episode["state"]    # (T, S)
    rewards = episode["rewards"]
    dones = episode["dones"]

    T = obs.shape[0]
    assert state.shape[0] == T
    assert rewards.shape[0] == T
    assert dones.shape[0] == T

    # 若 policy/MAC 提供了 log_probs/values，则 EpisodeBuffer 应该收集并导出
    assert "log_probs" in episode, "log_probs not collected in episode"
    assert "values" in episode, "values not collected in episode"

    log_probs = episode["log_probs"]
    values = episode["values"]
    assert log_probs.shape[0] == T
    assert values.shape[0] == T
    assert np.isfinite(log_probs).all()
    assert np.isfinite(values).all()


def test_trainer_postprocess_batch_keeps_state_and_values_log_probs():
    torch.manual_seed(1)
    np.random.seed(1)

    env, mac, worker = _make_mac_and_worker(num_agents=3)

    # 用 IPPOLearner + Trainer 跑一条 episode，并检查 batch 完整性
    obs_dim = env.obs_dim
    n_actions = env.n_actions
    policy = ActorCriticPolicy(obs_dim=obs_dim, n_actions=n_actions)
    learner = IPPOLearner(policy=policy)
    trainer = Trainer(rollout_worker=worker, learner=learner)

    buf, info = worker.collect_episode(seed=7)
    episode = buf.get_episode()
    batch = trainer._postprocess_episode(episode)  # type: ignore[attr-defined]

    # state 在 batch 中存在且时间对齐
    assert hasattr(batch, "state")
    state = np.asarray(batch.state)
    obs = np.asarray(batch.obs)
    assert state.shape[0] == obs.shape[0], "state time dim should match obs"

    # advantages / returns / values / log_probs 形状与 rewards 对齐
    rewards = np.asarray(batch.rewards)
    T = rewards.shape[0]

    for name in ("advantages", "returns", "values", "log_probs"):
        if hasattr(batch, name):
            arr = np.asarray(getattr(batch, name))
            assert arr.shape[0] == T, f"{name} time dim should match rewards"

