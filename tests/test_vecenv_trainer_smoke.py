"""Smoke test for vectorized PPO rollout collection."""

from __future__ import annotations

from pathlib import Path

from marl_uav.agents.mac import MAC
from marl_uav.envs.factories import build_env_from_config
from marl_uav.envs.vec_env_manager import VecEnvManager
from marl_uav.learners.on_policy.ippo_learner import IPPOLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy
from marl_uav.runners.vecenv_trainer import VecEnvTrainer


def test_vecenv_trainer_smoke_toy_uav():
    env_cfg_path = Path("configs/env/toy_uav.yaml")
    seed = 11

    probe_env = build_env_from_config(env_cfg_path, seed=seed, task_cfg={})
    probe_env.reset(seed=seed)

    policy = ActorCriticPolicy(
        obs_dim=probe_env.obs_dim,
        n_actions=probe_env.n_actions,
        action_space_type="discrete",
    )
    mac = MAC(
        obs_dim=probe_env.obs_dim,
        n_actions=probe_env.n_actions,
        n_agents=probe_env.num_agents,
    )
    mac.policy = policy
    learner = IPPOLearner(policy=policy, num_epochs=1)

    vec_env_manager = VecEnvManager(
        env_cfg_path=env_cfg_path,
        task_cfg={},
        num_envs=4,
        seed=seed,
        context="spawn",
        shared_memory=False,
        copy=True,
    )

    try:
        trainer = VecEnvTrainer(
            vec_env_manager=vec_env_manager,
            policy=mac,
            learner=learner,
            gamma=0.99,
            gae_lambda=0.95,
        )
        metrics = trainer.run(num_epochs=1, rollout_steps=4, seed=seed, log_interval=0)
    finally:
        vec_env_manager.close()
        probe_env.close()

    assert metrics["train/num_epochs"] == 1
