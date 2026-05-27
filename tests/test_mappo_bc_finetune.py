"""Tests for BC-specific MAPPO fine-tune (freeze actor, BC KL anchor)."""

from __future__ import annotations

import numpy as np
import torch

from marl_uav.data.batch import EpisodeBatch
from marl_uav.learners.on_policy.mappo_learner import MAPPOLearner
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy
from marl_uav.utils.mappo_finetune import (
    adaptive_bc_kl_coef,
    actor_lr_scale_for_epoch,
    bc_kl_coef_for_epoch,
    create_bc_policy_anchor,
    deterministic_rollout_for_epoch,
    entropy_coef_for_epoch,
    freeze_actor_for_epoch,
    ppo_inner_epochs_for_epoch,
    set_policy_actor_trainable,
)


def _make_continuous_batch(
    T: int = 4,
    N: int = 3,
    obs_dim: int = 6,
    state_dim: int = 10,
    action_dim: int = 4,
) -> EpisodeBatch:
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(T, N, obs_dim)).astype(np.float32)
    state = rng.normal(size=(T, state_dim)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(T, N, action_dim)).astype(np.float32)
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    dones = np.zeros((T,), dtype=np.float32)
    dones[-1] = 1.0
    values = rng.normal(size=(T, N)).astype(np.float32)
    advantages = rng.normal(size=(T, N)).astype(np.float32)
    returns = values + advantages
    log_probs = rng.normal(size=(T, N)).astype(np.float32)
    return EpisodeBatch(
        obs=obs,
        state=state,
        actions=actions,
        rewards=rewards,
        dones=dones,
        next_obs=obs,
        next_state=state,
        values=values,
        advantages=advantages,
        returns=returns,
        log_probs=log_probs,
    )


def _actor_param_ids(policy: CentralizedCriticPolicy) -> set[int]:
    ids: set[int] = set()
    for mod in (policy.actor_encoder, policy.policy_head):
        for p in mod.parameters():
            ids.add(id(p))
    return ids


def _critic_param_ids(policy: CentralizedCriticPolicy) -> set[int]:
    ids: set[int] = set()
    for mod in (policy.critic_encoder, policy.value_head):
        for p in mod.parameters():
            ids.add(id(p))
    return ids


def test_freeze_actor_for_epoch_helpers():
    cfg = {"freeze_actor_epochs": 3, "bc_kl_coef": 0.1, "bc_kl_coef_ramp_epochs": 10, "bc_kl_coef_end": 0.0}
    assert freeze_actor_for_epoch(cfg, 0) is True
    assert freeze_actor_for_epoch(cfg, 2) is True
    assert freeze_actor_for_epoch(cfg, 3) is False
    assert bc_kl_coef_for_epoch(cfg, 0) == 0.1
    assert abs(bc_kl_coef_for_epoch(cfg, 10) - 0.0) < 1e-6


def test_protected_epochs_extends_all_guards():
    cfg = {
        "protected_epochs": 10,
        "freeze_actor_epochs": 3,
        "deterministic_rollout_epochs": 5,
        "entropy_coef_start": 0.0,
        "entropy_coef_end": 0.01,
        "entropy_coef_ramp_epochs": 100,
        "actor_lr_scale_start": 0.1,
        "actor_lr_ramp_epochs": 20,
        "ppo_epochs_during_protection": 1,
    }
    assert freeze_actor_for_epoch(cfg, 9) is True
    assert deterministic_rollout_for_epoch(cfg, 9) is True
    assert entropy_coef_for_epoch(None, cfg, 9, 0.001) == 0.0
    assert actor_lr_scale_for_epoch(cfg, 9) == 0.0
    assert ppo_inner_epochs_for_epoch(cfg, 9, default_epochs=2) == 1
    assert freeze_actor_for_epoch(cfg, 10) is False
    assert actor_lr_scale_for_epoch(cfg, 10) == 0.1


def test_adaptive_bc_kl_coef_boosts_on_regression():
    cfg = {
        "bc_kl_adaptive": True,
        "bc_kl_capture_baseline": "auto",
        "bc_kl_regression_tolerance": 0.08,
        "bc_kl_regression_boost": 2.0,
        "bc_kl_relax_when_stable": 0.5,
        "bc_kl_coef_max": 0.5,
        "bc_kl_coef_min": 0.02,
    }
    base = 0.1
    mid = adaptive_bc_kl_coef(cfg, base_coef=base, rolling_capture=0.29, peak_capture=0.31)
    regressed = adaptive_bc_kl_coef(cfg, base_coef=base, rolling_capture=0.20, peak_capture=0.31)
    at_peak = adaptive_bc_kl_coef(cfg, base_coef=base, rolling_capture=0.31, peak_capture=0.31)
    assert mid == base
    assert regressed == 0.2
    assert at_peak == 0.05


def test_mappo_freeze_actor_only_updates_critic():
    torch.manual_seed(0)
    O, S, A_DIM = 6, 10, 4
    low = [-1.0] * A_DIM
    high = [1.0] * A_DIM
    policy = CentralizedCriticPolicy(
        obs_dim=O,
        state_dim=S,
        action_space_type="continuous",
        action_dim=A_DIM,
        action_low=low,
        action_high=high,
    )
    learner = MAPPOLearner(policy=policy, lr=3e-4, num_epochs=1)
    batch = _make_continuous_batch(obs_dim=O, state_dim=S, action_dim=A_DIM)

    actor_before = [p.detach().clone() for p in policy.actor_encoder.parameters()]
    critic_before = [p.detach().clone() for p in policy.critic_encoder.parameters()]

    learner.apply_finetune_epoch({"freeze_actor_epochs": 10}, epoch=0)
    assert learner._freeze_actor is True
    metrics = learner.update(batch)

    actor_after = [p.detach().clone() for p in policy.actor_encoder.parameters()]
    critic_after = [p.detach().clone() for p in policy.critic_encoder.parameters()]

    assert metrics["train/freeze_actor"] == 1.0
    assert all(torch.allclose(b, a) for b, a in zip(actor_before, actor_after))
    assert any(not torch.allclose(b, a) for b, a in zip(critic_before, critic_after))


def test_mappo_bc_kl_anchor_regularization():
    torch.manual_seed(2)
    O, S, A_DIM = 6, 10, 4
    low = [-1.0] * A_DIM
    high = [1.0] * A_DIM
    policy = CentralizedCriticPolicy(
        obs_dim=O,
        state_dim=S,
        action_space_type="continuous",
        action_dim=A_DIM,
        action_low=low,
        action_high=high,
    )
    anchor = create_bc_policy_anchor(policy)
    learner = MAPPOLearner(policy=policy, lr=3e-4, num_epochs=1)
    learner.set_bc_policy_anchor(anchor)
    learner.apply_finetune_epoch({"freeze_actor_epochs": 0, "bc_kl_coef": 0.05}, epoch=0)

    batch = _make_continuous_batch(obs_dim=O, state_dim=S, action_dim=A_DIM)
    metrics = learner.update(batch)
    assert "loss/bc_kl" in metrics
    assert np.isfinite(metrics["loss/bc_kl"])


def test_set_policy_actor_trainable_toggles_grad():
    policy = CentralizedCriticPolicy(
        obs_dim=4,
        state_dim=6,
        action_space_type="continuous",
        action_dim=3,
        action_low=[-1.0, -1.0, -1.0],
        action_high=[1.0, 1.0, 1.0],
    )
    actor_ids = _actor_param_ids(policy)
    critic_ids = _critic_param_ids(policy)

    set_policy_actor_trainable(policy, False)
    for p in policy.parameters():
        if id(p) in actor_ids:
            assert not p.requires_grad
        elif id(p) in critic_ids:
            assert p.requires_grad

    set_policy_actor_trainable(policy, True)
    for p in policy.parameters():
        assert p.requires_grad
