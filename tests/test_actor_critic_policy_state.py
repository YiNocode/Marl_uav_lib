"""Tests for ActorCriticPolicy: critic can take obs or state."""

from __future__ import annotations

import torch

from marl_uav.policies.actor_critic_policy import ActorCriticPolicy


def test_actor_critic_policy_critic_falls_back_to_obs_when_no_state():
    torch.manual_seed(0)
    B, N, O = 2, 3, 5
    n_actions = 4
    policy = ActorCriticPolicy(obs_dim=O, n_actions=n_actions)

    obs = torch.randn(B, N, O)
    actions, log_probs, values = policy.act(obs, deterministic=True)

    assert actions.shape == (B, N)
    assert log_probs.shape == (B, N)
    assert values.shape == (B, N)
    assert torch.isfinite(values).all()


def test_actor_critic_policy_critic_uses_state_when_provided():
    torch.manual_seed(1)
    B, N, O, S = 2, 4, 6, 11
    n_actions = 5
    policy = ActorCriticPolicy(obs_dim=O, n_actions=n_actions, state_dim=S)

    obs = torch.randn(B, N, O)
    state = torch.randn(B, S)  # global state
    actions, log_probs, values = policy.act(obs, state=state, deterministic=False)

    assert actions.shape == (B, N)
    assert log_probs.shape == (B, N)
    assert values.shape == (B, N)
    assert torch.isfinite(values).all()

