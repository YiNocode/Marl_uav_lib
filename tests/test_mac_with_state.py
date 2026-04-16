"""Tests for MAC.select_actions with optional state."""

from __future__ import annotations

import torch

from marl_uav.agents.mac import MAC
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy


def test_mac_select_actions_with_state_ippo_compatible():
    """默认 MAC(policy.state_dim=None) 时，即使传 state 也应忽略并正常工作。"""
    torch.manual_seed(0)
    B, N, O, S, A = 2, 3, 5, 7, 4
    mac = MAC(obs_dim=O, n_actions=A, n_agents=N)

    obs = torch.randn(B, N, O)
    state = torch.randn(B, S)  # env 的 global state

    actions, log_probs, values = mac.select_actions(obs, state=state, deterministic=True)
    assert actions.shape == (B, N)
    assert log_probs.shape == (B, N)
    assert values.shape == (B, N)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()


def test_mac_select_actions_without_state_still_works():
    """不传 state 时应保持与现有 IPPO 用法兼容。"""
    torch.manual_seed(1)
    B, N, O, A = 3, 4, 6, 5
    mac = MAC(obs_dim=O, n_actions=A, n_agents=N)

    obs = torch.randn(B, N, O)
    actions, log_probs, values = mac.select_actions(obs, deterministic=True)

    assert actions.shape == (B, N)
    assert log_probs.shape == (B, N)
    assert values.shape == (B, N)


def test_mac_select_actions_with_state_passes_to_policy_when_supported():
    """当 policy 有 state_dim 时，MAC 应透传 state 给 policy.act。"""
    torch.manual_seed(2)
    B, N, O, S, A = 2, 4, 5, 9, 6
    mac = MAC(obs_dim=O, n_actions=A, n_agents=N)

    # 替换成支持 state 的 policy（模拟 MAPPO centralized critic）
    mac.policy = ActorCriticPolicy(obs_dim=O, n_actions=A, state_dim=S)

    obs = torch.randn(B, N, O)
    state = torch.randn(B, S)
    actions, log_probs, values = mac.select_actions(obs, state=state, deterministic=True)

    assert actions.shape == (B, N)
    assert log_probs.shape == (B, N)
    assert values.shape == (B, N)
    assert torch.isfinite(values).all()

