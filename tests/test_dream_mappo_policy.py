"""Tests for Dream-MAPPO policy log-prob consistency."""

from __future__ import annotations

import torch

from marl_uav.policies.dream_mappo_policy import DreamMappoCentralizedCriticPolicy


def test_dream_policy_evaluate_actions_matches_act_log_probs_and_values():
    torch.manual_seed(7)
    B, N, O, S, ADIM = 3, 3, 18, 42, 4
    policy = DreamMappoCentralizedCriticPolicy(
        obs_dim=O,
        state_dim=S,
        action_dim=ADIM,
        action_low=[-0.25, -0.25, -0.01, -0.15],
        action_high=[0.25, 0.25, 0.01, 0.15],
        num_pursuers=N,
        a_max_geom=0.15,
        sigma_p=0.5,
        rho_scale=0.5,
        rho_min=0.05,
        psi_scale=3.14159265,
        a_max_residual=0.08,
    )

    obs = torch.randn(B, N, O)
    state = torch.randn(B, S)

    actions, old_log_probs, old_values = policy.act(obs, state=state, deterministic=True)
    new_log_probs, entropy, values = policy.evaluate_actions(obs, actions, state=state)

    assert actions.shape == (B, N, ADIM)
    assert old_log_probs.shape == (B, N)
    assert old_values.shape == (B, N)
    assert entropy.shape == (B, N)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(old_log_probs).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(values).all()
    torch.testing.assert_close(new_log_probs, old_log_probs)
    torch.testing.assert_close(values, old_values)
