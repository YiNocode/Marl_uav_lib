"""Tests for CentralizedCriticPolicy (actor obs, critic state)."""

from __future__ import annotations

import torch

from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy


def test_actor_depends_only_on_obs_critic_depends_on_state():
    torch.manual_seed(0)
    B, N, O, S, A = 2, 3, 5, 7, 4
    policy = CentralizedCriticPolicy(obs_dim=O, state_dim=S, n_actions=A)

    obs = torch.randn(B, N, O)
    state1 = torch.randn(B, S)
    state2 = state1 + 10.0  # 强烈扰动，确保 value 变化

    actor_out1, critic_out1 = policy.forward(obs, state1, deterministic=True)
    actor_out2, critic_out2 = policy.forward(obs, state2, deterministic=True)

    # actor 只依赖 obs：logits 必须完全一致（deterministic 不影响 logits）
    torch.testing.assert_close(actor_out1["logits"], actor_out2["logits"])

    # critic 依赖 state：values 应该发生变化
    v1, v2 = critic_out1["values"], critic_out2["values"]
    assert v1.shape == (B, N)
    assert v2.shape == (B, N)
    assert torch.isfinite(v1).all()
    assert torch.isfinite(v2).all()
    assert not torch.allclose(v1, v2), "Changing state should change critic values"


def test_act_outputs_complete_and_shapes():
    torch.manual_seed(1)
    B, N, O, S, A = 3, 4, 6, 11, 5
    policy = CentralizedCriticPolicy(obs_dim=O, state_dim=S, n_actions=A)

    obs = torch.randn(B, N, O)
    state = torch.randn(B, S)
    actions, log_probs, values = policy.act(obs, state=state, deterministic=True)

    assert actions.shape == (B, N)
    assert log_probs.shape == (B, N)
    assert values.shape == (B, N)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()


def test_evaluate_actions_recomputes_log_probs_and_values():
    torch.manual_seed(2)
    B, N, O, S, A = 2, 5, 4, 9, 6
    policy = CentralizedCriticPolicy(obs_dim=O, state_dim=S, n_actions=A)

    obs = torch.randn(B, N, O)
    state1 = torch.randn(B, S)
    state2 = state1 * 0.0 + 3.0

    # 用 deterministic=True 取动作，避免采样噪声影响对比
    actions, old_log_probs, old_values = policy.act(obs, state=state1, deterministic=True)

    new_log_probs1, entropy1, values1 = policy.evaluate_actions(
        obs, actions, state=state1
    )
    new_log_probs2, entropy2, values2 = policy.evaluate_actions(
        obs, actions, state=state2
    )

    assert new_log_probs1.shape == (B, N)
    assert entropy1.shape == (B, N)
    assert values1.shape == (B, N)
    assert torch.isfinite(new_log_probs1).all()
    assert torch.isfinite(entropy1).all()
    assert torch.isfinite(values1).all()

    # 同一 obs/actions/state 下，evaluate_actions 的 log_probs 应与 act 的 log_probs 一致（同一网络）
    torch.testing.assert_close(new_log_probs1, old_log_probs)
    torch.testing.assert_close(values1, old_values)

    # 改变 state：values 应改变；actor 侧 log_probs 只由 obs/actions 决定，应不变
    assert not torch.allclose(values1, values2), "Changing state should change critic values"
    torch.testing.assert_close(new_log_probs1, new_log_probs2)
    torch.testing.assert_close(entropy1, entropy2)


def test_continuous_act_and_evaluate_actions_shapes():
    torch.manual_seed(3)
    B, N, O, S, ADIM = 2, 3, 5, 8, 4
    low = [-1.0] * ADIM
    high = [1.0] * ADIM
    policy = CentralizedCriticPolicy(
        obs_dim=O,
        state_dim=S,
        action_space_type="continuous",
        action_dim=ADIM,
        action_low=low,
        action_high=high,
    )

    obs = torch.randn(B, N, O)
    state = torch.randn(B, S)
    actions, log_probs, values = policy.act(obs, state=state, deterministic=True)

    assert actions.shape == (B, N, ADIM)
    assert log_probs.shape == (B, N)
    assert values.shape == (B, N)

    new_lp, ent, vals = policy.evaluate_actions(obs, actions, state=state)
    assert new_lp.shape == (B, N)
    assert ent.shape == (B, N)
    assert vals.shape == (B, N)
    torch.testing.assert_close(new_lp, log_probs)
    torch.testing.assert_close(vals, values)

