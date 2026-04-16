"""Tests for MAPPOLearner: single update step sanity check."""

from __future__ import annotations

import copy

import numpy as np
import torch

from marl_uav.data.batch import EpisodeBatch
from marl_uav.learners.on_policy.mappo_learner import MAPPOLearner
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy


class DummyCentralizedCritic(CentralizedCriticPolicy):
    """Wrap CentralizedCriticPolicy to记录 evaluate_actions 的 state 调用."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_evaluate_state = None

    def evaluate_actions(self, obs, actions, *, state=None, avail_actions=None, **kwargs):
        self.last_evaluate_state = state
        return super().evaluate_actions(
            obs=obs, actions=actions, state=state, avail_actions=avail_actions, **kwargs
        )


def _make_fake_episode_batch(T: int = 5, N: int = 3, obs_dim: int = 4, state_dim: int = 7, n_actions: int = 5):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(T, N, obs_dim)).astype(np.float32)
    state = rng.normal(size=(T, state_dim)).astype(np.float32)
    actions = rng.integers(low=0, high=n_actions, size=(T, N), dtype=np.int64)
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    dones = np.zeros((T,), dtype=np.float32)
    dones[-1] = 1.0

    # 简单构造 advantages / returns / log_probs / values
    values = rng.normal(size=(T, N)).astype(np.float32)
    advantages = rng.normal(size=(T, N)).astype(np.float32)
    returns = values + advantages
    log_probs = rng.normal(size=(T, N)).astype(np.float32)

    data = {
        "obs": obs,
        "state": state,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "next_obs": obs,   # 对测试无关，随便填
        "next_state": state,
        "values": values,
        "advantages": advantages,
        "returns": returns,
        "log_probs": log_probs,
    }
    return EpisodeBatch(**data)


def test_mappo_learner_single_update_updates_params_and_metrics_finite():
    torch.manual_seed(0)
    np.random.seed(0)

    T, N, O, S, A = 6, 4, 5, 9, 7
    policy = DummyCentralizedCritic(obs_dim=O, state_dim=S, n_actions=A)
    learner = MAPPOLearner(policy=policy, lr=3e-4, num_epochs=2)

    batch = _make_fake_episode_batch(T=T, N=N, obs_dim=O, state_dim=S, n_actions=A)

    # 记录若干参数的拷贝，用于确认更新
    params_before = [p.detach().clone() for p in policy.parameters()]

    metrics = learner.update(batch)

    # metrics 字段完整且为有限值
    for key in ("loss/policy_loss", "loss/value_loss", "loss/entropy"):
        assert key in metrics, f"missing metric {key}"
        assert np.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"

    # 参数应发生变化（至少有一个 tensor 不再相等）
    params_after = [p.detach().clone() for p in policy.parameters()]
    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Expected policy parameters to change after MAPPO update"

    # centralized critic policy 被真正调用：evaluate_actions 收到非 None 的 state
    assert policy.last_evaluate_state is not None, "centralized critic was not called with state"
    st = np.asarray(policy.last_evaluate_state)
    assert st.ndim in (2, 3), f"unexpected state shape from evaluate_actions: {st.shape}"

