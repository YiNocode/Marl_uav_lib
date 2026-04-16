"""Tests for rl utilities: compute_returns, compute_gae."""

from __future__ import annotations

import numpy as np

from marl_uav.utils.rl import compute_gae, compute_returns


def test_compute_returns_numpy_simple():
    # 三步奖励，最后一步结束
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gamma = 0.9

    ret = compute_returns(rewards, dones, gamma=gamma, last_value=0.0)

    # 手动计算:
    # R2 = 1
    # R1 = 1 + 0.9 * 1 = 1.9
    # R0 = 1 + 0.9 * 1.9 = 2.71
    expected = np.array([2.71, 1.9, 1.0], dtype=np.float32)
    np.testing.assert_allclose(ret, expected, rtol=1e-5, atol=1e-5)


def test_compute_gae_numpy_matches_returns_when_lambda_1():
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.zeros_like(rewards)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gamma = 0.9
    lam = 1.0

    adv, ret = compute_gae(
        rewards,
        values,
        dones,
        gamma=gamma,
        gae_lambda=lam,
        last_value=0.0,
    )

    # lambda=1, values=0 时, GAE 退化为标准 returns
    expected_returns = compute_returns(rewards, dones, gamma=gamma, last_value=0.0)
    np.testing.assert_allclose(ret, expected_returns, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(adv, expected_returns, rtol=1e-5, atol=1e-5)


def test_compute_gae_handles_dones():
    # 在中间时间步 done, 之后的回报不应影响之前的 advantage
    rewards = np.array([1.0, 1.0, 10.0], dtype=np.float32)
    values = np.zeros_like(rewards)
    dones = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    gamma = 0.9
    lam = 0.95

    adv, ret = compute_gae(
        rewards,
        values,
        dones,
        gamma=gamma,
        gae_lambda=lam,
        last_value=0.0,
    )

    # t=1 done, 所以 t=0 的 advantage 不应包含 t>=2 的信息
    # 手动：delta1 = r1 - v1 = 1; A1 = 1
    #       R1 = A1 + v1 = 1
    #       delta0 = r0 + gamma * (1-d0) * v1 - v0 = 1
    #       A0 = delta0 + gamma*lam*(1-d0)*A1 = 1 + 0.9*0.95*1
    expected_A1 = 1.0
    expected_A0 = 1.0 + gamma * lam * 1.0
    np.testing.assert_allclose(adv[1], expected_A1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(adv[0], expected_A0, rtol=1e-5, atol=1e-5)

