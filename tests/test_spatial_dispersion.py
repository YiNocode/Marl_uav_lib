"""空间分散度工具与 SC-MAPPO learner 冒烟测试。"""

from __future__ import annotations

import numpy as np
import torch

from marl_uav.data.batch import EpisodeBatch
from marl_uav.learners.on_policy.sc_mappo_learner import SCMAPPOLearner
from marl_uav.learners.utils.spatial_dispersion import (
    pairwise_mean_cosine_similarity,
    spatial_dispersion_penalty_per_timestep,
)
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy


def test_pairwise_cosine_high_when_parallel():
    T, P, D = 2, 3, 3
    # 三架机在 evader 同侧同一方向：rels 相同 -> to_evader 相同 -> 高相似度
    rel = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
    rels = np.broadcast_to(rel, (T, P, D)).copy()
    p = pairwise_mean_cosine_similarity(rels)
    assert p.shape == (T,)
    assert float(p[0]) > 0.99


def test_pairwise_cosine_lower_when_spread():
    T, P, D = 1, 3, 3
    # 120° 水平均分（evader 在原点，pursuer 在三个方向）
    angles = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    rels = np.zeros((T, P, D), dtype=np.float32)
    for i, a in enumerate(angles):
        rels[0, i] = [np.cos(a), np.sin(a), 0.0]
    p = pairwise_mean_cosine_similarity(rels)
    assert float(p[0]) < 0.5


def test_spatial_dispersion_penalty_from_flat_state():
    # 末尾 9 维为 3*3 rels
    rng = np.random.default_rng(0)
    prefix = rng.normal(size=(4, 33)).astype(np.float32)
    rels = np.array(
        [
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        ]
        * 4,
        dtype=np.float32,
    )
    state = np.concatenate([prefix, rels.reshape(4, -1)], axis=1)
    pen = spatial_dispersion_penalty_per_timestep(state, num_pursuers=3, spatial_dim=3)
    assert pen.shape == (4,)
    assert float(np.mean(pen)) > 0.9


def test_sc_mappo_update_runs():
    torch.manual_seed(0)
    T, N, O, S, A = 5, 3, 8, 42, 7
    policy = CentralizedCriticPolicy(obs_dim=O, state_dim=S, n_actions=A)
    learner = SCMAPPOLearner(
        policy=policy,
        lr=3e-4,
        num_epochs=2,
        dispersion_coef=0.1,
        num_pursuers=3,
    )

    rng = np.random.default_rng(1)
    obs = rng.normal(size=(T, N, O)).astype(np.float32)
    # 末尾 9 维：平行 rels
    tail = np.array([[-1.0, 0.0, 0.0]] * 3, dtype=np.float32).reshape(-1)
    state = np.concatenate([rng.normal(size=(S - 9,)).astype(np.float32), tail])
    state = np.broadcast_to(state, (T, S)).copy()

    actions = rng.integers(low=0, high=A, size=(T, N), dtype=np.int64)
    values = rng.normal(size=(T, N)).astype(np.float32)
    advantages = rng.normal(size=(T, N)).astype(np.float32)
    returns = values + advantages
    log_probs = rng.normal(size=(T, N)).astype(np.float32)
    dones = np.zeros((T,), dtype=np.float32)
    dones[-1] = 1.0

    batch = EpisodeBatch(
        obs=obs,
        state=state,
        actions=actions,
        rewards=rng.normal(size=(T, N)).astype(np.float32),
        dones=dones,
        next_obs=obs,
        next_state=state,
        values=values,
        advantages=advantages,
        returns=returns,
        log_probs=log_probs,
    )
    metrics = learner.update(batch)
    assert "loss/dispersion_penalty" in metrics
    assert np.isfinite(metrics["loss/dispersion_penalty"])
    assert np.isfinite(metrics["loss/policy_loss"])
