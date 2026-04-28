"""IPPO 正确性检查：4 项指标 (参数更新 / old vs new log_probs / advantages / returns-values 尺度)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.agents.mac import MAC
from marl_uav.data.batch import Batch
from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.learners.on_policy import IPPOLearner
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.config import load_config
from marl_uav.utils.rl import compute_gae


def _actor_first_layer(policy):
    """Actor 第一层：CategoricalPolicyHead.linear."""
    return policy.policy_head.linear.weight, policy.policy_head.linear.bias


def _critic_first_layer(policy):
    """Critic 第一层：LinearValueHead.linear."""
    return policy.value_head.linear.weight, policy.value_head.linear.bias


def _weight_stats(w, b):
    """权重均值与范数 (L2)。"""
    with torch.no_grad():
        mean_w = float(w.mean().item())
        norm_w = float(w.norm().item())
        mean_b = float(b.mean().item()) if b is not None else float("nan")
        norm_b = float(b.norm().item()) if b is not None else float("nan")
    return mean_w, norm_w, mean_b, norm_b


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env_cfg = load_config(root / "configs" / "env" / "toy_uav.yaml")
    env = ToyUavEnv.from_config(env_cfg, seed=42)
    device = torch.device("cpu")

    num_agents = env.num_agents
    obs_dim = env.obs_dim
    n_actions = env.n_actions

    mac = MAC(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_agents=num_agents,
        device=device,
        encoder_hidden_dims=(64, 64),
    )
    policy = mac.policy
    worker = RolloutWorker(env=env, policy=mac)
    learner = IPPOLearner(
        policy=policy,
        lr=3e-4,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=2,
    )

    gamma = 0.99
    gae_lambda = 0.95

    # 收集一条 episode
    buf, info = worker.collect_episode(seed=123)
    episode = buf.get_episode()
    rewards = episode["rewards"]
    dones = episode["dones"]
    values = episode.get("values")
    if values is None:
        values = np.zeros_like(rewards, dtype=np.float32)
    adv, rets = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        last_value=0.0,
    )
    data = dict(episode)
    data["advantages"] = adv
    data["returns"] = rets
    batch = Batch(**data)

    # ---------- 检查 3：advantages 是否合理 ----------
    print("=== Check 3: Advantages ===")
    a = np.asarray(batch.advantages)
    print(f"  advantages.mean() = {float(a.mean()):.6f}")
    print(f"  advantages.std()  = {float(a.std()):.6f}")
    print(f"  advantages.min() = {float(a.min()):.6f}")
    print(f"  advantages.max() = {float(a.max()):.6f}")

    # ---------- 检查 4：returns 与 values 尺度 ----------
    print("\n=== Check 4: Returns vs Values scale ===")
    v = np.asarray(batch.values) if hasattr(batch, "values") else values
    r = np.asarray(batch.returns)
    print(f"  values.mean() = {float(v.mean()):.6f}, values.std() = {float(v.std()):.6f}")
    print(f"  returns.mean() = {float(r.mean()):.6f}, returns.std() = {float(r.std()):.6f}")

    # ---------- 检查 1：update 前后参数 ----------
    print("\n=== Check 1: Params before learner.update() ===")
    aw, ab = _actor_first_layer(policy)
    cw, cb = _critic_first_layer(policy)
    am_w, an_w, am_b, an_b = _weight_stats(aw, ab)
    cm_w, cn_w, cm_b, cn_b = _weight_stats(cw, cb)
    print(f"  actor  first layer: weight mean={am_w:.6f}, norm={an_w:.6f} | bias mean={am_b:.6f}, norm={an_b:.6f}")
    print(f"  critic first layer: weight mean={cm_w:.6f}, norm={cn_w:.6f} | bias mean={cm_b:.6f}, norm={cn_b:.6f}")

    learner.update(batch)

    print("\n=== Check 1: Params after learner.update() ===")
    aw2, ab2 = _actor_first_layer(policy)
    cw2, cb2 = _critic_first_layer(policy)
    am_w2, an_w2, am_b2, an_b2 = _weight_stats(aw2, ab2)
    cm_w2, cn_w2, cm_b2, cn_b2 = _weight_stats(cw2, cb2)
    print(f"  actor  first layer: weight mean={am_w2:.6f}, norm={an_w2:.6f} | bias mean={am_b2:.6f}, norm={an_b2:.6f}")
    print(f"  critic first layer: weight mean={cm_w2:.6f}, norm={cn_w2:.6f} | bias mean={cm_b2:.6f}, norm={cn_b2:.6f}")
    print(f"  Actor  weight changed: {not torch.allclose(aw, aw2)}")
    print(f"  Critic weight changed: {not torch.allclose(cw, cw2)}")

    # ---------- 检查 2：update 后用同一批 obs,actions 重算 log_probs 是否变化 ----------
    print("\n=== Check 2: old_log_probs vs new_log_probs (after update) ===")
    old_lp = np.asarray(batch.log_probs)  # rollout 时存的
    new_lp, _, _ = policy.evaluate_actions(obs=batch.obs, actions=batch.actions)
    new_lp_np = new_lp.detach().cpu().numpy()
    diff = np.abs(new_lp_np - old_lp)
    print(f"  old_log_probs (rollout): mean={float(old_lp.mean()):.6f}, std={float(old_lp.std()):.6f}")
    print(f"  new_log_probs (after update, same obs/actions): mean={float(new_lp_np.mean()):.6f}, std={float(new_lp_np.std()):.6f}")
    print(f"  |new - old|: mean={float(diff.mean()):.6f}, max={float(diff.max()):.6f}")
    print(f"  Different: {not np.allclose(old_lp, new_lp_np)}")

    print("\nIPPO correctness checks done.")


if __name__ == "__main__":
    main()
