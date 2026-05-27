"""Verify MAPPO BC fine-tune invariants: log-prob ratio ~1 and anchor sync."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marl_uav.agents.mac import MAC
from marl_uav.envs.factories import build_env_from_config
from marl_uav.learners.on_policy.mappo_learner import MAPPOLearner
from marl_uav.runners.bc_pretrainer import load_bc_policy_weights, set_policy_log_std
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.runners.trainer import Trainer
from marl_uav.utils.config import load_config
from marl_uav.utils.device import resolve_train_device
from marl_uav.utils.mappo_finetune import attach_bc_anchor_to_learner, create_bc_policy_anchor


def _build_policy(model_cfg_path: Path, env, algo_cfg_path: Path):
    from scripts.train import build_policy

    return build_policy(model_cfg_path, env, algo_cfg_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config",
        default=str(ROOT / "configs/experiment/e1_1_open_space_pyflyt_mappo_bc.yaml"),
    )
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    train_cfg = load_config(ROOT / args.train_config)
    env_cfg_path = ROOT / train_cfg["env"]
    algo_cfg_path = ROOT / train_cfg["algo"]
    model_cfg_path = ROOT / train_cfg["model"]
    task_cfg = dict(train_cfg.get("task") or {})
    seed = int(args.seed)

    env = build_env_from_config(env_cfg_path, seed=seed, task_cfg=task_cfg)
    device = resolve_train_device(train_cfg)
    policy = _build_policy(model_cfg_path, env, algo_cfg_path).to(device)

    bc_cfg = dict(train_cfg.get("bc_warmstart") or {})
    ckpt = (
        ROOT
        / train_cfg.get("train_results_dir", "results")
        / "checkpoints"
        / str(seed)
        / str(bc_cfg.get("checkpoint_name", "bc_pretrained.pt"))
    )
    if not ckpt.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {ckpt}")

    load_bc_policy_weights(policy, ckpt, actor_only=True)
    log_std_after = bc_cfg.get("log_std_after_bc")
    if log_std_after is not None:
        set_policy_log_std(policy, float(log_std_after))

    n_actions = getattr(policy, "action_dim", None) or env.n_actions
    mac = MAC(obs_dim=env.obs_dim, n_actions=int(n_actions), n_agents=env.num_agents)
    mac.policy = policy
    mac.set_test_mode(True)

    learner = MAPPOLearner(policy=policy, lr=1e-5, num_epochs=1, minibatch_size=0)
    attach_bc_anchor_to_learner(
        learner,
        policy=policy,
        bc_ckpt_path=ckpt,
        device=device,
        log_std_after_bc=float(log_std_after) if log_std_after is not None else None,
    )
    assert learner.bc_policy_anchor is not None

    worker = RolloutWorker(env=env, policy=mac, logger=None)
    trainer = Trainer(rollout_worker=worker, learner=learner, gamma=0.99, gae_lambda=0.95)
    buf, info = worker.collect_episode(seed=seed)
    episode = buf.get_episode()
    batch = trainer._postprocess_episode(episode)

    obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
    state = torch.as_tensor(batch.state, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=device)
    old_lp = np.asarray(batch.log_probs, dtype=np.float64)

    with torch.no_grad():
        new_lp_t, _, _ = policy.evaluate_actions(obs=obs, actions=actions, state=state)
        anchor_lp_t, _, _ = learner.bc_policy_anchor.evaluate_actions(  # type: ignore[union-attr]
            obs=obs, actions=actions, state=state
        )
    new_lp = new_lp_t.detach().cpu().numpy()
    anchor_lp = anchor_lp_t.detach().cpu().numpy()

    log_ratio = new_lp - old_lp
    ratio = np.exp(log_ratio)
    anchor_gap = np.abs(new_lp - anchor_lp)

    print(f"[verify] episode_len={info['episode_len']} capture={info.get('capture', False)}")
    print(
        "[verify] PPO ratio (new_logp / rollout_logp): "
        f"mean={ratio.mean():.6f} std={ratio.std():.6f} "
        f"min={ratio.min():.6f} max={ratio.max():.6f}"
    )
    print(
        "[verify] |policy_logp - anchor_logp|: "
        f"mean={anchor_gap.mean():.6f} max={anchor_gap.max():.6f}"
    )

    ok_ratio = bool(np.allclose(ratio, 1.0, rtol=1e-4, atol=1e-4))
    ok_anchor = bool(np.max(anchor_gap) < 1e-5)
    if not ok_ratio:
        print("[verify] FAIL: rollout log_probs disagree with evaluate_actions (PPO ratio != 1).")
    if not ok_anchor:
        print("[verify] FAIL: BC anchor log_probs disagree with live policy.")
    if ok_ratio and ok_anchor:
        print("[verify] PASS: BC policy consistent for MAPPO update.")

    # One frozen-critic update should not move actor weights.
    actor_before = [p.detach().clone() for p in policy.actor_encoder.parameters()]
    learner.apply_finetune_epoch({"freeze_actor_epochs": 10}, epoch=0)
    metrics = learner.update(batch)
    actor_after = [p.detach().clone() for p in policy.actor_encoder.parameters()]
    actor_unchanged = all(torch.allclose(a, b) for a, b in zip(actor_before, actor_after))
    print(f"[verify] freeze_actor update: actor_unchanged={actor_unchanged} metrics={metrics.get('train/freeze_actor')}")
    if not actor_unchanged:
        print("[verify] FAIL: actor moved while freeze_actor_epochs active.")

    env.close()
    if not (ok_ratio and ok_anchor and actor_unchanged):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
