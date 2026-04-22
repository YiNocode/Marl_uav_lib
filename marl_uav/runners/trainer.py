"""Trainer: main training loop (on-policy)."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch
from marl_uav.data.batch import Batch, EpisodeBatch
from marl_uav.learners.base_learner import BaseLearner
from marl_uav.runners.base_runner import BaseRunner
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.checkpoint import CheckpointManager
from marl_uav.utils.logger import Logger
from marl_uav.utils.rl import compute_gae, compute_returns


class Trainer(BaseRunner):
    """On-policy trainer: rollout -> postprocess -> learner.update -> log."""

    def __init__(
        self,
        rollout_worker: RolloutWorker,
        learner: BaseLearner,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        logger: Logger | None = None,
        checkpoint: CheckpointManager | None = None,
    ) -> None:
        self.rollout_worker = rollout_worker
        self.learner = learner
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.logger = logger
        self.checkpoint = checkpoint

    def _postprocess_episode(self, episode: dict[str, Any]) -> Batch:
        """从 episode 字典中计算 advantage / returns 并打包成 Batch/EpisodeBatch。

        期望的 episode 字段（来自 EpisodeBuffer.get_episode()）：
            - rewards: (T, n_agents)
            - dones:   (T,)
            - terminated: (T,) 若存在则用于 GAE last_value：terminated=True -> last_value=0
            - truncated: (T,) 若存在则用于 GAE last_value：truncated=True -> last_value=V(next_state_last)
            - values:  (T, n_agents)  (可选，没有则用 0)
            - 以及 obs/state/actions/next_obs/next_state
        """
        rewards = np.asarray(episode["rewards"])  # (T, n_agents) 或 (T,)
        dones = np.asarray(episode["dones"])  # (T,)
        values_arr = episode.get("values")
        if values_arr is None:
            values = np.zeros_like(rewards, dtype=np.float32)
        else:
            values = np.asarray(values_arr)
            # 确保 values 与 rewards 形状对齐，便于 GAE/returns 逐元素计算
            if values.shape != rewards.shape:
                try:
                    values = np.broadcast_to(values, rewards.shape)
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Shape mismatch between rewards {rewards.shape} and values {values.shape} "
                        "and cannot broadcast; check learner/policy value output."
                    ) from exc

        # GAE 的 last_value：terminated=True -> 0.0；truncated=True -> V(next_state_last)
        last_terminated = bool(episode["terminated"][-1]) if "terminated" in episode else bool(dones[-1])
        last_truncated = bool(episode["truncated"][-1]) if "truncated" in episode else False
        if last_terminated:
            last_value = 0.0
        elif last_truncated and hasattr(self.learner, "policy"):
            next_obs_last = np.asarray(episode["next_obs"][-1], dtype=np.float32)  # (N, O)
            next_state_last = np.asarray(episode["next_state"][-1], dtype=np.float32)  # (S,) 或 (N, S)
            if next_obs_last.ndim == 2:
                next_obs_batch = next_obs_last[np.newaxis, ...]  # (1, N, O)
            else:
                next_obs_batch = next_obs_last[np.newaxis, ...]
            if next_state_last.ndim == 1:
                next_state_batch = next_state_last[np.newaxis, :]  # (1, S)，policy 会 expand 到 (1, N, S)
            else:
                next_state_batch = next_state_last[np.newaxis, ...]  # (1, N, S)

            policy = self.learner.policy
            state_dim = getattr(policy, "state_dim", None)

            with torch.no_grad():
                policy.eval()
                # 当 policy.state_dim 为 None（如 IPPO 使用 obs 做 critic）时，不传 state
                state_arg = next_state_batch if state_dim is not None else None
                forward_out = policy.forward(
                    next_obs_batch,
                    state=state_arg,
                    avail_actions=None,
                    deterministic=True,
                )
                policy.train()
                # ActorCriticPolicy: (features, pi_out, values)；CentralizedCriticPolicy: (actor_out, critic_out)
                if isinstance(forward_out, tuple) and len(forward_out) == 2:
                    _critic_out = forward_out[1]
                    v_tensor = _critic_out["values"]
                else:
                    v_tensor = forward_out[2]
            last_value = v_tensor.cpu().numpy().flatten()  # (N,) 与 values 最后一维对齐
        else:
            last_value = 0.0

        # 使用 GAE 计算 advantage / returns
        adv, rets = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            last_value=last_value,
        )

        data = dict(episode)
        data["advantages"] = adv
        data["returns"] = rets
        data.pop("terminated", None)
        data.pop("truncated", None)
        # 使用 EpisodeBatch，使 batch 中包含更完整的信息（state/next_state/avail_actions 等）
        return EpisodeBatch(**data)

    def _call_learner(self, batch: Any) -> Dict[str, Any]:
        """兼容 learner.update 与 learner.train 两种接口。"""
        if hasattr(self.learner, "update"):
            return getattr(self.learner, "update")(batch)  # type: ignore[no-any-return]
        return self.learner.train(batch)

    def run(
        self,
        *,
        num_epochs: int = 10,
        rollout_steps: int = 1024,
        seed: int = 42,
        log_interval: int = 1,
    ) -> Dict[str, Any]:
        """主训练循环。

        每个 epoch:
            - 用 rollout_worker 收集至少 rollout_steps 环境步（可多出一整个 episode）
            - 对每条 episode 做 GAE/returns 后处理
            - 调 learner.update/train(batch)
            - 记录并打印指标
        """
        all_episode_returns: list[float] = []
        all_episode_lens: list[int] = []
        all_loss_vals: list[Dict[str, Any]] = []

        env_step_seed = seed
        total_env_steps = 0

        for epoch in range(num_epochs):
            steps_collected = 0
            epoch_returns: list[float] = []
            epoch_lens: list[int] = []
            epoch_losses: list[Dict[str, Any]] = []
            rollout_time = 0.0
            update_time = 0.0
            env_timing_totals: Dict[str, float] = {}

            # 收集至少 rollout_steps 个时间步
            while steps_collected < rollout_steps:
                t0 = time.time()
                buf, info = self.rollout_worker.collect_episode(seed=env_step_seed)
                t1 = time.time()
                rollout_time += t1 - t0
                env_step_seed += 1

                episode = buf.get_episode()
                T = int(episode["obs"].shape[0])
                steps_collected += T
                env_timing_info = info.get("env_timing_total_s")
                if isinstance(env_timing_info, dict):
                    for k, v in env_timing_info.items():
                        env_timing_totals[k] = env_timing_totals.get(k, 0.0) + float(v)

                epoch_returns.append(float(info["episode_return"]))
                epoch_lens.append(int(info["episode_len"]))

                batch = self._postprocess_episode(episode)
                loss_dict = self._call_learner(batch)
                t2 = time.time()
                update_time += t2 - t1
                epoch_losses.append(loss_dict)

                # on-policy: 一条 episode 用完后即丢弃 / 清空 buffer
                buf.clear()

            all_episode_returns.extend(epoch_returns)
            all_episode_lens.extend(epoch_lens)
            all_loss_vals.extend(epoch_losses)
            total_env_steps += steps_collected

            if log_interval > 0 and (epoch + 1) % log_interval == 0:
                avg_ret = float(np.mean(epoch_returns)) if epoch_returns else 0.0
                avg_len = float(np.mean(epoch_lens)) if epoch_lens else 0.0
                # 聚合 loss 字段的均值
                loss_mean: Dict[str, float] = {}
                for ld in epoch_losses:
                    for k, v in ld.items():
                        loss_mean.setdefault(k, 0.0)
                        loss_mean[k] += float(v)
                if epoch_losses:
                    for k in loss_mean:
                        loss_mean[k] /= len(epoch_losses)

                # 控制台打印
                msg = (
                    f"[train] epoch={epoch+1}/{num_epochs} "
                    f"steps={steps_collected} avg_return={avg_ret:.3f} avg_len={avg_len:.1f}"
                )
                if loss_mean:
                    loss_str = " ".join(f"{k}={v:.4f}" for k, v in loss_mean.items())
                    msg += " " + loss_str
                print(msg)

                # TensorBoard: 记录 PPO 相关指标到 ppo/*（若提供 logger）
                if self.logger is not None and loss_mean:
                    ppo_metrics: Dict[str, float] = {}
                    if "loss/policy_loss" in loss_mean:
                        ppo_metrics["policy_loss"] = float(loss_mean["loss/policy_loss"])
                    if "loss/value_loss" in loss_mean:
                        ppo_metrics["value_loss"] = float(loss_mean["loss/value_loss"])
                    if "loss/entropy" in loss_mean:
                        ppo_metrics["entropy"] = float(loss_mean["loss/entropy"])
                    if "train/approx_kl" in loss_mean:
                        ppo_metrics["approx_kl"] = float(loss_mean["train/approx_kl"])
                    if "train/clip_fraction" in loss_mean:
                        ppo_metrics["clip_fraction"] = float(loss_mean["train/clip_fraction"])
                    if "train/grad_norm" in loss_mean:
                        ppo_metrics["grad_norm"] = float(loss_mean["train/grad_norm"])

                    if ppo_metrics:
                        self.logger.log_ppo_metrics(ppo_metrics, step=epoch)

                # Checkpoint: 保存 latest & best 模型
                if self.checkpoint is not None:
                    metrics_for_ckpt: Dict[str, float] = {
                        "train/avg_return": avg_ret,
                        "train/avg_len": avg_len,
                    }
                    metrics_for_ckpt.update(loss_mean)
                    self.checkpoint.save(
                        learner=self.learner,
                        epoch=epoch,
                        global_step=total_env_steps,
                        metrics=metrics_for_ckpt,
                    )
                t3 = time.time()
                rollout_ms_per_step = 1000.0 * rollout_time / max(steps_collected, 1)
                update_ms_per_step = 1000.0 * update_time / max(steps_collected, 1)
                timing_msg = ""
                if env_timing_totals:
                    env_ms_per_step = {
                        k: 1000.0 * v / max(steps_collected, 1) for k, v in env_timing_totals.items()
                    }
                    timing_msg = (
                        " env_step_ms:"
                        f" total={env_ms_per_step.get('total_s', 0.0):.2f}"
                        f" backend={env_ms_per_step.get('backend_step_s', 0.0):.2f}"
                        f" reward={env_ms_per_step.get('compute_rewards_s', 0.0):.2f}"
                        f" done={env_ms_per_step.get('compute_done_s', 0.0):.2f}"
                        f" obs_state={env_ms_per_step.get('build_obs_state_s', 0.0):.2f}"
                        f" info={env_ms_per_step.get('build_info_s', 0.0):.2f}"
                        f" action={env_ms_per_step.get('action_to_setpoint_s', 0.0):.2f}"
                    )
                print(
                    f"rollout={rollout_time:.2f}s ({rollout_ms_per_step:.2f}ms/step) "
                    f"update={update_time:.2f}s ({update_ms_per_step:.2f}ms/step) "
                    f"log={t3 - t2:.2f}s episodes={len(epoch_lens)} steps={steps_collected}"
                    f"{timing_msg}"
                )


        # 汇总整体指标
        global_metrics: Dict[str, Any] = {
            "train/num_epochs": int(num_epochs),
            "train/avg_return": float(np.mean(all_episode_returns))
            if all_episode_returns
            else 0.0,
            "train/avg_len": float(np.mean(all_episode_lens))
            if all_episode_lens
            else 0.0,
            "train/num_episodes": int(len(all_episode_returns)),
        }
        return global_metrics
