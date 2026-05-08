"""Vectorized PPO trainer built on top of Gymnasium AsyncVectorEnv."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from marl_uav.buffers.vec_rollout_buffer import VecRolloutBuffer
from marl_uav.learners.base_learner import BaseLearner
from marl_uav.runners.base_runner import BaseRunner

if TYPE_CHECKING:
    from marl_uav.utils.checkpoint import CheckpointManager
    from marl_uav.utils.logger import Logger


class VecEnvTrainer(BaseRunner):
    """Collect batched rollouts with AsyncVectorEnv and update PPO on flattened batches."""

    def __init__(
        self,
        *,
        vec_env_manager: Any,
        policy: Any,
        learner: BaseLearner,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        logger: Logger | None = None,
        checkpoint: CheckpointManager | None = None,
    ) -> None:
        self.vec_env_manager = vec_env_manager
        self.policy = policy
        self.learner = learner
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.logger = logger
        self.checkpoint = checkpoint

    def _select_actions(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        avail_actions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions, log_probs, values = self.policy.select_actions(
            obs,
            state=state,
            avail_actions=avail_actions,
        )

        def _to_numpy(x: Any) -> np.ndarray:
            if hasattr(x, "detach") and hasattr(x, "cpu"):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        return (
            _to_numpy(actions),
            _to_numpy(log_probs).astype(np.float32),
            _to_numpy(values).astype(np.float32),
        )

    def _evaluate_values(self, obs: np.ndarray, state: np.ndarray) -> np.ndarray:
        policy = getattr(self.learner, "policy")
        state_arg = state if getattr(policy, "state_dim", None) is not None else None
        _, _, values = policy.act(
            obs,
            state=state_arg,
            avail_actions=None,
            deterministic=True,
        )
        if hasattr(values, "detach") and hasattr(values, "cpu"):
            return values.detach().cpu().numpy().astype(np.float32)
        return np.asarray(values, dtype=np.float32)

    def _aggregate_timing(self, infos: dict[str, Any], env_timing_totals: dict[str, float]) -> None:
        timing = infos.get("timing")
        timing_mask = infos.get("_timing")
        if not isinstance(timing, dict) or timing_mask is None:
            return
        valid_mask = np.asarray(timing_mask, dtype=np.bool_)
        for key, values in timing.items():
            if key.startswith("_"):
                continue
            value_mask = timing.get(f"_{key}", valid_mask)
            mask = np.asarray(value_mask, dtype=np.bool_)
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 0:
                env_timing_totals[key] = env_timing_totals.get(key, 0.0) + float(arr)
            else:
                env_timing_totals[key] = env_timing_totals.get(key, 0.0) + float(arr[mask].sum())

    def _call_learner(self, batch: Any) -> Dict[str, Any]:
        if hasattr(self.learner, "update"):
            return getattr(self.learner, "update")(batch)
        return self.learner.train(batch)

    def run(
        self,
        *,
        num_epochs: int = 10,
        rollout_steps: int = 256,
        seed: int = 42,
        log_interval: int = 1,
    ) -> Dict[str, Any]:
        obs, state, avail_actions, _ = self.vec_env_manager.reset(seed=seed)
        num_envs, num_agents, obs_dim = obs.shape
        state_dim = int(state.shape[-1])
        action_space = self.vec_env_manager.action_space
        discrete_actions = hasattr(action_space, "nvec")
        action_shape = () if discrete_actions else tuple(action_space.shape[1:])
        avail_action_dim = None if avail_actions is None else int(avail_actions.shape[-1])

        all_episode_returns: list[float] = []
        all_episode_lens: list[int] = []
        total_env_steps = 0
        episode_returns = np.zeros((num_envs,), dtype=np.float32)
        episode_lengths = np.zeros((num_envs,), dtype=np.int32)

        for epoch in range(num_epochs):
            buffer = VecRolloutBuffer(
                num_steps=rollout_steps,
                num_envs=num_envs,
                num_agents=num_agents,
                obs_dim=obs_dim,
                state_dim=state_dim,
                action_shape=action_shape,
                discrete_actions=discrete_actions,
                avail_action_dim=avail_action_dim,
            )

            rollout_time = 0.0
            update_time = 0.0
            env_timing_totals: dict[str, float] = {}
            epoch_returns: list[float] = []
            epoch_lens: list[int] = []

            for step in range(rollout_steps):
                t0 = time.time()
                actions, log_probs, values = self._select_actions(obs, state, avail_actions)
                step_result = self.vec_env_manager.step(actions)
                next_values = self._evaluate_values(step_result.gae_next_obs, step_result.gae_next_state)
                next_values = next_values * (1.0 - step_result.terminated[:, None].astype(np.float32))

                buffer.add(
                    step,
                    obs=obs,
                    state=state,
                    actions=np.asarray(actions),
                    rewards=step_result.rewards,
                    dones=step_result.dones,
                    terminated=step_result.terminated,
                    truncated=step_result.truncated,
                    log_probs=log_probs,
                    values=values,
                    next_values=next_values,
                    avail_actions=avail_actions,
                )
                rollout_time += time.time() - t0
                self._aggregate_timing(step_result.infos, env_timing_totals)

                episode_returns += step_result.rewards.sum(axis=1)
                episode_lengths += 1
                done_envs = np.flatnonzero(step_result.dones)
                for env_idx in done_envs:
                    epoch_returns.append(float(episode_returns[env_idx]))
                    epoch_lens.append(int(episode_lengths[env_idx]))
                    episode_returns[env_idx] = 0.0
                    episode_lengths[env_idx] = 0

                obs = step_result.obs
                state = step_result.state
                avail_actions = step_result.avail_actions

            buffer.compute_returns_and_advantages(gamma=self.gamma, gae_lambda=self.gae_lambda)
            t1 = time.time()
            loss_dict = self._call_learner(buffer.as_batch())
            update_time += time.time() - t1

            total_env_steps += rollout_steps * num_envs
            all_episode_returns.extend(epoch_returns)
            all_episode_lens.extend(epoch_lens)

            if log_interval > 0 and (epoch + 1) % log_interval == 0:
                avg_ret = float(np.mean(epoch_returns)) if epoch_returns else 0.0
                avg_len = float(np.mean(epoch_lens)) if epoch_lens else 0.0
                msg = (
                    f"[vec-train] epoch={epoch+1}/{num_epochs} "
                    f"num_envs={num_envs} steps={rollout_steps * num_envs} "
                    f"avg_return={avg_ret:.3f} avg_len={avg_len:.1f}"
                )
                if loss_dict:
                    msg += " " + " ".join(f"{k}={float(v):.4f}" for k, v in loss_dict.items())
                print(msg)

                if self.logger is not None and loss_dict:
                    ppo_metrics: Dict[str, float] = {}
                    for src, dst in (
                        ("loss/policy_loss", "policy_loss"),
                        ("loss/value_loss", "value_loss"),
                        ("loss/entropy", "entropy"),
                        ("train/approx_kl", "approx_kl"),
                        ("train/clip_fraction", "clip_fraction"),
                        ("train/grad_norm", "grad_norm"),
                    ):
                        if src in loss_dict:
                            ppo_metrics[dst] = float(loss_dict[src])
                    if ppo_metrics:
                        self.logger.log_ppo_metrics(ppo_metrics, step=epoch)

                if self.checkpoint is not None:
                    metrics_for_ckpt: Dict[str, float] = {
                        "train/avg_return": avg_ret,
                        "train/avg_len": avg_len,
                    }
                    metrics_for_ckpt.update({k: float(v) for k, v in loss_dict.items()})
                    self.checkpoint.save(
                        learner=self.learner,
                        epoch=epoch,
                        global_step=total_env_steps,
                        metrics=metrics_for_ckpt,
                    )

                rollout_ms_per_env_step = rollout_time * 1000.0 / max(rollout_steps * num_envs, 1)
                update_ms_per_env_step = update_time * 1000.0 / max(rollout_steps * num_envs, 1)
                timing_msg = ""
                if env_timing_totals:
                    env_ms_per_step = {
                        key: 1000.0 * val / max(rollout_steps * num_envs, 1)
                        for key, val in env_timing_totals.items()
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
                    f"rollout={rollout_time:.2f}s ({rollout_ms_per_env_step:.2f}ms/step) "
                    f"update={update_time:.2f}s ({update_ms_per_env_step:.2f}ms/step) "
                    f"episodes={len(epoch_lens)} env_steps={rollout_steps * num_envs}"
                    f"{timing_msg}"
                )

        return {
            "train/num_epochs": int(num_epochs),
            "train/avg_return": float(np.mean(all_episode_returns)) if all_episode_returns else 0.0,
            "train/avg_len": float(np.mean(all_episode_lens)) if all_episode_lens else 0.0,
            "train/num_episodes": int(len(all_episode_returns)),
        }
