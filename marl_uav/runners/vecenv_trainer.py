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

# 与 RolloutWorker 中 pursuit_structure 尾部均值窗口一致
_PE_STRUCTURE_TAIL_STEPS = 30


def _vec_info_pick(infos: dict[str, Any], key: str, env_idx: int) -> Any:
    """从 AsyncVectorEnv 返回的 infos 中取第 env_idx 个子环境的标量/对象。"""
    if key not in infos:
        return None
    v = infos[key]
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return v.flat[env_idx] if v.size > env_idx else None
        return v[env_idx]
    if isinstance(v, (list, tuple)):
        return v[env_idx] if len(v) > env_idx else None
    return v


def _vec_info_float(infos: dict[str, Any], key: str, env_idx: int, default: float = 0.0) -> float:
    x = _vec_info_pick(infos, key, env_idx)
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _vec_info_bool(infos: dict[str, Any], key: str, env_idx: int) -> bool:
    x = _vec_info_pick(infos, key, env_idx)
    if x is None:
        return False
    return bool(x)


def _cov_col_from_pursuit_structure(
    ps: Any, env_idx: int, num_envs: int
) -> tuple[float, float] | None:
    """Gymnasium 向量化后 ``pursuit_structure`` 可能是单 env 的 dict，也可能是 {C_cov: (N,), C_col: (N,)} 批处理 dict。"""
    if not isinstance(ps, dict) or "C_cov" not in ps or "C_col" not in ps:
        return None
    cc = np.asarray(ps["C_cov"], dtype=np.float64)
    cl = np.asarray(ps["C_col"], dtype=np.float64)
    if cc.size == 0 or cl.size == 0:
        return None
    if cc.size == num_envs and cl.size == num_envs:
        return float(cc[env_idx]), float(cl[env_idx])
    if cc.size == 1 and cl.size == 1:
        return float(cc.flat[0]), float(cl.flat[0])
    try:
        return float(cc.flat[env_idx]), float(cl.flat[env_idx])
    except (IndexError, ValueError, TypeError):
        return float(cc.flat[0]), float(cl.flat[0])


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

    def _reset_tb_trackers(self, num_envs: int) -> None:
        self._ep_any_capture = np.zeros(num_envs, dtype=np.bool_)
        self._ep_any_collision = np.zeros(num_envs, dtype=np.bool_)
        self._ep_any_pursuer_oob = np.zeros(num_envs, dtype=np.bool_)
        self._ep_any_timeout = np.zeros(num_envs, dtype=np.bool_)
        self._ep_any_obstacle_term = np.zeros(num_envs, dtype=np.bool_)
        self._ep_first_capture_step = np.full(num_envs, -1, dtype=np.int32)
        self._ep_mgd_sum = np.zeros((num_envs,), dtype=np.float64)
        self._ep_prog_sum = np.zeros((num_envs,), dtype=np.float64)
        self._ep_time_penalty_sum = np.zeros((num_envs,), dtype=np.float64)
        self._ep_reach_bonus_sum = np.zeros((num_envs,), dtype=np.float64)
        self._ep_collision_penalty_sum = np.zeros((num_envs,), dtype=np.float64)
        self._ep_ps_pairs: list[list[tuple[float, float]]] = [[] for _ in range(num_envs)]
        self._tb_episode_idx = 0

    def _update_tb_trackers(self, infos: dict[str, Any], num_envs: int) -> None:
        for e in range(num_envs):
            if _vec_info_bool(infos, "captured", e):
                self._ep_any_capture[e] = True
            if _vec_info_bool(infos, "has_collision", e):
                self._ep_any_collision[e] = True
            if _vec_info_bool(infos, "pursuer_oob", e):
                self._ep_any_pursuer_oob[e] = True
            if _vec_info_bool(infos, "timeout", e):
                self._ep_any_timeout[e] = True
            if _vec_info_bool(infos, "obstacle_terminated", e):
                self._ep_any_obstacle_term[e] = True
            cs = _vec_info_pick(infos, "capture_step", e)
            if self._ep_first_capture_step[e] < 0 and cs is not None:
                try:
                    csi = int(cs)
                    if csi >= 0:
                        self._ep_first_capture_step[e] = csi
                except (TypeError, ValueError):
                    pass
            self._ep_mgd_sum[e] += _vec_info_float(infos, "mean_goal_distance", e, 0.0)
            self._ep_prog_sum[e] += _vec_info_float(infos, "reward_progress", e, 0.0)
            self._ep_time_penalty_sum[e] += _vec_info_float(infos, "reward_time_penalty", e, 0.0)
            self._ep_reach_bonus_sum[e] += _vec_info_float(infos, "reward_reach_bonus", e, 0.0)
            self._ep_collision_penalty_sum[e] += _vec_info_float(infos, "reward_collision_penalty", e, 0.0)
            ps = _vec_info_pick(infos, "pursuit_structure", e)
            if isinstance(ps, dict) and "C_cov" in ps and "C_col" in ps:
                pair = _cov_col_from_pursuit_structure(ps, e, num_envs)
                if pair is not None:
                    self._ep_ps_pairs[e].append(pair)

    def _clear_tb_trackers_env(self, env_idx: int) -> None:
        self._ep_any_capture[env_idx] = False
        self._ep_any_collision[env_idx] = False
        self._ep_any_pursuer_oob[env_idx] = False
        self._ep_any_timeout[env_idx] = False
        self._ep_any_obstacle_term[env_idx] = False
        self._ep_first_capture_step[env_idx] = -1
        self._ep_mgd_sum[env_idx] = 0.0
        self._ep_prog_sum[env_idx] = 0.0
        self._ep_time_penalty_sum[env_idx] = 0.0
        self._ep_reach_bonus_sum[env_idx] = 0.0
        self._ep_collision_penalty_sum[env_idx] = 0.0
        self._ep_ps_pairs[env_idx].clear()

    def _log_tb_episode(self, env_idx: int, ep_ret: float, ep_len: int, infos: dict[str, Any]) -> None:
        if self.logger is None or ep_len <= 0:
            return
        last_reached = _vec_info_bool(infos, "all_reached", env_idx)
        last_oob = _vec_info_bool(infos, "out_of_bounds", env_idx)
        last_col = _vec_info_bool(infos, "has_collision", env_idx)
        last_cap = _vec_info_bool(infos, "captured", env_idx)
        last_p_oob = _vec_info_bool(infos, "pursuer_oob", env_idx)
        last_to = _vec_info_bool(infos, "timeout", env_idx)
        last_obs_term = _vec_info_bool(infos, "obstacle_terminated", env_idx)
        success = bool(last_reached)
        collision = bool(self._ep_any_collision[env_idx] or last_col)
        capture = bool(self._ep_any_capture[env_idx] or last_cap)
        pursuer_oob = bool(self._ep_any_pursuer_oob[env_idx] or last_p_oob)
        timeout = bool(self._ep_any_timeout[env_idx] or last_to)
        obs_term = bool(self._ep_any_obstacle_term[env_idx] or last_obs_term)
        train_metrics: dict[str, float] = {
            "episode_return": float(ep_ret),
            "episode_length": float(ep_len),
            "success_rate": 1.0 if success else 0.0,
            "out_of_bounds_rate": 1.0 if last_oob else 0.0,
            "collision_rate": 1.0 if collision else 0.0,
            "capture_rate": 1.0 if capture else 0.0,
            "timeout_rate": 1.0 if timeout else 0.0,
            "pursuer_oob_rate": 1.0 if pursuer_oob else 0.0,
            "obstacle_termination_rate": 1.0 if obs_term else 0.0,
        }
        self.logger.log_train_env_metrics(train_metrics, step=self._tb_episode_idx)
        env_metrics: dict[str, float] = {}
        mgd_mean = float(self._ep_mgd_sum[env_idx] / max(ep_len, 1))
        env_metrics["mean_goal_distance"] = mgd_mean
        env_metrics["final_goal_distance"] = _vec_info_float(infos, "mean_goal_distance", env_idx, mgd_mean)
        env_metrics["reward_progress"] = float(self._ep_prog_sum[env_idx])
        env_metrics["reward_time_penalty"] = float(self._ep_time_penalty_sum[env_idx])
        env_metrics["reward_reach_bonus"] = float(self._ep_reach_bonus_sum[env_idx])
        env_metrics["reward_collision_penalty"] = float(self._ep_collision_penalty_sum[env_idx])
        pairs = self._ep_ps_pairs[env_idx]
        if pairs:
            tail = pairs[-_PE_STRUCTURE_TAIL_STEPS:]
            covs = np.array([p[0] for p in tail], dtype=np.float64)
            cols = np.array([p[1] for p in tail], dtype=np.float64)
            env_metrics["mean_C_cov"] = float(np.mean(covs))
            env_metrics["mean_C_col"] = float(np.mean(cols))
        self.logger.log_env_diagnostics(env_metrics, step=self._tb_episode_idx)
        self._tb_episode_idx += 1

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
        profile_timing: bool = False,
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
        self._reset_tb_trackers(num_envs)

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
            policy_time = 0.0
            vec_step_time = 0.0
            value_boot_time = 0.0
            update_time = 0.0
            env_timing_totals: dict[str, float] = {}
            epoch_returns: list[float] = []
            epoch_lens: list[int] = []

            for step in range(rollout_steps):
                if profile_timing:
                    t_roll0 = time.perf_counter()
                    tp0 = time.perf_counter()
                    actions, log_probs, values = self._select_actions(obs, state, avail_actions)
                    policy_time += time.perf_counter() - tp0
                    ts0 = time.perf_counter()
                    step_result = self.vec_env_manager.step(actions)
                    vec_step_time += time.perf_counter() - ts0
                    tv0 = time.perf_counter()
                    next_values = self._evaluate_values(step_result.gae_next_obs, step_result.gae_next_state)
                    next_values = next_values * (1.0 - step_result.terminated[:, None].astype(np.float32))
                    value_boot_time += time.perf_counter() - tv0
                    rollout_time += time.perf_counter() - t_roll0
                else:
                    t0 = time.time()
                    actions, log_probs, values = self._select_actions(obs, state, avail_actions)
                    step_result = self.vec_env_manager.step(actions)
                    next_values = self._evaluate_values(step_result.gae_next_obs, step_result.gae_next_state)
                    next_values = next_values * (1.0 - step_result.terminated[:, None].astype(np.float32))
                    rollout_time += time.time() - t0

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
                self._aggregate_timing(step_result.infos, env_timing_totals)
                self._update_tb_trackers(step_result.infos, num_envs)

                episode_returns += step_result.rewards.sum(axis=1)
                episode_lengths += 1
                done_envs = np.flatnonzero(step_result.dones)
                for env_idx in done_envs:
                    epoch_returns.append(float(episode_returns[env_idx]))
                    epoch_lens.append(int(episode_lengths[env_idx]))
                    self._log_tb_episode(
                        env_idx,
                        float(episode_returns[env_idx]),
                        int(episode_lengths[env_idx]),
                        step_result.infos,
                    )
                    self._clear_tb_trackers_env(env_idx)
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
                if profile_timing:
                    steps_den = max(rollout_steps * num_envs, 1)
                    epoch_wall = rollout_time + update_time
                    rollout_frac = rollout_time / max(epoch_wall, 1e-9)
                    print(
                        f"[vec-profile] ms/env_step: policy={1000.0 * policy_time / steps_den:.3f} "
                        f"vec_env_ipc={1000.0 * vec_step_time / steps_den:.3f} "
                        f"bootstrap_V={1000.0 * value_boot_time / steps_den:.3f} "
                        f"ppo_update={update_ms_per_env_step:.3f} "
                        f"rollout_frac_of_epoch={rollout_frac:.3f}"
                    )

        return {
            "train/num_epochs": int(num_epochs),
            "train/avg_return": float(np.mean(all_episode_returns)) if all_episode_returns else 0.0,
            "train/avg_len": float(np.mean(all_episode_lens)) if all_episode_lens else 0.0,
            "train/num_episodes": int(len(all_episode_returns)),
        }
