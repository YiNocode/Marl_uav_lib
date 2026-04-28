"""Rollout worker: collect trajectories."""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch

from marl_uav.buffers.episode_buffer import EpisodeBuffer
from marl_uav.envs.base_env import BaseEnv
from marl_uav.runners.base_runner import BaseRunner
from marl_uav.utils.logger import Logger

# 3v1 围捕：episode 级 mean_C_cov / mean_C_col 仅对最后若干时刻（含 reset 后首帧）的指标取平均
PURSUIT_STRUCTURE_MEAN_LAST_STEPS = 30


class RolloutWorker(BaseRunner):
    """使用当前 policy 从 env 收集一条 episode 的 transition。"""

    def __init__(
        self,
        env: BaseEnv,
        policy: Any,
        *,
        get_actions_fn: Callable[..., np.ndarray] | None = None,
        logger: Logger | None = None,
    ) -> None:
        """
        Args:
            env: 多智能体环境（需实现 get_obs, get_state, get_avail_actions）
            policy: 策略对象，若提供 get_actions_fn 则用其取动作，否则需有 select_actions(obs, avail_actions=...) 方法
            get_actions_fn: 可选，签名 (obs, state, avail_actions) -> actions array，用于自定义取动作方式
        """
        self.env = env
        self.policy = policy
        self._get_actions_fn = get_actions_fn
        self._buffer: EpisodeBuffer | None = None
        self._logger = logger
        self._episode_idx = 0  # 用于 TensorBoard step（按 episode 计数）

    def _ensure_buffer(self) -> EpisodeBuffer:
        if self._buffer is None:
            n = getattr(self.env, "num_agents", 1)
            # 优先使用 env.obs_dim / env.state_dim，避免在没有 observation_space 时访问失败
            obs_dim = getattr(self.env, "obs_dim", None)
            if obs_dim is None:
                obs_space = getattr(self.env, "observation_space", None)
                if obs_space is None or not hasattr(obs_space, "shape"):
                    raise RuntimeError(
                        "Env must define obs_dim or observation_space.shape for RolloutWorker."
                    )
                obs_dim = obs_space.shape[0]
            state_dim = getattr(self.env, "state_dim", obs_dim * n)
            self._buffer = EpisodeBuffer(
                num_agents=n,
                obs_dim=obs_dim,
                state_dim=state_dim,
            )
        return self._buffer

    def _select_actions(
        self,
        obs: list[np.ndarray],
        state: np.ndarray,
        avail_actions: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """根据 policy 选择动作，并尽可能返回 log_probs / values。"""

        def _to_numpy(x: Any | None) -> np.ndarray | None:
            if x is None:
                return None
            # Torch tensor: 需要 detach().cpu().numpy()
            if hasattr(x, "detach") and hasattr(x, "cpu"):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        if self._get_actions_fn is not None:
            out = self._get_actions_fn(obs, state, avail_actions)
        else:
            # 兼容两类接口：
            # 1) legacy: select_actions(obs, avail_actions=...) -> actions
            # 2) MAC:   select_actions(...) -> (actions, log_probs, values[, entropy])
            out = self.policy.select_actions(  # type: ignore[call-arg]
                obs,
                state=state,
                avail_actions=avail_actions,
            )

        if isinstance(out, (tuple, list)):
            if len(out) >= 3:
                actions, log_probs, values = out[:3]
            elif len(out) == 2:
                actions, log_probs = out
                values = None
            else:
                actions = out[0]
                log_probs = None
                values = None
        else:
            actions = out
            log_probs = None
            values = None

        actions_np = _to_numpy(actions)
        if actions_np is None:
            raise TypeError("select_actions returned None for actions")
        # 单环境情况下，若 policy 返回 (1, n_agents)，去掉 batch 维
        if actions_np.ndim > 1 and actions_np.shape[0] == 1:
            actions_np = actions_np[0]

        log_probs_np = _to_numpy(log_probs)
        if isinstance(log_probs_np, np.ndarray) and log_probs_np.ndim > 1 and log_probs_np.shape[0] == 1:
            log_probs_np = log_probs_np[0]

        values_np = _to_numpy(values)
        if isinstance(values_np, np.ndarray) and values_np.ndim > 1 and values_np.shape[0] == 1:
            values_np = values_np[0]

        return actions_np, log_probs_np, values_np

    def _maybe_extract_dream_manifold_snapshot(self, state: np.ndarray) -> dict[str, Any] | None:
        """Return Dream-MAPPO manifold params for the current state if available."""
        policy_obj = getattr(self.policy, "policy", self.policy)
        if not (
            hasattr(policy_obj, "_prepare_state")
            and hasattr(policy_obj, "_state_b")
            and hasattr(policy_obj, "_geom_from_state")
        ):
            return None

        try:
            n_agents = int(getattr(self.env, "num_agents"))
        except (TypeError, ValueError):
            return None

        try:
            with torch.no_grad():
                state_tensor = policy_obj._prepare_state(state, B=1, N=n_agents)
                state_b = policy_obj._state_b(state_tensor)
                _, rho, psi = policy_obj._geom_from_state(state_b)
        except Exception:
            return None

        rho_np = rho.detach().cpu().numpy().reshape(-1)
        psi_np = psi.detach().cpu().numpy().reshape(-1)
        if rho_np.size == 0 or psi_np.size == 0:
            return None
        return {"rho": float(rho_np[0]), "psi": float(psi_np[0])}

    def collect_episode(
        self,
        seed: int | None = None,
        buffer: EpisodeBuffer | None = None,
        record_trajectory: bool = False,
    ) -> tuple[EpisodeBuffer, dict]:
        """
        收集一条完整 episode，存入 buffer 并返回。

        Args:
            record_trajectory: 若为 True 且 env 提供 prev_backend_state，则在 info 中附带
                info["trajectory"]，形状 [T+1, N, 3]，为各步全体 agent 的位置。

        Returns:
            buffer: 存满当前 episode 的 buffer
            info: 含 episode_return, episode_len, terminated, truncated，以及可选的 trajectory、
                pursuit_structure_series（3v1 时每步围捕结构指标，与 trajectory 时间对齐）、
                mean_C_cov / mean_C_col（3v1 时对 pursuit_structure 最后若干时刻的均值，见
                PURSUIT_STRUCTURE_MEAN_LAST_STEPS）、obstacle_termination（ex2 撞柱终局）、
                obstacle_xy / obstacle_r（本局柱布局，供可视化）
        """
        # 先 reset 环境，确保 env 已初始化好 obs_dim/state_dim 等属性
        obs_dict, env_info = self.env.reset(seed=seed)

        # ex2 圆柱：reset 时柱布局固定整局，供评估轨迹画图与日志
        obstacle_xy_snapshot = env_info.get("obstacle_xy")
        obstacle_r_snapshot = env_info.get("obstacle_r")

        buf = buffer if buffer is not None else self._ensure_buffer()
        buf.clear()

        obs_list = obs_dict["obs"]
        state = obs_dict["state"]
        avail_actions = self.env.get_avail_actions()
        episode_return = 0.0
        terminated = False
        truncated = False

        # 可选：记录轨迹（用于 3v1 追逃等任务的评估画图）
        traj_list: list[np.ndarray] = []
        pursuit_structure_series: List[Dict[str, Any]] = []
        dream_manifold_series: List[Dict[str, Any]] = []
        # 3v1 围捕：按时间顺序记录 C_cov / C_col，episode 末对最后 PURSUIT_STRUCTURE_MEAN_LAST_STEPS 步取均值
        pursuit_cov_col_pairs: list[tuple[float, float]] = []

        if getattr(self.env, "prev_backend_state", None) is not None:
            ps0 = env_info.get("pursuit_structure")
            if isinstance(ps0, dict):
                pursuit_cov_col_pairs.append(
                    (float(ps0["C_cov"]), float(ps0["C_col"]))
                )
            if record_trajectory:
                traj_list.append(
                    np.asarray(self.env.prev_backend_state.states[:, 3, :], dtype=np.float32)
                )
                if isinstance(ps0, dict):
                    pursuit_structure_series.append(ps0)
                manifold0 = self._maybe_extract_dream_manifold_snapshot(state)
                if manifold0 is not None:
                    dream_manifold_series.append(manifold0)

        # 用于聚合环境诊断（若 env 的 step_info 提供）
        mean_goal_distances: list[float] = []
        reward_progress_list: list[float] = []
        reward_time_penalty_list: list[float] = []
        reward_collision_penalty_list: list[float] = []
        reward_reach_bonus_list: list[float] = []
        env_timing_totals: dict[str, float] = {}
        # 追逃等任务的 episode 级统计
        any_capture = False
        first_capture_step = -1
        any_pursuer_oob = False
        any_collision = False
        any_timeout = False
        any_obstacle_termination = False
        last_step_info: dict = {}

        while True:
            actions, log_probs, values = self._select_actions(
                obs_list, state, avail_actions
            )
            next_obs_dict, rewards, terminated, truncated, step_info = self.env.step(
                actions
            )

            last_step_info = step_info
            if "mean_goal_distance" in step_info:
                mean_goal_distances.append(float(step_info["mean_goal_distance"]))
            if "reward_progress" in step_info:
                reward_progress_list.append(float(step_info["reward_progress"]))
            if "reward_time_penalty" in step_info:
                reward_time_penalty_list.append(float(step_info["reward_time_penalty"]))
            if "reward_collision_penalty" in step_info:
                reward_collision_penalty_list.append(float(step_info["reward_collision_penalty"]))
            if "reward_reach_bonus" in step_info:
                reward_reach_bonus_list.append(float(step_info["reward_reach_bonus"]))
            timing_info = step_info.get("timing")
            if isinstance(timing_info, dict):
                for k, v in timing_info.items():
                    env_timing_totals[k] = env_timing_totals.get(k, 0.0) + float(v)

            # 追逃任务相关 step 级信息
            if step_info.get("captured", False):
                any_capture = True
                if first_capture_step < 0 and step_info.get("capture_step", -1) >= 0:
                    first_capture_step = int(step_info["capture_step"])
            if step_info.get("pursuer_oob", False):
                any_pursuer_oob = True
            if step_info.get("has_collision", False):
                any_collision = True
            if step_info.get("timeout", False):
                any_timeout = True
            if step_info.get("obstacle_terminated", False):
                any_obstacle_termination = True

            ps_step = step_info.get("pursuit_structure")
            if isinstance(ps_step, dict):
                pursuit_cov_col_pairs.append(
                    (float(ps_step["C_cov"]), float(ps_step["C_col"]))
                )

            next_obs_list = next_obs_dict["obs"]
            next_state = next_obs_dict["state"]

            if record_trajectory and getattr(self.env, "prev_backend_state", None) is not None:
                traj_list.append(
                    np.asarray(self.env.prev_backend_state.states[:, 3, :], dtype=np.float32)
                )
                if isinstance(ps_step, dict):
                    pursuit_structure_series.append(ps_step)
                manifold_step = self._maybe_extract_dream_manifold_snapshot(next_state)
                if manifold_step is not None:
                    dream_manifold_series.append(manifold_step)
            done = terminated or truncated
            episode_return += sum(rewards)

            buf.add(
                obs=obs_list,
                state=state,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs_list,
                next_state=next_state,
                done=done,
                terminated=terminated,
                truncated=truncated,
                log_probs=log_probs,
                values=values,
                avail_actions=avail_actions,
            )

            if done:
                break
            obs_list = next_obs_list
            state = next_state
            avail_actions = self.env.get_avail_actions()

        info = {
            "episode_return": episode_return,
            "episode_len": buf.get_episode_length(),
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(last_step_info.get("all_reached", False)),
            # 对于导航任务：最后一步 has_collision 即 episode 级碰撞
            # 对于追逃任务：我们额外聚合 any_collision 作为 episode 级碰撞
            "collision": bool(any_collision or last_step_info.get("has_collision", False)),
            "out_of_bounds": bool(last_step_info.get("out_of_bounds", False)),
            "capture": bool(any_capture or last_step_info.get("captured", False)),
            "capture_step": int(first_capture_step),
            "pursuer_oob": bool(any_pursuer_oob or last_step_info.get("pursuer_oob", False)),
            "timeout": bool(any_timeout or last_step_info.get("timeout", False)),
            "obstacle_termination": bool(
                any_obstacle_termination or last_step_info.get("obstacle_terminated", False)
            ),
        }
        if obstacle_xy_snapshot is not None and obstacle_r_snapshot is not None:
            info["obstacle_xy"] = np.asarray(obstacle_xy_snapshot, dtype=np.float32).copy()
            info["obstacle_r"] = np.asarray(obstacle_r_snapshot, dtype=np.float32).copy()
        if traj_list:
            info["trajectory"] = np.stack(traj_list, axis=0)  # [T+1, N, 3]
        if pursuit_structure_series:
            info["pursuit_structure_series"] = pursuit_structure_series
        if dream_manifold_series:
            info["dream_manifold_series"] = dream_manifold_series
        if pursuit_cov_col_pairs:
            tail = pursuit_cov_col_pairs[-PURSUIT_STRUCTURE_MEAN_LAST_STEPS:]
            covs = np.array([p[0] for p in tail], dtype=np.float64)
            cols = np.array([p[1] for p in tail], dtype=np.float64)
            info["mean_C_cov"] = float(np.mean(covs))
            info["mean_C_col"] = float(np.mean(cols))
        if mean_goal_distances:
            info["env_mean_goal_distance"] = float(np.mean(mean_goal_distances))
            info["env_final_goal_distance"] = mean_goal_distances[-1]
        if reward_progress_list:
            info["env_reward_progress"] = sum(reward_progress_list)
        if reward_time_penalty_list:
            info["env_reward_time_penalty"] = sum(reward_time_penalty_list)
        if reward_collision_penalty_list:
            info["env_reward_collision_penalty"] = sum(reward_collision_penalty_list)
        if reward_reach_bonus_list:
            info["env_reward_reach_bonus"] = sum(reward_reach_bonus_list)

        # 若提供了 TensorBoard Logger，则在 episode 级别记录环境 & 诊断指标
        if env_timing_totals:
            info["env_timing_total_s"] = dict(env_timing_totals)
            episode_len = max(int(info["episode_len"]), 1)
            info["env_timing_mean_ms"] = {
                k: 1000.0 * v / episode_len for k, v in env_timing_totals.items()
            }

        if self._logger is not None:
            step = self._episode_idx

            # 1) 环境训练指标：train/*
            train_metrics = {
                "episode_return": float(info["episode_return"]),
                "episode_length": float(info["episode_len"]),
                # 成功 / 出界 / 碰撞 / 捕获 / timeout / 追捕方出界：
                # 按 episode 记 0/1，TensorBoard 中曲线即对应的比率
                "success_rate": 1.0 if info.get("success", False) else 0.0,
                "out_of_bounds_rate": 1.0 if info.get("out_of_bounds", False) else 0.0,
                "collision_rate": 1.0 if info.get("collision", False) else 0.0,
                "capture_rate": 1.0 if info.get("capture", False) else 0.0,
                "timeout_rate": 1.0 if info.get("timeout", False) else 0.0,
                "pursuer_oob_rate": 1.0 if info.get("pursuer_oob", False) else 0.0,
                "obstacle_termination_rate": 1.0 if info.get("obstacle_termination", False) else 0.0,
            }
            self._logger.log_train_env_metrics(train_metrics, step=step)

            # 2) 环境诊断指标：env/*
            env_metrics: dict[str, float] = {}
            if "env_mean_goal_distance" in info:
                env_metrics["mean_goal_distance"] = float(info["env_mean_goal_distance"])
            if "env_final_goal_distance" in info:
                env_metrics["final_goal_distance"] = float(info["env_final_goal_distance"])
            if "env_reward_progress" in info:
                env_metrics["reward_progress"] = float(info["env_reward_progress"])
            if "env_reward_time_penalty" in info:
                env_metrics["reward_time_penalty"] = float(info["env_reward_time_penalty"])
            if "env_reward_reach_bonus" in info:
                env_metrics["reward_reach_bonus"] = float(info["env_reward_reach_bonus"])
            if "env_reward_collision_penalty" in info:
                # 规格里未特别要求，但如有则一并记录
                env_metrics["reward_collision_penalty"] = float(
                    info["env_reward_collision_penalty"]
                )
            if "env_timing_mean_ms" in info:
                timing_mean_ms = info["env_timing_mean_ms"]
                if isinstance(timing_mean_ms, dict):
                    for k, v in timing_mean_ms.items():
                        env_metrics[f"time_{k.replace('_s', '')}_ms"] = float(v)
            if env_metrics:
                self._logger.log_env_diagnostics(env_metrics, step=step)

            self._episode_idx += 1

        return buf, info

    def run(
        self,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """BaseRunner 接口：执行一次 rollout，返回 (buffer, info)。"""
        return self.collect_episode(seed=seed, **kwargs)
