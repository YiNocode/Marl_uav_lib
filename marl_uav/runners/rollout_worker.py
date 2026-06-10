"""Rollout worker: collect trajectories."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List

import numpy as np
import torch

from marl_uav.buffers.episode_buffer import EpisodeBuffer
from marl_uav.envs.base_env import BaseEnv
from marl_uav.runners.base_runner import BaseRunner

if TYPE_CHECKING:
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
        self._capture_guard: Any | None = None
        self._expert_get_actions_fn: Callable[..., np.ndarray] | None = None
        self._bc_policy_for_diag: Any | None = None

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

        import time

        from marl_uav.utils.control_timing import publish_control_timing, should_record_control_timing

        record = should_record_control_timing(getattr(self, "env", None))
        t0 = time.perf_counter() if record else None

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

        if record and t0 is not None:
            publish_control_timing(self.env, total_decision_latency=time.perf_counter() - t0)

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
        from marl_uav.utils.debug_browser import get_debug_browser_hub, publish_episode_marker

        hub = get_debug_browser_hub()
        if hub is not None:
            ep_idx = hub.next_episode()
            total_eps = int(getattr(hub, "_total_episodes", 1))

        # reset 环境，确保 env 已初始化好 obs_dim/state_dim 等属性
        obs_dict, env_info = self.env.reset(seed=seed)

        if hub is not None:
            publish_episode_marker(
                "episode_start",
                episode=ep_idx,
                total_episodes=total_eps,
                seed=seed,
            )
            hub.arm_start_gate()
            hub.wait_if_blocked()

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
        reference_manifold_series: List[Dict[str, Any]] = []
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
                ref_targets0 = env_info.get("reference_manifold_targets")
                ref_curve0 = env_info.get("reference_manifold_curve")
                if ref_targets0 is not None or ref_curve0 is not None:
                    reference_manifold_series.append(
                        {
                            "targets": None if ref_targets0 is None else np.asarray(ref_targets0, dtype=np.float32).copy(),
                            "curve": None if ref_curve0 is None else np.asarray(ref_curve0, dtype=np.float32).copy(),
                        }
                    )

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
        obstacle_aware_series: list[dict[str, Any]] = []

        prev_min_dist = float(env_info.get("mean_goal_distance", 1e9))
        policy_action_mse_acc: list[float] = []
        policy_bc_kl_acc: list[float] = []
        action_deviation_sce_acc: list[float] = []

        while True:
            actions, log_probs, values = self._select_actions(
                obs_list, state, avail_actions
            )
            actions_np = np.asarray(actions, dtype=np.float32)
            if self._expert_get_actions_fn is not None:
                sce_actions = np.asarray(
                    self._expert_get_actions_fn(obs_list, state, avail_actions),
                    dtype=np.float32,
                )
                action_deviation_sce_acc.append(
                    float(np.mean((actions_np - sce_actions) ** 2))
                )
                guard = getattr(self, "_capture_guard", None)
                if guard is not None:
                    actions_np = guard.apply(
                        actions_np,
                        sce_actions,
                        min_pursuer_evader_dist=prev_min_dist,
                        sce_improves=True,
                    )
            if self._bc_policy_for_diag is not None:
                try:
                    import torch

                    obs_arr = np.asarray(obs_list, dtype=np.float32)
                    st_arr = np.asarray(state, dtype=np.float32)
                    with torch.no_grad():
                        out, _ = self._bc_policy_for_diag.forward(  # type: ignore[attr-defined]
                            obs_arr[np.newaxis, ...],
                            st_arr[np.newaxis, ...] if st_arr.ndim == 1 else st_arr,
                            deterministic=True,
                        )
                        bc_a = out["actions"][0].detach().cpu().numpy()
                    policy_action_mse_acc.append(float(np.mean((actions_np - bc_a) ** 2)))
                except Exception:
                    pass

            next_obs_dict, rewards, terminated, truncated, step_info = self.env.step(
                actions_np
            )
            if "mean_goal_distance" in step_info:
                prev_min_dist = float(step_info["mean_goal_distance"])

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

            oad = step_info.get("obstacle_aware_diagnostics")
            if isinstance(oad, dict):
                obstacle_aware_series.append(oad)

            ps_step = step_info.get("pursuit_structure")
            if isinstance(ps_step, dict):
                pursuit_cov_col_pairs.append(
                    (float(ps_step["C_cov"]), float(ps_step["C_col"]))
                )

            next_obs_list = next_obs_dict["obs"]
            next_state = next_obs_dict["state"]

            debug_hub = get_debug_browser_hub()
            if debug_hub is not None:
                from marl_uav.utils.debug_browser import publish_env_frame

                dream_step = self._maybe_extract_dream_manifold_snapshot(next_state)
                publish_env_frame(
                    self.env,
                    step_info,
                    event="step",
                    dream_manifold=dream_step,
                )
                debug_hub.wait_after_step()

            if record_trajectory and getattr(self.env, "prev_backend_state", None) is not None:
                traj_list.append(
                    np.asarray(self.env.prev_backend_state.states[:, 3, :], dtype=np.float32)
                )
                if isinstance(ps_step, dict):
                    pursuit_structure_series.append(ps_step)
                manifold_step = self._maybe_extract_dream_manifold_snapshot(next_state)
                if manifold_step is not None:
                    dream_manifold_series.append(manifold_step)
                ref_targets = step_info.get("reference_manifold_targets")
                ref_curve = step_info.get("reference_manifold_curve")
                if ref_targets is not None or ref_curve is not None:
                    reference_manifold_series.append(
                        {
                            "targets": None if ref_targets is None else np.asarray(ref_targets, dtype=np.float32).copy(),
                            "curve": None if ref_curve is None else np.asarray(ref_curve, dtype=np.float32).copy(),
                        }
                    )
            done = terminated or truncated
            episode_return += sum(rewards)

            buf.add(
                obs=obs_list,
                state=state,
                actions=actions_np,
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
        if reference_manifold_series:
            info["reference_manifold_series"] = reference_manifold_series
        if pursuit_cov_col_pairs:
            tail = pursuit_cov_col_pairs[-PURSUIT_STRUCTURE_MEAN_LAST_STEPS:]
            covs = np.array([p[0] for p in tail], dtype=np.float64)
            cols = np.array([p[1] for p in tail], dtype=np.float64)
            info["mean_C_cov"] = float(np.mean(covs))
            info["mean_C_col"] = float(np.mean(cols))
        if obstacle_aware_series:
            keys = (
                "assigned_pair_blocked_los",
                "path_cache_hit_rate",
                "num_replans_this_step",
                "stale_path_rate",
                "path_endpoint_error",
                "path_min_clearance",
                "path_tracking_error",
                "turn_safety_active_rate",
                "turn_arc_min_clearance",
                "turn_boundary_min_clearance",
                "turn_boundary_unsafe_rate",
                "turn_angle_rad",
                "slot_reachable_rate",
                "mean_time_to_slot",
                "max_time_to_slot",
                "path_clearance_min",
                "path_clearance_mean",
                "path_risk_integral",
                "slot_behind_obstacle_rate",
                "los_blocked_slot_rate",
                "unreachable_slot_rate",
                "fallback_slot_selection_rate",
                "assignment_switch_count",
                "cbf_active",
                "cbf_active_rate",
                "cbf_active_consecutive_steps",
                "cbf_correction_norm",
                "nominal_action_norm",
                "filtered_action_norm",
                "local_obstacle_count",
                "candidate_count",
                "valid_candidate_count",
                "local_planner_blocked",
                "local_planner_time_ms",
                "best_candidate_cost",
                "best_candidate_speed",
                "best_candidate_yaw_rate",
                "min_predicted_clearance",
                "assigned_slot_distance",
                "selected_action_norm",
                "cbf_filter_time_ms",
                "decision_time_ms",
            )
            for key in keys:
                vals = [
                    float(row[key])
                    for row in obstacle_aware_series
                    if key in row and row[key] is not None
                ]
                if vals:
                    info[f"mean_{key}"] = float(np.mean(vals))
            if traj_list:
                xy = np.asarray(traj_list, dtype=np.float64)[:, :3, :2]
                seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=2)
                info["avg_path_length"] = float(np.sum(seg_len))
            replans = [
                float(row.get("num_replans_this_step", 0.0))
                for row in obstacle_aware_series
            ]
            if replans:
                info["mean_num_replans_this_step"] = float(np.mean(replans))
            decision_vals = [
                float(row["decision_time_ms"])
                for row in obstacle_aware_series
                if "decision_time_ms" in row and row["decision_time_ms"] is not None
            ]
            if decision_vals:
                arr = np.asarray(decision_vals, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    info["avg_decision_ms"] = float(np.mean(arr))
                    info["p95_decision_ms"] = float(np.percentile(arr, 95))
                    info["max_decision_ms"] = float(np.max(arr))

        try:
            from marl_uav.control.obstacle_aware_sce_baselines import episode_timing_summary

            timing = episode_timing_summary(self.env)
            if timing:
                info.update(timing)
                info.setdefault("avg_decision_ms", timing.get("avg_decision_ms", timing.get("decision_total_avg_ms")))
                info.setdefault("p95_decision_ms", timing.get("p95_decision_ms", timing.get("decision_total_p95_ms")))
                info.setdefault("max_decision_ms", timing.get("max_decision_ms", timing.get("decision_total_max_ms")))
                info.setdefault("avg_los_ms", timing.get("los_check_avg_ms", 0.0))
                info.setdefault("avg_assignment_ms", timing.get("assignment_avg_ms", 0.0))
                info.setdefault("avg_path_planning_ms", timing.get("path_planning_avg_ms", 0.0))
                info.setdefault("avg_cbf_ms", timing.get("cbf_filter_avg_ms", 0.0))
        except Exception:
            pass

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
            self._logger.flush()

            self._episode_idx += 1

        if hub is not None:
            publish_episode_marker(
                "episode_end",
                episode=int(getattr(hub, "_episode_idx", 0)),
                total_episodes=int(getattr(hub, "_total_episodes", 1)),
                episode_return=float(info.get("episode_return", 0.0)),
                episode_len=int(info.get("episode_len", 0)),
                capture=bool(info.get("capture", False)),
            )

        if policy_action_mse_acc:
            info["policy_bc_action_mse"] = float(np.mean(policy_action_mse_acc))
        if action_deviation_sce_acc:
            info["action_deviation_from_sce"] = float(np.mean(action_deviation_sce_acc))
        guard = getattr(self, "_capture_guard", None)
        if guard is not None:
            guard.episode_end(captured=bool(any_capture))
            info.update(guard.stats.to_dict())
            guard.reset_stats()

        return buf, info

    def run(
        self,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """BaseRunner 接口：执行一次 rollout，返回 (buffer, info)。"""
        return self.collect_episode(seed=seed, **kwargs)
