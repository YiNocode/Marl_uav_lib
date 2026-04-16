from __future__ import annotations

import numpy as np
from gymnasium import spaces

from marl_uav.envs.base_env import BaseEnv
from marl_uav.envs.backends.pyflyt_aviary_backend import PyFlytAviaryBackend
from marl_uav.envs.tasks.base_task import BaseTask
from marl_uav.envs.tasks.navigation_task import NavigationTask
from marl_uav.envs.tasks.pursuit_evasion_3v1_task import (
    PursuitEvasion3v1Task,
    compute_pursuit_structure_metrics_3v1,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx1,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx2,
    PursuitEvasion3v1TaskEx2State,
)

# 追逃 3v1：标准任务与 ex1（扩展观测）/ ex2（圆柱障碍）共用同一套 env 诊断与离散动作数
PURSUIT_EVASION_3V1_TASK_TYPES = (PursuitEvasion3v1Task, PursuitEvasion3v1TaskEx1, PursuitEvasion3v1TaskEx2)


class PyFlytAviaryEnv(BaseEnv):
    """统一封装 PyFlyt Aviary 后端与任务的多智能体环境.

    支持 NavigationTask / PursuitEvasion3v1Task / PursuitEvasion3v1TaskEx1 / PursuitEvasion3v1TaskEx2 等 BaseTask 子类。
    支持离散/连续动作空间，由配置 action_space 与 action_dim 决定。
    """

    def __init__(
        self,
        backend: PyFlytAviaryBackend,
        task: BaseTask,
        seed: int | None = None,
        *,
        action_space: str = "discrete",
        action_dim: int = 4,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.task_state = None
        self.prev_backend_state = None
        self.step_count = 0

        self._action_space_type = str(action_space).lower()
        if self._action_space_type not in ("discrete", "continuous"):
            raise ValueError(f"action_space must be 'discrete' or 'continuous', got {action_space!r}")

        self.num_agents: int = self.backend.num_agents
        self.obs_dim: int | None = None
        self.state_dim: int | None = None

        if self._action_space_type == "discrete":
            if isinstance(self.task, PURSUIT_EVASION_3V1_TASK_TYPES):
                self.n_actions: int = 7
            else:
                self.n_actions: int = 9
            self.action_dim: int | None = None
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.n_actions = 0
            self.action_dim = int(action_dim)
            if action_low is None and action_high is None:
                low = np.full((self.action_dim,), -1.0, dtype=np.float32)
                high = np.full((self.action_dim,), 1.0, dtype=np.float32)
            else:
                if action_low is None or action_high is None:
                    raise ValueError(
                        "连续动作下 action_low 与 action_high 必须同时提供或同时省略（默认 [-1,1]）。"
                    )
                low = np.asarray(action_low, dtype=np.float32).reshape(-1)
                high = np.asarray(action_high, dtype=np.float32).reshape(-1)
                if low.size != self.action_dim or high.size != self.action_dim:
                    raise ValueError(
                        f"action_low/high 长度须等于 action_dim={self.action_dim}，"
                        f"当前为 {low.size} 与 {high.size}。"
                    )
            self.action_space = spaces.Box(
                low=low, high=high, shape=(self.action_dim,), dtype=np.float32
            )
            # 与策略 Gaussian 头、boxed_action_bounds 对齐（来自配置或默认）
            self.action_low_np = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
            self.action_high_np = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # backend 真实 UAV 数由 backend.num_agents 决定（例如 3v1 追逃时为 4）
        start_pos, start_orn, self.task_state = self.task.sample_initial_conditions(
            self.backend.num_agents,
            self.rng,
        )
        backend_state = self.backend.reset(start_pos, start_orn)
        self.prev_backend_state = backend_state
        self.step_count = 0

        obs = self.task.build_obs(backend_state, self.task_state)
        state = self.task.build_state(backend_state, self.task_state)
        # 在首次 reset 时记录 obs/state 维度，供 RolloutWorker / MAC 等使用
        # 其中 obs.shape[0] 是“策略控制的 agent 数”（例如 3v1 追逃时为 3）
        self.num_agents = int(obs.shape[0])
        self.obs_dim = int(obs.shape[1])
        self.state_dim = int(state.shape[0])
        self._last_obs = obs
        self._last_state = state
        info = {"state": state}
        if isinstance(self.task, PURSUIT_EVASION_3V1_TASK_TYPES):
            lin_pos0 = backend_state.states[:, 3, :]
            ps = lin_pos0[self.task_state.pursuer_ids]
            pe = lin_pos0[self.task_state.evader_id]
            info["pursuit_structure"] = compute_pursuit_structure_metrics_3v1(ps, pe)
        if isinstance(self.task, PursuitEvasion3v1TaskEx2) and isinstance(
            self.task_state, PursuitEvasion3v1TaskEx2State
        ):
            info["obstacle_xy"] = np.asarray(self.task_state.obstacle_xy, dtype=np.float32).copy()
            info["obstacle_r"] = np.asarray(self.task_state.obstacle_r, dtype=np.float32).copy()
        return {"obs": obs, "state": state}, info

    def step(self, actions):
        # 离散: actions (n_agents,) int; 连续: actions (n_agents, action_dim) float
        setpoints = self.task.action_to_setpoint(
            actions,
            self.prev_backend_state,
            self.task_state,
            action_space_type=self._action_space_type,
            action_dim=self.action_dim,
        )
        backend_state = self.backend.step(setpoints)
        self.step_count += 1

        # 环境诊断（在 compute_rewards 更新 task_state 之前计算）
        # 不同任务的诊断信息略有差异，这里按任务类型分别处理。
        mean_goal_distance = 0.0
        reward_progress = 0.0
        reward_time_penalty = 0.0
        reward_reach_bonus = 0.0
        reward_collision_penalty = 0.0

        lin_pos = backend_state.states[:, 3, :]  # [N_real, 3]

        if isinstance(self.task, NavigationTask):
            curr_dist = np.linalg.norm(lin_pos - self.task_state.goals, axis=1).astype(np.float32)
            mean_goal_distance = float(np.mean(curr_dist))
            progress = self.task_state.prev_dist - curr_dist
            reward_progress = float(np.sum(progress))
            reward_time_penalty = -0.01 * self.num_agents
            newly_reached = (curr_dist <= self.task.goal_reach_dist) & (~self.task_state.reached)
            reward_reach_bonus = 2.0 * float(np.sum(newly_reached))
            reward_collision_penalty = (
                -float(self.num_agents) if np.any(backend_state.contact_array) else 0.0
            )
        elif isinstance(self.task, PURSUIT_EVASION_3V1_TASK_TYPES):
            # 对追逃任务，给出基于 pursuer-evader 距离的简单诊断信息
            pursuer_pos = lin_pos[self.task_state.pursuer_ids]
            evader_pos = lin_pos[self.task_state.evader_id]
            dists = np.linalg.norm(pursuer_pos - evader_pos[None, :], axis=1).astype(np.float32)
            mean_goal_distance = float(np.mean(dists))  # 这里“目标”理解为 evader
            reward_time_penalty = -0.01 * self.num_agents
            reward_collision_penalty = -float(self.num_agents) if np.any(backend_state.contact_array) else 0.0
        # 其他任务暂不填充诊断细节

        # 追逃任务需要检测“是否在本步新发生捕获”
        prev_captured = bool(getattr(self.task_state, "captured", False))

        rewards = self.task.compute_rewards(
            self.prev_backend_state,
            backend_state,
            self.task_state,
        )
        terminated, truncated = self.task.compute_terminated_truncated(
            backend_state,
            self.task_state,
            self.step_count,
        )

        obs = self.task.build_obs(backend_state, self.task_state)
        state = self.task.build_state(backend_state, self.task_state)
        self._last_obs = obs
        self._last_state = state

        # 统计用信息：终止原因、碰撞/坠毁等
        # 注意这里 lin_pos 是 backend 实际 UAV 数（例如 3v1 时为 4）
        lin_pos = backend_state.states[:, 3, :]  # [N_real, 3]

        if isinstance(self.task, NavigationTask):
            all_reached = bool(np.all(self.task_state.reached))
        else:
            # 对非 Navigation 任务，保留字段但语义不同：例如追逃中用 captured 代表“成功”
            all_reached = bool(getattr(self.task_state, "captured", False))

        out_of_bounds = bool(
            np.any(np.abs(lin_pos[:, :2]) > getattr(self.task, "world_xy", 5.0) * 1.2)
            or np.any(
                (lin_pos[:, 2] < 0.1)
                | (lin_pos[:, 2] > getattr(self.task, "z_max", 2.0) * 1.5)
            )
        )
        has_collision = bool(np.any(backend_state.contact_array))
        crash = bool(np.any(lin_pos[:, 2] < 0.1))

        # 追逃任务专用的额外诊断：捕获 / 出界类型 / 是否 timeout 等
        captured = False
        newly_captured = False
        capture_step = -1
        pursuer_oob = False
        too_many_pursuers_oob = False
        evader_oob = False
        timeout = False

        pursuer_obstacle_hit = False
        obstacle_terminated = False

        if isinstance(self.task, PURSUIT_EVASION_3V1_TASK_TYPES):
            captured = bool(getattr(self.task_state, "captured", False))
            if captured and not prev_captured:
                newly_captured = True
                capture_step = int(self.step_count)

            pursuer_pos = lin_pos[self.task_state.pursuer_ids]
            evader_pos = lin_pos[self.task_state.evader_id]

            p_oob_mask = self.task._get_oob_mask(pursuer_pos)
            num_p_oob = int(np.sum(p_oob_mask))
            pursuer_oob = bool(num_p_oob >= 1)
            too_many_pursuers_oob = bool(num_p_oob >= 2)
            evader_oob = bool(self.task._get_oob_mask(evader_pos[None, :])[0])

            # timeout: 由于时间上限导致的截断（既不是捕获，也不是出界终止）
            if truncated and not (captured or too_many_pursuers_oob or evader_oob):
                timeout = True

            if isinstance(self.task, PursuitEvasion3v1TaskEx2) and isinstance(
                self.task_state, PursuitEvasion3v1TaskEx2State
            ):
                hit_mask = self.task._pursuer_obstacle_collision_mask(pursuer_pos, self.task_state)
                pursuer_obstacle_hit = bool(np.any(hit_mask))
                # 本步因几何碰柱而 terminated，且本步不是「刚完成捕获」的成功终局
                obstacle_terminated = bool(terminated and pursuer_obstacle_hit and not newly_captured)

        info = {
            "state": state,
            "all_reached": all_reached,
            "out_of_bounds": out_of_bounds,
            "has_collision": has_collision,
            "crash": crash,
            "mean_goal_distance": mean_goal_distance,
            "final_goal_distance": mean_goal_distance,
            "reward_progress": reward_progress,
            "reward_time_penalty": reward_time_penalty,
            "reward_reach_bonus": reward_reach_bonus,
            "reward_collision_penalty": reward_collision_penalty,
        }

        if isinstance(self.task, PURSUIT_EVASION_3V1_TASK_TYPES):
            ps = lin_pos[self.task_state.pursuer_ids]
            pe = lin_pos[self.task_state.evader_id]
            pursuit_structure = compute_pursuit_structure_metrics_3v1(ps, pe)
            info.update(
                {
                    "captured": captured,
                    "newly_captured": newly_captured,
                    "capture_step": capture_step,
                    "pursuer_oob": pursuer_oob,
                    "too_many_pursuers_oob": too_many_pursuers_oob,
                    "evader_oob": evader_oob,
                    "timeout": timeout,
                    "pursuit_structure": pursuit_structure,
                    "pursuer_obstacle_hit": pursuer_obstacle_hit,
                    "obstacle_terminated": obstacle_terminated,
                }
            )

        self.prev_backend_state = backend_state
        return {"obs": obs, "state": state}, rewards.tolist(), terminated, truncated, info

    def get_obs(self):
        """兼容 BaseEnv 接口，返回最近一次 step/reset 的 obs 列表形式。"""
        if not hasattr(self, "_last_obs"):
            raise RuntimeError("Env has not been reset yet.")
        # RolloutWorker 期望 list[np.ndarray]
        return list(np.asarray(self._last_obs))

    def get_state(self):
        """兼容 BaseEnv 接口，返回最近一次 step/reset 的全局 state。"""
        if not hasattr(self, "_last_state"):
            raise RuntimeError("Env has not been reset yet.")
        return np.asarray(self._last_state, dtype=np.float32)

    def get_avail_actions(self):
        """离散：全动作可用，返回全 1 mask；连续：返回全 1（策略侧会忽略）。"""
        if self._action_space_type == "continuous":
            return [np.ones(self.action_dim, dtype=np.float32) for _ in range(self.num_agents)]
        return [np.ones(self.n_actions, dtype=np.float32) for _ in range(self.num_agents)]

    def close(self):
        self.backend.close()

