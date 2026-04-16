from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl_uav.envs.tasks.base_task import BaseTask


@dataclass
class NavigationTaskState:
    goals: np.ndarray         # [N, 3]
    reached: np.ndarray       # [N]
    prev_dist: np.ndarray     # [N]


class NavigationTask(BaseTask):
    def __init__(
        self,
        world_xy: float = 1.5,
        z_min: float = 0.5,
        z_max: float = 2.0,
        goal_reach_dist: float = 0.3,
        episode_limit: int = 500,
    ) -> None:
        self.world_xy = world_xy
        self.z_min = z_min
        self.z_max = z_max
        self.goal_reach_dist = goal_reach_dist
        self.episode_limit = episode_limit

    def sample_initial_conditions(self, num_agents: int, rng: np.random.Generator):
        start_pos = np.zeros((num_agents, 3), dtype=np.float32)
        start_pos[:, 0] = rng.uniform(-self.world_xy, self.world_xy, size=num_agents)
        start_pos[:, 1] = rng.uniform(-self.world_xy, self.world_xy, size=num_agents)
        start_pos[:, 2] = rng.uniform(self.z_min, self.z_max, size=num_agents)
        # start_pos[:, 0] = rng.uniform(-self.world_xy/20, self.world_xy/20, size=num_agents)
        # start_pos[:, 1] = rng.uniform(-self.world_xy/20, self.world_xy/20, size=num_agents)
        # start_pos[:, 2] = rng.uniform(self.z_min, self.z_max/2, size=num_agents)
        start_orn = np.zeros((num_agents, 3), dtype=np.float32)

        goals = np.zeros((num_agents, 3), dtype=np.float32)
        goals[:, 0] = rng.uniform(-self.world_xy, self.world_xy, size=num_agents)
        goals[:, 1] = rng.uniform(-self.world_xy, self.world_xy, size=num_agents)
        goals[:, 2] = rng.uniform(self.z_min, self.z_max, size=num_agents)
        # goals[:, 0] = rng.uniform(2.0, 2.1, size=num_agents)
        # goals[:, 1] = rng.uniform(2.0, 2.1, size=num_agents)
        # goals[:, 2] = rng.uniform(self.z_min, self.z_max/2, size=num_agents)
        prev_dist = np.linalg.norm(start_pos - goals, axis=1).astype(np.float32)
        task_state = NavigationTaskState(
            goals=goals,
            reached=prev_dist <= self.goal_reach_dist,
            prev_dist=prev_dist,
        )
        return start_pos, start_orn, task_state

    def build_obs(self, backend_state, task_state: NavigationTaskState) -> np.ndarray:
        states = backend_state.states  # [N, 4, 3]
        ang_vel = states[:, 0, :]
        ang_pos = states[:, 1, :]
        lin_vel = states[:, 2, :]
        lin_pos = states[:, 3, :]
        goal_delta = task_state.goals - lin_pos
        obs = np.concatenate(
            [lin_pos, lin_vel, ang_pos, goal_delta],
            axis=1,
        ).astype(np.float32)
        return obs

    def build_state(self, backend_state, task_state: NavigationTaskState) -> np.ndarray:
        states = backend_state.states
        lin_pos = states[:, 3, :]
        lin_vel = states[:, 2, :]
        ang_pos = states[:, 1, :]
        # 目标相对位置：goal - lin_pos，与 build_obs 中的 goal_delta 一致
        goal_relative = (task_state.goals - lin_pos).reshape(-1)
        state = np.concatenate(
            [
                lin_pos.reshape(-1),
                lin_vel.reshape(-1),
                ang_pos.reshape(-1),
                goal_relative,
            ],
            axis=0,
        ).astype(np.float32)
        return state

    def compute_rewards(
            self,
            prev_backend_state,
            backend_state,
            task_state: NavigationTaskState,
    ) -> np.ndarray:
        """导航任务奖励函数。

        设计思路：
        1. progress reward: 朝目标靠近得到正奖励，远离目标得到负奖励
        2. step penalty: 每步小惩罚，鼓励更快完成任务
        3. reach bonus: 首次到达目标时给予较大奖励
        4. stay bonus: 已到达目标后，每步给一个很小的保持奖励（防止离开目标点）
        5. out-of-bounds penalty: 飞出边界给惩罚
        6. collision penalty: 碰撞给惩罚
        """

        curr_pos = backend_state.states[:, 3, :]  # [N, 3]
        curr_dist = np.linalg.norm(curr_pos - task_state.goals, axis=1).astype(np.float32)

        # ---------- 1) progress reward ----------
        # 正值表示靠近目标，负值表示远离目标
        progress = task_state.prev_dist - curr_dist

        # 做一个裁剪，避免偶发大跳变导致 reward spike 太大
        progress = np.clip(progress, -1.0, 1.0)

        # 放大 progress 信号
        rewards = 2.0 * progress

        # ---------- 2) time penalty ----------
        rewards -= 0.02

        # ---------- 3) reach bonus ----------
        newly_reached = (curr_dist <= self.goal_reach_dist) & (~task_state.reached)
        rewards[newly_reached] += 8.0

        # ---------- 4) stay bonus ----------
        # 已到达目标后，给小额保持奖励，并避免 progress 噪声继续干扰
        already_reached = task_state.reached & (~newly_reached)
        rewards[already_reached] = 0.1

        # ---------- 5) out-of-bounds penalty ----------
        # 这里假设任务类里有 world_xy / z_min / z_max
        out_of_bounds = (
                (np.abs(curr_pos[:, 0]) > self.world_xy) |
                (np.abs(curr_pos[:, 1]) > self.world_xy) |
                (curr_pos[:, 2] < self.z_min) |
                (curr_pos[:, 2] > self.z_max)
        )
        rewards[out_of_bounds] -= 3.0

        # ---------- 6) collision penalty ----------
        # 若 contact_array 可反映碰撞，则对所有 agent 扣分；后续可细化成 per-agent
        if np.any(backend_state.contact_array):
            rewards -= 2.0

        # ---------- 更新 task state ----------
        task_state.reached = task_state.reached | (curr_dist <= self.goal_reach_dist)
        task_state.prev_dist = curr_dist

        return rewards.astype(np.float32)

    def compute_terminated_truncated(
            self,
            backend_state,
            task_state: NavigationTaskState,
            step_count: int,
    ):
        """计算 episode 是否 terminated / truncated。

        设计原则：
        1. 全部 agent 到达目标 -> terminated = True
        2. 只有当“过多 agent 出界”时才终止 episode，避免单个 agent 偶发出界直接毁掉训练
        3. 时间上限 -> truncated = True
        """

        lin_pos = backend_state.states[:, 3, :]  # [N, 3]

        # ---------- 1) 边界判定 ----------
        oob_mask = (
                (np.abs(lin_pos[:, 0]) > self.world_xy) |
                (np.abs(lin_pos[:, 1]) > self.world_xy) |
                (lin_pos[:, 2] < self.z_min) |
                (lin_pos[:, 2] > self.z_max)
        )

        num_oob = int(np.sum(oob_mask))
        num_agents = lin_pos.shape[0]

        # ---------- 2) 成功判定 ----------
        all_reached = bool(np.all(task_state.reached))

        # ---------- 3) 失败终止判定 ----------
        # 更稳妥的做法：
        # - 单个 agent 出界不立刻结束
        # - 超过一半 agent 出界，或全部 agent 出界，再终止
        too_many_oob = bool(num_oob >= max(1, num_agents // 2 + 1))

        # 如果你想更严格，也可以改成：
        # too_many_oob = bool(num_oob >= 1)
        # 但当前训练阶段不建议这么做

        terminated = bool(all_reached or too_many_oob)

        # ---------- 4) 时间截断 ----------
        truncated = bool(step_count >= self.episode_limit)

        return terminated, truncated

    def action_to_setpoint(
        self,
        actions: np.ndarray,
        backend_state,
        task_state: NavigationTaskState,
        *,
        action_space_type: str = "discrete",
        action_dim: int | None = None,
    ) -> np.ndarray:
        """离散: 查表；连续: actions (n_agents, 4) 视为 [-1,1] 映射到 ±0.15 速度 setpoint."""
        actions = np.asarray(actions)
        if action_space_type == "continuous" and actions.ndim == 2 and actions.dtype in (
            np.float32,
            np.float64,
        ):
            scale = 0.15
            setpoints = np.clip(actions.astype(np.float32), -1.0, 1.0) * scale
            if setpoints.shape[1] >= 4:
                setpoints[:, 3] = np.clip(setpoints[:, 3], -0.15, 0.15)
            return setpoints
        table = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],   # hover
                [0.15, 0.0, 0.0, 0.0],   # +x
                [-0.15, 0.0, 0.0, 0.0],  # -x
                [0.0, 0.15, 0.0, 0.0],   # +y
                [0.0, -0.15, 0.0, 0.0],  # -y
                [0.15, -0.15, 0.0, 0.15],   # +z
                [-0.15, 0.15, 0.0, -0.15],  # -z
                [0.15, 0.15, 0.0, 0.0],   # yaw+
                [-0.15, -0.15, 0.0, 0.0],  # yaw-
            ],
            dtype=np.float32,
        )
        actions = np.asarray(actions, dtype=np.int64)
        return table[actions]

