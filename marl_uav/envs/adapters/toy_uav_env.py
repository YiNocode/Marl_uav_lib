"""Toy UAV environment adapter."""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from marl_uav.envs.base_env import BaseEnv


class ToyUavEnv(BaseEnv):
    """Toy multi-UAV environment for quick experiments.

    2D 平面，多智能体各自有目标点；每步选择离散动作（无操作/上/下/左/右）。
    观测为 [x, y, vx, vy, goal_dx, goal_dy]，全局 state 为所有智能体观测拼接。

    这版环境专门针对 IPPO 的最小验证做了修改：
    1. reward 使用“距离改善量 progress reward”
    2. 到达目标后 agent 冻结，不再移动
    3. 增加小时间惩罚
    4. 到达目标时给显式 bonus
    5. 全部到达时 episode 终止，否则到达上限时截断
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    ACTION_NOOP = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4

    def __init__(
        self,
        num_agents: int = 2,
        episode_limit: int = 30,
        world_size: float = 2.0,
        step_size: float = 0.1,
        goal_reach_dist: float = 0.1,
        time_penalty: float = 0.01,
        reach_bonus: float = 2.0,
        freeze_reached_agent: bool = True,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.episode_limit = episode_limit
        self.world_size = world_size
        self.step_size = step_size
        self.goal_reach_dist = goal_reach_dist
        self.time_penalty = time_penalty
        self.reach_bonus = reach_bonus
        self.freeze_reached_agent = freeze_reached_agent
        self.render_mode = render_mode

        # 每个智能体观测:
        # [x_norm, y_norm, vx_norm, vy_norm, goal_dx_norm, goal_dy_norm]
        self._obs_dim = 6
        obs_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self._state_dim = num_agents * self._obs_dim
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32
        )

        self._pos: np.ndarray               # (n_agents, 2)
        self._vel: np.ndarray               # (n_agents, 2)
        self._goals: np.ndarray             # (n_agents, 2)
        self._prev_dist: np.ndarray         # (n_agents,)
        self._reached: np.ndarray           # (n_agents,), bool
        self._step_count: int
        self._rng: np.random.Generator

        if seed is not None:
            self.reset(seed=seed)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()

        half = self.world_size / 2

        self._pos = self._rng.uniform(-half, half, (self.num_agents, 2)).astype(
            np.float32
        )
        self._goals = self._rng.uniform(-half, half, (self.num_agents, 2)).astype(
            np.float32
        )
        self._vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._step_count = 0

        self._prev_dist = np.linalg.norm(self._pos - self._goals, axis=1).astype(
            np.float32
        )
        self._reached = self._prev_dist <= self.goal_reach_dist

        obs = self.get_obs()
        state = self.get_state()

        infos = {
            "state": state,
            "all_reached": bool(np.all(self._reached)),
            "reached_mask": self._reached.copy(),
        }

        return {"obs": obs, "state": state}, infos

    def step(
        self, actions: list[int] | np.ndarray
    ) -> tuple[dict, list[float], bool, bool, dict]:
        if isinstance(actions, (list, tuple)):
            actions = np.asarray(actions, dtype=np.int32)
        else:
            actions = np.asarray(actions, dtype=np.int32)

        if actions.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} actions, got {len(actions)}"
            )

        prev_dist = self._prev_dist.copy()

        # 动作 -> 位移/速度
        move = np.zeros((self.num_agents, 2), dtype=np.float32)
        active_mask = ~self._reached

        move[(actions == self.ACTION_UP) & active_mask, 1] = self.step_size
        move[(actions == self.ACTION_DOWN) & active_mask, 1] = -self.step_size
        move[(actions == self.ACTION_LEFT) & active_mask, 0] = -self.step_size
        move[(actions == self.ACTION_RIGHT) & active_mask, 0] = self.step_size

        if self.freeze_reached_agent:
            move[self._reached] = 0.0

        self._vel = move.copy()
        self._pos = self._pos + self._vel

        # 边界裁剪
        half = self.world_size / 2
        self._pos = np.clip(self._pos, -half, half)

        self._step_count += 1

        new_dist = np.linalg.norm(self._pos - self._goals, axis=1).astype(np.float32)
        newly_reached = (new_dist <= self.goal_reach_dist) & (~self._reached)
        self._reached = self._reached | (new_dist <= self.goal_reach_dist)

        rewards = self._compute_rewards(prev_dist, new_dist, newly_reached)

        self._prev_dist = new_dist.copy()

        obs = self.get_obs()
        state = self.get_state()

        all_reached = bool(np.all(self._reached))
        terminated = all_reached
        truncated = self._step_count >= self.episode_limit

        infos = {
            "state": state,
            "all_reached": all_reached,
            "reached_mask": self._reached.copy(),
            "newly_reached": newly_reached.copy(),
            "mean_dist": float(new_dist.mean()),
        }

        return {"obs": obs, "state": state}, rewards, terminated, truncated, infos

    def _compute_rewards(
        self,
        prev_dist: np.ndarray,
        new_dist: np.ndarray,
        newly_reached: np.ndarray,
    ) -> list[float]:
        """
        reward 设计：
        1. progress = prev_dist - new_dist，朝目标靠近则为正
        2. 每步一个很小的时间惩罚，鼓励更快到达
        3. 首次到达目标时给予额外 bonus
        4. 已到达并冻结的 agent 后续不给持续 bonus，避免刷分
        """
        progress = prev_dist - new_dist
        rewards = progress - self.time_penalty

        # 首次到达目标时给额外奖励
        rewards[newly_reached] += self.reach_bonus

        # 对已经到达且冻结的 agent，后续 reward 设为 0 更稳定
        if self.freeze_reached_agent:
            already_reached_before = self._reached & (~newly_reached)
            rewards[already_reached_before] = 0.0

        return rewards.astype(np.float32).tolist()

    def get_obs(self) -> np.ndarray:
        """返回 shape=(num_agents, obs_dim) 的二维数组。"""
        half = self.world_size / 2

        x_norm = self._pos[:, 0] / half
        y_norm = self._pos[:, 1] / half
        vx = self._vel[:, 0] / max(self.step_size, 1e-6)
        vy = self._vel[:, 1] / max(self.step_size, 1e-6)
        goal_dx = (self._goals[:, 0] - self._pos[:, 0]) / half
        goal_dy = (self._goals[:, 1] - self._pos[:, 1]) / half

        obs = np.stack([x_norm, y_norm, vx, vy, goal_dx, goal_dy], axis=1).astype(
            np.float32
        )
        return obs

    def get_avail_actions(self) -> list[np.ndarray]:
        """当前所有动作均可用，返回全 1 mask。"""
        return [np.ones(self.n_actions, dtype=np.float32) for _ in range(self.num_agents)]

    def get_state(self) -> np.ndarray:
        """全局 state：所有智能体观测拼接。"""
        return self.get_obs().reshape(-1).astype(np.float32)

    def _check_all_goals_reached(self) -> bool:
        return bool(np.all(self._reached))

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def n_actions(self) -> int:
        return int(self.action_space.n)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        pass

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> "ToyUavEnv":
        return cls(
            num_agents=config.get("num_agents", 2),
            episode_limit=config.get("episode_limit", 30),
            world_size=config.get("world_size", 2.0),
            step_size=config.get("step_size", 0.1),
            goal_reach_dist=config.get("goal_reach_dist", 0.1),
            time_penalty=config.get("time_penalty", 0.01),
            reach_bonus=config.get("reach_bonus", 2.0),
            freeze_reached_agent=config.get("freeze_reached_agent", True),
            render_mode=render_mode,
            seed=seed,
        )