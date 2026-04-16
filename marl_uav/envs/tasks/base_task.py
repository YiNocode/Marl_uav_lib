from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTask(ABC):
    """任务抽象层接口，用于在物理后端之上定义任务逻辑."""

    @abstractmethod
    def sample_initial_conditions(
        self,
        num_agents: int,
        rng: np.random.Generator,
    ):
        """采样任务的初始条件（例如目标点、初始位姿等）."""
        raise NotImplementedError

    @abstractmethod
    def build_obs(self, backend_state, task_state) -> np.ndarray:
        """根据后端状态与任务状态构造多智能体观测."""
        raise NotImplementedError

    @abstractmethod
    def build_state(self, backend_state, task_state) -> np.ndarray:
        """根据后端状态与任务状态构造全局 state."""
        raise NotImplementedError

    @abstractmethod
    def compute_rewards(
        self,
        prev_backend_state,
        backend_state,
        task_state,
    ) -> np.ndarray:
        """根据前后两帧后端状态与任务状态计算每个智能体奖励."""
        raise NotImplementedError

    @abstractmethod
    def compute_terminated_truncated(
        self,
        backend_state,
        task_state,
        step_count: int,
    ):
        """根据当前状态与步数，计算 episode 是否终止 / 截断."""
        raise NotImplementedError

    @abstractmethod
    def action_to_setpoint(
        self,
        actions: np.ndarray,
        backend_state,
        task_state,
        *,
        action_space_type: str = "discrete",
        action_dim: int | None = None,
    ) -> np.ndarray:
        """将高层离散/连续动作映射为底层控制 setpoint。

        - 离散: actions 形状 (n_agents,) int，查表得到 setpoint
        - 连续: actions 形状 (n_agents, action_dim) float，按任务约定缩放/裁剪后作为 setpoint
        """
        raise NotImplementedError

