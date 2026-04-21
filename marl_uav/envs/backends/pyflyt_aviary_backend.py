from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PyFlyt.core import Aviary


@dataclass
class BackendState:
    """轻量级状态打包结构，用于上层 Env / Runner."""

    # 形状约定：[num_agents, 4, 3]
    # 参考文档：state[i] 的含义为：
    #   state[0, :] -> 机体系角速度
    #   state[1, :] -> 地面系角度
    #   state[2, :] -> 机体系线速度
    #   state[3, :] -> 地面系位置
    states: np.ndarray
    aux_states: list[np.ndarray]
    contact_array: np.ndarray
    elapsed_time: float


class PyFlytAviaryBackend:
    """基于 PyFlyt.core.Aviary 的多 UAV 仿真后端封装.

    功能目标：
    - 通过 ``set_mode`` 配置飞控模式（例如 QuadX 的位置控制模式 7）
    - 通过 ``set_setpoint`` / ``set_all_setpoints`` 下发控制目标
    - 通过 ``step()`` 推进一个控制周期（与 control_hz 对齐）
    - 通过 ``state(i)`` / ``all_states`` 读取状态
    - 通过 ``contact_array`` 读取碰撞信息
    - 通过 ``reset()`` 重置仿真
    """

    def __init__(
        self,
        num_agents: int,
        drone_type: str = "quadx",
        render: bool = False,
        physics_hz: int = 240,
        control_hz: int = 30,
        world_scale: float = 5.0,
        drone_options: dict[str, Any] | None = None,
        seed: int | None = None,
        flight_mode: int = 9,
    ) -> None:
        """构造函数.

        参数说明基本与 Aviary 保持一致，新增:
        - num_agents: 智能体 / UAV 数量
        - control_hz: 控制回路频率，必须整除 physics_hz
        - flight_mode: 通过 ``Aviary.set_mode`` 设置的飞控模式编号
        """
        if physics_hz % control_hz != 0:
            raise ValueError(
                f"control_hz={control_hz} 必须是 physics_hz={physics_hz} 的因子，"
                "以确保每次 step() 都对应完整的一个控制周期。"
            )

        self.num_agents = int(num_agents)
        self.drone_type = drone_type
        self.render = render
        self.physics_hz = int(physics_hz)
        self.control_hz = int(control_hz)
        self.world_scale = float(world_scale)
        self.seed = seed
        self.drone_options = drone_options or {}
        self.flight_mode = int(flight_mode)

        self.env: Aviary | None = None

    # ------------------------------------------------------------------
    # 生命周期控制
    # ------------------------------------------------------------------
    def reset(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
    ) -> BackendState:
        """重置仿真并返回当前 BackendState.

        参数:
        - start_pos: 形状为 (num_agents, 3) 的初始位置 [x, y, z]
        - start_orn: 形状为 (num_agents, 3) 的初始欧拉角 [roll, pitch, yaw]
        """
        if self.env is not None:
            # 确保上一次的仿真被正确释放
            self.env.disconnect()
            self.env = None

        start_pos = np.asarray(start_pos, dtype=np.float32)
        start_orn = np.asarray(start_orn, dtype=np.float32)

        if start_pos.shape != (self.num_agents, 3):
            raise ValueError(
                f"start_pos 形状应为 {(self.num_agents, 3)}，实际为 {start_pos.shape}"
            )
        if start_orn.shape != (self.num_agents, 3):
            raise ValueError(
                f"start_orn 形状应为 {(self.num_agents, 3)}，实际为 {start_orn.shape}"
            )

        # 为每个 UAV 构造独立的 drone_options，并写入 control_hz
        drone_options_list: list[dict[str, Any]] = []
        for _ in range(self.num_agents):
            opt = dict(self.drone_options)
            opt["control_hz"] = self.control_hz
            drone_options_list.append(opt)

        # 实例化 Aviary，多机统一类型 / 统一控制频率
        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type=[self.drone_type] * self.num_agents,
            drone_options=drone_options_list,
            render=self.render,
            physics_hz=self.physics_hz,
            world_scale=self.world_scale,
            seed=self.seed,
        )
        if self.render:
            self.env.resetDebugVisualizerCamera(cameraDistance=5,
                                               cameraYaw=0,
                                               cameraPitch=-45,
                                               cameraTargetPosition=[0, 0, 5] )
        # 通过 set_mode 设置飞控模式
        print("flight_mode:", self.flight_mode)
        self.env.set_mode([self.flight_mode] * self.num_agents)

        # 如果使用 QuadX 位置控制模式（例如 7），
        # 上层应在 reset 之后立刻通过 step(setpoints) 下发期望 [x, y, z, yaw]
        return self.get_backend_state()

    def step(self, setpoints: np.ndarray) -> BackendState:
        """下发控制目标并推进一个控制周期."""
        if self.env is None:
            raise RuntimeError("Aviary 尚未初始化，请先调用 reset()。")

        setpoints = np.asarray(setpoints, dtype=np.float32)
        if setpoints.shape[0] != self.num_agents:
            raise ValueError(
                f"setpoints 第一维大小应为 num_agents={self.num_agents}，实际为 {setpoints.shape[0]}"
            )

        # 通过 set_all_setpoints 一次性下发所有 UAV 的控制目标
        self.env.set_all_setpoints(setpoints)
        # 调用 Aviary.step()，内部会按照 physics_hz / control_hz 比例推进物理仿真
        self.env.step()
        return self.get_backend_state()

    # ------------------------------------------------------------------
    # 状态访问与封装
    # ------------------------------------------------------------------
    def get_backend_state(self) -> BackendState:
        """从 Aviary 读取当前仿真状态并封装为 BackendState."""
        if self.env is None:
            raise RuntimeError("Aviary 尚未初始化，请先调用 reset()。")



        # all_states: list[np.ndarray]，每个元素形状为 (4, 3)
        all_states = np.asarray(self.env.all_states, dtype=np.float32)
        # all_aux_states: list[np.ndarray]，每个 UAV 的辅助状态
        all_aux = list(self.env.all_aux_states)
        # contact_array: numpy.ndarray，碰撞矩阵
        contact_array = np.asarray(self.env.contact_array, dtype=np.int8)

        # 仿真已过去的时间（单位：秒）
        # 优先使用 Aviary 自带的 elapsed_time 属性；若不存在，则根据 aviary_steps / control_hz 估算。
        if hasattr(self.env, "elapsed_time"):
            elapsed_time = float(self.env.elapsed_time)
        elif hasattr(self.env, "aviary_steps"):
            elapsed_time = float(self.env.aviary_steps) / float(self.control_hz)
        else:
            elapsed_time = 0.0

        return BackendState(
            states=all_states,
            aux_states=all_aux,
            contact_array=contact_array,
            elapsed_time=elapsed_time,
        )

    # ------------------------------------------------------------------
    # 低级接口直通（可选）
    # ------------------------------------------------------------------
    def set_mode(self, modes: int | list[int]) -> None:
        """直接修改 Aviary 的飞控模式."""
        if self.env is None:
            raise RuntimeError("Aviary 尚未初始化，请先调用 reset()。")
        self.env.set_mode(modes)

    def set_setpoint(self, index: int, setpoint: np.ndarray) -> None:
        """为单个 UAV 下发控制目标."""
        if self.env is None:
            raise RuntimeError("Aviary 尚未初始化，请先调用 reset()。")
        setpoint = np.asarray(setpoint, dtype=np.float32)
        self.env.set_setpoint(int(index), setpoint)

    def state(self, index: int) -> np.ndarray:
        """读取单个 UAV 的状态，相当于 Aviary.state(i)."""
        if self.env is None:
            raise RuntimeError("Aviary 尚未初始化，请先调用 reset()。")
        return np.asarray(self.env.state(int(index)), dtype=np.float32)

    @property
    def all_states(self) -> np.ndarray:
        """读取所有 UAV 的状态，相当于 Aviary.all_states."""
        if self.env is None:
            raise RuntimeError("Aviary 尚未初始化，请先调用 reset()。")
        return np.asarray(self.env.all_states, dtype=np.float32)

    @property
    def contact_array(self) -> np.ndarray:
        """读取当前的碰撞信息矩阵."""
        if self.env is None:
            raise RuntimeError("Aviary 尚未初始化，请先调用 reset()。")
        return np.asarray(self.env.contact_array)

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------
    def close(self) -> None:
        """关闭当前 Aviary 仿真."""
        if self.env is not None:
            self.env.disconnect()
            self.env = None

