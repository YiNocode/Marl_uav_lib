from __future__ import annotations

import time

import numpy as np

from marl_uav.envs.backends.pyflyt_aviary_backend import PyFlytAviaryBackend


def main() -> None:
    # 创建一个简单的 PyFlyt 后端，2 架 quadx，无渲染可设为 False
    backend = PyFlytAviaryBackend(
        num_agents=2,
        drone_type="quadx",
        render=True,          # 若只想跑速度测试可改成 False
        physics_hz=240,
        control_hz=60,
        world_scale=5.0,
        drone_options={},     # 可按需传入 use_camera 等选项
        seed=0,
        flight_mode=7,        # QuadX 位置控制模式，setpoint=[x, y, z, yaw]
    )

    # 初始位置：两架机在原点左右各 1 米，高度 1 米
    start_pos = np.array(
        [
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    # 初始姿态全 0（roll, pitch, yaw）
    start_orn = np.zeros((2, 3), dtype=np.float32)

    state = backend.reset(start_pos, start_orn)
    print("Initial positions (x, y, z):")
    print(state.states[:, 3, :])

    # 目标位置：两架机各自飞到 y 方向正负 2 米处，高度保持 1 米，yaw=0
    target_setpoints = np.array(
        [
            [-1.0, 2.0, 0.0, 1.0],   # UAV 0: 向正 y 方向飞
            [1.0, -2.0, 0.0, 1.0],   # UAV 1: 向负 y 方向飞
        ],
        dtype=np.float32,
    )

    num_steps = 600
    print_interval = 30

    for step in range(num_steps):
        state = backend.step(target_setpoints)

        # 读取当前位置
        positions = state.states[:, 3, :]  # [N, 3], ground-frame position

        if step % print_interval == 0 or step == num_steps - 1:
            print(f"Step {step}: positions (x, y, z)")
            print(positions)

        # 适当 sleep，使可视化更友好（与 control_hz 大致对应）
        time.sleep(1.0 / backend.control_hz)

    backend.close()


if __name__ == "__main__":
    main()

