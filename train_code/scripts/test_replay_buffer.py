"""使用 RandomPolicy 收集经验并测试 ReplayBuffer：add_episode + sample。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.policies.random_policy import RandomPolicy
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.buffers.replay_buffer import ReplayBuffer
from marl_uav.utils.config import load_config
import numpy as np

def main():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "env" / "toy_uav.yaml")
    env = ToyUavEnv.from_config(config, seed=42)

    policy = RandomPolicy(
        num_agents=env.num_agents,
        n_actions=env.n_actions,
        seed=42,
    )
    worker = RolloutWorker(env, policy)

    # ReplayBuffer 参数与 env 一致
    capacity = 5000
    replay = ReplayBuffer(
        capacity=capacity,
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
    )

    # 用 RandomPolicy 收集多条 episode 并写入 replay
    num_episodes = 5
    total_steps = 0
    for i in range(num_episodes):
        buffer, info = worker.collect_episode(seed=42 + i)
        replay.add_episode(buffer)
        total_steps += info["episode_len"]
        print(f"  episode {i+1}: len={info['episode_len']}, return={info['episode_return']:.2f}")

    print(f"\n=== ReplayBuffer 状态 ===")
    print(f"  len(replay) = {len(replay)}")
    print(f"  total steps collected = {total_steps}")

    # 随机采样
    batch_size = 32
    batch = replay.sample(batch_size)
    if batch is None:
        print("  sample 失败：buffer 中样本不足 batch_size")
        return

    print(f"\n=== Sample(batch_size={batch_size}) 形状 ===")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} dtype={v.dtype}")

    # 再采样一次确保可重复采样
    batch2 = replay.sample(batch_size)
    print(f"\n  二次采样 OK，obs 与第一次不同: {not np.allclose(batch['obs'], batch2['obs'])}")

    print("\nOK: ReplayBuffer 已跑通（RandomPolicy 收集 + add_episode + sample）。")


if __name__ == "__main__":
    main()
