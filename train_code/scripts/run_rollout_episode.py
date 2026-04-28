"""跑通一条完整 episode 收集流程：ToyUavEnv + RandomPolicy + RolloutWorker。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.policies.random_policy import RandomPolicy
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.config import load_config


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

    buffer, info = worker.collect_episode(seed=123)
    print("=== Episode 收集完成 ===")
    print("episode_return:", info["episode_return"])
    print("episode_len:  ", info["episode_len"])
    print("terminated:   ", info["terminated"])
    print("truncated:    ", info["truncated"])

    batch = buffer.get_episode()
    print("\n=== Batch 形状 ===")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} dtype={v.dtype}")

    print("\nOK: 一条完整 episode 收集流程已跑通。")


if __name__ == "__main__":
    main()
