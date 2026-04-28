"""Test environment script."""
import sys
from pathlib import Path

# 保证项目根在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.utils.config import load_config


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "env" / "toy_uav.yaml"
    config = load_config(config_path)
    env = ToyUavEnv.from_config(config, seed=42)

    obs, info = env.reset(seed=42)
    print("obs list length:", len(obs["obs"]))
    print("obs[0].shape:", obs["obs"][0].shape)
    print("state.shape:", obs["state"].shape)
    print("avail_actions:", [a.shape for a in env.get_avail_actions()])

    actions = [env.action_space.sample() for _ in range(env.num_agents)]
    next_obs, rewards, term, trunc, info = env.step(actions)
    print("rewards:", rewards)
    print("terminated:", term, "truncated:", trunc)
    print("OK: ToyUavEnv runs.")


if __name__ == "__main__":
    main()
