"""Run a minimal Genesis 3v1 environment smoke test without training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marl_uav.envs.factories import build_env_from_config
from marl_uav.utils.config import load_config
from marl_uav.utils.stdio import configure_utf8_stdio


configure_utf8_stdio()


def main() -> int:
    try:
        import genesis  # noqa: F401
    except ImportError:
        print("SKIP: Genesis is not installed.")
        return 0

    train_cfg = load_config(ROOT / "configs/train/genesis_3v1.yaml")
    env_cfg_path = ROOT / train_cfg["env"]
    task_cfg = dict(train_cfg.get("task", {}))
    env = build_env_from_config(env_cfg_path, seed=int(train_cfg.get("seed", 0)), task_cfg=task_cfg)
    try:
        obs_dict, info = env.reset(seed=int(train_cfg.get("seed", 0)))
        print(
            "reset:",
            "obs_shape=", np.asarray(obs_dict["obs"]).shape,
            "state_shape=", np.asarray(obs_dict["state"]).shape,
            "info_keys=", sorted(info.keys())[:8],
        )
        rng = np.random.default_rng(0)
        low = np.broadcast_to(env.action_low_np, (env.num_agents, env.action_dim))
        high = np.broadcast_to(env.action_high_np, (env.num_agents, env.action_dim))
        for step in range(10000):
            actions = rng.uniform(low, high).astype(np.float32)
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            obs = np.asarray(obs_dict["obs"], dtype=np.float32)
            state = np.asarray(obs_dict["state"], dtype=np.float32)
            reward_arr = np.asarray(reward, dtype=np.float32)
            if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(state)) and np.all(np.isfinite(reward_arr))):
                raise RuntimeError(f"Non-finite value detected at step {step}.")
            print(
                f"step={step + 1} reward_sum={float(np.sum(reward_arr)):.3f} "
                f"terminated={bool(terminated)} truncated={bool(truncated)} "
                f"capture={bool(info.get('capture', False))} oob={bool(info.get('oob', False))}"
            )
            # if terminated or truncated:
            #     break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
