"""Thin adapter around PyFlyt env for slot execution MAPPO."""

from __future__ import annotations

from typing import Any

import numpy as np

from slot_exec_mappo.config import SlotExecConfig
from slot_exec_mappo.obs import build_global_state, build_local_obs, global_state_dim, local_obs_dim
from slot_exec_mappo.policy import SlotExecPolicyBundle
from slot_exec_mappo.reward import ExecRewardState, compute_exec_rewards


class SlotExecEnvWrapper:
    """Wrap pursuit env: execution obs/state + slot navigation rewards."""

    def __init__(self, env: Any, *, cfg: SlotExecConfig | None = None) -> None:
        self.env = env
        self.cfg = cfg or SlotExecConfig()
        self._rw_state = ExecRewardState()
        self._last_exec_obs: np.ndarray | None = None
        self._last_exec_state: np.ndarray | None = None
        self._last_actions = np.zeros((3, 4), dtype=np.float32)

        self.obs_dim = local_obs_dim(self.cfg.obs)
        self.state_dim = global_state_dim(self.cfg.obs)
        task = getattr(env, "task", None)
        self.num_agents = int(getattr(task, "num_pursuers", 3) or 3)
        self.action_dim = int(getattr(env, "action_dim", 4) or 4)
        self.action_low_np = np.asarray(getattr(env, "action_low_np", -0.25), dtype=np.float32)
        self.action_high_np = np.asarray(getattr(env, "action_high_np", 0.25), dtype=np.float32)
        self._action_space_type = getattr(env, "_action_space_type", "continuous")

    def _sync_exec_buffers(self) -> None:
        self._last_exec_obs = build_local_obs(
            self.env,
            cfg=self.cfg.obs,
            prev_actions=self._rw_state.prev_actions,
        )
        self._last_exec_state = build_global_state(
            self.env,
            cfg=self.cfg.obs,
            slot_dists=self._rw_state.prev_slot_dists,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        out = self.env.reset(seed=seed, options=options)
        self._rw_state.reset()
        self._last_actions = np.zeros((self.num_agents, 4), dtype=np.float32)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict):
            info = dict(out[1])
        else:
            info = {}
        self._sync_exec_buffers()
        return self._last_exec_obs.copy(), info

    def step(self, actions: np.ndarray):
        inner_out = self.env.step(actions)
        if isinstance(inner_out, tuple) and len(inner_out) == 5:
            _payload, _task_rewards, terminated, truncated, info = inner_out
        else:
            raise RuntimeError("Unsupported env.step return format")

        exec_rewards, rw_diag = compute_exec_rewards(
            self.env,
            actions,
            cfg=self.cfg.reward,
            rw_state=self._rw_state,
            step_info=info if isinstance(info, dict) else None,
        )
        self._last_actions = np.asarray(actions, dtype=np.float32).reshape(self.num_agents, -1)
        self._sync_exec_buffers()

        if isinstance(info, dict):
            info = dict(info)
            info["slot_exec_reward"] = rw_diag
            info["rewards"] = exec_rewards.astype(float).tolist()

        if bool(self.cfg.reward.success_all_agents):
            dists = np.asarray(rw_diag.get("slot_dist_xy", [1.0, 1.0, 1.0]), dtype=np.float32)
            if bool(np.all(dists < float(self.cfg.reward.arrive_dist))):
                terminated = True
                if isinstance(info, dict):
                    info["termination_reason"] = "slot_success"

        return self._last_exec_obs.copy(), exec_rewards, bool(terminated), bool(truncated), info

    def get_obs(self) -> np.ndarray:
        if self._last_exec_obs is None:
            raise RuntimeError("Env has not been reset yet.")
        return np.asarray(self._last_exec_obs, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        if self._last_exec_state is None:
            raise RuntimeError("Env has not been reset yet.")
        return np.asarray(self._last_exec_state, dtype=np.float32)

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def make_slot_exec_get_actions_fn(
    env: Any,
    *,
    checkpoint: str | None = None,
    policy: SlotExecPolicyBundle | None = None,
    cfg: SlotExecConfig | None = None,
    deterministic: bool = True,
    device: str = "cpu",
):
    """RolloutWorker-compatible action fn using standalone slot_exec policy."""
    raw_env = env.env if isinstance(env, SlotExecEnvWrapper) else env
    slot_cfg = cfg or SlotExecConfig()
    if policy is None:
        if not checkpoint:
            raise ValueError("checkpoint or policy is required for slot_exec_mappo inference")
        policy = SlotExecPolicyBundle.load(
            checkpoint,
            action_low=np.asarray(raw_env.action_low_np, dtype=np.float32),
            action_high=np.asarray(raw_env.action_high_np, dtype=np.float32),
            hidden_sizes=slot_cfg.train.hidden_sizes,
            device=device,
        )

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if raw_env.prev_backend_state is None or raw_env.task_state is None:
            raise RuntimeError("Environment must be reset before selecting slot_exec_mappo actions.")
        exec_obs = build_local_obs(
            raw_env,
            cfg=slot_cfg.obs,
            prev_actions=getattr(raw_env.task_state, "prev_pursuer_actions", None),
        )
        actions, _logp, _vals = policy.act(exec_obs, deterministic=deterministic)
        return actions.astype(np.float32)

    return get_actions
