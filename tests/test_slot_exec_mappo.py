"""Tests for standalone slot execution MAPPO."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from slot_exec_mappo.adapter import SlotExecEnvWrapper
from slot_exec_mappo.config import ExecObsConfig, ExecRewardConfig, SlotExecConfig
from slot_exec_mappo.obs import build_local_obs, global_state_dim, local_obs_dim
from slot_exec_mappo.reward import ExecRewardState, compute_exec_rewards


class _FakeTask:
    world_xy = 20.0
    z_min = 0.0
    z_max = 5.0
    pursuer_speed_xy = 0.25
    pursuer_speed_z = 0.15
    episode_limit = 400
    evader_margin_xy_ratio = 0.25

    def _assigned_targets_from_state(self, pursuer_pos, evader_pos, task_state=None):
        del evader_pos, task_state
        targets = pursuer_pos + np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        assignment = np.array([0, 1, 2], dtype=np.int64)
        return targets, assignment, targets.copy()

    def _pursuer_obstacle_obs_block(self, pursuer_xy, obstacle_xy, obstacle_r):
        del pursuer_xy, obstacle_xy, obstacle_r
        return np.zeros((16,), dtype=np.float32)

    def _global_obstacle_state_block(self, obstacle_xy, obstacle_r):
        del obstacle_xy, obstacle_r
        return np.zeros((16,), dtype=np.float32)

    def _get_oob_mask(self, pursuer_pos):
        del pursuer_pos
        return np.zeros((3,), dtype=bool)

    def _pursuer_obstacle_collision_mask(self, pursuer_pos, task_state):
        del pursuer_pos, task_state
        return np.zeros((3,), dtype=bool)

    def _pursuer_obstacle_hit_radius(self):
        return 0.15


def _fake_env():
    backend = SimpleNamespace(
        states=np.zeros((4, 4, 3), dtype=np.float32),
    )
    backend.states[:, 3, :] = np.array(
        [[-2.0, -1.0, 1.0], [-2.0, 0.0, 1.0], [-2.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    task_state = SimpleNamespace(
        pursuer_ids=np.array([0, 1, 2], dtype=np.int64),
        evader_id=3,
        assigned_target_indices=np.array([0, 1, 2], dtype=np.int64),
        obstacle_xy=np.array([[1.0, 0.0]], dtype=np.float32),
        obstacle_r=np.array([0.5], dtype=np.float32),
    )
    env = SimpleNamespace(
        task=_FakeTask(),
        task_state=task_state,
        prev_backend_state=backend,
        step_count=1,
        num_agents=3,
        action_dim=4,
        action_low_np=np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32),
        action_high_np=np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32),
        _action_space_type="continuous",
    )
    return env


def test_local_and_global_obs_dims() -> None:
    cfg = ExecObsConfig(obstacle_slots=4, include_prev_action=True)
    assert local_obs_dim(cfg) == 9 + 7 + 16 + 4 + 4
    assert global_state_dim(cfg) == 21 + 9 + 16 + 3
    env = _fake_env()
    obs = build_local_obs(env, cfg=cfg)
    assert obs.shape == (3, local_obs_dim(cfg))


def test_exec_reward_progress_and_collision() -> None:
    env = _fake_env()
    rw = ExecRewardState()
    cfg = ExecRewardConfig()
    actions = np.zeros((3, 4), dtype=np.float32)
    rewards, diag = compute_exec_rewards(env, actions, cfg=cfg, rw_state=rw)
    assert rewards.shape == (3,)
    assert "slot_dist_xy" in diag

    env.task._pursuer_obstacle_collision_mask = lambda p, ts: np.array([True, False, False])
    rewards_hit, _diag = compute_exec_rewards(env, actions, cfg=cfg, rw_state=rw)
    assert float(rewards_hit[0]) < float(rewards[0])


class _InnerEnv:
    def __init__(self) -> None:
        self.task = _FakeTask()
        self.task_state = None
        self.prev_backend_state = None
        self.step_count = 0
        self.num_agents = 3
        self.action_dim = 4
        self.action_low_np = np.array([-0.25, -0.25, -0.25, -0.15], dtype=np.float32)
        self.action_high_np = np.array([0.25, 0.25, 0.25, 0.15], dtype=np.float32)
        self._action_space_type = "continuous"

    def reset(self, seed=None, options=None):
        del seed, options
        self.step_count = 0
        self.prev_backend_state = _fake_env().prev_backend_state
        self.task_state = _fake_env().task_state
        obs = np.zeros((3, 10), dtype=np.float32)
        state = np.zeros((20,), dtype=np.float32)
        return {"obs": obs, "state": state}, {"event": "reset"}

    def step(self, actions):
        del actions
        self.step_count += 1
        obs = np.zeros((3, 10), dtype=np.float32)
        state = np.zeros((20,), dtype=np.float32)
        return {"obs": obs, "state": state}, [0.0, 0.0, 0.0], False, False, {"termination_reason": "running"}

    def get_avail_actions(self):
        return [np.ones(4, dtype=np.float32) for _ in range(3)]

    def close(self):
        return None


def test_wrapper_reset_step_exposes_exec_obs() -> None:
    wrapped = SlotExecEnvWrapper(_InnerEnv(), cfg=SlotExecConfig())
    obs, info = wrapped.reset()
    assert obs.shape[0] == 3
    assert obs.shape[1] == wrapped.obs_dim
    assert isinstance(info, dict)

    obs2, rewards, terminated, truncated, info2 = wrapped.step(np.zeros((3, 4), dtype=np.float32))
    assert obs2.shape == obs.shape
    assert rewards.shape == (3,)
    assert terminated is False
    assert truncated is False
    assert "slot_exec_reward" in info2


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="PyTorch not installed",
)
def test_policy_bundle_act_shape() -> None:
    pytest.importorskip("torch")
    from slot_exec_mappo.policy import SlotExecPolicyBundle

    bundle = SlotExecPolicyBundle(obs_dim=53, state_dim=55, action_dim=4, device="cpu")
    obs = np.zeros((3, 53), dtype=np.float32)
    actions, logp, vals = bundle.act(obs, deterministic=True)
    assert actions.shape == (3, 4)
    assert logp is None
    v = bundle.value(np.zeros((55,), dtype=np.float32), obs[0])
    assert v.shape == (1,)
