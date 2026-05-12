"""Genesis-backed 3v1 pursuit-evasion environment adapter."""

from __future__ import annotations

import time

import numpy as np
from gymnasium import spaces

from marl_uav.envs.base_env import BaseEnv
from marl_uav.envs.backends.genesis_backend import GenesisBackend
from marl_uav.envs.tasks.pursuit_evasion_3v1_task import (
    PursuitEvasion3v1Task,
    compute_pursuit_structure_metrics_3v1,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx1,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx2,
    PursuitEvasion3v1TaskEx2State,
)


PURSUIT_EVASION_3V1_TASK_TYPES = (
    PursuitEvasion3v1Task,
    PursuitEvasion3v1TaskEx1,
    PursuitEvasion3v1TaskEx2,
)


class GenesisPursuitEvasionEnv(BaseEnv):
    """Gymnasium-style MARL env that connects Genesis to existing 3v1 tasks."""

    def __init__(
        self,
        *,
        backend: GenesisBackend,
        task: PursuitEvasion3v1Task | PursuitEvasion3v1TaskEx1 | PursuitEvasion3v1TaskEx2,
        seed: int | None = None,
        action_space: str = "continuous",
        action_dim: int = 4,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.task_state = None
        self.prev_backend_state = None
        self.step_count = 0
        self._episode_return = 0.0
        self._episode_len = 0
        self._avail_actions_cache: list[np.ndarray] | None = None

        self._action_space_type = str(action_space).lower()
        if self._action_space_type not in ("discrete", "continuous"):
            raise ValueError(f"action_space must be 'discrete' or 'continuous', got {action_space!r}")

        self.num_agents = 3
        self.obs_dim: int | None = None
        self.state_dim: int | None = None

        if self._action_space_type == "discrete":
            self.n_actions = 7
            self.action_dim = None
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.n_actions = 0
            self.action_dim = int(action_dim)
            low = (
                np.full((self.action_dim,), -1.0, dtype=np.float32)
                if action_low is None
                else np.asarray(action_low, dtype=np.float32).reshape(-1)
            )
            high = (
                np.full((self.action_dim,), 1.0, dtype=np.float32)
                if action_high is None
                else np.asarray(action_high, dtype=np.float32).reshape(-1)
            )
            if low.size != self.action_dim or high.size != self.action_dim:
                raise ValueError(
                    f"action_low/high must match action_dim={self.action_dim}, got {low.size}/{high.size}"
                )
            self.action_space = spaces.Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)
            self.action_low_np = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
            self.action_high_np = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset task and Genesis backend, returning existing project obs/state format."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        start_pos, start_orn, self.task_state = self.task.sample_initial_conditions(
            self.backend.num_agents,
            self.rng,
        )
        backend_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        backend_state = self.backend.reset(start_pos, start_orn, seed=backend_seed)
        self.prev_backend_state = backend_state
        self.step_count = 0
        self._episode_return = 0.0
        self._episode_len = 0

        obs = self.task.build_obs(backend_state, self.task_state)
        state = self.task.build_state(backend_state, self.task_state)
        self.num_agents = int(obs.shape[0])
        self.obs_dim = int(obs.shape[1])
        self.state_dim = int(state.shape[0])
        avail = np.ones(self.action_dim if self._action_space_type == "continuous" else self.n_actions, dtype=np.float32)
        self._avail_actions_cache = [avail.copy() for _ in range(self.num_agents)]
        self._last_obs = obs
        self._last_state = state

        info = {"state": state}
        info.update(self._pursuit_info(backend_state, terminated=False, truncated=False, prev_captured=False))
        return {"obs": obs, "state": state}, info

    def step(self, actions):
        """Execute one MARL step through task setpoints and Genesis physics."""
        if self.prev_backend_state is None:
            raise RuntimeError("Environment has not been reset yet.")

        step_t0 = time.perf_counter()
        setpoints = self.task.action_to_setpoint(
            actions,
            self.prev_backend_state,
            self.task_state,
            action_space_type=self._action_space_type,
            action_dim=self.action_dim,
        )
        t_after_action = time.perf_counter()
        backend_state = self.backend.step(setpoints)
        t_after_backend = time.perf_counter()
        self.step_count += 1

        prev_captured = bool(getattr(self.task_state, "captured", False))
        rewards = self.task.compute_rewards(self.prev_backend_state, backend_state, self.task_state)
        self._episode_return += float(np.sum(rewards))
        self._episode_len += 1
        t_after_rewards = time.perf_counter()
        terminated, truncated = self.task.compute_terminated_truncated(
            backend_state,
            self.task_state,
            self.step_count,
        )
        t_after_done = time.perf_counter()

        obs = self.task.build_obs(backend_state, self.task_state)
        state = self.task.build_state(backend_state, self.task_state)
        t_after_obs_state = time.perf_counter()
        self._last_obs = obs
        self._last_state = state

        info = self._pursuit_info(
            backend_state,
            terminated=bool(terminated),
            truncated=bool(truncated),
            prev_captured=prev_captured,
        )
        t_after_info = time.perf_counter()
        info["timing"] = {
            "total_s": float(t_after_info - step_t0),
            "action_to_setpoint_s": float(t_after_action - step_t0),
            "backend_step_s": float(t_after_backend - t_after_action),
            "compute_rewards_s": float(t_after_rewards - t_after_backend),
            "compute_done_s": float(t_after_done - t_after_rewards),
            "build_obs_state_s": float(t_after_obs_state - t_after_done),
            "build_info_s": float(t_after_info - t_after_obs_state),
        }

        self.prev_backend_state = backend_state
        return {"obs": obs, "state": state}, rewards.tolist(), bool(terminated), bool(truncated), info

    def get_obs(self):
        """Return latest per-agent observations."""
        if not hasattr(self, "_last_obs"):
            raise RuntimeError("Env has not been reset yet.")
        return list(np.asarray(self._last_obs))

    def get_state(self):
        """Return latest global state."""
        if not hasattr(self, "_last_state"):
            raise RuntimeError("Env has not been reset yet.")
        return np.asarray(self._last_state, dtype=np.float32)

    def get_avail_actions(self):
        """All actions are available for both discrete and continuous heads."""
        if self._avail_actions_cache is None:
            dim = self.action_dim if self._action_space_type == "continuous" else self.n_actions
            avail = np.ones(dim, dtype=np.float32)
            self._avail_actions_cache = [avail.copy() for _ in range(self.num_agents)]
        return self._avail_actions_cache

    def close(self) -> None:
        self.backend.close()

    def set_training_progress(self, epoch: int, num_epochs: int):
        if hasattr(self.task, "set_training_progress"):
            return self.task.set_training_progress(epoch=epoch, num_epochs=num_epochs)
        return None

    def _pursuit_info(
        self,
        backend_state,
        *,
        terminated: bool,
        truncated: bool,
        prev_captured: bool,
    ) -> dict:
        """Build info keys compatible with the existing PyFlyt pursuit adapter."""
        state = self.task.build_state(backend_state, self.task_state)
        lin_pos = backend_state.states[:, 3, :]
        pursuer_pos = lin_pos[self.task_state.pursuer_ids]
        evader_pos = lin_pos[self.task_state.evader_id]
        dists = np.linalg.norm(pursuer_pos - evader_pos[None, :], axis=1).astype(np.float32)
        mean_goal_distance = float(np.mean(dists))

        captured = bool(getattr(self.task_state, "captured", False))
        newly_captured = bool(captured and not prev_captured)
        capture_step = int(self.step_count) if newly_captured else -1

        p_oob_mask = self.task._get_oob_mask(pursuer_pos)
        num_p_oob = int(np.sum(p_oob_mask))
        pursuer_oob = bool(num_p_oob >= 1)
        too_many_pursuers_oob = bool(
            num_p_oob >= int(getattr(self.task, "max_pursuers_oob_before_terminate", 1))
        )
        evader_oob = bool(self.task._get_oob_mask(evader_pos[None, :])[0])
        out_of_bounds = bool(
            np.any(np.abs(lin_pos[:, :2]) > getattr(self.task, "world_xy", 5.0) * 1.2)
            or np.any((lin_pos[:, 2] < 0.1) | (lin_pos[:, 2] > getattr(self.task, "z_max", 2.0) * 1.5))
        )
        has_collision = bool(np.any(backend_state.contact_array))
        crash = bool(np.any(lin_pos[:, 2] < 0.1))
        timeout = bool(truncated and not (captured or too_many_pursuers_oob or evader_oob))

        latest_struct = np.asarray(
            getattr(self.task_state, "latest_structure_metrics", None), dtype=np.float32
        ).reshape(-1)
        if latest_struct.shape[0] == 3:
            pursuit_structure = {
                "C_cov": float(latest_struct[0]),
                "C_col": float(latest_struct[1]),
                "D_ang": float(latest_struct[2]),
            }
        else:
            pursuit_structure = compute_pursuit_structure_metrics_3v1(pursuer_pos, evader_pos)

        pursuer_obstacle_hit = False
        obstacle_terminated = False
        if isinstance(self.task, PursuitEvasion3v1TaskEx2) and isinstance(
            self.task_state, PursuitEvasion3v1TaskEx2State
        ):
            hit_mask = self.task._pursuer_obstacle_collision_mask(pursuer_pos, self.task_state)
            pursuer_obstacle_hit = bool(np.any(hit_mask))
            obstacle_terminated = bool(terminated and pursuer_obstacle_hit and not newly_captured)

        is_success = bool(terminated and captured)
        termination_reason = "running"
        if terminated:
            if is_success:
                termination_reason = "capture"
            elif obstacle_terminated or has_collision:
                termination_reason = "collision"
            elif too_many_pursuers_oob:
                termination_reason = "pursuer_oob"
            elif evader_oob:
                termination_reason = "evader_oob"
            elif out_of_bounds:
                termination_reason = "out_of_bounds"
            else:
                termination_reason = "terminated"
        elif truncated:
            termination_reason = "timeout" if not is_success else "truncated_success"

        info = {
            "state": state,
            "all_reached": captured,
            "is_success": is_success,
            "out_of_bounds": out_of_bounds,
            "has_collision": has_collision,
            "crash": crash,
            "mean_goal_distance": mean_goal_distance,
            "final_goal_distance": mean_goal_distance,
            "reward_progress": 0.0,
            "reward_time_penalty": -float(getattr(self.task, "time_penalty", 0.0)) * self.num_agents,
            "reward_reach_bonus": 0.0,
            "reward_collision_penalty": -float(self.num_agents) if has_collision else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": termination_reason,
            "episode_return": float(self._episode_return),
            "episode_len": int(self._episode_len),
            "capture": captured,
            "oob": bool(pursuer_oob or evader_oob or out_of_bounds),
            "captured": captured,
            "newly_captured": newly_captured,
            "capture_step": capture_step,
            "pursuer_oob": pursuer_oob,
            "too_many_pursuers_oob": too_many_pursuers_oob,
            "evader_oob": evader_oob,
            "timeout": timeout,
            "pursuit_structure": pursuit_structure,
            "metrics": {"pursuit_structure": pursuit_structure},
            "pursuer_obstacle_hit": pursuer_obstacle_hit,
            "obstacle_terminated": obstacle_terminated,
        }
        info.update(self._build_reference_manifold_info(backend_state))
        if isinstance(self.task, PursuitEvasion3v1TaskEx2) and isinstance(
            self.task_state, PursuitEvasion3v1TaskEx2State
        ):
            info["obstacle_xy"] = np.asarray(self.task_state.obstacle_xy, dtype=np.float32).copy()
            info["obstacle_r"] = np.asarray(self.task_state.obstacle_r, dtype=np.float32).copy()
        return info

    def _build_reference_manifold_info(self, backend_state) -> dict[str, np.ndarray]:
        if not (
            hasattr(self.task, "_reference_manifold_targets")
            and hasattr(self.task, "_reference_manifold_curve")
        ):
            return {}
        lin_pos = backend_state.states[:, 3, :]
        pursuer_pos = np.asarray(lin_pos[self.task_state.pursuer_ids], dtype=np.float32)
        evader_pos = np.asarray(lin_pos[self.task_state.evader_id], dtype=np.float32)
        targets = self.task._reference_manifold_targets(
            pursuer_pos,
            evader_pos,
            task_state=self.task_state,
        ).astype(np.float32)
        curve = self.task._reference_manifold_curve(
            pursuer_pos,
            evader_pos,
            task_state=self.task_state,
        ).astype(np.float32)
        return {
            "reference_manifold_targets": targets,
            "reference_manifold_curve": curve,
        }
