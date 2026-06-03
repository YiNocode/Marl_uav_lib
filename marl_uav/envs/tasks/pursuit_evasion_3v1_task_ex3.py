"""ex3: ex2 obstacles + optional random spawn + pre-planned evader sharp-turn path."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from marl_uav.control.altitude_hold import hard_altitude_hold
from marl_uav.envs.tasks.evader_trajectory_planner import (
    path_hold_altitude,
    plan_sharp_turn_evader_path,
    select_path_tracking_target,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx2Base,
    PursuitEvasion3v1TaskEx2State,
)


@dataclass
class PursuitEvasion3v1TaskEx3State(PursuitEvasion3v1TaskEx2State):
    """ex2 state + optional pre-planned evader trajectory."""

    evader_planned_path: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    evader_path_cursor: int = 0


class PursuitEvasion3v1Task(PursuitEvasion3v1TaskEx2Base):
    """
    ex2 基础上可选：

    - ``evader_sharp_turn_enabled``：reset 前为 evader 规划障碍感知、含急转的折线路径；
    - ``init_random_bias_enabled``：pursuer / evader 在场景内随机初始化（仍满足避障与间距）。

    两开关可独立或同时开启；控制算法与 env 结构不变。
    """

    def __init__(
        self,
        *args,
        evader_sharp_turn_enabled: bool = False,
        init_random_bias_enabled: bool = False,
        evader_planned_path_num_legs: int = 4,
        evader_planned_path_min_leg_m: float = 8.0,
        evader_planned_path_min_turn_deg: float = 60.0,
        evader_planned_path_lookahead_m: float = 1.5,
        evader_planned_path_accept_radius: float = 0.35,
        init_random_bias_margin_ratio: float = 0.12,
        init_random_bias_min_pursuer_sep_scale: float = 1.5,
        evader_path_plan_safety_margin: float = 0.35,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evader_sharp_turn_enabled = bool(evader_sharp_turn_enabled)
        self.init_random_bias_enabled = bool(init_random_bias_enabled)
        self.evader_planned_path_num_legs = max(int(evader_planned_path_num_legs), 1)
        self.evader_planned_path_min_leg_m = max(float(evader_planned_path_min_leg_m), 1.0)
        self.evader_planned_path_min_turn_deg = float(evader_planned_path_min_turn_deg)
        self.evader_planned_path_lookahead_m = max(float(evader_planned_path_lookahead_m), 0.2)
        self.evader_planned_path_accept_radius = max(float(evader_planned_path_accept_radius), 0.05)
        self.init_random_bias_margin_ratio = float(init_random_bias_margin_ratio)
        self.init_random_bias_min_pursuer_sep_scale = max(float(init_random_bias_min_pursuer_sep_scale), 1.0)
        self.evader_path_plan_safety_margin = max(float(evader_path_plan_safety_margin), 0.05)

    def _ex3_state_from_ex2(
        self,
        state: PursuitEvasion3v1TaskEx2State,
        *,
        planned_path: np.ndarray | None = None,
    ) -> PursuitEvasion3v1TaskEx3State:
        data = {f.name: getattr(state, f.name) for f in fields(PursuitEvasion3v1TaskEx2State)}
        path = np.zeros((0, 3), dtype=np.float32) if planned_path is None else np.asarray(planned_path, dtype=np.float32)
        return PursuitEvasion3v1TaskEx3State(
            **data,
            evader_planned_path=path,
            evader_path_cursor=0,
        )

    def _plan_evader_path(
        self,
        start_xyz: np.ndarray,
        pursuer_xyz: np.ndarray,
        obstacle_xy: np.ndarray,
        obstacle_r: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        z_lo, z_hi = self._random_bias_flight_z_bounds()
        return plan_sharp_turn_evader_path(
            start_xyz,
            pursuer_xyz,
            obstacle_xy,
            obstacle_r,
            world_xy=self.world_xy,
            rng=rng,
            num_legs=self.evader_planned_path_num_legs,
            min_leg_m=self.evader_planned_path_min_leg_m,
            min_turn_deg=self.evader_planned_path_min_turn_deg,
            safety_margin=self.evader_path_plan_safety_margin,
            uav_radius=self._pursuer_obstacle_hit_radius(),
            arena_margin_ratio=self.init_random_bias_margin_ratio,
            z_min=z_lo,
            z_max=z_hi,
        )

    def _random_bias_flight_z_bounds(self) -> tuple[float, float]:
        """Same narrow flight band as ex2 (not the full arena z span)."""
        return self._initial_flight_z_bounds()

    def _sample_random_bias_initial_conditions(
        self,
        num_agents: int,
        rng: np.random.Generator,
    ):
        assert num_agents == 4
        pursuer_ids = np.array([0, 1, 2], dtype=np.int64)
        evader_id = 3
        start_pos = np.zeros((num_agents, 3), dtype=np.float32)
        start_orn = np.zeros((num_agents, 3), dtype=np.float32)

        margin_xy = max(self.init_random_bias_margin_ratio * self.world_xy, 0.05)
        init_z_low, init_z_high = self._random_bias_flight_z_bounds()
        clear = self._init_obstacle_clearance()
        min_sep = self.init_random_bias_min_pursuer_sep_scale * self.min_pursuer_sep
        min_mean_dist = self.init_mean_dist_range_ratio[0] * self.world_xy
        max_mean_dist = self.init_mean_dist_range_ratio[1] * self.world_xy
        z_noise = float(self.init_pursuer_noise_z)

        outer_tries = max(self.max_init_resample, 50)
        for _ in range(outer_tries):
            obstacle_xy, obstacle_r = self._sample_obstacles(rng)
            if obstacle_xy.shape[0] < self.num_obstacles_min:
                continue

            evader_xy = rng.uniform(-self.world_xy + margin_xy, self.world_xy - margin_xy, size=2)
            evader_z = float(rng.uniform(init_z_low, init_z_high))
            evader_pos = np.array([evader_xy[0], evader_xy[1], evader_z], dtype=np.float32)
            if not bool(self._xy_clear_of_obstacles(evader_pos[:2].reshape(1, 2), obstacle_xy, obstacle_r, clear)[0]):
                continue

            ok = True
            for i in range(3):
                placed = False
                for _trial in range(self.max_init_resample):
                    xy = rng.uniform(-self.world_xy + margin_xy, self.world_xy - margin_xy, size=2)
                    if not bool(self._xy_clear_of_obstacles(xy.reshape(1, 2), obstacle_xy, obstacle_r, clear)[0]):
                        continue
                    if i > 0 and np.any(
                        np.linalg.norm(start_pos[pursuer_ids[:i], :2] - xy.reshape(1, 2), axis=1) < min_sep
                    ):
                        continue
                    pz = float(np.clip(
                        evader_z + rng.uniform(-z_noise, z_noise),
                        init_z_low,
                        init_z_high,
                    ))
                    cand = np.array([xy[0], xy[1], pz], dtype=np.float32)
                    dists = np.linalg.norm(start_pos[pursuer_ids[:i]] - cand[None, :], axis=1) if i > 0 else np.array([])
                    if i > 0 and np.any(dists < min_sep):
                        continue
                    dist_ev = float(np.linalg.norm(cand - evader_pos))
                    if dist_ev < 1.5 * self.capture_dist:
                        continue
                    start_pos[pursuer_ids[i]] = cand
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok:
                continue

            init_dists = np.linalg.norm(start_pos[pursuer_ids] - evader_pos[None, :], axis=1)
            mean_dist = float(np.mean(init_dists))
            if not (min_mean_dist <= mean_dist <= max_mean_dist):
                continue

            start_pos[evader_id] = evader_pos
            start_pos[pursuer_ids] = self._clip_positions_inside(
                start_pos[pursuer_ids], margin_xy=0.02, margin_z=0.02,
            )
            start_pos[evader_id] = self._clip_positions_inside(
                start_pos[evader_id].reshape(1, 3), margin_xy=0.02, margin_z=0.02,
            ).reshape(3)

            init_dists = np.linalg.norm(
                start_pos[pursuer_ids] - start_pos[evader_id][None, :],
                axis=1,
            ).astype(np.float32)
            state = PursuitEvasion3v1TaskEx3State(
                pursuer_ids=pursuer_ids,
                evader_id=evader_id,
                captured=bool(np.any(init_dists <= self.capture_dist)),
                capture_agent=int(pursuer_ids[np.argmin(init_dists)]) if np.any(init_dists <= self.capture_dist) else -1,
                prev_pursuer_dists=init_dists.copy(),
                obstacle_xy=obstacle_xy,
                obstacle_r=obstacle_r,
            )
            return start_pos, start_orn, state

        return super().sample_initial_conditions(num_agents, rng)

    def sample_initial_conditions(self, num_agents: int, rng: np.random.Generator):
        if self.init_random_bias_enabled:
            start_pos, start_orn, state = self._sample_random_bias_initial_conditions(num_agents, rng)
            if not isinstance(state, PursuitEvasion3v1TaskEx3State):
                state = self._ex3_state_from_ex2(state)
        else:
            start_pos, start_orn, state2 = super().sample_initial_conditions(num_agents, rng)
            state = self._ex3_state_from_ex2(state2)

        if self.evader_sharp_turn_enabled:
            evader_id = int(state.evader_id)
            pursuer_ids = np.asarray(state.pursuer_ids, dtype=np.int64).reshape(3)
            planned = self._plan_evader_path(
                start_pos[evader_id],
                start_pos[pursuer_ids],
                state.obstacle_xy,
                state.obstacle_r,
                rng,
            )
            state.evader_planned_path = planned
            state.evader_path_cursor = 0

        return start_pos, start_orn, state

    def _compute_evader_setpoint(self, backend_state, task_state):
        if (
            self.evader_sharp_turn_enabled
            and isinstance(task_state, PursuitEvasion3v1TaskEx3State)
            and task_state.evader_planned_path.shape[0] >= 2
        ):
            lin_pos = backend_state.states[:, 3, :]
            evader_pos = np.asarray(lin_pos[int(task_state.evader_id)], dtype=np.float32).reshape(3)
            path = task_state.evader_planned_path
            if int(task_state.evader_path_cursor) >= path.shape[0] - 1:
                dist_end = float(np.linalg.norm(evader_pos[:2] - path[-1, :2]))
                if dist_end <= self.evader_planned_path_accept_radius:
                    return super()._compute_evader_setpoint(backend_state, task_state)

            target, cursor = select_path_tracking_target(
                evader_pos,
                path,
                int(task_state.evader_path_cursor),
                lookahead_m=self.evader_planned_path_lookahead_m,
                accept_radius=self.evader_planned_path_accept_radius,
            )
            task_state.evader_path_cursor = int(cursor)
            hold_z = path_hold_altitude(path)
            move_xy = np.asarray(target[:2], dtype=np.float64) - np.asarray(evader_pos[:2], dtype=np.float64)
            move_dir_xy = self._safe_normalize(move_xy.astype(np.float32))
            if float(np.linalg.norm(move_dir_xy)) < 1e-8:
                fallback = np.asarray(target[:2], dtype=np.float32) - np.asarray(path[0, :2], dtype=np.float32)
                move_dir_xy = self._safe_normalize(np.array([fallback[0], fallback[1], 0.0], dtype=np.float32))
            low_z = -self.evader_speed_z
            high_z = self.evader_speed_z
            vz, gate = hard_altitude_hold(
                float(evader_pos[2]),
                hold_z,
                low_z,
                high_z,
            )
            return np.array(
                [
                    move_dir_xy[0] * self.evader_speed_xy * gate,
                    move_dir_xy[1] * self.evader_speed_xy * gate,
                    0.0,
                    vz,
                ],
                dtype=np.float32,
            )
        return super()._compute_evader_setpoint(backend_state, task_state)
