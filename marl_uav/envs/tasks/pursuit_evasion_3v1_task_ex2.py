from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx1Base,
    PursuitEvasion3v1TaskState,
)


@dataclass
class PursuitEvasion3v1TaskEx2State(PursuitEvasion3v1TaskState):
    """ex1 状态 + 本回合随机生成的圆柱障碍物（竖直轴平行 z，贯穿飞行高度带）。"""

    obstacle_xy: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )  # [K, 2]
    obstacle_r: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )  # [K]


class PursuitEvasion3v1Task(PursuitEvasion3v1TaskEx1Base):
    """
    在 ex1（结构感知观测、自适应尺度等）基础上：
    - 每回合在场地内随机生成 num_obstacles_min~num_obstacles_max 根圆柱障碍物；
    - 圆柱在 xy 上为圆域，z 方向覆盖 [z_min, z_max]（与有效飞行高度一致）；
    - 任一架 pursuer 机体中心在 xy 上进入「柱半径 + pursuer_obstacle_hit_radius」内即视为碰撞，
      本步起 terminated=True（回合终结）。

    观测 / 全局状态在 ex1 基础上增加固定维度的柱特征（槽数 = num_obstacles_max）：
    - build_obs：每架 pursuer 后接 4 * num_obstacles_max 维；各槽按「与该机水平距离」升序取最近柱，
      每槽 [Δx, Δy, r, valid]（xy 相对柱心、除以 pos_xy_norm；r 同尺度；valid∈{0,1}）。
    - build_state：在 ex1 全局向量末尾接 4 * num_obstacles_max 维；各槽按柱心 (cx, cy) 字典序排列，
      每槽 [cx, cy, r, valid]（cx, cy, r 均除以 pos_xy_norm）。
    """

    def __init__(
        self,
        *args,
        num_obstacles_min: int = 5,
        num_obstacles_max: int = 10,
        obstacle_radius_min_ratio: float = 0.012,
        obstacle_radius_max_ratio: float = 0.045,
        obstacle_min_gap_xy_ratio: float = 0.02,
        arena_edge_margin_ratio: float = 0.08,
        pursuer_obstacle_hit_radius_ratio: float = 0.015,
        init_clearance_extra_ratio: float = 0.02,
        obstacle_collision_penalty: float = 15.0,
        max_obstacle_place_trials: int = 8000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_obstacles_min = int(max(1, num_obstacles_min))
        self.num_obstacles_max = int(max(self.num_obstacles_min, num_obstacles_max))
        self.obstacle_radius_min_ratio = float(obstacle_radius_min_ratio)
        self.obstacle_radius_max_ratio = float(obstacle_radius_max_ratio)
        self.obstacle_min_gap_xy_ratio = float(obstacle_min_gap_xy_ratio)
        self.arena_edge_margin_ratio = float(arena_edge_margin_ratio)
        self.pursuer_obstacle_hit_radius_ratio = float(pursuer_obstacle_hit_radius_ratio)
        self.init_clearance_extra_ratio = float(init_clearance_extra_ratio)
        self.obstacle_collision_penalty = float(obstacle_collision_penalty)
        self.max_obstacle_place_trials = int(max_obstacle_place_trials)
        # 与 ex1 的 pos_xy_norm 一致，用于柱坐标 / 半径归一化
        self._obstacle_feature_slots = int(self.num_obstacles_max)
        self.obstacle_obs_extra_dim = 4 * self._obstacle_feature_slots

    def _obstacle_r_min_max(self) -> tuple[float, float]:
        lo = self.obstacle_radius_min_ratio * self.world_xy
        hi = self.obstacle_radius_max_ratio * self.world_xy
        if lo > hi:
            lo, hi = hi, lo
        return float(lo), float(hi)

    def _pursuer_obstacle_hit_radius(self) -> float:
        return max(1e-3, self.pursuer_obstacle_hit_radius_ratio * self.world_xy)

    def _init_obstacle_clearance(self) -> float:
        return self._pursuer_obstacle_hit_radius() + self.init_clearance_extra_ratio * self.world_xy

    def _obstacle_min_gap_xy(self) -> float:
        return max(1e-3, self.obstacle_min_gap_xy_ratio * self.world_xy)

    def _arena_inner_margin(self) -> float:
        return max(0.05, self.arena_edge_margin_ratio * self.world_xy)

    def _sample_obstacles(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        k = int(rng.integers(self.num_obstacles_min, self.num_obstacles_max + 1))
        r_lo, r_hi = self._obstacle_r_min_max()
        min_gap = self._obstacle_min_gap_xy()
        edge_m = self._arena_inner_margin()

        centers: list[list[float]] = []
        radii: list[float] = []
        trials = 0
        while len(centers) < k and trials < self.max_obstacle_place_trials:
            trials += 1
            r = float(rng.uniform(r_lo, r_hi))
            x_max = self.world_xy - edge_m - r
            x_min = -self.world_xy + edge_m + r
            y_max = self.world_xy - edge_m - r
            y_min = -self.world_xy + edge_m + r
            if x_min >= x_max or y_min >= y_max:
                break
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))
            ok = True
            for (cx, cy), cr in zip(centers, radii):
                if np.hypot(x - cx, y - cy) < r + cr + min_gap:
                    ok = False
                    break
            if ok:
                centers.append([x, y])
                radii.append(r)

        if not centers:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        return (
            np.asarray(centers, dtype=np.float32),
            np.asarray(radii, dtype=np.float32),
        )

    def _xy_clear_of_obstacles(
        self,
        xy: np.ndarray,
        obstacle_xy: np.ndarray,
        obstacle_r: np.ndarray,
        clearance: float,
    ) -> np.ndarray:
        """xy: [M, 2]，返回 shape [M] 的 bool，True 表示该点距所有柱面足够远。"""
        if obstacle_xy.size == 0:
            return np.ones((xy.shape[0],), dtype=bool)
        d = np.linalg.norm(xy[:, None, :] - obstacle_xy[None, :, :], axis=2)
        thresh = obstacle_r[None, :] + float(clearance)
        return np.all(d > thresh, axis=1)

    def _sample_valid_evader_position(
        self,
        pursuer_pos: np.ndarray,
        rng: np.random.Generator,
        obstacle_xy: np.ndarray,
        obstacle_r: np.ndarray,
    ) -> np.ndarray | None:
        """与 ex1 相同分布，但要求 evader xy 远离圆柱；失败返回 None。"""
        safe_x_min = -self.world_xy + self.evader_margin_xy + 0.05
        safe_x_max = self.world_xy - self.evader_margin_xy - 0.05
        safe_y_abs = max(self.world_xy - self.evader_margin_xy - 0.05, 0.05)
        safe_z_min = self.z_min + self.evader_margin_z + 0.02
        safe_z_max = self.z_max - self.evader_margin_z - 0.02
        if safe_z_min > safe_z_max:
            safe_z_min, safe_z_max = self.z_min + 0.02, self.z_max - 0.02

        evader_x_low = float(np.clip(self.init_evader_x_range_ratio[0] * self.world_xy, safe_x_min, safe_x_max))
        evader_x_high = float(np.clip(self.init_evader_x_range_ratio[1] * self.world_xy, safe_x_min, safe_x_max))
        if evader_x_low > evader_x_high:
            evader_x_low, evader_x_high = evader_x_high, evader_x_low

        evader_y_abs = min(self.init_evader_y_range_ratio * self.world_xy, safe_y_abs)
        if evader_y_abs < 1e-6:
            evader_y_abs = min(0.05, safe_y_abs)

        init_z_low = float(np.clip(self.init_evader_z_range[0], safe_z_min, safe_z_max))
        init_z_high = float(np.clip(self.init_evader_z_range[1], safe_z_min, safe_z_max))
        if init_z_low > init_z_high:
            init_z_low, init_z_high = init_z_high, init_z_low

        min_mean_dist = self.init_mean_dist_range_ratio[0] * self.world_xy
        max_mean_dist = self.init_mean_dist_range_ratio[1] * self.world_xy
        if min_mean_dist > max_mean_dist:
            min_mean_dist, max_mean_dist = max_mean_dist, min_mean_dist

        clear = self._init_obstacle_clearance()

        for _ in range(self.max_init_resample):
            evader_pos = np.array(
                [
                    rng.uniform(evader_x_low, evader_x_high),
                    rng.uniform(-evader_y_abs, evader_y_abs),
                    rng.uniform(init_z_low, init_z_high),
                ],
                dtype=np.float32,
            )
            dists = np.linalg.norm(pursuer_pos - evader_pos[None, :], axis=1)
            mean_dist = float(np.mean(dists))
            all_far_enough = bool(np.all(dists >= 1.5 * self.capture_dist))
            xy_ok = bool(self._xy_clear_of_obstacles(evader_pos[:2].reshape(1, 2), obstacle_xy, obstacle_r, clear)[0])
            if all_far_enough and (min_mean_dist <= mean_dist <= max_mean_dist) and xy_ok:
                return evader_pos
        return None

    def sample_initial_conditions(self, num_agents: int, rng: np.random.Generator):
        assert num_agents == 4, f"PursuitEvasion3v1Task requires num_agents=4, got {num_agents}"

        pursuer_ids = np.array([0, 1, 2], dtype=np.int64)
        evader_id = 3

        start_pos = np.zeros((num_agents, 3), dtype=np.float32)
        start_orn = np.zeros((num_agents, 3), dtype=np.float32)

        z0 = float(np.clip(rng.uniform(0.95, 1.05), self.z_min + 0.03, self.z_max - 0.03))

        pursuer_x = -self.init_pursuer_x_ratio * self.world_xy
        pursuer_y_spread = self.init_pursuer_y_spread_ratio * self.world_xy
        pursuer_noise_xy = self.init_pursuer_noise_xy_ratio * self.world_xy

        base_pursuers = np.array(
            [
                [pursuer_x, -pursuer_y_spread, z0],
                [pursuer_x, 0.0, z0],
                [pursuer_x, pursuer_y_spread, z0],
            ],
            dtype=np.float32,
        )

        clear = self._init_obstacle_clearance()
        outer_tries = max(self.max_init_resample, 50)

        for _ in range(outer_tries):
            obstacle_xy, obstacle_r = self._sample_obstacles(rng)
            if obstacle_xy.shape[0] < self.num_obstacles_min:
                continue

            noise_xy = rng.uniform(-pursuer_noise_xy, pursuer_noise_xy, size=(3, 2)).astype(np.float32)
            noise_z = rng.uniform(-self.init_pursuer_noise_z, self.init_pursuer_noise_z, size=(3, 1)).astype(np.float32)
            start_pos[pursuer_ids, :2] = base_pursuers[:, :2] + noise_xy
            start_pos[pursuer_ids, 2:] = base_pursuers[:, 2:] + noise_z
            start_pos[pursuer_ids] = self._clip_positions_inside(start_pos[pursuer_ids], margin_xy=0.02, margin_z=0.02)

            p_xy = start_pos[pursuer_ids, :2]
            if not bool(np.all(self._xy_clear_of_obstacles(p_xy, obstacle_xy, obstacle_r, clear))):
                continue

            evader_pos = self._sample_valid_evader_position(start_pos[pursuer_ids], rng, obstacle_xy, obstacle_r)
            if evader_pos is None:
                continue

            start_pos[evader_id] = evader_pos

            init_dists = np.linalg.norm(
                start_pos[pursuer_ids] - start_pos[evader_id][None, :],
                axis=1,
            ).astype(np.float32)

            if self.debug:
                print(
                    "[reset ex2] n_obs=", obstacle_xy.shape[0],
                    "world_xy=", self.world_xy,
                    "init_dists=", init_dists,
                )

            task_state = PursuitEvasion3v1TaskEx2State(
                pursuer_ids=pursuer_ids,
                evader_id=evader_id,
                captured=bool(np.any(init_dists <= self.capture_dist)),
                capture_agent=int(pursuer_ids[np.argmin(init_dists)]) if np.any(init_dists <= self.capture_dist) else -1,
                prev_pursuer_dists=init_dists.astype(np.float32).copy(),
                obstacle_xy=obstacle_xy,
                obstacle_r=obstacle_r,
            )
            return start_pos, start_orn, task_state

        # 极端情况下退化为无障碍物，保证 reset 总能返回（训练不中断）
        obstacle_xy = np.zeros((0, 2), dtype=np.float32)
        obstacle_r = np.zeros((0,), dtype=np.float32)
        noise_xy = rng.uniform(-pursuer_noise_xy, pursuer_noise_xy, size=(3, 2)).astype(np.float32)
        noise_z = rng.uniform(-self.init_pursuer_noise_z, self.init_pursuer_noise_z, size=(3, 1)).astype(np.float32)
        start_pos[pursuer_ids, :2] = base_pursuers[:, :2] + noise_xy
        start_pos[pursuer_ids, 2:] = base_pursuers[:, 2:] + noise_z
        start_pos[pursuer_ids] = self._clip_positions_inside(start_pos[pursuer_ids], margin_xy=0.02, margin_z=0.02)
        start_pos[evader_id] = super()._sample_valid_evader_position(start_pos[pursuer_ids], rng)
        init_dists = np.linalg.norm(
            start_pos[pursuer_ids] - start_pos[evader_id][None, :],
            axis=1,
        ).astype(np.float32)
        task_state = PursuitEvasion3v1TaskEx2State(
            pursuer_ids=pursuer_ids,
            evader_id=evader_id,
            captured=bool(np.any(init_dists <= self.capture_dist)),
            capture_agent=int(pursuer_ids[np.argmin(init_dists)]) if np.any(init_dists <= self.capture_dist) else -1,
            prev_pursuer_dists=init_dists.astype(np.float32).copy(),
            obstacle_xy=obstacle_xy,
            obstacle_r=obstacle_r,
        )
        return start_pos, start_orn, task_state

    def _pursuer_obstacle_collision_mask(
        self,
        pursuer_pos: np.ndarray,
        task_state: PursuitEvasion3v1TaskEx2State,
    ) -> np.ndarray:
        """shape [P] bool，True 表示该 pursuer 与某圆柱在 xy 上重叠（z 全高度）。"""
        if not isinstance(task_state, PursuitEvasion3v1TaskEx2State):
            return np.zeros((pursuer_pos.shape[0],), dtype=bool)
        obs_xy = task_state.obstacle_xy
        obs_r = task_state.obstacle_r
        if obs_xy.size == 0:
            return np.zeros((pursuer_pos.shape[0],), dtype=bool)
        xy = pursuer_pos[:, :2]
        hit_r = self._pursuer_obstacle_hit_radius()
        d = np.linalg.norm(xy[:, None, :] - obs_xy[None, :, :], axis=2)
        return np.any(d <= obs_r[None, :] + hit_r, axis=1)

    def _pursuer_obstacle_obs_block(
        self,
        pursuer_xy: np.ndarray,
        obstacle_xy: np.ndarray,
        obstacle_r: np.ndarray,
    ) -> np.ndarray:
        """
        单机观测用：按该机到柱轴水平距离升序取最多 num_obstacles_max 根柱。
        每槽 4 维：[ (cx-px)/pos_xy_norm, (cy-py)/pos_xy_norm, r/pos_xy_norm, valid ]。
        """
        m_max = self._obstacle_feature_slots
        scale = float(self.pos_xy_norm)
        out = np.zeros((m_max * 4,), dtype=np.float32)
        if obstacle_xy.size == 0:
            return out
        pxy = np.asarray(pursuer_xy, dtype=np.float64).reshape(2)
        oxy = np.asarray(obstacle_xy, dtype=np.float64)
        orad = np.asarray(obstacle_r, dtype=np.float64).reshape(-1)
        k = int(oxy.shape[0])
        d = np.linalg.norm(oxy - pxy.reshape(1, 2), axis=1)
        order = np.argsort(d)
        take = min(k, m_max)
        for slot in range(take):
            idx = int(order[slot])
            c0, c1 = float(oxy[idx, 0]), float(oxy[idx, 1])
            r = float(orad[idx])
            base = slot * 4
            out[base + 0] = np.float32((c0 - pxy[0]) / scale)
            out[base + 1] = np.float32((c1 - pxy[1]) / scale)
            out[base + 2] = np.float32(r / scale)
            out[base + 3] = np.float32(1.0)
        return out

    def _global_obstacle_state_block(
        self,
        obstacle_xy: np.ndarray,
        obstacle_r: np.ndarray,
    ) -> np.ndarray:
        """
        Critic 全局状态：柱心按 (cx, cy) 字典序排列，与各 pursuer 局部顺序无关。
        每槽 4 维：[ cx/pos_xy_norm, cy/pos_xy_norm, r/pos_xy_norm, valid ]。
        """
        m_max = self._obstacle_feature_slots
        scale = float(self.pos_xy_norm)
        out = np.zeros((m_max * 4,), dtype=np.float32)
        if obstacle_xy.size == 0:
            return out
        oxy = np.asarray(obstacle_xy, dtype=np.float64)
        orad = np.asarray(obstacle_r, dtype=np.float64).reshape(-1)
        k = int(oxy.shape[0])
        order = np.lexsort((oxy[:, 1], oxy[:, 0]))
        take = min(k, m_max)
        for slot in range(take):
            idx = int(order[slot])
            c0, c1 = float(oxy[idx, 0]), float(oxy[idx, 1])
            r = float(orad[idx])
            base = slot * 4
            out[base + 0] = np.float32(c0 / scale)
            out[base + 1] = np.float32(c1 / scale)
            out[base + 2] = np.float32(r / scale)
            out[base + 3] = np.float32(1.0)
        return out

    def build_obs(self, backend_state, task_state: PursuitEvasion3v1TaskState) -> np.ndarray:
        base = super().build_obs(backend_state, task_state)
        if not isinstance(task_state, PursuitEvasion3v1TaskEx2State):
            return base
        states = backend_state.states
        lin_pos = states[:, 3, :]
        pursuer_ids = task_state.pursuer_ids
        obs_xy = task_state.obstacle_xy
        obs_r = task_state.obstacle_r
        extras = [
            self._pursuer_obstacle_obs_block(lin_pos[int(i), :2], obs_xy, obs_r)
            for i in pursuer_ids
        ]
        return np.concatenate([base, np.stack(extras, axis=0)], axis=1).astype(np.float32)

    def build_state(self, backend_state, task_state: PursuitEvasion3v1TaskState) -> np.ndarray:
        base = super().build_state(backend_state, task_state)
        if not isinstance(task_state, PursuitEvasion3v1TaskEx2State):
            return base
        glob_obs = self._global_obstacle_state_block(task_state.obstacle_xy, task_state.obstacle_r)
        return np.concatenate([base, glob_obs], axis=0).astype(np.float32)

    def compute_rewards(
        self,
        prev_backend_state,
        backend_state,
        task_state: PursuitEvasion3v1TaskState,
    ) -> np.ndarray:
        rewards = super().compute_rewards(prev_backend_state, backend_state, task_state)
        if isinstance(task_state, PursuitEvasion3v1TaskEx2State):
            lin_pos = backend_state.states[:, 3, :]
            pursuer_ids = task_state.pursuer_ids
            pursuer_pos = lin_pos[pursuer_ids]
            obs_hit = self._pursuer_obstacle_collision_mask(pursuer_pos, task_state)
            if np.any(obs_hit):
                rewards = rewards - obs_hit.astype(np.float32) * np.float32(self.obstacle_collision_penalty)
        return rewards

    def compute_terminated_truncated(
        self,
        backend_state,
        task_state: PursuitEvasion3v1TaskState,
        step_count: int,
    ):
        terminated, truncated = super().compute_terminated_truncated(backend_state, task_state, step_count)
        if isinstance(task_state, PursuitEvasion3v1TaskEx2State):
            lin_pos = backend_state.states[:, 3, :]
            pursuer_pos = lin_pos[task_state.pursuer_ids]
            if np.any(self._pursuer_obstacle_collision_mask(pursuer_pos, task_state)):
                terminated = True
        return terminated, truncated
