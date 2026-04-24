from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np

from marl_uav.envs.tasks.base_task import BaseTask


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def mod_2pi(angle: float) -> float:
    """Wrap angle to [0, 2pi)."""
    return angle % (2.0 * np.pi)


def build_structure_aware_state_19d(
    pursuer_pos: np.ndarray,  # [3, 3], each row = [x, y, z]
    pursuer_vel: np.ndarray,  # [3, 3]
    evader_pos: np.ndarray,  # [3]
    evader_vel: np.ndarray,  # [3]
    env_xy_scale: float,
    vel_scale: float,
    prev_structure_metrics: np.ndarray | None = None,
    current_structure_metrics: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Returns struct_obs of shape [3, 19]: structure-aware features per pursuer row.

    The last 3 dims are structure deltas relative to the previous step:
    [delta_C_cov, delta_C_col, delta_D_ang].
    (row k matches pursuer_pos[k] / pursuer_vel[k]).
    """
    num_pursuers = 3
    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
    pv = np.asarray(pursuer_vel, dtype=np.float64).reshape(3, 3)
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    ev = np.asarray(evader_vel, dtype=np.float64).reshape(3)

    rel_xy = p[:, :2] - e[None, :2]
    rel_vel_xy = pv[:, :2] - ev[None, :2]

    d = np.linalg.norm(rel_xy, axis=1)
    theta = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])

    u = rel_xy / (d[:, None] + eps)
    u_perp = np.stack([-u[:, 1], u[:, 0]], axis=1)

    v_r = -np.sum(rel_vel_xy * u, axis=1)
    v_t = np.sum(rel_vel_xy * u_perp, axis=1)

    order = np.argsort(theta)
    theta_sorted = theta[order]

    phi = np.zeros(3, dtype=np.float64)
    phi[0] = theta_sorted[1] - theta_sorted[0]
    phi[1] = theta_sorted[2] - theta_sorted[1]
    phi[2] = 2.0 * np.pi + theta_sorted[0] - theta_sorted[2]

    phi_max = float(np.max(phi))
    d_mean = float(np.mean(d))
    collapse = float(np.linalg.norm(np.mean(u, axis=0)))

    k_max = int(np.argmax(phi))
    theta_gap_start = float(theta_sorted[k_max])
    theta_gap_center = wrap_to_pi(theta_gap_start + 0.5 * phi_max)

    pred = np.zeros(num_pursuers, dtype=np.int64)
    succ = np.zeros(num_pursuers, dtype=np.int64)

    for rank, idx in enumerate(order):
        pred_idx = int(order[(rank - 1) % num_pursuers])
        succ_idx = int(order[(rank + 1) % num_pursuers])
        pred[idx] = pred_idx
        succ[idx] = succ_idx

    env_s = max(float(env_xy_scale), eps)
    vel_s = max(float(vel_scale), eps)

    curr_struct = np.asarray(
        current_structure_metrics, dtype=np.float32
    ).reshape(-1) if current_structure_metrics is not None else np.zeros((0,), dtype=np.float32)
    if curr_struct.shape[0] != 3:
        metrics = compute_pursuit_structure_metrics_3v1(p, e, eps=eps)
        curr_struct = np.array(
            [metrics["C_cov"], metrics["C_col"], metrics["D_ang"]],
            dtype=np.float32,
        )
    prev_struct = np.asarray(
        curr_struct if prev_structure_metrics is None else prev_structure_metrics,
        dtype=np.float32,
    ).reshape(-1)
    if prev_struct.shape[0] != 3:
        prev_struct = curr_struct.copy()
    struct_delta = np.clip(curr_struct - prev_struct, -1.0, 1.0).astype(np.float32)

    struct_obs = np.zeros((num_pursuers, 19), dtype=np.float32)

    for i in range(num_pursuers):
        i_pred = int(pred[i])
        i_succ = int(succ[i])

        delta_theta_pred = wrap_to_pi(float(theta[i_pred]) - float(theta[i]))
        delta_theta_succ = wrap_to_pi(float(theta[i_succ]) - float(theta[i]))

        phi_left = mod_2pi(float(theta[i]) - float(theta[i_pred]))
        phi_right = mod_2pi(float(theta[i_succ]) - float(theta[i]))

        delta_theta_gap = wrap_to_pi(theta_gap_center - float(theta[i]))

        d_i = float(d[i]) / env_s
        d_pred = float(d[i_pred]) / env_s
        d_succ = float(d[i_succ]) / env_s

        vr_i = float(v_r[i]) / vel_s
        vt_i = float(v_t[i]) / vel_s

        phi_left_n = phi_left / np.pi
        phi_right_n = phi_right / np.pi
        phi_max_n = phi_max / np.pi
        d_mean_n = d_mean / env_s

        struct_obs[i] = np.array(
            [
                d_i,
                vr_i,
                vt_i,
                d_pred,
                d_succ,
                np.sin(delta_theta_pred),
                np.cos(delta_theta_pred),
                np.sin(delta_theta_succ),
                np.cos(delta_theta_succ),
                phi_left_n,
                phi_right_n,
                phi_max_n,
                collapse,
                d_mean_n,
                np.sin(delta_theta_gap),
                np.cos(delta_theta_gap),
                struct_delta[0],
                struct_delta[1],
                struct_delta[2],
            ],
            dtype=np.float32,
        )

    return struct_obs


def compute_pursuit_structure_metrics_3v1(
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    *,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """
    3v1 围捕在水平面 (xy) 上的结构化指标（逐时刻）。

    记 evader 位置 e，第 i 个 pursuer 位置 p_i，相对位置 xy 投影
    r̃_i = Π_xy(p_i - e)，方位角 θ_i = atan2(r̃_{i,y}, r̃_{i,x})，排序后得
    θ_(1) ≤ θ_(2) ≤ θ_(3)，循环角间隔
    φ_1 = θ_(2)-θ_(1), φ_2 = θ_(3)-θ_(2), φ_3 = 2π+θ_(1)-θ_(3)。

    返回:
        C_cov: (2π - φ_max) / (4π/3)，φ_max = max(φ_k)
        C_col: ‖(1/3)Σ u_i‖，u_i = r̃_i / (‖r̃_i‖ + ε)
        D_ang: 1 - E_ang / (2π²/3)，E_ang = (1/3) Σ_k (φ_k - 2π/3)²
    以及 φ_1, φ_2, φ_3, phi_max, E_ang 便于记录。
    """
    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    r_tilde = p[:, :2] - e[:2]

    theta = np.arctan2(r_tilde[:, 1], r_tilde[:, 0])
    theta_sorted = np.sort(theta)
    t1, t2, t3 = float(theta_sorted[0]), float(theta_sorted[1]), float(theta_sorted[2])

    two_pi = 2.0 * np.pi
    phi1 = t2 - t1
    phi2 = t3 - t2
    phi3 = two_pi + t1 - t3
    phi = np.array([phi1, phi2, phi3], dtype=np.float64)
    phi_max = float(np.max(phi))

    four_pi_over_3 = 4.0 * np.pi / 3.0
    c_cov = (two_pi - phi_max) / four_pi_over_3
    c_cov = float(np.clip(c_cov, 0.0, 1.0))

    norms = np.linalg.norm(r_tilde, axis=1)
    u = r_tilde / (norms[:, None] + eps)
    mean_u = np.mean(u, axis=0)
    c_col = float(np.linalg.norm(mean_u))
    c_col = float(np.clip(c_col, 0.0, 1.0))

    e_ang = float(np.mean((phi - two_pi / 3.0) ** 2))
    denom = 2.0 * np.pi**2 / 3.0
    d_ang = 1.0 - e_ang / denom
    d_ang = float(np.clip(d_ang, 0.0, 1.0))

    return {
        "C_cov": c_cov,
        "C_col": c_col,
        "D_ang": d_ang,
        "phi_1": float(phi1),
        "phi_2": float(phi2),
        "phi_3": float(phi3),
        "phi_max": phi_max,
        "E_ang": e_ang,
        "theta_sorted": [t1, t2, t3],
        "theta_pursuer": [float(theta[0]), float(theta[1]), float(theta[2])],
    }


@dataclass
class PursuitEvasion3v1TaskState:
    """阶段1任务状态：固定 3 pursuers + 1 heuristic evader."""

    pursuer_ids: np.ndarray  # [3], e.g. [0,1,2]
    evader_id: int  # e.g. 3
    captured: bool  # episode 内是否已捕获
    capture_agent: int  # 捕获者 id, 未捕获为 -1
    prev_pursuer_dists: np.ndarray  # [P] 上一步各 pursuer 到 evader 的距离
    prev_structure_metrics: np.ndarray = field(
        default_factory=lambda: np.zeros((3,), dtype=np.float32)
    )  # [C_cov, C_col, D_ang]
    latest_structure_metrics: np.ndarray = field(
        default_factory=lambda: np.zeros((3,), dtype=np.float32)
    )
    structure_hold_steps: int = 0  # 连续满足结构阈值的步数
    assigned_target_indices: np.ndarray = field(
        default_factory=lambda: np.arange(3, dtype=np.int64)
    )
    prev_role_target_dists: np.ndarray = field(
        default_factory=lambda: np.zeros((3,), dtype=np.float32)
    )
    prev_mean_radius_xy: float = 0.0
    initial_mean_radius_xy: float = 0.0
    latest_target_radius_xy: float = 0.0
    elapsed_steps: int = 0


class PursuitEvasion3v1Task(BaseTask):
    """
    修正版自适应边界任务：
    - 修复 reset 几何条件不可达的问题
    - 对 obs / state 做尺度归一化，避免 world_xy 改变后输入分布漂移
    - 对 capture_dist / xy_speed / min_sep 做一致尺度缩放
    - 默认任一 pursuer 出界即可终止，避免“单机牺牲式”策略

    说明：
    1) world_xy 主要控制 x/y 平面尺度；z 方向范围通常不随之放大。
    2) 为保持“相似任务”，x/y 相关几何量按 scene_scale = world_xy / reference_world_xy 缩放：
       - capture_dist
       - pursuer / evader 的 xy 速度
       - min_pursuer_sep
    3) 垂直速度单独控制，不随 world_xy 线性放大，避免 z 方向过激动作。
    4) 默认不放大 episode_limit，因为初始化距离和 xy 速度都已同步缩放，
       到达时间尺度基本保持一致；如需更大场景更长回合，可打开 adaptive_episode_limit。
    """

    def __init__(
        self,
        world_xy: float = 2.0,
        z_min: float = 0.8,
        z_max: float = 1.2,
        capture_dist: float = 0.30,
        episode_limit: int = 400,
        pursuer_speed: float = 0.20,
        evader_speed: float = 0.16,
        pursuer_vertical_speed: float = 0.12,
        evader_vertical_speed: float = 0.16,
        min_pursuer_sep: float = 0.25,
        progress_reward_scale: float = 1.0,
        min_progress_reward_scale: float = 0.5,
        time_penalty: float = 0.05,
        capture_bonus: float = 20.0,
        collision_penalty: float = 2.0,
        oob_penalty: float = 3.0,
        evader_boundary_gain: float = 2.8,
        evader_apf_pursuer_gain: float = 0.8,
        evader_nearest_pursuer_gain: float = 1.4,
        evader_center_pull_gain: float = 0.7,
        evader_boundary_tangent_gain: float = 0.9,
        evader_boundary_slowdown_min: float = 0.45,
        *,
        init_pursuer_x_ratio: float = 0.30,
        init_pursuer_y_spread_ratio: float = 0.10,
        init_pursuer_noise_xy_ratio: float = 0.03,
        init_pursuer_noise_z: float = 0.03,
        init_evader_x_range_ratio: tuple[float, float] = (0.12, 0.28),
        init_evader_y_range_ratio: float = 0.12,
        init_evader_z_range: tuple[float, float] = (0.95, 1.10),
        init_mean_dist_range_ratio: tuple[float, float] = (0.45, 0.75),
        evader_margin_xy_ratio: float = 0.25,
        evader_margin_z_ratio: float = 0.15,
        adaptive_episode_limit: bool = False,
        reference_world_xy: float = 2.0,
        max_pursuers_oob_before_terminate: int = 1,
        max_init_resample: int = 200,
        debug: bool = False,
        mean_progress_reward_scale: float = 0.5,
        progress_dist_norm: float = 2.0,
        capture_bonus_team: float = 30.0,
        capture_bonus_individual: float = 10.0,
        structure_reward_scale: float = 0.25,
        structure_improve_scale: float = 0.2,
        structure_hold_reward_scale: float = 0.05,
        structure_cov_weight: float = 1.0,
        structure_col_weight: float = 1.0,
        structure_ang_weight: float = 1.0,
        structure_cov_threshold: float = 0.75,
        structure_col_threshold: float = 0.35,
        structure_ang_threshold: float = 0.75,
        structure_hold_steps_cap: int = 30,
        structure_gate_near_dist_ratio: float = 3.0,
        structure_gate_far_dist_ratio: float = 6.0,
        progress_gate_min_scale: float = 0.35,
        structure_obs_include_deltas: bool = True,
        role_assignment_mode: str = "nearest",
        manifold_target_phase: float = 0.0,
        manifold_target_radius_scale: float = 1.0,
        manifold_target_rho_min: float | None = None,
        manifold_target_rho_max: float | None = None,
        manifold_contraction_rate: float = 0.01,
        manifold_structure_gate_scale: float = 0.75,
        assignment_inertia_margin: float = 0.05,
        role_progress_reward_scale: float = 0.75,
        residual_control_gain: float = 0.5,
        radial_compress_reward_scale: float = 1.0,
        radial_overshoot_penalty_scale: float = 0.5,
        contraction_reward_norm: float | None = None,
        contraction_phase_structure_scale: float = 0.3,
        contraction_phase_compress_scale: float = 2.0,
    ) -> None:
        self.world_xy = float(world_xy)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.z_span = max(self.z_max - self.z_min, 1e-6)
        self.z_center = 0.5 * (self.z_min + self.z_max)
        self.z_half_span = 0.5 * self.z_span

        self.reference_world_xy = max(float(reference_world_xy), 1e-6)
        self.scene_scale = max(self.world_xy / self.reference_world_xy, 1e-6)

        # 基础参数（以 reference_world_xy 对应的标称场景为基准）
        self.base_capture_dist = float(capture_dist)
        self.base_pursuer_speed = float(pursuer_speed)
        self.base_evader_speed = float(evader_speed)
        self.base_pursuer_vertical_speed = float(pursuer_vertical_speed)
        self.base_evader_vertical_speed = float(evader_vertical_speed)
        self.base_min_pursuer_sep = float(min_pursuer_sep)

        # 一致尺度缩放：只缩放 x/y 相关几何和速度；z 方向单独保留
        self.capture_dist = self.base_capture_dist * self.scene_scale * 0.1
        self.pursuer_speed_xy = self.base_pursuer_speed * self.scene_scale
        self.evader_speed_xy = self.base_evader_speed * self.scene_scale
        self.pursuer_speed_z = self.base_pursuer_vertical_speed
        self.evader_speed_z = self.base_evader_vertical_speed
        self.min_pursuer_sep = self.base_min_pursuer_sep * self.scene_scale

        # 保留旧属性名，兼容外部可能的直接访问
        self.pursuer_speed = self.pursuer_speed_xy
        self.evader_speed = self.evader_speed_xy

        self.progress_reward_scale = float(progress_reward_scale)
        self.min_progress_reward_scale = float(min_progress_reward_scale)
        self.time_penalty = float(time_penalty)
        self.capture_bonus = float(capture_bonus)
        self.collision_penalty = float(collision_penalty)
        self.oob_penalty = float(oob_penalty)
        self.evader_boundary_gain = float(evader_boundary_gain)
        self.evader_apf_pursuer_gain = float(evader_apf_pursuer_gain)
        self.evader_nearest_pursuer_gain = float(evader_nearest_pursuer_gain)
        self.evader_center_pull_gain = float(evader_center_pull_gain)
        self.evader_boundary_tangent_gain = float(evader_boundary_tangent_gain)
        self.evader_boundary_slowdown_min = float(np.clip(evader_boundary_slowdown_min, 0.10, 1.00))

        self.init_pursuer_x_ratio = float(init_pursuer_x_ratio)
        self.init_pursuer_y_spread_ratio = float(init_pursuer_y_spread_ratio)
        self.init_pursuer_noise_xy_ratio = float(init_pursuer_noise_xy_ratio)
        self.init_pursuer_noise_z = float(init_pursuer_noise_z)
        self.init_evader_x_range_ratio = tuple(float(x) for x in init_evader_x_range_ratio)
        self.init_evader_y_range_ratio = float(init_evader_y_range_ratio)
        self.init_evader_z_range = tuple(float(z) for z in init_evader_z_range)
        self.init_mean_dist_range_ratio = tuple(float(x) for x in init_mean_dist_range_ratio)
        self.evader_margin_xy_ratio = float(evader_margin_xy_ratio)
        self.evader_margin_z_ratio = float(evader_margin_z_ratio)
        self.adaptive_episode_limit = bool(adaptive_episode_limit)
        self.max_pursuers_oob_before_terminate = int(max_pursuers_oob_before_terminate)
        self.max_init_resample = int(max_init_resample)
        self.debug = bool(debug)

        if self.adaptive_episode_limit:
            self.episode_limit = int(np.ceil(float(episode_limit) * self.scene_scale))
        else:
            self.episode_limit = int(episode_limit)

        self.evader_margin_xy = max(0.05, self.evader_margin_xy_ratio * self.world_xy)
        self.evader_margin_z = max(0.05, self.evader_margin_z_ratio * self.z_span)

        # 归一化尺度
        self.pos_xy_norm = max(self.world_xy, 1e-6)
        self.pos_z_norm = max(self.z_half_span, 1e-6)
        self.vel_xy_norm = max(self.pursuer_speed_xy, self.evader_speed_xy, 1e-6)
        self.vel_z_norm = max(self.pursuer_speed_z, self.evader_speed_z, 1e-6)
        self.dist_norm = max(self.world_xy, self.capture_dist, 1e-6)
        self.angle_norm = np.pi

        self.mean_progress_reward_scale = float(mean_progress_reward_scale)
        self.progress_dist_norm = float(progress_dist_norm)
        self.capture_bonus_team = float(capture_bonus_team)
        self.capture_bonus_individual = float(capture_bonus_individual)
        self.structure_reward_scale = float(structure_reward_scale)
        self.structure_improve_scale = float(structure_improve_scale)
        self.structure_hold_reward_scale = float(structure_hold_reward_scale)
        self.structure_cov_weight = max(float(structure_cov_weight), 0.0)
        self.structure_col_weight = max(float(structure_col_weight), 0.0)
        self.structure_ang_weight = max(float(structure_ang_weight), 0.0)
        self.structure_cov_threshold = float(np.clip(structure_cov_threshold, 0.0, 1.0))
        self.structure_col_threshold = float(np.clip(structure_col_threshold, 0.0, 1.0))
        self.structure_ang_threshold = float(np.clip(structure_ang_threshold, 0.0, 1.0))
        self.structure_hold_steps_cap = max(int(structure_hold_steps_cap), 1)
        self.structure_gate_near_dist_ratio = max(float(structure_gate_near_dist_ratio), 0.0)
        self.structure_gate_far_dist_ratio = max(
            float(structure_gate_far_dist_ratio),
            self.structure_gate_near_dist_ratio + 1e-6,
        )
        self.progress_gate_min_scale = float(np.clip(progress_gate_min_scale, 0.0, 1.0))
        self.structure_obs_include_deltas = bool(structure_obs_include_deltas)
        role_mode = str(role_assignment_mode).strip().lower()
        if role_mode not in {"fixed", "nearest"}:
            raise ValueError(
                f"role_assignment_mode must be 'fixed' or 'nearest', got {role_assignment_mode!r}"
            )
        self.role_assignment_mode = role_mode
        self.manifold_target_phase = float(manifold_target_phase)
        self.manifold_target_radius_scale = max(float(manifold_target_radius_scale), 0.05)
        self.manifold_target_rho_min = (
            float(self.capture_dist)
            if manifold_target_rho_min is None
            else max(float(manifold_target_rho_min), 0.0)
        )
        self.manifold_target_rho_max = (
            None
            if manifold_target_rho_max is None
            else max(float(manifold_target_rho_max), self.manifold_target_rho_min)
        )
        self.manifold_contraction_rate = max(float(manifold_contraction_rate), 0.0)
        self.manifold_structure_gate_scale = float(np.clip(manifold_structure_gate_scale, 0.0, 1.0))
        self.assignment_inertia_margin = max(float(assignment_inertia_margin), 0.0)
        self.role_progress_reward_scale = float(role_progress_reward_scale)
        self.residual_control_gain = float(residual_control_gain)
        self.radial_compress_reward_scale = float(radial_compress_reward_scale)
        self.radial_overshoot_penalty_scale = float(radial_overshoot_penalty_scale)
        self.contraction_reward_norm = (
            max(float(self.capture_dist), 1e-6)
            if contraction_reward_norm is None
            else max(float(contraction_reward_norm), 1e-6)
        )
        self.contraction_phase_structure_scale = max(float(contraction_phase_structure_scale), 0.0)
        self.contraction_phase_compress_scale = max(float(contraction_phase_compress_scale), 0.0)
        # 离散动作：[vx, vy, yaw, vz]
        self._action_table = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [self.pursuer_speed_xy, 0.0, 0.0, 0.0],
                [-self.pursuer_speed_xy, 0.0, 0.0, 0.0],
                [0.0, self.pursuer_speed_xy, 0.0, 0.0],
                [0.0, -self.pursuer_speed_xy, 0.0, 0.0],
                [self.pursuer_speed_xy, -self.pursuer_speed_xy, 0.0, self.pursuer_speed_z],
                [-self.pursuer_speed_xy, self.pursuer_speed_xy, 0.0, -self.pursuer_speed_z],
                [self.pursuer_speed_xy, self.pursuer_speed_xy, 0.0, 0.0],
                [-self.pursuer_speed_xy, -self.pursuer_speed_xy, 0.0, 0.0],
            ],
            dtype=np.float32,

        )
        self.role_obs_dim = 7
        self.role_state_dim = 12

    # ---------------------------------------------------------------------
    # reset / init
    # ---------------------------------------------------------------------
    def sample_initial_conditions(self, num_agents: int, rng: np.random.Generator):
        """
        约定总 agent 数为 4：
        agents 0,1,2 -> pursuers
        agent 3       -> evader
        """
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

        noise_xy = rng.uniform(-pursuer_noise_xy, pursuer_noise_xy, size=(3, 2)).astype(np.float32)
        noise_z = rng.uniform(-self.init_pursuer_noise_z, self.init_pursuer_noise_z, size=(3, 1)).astype(np.float32)
        start_pos[pursuer_ids, :2] = base_pursuers[:, :2] + noise_xy
        start_pos[pursuer_ids, 2:] = base_pursuers[:, 2:] + noise_z
        start_pos[pursuer_ids] = self._clip_positions_inside(start_pos[pursuer_ids], margin_xy=0.02, margin_z=0.02)

        evader_pos = self._sample_valid_evader_position(start_pos[pursuer_ids], rng)
        start_pos[evader_id] = evader_pos

        init_dists = np.linalg.norm(
            start_pos[pursuer_ids] - start_pos[evader_id][None, :],
            axis=1,
        ).astype(np.float32)

        if self.debug:
            print(
                "[reset] world_xy=", self.world_xy,
                "scene_scale=", self.scene_scale,
                "episode_limit=", self.episode_limit,
                "capture_dist=", self.capture_dist,
                "init_dists=", init_dists,
                "pursuer_pos=", start_pos[pursuer_ids],
                "evader_pos=", start_pos[evader_id],
                "z_max=", self.z_max,
                "z_min=", self.z_min,
                "pursuer_speed_xy=", self.pursuer_speed_xy, self.pursuer_speed_z,
                "evader_speed_xy=", self.evader_speed_xy, self.evader_speed_z,
            )

        init_struct = compute_pursuit_structure_metrics_3v1(start_pos[pursuer_ids], start_pos[evader_id])
        init_targets, init_assign = self._compute_role_targets_and_assignment(
            start_pos[pursuer_ids],
            start_pos[evader_id],
            task_state=None,
        )
        init_rel_xy = start_pos[pursuer_ids, :2] - start_pos[evader_id][None, :2]
        init_mean_radius_xy = float(np.mean(np.linalg.norm(init_rel_xy, axis=1)))
        init_role_target_dists = np.linalg.norm(
            init_targets[init_assign] - start_pos[pursuer_ids],
            axis=1,
        ).astype(np.float32)
        task_state = PursuitEvasion3v1TaskState(
            pursuer_ids=pursuer_ids,
            evader_id=evader_id,
            captured=bool(np.any(init_dists <= self.capture_dist)),
            capture_agent=int(pursuer_ids[np.argmin(init_dists)]) if np.any(init_dists <= self.capture_dist) else -1,
            prev_pursuer_dists=init_dists.astype(np.float32).copy(),
            prev_structure_metrics=np.array(
                [
                    init_struct["C_cov"],
                    init_struct["C_col"],
                    init_struct["D_ang"],
                ],
                dtype=np.float32,
            ),
            latest_structure_metrics=np.array(
                [
                    init_struct["C_cov"],
                    init_struct["C_col"],
                    init_struct["D_ang"],
                ],
                dtype=np.float32,
            ),
            structure_hold_steps=1 if self._structure_hold_satisfied(init_struct) else 0,
            assigned_target_indices=init_assign.astype(np.int64).copy(),
            prev_role_target_dists=init_role_target_dists.copy(),
            prev_mean_radius_xy=init_mean_radius_xy,
            initial_mean_radius_xy=init_mean_radius_xy,
            latest_target_radius_xy=max(
                self.manifold_target_rho_min,
                init_mean_radius_xy * self.manifold_target_radius_scale,
            ),
            elapsed_steps=0,
        )
        return start_pos, start_orn, task_state

    def _sample_valid_evader_position(self, pursuer_pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """采样一个满足约束、且不靠近边界的 evader 初始位置。"""
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
            if all_far_enough and (min_mean_dist <= mean_dist <= max_mean_dist):
                return evader_pos

        # fallback：构造一个合法且稳定的位置，避免 reset 分布退化为错误点
        fallback_x = float(np.clip(0.5 * (evader_x_low + evader_x_high), safe_x_min, safe_x_max))
        fallback_z = float(np.clip(0.5 * (init_z_low + init_z_high), safe_z_min, safe_z_max))
        return np.array([fallback_x, 0.0, fallback_z], dtype=np.float32)

    # ---------------------------------------------------------------------
    # observation / state
    # ---------------------------------------------------------------------
    def _structure_aware_features_19d(
        self,
        lin_pos: np.ndarray,
        lin_vel: np.ndarray,
        pursuer_ids: np.ndarray,
        evader_id: int,
        task_state: PursuitEvasion3v1TaskState,
    ) -> np.ndarray:
        """与 build_obs 行顺序一致：第 k 行对应 pursuer_ids[k]。"""
        pp = lin_pos[pursuer_ids]
        pv = lin_vel[pursuer_ids]
        ep = lin_pos[evader_id]
        ev = lin_vel[evader_id]
        vel_scale = max(self.pursuer_speed_xy + self.evader_speed_xy, 1e-6)
        return build_structure_aware_state_19d(
            pp,
            pv,
            ep,
            ev,
            env_xy_scale=float(self.pos_xy_norm),
            vel_scale=float(vel_scale),
            prev_structure_metrics=getattr(task_state, "prev_structure_metrics", None),
            current_structure_metrics=getattr(task_state, "latest_structure_metrics", None),
        )

    def _reference_manifold_targets(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        task_state: PursuitEvasion3v1TaskState | None = None,
    ) -> np.ndarray:
        pursuer_pos = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
        evader_pos = np.asarray(evader_pos, dtype=np.float32).reshape(3)
        rho = self._compute_target_radius_xy(
            pursuer_pos,
            evader_pos,
            task_state=task_state,
        )
        if task_state is not None:
            task_state.latest_target_radius_xy = float(rho)
        ang = self.manifold_target_phase + (2.0 * np.pi / 3.0) * np.arange(3, dtype=np.float32)
        targets = np.zeros((3, 3), dtype=np.float32)
        targets[:, 0] = evader_pos[0] + rho * np.cos(ang)
        targets[:, 1] = evader_pos[1] + rho * np.sin(ang)
        targets[:, 2] = evader_pos[2]
        return targets

    def _compute_role_targets_and_assignment(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        task_state: PursuitEvasion3v1TaskState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pursuer_pos = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
        targets = self._reference_manifold_targets(pursuer_pos, evader_pos, task_state=task_state)
        if self.role_assignment_mode == "fixed":
            return targets, np.arange(3, dtype=np.int64)

        dist_mat = np.linalg.norm(
            pursuer_pos[:, None, :] - targets[None, :, :],
            axis=2,
        )
        best_perm = (0, 1, 2)
        best_cost = np.inf
        for perm in permutations(range(3)):
            cost = float(sum(dist_mat[i, perm[i]] for i in range(3)))
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        best_assignment = np.asarray(best_perm, dtype=np.int64)

        prev_assignment = None if task_state is None else getattr(task_state, "assigned_target_indices", None)
        if prev_assignment is None:
            return targets, best_assignment
        prev_assignment = np.asarray(prev_assignment, dtype=np.int64).reshape(-1)
        if prev_assignment.shape[0] != 3 or len(np.unique(prev_assignment)) != 3:
            return targets, best_assignment

        old_cost = float(sum(dist_mat[i, int(prev_assignment[i])] for i in range(3)))
        if best_cost < old_cost - self.assignment_inertia_margin:
            return targets, best_assignment
        return targets, prev_assignment.copy()

    def _assigned_targets_from_state(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        task_state: PursuitEvasion3v1TaskState | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        targets, assignment = self._compute_role_targets_and_assignment(
            pursuer_pos,
            evader_pos,
            task_state=task_state,
        )
        assigned_targets = targets[assignment]
        return targets, assignment, assigned_targets

    def _role_feature_block(
        self,
        pursuer_pos: np.ndarray,
        assigned_target: np.ndarray,
    ) -> np.ndarray:
        rel_target = self._normalize_delta(assigned_target - pursuer_pos).astype(np.float32)
        norm = float(np.linalg.norm(rel_target))
        if norm < 1e-6:
            slot_dir = np.zeros((3,), dtype=np.float32)
        else:
            slot_dir = (rel_target / norm).astype(np.float32)
        return np.concatenate(
            [
                rel_target,
                slot_dir,
                np.array([1.0], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def _clip_pursuer_setpoints(self, pursuer_setpoints: np.ndarray) -> np.ndarray:
        sp = np.asarray(pursuer_setpoints, dtype=np.float32).copy()
        sp[:, 0] = np.clip(sp[:, 0], -self.pursuer_speed_xy, self.pursuer_speed_xy)
        sp[:, 1] = np.clip(sp[:, 1], -self.pursuer_speed_xy, self.pursuer_speed_xy)
        if sp.shape[1] >= 4:
            sp[:, 3] = np.clip(sp[:, 3], -self.pursuer_speed_z, self.pursuer_speed_z)
        return sp

    def build_obs(self, backend_state, task_state: PursuitEvasion3v1TaskState) -> np.ndarray:
        """
        只为 pursuers 构造 obs，shape = [3, obs_dim]
        obs_i = [
            self_pos(3), self_vel(3), self_ang(3),
            rel_evader_pos(3), rel_evader_vel(3),
            rel_teammate1_pos(3), rel_teammate1_vel(3),
            rel_teammate2_pos(3), rel_teammate2_vel(3),
            structure_aware(19),  # 原 16 维几何结构 + 3 维结构变化量
            assigned_slot_rel(3), assigned_slot_dir(3), assignment_weight(1),
        ]

        全部做归一化：
        - x/y 位置除以 world_xy
        - z 位置以 z_center 为中心、除以 z_half_span
        - x/y 速度除以 max_xy_speed
        - z 速度除以 max_z_speed
        - 角度除以 pi
        - 结构 16 维：距离按 pos_xy_norm，平面相对速度按 pursuer_xy+evader_xy
        """
        states = backend_state.states  # [N, 4, 3]
        ang_pos = states[:, 1, :]
        lin_vel = states[:, 2, :]
        lin_pos = states[:, 3, :]

        pursuer_ids = task_state.pursuer_ids
        evader_id = task_state.evader_id

        pursuer_pos = lin_pos[pursuer_ids]
        evader_pos = lin_pos[evader_id]
        evader_vel = lin_vel[evader_id]
        manifold_targets, assignment, assigned_targets = self._assigned_targets_from_state(
            pursuer_pos,
            evader_pos,
            task_state=task_state,
        )
        task_state.assigned_target_indices = assignment.astype(np.int64).copy()

        struct19 = self._structure_aware_features_19d(
            lin_pos, lin_vel, pursuer_ids, evader_id, task_state
        )

        obs_list = []
        for row, i in enumerate(pursuer_ids):
            teammates = [j for j in pursuer_ids if j != i]
            j1, j2 = teammates[0], teammates[1]

            obs_i = np.concatenate(
                [
                    self._normalize_position(lin_pos[i]),
                    self._normalize_velocity(lin_vel[i]),
                    self._normalize_angle(ang_pos[i]),
                    self._normalize_delta(evader_pos - lin_pos[i]),
                    self._normalize_velocity(evader_vel - lin_vel[i]),
                    self._normalize_delta(lin_pos[j1] - lin_pos[i]),
                    self._normalize_velocity(lin_vel[j1] - lin_vel[i]),
                    self._normalize_delta(lin_pos[j2] - lin_pos[i]),
                    self._normalize_velocity(lin_vel[j2] - lin_vel[i]),
                    (
                        struct19[row]
                        if self.structure_obs_include_deltas
                        else struct19[row, :16]
                    ),
                    self._role_feature_block(
                        lin_pos[i],
                        assigned_targets[row],
                    ),
                ],
                axis=0,
            ).astype(np.float32)
            obs_list.append(obs_i)

        return np.stack(obs_list, axis=0)

    def build_state(self, backend_state, task_state: PursuitEvasion3v1TaskState) -> np.ndarray:
        """centralized critic 全局状态（归一化后）。不含结构特征，末尾追加已分配 manifold target 与 slot index。"""
        states = backend_state.states
        lin_pos = states[:, 3, :]
        lin_vel = states[:, 2, :]
        ang_pos = states[:, 1, :]

        pursuer_ids = task_state.pursuer_ids
        evader_id = task_state.evader_id

        pursuer_pos = self._normalize_position(lin_pos[pursuer_ids]).reshape(-1)
        pursuer_vel = self._normalize_velocity(lin_vel[pursuer_ids]).reshape(-1)
        pursuer_ang = self._normalize_angle(ang_pos[pursuer_ids]).reshape(-1)
        evader_pos_raw = lin_pos[evader_id]
        evader_pos = self._normalize_position(evader_pos_raw).reshape(-1)
        evader_vel = self._normalize_velocity(lin_vel[evader_id]).reshape(-1)
        rels = self._normalize_delta(lin_pos[pursuer_ids] - lin_pos[evader_id][None, :]).reshape(-1)
        _, assignment, assigned_targets_world = self._assigned_targets_from_state(
            lin_pos[pursuer_ids],
            evader_pos_raw,
            task_state=task_state,
        )
        task_state.assigned_target_indices = assignment.astype(np.int64).copy()
        assigned_targets = self._normalize_position(assigned_targets_world).reshape(-1)
        assignment_feat = (assignment.astype(np.float32) / 2.0).reshape(-1)

        state = np.concatenate(
            [
                pursuer_pos,
                pursuer_vel,
                pursuer_ang,
                evader_pos,
                evader_vel,
                rels,
                assigned_targets,
                assignment_feat,
            ],
            axis=0,
        ).astype(np.float32)
        return state

    def _structure_score_from_metrics(self, metrics: dict[str, Any]) -> float:
        cov = float(np.clip(metrics.get("C_cov", 0.0), 0.0, 1.0))
        col = float(np.clip(metrics.get("C_col", 1.0), 0.0, 1.0))
        ang = float(np.clip(metrics.get("D_ang", 0.0), 0.0, 1.0))
        w_cov = self.structure_cov_weight
        w_col = self.structure_col_weight
        w_ang = self.structure_ang_weight
        w_sum = max(w_cov + w_col + w_ang, 1e-6)
        score = (
            w_cov * cov
            + w_col * (1.0 - col)
            + w_ang * ang
        ) / w_sum
        return float(np.clip(score, 0.0, 1.0))

    def _structure_score_from_array(self, metrics_arr: np.ndarray) -> float:
        arr = np.asarray(metrics_arr, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 3:
            return 0.0
        return self._structure_score_from_metrics(
            {
                "C_cov": float(arr[0]),
                "C_col": float(arr[1]),
                "D_ang": float(arr[2]),
            }
        )

    def _structure_hold_satisfied(self, metrics: dict[str, Any]) -> bool:
        cov = float(np.clip(metrics.get("C_cov", 0.0), 0.0, 1.0))
        col = float(np.clip(metrics.get("C_col", 1.0), 0.0, 1.0))
        ang = float(np.clip(metrics.get("D_ang", 0.0), 0.0, 1.0))
        return bool(
            cov >= self.structure_cov_threshold
            and col <= self.structure_col_threshold
            and ang >= self.structure_ang_threshold
        )

    def _structure_reward_gate(self, min_dist: float) -> float:
        near_dist = self.structure_gate_near_dist_ratio * self.capture_dist
        far_dist = self.structure_gate_far_dist_ratio * self.capture_dist
        gate_input = (far_dist - float(min_dist)) / max(far_dist - near_dist, 1e-6)
        return float(self._smoothstep01(gate_input))

    def _mean_radius_xy(self, pursuer_pos: np.ndarray, evader_pos: np.ndarray) -> float:
        pursuer_pos = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
        evader_pos = np.asarray(evader_pos, dtype=np.float32).reshape(3)
        rel_xy = pursuer_pos[:, :2] - evader_pos[None, :2]
        return float(np.mean(np.linalg.norm(rel_xy, axis=1)))

    def _compute_target_radius_xy(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        task_state: PursuitEvasion3v1TaskState | None = None,
    ) -> float:
        pursuer_pos = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
        evader_pos = np.asarray(evader_pos, dtype=np.float32).reshape(3)
        mean_radius_xy = self._mean_radius_xy(pursuer_pos, evader_pos)
        rho_min = float(self.manifold_target_rho_min)
        if task_state is None:
            rho0 = mean_radius_xy
        else:
            rho0 = float(getattr(task_state, "initial_mean_radius_xy", mean_radius_xy))
        rho0 = max(rho0 * self.manifold_target_radius_scale, rho_min)
        rho_max = rho0 if self.manifold_target_rho_max is None else max(float(self.manifold_target_rho_max), rho_min)

        elapsed_steps = 0 if task_state is None else int(getattr(task_state, "elapsed_steps", 0))
        decay = np.exp(-self.manifold_contraction_rate * float(elapsed_steps))
        rho_decay = rho_min + (rho_max - rho_min) * float(decay)

        struct_metrics = compute_pursuit_structure_metrics_3v1(pursuer_pos, evader_pos)
        struct_score = self._structure_score_from_metrics(struct_metrics)
        hold_ready = 1.0 if self._structure_hold_satisfied(struct_metrics) else 0.0
        structure_gate = hold_ready * struct_score
        rho_struct = rho_max - structure_gate * (rho_max - rho_min)

        gate_mix = self.manifold_structure_gate_scale
        rho_target = (1.0 - gate_mix) * rho_decay + gate_mix * min(rho_decay, rho_struct)
        return float(np.clip(rho_target, rho_min, rho_max))

    # ---------------------------------------------------------------------
    # reward / termination
    # ---------------------------------------------------------------------
    def compute_rewards(
            self,
            prev_backend_state,
            backend_state,
            task_state: PursuitEvasion3v1TaskState,
    ) -> np.ndarray:
        """
        每架 pursuer 独立奖励，返回 shape = [3]

        组成：
        1) per-agent progress: 各自距离缩短
        2) mean progress: 团队平均距离缩短
        3) min progress: 最近 pursuer 的关键逼近
        4) role progress: 各自相对分配槽位的距离改善（而不是绝对距离惩罚）
        5) structure reward: 当前结构质量 + 相对上一时刻改善
        6) structure hold reward: 连续保持包围结构
        7) time penalty
        8) capture team bonus + capturer extra bonus
        9) collision / oob penalty
        """
        lin_pos = backend_state.states[:, 3, :]
        pursuer_ids = task_state.pursuer_ids
        evader_id = task_state.evader_id

        pursuer_pos = lin_pos[pursuer_ids]
        evader_pos = lin_pos[evader_id]

        dists = np.linalg.norm(pursuer_pos - evader_pos[None, :], axis=1).astype(np.float32)
        _, assignment, assigned_targets = self._assigned_targets_from_state(
            pursuer_pos,
            evader_pos,
            task_state=task_state,
        )
        role_target_dists = np.linalg.norm(
            assigned_targets - pursuer_pos,
            axis=1,
        ).astype(np.float32)
        min_dist = float(np.min(dists))
        mean_dist = float(np.mean(dists))
        mean_radius_xy = self._mean_radius_xy(pursuer_pos, evader_pos)
        p = len(pursuer_ids)

        prev = np.asarray(task_state.prev_pursuer_dists, dtype=np.float32).reshape(-1)
        if prev.shape[0] != p:
            prev = dists.copy()
        prev_min_dist = float(np.min(prev))
        prev_mean_dist = float(np.mean(prev))

        # 关键：不要再用 world_xy 做归一化
        progress_norm = float(getattr(self, "progress_dist_norm", 2.0))

        per_progress = np.clip((prev - dists) / progress_norm, -1.0, 1.0).astype(np.float32)
        mean_progress = np.clip((prev_mean_dist - mean_dist) / progress_norm, -1.0, 1.0)
        min_progress = np.clip((prev_min_dist - min_dist) / progress_norm, -1.0, 1.0)

        progress_reward_scale = float(getattr(self, "progress_reward_scale", 4.0))
        mean_progress_reward_scale = float(getattr(self, "mean_progress_reward_scale", 2.0))
        min_progress_reward_scale = float(getattr(self, "min_progress_reward_scale", 2.0))
        time_penalty = float(getattr(self, "time_penalty", 0.005))
        prev_role_target_dists = np.asarray(
            getattr(task_state, "prev_role_target_dists", role_target_dists),
            dtype=np.float32,
        ).reshape(-1)
        if prev_role_target_dists.shape[0] != p:
            prev_role_target_dists = role_target_dists.copy()
        role_progress = np.clip(
            (prev_role_target_dists - role_target_dists) / progress_norm,
            -1.0,
            1.0,
        ).astype(np.float32)
        prev_mean_radius_xy = float(getattr(task_state, "prev_mean_radius_xy", mean_radius_xy))

        structure_gate = self._structure_reward_gate(min_dist)
        progress_gate_scale = 1.0 - (1.0 - self.progress_gate_min_scale) * structure_gate

        rewards = progress_reward_scale * per_progress
        rewards += mean_progress_reward_scale * np.float32(mean_progress)
        rewards += min_progress_reward_scale * np.float32(min_progress)
        rewards += np.float32(self.role_progress_reward_scale) * role_progress
        rewards *= np.float32(progress_gate_scale)

        struct_metrics = compute_pursuit_structure_metrics_3v1(pursuer_pos, evader_pos)
        struct_arr = np.array(
            [
                struct_metrics["C_cov"],
                struct_metrics["C_col"],
                struct_metrics["D_ang"],
            ],
            dtype=np.float32,
        )
        prev_struct_arr = np.asarray(
            getattr(task_state, "prev_structure_metrics", struct_arr),
            dtype=np.float32,
        ).reshape(-1)
        if prev_struct_arr.shape[0] != 3:
            prev_struct_arr = struct_arr.copy()

        struct_score = self._structure_score_from_metrics(struct_metrics)
        prev_struct_score = self._structure_score_from_array(prev_struct_arr)
        struct_improve = float(np.clip(struct_score - prev_struct_score, -1.0, 1.0))

        hold_ok = self._structure_hold_satisfied(struct_metrics)
        hold_steps = (int(getattr(task_state, "structure_hold_steps", 0)) + 1) if hold_ok else 0
        hold_ratio = float(np.clip(hold_steps / self.structure_hold_steps_cap, 0.0, 1.0))
        contraction_phase = 1.0 if hold_ok else 0.0
        structure_scale = (
            self.contraction_phase_structure_scale if contraction_phase > 0.0 else 1.0
        )
        compress_scale = (
            self.contraction_phase_compress_scale if contraction_phase > 0.0 else 1.0
        )
        target_radius_xy = self._compute_target_radius_xy(
            pursuer_pos,
            evader_pos,
            task_state=task_state,
        )
        radial_compress = float(
            np.clip(
                (prev_mean_radius_xy - mean_radius_xy) / self.contraction_reward_norm,
                -1.0,
                1.0,
            )
        )
        radial_gap = max(0.0, mean_radius_xy - target_radius_xy)

        rewards += np.float32(self.structure_reward_scale * structure_scale * structure_gate * struct_score)
        rewards += np.float32(self.structure_improve_scale * structure_scale * structure_gate * struct_improve)
        rewards += np.float32(self.structure_hold_reward_scale * structure_scale * structure_gate * hold_ratio)
        rewards += np.float32(
            self.radial_compress_reward_scale
            * compress_scale
            * contraction_phase
            * structure_gate
            * radial_compress
        )
        rewards -= np.float32(
            self.radial_overshoot_penalty_scale
            * compress_scale
            * contraction_phase
            * structure_gate
            * (radial_gap / self.contraction_reward_norm)
        )
        rewards -= np.float32(time_penalty)

        newly_captured = (min_dist <= self.capture_dist) and (not task_state.captured)
        if newly_captured:
            capturer_idx = int(np.argmin(dists))
            task_state.captured = True
            task_state.capture_agent = int(pursuer_ids[capturer_idx])

            capture_bonus_team = float(getattr(self, "capture_bonus_team", 30.0))
            capture_bonus_individual = float(getattr(self, "capture_bonus_individual", 10.0))

            rewards += np.full((p,), np.float32(capture_bonus_team), dtype=np.float32)
            rewards[capturer_idx] += np.float32(capture_bonus_individual)

            task_state.prev_pursuer_dists = dists.copy()
            task_state.prev_structure_metrics = struct_arr.copy()
            task_state.latest_structure_metrics = struct_arr.copy()
            task_state.structure_hold_steps = int(hold_steps)
            task_state.assigned_target_indices = assignment.astype(np.int64).copy()
            task_state.prev_role_target_dists = role_target_dists.copy()
            task_state.prev_mean_radius_xy = float(mean_radius_xy)
            task_state.latest_target_radius_xy = float(target_radius_xy)
            task_state.elapsed_steps = int(getattr(task_state, "elapsed_steps", 0)) + 1
            return rewards.astype(np.float32)

        p_oob_mask = self._get_oob_mask(pursuer_pos)
        rewards -= p_oob_mask.astype(np.float32) * np.float32(self.oob_penalty)

        if hasattr(backend_state, "contact_array"):
            ca = np.asarray(backend_state.contact_array)
            if ca.ndim == 2 and ca.shape[0] > int(np.max(pursuer_ids)):
                p_coll_mask = np.any(ca[pursuer_ids, :] != 0, axis=1)
            else:
                p_coll_mask = np.full((p,), bool(np.any(ca)), dtype=bool)
            rewards -= p_coll_mask.astype(np.float32) * np.float32(self.collision_penalty)

        if self.debug:
            print(
                "[reward] min_dist=", min_dist,
                "mean_dist=", mean_dist,
                "per_progress=", per_progress,
                "mean_progress=", mean_progress,
                "min_progress=", min_progress,
                "role_progress=", role_progress,
                "role_target_dists=", role_target_dists,
                "C_cov=", struct_metrics["C_cov"],
                "C_col=", struct_metrics["C_col"],
                "D_ang=", struct_metrics["D_ang"],
                "structure_gate=", structure_gate,
                "progress_gate_scale=", progress_gate_scale,
                "struct_score=", struct_score,
                "struct_improve=", struct_improve,
                "hold_steps=", hold_steps,
                "contraction_phase=", contraction_phase,
                "mean_radius_xy=", mean_radius_xy,
                "target_radius_xy=", target_radius_xy,
                "radial_compress=", radial_compress,
                "radial_gap=", radial_gap,
                "rewards=", rewards,
            )

        task_state.prev_pursuer_dists = dists.copy()
        task_state.prev_structure_metrics = struct_arr.copy()
        task_state.latest_structure_metrics = struct_arr.copy()
        task_state.structure_hold_steps = int(hold_steps)
        task_state.assigned_target_indices = assignment.astype(np.int64).copy()
        task_state.prev_role_target_dists = role_target_dists.copy()
        task_state.prev_mean_radius_xy = float(mean_radius_xy)
        task_state.latest_target_radius_xy = float(target_radius_xy)
        task_state.elapsed_steps = int(getattr(task_state, "elapsed_steps", 0)) + 1
        return rewards.astype(np.float32)

    def compute_terminated_truncated(
        self,
        backend_state,
        task_state: PursuitEvasion3v1TaskState,
        step_count: int,
    ):
        lin_pos = backend_state.states[:, 3, :]
        pursuer_pos = lin_pos[task_state.pursuer_ids]
        evader_pos = lin_pos[task_state.evader_id]

        captured = bool(task_state.captured)
        p_oob_mask = self._get_oob_mask(pursuer_pos)
        num_p_oob = int(np.sum(p_oob_mask))
        too_many_pursuers_oob = bool(num_p_oob >= self.max_pursuers_oob_before_terminate)
        evader_oob = bool(self._get_oob_mask(evader_pos[None, :])[0])

        if self.debug and (np.any(p_oob_mask) or evader_oob):
            print(
                "[term] p_oob_mask=", p_oob_mask,
                "num_p_oob=", num_p_oob,
                "evader_oob=", evader_oob,
                "evader_pos=", evader_pos,
            )

        terminated = bool(captured or evader_oob or too_many_pursuers_oob)
        truncated = bool(step_count >= self.episode_limit)
        return terminated, truncated

    # ---------------------------------------------------------------------
    # action mapping
    # ---------------------------------------------------------------------
    def action_to_setpoint(
        self,
        actions: np.ndarray,
        backend_state,
        task_state: PursuitEvasion3v1TaskState,
        *,
        action_space_type: str = "discrete",
        action_dim: int | None = None,
    ) -> np.ndarray:
        """
        只接收 pursuers 的动作。
        离散: shape = [3] int，查表；
        连续: shape = [3, action_dim] float，默认原样作为 pursuer setpoint。

        若你在 continuous 模式下训练，请确保 policy/action_head 的 action_low/high
        已按新的有效速度尺度同步调整。
        """
        actions = np.asarray(actions)
        lin_pos = backend_state.states[:, 3, :]
        pursuer_pos = lin_pos[task_state.pursuer_ids]
        evader_pos = lin_pos[task_state.evader_id]
        _, assignment, assigned_targets = self._assigned_targets_from_state(
            pursuer_pos,
            evader_pos,
            task_state=task_state,
        )
        task_state.assigned_target_indices = assignment.astype(np.int64).copy()
        residual = np.zeros((3, 4), dtype=np.float32)
        pos_error = np.float32(self.residual_control_gain) * (
            assigned_targets - pursuer_pos
        ).astype(np.float32)
        residual[:, 0] = pos_error[:, 0]
        residual[:, 1] = pos_error[:, 1]
        residual[:, 3] = pos_error[:, 2]

        if action_space_type == "continuous" and actions.ndim == 2 and actions.dtype in (np.float32, np.float64):
            assert actions.shape[0] == 3 and actions.shape[1] == (action_dim or 4), (
                f"Expected continuous actions [3, {action_dim or 4}], got {actions.shape}"
            )
            pursuer_setpoints = actions.astype(np.float32) + residual
            pursuer_setpoints = self._clip_pursuer_setpoints(pursuer_setpoints)
            evader_setpoint = self._compute_evader_setpoint(backend_state, task_state)[None, :]
            # if self.debug:
            #     print("evader_setpoint=", evader_setpoint)
            joint_setpoints = np.concatenate([pursuer_setpoints, evader_setpoint], axis=0).astype(np.float32)
            return joint_setpoints

        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape[0] == 3, f"Expected 3 pursuer actions, got {actions.shape}"
        pursuer_setpoints = self._action_table[actions] + residual
        pursuer_setpoints = self._clip_pursuer_setpoints(pursuer_setpoints)
        evader_setpoint = self._compute_evader_setpoint(backend_state, task_state)[None, :]
        joint_setpoints = np.concatenate([pursuer_setpoints, evader_setpoint], axis=0).astype(np.float32)
        return joint_setpoints

    # ---------------------------------------------------------------------
    # helper: normalization / clipping / OOB
    # ---------------------------------------------------------------------
    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        p = np.asarray(pos, dtype=np.float32)
        out = np.empty_like(p, dtype=np.float32)
        out[..., 0] = p[..., 0] / self.pos_xy_norm
        out[..., 1] = p[..., 1] / self.pos_xy_norm
        out[..., 2] = (p[..., 2] - self.z_center) / self.pos_z_norm
        return out

    def _normalize_delta(self, delta: np.ndarray) -> np.ndarray:
        d = np.asarray(delta, dtype=np.float32)
        out = np.empty_like(d, dtype=np.float32)
        out[..., 0] = d[..., 0] / self.pos_xy_norm
        out[..., 1] = d[..., 1] / self.pos_xy_norm
        out[..., 2] = d[..., 2] / self.pos_z_norm
        return out

    def _normalize_velocity(self, vel: np.ndarray) -> np.ndarray:
        v = np.asarray(vel, dtype=np.float32)
        out = np.empty_like(v, dtype=np.float32)
        out[..., 0] = v[..., 0] / self.vel_xy_norm
        out[..., 1] = v[..., 1] / self.vel_xy_norm
        out[..., 2] = v[..., 2] / self.vel_z_norm
        return out

    def _normalize_angle(self, ang: np.ndarray) -> np.ndarray:
        a = np.asarray(ang, dtype=np.float32)
        return (a / self.angle_norm).astype(np.float32)

    def _get_oob_mask(self, pos: np.ndarray) -> np.ndarray:
        """pos: [M, 3]"""
        p = np.asarray(pos, dtype=np.float32)
        return (
            (np.abs(p[:, 0]) > self.world_xy)
            | (np.abs(p[:, 1]) > self.world_xy)
            | (p[:, 2] < self.z_min)
            | (p[:, 2] > self.z_max)
        )

    def _clip_positions_inside(self, pos: np.ndarray, margin_xy: float = 0.0, margin_z: float = 0.0) -> np.ndarray:
        p = np.asarray(pos, dtype=np.float32).copy()
        x_low = -self.world_xy + margin_xy
        x_high = self.world_xy - margin_xy
        z_low = self.z_min + margin_z
        z_high = self.z_max - margin_z
        p[..., 0] = np.clip(p[..., 0], x_low, x_high)
        p[..., 1] = np.clip(p[..., 1], x_low, x_high)
        p[..., 2] = np.clip(p[..., 2], z_low, z_high)
        return p.astype(np.float32)

    def _nearest_point_in_arena(self, pos: np.ndarray) -> np.ndarray:
        """将位置投影到合法盒 [|x|,|y|<=world_xy, z_min<=z<=z_max] 内。"""
        p = np.asarray(pos, dtype=np.float32).reshape(3)
        return np.array(
            [
                float(np.clip(p[0], -self.world_xy, self.world_xy)),
                float(np.clip(p[1], -self.world_xy, self.world_xy)),
                float(np.clip(p[2], self.z_min, self.z_max)),
            ],
            dtype=np.float32,
        )

    # ---------------------------------------------------------------------
    # helper: evader APF
    # ---------------------------------------------------------------------
    def _smoothstep01(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _wall_threat(self, clearance: float, margin: float) -> float:
        if margin <= 1e-8:
            return 0.0
        return self._smoothstep01((margin - float(clearance)) / float(margin))

    def _evader_apf_pursuer_repulsion(self, evader_pos: np.ndarray, pursuer_pos: np.ndarray) -> np.ndarray:
        """
        更稳健的 pursuer 排斥：
        - 同时考虑最近 pursuer 与 pursuer 质心；
        - 最近 pursuer 权重大于质心，避免只看质心导致被单机贴边驱赶。
        """
        e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
        p = np.asarray(pursuer_pos, dtype=np.float64).reshape(-1, 3)
        if p.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        center = np.mean(p, axis=0)
        vec_center = e - center
        dist_center = float(np.linalg.norm(vec_center))
        u_center = vec_center / max(dist_center, 1e-8)

        dists = np.linalg.norm(p - e[None, :], axis=1)
        nearest = p[int(np.argmin(dists))]
        vec_nearest = e - nearest
        dist_nearest = float(np.linalg.norm(vec_nearest))
        u_nearest = vec_nearest / max(dist_nearest, 1e-8)

        nearest_boost = 1.0 / max(dist_nearest / max(self.capture_dist, 1e-6), 1.0)
        rep = (
            float(self.evader_apf_pursuer_gain) * u_center
            + float(self.evader_nearest_pursuer_gain) * nearest_boost * u_nearest
        )
        return rep.astype(np.float32)

    def _evader_apf_boundary_repulsion(self, evader_pos: np.ndarray) -> tuple[np.ndarray, float]:
        """
        连续边界势：
        - 返回指向场内的连续 inward 向量，而非二值跳变；
        - 同时返回 threat in [0, 1]，表示离边界有多危险。
        """
        p = np.asarray(evader_pos, dtype=np.float64).reshape(3)
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        wx = float(self.world_xy)
        z_lo, z_hi = float(self.z_min), float(self.z_max)
        mxy = float(self.evader_margin_xy)
        mz = float(self.evader_margin_z)

        clear_px = wx - x
        clear_nx = x + wx
        clear_py = wx - y
        clear_ny = y + wx
        clear_pz = z_hi - z
        clear_nz = z - z_lo

        w_px = self._wall_threat(clear_px, mxy)
        w_nx = self._wall_threat(clear_nx, mxy)
        w_py = self._wall_threat(clear_py, mxy)
        w_ny = self._wall_threat(clear_ny, mxy)
        w_pz = self._wall_threat(clear_pz, mz)
        w_nz = self._wall_threat(clear_nz, mz)

        inward = np.array(
            [
                -w_px + w_nx,
                -w_py + w_ny,
                -w_pz + w_nz,
            ],
            dtype=np.float32,
        )
        threat = float(max(w_px, w_nx, w_py, w_ny, w_pz, w_nz))
        return inward, threat

    def _compute_evader_setpoint(
        self,
        backend_state,
        task_state: PursuitEvasion3v1TaskState,
    ) -> np.ndarray:
        """
        更稳健的 evader 策略：
        1) 远离 pursuer：同时考虑最近 pursuer 和 pursuer 质心；
        2) 靠近边界时使用连续 inward barrier，而非进入 margin 才触发的二值力；
        3) 靠近边界时去掉“朝墙外”的分量，并加入沿墙切向滑移项，减少角点冲出；
        4) 靠近边界时主动减速，避免惯性式 OOB。
        """
        states = backend_state.states
        lin_pos = states[:, 3, :]
        pursuer_pos = lin_pos[task_state.pursuer_ids]
        evader_pos = np.asarray(lin_pos[task_state.evader_id], dtype=np.float32).reshape(3)

        nearest = self._nearest_point_in_arena(evader_pos)
        to_inside = nearest - evader_pos
        dist_out = float(np.linalg.norm(to_inside))
        if dist_out > 1e-5:
            move_dir = self._safe_normalize(to_inside)
            return np.array(
                [
                    move_dir[0] * self.evader_speed_xy,
                    move_dir[1] * self.evader_speed_xy,
                    0.0,
                    move_dir[2] * self.evader_speed_z,
                ],
                dtype=np.float32,
            )

        f_p = np.asarray(self._evader_apf_pursuer_repulsion(evader_pos, pursuer_pos), dtype=np.float32)
        f_b, boundary_threat = self._evader_apf_boundary_repulsion(evader_pos)

        center_pull = np.array(
            [
                -evader_pos[0] / max(self.world_xy, 1e-6),
                -evader_pos[1] / max(self.world_xy, 1e-6),
                (self.z_center - evader_pos[2]) / max(self.z_half_span, 1e-6),
            ],
            dtype=np.float32,
        )
        center_pull = self._safe_normalize(center_pull)

        inward_hat = self._safe_normalize(f_b)
        safe_escape = f_p.copy()
        tangent = np.zeros(3, dtype=np.float32)
        if boundary_threat > 1e-6 and float(np.linalg.norm(inward_hat)) > 1e-8:
            proj_inward = float(np.dot(safe_escape, inward_hat))
            if proj_inward < 0.0:
                safe_escape = safe_escape - proj_inward * inward_hat

            tangent = safe_escape - float(np.dot(safe_escape, inward_hat)) * inward_hat
            if float(np.linalg.norm(tangent[:2])) < 1e-8 and float(np.linalg.norm(inward_hat[:2])) > 1e-8:
                tangent = np.array([-inward_hat[1], inward_hat[0], 0.0], dtype=np.float32)
            tangent = self._safe_normalize(tangent)

        desired = (
            safe_escape
            + float(self.evader_boundary_gain) * boundary_threat * inward_hat
            + float(self.evader_boundary_tangent_gain) * boundary_threat * tangent
            + float(self.evader_center_pull_gain) * (boundary_threat ** 2) * center_pull
        )

        move_dir = self._safe_normalize(desired)
        if float(np.linalg.norm(move_dir)) < 1e-8:
            move_dir = self._safe_normalize(center_pull)

        slowdown = 1.0 - (1.0 - float(self.evader_boundary_slowdown_min)) * boundary_threat
        slowdown = float(np.clip(slowdown, self.evader_boundary_slowdown_min, 1.0))

        return np.array(
            [
                move_dir[0] * self.evader_speed_xy * slowdown,
                move_dir[1] * self.evader_speed_xy * slowdown,
                0.0,
                move_dir[2] * self.evader_speed_z * slowdown,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _safe_normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < eps:
            return np.zeros_like(vec, dtype=np.float32)
        return (vec / norm).astype(np.float32)
