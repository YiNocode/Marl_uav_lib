"""Control Barrier Function (CBF) lightweight projection filter (deployable)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle, inflate_obstacle
from marl_uav.framework.geometry.obstacle_query import select_cbf_obstacles


@dataclass
class CBFConfig:
    enabled: bool = True
    solver: str = "projection"
    alpha_obstacle: float = 2.0
    alpha_agent: float = 2.0
    obstacle_activation_radius: float = 2.0
    agent_activation_radius: float = 1.5
    max_projection_iters: int = 5
    safety_margin: float = 0.3
    min_agent_sep: float = 0.6
    speed_clip: bool = True
    infeasible_fallback: str = "zero_velocity"
    max_filter_time_ms: float = 2.0
    uav_radius: float = 0.15
    obstacle_query: str = "radius_forward"
    forward_range: float = 3.0
    forward_cone_half_deg: float = 90.0
    predictive_enabled: bool = True
    prediction_horizon_s: float = 1.5
    predictive_extra_margin: float = 0.15
    predictive_min_speed: float = 0.03
    emergency_clearance: float = 0.0
    emergency_escape_speed: float = 0.12
    local_escape_enabled: bool = True
    local_escape_extra_clearance: float = 0.15

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> CBFConfig:
        raw = dict(cfg or {})
        solver = str(raw.get("solver", "projection")).strip().lower()
        if solver in ("auto", "slsqp", "qp"):
            solver = "projection"
        return cls(
            enabled=bool(raw.get("enabled", True)),
            solver=solver,
            alpha_obstacle=float(raw.get("alpha_obstacle", 2.0)),
            alpha_agent=float(raw.get("alpha_agent", 2.0)),
            obstacle_activation_radius=float(raw.get("obstacle_activation_radius", 2.0)),
            agent_activation_radius=float(raw.get("agent_activation_radius", 1.5)),
            max_projection_iters=int(raw.get("max_projection_iters", raw.get("max_proj_iters", 5))),
            safety_margin=float(raw.get("safety_margin", 0.3)),
            min_agent_sep=float(raw.get("min_agent_sep", 0.6)),
            speed_clip=bool(raw.get("speed_clip", True)),
            infeasible_fallback=str(raw.get("infeasible_fallback", "zero_velocity")),
            max_filter_time_ms=float(raw.get("max_filter_time_ms", 2.0)),
            uav_radius=float(raw.get("uav_radius", 0.15)),
            obstacle_query=str(raw.get("obstacle_query", "radius_forward")).strip().lower(),
            forward_range=float(raw.get("forward_range", 3.0)),
            forward_cone_half_deg=float(raw.get("forward_cone_half_deg", 90.0)),
            predictive_enabled=bool(raw.get("predictive_enabled", True)),
            prediction_horizon_s=float(raw.get("prediction_horizon_s", 1.5)),
            predictive_extra_margin=float(raw.get("predictive_extra_margin", 0.15)),
            predictive_min_speed=float(raw.get("predictive_min_speed", 0.03)),
            emergency_clearance=float(raw.get("emergency_clearance", 0.0)),
            emergency_escape_speed=float(raw.get("emergency_escape_speed", 0.12)),
            local_escape_enabled=bool(raw.get("local_escape_enabled", True)),
            local_escape_extra_clearance=float(raw.get("local_escape_extra_clearance", 0.15)),
        )


def _count_violations(A_rows: list[np.ndarray], b_vals: list[float], u: np.ndarray) -> int:
    n = 0
    for a, b in zip(A_rows, b_vals):
        if float(np.dot(a, u)) < b - 1e-6:
            n += 1
    return n


def _project_halfplanes(
    u0: np.ndarray,
    A_rows: list[np.ndarray],
    b_vals: list[float],
    *,
    u_min: np.ndarray,
    u_max: np.ndarray,
    max_iters: int,
) -> np.ndarray:
    u = np.asarray(u0, dtype=np.float64).reshape(2).copy()
    for _ in range(max_iters):
        changed = False
        for a, b in zip(A_rows, b_vals):
            val = float(np.dot(a, u))
            if val < b:
                denom = float(np.dot(a, a))
                if denom < 1e-12:
                    continue
                u = u + ((b - val) / denom) * a
                changed = True
        u = np.clip(u, u_min, u_max)
        if not changed:
            break
    return u


def _predicted_obstacle_hit(
    p: np.ndarray,
    u: np.ndarray,
    obstacle: Obstacle,
    *,
    uav_radius: float,
    safety_margin: float,
    extra_margin: float,
    horizon_s: float,
    min_speed: float,
) -> tuple[bool, float, float]:
    """Predict whether current velocity sweeps into an inflated obstacle."""
    speed = float(np.linalg.norm(u))
    if speed < max(float(min_speed), 1e-9) or obstacle.kind != "circle":
        return False, float("inf"), float("inf")
    infl = inflate_obstacle(obstacle, uav_radius=uav_radius, safety_margin=safety_margin)
    c = np.asarray(infl.center, dtype=np.float64).reshape(2)
    rel = p - c
    uu = float(np.dot(u, u))
    t_star = -float(np.dot(rel, u)) / max(uu, 1e-12)
    t_hit = float(np.clip(t_star, 0.0, max(float(horizon_s), 0.0)))
    closest = rel + t_hit * u
    d_closest = float(np.linalg.norm(closest))
    hit_radius = float(infl.radius) + max(float(extra_margin), 0.0)
    closing = float(np.dot(rel, u)) < 0.0
    return bool(closing and d_closest <= hit_radius), t_hit, d_closest - hit_radius


def _clip_speed_to_box(u: np.ndarray, u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    out = np.asarray(u, dtype=np.float64).reshape(2).copy()
    out = np.clip(out, u_min, u_max)
    if np.all(np.isfinite(u_max)):
        vmax = float(np.max(np.abs(u_max)))
        nrm = float(np.linalg.norm(out))
        if nrm > vmax and nrm > 1e-9:
            out = out * (vmax / nrm)
    return out


class LightweightCBFFilter:
    """Deployable CBF via half-plane projection; no scipy/cvxpy."""

    def filter(
        self,
        agent_pos: np.ndarray,
        u_nominal: np.ndarray,
        obstacles: list[Obstacle],
        other_agents: np.ndarray,
        cfg: CBFConfig,
        *,
        uav_radius: float | None = None,
        action_low_xy: np.ndarray | None = None,
        action_high_xy: np.ndarray | None = None,
        agent_yaw: float | None = None,
        all_obstacles: list[Obstacle] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        t0 = time.perf_counter()
        u_nom = np.asarray(u_nominal, dtype=np.float64).reshape(2)
        if not cfg.enabled or cfg.solver != "projection":
            return u_nom.copy(), {
                "cbf_active": False,
                "num_constraints": 0,
                "cbf_filter_time_ms": 0.0,
                "cbf_delta_norm": 0.0,
                "cbf_timeout_or_fallback": False,
            }

        p = np.asarray(agent_pos, dtype=np.float64).reshape(2)
        uav_r = float(uav_radius if uav_radius is not None else cfg.uav_radius)
        u_min = np.full(2, -np.inf) if action_low_xy is None else np.asarray(action_low_xy, dtype=np.float64).reshape(2)
        u_max = np.full(2, np.inf) if action_high_xy is None else np.asarray(action_high_xy, dtype=np.float64).reshape(2)

        A_rows: list[np.ndarray] = []
        b_vals: list[float] = []
        active_obstacle_indices: list[int] = []
        active_agent_indices: list[int] = []
        act_r_obs = float(cfg.obstacle_activation_radius)
        act_r_ag = float(cfg.agent_activation_radius)
        closest_obstacle_clearance = float("inf")
        closest_obstacle_dir: np.ndarray | None = None

        obs_pool = list(all_obstacles if all_obstacles is not None else obstacles)
        yaw_val = float(agent_yaw) if agent_yaw is not None else float(np.arctan2(u_nom[1], u_nom[0]))
        queried, global_idx = select_cbf_obstacles(
            p, yaw_val, u_nom, obs_pool,
            activation_radius=act_r_obs,
            forward_range=float(cfg.forward_range),
            forward_cone_half_deg=float(cfg.forward_cone_half_deg),
            mode=str(cfg.obstacle_query),
        )
        if cfg.predictive_enabled:
            seen = {int(i) for i in global_idx}
            speed = float(np.linalg.norm(u_nom))
            reach = (
                speed * max(float(cfg.prediction_horizon_s), 0.0)
                + uav_r
                + float(cfg.safety_margin)
                + float(cfg.predictive_extra_margin)
            )
            for obs_i, obs in enumerate(obs_pool):
                if int(obs_i) in seen:
                    continue
                c = np.asarray(obs.center, dtype=np.float64).reshape(2)
                if float(np.linalg.norm(c - p)) > reach + float(obs.radius):
                    continue
                predicted, _ttc, _clearance = _predicted_obstacle_hit(
                    p,
                    u_nom,
                    obs,
                    uav_radius=uav_r,
                    safety_margin=cfg.safety_margin,
                    extra_margin=cfg.predictive_extra_margin,
                    horizon_s=cfg.prediction_horizon_s,
                    min_speed=cfg.predictive_min_speed,
                )
                if predicted:
                    queried.append(obs)
                    global_idx.append(int(obs_i))
                    seen.add(int(obs_i))

        for obs_i, obs in zip(global_idx, queried):
            if obs.kind != "circle":
                continue
            infl = inflate_obstacle(obs, uav_radius=uav_r, safety_margin=cfg.safety_margin)
            c = np.asarray(infl.center, dtype=np.float64).reshape(2)
            diff = p - c
            dist = float(np.linalg.norm(diff))
            clearance = dist - float(infl.radius)
            if clearance < closest_obstacle_clearance:
                closest_obstacle_clearance = clearance
                if dist > 1e-9:
                    closest_obstacle_dir = diff / dist
                else:
                    fallback_dir = -u_nom
                    fallback_n = float(np.linalg.norm(fallback_dir))
                    closest_obstacle_dir = (
                        fallback_dir / fallback_n
                        if fallback_n > 1e-9
                        else np.array([1.0, 0.0], dtype=np.float64)
                    )
            h = float(np.dot(diff, diff) - float(infl.radius) ** 2)
            A_rows.append(2.0 * diff)
            b_vals.append(-cfg.alpha_obstacle * h)
            active_obstacle_indices.append(int(obs_i))
            if cfg.predictive_enabled:
                predicted, _ttc, _clearance = _predicted_obstacle_hit(
                    p,
                    u_nom,
                    obs,
                    uav_radius=uav_r,
                    safety_margin=cfg.safety_margin,
                    extra_margin=cfg.predictive_extra_margin,
                    horizon_s=cfg.prediction_horizon_s,
                    min_speed=cfg.predictive_min_speed,
                )
                if predicted:
                    A_rows.append(diff.copy())
                    b_vals.append(0.0)

        others = np.asarray(other_agents, dtype=np.float64).reshape(-1, 2)
        d_min = float(cfg.min_agent_sep)
        for k in range(int(others.shape[0])):
            pk = others[k]
            dist = float(np.linalg.norm(p - pk))
            if dist > act_r_ag + d_min:
                continue
            diff = p - pk
            h = float(np.dot(diff, diff) - d_min ** 2)
            A_rows.append(2.0 * diff)
            b_vals.append(-cfg.alpha_agent * h)
            active_agent_indices.append(int(k))

        violated_before = _count_violations(A_rows, b_vals, u_nom)
        u_safe = _project_halfplanes(
            u_nom, A_rows, b_vals,
            u_min=u_min, u_max=u_max,
            max_iters=cfg.max_projection_iters,
        )
        if cfg.speed_clip and np.all(np.isfinite(u_max)):
            vmax = float(np.max(np.abs(u_max)))
            nrm = float(np.linalg.norm(u_safe))
            if nrm > vmax and nrm > 1e-9:
                u_safe = u_safe * (vmax / nrm)

        if (
            closest_obstacle_dir is not None
            and closest_obstacle_clearance < float(cfg.emergency_clearance)
        ):
            emergency_band = max(float(cfg.emergency_clearance), 1e-6)
            severity = float(np.clip((emergency_band - closest_obstacle_clearance) / emergency_band, 0.0, 1.0))
            min_outward = max(float(cfg.emergency_escape_speed), 0.0) * severity
            outward = float(np.dot(u_safe, closest_obstacle_dir))
            if outward < min_outward:
                u_safe = u_safe + (min_outward - outward) * closest_obstacle_dir
                u_safe = _clip_speed_to_box(u_safe, u_min, u_max)

        violated_after = _count_violations(A_rows, b_vals, u_safe)
        fallback = False
        fallback_mode = str(cfg.infeasible_fallback).strip().lower()
        if violated_after > 0 and fallback_mode in {"outward_velocity", "escape", "repulsive"}:
            if closest_obstacle_dir is not None:
                speed = max(float(cfg.emergency_escape_speed), float(np.linalg.norm(u_nom)))
                if np.all(np.isfinite(u_max)):
                    speed = min(speed, float(np.max(np.abs(u_max))))
                u_safe = _clip_speed_to_box(speed * closest_obstacle_dir, u_min, u_max)
            else:
                u_safe = np.zeros(2, dtype=np.float64)
            fallback = True
        elif violated_after > 0 and fallback_mode == "zero_velocity":
            u_safe = np.zeros(2, dtype=np.float64)
            fallback = True

        elapsed = (time.perf_counter() - t0) * 1000.0
        timed_out = elapsed > cfg.max_filter_time_ms

        return u_safe, {
            "cbf_active": bool(violated_before > 0 or len(A_rows) > 0),
            "num_constraints": len(A_rows),
            "num_violated_before": int(violated_before),
            "num_violated_after": int(_count_violations(A_rows, b_vals, u_safe)),
            "cbf_delta_norm": float(np.linalg.norm(u_safe - u_nom)),
            "cbf_filter_time_ms": elapsed,
            "cbf_timeout_or_fallback": bool(fallback or timed_out),
            "qp_success": _count_violations(A_rows, b_vals, u_safe) == 0,
            "cbf_action_delta_norm": float(np.linalg.norm(u_safe - u_nom)),
            "cbf_active_obstacle_indices": active_obstacle_indices,
            "cbf_active_agent_indices": active_agent_indices,
            "cbf_nominal_speed_xy": float(np.linalg.norm(u_nom)),
            "cbf_safe_speed_xy": float(np.linalg.norm(u_safe)),
        }


_CBF_SINGLETON = LightweightCBFFilter()


def apply_cbf_filter(
    agent_pos_xy: np.ndarray,
    u_nominal_xy: np.ndarray,
    obstacles: list[Obstacle],
    other_agent_positions_xy: np.ndarray,
    other_agent_nominal_actions_xy: np.ndarray | None = None,
    cfg: CBFConfig | dict[str, Any] | None = None,
    *,
    uav_radius: float = 0.15,
    action_low_xy: np.ndarray | None = None,
    action_high_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible wrapper around ``LightweightCBFFilter``."""
    del other_agent_nominal_actions_xy
    ccfg = cfg if isinstance(cfg, CBFConfig) else CBFConfig.from_dict(cfg)
    return _CBF_SINGLETON.filter(
        agent_pos_xy, u_nominal_xy, obstacles, other_agent_positions_xy, ccfg,
        uav_radius=uav_radius,
        action_low_xy=action_low_xy,
        action_high_xy=action_high_xy,
    )
