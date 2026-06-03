"""Turn-radius swept-corridor obstacle queries for local slot tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TurnRadiusObstacleQueryConfig:
    horizon_s: float = 1.5
    dt: float = 0.1
    vmax: float = 0.25
    omega_max: float = 1.0
    num_yaw_samples: int = 11
    speed_samples: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    min_turn_radius: float = 0.25
    lookahead_dist: float = 1.5
    uav_radius: float = 0.15
    safety_margin: float = 0.30
    query_extra_margin: float = 0.20
    use_candidate_rollout_filter: bool = True
    amax_xy: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TurnRadiusObstacleQueryConfig":
        d = dict(raw or {})
        speeds = d.get("speed_samples", cls.speed_samples)
        return cls(
            horizon_s=float(d.get("horizon_s", cls.horizon_s)),
            dt=float(d.get("dt", cls.dt)),
            vmax=float(d.get("vmax", cls.vmax)),
            omega_max=float(d.get("omega_max", cls.omega_max)),
            num_yaw_samples=int(d.get("num_yaw_samples", cls.num_yaw_samples)),
            speed_samples=tuple(float(x) for x in speeds),
            min_turn_radius=float(d.get("min_turn_radius", cls.min_turn_radius)),
            lookahead_dist=float(d.get("lookahead_dist", cls.lookahead_dist)),
            uav_radius=float(d.get("uav_radius", cls.uav_radius)),
            safety_margin=float(d.get("safety_margin", cls.safety_margin)),
            query_extra_margin=float(d.get("query_extra_margin", cls.query_extra_margin)),
            use_candidate_rollout_filter=bool(
                d.get("use_candidate_rollout_filter", cls.use_candidate_rollout_filter)
            ),
            amax_xy=None if d.get("amax_xy") is None else float(d["amax_xy"]),
        )


def wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def effective_omega_max(vmax: float, omega_max: float, min_turn_radius: float) -> float:
    vmax = max(float(vmax), 0.0)
    omega = max(float(omega_max), 0.0)
    radius = max(float(min_turn_radius), 1e-6)
    if vmax <= 1e-12:
        return omega
    return float(min(omega, vmax / radius))


def resolve_speed_samples(
    vmax: float,
    samples: tuple[float, ...] | list[float],
    *,
    include_brake: bool = False,
) -> np.ndarray:
    vmax = max(float(vmax), 0.0)
    vals: list[float] = []
    for raw in samples:
        x = max(float(raw), 0.0)
        vals.append(x * vmax if x <= 1.0 + 1e-9 else min(x, vmax))
    if include_brake:
        vals.append(0.0)
    vals = sorted(set(round(v, 10) for v in vals), reverse=True)
    return np.asarray(vals, dtype=np.float64)


def default_amax_xy(vmax: float, horizon_s: float, dt: float) -> float:
    """Conservative XY acceleration limit for inertial rollout."""
    vmax = max(float(vmax), 0.0)
    horizon = max(float(horizon_s), float(dt), 1e-3)
    step = max(float(dt), 1e-3)
    return max(vmax / max(0.5 * horizon, step), 1e-3)


def rollout_unicycle_paths_batch_inertial(
    pos_xy: np.ndarray,
    yaw: float,
    v_world_xy: np.ndarray,
    speeds: np.ndarray,
    yaw_rates: np.ndarray,
    *,
    horizon_s: float,
    dt: float,
    amax_xy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out unicycle candidates with bounded acceleration from current world velocity."""
    pos0 = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    v0 = np.asarray(v_world_xy, dtype=np.float64).reshape(2)
    speed_arr = np.asarray(speeds, dtype=np.float64).reshape(-1)
    yaw_rate_arr = np.asarray(yaw_rates, dtype=np.float64).reshape(-1)
    step = max(float(dt), 1e-3)
    n = max(1, int(np.ceil(max(float(horizon_s), step) / step)))
    amax = max(float(amax_xy), 1e-6)
    max_dv = amax * step

    n_cand = speed_arr.shape[0]
    pos = np.tile(pos0[None, :], (n_cand, 1))
    vel = np.tile(v0[None, :], (n_cand, 1))
    yaw_h = np.full(n_cand, float(yaw), dtype=np.float64)
    pts = np.zeros((n_cand, n + 1, 2), dtype=np.float64)
    pts[:, 0, :] = pos0[None, :]
    for k in range(n):
        cmd = speed_arr[:, None] * np.stack([np.cos(yaw_h), np.sin(yaw_h)], axis=1)
        dv = cmd - vel
        dv_norm = np.linalg.norm(dv, axis=1)
        scale = np.where(dv_norm > max_dv, max_dv / np.maximum(dv_norm, 1e-9), 1.0)
        vel = vel + dv * scale[:, None]
        yaw_h = np.asarray([wrap_to_pi(y + step * yr) for y, yr in zip(yaw_h, yaw_rate_arr)])
        pos = pos + vel * step
        pts[:, k + 1, :] = pos
    return pts, yaw_h.astype(np.float64)


def rollout_unicycle_path(
    pos_xy: np.ndarray,
    yaw: float,
    speed: float,
    yaw_rate: float,
    *,
    horizon_s: float,
    dt: float,
) -> tuple[np.ndarray, float]:
    pos = np.asarray(pos_xy, dtype=np.float64).reshape(2).copy()
    yaw_i = float(yaw)
    step = max(float(dt), 1e-3)
    n = max(1, int(np.ceil(max(float(horizon_s), step) / step)))
    pts = np.zeros((n + 1, 2), dtype=np.float64)
    pts[0] = pos
    for k in range(1, n + 1):
        pos = pos + step * float(speed) * np.array([np.cos(yaw_i), np.sin(yaw_i)])
        yaw_i = wrap_to_pi(yaw_i + step * float(yaw_rate))
        pts[k] = pos
    return pts, yaw_i


def rollout_unicycle_paths_batch(
    pos_xy: np.ndarray,
    yaw: float,
    speeds: np.ndarray,
    yaw_rates: np.ndarray,
    *,
    horizon_s: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    speed_arr = np.asarray(speeds, dtype=np.float64).reshape(-1)
    yaw_rate_arr = np.asarray(yaw_rates, dtype=np.float64).reshape(-1)
    step = max(float(dt), 1e-3)
    n = max(1, int(np.ceil(max(float(horizon_s), step) / step)))
    t = np.arange(n, dtype=np.float64) * step
    headings = float(yaw) + yaw_rate_arr[:, None] * t[None, :]
    delta = step * speed_arr[:, None, None] * np.stack(
        [np.cos(headings), np.sin(headings)],
        axis=2,
    )
    pts = np.zeros((speed_arr.shape[0], n + 1, 2), dtype=np.float64)
    pts[:, 0, :] = pos[None, :]
    pts[:, 1:, :] = pos[None, None, :] + np.cumsum(delta, axis=1)
    final_yaws = np.asarray([wrap_to_pi(float(yaw) + step * n * yr) for yr in yaw_rate_arr], dtype=np.float64)
    return pts, final_yaws


def point_segment_distance(point: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    q = np.asarray(point, dtype=np.float64).reshape(2)
    a = np.asarray(p0, dtype=np.float64).reshape(2)
    b = np.asarray(p1, dtype=np.float64).reshape(2)
    v = b - a
    den = float(np.dot(v, v))
    if den <= 1e-18:
        return float(np.linalg.norm(q - a))
    t = float(np.clip(np.dot(q - a, v) / den, 0.0, 1.0))
    return float(np.linalg.norm(q - (a + t * v)))


def path_min_clearance(
    path_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
) -> float:
    pts = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2 or not obstacles:
        return float("inf")
    clearances = path_clearance_per_obstacle(
        pts,
        obstacles,
        uav_radius=uav_radius,
        safety_margin=safety_margin,
    )
    if clearances.size == 0:
        return float("inf")
    return float(np.min(clearances))


def path_clearance_per_obstacle(
    path_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
) -> np.ndarray:
    pts = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2 or not obstacles:
        return np.zeros(0, dtype=np.float64)
    centers: list[np.ndarray] = []
    radii: list[float] = []
    pad = float(uav_radius) + float(safety_margin)
    for obs in obstacles:
        if getattr(obs, "kind", "circle") != "circle":
            continue
        centers.append(np.asarray(obs.center, dtype=np.float64).reshape(2))
        radii.append(float(getattr(obs, "radius", 0.0)) + pad)
    if not centers:
        return np.zeros(0, dtype=np.float64)
    c = np.stack(centers, axis=0)
    r = np.asarray(radii, dtype=np.float64)
    p0 = pts[:-1]
    p1 = pts[1:]
    v = p1 - p0
    den = np.sum(v * v, axis=1)
    safe_den = np.maximum(den, 1e-18)
    w = c[:, None, :] - p0[None, :, :]
    t = np.sum(w * v[None, :, :], axis=2) / safe_den[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = p0[None, :, :] + t[:, :, None] * v[None, :, :]
    dist = np.linalg.norm(c[:, None, :] - proj, axis=2)
    return np.min(dist, axis=1) - r


def batch_path_min_clearance(
    paths_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
) -> np.ndarray:
    paths = np.asarray(paths_xy, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[1] < 2:
        return np.full(paths.shape[0] if paths.ndim >= 1 else 0, np.inf, dtype=np.float64)
    if not obstacles:
        return np.full(paths.shape[0], np.inf, dtype=np.float64)
    centers: list[np.ndarray] = []
    radii: list[float] = []
    pad = float(uav_radius) + float(safety_margin)
    for obs in obstacles:
        if getattr(obs, "kind", "circle") != "circle":
            continue
        centers.append(np.asarray(obs.center, dtype=np.float64).reshape(2))
        radii.append(float(getattr(obs, "radius", 0.0)) + pad)
    if not centers:
        return np.full(paths.shape[0], np.inf, dtype=np.float64)
    c = np.stack(centers, axis=0)
    r = np.asarray(radii, dtype=np.float64)
    p0 = paths[:, :-1, :]
    p1 = paths[:, 1:, :]
    v = p1 - p0
    den = np.sum(v * v, axis=2)
    safe_den = np.maximum(den, 1e-18)
    w = c[None, :, None, :] - p0[:, None, :, :]
    t = np.sum(w * v[:, None, :, :], axis=3) / safe_den[:, None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = p0[:, None, :, :] + t[:, :, :, None] * v[:, None, :, :]
    dist = np.linalg.norm(c[None, :, None, :] - proj, axis=3)
    clear = dist - r[None, :, None]
    return np.min(clear, axis=(1, 2))


def batch_paths_obstacle_collision_mask(
    paths_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
) -> np.ndarray:
    paths = np.asarray(paths_xy, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[1] < 2 or not obstacles:
        return np.zeros(len(obstacles), dtype=bool)
    centers: list[np.ndarray] = []
    radii: list[float] = []
    valid_indices: list[int] = []
    pad = float(uav_radius) + float(safety_margin)
    for idx, obs in enumerate(obstacles):
        if getattr(obs, "kind", "circle") != "circle":
            continue
        centers.append(np.asarray(obs.center, dtype=np.float64).reshape(2))
        radii.append(float(getattr(obs, "radius", 0.0)) + pad)
        valid_indices.append(idx)
    out = np.zeros(len(obstacles), dtype=bool)
    if not centers:
        return out
    c = np.stack(centers, axis=0)
    r = np.asarray(radii, dtype=np.float64)
    p0 = paths[:, :-1, :]
    p1 = paths[:, 1:, :]
    v = p1 - p0
    den = np.sum(v * v, axis=2)
    safe_den = np.maximum(den, 1e-18)
    w = c[None, :, None, :] - p0[:, None, :, :]
    t = np.sum(w * v[:, None, :, :], axis=3) / safe_den[:, None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = p0[:, None, :, :] + t[:, :, :, None] * v[:, None, :, :]
    dist = np.linalg.norm(c[None, :, None, :] - proj, axis=3)
    clear = dist - r[None, :, None]
    hit = np.min(clear, axis=(0, 2)) <= 0.0
    for idx, h in zip(valid_indices, hit):
        out[idx] = bool(h)
    return out


def path_collides_swept_circle(
    path_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
) -> bool:
    return path_min_clearance(
        path_xy,
        obstacles,
        uav_radius=uav_radius,
        safety_margin=safety_margin,
    ) <= 0.0


def path_collision_mask(
    path_xy: np.ndarray,
    obstacles: list[Any],
    *,
    uav_radius: float,
    safety_margin: float,
) -> np.ndarray:
    clearances = path_clearance_per_obstacle(
        path_xy,
        obstacles,
        uav_radius=uav_radius,
        safety_margin=safety_margin,
    )
    return clearances <= 0.0


def _candidate_rollouts(
    pos_xy: np.ndarray,
    yaw: float,
    cfg: TurnRadiusObstacleQueryConfig,
) -> list[np.ndarray]:
    omega = effective_omega_max(cfg.vmax, cfg.omega_max, cfg.min_turn_radius)
    yaws = np.linspace(-omega, omega, max(int(cfg.num_yaw_samples), 1), dtype=np.float64)
    speeds = resolve_speed_samples(cfg.vmax, cfg.speed_samples)
    out: list[np.ndarray] = []
    for speed in speeds:
        for yaw_rate in yaws:
            path, _ = rollout_unicycle_path(
                pos_xy,
                yaw,
                float(speed),
                float(yaw_rate),
                horizon_s=cfg.horizon_s,
                dt=cfg.dt,
            )
            out.append(path)
    return out


def query_turn_radius_obstacles(
    pos_xy: np.ndarray,
    yaw: float,
    target_xy: np.ndarray,
    obstacles: list[Any],
    cfg: dict[str, Any] | TurnRadiusObstacleQueryConfig | None,
    candidate_paths: list[np.ndarray] | np.ndarray | None = None,
) -> list[Any]:
    """Return obstacles intersecting the local turn-radius swept corridor.

    This is a geometric local query, not nearest-k filtering.  Obstacles keep
    their input order, and no fixed top-k truncation is applied.
    """
    qcfg = cfg if isinstance(cfg, TurnRadiusObstacleQueryConfig) else TurnRadiusObstacleQueryConfig.from_dict(cfg)
    pos = np.asarray(pos_xy, dtype=np.float64).reshape(2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(2)
    if not obstacles:
        return []

    max_obs_r = max(
        (float(getattr(o, "radius", 0.0)) for o in obstacles if getattr(o, "kind", "circle") == "circle"),
        default=0.0,
    )
    turn_radius = max(float(qcfg.min_turn_radius), float(qcfg.vmax) / max(float(qcfg.omega_max), 1e-6))
    query_radius = (
        float(qcfg.lookahead_dist)
        + turn_radius
        + max_obs_r
        + float(qcfg.uav_radius)
        + float(qcfg.safety_margin)
        + float(qcfg.query_extra_margin)
    )

    coarse: list[Any] = []
    for obs in obstacles:
        if getattr(obs, "kind", "circle") != "circle":
            continue
        center = np.asarray(obs.center, dtype=np.float64).reshape(2)
        if float(np.linalg.norm(center - pos)) <= query_radius:
            coarse.append(obs)
    if not coarse:
        return []

    rel = target - pos
    goal_dist = float(np.linalg.norm(rel))
    if goal_dist > 1e-9:
        goal_bearing = float(np.arctan2(rel[1], rel[0]))
    else:
        goal_bearing = float(yaw)
    _turn_need = wrap_to_pi(goal_bearing - float(yaw))

    forward_dist = min(max(float(qcfg.lookahead_dist), 0.0), max(goal_dist, float(qcfg.lookahead_dist)))
    forward = np.array([np.cos(float(yaw)), np.sin(float(yaw))], dtype=np.float64)
    forward_path = np.stack([pos, pos + forward_dist * forward], axis=0)
    selected: list[Any] = []
    selected_ids: set[int] = set()

    forward_mask = path_collision_mask(
        forward_path,
        coarse,
        uav_radius=qcfg.uav_radius,
        safety_margin=qcfg.safety_margin,
    )
    for obs, hit in zip(coarse, forward_mask):
        if bool(hit):
            selected.append(obs)
            selected_ids.add(id(obs))

    if qcfg.use_candidate_rollout_filter:
        paths = candidate_paths if candidate_paths is not None else _candidate_rollouts(pos, float(yaw), qcfg)
        remaining = [obs for obs in coarse if id(obs) not in selected_ids]
        if remaining:
            mask = batch_paths_obstacle_collision_mask(
                np.asarray(paths, dtype=np.float64),
                remaining,
                uav_radius=qcfg.uav_radius,
                safety_margin=qcfg.safety_margin,
            )
            for obs, hit in zip(remaining, mask):
                if bool(hit):
                    selected.append(obs)
                    selected_ids.add(id(obs))

    return selected
