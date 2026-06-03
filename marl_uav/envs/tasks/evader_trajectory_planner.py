"""Pre-simulation evader escape trajectory with sharp turns (pursuer + obstacle aware)."""

from __future__ import annotations

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import (
    Obstacle,
    collision_check_path,
)
from marl_uav.framework.planning.visibility_path_planner import plan_path


def _obstacles_from_arrays(
    obstacle_xy: np.ndarray,
    obstacle_r: np.ndarray,
) -> list[Obstacle]:
    xy = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1, 2)
    rr = np.asarray(obstacle_r, dtype=np.float64).reshape(-1)
    if xy.shape[0] == 0:
        return []
    n = min(xy.shape[0], rr.shape[0])
    return [
        Obstacle(kind="circle", center=xy[i].copy(), radius=float(rr[i]))
        for i in range(n)
    ]


def _wrap_angle_diff(a: float, b: float) -> float:
    return float(abs((a - b + np.pi) % (2.0 * np.pi) - np.pi))


def _clip_xy(xy: np.ndarray, world_xy: float, margin: float) -> np.ndarray:
    lim = float(world_xy) - float(margin)
    out = np.asarray(xy, dtype=np.float64).reshape(2).copy()
    out[0] = float(np.clip(out[0], -lim, lim))
    out[1] = float(np.clip(out[1], -lim, lim))
    return out


def _xy_clear(
    xy: np.ndarray,
    obstacles: list[Obstacle],
    clearance: float,
) -> bool:
    p = np.asarray(xy, dtype=np.float64).reshape(2)
    for obs in obstacles:
        c = np.asarray(obs.center, dtype=np.float64).reshape(2)
        if float(np.linalg.norm(p - c)) <= float(obs.radius) + clearance:
            return False
    return True


def _min_pursuer_dist(evader_xy: np.ndarray, pursuer_xy: np.ndarray) -> float:
    e = np.asarray(evader_xy, dtype=np.float64).reshape(2)
    p = np.asarray(pursuer_xy, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(p - e.reshape(1, 2), axis=1)))


def _clip_flight_altitude(z: float, z_min: float | None, z_max: float | None) -> float:
    out = float(z)
    if z_min is not None:
        out = max(out, float(z_min))
    if z_max is not None:
        out = min(out, float(z_max))
    return out


def path_hold_altitude(path: np.ndarray) -> float:
    """Reference hold altitude for a planned evader polyline."""
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0
    return float(pts[0, 2])


def apply_path_altitude(path: np.ndarray, hold_z: float) -> np.ndarray:
    """Force every waypoint to the same flight altitude."""
    out = np.asarray(path, dtype=np.float32).reshape(-1, 3).copy()
    if out.shape[0] == 0:
        return out
    out[:, 2] = np.float32(hold_z)
    return out


def _escape_direction(evader_xy: np.ndarray, pursuer_xy: np.ndarray) -> np.ndarray:
    """
    Reference escape heading: away from pursuer centroid + nearest pursuer
    (consistent with evader APF repulsion logic).
    """
    e = np.asarray(evader_xy, dtype=np.float64).reshape(2)
    p = np.asarray(pursuer_xy, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] == 0:
        return np.array([1.0, 0.0], dtype=np.float64)

    center = np.mean(p, axis=0)
    to_evader = e - center
    dist_c = float(np.linalg.norm(to_evader))
    u_center = to_evader / max(dist_c, 1e-8)

    dists = np.linalg.norm(p - e.reshape(1, 2), axis=1)
    nearest = p[int(np.argmin(dists))]
    to_nearest = e - nearest
    dist_n = float(np.linalg.norm(to_nearest))
    u_nearest = to_nearest / max(dist_n, 1e-8)

    rep = u_center + 0.8 * u_nearest
    nrm = float(np.linalg.norm(rep))
    if nrm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (rep / nrm).astype(np.float64)


def _candidate_escape_headings(
    escape_hat: np.ndarray,
    prev_heading: float | None,
    min_turn_rad: float,
    jitter_deg: float,
) -> list[float]:
    base = float(np.arctan2(float(escape_hat[1]), float(escape_hat[0])))
    jitter = np.deg2rad(float(jitter_deg))
    offsets = [0.0, 0.5 * jitter, -0.5 * jitter, jitter, -jitter, 1.5 * jitter, -1.5 * jitter]
    headings: list[float] = []
    for off in offsets:
        h = base + float(off)
        if prev_heading is not None and _wrap_angle_diff(h, prev_heading) < min_turn_rad:
            continue
        headings.append(h)
    if not headings:
        if prev_heading is None:
            headings = [base]
        else:
            for sign in (1.0, -1.0):
                h = prev_heading + sign * min_turn_rad
                headings.append(h)
    return headings


def plan_sharp_turn_evader_path(
    start_xyz: np.ndarray,
    pursuer_xyz: np.ndarray,
    obstacle_xy: np.ndarray,
    obstacle_r: np.ndarray,
    *,
    world_xy: float,
    rng: np.random.Generator,
    num_legs: int = 4,
    min_leg_m: float = 8.0,
    min_turn_deg: float = 60.0,
    safety_margin: float = 0.35,
    uav_radius: float = 0.15,
    arena_margin_ratio: float = 0.12,
    z_min: float | None = None,
    z_max: float | None = None,
    escape_angle_jitter_deg: float = 40.0,
    w_escape_dist: float = 3.0,
    w_escape_progress: float = 1.0,
    max_goal_trials: int = 80,
    planner_cfg: dict | None = None,
) -> np.ndarray:
    """
    Build a pre-sim escape polyline:

    - **Primary objective**: each leg moves away from pursuers (min distance gain + progress along escape axis).
    - **Constraints**: obstacle-free visibility segments, deliberate heading changes between legs.
    - **Altitude**: entire polyline stays at clipped start altitude (horizontal escape only).
    """
    start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    pursuer_xy = np.asarray(pursuer_xyz, dtype=np.float64).reshape(-1, 3)[:, :2]
    hold_z = _clip_flight_altitude(float(start[2]), z_min, z_max)
    margin = float(arena_margin_ratio) * float(world_xy)
    bounds = (-world_xy, -world_xy, world_xy, world_xy)
    obstacles = _obstacles_from_arrays(obstacle_xy, obstacle_r)
    cfg = dict(planner_cfg or {})
    cfg.setdefault("num_obstacle_samples", 16)
    cfg.setdefault("clearance", 0.25)
    cfg.setdefault("shortcut_smoothing", False)

    poly_xy: list[np.ndarray] = [_clip_xy(start[:2], world_xy, margin)]
    current = poly_xy[-1].copy()
    prev_heading: float | None = None
    min_turn_rad = np.deg2rad(float(min_turn_deg))
    min_leg = max(float(min_leg_m), 1.0)
    obs_clear = safety_margin + uav_radius

    for _leg in range(max(int(num_legs), 1)):
        escape_hat = _escape_direction(current, pursuer_xy)
        headings = _candidate_escape_headings(
            escape_hat, prev_heading, min_turn_rad, escape_angle_jitter_deg,
        )
        dists = [float(rng.uniform(min_leg, min_leg * 1.35)) for _ in headings]
        rng.shuffle(headings)

        best_score = -float("inf")
        best_segment: list[np.ndarray] | None = None
        best_goal: np.ndarray | None = None
        best_heading: float | None = None
        base_min_dist = _min_pursuer_dist(current, pursuer_xy)

        trials = 0
        for heading in headings:
            for dist_scale in (1.0, 1.15, 0.9):
                if trials >= max_goal_trials:
                    break
                trials += 1
                dist = min_leg * dist_scale if dist_scale != 1.0 else float(rng.uniform(min_leg, min_leg * 1.35))
                goal = _clip_xy(
                    current + dist * np.array([np.cos(heading), np.sin(heading)], dtype=np.float64),
                    world_xy,
                    margin,
                )
                if float(np.linalg.norm(goal - current)) < min_leg * 0.75:
                    continue
                if not _xy_clear(goal, obstacles, clearance=obs_clear):
                    continue

                segment = plan_path(
                    current, goal, obstacles, bounds=bounds, cfg=cfg,
                    safety_margin=safety_margin, uav_radius=uav_radius,
                )
                if segment is None or len(segment) < 2:
                    continue
                if not collision_check_path(
                    segment, obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
                ):
                    continue

                seg_end = np.asarray(segment[-1], dtype=np.float64).reshape(2)
                min_dist_gain = _min_pursuer_dist(seg_end, pursuer_xy) - base_min_dist
                progress = float(np.dot(seg_end - current, escape_hat))
                turn_bonus = 0.0
                if prev_heading is not None:
                    turn_bonus = 0.15 * _wrap_angle_diff(heading, prev_heading)

                score = (
                    w_escape_dist * min_dist_gain
                    + w_escape_progress * progress
                    + turn_bonus
                )
                if score > best_score:
                    best_score = score
                    best_segment = segment
                    best_goal = seg_end.copy()
                    best_heading = heading
            if trials >= max_goal_trials:
                break

        if best_segment is None or best_goal is None:
            # fallback: one escape step along repulsion direction
            fallback_goal = _clip_xy(
                current + min_leg * escape_hat,
                world_xy,
                margin,
            )
            segment = plan_path(
                current, fallback_goal, obstacles, bounds=bounds, cfg=cfg,
                safety_margin=safety_margin, uav_radius=uav_radius,
            )
            if segment is not None and len(segment) >= 2:
                for pt in segment[1:]:
                    poly_xy.append(np.asarray(pt, dtype=np.float64).reshape(2).copy())
                current = poly_xy[-1].copy()
                prev_heading = float(np.arctan2(escape_hat[1], escape_hat[0]))
            break

        for pt in best_segment[1:]:
            poly_xy.append(np.asarray(pt, dtype=np.float64).reshape(2).copy())
        current = poly_xy[-1].copy()
        prev_heading = float(best_heading if best_heading is not None else np.arctan2(
            current[1] - best_segment[0][1], current[0] - best_segment[0][0],
        ))

    if len(poly_xy) < 2:
        esc = _escape_direction(start[:2], pursuer_xy)
        fallback = np.stack(
            [start[:2], _clip_xy(start[:2] + min_leg * esc, world_xy, margin)],
            axis=0,
        )
        out = np.zeros((2, 3), dtype=np.float32)
        out[:, :2] = fallback.astype(np.float32)
        return apply_path_altitude(out, hold_z)

    out = np.zeros((len(poly_xy), 3), dtype=np.float32)
    out[:, :2] = np.stack(poly_xy, axis=0).astype(np.float32)
    return apply_path_altitude(out, hold_z)


def select_path_tracking_target(
    evader_pos: np.ndarray,
    path: np.ndarray,
    cursor: int,
    *,
    lookahead_m: float,
    accept_radius: float,
) -> tuple[np.ndarray, int]:
    """Advance cursor and pick lookahead target on the polyline (xy only; z = hold altitude)."""
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.asarray(evader_pos, dtype=np.float64).reshape(3), 0
    hold_z = path_hold_altitude(pts)
    pos = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    cur = int(np.clip(cursor, 0, pts.shape[0] - 1))
    accept = max(float(accept_radius), 0.05)
    while cur < pts.shape[0] - 1 and float(np.linalg.norm(pts[cur, :2] - pos[:2])) <= accept:
        cur += 1

    target = pts[cur].copy()
    remaining = float(lookahead_m)
    idx = cur
    while remaining > 1e-6 and idx < pts.shape[0] - 1:
        seg = pts[idx + 1, :2] - pts[idx, :2]
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-6:
            idx += 1
            continue
        if remaining >= seg_len:
            remaining -= seg_len
            idx += 1
            target = pts[idx].copy()
        else:
            t = remaining / seg_len
            target[:2] = pts[idx, :2] + t * seg
            remaining = 0.0
    target[2] = hold_z
    return target, cur
