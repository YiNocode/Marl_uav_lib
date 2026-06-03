"""Geometric non-learning baselines for 3v1 pursuit."""

from __future__ import annotations

from typing import Any

import numpy as np

from marl_uav.framework.planning.path_tracking import closest_point_on_polyline, path_tangent_bearing
from marl_uav.utils.control_timing import publish_control_timing, should_record_control_timing


def _empty_actions(action_low: np.ndarray) -> np.ndarray:
    adim = int(np.asarray(action_low).reshape(-1).shape[0])
    return np.zeros((3, adim), dtype=np.float32)


def _clip_actions(actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    return np.clip(np.asarray(actions, dtype=np.float32), low[None, :], high[None, :]).astype(np.float32)


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    ang = np.asarray(angles, dtype=np.float32)
    return np.arctan2(np.sin(ang), np.cos(ang)).astype(np.float32)


def pursuer_yaws_from_backend(backend_state: Any, pursuer_ids: np.ndarray) -> np.ndarray:
    """Read pursuer yaw (rad) from PyFlyt ang_pos channel."""
    ang_pos = np.asarray(backend_state.states[:, 1, :], dtype=np.float32)
    pids = np.asarray(pursuer_ids, dtype=np.int64).reshape(-1)
    return ang_pos[pids, 2].copy()


def default_proportional_gains(
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float | None,
    z_gain: float | None,
    yaw_gain: float | None,
) -> tuple[float, float, float]:
    """Default heuristic gains to env action bounds (unit error -> saturated command)."""
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    xy = float(xy_gain) if xy_gain is not None else float(high[0])
    zg = float(z_gain) if z_gain is not None else (float(high[3]) if high.shape[0] >= 4 else 0.20)
    yg = float(yaw_gain) if yaw_gain is not None else (float(high[2]) if high.shape[0] >= 3 else 0.0)
    return xy, zg, yg


def body_vel_to_world(vx_b: float, vy_b: float, yaw: float) -> tuple[float, float]:
    """Rotate body-frame XY velocity into world frame (yaw about +Z)."""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return vx_b * c - vy_b * s, vx_b * s + vy_b * c


def world_vel_to_body(vx_w: float, vy_w: float, yaw: float) -> tuple[float, float]:
    """Rotate world-frame XY velocity into body frame (yaw about +Z)."""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return vx_w * c + vy_w * s, -vx_w * s + vy_w * c


def analyze_tube_tracking_speed(
    pursuer_pos: np.ndarray,
    actions: np.ndarray,
    agent_paths_xy: list[list[list[float]] | None],
    action_high: np.ndarray,
    *,
    tube_radius: float,
    approach_dist: float,
    xy_gain: float,
    pursuer_yaw: np.ndarray | None,
) -> list[dict[str, float | str]]:
    """Speed breakdown for tube (cross-track constrained) path tracking."""
    p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    acts = np.asarray(actions, dtype=np.float32).reshape(3, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    vmax = float(high[0])
    yaw = (
        np.asarray(pursuer_yaw, dtype=np.float32).reshape(3)
        if pursuer_yaw is not None
        else np.zeros(3, dtype=np.float32)
    )
    tube_r = max(float(tube_radius), 0.05)
    out: list[dict[str, float | str]] = []
    for i in range(3):
        path = agent_paths_xy[i] if i < len(agent_paths_xy) else None
        act_xy = float(np.hypot(float(acts[i, 0]), float(acts[i, 1])))
        cross = 0.0
        tangent = 0.0
        if path and len(path) >= 2:
            _, tang, cross_signed, _ = closest_point_on_polyline(path, p[i, :2])
            cross = float(abs(cross_signed))
            tangent = float(np.arctan2(float(tang[1]), float(tang[0])))
        yaw_err = float(_wrap_to_pi(tangent - yaw[i])) if path else 0.0
        if path and len(path) >= 2:
            if cross <= tube_r:
                world_speed = vmax
                limit = "tube_cruise"
            elif cross <= tube_r * 2.0:
                world_speed = vmax * float(1.0 - 0.5 * (cross - tube_r) / tube_r)
                limit = "tube_slowdown"
            else:
                world_speed = vmax * 0.35
                limit = "tube_off_path"
        else:
            world_speed = min(float(xy_gain) * float(np.hypot(0.0, 0.0)), vmax)
            limit = "approach"
        out.append(
            {
                "dist_xy": cross,
                "cross_track_xy": cross,
                "tube_radius": tube_r,
                "tangent_bearing_deg": float(np.rad2deg(tangent)),
                "yaw_err_deg": float(np.rad2deg(yaw_err)),
                "align_factor": float(np.cos(yaw_err)),
                "speed_cmd_xy": act_xy,
                "world_speed_cmd_xy": float(world_speed),
                "speed_cap_xy": vmax,
                "limit_reason": limit,
            }
        )
    return out


def tube_tracking_body_actions(
    pursuer_pos: np.ndarray,
    track_targets: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    agent_paths_xy: list[list[list[float]] | None],
    *,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None,
    yaw_gain: float = 0.0,
    tube_radius: float = 0.35,
    cross_track_gain: float = 1.0,
    approach_dist: float = 0.5,
) -> np.ndarray:
    """Tube tracking: along-path speed + cross-track correction within ``tube_radius``."""
    p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    g = np.asarray(track_targets, dtype=np.float32).reshape(3, 3)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    out = _empty_actions(action_low)
    yaw = (
        np.asarray(pursuer_yaw, dtype=np.float32).reshape(3)
        if pursuer_yaw is not None
        else np.zeros(3, dtype=np.float32)
    )
    vmax = float(high[0])
    tube_r = max(float(tube_radius), 0.05)
    k_ct = float(cross_track_gain)

    for i in range(3):
        err = g[i] - p[i]
        track_dist = float(np.hypot(float(err[0]), float(err[1])))
        path = agent_paths_xy[i] if i < len(agent_paths_xy) else None

        if path and len(path) >= 2 and track_dist >= float(approach_dist):
            _, tangent, cross_signed, _ = closest_point_on_polyline(path, p[i, :2])
            cross = float(abs(cross_signed))
            tangent_b = float(np.arctan2(float(tangent[1]), float(tangent[0])))
            left_n = np.array([-float(tangent[1]), float(tangent[0])], dtype=np.float64)

            if cross <= tube_r:
                along_speed = vmax
            elif cross <= tube_r * 2.0:
                along_speed = vmax * float(1.0 - 0.5 * (cross - tube_r) / tube_r)
            else:
                along_speed = vmax * 0.35

            corr = k_ct * (-cross_signed)
            vwx = along_speed * float(tangent[0]) + corr * left_n[0]
            vwy = along_speed * float(tangent[1]) + corr * left_n[1]
            nrm = float(np.hypot(vwx, vwy))
            if nrm > vmax and nrm > 1e-9:
                vwx *= vmax / nrm
                vwy *= vmax / nrm
            yaw_ref = tangent_b
        else:
            bearing = float(np.arctan2(float(err[1]), float(err[0])))
            speed = min(float(xy_gain) * track_dist, vmax)
            vwx = speed * float(np.cos(bearing))
            vwy = speed * float(np.sin(bearing))
            yaw_ref = bearing

        vx_b, vy_b = world_vel_to_body(vwx, vwy, float(yaw[i]))
        out[i, 0] = np.float32(vx_b)
        out[i, 1] = np.float32(vy_b)
        if out.shape[1] >= 3 and float(yaw_gain) > 0.0:
            out[i, 2] = np.float32(float(yaw_gain) * _wrap_to_pi(yaw_ref - yaw[i]))
        if out.shape[1] >= 4:
            out[i, 3] = np.float32(float(z_gain) * err[2])

    return _clip_actions(out, low, high)


def analyze_path_following_speed(
    pursuer_pos: np.ndarray,
    track_targets: np.ndarray,
    actions: np.ndarray,
    agent_paths_xy: list[list[list[float]] | None],
    action_high: np.ndarray,
    *,
    xy_gain: float,
    pursuer_yaw: np.ndarray | None,
    approach_dist: float,
) -> list[dict[str, float | str]]:
    """Speed breakdown for holonomic path-tangent body-frame tracking."""
    p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    g = np.asarray(track_targets, dtype=np.float32).reshape(3, 3)
    acts = np.asarray(actions, dtype=np.float32).reshape(3, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    vmax = float(high[0])
    yaw = (
        np.asarray(pursuer_yaw, dtype=np.float32).reshape(3)
        if pursuer_yaw is not None
        else np.zeros(3, dtype=np.float32)
    )
    out: list[dict[str, float | str]] = []
    for i in range(3):
        path = agent_paths_xy[i] if i < len(agent_paths_xy) else None
        dx = float(g[i, 0] - p[i, 0])
        dy = float(g[i, 1] - p[i, 1])
        track_dist = float(np.hypot(dx, dy))
        tangent = path_tangent_bearing(path, p[i, :2]) if path and len(path) >= 2 else float(np.arctan2(dy, dx))
        yaw_err = float(_wrap_to_pi(tangent - yaw[i]))
        act_xy = float(np.hypot(float(acts[i, 0]), float(acts[i, 1])))
        if path and len(path) >= 2 and track_dist >= float(approach_dist):
            limit = "path_tangent_cruise"
            world_speed = vmax
        else:
            world_speed = min(float(xy_gain) * track_dist, vmax)
            limit = "approach"
        out.append(
            {
                "dist_xy": track_dist,
                "tangent_bearing_deg": float(np.rad2deg(tangent)),
                "yaw_err_deg": float(np.rad2deg(yaw_err)),
                "align_factor": float(np.cos(yaw_err)),
                "speed_cmd_xy": act_xy,
                "world_speed_cmd_xy": float(world_speed),
                "speed_cap_xy": vmax,
                "limit_reason": limit,
            }
        )
    return out


def path_following_body_actions(
    pursuer_pos: np.ndarray,
    track_targets: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    agent_paths_xy: list[list[list[float]] | None],
    *,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None,
    yaw_gain: float = 0.0,
    approach_dist: float = 0.5,
) -> np.ndarray:
    """Holonomic path tracking: full world speed along path tangent (body-frame vx/vy).

    Unlike bearing-aligned control, forward speed is **not** gated by yaw alignment;
    the drone may use lateral body velocity while yaw catches up.
    """
    p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    g = np.asarray(track_targets, dtype=np.float32).reshape(3, 3)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    out = _empty_actions(action_low)
    yaw = (
        np.asarray(pursuer_yaw, dtype=np.float32).reshape(3)
        if pursuer_yaw is not None
        else np.zeros(3, dtype=np.float32)
    )
    vmax = float(high[0])

    for i in range(3):
        err = g[i] - p[i]
        track_dist = float(np.hypot(float(err[0]), float(err[1])))
        path = agent_paths_xy[i] if i < len(agent_paths_xy) else None

        if path and len(path) >= 2 and track_dist >= float(approach_dist):
            tangent = path_tangent_bearing(path, p[i, :2])
            speed = vmax
        else:
            tangent = float(np.arctan2(float(err[1]), float(err[0])))
            speed = min(float(xy_gain) * track_dist, vmax)

        vwx = speed * float(np.cos(tangent))
        vwy = speed * float(np.sin(tangent))
        vx_b, vy_b = world_vel_to_body(vwx, vwy, float(yaw[i]))
        out[i, 0] = np.float32(vx_b)
        out[i, 1] = np.float32(vy_b)
        if out.shape[1] >= 3 and float(yaw_gain) > 0.0:
            out[i, 2] = np.float32(float(yaw_gain) * _wrap_to_pi(tangent - yaw[i]))
        if out.shape[1] >= 4:
            out[i, 3] = np.float32(float(z_gain) * err[2])

    return _clip_actions(out, low, high)


def analyze_proportional_speed(
    pursuer_pos: np.ndarray,
    target_pos: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_align_min_speed: float = 0.25,
    path_cruise: bool = False,
    path_cruise_dist: float | None = None,
) -> list[dict[str, float | str]]:
    """Per-pursuer speed breakdown for debug visualization."""
    p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    g = np.asarray(target_pos, dtype=np.float32).reshape(3, 3)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    vmax = float(high[0])
    err = g - p
    use_yaw = pursuer_yaw is not None
    yaw = np.asarray(pursuer_yaw, dtype=np.float32).reshape(3) if use_yaw else np.zeros(3, dtype=np.float32)

    out: list[dict[str, float | str]] = []
    cruise_thr = float(path_cruise_dist) if path_cruise_dist is not None else 0.0
    for i in range(3):
        dx = float(err[i, 0])
        dy = float(err[i, 1])
        dist_xy = float(np.hypot(dx, dy))
        bearing = float(np.arctan2(dy, dx))
        yaw_err = float(_wrap_to_pi(bearing - yaw[i])) if use_yaw else 0.0
        align_raw = float(np.cos(yaw_err)) if use_yaw else 1.0
        align = float(np.clip(align_raw, float(yaw_align_min_speed), 1.0))
        if path_cruise and dist_xy >= cruise_thr * 0.85:
            speed_cmd = min(vmax * align, vmax)
            limit = "path_cruise"
        else:
            speed_cmd = min(float(xy_gain) * dist_xy * align, vmax)
            if dist_xy * align < vmax / max(float(xy_gain), 1e-6):
                limit = "lookahead_dist" if path_cruise else "proportional"
            elif align < 0.999:
                limit = "yaw_misalign"
            else:
                limit = "saturated"
        out.append(
            {
                "dist_xy": dist_xy,
                "bearing_deg": float(np.rad2deg(bearing)),
                "yaw_err_deg": float(np.rad2deg(yaw_err)),
                "align_factor": align,
                "speed_cmd_xy": float(speed_cmd),
                "speed_cap_xy": vmax,
                "limit_reason": limit,
            }
        )
    return out


def proportional_actions_to_targets(
    pursuer_pos: np.ndarray,
    target_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
    path_cruise: bool = False,
    path_cruise_dist: float | None = None,
) -> np.ndarray:
    """Map pursuer-to-target position error to [vx, vy, yaw_rate, vz] actions.

    When ``yaw_gain > 0`` and ``pursuer_yaw`` is provided, uses bearing-aligned
    horizontal speed plus proportional yaw tracking. Otherwise falls back to
    decoupled xy P-control with zero yaw (legacy RL action-box semantics).

    With ``path_cruise=True``, once the track point is at least
    ``path_cruise_dist`` ahead, command full horizontal speed (subject to yaw
    alignment) instead of scaling by short lookahead distance.
    """
    p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    g = np.asarray(target_pos, dtype=np.float32).reshape(3, 3)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    out = _empty_actions(action_low)
    err = g - p

    use_yaw_align = pursuer_yaw is not None and float(yaw_gain) > 0.0
    cruise_thr = float(path_cruise_dist) if path_cruise_dist is not None else 0.0
    if use_yaw_align:
        yaw = np.asarray(pursuer_yaw, dtype=np.float32).reshape(3)
        dx = err[:, 0]
        dy = err[:, 1]
        bearings = np.arctan2(dy, dx).astype(np.float32)
        yaw_err = _wrap_to_pi(bearings - yaw)
        out[:, 2] = float(yaw_gain) * yaw_err

        dist_xy = np.hypot(dx, dy).astype(np.float32)
        align = np.cos(yaw_err)
        align = np.clip(align, float(yaw_align_min_speed), 1.0).astype(np.float32)
        if path_cruise:
            cruise = dist_xy >= np.float32(cruise_thr * 0.85)
            prop_speed = np.minimum(float(xy_gain) * dist_xy * align, float(high[0])).astype(np.float32)
            full_speed = np.minimum(float(high[0]) * align, float(high[0])).astype(np.float32)
            speed_cmd = np.where(cruise, full_speed, prop_speed).astype(np.float32)
        else:
            speed_cmd = np.minimum(float(xy_gain) * dist_xy * align, float(high[0])).astype(np.float32)
        out[:, 0] = speed_cmd * np.cos(bearings)
        out[:, 1] = speed_cmd * np.sin(bearings)
    else:
        out[:, 0] = float(xy_gain) * err[:, 0]
        out[:, 1] = float(xy_gain) * err[:, 1]
        if out.shape[1] >= 3:
            out[:, 2] = 0.0

    if out.shape[1] >= 4:
        out[:, 3] = float(z_gain) * err[:, 2]
    return _clip_actions(out, low, high)


def pure_pursuit_actions_from_state(
    lin_pos: np.ndarray,
    pursuer_ids: np.ndarray,
    evader_id: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
) -> np.ndarray:
    """Pure pursuit: each pursuer directly moves toward the evader."""
    pos = np.asarray(lin_pos, dtype=np.float32)
    pids = np.asarray(pursuer_ids, dtype=np.int64).reshape(3)
    evader = pos[int(evader_id)]
    targets = np.repeat(evader[None, :], 3, axis=0)
    return proportional_actions_to_targets(
        pos[pids],
        targets,
        action_low,
        action_high,
        xy_gain=xy_gain,
        z_gain=z_gain,
        pursuer_yaw=pursuer_yaw,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )


def deployable_slot_actions_from_state(
    env: Any,
    lin_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    role_assignment_mode: str | None,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
) -> np.ndarray:
    """Proportional slot tracking with explicit manifold role allocation.

    ``role_assignment_mode`` selects the deployable allocator (e.g. ``nearest``
    for Hungarian / min-cost matching, ``entropic_ot`` for OT). When ``None``,
    falls back to ``env.task.role_assignment_mode`` (legacy oracle-slot configs).
    """
    pos = np.asarray(lin_pos, dtype=np.float32)
    task_state = env.task_state
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
    pursuer_pos = pos[pursuer_ids]
    evader_pos = pos[int(task_state.evader_id)]

    if not hasattr(env.task, "_assigned_targets_from_state"):
        raise TypeError("Slot baseline requires a pursuit task with slot targets.")

    import time

    record = should_record_control_timing(env)
    _, assignment, assigned_targets = env.task._assigned_targets_from_state(
        pursuer_pos,
        evader_pos,
        task_state=task_state,
        role_assignment_mode=role_assignment_mode,
        record_timing=record,
    )
    t_after_role = time.perf_counter()
    task_state.assigned_target_indices = np.asarray(assignment, dtype=np.int64).copy()
    actions = proportional_actions_to_targets(
        pursuer_pos,
        assigned_targets,
        action_low,
        action_high,
        xy_gain=xy_gain,
        z_gain=z_gain,
        pursuer_yaw=pursuer_yaw,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )
    if record:
        publish_control_timing(env, action_mapping_time=time.perf_counter() - t_after_role)
    return actions


def hungarian_slot_actions_from_state(
    env: Any,
    lin_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
) -> np.ndarray:
    """Hungarian slot controller: min-cost UAV→slot matching on the manifold."""
    return deployable_slot_actions_from_state(
        env,
        lin_pos,
        action_low,
        action_high,
        role_assignment_mode="nearest",
        xy_gain=xy_gain,
        z_gain=z_gain,
        pursuer_yaw=pursuer_yaw,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )


def ot_slot_actions_from_state(
    env: Any,
    lin_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
) -> np.ndarray:
    """OT slot controller: entropic optimal-transport role allocation on the manifold."""
    return deployable_slot_actions_from_state(
        env,
        lin_pos,
        action_low,
        action_high,
        role_assignment_mode="entropic_ot",
        xy_gain=xy_gain,
        z_gain=z_gain,
        pursuer_yaw=pursuer_yaw,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )


def oracle_slot_actions_from_state(
    env: Any,
    lin_pos: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    xy_gain: float,
    z_gain: float,
    pursuer_yaw: np.ndarray | None = None,
    yaw_gain: float = 0.0,
    yaw_align_min_speed: float = 0.25,
) -> np.ndarray:
    """Legacy oracle slot: uses ``task.role_assignment_mode`` from experiment YAML."""
    return deployable_slot_actions_from_state(
        env,
        lin_pos,
        action_low,
        action_high,
        role_assignment_mode=None,
        xy_gain=xy_gain,
        z_gain=z_gain,
        pursuer_yaw=pursuer_yaw,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )


def make_pure_pursuit_get_actions_fn(
    env: Any,
    *,
    xy_gain: float | None = 0.25,
    z_gain: float | None = 0.20,
    yaw_gain: float | None = 0.25,
    yaw_align_min_speed: float = 0.25,
):
    """Build a RolloutWorker ``get_actions_fn`` for pure pursuit."""
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError("pure-pursuit baseline requires a continuous action-space env")
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)
    xy_gain, z_gain, yaw_gain = default_proportional_gains(
        low, high, xy_gain=xy_gain, z_gain=z_gain, yaw_gain=yaw_gain
    )

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError("Environment must be reset before selecting pure-pursuit actions.")
        backend = env.prev_backend_state
        lin_pos = np.asarray(backend.states[:, 3, :], dtype=np.float32)
        yaws = pursuer_yaws_from_backend(backend, env.task_state.pursuer_ids)
        import time

        record = should_record_control_timing(env)
        t_map = time.perf_counter() if record else None
        actions = pure_pursuit_actions_from_state(
            lin_pos,
            env.task_state.pursuer_ids,
            env.task_state.evader_id,
            low,
            high,
            xy_gain=xy_gain,
            z_gain=z_gain,
            pursuer_yaw=yaws,
            yaw_gain=yaw_gain,
            yaw_align_min_speed=yaw_align_min_speed,
        )
        if record and t_map is not None:
            publish_control_timing(env, action_mapping_time=time.perf_counter() - t_map)
        return actions

    return get_actions


def _make_slot_get_actions_fn(
    env: Any,
    *,
    role_assignment_mode: str | None,
    actions_from_state_fn: Any,
    label: str,
    xy_gain: float | None = 0.25,
    z_gain: float | None = 0.20,
    yaw_gain: float | None = 0.25,
    yaw_align_min_speed: float = 0.25,
):
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError(f"{label} baseline requires a continuous action-space env")
    low = np.asarray(env.action_low_np, dtype=np.float32)
    high = np.asarray(env.action_high_np, dtype=np.float32)
    xy_gain, z_gain, yaw_gain = default_proportional_gains(
        low, high, xy_gain=xy_gain, z_gain=z_gain, yaw_gain=yaw_gain
    )
    del role_assignment_mode

    def get_actions(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        if env.prev_backend_state is None or env.task_state is None:
            raise RuntimeError(f"Environment must be reset before selecting {label} actions.")
        backend = env.prev_backend_state
        lin_pos = np.asarray(backend.states[:, 3, :], dtype=np.float32)
        yaws = pursuer_yaws_from_backend(backend, env.task_state.pursuer_ids)
        return actions_from_state_fn(
            env,
            lin_pos,
            low,
            high,
            xy_gain=xy_gain,
            z_gain=z_gain,
            pursuer_yaw=yaws,
            yaw_gain=yaw_gain,
            yaw_align_min_speed=yaw_align_min_speed,
        )

    return get_actions


def make_hungarian_slot_get_actions_fn(
    env: Any,
    *,
    xy_gain: float | None = 0.25,
    z_gain: float | None = 0.20,
    yaw_gain: float | None = 0.25,
    yaw_align_min_speed: float = 0.25,
):
    """Build a RolloutWorker ``get_actions_fn`` for Hungarian slot tracking."""
    return _make_slot_get_actions_fn(
        env,
        role_assignment_mode="nearest",
        actions_from_state_fn=hungarian_slot_actions_from_state,
        label="hungarian-slot",
        xy_gain=xy_gain,
        z_gain=z_gain,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )


def make_ot_slot_get_actions_fn(
    env: Any,
    *,
    xy_gain: float | None = 0.25,
    z_gain: float | None = 0.20,
    yaw_gain: float | None = 0.25,
    yaw_align_min_speed: float = 0.25,
):
    """Build a RolloutWorker ``get_actions_fn`` for OT slot tracking."""
    return _make_slot_get_actions_fn(
        env,
        role_assignment_mode="entropic_ot",
        actions_from_state_fn=ot_slot_actions_from_state,
        label="ot-slot",
        xy_gain=xy_gain,
        z_gain=z_gain,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )


def make_oracle_slot_get_actions_fn(
    env: Any,
    *,
    xy_gain: float | None = 0.25,
    z_gain: float | None = 0.20,
    yaw_gain: float | None = 0.25,
    yaw_align_min_speed: float = 0.25,
):
    """Build a RolloutWorker ``get_actions_fn`` for the legacy oracle slot controller."""
    return _make_slot_get_actions_fn(
        env,
        role_assignment_mode=None,
        actions_from_state_fn=oracle_slot_actions_from_state,
        label="oracle-slot",
        xy_gain=xy_gain,
        z_gain=z_gain,
        yaw_gain=yaw_gain,
        yaw_align_min_speed=yaw_align_min_speed,
    )
