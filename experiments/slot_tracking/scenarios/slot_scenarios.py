"""Single-UAV and shadow-SCE slot trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from experiments.slot_tracking.scenarios.obstacle_maps import (
    ObstacleMap,
    build_obstacle_map,
    empty_obstacles,
)
from marl_uav.framework.geometry.obstacle_geometry import has_line_of_sight


@dataclass(frozen=True)
class ScenarioSpec:
    """Fully deterministic benchmark scenario description."""

    name: str
    group: str
    obstacle_map: str = "none"
    speed_scale: float = 1.0
    noise_std: float = 0.0
    obstacle_noise_std: float = 0.0
    action_delay_steps: int = 0
    actual_vmax_scale: float = 1.0
    wind_std: float = 0.0
    obstacle_dropout_prob: float = 0.0
    num_agents: int = 1
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class ScenarioInstance:
    """Concrete initial state, slot path, and obstacle map for one seed."""

    spec: ScenarioSpec
    init_positions: np.ndarray
    init_velocities: np.ndarray
    slot_positions: np.ndarray
    slot_velocities: np.ndarray
    obstacle_map: ObstacleMap
    feasible: bool
    infeasible_reason: str = ""
    feasibility_metrics: dict[str, float | bool | str] = field(default_factory=dict)
    jump_steps: list[int] = field(default_factory=list)


def _clip_to_soft_world(path: np.ndarray, world_xy: float, margin: float = 0.6) -> np.ndarray:
    out = np.asarray(path, dtype=np.float64).copy()
    lim = max(float(world_xy) - float(margin), 0.1)
    out[..., 0] = np.clip(out[..., 0], -lim, lim)
    out[..., 1] = np.clip(out[..., 1], -lim, lim)
    return out


def _vel_from_path(path: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(path)
    if path.shape[0] > 1:
        vel[1:] = (path[1:] - path[:-1]) / max(float(dt), 1e-9)
        vel[0] = vel[1]
    return vel


def _constant_z(path_xy: np.ndarray, z: float) -> np.ndarray:
    z_col = np.full((*path_xy.shape[:-1], 1), float(z), dtype=np.float64)
    return np.concatenate([path_xy, z_col], axis=-1)


def _as_xy(value: object, *, default: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(2)


def _slot_path_for_spec(
    spec: ScenarioSpec,
    rng: np.random.Generator,
    *,
    steps: int,
    dt: float,
    world_xy: float,
    z: float,
    uav_vmax: float,
) -> tuple[np.ndarray, list[int]]:
    t = np.arange(int(steps), dtype=np.float64) * float(dt)
    speed = float(spec.speed_scale) * float(uav_vmax)
    name = spec.name
    jump_steps: list[int] = []

    trajectory_name = "linear_slot" if name in {
        "robust_linear_slot",
        "obs_noise",
        "action_delay",
        "reduced_actual_vmax",
        "wind_disturbance",
        "obstacle_dropout",
    } else name

    if trajectory_name == "static_slot":
        xy = np.repeat(np.array([[3.0, 0.0]], dtype=np.float64), steps, axis=0)
    elif trajectory_name == "linear_slot":
        xy = np.column_stack((-4.0 + speed * t, 2.0 + 0.0 * t))
    elif trajectory_name == "circular_slot":
        radius = 3.0
        omega = speed / radius
        xy = np.column_stack((radius * np.cos(omega * t), radius * np.sin(omega * t)))
    elif trajectory_name == "sinusoidal_slot":
        vx = 0.65 * speed
        amp = 2.0
        # Keep max trajectory speed within the configured slot speed label.
        omega = 0.0 if speed <= 1e-9 else (speed * np.sqrt(max(1.0 - 0.65**2, 0.0))) / amp
        xy = np.column_stack((-4.0 + vx * t, amp * np.sin(omega * t)))
    elif trajectory_name == "random_walk_slot":
        xy = np.zeros((steps, 2), dtype=np.float64)
        xy[0] = [-4.0, -1.0]
        heading = float(rng.uniform(-np.pi, np.pi))
        hold = max(int(1.5 / max(dt, 1e-6)), 1)
        for k in range(1, steps):
            if k % hold == 0:
                heading += float(rng.uniform(-1.2, 1.2))
            xy[k] = xy[k - 1] + speed * dt * np.array([np.cos(heading), np.sin(heading)])
            xy[k] = _clip_to_soft_world(xy[k], world_xy, margin=2.0)
    elif trajectory_name == "boundary_parallel_slot":
        y = world_xy - 1.0
        xy = np.column_stack((-6.0 + speed * t, np.full_like(t, y)))
    elif trajectory_name == "boundary_corner_turn":
        xy = np.zeros((steps, 2), dtype=np.float64)
        p = np.array([world_xy - 6.0, world_xy - 1.2], dtype=np.float64)
        for k in range(steps):
            if k < steps // 2:
                p = p + np.array([speed * dt, 0.0])
            else:
                p = p + np.array([0.0, -speed * dt])
            xy[k] = p
    elif trajectory_name == "outside_boundary_inducing_slot":
        xy = np.column_stack((world_xy - 2.5 + 4.5 * np.sin(0.4 * t), 2.0 * np.sin(0.2 * t)))
    elif trajectory_name == "outward_initial_velocity":
        xy = np.repeat(np.array([[world_xy - 4.0, 0.0]], dtype=np.float64), steps, axis=0)
    elif trajectory_name == "single_blocking_obstacle":
        y = 0.20 * np.sin(0.35 * t)
        xy = np.column_stack((np.full_like(t, 1.45), y))
    elif trajectory_name == "sparse_random_obstacles":
        vx = 0.7 * speed
        y = -2.8 + 0.35 * speed * t
        xy = np.column_stack((-5.2 + vx * t, y))
    elif trajectory_name == "narrow_passage":
        vx = 0.8 * speed
        y = 0.25 * np.sin(0.45 * t)
        xy = np.column_stack((-4.8 + vx * t, y))
    elif trajectory_name == "boundary_obstacle_combo":
        x = np.full_like(t, world_xy - 1.55)
        y = -1.2 + 0.25 * np.sin(0.32 * t)
        xy = np.column_stack((x, y))
    elif trajectory_name == "u_shaped_trap":
        x = 2.15 + 0.25 * np.sin(0.18 * t)
        y = 0.25 * np.sin(0.25 * t)
        xy = np.column_stack((x, y))
    elif trajectory_name.startswith("smooth_slot_deformation"):
        suffix_speed = {
            "smooth_slot_deformation": 0.5,
            "smooth_slot_deformation_v02": 0.2,
            "smooth_slot_deformation_v05": 0.5,
            "smooth_slot_deformation_v08": 0.8,
        }.get(trajectory_name, 0.5)
        local_speed = min(float(suffix_speed), max(speed, 1e-6))
        radius = 3.0 + 0.8 * np.sin(0.18 * t)
        omega = local_speed / 3.2
        xy = np.column_stack((radius * np.cos(omega * t), radius * np.sin(omega * t)))
    elif trajectory_name in (
        "small_slot_jump",
        "medium_slot_jump",
        "large_slot_jump",
        "repeated_medium_jumps",
        "random_interval_jumps",
        "high_frequency_jumps",
        "two_slot_swap",
        "far_slot_swap",
        "cyclic_slot_swap",
        "three_slot_cyclic_swap",
        "nearby_slot_swap",
        "jump_outside_boundary",
        "jump_inside_obstacle",
        "jump_too_far_for_available_window",
        "jump_behind_obstacle_without_enough_time",
        "slot_jump_behind_obstacle",
        "slot_jump_near_boundary",
        "slot_jump_through_narrow_passage",
        "slot_jump_boundary_obstacle_combo",
        "infeasible_slot_jump",
    ):
        xy = np.repeat(np.array([[-1.5, -2.0]], dtype=np.float64), steps, axis=0)
        current = xy[0].copy()

        def apply_jump(k: int, target: np.ndarray | None = None, distance: float = 1.5) -> None:
            nonlocal current
            jump_steps.append(int(k))
            if target is None:
                direction = rng.uniform(-np.pi, np.pi)
                current = current + float(distance) * np.array([np.cos(direction), np.sin(direction)])
            else:
                current = np.asarray(target, dtype=np.float64).reshape(2).copy()
            if trajectory_name not in {"jump_outside_boundary", "jump_inside_obstacle", "jump_too_far_for_available_window", "jump_behind_obstacle_without_enough_time", "infeasible_slot_jump"}:
                current = _clip_to_soft_world(current, world_xy, margin=1.5)

        if trajectory_name == "small_slot_jump":
            schedule = [(max(int(5.0 / dt), 1), np.array([-0.8, -1.5]), 0.7)]
        elif trajectory_name == "medium_slot_jump":
            schedule = [(max(int(5.0 / dt), 1), np.array([0.2, -1.0]), 1.5)]
        elif trajectory_name == "large_slot_jump":
            schedule = [(max(int(5.0 / dt), 1), np.array([2.2, -0.6]), 3.8)]
        elif trajectory_name == "repeated_medium_jumps":
            schedule = [
                (max(int(4.0 / dt), 1), None, 1.4),
                (max(int(8.0 / dt), 1), None, 1.6),
                (max(int(12.0 / dt), 1), None, 1.5),
            ]
        elif trajectory_name == "random_interval_jumps":
            base = max(int(3.0 / dt), 1)
            schedule = [(base + int(x), None, float(rng.uniform(1.0, 2.0))) for x in np.cumsum(rng.integers(max(int(1.5 / dt), 1), max(int(3.5 / dt), 2), size=4))]
        elif trajectory_name == "high_frequency_jumps":
            schedule = [(max(int((3.0 + j * 0.8) / dt), 1), None, 1.2) for j in range(6)]
        elif trajectory_name == "two_slot_swap":
            schedule = [(max(int(5.0 / dt), 1), np.array([1.5, 2.0]), 0.0)]
        elif trajectory_name == "far_slot_swap":
            schedule = [(max(int(5.0 / dt), 1), np.array([4.5, 2.5]), 0.0)]
        elif trajectory_name in {"cyclic_slot_swap", "three_slot_cyclic_swap"}:
            schedule = [
                (max(int(4.0 / dt), 1), np.array([1.5, -1.0]), 0.0),
                (max(int(8.0 / dt), 1), np.array([1.5, 2.0]), 0.0),
                (max(int(12.0 / dt), 1), np.array([-1.5, 2.0]), 0.0),
            ]
        elif trajectory_name == "nearby_slot_swap":
            schedule = [(max(int(5.0 / dt), 1), np.array([-0.4, -1.35]), 0.0)]
        elif trajectory_name in {"jump_outside_boundary", "infeasible_slot_jump"}:
            schedule = [(max(int(4.0 / dt), 1), np.array([world_xy + 4.0, 0.0]), 0.0)]
        elif trajectory_name == "jump_inside_obstacle":
            schedule = [(max(int(4.0 / dt), 1), np.array([0.0, 0.0]), 0.0)]
        elif trajectory_name == "jump_too_far_for_available_window":
            schedule = [(max(int((steps * dt - 1.0) / dt), 1), np.array([world_xy - 1.0, world_xy - 1.0]), 0.0)]
        elif trajectory_name == "jump_behind_obstacle_without_enough_time":
            schedule = [(max(int((steps * dt - 1.0) / dt), 1), np.array([0.0, 0.0]), 0.0)]
        elif trajectory_name == "slot_jump_behind_obstacle":
            schedule = [(max(int(4.0 / dt), 1), np.array([1.8, 0.0]), 0.0)]
        elif trajectory_name == "slot_jump_near_boundary":
            schedule = [(max(int(4.0 / dt), 1), np.array([world_xy - 1.2, 2.0]), 0.0)]
        elif trajectory_name == "slot_jump_through_narrow_passage":
            schedule = [
                (max(int(4.0 / dt), 1), np.array([-1.2, 0.0]), 0.0),
                (max(int(8.0 / dt), 1), np.array([1.2, 0.0]), 0.0),
            ]
        elif trajectory_name == "slot_jump_boundary_obstacle_combo":
            schedule = [(max(int(4.0 / dt), 1), np.array([world_xy - 1.6, 0.3]), 0.0)]
        else:
            schedule = []

        schedule_by_step = {int(k): (target, dist) for k, target, dist in schedule if 0 < int(k) < steps}
        for k in range(steps):
            if k in schedule_by_step:
                target, dist = schedule_by_step[k]
                apply_jump(k, target=target, distance=dist)
            xy[k] = current
    elif trajectory_name == "shadow_sce_three_slots":
        center = np.column_stack((1.5 * np.sin(0.25 * t), 1.5 * np.cos(0.2 * t)))
        radius = 3.0
        phase = 0.35 * t
        xy3 = np.zeros((steps, 3, 2), dtype=np.float64)
        for j in range(3):
            ang = phase + 2.0 * np.pi * j / 3.0
            xy3[:, j, 0] = center[:, 0] + radius * np.cos(ang)
            xy3[:, j, 1] = center[:, 1] + radius * np.sin(ang)
        return _constant_z(_clip_to_soft_world(xy3, world_xy, margin=1.0), z), jump_steps
    else:
        raise ValueError(f"Unknown slot scenario: {name}")

    if trajectory_name not in (
        "outside_boundary_inducing_slot",
        "infeasible_slot_jump",
        "jump_outside_boundary",
        "jump_inside_obstacle",
        "jump_too_far_for_available_window",
        "jump_behind_obstacle_without_enough_time",
    ):
        xy = _clip_to_soft_world(xy, world_xy, margin=0.6)
    return _constant_z(xy, z), jump_steps


def scenario_specs(
    groups: list[str],
    speed_levels: list[float],
    robustness_cfg: dict,
    scenario_names: list[str] | None = None,
) -> list[ScenarioSpec]:
    """Enumerate reproducible scenario specifications for requested groups."""
    selected = set("ABCDEF" if "all" in groups else [g.upper() for g in groups])
    name_filter = None if scenario_names is None else {str(x) for x in scenario_names}
    specs: list[ScenarioSpec] = []
    if "A" in selected:
        for name in ["static_slot", "linear_slot", "circular_slot", "sinusoidal_slot", "random_walk_slot"]:
            if name_filter is not None and name not in name_filter:
                continue
            for s in speed_levels:
                specs.append(ScenarioSpec(name=name, group="A", speed_scale=float(s)))
    if "B" in selected:
        for name in ["boundary_parallel_slot", "boundary_corner_turn", "outside_boundary_inducing_slot", "outward_initial_velocity"]:
            if name_filter is not None and name not in name_filter:
                continue
            for s in speed_levels:
                specs.append(ScenarioSpec(name=name, group="B", speed_scale=float(s)))
    if "C" in selected:
        for obstacle_name in ["single_blocking_obstacle", "sparse_random_obstacles", "narrow_passage", "boundary_obstacle_combo", "u_shaped_trap"]:
            if name_filter is not None and obstacle_name not in name_filter:
                continue
            specs.append(ScenarioSpec(name=obstacle_name, group="C", obstacle_map=obstacle_name, speed_scale=0.8))
    if "D" in selected:
        d_names = [
            "smooth_slot_deformation_v02",
            "smooth_slot_deformation_v05",
            "smooth_slot_deformation_v08",
            "small_slot_jump",
            "medium_slot_jump",
            "large_slot_jump",
            "repeated_medium_jumps",
            "random_interval_jumps",
            "high_frequency_jumps",
            "two_slot_swap",
            "far_slot_swap",
            "three_slot_cyclic_swap",
            "nearby_slot_swap",
            "jump_outside_boundary",
            "jump_inside_obstacle",
            "jump_too_far_for_available_window",
            "jump_behind_obstacle_without_enough_time",
            "slot_jump_behind_obstacle",
            "slot_jump_near_boundary",
            "slot_jump_through_narrow_passage",
            "slot_jump_boundary_obstacle_combo",
        ]
        for name in d_names:
            if name_filter is not None and name not in name_filter:
                continue
            obstacle_map = (
                "single_blocking_obstacle"
                if name in {"jump_inside_obstacle", "jump_behind_obstacle_without_enough_time", "slot_jump_behind_obstacle"}
                else "narrow_passage"
                if name == "slot_jump_through_narrow_passage"
                else "boundary_obstacle_combo"
                if name == "slot_jump_boundary_obstacle_combo"
                else "none"
            )
            specs.append(ScenarioSpec(name=name, group="D", obstacle_map=obstacle_map, speed_scale=0.8))
    if "E" in selected:
        base = ScenarioSpec(name="robust_linear_slot", group="E", speed_scale=0.8, obstacle_map="sparse_random_obstacles")
        for noise in robustness_cfg.get("observation_noise_std", [0.0]):
            if float(noise) > 0.0:
                specs.append(_replace(base, name="obs_noise", noise_std=float(noise)))
        for delay in robustness_cfg.get("action_delay_steps", [0]):
            if int(delay) > 0:
                specs.append(_replace(base, name="action_delay", action_delay_steps=int(delay)))
        for scale in robustness_cfg.get("actual_vmax_scale", [1.0]):
            if float(scale) < 1.0:
                specs.append(_replace(base, name="reduced_actual_vmax", actual_vmax_scale=float(scale)))
        for wind in robustness_cfg.get("wind_std", [0.0]):
            if float(wind) > 0.0:
                specs.append(_replace(base, name="wind_disturbance", wind_std=float(wind)))
        for drop in robustness_cfg.get("obstacle_dropout_prob", [0.0]):
            if float(drop) > 0.0:
                specs.append(_replace(base, name="obstacle_dropout", obstacle_dropout_prob=float(drop)))
    if "F" in selected:
        specs.append(ScenarioSpec(name="shadow_sce_three_slots", group="F", speed_scale=0.8, num_agents=3))
    return specs


def _replace(spec: ScenarioSpec, **kwargs) -> ScenarioSpec:
    data = {
        "name": spec.name,
        "group": spec.group,
        "obstacle_map": spec.obstacle_map,
        "speed_scale": spec.speed_scale,
        "noise_std": spec.noise_std,
        "obstacle_noise_std": spec.obstacle_noise_std,
        "action_delay_steps": spec.action_delay_steps,
        "actual_vmax_scale": spec.actual_vmax_scale,
        "wind_std": spec.wind_std,
        "obstacle_dropout_prob": spec.obstacle_dropout_prob,
        "num_agents": spec.num_agents,
        "metadata": dict(spec.metadata),
    }
    data.update(kwargs)
    return ScenarioSpec(**data)


def instantiate_scenario(spec: ScenarioSpec, cfg: dict, seed: int) -> ScenarioInstance:
    """Instantiate a scenario with deterministic initial states and slot path."""
    rng = np.random.default_rng(int(seed))
    dyn = cfg["dynamics"]
    world = cfg["world"]
    steps = int(dyn["episode_steps"])
    dt = float(dyn["dt"])
    world_xy = float(world["world_xy"])
    z = float(world.get("z", 1.0))
    uav_vmax = float(dyn["uav_vmax"])

    path, jumps = _slot_path_for_spec(
        spec,
        rng,
        steps=steps,
        dt=dt,
        world_xy=world_xy,
        z=z,
        uav_vmax=uav_vmax,
    )
    vel = _vel_from_path(path, dt)
    n = int(spec.num_agents)
    if n == 1:
        init_pos = np.array([[-6.0, -3.0, z]], dtype=np.float64)
        if spec.name == "outward_initial_velocity":
            init_pos = np.array([[world_xy - 0.8, 0.0, z]], dtype=np.float64)
        init_vel = np.zeros((1, 3), dtype=np.float64)
        if spec.name == "outward_initial_velocity":
            init_vel[0, 0] = uav_vmax
        slot_positions = path.reshape(steps, 1, 3)
        slot_velocities = vel.reshape(steps, 1, 3)
    else:
        init_pos = np.array([[-6.0, -4.0, z], [-6.0, 0.0, z], [-6.0, 4.0, z]], dtype=np.float64)
        init_vel = np.zeros((3, 3), dtype=np.float64)
        slot_positions = path.reshape(steps, n, 3)
        slot_velocities = vel.reshape(steps, n, 3)

    init_pos, init_vel = _apply_initial_state_config(
        init_pos,
        init_vel,
        slot_positions,
        spec=spec,
        cfg=cfg,
    )

    obstacle_cfg = dict((cfg.get("obstacles") or {}).get(spec.obstacle_map) or {})
    obstacle_map = empty_obstacles() if spec.obstacle_map == "none" else build_obstacle_map(spec.obstacle_map, rng, world_xy, obstacle_cfg)
    init_pos = _repair_initial_positions_for_obstacles(init_pos, obstacle_map, cfg=cfg)
    feasibility = compute_slot_feasibility(
        slot_positions,
        obstacle_map,
        cfg=cfg,
        init_positions=init_pos,
    )
    infeasible = bool(feasibility["target_infeasible"])
    reason = str(feasibility["infeasible_reason"])
    if spec.name == "static_slot" and float(feasibility["actual_slot_speed_p95"]) > float(
        cfg.get("failure_classifier", {}).get("static_slot_speed_epsilon", 1e-6)
    ):
        print("WARNING: static_slot has nonzero actual speed.")
    if spec.group == "D" and spec.name.startswith("smooth_slot_deformation"):
        smooth_limit = float(cfg.get("failure_classifier", {}).get("smooth_speed_limit", cfg["dynamics"]["uav_vmax"]))
        if int(feasibility.get("number_of_generator_jump_events", 0)) != 0:
            print(f"WARNING: {spec.name} generated jump events.")
        if float(feasibility.get("slot_out_of_bounds_ratio", 0.0)) != 0.0:
            print(f"WARNING: {spec.name} leaves boundary.")
        if float(feasibility.get("actual_slot_speed_p95", 0.0)) > smooth_limit:
            print(f"WARNING: {spec.name} continuous speed p95 exceeds smooth_speed_limit.")
    return ScenarioInstance(
        spec=spec,
        init_positions=init_pos,
        init_velocities=init_vel,
        slot_positions=slot_positions,
        slot_velocities=slot_velocities,
        obstacle_map=obstacle_map,
        feasible=not infeasible,
        infeasible_reason=reason,
        feasibility_metrics=feasibility,
        jump_steps=jumps,
    )


def _apply_initial_state_config(
    init_pos: np.ndarray,
    init_vel: np.ndarray,
    slot_positions: np.ndarray,
    *,
    spec: ScenarioSpec,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply benchmark-level initial-state overrides.

    Legacy ``benchmark.initial_offset_xy`` remains supported. Newer configs can
    use ``benchmark.initial_state.mode: near_slot`` plus per-scenario overrides
    to isolate local tracking/safety behavior from long approach transients.
    """
    bench = cfg.get("benchmark", {})
    pos = np.asarray(init_pos, dtype=np.float64).copy()
    vel = np.asarray(init_vel, dtype=np.float64).copy()

    state_cfg = dict(bench.get("initial_state") or {})
    if not state_cfg and "initial_offset_xy" in bench:
        state_cfg = {"mode": "near_slot", "offset_xy": bench["initial_offset_xy"]}
    if not state_cfg:
        return pos, vel

    per_scenario = state_cfg.get("per_scenario") or {}
    override = dict(per_scenario.get(spec.name) or {}) if isinstance(per_scenario, dict) else {}
    merged = {**state_cfg, **override}
    mode = str(merged.get("mode", state_cfg.get("mode", "default"))).strip().lower()

    if mode in ("near_slot", "slot_offset"):
        offset = _as_xy(merged.get("offset_xy", merged.get("default_offset_xy")), default=(0.0, 0.0))
        pos[:, :2] = np.asarray(slot_positions[0, :, :2], dtype=np.float64) + offset.reshape(1, 2)
    elif mode in ("fixed", "absolute"):
        xy = _as_xy(merged.get("position_xy"))
        pos[:, :2] = xy.reshape(1, 2)

    if "velocity_xy" in merged:
        vel[:, :2] = _as_xy(merged.get("velocity_xy")).reshape(1, 2)
    elif bool(merged.get("outward_velocity", False)):
        speed = float(merged.get("velocity_speed", cfg["dynamics"]["uav_vmax"]))
        for i in range(pos.shape[0]):
            axis = int(np.argmax(np.abs(pos[i, :2])))
            sign = 1.0 if pos[i, axis] >= 0.0 else -1.0
            vel[i, :2] = 0.0
            vel[i, axis] = sign * speed

    if bool(merged.get("clip_to_valid", False)):
        world_xy = float(cfg["world"]["world_xy"])
        uav_radius = float(cfg["dynamics"].get("uav_radius", 0.0))
        min_margin = float(merged.get("min_boundary_margin", cfg["dynamics"].get("safety_margin", 0.0)))
        limit = max(world_xy - uav_radius - max(min_margin, 0.0), 0.0)
        pos[:, 0] = np.clip(pos[:, 0], -limit, limit)
        pos[:, 1] = np.clip(pos[:, 1], -limit, limit)

    return pos, vel


def _repair_initial_positions_for_obstacles(
    init_pos: np.ndarray,
    obstacle_map: ObstacleMap,
    *,
    cfg: dict,
) -> np.ndarray:
    bench = cfg.get("benchmark", {})
    state_cfg = dict(bench.get("initial_state") or {})
    if not bool(state_cfg.get("clip_to_valid", False)):
        return init_pos
    if not obstacle_map.obstacles:
        return init_pos
    pos = np.asarray(init_pos, dtype=np.float64).copy()
    world_xy = float(cfg["world"]["world_xy"])
    uav_radius = float(cfg["dynamics"].get("uav_radius", 0.0))
    safety_margin = float(cfg["dynamics"].get("safety_margin", 0.0))
    min_boundary_margin = float(state_cfg.get("min_boundary_margin", safety_margin))
    limit = max(world_xy - uav_radius - max(min_boundary_margin, 0.0), 0.0)
    for i in range(pos.shape[0]):
        xy = pos[i, :2].copy()
        for _ in range(8):
            changed = False
            for obs in obstacle_map.obstacles:
                clear = _obstacle_clearance_xy(xy, obs, uav_radius=uav_radius)
                if clear >= safety_margin:
                    continue
                center = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
                rel = xy - center
                dist = float(np.linalg.norm(rel))
                normal = np.array([1.0, 0.0], dtype=np.float64) if dist <= 1e-9 else rel / dist
                radius = float(getattr(obs, "radius", 0.0))
                xy = center + normal * (radius + uav_radius + safety_margin)
                xy[0] = float(np.clip(xy[0], -limit, limit))
                xy[1] = float(np.clip(xy[1], -limit, limit))
                changed = True
            if not changed:
                break
        pos[i, :2] = xy
    return pos


def compute_slot_feasibility(
    slot_positions: np.ndarray,
    obstacle_map: ObstacleMap,
    *,
    cfg: dict,
    init_positions: np.ndarray | None = None,
) -> dict[str, float | bool | str]:
    """Compute target feasibility from the actual generated slot trajectory."""
    dyn = cfg["dynamics"]
    fc = cfg.get("failure_classifier", {})
    dt = max(float(dyn["dt"]), 1e-9)
    uav_vmax = float(dyn["uav_vmax"])
    world_xy = float(cfg["world"]["world_xy"])
    uav_radius = float(dyn.get("uav_radius", 0.0))
    safety_margin = float(dyn.get("safety_margin", 0.0))
    speed_tol = float(fc.get("infeasible_speed_tolerance", 1.05))
    outside_threshold = float(fc.get("slot_out_of_bounds_threshold", fc.get("infeasible_outside_ratio", 0.35)))
    unreachable_threshold = float(fc.get("slot_unreachable_ratio_threshold", 0.50))
    slot_safety_margin = float(fc.get("slot_boundary_safety_margin", safety_margin))

    slots = np.asarray(slot_positions, dtype=np.float64).reshape(slot_positions.shape[0], -1, 3)
    if slots.shape[0] > 1:
        displacements = np.linalg.norm(np.diff(slots[..., :2], axis=0), axis=-1)
        speeds = displacements / dt
    else:
        displacements = np.zeros((1, slots.shape[1]), dtype=np.float64)
        speeds = np.zeros((1, slots.shape[1]), dtype=np.float64)
    jump_threshold = float(cfg.get("slot_transition", {}).get("jump_detection_threshold", dyn.get("jump_detection_threshold", 0.75)))
    jump_events = displacements > jump_threshold
    continuous_speeds = speeds[~jump_events] if speeds.size else speeds
    margins = world_xy - np.max(np.abs(slots[..., :2]), axis=-1)
    outside = margins < 0.0
    too_close = (margins >= 0.0) & (margins < slot_safety_margin)
    obstacle_blocked = np.zeros_like(margins, dtype=bool)
    obstacle_too_close = np.zeros_like(margins, dtype=bool)
    obstacle_clearances = np.full_like(margins, float("inf"), dtype=np.float64)
    if obstacle_map.obstacles:
        for t in range(slots.shape[0]):
            for j in range(slots.shape[1]):
                p = slots[t, j, :2]
                for obs in obstacle_map.obstacles:
                    c = np.asarray(obs.center, dtype=np.float64).reshape(2)
                    clearance = _obstacle_clearance_xy(p, obs, uav_radius=uav_radius)
                    obstacle_clearances[t, j] = min(float(obstacle_clearances[t, j]), clearance)
                    if clearance < 0.0:
                        obstacle_blocked[t, j] = True
                    if clearance < safety_margin:
                        obstacle_too_close[t, j] = True

    line_of_sight_blocked = np.zeros_like(margins, dtype=bool)
    start_goal_valid = True
    if obstacle_map.obstacles and init_positions is not None:
        starts = np.asarray(init_positions, dtype=np.float64).reshape(-1, 3)
        for j in range(slots.shape[1]):
            start = starts[min(j, starts.shape[0] - 1), :2]
            goal = slots[0, j, :2]
            los = has_line_of_sight(start, goal, obstacle_map.obstacles, safety_margin=safety_margin, uav_radius=uav_radius)
            start_free = (
                _obstacle_clearance_xy(start, obstacle_map.obstacles[0], uav_radius=uav_radius) >= 0.0
                if len(obstacle_map.obstacles) == 1
                else all(_obstacle_clearance_xy(start, obs, uav_radius=uav_radius) >= 0.0 for obs in obstacle_map.obstacles)
            )
            goal_free = all(_obstacle_clearance_xy(goal, obs, uav_radius=uav_radius) >= 0.0 for obs in obstacle_map.obstacles)
            start_goal_valid = bool(start_goal_valid and start_free and goal_free)
            for t in range(slots.shape[0]):
                line_of_sight_blocked[t, j] = not has_line_of_sight(
                    start,
                    slots[t, j, :2],
                    obstacle_map.obstacles,
                    safety_margin=safety_margin,
                    uav_radius=uav_radius,
                )

    actual_mean = float(np.mean(continuous_speeds)) if continuous_speeds.size else 0.0
    actual_p95 = float(np.percentile(continuous_speeds, 95)) if continuous_speeds.size else 0.0
    actual_max = float(np.max(speeds)) if speeds.size else 0.0
    jump_distances = displacements[jump_events] if displacements.size else np.zeros(0, dtype=np.float64)
    jump_steps = np.argwhere(jump_events)
    min_reach_times: list[float] = []
    available_windows: list[float] = []
    if jump_steps.size:
        steps_per_agent: dict[int, list[int]] = {}
        for step_idx, agent_idx in jump_steps:
            steps_per_agent.setdefault(int(agent_idx), []).append(int(step_idx) + 1)
        for agent_idx, event_steps in steps_per_agent.items():
            for idx, step_idx in enumerate(event_steps):
                next_step = event_steps[idx + 1] if idx + 1 < len(event_steps) else slots.shape[0] - 1
                distance = float(displacements[max(step_idx - 1, 0), agent_idx])
                min_reach_times.append(distance / max(uav_vmax, 1e-9))
                available_windows.append(max(float(next_step - step_idx) * dt, 0.0))
    outside_ratio = float(np.mean(outside)) if outside.size else 0.0
    too_close_ratio = float(np.mean(too_close)) if too_close.size else 0.0
    unreachable_ratio = float(np.mean(outside | obstacle_blocked)) if outside.size else 0.0
    feasible_ratio = float(np.mean((~outside) & (~obstacle_blocked))) if outside.size else 1.0
    min_margin = float(np.min(margins)) if margins.size else float("inf")
    inside_obstacle_ratio = float(np.mean(obstacle_blocked)) if obstacle_blocked.size else 0.0
    slot_obstacle_too_close_ratio = float(np.mean(obstacle_too_close)) if obstacle_too_close.size else 0.0
    min_obstacle_clearance = float(np.min(obstacle_clearances)) if obstacle_clearances.size else float("inf")
    los_blocked_ratio = float(np.mean(line_of_sight_blocked)) if line_of_sight_blocked.size else 0.0

    reason = ""
    target_infeasible = False
    feasibility_factor = float(fc.get("jump_feasibility_factor", cfg.get("success", {}).get("jump_reacquisition_budget_factor", 1.6)))
    jump_too_large = bool(
        min_reach_times
        and any(float(m) > float(w) * feasibility_factor for m, w in zip(min_reach_times, available_windows))
    )
    if outside_ratio > outside_threshold:
        target_infeasible = True
        reason = "slot_outside_boundary"
    elif unreachable_ratio > unreachable_threshold:
        target_infeasible = True
        reason = "slot_unreachable_obstacle_or_boundary"
    elif jump_too_large:
        target_infeasible = True
        reason = "jump_too_large_for_available_window"
    elif actual_p95 > uav_vmax * speed_tol:
        target_infeasible = True
        reason = "continuous_slot_speed_exceeds_uav_vmax"

    return {
        "actual_slot_speed_mean": actual_mean,
        "actual_slot_speed_p95": actual_p95,
        "actual_slot_speed_max": actual_max,
        "continuous_slot_speed_p95": actual_p95,
        "number_of_generator_jump_events": int(jump_distances.size),
        "jump_event_displacement_mean": float(np.mean(jump_distances)) if jump_distances.size else 0.0,
        "jump_event_displacement_max": float(np.max(jump_distances)) if jump_distances.size else 0.0,
        "minimum_reach_time": float(np.mean(min_reach_times)) if min_reach_times else 0.0,
        "available_reacquisition_window": float(np.mean(available_windows)) if available_windows else 0.0,
        "jump_too_large_for_window": bool(jump_too_large),
        "slot_out_of_bounds_ratio": outside_ratio,
        "slot_inside_obstacle_ratio": inside_obstacle_ratio,
        "slot_min_obstacle_clearance": min_obstacle_clearance,
        "slot_obstacle_too_close_ratio": slot_obstacle_too_close_ratio,
        "start_goal_free_space_valid": bool(start_goal_valid),
        "line_of_sight_blocked_ratio": los_blocked_ratio,
        "slot_min_boundary_margin": min_margin,
        "slot_feasible_ratio": feasible_ratio,
        "slot_too_close_ratio": too_close_ratio,
        "slot_unreachable_ratio": unreachable_ratio,
        "target_infeasible": bool(target_infeasible),
        "infeasible_reason": reason,
    }


def _obstacle_clearance_xy(position_xy: np.ndarray, obs, *, uav_radius: float) -> float:
    p = np.asarray(position_xy, dtype=np.float64).reshape(2)
    c = np.asarray(obs.center, dtype=np.float64).reshape(2)
    if getattr(obs, "kind", "circle") == "aabb" and getattr(obs, "half_extents", None) is not None:
        half = np.asarray(obs.half_extents, dtype=np.float64).reshape(2)
        q = np.maximum(np.abs(p - c) - half, 0.0)
        return float(np.linalg.norm(q) - float(uav_radius))
    return float(np.linalg.norm(p - c) - float(obs.radius) - float(uav_radius))
