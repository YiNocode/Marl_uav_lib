"""Deterministic manifold-generator debug cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


DEFAULT_BOUNDARY = {"xmin": 0.0, "xmax": 20.0, "ymin": 0.0, "ymax": 20.0}


@dataclass(frozen=True)
class DebugCase:
    case_id: str
    description: str
    expected: str
    steps: list[dict]


def _obstacle(x: float, y: float, r: float) -> dict:
    return {"center": np.array([x, y], dtype=np.float64), "radius": float(r)}


def _step(xy: tuple[float, float], obstacles: list[dict] | None = None, boundary: dict | None = None) -> dict:
    return {
        "evader_state": np.array([float(xy[0]), float(xy[1])], dtype=np.float64),
        "obstacles": list(obstacles or []),
        "boundary": dict(boundary or DEFAULT_BOUNDARY),
    }


def _repeat_to_length(items: list, num_steps: int) -> list:
    if num_steps <= 0:
        return []
    out = []
    for i in range(num_steps):
        out.append(items[min(len(items) - 1, int(i * len(items) / max(num_steps, 1)))])
    return out


def g0_static_no_obstacle(num_steps: int, seed: int = 0, boundary: dict | None = None) -> DebugCase:
    del seed
    return DebugCase(
        "g0_static_no_obstacle",
        "Fixed evader at the arena center with no obstacles.",
        "Closed curve, target inside, no self-intersection.",
        [_step((10.0, 10.0), boundary=boundary) for _ in range(num_steps)],
    )


def g1_dynamic_no_obstacle(num_steps: int, seed: int = 0, boundary: dict | None = None) -> DebugCase:
    del seed
    xs = np.linspace(5.0, 15.0, max(num_steps, 1), dtype=np.float64)
    return DebugCase(
        "g1_dynamic_no_obstacle",
        "Evader translates horizontally across the arena with no obstacles.",
        "Manifold follows smoothly without large jumps.",
        [_step((float(x), 10.0), boundary=boundary) for x in xs],
    )


def g2_boundary_stress(num_steps: int, seed: int = 0, boundary: dict | None = None) -> DebugCase:
    del seed
    positions = [(1.5, 10.0), (18.5, 10.0), (10.0, 1.5), (10.0, 18.5), (1.5, 1.5)]
    return DebugCase(
        "g2_boundary_stress",
        "Evader is placed near edges and a corner.",
        "No boundary violation if feasible; no broken clipped curve.",
        [_step(pos, boundary=boundary) for pos in _repeat_to_length(positions, num_steps)],
    )


def g3_single_obstacle(num_steps: int, seed: int = 0, boundary: dict | None = None) -> DebugCase:
    del seed
    obstacle_positions = [(15.0, 10.0), (13.0, 10.0), (11.5, 10.0), (10.8, 10.0)]
    steps = [
        _step((10.0, 10.0), [_obstacle(x, y, 0.8)], boundary=boundary)
        for x, y in _repeat_to_length(obstacle_positions, num_steps)
    ]
    return DebugCase(
        "g3_single_obstacle",
        "One obstacle is placed outside, on, inside, and very close to the nominal manifold.",
        "No obstacle penetration or self-intersection, or explicit INFEASIBLE.",
        steps,
    )


def g4_multi_obstacle(num_steps: int, seed: int = 0, boundary: dict | None = None) -> DebugCase:
    rng = np.random.default_rng(int(seed))
    layouts: list[list[dict]] = [
        [_obstacle(13.0, 10.0, 0.75), _obstacle(7.0, 10.0, 0.75)],
        [_obstacle(8.5, 10.0, 0.55), _obstacle(10.0, 10.0, 0.55), _obstacle(11.5, 10.0, 0.55)],
        [_obstacle(10.0, 8.2, 0.7), _obstacle(10.0, 11.8, 0.7), _obstacle(13.0, 10.0, 0.7)],
        [_obstacle(8.0, 8.0, 0.65), _obstacle(8.0, 10.0, 0.65), _obstacle(8.0, 12.0, 0.65), _obstacle(10.0, 8.0, 0.65), _obstacle(12.0, 8.0, 0.65)],
    ]
    random_layout = [
        _obstacle(float(rng.uniform(6.0, 14.0)), float(rng.uniform(6.0, 14.0)), float(rng.uniform(0.35, 0.75)))
        for _ in range(8)
    ]
    layouts.append(random_layout)
    steps = [_step((10.0, 10.0), obs, boundary=boundary) for obs in _repeat_to_length(layouts, num_steps)]
    return DebugCase(
        "g4_multi_obstacle",
        "Multiple obstacle layouts: opposite obstacles, wall, narrow corridor, U-shape, fixed random field.",
        "Valid closed curve if feasible; otherwise explicit INFEASIBLE.",
        steps,
    )


def g5_moving_obstacle(num_steps: int, seed: int = 0, boundary: dict | None = None) -> DebugCase:
    del seed
    xs = np.linspace(16.0, 12.0, max(num_steps, 1), dtype=np.float64)
    steps = [_step((10.0, 10.0), [_obstacle(float(x), 10.0, 0.8)], boundary=boundary) for x in xs]
    return DebugCase(
        "g5_moving_obstacle",
        "One obstacle moves slowly from outside toward the nominal manifold.",
        "Smooth deformation without sudden jumps when obstacle influence starts.",
        steps,
    )


CASE_BUILDERS: dict[str, Callable[[int, int, dict | None], DebugCase]] = {
    "g0_static_no_obstacle": g0_static_no_obstacle,
    "g1_dynamic_no_obstacle": g1_dynamic_no_obstacle,
    "g2_boundary_stress": g2_boundary_stress,
    "g3_single_obstacle": g3_single_obstacle,
    "g4_multi_obstacle": g4_multi_obstacle,
    "g5_moving_obstacle": g5_moving_obstacle,
}


ALIASES = {
    "g0": "g0_static_no_obstacle",
    "g1": "g1_dynamic_no_obstacle",
    "g2": "g2_boundary_stress",
    "g3": "g3_single_obstacle",
    "g4": "g4_multi_obstacle",
    "g5": "g5_moving_obstacle",
}


def list_case_ids() -> list[str]:
    return list(CASE_BUILDERS.keys())


def resolve_case_id(case_id: str) -> str:
    key = str(case_id).strip().lower()
    return ALIASES.get(key, key)


def make_case(case_id: str, num_steps: int, seed: int = 0, boundary: dict | None = None) -> DebugCase:
    resolved = resolve_case_id(case_id)
    if resolved not in CASE_BUILDERS:
        known = ", ".join(["all", *list_case_ids()])
        raise KeyError(f"unknown case '{case_id}'. Known cases: {known}")
    return CASE_BUILDERS[resolved](int(num_steps), int(seed), boundary)

