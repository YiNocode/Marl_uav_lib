"""Obstacle-map generators for the independent slot-tracking benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import Obstacle


@dataclass(frozen=True)
class ObstacleMap:
    """A named collection of 2D circular obstacles."""

    name: str
    obstacles: list[Obstacle]


def circle(x: float, y: float, r: float) -> Obstacle:
    return Obstacle(kind="circle", center=np.array([x, y], dtype=np.float64), radius=float(r))


def box(x: float, y: float, hx: float, hy: float) -> Obstacle:
    half = np.array([float(hx), float(hy)], dtype=np.float64)
    return Obstacle(
        kind="aabb",
        center=np.array([float(x), float(y)], dtype=np.float64),
        radius=float(np.linalg.norm(half)),
        half_extents=half,
    )


def obstacle_records(obstacles: list[Obstacle]) -> list[dict[str, float]]:
    """Convert obstacle objects to serializable records."""
    out: list[dict[str, float]] = []
    for obs in obstacles:
        out.append(
            {
                "x": float(obs.center[0]),
                "y": float(obs.center[1]),
                "radius": float(obs.radius),
            }
        )
    return out


def sparse_random_obstacles(rng: np.random.Generator, world_xy: float, cfg: dict[str, Any] | None = None) -> ObstacleMap:
    """Sample a sparse map while keeping the middle corridor mostly usable."""
    raw = dict(cfg or {})
    n_min = int(raw.get("count_min", 3))
    n_max = int(raw.get("count_max", 6))
    n = int(rng.integers(n_min, n_max + 1))
    obstacles: list[Obstacle] = []
    for _ in range(n):
        for _attempt in range(100):
            xy = rng.uniform(-0.65 * world_xy, 0.65 * world_xy, size=2)
            if np.linalg.norm(xy) < 2.0:
                continue
            if abs(float(xy[1])) < 0.9 and -6.0 < float(xy[0]) < 6.0:
                continue
            r = float(rng.uniform(float(raw.get("radius_min", 0.35)), float(raw.get("radius_max", 0.9))))
            obstacles.append(circle(float(xy[0]), float(xy[1]), r))
            break
    return ObstacleMap("sparse_random_obstacles", obstacles)


def single_blocking_obstacle(_rng: np.random.Generator, _world_xy: float, cfg: dict[str, Any] | None = None) -> ObstacleMap:
    raw = dict(cfg or {})
    radius = float(raw.get("radius", 1.1))
    return ObstacleMap("single_blocking_obstacle", [circle(0.0, 0.0, radius)])


def narrow_passage(_rng: np.random.Generator, _world_xy: float, cfg: dict[str, Any] | None = None) -> ObstacleMap:
    raw = dict(cfg or {})
    radius = float(raw.get("radius", 1.0))
    passage_width = float(raw.get("passage_width", 1.2))
    center_y = radius + 0.5 * max(passage_width, 0.05)
    return ObstacleMap("narrow_passage", [circle(0.0, center_y, radius), circle(0.0, -center_y, radius)])


def u_shaped_trap(_rng: np.random.Generator, _world_xy: float, cfg: dict[str, Any] | None = None) -> ObstacleMap:
    raw = dict(cfg or {})
    radius = float(raw.get("radius", 0.65))
    return ObstacleMap(
        "u_shaped_trap",
        [
            circle(3.2, 0.0, radius),
            circle(1.8, 1.35, radius),
            circle(1.8, -1.35, radius),
            circle(2.7, 1.35, radius),
            circle(2.7, -1.35, radius),
        ],
    )


def boundary_obstacle_combo(_rng: np.random.Generator, world_xy: float, cfg: dict[str, Any] | None = None) -> ObstacleMap:
    raw = dict(cfg or {})
    edge = float(raw.get("edge_x", 0.72 * world_xy))
    main_y = float(raw.get("main_y", 0.0))
    secondary_y = float(raw.get("secondary_y", 1.0))
    return ObstacleMap(
        "boundary_obstacle_combo",
        [
            circle(edge, main_y, float(raw.get("main_radius", 1.1))),
            circle(edge - 1.4, secondary_y, float(raw.get("secondary_radius", 0.7))),
        ],
    )


def empty_obstacles() -> ObstacleMap:
    return ObstacleMap("none", [])


def build_obstacle_map(name: str, rng: np.random.Generator, world_xy: float, cfg: dict[str, Any] | None = None) -> ObstacleMap:
    """Build a configured obstacle map by name."""
    builders: dict[str, Any] = {
        "none": lambda _rng, _w: empty_obstacles(),
        "sparse_random_obstacles": sparse_random_obstacles,
        "single_blocking_obstacle": single_blocking_obstacle,
        "narrow_passage": narrow_passage,
        "u_shaped_trap": u_shaped_trap,
        "boundary_obstacle_combo": boundary_obstacle_combo,
    }
    if name not in builders:
        raise ValueError(f"Unknown obstacle map: {name}")
    if name == "none":
        return builders[name](rng, float(world_xy))
    return builders[name](rng, float(world_xy), cfg)
