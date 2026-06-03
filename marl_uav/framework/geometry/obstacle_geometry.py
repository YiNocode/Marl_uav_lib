"""2D obstacle geometry: inflation, line-of-sight, and path collision checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ObstacleKind = Literal["circle", "aabb", "polygon"]


@dataclass(frozen=True)
class Obstacle:
    """Unified 2D obstacle representation for planning and reachability."""

    kind: ObstacleKind
    center: np.ndarray  # [2]
    radius: float = 0.0  # circle or conservative bound
    half_extents: np.ndarray | None = None  # [2] for AABB
    vertices: np.ndarray | None = None  # [M, 2] for polygon

    def copy(self) -> Obstacle:
        return Obstacle(
            kind=self.kind,
            center=np.asarray(self.center, dtype=np.float64).copy(),
            radius=float(self.radius),
            half_extents=None
            if self.half_extents is None
            else np.asarray(self.half_extents, dtype=np.float64).copy(),
            vertices=None
            if self.vertices is None
            else np.asarray(self.vertices, dtype=np.float64).copy(),
        )


@dataclass(frozen=True)
class CircleObstacle:
    """Convenience wrapper for circular obstacles."""

    center: np.ndarray
    radius: float

    def to_obstacle(self) -> Obstacle:
        return Obstacle(
            kind="circle",
            center=np.asarray(self.center, dtype=np.float64).reshape(2),
            radius=float(self.radius),
        )


def inflate_obstacle(
    obstacle: Obstacle,
    *,
    uav_radius: float,
    safety_margin: float,
) -> Obstacle:
    """Inflate obstacle by ``uav_radius + safety_margin`` (conservative for non-circles)."""
    pad = max(float(uav_radius), 0.0) + max(float(safety_margin), 0.0)
    obs = obstacle.copy()
    if obs.kind == "circle":
        return Obstacle(
            kind="circle",
            center=obs.center.copy(),
            radius=float(obs.radius) + pad,
        )
    if obs.kind == "aabb":
        he = np.asarray(obs.half_extents, dtype=np.float64).reshape(2)
        return Obstacle(
            kind="aabb",
            center=obs.center.copy(),
            radius=float(np.linalg.norm(he + pad)),
            half_extents=he + pad,
        )
    # polygon: conservative bounding-circle inflation
    verts = np.asarray(obs.vertices, dtype=np.float64)
    if verts.size == 0:
        return Obstacle(kind="polygon", center=obs.center.copy(), radius=pad, vertices=verts)
    dists = np.linalg.norm(verts - obs.center.reshape(1, 2), axis=1)
    return Obstacle(
        kind="polygon",
        center=obs.center.copy(),
        radius=float(np.max(dists)) + pad,
        vertices=verts.copy(),
    )


def _segment_point_distance_sq(p0: np.ndarray, p1: np.ndarray, q: np.ndarray) -> float:
    """Squared distance from point q to segment p0->p1."""
    v = p1 - p0
    w = q - p0
    c1 = float(np.dot(w, v))
    if c1 <= 0.0:
        return float(np.dot(w, w))
    c2 = float(np.dot(v, v))
    if c2 <= c1:
        d = q - p1
        return float(np.dot(d, d))
    t = c1 / c2
    proj = p0 + t * v
    d = q - proj
    return float(np.dot(d, d))


def _segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, half: np.ndarray) -> bool:
    """Liang-Barsky style segment-AABB intersection (2D)."""
    min_c = center - half
    max_c = center + half
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if p0[i] < min_c[i] or p0[i] > max_c[i]:
                return False
        else:
            inv = 1.0 / d[i]
            t_near = (min_c[i] - p0[i]) * inv
            t_far = (max_c[i] - p0[i]) * inv
            if t_near > t_far:
                t_near, t_far = t_far, t_near
            t0 = max(t0, t_near)
            t1 = min(t1, t_far)
            if t0 > t1:
                return False
    return True


def _segments_intersect(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> bool:
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1 = cross(a0, a1, b0)
    d2 = cross(a0, a1, b1)
    d3 = cross(b0, b1, a0)
    d4 = cross(b0, b1, a1)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    return False


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = int(vertices.shape[0])
    j = n - 1
    for i in range(n):
        xi, yi = float(vertices[i, 0]), float(vertices[i, 1])
        xj, yj = float(vertices[j, 0]), float(vertices[j, 1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def line_segment_intersects_obstacle(
    p0: np.ndarray,
    p1: np.ndarray,
    obstacle: Obstacle,
    *,
    treat_tangent_as_blocked: bool = True,
) -> bool:
    """Return True if segment p0->p1 intersects the (already inflated) obstacle."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    p1 = np.asarray(p1, dtype=np.float64).reshape(2)
    eps = 1e-9 if treat_tangent_as_blocked else 0.0

    if obstacle.kind == "circle":
        c = np.asarray(obstacle.center, dtype=np.float64).reshape(2)
        r = float(obstacle.radius)
        dist_sq = _segment_point_distance_sq(p0, p1, c)
        return dist_sq <= (r + eps) ** 2

    if obstacle.kind == "aabb":
        he = np.asarray(obstacle.half_extents, dtype=np.float64).reshape(2)
        c = np.asarray(obstacle.center, dtype=np.float64).reshape(2)
        return _segment_intersects_aabb(p0, p1, c, he)

    # polygon: edge intersections or midpoint inside conservative circle / polygon
    verts = np.asarray(obstacle.vertices, dtype=np.float64)
    if verts.shape[0] >= 3:
        for i in range(int(verts.shape[0])):
            j = (i + 1) % int(verts.shape[0])
            if _segments_intersect(p0, p1, verts[i], verts[j]):
                return True
        mid = 0.5 * (p0 + p1)
        if _point_in_polygon(mid, verts):
            return True
    # fallback: conservative circle
    c = np.asarray(obstacle.center, dtype=np.float64).reshape(2)
    r = float(obstacle.radius)
    dist_sq = _segment_point_distance_sq(p0, p1, c)
    return dist_sq <= (r + eps) ** 2


def has_line_of_sight(
    p0: np.ndarray,
    p1: np.ndarray,
    obstacles: list[Obstacle],
    *,
    safety_margin: float = 0.0,
    uav_radius: float = 0.0,
    treat_tangent_as_blocked: bool = True,
) -> bool:
    """True when segment p0->p1 does not intersect any inflated obstacle."""
    inflated = [
        inflate_obstacle(o, uav_radius=uav_radius, safety_margin=safety_margin) for o in obstacles
    ]
    for obs in inflated:
        if line_segment_intersects_obstacle(
            p0, p1, obs, treat_tangent_as_blocked=treat_tangent_as_blocked
        ):
            return False
    return True


def path_length(path: list[np.ndarray] | np.ndarray) -> float:
    """Polyline length for a waypoint list."""
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    diffs = np.diff(pts[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def collision_check_path(
    path: list[np.ndarray] | np.ndarray,
    obstacles: list[Obstacle],
    *,
    safety_margin: float = 0.0,
    uav_radius: float = 0.0,
    treat_tangent_as_blocked: bool = True,
) -> bool:
    """True when every segment of path is collision-free w.r.t. inflated obstacles."""
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return True
    inflated = [
        inflate_obstacle(o, uav_radius=uav_radius, safety_margin=safety_margin) for o in obstacles
    ]
    for i in range(int(pts.shape[0]) - 1):
        p0, p1 = pts[i, :2], pts[i + 1, :2]
        for obs in inflated:
            if line_segment_intersects_obstacle(
                p0, p1, obs, treat_tangent_as_blocked=treat_tangent_as_blocked
            ):
                return False
    return True
